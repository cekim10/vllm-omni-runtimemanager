from __future__ import annotations

import copy
import os
import tempfile
import uuid
from pathlib import Path

import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.state.diffusion_state import DiffusionState, Fidelity, Placement
from vllm_omni.diffusion.state.fidelity_policy import FidelityPolicy
from vllm_omni.diffusion.state.placement_engine import PlacementEngine


class DiffusionStateManager:
    """Persist resumable diffusion checkpoints across denoise steps."""

    def __init__(
        self,
        noise_scheduler: object | None = None,
        gpu_budget_bytes: int = 0,
        cpu_budget_bytes: int = 0,
        theta_h: float = 0.7,
        theta_w: float = 0.3,
        disk_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.noise_scheduler = noise_scheduler
        self.fid_policy = FidelityPolicy(theta_h, theta_w)
        if not torch.cuda.is_available():
            gpu_budget_bytes = 0
        self.placement = PlacementEngine(gpu_budget_bytes, cpu_budget_bytes)
        self.disk_path = Path(disk_path or os.path.join(tempfile.gettempdir(), "vllm_omni_diffusion_states"))
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, DiffusionState] = {}

    def on_step_complete(
        self,
        request_id: str,
        step_idx: int,
        total_steps: int,
        latent: torch.Tensor,
        value_score: float,
    ) -> DiffusionState:
        previous_state = self._store.get(request_id)
        fidelity = self.fid_policy.assign(value_score)
        compressed, scale = self.fid_policy.compress(latent, fidelity)
        size_bytes = self.fid_policy.estimate_size_bytes(compressed, scale)
        state = DiffusionState(
            request_id=request_id,
            step_idx=step_idx,
            total_steps=total_steps,
            latent=compressed,
            fidelity=fidelity,
            placement=Placement.CPU,
            value_score=value_score,
            size_bytes=size_bytes,
            scale=scale,
        )
        state.placement = self.placement.place(state, previous_state=previous_state)
        self._materialize_state(state)
        self._cleanup_persisted_state(previous_state)
        self._store[request_id] = state
        return state

    def on_failure(self, request_id: str) -> DiffusionState | None:
        return self._store.get(request_id)

    def restore(self, state: DiffusionState) -> torch.Tensor:
        latent, scale = self._load_state_payload(state)
        return self.fid_policy.decompress(latent, scale, state.fidelity)

    def restore_request(self, request: OmniDiffusionRequest, request_id: str | None = None) -> OmniDiffusionRequest:
        target_id = request_id or request.request_id
        state = self.on_failure(target_id)
        if state is None:
            raise KeyError(f"No diffusion checkpoint found for request {target_id!r}.")

        restored_request = copy.deepcopy(request)
        restored_request.sampling_params.latents = self.restore(state)
        restored_request.sampling_params.step_index = state.step_idx
        restored_request.request_id = target_id
        return restored_request

    def release_request(self, request_id: str) -> None:
        state = self._store.pop(request_id, None)
        if state is None:
            return
        self.placement.release(state)
        self._cleanup_persisted_state(state)

    def clear(self) -> None:
        for request_id in list(self._store):
            self.release_request(request_id)

    def _materialize_state(self, state: DiffusionState) -> None:
        if state.placement == Placement.DISK:
            disk_path = self.disk_path / f"{state.request_id}_{state.step_idx}_{uuid.uuid4().hex}.pt"
            torch.save({"latent": state.latent, "scale": state.scale}, disk_path)
            state.disk_path = str(disk_path)
            state.latent = None
            state.scale = None
            return

        if state.placement == Placement.GPU and torch.cuda.is_available():
            state.latent = state.latent.to(device="cuda")
            if isinstance(state.scale, torch.Tensor):
                state.scale = state.scale.to(device="cuda")
            return

        state.placement = Placement.CPU
        state.latent = state.latent.to(device="cpu")
        if isinstance(state.scale, torch.Tensor):
            state.scale = state.scale.to(device="cpu")

    def _load_state_payload(self, state: DiffusionState) -> tuple[torch.Tensor, torch.Tensor | float | None]:
        if state.placement == Placement.DISK:
            if state.disk_path is None:
                raise ValueError(f"Disk state for request {state.request_id!r} is missing its payload path.")
            payload = torch.load(state.disk_path, map_location="cpu")
            return payload["latent"], payload.get("scale")

        if state.latent is None:
            raise ValueError(f"In-memory state for request {state.request_id!r} is missing its latent tensor.")
        return state.latent, state.scale

    def _cleanup_persisted_state(self, state: DiffusionState | None) -> None:
        if state is None:
            return
        if state.placement == Placement.DISK and state.disk_path is not None:
            try:
                os.remove(state.disk_path)
            except FileNotFoundError:
                pass
