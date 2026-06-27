from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState


class ValueEstimator:
    """Estimate checkpoint value from the request's diffusion noise schedule."""

    def __init__(
        self,
        noise_scheduler: object,
        timesteps: torch.Tensor | Sequence[torch.Tensor | float | int] | None = None,
    ) -> None:
        self._value_table = self._build_value_table(noise_scheduler, timesteps)

    @classmethod
    def from_request_state(cls, state: DiffusionRequestState) -> "ValueEstimator":
        scheduler = state.scheduler
        if scheduler is None:
            raise ValueError(f"Request {state.request_id} has no scheduler; cannot estimate value.")
        return cls(scheduler, state.timesteps)

    def get_value(self, step_idx: int) -> float:
        if self._value_table.numel() == 0:
            return 0.0
        clamped_step = min(max(step_idx, 0), self._value_table.numel() - 1)
        return float(self._value_table[clamped_step].item())

    @staticmethod
    def _build_value_table(
        noise_scheduler: object,
        timesteps: torch.Tensor | Sequence[torch.Tensor | float | int] | None,
    ) -> torch.Tensor:
        alpha_snr = ValueEstimator._snr_from_alphas(noise_scheduler, timesteps)
        if alpha_snr is None:
            sigma_snr = ValueEstimator._snr_from_sigmas(noise_scheduler, timesteps)
            if sigma_snr is None:
                raise ValueError(
                    "Noise scheduler does not expose a usable `alphas_cumprod` or `sigmas` schedule."
                )
            snr = sigma_snr
        else:
            snr = alpha_snr

        denom = snr[0].clamp_min(1e-8)
        return (snr / denom).clamp_min(0.0)

    @staticmethod
    def _normalize_timesteps(
        timesteps: torch.Tensor | Sequence[torch.Tensor | float | int] | None,
    ) -> torch.Tensor | None:
        if timesteps is None:
            return None
        if isinstance(timesteps, torch.Tensor):
            if timesteps.ndim == 0:
                return timesteps.reshape(1).detach().cpu()
            return timesteps.detach().cpu().reshape(-1)

        values: list[float] = []
        for timestep in timesteps:
            if isinstance(timestep, torch.Tensor):
                values.append(float(timestep.detach().cpu().item()))
            else:
                values.append(float(timestep))
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _snr_from_alphas(
        noise_scheduler: object,
        timesteps: torch.Tensor | Sequence[torch.Tensor | float | int] | None,
    ) -> torch.Tensor | None:
        alphas_cumprod = getattr(noise_scheduler, "alphas_cumprod", None)
        if alphas_cumprod is None:
            return None

        alphas = torch.as_tensor(alphas_cumprod, dtype=torch.float32).reshape(-1)
        if alphas.numel() == 0:
            return None

        schedule_timesteps = ValueEstimator._normalize_timesteps(timesteps)
        if schedule_timesteps is not None:
            indices = schedule_timesteps.round().to(torch.long).clamp_(0, alphas.numel() - 1)
            alpha_bar = alphas.index_select(0, indices)
        else:
            alpha_bar = alphas

        return alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)

    @staticmethod
    def _snr_from_sigmas(
        noise_scheduler: object,
        timesteps: torch.Tensor | Sequence[torch.Tensor | float | int] | None,
    ) -> torch.Tensor | None:
        sigmas = getattr(noise_scheduler, "sigmas", None)
        if sigmas is None:
            return None

        sigma_values = torch.as_tensor(sigmas, dtype=torch.float32).reshape(-1)
        if sigma_values.numel() == 0:
            return None

        schedule_timesteps = ValueEstimator._normalize_timesteps(timesteps)
        if schedule_timesteps is not None:
            expected = schedule_timesteps.numel()
            if sigma_values.numel() == expected + 1:
                sigma_values = sigma_values[:-1]
            elif sigma_values.numel() != expected:
                sigma_values = sigma_values[:expected]

        sigma_values = sigma_values.clamp_min(1e-8)
        return 1.0 / sigma_values.square()
