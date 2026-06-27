from __future__ import annotations

from typing import cast

from vllm_omni.diffusion.state.diffusion_state import DiffusionState, Placement


class PlacementEngine:
    """Track tier budgets for diffusion checkpoints."""

    def __init__(self, gpu_budget_bytes: int, cpu_budget_bytes: int) -> None:
        self.gpu_budget = max(0, gpu_budget_bytes)
        self.cpu_budget = max(0, cpu_budget_bytes)
        self.gpu_used = 0
        self.cpu_used = 0
        self._gpu_states: dict[str, DiffusionState] = {}
        self._cpu_states: dict[str, DiffusionState] = {}

    def place(self, state: DiffusionState, previous_state: DiffusionState | None = None) -> Placement:
        if previous_state is not None:
            self.release(previous_state)

        if self.gpu_budget > 0 and self.gpu_used + state.size_bytes <= self.gpu_budget:
            self.gpu_used += state.size_bytes
            self._gpu_states[state.request_id] = state
            return Placement.GPU

        if self.cpu_budget > 0 and self.cpu_used + state.size_bytes <= self.cpu_budget:
            self.cpu_used += state.size_bytes
            self._cpu_states[state.request_id] = state
            return Placement.CPU

        return Placement.DISK

    def release(self, state: DiffusionState) -> None:
        if state.placement == Placement.GPU and self._gpu_states.pop(state.request_id, None) is not None:
            self.gpu_used = max(0, self.gpu_used - state.size_bytes)
        elif state.placement == Placement.CPU and self._cpu_states.pop(state.request_id, None) is not None:
            self.cpu_used = max(0, self.cpu_used - state.size_bytes)

    def evict_one(self) -> DiffusionState | None:
        if not self._gpu_states:
            return None
        victim = min(
            self._gpu_states.values(),
            key=lambda state: state.value_score / max(state.size_bytes, 1),
        )
        self.release(victim)
        return victim

    def gpu_states(self) -> dict[str, DiffusionState]:
        return cast(dict[str, DiffusionState], dict(self._gpu_states))

    def cpu_states(self) -> dict[str, DiffusionState]:
        return cast(dict[str, DiffusionState], dict(self._cpu_states))
