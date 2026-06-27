# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.state import DiffusionStateManager, FidelityPolicy, Placement, ValueEstimator
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(request_id: str = "req-1") -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["prompt"],
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4),
    )


def test_value_estimator_uses_alphas_cumprod_schedule() -> None:
    scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.9, 0.6, 0.3], dtype=torch.float32))
    estimator = ValueEstimator(scheduler, timesteps=torch.tensor([0, 1, 2]))

    values = [estimator.get_value(i) for i in range(3)]
    assert values[0] == pytest.approx(1.0)
    assert values[1] < values[0]
    assert values[2] < values[1]


def test_value_estimator_falls_back_to_sigmas() -> None:
    scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32))
    estimator = ValueEstimator(scheduler, timesteps=torch.tensor([1000, 500, 0]))

    values = [estimator.get_value(i) for i in range(3)]
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.25)
    assert values[2] == pytest.approx(0.0625)


def test_fidelity_policy_round_trips_lossy_and_lossless_tiers() -> None:
    policy = FidelityPolicy(theta_h=0.7, theta_w=0.3)
    latent = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32).reshape(1, 4, 8)

    for value_score in (0.9, 0.5, 0.1):
        fidelity = policy.assign(value_score)
        compressed, scale = policy.compress(latent, fidelity)
        restored = policy.decompress(compressed, scale, fidelity)
        assert restored.shape == latent.shape
        if fidelity.name == "LOSSLESS":
            assert restored.dtype == latent.dtype
            torch.testing.assert_close(restored, latent, atol=0.0, rtol=0.0)
        else:
            torch.testing.assert_close(restored, latent, atol=0.05, rtol=0.05)


def test_fidelity_policy_preserves_bfloat16_for_lossless_tier() -> None:
    policy = FidelityPolicy(theta_h=0.7, theta_w=0.3)
    latent = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8).to(torch.bfloat16)

    compressed, scale = policy.compress(latent, Fidelity.LOSSLESS)
    restored = policy.decompress(compressed, scale, Fidelity.LOSSLESS)

    assert compressed.dtype == torch.bfloat16
    assert restored.dtype == torch.bfloat16
    torch.testing.assert_close(restored, latent, atol=0.0, rtol=0.0)


def test_state_manager_restores_latest_checkpoint_from_memory() -> None:
    manager = DiffusionStateManager(gpu_budget_bytes=0, cpu_budget_bytes=1 << 20)
    latent = torch.randn(1, 4, 8, dtype=torch.float32)

    state = manager.on_step_complete(
        request_id="req-1",
        step_idx=2,
        total_steps=6,
        latent=latent,
        value_score=0.8,
    )

    restored = manager.restore(state)
    assert state.placement == Placement.CPU
    assert restored.dtype == latent.dtype
    torch.testing.assert_close(restored, latent, atol=0.0, rtol=0.0)


def test_state_manager_restores_checkpoint_from_disk_and_builds_resume_request(tmp_path) -> None:
    manager = DiffusionStateManager(
        gpu_budget_bytes=0,
        cpu_budget_bytes=8,
        disk_path=tmp_path,
    )
    latent = torch.randn(1, 4, 8, dtype=torch.float32)

    state = manager.on_step_complete(
        request_id="req-disk",
        step_idx=3,
        total_steps=8,
        latent=latent,
        value_score=0.2,
    )

    assert state.placement == Placement.DISK
    assert state.disk_path is not None

    request = _make_request("req-disk")
    resumed = manager.restore_request(request)
    assert resumed.request_id == "req-disk"
    assert resumed.sampling_params.step_index == 3
    assert resumed.sampling_params.latents is not None
    restored = manager.restore(state)
    torch.testing.assert_close(resumed.sampling_params.latents, restored)
