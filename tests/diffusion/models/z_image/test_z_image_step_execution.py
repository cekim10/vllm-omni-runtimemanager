from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    MingImagePipeline,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = {}
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.begin_index = 0
        self.sigma_min = 0.0

    def __deepcopy__(self, memo):
        del memo
        other = _FakeScheduler()
        other.config = dict(self.config)
        other.timesteps = self.timesteps.clone()
        other.begin_index = self.begin_index
        other.sigma_min = self.sigma_min
        return other

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs) -> None:
        del sigmas, kwargs
        if timesteps is not None:
            self.timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
        else:
            assert num_inference_steps is not None
            self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device, dtype=torch.float32)

    def set_begin_index(self, index: int) -> None:
        self.begin_index = index

    def step(self, noise_pred, timestep, latents, return_dict=False):
        del timestep, return_dict
        return (latents - noise_pred.to(latents.dtype),)


class _FakeTransformer:
    in_channels = 4

    def __call__(self, latent_list, timestep, prompt_embeds):
        del timestep
        outputs = []
        for latent, prompt_embed in zip(latent_list, prompt_embeds, strict=True):
            outputs.append(latent + prompt_embed.mean().to(latent.dtype))
        return outputs, {}


class _FakeVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5)

    def decode(self, latents, return_dict=False):
        del return_dict
        return (latents + 7.0,)


def _pipeline() -> ZImagePipeline:
    pipeline = object.__new__(ZImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline._execution_device = torch.device("cpu")
    pipeline.scheduler = _FakeScheduler()
    pipeline.transformer = _FakeTransformer()
    pipeline.vae = _FakeVAE()
    pipeline.vae_scale_factor = 8
    pipeline._profiler_lock = Lock()
    pipeline._stage_durations = {}
    pipeline.encode_prompt = lambda **kwargs: (
        [
            torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        ],
        [
            torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        ],
    )
    pipeline.prepare_latents = (
        lambda batch_size, num_channels_latents, height, width, dtype, device, generator, latents, image=None, timestep=None: (
            latents.to(device=device, dtype=dtype)
            if latents is not None
            else torch.zeros((batch_size, num_channels_latents, 4, 4), dtype=dtype, device=device)
        )
    )
    return pipeline


def _state(
    *,
    sampling: OmniDiffusionSamplingParams | None = None,
    prompts: list[str | dict] | None = None,
) -> DiffusionRequestState:
    return DiffusionRequestState(
        request_id="req-z",
        sampling=sampling or OmniDiffusionSamplingParams(num_inference_steps=4, guidance_scale=2.0),
        prompts=prompts or ["a brass astrolabe"],
    )


def test_z_image_step_execution_is_opt_in_and_ming_stays_opted_out() -> None:
    assert supports_step_execution(object.__new__(ZImagePipeline))
    assert not supports_step_execution(object.__new__(MingImagePipeline))


def test_prepare_encode_supports_resume_state_for_text_to_image() -> None:
    pipeline = _pipeline()
    restored_latents = torch.full((1, 4, 4, 4), 2.0, dtype=torch.float32)
    state = _state(
        sampling=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=2.0,
            guidance_scale_provided=True,
            latents=restored_latents,
            step_index=2,
        )
    )

    pipeline.prepare_encode(state)

    assert state.step_index == 2
    assert state.scheduler.begin_index == 2
    assert state.do_true_cfg is True
    assert state.current_timestep is not None
    torch.testing.assert_close(state.latents, restored_latents)
    assert state.prompt_embeds.shape == (1, 2, 2)
    assert state.prompt_embeds_mask.shape == (1, 2)
    assert state.negative_prompt_embeds.shape == (1, 1, 2)


def test_prepare_encode_rejects_image_conditioned_step_execution() -> None:
    pipeline = _pipeline()
    state = _state(prompts=[{"prompt": "edit this", "multi_modal_data": {"image": "image.png"}}])

    with pytest.raises(ValueError, match="text-to-image requests only"):
        pipeline.prepare_encode(state)


def test_z_image_step_contract_round_trips_cfg_and_decode() -> None:
    pipeline = _pipeline()
    state = _state(
        sampling=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=2.0,
            guidance_scale_provided=True,
            output_type="pil",
        )
    )
    pipeline.prepare_encode(state)

    batch = InputBatch.make_batch([state])
    noise_pred = pipeline.denoise_step(batch, states=[state])

    expected_noise = torch.full((1, 4, 4, 4), -3.0, dtype=torch.float32)
    torch.testing.assert_close(noise_pred, expected_noise)

    pipeline.step_scheduler(state, noise_pred)
    torch.testing.assert_close(state.latents, torch.full((1, 4, 4, 4), 3.0, dtype=torch.float32))
    assert state.step_index == 1

    out = pipeline.post_decode(state)
    torch.testing.assert_close(out.output, torch.full((1, 4, 4, 4), 9.0, dtype=torch.float32))
