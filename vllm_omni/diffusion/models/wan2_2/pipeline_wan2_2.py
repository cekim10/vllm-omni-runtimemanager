# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)
DEBUG_PERF = False
WAN_SAMPLE_SOLVER_CHOICES = {"unipc", "euler"}


def build_wan_scheduler(sample_solver: str, flow_shift: float) -> Any:
    if sample_solver == "unipc":
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )
    if sample_solver == "euler":
        return WanEulerScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

    raise ValueError(
        f"Unsupported Wan sample_solver: {sample_solver}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}"
    )


def resolve_wan_sample_solver(req: OmniDiffusionRequest, default: str = "unipc") -> str:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN_SAMPLE_SOLVER_CHOICES:
        raise ValueError(f"Invalid sample_solver={raw!r}. Expected one of: {sorted(WAN_SAMPLE_SOLVER_CHOICES)}")
    return sample_solver


def resolve_wan_flow_shift(req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> float:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}. flow_shift must be a float.") from exc


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None, prefix: str = ""
) -> WanTransformer3DModel:
    """Create WanTransformer3DModel from config dict."""
    kwargs: dict = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]

    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, config["quantization_config"])

    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    if prefix:
        kwargs["prefix"] = prefix

    return WanTransformer3DModel(**kwargs)


def get_wan22_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video
        custom_output = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            custom_output["video_fps_multiplier"] = multiplier
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "custom_output": custom_output,
        }

    return post_process_func


def get_wan22_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for Wan2.2: optionally load and resize input image for I2V mode."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                continue

            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"""Unsupported image format {raw_image.__class__}.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            # Calculate dimensions based on aspect ratio if not provided
            if request.sampling_params.height is None or request.sampling_params.width is None:
                # Default max area for 720P
                max_area = 720 * 1280
                aspect_ratio = image.height / image.width

                # Calculate dimensions maintaining aspect ratio
                mod_value = 16  # Must be divisible by 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize image to target dimensions
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore # Above has ensured that width & height are not None
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence already checked above

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22Pipeline(
    nn.Module,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer", "transformer_2"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Read model_index.json to detect expand_timesteps mode (for TI2V-5B)
        self.expand_timesteps = False
        self.has_transformer_2 = False
        if local_files_only:
            model_index_path = os.path.join(model, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
            # Check if this is a two-stage model (MoE with transformer_2)
            transformer_2_path = os.path.join(model, "transformer_2")
            self.has_transformer_2 = os.path.exists(transformer_2_path)
        else:
            # For remote models, download and read model_index.json
            try:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    model_index = json.load(f)
                    self.expand_timesteps = model_index.get("expand_timesteps", False)
                    # Check transformer_2 from model_index
                    transformer_2_info = model_index.get("transformer_2", [None, None])
                    self.has_transformer_2 = transformer_2_info[0] is not None
            except Exception:
                pass

        self.boundary_ratio = od_config.boundary_ratio

        # Determine which transformers to load based on boundary_ratio
        # boundary_ratio=1.0: only load transformer_2 (low-noise stage only)
        # boundary_ratio=0.0: only load transformer (high-noise stage only)
        # otherwise: load both transformers
        load_transformer = self.boundary_ratio != 1.0 if self.boundary_ratio is not None else True
        load_transformer_2 = self.has_transformer_2 and (
            self.boundary_ratio != 0.0 if self.boundary_ratio is not None else True
        )

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        if load_transformer:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer",
                    revision=None,
                    prefix="transformer.",
                    fall_back_to_pt=True,
                )
            )
        if load_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(
            model,
            component_subfolders,
            local_files_only=local_files_only,
        )

        # ``from_pretrained_with_prefetch`` re-prefetches and retries if the
        # cache is still half-written (the missing-shard ``OSError`` and the
        # default-``UMT5Config`` size-mismatch ``RuntimeError`` seen on multi
        # -worker HSDP / ring launches), instead of crashing the worker.
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        # Initialize transformers with correct config (weights loaded via load_weights)
        if load_transformer:
            transformer_config = load_transformer_config(model, "transformer", local_files_only)
            self.transformer = self._create_transformer(transformer_config)
        else:
            self.transformer = None

        if load_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = self._create_transformer(transformer_2_config)
        else:
            self.transformer_2 = None

        # Store the active transformer config
        if load_transformer:
            self.transformer_config = self.transformer.config
        elif load_transformer_2:
            self.transformer_config = self.transformer_2.config
        else:
            raise RuntimeError("No transformer loaded")

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanTransformer3DModel:
        """Create a transformer from a config dict. Respects od_config.quantization_config."""
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(config, quant_config=quant_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def _build_trajectory_probe_state(
        self,
        req: OmniDiffusionRequest,
        timesteps: torch.Tensor,
    ) -> dict[str, Any] | None:
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        probe_cfg = extra_args.get("trajectory_probe")
        if not isinstance(probe_cfg, dict):
            return None

        artifact_dir = probe_cfg.get("artifact_dir")
        if not artifact_dir:
            raise ValueError("trajectory_probe.artifact_dir is required.")

        num_steps = len(timesteps)
        capture_steps_raw = probe_cfg.get("capture_steps")
        capture_progress_raw = probe_cfg.get("capture_progress")
        capture_steps: set[int] = set()

        if isinstance(capture_steps_raw, list) and capture_steps_raw:
            for value in capture_steps_raw:
                step = int(value)
                capture_steps.add(min(max(step, 0), num_steps))
        elif isinstance(capture_progress_raw, list) and capture_progress_raw:
            for value in capture_progress_raw:
                progress = float(value)
                progress = min(max(progress, 0.0), 1.0)
                capture_steps.add(int(round(progress * num_steps)))
        else:
            default_progress = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0]
            for progress in default_progress:
                capture_steps.add(int(round(progress * num_steps)))

        capture_steps.add(0)
        capture_steps.add(num_steps)

        label = str(probe_cfg.get("request_label") or req.request_id)
        fps = float(
            probe_cfg.get("fps")
            or req.sampling_params.resolved_frame_rate
            or req.sampling_params.fps
            or 16.0
        )

        return {
            "artifact_dir": Path(str(artifact_dir)),
            "capture_steps": sorted(capture_steps),
            "capture_steps_set": set(capture_steps),
            "records": [],
            "label": label,
            "fps": fps,
            "save_decoded": bool(probe_cfg.get("save_decoded", True)),
            "save_latents": bool(probe_cfg.get("save_latents", False)),
            "save_mp4": bool(probe_cfg.get("save_mp4", True)),
            "sampling_seed": getattr(req.sampling_params, "seed", None),
            "prompt": req.prompts[0] if req.prompts else None,
            "num_steps": num_steps,
        }

    def _capture_trajectory_probe_checkpoint(
        self,
        probe_state: dict[str, Any] | None,
        *,
        step_index: int,
        timestep: torch.Tensor | None,
        latents: torch.Tensor,
        step_latency_ms: float,
        cumulative_dit_ms: float,
    ) -> None:
        if probe_state is None or step_index not in probe_state["capture_steps_set"]:
            return

        timestep_value: float | None = None
        if timestep is not None:
            timestep_value = float(torch.as_tensor(timestep).detach().float().cpu().reshape(-1)[0].item())

        free_gpu_bytes: int | None = None
        peak_reserved_bytes: int | None = None
        peak_allocated_bytes: int | None = None
        if current_omni_platform.is_available():
            free_gpu_bytes = int(current_omni_platform.get_free_memory(self.device))
            peak_reserved_bytes = int(current_omni_platform.max_memory_reserved())
            peak_allocated_bytes = int(current_omni_platform.max_memory_allocated())

        probe_state["records"].append(
            {
                "step_index": int(step_index),
                "progress": float(step_index / max(probe_state["num_steps"], 1)),
                "timestep": timestep_value,
                "latent_shape": list(latents.shape),
                "latent_dtype": str(latents.dtype),
                "latent_cpu": latents.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                "step_latency_ms": float(step_latency_ms),
                "cumulative_dit_ms": float(cumulative_dit_ms),
                "free_gpu_bytes": free_gpu_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
            }
        )

    def _decode_probe_video_frames(
        self,
        latents: torch.Tensor,
    ) -> np.ndarray:
        decode_latents = latents.to(device=self.device, dtype=self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(decode_latents.device, decode_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            decode_latents.device, decode_latents.dtype
        )
        decode_latents = decode_latents / latents_std + latents_mean
        decoded = self.vae.decode(decode_latents, return_dict=False)[0]
        decoded = decoded.detach().clamp(-1.0, 1.0)
        decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
        decoded = decoded.permute(0, 2, 3, 4, 1).contiguous()
        return decoded[0].cpu().numpy()

    def _persist_trajectory_probe(
        self,
        probe_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if probe_state is None:
            return {}

        artifact_dir = probe_state["artifact_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        saved_records: list[dict[str, Any]] = []

        for record in probe_state["records"]:
            step_index = int(record["step_index"])
            step_prefix = f"{probe_state['label']}_step{step_index:03d}"
            latent_path = artifact_dir / f"{step_prefix}_latents.pt"
            frames_path = artifact_dir / f"{step_prefix}_frames.pt"
            mp4_path = artifact_dir / f"{step_prefix}.mp4"

            if probe_state["save_latents"]:
                torch.save(record["latent_cpu"], latent_path)

            frames_path_str: str | None = None
            mp4_path_str: str | None = None
            if probe_state["save_decoded"]:
                frames_u8 = self._decode_probe_video_frames(record["latent_cpu"])
                torch.save(torch.from_numpy(frames_u8), frames_path)
                frames_path_str = str(frames_path)
                if probe_state["save_mp4"] and _is_rank_zero():
                    mp4_bytes = mux_video_audio_bytes(frames_u8, None, fps=probe_state["fps"])
                    mp4_path.write_bytes(mp4_bytes)
                    mp4_path_str = str(mp4_path)

            saved_records.append(
                {
                    "step_index": step_index,
                    "progress": record["progress"],
                    "timestep": record["timestep"],
                    "latent_shape": record["latent_shape"],
                    "latent_dtype": record["latent_dtype"],
                    "step_latency_ms": record["step_latency_ms"],
                    "cumulative_dit_ms": record["cumulative_dit_ms"],
                    "free_gpu_bytes": record["free_gpu_bytes"],
                    "peak_reserved_bytes": record["peak_reserved_bytes"],
                    "peak_allocated_bytes": record["peak_allocated_bytes"],
                    "latent_path": str(latent_path) if probe_state["save_latents"] else None,
                    "frames_path": frames_path_str,
                    "mp4_path": mp4_path_str,
                }
            )

        metadata = {
            "label": probe_state["label"],
            "num_steps": probe_state["num_steps"],
            "fps": probe_state["fps"],
            "sampling_seed": probe_state["sampling_seed"],
            "records": saved_records,
        }
        metadata_path = artifact_dir / f"{probe_state['label']}_trajectory_probe.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return {
            "trajectory_probe_metadata": metadata,
            "trajectory_probe_metadata_path": str(metadata_path),
            "trajectory_probe_artifact_dir": str(artifact_dir),
        }

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any],
        latent_condition: torch.Tensor | None = None,
        first_frame_mask: torch.Tensor | None = None,
        probe_state: dict[str, Any] | None = None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        cumulative_dit_ms = 0.0
        self._capture_trajectory_probe_checkpoint(
            probe_state,
            step_index=0,
            timestep=timesteps[0] if len(timesteps) > 0 else None,
            latents=latents,
            step_latency_ms=0.0,
            cumulative_dit_ms=0.0,
        )
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                set_forward_context_denoise_step_idx(step_idx)
                step_start = time.perf_counter()

                # Select model based on timestep and boundary_ratio
                # High noise stage (t >= boundary_timestep): use transformer
                # Low noise stage (t < boundary_timestep): use transformer_2
                if boundary_timestep is not None and t < boundary_timestep:
                    # Low noise stage - always use guidance_high for this stage
                    current_guidance_scale = guidance_high
                    if self.transformer_2 is not None:
                        current_model = self.transformer_2
                    elif self.transformer is not None:
                        # Fallback to transformer if transformer_2 not loaded
                        current_model = self.transformer
                    else:
                        raise RuntimeError("No transformer available for low-noise stage")
                else:
                    # High noise stage - always use guidance_low for this stage
                    current_guidance_scale = guidance_low
                    if self.transformer is not None:
                        current_model = self.transformer
                    elif self.transformer_2 is not None:
                        # Fallback to transformer_2 if transformer not loaded
                        current_model = self.transformer_2
                    else:
                        raise RuntimeError("No transformer available for high-noise stage")

                if self.expand_timesteps and latent_condition is not None:
                    # I2V mode: blend condition with latents using mask
                    latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(dtype)

                    # Expand timesteps per patch - use floor division to match patch embedding
                    patch_size = self.transformer_config.patch_size
                    patch_height = latents.shape[3] // patch_size[1]
                    patch_width = latents.shape[4] // patch_size[2]

                    # Create mask at patch resolution (same as hidden states sequence length)
                    patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                    patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]  # Ensure correct dimensions
                    temp_ts = (patch_mask[0][0] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # T2V mode: standard forward
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])

                do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": current_model,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                if current_omni_platform.is_available():
                    current_omni_platform.synchronize()
                step_latency_ms = (time.perf_counter() - step_start) * 1000.0
                cumulative_dit_ms += step_latency_ms
                self._capture_trajectory_probe_checkpoint(
                    probe_state,
                    step_index=step_idx + 1,
                    timestep=t,
                    latents=latents,
                    step_latency_ms=step_latency_ms,
                    cumulative_dit_ms=cumulative_dit_ms,
                )
                pbar.update()

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float | tuple[float, float] = 4.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Get parameters from request or arguments
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Wan2.2 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # Respect per-request guidance_scale when explicitly provided.
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        # record guidance for properties
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Prefer engine-configured boundary_ratio, but allow per-request fallback.
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else req.sampling_params.boundary_ratio

        if boundary_ratio is None:
            boundary_ratio = 0.875
            logger.warning("boundary_ratio is required for T2V generation. using default value 0.875")

        # validate shapes
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale_2=guidance_high if boundary_ratio is not None else None,
            boundary_ratio=boundary_ratio,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            # Fallback to text_encoder dtype if no transformer loaded
            dtype = self.text_encoder.dtype

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        if DEBUG_PERF:
            # Sync GPU before timing to ensure accurate measurements
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif guidance_low > 1.0 or guidance_high > 1.0:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are given and guidance > 1."
                )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        resume_step_index = int(req.sampling_params.step_index or 0)
        if resume_step_index < 0 or resume_step_index >= num_steps:
            raise ValueError(f"Resume step_index must be in [0, {num_steps - 1}], got {resume_step_index}.")
        if resume_step_index > 0 and req.sampling_params.latents is None:
            raise ValueError("Resuming Wan generation requires sampling.latents to be populated.")
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(resume_step_index)
        if resume_step_index > 0:
            timesteps = timesteps[resume_step_index:]

        self._num_timesteps = len(timesteps)
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()
        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else None
        raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
        if isinstance(raw_image, list):
            if len(raw_image) > 1:
                logger.warning(
                    """Received a list of image. Only a single image is supported by this model."""
                    """Taking only the first image for now."""
                )
            raw_image = raw_image[0]
        if raw_image is None:
            image = None
        elif isinstance(raw_image, str):
            image = PIL.Image.open(raw_image)
        else:
            image = cast(PIL.Image.Image | torch.Tensor, raw_image)

        latent_condition = None
        first_frame_mask = None

        if self.expand_timesteps and image is not None:
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            # Preprocess image
            if isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels
            batch_size = prompt_embeds.shape[0]

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=prompt_embeds.shape[0],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=req.sampling_params.latents,
            )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if attention_kwargs is None:
            attention_kwargs = {}

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        probe_state = None
        if not (self.expand_timesteps and latent_condition is not None):
            probe_state = self._build_trajectory_probe_state(req, timesteps)
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
            probe_state=probe_state,
        )

        # Wan2.2 is prone to out of memory errors when predicting large videos
        # so we empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        # For I2V mode: blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms

            if _is_rank_zero():
                logger.info(
                    "Pipeline stage timing summary: "
                    "TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, "
                    "StagesSum=%.2f ms, PipelineWall=%.2f ms, Unaccounted=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                    _t_pipeline_wall_ms - _t_stages_sum,
                )

        custom_output = self._persist_trajectory_probe(probe_state)
        return DiffusionOutput(
            output=output,
            custom_output=custom_output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        return result if isinstance(result, IntermediateTensors) else result[0]

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        guidance_scale_2=None,
        boundary_ratio=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")


# ---------------------------------------------------------------------------
# DMD2-distilled variant
# ---------------------------------------------------------------------------


class WanT2VDMD2Pipeline(DMD2PipelineMixin, Wan22Pipeline):
    """Wan 2.x T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
