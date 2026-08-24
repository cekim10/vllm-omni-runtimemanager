# Repository Map: Temporal-Dimension Kill Test

This map documents the existing video-generation path in `/Users/cekim/Desktop/git/vllm-omni-runtimemanager` for the initial temporal-structure kill test. It is intentionally narrow and only covers the files needed for trajectory instrumentation and preflight analysis.

## Primary model choice

- Primary preflight model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Why this model:
  - already supported in the native diffusion stack,
  - has an existing offline and online video-serving path,
  - exposes a clear text-to-video denoising loop,
  - decodes to full video through the normal VAE path,
  - avoids the extra conditioning complexity of image-to-video for the first kill test.

## Supported video diffusion models

- `Wan22Pipeline` / `WanT2VDMD2Pipeline`
  - File: `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
  - Public T2V path with existing examples.
- `WanI2VPipeline`
  - File: `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py`
  - Image-to-video variant; not the first preflight target.
- `WanS2VPipeline`
  - File: `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_s2v.py`
  - Speech/audio-conditioned video; more complex autoregressive clip loop.
- `HunyuanVideo15Pipeline`
  - File: `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py`
  - Supported, but no native step-protocol path and heavier baseline config.
- `HeliosPipeline`
  - File: `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/helios/pipeline_helios.py`
  - Has step-execution support, but Wan2.2 is the safer first baseline because the repo already exposes public serving examples for it.

## Current video generation entry points

- Offline text-to-video example:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/examples/offline_inference/text_to_video/text_to_video.py`
- Online video-serving API:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/entrypoints/openai/serving_video.py`
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/examples/online_serving/image_to_video/README.md`
- Generic synchronous user-facing API:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/entrypoints/omni.py`
  - `Omni.generate(...)` returns `OmniRequestOutput`.

## Engine and denoising execution path

- Diffusion engine:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/diffusion_engine.py`
  - Handles request lifecycle, scheduling, postprocess formatting, and worker communication.
- Worker/model runner:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/worker/diffusion_worker.py`
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/worker/diffusion_model_runner.py`
  - Non-stepwise mode calls `pipeline.forward(...)`.
  - Stepwise mode exists for some image/video pipelines, but not for Wan2.2 T2V.
- Scheduler:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/sched/request_scheduler.py`
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/sched/step_scheduler.py`

## Denoising loop implementation

- For the chosen model, the hot denoising loop is:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
  - `Wan22Pipeline.diffuse(...)`
- This is the correct instrumentation point for:
  - per-step latency,
  - selected-timestep latent capture,
  - cumulative DiT-time accounting.

## Latent tensor layout

- Wan2.2 latent preparation:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
  - `prepare_latents(...)`
- T2V latent layout is:
  - `[B, C, T_lat, H_lat, W_lat]`
- Temporal compression:
  - `T_lat = (num_frames - 1) // vae_scale_factor_temporal + 1`
- Spatial compression:
  - `H_lat = height // vae_scale_factor_spatial`
  - `W_lat = width // vae_scale_factor_spatial`

## Temporal / spatial token organization

- Wan2.2 uses a 3D DiT-style latent tensor where temporal and spatial positions are jointly represented in the latent volume.
- Temporal positions persist throughout denoising; there is no separate late-stage temporal shutdown in the current implementation.
- In `expand_timesteps` mode, image-conditioned requests expand timestep values per patch, but the first preflight should stay in pure T2V mode to avoid this complication.

## Attention implementation

- Backend selection and platform defaults:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/data.py`
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/platforms/cuda/platform.py`
- Wan2.2 transformer:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py`
- Attention backend is runtime-selectable, but the kill test should not tune this. Just record the active backend from the normal run configuration.

## Scheduler / timestep representation

- Wan2.2 scheduler construction:
  - `build_wan_scheduler(...)` in `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
- Supported samplers:
  - `unipc`
  - `euler`
- Timesteps are stored in `self.scheduler.timesteps` and iterated directly in `diffuse(...)`.
- Boundary split for two-stage Wan models is controlled by `boundary_ratio`; for the preflight, this is part of the baseline configuration and should simply be recorded.

## VAE decode path

- Final decode path:
  - `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`
  - inside `forward(...)`, after denoising:
    - denormalize latents using `latents_mean` and `latents_std`
    - `self.vae.decode(latents, return_dict=False)[0]`
- Video postprocess:
  - `get_wan22_post_process_func(...)` in the same file.

## Can intermediate latents be decoded?

- Yes, for Wan2.2 T2V latents.
- The same final-latent decode path can be applied to saved intermediate latents as long as they are in the model’s latent space and denormalized with the same VAE config.
- For the preflight study, this is now exposed through a measurement-only `trajectory_probe` hook in `Wan22Pipeline`.

## Safe instrumentation points

- Best hook for per-step timing and latent capture:
  - `Wan22Pipeline.diffuse(...)`
- Best hook for writing measurement artifacts:
  - `Wan22Pipeline.forward(...)` after denoising, before returning `DiffusionOutput`
- Existing result transport path:
  - `DiffusionOutput.custom_output`
  - formatted through `/Users/cekim/Desktop/git/vllm-omni-runtimemanager/vllm_omni/diffusion/output_formatter.py`

## Why not use step-execution first?

- Step-execution support currently exists for image pipelines such as:
  - Z-Image
  - Qwen-Image
  - HunyuanImage3
  - Helios
- Wan2.2 T2V does not currently implement the step protocol.
- For this kill test, a narrow measurement-only hook inside the real Wan2.2 forward path is less invasive than porting Wan2.2 to the full step interface before we know the opportunity exists.
