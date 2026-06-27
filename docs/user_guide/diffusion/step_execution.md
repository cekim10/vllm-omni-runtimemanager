# Step Execution

Step execution is an opt-in diffusion execution mode enabled with
`step_execution=True` when constructing `Omni`.

It is not a generic diffusion toggle for every pipeline. Only pipelines that
implement the stepwise contract support it today.

## Quick Start

### Python API

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    step_execution=True,
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
    ),
)
```

### Serving

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

For serving, `--step-execution` enables the step-wise runtime. Continuous
batching only becomes relevant when `--max-num-seqs > 1`.

## Supported Pipelines

| Pipeline | Example models | Step execution |
|----------|----------------|----------------|
| `QwenImagePipeline` | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes |
| All other diffusion pipelines | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`, `QwenImageLayeredPipeline`, GLM-Image, Wan, Flux, etc. | No |

!!! warning "Experimental continuous batching"
    When `--step-execution` is enabled and `max_num_seqs > 1` is configured,
    the step-wise path can batch
    compatible requests together. This is experimental. Requests with
    incompatible sampling parameters are intentionally kept in separate batches,
    and `max_num_seqs=1` remains the conservative default.

## Current Limitations

- Continuous batching under `step_execution` is experimental and only batches
  compatible requests.
- `cache_backend` is not supported together with step execution.
- Unsupported pipelines fail early during model loading.
- Request-mode extras such as KV transfer are not wired into step mode yet.
- LoRA is supported in step mode, but each batch must use a single adapter:
  requests with different `lora_request` or `lora_scale` are scheduled into
  separate batches.

## When To Use It

Use step execution only when you specifically need the pipeline to run through
its stepwise request state machine. For normal diffusion inference, leave it
disabled unless your workflow depends on this mode.

For Qwen-Image online serving, the usual progression is:

- start with `--step-execution --max-num-seqs 1` if you only need the step-wise path
- increase `--max-num-seqs` after that if you want the experimental compatible-request batching behavior

If you are looking for general diffusion speedups, see
[Diffusion Features Overview](../diffusion_features.md).

## Troubleshooting

If model loading fails with a message mentioning `prepare_encode()`,
`denoise_step()`, `step_scheduler()`, and `post_decode()`, the selected
pipeline does not support step execution.

## Recovery Smoke Test

The diffusion state manager is built on top of step execution. To run a local
smoke test that checkpoints a request, aborts it mid-denoise, restores from the
captured `x_t`, and compares the resumed output against a baseline run:

```bash
python tools/diffusion_state_recovery_smoke.py \
  --prompt "A brass astrolabe on a wooden desk" \
  --num-inference-steps 20 \
  --failure-step 10 \
  --output-dir /tmp/diffusion-state-smoke
```

By default this uses a lightweight runtime stub backend, so it exercises the
checkpoint/abort/restore/resume control path without loading a large model.
This is the recommended smoke path for single-GPU development and server bringup.

If you explicitly want to run the same smoke against a real step-execution
model, switch to the real-model backend:

```bash
python tools/diffusion_state_recovery_smoke.py \
  --backend real-model \
  --model Qwen/Qwen-Image \
  --prompt "A brass astrolabe on a wooden desk" \
  --num-inference-steps 20 \
  --failure-step 10 \
  --output-dir /tmp/diffusion-state-smoke
```

On a single 48GB GPU, `Qwen/Qwen-Image` may initialize close to the memory
limit. The helper enables CPU and layerwise offload automatically for that
configuration, and still retries with offload after a CUDA OOM if needed.

Small text-to-image models such as `Tongyi-MAI/Z-Image-Turbo` and
`stabilityai/stable-diffusion-3.5-medium` are valid vLLM-Omni models, but they
do not currently support `step_execution=True`. That means they cannot be used
with the recovery smoke test's real-model backend yet. For single-GPU
development on those models, keep using the default `--backend stub` path.

## For Model Authors

If you want to add step execution support to a new diffusion pipeline, see the
implementation guide:
[Diffusion Step Execution Design](../../design/feature/diffusion_step_execution.md).

If you also want that pipeline to participate in the experimental batched
step-wise path, see:
[Continuous Batching for Step-Wise Diffusion](../../design/feature/diffusion_continuous_batching.md).
