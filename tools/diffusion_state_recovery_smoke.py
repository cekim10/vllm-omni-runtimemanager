#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test diffusion checkpoint capture, abort, restore, and resume."
    )
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Diffusion model name or local path.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Optional Omni stage config YAML.")
    parser.add_argument("--prompt", default="A brass astrolabe on a wooden desk", help="Text prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic generation seed.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Total denoising steps.")
    parser.add_argument("--failure-step", type=int, default=10, help="Abort after checkpointing this step index.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/diffusion-state-smoke"),
        help="Directory for baseline/resumed outputs and summary JSON.",
    )
    parser.add_argument(
        "--disk-path",
        type=Path,
        default=Path("/tmp/diffusion-state-smoke/checkpoints"),
        help="Checkpoint spill directory for the diffusion state manager.",
    )
    parser.add_argument("--gpu-budget-bytes", type=int, default=0, help="State manager GPU tier budget.")
    parser.add_argument("--cpu-budget-bytes", type=int, default=0, help="State manager CPU tier budget.")
    parser.add_argument("--theta-h", type=float, default=0.7, help="LOSSLESS threshold.")
    parser.add_argument("--theta-w", type=float, default=0.3, help="COMPRESSED threshold.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Omni init timeout in seconds.")
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Per-stage init timeout in seconds.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile for easier debugging.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Polling interval in seconds while waiting for the target checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds while waiting for the target checkpoint step.",
    )
    parser.add_argument(
        "--strict-equality",
        action="store_true",
        help="Fail if the resumed output is not pixel-identical to the baseline output.",
    )
    return parser.parse_args()


def _build_sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=1,
    )
    sampling.extra_args = {
        "cfg_scale": args.true_cfg_scale,
        "cfg_text_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
    }
    return sampling


def _build_prompt(omni: Omni, args: argparse.Namespace) -> Any:
    model_class_name = get_model_class_name(omni)
    return build_text_to_image_prompt(
        model_class_name=model_class_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
    )


def _make_request(
    request_id: str,
    prompt: Any,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[copy.deepcopy(prompt)],
        request_id=request_id,
        sampling_params=copy.deepcopy(sampling_params),
    )


def _get_inline_diffusion_engine(omni: Omni) -> DiffusionEngine:
    stage_clients = getattr(omni.engine, "stage_clients", [])
    diffusion_clients = [client for client in stage_clients if getattr(client, "stage_type", None) == "diffusion"]
    if len(diffusion_clients) != 1:
        raise RuntimeError(
            f"Expected exactly one diffusion stage client, found {len(diffusion_clients)}. "
            "This smoke test currently targets single-stage diffusion setups."
        )

    client = diffusion_clients[0]
    engine = getattr(client, "_engine", None)
    if engine is None:
        raise RuntimeError(
            "The diffusion stage is not running inline, so the internal DiffusionEngine is not directly accessible. "
            "Use a single-stage local diffusion configuration for this smoke test."
        )
    if not isinstance(engine, DiffusionEngine):
        raise TypeError(f"Unexpected inline diffusion engine type: {type(engine)!r}")
    if getattr(engine, "state_manager", None) is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")
    return engine


async def _wait_for_checkpoint(
    engine: DiffusionEngine,
    request_id: str,
    target_step: int,
    poll_interval: float,
    timeout_s: float,
) -> Any:
    start = time.monotonic()
    state_manager = engine.state_manager
    assert state_manager is not None
    last_state = None
    while True:
        last_state = state_manager.on_failure(request_id)
        if last_state is not None and last_state.step_idx >= target_step:
            return last_state
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for checkpoint step>={target_step}; "
                f"last_seen_step={getattr(last_state, 'step_idx', None)}"
            )
        await asyncio.sleep(poll_interval)


def _save_first_image(outputs: list[Any], path: Path) -> None:
    if not outputs or not outputs[0].images:
        raise RuntimeError("No image output produced.")
    outputs[0].images[0].save(path)


def _image_metrics(lhs_path: Path, rhs_path: Path) -> dict[str, Any]:
    from PIL import Image

    lhs = np.asarray(Image.open(lhs_path).convert("RGB"), dtype=np.float32)
    rhs = np.asarray(Image.open(rhs_path).convert("RGB"), dtype=np.float32)
    diff = lhs - rhs
    mse = float(np.mean(diff * diff))
    max_abs = float(np.max(np.abs(diff)))
    identical = bool(np.array_equal(lhs, rhs))
    return {
        "pixel_identical": identical,
        "mse": mse,
        "max_abs_diff": max_abs,
    }


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.failure_step < 0 or args.failure_step >= args.num_inference_steps:
        raise ValueError(
            f"--failure-step must be in [0, {args.num_inference_steps - 1}], got {args.failure_step}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.disk_path.mkdir(parents=True, exist_ok=True)

    omni_kwargs = {
        "model": args.model,
        "mode": "text-to-image",
        "step_execution": True,
        "enable_diffusion_state_manager": True,
        "diffusion_state_manager_gpu_budget_bytes": args.gpu_budget_bytes,
        "diffusion_state_manager_cpu_budget_bytes": args.cpu_budget_bytes,
        "diffusion_state_manager_theta_h": args.theta_h,
        "diffusion_state_manager_theta_w": args.theta_w,
        "diffusion_state_manager_disk_path": str(args.disk_path),
        "tensor_parallel_size": args.tensor_parallel_size,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "enforce_eager": args.enforce_eager,
    }
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path

    omni = Omni(**omni_kwargs)
    try:
        prompt = _build_prompt(omni, args)
        sampling_params = _build_sampling_params(args)
        engine = _get_inline_diffusion_engine(omni)

        initial_request = _make_request("recovery-smoke", prompt, sampling_params)
        await engine._check_and_start_background_loop()
        run_task = asyncio.create_task(engine.async_add_req_and_wait_for_response(initial_request))

        checkpoint_state = await _wait_for_checkpoint(
            engine,
            request_id=initial_request.request_id,
            target_step=args.failure_step,
            poll_interval=args.poll_interval,
            timeout_s=args.checkpoint_timeout,
        )

        engine.abort(initial_request.request_id)
        aborted_output = await run_task
        if not aborted_output.aborted:
            raise RuntimeError(f"Expected aborted output after injected failure, got: {aborted_output!r}")

        resume_template = _make_request(initial_request.request_id, prompt, sampling_params)
        resumed_request = engine.restore_request_from_state(resume_template, request_id=initial_request.request_id)
        resumed_outputs = await engine.step(resumed_request)
        resumed_path = args.output_dir / "resumed.png"
        _save_first_image(resumed_outputs, resumed_path)

        baseline_request = _make_request("baseline-smoke", prompt, sampling_params)
        baseline_outputs = await engine.step(baseline_request)
        baseline_path = args.output_dir / "baseline.png"
        _save_first_image(baseline_outputs, baseline_path)

        metrics = _image_metrics(baseline_path, resumed_path)
        if args.strict_equality and not metrics["pixel_identical"]:
            raise RuntimeError(f"Resumed output diverged from baseline: {metrics}")

        summary = {
            "model": args.model,
            "prompt": args.prompt,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "failure_step": args.failure_step,
            "checkpoint": {
                "step_idx": checkpoint_state.step_idx,
                "total_steps": checkpoint_state.total_steps,
                "fidelity": checkpoint_state.fidelity.value,
                "placement": checkpoint_state.placement.value,
                "value_score": checkpoint_state.value_score,
                "size_bytes": checkpoint_state.size_bytes,
                "disk_path": checkpoint_state.disk_path,
            },
            "aborted": {
                "aborted": aborted_output.aborted,
                "abort_message": aborted_output.abort_message,
            },
            "restore": {
                "restored_step_index": resumed_request.sampling_params.step_index,
                "restored_latents_shape": list(resumed_request.sampling_params.latents.shape),
            },
            "comparison": metrics,
            "artifacts": {
                "baseline_image": str(baseline_path),
                "resumed_image": str(resumed_path),
                "checkpoint_dir": str(args.disk_path),
            },
        }

        summary_path = args.output_dir / "summary.json"
        summary["artifacts"]["summary_json"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        omni.close()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_smoke(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
