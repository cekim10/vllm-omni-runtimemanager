#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_VARIANTS = [
    "full",
    "fp16",
    "int8",
    "int4",
    "spatial_down2",
    "temporal_down2",
    "low_rank_25",
]


CSV_FIELDS = [
    "model",
    "prompt_id",
    "category",
    "prompt",
    "seed",
    "checkpoint_step",
    "total_steps",
    "variant",
    "retained_bytes",
    "retained_ratio",
    "checkpoint_size_bytes",
    "checkpoint_value_score",
    "resume_latency_ms",
    "exact_equal",
    "video_mse",
    "max_abs_diff",
    "spatial_metric",
    "temporal_shape_composite",
    "temporal_dynamic_composite",
    "temporal_metric_1",
    "temporal_metric_2",
    "motion_metric",
    "motion_energy_ratio",
    "flow_magnitude_cosine",
    "flow_magnitude_ratio",
    "flow_direction_cosine",
    "flicker_similarity",
    "semantic_metric_raw",
    "final_semantic_metric_raw",
    "semantic_metric",
    "artifact_path",
    "variant_metadata_json",
]


SUMMARY_FIELDS = [
    "checkpoint_step",
    "variant",
    "num_rows",
    "mean_retained_ratio",
    "mean_resume_latency_ms",
    "mean_spatial_metric",
    "mean_temporal_shape_composite",
    "mean_temporal_dynamic_composite",
    "mean_semantic_metric",
    "mean_video_mse",
    "exact_equal_rate",
]


@dataclass
class PerturbedState:
    restored_latent: torch.Tensor
    retained_bytes: int
    metadata: dict[str, Any]


def _preflight_helpers() -> Any:
    from experiments import temporal_dimension_killtest_preflight as preflight

    return preflight


def _wait_for_checkpoint_helper() -> Any:
    from tools.diffusion_state_recovery_smoke import _wait_for_checkpoint

    return _wait_for_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct video diffusion recovery frontier: perturb checkpoint state, resume, and compare final videos."
    )
    parser.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument(
        "--prompt-set",
        default="experiments/temporal_dimension_prompt_set.json",
        help="Prompt JSON with prompt_id, prompt, motion_category.",
    )
    parser.add_argument("--output-dir", default="results/video_recovery_frontier")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[10, 20, 30])
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--checkpoint-timeout", type=float, default=600.0)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help=f"Perturbation variants. Default: {' '.join(DEFAULT_VARIANTS)}",
    )
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument(
        "--disk-path",
        type=Path,
        default=None,
        help="Optional state-manager spill directory. Defaults under output-dir/checkpoints.",
    )
    return parser.parse_args()


def _make_sampling_params(args: argparse.Namespace, seed: int) -> Any:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def _make_request(
    request_id: str,
    prompt: dict[str, Any],
    sampling_params: Any,
) -> Any:
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    return OmniDiffusionRequest(
        prompts=[copy.deepcopy(prompt)],
        request_id=request_id,
        sampling_params=copy.deepcopy(sampling_params),
    )


def _unwrap_video_outputs(outputs: Any) -> np.ndarray:
    from vllm_omni.outputs import OmniRequestOutput

    output = OmniRequestOutput.unwrap_result(outputs)
    return _preflight_helpers()._normalize_frames(output)


def _video_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    lhs_f = lhs.astype(np.float32)
    rhs_f = rhs.astype(np.float32)
    diff = lhs_f - rhs_f
    return {
        "exact_equal": bool(np.array_equal(lhs, rhs)),
        "video_mse": float(np.mean(diff * diff)),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _full_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.nelement() * tensor.element_size())


def _quantize_restore(
    latent: torch.Tensor,
    *,
    levels: int,
    packed_bits: int,
) -> PerturbedState:
    tensor = latent.detach().to(torch.float32).cpu().contiguous()
    max_q = float(levels)
    scale = tensor.abs().max().clamp_min(1e-8) / max_q
    quantized = torch.clamp(torch.round(tensor / scale), -max_q, max_q).to(torch.int8)
    restored = (quantized.to(torch.float32) * scale).to(dtype=latent.dtype)
    payload_bytes = int(math.ceil(tensor.numel() * packed_bits / 8))
    scale_bytes = int(scale.nelement() * scale.element_size())
    return PerturbedState(
        restored_latent=restored,
        retained_bytes=payload_bytes + scale_bytes,
        metadata={
            "type": f"int{packed_bits}",
            "scale_bytes": scale_bytes,
            "payload_bytes": payload_bytes,
        },
    )


def _spatial_downsample_restore(latent: torch.Tensor, factor: int) -> PerturbedState:
    tensor = latent.detach().to(torch.float32).cpu().contiguous()
    original_shape = tuple(tensor.shape)
    if tensor.ndim == 5:
        _, _, t_dim, h_dim, w_dim = tensor.shape
        down = F.interpolate(
            tensor,
            size=(t_dim, max(1, h_dim // factor), max(1, w_dim // factor)),
            mode="trilinear",
            align_corners=False,
        )
        restored = F.interpolate(
            down,
            size=(t_dim, h_dim, w_dim),
            mode="trilinear",
            align_corners=False,
        )
    elif tensor.ndim == 4:
        _, _, h_dim, w_dim = tensor.shape
        down = F.interpolate(
            tensor,
            size=(max(1, h_dim // factor), max(1, w_dim // factor)),
            mode="bilinear",
            align_corners=False,
        )
        restored = F.interpolate(
            down,
            size=(h_dim, w_dim),
            mode="bilinear",
            align_corners=False,
        )
    else:
        down = tensor
        restored = tensor
    return PerturbedState(
        restored_latent=restored.to(dtype=latent.dtype),
        retained_bytes=_full_bytes(down),
        metadata={
            "type": "spatial_downsample",
            "factor": factor,
            "stored_shape": list(down.shape),
            "original_shape": list(original_shape),
        },
    )


def _temporal_downsample_restore(latent: torch.Tensor, factor: int) -> PerturbedState:
    tensor = latent.detach().to(torch.float32).cpu().contiguous()
    original_shape = tuple(tensor.shape)
    if tensor.ndim == 5 and tensor.shape[2] > 1:
        _, _, t_dim, h_dim, w_dim = tensor.shape
        down = F.interpolate(
            tensor,
            size=(max(1, math.ceil(t_dim / factor)), h_dim, w_dim),
            mode="trilinear",
            align_corners=False,
        )
        restored = F.interpolate(
            down,
            size=(t_dim, h_dim, w_dim),
            mode="trilinear",
            align_corners=False,
        )
    else:
        down = tensor
        restored = tensor
    return PerturbedState(
        restored_latent=restored.to(dtype=latent.dtype),
        retained_bytes=_full_bytes(down),
        metadata={
            "type": "temporal_downsample",
            "factor": factor,
            "stored_shape": list(down.shape),
            "original_shape": list(original_shape),
        },
    )


def _low_rank_restore(latent: torch.Tensor, ratio: float) -> PerturbedState:
    tensor = latent.detach().to(torch.float32).cpu().contiguous()
    original_shape = tuple(tensor.shape)
    if tensor.ndim == 5:
        bsz, channels, t_dim, h_dim, w_dim = tensor.shape
        matrix = tensor.reshape(bsz * channels * t_dim, h_dim * w_dim)
    elif tensor.ndim == 4:
        bsz, channels, h_dim, w_dim = tensor.shape
        matrix = tensor.reshape(bsz * channels, h_dim * w_dim)
    else:
        matrix = tensor.reshape(tensor.shape[0], -1)

    rank = max(1, min(min(matrix.shape), int(round(min(matrix.shape) * ratio))))
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u_k = u[:, :rank].contiguous()
    s_k = s[:rank].contiguous()
    vh_k = vh[:rank, :].contiguous()
    reconstructed = (u_k * s_k.unsqueeze(0)) @ vh_k
    restored = reconstructed.reshape(original_shape).to(dtype=latent.dtype)
    retained_bytes = _full_bytes(u_k) + _full_bytes(s_k) + _full_bytes(vh_k)
    return PerturbedState(
        restored_latent=restored,
        retained_bytes=retained_bytes,
        metadata={
            "type": "low_rank",
            "rank": rank,
            "ratio": ratio,
            "matrix_shape": list(matrix.shape),
        },
    )


def _apply_variant(latent: torch.Tensor, variant: str) -> PerturbedState:
    if variant == "full":
        restored = latent.detach().cpu().clone()
        return PerturbedState(
            restored_latent=restored,
            retained_bytes=_full_bytes(restored),
            metadata={"type": "full"},
        )
    if variant == "fp16":
        packed = latent.detach().cpu().to(torch.float16).contiguous()
        return PerturbedState(
            restored_latent=packed.to(dtype=latent.dtype),
            retained_bytes=_full_bytes(packed),
            metadata={"type": "fp16"},
        )
    if variant == "int8":
        return _quantize_restore(latent, levels=127, packed_bits=8)
    if variant == "int4":
        return _quantize_restore(latent, levels=7, packed_bits=4)
    if variant == "spatial_down2":
        return _spatial_downsample_restore(latent, factor=2)
    if variant == "temporal_down2":
        return _temporal_downsample_restore(latent, factor=2)
    if variant == "low_rank_25":
        return _low_rank_restore(latent, ratio=0.25)
    raise ValueError(f"Unsupported variant: {variant}")


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["checkpoint_step"]), str(row["variant"])), []).append(row)

    summary_rows = []
    for (checkpoint_step, variant), group_rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "variant": variant,
                "num_rows": len(group_rows),
                "mean_retained_ratio": float(np.mean([float(row["retained_ratio"]) for row in group_rows])),
                "mean_resume_latency_ms": float(np.mean([float(row["resume_latency_ms"]) for row in group_rows])),
                "mean_spatial_metric": float(np.mean([float(row["spatial_metric"]) for row in group_rows])),
                "mean_temporal_shape_composite": float(
                    np.mean([float(row["temporal_shape_composite"]) for row in group_rows])
                ),
                "mean_temporal_dynamic_composite": float(
                    np.mean([float(row["temporal_dynamic_composite"]) for row in group_rows])
                ),
                "mean_semantic_metric": float(np.mean([float(row["semantic_metric"]) for row in group_rows])),
                "mean_video_mse": float(np.mean([float(row["video_mse"]) for row in group_rows])),
                "exact_equal_rate": float(np.mean([1.0 if row["exact_equal"] else 0.0 for row in group_rows])),
            }
        )
    return summary_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    best_dynamic = sorted(
        aggregate_rows,
        key=lambda row: (-float(row["mean_temporal_dynamic_composite"]), float(row["mean_retained_ratio"])),
    )[:5]
    best_semantic = sorted(
        aggregate_rows,
        key=lambda row: (-float(row["mean_semantic_metric"]), float(row["mean_retained_ratio"])),
    )[:5]

    lines = [
        "# Video Recovery Frontier",
        "",
        f"- Model: `{args.model}`",
        f"- Resolution: `{args.width}x{args.height}`",
        f"- Frames: `{args.num_frames}`",
        f"- Denoising steps: `{args.num_inference_steps}`",
        f"- Checkpoint steps: `{args.checkpoint_steps}`",
        f"- Variants: `{' '.join(args.variants)}`",
        "",
        "## Best Dynamic Retention",
        "",
    ]
    for row in best_dynamic:
        lines.append(
            f"- step `{row['checkpoint_step']}` / `{row['variant']}`:"
            f" dynamic=`{float(row['mean_temporal_dynamic_composite']):.4f}`"
            f", spatial=`{float(row['mean_spatial_metric']):.4f}`"
            f", semantic=`{float(row['mean_semantic_metric']):.4f}`"
            f", retained=`{float(row['mean_retained_ratio']):.4f}`"
        )
    lines.extend(["", "## Best Semantic Retention", ""])
    for row in best_semantic:
        lines.append(
            f"- step `{row['checkpoint_step']}` / `{row['variant']}`:"
            f" semantic=`{float(row['mean_semantic_metric']):.4f}`"
            f", dynamic=`{float(row['mean_temporal_dynamic_composite']):.4f}`"
            f", retained=`{float(row['mean_retained_ratio']):.4f}`"
        )
    lines.extend(["", f"Rows collected: `{len(rows)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_omni(args: argparse.Namespace, disk_path: Path) -> Omni:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    return Omni(
        model=args.model,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        parallel_config=DiffusionParallelConfig(),
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
        step_execution=True,
        enable_diffusion_state_manager=True,
        diffusion_state_manager_gpu_budget_bytes=0,
        diffusion_state_manager_cpu_budget_bytes=0,
        diffusion_state_manager_theta_h=0.0,
        diffusion_state_manager_theta_w=0.0,
        diffusion_state_manager_disk_path=str(disk_path),
    )


async def _run_baseline_video(
    engine: Any,
    prompt: dict[str, Any],
    sampling_params: OmniDiffusionSamplingParams,
    artifact_dir: Path,
) -> tuple[np.ndarray, Path, float]:
    request = _make_request("baseline-frontier", prompt, sampling_params)
    start = time.perf_counter()
    outputs = await engine.step(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    video = _unwrap_video_outputs(outputs)
    output_path = artifact_dir / "baseline.mp4"
    _save_video(output_path, video, fps=float(sampling_params.fps or 16.0))
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is not None:
        state_manager.release_request(request.request_id)
    return video, output_path, latency_ms


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disk_path = args.disk_path or (output_dir / "checkpoints")
    disk_path.mkdir(parents=True, exist_ok=True)

    preflight = _preflight_helpers()
    wait_for_checkpoint = _wait_for_checkpoint_helper()

    prompt_entries = preflight._read_prompt_set(Path(args.prompt_set), args.num_prompts)
    semantic_evaluator = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )

    omni = _make_omni(args, disk_path)
    rows: list[dict[str, Any]] = []
    try:
        stage_clients = getattr(omni.engine, "stage_clients", [])
        diffusion_clients = [client for client in stage_clients if getattr(client, "stage_type", None) == "diffusion"]
        if len(diffusion_clients) != 1 or getattr(diffusion_clients[0], "_engine", None) is None:
            raise RuntimeError("Expected a single inline diffusion stage for frontier experiment.")
        engine = diffusion_clients[0]._engine
        state_manager = getattr(engine, "state_manager", None)
        if state_manager is None:
            raise RuntimeError("Diffusion state manager is disabled; cannot capture frontier checkpoints.")

        for prompt_idx, prompt_entry in enumerate(prompt_entries):
            prompt_id = str(prompt_entry["prompt_id"])
            category = str(prompt_entry["motion_category"])
            prompt_text = str(prompt_entry["prompt"])
            prompt = {"prompt": prompt_text}
            seed = args.seed + prompt_idx
            artifact_dir = output_dir / "artifacts" / f"{prompt_id}_seed{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            sampling_params = _make_sampling_params(args, seed)
            baseline_video, baseline_path, baseline_latency_ms = await _run_baseline_video(
                engine,
                prompt,
                sampling_params,
                artifact_dir,
            )
            final_semantic_metric_raw = (
                semantic_evaluator.score_video(prompt_text, baseline_video)
                if semantic_evaluator is not None
                else float("nan")
            )

            for checkpoint_step in args.checkpoint_steps:
                request_id = f"frontier-{prompt_id}-step{checkpoint_step}"
                request = _make_request(request_id, prompt, sampling_params)
                run_task = asyncio.create_task(engine.async_add_req_and_wait_for_response(request))
                checkpoint_state = await wait_for_checkpoint(
                    engine,
                    request_id=request_id,
                    target_step=checkpoint_step,
                    poll_interval=args.poll_interval,
                    timeout_s=args.checkpoint_timeout,
                )
                exact_latent = state_manager.restore(checkpoint_state).detach().cpu().clone()
                checkpoint_size_bytes = int(checkpoint_state.size_bytes)
                checkpoint_value_score = float(checkpoint_state.value_score)
                full_state_bytes = _full_bytes(exact_latent)

                engine.abort(request_id)
                aborted_output = await run_task
                if not aborted_output.aborted:
                    raise RuntimeError(f"Expected aborted output for {request_id}, got {aborted_output!r}")
                state_manager.release_request(request_id)

                for variant in args.variants:
                    perturb = _apply_variant(exact_latent, variant)
                    variant_sampling = _make_sampling_params(args, seed)
                    variant_sampling.latents = perturb.restored_latent.detach().cpu().clone()
                    variant_sampling.step_index = int(checkpoint_state.step_idx)
                    variant_request_id = f"{request_id}-{variant}"
                    variant_request = _make_request(variant_request_id, prompt, variant_sampling)

                    resume_start = time.perf_counter()
                    variant_outputs = await engine.step(variant_request)
                    resume_latency_ms = (time.perf_counter() - resume_start) * 1000.0
                    resumed_video = _unwrap_video_outputs(variant_outputs)
                    artifact_path = artifact_dir / f"step{checkpoint_step:02d}_{variant}.mp4"
                    preflight._save_video(artifact_path, resumed_video, fps=args.fps)

                    temporal_metrics = preflight._temporal_metrics(resumed_video, baseline_video)
                    spatial_metric = preflight._spatial_metric(resumed_video, baseline_video)
                    semantic_metric_raw = (
                        semantic_evaluator.score_video(prompt_text, resumed_video)
                        if semantic_evaluator is not None
                        else float("nan")
                    )
                    video_metrics = _video_metrics(resumed_video, baseline_video)
                    row = {
                        "model": args.model,
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt": prompt_text,
                        "seed": seed,
                        "checkpoint_step": int(checkpoint_state.step_idx),
                        "total_steps": int(args.num_inference_steps),
                        "variant": variant,
                        "retained_bytes": int(perturb.retained_bytes),
                        "retained_ratio": float(perturb.retained_bytes / max(full_state_bytes, 1)),
                        "checkpoint_size_bytes": checkpoint_size_bytes,
                        "checkpoint_value_score": checkpoint_value_score,
                        "resume_latency_ms": resume_latency_ms,
                        "exact_equal": video_metrics["exact_equal"],
                        "video_mse": video_metrics["video_mse"],
                        "max_abs_diff": video_metrics["max_abs_diff"],
                        "spatial_metric": spatial_metric,
                        "temporal_shape_composite": temporal_metrics["temporal_shape_composite"],
                        "temporal_dynamic_composite": temporal_metrics["temporal_dynamic_composite"],
                        "temporal_metric_1": temporal_metrics["temporal_metric_1"],
                        "temporal_metric_2": temporal_metrics["temporal_metric_2"],
                        "motion_metric": temporal_metrics["motion_metric"],
                        "motion_energy_ratio": temporal_metrics["motion_energy_ratio"],
                        "flow_magnitude_cosine": temporal_metrics["flow_magnitude_cosine"],
                        "flow_magnitude_ratio": temporal_metrics["flow_magnitude_ratio"],
                        "flow_direction_cosine": temporal_metrics["flow_direction_cosine"],
                        "flicker_similarity": temporal_metrics["flicker_similarity"],
                        "semantic_metric_raw": semantic_metric_raw,
                        "final_semantic_metric_raw": final_semantic_metric_raw,
                        "semantic_metric": preflight._safe_relative(semantic_metric_raw, final_semantic_metric_raw)
                        if not math.isnan(semantic_metric_raw) and not math.isnan(final_semantic_metric_raw)
                        else float("nan"),
                        "artifact_path": str(artifact_path),
                        "variant_metadata_json": json.dumps(perturb.metadata, sort_keys=True),
                    }
                    rows.append(row)
                    print(
                        f"[frontier] prompt_id={prompt_id} step={checkpoint_step} variant={variant} "
                        f"retained_ratio={row['retained_ratio']:.4f} "
                        f"dynamic={row['temporal_dynamic_composite']:.4f} "
                        f"spatial={row['spatial_metric']:.4f} "
                        f"semantic={row['semantic_metric']:.4f}",
                        flush=True,
                    )
                    state_manager.release_request(variant_request_id)

            baseline_meta = {
                "prompt_id": prompt_id,
                "category": category,
                "prompt": prompt_text,
                "seed": seed,
                "baseline_path": str(baseline_path),
                "baseline_latency_ms": baseline_latency_ms,
                "baseline_semantic_metric_raw": final_semantic_metric_raw,
            }
            (artifact_dir / "baseline_meta.json").write_text(json.dumps(baseline_meta, indent=2), encoding="utf-8")
    finally:
        omni.close()

    aggregate_rows = _aggregate_rows(rows)
    _write_csv(output_dir / "frontier_rows.csv", CSV_FIELDS, rows)
    _write_csv(output_dir / "frontier_summary.csv", SUMMARY_FIELDS, aggregate_rows)
    _write_report(output_dir / "video_recovery_frontier.md", args=args, rows=rows, aggregate_rows=aggregate_rows)

    summary = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "checkpoint_steps": args.checkpoint_steps,
        "variants": args.variants,
        "rows_path": str(output_dir / "frontier_rows.csv"),
        "summary_path": str(output_dir / "frontier_summary.csv"),
        "report_path": str(output_dir / "video_recovery_frontier.md"),
        "num_rows": len(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if any(step < 0 or step >= args.num_inference_steps for step in args.checkpoint_steps):
        raise ValueError(
            f"--checkpoint-steps must all be in [0, {args.num_inference_steps - 1}] "
            f"but got {args.checkpoint_steps}"
        )
    summary = asyncio.run(run_experiment(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
