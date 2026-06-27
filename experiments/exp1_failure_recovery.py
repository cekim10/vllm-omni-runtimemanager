#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diffusion_state_recovery_smoke import (
    _build_prompt,
    _build_sampling_params,
    _get_inline_diffusion_engine,
    _initialize_omni_with_retry,
    _make_request,
    _preflight_real_model_args,
    _wait_for_checkpoint,
)

DEFAULT_PROMPTS = [
    "A brass astrolabe on a wooden desk",
    "A neon-lit alley in the rain at midnight",
    "An observatory on a snowy mountain at sunrise",
    "A ceramic teapot with sliced citrus on a linen tablecloth",
    "A retro robot tending orchids inside a glass greenhouse",
]

CSV_FIELDS = [
    "model",
    "prompt_id",
    "prompt",
    "run_idx",
    "strategy",
    "failure_step_frac",
    "failure_step_idx",
    "total_steps",
    "ttfv_sec",
    "wasted_gpu_sec",
    "recovery_latency_ms",
    "output_ssim",
    "output_mse",
    "exact_equal",
]


@dataclass
class BaselineResult:
    image: np.ndarray
    ttfv_sec: float
    image_path: Path
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: diffusion failure-recovery metrics.")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for baseline/restart/resume image artifacts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt file. Defaults to 5 built-in prompts.",
    )
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed; prompt_id is added per prompt.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Total denoising steps.")
    parser.add_argument(
        "--failure-fracs",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Failure-step fractions of total denoising steps.",
    )
    parser.add_argument("--runs-per-condition", type=int, default=3, help="Number of runs per condition.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Optional Omni stage config YAML.")
    parser.add_argument("--gpu-budget-bytes", type=int, default=0, help="State manager GPU tier budget.")
    parser.add_argument("--cpu-budget-bytes", type=int, default=0, help="State manager CPU tier budget.")
    parser.add_argument("--theta-h", type=float, default=0.7, help="LOSSLESS threshold.")
    parser.add_argument("--theta-w", type=float, default=0.3, help="COMPRESSED threshold.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Omni init timeout in seconds.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Per-stage init timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile for easier debugging.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offload from the first initialization attempt.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise offload from the first initialization attempt.",
    )
    parser.add_argument(
        "--retry-with-offload",
        action="store_true",
        default=True,
        help="Retry model initialization with CPU/layerwise offload after a CUDA OOM.",
    )
    parser.add_argument(
        "--no-retry-with-offload",
        dest="retry_with_offload",
        action="store_false",
        help="Disable the automatic OOM retry path.",
    )
    parser.add_argument(
        "--disable-auto-offload",
        action="store_true",
        help="Disable the conservative single-GPU auto-offload heuristic.",
    )
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
    parser.set_defaults(backend="real-model", disk_path=None, strict_equality=False, step_delay=0.02)
    return parser.parse_args()


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file is None:
        return list(DEFAULT_PROMPTS)

    prompts = [line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_file}.")
    return prompts


def _ensure_failure_step(frac: float, total_steps: int) -> int:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"Failure fraction must be in (0, 1), got {frac}.")
    return max(1, min(total_steps - 1, int(total_steps * frac)))


def _artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return args.artifact_dir
    return args.output.with_suffix("")


def _prompt_args(args: argparse.Namespace, prompt: str, prompt_id: int) -> argparse.Namespace:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prompt = prompt
    prompt_args.seed = args.seed + prompt_id
    if prompt_args.disk_path is None:
        prompt_args.disk_path = _artifact_dir(args) / "checkpoints"
    return prompt_args


def _extract_first_image(outputs: list[Any]) -> np.ndarray:
    if not outputs or not outputs[0].images:
        raise RuntimeError("No image output produced.")
    return np.asarray(outputs[0].images[0].convert("RGB"), dtype=np.uint8)


def _save_image(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _compute_ssim(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if np.array_equal(lhs, rhs):
        return 1.0

    try:
        from skimage.metrics import structural_similarity as structural_similarity

        return float(structural_similarity(lhs, rhs, channel_axis=-1, data_range=255))
    except ImportError:
        pass

    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        metric = StructuralSimilarityIndexMeasure(data_range=255.0)
        lhs_tensor = torch.from_numpy(lhs).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        rhs_tensor = torch.from_numpy(rhs).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        return float(metric(lhs_tensor, rhs_tensor).item())
    except ImportError:
        pass

    lhs_float = lhs.astype(np.float64)
    rhs_float = rhs.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = lhs_float.mean()
    mu_y = rhs_float.mean()
    sigma_x = lhs_float.var()
    sigma_y = rhs_float.var()
    sigma_xy = ((lhs_float - mu_x) * (rhs_float - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)


def _image_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diff = candidate.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return {
        "exact_equal": bool(np.array_equal(candidate, reference)),
        "output_mse": mse,
        "output_ssim": _compute_ssim(candidate, reference),
    }


async def _run_full_baseline(
    engine: Any,
    prompt: Any,
    sampling_params: Any,
    *,
    prompt_id: int,
    prompt_seed: int,
    artifact_dir: Path,
) -> BaselineResult:
    request = _make_request(f"baseline-p{prompt_id}", prompt, sampling_params)
    start = time.perf_counter()
    outputs = await engine.step(request)
    elapsed = time.perf_counter() - start
    image = _extract_first_image(outputs)
    image_path = artifact_dir / f"prompt_{prompt_id:02d}" / "baseline.png"
    _save_image(image, image_path)
    return BaselineResult(image=image, ttfv_sec=elapsed, image_path=image_path, seed=prompt_seed)


async def _run_restart(
    engine: Any,
    prompt: Any,
    sampling_params: Any,
    *,
    prompt_id: int,
    run_idx: int,
    failure_step_idx: int,
    failure_step_frac: float,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    request_id = f"restart-p{prompt_id}-r{run_idx}-s{failure_step_idx}"
    start = time.perf_counter()
    run_task = asyncio.create_task(engine.async_add_req_and_wait_for_response(_make_request(request_id, prompt, sampling_params)))
    checkpoint_state = await _wait_for_checkpoint(
        engine,
        request_id=request_id,
        target_step=failure_step_idx,
        poll_interval=args.poll_interval,
        timeout_s=args.checkpoint_timeout,
    )
    wasted_gpu_sec = time.perf_counter() - start
    engine.abort(request_id)
    aborted_output = await run_task
    if not aborted_output.aborted:
        raise RuntimeError(f"Restart baseline expected an aborted request, got: {aborted_output!r}")

    state_manager = getattr(engine, "state_manager", None)
    if state_manager is not None:
        state_manager.release_request(request_id)

    restarted_outputs = await engine.step(_make_request(f"{request_id}-fresh", prompt, sampling_params))
    ttfv_sec = time.perf_counter() - start
    image = _extract_first_image(restarted_outputs)
    image_path = artifact_dir / f"prompt_{prompt_id:02d}" / f"restart_f{failure_step_idx}_r{run_idx}.png"
    _save_image(image, image_path)
    return (
        {
            "strategy": "restart",
            "failure_step_frac": failure_step_frac,
            "failure_step_idx": checkpoint_state.step_idx,
            "ttfv_sec": ttfv_sec,
            "wasted_gpu_sec": wasted_gpu_sec,
            "recovery_latency_ms": 0.0,
        },
        image,
    )


async def _run_ours(
    engine: Any,
    prompt: Any,
    sampling_params: Any,
    *,
    prompt_id: int,
    run_idx: int,
    failure_step_idx: int,
    failure_step_frac: float,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    request_id = f"ours-p{prompt_id}-r{run_idx}-s{failure_step_idx}"
    start = time.perf_counter()
    run_task = asyncio.create_task(engine.async_add_req_and_wait_for_response(_make_request(request_id, prompt, sampling_params)))
    checkpoint_state = await _wait_for_checkpoint(
        engine,
        request_id=request_id,
        target_step=failure_step_idx,
        poll_interval=args.poll_interval,
        timeout_s=args.checkpoint_timeout,
    )
    engine.abort(request_id)
    aborted_output = await run_task
    if not aborted_output.aborted:
        raise RuntimeError(f"Ours expected an aborted request, got: {aborted_output!r}")

    restore_start = time.perf_counter()
    resumed_request = engine.restore_request_from_state(_make_request(request_id, prompt, sampling_params), request_id=request_id)
    resumed_outputs = await engine.step(resumed_request)
    recovery_latency_ms = (time.perf_counter() - restore_start) * 1000.0
    ttfv_sec = time.perf_counter() - start
    image = _extract_first_image(resumed_outputs)
    image_path = artifact_dir / f"prompt_{prompt_id:02d}" / f"ours_f{failure_step_idx}_r{run_idx}.png"
    _save_image(image, image_path)
    return (
        {
            "strategy": "ours",
            "failure_step_frac": failure_step_frac,
            "failure_step_idx": checkpoint_state.step_idx,
            "ttfv_sec": ttfv_sec,
            "wasted_gpu_sec": 0.0,
            "recovery_latency_ms": recovery_latency_ms,
        },
        image,
    )


def _write_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["failure_step_frac"])].append(row)

    aggregates = []
    for (strategy, frac), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        aggregates.append(
            {
                "strategy": strategy,
                "failure_step_frac": frac,
                "num_rows": len(group_rows),
                "mean_ttfv_sec": mean(float(row["ttfv_sec"]) for row in group_rows),
                "mean_wasted_gpu_sec": mean(float(row["wasted_gpu_sec"]) for row in group_rows),
                "mean_recovery_latency_ms": mean(float(row["recovery_latency_ms"]) for row in group_rows),
                "mean_output_ssim": mean(float(row["output_ssim"]) for row in group_rows),
                "mean_output_mse": mean(float(row["output_mse"]) for row in group_rows),
                "exact_equal_rate": mean(1.0 if row["exact_equal"] else 0.0 for row in group_rows),
            }
        )
    return aggregates


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompts(args)
    _preflight_real_model_args(args)

    artifact_dir = _artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.disk_path is None:
        args.disk_path = artifact_dir / "checkpoints"
    Path(args.disk_path).mkdir(parents=True, exist_ok=True)
    _write_csv_header(args.output)

    omni, init_config = _initialize_omni_with_retry(args)
    rows: list[dict[str, Any]] = []
    baseline_summary: list[dict[str, Any]] = []
    try:
        engine = _get_inline_diffusion_engine(omni)

        for prompt_id, prompt_text in enumerate(prompts):
            prompt_args = _prompt_args(args, prompt_text, prompt_id)
            prompt = _build_prompt(omni, prompt_args)
            sampling_params = _build_sampling_params(prompt_args)
            baseline = await _run_full_baseline(
                engine,
                prompt,
                sampling_params,
                prompt_id=prompt_id,
                prompt_seed=prompt_args.seed,
                artifact_dir=artifact_dir,
            )
            baseline_summary.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "seed": baseline.seed,
                    "baseline_ttfv_sec": baseline.ttfv_sec,
                    "baseline_image": str(baseline.image_path),
                }
            )
            print(
                f"[baseline] prompt_id={prompt_id} seed={baseline.seed} ttfv_sec={baseline.ttfv_sec:.3f}",
                flush=True,
            )

            for frac in args.failure_fracs:
                failure_step_idx = _ensure_failure_step(frac, args.num_inference_steps)
                for run_idx in range(args.runs_per_condition):
                    for runner in (_run_restart, _run_ours):
                        row, image = await runner(
                            engine,
                            prompt,
                            sampling_params,
                            prompt_id=prompt_id,
                            run_idx=run_idx,
                            failure_step_idx=failure_step_idx,
                            failure_step_frac=frac,
                            artifact_dir=artifact_dir,
                            args=args,
                        )
                        metrics = _image_metrics(image, baseline.image)
                        csv_row = {
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "prompt": prompt_text,
                            "run_idx": run_idx,
                            "strategy": row["strategy"],
                            "failure_step_frac": frac,
                            "failure_step_idx": row["failure_step_idx"],
                            "total_steps": args.num_inference_steps,
                            "ttfv_sec": row["ttfv_sec"],
                            "wasted_gpu_sec": row["wasted_gpu_sec"],
                            "recovery_latency_ms": row["recovery_latency_ms"],
                            "output_ssim": metrics["output_ssim"],
                            "output_mse": metrics["output_mse"],
                            "exact_equal": metrics["exact_equal"],
                        }
                        rows.append(csv_row)
                        _append_csv_row(args.output, csv_row)
                        print(
                            f"[{csv_row['strategy']}] prompt_id={prompt_id} run_idx={run_idx} "
                            f"failure_frac={frac:.2f} ttfv_sec={csv_row['ttfv_sec']:.3f} "
                            f"wasted_gpu_sec={csv_row['wasted_gpu_sec']:.3f} "
                            f"recovery_latency_ms={csv_row['recovery_latency_ms']:.2f} "
                            f"ssim={csv_row['output_ssim']:.6f} exact={csv_row['exact_equal']}",
                            flush=True,
                        )
    finally:
        omni.close()

    aggregates = _aggregate_rows(rows)
    summary = {
        "model": args.model,
        "num_inference_steps": args.num_inference_steps,
        "failure_fracs": args.failure_fracs,
        "runs_per_condition": args.runs_per_condition,
        "init_config": init_config,
        "prompts": baseline_summary,
        "csv_path": str(args.output),
        "artifact_dir": str(artifact_dir),
        "aggregates": aggregates,
        "num_rows": len(rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_experiment(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
