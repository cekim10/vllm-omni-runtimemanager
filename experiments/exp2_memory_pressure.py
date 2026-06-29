#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

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
)
from vllm_omni.diffusion.state import Fidelity, Placement

DEFAULT_PROMPTS = [
    "A brass astrolabe on a wooden desk",
    "A neon-lit alley in the rain at midnight",
    "An observatory on a snowy mountain at sunrise",
    "A ceramic teapot with sliced citrus on a linen tablecloth",
    "A retro robot tending orchids inside a glass greenhouse",
]

FIDELITY_MODES = (
    "value_aware",
    "always_lossless",
    "always_int8",
    "always_int8_channel",
)

FIXED_MODE_TO_FIDELITY = {
    "always_lossless": Fidelity.LOSSLESS,
    "always_int8": Fidelity.COMPRESSED,
    "always_int8_channel": Fidelity.SKETCH,
}

REQUEST_CSV_FIELDS = [
    "model",
    "run_idx",
    "concurrency",
    "fidelity_mode",
    "slot_idx",
    "prompt_id",
    "prompt",
    "seed",
    "launch_offset_sec",
    "target_step_idx",
    "actual_step_idx",
    "assigned_fidelity",
    "placement",
    "checkpoint_size_bytes",
    "checkpoint_value_score",
    "request_latency_sec",
    "recovery_latency_sec",
    "slo_threshold_sec",
    "slo_met",
    "output_ssim",
    "output_mse",
    "exact_equal",
]

BATCH_CSV_FIELDS = [
    "model",
    "run_idx",
    "concurrency",
    "fidelity_mode",
    "cpu_budget_bytes",
    "gpu_budget_bytes",
    "phase_min_step",
    "phase_max_step",
    "batch_makespan_sec",
    "batch_recovery_makespan_sec",
    "throughput_rps",
    "mean_request_latency_sec",
    "max_request_latency_sec",
    "mean_recovery_latency_sec",
    "max_recovery_latency_sec",
    "slo_threshold_mean_sec",
    "slo_attainment",
    "mean_output_ssim",
    "min_output_ssim",
    "mean_output_mse",
    "max_output_mse",
    "exact_equal_rate",
    "total_checkpoint_size_bytes",
    "mean_checkpoint_size_bytes",
    "placement_gpu_count",
    "placement_cpu_count",
    "placement_disk_count",
    "disk_spill_rate",
    "assigned_fidelity_counts",
]


@dataclass
class BaselineCase:
    slot_idx: int
    prompt: str
    seed: int
    baseline_image: np.ndarray
    baseline_ttfv_sec: float
    image_path: Path


@dataclass
class ConditionRow:
    request_row: dict[str, Any]
    batch_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 2: diffusion memory-pressure experiment.")
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--output", type=Path, required=True, help="Per-request CSV output path.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for baseline/resumed image artifacts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt file. Defaults to 5 built-in prompts.",
    )
    parser.add_argument(
        "--fidelity-modes",
        nargs="+",
        choices=FIDELITY_MODES,
        default=list(FIDELITY_MODES),
        help="Fidelity assignment policies to evaluate.",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Concurrent request counts to evaluate.",
    )
    parser.add_argument("--runs-per-condition", type=int, default=3, help="Number of runs per condition.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed; slot_idx is added per request slot.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Total denoising steps.")
    parser.add_argument("--phase-min-step", type=int, default=10, help="Newest request target step at failure.")
    parser.add_argument("--phase-max-step", type=int, default=40, help="Oldest request target step at failure.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Optional Omni stage config YAML.")
    parser.add_argument(
        "--cpu-budget-bytes",
        type=int,
        default=524288,
        help="Checkpoint CPU tier budget. Defaults to ~2 lossless Z-Image checkpoints at 512x512.",
    )
    parser.add_argument("--gpu-budget-bytes", type=int, default=0, help="Checkpoint GPU tier budget.")
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
        help="Polling interval in seconds while waiting for checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds while waiting for target checkpoint steps.",
    )
    parser.add_argument(
        "--slo-multiplier",
        type=float,
        default=1.25,
        help="SLO threshold multiplier relative to serialized no-failure latency.",
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


def _artifact_dir(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return args.artifact_dir
    return args.output.with_suffix("")


def _write_csv_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _normalize_image(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu().to(torch.float32)
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected a single image tensor, got batch shape {tuple(tensor.shape)}.")
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                if image.min() < 0:
                    image = np.clip(image, -1.0, 1.0) * 0.5 + 0.5
                else:
                    image = np.clip(image, 0.0, 1.0)
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    return image


def _extract_images(result: Any) -> list[Any]:
    images: list[Any] = []

    if isinstance(result, list):
        for item in result:
            extracted = _extract_images(item)
            if extracted:
                return extracted
        return []

    if hasattr(result, "images") and result.images:
        images = result.images
    elif hasattr(result, "request_output"):
        request_output = result.request_output
        if isinstance(request_output, dict) and request_output.get("images"):
            images = request_output["images"]
        elif hasattr(request_output, "images") and request_output.images:
            images = request_output.images
    elif hasattr(result, "output"):
        output = result.output
        if isinstance(output, dict):
            if output.get("images"):
                images = output["images"]
            elif output.get("image") is not None:
                images = [output["image"]]
        elif isinstance(output, (list, tuple)):
            images = list(output)
        elif output is not None:
            images = [output]

    if images and isinstance(images[0], np.ndarray) and images[0].ndim == 5 and images[0].shape[0] > 1:
        images = list(images[0])

    flattened: list[Any] = []
    for image in images:
        if isinstance(image, list):
            flattened.extend(image)
        else:
            flattened.append(image)
    return [_normalize_image(image) for image in flattened]


def _extract_first_image(result: Any) -> np.ndarray:
    images = _extract_images(result)
    if not images:
        raise RuntimeError("No image output produced.")
    return np.asarray(images[0], dtype=np.uint8)


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


async def _run_request_with_finish_time(engine: Any, request: Any) -> tuple[Any, float]:
    output = await engine.async_add_req_and_wait_for_response(request)
    return output, time.perf_counter()


def _case_args(args: argparse.Namespace, prompt: str, slot_idx: int) -> argparse.Namespace:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prompt = prompt
    prompt_args.seed = args.seed + slot_idx
    if prompt_args.disk_path is None:
        prompt_args.disk_path = _artifact_dir(args) / "checkpoints"
    return prompt_args


def _phase_targets(concurrency: int, phase_min_step: int, phase_max_step: int) -> list[int]:
    if concurrency <= 0:
        raise ValueError(f"Concurrency must be positive, got {concurrency}.")
    if concurrency == 1:
        return [phase_max_step]
    steps = np.linspace(phase_max_step, phase_min_step, concurrency)
    return [max(1, int(round(step))) for step in steps.tolist()]


async def _wait_for_checkpoint(
    engine: Any,
    request_id: str,
    target_step: int,
    poll_interval: float,
    timeout_s: float,
) -> Any:
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")

    start = time.monotonic()
    last_state = None
    while True:
        last_state = state_manager.on_failure(request_id)
        if last_state is not None and last_state.step_idx >= target_step:
            return last_state
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for request {request_id} to reach step>={target_step}; "
                f"last_seen_step={getattr(last_state, 'step_idx', None)}"
            )
        await asyncio.sleep(poll_interval)


@contextmanager
def _fidelity_mode_override(engine: Any, fidelity_mode: str) -> Iterator[None]:
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")

    original_assign = state_manager.fid_policy.assign
    if fidelity_mode == "value_aware":
        yield
        return

    fixed_fidelity = FIXED_MODE_TO_FIDELITY[fidelity_mode]
    state_manager.fid_policy.assign = lambda _value_score: fixed_fidelity
    try:
        yield
    finally:
        state_manager.fid_policy.assign = original_assign


async def _run_full_baseline(
    engine: Any,
    args: argparse.Namespace,
    *,
    slot_idx: int,
    prompt: str,
    artifact_dir: Path,
) -> BaselineCase:
    prompt_args = _case_args(args, prompt, slot_idx)
    prompt_obj = _build_prompt(engine.omni, prompt_args) if hasattr(engine, "omni") else None
    if prompt_obj is None:
        raise RuntimeError("Inline diffusion engine is missing its Omni owner reference.")
    sampling_params = _build_sampling_params(prompt_args)
    request = _make_request(f"baseline-slot{slot_idx}", prompt_obj, sampling_params)
    start = time.perf_counter()
    outputs = await engine.step(request)
    elapsed = time.perf_counter() - start
    image = _extract_first_image(outputs)
    image_path = artifact_dir / f"baseline_slot_{slot_idx:02d}.png"
    _save_image(image, image_path)
    return BaselineCase(
        slot_idx=slot_idx,
        prompt=prompt,
        seed=prompt_args.seed,
        baseline_image=image,
        baseline_ttfv_sec=elapsed,
        image_path=image_path,
    )


async def _build_baselines(
    engine: Any,
    args: argparse.Namespace,
    prompts: list[str],
    artifact_dir: Path,
) -> list[BaselineCase]:
    max_concurrency = max(args.concurrency_levels)
    baselines: list[BaselineCase] = []
    for slot_idx in range(max_concurrency):
        prompt = prompts[slot_idx % len(prompts)]
        baseline = await _run_full_baseline(
            engine,
            args,
            slot_idx=slot_idx,
            prompt=prompt,
            artifact_dir=artifact_dir,
        )
        baselines.append(baseline)
        print(
            f"[baseline] slot_idx={slot_idx} seed={baseline.seed} ttfv_sec={baseline.baseline_ttfv_sec:.3f}",
            flush=True,
        )
    return baselines


def _attach_engine_owner(omni: Any, engine: Any) -> None:
    setattr(engine, "omni", omni)


async def _run_condition(
    engine: Any,
    args: argparse.Namespace,
    *,
    run_idx: int,
    concurrency: int,
    fidelity_mode: str,
    baselines: list[BaselineCase],
    artifact_dir: Path,
) -> list[ConditionRow]:
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")

    cases = baselines[:concurrency]
    target_steps = _phase_targets(concurrency, args.phase_min_step, args.phase_max_step)
    request_ids = [f"{fidelity_mode}-c{concurrency}-r{run_idx}-slot{case.slot_idx}" for case in cases]
    launch_times: dict[str, float] = {}
    launch_offsets: dict[str, float] = {}
    initial_tasks: dict[str, asyncio.Task[tuple[Any, float]]] = {}
    prompt_objs: dict[str, Any] = {}
    sampling_params_by_id: dict[str, Any] = {}
    batch_start = time.perf_counter()

    def _make_prompt_and_sampling(case: BaselineCase) -> tuple[Any, Any]:
        prompt_args = _case_args(args, case.prompt, case.slot_idx)
        return _build_prompt(engine.omni, prompt_args), _build_sampling_params(prompt_args)

    async def _start_request(case: BaselineCase, request_id: str) -> None:
        prompt_obj, sampling_params = _make_prompt_and_sampling(case)
        prompt_objs[request_id] = prompt_obj
        sampling_params_by_id[request_id] = sampling_params
        launch_times[request_id] = time.perf_counter()
        launch_offsets[request_id] = launch_times[request_id] - batch_start
        initial_tasks[request_id] = asyncio.create_task(
            _run_request_with_finish_time(engine, _make_request(request_id, prompt_obj, sampling_params))
        )

    with _fidelity_mode_override(engine, fidelity_mode):
        for idx in range(concurrency):
            await _start_request(cases[idx], request_ids[idx])

        target_by_request = {request_ids[idx]: target_steps[idx] for idx in range(concurrency)}
        checkpoint_states = {}
        checkpoint_wait_tasks = {
            request_id: asyncio.create_task(
                _wait_for_checkpoint(
                    engine,
                    request_id=request_id,
                    target_step=target_step,
                    poll_interval=args.poll_interval,
                    timeout_s=args.checkpoint_timeout,
                )
            )
            for request_id, target_step in target_by_request.items()
        }
        wait_task_to_request = {task: request_id for request_id, task in checkpoint_wait_tasks.items()}

        pending_waits = set(checkpoint_wait_tasks.values())
        while pending_waits:
            done, pending_waits = await asyncio.wait(pending_waits, return_when=asyncio.FIRST_COMPLETED)
            for wait_task in done:
                request_id = wait_task_to_request[wait_task]
                checkpoint_states[request_id] = wait_task.result()
                engine.abort(request_id)
                output, _finish_time = await initial_tasks[request_id]
                if not output.aborted:
                    raise RuntimeError(f"Expected aborted output for {request_id}, got: {output!r}")

        for request_id in request_ids:
            if request_id not in checkpoint_states:
                raise RuntimeError(f"Missing checkpoint for request {request_id}.")

        for request_id in request_ids:
            output, _finish_time = await initial_tasks[request_id]
            if not output.aborted:
                raise RuntimeError(f"Expected aborted output for {request_id}, got: {output!r}")

        recovery_start = time.perf_counter()
        resumed_tasks: dict[str, asyncio.Task[tuple[Any, float]]] = {}
        resume_launch_times: dict[str, float] = {}
        for request_id in request_ids:
            resume_template = _make_request(request_id, prompt_objs[request_id], sampling_params_by_id[request_id])
            resumed_request = engine.restore_request_from_state(resume_template, request_id=request_id)
            resume_launch_times[request_id] = time.perf_counter()
            resumed_tasks[request_id] = asyncio.create_task(_run_request_with_finish_time(engine, resumed_request))

        resumed_results = await asyncio.gather(*(resumed_tasks[request_id] for request_id in request_ids))
        recovery_end = time.perf_counter()

    request_rows: list[ConditionRow] = []
    request_latencies: list[float] = []
    request_recovery_latencies: list[float] = []
    finish_times: list[float] = []
    request_ssims: list[float] = []
    request_mses: list[float] = []
    request_exacts: list[float] = []
    checkpoint_sizes: list[int] = []
    placements = Counter()
    fidelity_counts = Counter()

    for idx, (case, request_id, resumed_result) in enumerate(zip(cases, request_ids, resumed_results, strict=True)):
        resumed_output, finish_time = resumed_result
        if resumed_output.error is not None:
            raise RuntimeError(f"Resumed request {request_id} failed: {resumed_output.error}")

        checkpoint_state = checkpoint_states[request_id]
        output_image = _extract_first_image([resumed_output])
        image_path = (
            artifact_dir
            / f"concurrency_{concurrency:02d}"
            / fidelity_mode
            / f"run_{run_idx:02d}"
            / f"slot_{case.slot_idx:02d}.png"
        )
        _save_image(output_image, image_path)
        metrics = _image_metrics(output_image, case.baseline_image)
        request_latency_sec = finish_time - launch_times[request_id]
        recovery_latency_sec = finish_time - resume_launch_times[request_id]
        slo_threshold_sec = case.baseline_ttfv_sec * concurrency * args.slo_multiplier
        slo_met = request_latency_sec <= slo_threshold_sec

        request_latencies.append(request_latency_sec)
        request_recovery_latencies.append(recovery_latency_sec)
        finish_times.append(finish_time)
        request_ssims.append(float(metrics["output_ssim"]))
        request_mses.append(float(metrics["output_mse"]))
        request_exacts.append(1.0 if metrics["exact_equal"] else 0.0)
        checkpoint_sizes.append(int(checkpoint_state.size_bytes))
        placements[checkpoint_state.placement.value] += 1
        fidelity_counts[checkpoint_state.fidelity.value] += 1

        request_row = {
            "model": args.model,
            "run_idx": run_idx,
            "concurrency": concurrency,
            "fidelity_mode": fidelity_mode,
            "slot_idx": case.slot_idx,
            "prompt_id": case.slot_idx,
            "prompt": case.prompt,
            "seed": case.seed,
            "launch_offset_sec": launch_offsets[request_id],
            "target_step_idx": target_steps[idx],
            "actual_step_idx": checkpoint_state.step_idx,
            "assigned_fidelity": checkpoint_state.fidelity.value,
            "placement": checkpoint_state.placement.value,
            "checkpoint_size_bytes": checkpoint_state.size_bytes,
            "checkpoint_value_score": checkpoint_state.value_score,
            "request_latency_sec": request_latency_sec,
            "recovery_latency_sec": recovery_latency_sec,
            "slo_threshold_sec": slo_threshold_sec,
            "slo_met": slo_met,
            "output_ssim": metrics["output_ssim"],
            "output_mse": metrics["output_mse"],
            "exact_equal": metrics["exact_equal"],
        }
        request_rows.append(ConditionRow(request_row=request_row, batch_row={}))

    batch_makespan_sec = max(finish_times) - batch_start if finish_times else 0.0
    batch_recovery_makespan_sec = recovery_end - recovery_start
    batch_row = {
        "model": args.model,
        "run_idx": run_idx,
        "concurrency": concurrency,
        "fidelity_mode": fidelity_mode,
        "cpu_budget_bytes": args.cpu_budget_bytes,
        "gpu_budget_bytes": args.gpu_budget_bytes,
        "phase_min_step": args.phase_min_step,
        "phase_max_step": args.phase_max_step,
        "batch_makespan_sec": batch_makespan_sec,
        "batch_recovery_makespan_sec": batch_recovery_makespan_sec,
        "throughput_rps": concurrency / max(batch_makespan_sec, 1e-8),
        "mean_request_latency_sec": mean(request_latencies),
        "max_request_latency_sec": max(request_latencies),
        "mean_recovery_latency_sec": mean(request_recovery_latencies),
        "max_recovery_latency_sec": max(request_recovery_latencies),
        "slo_threshold_mean_sec": mean(float(row.request_row["slo_threshold_sec"]) for row in request_rows),
        "slo_attainment": mean(1.0 if row.request_row["slo_met"] else 0.0 for row in request_rows),
        "mean_output_ssim": mean(request_ssims),
        "min_output_ssim": min(request_ssims),
        "mean_output_mse": mean(request_mses),
        "max_output_mse": max(request_mses),
        "exact_equal_rate": mean(request_exacts),
        "total_checkpoint_size_bytes": sum(checkpoint_sizes),
        "mean_checkpoint_size_bytes": mean(checkpoint_sizes),
        "placement_gpu_count": placements.get(Placement.GPU.value, 0),
        "placement_cpu_count": placements.get(Placement.CPU.value, 0),
        "placement_disk_count": placements.get(Placement.DISK.value, 0),
        "disk_spill_rate": placements.get(Placement.DISK.value, 0) / concurrency,
        "assigned_fidelity_counts": json.dumps(dict(sorted(fidelity_counts.items())), sort_keys=True),
    }

    for row in request_rows:
        row.batch_row = batch_row
    return request_rows


def _aggregate_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["fidelity_mode"], int(row["concurrency"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (fidelity_mode, concurrency), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        aggregates.append(
            {
                "fidelity_mode": fidelity_mode,
                "concurrency": concurrency,
                "num_rows": len(group_rows),
                "mean_throughput_rps": mean(float(row["throughput_rps"]) for row in group_rows),
                "mean_batch_makespan_sec": mean(float(row["batch_makespan_sec"]) for row in group_rows),
                "mean_batch_recovery_makespan_sec": mean(float(row["batch_recovery_makespan_sec"]) for row in group_rows),
                "mean_recovery_latency_sec": mean(float(row["mean_recovery_latency_sec"]) for row in group_rows),
                "mean_slo_attainment": mean(float(row["slo_attainment"]) for row in group_rows),
                "mean_output_ssim": mean(float(row["mean_output_ssim"]) for row in group_rows),
                "mean_output_mse": mean(float(row["mean_output_mse"]) for row in group_rows),
                "mean_disk_spill_rate": mean(float(row["disk_spill_rate"]) for row in group_rows),
                "mean_checkpoint_size_bytes": mean(float(row["mean_checkpoint_size_bytes"]) for row in group_rows),
                "mean_total_checkpoint_size_bytes": mean(float(row["total_checkpoint_size_bytes"]) for row in group_rows),
            }
        )
    return aggregates


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompts(args)
    _preflight_real_model_args(args)

    if args.phase_min_step <= 0 or args.phase_max_step >= args.num_inference_steps:
        raise ValueError(
            f"Phase steps must satisfy 1 <= phase_min_step < phase_max_step < num_inference_steps; got "
            f"min={args.phase_min_step}, max={args.phase_max_step}, total={args.num_inference_steps}."
        )

    artifact_dir = _artifact_dir(args)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.disk_path is None:
        args.disk_path = artifact_dir / "checkpoints"
    Path(args.disk_path).mkdir(parents=True, exist_ok=True)

    request_csv_path = args.output
    batch_csv_path = args.output.with_name(f"{args.output.stem}.batch{args.output.suffix}")
    _write_csv_header(request_csv_path, REQUEST_CSV_FIELDS)
    _write_csv_header(batch_csv_path, BATCH_CSV_FIELDS)

    omni, init_config = _initialize_omni_with_retry(args)
    request_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    try:
        engine = _get_inline_diffusion_engine(omni)
        _attach_engine_owner(omni, engine)
        baselines = await _build_baselines(engine, args, prompts, artifact_dir)

        for concurrency in args.concurrency_levels:
            for fidelity_mode in args.fidelity_modes:
                for run_idx in range(args.runs_per_condition):
                    condition_rows = await _run_condition(
                        engine,
                        args,
                        run_idx=run_idx,
                        concurrency=concurrency,
                        fidelity_mode=fidelity_mode,
                        baselines=baselines,
                        artifact_dir=artifact_dir,
                    )
                    for row in condition_rows:
                        request_rows.append(row.request_row)
                        _append_csv_row(request_csv_path, row.request_row, REQUEST_CSV_FIELDS)

                    batch_row = condition_rows[0].batch_row
                    batch_rows.append(batch_row)
                    _append_csv_row(batch_csv_path, batch_row, BATCH_CSV_FIELDS)
                    print(
                        f"[{fidelity_mode}] concurrency={concurrency} run_idx={run_idx} "
                        f"throughput_rps={batch_row['throughput_rps']:.3f} "
                        f"disk_spill_rate={batch_row['disk_spill_rate']:.3f} "
                        f"mean_ssim={batch_row['mean_output_ssim']:.6f} "
                        f"slo_attainment={batch_row['slo_attainment']:.3f} "
                        f"fidelity_counts={batch_row['assigned_fidelity_counts']}",
                        flush=True,
                    )
    finally:
        omni.close()

    summary = {
        "model": args.model,
        "num_inference_steps": args.num_inference_steps,
        "concurrency_levels": args.concurrency_levels,
        "fidelity_modes": args.fidelity_modes,
        "runs_per_condition": args.runs_per_condition,
        "phase_min_step": args.phase_min_step,
        "phase_max_step": args.phase_max_step,
        "cpu_budget_bytes": args.cpu_budget_bytes,
        "gpu_budget_bytes": args.gpu_budget_bytes,
        "slo_multiplier": args.slo_multiplier,
        "init_config": init_config,
        "request_csv_path": str(request_csv_path),
        "batch_csv_path": str(batch_csv_path),
        "artifact_dir": str(artifact_dir),
        "batch_aggregates": _aggregate_batch_rows(batch_rows),
        "num_request_rows": len(request_rows),
        "num_batch_rows": len(batch_rows),
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
