#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp2_memory_pressure import (
    _append_csv_row,
    _artifact_dir,
    _attach_engine_owner,
    _case_args,
    _extract_first_image,
    _image_metrics,
    _load_prompts,
    _phase_targets,
    _run_full_baseline,
    _run_request_with_finish_time,
    _wait_for_checkpoint,
    _write_csv_header,
)
from tools.diffusion_state_recovery_smoke import (
    _build_prompt,
    _build_sampling_params,
    _get_inline_diffusion_engine,
    _initialize_omni_with_retry,
    _make_request,
    _preflight_real_model_args,
)
from vllm_omni.diffusion.state import Fidelity, Placement

POLICIES = (
    "restart",
    "gpu_resident",
    "cpu_lossless",
    "cpu_value_aware",
)

REQUEST_CSV_FIELDS = [
    "model",
    "run_idx",
    "policy",
    "paused_request_count",
    "pause_duration_sec",
    "foreground_level",
    "foreground_concurrency",
    "request_role",
    "slot_idx",
    "prompt_id",
    "prompt",
    "seed",
    "request_id",
    "target_step_idx",
    "actual_step_idx",
    "assigned_fidelity",
    "placement",
    "checkpoint_size_bytes",
    "checkpoint_value_score",
    "admitted",
    "admission_latency_sec",
    "total_latency_sec",
    "resume_or_restart_latency_sec",
    "output_ssim",
    "output_mse",
    "exact_equal",
]

BATCH_CSV_FIELDS = [
    "model",
    "run_idx",
    "policy",
    "paused_request_count",
    "pause_duration_sec",
    "foreground_level",
    "foreground_concurrency",
    "phase_min_step",
    "phase_max_step",
    "pre_pause_gpu_free_bytes",
    "post_pause_gpu_free_bytes",
    "post_foreground_gpu_free_bytes",
    "post_resume_gpu_free_bytes",
    "paused_gpu_count",
    "paused_cpu_count",
    "paused_disk_count",
    "paused_total_checkpoint_bytes",
    "paused_mean_checkpoint_bytes",
    "paused_assigned_fidelity_counts",
    "paused_estimated_d2h_bytes",
    "paused_estimated_h2d_bytes",
    "foreground_admission_rate",
    "foreground_mean_admission_latency_sec",
    "foreground_mean_latency_sec",
    "foreground_max_latency_sec",
    "foreground_success_rate",
    "paused_mean_completion_latency_sec",
    "paused_max_completion_latency_sec",
    "paused_mean_resume_or_restart_latency_sec",
    "paused_mean_output_ssim",
    "paused_min_output_ssim",
    "paused_mean_output_mse",
    "paused_max_output_mse",
    "paused_exact_equal_rate",
]


@dataclass
class ForegroundRequestResult:
    request_id: str
    slot_idx: int
    prompt: str
    seed: int
    target_step_idx: int
    admitted: bool
    admission_latency_sec: float | None
    total_latency_sec: float | None
    error_text: str | None


@dataclass
class PausedRequestResult:
    request_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go/No-Go experiment: when does GPU-resident paused diffusion state become a serving bottleneck?"
    )
    parser.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo", help="Diffusion model name or local path.")
    parser.add_argument("--output", type=Path, required=True, help="Per-request CSV output path.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for baseline/foreground/resumed image artifacts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt file.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Preservation strategies to compare.",
    )
    parser.add_argument(
        "--paused-request-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of paused background requests to retain before admitting foreground work.",
    )
    parser.add_argument(
        "--pause-durations-sec",
        type=float,
        nargs="+",
        default=[0.1, 1.0],
        help="How long to keep requests paused before foreground admission.",
    )
    parser.add_argument(
        "--foreground-levels",
        nargs="+",
        choices=["low", "high"],
        default=["low", "high"],
        help="Foreground memory-pressure presets to evaluate.",
    )
    parser.add_argument("--foreground-low-concurrency", type=int, default=1, help="Foreground request count for low pressure.")
    parser.add_argument("--foreground-high-concurrency", type=int, default=2, help="Foreground request count for high pressure.")
    parser.add_argument("--runs-per-condition", type=int, default=1, help="Number of runs per condition.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Total denoising steps.")
    parser.add_argument("--phase-min-step", type=int, default=10, help="Newest paused request target step.")
    parser.add_argument("--phase-max-step", type=int, default=40, help="Oldest paused request target step.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen true CFG scale.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Optional Omni stage config YAML.")
    parser.add_argument(
        "--cpu-budget-bytes",
        type=int,
        default=1 << 30,
        help="CPU budget for offloaded paused states. Defaults high enough to avoid accidental disk spill for image workloads.",
    )
    parser.add_argument(
        "--gpu-resident-budget-bytes",
        type=int,
        default=1 << 30,
        help="Logical GPU checkpoint budget for the gpu_resident policy.",
    )
    parser.add_argument(
        "--restart-observation-budget-bytes",
        type=int,
        default=1 << 20,
        help="Temporary checkpoint budget used only to observe progress before dropping state in restart mode.",
    )
    parser.add_argument("--gpu-budget-bytes", type=int, default=0, help="Initial state-manager GPU budget at engine init.")
    parser.add_argument("--theta-h", type=float, default=0.7, help="LOSSLESS threshold.")
    parser.add_argument("--theta-w", type=float, default=0.3, help="COMPRESSED threshold.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Omni init timeout in seconds.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Per-stage init timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile for easier debugging.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offload from the first init attempt.")
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise offload from the first init attempt.",
    )
    parser.add_argument(
        "--retry-with-offload",
        action="store_true",
        default=True,
        help="Retry model initialization with CPU/layerwise offload after CUDA OOM.",
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
        help="Timeout in seconds while waiting for paused-request checkpoints.",
    )
    parser.add_argument(
        "--foreground-admission-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for a foreground request to reach step 1.",
    )
    parser.set_defaults(backend="real-model", disk_path=None, strict_equality=False, step_delay=0.02)
    return parser.parse_args()


def _foreground_concurrency(level: str, args: argparse.Namespace) -> int:
    if level == "low":
        return args.foreground_low_concurrency
    if level == "high":
        return args.foreground_high_concurrency
    raise ValueError(f"Unexpected foreground level: {level}")


def _gpu_mem_snapshot() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"free_bytes": None, "total_bytes": None, "used_bytes": None}

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
    }


@contextmanager
def _policy_override(
    engine: Any,
    policy: str,
    args: argparse.Namespace,
) -> Iterator[None]:
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")

    state_manager.clear()

    original_assign = state_manager.fid_policy.assign
    original_gpu_budget = state_manager.placement.gpu_budget
    original_cpu_budget = state_manager.placement.cpu_budget

    try:
        if policy == "restart":
            state_manager.fid_policy.assign = lambda _value_score: Fidelity.LOSSLESS
            state_manager.placement.gpu_budget = 0
            state_manager.placement.cpu_budget = args.restart_observation_budget_bytes
        elif policy == "gpu_resident":
            state_manager.fid_policy.assign = lambda _value_score: Fidelity.LOSSLESS
            state_manager.placement.gpu_budget = args.gpu_resident_budget_bytes
            state_manager.placement.cpu_budget = 0
        elif policy == "cpu_lossless":
            state_manager.fid_policy.assign = lambda _value_score: Fidelity.LOSSLESS
            state_manager.placement.gpu_budget = 0
            state_manager.placement.cpu_budget = args.cpu_budget_bytes
        elif policy == "cpu_value_aware":
            state_manager.placement.gpu_budget = 0
            state_manager.placement.cpu_budget = args.cpu_budget_bytes
        else:
            raise ValueError(f"Unsupported policy {policy!r}")
        yield
    finally:
        state_manager.clear()
        state_manager.fid_policy.assign = original_assign
        state_manager.placement.gpu_budget = original_gpu_budget
        state_manager.placement.cpu_budget = original_cpu_budget


def _prompt_and_sampling(engine: Any, args: argparse.Namespace, *, prompt: str, seed: int) -> tuple[Any, Any]:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prompt = prompt
    prompt_args.seed = seed
    return _build_prompt(engine.omni, prompt_args), _build_sampling_params(prompt_args)


async def _build_background_baselines(
    engine: Any,
    args: argparse.Namespace,
    prompts: list[str],
    artifact_dir: Path,
    count: int,
) -> list[Any]:
    baselines: list[Any] = []
    for slot_idx in range(count):
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


async def _wait_for_admission(
    engine: Any,
    request_id: str,
    *,
    poll_interval: float,
    timeout_s: float,
) -> tuple[float, Any]:
    start = time.perf_counter()
    state = await _wait_for_checkpoint(
        engine,
        request_id=request_id,
        target_step=1,
        poll_interval=poll_interval,
        timeout_s=timeout_s,
    )
    return time.perf_counter() - start, state


def _estimated_transfer_bytes(policy: str, placement: Placement, size_bytes: int) -> tuple[int, int]:
    if policy == "restart":
        return 0, 0
    if placement == Placement.GPU:
        return 0, 0
    return size_bytes, size_bytes


async def _pause_background_requests(
    engine: Any,
    args: argparse.Namespace,
    *,
    policy: str,
    paused_request_count: int,
    baselines: list[Any],
    target_steps: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_manager = getattr(engine, "state_manager", None)
    assert state_manager is not None

    request_ids = [f"paused-{policy}-slot{case.slot_idx}" for case in baselines[:paused_request_count]]
    prompt_objs: dict[str, Any] = {}
    sampling_params_by_id: dict[str, Any] = {}
    launch_times: dict[str, float] = {}
    initial_tasks: dict[str, asyncio.Task[tuple[Any, float]]] = {}

    for idx, case in enumerate(baselines[:paused_request_count]):
        prompt_obj, sampling_params = _prompt_and_sampling(
            engine,
            args,
            prompt=case.prompt,
            seed=case.seed,
        )
        request_id = request_ids[idx]
        prompt_objs[request_id] = prompt_obj
        sampling_params_by_id[request_id] = sampling_params
        launch_times[request_id] = time.perf_counter()
        initial_tasks[request_id] = asyncio.create_task(
            _run_request_with_finish_time(engine, _make_request(request_id, prompt_obj, sampling_params))
        )

    wait_tasks = {
        request_id: asyncio.create_task(
            _wait_for_checkpoint(
                engine,
                request_id=request_id,
                target_step=target_steps[idx],
                poll_interval=args.poll_interval,
                timeout_s=args.checkpoint_timeout,
            )
        )
        for idx, request_id in enumerate(request_ids)
    }

    checkpoint_states: dict[str, Any] = {}
    for request_id, wait_task in wait_tasks.items():
        checkpoint_states[request_id] = await wait_task
        engine.abort(request_id)

    for request_id, task in initial_tasks.items():
        output, _ = await task
        if not output.aborted:
            raise RuntimeError(f"Expected aborted output for {request_id}, got: {output!r}")

    if policy == "restart":
        for request_id in request_ids:
            state_manager.release_request(request_id)

    post_pause_mem = _gpu_mem_snapshot()
    pause_started_at = time.perf_counter()
    return (
        [
            {
                "request_id": request_id,
                "prompt_obj": prompt_objs[request_id],
                "sampling_params": sampling_params_by_id[request_id],
                "launch_time": launch_times[request_id],
                "checkpoint_state": checkpoint_states[request_id],
            }
            for request_id in request_ids
        ],
        post_pause_mem,
        {"pause_started_at": pause_started_at},
        {"request_ids": request_ids},
    )


async def _run_foreground_requests(
    engine: Any,
    args: argparse.Namespace,
    *,
    foreground_level: str,
    foreground_concurrency: int,
    prompts: list[str],
    seed_offset: int,
) -> list[ForegroundRequestResult]:
    request_ids = [f"foreground-{foreground_level}-slot{slot_idx}" for slot_idx in range(foreground_concurrency)]
    request_tasks: dict[str, asyncio.Task[tuple[Any, float]]] = {}
    admission_tasks: dict[str, asyncio.Task[tuple[float, Any]]] = {}
    launch_times: dict[str, float] = {}
    request_meta: dict[str, tuple[int, str, int]] = {}

    for slot_idx in range(foreground_concurrency):
        prompt = prompts[(seed_offset + slot_idx) % len(prompts)]
        seed = args.seed + 100_000 + seed_offset + slot_idx
        prompt_obj, sampling_params = _prompt_and_sampling(engine, args, prompt=prompt, seed=seed)
        request_id = request_ids[slot_idx]
        request_meta[request_id] = (slot_idx, prompt, seed)
        launch_times[request_id] = time.perf_counter()
        request_tasks[request_id] = asyncio.create_task(
            _run_request_with_finish_time(engine, _make_request(request_id, prompt_obj, sampling_params))
        )
        admission_tasks[request_id] = asyncio.create_task(
            _wait_for_admission(
                engine,
                request_id,
                poll_interval=args.poll_interval,
                timeout_s=args.foreground_admission_timeout,
            )
        )

    admission_results: dict[str, tuple[bool, float | None]] = {}
    for request_id, task in admission_tasks.items():
        try:
            admission_latency_sec, _state = await task
            admission_results[request_id] = (True, admission_latency_sec)
        except Exception:
            admission_results[request_id] = (False, None)
            engine.abort(request_id)

    results: list[ForegroundRequestResult] = []
    for request_id, task in request_tasks.items():
        admitted, admission_latency_sec = admission_results[request_id]
        slot_idx, prompt, seed = request_meta[request_id]
        try:
            output, finish_time = await task
            if output.error is not None:
                raise RuntimeError(str(output.error))
            if output.aborted:
                admitted = False
            total_latency_sec = finish_time - launch_times[request_id]
            error_text = None if admitted else "aborted_before_admission"
        except Exception as exc:
            total_latency_sec = None
            error_text = str(exc)
            admitted = False
        results.append(
            ForegroundRequestResult(
                request_id=request_id,
                slot_idx=slot_idx,
                prompt=prompt,
                seed=seed,
                target_step_idx=1,
                admitted=admitted,
                admission_latency_sec=admission_latency_sec,
                total_latency_sec=total_latency_sec,
                error_text=error_text,
            )
        )
    return results


async def _resume_background_requests(
    engine: Any,
    args: argparse.Namespace,
    *,
    policy: str,
    paused_records: list[dict[str, Any]],
    baselines: list[Any],
    artifact_dir: Path,
    paused_request_count: int,
    pause_duration_sec: float,
    foreground_level: str,
    foreground_concurrency: int,
    run_idx: int,
) -> list[PausedRequestResult]:
    resume_launch_times: dict[str, float] = {}
    resumed_tasks: dict[str, asyncio.Task[tuple[Any, float]]] = {}

    for idx, record in enumerate(paused_records):
        request_id = record["request_id"]
        if policy == "restart":
            launch_request = _make_request(
                f"{request_id}-restart",
                record["prompt_obj"],
                record["sampling_params"],
            )
        else:
            resume_template = _make_request(request_id, record["prompt_obj"], record["sampling_params"])
            launch_request = engine.restore_request_from_state(resume_template, request_id=request_id)
        resume_launch_times[request_id] = time.perf_counter()
        resumed_tasks[request_id] = asyncio.create_task(_run_request_with_finish_time(engine, launch_request))

    rows: list[PausedRequestResult] = []
    for idx, (record, case) in enumerate(zip(paused_records, baselines[:paused_request_count], strict=True)):
        request_id = record["request_id"]
        checkpoint_state = record["checkpoint_state"]
        resumed_output, finish_time = await resumed_tasks[request_id]
        if resumed_output.error is not None:
            raise RuntimeError(f"Paused request {request_id} failed after {policy}: {resumed_output.error}")
        output_image = _extract_first_image([resumed_output])
        output_path = (
            artifact_dir
            / "paused_resume"
            / policy
            / f"paused_{paused_request_count:02d}"
            / f"pause_{str(pause_duration_sec).replace('.', 'p')}"
            / foreground_level
            / f"run_{run_idx:02d}"
            / f"slot_{case.slot_idx:02d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(output_image, mode="RGB").save(output_path)
        metrics = _image_metrics(output_image, case.baseline_image)

        row = {
            "model": args.model,
            "run_idx": run_idx,
            "policy": policy,
            "paused_request_count": paused_request_count,
            "pause_duration_sec": pause_duration_sec,
            "foreground_level": foreground_level,
            "foreground_concurrency": foreground_concurrency,
            "request_role": "paused",
            "slot_idx": case.slot_idx,
            "prompt_id": case.slot_idx,
            "prompt": case.prompt,
            "seed": case.seed,
            "request_id": request_id,
            "target_step_idx": checkpoint_state.step_idx,
            "actual_step_idx": checkpoint_state.step_idx,
            "assigned_fidelity": checkpoint_state.fidelity.value,
            "placement": checkpoint_state.placement.value,
            "checkpoint_size_bytes": checkpoint_state.size_bytes,
            "checkpoint_value_score": checkpoint_state.value_score,
            "admitted": True,
            "admission_latency_sec": None,
            "total_latency_sec": finish_time - record["launch_time"],
            "resume_or_restart_latency_sec": finish_time - resume_launch_times[request_id],
            "output_ssim": metrics["output_ssim"],
            "output_mse": metrics["output_mse"],
            "exact_equal": metrics["exact_equal"],
        }
        rows.append(PausedRequestResult(request_row=row))

    return rows


def _aggregate_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["policy"],
            int(row["paused_request_count"]),
            float(row["pause_duration_sec"]),
            row["foreground_level"],
        )
        grouped[key].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][3], item[0][0])):
        policy, paused_request_count, pause_duration_sec, foreground_level = key
        foreground_latencies = [
            float(row["foreground_mean_latency_sec"])
            for row in group_rows
            if row["foreground_mean_latency_sec"] is not None
        ]
        aggregates.append(
            {
                "policy": policy,
                "paused_request_count": paused_request_count,
                "pause_duration_sec": pause_duration_sec,
                "foreground_level": foreground_level,
                "num_rows": len(group_rows),
                "mean_foreground_admission_rate": mean(float(row["foreground_admission_rate"]) for row in group_rows),
                "mean_foreground_latency_sec": mean(foreground_latencies) if foreground_latencies else None,
                "mean_foreground_success_rate": mean(float(row["foreground_success_rate"]) for row in group_rows),
                "mean_paused_resume_or_restart_latency_sec": mean(
                    float(row["paused_mean_resume_or_restart_latency_sec"]) for row in group_rows
                ),
                "mean_paused_output_ssim": mean(float(row["paused_mean_output_ssim"]) for row in group_rows),
                "mean_post_pause_gpu_free_bytes": mean(float(row["post_pause_gpu_free_bytes"]) for row in group_rows),
                "mean_post_foreground_gpu_free_bytes": mean(float(row["post_foreground_gpu_free_bytes"]) for row in group_rows),
            }
        )
    return aggregates


def _detect_gpu_resident_bottlenecks(batch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in batch_rows:
        key = (
            int(row["paused_request_count"]),
            float(row["pause_duration_sec"]),
            row["foreground_level"],
        )
        grouped[key][row["policy"]].append(row)

    findings: list[dict[str, Any]] = []
    for key, by_policy in grouped.items():
        if "gpu_resident" not in by_policy:
            continue
        if "cpu_lossless" not in by_policy and "cpu_value_aware" not in by_policy:
            continue

        gpu_rows = by_policy["gpu_resident"]
        cpu_candidates = by_policy.get("cpu_value_aware", []) + by_policy.get("cpu_lossless", [])
        gpu_admission = mean(float(row["foreground_admission_rate"]) for row in gpu_rows)
        gpu_latencies = [
            float(row["foreground_mean_latency_sec"])
            for row in gpu_rows
            if row["foreground_mean_latency_sec"] is not None
        ]
        gpu_latency = mean(gpu_latencies) if gpu_latencies else float("inf")
        gpu_free = mean(float(row["post_pause_gpu_free_bytes"]) for row in gpu_rows)

        cpu_admission = mean(float(row["foreground_admission_rate"]) for row in cpu_candidates)
        cpu_latencies = [
            float(row["foreground_mean_latency_sec"])
            for row in cpu_candidates
            if row["foreground_mean_latency_sec"] is not None
        ]
        cpu_latency = mean(cpu_latencies) if cpu_latencies else float("inf")
        cpu_free = mean(float(row["post_pause_gpu_free_bytes"]) for row in cpu_candidates)

        if gpu_admission + 1e-9 < cpu_admission or gpu_latency > cpu_latency * 1.05:
            paused_request_count, pause_duration_sec, foreground_level = key
            findings.append(
                {
                    "paused_request_count": paused_request_count,
                    "pause_duration_sec": pause_duration_sec,
                    "foreground_level": foreground_level,
                    "gpu_resident_mean_admission_rate": gpu_admission,
                    "cpu_offload_mean_admission_rate": cpu_admission,
                    "gpu_resident_mean_foreground_latency_sec": gpu_latency,
                    "cpu_offload_mean_foreground_latency_sec": cpu_latency,
                    "gpu_resident_post_pause_gpu_free_bytes": gpu_free,
                    "cpu_offload_post_pause_gpu_free_bytes": cpu_free,
                }
            )
    return findings


async def _run_condition(
    engine: Any,
    args: argparse.Namespace,
    *,
    run_idx: int,
    policy: str,
    paused_request_count: int,
    pause_duration_sec: float,
    foreground_level: str,
    prompts: list[str],
    baselines: list[Any],
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_manager = getattr(engine, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Diffusion state manager is disabled for this engine.")

    foreground_concurrency = _foreground_concurrency(foreground_level, args)
    target_steps = _phase_targets(paused_request_count, args.phase_min_step, args.phase_max_step)
    request_rows: list[dict[str, Any]] = []

    with _policy_override(engine, policy, args):
        pre_pause_mem = _gpu_mem_snapshot()
        paused_records, post_pause_mem, pause_meta, _extra = await _pause_background_requests(
            engine,
            args,
            policy=policy,
            paused_request_count=paused_request_count,
            baselines=baselines,
            target_steps=target_steps,
        )

        await asyncio.sleep(pause_duration_sec)

        foreground_results = await _run_foreground_requests(
            engine,
            args,
            foreground_level=foreground_level,
            foreground_concurrency=foreground_concurrency,
            prompts=prompts,
            seed_offset=run_idx * 1_000,
        )
        post_foreground_mem = _gpu_mem_snapshot()

        paused_results = await _resume_background_requests(
            engine,
            args,
            policy=policy,
            paused_records=paused_records,
            baselines=baselines,
            artifact_dir=artifact_dir,
            paused_request_count=paused_request_count,
            pause_duration_sec=pause_duration_sec,
            foreground_level=foreground_level,
            foreground_concurrency=foreground_concurrency,
            run_idx=run_idx,
        )
        post_resume_mem = _gpu_mem_snapshot()

    placements = Counter()
    fidelity_counts = Counter()
    checkpoint_sizes: list[int] = []
    paused_ssims: list[float] = []
    paused_mses: list[float] = []
    paused_exacts: list[float] = []
    paused_completion_latencies: list[float] = []
    paused_resume_latencies: list[float] = []
    d2h_bytes = 0
    h2d_bytes = 0

    for result in paused_results:
        row = result.request_row
        request_rows.append(row)
        placements[row["placement"]] += 1
        fidelity_counts[row["assigned_fidelity"]] += 1
        checkpoint_sizes.append(int(row["checkpoint_size_bytes"]))
        paused_ssims.append(float(row["output_ssim"]))
        paused_mses.append(float(row["output_mse"]))
        paused_exacts.append(1.0 if row["exact_equal"] else 0.0)
        paused_completion_latencies.append(float(row["total_latency_sec"]))
        paused_resume_latencies.append(float(row["resume_or_restart_latency_sec"]))
        request_d2h_bytes, request_h2d_bytes = _estimated_transfer_bytes(
            policy,
            Placement(row["placement"]),
            int(row["checkpoint_size_bytes"]),
        )
        d2h_bytes += request_d2h_bytes
        h2d_bytes += request_h2d_bytes

    foreground_latencies = [row.total_latency_sec for row in foreground_results if row.total_latency_sec is not None]
    foreground_admission_latencies = [
        row.admission_latency_sec for row in foreground_results if row.admission_latency_sec is not None
    ]
    foreground_success_rate = mean(1.0 if row.error_text is None else 0.0 for row in foreground_results)
    foreground_admission_rate = mean(1.0 if row.admitted else 0.0 for row in foreground_results)

    for foreground_result in foreground_results:
        request_rows.append(
            {
                "model": args.model,
                "run_idx": run_idx,
                "policy": policy,
                "paused_request_count": paused_request_count,
                "pause_duration_sec": pause_duration_sec,
                "foreground_level": foreground_level,
                "foreground_concurrency": foreground_concurrency,
                "request_role": "foreground",
                "slot_idx": foreground_result.slot_idx,
                "prompt_id": foreground_result.slot_idx,
                "prompt": foreground_result.prompt,
                "seed": foreground_result.seed,
                "request_id": foreground_result.request_id,
                "target_step_idx": foreground_result.target_step_idx,
                "actual_step_idx": None,
                "assigned_fidelity": None,
                "placement": None,
                "checkpoint_size_bytes": None,
                "checkpoint_value_score": None,
                "admitted": foreground_result.admitted,
                "admission_latency_sec": foreground_result.admission_latency_sec,
                "total_latency_sec": foreground_result.total_latency_sec,
                "resume_or_restart_latency_sec": None,
                "output_ssim": None,
                "output_mse": None,
                "exact_equal": None,
            }
        )

    batch_row = {
        "model": args.model,
        "run_idx": run_idx,
        "policy": policy,
        "paused_request_count": paused_request_count,
        "pause_duration_sec": pause_duration_sec,
        "foreground_level": foreground_level,
        "foreground_concurrency": foreground_concurrency,
        "phase_min_step": args.phase_min_step,
        "phase_max_step": args.phase_max_step,
        "pre_pause_gpu_free_bytes": pre_pause_mem["free_bytes"],
        "post_pause_gpu_free_bytes": post_pause_mem["free_bytes"],
        "post_foreground_gpu_free_bytes": post_foreground_mem["free_bytes"],
        "post_resume_gpu_free_bytes": post_resume_mem["free_bytes"],
        "paused_gpu_count": placements.get(Placement.GPU.value, 0),
        "paused_cpu_count": placements.get(Placement.CPU.value, 0),
        "paused_disk_count": placements.get(Placement.DISK.value, 0),
        "paused_total_checkpoint_bytes": sum(checkpoint_sizes),
        "paused_mean_checkpoint_bytes": mean(checkpoint_sizes) if checkpoint_sizes else 0.0,
        "paused_assigned_fidelity_counts": json.dumps(dict(sorted(fidelity_counts.items())), sort_keys=True),
        "paused_estimated_d2h_bytes": d2h_bytes,
        "paused_estimated_h2d_bytes": h2d_bytes,
        "foreground_admission_rate": foreground_admission_rate,
        "foreground_mean_admission_latency_sec": mean(foreground_admission_latencies)
        if foreground_admission_latencies
        else None,
        "foreground_mean_latency_sec": mean(foreground_latencies) if foreground_latencies else None,
        "foreground_max_latency_sec": max(foreground_latencies) if foreground_latencies else None,
        "foreground_success_rate": foreground_success_rate,
        "paused_mean_completion_latency_sec": mean(paused_completion_latencies),
        "paused_max_completion_latency_sec": max(paused_completion_latencies),
        "paused_mean_resume_or_restart_latency_sec": mean(paused_resume_latencies),
        "paused_mean_output_ssim": mean(paused_ssims),
        "paused_min_output_ssim": min(paused_ssims),
        "paused_mean_output_mse": mean(paused_mses),
        "paused_max_output_mse": max(paused_mses),
        "paused_exact_equal_rate": mean(paused_exacts),
    }
    return request_rows, batch_row


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompts(args)
    _preflight_real_model_args(args)

    if args.phase_min_step <= 0 or args.phase_max_step >= args.num_inference_steps:
        raise ValueError(
            "Phase steps must satisfy 1 <= phase_min_step < phase_max_step < num_inference_steps; "
            f"got min={args.phase_min_step}, max={args.phase_max_step}, total={args.num_inference_steps}."
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

        max_background = max(args.paused_request_counts)
        baseline_prompts = [prompts[idx % len(prompts)] for idx in range(max_background)]
        baselines = await _build_background_baselines(
            engine,
            args,
            baseline_prompts,
            artifact_dir,
            count=max_background,
        )

        for paused_request_count in args.paused_request_counts:
            for pause_duration_sec in args.pause_durations_sec:
                for foreground_level in args.foreground_levels:
                    for policy in args.policies:
                        for run_idx in range(args.runs_per_condition):
                            condition_request_rows, batch_row = await _run_condition(
                                engine,
                                args,
                                run_idx=run_idx,
                                policy=policy,
                                paused_request_count=paused_request_count,
                                pause_duration_sec=pause_duration_sec,
                                foreground_level=foreground_level,
                                prompts=prompts,
                                baselines=baselines,
                                artifact_dir=artifact_dir,
                            )
                            for row in condition_request_rows:
                                request_rows.append(row)
                                _append_csv_row(request_csv_path, row, REQUEST_CSV_FIELDS)
                            batch_rows.append(batch_row)
                            _append_csv_row(batch_csv_path, batch_row, BATCH_CSV_FIELDS)
                            print(
                                f"[{policy}] paused={paused_request_count} pause_s={pause_duration_sec:.3f} "
                                f"foreground={foreground_level}:{batch_row['foreground_concurrency']} "
                                f"admission={batch_row['foreground_admission_rate']:.3f} "
                                f"fg_latency={batch_row['foreground_mean_latency_sec']} "
                                f"paused_ssim={batch_row['paused_mean_output_ssim']:.6f} "
                                f"placements=gpu:{batch_row['paused_gpu_count']} "
                                f"cpu:{batch_row['paused_cpu_count']} disk:{batch_row['paused_disk_count']}",
                                flush=True,
                            )
    finally:
        omni.close()

    summary = {
        "model": args.model,
        "num_inference_steps": args.num_inference_steps,
        "policies": args.policies,
        "paused_request_counts": args.paused_request_counts,
        "pause_durations_sec": args.pause_durations_sec,
        "foreground_levels": args.foreground_levels,
        "foreground_low_concurrency": args.foreground_low_concurrency,
        "foreground_high_concurrency": args.foreground_high_concurrency,
        "runs_per_condition": args.runs_per_condition,
        "phase_min_step": args.phase_min_step,
        "phase_max_step": args.phase_max_step,
        "cpu_budget_bytes": args.cpu_budget_bytes,
        "gpu_resident_budget_bytes": args.gpu_resident_budget_bytes,
        "restart_observation_budget_bytes": args.restart_observation_budget_bytes,
        "init_config": init_config,
        "request_csv_path": str(request_csv_path),
        "batch_csv_path": str(batch_csv_path),
        "artifact_dir": str(artifact_dir),
        "batch_aggregates": _aggregate_batch_rows(batch_rows),
        "gpu_resident_bottleneck_conditions": [],
        "go_no_go_detected": False,
        "num_request_rows": len(request_rows),
        "num_batch_rows": len(batch_rows),
    }
    summary["gpu_resident_bottleneck_conditions"] = _detect_gpu_resident_bottlenecks(batch_rows)
    summary["go_no_go_detected"] = bool(summary["gpu_resident_bottleneck_conditions"])
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_experiment(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
