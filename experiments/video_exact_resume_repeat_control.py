#!/usr/bin/env python3
"""Measure run-to-run trajectory divergence from one fixed Wan checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight
from experiments import video_denoising_error_correction_killtest as correction
from experiments import video_state_protection_killtest as protection


TRAJECTORY_FIELDS = [
    "repeat_a",
    "repeat_b",
    "checkpoint_step",
    "denoising_step",
    "remaining_step_index",
    "latent_mse",
    "normalized_l2",
    "cosine_similarity",
    "latent_path_a",
    "latent_path_b",
]
FINAL_FIELDS = [
    "repeat_a",
    "repeat_b",
    "video_mse",
    "video_max_abs_diff",
    "spatial_quality",
    "temporal_shape_quality",
    "temporal_dynamic_quality",
    "motion_energy_ratio",
    "flow_magnitude_cosine",
    "flow_magnitude_ratio",
    "flow_direction_cosine",
    "flicker_similarity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="results/video_denoising_error_correction_killtest",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    return parser.parse_args()


def classify_control(
    pairwise_final_errors: list[float],
    *,
    small_initial_error: float,
    medium_initial_error: float,
    uninterrupted_resume_error: float,
) -> dict[str, Any]:
    if not pairwise_final_errors:
        raise ValueError("At least one repeat pair is required")
    sorted_errors = sorted(pairwise_final_errors)
    midpoint = len(sorted_errors) // 2
    if len(sorted_errors) % 2:
        median_error = sorted_errors[midpoint]
    else:
        median_error = (sorted_errors[midpoint - 1] + sorted_errors[midpoint]) / 2.0
    max_error = max(sorted_errors)
    small_ratio = median_error / max(small_initial_error, 1e-12)
    medium_ratio = median_error / max(medium_initial_error, 1e-12)
    repeat_stable = max_error <= 1e-5
    if repeat_stable and uninterrupted_resume_error > 1e-3:
        diagnosis = "RESUME-PATH MISMATCH"
    elif small_ratio >= 1.0:
        diagnosis = "SMALL-ERROR NOISE FLOOR DOMINATES"
    elif small_ratio >= 0.25:
        diagnosis = "SMALL-ERROR MATERIALLY CONFOUNDED"
    else:
        diagnosis = "REPEAT NOISE BELOW SMALL-ERROR CONTROL LIMIT"
    return {
        "pairwise_final_normalized_l2_median": median_error,
        "pairwise_final_normalized_l2_max": max_error,
        "noise_floor_over_small_initial_error": small_ratio,
        "noise_floor_over_medium_initial_error": medium_ratio,
        "repeat_stable_at_1e_5": repeat_stable,
        "small_error_materially_confounded": small_ratio >= 0.25,
        "small_error_noise_floor_dominates": small_ratio >= 1.0,
        "medium_error_materially_confounded": medium_ratio >= 0.25,
        "diagnosis": diagnosis,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _runtime_args(config: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=config["model"],
        height=int(config["height"]),
        width=int(config["width"]),
        num_frames=int(config["num_frames"]),
        num_inference_steps=int(config["num_inference_steps"]),
        guidance_scale=float(config["guidance_scale"]),
        fps=float(config["fps"]),
        flow_shift=float(config["flow_shift"]),
        boundary_ratio=float(config["boundary_ratio"]),
        enable_cpu_offload=bool(config["enable_cpu_offload"]),
        enable_layerwise_offload=bool(config["enable_layerwise_offload"]),
        enforce_eager=False,
        init_timeout=cli.init_timeout,
        stage_init_timeout=cli.stage_init_timeout,
    )


def _load_sources(
    source_dir: Path,
    checkpoint_step: int,
) -> tuple[dict[str, Any], dict[str, Any], int, torch.Tensor, dict[str, Any], np.ndarray]:
    config = json.loads((source_dir / "perturbation_config.json").read_text(encoding="utf-8"))
    summary = json.loads((source_dir / "error_contraction_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete") or int(summary.get("completed_conditions", 0)) != 8:
        raise ValueError("The source smoke experiment is incomplete")
    if config.get("stage") != "smoke" or checkpoint_step not in config.get("checkpoint_steps", []):
        raise ValueError("The source is not the registered step-20 smoke experiment")
    entry = config["prompts"][0]
    seed = int(config["seed"])
    artifact_dir = source_dir / "artifacts" / f"{entry['prompt_id']}_seed{seed}"
    baseline_label = f"{entry['prompt_id']}_seed{seed}_uninterrupted"
    baseline_meta = json.loads(
        (artifact_dir / f"{baseline_label}_trajectory_probe.json").read_text(encoding="utf-8")
    )
    records = correction._metadata_records(baseline_meta)
    latent_path = Path(records[checkpoint_step]["latent_path"])
    if not latent_path.exists():
        raise FileNotFoundError(latent_path)
    latent = torch.load(latent_path, map_location="cpu")
    baseline_video = np.load(artifact_dir / f"{baseline_label}.npy", allow_pickle=False)
    return config, entry, seed, latent, baseline_meta, baseline_video


def _load_or_run_repeat(
    omni: Any | None,
    runtime_args: argparse.Namespace,
    entry: dict[str, Any],
    seed: int,
    checkpoint_step: int,
    latent: torch.Tensor,
    artifact_dir: Path,
    repeat_index: int,
    resume: bool,
) -> tuple[np.ndarray | None, dict[str, Any] | None, Any | None]:
    remaining = runtime_args.num_inference_steps - checkpoint_step
    label = f"{entry['prompt_id']}_seed{seed}_step{checkpoint_step:02d}_exact_repeat{repeat_index}"
    metadata_path = artifact_dir / f"{label}_trajectory_probe.json"
    video_path = artifact_dir / f"{label}.npy"
    cached = correction._valid_probe(metadata_path, remaining) if resume else None
    if cached is not None and video_path.exists():
        return np.load(video_path, allow_pickle=False), cached, omni
    if omni is None:
        omni = protection._make_omni(runtime_args)
    video, metadata, _ = correction._run_with_probe(
        omni,
        runtime_args,
        entry["prompt"],
        seed,
        latent.clone(),
        checkpoint_step,
        artifact_dir,
        label,
    )
    np.save(video_path, video, allow_pickle=False)
    preflight._save_video(artifact_dir / f"{label}.mp4", video, fps=runtime_args.fps)
    return video, metadata, omni


def _pairwise_trajectory_rows(
    checkpoint_step: int,
    repeat_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for repeat_a, repeat_b in combinations(range(len(repeat_metadata)), 2):
        records_a = correction._metadata_records(repeat_metadata[repeat_a])
        records_b = correction._metadata_records(repeat_metadata[repeat_b])
        if set(records_a) != set(records_b):
            raise ValueError(f"Repeat trajectory mismatch: {repeat_a} vs {repeat_b}")
        for local_step in sorted(records_a):
            path_a = Path(records_a[local_step]["latent_path"])
            path_b = Path(records_b[local_step]["latent_path"])
            latent_a = torch.load(path_a, map_location="cpu")
            latent_b = torch.load(path_b, map_location="cpu")
            metrics = correction.latent_error(latent_a, latent_b)
            output.append(
                {
                    "repeat_a": repeat_a,
                    "repeat_b": repeat_b,
                    "checkpoint_step": checkpoint_step,
                    "denoising_step": checkpoint_step + local_step,
                    "remaining_step_index": local_step,
                    "latent_mse": metrics["mse"],
                    "normalized_l2": metrics["normalized_l2"],
                    "cosine_similarity": metrics["cosine_similarity"],
                    "latent_path_a": str(path_a),
                    "latent_path_b": str(path_b),
                }
            )
    return output


def _pairwise_final_rows(videos: list[np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for repeat_a, repeat_b in combinations(range(len(videos)), 2):
        metrics = correction._quality_metrics(
            videos[repeat_a], videos[repeat_b], "", None, float("nan")
        )
        output.append(
            {
                "repeat_a": repeat_a,
                "repeat_b": repeat_b,
                "video_mse": metrics["video_mse_vs_uninterrupted"],
                "video_max_abs_diff": metrics["video_max_abs_diff_vs_uninterrupted"],
                "spatial_quality": metrics["spatial_quality"],
                "temporal_shape_quality": metrics["temporal_shape_quality"],
                "temporal_dynamic_quality": metrics["temporal_dynamic_quality"],
                "motion_energy_ratio": metrics["motion_energy_ratio"],
                "flow_magnitude_cosine": metrics["flow_magnitude_cosine"],
                "flow_magnitude_ratio": metrics["flow_magnitude_ratio"],
                "flow_direction_cosine": metrics["flow_direction_cosine"],
                "flicker_similarity": metrics["flicker_similarity"],
            }
        )
    return output


def _source_error_levels(source_dir: Path) -> tuple[float, float]:
    rows = _read_csv(source_dir / "final_recovery_quality.csv")
    values = {
        row["error_strength"]: float(row["initial_normalized_l2"])
        for row in rows
        if row["error_family"] == "quantization"
    }
    return values["small"], values["medium"]


def _uninterrupted_resume_error(source_dir: Path, checkpoint_step: int) -> float:
    rows = _read_csv(source_dir / "exact_resume_trajectory_control.csv")
    selected = [
        float(row["normalized_l2"])
        for row in rows
        if int(row["checkpoint_step"]) == checkpoint_step
    ]
    return selected[-1]


def _plot(output_dir: Path, rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    pairs = sorted({(int(row["repeat_a"]), int(row["repeat_b"])) for row in rows})
    for pair in pairs:
        selected = [
            row for row in rows
            if (int(row["repeat_a"]), int(row["repeat_b"])) == pair
        ]
        selected.sort(key=lambda row: int(row["denoising_step"]))
        axis.plot(
            [int(row["denoising_step"]) for row in selected],
            [float(row["normalized_l2"]) for row in selected],
            marker="o",
            markersize=2.5,
            label=f"repeat {pair[0]} vs {pair[1]}",
        )
    axis.set_xlabel("Global denoising step")
    axis.set_ylabel("Pairwise normalized L2")
    axis.set_title("Exact-resume repeat divergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "exact_resume_repeat_divergence.pdf"
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exact-Resume Repeat Control",
        "",
        "## CONFIRMED",
        "",
        f"- All `{summary['repeats']}` resumes used the same checkpoint SHA256: "
        f"`{summary['checkpoint_sha256']}`.",
        f"- Median pairwise final normalized L2: "
        f"`{summary['control']['pairwise_final_normalized_l2_median']:.6g}`.",
        f"- Uninterrupted vs exact-resume final normalized L2: "
        f"`{summary['uninterrupted_resume_final_normalized_l2']:.6g}`.",
        "",
        "## INTERPRETATION",
        "",
        f"- Small-error floor ratio: `{summary['control']['noise_floor_over_small_initial_error']:.3f}`.",
        f"- Medium-error floor ratio: `{summary['control']['noise_floor_over_medium_initial_error']:.3f}`.",
        f"- Diagnosis: **{summary['control']['diagnosis']}**.",
        "",
        "This control diagnoses measurement validity only. It does not establish denoising correction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_control(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "0,"}:
        raise EnvironmentError("Run with CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EnvironmentError("Exactly one CUDA device must be visible")

    source_dir = Path(args.source_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else source_dir / "exact_repeat_control"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config, entry, seed, latent, _, _ = _load_sources(source_dir, args.checkpoint_step)
    runtime_args = _runtime_args(config, args)
    source_records = correction._metadata_records(
        json.loads(
            (
                source_dir
                / "artifacts"
                / f"{entry['prompt_id']}_seed{seed}"
                / f"{entry['prompt_id']}_seed{seed}_uninterrupted_trajectory_probe.json"
            ).read_text(encoding="utf-8")
        )
    )
    checkpoint_path = Path(source_records[args.checkpoint_step]["latent_path"])
    preregistration = {
        "purpose": "repeat exact resume from one fixed checkpoint without perturbation",
        "source_dir": str(source_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": args.checkpoint_step,
        "repeats": args.repeats,
        "small_error_material_threshold": 0.25,
        "small_error_dominance_threshold": 1.0,
        "repeat_stability_threshold": 1e-5,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    config_path = output_dir / "exact_repeat_control_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != preregistration:
        raise ValueError(f"Existing control configuration differs: {config_path}")
    correction._atomic_json(config_path, preregistration)

    videos = []
    metadata = []
    omni = None
    try:
        for repeat_index in range(args.repeats):
            video, repeat_meta, omni = _load_or_run_repeat(
                omni,
                runtime_args,
                entry,
                seed,
                args.checkpoint_step,
                latent,
                artifact_dir,
                repeat_index,
                args.resume,
            )
            if video is None or repeat_meta is None:
                raise RuntimeError(f"Repeat {repeat_index} did not produce output")
            videos.append(video)
            metadata.append(repeat_meta)
            print(f"[exact-repeat-control] complete repeat={repeat_index}", flush=True)
    finally:
        if omni is not None and hasattr(omni, "shutdown"):
            omni.shutdown()

    trajectory_rows = _pairwise_trajectory_rows(args.checkpoint_step, metadata)
    final_rows = _pairwise_final_rows(videos)
    correction._atomic_csv(
        output_dir / "exact_resume_repeat_trajectory.csv",
        TRAJECTORY_FIELDS,
        trajectory_rows,
    )
    correction._atomic_csv(
        output_dir / "exact_resume_repeat_final_quality.csv",
        FINAL_FIELDS,
        final_rows,
    )
    final_step = runtime_args.num_inference_steps
    final_errors = [
        float(row["normalized_l2"])
        for row in trajectory_rows
        if int(row["denoising_step"]) == final_step
    ]
    initial_pairwise_errors = [
        float(row["normalized_l2"])
        for row in trajectory_rows
        if int(row["denoising_step"]) == args.checkpoint_step
    ]
    small_error, medium_error = _source_error_levels(source_dir)
    uninterrupted_error = _uninterrupted_resume_error(source_dir, args.checkpoint_step)
    control = classify_control(
        final_errors,
        small_initial_error=small_error,
        medium_initial_error=medium_error,
        uninterrupted_resume_error=uninterrupted_error,
    )
    summary = {
        "repeats": args.repeats,
        "pair_count": math.comb(args.repeats, 2),
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_sha256": preregistration["checkpoint_sha256"],
        "max_pairwise_initial_normalized_l2": max(initial_pairwise_errors),
        "small_initial_error": small_error,
        "medium_initial_error": medium_error,
        "uninterrupted_resume_final_normalized_l2": uninterrupted_error,
        "control": control,
        "pairwise_final_quality": final_rows,
        "figure_path": _plot(output_dir, trajectory_rows),
    }
    correction._atomic_json(output_dir / "exact_resume_repeat_summary.json", summary)
    _write_report(output_dir / "exact_resume_repeat_control.md", summary)
    return summary


def main() -> None:
    summary = run_control(parse_args())
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
