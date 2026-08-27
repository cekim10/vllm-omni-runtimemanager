#!/usr/bin/env python3
"""Kill test for structure-dependent correction of checkpoint-state errors.

This experiment perturbs only a serialized/resumed Wan latent. It does not
quantize model weights or normal inference activations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight
from experiments import video_state_protection_killtest as protection


MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
ERROR_FAMILIES = ("quantization", "spatial_lowpass", "temporal_lowpass", "random_noise")
STRENGTHS = ("small", "medium")

TRAJECTORY_FIELDS = [
    "prompt_id", "motion_category", "seed", "checkpoint_step", "error_family",
    "error_strength", "quantization_bits", "denoising_step", "remaining_step_index",
    "initial_latent_mse", "initial_normalized_l2", "initial_cosine_similarity",
    "latent_mse", "normalized_l2", "cosine_similarity", "normalized_contraction",
    "reference_latent_path", "perturbed_latent_path",
]
FINAL_FIELDS = [
    "prompt_id", "motion_category", "seed", "checkpoint_step", "error_family",
    "error_strength", "quantization_bits", "target_latent_mse", "target_normalized_l2",
    "initial_latent_mse", "initial_normalized_l2", "initial_cosine_similarity",
    "relative_mse_mismatch", "relative_error_mismatch", "calibration_scale", "resume_latency_ms",
    "final_latent_mse", "final_normalized_l2", "final_cosine_similarity",
    "final_error_ratio", "contraction_fraction", "trajectory_monotonicity_fraction",
    "first_half_step", "persistent_half_life_steps", "correctability_class",
    "video_mse_vs_uninterrupted", "video_max_abs_diff_vs_uninterrupted",
    "spatial_quality", "temporal_shape_quality", "temporal_dynamic_quality",
    "video_mse_vs_exact_resume", "video_max_abs_diff_vs_exact_resume",
    "spatial_quality_vs_exact_resume", "temporal_shape_quality_vs_exact_resume",
    "temporal_dynamic_quality_vs_exact_resume",
    "motion_energy_ratio", "flow_magnitude_cosine", "flow_magnitude_ratio",
    "flow_direction_cosine", "flicker_similarity", "semantic_quality",
    "semantic_quality_vs_uninterrupted", "artifact_path", "trajectory_metadata_path",
]
ISO_FIELDS = [
    "prompt_id", "checkpoint_step", "error_strength", "family_a", "family_b",
    "initial_mse_a", "initial_mse_b", "relative_mse_mismatch",
    "initial_normalized_l2_a", "initial_normalized_l2_b", "relative_error_mismatch",
    "within_matching_tolerance", "contraction_fraction_a", "contraction_fraction_b",
    "contraction_delta_a_minus_b", "temporal_dynamic_a", "temporal_dynamic_b",
    "temporal_dynamic_delta_a_minus_b", "spatial_a", "spatial_b",
]


@dataclass(frozen=True)
class Perturbation:
    family: str
    strength: str
    tensor: torch.Tensor
    target_mse: float
    target_normalized_l2: float
    actual_mse: float
    actual_normalized_l2: float
    cosine_similarity: float
    relative_mismatch: float
    calibration_scale: float
    quantization_bits: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--prompt-set",
        default="experiments/video_denoising_error_correction_prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/video_denoising_error_correction_killtest",
    )
    parser.add_argument("--stage", choices=("smoke", "progress", "content"), default="smoke")
    parser.add_argument("--prior-summary")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--sample-solver", choices=("unipc", "euler"), default="unipc")
    parser.add_argument(
        "--require-exact-resume",
        action="store_true",
        help="Abort before perturbation runs if uninterrupted and resumed trajectories differ.",
    )
    parser.add_argument("--exact-resume-tolerance", type=float, default=1e-6)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--iso-error-tolerance", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def latent_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        raise ValueError(f"Latent shape mismatch: {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    mse = float(torch.mean(delta.square()).item())
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    error_norm = float(torch.linalg.vector_norm(delta).item())
    normalized_l2 = error_norm / max(ref_norm, 1e-12)
    cosine = float(F.cosine_similarity(reference.reshape(1, -1), candidate.reshape(1, -1)).item())
    return {"mse": mse, "normalized_l2": normalized_l2, "cosine_similarity": cosine}


def symmetric_quantize_dequantize(latent: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2 or bits > 16:
        raise ValueError(f"Unsupported checkpoint quantization bit depth: {bits}")
    source = latent.detach().float().cpu()
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(source.abs().max().item())
    if max_abs == 0.0:
        return source.clone()
    scale = max_abs / qmax
    quantized = torch.round(source / scale).clamp(-qmax, qmax)
    return (quantized * scale).contiguous()


def _spatial_lowpass(latent: torch.Tensor) -> torch.Tensor:
    source = latent.detach().float().cpu()
    _, _, temporal, height, width = source.shape
    down = F.interpolate(
        source,
        size=(temporal, max(1, math.ceil(height / 2)), max(1, math.ceil(width / 2))),
        mode="trilinear",
        align_corners=False,
    )
    return F.interpolate(down, size=(temporal, height, width), mode="trilinear", align_corners=False)


def _temporal_lowpass(latent: torch.Tensor) -> torch.Tensor:
    source = latent.detach().float().cpu()
    _, _, temporal, height, width = source.shape
    down = F.interpolate(
        source,
        size=(max(1, math.ceil(temporal / 2)), height, width),
        mode="trilinear",
        align_corners=False,
    )
    return F.interpolate(down, size=(temporal, height, width), mode="trilinear", align_corners=False)


def calibrate_direction(
    latent: torch.Tensor,
    direction: torch.Tensor,
    target_normalized_l2: float,
) -> tuple[torch.Tensor, float]:
    source = latent.detach().float().cpu()
    direction = direction.detach().float().cpu()
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    source_norm = float(torch.linalg.vector_norm(source).item())
    if direction_norm <= 1e-12:
        raise ValueError("Cannot calibrate a zero perturbation direction")
    scale = target_normalized_l2 * source_norm / direction_norm
    return (source + direction * scale).contiguous(), float(scale)


def build_perturbations(
    latent: torch.Tensor,
    *,
    random_seed: int,
    matching_tolerance: float,
) -> list[Perturbation]:
    source = latent.detach().float().cpu().contiguous()
    quantized = {
        "small": (8, symmetric_quantize_dequantize(source, 8)),
        "medium": (4, symmetric_quantize_dequantize(source, 4)),
    }
    targets = {
        strength: latent_error(source, candidate)["normalized_l2"]
        for strength, (_, candidate) in quantized.items()
    }
    target_mse = {
        strength: latent_error(source, candidate)["mse"]
        for strength, (_, candidate) in quantized.items()
    }
    if targets["small"] <= 0.0 or targets["medium"] <= targets["small"]:
        raise ValueError(f"Invalid quantization-derived targets: {targets}")

    spatial_direction = _spatial_lowpass(source) - source
    temporal_direction = _temporal_lowpass(source) - source
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random_direction = torch.randn(source.shape, generator=generator, dtype=source.dtype)

    perturbations = []
    for strength in STRENGTHS:
        target = targets[strength]
        bits, quantized_tensor = quantized[strength]
        candidates = {"quantization": (quantized_tensor, 1.0, bits)}
        for family, direction in (
            ("spatial_lowpass", spatial_direction),
            ("temporal_lowpass", temporal_direction),
            ("random_noise", random_direction),
        ):
            calibrated, scale = calibrate_direction(source, direction, target)
            candidates[family] = (calibrated, scale, None)

        for family in ERROR_FAMILIES:
            candidate, scale, candidate_bits = candidates[family]
            metrics = latent_error(source, candidate)
            mismatch = abs(metrics["normalized_l2"] - target) / max(target, 1e-12)
            if mismatch > matching_tolerance:
                raise ValueError(
                    f"Iso-error calibration failed for {family}/{strength}: "
                    f"target={target:.8g}, actual={metrics['normalized_l2']:.8g}, mismatch={mismatch:.3%}"
                )
            perturbations.append(
                Perturbation(
                    family=family,
                    strength=strength,
                    tensor=candidate,
                    target_mse=target_mse[strength],
                    target_normalized_l2=target,
                    actual_mse=metrics["mse"],
                    actual_normalized_l2=metrics["normalized_l2"],
                    cosine_similarity=metrics["cosine_similarity"],
                    relative_mismatch=mismatch,
                    calibration_scale=scale,
                    quantization_bits=candidate_bits,
                )
            )
    return perturbations


def contraction_statistics(ratios: list[float]) -> dict[str, Any]:
    if not ratios or ratios[0] <= 0.0:
        raise ValueError("Contraction ratios require a positive initial error")
    transitions = [later <= earlier + 1e-6 for earlier, later in zip(ratios, ratios[1:])]
    monotonicity = sum(transitions) / max(len(transitions), 1)
    first_half = next((index for index, value in enumerate(ratios) if value <= 0.5), None)
    persistent_half = next(
        (index for index, value in enumerate(ratios) if value <= 0.5 and all(x <= 0.5 for x in ratios[index:])),
        None,
    )
    contraction = 1.0 - ratios[-1]
    if contraction >= 0.5:
        classification = "highly_correctable"
    elif contraction >= 0.1:
        classification = "partially_correctable"
    else:
        classification = "persistent_or_amplifying"
    return {
        "final_error_ratio": ratios[-1],
        "contraction_fraction": contraction,
        "trajectory_monotonicity_fraction": monotonicity,
        "first_half_step": first_half,
        "persistent_half_life_steps": persistent_half if monotonicity >= 0.8 else None,
        "correctability_class": classification,
    }


def _read_prompts(path_string: str) -> tuple[Path, str, list[dict[str, str]]]:
    path = Path(path_string)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    raw = path.read_text(encoding="utf-8")
    entries = json.loads(raw)
    if not isinstance(entries, list) or len(entries) < 3:
        raise ValueError("The preregistered prompt set must contain at least three prompts")
    required = {"prompt_id", "motion_category", "prompt"}
    if any(not required <= set(entry) for entry in entries):
        raise ValueError(f"Malformed prompt set: {path}")
    return path, hashlib.sha256(raw.encode("utf-8")).hexdigest(), entries


def _stage_scope(stage: str, entries: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[int]]:
    if stage == "smoke":
        return entries[:1], [20]
    if stage == "progress":
        return entries[:1], [10, 20, 30]
    return entries[:3], [10, 20, 30]


def _check_stage_gate(args: argparse.Namespace) -> None:
    if args.stage == "smoke":
        return
    if not args.prior_summary:
        raise ValueError(f"--stage {args.stage} requires --prior-summary")
    summary = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))
    required = "progress" if args.stage == "progress" else "content"
    if summary.get("eligible_next_stage") != required:
        raise ValueError(
            f"Stage {args.stage} blocked by prior summary: eligible_next_stage="
            f"{summary.get('eligible_next_stage')!r}"
        )


def _validate_runtime_config(args: argparse.Namespace) -> None:
    expected = {
        "model": MODEL,
        "height": 480,
        "width": 832,
        "num_frames": 33,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "fps": 16.0,
        "flow_shift": 12.0,
        "boundary_ratio": 0.875,
        "enable_cpu_offload": True,
        "enable_layerwise_offload": False,
        "iso_error_tolerance": 0.05,
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Kill-test configuration changed from preregistration: {actual} != {expected}")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validated_resume_rows(
    final_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    total_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory_by_key: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        key = (
            str(row.get("prompt_id")),
            int(row.get("checkpoint_step", -1)),
            str(row.get("error_family")),
            str(row.get("error_strength")),
        )
        trajectory_by_key.setdefault(key, []).append(row)

    valid_final = []
    valid_keys = set()
    for row in final_rows:
        try:
            key = (
                str(row["prompt_id"]),
                int(row["checkpoint_step"]),
                str(row["error_family"]),
                str(row["error_strength"]),
            )
            checkpoint_step = key[1]
            expected_steps = total_steps - checkpoint_step + 1
            matching_rows = trajectory_by_key.get(key, [])
            metadata_path = Path(str(row["trajectory_metadata_path"]))
            artifact_path = Path(str(row["artifact_path"]))
            metadata = _valid_probe(metadata_path, total_steps - checkpoint_step)
            observed_steps = {int(item["denoising_step"]) for item in matching_rows}
            expected_global_steps = set(range(checkpoint_step, total_steps + 1))
            numeric_values = (
                float(row["initial_normalized_l2"]),
                float(row["final_normalized_l2"]),
                float(row["contraction_fraction"]),
                float(row["temporal_dynamic_quality"]),
                float(row["temporal_dynamic_quality_vs_exact_resume"]),
            )
            paths_valid = all(
                Path(str(item["reference_latent_path"])).exists()
                and Path(str(item["perturbed_latent_path"])).exists()
                for item in matching_rows
            )
            valid = (
                artifact_path.exists()
                and metadata is not None
                and len(matching_rows) == expected_steps
                and observed_steps == expected_global_steps
                and paths_valid
                and all(math.isfinite(value) for value in numeric_values)
            )
        except (KeyError, TypeError, ValueError, OSError):
            valid = False
        if valid and key not in valid_keys:
            valid_final.append(row)
            valid_keys.add(key)

    valid_trajectory = []
    for key in valid_keys:
        valid_trajectory.extend(trajectory_by_key[key])
    valid_trajectory.sort(
        key=lambda row: (
            str(row["prompt_id"]),
            int(row["checkpoint_step"]),
            str(row["error_strength"]),
            str(row["error_family"]),
            int(row["denoising_step"]),
        )
    )
    valid_final.sort(
        key=lambda row: (
            str(row["prompt_id"]),
            int(row["checkpoint_step"]),
            str(row["error_strength"]),
            str(row["error_family"]),
        )
    )
    return valid_final, valid_trajectory


def _probe_sampling(
    args: argparse.Namespace,
    *,
    seed: int,
    latents: torch.Tensor | None,
    checkpoint_step: int,
    artifact_dir: Path,
    label: str,
) -> Any:
    if latents is None:
        sampling = protection._build_probe_sampling_params(
            args,
            seed=seed,
            artifact_dir=artifact_dir,
            request_label=label,
            checkpoint_steps=list(range(args.num_inference_steps + 1)),
        )
    else:
        sampling = protection._build_resume_sampling_params(
            args, seed=seed, checkpoint_step=checkpoint_step, latents=latents
        )
        remaining = args.num_inference_steps - checkpoint_step
        sampling.extra_args = {
            "flow_shift": args.flow_shift,
            "sample_solver": args.sample_solver,
            "trajectory_probe": {
                "artifact_dir": str(artifact_dir),
                "request_label": label,
                "capture_steps": list(range(remaining + 1)),
                "fps": args.fps,
                "save_decoded": False,
                "save_latents": True,
                "save_mp4": False,
            },
        }
    sampling.extra_args = dict(sampling.extra_args or {})
    sampling.extra_args["sample_solver"] = args.sample_solver
    return sampling


def _require_exact_resume(
    trajectory_rows: list[dict[str, Any]],
    quality: dict[str, float],
    tolerance: float,
) -> None:
    max_latent_error = max(float(row["normalized_l2"]) for row in trajectory_rows)
    video_mse = float(quality["video_mse_vs_uninterrupted"])
    if max_latent_error > tolerance or video_mse > tolerance:
        raise RuntimeError(
            "Exact-resume validation failed before perturbation runs: "
            f"max trajectory normalized L2={max_latent_error:.6g}, "
            f"video MSE={video_mse:.6g}, tolerance={tolerance:.6g}."
        )


def _run_with_probe(
    omni: Any,
    args: argparse.Namespace,
    prompt: str,
    seed: int,
    latents: torch.Tensor | None,
    checkpoint_step: int,
    artifact_dir: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], float]:
    sampling = _probe_sampling(
        args,
        seed=seed,
        latents=latents,
        checkpoint_step=checkpoint_step,
        artifact_dir=artifact_dir,
        label=label,
    )
    start = time.perf_counter()
    outputs = omni.generate({"prompt": prompt}, sampling)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    video, output = protection._normalize_output_video(outputs)
    metadata_path = output.custom_output.get("trajectory_probe_metadata_path")
    if not metadata_path:
        raise ValueError(f"Missing trajectory metadata for {label}")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return video, metadata, elapsed_ms


def _metadata_records(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["step_index"]): row for row in metadata.get("records", [])}


def _valid_probe(metadata_path: Path, expected_steps: int) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = _metadata_records(metadata)
        if set(records) != set(range(expected_steps + 1)):
            return None
        for record in records.values():
            latent_path = record.get("latent_path")
            if not latent_path or not Path(latent_path).exists():
                return None
        return metadata
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _load_or_run_baseline(
    omni: Any,
    args: argparse.Namespace,
    entry: dict[str, str],
    seed: int,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    label = f"{entry['prompt_id']}_seed{seed}_uninterrupted"
    frames_path = artifact_dir / f"{label}.npy"
    metadata_path = artifact_dir / f"{label}_trajectory_probe.json"
    cached = _valid_probe(metadata_path, args.num_inference_steps) if args.resume else None
    if cached is not None and frames_path.exists():
        return np.load(frames_path, allow_pickle=False), cached
    video, metadata, _ = _run_with_probe(
        omni, args, entry["prompt"], seed, None, 0, artifact_dir, label
    )
    np.save(frames_path, video, allow_pickle=False)
    preflight._save_video(artifact_dir / f"{label}.mp4", video, fps=args.fps)
    return video, metadata


def _load_or_run_exact_resume(
    omni: Any,
    args: argparse.Namespace,
    entry: dict[str, str],
    seed: int,
    checkpoint_step: int,
    exact_latent: torch.Tensor,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    label = f"{entry['prompt_id']}_seed{seed}_step{checkpoint_step:02d}_exact_resume"
    frames_path = artifact_dir / f"{label}.npy"
    metadata_path = artifact_dir / f"{label}_trajectory_probe.json"
    remaining = args.num_inference_steps - checkpoint_step
    cached = _valid_probe(metadata_path, remaining) if args.resume else None
    if cached is not None and frames_path.exists():
        return np.load(frames_path, allow_pickle=False), cached
    video, metadata, _ = _run_with_probe(
        omni,
        args,
        entry["prompt"],
        seed,
        exact_latent,
        checkpoint_step,
        artifact_dir,
        label,
    )
    np.save(frames_path, video, allow_pickle=False)
    preflight._save_video(artifact_dir / f"{label}.mp4", video, fps=args.fps)
    return video, metadata


def _trajectory_rows(
    entry: dict[str, str],
    seed: int,
    checkpoint_step: int,
    perturbation: Perturbation,
    reference_meta: dict[str, Any],
    perturbed_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = _metadata_records(reference_meta)
    perturbed = _metadata_records(perturbed_meta)
    if set(reference) != set(perturbed):
        raise ValueError("Reference and perturbed trajectories have different captured steps")
    observations = []
    for local_step in sorted(reference):
        reference_path = Path(reference[local_step]["latent_path"])
        perturbed_path = Path(perturbed[local_step]["latent_path"])
        reference_latent = torch.load(reference_path, map_location="cpu")
        perturbed_latent = torch.load(perturbed_path, map_location="cpu")
        metrics = latent_error(reference_latent, perturbed_latent)
        observations.append((local_step, reference_path, perturbed_path, metrics))
    initial_metrics = observations[0][3]
    ratios = [
        metrics["normalized_l2"] / max(initial_metrics["normalized_l2"], 1e-12)
        for _, _, _, metrics in observations
    ]
    rows = []
    for (local_step, reference_path, perturbed_path, metrics), ratio in zip(observations, ratios, strict=True):
        rows.append(
            {
                "prompt_id": entry["prompt_id"],
                "motion_category": entry["motion_category"],
                "seed": seed,
                "checkpoint_step": checkpoint_step,
                "error_family": perturbation.family,
                "error_strength": perturbation.strength,
                "quantization_bits": perturbation.quantization_bits or "",
                "denoising_step": checkpoint_step + local_step,
                "remaining_step_index": local_step,
                "initial_latent_mse": initial_metrics["mse"],
                "initial_normalized_l2": initial_metrics["normalized_l2"],
                "initial_cosine_similarity": initial_metrics["cosine_similarity"],
                "latent_mse": metrics["mse"],
                "normalized_l2": metrics["normalized_l2"],
                "cosine_similarity": metrics["cosine_similarity"],
                "normalized_contraction": ratio,
                "reference_latent_path": str(reference_path),
                "perturbed_latent_path": str(perturbed_path),
            }
        )
    return rows, contraction_statistics(ratios)


def _reference_trajectory_rows(
    entry: dict[str, str],
    seed: int,
    checkpoint_step: int,
    baseline_meta: dict[str, Any],
    exact_resume_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = _metadata_records(baseline_meta)
    exact_resume = _metadata_records(exact_resume_meta)
    rows = []
    for local_step, record in sorted(exact_resume.items()):
        global_step = checkpoint_step + local_step
        baseline_latent = torch.load(baseline[global_step]["latent_path"], map_location="cpu")
        resumed_latent = torch.load(record["latent_path"], map_location="cpu")
        metrics = latent_error(baseline_latent, resumed_latent)
        rows.append(
            {
                "prompt_id": entry["prompt_id"],
                "motion_category": entry["motion_category"],
                "seed": seed,
                "checkpoint_step": checkpoint_step,
                "denoising_step": global_step,
                "latent_mse": metrics["mse"],
                "normalized_l2": metrics["normalized_l2"],
                "cosine_similarity": metrics["cosine_similarity"],
                "uninterrupted_latent_path": baseline[global_step]["latent_path"],
                "exact_resume_latent_path": record["latent_path"],
            }
        )
    return rows


def _quality_metrics(
    video: np.ndarray,
    baseline: np.ndarray,
    prompt: str,
    semantic: Any | None,
    baseline_semantic: float,
) -> dict[str, float]:
    temporal = preflight._temporal_metrics(video, baseline)
    semantic_abs = semantic.score_video(prompt, video) if semantic is not None else float("nan")
    semantic_relative = (
        semantic_abs / baseline_semantic
        if math.isfinite(semantic_abs) and math.isfinite(baseline_semantic) and baseline_semantic != 0.0
        else float("nan")
    )
    delta = video.astype(np.float32) - baseline.astype(np.float32)
    return {
        "video_mse_vs_uninterrupted": float(np.mean(np.square(delta))),
        "video_max_abs_diff_vs_uninterrupted": float(np.max(np.abs(delta))),
        "spatial_quality": preflight._spatial_metric(video, baseline),
        "temporal_shape_quality": temporal["temporal_shape_composite"],
        "temporal_dynamic_quality": temporal["temporal_dynamic_composite"],
        "motion_energy_ratio": temporal["motion_energy_ratio"],
        "flow_magnitude_cosine": temporal["flow_magnitude_cosine"],
        "flow_magnitude_ratio": temporal["flow_magnitude_ratio"],
        "flow_direction_cosine": temporal["flow_direction_cosine"],
        "flicker_similarity": temporal["flicker_similarity"],
        "semantic_quality": semantic_abs,
        "semantic_quality_vs_uninterrupted": semantic_relative,
    }


def _iso_rows(final_rows: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    output = []
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in final_rows:
        key = (str(row["prompt_id"]), int(row["checkpoint_step"]), str(row["error_strength"]))
        grouped.setdefault(key, {})[str(row["error_family"])] = row
    for (prompt_id, step, strength), families in sorted(grouped.items()):
        family_names = sorted(families)
        for index, family_a in enumerate(family_names):
            for family_b in family_names[index + 1 :]:
                row_a, row_b = families[family_a], families[family_b]
                error_a = float(row_a["initial_normalized_l2"])
                error_b = float(row_b["initial_normalized_l2"])
                mismatch = abs(error_a - error_b) / max(error_a, error_b, 1e-12)
                mse_a = float(row_a["initial_latent_mse"])
                mse_b = float(row_b["initial_latent_mse"])
                mse_mismatch = abs(mse_a - mse_b) / max(mse_a, mse_b, 1e-12)
                output.append(
                    {
                        "prompt_id": prompt_id,
                        "checkpoint_step": step,
                        "error_strength": strength,
                        "family_a": family_a,
                        "family_b": family_b,
                        "initial_mse_a": mse_a,
                        "initial_mse_b": mse_b,
                        "relative_mse_mismatch": mse_mismatch,
                        "initial_normalized_l2_a": error_a,
                        "initial_normalized_l2_b": error_b,
                        "relative_error_mismatch": mismatch,
                        "within_matching_tolerance": mismatch <= tolerance and mse_mismatch <= tolerance,
                        "contraction_fraction_a": float(row_a["contraction_fraction"]),
                        "contraction_fraction_b": float(row_b["contraction_fraction"]),
                        "contraction_delta_a_minus_b": float(row_a["contraction_fraction"])
                        - float(row_b["contraction_fraction"]),
                        "temporal_dynamic_a": float(row_a["temporal_dynamic_quality_vs_exact_resume"]),
                        "temporal_dynamic_b": float(row_b["temporal_dynamic_quality_vs_exact_resume"]),
                        "temporal_dynamic_delta_a_minus_b": float(
                            row_a["temporal_dynamic_quality_vs_exact_resume"]
                        ) - float(row_b["temporal_dynamic_quality_vs_exact_resume"]),
                        "spatial_a": float(row_a["spatial_quality_vs_exact_resume"]),
                        "spatial_b": float(row_b["spatial_quality_vs_exact_resume"]),
                    }
                )
    return output


def smoke_gate(final_rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in final_rows:
        key = (str(row["prompt_id"]), int(row["checkpoint_step"]), str(row["error_strength"]))
        grouped.setdefault(key, {})[str(row["error_family"])] = row
    effects = []
    for key, families in sorted(grouped.items()):
        if set(families) != set(ERROR_FAMILIES):
            continue
        contractions = [float(row["contraction_fraction"]) for row in families.values()]
        dynamics = [float(row["temporal_dynamic_quality_vs_exact_resume"]) for row in families.values()]
        quant = families["quantization"]
        structured = [families["spatial_lowpass"], families["temporal_lowpass"]]
        match_ok = all(
            float(row["relative_error_mismatch"]) <= tolerance
            and float(row.get("relative_mse_mismatch", row["relative_error_mismatch"])) <= tolerance
            for row in families.values()
        )
        quant_contraction_advantage = max(
            float(quant["contraction_fraction"]) - float(row["contraction_fraction"])
            for row in structured
        )
        quant_dynamic_advantage = max(
            float(quant["temporal_dynamic_quality_vs_exact_resume"])
            - float(row["temporal_dynamic_quality_vs_exact_resume"])
            for row in structured
        )
        passes = match_ok and (
            (max(contractions) - min(contractions) >= 0.20)
            or (max(dynamics) - min(dynamics) >= 0.10)
        ) and (
            quant_contraction_advantage >= 0.15 or quant_dynamic_advantage >= 0.10
        )
        effects.append(
            {
                "cell": key,
                "match_ok": match_ok,
                "contraction_range": max(contractions) - min(contractions),
                "dynamic_quality_range": max(dynamics) - min(dynamics),
                "quantization_contraction_advantage_vs_worst_structural": quant_contraction_advantage,
                "quantization_dynamic_advantage_vs_worst_structural": quant_dynamic_advantage,
                "passes_type_dependence_gate": passes,
            }
        )
    passing_cells = sum(bool(row["passes_type_dependence_gate"]) for row in effects)
    return {"effects": effects, "passing_cells": passing_cells, "evaluated_cells": len(effects)}


def _plot_results(
    output_dir: Path,
    trajectory_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    combined, combined_axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for strength in STRENGTHS:
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        combined_axis = combined_axes[STRENGTHS.index(strength)]
        for family in ERROR_FAMILIES:
            selected = [
                row for row in trajectory_rows
                if row["error_strength"] == strength and row["error_family"] == family
            ]
            if selected:
                selected.sort(key=lambda row: int(row["denoising_step"]))
                axis.plot(
                    [int(row["denoising_step"]) for row in selected],
                    [float(row["normalized_contraction"]) for row in selected],
                    marker="o", markersize=2.5, label=family,
                )
                combined_axis.plot(
                    [int(row["denoising_step"]) for row in selected],
                    [float(row["normalized_contraction"]) for row in selected],
                    marker="o", markersize=2.2, label=family,
                )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Global denoising step")
        axis.set_ylabel("Normalized latent error $E_k/E_t$")
        axis.set_title(f"Matched {strength} checkpoint error")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"error_contraction_curves_{strength}.pdf"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
        combined_axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        combined_axis.set_xlabel("Global denoising step")
        combined_axis.set_title(f"Matched {strength} error")
        combined_axis.grid(alpha=0.25)
    combined_axes[0].set_ylabel("Normalized latent error $E_k/E_t$")
    combined_axes[1].legend(fontsize=8)
    combined.tight_layout()
    combined_path = output_dir / "error_contraction_curves.pdf"
    combined.savefig(combined_path)
    plt.close(combined)
    paths.append(str(combined_path))

    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for family in ERROR_FAMILIES:
        selected = [row for row in final_rows if row["error_family"] == family]
        axis.scatter(
            [float(row["initial_normalized_l2"]) for row in selected],
            [float(row["temporal_dynamic_quality_vs_exact_resume"]) for row in selected],
            label=family, s=45,
        )
    axis.set_xlabel("Initial checkpoint normalized L2 error")
    axis.set_ylabel("Final temporal/dynamic quality vs exact resume")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "final_quality_vs_initial_error.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    judgment = summary["judgment"]
    lines = [
        "# Video Denoising Error-Correction Kill Test",
        "",
        "## Scope",
        "",
        f"- Stage: `{summary['stage']}`",
        f"- Prompt count: `{summary['prompt_count']}`",
        f"- Checkpoint steps: `{summary['checkpoint_steps']}`",
        "- Only checkpoint latents were perturbed; model weights and ordinary activations were unchanged.",
        "- Small and medium error magnitudes were fixed by the actual INT8-like and INT4-like errors, respectively.",
        "",
        "## CONFIRMED",
        "",
        f"- Completed perturbation conditions: `{summary['completed_conditions']}/{summary['expected_conditions']}`",
        f"- Maximum observed iso-error mismatch: `{summary['max_iso_error_mismatch']:.4%}`",
        f"- Maximum observed iso-MSE mismatch: `{summary['max_iso_mse_mismatch']:.4%}`",
        f"- Maximum exact-resume vs uninterrupted trajectory normalized L2: "
        f"`{summary['max_exact_resume_trajectory_normalized_l2']:.6g}`",
        f"- Error-type gate passing cells: `{summary['gate']['passing_cells']}/{summary['gate']['evaluated_cells']}`",
        "",
        "## INFERRED",
        "",
        "- A passing smoke gate is evidence worth testing across progress, not proof of a deployable "
        "recovery representation.",
        "- Different contraction curves at matched initial error are consistent with structure-dependent "
        "denoising correction.",
        "",
        "## UNKNOWN",
        "",
        "- Repeatability across checkpoint progress and content remains unknown until the gated follow-up stages run.",
        "- Storage benefits of a recovery-aware encoder are not measured in this mechanism kill test.",
        "- Causal frequency/component attribution is not established by low-pass direction probes alone.",
        "",
        "## Preregistered Gate",
        "",
        "A cell passes only when error matching is within 5%, the across-family contraction range is at least 0.20 "
        "or dynamic-quality range is at least 0.10, and quantization exceeds at least one structural family by "
        "0.15 contraction or 0.10 dynamic quality.",
        "",
        f"Eligible next stage: `{summary['eligible_next_stage']}`",
        "",
        "## Judgment",
        "",
        judgment,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    _validate_runtime_config(args)
    _check_stage_gate(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "0,"}:
        raise EnvironmentError("Run this experiment with CUDA_VISIBLE_DEVICES=0 on the exclusive second-server GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EnvironmentError(
            "The kill test requires exactly one visible CUDA device; reserve GPU0 and set CUDA_VISIBLE_DEVICES=0"
        )

    prompt_path, prompt_hash, all_entries = _read_prompts(args.prompt_set)
    entries, checkpoint_steps = _stage_scope(args.stage, all_entries)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "hypothesis": "matched checkpoint errors with different structure have different denoising contraction",
        "stage": args.stage,
        "model": args.model,
        "prompt_set_path": str(prompt_path),
        "prompt_set_sha256": prompt_hash,
        "prompt_ids": [entry["prompt_id"] for entry in entries],
        "prompts": entries,
        "checkpoint_steps": checkpoint_steps,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "flow_shift": args.flow_shift,
        "boundary_ratio": args.boundary_ratio,
        "sample_solver": args.sample_solver,
        "require_exact_resume": args.require_exact_resume,
        "exact_resume_tolerance": args.exact_resume_tolerance,
        "enable_cpu_offload": args.enable_cpu_offload,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "semantic_metric": {
            "disabled": args.disable_semantic_metric,
            "model": args.semantic_model,
            "frame_count": args.semantic_frame_count,
            "device": args.semantic_device,
        },
        "error_families": list(ERROR_FAMILIES),
        "error_strengths": {
            "small": "actual symmetric INT8-like normalized L2 error",
            "medium": "actual symmetric INT4-like normalized L2 error",
        },
        "structural_calibration": (
            "spatial/temporal low-pass residual directions are linearly scaled to the quantization-derived "
            "MSE and normalized-L2 targets; random control uses one fixed direction per prompt/step"
        ),
        "iso_error_tolerance": args.iso_error_tolerance,
        "gate": {
            "family_contraction_range": 0.20,
            "family_dynamic_quality_range": 0.10,
            "quantization_contraction_advantage_vs_structural": 0.15,
            "quantization_dynamic_advantage_vs_structural": 0.10,
        },
        "execution": "exclusive GPU0; one generation at a time",
    }
    config_path = output_dir / "perturbation_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"Existing preregistration differs: {config_path}")
    else:
        _atomic_json(config_path, config)

    trajectory_path = output_dir / "trajectory_error_rows.csv"
    final_path = output_dir / "final_recovery_quality.csv"
    trajectory_rows: list[dict[str, Any]] = _read_csv(trajectory_path) if args.resume else []
    final_rows: list[dict[str, Any]] = _read_csv(final_path) if args.resume else []
    if args.resume:
        final_rows, trajectory_rows = _validated_resume_rows(
            final_rows, trajectory_rows, args.num_inference_steps
        )
        _atomic_csv(trajectory_path, TRAJECTORY_FIELDS, trajectory_rows)
        _atomic_csv(final_path, FINAL_FIELDS, final_rows)
    completed = {
        (str(row["prompt_id"]), int(row["checkpoint_step"]), str(row["error_family"]), str(row["error_strength"]))
        for row in final_rows
    }

    semantic = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )
    omni = protection._make_omni(args)
    reference_validation = []
    reference_trajectory_rows = []
    try:
        for prompt_index, entry in enumerate(entries):
            seed = args.seed + prompt_index
            artifact_dir = output_dir / "artifacts" / f"{entry['prompt_id']}_seed{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            baseline, baseline_meta = _load_or_run_baseline(omni, args, entry, seed, artifact_dir)
            baseline_semantic = (
                semantic.score_video(entry["prompt"], baseline)
                if semantic is not None
                else float("nan")
            )
            baseline_records = _metadata_records(baseline_meta)

            for checkpoint_step in checkpoint_steps:
                exact_latent = torch.load(baseline_records[checkpoint_step]["latent_path"], map_location="cpu")
                exact_video, exact_meta = _load_or_run_exact_resume(
                    omni, args, entry, seed, checkpoint_step, exact_latent, artifact_dir
                )
                exact_quality = _quality_metrics(
                    exact_video, baseline, entry["prompt"], semantic, baseline_semantic
                )
                exact_trajectory = _reference_trajectory_rows(
                    entry, seed, checkpoint_step, baseline_meta, exact_meta
                )
                if args.require_exact_resume:
                    _require_exact_resume(
                        exact_trajectory,
                        exact_quality,
                        args.exact_resume_tolerance,
                    )
                reference_trajectory_rows.extend(exact_trajectory)
                reference_validation.append(
                    {
                        "prompt_id": entry["prompt_id"],
                        "checkpoint_step": checkpoint_step,
                        "max_trajectory_normalized_l2": max(
                            float(row["normalized_l2"]) for row in exact_trajectory
                        ),
                        **exact_quality,
                    }
                )
                perturbations = build_perturbations(
                    exact_latent,
                    random_seed=seed * 100 + checkpoint_step,
                    matching_tolerance=args.iso_error_tolerance,
                )
                for perturbation in perturbations:
                    key = (entry["prompt_id"], checkpoint_step, perturbation.family, perturbation.strength)
                    if key in completed:
                        print(
                            f"[correction-killtest] skip complete prompt={entry['prompt_id']} "
                            f"step={checkpoint_step} family={perturbation.family} strength={perturbation.strength}",
                            flush=True,
                        )
                        continue
                    label = (
                        f"{entry['prompt_id']}_seed{seed}_step{checkpoint_step:02d}_"
                        f"{perturbation.family}_{perturbation.strength}"
                    )
                    perturbation_path = artifact_dir / f"{label}_initial_latent.pt"
                    torch.save(perturbation.tensor, perturbation_path)
                    video, metadata, resume_ms = _run_with_probe(
                        omni,
                        args,
                        entry["prompt"],
                        seed,
                        perturbation.tensor,
                        checkpoint_step,
                        artifact_dir,
                        label,
                    )
                    video_path = artifact_dir / f"{label}.mp4"
                    preflight._save_video(video_path, video, fps=args.fps)
                    np.save(artifact_dir / f"{label}.npy", video, allow_pickle=False)
                    new_trajectory, contraction = _trajectory_rows(
                        entry, seed, checkpoint_step, perturbation, exact_meta, metadata
                    )
                    quality = _quality_metrics(
                        video, baseline, entry["prompt"], semantic, baseline_semantic
                    )
                    quality_vs_exact = _quality_metrics(
                        video, exact_video, entry["prompt"], None, float("nan")
                    )
                    observed_initial_mse = float(new_trajectory[0]["initial_latent_mse"])
                    observed_initial_l2 = float(new_trajectory[0]["initial_normalized_l2"])
                    last = new_trajectory[-1]
                    final_row = {
                        "prompt_id": entry["prompt_id"],
                        "motion_category": entry["motion_category"],
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "error_family": perturbation.family,
                        "error_strength": perturbation.strength,
                        "quantization_bits": perturbation.quantization_bits or "",
                        "target_latent_mse": perturbation.target_mse,
                        "target_normalized_l2": perturbation.target_normalized_l2,
                        "initial_latent_mse": observed_initial_mse,
                        "initial_normalized_l2": observed_initial_l2,
                        "initial_cosine_similarity": new_trajectory[0]["initial_cosine_similarity"],
                        "relative_mse_mismatch": abs(observed_initial_mse - perturbation.target_mse)
                        / max(perturbation.target_mse, 1e-12),
                        "relative_error_mismatch": abs(observed_initial_l2 - perturbation.target_normalized_l2)
                        / max(perturbation.target_normalized_l2, 1e-12),
                        "calibration_scale": perturbation.calibration_scale,
                        "resume_latency_ms": resume_ms,
                        "final_latent_mse": last["latent_mse"],
                        "final_normalized_l2": last["normalized_l2"],
                        "final_cosine_similarity": last["cosine_similarity"],
                        **contraction,
                        **quality,
                        "video_mse_vs_exact_resume": quality_vs_exact["video_mse_vs_uninterrupted"],
                        "video_max_abs_diff_vs_exact_resume": quality_vs_exact[
                            "video_max_abs_diff_vs_uninterrupted"
                        ],
                        "spatial_quality_vs_exact_resume": quality_vs_exact["spatial_quality"],
                        "temporal_shape_quality_vs_exact_resume": quality_vs_exact[
                            "temporal_shape_quality"
                        ],
                        "temporal_dynamic_quality_vs_exact_resume": quality_vs_exact[
                            "temporal_dynamic_quality"
                        ],
                        "artifact_path": str(video_path),
                        "trajectory_metadata_path": str(
                            artifact_dir / f"{label}_trajectory_probe.json"
                        ),
                    }
                    trajectory_rows.extend(new_trajectory)
                    final_rows.append(final_row)
                    completed.add(key)
                    _atomic_csv(trajectory_path, TRAJECTORY_FIELDS, trajectory_rows)
                    _atomic_csv(final_path, FINAL_FIELDS, final_rows)
                    print(
                        f"[correction-killtest] prompt={entry['prompt_id']} step={checkpoint_step} "
                        f"family={perturbation.family} strength={perturbation.strength} "
                        f"initial={perturbation.actual_normalized_l2:.6f} "
                        f"final_ratio={contraction['final_error_ratio']:.4f} "
                        f"dynamic_vs_exact={quality_vs_exact['temporal_dynamic_quality']:.4f}",
                        flush=True,
                    )
    finally:
        if hasattr(omni, "shutdown"):
            omni.shutdown()

    iso_rows = _iso_rows(final_rows, args.iso_error_tolerance)
    _atomic_csv(output_dir / "iso_error_comparisons.csv", ISO_FIELDS, iso_rows)
    _atomic_csv(
        output_dir / "exact_resume_trajectory_control.csv",
        [
            "prompt_id", "motion_category", "seed", "checkpoint_step", "denoising_step",
            "latent_mse", "normalized_l2", "cosine_similarity", "uninterrupted_latent_path",
            "exact_resume_latent_path",
        ],
        reference_trajectory_rows,
    )
    gate = smoke_gate(final_rows, args.iso_error_tolerance)
    expected = len(entries) * len(checkpoint_steps) * len(ERROR_FAMILIES) * len(STRENGTHS)
    complete = len(completed) == expected
    if not complete:
        judgment = "NO-GO"
        eligible_next = None
    elif args.stage == "smoke":
        passed = gate["passing_cells"] >= 1
        judgment = "CONDITIONAL GO" if passed else "NO-GO"
        eligible_next = "progress" if passed else None
    elif args.stage == "progress":
        passed_steps = len({effect["cell"][1] for effect in gate["effects"] if effect["passes_type_dependence_gate"]})
        judgment = "CONDITIONAL GO" if passed_steps >= 2 else "NO-GO"
        eligible_next = "content" if passed_steps >= 2 else None
    else:
        passed_prompts = len({effect["cell"][0] for effect in gate["effects"] if effect["passes_type_dependence_gate"]})
        passed_steps = len({effect["cell"][1] for effect in gate["effects"] if effect["passes_type_dependence_gate"]})
        judgment = "STRONG GO" if passed_prompts >= 2 and passed_steps >= 2 else (
            "CONDITIONAL GO" if gate["passing_cells"] else "NO-GO"
        )
        eligible_next = None

    max_mismatch = max((float(row["relative_error_mismatch"]) for row in final_rows), default=float("nan"))
    max_mse_mismatch = max((float(row["relative_mse_mismatch"]) for row in final_rows), default=float("nan"))
    max_exact_resume_error = max(
        (float(row["max_trajectory_normalized_l2"]) for row in reference_validation),
        default=float("nan"),
    )
    figure_paths = _plot_results(output_dir, trajectory_rows, final_rows) if complete else []
    summary = {
        "stage": args.stage,
        "prompt_count": len(entries),
        "checkpoint_steps": checkpoint_steps,
        "completed_conditions": len(completed),
        "expected_conditions": expected,
        "complete": complete,
        "max_iso_error_mismatch": max_mismatch,
        "max_iso_mse_mismatch": max_mse_mismatch,
        "max_exact_resume_trajectory_normalized_l2": max_exact_resume_error,
        "gate": gate,
        "reference_resume_validation": reference_validation,
        "figure_paths": figure_paths,
        "judgment": judgment,
        "eligible_next_stage": eligible_next,
    }
    _atomic_json(output_dir / "error_contraction_summary.json", summary)
    _write_report(output_dir / "video_denoising_error_correction_killtest.md", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
