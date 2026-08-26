#!/usr/bin/env python3
"""Measure repeated recovery/evaluation variability from fixed checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight
from experiments import video_state_protection_killtest as killtest
from experiments import video_state_protection_analysis as analysis


PROMPT_IDS = ["recovery_000", "recovery_004", "recovery_006"]
CHECKPOINT_STEPS = [10, 20, 30]
VARIANTS = ["full", "fp16", "int8"]
METRICS = {
    "spatial_vs_full": "spatial_metric_abs",
    "temporal_dynamic_vs_full": "temporal_dynamic_composite_abs",
    "semantic_vs_full": "semantic_metric_abs",
}

RAW_FIELDS = [
    "prompt_set_sha256",
    "model",
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "flow_shift",
    "boundary_ratio",
    "prompt_id",
    "category",
    "seed",
    "checkpoint_step",
    "variant",
    "repeat_index",
    "serialized_payload_sha256",
    "serialized_metadata_sha256",
    "resume_latency_ms",
    "spatial_metric_abs",
    "temporal_dynamic_composite_abs",
    "semantic_metric_abs",
    "artifact_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--prompt-set", default="experiments/video_recovery_prompt_set.json")
    parser.add_argument("--frontier-dir", default="results/video_state_protection_killtest_gpu0/run")
    parser.add_argument("--output-dir", default="results/video_state_protection_killtest_gpu0/run/noise_floor")
    parser.add_argument("--prompt-ids", nargs="+", default=PROMPT_IDS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=CHECKPOINT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = _mean(values)
    if len(values) == 1:
        return (mean, mean)
    half = 1.96 * _std(values) / math.sqrt(len(values))
    return (mean - half, mean + half)


def _prompt_index(prompt_id: str, entries: list[dict[str, Any]]) -> int:
    for index, entry in enumerate(entries):
        if entry["prompt_id"] == prompt_id:
            return index
    raise ValueError(f"Prompt ID is not registered: {prompt_id}")


def _metric_values(video: np.ndarray, baseline: np.ndarray, semantic: Any, prompt: str) -> dict[str, float]:
    temporal = preflight._temporal_metrics(video, baseline)
    metrics = {
        "spatial_metric_abs": preflight._spatial_metric(video, baseline),
        "temporal_dynamic_composite_abs": temporal["temporal_dynamic_composite"],
        "semantic_metric_abs": semantic.score_video(prompt, video) if semantic is not None else float("nan"),
    }
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"Non-finite recovery metric for prompt: {prompt}")
    return metrics


def _save_visual_grid(
    path: Path,
    videos: list[tuple[str, np.ndarray]],
    fps: float,
) -> None:
    from PIL import Image, ImageDraw

    target_width = 320
    frames = []
    frame_count = min(len(video) for _, video in videos)
    for frame_index in range(frame_count):
        panels = []
        for label, video in videos:
            image = Image.fromarray(video[frame_index]).convert("RGB")
            target_height = max(1, round(image.height * target_width / image.width))
            image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (target_width, target_height + 28), "black")
            canvas.paste(image, (0, 28))
            ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
            panels.append(np.asarray(canvas, dtype=np.uint8))
        frames.append(np.concatenate(panels, axis=1))
    preflight._save_video(path, np.stack(frames), fps=fps)


def _summarize(raw_rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["prompt_id"], int(row["checkpoint_step"]), row["variant"])].append(row)
    output = []
    for prompt_id, step, variant in sorted(grouped):
        rows = sorted(grouped[(prompt_id, step, variant)], key=lambda row: int(row["repeat_index"]))
        if len(rows) != repeats:
            raise ValueError(f"noise floor cell {prompt_id}/step{step}/{variant} has {len(rows)} repeats")
        full_rows = sorted(grouped[(prompt_id, step, "full")], key=lambda row: int(row["repeat_index"]))
        for metric, absolute_field in METRICS.items():
            values = [float(row[absolute_field]) for row in rows]
            full_values = [float(row[absolute_field]) for row in full_rows]
            full_mean = _mean(full_values)
            relative_values = analysis.paired_relative_values(values, full_values)
            low, high = _ci(values)
            std = _std(values)
            full_std = _std(full_values)
            compression_gap = full_mean - _mean(values)
            natural_noise = max(std, full_std)
            if natural_noise == 0.0:
                ratio = float("inf") if compression_gap != 0.0 else 0.0
            else:
                ratio = abs(compression_gap) / natural_noise
            orderings = []
            rows_by_variant_repeat = {
                candidate: {
                    int(item["repeat_index"]): item
                    for item in grouped[(prompt_id, step, candidate)]
                }
                for candidate in VARIANTS
            }
            for repeat_index in range(repeats):
                same_repeat = {
                    candidate: float(rows_by_variant_repeat[candidate][repeat_index][absolute_field])
                    for candidate in VARIANTS
                }
                orderings.append(">".join(sorted(VARIANTS, key=lambda name: same_repeat[name], reverse=True)))
            output.append(
                {
                    "prompt_set_sha256": rows[0]["prompt_set_sha256"],
                    "model": rows[0]["model"],
                    "prompt_id": prompt_id,
                    "category": rows[0]["category"],
                    "seed": rows[0]["seed"],
                    "checkpoint_step": step,
                    "variant": variant,
                    "metric": metric,
                    "repeat_count": repeats,
                    "mean": _mean(values),
                    "std": std,
                    "coefficient_of_variation": std / abs(_mean(values)) if _mean(values) else float("nan"),
                    "ci_low": low,
                    "ci_high": high,
                    "min": min(values),
                    "max": max(values),
                    "relative_to_full_mean": _mean(relative_values),
                    "probability_ge_0_95": _mean([1.0 if value >= 0.95 else 0.0 for value in relative_values]),
                    "probability_ge_0_975": _mean([1.0 if value >= 0.975 else 0.0 for value in relative_values]),
                    "probability_ge_0_99": _mean([1.0 if value >= 0.99 else 0.0 for value in relative_values]),
                    "full_mean": full_mean,
                    "compression_gap": compression_gap,
                    "natural_noise_std": natural_noise,
                    "gap_to_noise_ratio": ratio,
                    "gap_above_noise_floor": abs(compression_gap) > 2.0 * natural_noise,
                    "ordering_unique_count": len(set(orderings)),
                    "ordering_stable": len(set(orderings)) == 1,
                    "ordering_examples": json.dumps(sorted(set(orderings))),
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    frontier_dir = Path(args.frontier_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.disable_semantic_metric:
        raise ValueError("The preregistered noise-floor run requires the semantic metric")
    frontier_rows = analysis._read_csv(frontier_dir / "frontier_raw.csv")
    frontier_validation = analysis.validate_frontier(frontier_rows, 12, 5)
    analysis.validate_preregistered_config(
        frontier_dir / "preregistered_config.json",
        frontier_validation,
        expected_seeds=5,
    )
    expected_args = {
        "model": analysis.EXPECTED_MODEL,
        "height": 480,
        "width": 832,
        "num_frames": 33,
        "num_inference_steps": 40,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "guidance_scale": 4.0,
        "fps": 16.0,
        "flow_shift": 12.0,
        "boundary_ratio": 0.875,
        "enable_cpu_offload": True,
        "enable_layerwise_offload": False,
        "prompt_ids": PROMPT_IDS,
        "repeats": 5,
    }
    mismatches = {
        key: {"expected": expected, "actual": getattr(args, key)}
        for key, expected in expected_args.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(f"Noise-floor configuration differs from preregistration: {mismatches}")
    raw_path = output_dir / "noise_floor_raw.csv"
    raw_rows: list[dict[str, Any]] = _read_csv(raw_path)
    if raw_rows:
        required_raw_fields = set(RAW_FIELDS) - {"artifact_path"}
        missing = required_raw_fields - set(raw_rows[0])
        if missing:
            raise ValueError(
                f"Existing noise_floor_raw.csv predates required provenance fields: {sorted(missing)}"
            )
        if {row["prompt_set_sha256"] for row in raw_rows} != {
            frontier_validation["prompt_set_sha256"][0]
        }:
            raise ValueError("Existing noise-floor rows use a different prompt set")
        if {row["model"] for row in raw_rows} != {analysis.EXPECTED_MODEL}:
            raise ValueError("Existing noise-floor rows use a different model")
    existing_keys = [
        (row["prompt_id"], int(row["checkpoint_step"]), row["variant"], int(row["repeat_index"]))
        for row in raw_rows
    ]
    if len(existing_keys) != len(set(existing_keys)):
        raise ValueError("noise_floor_raw.csv contains duplicate repeat rows")
    existing = set(existing_keys)
    provenance = killtest._strict_prompt_provenance(args.prompt_set, 12)
    if provenance.sha256 != frontier_validation["prompt_set_sha256"][0]:
        raise ValueError("Noise-floor prompt set differs from the completed n=5 frontier")
    entries = {entry["prompt_id"]: entry for entry in provenance.entries}
    missing_prompts = set(args.prompt_ids) - set(entries)
    if missing_prompts:
        raise ValueError(f"Noise-floor prompts are outside the registered first 12: {sorted(missing_prompts)}")
    prereg = {
        "prompt_set_sha256": provenance.sha256,
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "flow_shift": args.flow_shift,
        "boundary_ratio": args.boundary_ratio,
        "enable_cpu_offload": args.enable_cpu_offload,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "semantic_model": args.semantic_model,
        "semantic_frame_count": args.semantic_frame_count,
        "semantic_device": args.semantic_device,
        "prompt_ids": args.prompt_ids,
        "checkpoint_steps": args.checkpoint_steps,
        "variants": VARIANTS,
        "repeats": args.repeats,
        "fixed_state_rule": "serialize once per prompt/step/variant and reuse restored bytes for every repeat",
        "noise_floor_rule": "abs(full-compressed mean) > 2 * max(full std, compressed std)",
        "threshold_decision_rule": (
            "selected high-fidelity tier passes every metric in 5/5 paired repeats and every "
            "cheaper high-fidelity tier rejected by n=5 fails at least one metric in 5/5 repeats"
        ),
        "quality_thresholds": [0.95, 0.975, 0.99],
        "visual_sanity_variants": ["baseline", "full", "fp16", "int8", "spatial_down2"],
    }
    prereg_path = output_dir / "noise_floor_preregistered_config.json"
    if prereg_path.exists() and json.loads(prereg_path.read_text(encoding="utf-8")) != prereg:
        raise ValueError("Noise-floor configuration differs from existing preregistration")
    prereg_tmp = prereg_path.with_suffix(prereg_path.suffix + ".tmp")
    prereg_tmp.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    prereg_tmp.replace(prereg_path)

    source_cache = {}
    for prompt_id in args.prompt_ids:
        entry = entries[prompt_id]
        prompt_index = _prompt_index(prompt_id, provenance.entries)
        seed = args.seed_base + prompt_index * 1000
        artifact_dir = frontier_dir / "artifacts" / f"{prompt_id}_seed{seed}"
        cached = killtest._load_cached_baseline(args, entry, seed, artifact_dir)
        if cached is None:
            raise FileNotFoundError(
                f"Stage B trajectory artifacts are missing or corrupt for {prompt_id} seed {seed}"
            )
        source_cache[prompt_id] = (seed, artifact_dir, cached)

    semantic = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )
    omni = killtest._make_omni(args)
    visual_manifest = []
    try:
        for prompt_id in args.prompt_ids:
            entry = entries[prompt_id]
            seed, artifact_dir, cached = source_cache[prompt_id]
            baseline, probe_meta, _, _ = cached
            probe_records = killtest._load_probe_records_by_step(probe_meta)
            for step in args.checkpoint_steps:
                latent_path = Path(probe_records[int(step)]["latent_path"])
                exact_latent = torch.load(latent_path, map_location="cpu")
                first_videos: dict[str, np.ndarray] = {}
                for variant in VARIANTS:
                    serialized_dir = output_dir / "fixed_serialized" / prompt_id / f"step{step:03d}" / variant
                    encoded = killtest._encode_representation(exact_latent, variant, serialized_dir)
                    payload_hash = _sha256(serialized_dir / f"{variant}.payload.bin")
                    metadata_hash = _sha256(serialized_dir / f"{variant}.metadata.json")
                    prior_cell_rows = [
                        row
                        for row in raw_rows
                        if row["prompt_id"] == prompt_id
                        and int(row["checkpoint_step"]) == int(step)
                        and row["variant"] == variant
                    ]
                    if any(
                        row["serialized_payload_sha256"] != payload_hash
                        or row["serialized_metadata_sha256"] != metadata_hash
                        for row in prior_cell_rows
                    ):
                        raise ValueError(
                            f"Serialized checkpoint changed across retries: {prompt_id}/step{step}/{variant}"
                        )
                    for repeat_index in range(args.repeats):
                        key = (prompt_id, int(step), variant, repeat_index)
                        if key in existing:
                            continue
                        sampling = killtest._build_resume_sampling_params(
                            args, seed=seed, checkpoint_step=int(step), latents=encoded.restored_latent
                        )
                        start = time.perf_counter()
                        outputs = omni.generate({"prompt": entry["prompt"]}, sampling)
                        resume_ms = (time.perf_counter() - start) * 1000.0
                        video, _ = killtest._normalize_output_video(outputs)
                        metrics = _metric_values(video, baseline, semantic, entry["prompt"])
                        artifact_path = output_dir / "videos" / prompt_id / f"step{step:03d}_{variant}_repeat{repeat_index}.mp4"
                        if repeat_index == 0:
                            artifact_path.parent.mkdir(parents=True, exist_ok=True)
                            preflight._save_video(artifact_path, video, fps=args.fps)
                            np.save(artifact_path.with_suffix(".npy"), video, allow_pickle=False)
                            first_videos[variant] = video
                        raw_rows.append(
                            {
                                "prompt_set_sha256": provenance.sha256,
                                "model": args.model,
                                "height": args.height,
                                "width": args.width,
                                "num_frames": args.num_frames,
                                "num_inference_steps": args.num_inference_steps,
                                "guidance_scale": args.guidance_scale,
                                "flow_shift": args.flow_shift,
                                "boundary_ratio": args.boundary_ratio,
                                "prompt_id": prompt_id,
                                "category": entry["motion_category"],
                                "seed": seed,
                                "checkpoint_step": step,
                                "variant": variant,
                                "repeat_index": repeat_index,
                                "serialized_payload_sha256": payload_hash,
                                "serialized_metadata_sha256": metadata_hash,
                                "resume_latency_ms": resume_ms,
                                **metrics,
                                "artifact_path": str(artifact_path) if repeat_index == 0 else "",
                            }
                        )
                        existing.add(key)
                        _write_csv_atomic(raw_path, raw_rows, RAW_FIELDS)
                        print(
                            f"[noise-floor] prompt={prompt_id} step={step} variant={variant} "
                            f"repeat={repeat_index + 1}/{args.repeats}",
                            flush=True,
                        )
                grid_path = output_dir / "visual_sanity" / f"{prompt_id}_step{step:03d}.mp4"
                if not grid_path.exists():
                    # Generate the visual-only structural baseline once; it is not used in noise statistics.
                    spatial_dir = output_dir / "fixed_serialized" / prompt_id / f"step{step:03d}" / "spatial_down2"
                    spatial = killtest._encode_representation(exact_latent, "spatial_down2", spatial_dir)
                    spatial_sampling = killtest._build_resume_sampling_params(
                        args, seed=seed, checkpoint_step=int(step), latents=spatial.restored_latent
                    )
                    spatial_outputs = omni.generate({"prompt": entry["prompt"]}, spatial_sampling)
                    spatial_video, _ = killtest._normalize_output_video(spatial_outputs)
                    for variant in VARIANTS:
                        if variant not in first_videos:
                            npy = output_dir / "videos" / prompt_id / f"step{step:03d}_{variant}_repeat0.npy"
                            first_videos[variant] = np.load(npy, allow_pickle=False)
                    grid_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_visual_grid(
                        grid_path,
                        [
                            ("baseline", baseline),
                            ("full", first_videos["full"]),
                            ("fp16", first_videos["fp16"]),
                            ("int8", first_videos["int8"]),
                            ("spatial_down2", spatial_video),
                        ],
                        args.fps,
                    )
                visual_manifest.append(
                    {
                        "prompt_id": prompt_id,
                        "category": entry["motion_category"],
                        "checkpoint_step": step,
                        "video_path": str(grid_path),
                        "panel_order": "baseline|full|fp16|int8|spatial_down2",
                        "formal_evidence": False,
                    }
                )
    finally:
        omni.close()

    expected = len(args.prompt_ids) * len(args.checkpoint_steps) * len(VARIANTS) * args.repeats
    if len(raw_rows) != expected:
        raise RuntimeError(f"expected {expected} repeated-recovery rows, found {len(raw_rows)}")
    summary_rows = _summarize(raw_rows, args.repeats)
    summary_fields = list(summary_rows[0])
    _write_csv_atomic(output_dir / "noise_floor_results.csv", summary_rows, summary_fields)
    if visual_manifest:
        _write_csv_atomic(output_dir / "visual_sanity_manifest.csv", visual_manifest, list(visual_manifest[0]))
    summary = {
        "raw_rows": len(raw_rows),
        "summary_rows": len(summary_rows),
        "fixed_cells": len(args.prompt_ids) * len(args.checkpoint_steps) * len(VARIANTS),
        "repeats": args.repeats,
        "noise_floor_csv": str(output_dir / "noise_floor_results.csv"),
        "visual_sanity_manifest": str(output_dir / "visual_sanity_manifest.csv"),
    }
    (output_dir / "noise_floor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
