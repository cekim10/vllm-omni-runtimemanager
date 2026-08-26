#!/usr/bin/env python3
"""Corrected hierarchical analysis for the video state-protection kill test.

This module intentionally contains no model/runtime code. It consumes the
immutable frontier CSV, preserves seeds as within-prompt repeats, and treats
prompts as the population sampling unit for aggregate inference.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIANTS = [
    "full",
    "fp16",
    "int8",
    "spatial_down2",
    "temporal_down2",
    "low_rank_25",
]
QUALITY_METRICS = [
    "spatial_vs_full",
    "temporal_dynamic_vs_full",
    "semantic_vs_full",
]
QUALITY_TARGETS = [0.95, 0.975, 0.99]
CHECKPOINT_STEPS = [10, 20, 30]
EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
NOISE_PROMPT_IDS = ["recovery_000", "recovery_004", "recovery_006"]
NOISE_VARIANTS = ["full", "fp16", "int8"]
ISO_STORAGE_PAIRS = [
    ("int8", "spatial_down2"),
    ("int8", "low_rank_25"),
    ("spatial_down2", "low_rank_25"),
    ("fp16", "temporal_down2"),
]
POLICIES = [
    "full",
    "uniform_int8",
    "progress_only",
    "content_only",
    "simple_separable",
    "joint_oracle",
]
DEFAULT_ANALYSIS_CONFIG = Path(__file__).with_name(
    "video_state_protection_stage_b_analysis_config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-prompts", type=int, default=12)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--iso-storage-tolerance", type=float, default=0.05)
    parser.add_argument("--complexity-bins", type=int, default=3)
    parser.add_argument("--simple-policy-max-training-violation", type=float, default=0.05)
    parser.add_argument("--budget-trials", type=int, default=200)
    parser.add_argument("--budget-session-counts", type=int, nargs="+", default=[3, 8, 20])
    parser.add_argument("--budget-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    parser.add_argument("--noise-floor-csv")
    parser.add_argument("--analysis-config", default=str(DEFAULT_ANALYSIS_CONFIG))
    parser.add_argument(
        "--allow-missing-noise-floor",
        action="store_true",
        help="Permit analysis-only dry runs. Final n=5 reports require noise-floor data.",
    )
    return parser.parse_args()


def load_analysis_config(path: Path) -> dict[str, Any]:
    """Load the preregistration; JSON syntax is intentionally valid YAML."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Stage B analysis preregistration: {path}") from exc
    required_judgment = {
        "strong_iso_minimum_resolved_cells",
        "strong_iso_minimum_absolute_dynamic_delta",
        "heterogeneity_minimum_progress_change_fraction",
        "heterogeneity_minimum_representations_in_content_cell",
        "noise_minimum_measured_99pct_decisions",
        "noise_minimum_stable_99pct_decision_fraction",
        "noise_minimum_stable_99pct_representations",
        "effect_not_confined_to_99pct_minimum_mean_relative_oracle_gap",
        "strong_gap_minimum_relative_improvement",
        "strong_gap_minimum_budget_levels",
        "strong_gap_minimum_mixtures",
        "strong_gap_minimum_session_counts",
        "conditional_gap_minimum_relative_improvement",
        "conditional_gap_minimum_budget_levels",
        "conditional_gap_minimum_mixtures",
        "conditional_gap_minimum_session_counts",
        "runtime_design_forbidden_unless",
    }
    missing = required_judgment - set(config.get("judgment", {}))
    if missing:
        raise ValueError(f"Analysis preregistration omits judgment keys: {sorted(missing)}")
    if config.get("interaction_crossing", {}).get(
        "minimum_noise_resolved_steps_for_pair"
    ) != 2:
        raise ValueError("Interaction crossing preregistration is incomplete or changed")
    return config


def validate_analysis_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "raw_frontier.prompts": (config["raw_frontier"]["prompts"], args.expected_prompts),
        "raw_frontier.seeds_per_prompt": (
            config["raw_frontier"]["seeds_per_prompt"], args.expected_seeds
        ),
        "statistical_units.bootstrap_samples": (
            config["statistical_units"]["bootstrap_samples"], args.bootstrap_samples
        ),
        "statistical_units.bootstrap_seed": (
            config["statistical_units"]["bootstrap_seed"], args.bootstrap_seed
        ),
        "iso_storage.maximum_relative_byte_mismatch": (
            config["iso_storage"]["maximum_relative_byte_mismatch"], args.iso_storage_tolerance
        ),
        "simple_policies.content_complexity_bins": (
            config["simple_policies"]["content_complexity_bins"], args.complexity_bins
        ),
        "simple_policies.maximum_training_quality_violation_rate": (
            config["simple_policies"]["maximum_training_quality_violation_rate"],
            args.simple_policy_max_training_violation,
        ),
        "budget_simulation.trials": (config["budget_simulation"]["trials"], args.budget_trials),
        "budget_simulation.session_counts": (
            config["budget_simulation"]["session_counts"], args.budget_session_counts
        ),
        "budget_simulation.budget_fractions_of_all_full": (
            config["budget_simulation"]["budget_fractions_of_all_full"], args.budget_fractions
        ),
    }
    mismatches = [name for name, (registered, actual) in expected.items() if registered != actual]
    if mismatches:
        raise ValueError(f"Analysis arguments differ from preregistration: {mismatches}")


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    """Percentile CI whose caller supplies values at the independent unit."""
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    draws = [
        _mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return (_percentile(draws, 0.025), _percentile(draws, 0.975))


def metric_stats(values: list[float], samples: int, seed: int) -> dict[str, float]:
    low, high = bootstrap_mean_ci(values, samples, seed)
    return {
        "mean": _mean(values),
        "median": _median(values),
        "std": _std(values),
        "ci_low": low,
        "ci_high": high,
    }


def paired_relative_values(values: list[float], full_values: list[float]) -> list[float]:
    if len(values) != len(full_values):
        raise ValueError("paired recovery and full metric vectors must have equal length")
    if any(not math.isfinite(value) for value in [*values, *full_values]):
        raise ValueError("paired recovery metrics must be finite")
    if any(full_value == 0.0 for full_value in full_values):
        raise ValueError("paired full recovery metrics cannot be zero")
    return [
        value / full_value for value, full_value in zip(values, full_values, strict=True)
    ]


def _group(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def validate_frontier(
    rows: list[dict[str, str]], expected_prompts: int, expected_seeds: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("frontier_raw.csv is empty")
    required = {
        "prompt_id",
        "category",
        "seed",
        "seed_index",
        "checkpoint_step",
        "variant",
        "raw_latent_bytes",
        "encoded_payload_bytes",
        "metadata_bytes",
        "total_checkpoint_bytes",
        "compression_ratio_vs_full",
        "encode_prepare_latency_ms",
        "storage_write_latency_ms",
        "load_read_latency_ms",
        "decode_reconstruction_latency_ms",
        "content_complexity_score",
        "prompt_set_sha256",
        "model",
        "total_steps",
        "progress_fraction",
        *QUALITY_METRICS,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"frontier_raw.csv is missing columns: {sorted(missing)}")
    keys = [
        (row["prompt_id"], _int(row, "seed_index"), _int(row, "checkpoint_step"), row["variant"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("frontier_raw.csv contains duplicate prompt/seed/step/variant rows")
    prompts = sorted({row["prompt_id"] for row in rows})
    if len(prompts) != expected_prompts:
        raise ValueError(f"expected {expected_prompts} prompts, found {len(prompts)}")
    if sorted({_int(row, "checkpoint_step") for row in rows}) != CHECKPOINT_STEPS:
        raise ValueError("checkpoint steps are not exactly 10, 20, 30")
    if sorted({row["variant"] for row in rows}) != sorted(VARIANTS):
        raise ValueError(f"representation coverage differs from {VARIANTS}")
    expected_seed_indexes = list(range(expected_seeds))
    for prompt_id in prompts:
        indexes = sorted({_int(row, "seed_index") for row in rows if row["prompt_id"] == prompt_id})
        if indexes != expected_seed_indexes:
            raise ValueError(
                f"{prompt_id} has seed indexes {indexes}; expected {expected_seed_indexes}. "
                "An n=2 dataset cannot produce an n=5 report."
            )
    expected_rows = expected_prompts * expected_seeds * len(CHECKPOINT_STEPS) * len(VARIANTS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected exactly {expected_rows} rows, found {len(rows)}")
    if expected_seeds == 5:
        hash_fields = {"serialized_payload_sha256", "serialized_metadata_sha256"}
        missing_hash_fields = hash_fields - set(rows[0])
        if missing_hash_fields:
            raise ValueError(
                f"n=5 frontier lacks artifact integrity columns: {sorted(missing_hash_fields)}"
            )
        if any(
            len(str(row.get(field, ""))) != 64
            for row in rows
            for field in hash_fields
        ):
            raise ValueError("n=5 frontier contains missing or malformed artifact hashes")
    prompt_ids = [row["prompt_id"] for row in rows]
    if any(prompt_id.startswith("preflight_") for prompt_id in prompt_ids):
        raise ValueError("preflight prompt IDs found in recovery frontier")
    hashes = sorted({row.get("prompt_set_sha256", "") for row in rows})
    if len(hashes) != 1 or not hashes[0]:
        raise ValueError(f"frontier rows contain mixed or empty prompt-set hashes: {hashes}")
    models = sorted({row.get("model", "") for row in rows})
    if models != [EXPECTED_MODEL]:
        raise ValueError(f"frontier model mismatch: expected {EXPECTED_MODEL}, found {models}")
    if {_int(row, "total_steps") for row in rows} != {40}:
        raise ValueError("frontier total_steps must be exactly 40")
    for row in rows:
        expected_progress = _int(row, "checkpoint_step") / _int(row, "total_steps")
        if not math.isclose(_float(row, "progress_fraction"), expected_progress, abs_tol=1e-12):
            raise ValueError(f"invalid progress_fraction in row {row['prompt_id']}/{row['seed_index']}")
        numeric_fields = [
            "raw_latent_bytes",
            "encoded_payload_bytes",
            "metadata_bytes",
            "total_checkpoint_bytes",
            "compression_ratio_vs_full",
            "encode_prepare_latency_ms",
            "storage_write_latency_ms",
            "load_read_latency_ms",
            "decode_reconstruction_latency_ms",
            "content_complexity_score",
            *QUALITY_METRICS,
        ]
        if any(not math.isfinite(_float(row, field)) for field in numeric_fields):
            raise ValueError(
                f"non-finite frontier metric in {row['prompt_id']}/seed{row['seed_index']}/"
                f"step{row['checkpoint_step']}/{row['variant']}"
            )
        if _float(row, "total_checkpoint_bytes") <= 0:
            raise ValueError("checkpoint byte counts must be positive")
        if _int(row, "raw_latent_bytes") <= 0 or _int(row, "encoded_payload_bytes") <= 0:
            raise ValueError("raw and encoded payload byte counts must be positive")
        if _int(row, "metadata_bytes") < 0:
            raise ValueError("metadata byte counts cannot be negative")
        if _int(row, "total_checkpoint_bytes") != (
            _int(row, "encoded_payload_bytes") + _int(row, "metadata_bytes")
        ):
            raise ValueError("total checkpoint bytes must equal encoded payload plus metadata")
        latency_fields = [
            "encode_prepare_latency_ms",
            "storage_write_latency_ms",
            "load_read_latency_ms",
            "decode_reconstruction_latency_ms",
        ]
        if any(_float(row, field) < 0.0 for field in latency_fields):
            raise ValueError("checkpoint preparation/storage/recovery latencies cannot be negative")
    paths = sorted({row.get("prompt_set_path", "") for row in rows})
    return {
        "row_count": len(rows),
        "prompt_count": len(prompts),
        "seed_count_per_prompt": expected_seeds,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "variants": VARIANTS,
        "prompt_ids": prompts,
        "prompt_set_sha256": hashes,
        "prompt_set_paths": paths,
    }


def validate_preregistered_config(
    path: Path,
    validation: dict[str, Any],
    expected_seeds: int,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing preregistered configuration: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    stopping = config.get("stopping_rule", {})
    if int(stopping.get("primary_num_prompts", -1)) != validation["prompt_count"]:
        raise ValueError("preregistered prompt count differs from frontier")
    if expected_seeds == 5 and int(stopping.get("primary_num_seeds", -1)) != 5:
        raise ValueError("preregistered primary seed count is not 5")
    provenance = config.get("prompt_provenance", {})
    if provenance.get("sha256") != validation["prompt_set_sha256"][0]:
        raise ValueError("preregistered prompt hash differs from frontier")
    if list(provenance.get("prompt_ids", [])) != validation["prompt_ids"]:
        raise ValueError("preregistered prompt IDs differ from frontier")
    experiment = config.get("experiment_config", {})
    expected = {
        "model": EXPECTED_MODEL,
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
        "variants": VARIANTS,
        "quality_targets": QUALITY_TARGETS,
    }
    mismatches = {
        key: {"expected": value, "actual": experiment.get(key)}
        for key, value in expected.items()
        if experiment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"preregistered experiment configuration mismatch: {mismatches}")
    return {
        "path": str(path),
        "prompt_set_sha256": provenance["sha256"],
        "primary_num_seeds": int(stopping["primary_num_seeds"]),
        "configuration_verified": True,
    }


def validate_noise_floor_rows(
    rows: list[dict[str, str]],
    expected_prompt_hash: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("noise_floor_results.csv is empty")
    required = {
        "prompt_set_sha256",
        "model",
        "prompt_id",
        "checkpoint_step",
        "variant",
        "metric",
        "repeat_count",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "min",
        "max",
        "relative_to_full_mean",
        "probability_ge_0_95",
        "probability_ge_0_975",
        "probability_ge_0_99",
        "full_mean",
        "compression_gap",
        "natural_noise_std",
        "gap_to_noise_ratio",
        "ordering_stable",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"noise-floor summary is missing columns: {sorted(missing)}")
    keys = [
        (row["prompt_id"], _int(row, "checkpoint_step"), row["variant"], row["metric"])
        for row in rows
    ]
    expected_count = len(NOISE_PROMPT_IDS) * len(CHECKPOINT_STEPS) * len(NOISE_VARIANTS) * len(QUALITY_METRICS)
    if len(rows) != expected_count or len(keys) != len(set(keys)):
        raise ValueError(f"noise-floor summary must contain {expected_count} unique cells, found {len(rows)}")
    if sorted({row["prompt_id"] for row in rows}) != sorted(NOISE_PROMPT_IDS):
        raise ValueError("noise-floor prompt coverage differs from preregistration")
    if sorted({_int(row, "checkpoint_step") for row in rows}) != CHECKPOINT_STEPS:
        raise ValueError("noise-floor checkpoint steps differ from preregistration")
    if sorted({row["variant"] for row in rows}) != sorted(NOISE_VARIANTS):
        raise ValueError("noise-floor representation coverage differs from preregistration")
    if sorted({row["metric"] for row in rows}) != sorted(QUALITY_METRICS):
        raise ValueError("noise-floor metric coverage differs from preregistration")
    if {row.get("model", "") for row in rows} != {EXPECTED_MODEL}:
        raise ValueError("noise-floor model provenance is missing or inconsistent")
    if {row.get("prompt_set_sha256", "") for row in rows} != {expected_prompt_hash}:
        raise ValueError("noise-floor prompt hash differs from frontier")
    numeric_fields = [
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "min",
        "max",
        "relative_to_full_mean",
        "probability_ge_0_95",
        "probability_ge_0_975",
        "probability_ge_0_99",
        "full_mean",
        "compression_gap",
        "natural_noise_std",
    ]
    for row in rows:
        if _int(row, "repeat_count") != 5:
            raise ValueError("every noise-floor cell must contain five repeats")
        for field in numeric_fields:
            value = _float(row, field)
            if not math.isfinite(value):
                raise ValueError(f"non-finite noise-floor value: {row['prompt_id']}/{row['variant']}/{field}")
        for field in ("probability_ge_0_95", "probability_ge_0_975", "probability_ge_0_99"):
            if not 0.0 <= _float(row, field) <= 1.0:
                raise ValueError(f"noise-floor probability outside [0,1]: {field}")
        if _float(row, "gap_to_noise_ratio") < 0.0 or math.isnan(_float(row, "gap_to_noise_ratio")):
            raise ValueError("noise-floor gap_to_noise_ratio must be nonnegative")
    return {"row_count": len(rows), "repeat_count": 5, "configuration_verified": True}


def validate_noise_preregistered_config(path: Path, expected_prompt_hash: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing noise-floor preregistration: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "prompt_set_sha256": expected_prompt_hash,
        "model": EXPECTED_MODEL,
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
        "prompt_ids": NOISE_PROMPT_IDS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "variants": NOISE_VARIANTS,
        "repeats": 5,
        "quality_thresholds": QUALITY_TARGETS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"noise-floor preregistration mismatch: {mismatches}")
    return {"path": str(path), "configuration_verified": True}


def _seed_cell_stats(
    rows: list[dict[str, Any]], metric: str, samples: int, seed: int
) -> dict[str, float]:
    return metric_stats([_float(row, metric) for row in rows], samples, seed)


def aggregate_frontier(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    by_prompt = _group(rows, "prompt_id", "checkpoint_step", "variant")
    output: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        for variant in VARIANTS:
            prompt_rows = [
                group
                for (prompt_id, grouped_step, grouped_variant), group in by_prompt.items()
                if int(grouped_step) == step and grouped_variant == variant
            ]
            key = f"step_{step}_{variant}"
            entry: dict[str, Any] = {
                "prompt_count": len(prompt_rows),
                "seed_count_per_prompt": len(prompt_rows[0]) if prompt_rows else 0,
                "sampling_unit": "prompt_mean",
            }
            for metric_index, metric in enumerate(["total_checkpoint_bytes", *QUALITY_METRICS]):
                prompt_means = [_mean(_float(row, metric) for row in group) for group in prompt_rows]
                entry[metric] = metric_stats(prompt_means, samples, seed + metric_index)
            output[key] = entry
    return output


def build_checkpoint_sizes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "prompt_id",
        "category",
        "seed",
        "seed_index",
        "checkpoint_step",
        "variant",
        "raw_latent_bytes",
        "encoded_payload_bytes",
        "metadata_bytes",
        "total_checkpoint_bytes",
        "compression_ratio_vs_full",
        "checkpoint_cpu_copy_ms",
        "checkpoint_save_ms",
        "checkpoint_protection_ms",
        "encode_prepare_latency_ms",
        "storage_write_latency_ms",
        "load_read_latency_ms",
        "decode_reconstruction_latency_ms",
        "serialized_artifact_dir",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def build_iso_storage(
    rows: list[dict[str, Any]],
    tolerance: float,
    samples: int,
    seed: int,
    expected_seeds: int,
) -> list[dict[str, Any]]:
    by_key = {
        (row["prompt_id"], _int(row, "seed_index"), _int(row, "checkpoint_step"), row["variant"]): row
        for row in rows
    }
    prompts = sorted({row["prompt_id"] for row in rows})
    output = []
    for step in CHECKPOINT_STEPS:
        for pair_index, (variant_a, variant_b) in enumerate(ISO_STORAGE_PAIRS):
            prompt_effects: dict[str, dict[str, float]] = {}
            for prompt_id in prompts:
                seed_deltas = {metric: [] for metric in QUALITY_METRICS}
                mismatches: list[float] = []
                bytes_a: list[float] = []
                bytes_b: list[float] = []
                for seed_index in range(expected_seeds):
                    row_a = by_key[(prompt_id, seed_index, step, variant_a)]
                    row_b = by_key[(prompt_id, seed_index, step, variant_b)]
                    a_bytes = _float(row_a, "total_checkpoint_bytes")
                    b_bytes = _float(row_b, "total_checkpoint_bytes")
                    mismatch = abs(a_bytes - b_bytes) / max(a_bytes, b_bytes)
                    if mismatch > tolerance:
                        continue
                    mismatches.append(mismatch)
                    bytes_a.append(a_bytes)
                    bytes_b.append(b_bytes)
                    for metric in QUALITY_METRICS:
                        seed_deltas[metric].append(_float(row_a, metric) - _float(row_b, metric))
                if len(mismatches) == expected_seeds:
                    prompt_effects[prompt_id] = {
                        "byte_mismatch": _mean(mismatches),
                        "bytes_a": _mean(bytes_a),
                        "bytes_b": _mean(bytes_b),
                        **{metric: _mean(values) for metric, values in seed_deltas.items()},
                    }
            base = {
                "checkpoint_step": step,
                "pair": f"{variant_a}__vs__{variant_b}",
                "variant_a": variant_a,
                "variant_b": variant_b,
                "prompt_count": len(prompt_effects),
                "seed_count": expected_seeds,
                "sample_unit": "prompt_mean_of_paired_seed_deltas",
                "mean_bytes_a": _mean(effect["bytes_a"] for effect in prompt_effects.values()),
                "mean_bytes_b": _mean(effect["bytes_b"] for effect in prompt_effects.values()),
                "mean_relative_byte_mismatch": _mean(
                    effect["byte_mismatch"] for effect in prompt_effects.values()
                ),
            }
            metric_prefixes = {
                "temporal_dynamic_vs_full": "dynamic",
                "spatial_vs_full": "spatial",
                "semantic_vs_full": "semantic",
            }
            for metric_index, (metric, prefix) in enumerate(metric_prefixes.items()):
                effects = [effect[metric] for effect in prompt_effects.values()]
                stats = metric_stats(effects, samples, seed + pair_index * 11 + metric_index)
                for stat_name, value in stats.items():
                    base[f"{prefix}_delta_{stat_name}"] = value
            low = float(base["dynamic_delta_ci_low"])
            high = float(base["dynamic_delta_ci_high"])
            resolved = bool(prompt_effects) and not (low <= 0.0 <= high)
            base.update(
                {
                    "resolved_at_seed_count": resolved,
                    "resolution_seed_count": expected_seeds,
                    "eligible_for_extension_to_15": expected_seeds == 5 and not resolved,
                }
            )
            output.append(base)
    return output


def load_noise_floor(path: Path | None) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    return {
        (row["prompt_id"], _int(row, "checkpoint_step"), row["variant"], row["metric"]): row
        for row in rows
    }


def _threshold_probability_field(target: float) -> str:
    mapping = {
        0.95: "probability_ge_0_95",
        0.975: "probability_ge_0_975",
        0.99: "probability_ge_0_99",
    }
    for registered_target, field in mapping.items():
        if math.isclose(target, registered_target):
            return field
    raise ValueError(f"No noise-floor threshold probability registered for {target}")


def _resolve_threshold_noise_decision(
    prompt_id: str,
    step: int,
    target: float,
    selected_representation: str,
    variant_stats: dict[str, dict[str, Any]],
    noise_floor: dict[tuple[str, int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Resolve a minimum-representation decision using fixed-state repeats."""
    high_fidelity = [
        variant for variant in ("int8", "fp16", "full") if variant in variant_stats
    ]
    if selected_representation not in high_fidelity:
        return {
            "decision_above_noise_floor": "",
            "noise_floor_status": "not_measured_for_representation",
            "noise_selected_min_pass_probability": "",
            "noise_cheaper_failed_tiers_resolved": "",
        }
    probability_field = _threshold_probability_field(target)

    def metric_rows(variant: str) -> list[dict[str, Any]] | None:
        found = [noise_floor.get((prompt_id, step, variant, metric)) for metric in QUALITY_METRICS]
        return None if any(row is None for row in found) else [row for row in found if row is not None]

    selected_rows = metric_rows(selected_representation)
    if selected_rows is None:
        return {
            "decision_above_noise_floor": "",
            "noise_floor_status": "unmeasured",
            "noise_selected_min_pass_probability": "",
            "noise_cheaper_failed_tiers_resolved": "",
        }
    selected_min_probability = min(_float(row, probability_field) for row in selected_rows)
    selected_stable = selected_min_probability == 1.0 and all(
        str(row.get("ordering_stable", "")).lower() == "true" for row in selected_rows
    )
    selected_bytes = float(variant_stats[selected_representation]["bytes"])
    cheaper_failed = [
        variant
        for variant in high_fidelity
        if float(variant_stats[variant]["bytes"]) < selected_bytes
        and not bool(variant_stats[variant][f"safe_{target}"])
    ]
    cheaper_resolved = True
    for variant in cheaper_failed:
        rows = metric_rows(variant)
        if rows is None:
            return {
                "decision_above_noise_floor": "",
                "noise_floor_status": "unmeasured",
                "noise_selected_min_pass_probability": selected_min_probability,
                "noise_cheaper_failed_tiers_resolved": "",
            }
        ordering_stable = all(
            str(row.get("ordering_stable", "")).lower() == "true" for row in rows
        )
        probabilities = [_float(row, probability_field) for row in rows]
        threshold_stable = all(probability in {0.0, 1.0} for probability in probabilities)
        has_stable_failure = any(probability == 0.0 for probability in probabilities)
        cheaper_resolved = (
            cheaper_resolved
            and ordering_stable
            and threshold_stable
            and has_stable_failure
        )
    resolved = selected_stable and cheaper_resolved
    return {
        "decision_above_noise_floor": resolved,
        "noise_floor_status": "threshold_stable" if resolved else "threshold_ambiguous",
        "noise_selected_min_pass_probability": selected_min_probability,
        "noise_cheaper_failed_tiers_resolved": cheaper_resolved,
    }


def build_minimum_safe(
    rows: list[dict[str, Any]],
    targets: list[float],
    samples: int,
    seed: int,
    expected_seeds: int,
    noise_floor: dict[tuple[str, int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, float], dict[str, Any]]]:
    grouped = _group(rows, "prompt_id", "category", "checkpoint_step", "variant")
    prompts = sorted({(row["prompt_id"], row["category"]) for row in rows})
    output = []
    cells: dict[tuple[str, int, float], dict[str, Any]] = {}
    for prompt_id, category in prompts:
        for step in CHECKPOINT_STEPS:
            variant_stats: dict[str, dict[str, Any]] = {}
            for variant_index, variant in enumerate(VARIANTS):
                group = sorted(
                    grouped[(prompt_id, category, str(step), variant)],
                    key=lambda row: _int(row, "seed_index"),
                )
                if len(group) != expected_seeds:
                    raise AssertionError(f"{prompt_id}/step{step}/{variant} has {len(group)} seeds")
                metric_values = {
                    metric: [_float(row, metric) for row in group]
                    for metric in QUALITY_METRICS
                }
                stats = {
                    metric: metric_stats(values, samples, seed + variant_index * 13 + metric_index)
                    for metric_index, (metric, values) in enumerate(metric_values.items())
                }
                variant_stats[variant] = {
                    "variant": variant,
                    "bytes": _mean(_float(row, "total_checkpoint_bytes") for row in group),
                    "compression_ratio": _mean(_float(row, "compression_ratio_vs_full") for row in group),
                    "metric_values": metric_values,
                    "metric_stats": stats,
                }
            for target in targets:
                safe_candidates = []
                for variant in VARIANTS:
                    info = variant_stats[variant]
                    safe = all(info["metric_stats"][metric]["ci_low"] >= target for metric in QUALITY_METRICS)
                    seed_safe_fraction = _mean(
                        1.0
                        if all(info["metric_values"][metric][index] >= target for metric in QUALITY_METRICS)
                        else 0.0
                        for index in range(expected_seeds)
                    )
                    info[f"safe_{target}"] = safe
                    info[f"seed_safe_fraction_{target}"] = seed_safe_fraction
                    if safe:
                        safe_candidates.append(info)
                selected = min(safe_candidates, key=lambda item: (item["bytes"], VARIANTS.index(item["variant"]))) if safe_candidates else None
                if selected is None:
                    selected_rep = "none"
                    selected_bytes = float("nan")
                    selected_ratio = float("nan")
                    stability = 0.0
                    noise_decision = {
                        "decision_above_noise_floor": "",
                        "noise_floor_status": "not_applicable",
                        "noise_selected_min_pass_probability": "",
                        "noise_cheaper_failed_tiers_resolved": "",
                    }
                else:
                    selected_rep = selected["variant"]
                    selected_bytes = selected["bytes"]
                    selected_ratio = selected["compression_ratio"]
                    stability = selected[f"seed_safe_fraction_{target}"]
                    noise_decision = _resolve_threshold_noise_decision(
                        prompt_id,
                        step,
                        target,
                        selected_rep,
                        variant_stats,
                        noise_floor,
                    )
                selected_stats = selected["metric_stats"] if selected else {}
                row = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "checkpoint_step": step,
                    "quality_target": target,
                    "selected_representation": selected_rep,
                    "selected_total_checkpoint_bytes": selected_bytes,
                    "selected_compression_ratio_vs_full": selected_ratio,
                    "selection_rule": "within_prompt_seed_bootstrap_lower_95ci_all_metrics>=target",
                    "safe_candidate_count": len(safe_candidates),
                    "seed_count": expected_seeds,
                    "selected_seed_safe_fraction": stability,
                    "selected_dynamic_mean": selected_stats.get("temporal_dynamic_vs_full", {}).get("mean", float("nan")),
                    "selected_dynamic_ci_low": selected_stats.get("temporal_dynamic_vs_full", {}).get("ci_low", float("nan")),
                    "selected_spatial_mean": selected_stats.get("spatial_vs_full", {}).get("mean", float("nan")),
                    "selected_spatial_ci_low": selected_stats.get("spatial_vs_full", {}).get("ci_low", float("nan")),
                    "selected_semantic_mean": selected_stats.get("semantic_vs_full", {}).get("mean", float("nan")),
                    "selected_semantic_ci_low": selected_stats.get("semantic_vs_full", {}).get("ci_low", float("nan")),
                    **noise_decision,
                }
                output.append(row)
                cells[(prompt_id, step, target)] = {
                    "row": row,
                    "variants": variant_stats,
                    "content_complexity": _mean(
                        _float(source, "content_complexity_score")
                        for source in grouped[(prompt_id, category, str(step), "full")]
                    ),
                }
    return output, cells


def _quantile_edges(values: list[float], bin_count: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return [0.0, 1.0]
    edges = [_percentile(ordered, index / bin_count) for index in range(bin_count + 1)]
    edges[-1] += 1e-12
    return edges


def _bin(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges[1:]):
        if value <= edge:
            return index
    return len(edges) - 2


def _choose_group_representation(
    cells: list[dict[str, Any]], max_violation: float
) -> str:
    candidates = []
    for variant in VARIANTS:
        violation = _mean(
            0.0 if cell["variants"][variant][f"safe_{cell['target']}"] else 1.0
            for cell in cells
        )
        mean_bytes = _mean(cell["variants"][variant]["bytes"] for cell in cells)
        candidates.append((variant, violation, mean_bytes))
    feasible = [item for item in candidates if item[1] <= max_violation]
    if feasible:
        return min(feasible, key=lambda item: (item[2], item[1], VARIANTS.index(item[0])))[0]
    return min(candidates, key=lambda item: (item[1], item[2], VARIANTS.index(item[0])))[0]


def _fit_monotonic_separable_rule(
    train: list[dict[str, Any]],
    complexity_by_prompt: dict[str, float],
    edges: list[float],
    max_violation: float,
) -> tuple[list[str], dict[int, int], list[int]]:
    """Fit rank(progress) + offset(complexity), without an interaction term."""
    byte_order = sorted(
        VARIANTS,
        key=lambda variant: _mean(cell["variants"][variant]["bytes"] for cell in train),
    )
    rank_count = len(byte_order)
    best: tuple[tuple[float, float, float], dict[int, int], list[int]] | None = None
    progress_candidates = (
        ranks
        for ranks in itertools.product(range(rank_count), repeat=len(CHECKPOINT_STEPS))
        if all(ranks[index] >= ranks[index + 1] for index in range(len(ranks) - 1))
    )
    bin_count = len(edges) - 1
    offset_candidates = [
        [0, *tail]
        for tail in itertools.combinations_with_replacement(range(rank_count), max(0, bin_count - 1))
    ]
    for progress_ranks in progress_candidates:
        progress_rule = dict(zip(CHECKPOINT_STEPS, progress_ranks))
        for offsets in offset_candidates:
            violations = 0
            total_bytes = 0.0
            representation_errors = 0
            for cell in train:
                bin_index = _bin(complexity_by_prompt[cell["prompt_id"]], edges)
                rank = min(rank_count - 1, progress_rule[cell["step"]] + offsets[bin_index])
                variant = byte_order[rank]
                if not cell["variants"][variant][f"safe_{cell['target']}"]:
                    violations += 1
                total_bytes += cell["variants"][variant]["bytes"]
                representation_errors += int(variant != cell["row"]["selected_representation"])
            violation_rate = violations / len(train)
            mean_bytes = total_bytes / len(train)
            representation_error_rate = representation_errors / len(train)
            objective = (
                0.0 if violation_rate <= max_violation else 1.0,
                mean_bytes if violation_rate <= max_violation else violation_rate,
                representation_error_rate if violation_rate <= max_violation else mean_bytes,
            )
            if best is None or objective < best[0]:
                best = (objective, progress_rule, offsets)
    if best is None:
        raise ValueError("Unable to fit simple separable rule")
    return byte_order, best[1], best[2]


def build_policy_predictions(
    cells: dict[tuple[str, int, float], dict[str, Any]],
    targets: list[float],
    complexity_bins: int,
    max_training_violation: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, float, str], str]]:
    prompts = sorted({key[0] for key in cells})
    predictions: dict[tuple[str, int, float, str], str] = {}
    detailed = []
    for target in targets:
        for held_out in prompts:
            train = []
            for (prompt_id, step, cell_target), cell in cells.items():
                if prompt_id != held_out and math.isclose(cell_target, target):
                    train.append({**cell, "prompt_id": prompt_id, "step": step, "target": target})
            complexity_by_prompt = {
                prompt_id: _mean(
                    cells[(prompt_id, step, target)]["content_complexity"] for step in CHECKPOINT_STEPS
                )
                for prompt_id in prompts
                if prompt_id != held_out
            }
            edges = _quantile_edges(list(complexity_by_prompt.values()), complexity_bins)
            progress_rules = {
                step: _choose_group_representation(
                    [cell for cell in train if cell["step"] == step], max_training_violation
                )
                for step in CHECKPOINT_STEPS
            }
            content_rules = {}
            for bin_index in range(complexity_bins):
                group_cells = [
                    cell
                    for cell in train
                    if _bin(complexity_by_prompt[cell["prompt_id"]], edges) == bin_index
                ]
                content_rules[bin_index] = _choose_group_representation(
                    group_cells or train, max_training_violation
                )
            separable_order, separable_progress, separable_offsets = _fit_monotonic_separable_rule(
                train,
                complexity_by_prompt,
                edges,
                max_training_violation,
            )
            held_complexity = _mean(
                cells[(held_out, step, target)]["content_complexity"] for step in CHECKPOINT_STEPS
            )
            held_bin = _bin(held_complexity, edges)
            for step in CHECKPOINT_STEPS:
                cell = cells[(held_out, step, target)]
                progress_rep = progress_rules[step]
                content_rep = content_rules[held_bin]
                separable_rank = min(
                    len(separable_order) - 1,
                    separable_progress[step] + separable_offsets[held_bin],
                )
                separable_rep = separable_order[separable_rank]
                for policy, variant in (
                    ("progress_only", progress_rep),
                    ("content_only", content_rep),
                    ("simple_separable", separable_rep),
                ):
                    predictions[(held_out, step, target, policy)] = variant
                    oracle = cell["row"]["selected_representation"]
                    info = cell["variants"][variant]
                    safe = bool(info[f"safe_{target}"])
                    oracle_bytes = float(cell["row"]["selected_total_checkpoint_bytes"])
                    chosen_bytes = float(info["bytes"])
                    detailed.append(
                        {
                            "quality_target": target,
                            "policy": policy,
                            "prompt_id": held_out,
                            "checkpoint_step": step,
                            "predicted_representation": variant,
                            "oracle_representation": oracle,
                            "representation_correct": variant == oracle,
                            "quality_slo_satisfied": safe,
                            "chosen_checkpoint_bytes": chosen_bytes,
                            "oracle_checkpoint_bytes": oracle_bytes,
                            "excess_checkpoint_bytes": max(0.0, chosen_bytes - oracle_bytes),
                            "oracle_gap_bytes": chosen_bytes - oracle_bytes,
                            "evaluation": "leave_one_prompt_out",
                        }
                    )
    return detailed, predictions


def build_global_policy_predictions(
    cells: dict[tuple[str, int, float], dict[str, Any]],
    targets: list[float],
    complexity_bins: int,
    max_training_violation: float,
) -> tuple[dict[tuple[str, int, float, str], str], list[dict[str, Any]]]:
    """Fit one common policy per target for finite-budget evaluation.

    This intentionally fits on all measured prompts. It is a conservative,
    privileged upper bound for a simple policy, not an out-of-sample or
    deployable predictor: content complexity is measured from the final video.
    """
    prompts = sorted({key[0] for key in cells})
    predictions: dict[tuple[str, int, float, str], str] = {}
    definitions: list[dict[str, Any]] = []
    for target in targets:
        train = [
            {**cell, "prompt_id": prompt_id, "step": step, "target": target}
            for (prompt_id, step, cell_target), cell in cells.items()
            if math.isclose(cell_target, target)
        ]
        complexity_by_prompt = {
            prompt_id: _mean(
                cells[(prompt_id, step, target)]["content_complexity"]
                for step in CHECKPOINT_STEPS
            )
            for prompt_id in prompts
        }
        edges = _quantile_edges(list(complexity_by_prompt.values()), complexity_bins)
        progress_rules = {
            step: _choose_group_representation(
                [cell for cell in train if cell["step"] == step],
                max_training_violation,
            )
            for step in CHECKPOINT_STEPS
        }
        content_rules = {}
        for bin_index in range(complexity_bins):
            group_cells = [
                cell
                for cell in train
                if _bin(complexity_by_prompt[cell["prompt_id"]], edges) == bin_index
            ]
            content_rules[bin_index] = _choose_group_representation(
                group_cells or train, max_training_violation
            )
        separable_order, separable_progress, separable_offsets = _fit_monotonic_separable_rule(
            train, complexity_by_prompt, edges, max_training_violation
        )
        for prompt_id in prompts:
            content_bin = _bin(complexity_by_prompt[prompt_id], edges)
            for step in CHECKPOINT_STEPS:
                separable_rank = min(
                    len(separable_order) - 1,
                    separable_progress[step] + separable_offsets[content_bin],
                )
                predictions[(prompt_id, step, target, "progress_only")] = progress_rules[step]
                predictions[(prompt_id, step, target, "content_only")] = content_rules[content_bin]
                predictions[(prompt_id, step, target, "simple_separable")] = separable_order[
                    separable_rank
                ]
        definitions.append(
            {
                "quality_target": target,
                "evaluation": "global_in_sample_privileged_upper_bound",
                "deployable": False,
                "privileged_feature": "final_video_content_complexity",
                "complexity_bin_edges": edges,
                "progress_only_rule": progress_rules,
                "content_only_rule": content_rules,
                "simple_separable_byte_order": separable_order,
                "simple_separable_progress_rank": separable_progress,
                "simple_separable_content_offsets": separable_offsets,
            }
        )
    return predictions, definitions


def summarize_separability(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (target, policy), group in sorted(_group(details, "quality_target", "policy").items()):
        output.append(
            {
                "quality_target": target,
                "policy": policy,
                "num_prompt_step_cells": len(group),
                "representation_prediction_accuracy": _mean(
                    1.0 if row["representation_correct"] else 0.0 for row in group
                ),
                "quality_slo_violation_rate": _mean(
                    0.0 if row["quality_slo_satisfied"] else 1.0 for row in group
                ),
                "mean_excess_checkpoint_bytes": _mean(float(row["excess_checkpoint_bytes"]) for row in group),
                "median_excess_checkpoint_bytes": _median(float(row["excess_checkpoint_bytes"]) for row in group),
                "mean_oracle_gap_bytes": _mean(float(row["oracle_gap_bytes"]) for row in group),
                "evaluation": "leave_one_prompt_out",
            }
        )
    return output


def build_interaction_crossings(
    minimum_rows: list[dict[str, Any]],
    cells: dict[tuple[str, int, float], dict[str, Any]],
    expected_seeds: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_key = {
        (row["prompt_id"], int(row["checkpoint_step"]), float(row["quality_target"])): row
        for row in minimum_rows
    }
    prompts = sorted({row["prompt_id"] for row in minimum_rows})
    categories = {row["prompt_id"]: row["category"] for row in minimum_rows}
    output = []
    for target in QUALITY_TARGETS:
        for index, prompt_a in enumerate(prompts):
            for prompt_b in prompts[index + 1 :]:
                orders: dict[int, str] = {}
                filtered_orders: dict[int, str] = {}
                stability = 1.0
                seed_minimum_bytes = {
                    prompt_a: {step: [] for step in CHECKPOINT_STEPS},
                    prompt_b: {step: [] for step in CHECKPOINT_STEPS},
                }
                for step in CHECKPOINT_STEPS:
                    row_a = by_key[(prompt_a, step, target)]
                    row_b = by_key[(prompt_b, step, target)]
                    a = float(row_a["selected_total_checkpoint_bytes"])
                    b = float(row_b["selected_total_checkpoint_bytes"])
                    if not math.isfinite(a) or not math.isfinite(b):
                        order = "ineligible"
                    elif a < b:
                        order = "a_lt_b"
                    elif a > b:
                        order = "a_gt_b"
                    else:
                        order = "equal"
                    orders[step] = order
                    noise_ok = row_a["decision_above_noise_floor"] is True and row_b["decision_above_noise_floor"] is True
                    filtered_orders[step] = order if noise_ok else "noise_unresolved"
                    stability = min(
                        stability,
                        float(row_a["selected_seed_safe_fraction"]),
                        float(row_b["selected_seed_safe_fraction"]),
                    )
                    for prompt_id in (prompt_a, prompt_b):
                        for seed_index in range(expected_seeds):
                            variant_stats = cells[(prompt_id, step, target)]["variants"]
                            safe = [
                                info
                                for info in variant_stats.values()
                                if all(
                                    info["metric_values"][metric][seed_index] >= target
                                    for metric in QUALITY_METRICS
                                )
                            ]
                            seed_minimum_bytes[prompt_id][step].append(min(
                                (float(info["bytes"]) for info in safe), default=float("inf")
                            ))
                non_equal = [value for value in orders.values() if value in {"a_lt_b", "a_gt_b"}]
                filtered_non_equal = [
                    value for value in filtered_orders.values() if value in {"a_lt_b", "a_gt_b"}
                ]
                def seed_pair_crossing(seed_a: int, seed_b: int) -> float:
                    pair_orders = []
                    for step in CHECKPOINT_STEPS:
                        a_bytes = seed_minimum_bytes[prompt_a][step][seed_a]
                        b_bytes = seed_minimum_bytes[prompt_b][step][seed_b]
                        if a_bytes < b_bytes:
                            pair_orders.append("a_lt_b")
                        elif a_bytes > b_bytes:
                            pair_orders.append("a_gt_b")
                    return 1.0 if len(set(pair_orders)) > 1 else 0.0

                seed_crossings = [
                    seed_pair_crossing(seed_a, seed_b)
                    for seed_a in range(expected_seeds)
                    for seed_b in range(expected_seeds)
                ]
                rng = random.Random(
                    bootstrap_seed + index * 101 + int(round(target * 1000))
                )
                bootstrap_crossing_fractions = []
                for _ in range(bootstrap_samples):
                    sampled_a = [rng.randrange(expected_seeds) for _ in range(expected_seeds)]
                    sampled_b = [rng.randrange(expected_seeds) for _ in range(expected_seeds)]
                    bootstrap_crossing_fractions.append(
                        _mean(seed_pair_crossing(seed_a, seed_b) for seed_a in sampled_a for seed_b in sampled_b)
                    )
                crossing_ci = (
                    _percentile(bootstrap_crossing_fractions, 0.025),
                    _percentile(bootstrap_crossing_fractions, 0.975),
                )
                output.append(
                    {
                        "quality_target": target,
                        "prompt_id_a": prompt_a,
                        "prompt_id_b": prompt_b,
                        "category_a": categories[prompt_a],
                        "category_b": categories[prompt_b],
                        **{f"step_{step}_order": orders[step] for step in CHECKPOINT_STEPS},
                        "has_crossing": len(set(non_equal)) > 1,
                        **{f"step_{step}_noise_filtered_order": filtered_orders[step] for step in CHECKPOINT_STEPS},
                        "has_noise_filtered_crossing": len(set(filtered_non_equal)) > 1,
                        "eligible_noise_filtered_steps": len(filtered_non_equal),
                        "minimum_seed_safe_fraction": stability,
                        "seed_crossing_count": int(sum(seed_crossings)),
                        "seed_crossing_fraction": _mean(seed_crossings),
                        "seed_crossing_ci_low": crossing_ci[0],
                        "seed_crossing_ci_high": crossing_ci[1],
                        "seed_count_per_prompt": expected_seeds,
                        "seed_pair_count": len(seed_crossings),
                        "seed_bootstrap_method": "independent_two_way_seed_resampling",
                    }
                )
    return output


def build_progress_dependence(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> list[dict[str, Any]]:
    grouped = _group(rows, "prompt_id", "checkpoint_step", "variant")
    prompts = sorted({row["prompt_id"] for row in rows})
    output = []
    for variant_index, variant in enumerate(VARIANTS):
        if variant == "full":
            continue
        for metric_index, metric in enumerate(QUALITY_METRICS):
            prompt_effects = []
            for prompt_id in prompts:
                early = grouped[(prompt_id, "10", variant)]
                late = grouped[(prompt_id, "30", variant)]
                prompt_effects.append(
                    _mean(_float(row, metric) for row in late)
                    - _mean(_float(row, metric) for row in early)
                )
            stats = metric_stats(prompt_effects, samples, seed + variant_index * 11 + metric_index)
            output.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "effect": "step30_minus_step10_relative_quality",
                    "prompt_count": len(prompt_effects),
                    **stats,
                }
            )
    return output


def run_budget_simulation(
    cells: dict[tuple[str, int, float], dict[str, Any]],
    predictions: dict[tuple[str, int, float, str], str],
    session_counts: list[int],
    budget_fractions: list[float],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_cells = []
    for (prompt_id, step, target), cell in cells.items():
        if cell["row"]["selected_representation"] == "none":
            continue
        all_cells.append({**cell, "prompt_id": prompt_id, "step": step, "target": target})
    scenarios: dict[str, list[dict[str, Any]]] = {
        f"target_{target}": [cell for cell in all_cells if math.isclose(cell["target"], target)]
        for target in QUALITY_TARGETS
    }
    scenarios["mixed_targets"] = all_cells
    mixed_by_session: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in all_cells:
        mixed_by_session[(cell["prompt_id"], cell["step"])].append(cell)
    output = []
    for scenario, pool in scenarios.items():
        for session_count in session_counts:
            available_sessions = len(mixed_by_session) if scenario == "mixed_targets" else len(pool)
            if available_sessions < session_count:
                raise ValueError(f"scenario {scenario} has only {len(pool)} cells for N={session_count}")
            for trial_index in range(trials):
                if scenario == "mixed_targets":
                    session_keys = rng.sample(sorted(mixed_by_session), session_count)
                    sampled = [rng.choice(mixed_by_session[key]) for key in session_keys]
                else:
                    sampled = rng.sample(pool, session_count)
                sampled_session_ids = [f"{cell['prompt_id']}:step{cell['step']}" for cell in sampled]
                if len(sampled_session_ids) != len(set(sampled_session_ids)):
                    raise AssertionError("a budget trial sampled the same base session more than once")
                full_total = sum(cell["variants"]["full"]["bytes"] for cell in sampled)
                for budget_fraction in budget_fractions:
                    budget = full_total * budget_fraction
                    trial_rows = []
                    for policy in POLICIES:
                        candidates = []
                        for cell in sampled:
                            key = (cell["prompt_id"], cell["step"], cell["target"])
                            if policy == "joint_oracle":
                                variant = cell["row"]["selected_representation"]
                            elif policy == "full":
                                variant = "full"
                            elif policy == "uniform_int8":
                                variant = "int8"
                            else:
                                variant = predictions[(*key, policy)]
                            info = cell["variants"][variant]
                            candidates.append((cell, variant, info))
                        candidates.sort(key=lambda item: (item[2]["bytes"], item[0]["prompt_id"], item[0]["step"]))
                        selected = []
                        used = 0.0
                        for cell, variant, info in candidates:
                            if used + info["bytes"] <= budget:
                                selected.append((cell, variant, info))
                                used += info["bytes"]
                        satisfying = [
                            item for item in selected if item[2][f"safe_{item[0]['target']}"]
                        ]
                        violations = len(selected) - len(satisfying)
                        wasted = sum(
                            max(0.0, info["bytes"] - float(cell["row"]["selected_total_checkpoint_bytes"]))
                            for cell, _, info in selected
                        )
                        trial_rows.append(
                            {
                                "mixture": scenario,
                                "session_count": session_count,
                                "budget_fraction_of_all_full": budget_fraction,
                                "trial_index": trial_index,
                                "policy": policy,
                                "sampled_unique_session_count": len(set(sampled_session_ids)),
                                "sampled_session_ids_json": json.dumps(sampled_session_ids),
                                "budget_bytes": budget,
                                "selected_sessions": len(selected),
                                "sessions_satisfying_quality_target": len(satisfying),
                                "fraction_sessions_satisfying_quality_target": len(satisfying) / session_count,
                                "quality_violation_count": violations,
                                "quality_violation_rate_among_selected": violations / len(selected) if selected else 0.0,
                                "total_checkpoint_bytes": used,
                                "average_excess_bytes_per_selected_session": wasted / len(selected) if selected else 0.0,
                                "wasted_bytes_above_minimum_safe": wasted,
                            }
                        )
                    oracle = next(row for row in trial_rows if row["policy"] == "joint_oracle")
                    for row in trial_rows:
                        gap = oracle["sessions_satisfying_quality_target"] - row["sessions_satisfying_quality_target"]
                        row["absolute_oracle_gap_sessions"] = gap
                        row["relative_oracle_gap"] = gap / max(oracle["sessions_satisfying_quality_target"], 1)
                        if row["policy"] == "simple_separable":
                            row["oracle_relative_improvement_over_simple"] = gap / max(
                                row["sessions_satisfying_quality_target"], 1
                            )
                        else:
                            row["oracle_relative_improvement_over_simple"] = ""
                    if [row["policy"] for row in trial_rows] != POLICIES:
                        raise AssertionError("budget policy set/order is not the preregistered six-policy set")
                    output.extend(trial_rows)
    unique_keys = {
        (row["mixture"], row["session_count"], row["budget_fraction_of_all_full"], row["trial_index"], row["policy"])
        for row in output
    }
    if len(unique_keys) != len(output):
        raise AssertionError("budget simulator emitted duplicate policy rows")
    return output


def _fraction(rows: list[dict[str, Any]], predicate) -> float:
    return _mean(1.0 if predicate(row) else 0.0 for row in rows) if rows else float("nan")


def _cramers_v(rows: list[dict[str, Any]]) -> float:
    categories = sorted({row["category"] for row in rows})
    representations = sorted({row["selected_representation"] for row in rows})
    if len(categories) < 2 or len(representations) < 2:
        return 0.0
    counts = Counter((row["category"], row["selected_representation"]) for row in rows)
    category_totals = Counter(row["category"] for row in rows)
    representation_totals = Counter(row["selected_representation"] for row in rows)
    total = len(rows)
    chi_square = 0.0
    for category in categories:
        for representation in representations:
            expected = category_totals[category] * representation_totals[representation] / total
            if expected:
                observed = counts[(category, representation)]
                chi_square += (observed - expected) ** 2 / expected
    denominator = total * min(len(categories) - 1, len(representations) - 1)
    return math.sqrt(chi_square / denominator) if denominator else 0.0


def build_summary_and_judgment(
    validation: dict[str, Any],
    frontier_summary: dict[str, Any],
    progress_effects: list[dict[str, Any]],
    iso_rows: list[dict[str, Any]],
    minimum_rows: list[dict[str, Any]],
    crossings: list[dict[str, Any]],
    separability: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    analysis_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    thresholds = analysis_config["judgment"]
    strong_iso = [
        row
        for row in iso_rows
        if bool(row["resolved_at_seed_count"])
        and abs(float(row["dynamic_delta_mean"]))
        >= float(thresholds["strong_iso_minimum_absolute_dynamic_delta"])
    ]
    progress_change_by_target = {}
    aggressive_safe_later_by_target = {}
    for target in QUALITY_TARGETS:
        target_rows = [row for row in minimum_rows if math.isclose(float(row["quality_target"]), target)]
        grouped = _group(target_rows, "prompt_id")
        progress_change_by_target[str(target)] = _fraction(
            [{"changes": len({row["selected_representation"] for row in group}) > 1} for group in grouped.values()],
            lambda row: row["changes"],
        )
        byte_changes = []
        for group in grouped.values():
            by_step = {int(row["checkpoint_step"]): row for row in group}
            early = float(by_step[10]["selected_total_checkpoint_bytes"])
            late = float(by_step[30]["selected_total_checkpoint_bytes"])
            if math.isfinite(early) and math.isfinite(late):
                byte_changes.append((early - late) / early)
        aggressive_safe_later_by_target[str(target)] = {
            "fraction_prompts_smaller_Rstar_at_step30": _mean(
                1.0 if change > 0.0 else 0.0 for change in byte_changes
            ),
            "fraction_prompts_not_smaller_at_step30": _mean(
                1.0 if change <= 0.0 else 0.0 for change in byte_changes
            ),
            "mean_relative_Rstar_byte_reduction_step10_to_step30": _mean(byte_changes),
        }
    content_variants_by_target_step = {}
    content_category_association = {}
    content_category_distributions = {}
    minimum_representation_counts: dict[str, dict[str, int]] = {}
    for target in QUALITY_TARGETS:
        target_rows = [row for row in minimum_rows if math.isclose(float(row["quality_target"]), target)]
        minimum_representation_counts[str(target)] = dict(
            sorted(Counter(row["selected_representation"] for row in target_rows).items())
        )
        for step in CHECKPOINT_STEPS:
            target_step_rows = [
                row
                for row in minimum_rows
                if math.isclose(float(row["quality_target"]), target)
                and int(row["checkpoint_step"]) == step
            ]
            reps = {
                row["selected_representation"] for row in target_step_rows
            }
            content_variants_by_target_step[f"{target}_step_{step}"] = sorted(reps)
            content_category_association[f"{target}_step_{step}"] = _cramers_v(target_step_rows)
            content_category_distributions[f"{target}_step_{step}"] = {
                category: dict(
                    sorted(
                        Counter(
                            row["selected_representation"]
                            for row in target_step_rows
                            if row["category"] == category
                        ).items()
                    )
                )
                for category in sorted({row["category"] for row in target_step_rows})
            }
    crossing_summary = {}
    for target in QUALITY_TARGETS:
        rows = [row for row in crossings if math.isclose(float(row["quality_target"]), target)]
        crossing_summary[str(target)] = {
            "eligible_pairs": len(rows),
            "raw_crossing_fraction": _fraction(rows, lambda row: bool(row["has_crossing"])),
            "mean_seed_crossing_fraction": _mean(
                float(row["seed_crossing_fraction"]) for row in rows
            ),
            "noise_filtered_crossing_fraction": _fraction(
                rows, lambda row: bool(row["has_noise_filtered_crossing"])
            ),
            "noise_filter_resolved_pair_fraction": _fraction(
                rows,
                lambda row: int(row["eligible_noise_filtered_steps"])
                >= int(
                    analysis_config["interaction_crossing"][
                        "minimum_noise_resolved_steps_for_pair"
                    ]
                ),
            ),
        }
    separability_map = {
        (float(row["quality_target"]), row["policy"]): row for row in separability
    }
    budget_simple = [row for row in budget_rows if row["policy"] == "simple_separable"]
    budget_aggregate = []
    budget_groups = _group(
        budget_rows,
        "mixture",
        "session_count",
        "budget_fraction_of_all_full",
        "policy",
    )
    for (mixture, session_count, budget_fraction, policy), group in sorted(budget_groups.items()):
        budget_aggregate.append(
            {
                "mixture": mixture,
                "session_count": int(session_count),
                "budget_fraction_of_all_full": float(budget_fraction),
                "policy": policy,
                "mean_sessions_satisfying_quality_target": _mean(
                    float(row["sessions_satisfying_quality_target"]) for row in group
                ),
                "mean_quality_violation_rate_among_selected": _mean(
                    float(row["quality_violation_rate_among_selected"]) for row in group
                ),
                "mean_total_checkpoint_bytes": _mean(float(row["total_checkpoint_bytes"]) for row in group),
                "mean_relative_oracle_gap": _mean(float(row["relative_oracle_gap"]) for row in group),
            }
        )
    budget_signal_by_n = {}
    for n in (3, 8, 20):
        rows = [row for row in budget_simple if int(row["session_count"]) == n]
        budget_signal_by_n[str(n)] = {
            "mean_relative_oracle_gap": _mean(float(row["relative_oracle_gap"]) for row in rows),
            "mean_oracle_relative_improvement_over_simple": _mean(
                float(row["oracle_relative_improvement_over_simple"] or 0.0) for row in rows
            ),
            "fraction_conditions_meeting_preregistered_strong_gap": _fraction(
                rows,
                lambda row: float(row["oracle_relative_improvement_over_simple"] or 0.0)
                >= float(thresholds["strong_gap_minimum_relative_improvement"]),
            ),
        }
    compression_noise_rows = [
        row
        for row in noise_rows
        if row.get("metric") in QUALITY_METRICS and row.get("variant") in {"fp16", "int8"}
    ]
    noise_above_fraction = _fraction(
        compression_noise_rows,
        lambda row: str(row.get("gap_above_noise_floor", "")).lower() == "true",
    ) if compression_noise_rows else float("nan")
    noise_ordering_stable_fraction = _fraction(
        compression_noise_rows,
        lambda row: str(row.get("ordering_stable", "")).lower() == "true",
    ) if compression_noise_rows else float("nan")
    noise_99_ambiguous_fraction = _fraction(
        compression_noise_rows,
        lambda row: 0.0 < float(row.get("probability_ge_0_99", 0.0)) < 1.0,
    ) if compression_noise_rows else float("nan")
    measured_minimum_decisions = [
        row
        for row in minimum_rows
        if row["noise_floor_status"] in {"threshold_stable", "threshold_ambiguous"}
    ]
    stable_minimum_decision_fraction = _fraction(
        measured_minimum_decisions,
        lambda row: row["decision_above_noise_floor"] is True,
    ) if measured_minimum_decisions else float("nan")
    measured_99_decisions = [
        row
        for row in measured_minimum_decisions
        if math.isclose(float(row["quality_target"]), 0.99)
    ]
    stable_99_decisions = [
        row for row in measured_99_decisions if row["decision_above_noise_floor"] is True
    ]
    stable_99_decision_fraction = _fraction(
        measured_99_decisions,
        lambda row: row["decision_above_noise_floor"] is True,
    ) if measured_99_decisions else float("nan")
    stable_99_representations = {
        row["selected_representation"] for row in stable_99_decisions
    }
    non_99_budget = [
        row for row in budget_simple if row["mixture"] in {"target_0.95", "target_0.975"}
    ]
    non_99_mean_gap = _mean(
        float(row["oracle_relative_improvement_over_simple"] or 0.0)
        for row in non_99_budget
    )

    def budget_gap_is_broad(
        threshold: float,
        minimum_budget_levels: int,
        minimum_mixtures: int,
        minimum_session_counts: int,
    ) -> bool:
        qualifying_n = 0
        for n in (3, 8, 20):
            rows = [row for row in budget_simple if int(row["session_count"]) == n]
            condition_means = {
                (mixture, float(budget_fraction)): _mean(
                    float(row["oracle_relative_improvement_over_simple"] or 0.0)
                    for row in group
                )
                for (mixture, budget_fraction), group in _group(
                    rows, "mixture", "budget_fraction_of_all_full"
                ).items()
            }
            qualifying = {
                condition for condition, mean_gap in condition_means.items() if mean_gap >= threshold
            }
            budget_levels = {budget_fraction for _, budget_fraction in qualifying}
            mixtures = {mixture for mixture, _ in qualifying}
            if len(budget_levels) >= minimum_budget_levels and len(mixtures) >= minimum_mixtures:
                qualifying_n += 1
        return qualifying_n >= minimum_session_counts

    strong_simple_gap = budget_gap_is_broad(
        float(thresholds["strong_gap_minimum_relative_improvement"]),
        int(thresholds["strong_gap_minimum_budget_levels"]),
        int(thresholds["strong_gap_minimum_mixtures"]),
        int(thresholds["strong_gap_minimum_session_counts"]),
    )
    moderate_simple_gap = budget_gap_is_broad(
        float(thresholds["conditional_gap_minimum_relative_improvement"]),
        int(thresholds["conditional_gap_minimum_budget_levels"]),
        int(thresholds["conditional_gap_minimum_mixtures"]),
        int(thresholds["conditional_gap_minimum_session_counts"]),
    )
    representation_real = len(strong_iso) >= int(
        thresholds["strong_iso_minimum_resolved_cells"]
    )
    heterogeneity_real = any(
        value >= float(thresholds["heterogeneity_minimum_progress_change_fraction"])
        for value in progress_change_by_target.values()
    ) and any(
        len(reps) >= int(thresholds["heterogeneity_minimum_representations_in_content_cell"])
        for reps in content_variants_by_target_step.values()
    )
    noise_supported = (
        bool(noise_rows)
        and len(measured_99_decisions)
        >= int(thresholds["noise_minimum_measured_99pct_decisions"])
        and stable_99_decision_fraction
        >= float(thresholds["noise_minimum_stable_99pct_decision_fraction"])
        and len(stable_99_representations)
        >= int(thresholds["noise_minimum_stable_99pct_representations"])
    )
    not_only_99 = non_99_mean_gap >= float(
        thresholds["effect_not_confined_to_99pct_minimum_mean_relative_oracle_gap"]
    )
    if representation_real and heterogeneity_real and noise_supported and strong_simple_gap and not_only_99:
        judgment = "STRONG GO"
    elif representation_real and heterogeneity_real and noise_supported and moderate_simple_gap:
        judgment = "CONDITIONAL GO"
    else:
        judgment = "NO-GO"
    summary = {
        "validation": validation,
        "statistical_unit": {
            "within_prompt": "seed",
            "population_aggregate": "prompt",
            "population_ci": "bootstrap_over_prompt_level_effects",
        },
        "preregistered_analysis_config": analysis_config,
        "frontier_summary": frontier_summary,
        "iso_storage": {
            "resolved_cells": sum(bool(row["resolved_at_seed_count"]) for row in iso_rows),
            "strong_dynamic_cells_abs_delta_gte_0_10": len(strong_iso),
            "rows": iso_rows,
        },
        "minimum_safe_representation_counts": minimum_representation_counts,
        "progress_change_fraction_by_target": progress_change_by_target,
        "aggressive_compression_safe_later_by_target": aggressive_safe_later_by_target,
        "progress_quality_effects": progress_effects,
        "content_representations_by_target_step": content_variants_by_target_step,
        "content_category_cramers_v_by_target_step": content_category_association,
        "content_category_representation_distribution": content_category_distributions,
        "interaction_crossings": crossing_summary,
        "separability": separability,
        "simple_separable_99": separability_map.get((0.99, "simple_separable")),
        "budget_simple_vs_joint_by_n": budget_signal_by_n,
        "budget_aggregate": budget_aggregate,
        "non_99_mean_relative_oracle_gap": non_99_mean_gap,
        "noise_floor": {
            "row_count": len(noise_rows),
            "compression_metric_cells_above_noise_fraction": noise_above_fraction,
            "ordering_stable_fraction": noise_ordering_stable_fraction,
            "ambiguous_99pct_threshold_crossing_fraction": noise_99_ambiguous_fraction,
            "measured_minimum_decision_count": len(measured_minimum_decisions),
            "stable_minimum_decision_fraction": stable_minimum_decision_fraction,
            "measured_99pct_minimum_decision_count": len(measured_99_decisions),
            "stable_99pct_minimum_decision_fraction": stable_99_decision_fraction,
            "stable_99pct_representations": sorted(stable_99_representations),
        },
        "judgment_evidence": {
            "representation_dependent_recoverability": representation_real,
            "progress_or_content_heterogeneity": heterogeneity_real,
            "high_fidelity_effect_above_noise": noise_supported,
            "simple_vs_joint_gap_ge_15pct_for_multiple_N": strong_simple_gap,
            "simple_vs_joint_gap_ge_5pct_for_multiple_N": moderate_simple_gap,
            "effect_not_confined_to_99pct": not_only_99,
        },
        "judgment": judgment,
    }
    return summary, judgment


def maybe_make_figures(
    output_dir: Path,
    frontier_summary: dict[str, Any],
    minimum_rows: list[dict[str, Any]],
    iso_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[str] = []
    markers = {10: "o", 20: "s", 30: "^"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for step in CHECKPOINT_STEPS:
        xs = [
            frontier_summary[f"step_{step}_{variant}"]["total_checkpoint_bytes"]["mean"]
            for variant in VARIANTS
        ]
        ys = [
            frontier_summary[f"step_{step}_{variant}"]["temporal_dynamic_vs_full"]["mean"]
            for variant in VARIANTS
        ]
        ax.scatter(xs, ys, marker=markers[step], label=f"step {step}")
        for x, y, variant in zip(xs, ys, VARIANTS, strict=True):
            ax.annotate(variant, (x, y), fontsize=6)
    ax.set_xlabel("Mean serialized checkpoint bytes")
    ax.set_ylabel("Mean temporal/dynamic quality vs full")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "figure1_footprint_quality_frontier.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    target = 0.99
    prompts = sorted({row["prompt_id"] for row in minimum_rows})
    representation_index = {"none": -1, **{variant: index for index, variant in enumerate(VARIANTS)}}
    by_minimum = {
        (row["prompt_id"], int(row["checkpoint_step"])): row
        for row in minimum_rows
        if math.isclose(float(row["quality_target"]), target)
    }
    heat = [
        [representation_index[by_minimum[(prompt_id, step)]["selected_representation"]] for step in CHECKPOINT_STEPS]
        for prompt_id in prompts
    ]
    fig, ax = plt.subplots(figsize=(6, max(4, len(prompts) * 0.35)))
    image = ax.imshow(heat, aspect="auto", vmin=-1, vmax=len(VARIANTS) - 1)
    ax.set_xticks(range(len(CHECKPOINT_STEPS)), [str(step) for step in CHECKPOINT_STEPS])
    ax.set_yticks(range(len(prompts)), prompts, fontsize=6)
    ax.set_xlabel("Checkpoint step")
    ax.set_title("Minimum safe representation at 99% target")
    colorbar = fig.colorbar(image, ax=ax, ticks=range(-1, len(VARIANTS)))
    colorbar.ax.set_yticklabels(["none", *VARIANTS])
    fig.tight_layout()
    path = output_dir / "figure2_minimum_safe_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, session_count in zip(axes, (3, 8, 20), strict=True):
        for policy in ("simple_separable", "joint_oracle"):
            policy_rows = [
                row
                for row in budget_rows
                if row["mixture"] == "mixed_targets"
                and int(row["session_count"]) == session_count
                and row["policy"] == policy
            ]
            xs = sorted({float(row["budget_fraction_of_all_full"]) for row in policy_rows})
            ys = [
                _mean(
                    float(row["sessions_satisfying_quality_target"])
                    for row in policy_rows
                    if math.isclose(float(row["budget_fraction_of_all_full"]), fraction)
                )
                for fraction in xs
            ]
            ax.plot(xs, ys, marker="o", label=policy)
        ax.set_title(f"N={session_count}")
        ax.set_xlabel("Budget / all-full bytes")
    axes[0].set_ylabel("Quality-satisfying sessions")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "figure3_separable_vs_oracle.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in sorted({row["pair"] for row in iso_rows}):
        pair_rows = sorted(
            [row for row in iso_rows if row["pair"] == pair],
            key=lambda row: int(row["checkpoint_step"]),
        )
        xs = [int(row["checkpoint_step"]) for row in pair_rows]
        ys = [float(row["dynamic_delta_mean"]) for row in pair_rows]
        lower = [y - float(row["dynamic_delta_ci_low"]) for y, row in zip(ys, pair_rows, strict=True)]
        upper = [float(row["dynamic_delta_ci_high"]) - y for y, row in zip(ys, pair_rows, strict=True)]
        ax.errorbar(xs, ys, yerr=[lower, upper], marker="o", label=pair)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Paired temporal/dynamic delta (A - B)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "figure4_iso_storage_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_final_report(path: Path, summary: dict[str, Any], judgment: str) -> None:
    evidence = summary["judgment_evidence"]
    lines = [
        "# Video State Protection Final Kill Test",
        "",
        "## Experimental Configuration",
        "",
        f"- Complete frontier rows: {summary['validation']['row_count']}.",
        f"- Prompts: {summary['validation']['prompt_count']}; seeds per prompt: {summary['validation']['seed_count_per_prompt']}.",
        "- Checkpoint steps: 10, 20, 30 of 40; representations: full, FP16, INT8, spatial/temporal reduction, and low-rank.",
        "- Within-prompt uncertainty uses seeds; population CIs bootstrap prompt-level effects.",
        "- Simple-policy evaluation is leave-one-prompt-out. Budget simulation uses fixed random mixtures and six unique policies.",
        "",
        "## Exact Checkpoint Footprints",
        "",
        "| Step | Representation | Mean bytes | Compression ratio |",
        "|---:|---|---:|---:|",
    ]
    for step in CHECKPOINT_STEPS:
        full_bytes = summary["frontier_summary"][f"step_{step}_full"]["total_checkpoint_bytes"]["mean"]
        for variant in VARIANTS:
            mean_bytes = summary["frontier_summary"][f"step_{step}_{variant}"]["total_checkpoint_bytes"]["mean"]
            lines.append(f"| {step} | {variant} | {mean_bytes:.1f} | {mean_bytes / full_bytes:.4f} |")
    lines.extend(
        [
            "",
            "## Prompt-Level Quality Distributions",
            "",
            "Values are means of prompt means with 95% prompt-bootstrap CIs.",
            "",
            "| Step | Representation | Spatial | Dynamic | Semantic |",
            "|---:|---|---|---|---|",
        ]
    )
    for step in CHECKPOINT_STEPS:
        for variant in VARIANTS:
            entry = summary["frontier_summary"][f"step_{step}_{variant}"]
            cells = []
            for metric in QUALITY_METRICS:
                stats = entry[metric]
                cells.append(
                    f"{stats['mean']:.4f} [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
                )
            lines.append(f"| {step} | {variant} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
        "",
        "## Iso-Storage Effects",
        "",
        "| Step | Pair | Dynamic delta mean | 95% prompt-bootstrap CI | Resolved |",
        "|---:|---|---:|---|---|",
        ]
    )
    for row in summary["iso_storage"]["rows"]:
        lines.append(
            f"| {row['checkpoint_step']} | {row['pair']} | {float(row['dynamic_delta_mean']):.4f} | "
            f"[{float(row['dynamic_delta_ci_low']):.4f}, {float(row['dynamic_delta_ci_high']):.4f}] | "
            f"{row['resolved_at_seed_count']} |"
        )
    lines.extend(
        [
        "",
        "## Minimum Safe Representation",
        "",
        f"- Counts by target: `{summary['minimum_safe_representation_counts']}`",
        f"- Progress-change fraction by target: `{summary['progress_change_fraction_by_target']}`",
        f"- Smaller safe state at step 30: `{summary['aggressive_compression_safe_later_by_target']}`",
        f"- Representations observed by target/step: `{summary['content_representations_by_target_step']}`",
        f"- Coarse-category association (Cramer's V): `{summary['content_category_cramers_v_by_target_step']}`",
        "",
        "## Noise Floor",
        "",
        f"- Repeated-recovery summary rows: {summary['noise_floor']['row_count']}.",
        f"- Compression metric cells above the preregistered 2-sigma noise rule: {summary['noise_floor']['compression_metric_cells_above_noise_fraction']}.",
        f"- Stable full/FP16/INT8 ordering fraction: {summary['noise_floor']['ordering_stable_fraction']}.",
        f"- Ambiguous 99% threshold-crossing fraction: {summary['noise_floor']['ambiguous_99pct_threshold_crossing_fraction']}.",
        f"- Threshold-resolved minimum-representation decisions: {summary['noise_floor']['measured_minimum_decision_count']} measured, stable fraction {summary['noise_floor']['stable_minimum_decision_fraction']}.",
        f"- At the 99% target: {summary['noise_floor']['measured_99pct_minimum_decision_count']} measured, stable fraction {summary['noise_floor']['stable_99pct_minimum_decision_fraction']}, stable representations {summary['noise_floor']['stable_99pct_representations']}.",
        "- Subjective side-by-side videos are sanity checks only and do not define a threshold.",
        "",
        "## Separability",
        "",
        "| Target | Policy | Accuracy | SLO violation | Mean excess bytes |",
        "|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["separability"]:
        lines.append(
            f"| {row['quality_target']} | {row['policy']} | "
            f"{float(row['representation_prediction_accuracy']):.4f} | "
            f"{float(row['quality_slo_violation_rate']):.4f} | "
            f"{float(row['mean_excess_checkpoint_bytes']):.1f} |"
        )
    lines.extend(
        [
        "",
        "The finite-budget simple policies use one common model per quality target. The content feature is computed from the final baseline video, so this is a privileged, non-deployable upper bound for a simple separable policy.",
        "",
        "## Interaction Crossings",
        "",
        f"- Raw and noise-filtered crossing results: `{summary['interaction_crossings']}`",
        "",
        "## Finite-Budget Simulation",
        "",
        f"- Simple-separable versus joint-oracle signal by N: `{summary['budget_simple_vs_joint_by_n']}`",
        f"- Mean non-99 relative oracle gap: {summary['non_99_mean_relative_oracle_gap']}",
        "",
        "Mixed-target means (all target-specific rows remain in `budget_simulation_fixed.csv`):",
        "",
        "| N | Budget | Policy | Quality-satisfying sessions | Violation rate | Relative oracle gap |",
        "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["budget_aggregate"]:
        if row["mixture"] != "mixed_targets" or row["policy"] not in {"simple_separable", "joint_oracle"}:
            continue
        lines.append(
            f"| {row['session_count']} | {float(row['budget_fraction_of_all_full']):.2f} | {row['policy']} | "
            f"{float(row['mean_sessions_satisfying_quality_target']):.3f} | "
            f"{float(row['mean_quality_violation_rate_among_selected']):.4f} | "
            f"{float(row['mean_relative_oracle_gap']):.4f} |"
        )
    lines.extend(
        [
        "",
        "## Questions",
        "",
        f"- Q1 representation dependence: {evidence['representation_dependent_recoverability']}.",
        f"- Q2 progress dependence: R* changes are {summary['progress_change_fraction_by_target']}; late-state reductions are {summary['aggressive_compression_safe_later_by_target']}; paired quality effects with CIs are in `final_frontier_summary.json`.",
        f"- Q3 content dependence: R* distributions are {summary['content_category_representation_distribution']}; Cramer's V is {summary['content_category_cramers_v_by_target_step']}.",
        f"- Q4 above noise floor: {evidence['high_fidelity_effect_above_noise']}.",
        f"- Q5 non-separability: noise-filtered crossings are {summary['interaction_crossings']}.",
        f"- Q6 strongest simple separable policy: {summary.get('simple_separable_99')}.",
        f"- Q7 joint allocation advantage: {summary['budget_simple_vs_joint_by_n']}.",
        f"- Q8 dependence on 99%: non-99 mean relative oracle gap is {summary['non_99_mean_relative_oracle_gap']}.",
        f"- Q9 runtime justification: {judgment}; no runtime is designed in this round.",
        "",
        "## CONFIRMED",
        "",
        "- Exact serialized checkpoint bytes and separate spatial, dynamic, and semantic recovery metrics are used.",
        "- Iso-storage population CIs are computed from prompt-level paired effects, not pooled prompt-seed rows.",
        "- Every simulator cell contains exactly FULL, UNIFORM INT8, PROGRESS ONLY, CONTENT ONLY, SIMPLE SEPARABLE, and JOINT ORACLE once.",
        "- Every budget trial uses one globally fitted policy per quality target; prompt-specific LOO models are used only for separability evaluation.",
        "- Mixed-target trials sample unique prompt-plus-step sessions before assigning one target per session.",
        "",
        "## INFERRED",
        "",
        "- A systems allocation problem is justified only when measured heterogeneity survives noise filtering and simple separable policies retain a material oracle gap.",
        "",
        "## UNKNOWN",
        "",
        "- Results generalize only to the registered 12 prompts, Wan2.2 configuration, representations, and budgets tested here.",
        "- The registered first 12 prompts omit the two occlusion-labeled entries that occur later in the prompt JSON.",
        "",
        "## INVALIDATED PREVIOUS RESULTS",
        "",
        "- Legacy `resolved_at_n5` labels from Stage A used only two seeds and a flat prompt-seed bootstrap.",
        "- Legacy iso-storage confidence intervals, interaction crossing percentages, separability estimates, and all old budget-simulator session counts are not used.",
        "",
        ]
    )
    if summary.get("figure_paths"):
        lines.extend(["## Figures", ""])
        lines.extend(f"- `{path}`" for path in summary["figure_paths"])
        lines.append("")
    lines.extend(["## Judgment", "", judgment])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_config_path = Path(args.analysis_config)
    analysis_config = load_analysis_config(analysis_config_path)
    validate_analysis_config(analysis_config, args)
    raw_path = input_dir / "frontier_raw.csv"
    rows = _read_csv(raw_path)
    validation = validate_frontier(rows, args.expected_prompts, args.expected_seeds)
    validation["preregistered_config"] = validate_preregistered_config(
        input_dir / "preregistered_config.json",
        validation,
        args.expected_seeds,
    )
    noise_path = Path(args.noise_floor_csv) if args.noise_floor_csv else output_dir / "noise_floor_results.csv"
    if args.expected_seeds == 5 and not noise_path.exists() and not args.allow_missing_noise_floor:
        raise FileNotFoundError(
            f"Final n=5 analysis requires the preregistered noise-floor file: {noise_path}"
        )
    noise_rows = _read_csv(noise_path) if noise_path.exists() else []
    noise_prereg_path = noise_path.parent / "noise_floor_preregistered_config.json"
    if noise_rows:
        validation["noise_floor"] = validate_noise_floor_rows(
            noise_rows,
            validation["prompt_set_sha256"][0],
        )
        validation["noise_floor"]["preregistered_config"] = (
            validate_noise_preregistered_config(
                noise_prereg_path,
                validation["prompt_set_sha256"][0],
            )
        )
    noise_floor = load_noise_floor(noise_path if noise_path.exists() else None)
    if noise_path.exists() and noise_path.resolve() != (output_dir / "noise_floor_results.csv").resolve():
        shutil.copyfile(noise_path, output_dir / "noise_floor_results.csv")
        shutil.copyfile(
            noise_prereg_path,
            output_dir / "noise_floor_preregistered_config.json",
        )

    frontier_summary = aggregate_frontier(rows, args.bootstrap_samples, args.bootstrap_seed)
    progress_effects = build_progress_dependence(rows, args.bootstrap_samples, args.bootstrap_seed)
    checkpoint_rows = build_checkpoint_sizes(rows)
    iso_rows = build_iso_storage(
        rows,
        args.iso_storage_tolerance,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.expected_seeds,
    )
    minimum_rows, cells = build_minimum_safe(
        rows,
        QUALITY_TARGETS,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.expected_seeds,
        noise_floor,
    )
    policy_details, _ = build_policy_predictions(
        cells,
        QUALITY_TARGETS,
        args.complexity_bins,
        args.simple_policy_max_training_violation,
    )
    budget_predictions, budget_policy_definitions = build_global_policy_predictions(
        cells,
        QUALITY_TARGETS,
        args.complexity_bins,
        args.simple_policy_max_training_violation,
    )
    separability_rows = summarize_separability(policy_details)
    crossing_rows = build_interaction_crossings(
        minimum_rows,
        cells,
        args.expected_seeds,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    budget_rows = run_budget_simulation(
        cells,
        budget_predictions,
        args.budget_session_counts,
        args.budget_fractions,
        args.budget_trials,
        args.bootstrap_seed,
    )
    summary, judgment = build_summary_and_judgment(
        validation,
        frontier_summary,
        progress_effects,
        iso_rows,
        minimum_rows,
        crossing_rows,
        separability_rows,
        budget_rows,
        noise_rows,
        analysis_config,
    )
    summary["budget_policy_definitions"] = budget_policy_definitions
    summary["budget_policy_scope"] = (
        "one global model per quality target; privileged final-video complexity upper bound, not deployable"
    )
    summary["figure_paths"] = maybe_make_figures(
        output_dir,
        frontier_summary,
        minimum_rows,
        iso_rows,
        budget_rows,
    )

    suffix = f"n{args.expected_seeds}"
    shutil.copyfile(raw_path, output_dir / f"frontier_raw_{suffix}.csv")
    _write_csv(output_dir / f"checkpoint_sizes_{suffix}.csv", checkpoint_rows)
    _write_csv(output_dir / f"iso_storage_frontier_{suffix}.csv", iso_rows)
    _write_csv(output_dir / f"minimum_safe_representation_{suffix}.csv", minimum_rows)
    _write_csv(output_dir / f"interaction_crossings_{suffix}.csv", crossing_rows)
    _write_csv(output_dir / f"separability_results_{suffix}.csv", separability_rows)
    _write_csv(output_dir / f"separability_predictions_{suffix}.csv", policy_details)
    _write_csv(output_dir / "budget_simulation_fixed.csv", budget_rows)
    audit_source = Path(__file__).with_name("analysis_audit.md")
    if audit_source.exists():
        shutil.copyfile(audit_source, output_dir / "analysis_audit.md")
    shutil.copyfile(analysis_config_path, output_dir / analysis_config_path.name)
    (output_dir / "final_frontier_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_final_report(output_dir / "video_state_protection_final_killtest.md", summary, judgment)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(args)
    print(json.dumps(_json_safe({"judgment": summary["judgment"], "validation": summary["validation"]}), indent=2))


if __name__ == "__main__":
    main()
