#!/usr/bin/env python3
"""CPU-only kill test for deadline-bounded video-session evacuation.

Measured checkpoint costs come from the Wan2.2 Stage-B frontier at
480x832x33. Costs for other shapes are explicitly marked model-based
extrapolations using Wan latent geometry; they are not GPU measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUALITY_FIELDS = ("spatial_vs_full", "temporal_dynamic_vs_full", "semantic_vs_full")


@dataclass(frozen=True)
class Shape:
    name: str
    height: int
    width: int
    frames: int


@dataclass(frozen=True)
class CostProfile:
    variant: str
    payload_bytes: float
    metadata_bytes: float
    d2h_ms: float
    encode_ms: float


@dataclass(frozen=True)
class Session:
    session_id: int
    prompt_id: str
    checkpoint_step: int
    shape: Shape
    shape_scale: float
    value_ms: float
    costs_ms: dict[str, float]
    safe_variants: frozenset[str]

    def minimum_safe(self) -> str:
        if not self.safe_variants:
            raise ValueError(f"Session {self.session_id} has no safe representation")
        return min(self.safe_variants, key=lambda variant: (self.costs_ms[variant], variant))


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    protected_sessions: float
    protected_work_ms: float
    evacuation_time_ms: float


def latent_elements(shape: Shape, channels: int = 16) -> int:
    if shape.height <= 0 or shape.width <= 0 or shape.frames <= 0:
        raise ValueError(f"Invalid shape: {shape}")
    latent_t = math.ceil(shape.frames / 4)
    latent_h = math.ceil(shape.height / 8)
    latent_w = math.ceil(shape.width / 8)
    return channels * latent_t * latent_h * latent_w


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("status") != "preregistered_before_evacuation_triage_killtest":
        raise ValueError("Evacuation configuration is not preregistered")
    if config["primary_quality_target"] not in config["quality_targets"]:
        raise ValueError("Primary target must be part of quality_targets")
    return config


def build_profiles(rows: list[dict[str, str]]) -> dict[str, CostProfile]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    required = {"full", "fp16", "int8", "spatial_down2", "temporal_down2", "low_rank_25"}
    if set(grouped) != required:
        raise ValueError(f"Unexpected representation set: {sorted(grouped)}")
    profiles = {}
    for variant, group in grouped.items():
        profiles[variant] = CostProfile(
            variant=variant,
            payload_bytes=statistics.mean(float(row["encoded_payload_bytes"]) for row in group),
            metadata_bytes=statistics.mean(float(row["metadata_bytes"]) for row in group),
            d2h_ms=statistics.mean(float(row["checkpoint_cpu_copy_ms"]) for row in group),
            encode_ms=statistics.mean(float(row["encode_prepare_latency_ms"]) for row in group),
        )
    return profiles


def build_quality_frontier(
    rows: list[dict[str, str]], targets: list[float]
) -> dict[tuple[str, int, float], frozenset[str]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_id"], int(row["checkpoint_step"]), row["variant"])].append(row)
    frontier: dict[tuple[str, int, float], frozenset[str]] = {}
    prompt_steps = sorted({(key[0], key[1]) for key in grouped})
    for prompt_id, step in prompt_steps:
        variants = {variant for p, s, variant in grouped if p == prompt_id and s == step}
        for target in targets:
            safe = set()
            for variant in variants:
                measurements = grouped[(prompt_id, step, variant)]
                if len(measurements) != 5:
                    raise ValueError(f"Expected five seeds for {prompt_id}/{step}/{variant}")
                if all(min(float(row[field]) for field in QUALITY_FIELDS) >= target for row in measurements):
                    safe.add(variant)
            if "full" not in safe:
                raise ValueError(f"Full recovery is not safe for {prompt_id}/{step}/{target}")
            frontier[(prompt_id, step, target)] = frozenset(safe)
    return frontier


def evacuation_cost_ms(
    profile: CostProfile,
    shape_scale: float,
    bandwidth_gbps: float,
    fixed_latency_ms: float,
) -> float:
    if bandwidth_gbps <= 0:
        raise ValueError("Bandwidth must be positive")
    payload = profile.payload_bytes * shape_scale + profile.metadata_bytes
    transfer_ms = payload * 8.0 / (bandwidth_gbps * 1e6)
    return (profile.d2h_ms + profile.encode_ms) * shape_scale + transfer_ms + fixed_latency_ms


def build_sessions(
    rng: random.Random,
    count: int,
    shapes: list[Shape],
    reference_shape: Shape,
    profiles: dict[str, CostProfile],
    frontier: dict[tuple[str, int, float], frozenset[str]],
    target: float,
    bandwidth_gbps: float,
    fixed_latency_ms: float,
    base_step_ms: float,
) -> list[Session]:
    cells = sorted(key for key in frontier if key[2] == target)
    reference_elements = latent_elements(reference_shape)
    sessions = []
    for session_id in range(count):
        prompt_id, step, _ = rng.choice(cells)
        shape = rng.choice(shapes)
        scale = latent_elements(shape) / reference_elements
        costs = {
            variant: evacuation_cost_ms(profile, scale, bandwidth_gbps, fixed_latency_ms)
            for variant, profile in profiles.items()
        }
        sessions.append(
            Session(
                session_id=session_id,
                prompt_id=prompt_id,
                checkpoint_step=step,
                shape=shape,
                shape_scale=scale,
                value_ms=base_step_ms * step * scale,
                costs_ms=costs,
                safe_variants=frontier[(prompt_id, step, target)],
            )
        )
    return sessions


def _pack_ordered(
    policy: str,
    candidates: list[tuple[Session, str]],
    deadline_ms: float,
) -> PolicyResult:
    used = 0.0
    protected = 0
    work = 0.0
    for session, variant in candidates:
        cost = session.costs_ms[variant]
        if used + cost <= deadline_ms:
            used += cost
            protected += 1
            work += session.value_ms
    return PolicyResult(policy, float(protected), work, used)


def evaluate_policies(sessions: list[Session], deadline_ms: float) -> list[PolicyResult]:
    fifo = list(sessions)

    def safe_int8(session: Session) -> bool:
        return "int8" in session.safe_variants

    int8 = [(session, "int8") for session in fifo if safe_int8(session)]
    minimum = [(session, session.minimum_safe()) for session in fifo]
    results = [
        _pack_ordered("full_fifo", [(session, "full") for session in fifo], deadline_ms),
        _pack_ordered("uniform_int8_fifo", int8, deadline_ms),
        _pack_ordered(
            "uniform_int8_progress",
            sorted(int8, key=lambda item: (-item[0].checkpoint_step, item[0].session_id)),
            deadline_ms,
        ),
        _pack_ordered(
            "uniform_int8_value_density",
            sorted(int8, key=lambda item: (-item[0].value_ms / item[0].costs_ms[item[1]], item[0].session_id)),
            deadline_ms,
        ),
        _pack_ordered("minimum_safe_fifo", minimum, deadline_ms),
        _pack_ordered(
            "minimum_safe_value_density",
            sorted(minimum, key=lambda item: (-item[0].value_ms / item[0].costs_ms[item[1]], item[0].session_id)),
            deadline_ms,
        ),
    ]

    # Fractional knapsack is an optimistic upper bound, not a deployable policy.
    remaining = deadline_ms
    fractional_sessions = 0.0
    fractional_work = 0.0
    used = 0.0
    for session, variant in sorted(
        minimum,
        key=lambda item: (-item[0].value_ms / item[0].costs_ms[item[1]], item[0].session_id),
    ):
        if remaining <= 0:
            break
        cost = session.costs_ms[variant]
        fraction = min(1.0, remaining / cost)
        fractional_sessions += fraction
        fractional_work += session.value_ms * fraction
        used += cost * fraction
        remaining -= cost * fraction
    results.append(PolicyResult("fractional_oracle_upper_bound", fractional_sessions, fractional_work, used))
    return results


def _relative_gap(upper: float, baseline: float) -> float:
    return 0.0 if upper <= 0 else max(0.0, (upper - baseline) / upper)


def judge(summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    primary = float(config["primary_quality_target"])
    realistic = [
        row
        for row in summary_rows
        if float(row["quality_target"]) == primary
        and int(row["deadline_sec"]) in config["realistic_deadlines_sec"]
        and float(row["bandwidth_gbps"]) in config["realistic_bandwidth_gbps"]
        and row["policy"] == "minimum_safe_value_density"
    ]
    if not realistic:
        raise ValueError("No realistic primary cells were evaluated")
    gate = config["judgment"]
    trivial = [float(row["mean_protected_fraction"]) >= gate["trivial_all_saved_fraction"] for row in realistic]
    trivial_fraction = sum(trivial) / len(trivial)
    strong = [row for row in realistic if float(row["relative_work_gap_to_upper_bound"]) >= gate["strong_gap_minimum_relative_saved_work"]]
    deadlines = {int(row["deadline_sec"]) for row in strong}
    counts = {int(row["session_count"]) for row in strong}
    if trivial_fraction >= gate["no_go_if_realistic_primary_cells_trivial_fraction_gte"]:
        judgment = "NO-GO"
    elif (
        len(strong) >= gate["minimum_realistic_cells_with_strong_gap"]
        and len(deadlines) >= gate["minimum_distinct_realistic_deadlines_with_strong_gap"]
        and len(counts) >= gate["minimum_distinct_session_counts_with_strong_gap"]
    ):
        judgment = "STRONG GO REQUIRES EXACT-ORACLE FOLLOW-UP"
    elif any(
        float(row["relative_work_gap_to_upper_bound"]) >= gate["conditional_gap_minimum_relative_saved_work"]
        for row in realistic
    ):
        judgment = "CONDITIONAL GO"
    else:
        judgment = "NO-GO"
    evidence = {
        "realistic_primary_cell_count": len(realistic),
        "trivial_all_saved_cell_fraction": trivial_fraction,
        "strong_upper_bound_gap_cell_count": len(strong),
        "strong_gap_deadlines": sorted(deadlines),
        "strong_gap_session_counts": sorted(counts),
        "maximum_realistic_relative_work_gap_to_upper_bound": max(
            float(row["relative_work_gap_to_upper_bound"]) for row in realistic
        ),
    }
    return judgment, evidence


def run(config: dict[str, Any], input_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read_csv(input_path)
    profiles = build_profiles(rows)
    frontier = build_quality_frontier(rows, [float(value) for value in config["quality_targets"]])
    shapes = [Shape(**item) for item in config["shape_catalog"]]
    reference_shape = Shape(name="reference", **config["reference_shape"])
    trial_rows: list[dict[str, Any]] = []
    aggregates: dict[tuple[Any, ...], list[PolicyResult]] = defaultdict(list)
    for target in config["quality_targets"]:
        for count in config["session_counts"]:
            for deadline in config["deadlines_sec"]:
                for bandwidth in config["bandwidth_gbps"]:
                    for trial in range(config["trials"]):
                        seed = config["random_seed"] + int(target * 1000) * 10_000_000 + count * 10_000 + deadline * 100 + int(bandwidth) * 10 + trial
                        sessions = build_sessions(
                            random.Random(seed), count, shapes, reference_shape, profiles, frontier,
                            float(target), float(bandwidth), float(config["fixed_remote_write_latency_ms"]),
                            float(config["base_denoising_step_ms"]),
                        )
                        for result in evaluate_policies(sessions, deadline * 1000.0):
                            key = (target, count, deadline, bandwidth, result.policy)
                            aggregates[key].append(result)
                            trial_rows.append({
                                "quality_target": target,
                                "session_count": count,
                                "deadline_sec": deadline,
                                "bandwidth_gbps": bandwidth,
                                "trial": trial,
                                "policy": result.policy,
                                "protected_sessions": result.protected_sessions,
                                "protected_fraction": result.protected_sessions / count,
                                "protected_work_ms": result.protected_work_ms,
                                "evacuation_time_ms": result.evacuation_time_ms,
                            })
    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(aggregates.items(), key=lambda item: tuple(map(str, item[0]))):
        target, count, deadline, bandwidth, policy = key
        upper = aggregates[(target, count, deadline, bandwidth, "fractional_oracle_upper_bound")]
        mean_upper_work = statistics.mean(item.protected_work_ms for item in upper)
        mean_work = statistics.mean(item.protected_work_ms for item in values)
        summary_rows.append({
            "quality_target": target,
            "session_count": count,
            "deadline_sec": deadline,
            "bandwidth_gbps": bandwidth,
            "policy": policy,
            "mean_protected_sessions": statistics.mean(item.protected_sessions for item in values),
            "mean_protected_fraction": statistics.mean(item.protected_sessions for item in values) / count,
            "mean_protected_work_ms": mean_work,
            "mean_evacuation_time_ms": statistics.mean(item.evacuation_time_ms for item in values),
            "relative_work_gap_to_upper_bound": _relative_gap(mean_upper_work, mean_work),
        })
    judgment, evidence = judge(summary_rows, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "evacuation_triage_trials.csv", trial_rows)
    _write_csv(output_dir / "evacuation_triage_summary.csv", summary_rows)
    summary = {
        "judgment": judgment,
        "evidence": evidence,
        "configuration": config,
        "input_frontier": str(input_path),
        "input_rows": len(rows),
        "measured_reference_shape_only": True,
        "non_reference_shape_costs_are_extrapolated": True,
    }
    (output_dir / "evacuation_triage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Video Evacuation Triage Kill Test",
        "",
        f"- Judgment: **{judgment}**",
        f"- Input rows: `{len(rows)}`",
        "- Costs are measured only at `480x832x33`; all other shapes are latent-element-linear extrapolations.",
        "- The fractional oracle is an optimistic upper bound, not a deployable integral policy.",
        "",
        "## Evidence",
        "",
    ] + [f"- {key}: `{value}`" for key, value in evidence.items()] + [
        "",
        "## Interpretation",
        "",
        "- A small upper-bound gap is sufficient to reject deadline-aware triage as a system mechanism.",
        "- A large gap is not final evidence; it requires measured multi-shape costs and an exact integral oracle.",
        "- Results at 5 seconds or 1 Gbps are stress cases and cannot alone justify GO.",
    ]
    (output_dir / "video_evacuation_triage_killtest.md").write_text("\n".join(report) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_evacuation_triage_config.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_evacuation_triage_killtest"))
    parser.add_argument("--trials", type=int, help="Testing override; omit for preregistered execution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.trials is not None:
        config["trials"] = args.trials
        config["testing_override"] = True
    input_path = args.input or Path(config["input_frontier"])
    summary = run(config, input_path, args.output_dir)
    print(json.dumps({"judgment": summary["judgment"], "evidence": summary["evidence"]}, indent=2))


if __name__ == "__main__":
    main()
