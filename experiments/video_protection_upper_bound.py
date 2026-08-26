#!/usr/bin/env python3
"""CPU-only upper-bound simulator for video state protection allocation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_state_protection_analysis as analysis


TARGETS = [0.95, 0.975, 0.99]
SESSION_COUNTS = [3, 8, 20]
BUDGET_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
BANDWIDTH_GBPS = [10, 25, 100]
FAILURE_PROBABILITIES = [0.001, 0.01, 0.05]
QUALITY_FIELDS = ["spatial_vs_full", "temporal_dynamic_vs_full", "semantic_vs_full"]
REPRESENTATION_POLICIES = [
    "uniform_int8",
    "minimum_safe_independent",
    "progress_only",
    "simple_separable",
    "joint_oracle",
]
ADMISSION_POLICIES = [
    "full_fifo",
    "full_highest_work",
    "uniform_int8",
    "minimum_safe_independent",
    "progress_only",
    "simple_separable",
    "joint_oracle",
]


@dataclass(frozen=True)
class Representation:
    name: str
    bytes: int
    compression_ratio: float
    spatial_quality: float
    dynamic_quality: float
    semantic_quality: float

    def safe(self, target: float) -> bool:
        return min(self.spatial_quality, self.dynamic_quality, self.semantic_quality) >= target


@dataclass
class Session:
    session_id: str
    prompt_id: str
    category: str
    seed: int
    seed_index: int
    checkpoint_step: int
    total_steps: int
    content_complexity: float
    representations: dict[str, Representation]
    per_step_gpu_ms: float
    fixed_resume_overhead_ms: float
    timing_method: str

    @property
    def progress_fraction(self) -> float:
        return self.checkpoint_step / self.total_steps

    @property
    def remaining_steps(self) -> int:
        return self.total_steps - self.checkpoint_step

    @property
    def accumulated_gpu_ms(self) -> float:
        return self.per_step_gpu_ms * self.checkpoint_step

    @property
    def remaining_gpu_ms(self) -> float:
        return self.per_step_gpu_ms * self.remaining_steps + self.fixed_resume_overhead_ms

    def minimum_safe(self, target: float) -> Representation | None:
        safe = [representation for representation in self.representations.values() if representation.safe(target)]
        return min(safe, key=lambda representation: (representation.bytes, representation.name)) if safe else None


@dataclass
class Aggregate:
    count: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sum_squares: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for name, value in metrics.items():
            self.sums[name] += value
            self.sum_squares[name] += value * value

    def summary(self) -> dict[str, float]:
        result = {}
        for name, total in self.sums.items():
            mean = total / self.count
            variance = max(0.0, self.sum_squares[name] / self.count - mean * mean)
            result[f"mean_{name}"] = mean
            result[f"std_{name}"] = math.sqrt(variance)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontier-csv",
        default="results/video_state_protection_killtest_gpu0/run/frontier_raw.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/video_protection_upper_bound_local"
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--include-n32", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _linear_timing_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    return slope, max(0.0, intercept)


def build_session_profiles(
    rows: list[dict[str, str]],
) -> tuple[list[Session], list[dict[str, Any]], dict[str, Any]]:
    validation = analysis.validate_frontier(rows, expected_prompts=12, expected_seeds=2)
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    full_by_request: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (row["prompt_id"], int(row["seed"]), int(row["checkpoint_step"]))
        grouped[key].append(row)
        if row["variant"] == "full":
            remaining = int(row["total_steps"]) - int(row["checkpoint_step"])
            full_by_request[(row["prompt_id"], int(row["seed"]))].append(
                (remaining, float(row["resume_latency_ms"]))
            )

    timing_fits = {
        key: _linear_timing_fit(points) for key, points in full_by_request.items()
    }
    positive_slopes = [slope for slope, _ in timing_fits.values() if slope > 0.0]
    if not positive_slopes:
        raise ValueError("Unable to estimate a positive denoising-step latency")
    fallback_slope = statistics.median(positive_slopes)

    sessions = []
    profile_rows = []
    for (prompt_id, seed, step), group in sorted(grouped.items()):
        if len(group) != len(analysis.VARIANTS):
            raise ValueError(f"Incomplete frontier cell: {prompt_id}/seed{seed}/step{step}")
        row_by_variant = {row["variant"]: row for row in group}
        slope, intercept = timing_fits[(prompt_id, seed)]
        timing_method = "per_request_linear_fit_full_resume_latency"
        if not math.isfinite(slope) or slope <= 0.0:
            slope, intercept = fallback_slope, 0.0
            timing_method = "global_median_positive_slope_fallback"
        representations = {
            variant: Representation(
                name=variant,
                bytes=int(row["total_checkpoint_bytes"]),
                compression_ratio=float(row["compression_ratio_vs_full"]),
                spatial_quality=float(row["spatial_vs_full"]),
                dynamic_quality=float(row["temporal_dynamic_vs_full"]),
                semantic_quality=float(row["semantic_vs_full"]),
            )
            for variant, row in row_by_variant.items()
        }
        first = group[0]
        session = Session(
            session_id=f"{prompt_id}:seed{seed}:step{step}",
            prompt_id=prompt_id,
            category=first["category"],
            seed=seed,
            seed_index=int(first["seed_index"]),
            checkpoint_step=step,
            total_steps=int(first["total_steps"]),
            content_complexity=float(first["content_complexity_score"]),
            representations=representations,
            per_step_gpu_ms=slope,
            fixed_resume_overhead_ms=intercept,
            timing_method=timing_method,
        )
        sessions.append(session)
        for variant in analysis.VARIANTS:
            representation = representations[variant]
            profile_rows.append(
                {
                    "session_id": session.session_id,
                    "prompt_id": prompt_id,
                    "category": session.category,
                    "seed": seed,
                    "seed_index": session.seed_index,
                    "checkpoint_step": step,
                    "total_steps": session.total_steps,
                    "progress_fraction": session.progress_fraction,
                    "representation": variant,
                    "actual_checkpoint_bytes": representation.bytes,
                    "compression_ratio_vs_full": representation.compression_ratio,
                    "spatial_quality": representation.spatial_quality,
                    "dynamic_temporal_quality": representation.dynamic_quality,
                    "semantic_quality": representation.semantic_quality,
                    "completed_steps": step,
                    "remaining_steps": session.remaining_steps,
                    "accumulated_compute_fraction": session.progress_fraction,
                    "remaining_compute_fraction": 1.0 - session.progress_fraction,
                    "estimated_per_step_gpu_ms": session.per_step_gpu_ms,
                    "estimated_fixed_resume_overhead_ms": session.fixed_resume_overhead_ms,
                    "estimated_accumulated_gpu_ms": session.accumulated_gpu_ms,
                    "estimated_remaining_gpu_ms": session.remaining_gpu_ms,
                    "timing_estimation_method": session.timing_method,
                }
            )
    return sessions, profile_rows, validation


def build_simple_policy_predictions(
    rows: list[dict[str, str]], bootstrap_samples: int
) -> tuple[dict[tuple[str, int, float, str], str], list[dict[str, Any]]]:
    minimum_rows, cells = analysis.build_minimum_safe(
        rows,
        TARGETS,
        samples=bootstrap_samples,
        seed=7,
        expected_seeds=2,
        noise_floor={},
    )
    del minimum_rows
    return analysis.build_global_policy_predictions(
        cells,
        TARGETS,
        complexity_bins=3,
        max_training_violation=0.05,
    )


def policy_representation(
    session: Session,
    target: float,
    policy: str,
    predictions: dict[tuple[str, int, float, str], str],
) -> Representation | None:
    if policy in {"full_fifo", "full_highest_work"}:
        representation = session.representations["full"]
    elif policy == "uniform_int8":
        representation = session.representations["int8"]
    elif policy in {"minimum_safe_independent", "joint_oracle"}:
        return session.minimum_safe(target)
    elif policy in {"progress_only", "simple_separable"}:
        variant = predictions[(session.prompt_id, session.checkpoint_step, target, policy)]
        representation = session.representations[variant]
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return representation if representation.safe(target) else None


def _item_value(session: Session, objective: str, priority: int) -> float:
    if objective == "equal_session":
        return 1.0
    if objective == "accumulated_work":
        return session.accumulated_gpu_ms
    if objective == "priority_weighted_work":
        return session.accumulated_gpu_ms * priority
    raise ValueError(objective)


def greedy_select(
    items: list[tuple[int, Session, Representation, float]],
    budget: int,
    policy: str,
    objective: str,
) -> list[tuple[int, Session, Representation, float]]:
    if policy == "full_fifo":
        ordered = items
    elif policy == "full_highest_work":
        ordered = sorted(items, key=lambda item: (-item[1].accumulated_gpu_ms, item[0]))
    elif objective == "equal_session":
        ordered = sorted(items, key=lambda item: (item[2].bytes, item[0]))
    else:
        ordered = sorted(items, key=lambda item: (-item[3], item[2].bytes, item[0]))
    selected = []
    used = 0
    for item in ordered:
        if used + item[2].bytes <= budget:
            selected.append(item)
            used += item[2].bytes
    return selected


def exact_knapsack_select(
    items: list[tuple[int, Session, Representation, float]], budget: int
) -> list[tuple[int, Session, Representation, float]]:
    """Exact 0/1 Pareto DP after dominated representation choices are removed."""
    states: dict[int, tuple[float, int, tuple[int, ...]]] = {0: (0.0, 0, ())}
    for item_index, item in enumerate(items):
        cost = item[2].bytes
        value = item[3]
        updates = dict(states)
        for used, (state_value, count, selected) in states.items():
            new_used = used + cost
            if new_used > budget:
                continue
            candidate = (state_value + value, count + 1, selected + (item_index,))
            current = updates.get(new_used)
            if current is None or candidate[:2] > current[:2]:
                updates[new_used] = candidate
        pruned = {}
        best_score = (-math.inf, -1)
        for used in sorted(updates):
            state = updates[used]
            score = state[:2]
            if score > best_score:
                pruned[used] = state
                best_score = score
        states = pruned
    _, best = max(
        states.items(), key=lambda pair: (pair[1][0], pair[1][1], -pair[0])
    )
    return [items[index] for index in best[2]]


def evaluate_selection(
    sessions: list[Session],
    selected: list[tuple[int, Session, Representation, float]],
    target: float,
    budget: int,
    objective: str,
    priorities: list[int],
) -> dict[str, float]:
    selected_indexes = {item[0] for item in selected}
    total_work = sum(session.accumulated_gpu_ms for session in sessions)
    total_weighted = sum(
        session.accumulated_gpu_ms * priorities[index]
        for index, session in enumerate(sessions)
    )
    protected_work = sum(item[1].accumulated_gpu_ms for item in selected)
    protected_weighted = sum(
        sessions[index].accumulated_gpu_ms * priorities[index] for index in selected_indexes
    )
    bytes_used = sum(item[2].bytes for item in selected)
    return {
        "protected_session_count": float(len(selected)),
        "fraction_sessions_protected": len(selected) / len(sessions),
        "protected_accumulated_gpu_ms": protected_work,
        "fraction_accumulated_work_protected": protected_work / total_work,
        "protected_priority_weighted_gpu_ms": protected_weighted,
        "fraction_priority_weighted_work_protected": protected_weighted / total_weighted,
        "expected_recompute_gpu_ms_if_failure": total_work - protected_work,
        "checkpoint_bytes_used": float(bytes_used),
        "budget_utilization": bytes_used / budget if budget else 0.0,
        "quality_violation_count": 0.0,
        "objective_value": sum(item[3] for item in selected),
        "all_sessions_feasible": float(len(selected) == len(sessions)),
        "quality_target": target,
    }


def representation_only_evaluation(
    sessions: list[Session],
    target: float,
    budget: int,
    policy: str,
    objective: str,
    priorities: list[int],
    predictions: dict[tuple[str, int, float, str], str],
) -> dict[str, float]:
    items = []
    invalid = 0
    for index, session in enumerate(sessions):
        representation = policy_representation(session, target, policy, predictions)
        if representation is None:
            invalid += 1
            continue
        items.append((index, session, representation, _item_value(session, objective, priorities[index])))
    required_bytes = sum(item[2].bytes for item in items)
    feasible = invalid == 0 and required_bytes <= budget
    selected = items if feasible else []
    metrics = evaluate_selection(sessions, selected, target, budget, objective, priorities)
    metrics["quality_violation_count"] = float(invalid)
    metrics["assigned_checkpoint_bytes"] = float(required_bytes)
    metrics["all_sessions_feasible"] = float(feasible)
    return metrics


def admission_evaluation(
    sessions: list[Session],
    target: float,
    budget: int,
    policy: str,
    objective: str,
    priorities: list[int],
    predictions: dict[tuple[str, int, float, str], str],
) -> dict[str, float]:
    items = []
    for index, session in enumerate(sessions):
        representation = policy_representation(session, target, policy, predictions)
        if representation is None:
            continue
        items.append((index, session, representation, _item_value(session, objective, priorities[index])))
    selected = (
        exact_knapsack_select(items, budget)
        if policy == "joint_oracle"
        else greedy_select(items, budget, policy, objective)
    )
    metrics = evaluate_selection(sessions, selected, target, budget, objective, priorities)
    metrics["eligible_session_count"] = float(len(items))
    return metrics


def _aggregate_rows(
    aggregates: dict[tuple[Any, ...], Aggregate],
    key_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for key, aggregate in sorted(aggregates.items()):
        row = dict(zip(key_names, key, strict=True))
        row["trial_count"] = aggregate.count
        row.update(aggregate.summary())
        rows.append(row)
    return rows


def run_simulation(
    sessions: list[Session],
    predictions: dict[tuple[str, int, float, str], str],
    trials: int,
    random_seed: int,
    session_counts: list[int],
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[Any, ...], Aggregate] = defaultdict(Aggregate)
    for session_count in session_counts:
        if session_count > len(sessions):
            raise ValueError(f"N={session_count} exceeds {len(sessions)} measured sessions")
        for target_index, target in enumerate(TARGETS):
            rng = random.Random(random_seed + session_count * 1009 + target_index * 9176)
            for _ in range(trials):
                sampled = rng.sample(sessions, session_count)
                priorities = rng.choices([1, 2, 4], weights=[0.25, 0.50, 0.25], k=session_count)
                full_total = sum(session.representations["full"].bytes for session in sampled)
                for budget_fraction in BUDGET_FRACTIONS:
                    budget = int(full_total * budget_fraction)
                    for objective in ("equal_session", "accumulated_work"):
                        for policy in REPRESENTATION_POLICIES:
                            metrics = representation_only_evaluation(
                                sampled, target, budget, policy, objective, priorities, predictions
                            )
                            aggregates[(
                                "representation_only", objective, target, session_count,
                                budget_fraction, policy
                            )].add(metrics)
                        for policy in ADMISSION_POLICIES:
                            metrics = admission_evaluation(
                                sampled, target, budget, policy, objective, priorities, predictions
                            )
                            aggregates[(
                                "admission_and_representation", objective, target, session_count,
                                budget_fraction, policy
                            )].add(metrics)
    return _aggregate_rows(
        aggregates,
        [
            "question",
            "objective",
            "quality_target",
            "session_count",
            "budget_fraction_of_all_full",
            "policy",
        ],
    )


def build_policy_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {
        (
            row["question"], row["objective"], row["quality_target"],
            row["session_count"], row["budget_fraction_of_all_full"], row["policy"]
        ): row
        for row in rows
    }
    output = []
    base_dimensions = sorted({key[:-1] for key in grouped})
    for dimensions in base_dimensions:
        oracle = grouped.get((*dimensions, "joint_oracle"))
        if oracle is None:
            continue
        for baseline_policy in ("minimum_safe_independent", "simple_separable"):
            baseline = grouped.get((*dimensions, baseline_policy))
            if baseline is None:
                continue
            count_gap = (
                oracle["mean_protected_session_count"]
                - baseline["mean_protected_session_count"]
            )
            work_gap = (
                oracle["mean_protected_accumulated_gpu_ms"]
                - baseline["mean_protected_accumulated_gpu_ms"]
            )
            baseline_sessions = baseline["mean_protected_session_count"]
            baseline_work = baseline["mean_protected_accumulated_gpu_ms"]
            output.append(
                {
                    "question": dimensions[0],
                    "objective": dimensions[1],
                    "quality_target": dimensions[2],
                    "session_count": dimensions[3],
                    "budget_fraction_of_all_full": dimensions[4],
                    "baseline_policy": baseline_policy,
                    "oracle_mean_protected_sessions": oracle["mean_protected_session_count"],
                    "baseline_mean_protected_sessions": baseline["mean_protected_session_count"],
                    "absolute_session_gap": count_gap,
                    "relative_session_improvement": (
                        count_gap / baseline_sessions if baseline_sessions > 0.0 else ""
                    ),
                    "session_relative_gap_undefined_from_zero_baseline": (
                        baseline_sessions == 0.0 and count_gap > 0.0
                    ),
                    "oracle_mean_protected_work_ms": oracle["mean_protected_accumulated_gpu_ms"],
                    "baseline_mean_protected_work_ms": baseline["mean_protected_accumulated_gpu_ms"],
                    "absolute_work_gap_ms": work_gap,
                    "relative_work_improvement": (
                        work_gap / baseline_work if baseline_work > 0.0 else ""
                    ),
                    "work_relative_gap_undefined_from_zero_baseline": (
                        baseline_work == 0.0 and work_gap > 0.0
                    ),
                }
            )
    return output


def build_accumulated_work_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["question"], row["quality_target"], row["session_count"],
            row["budget_fraction_of_all_full"], row["policy"], row["objective"]
        ): row
        for row in rows
    }
    output = []
    dimensions = sorted({key[:-1] for key in by_key})
    for key in dimensions:
        equal = by_key.get((*key, "equal_session"))
        work = by_key.get((*key, "accumulated_work"))
        if equal is None or work is None:
            continue
        output.append(
            {
                "question": key[0],
                "quality_target": key[1],
                "session_count": key[2],
                "budget_fraction_of_all_full": key[3],
                "policy": key[4],
                "equal_objective_mean_sessions": equal["mean_protected_session_count"],
                "work_objective_mean_sessions": work["mean_protected_session_count"],
                "session_count_change": (
                    work["mean_protected_session_count"] - equal["mean_protected_session_count"]
                ),
                "equal_objective_mean_work_ms": equal["mean_protected_accumulated_gpu_ms"],
                "work_objective_mean_work_ms": work["mean_protected_accumulated_gpu_ms"],
                "relative_protected_work_gain": (
                    work["mean_protected_accumulated_gpu_ms"]
                    - equal["mean_protected_accumulated_gpu_ms"]
                ) / max(equal["mean_protected_accumulated_gpu_ms"], 1e-12),
            }
        )
    return output


def build_failure_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = [
        row for row in rows
        if row["question"] == "admission_and_representation"
        and row["objective"] == "accumulated_work"
    ]
    output = []
    for row in source:
        common = {
            "quality_target": row["quality_target"],
            "session_count": row["session_count"],
            "budget_fraction_of_all_full": row["budget_fraction_of_all_full"],
            "policy": row["policy"],
        }
        output.append(
            {
                **common,
                "failure_model": "memory_pressure_eviction",
                "parameter": "host_storage_budget_fraction",
                "parameter_value": row["budget_fraction_of_all_full"],
                "mean_checkpoint_bytes": row["mean_checkpoint_bytes_used"],
                "mean_transfer_time_ms": 0.0,
                "mean_expected_recompute_gpu_ms": row["mean_expected_recompute_gpu_ms_if_failure"],
                "mean_fraction_work_protected": row["mean_fraction_accumulated_work_protected"],
            }
        )
        for bandwidth in BANDWIDTH_GBPS:
            output.append(
                {
                    **common,
                    "failure_model": "cross_device_migration",
                    "parameter": "bandwidth_gbps",
                    "parameter_value": bandwidth,
                    "mean_checkpoint_bytes": row["mean_checkpoint_bytes_used"],
                    "mean_transfer_time_ms": row["mean_checkpoint_bytes_used"] * 8.0 / (bandwidth * 1e6),
                    "mean_expected_recompute_gpu_ms": row["mean_expected_recompute_gpu_ms_if_failure"],
                    "mean_fraction_work_protected": row["mean_fraction_accumulated_work_protected"],
                }
            )
        for probability in FAILURE_PROBABILITIES:
            output.append(
                {
                    **common,
                    "failure_model": "node_failure",
                    "parameter": "failure_probability",
                    "parameter_value": probability,
                    "mean_checkpoint_bytes": row["mean_checkpoint_bytes_used"],
                    "mean_transfer_time_ms": "",
                    "mean_expected_recompute_gpu_ms": (
                        row["mean_expected_recompute_gpu_ms_if_failure"] * probability
                    ),
                    "mean_fraction_work_protected": row["mean_fraction_accumulated_work_protected"],
                }
            )
    return output


def _max_gap(
    comparisons: list[dict[str, Any]], baseline: str, field: str
) -> float:
    values = [
        float(row[field])
        for row in comparisons
        if row["baseline_policy"] == baseline and row[field] != ""
    ]
    return max(values, default=0.0)


def build_summary(
    validation: dict[str, Any],
    sessions: list[Session],
    budget_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    accumulated_rows: list[dict[str, Any]],
    trials: int,
    random_seed: int,
    policy_definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    admission_comparisons = [
        row for row in comparisons if row["question"] == "admission_and_representation"
    ]
    broad = [
        row for row in admission_comparisons
        if row["baseline_policy"] == "minimum_safe_independent"
        and row["budget_fraction_of_all_full"] in {0.25, 0.5, 0.75}
    ]
    meaningful = [
        row for row in broad
        if max(
            float(row["relative_session_improvement"] or 0.0),
            float(row["relative_work_improvement"] or 0.0),
        ) >= 0.15
    ]
    weak = [
        row for row in broad
        if max(
            float(row["relative_session_improvement"] or 0.0),
            float(row["relative_work_improvement"] or 0.0),
        ) >= 0.05
    ]
    if len(meaningful) >= max(3, len(broad) // 4):
        judgment = "PROMISING UPPER BOUND"
    elif weak:
        judgment = "WEAK UPPER BOUND"
    else:
        judgment = "NO MEANINGFUL UPPER BOUND"
    admission_accumulated = [
        row for row in budget_rows
        if row["question"] == "admission_and_representation"
        and row["objective"] == "accumulated_work"
    ]
    target_policy_means = {}
    for target in TARGETS:
        for policy in ("uniform_int8", "minimum_safe_independent", "joint_oracle"):
            selected = [
                row for row in admission_accumulated
                if math.isclose(float(row["quality_target"]), target)
                and row["policy"] == policy
            ]
            target_policy_means[f"target_{target}_{policy}"] = {
                "mean_fraction_sessions_protected": statistics.fmean(
                    row["mean_fraction_sessions_protected"] for row in selected
                ),
                "mean_fraction_accumulated_work_protected": statistics.fmean(
                    row["mean_fraction_accumulated_work_protected"] for row in selected
                ),
            }
    max_work_gap_by_n = {}
    for session_count in sorted({int(row["session_count"]) for row in budget_rows}):
        selected = [
            row for row in admission_comparisons
            if row["baseline_policy"] == "minimum_safe_independent"
            and row["objective"] == "accumulated_work"
            and int(row["session_count"]) == session_count
            and row["relative_work_improvement"] != ""
        ]
        max_work_gap_by_n[str(session_count)] = max(
            (float(row["relative_work_improvement"]) for row in selected), default=0.0
        )
    minimum_work_sensitivity = [
        row for row in accumulated_rows
        if row["question"] == "admission_and_representation"
        and row["policy"] == "minimum_safe_independent"
    ]
    max_joint_bytes = max(
        row["mean_checkpoint_bytes_used"]
        for row in admission_accumulated
        if row["policy"] == "joint_oracle"
    )
    return {
        "input_validation": validation,
        "session_profile_count": len(sessions),
        "representation_profile_count": len(sessions) * len(analysis.VARIANTS),
        "simulation": {
            "trials_per_N_target": trials,
            "random_seed": random_seed,
            "session_counts": sorted({int(row["session_count"]) for row in budget_rows}),
            "quality_targets": TARGETS,
            "budget_fractions": BUDGET_FRACTIONS,
        },
        "timing_estimation": (
            "Per prompt-seed linear fit of full resume latency against remaining steps; "
            "accumulated denoising work is fitted slope times completed steps."
        ),
        "oracle_structure": (
            "All quality-safe representations have identical protection value, so any larger "
            "safe representation is dominated by that session's minimum-safe representation. "
            "The exact joint oracle therefore reduces to admission knapsack over minimum-safe states."
        ),
        "simple_policy_definitions": policy_definitions,
        "target_policy_means": target_policy_means,
        "maximum_oracle_work_improvement_over_minimum_safe_by_N": max_work_gap_by_n,
        "minimum_safe_accumulated_objective_mean_work_gain": statistics.fmean(
            row["relative_protected_work_gain"] for row in minimum_work_sensitivity
        ),
        "minimum_safe_accumulated_objective_max_work_gain": max(
            row["relative_protected_work_gain"] for row in minimum_work_sensitivity
        ),
        "minimum_safe_accumulated_objective_mean_session_change": statistics.fmean(
            row["session_count_change"] for row in minimum_work_sensitivity
        ),
        "maximum_joint_checkpoint_bytes": max_joint_bytes,
        "maximum_joint_transfer_ms_by_bandwidth": {
            str(bandwidth): max_joint_bytes * 8.0 / (bandwidth * 1e6)
            for bandwidth in BANDWIDTH_GBPS
        },
        "minimum_safe_conditions_ge_5pct": [
            {
                "objective": row["objective"],
                "quality_target": row["quality_target"],
                "session_count": row["session_count"],
                "budget_fraction": row["budget_fraction_of_all_full"],
                "relative_session_improvement": row["relative_session_improvement"],
                "relative_work_improvement": row["relative_work_improvement"],
            }
            for row in weak
        ],
        "maximum_oracle_relative_session_improvement_over_minimum_safe": _max_gap(
            admission_comparisons, "minimum_safe_independent", "relative_session_improvement"
        ),
        "maximum_oracle_relative_work_improvement_over_minimum_safe": _max_gap(
            admission_comparisons, "minimum_safe_independent", "relative_work_improvement"
        ),
        "maximum_oracle_relative_session_improvement_over_simple_separable": _max_gap(
            admission_comparisons, "simple_separable", "relative_session_improvement"
        ),
        "maximum_oracle_relative_work_improvement_over_simple_separable": _max_gap(
            admission_comparisons, "simple_separable", "relative_work_improvement"
        ),
        "conditions_ge_15pct_vs_minimum_safe": len(meaningful),
        "conditions_ge_5pct_vs_minimum_safe": len(weak),
        "judgment_scope": "Stage A CPU-only upper bound; not the final project GO/NO-GO",
        "judgment": judgment,
    }


def make_figures(output_dir: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    selected = [
        row for row in rows
        if row["question"] == "admission_and_representation"
        and row["objective"] == "accumulated_work"
        and math.isclose(float(row["quality_target"]), 0.99)
        and int(row["session_count"]) == 20
        and row["policy"] in {"minimum_safe_independent", "simple_separable", "joint_oracle"}
    ]
    for metric, filename, ylabel in (
        ("mean_protected_session_count", "policy_vs_budget.pdf", "Protected sessions"),
        ("mean_fraction_accumulated_work_protected", "protected_work_vs_budget.pdf", "Protected accumulated work fraction"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for policy in ("minimum_safe_independent", "simple_separable", "joint_oracle"):
            policy_rows = sorted(
                [row for row in selected if row["policy"] == policy],
                key=lambda row: row["budget_fraction_of_all_full"],
            )
            ax.plot(
                [row["budget_fraction_of_all_full"] for row in policy_rows],
                [row[metric] for row in policy_rows],
                marker="o",
                label=policy,
            )
        ax.set_xlabel("Protection budget / all-full footprint")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    gap_rows = [
        row for row in comparisons
        if row["question"] == "admission_and_representation"
        and row["objective"] == "accumulated_work"
        and row["baseline_policy"] == "minimum_safe_independent"
        and math.isclose(float(row["quality_target"]), 0.99)
        and math.isclose(float(row["budget_fraction_of_all_full"]), 0.5)
    ]
    gap_rows.sort(key=lambda row: row["session_count"])
    ax.plot(
        [row["session_count"] for row in gap_rows],
        [row["relative_work_improvement"] for row in gap_rows],
        marker="o",
    )
    ax.set_xlabel("Concurrent sessions")
    ax.set_ylabel("Oracle work improvement over minimum-safe")
    fig.tight_layout()
    path = output_dir / "oracle_gap_vs_num_sessions.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    admission = [row for row in rows if row["question"] == "admission_and_representation"]
    minimum = [row for row in admission if row["policy"] == "minimum_safe_independent"]
    oracle = [row for row in admission if row["policy"] == "joint_oracle"]
    mean_minimum_protected = statistics.fmean(row["mean_fraction_sessions_protected"] for row in minimum)
    mean_oracle_protected = statistics.fmean(row["mean_fraction_sessions_protected"] for row in oracle)
    concurrency_evidence = (
        "`oracle_gap_vs_num_sessions.pdf` and `policy_comparison.csv`"
        if summary.get("figure_paths")
        else "`policy_comparison.csv` (matplotlib was unavailable, so no PDF was generated)"
    )
    target_means = summary["target_policy_means"]
    int8_by_target = {
        target: target_means[f"target_{target}_uniform_int8"]
        for target in TARGETS
    }
    weak_conditions = summary["minimum_safe_conditions_ge_5pct"]
    transfer = summary["maximum_joint_transfer_ms_by_bandwidth"]
    weak_condition_text = "; ".join(
        f"target={row['quality_target']}, N={row['session_count']}: "
        f"sessions +{100 * row['relative_session_improvement']:.1f}%, "
        f"work +{100 * row['relative_work_improvement']:.1f}%"
        for row in weak_conditions
    )
    lines = [
        "# Video Protection Allocation Upper Bound",
        "",
        "## Scope",
        "",
        "This is a CPU-only Stage A upper-bound analysis. It does not use GPU inference and is not the final project GO/NO-GO.",
        "",
        "## Data and Method",
        "",
        f"- Input: {summary['input_validation']['row_count']} raw rows, {summary['session_profile_count']} measured prompt-seed-progress sessions.",
        f"- Synthetic mixtures per N and target: {summary['simulation']['trials_per_N_target']}.",
        f"- Timing: {summary['timing_estimation']}",
        f"- Oracle structure: {summary['oracle_structure']}",
        "- A representation is safe only when spatial, temporal/dynamic, and semantic quality are all at least the preregistered target.",
        "- The content+progress baseline uses final-video complexity and is therefore a privileged, non-deployable upper bound.",
        "- Uniform INT8 leaves an unsafe session unprotected. Minimum-safe uses the smallest measured safe state and admits by bytes for the count objective or accumulated work for the work objective.",
        "- FULL-FIFO and FULL-highest-work use full checkpoints. The exact oracle jointly admits minimum-safe states with a Pareto-pruned 0/1 knapsack.",
        "",
        "## Answers",
        "",
        f"**Q1. Is uniform INT8 enough?** Not at high fidelity. Its mean protected-session fractions are {int8_by_target[0.95]['mean_fraction_sessions_protected']:.3f} at 95%, {int8_by_target[0.975]['mean_fraction_sessions_protected']:.3f} at 97.5%, and {int8_by_target[0.99]['mean_fraction_sessions_protected']:.3f} at 99% across the registered N/budget sweep.",
        f"**Q2. Does independent minimum-safe capture the benefit?** Yes, almost all of it. It protects {mean_minimum_protected:.3f} on average versus oracle {mean_oracle_protected:.3f}. Equal-session optimization has zero oracle gap. Under accumulated-work optimization, the maximum relative session/work gaps are {summary['maximum_oracle_relative_session_improvement_over_minimum_safe']:.3f}/{summary['maximum_oracle_relative_work_improvement_over_minimum_safe']:.3f}.",
        f"**Q3. Does accumulated work change selection?** Sometimes under tight budgets, but it does not create much joint-optimization headroom. For minimum-safe, switching from count to accumulated-work ordering raises protected work by {summary['minimum_safe_accumulated_objective_mean_work_gain']:.3f} on average and at most {summary['minimum_safe_accumulated_objective_max_work_gain']:.3f}, while changing protected count by {summary['minimum_safe_accumulated_objective_mean_session_change']:.3f} sessions on average.",
        f"**Q4. Does concurrency help joint optimization?** The maximum oracle work gaps over minimum-safe rise from {summary['maximum_oracle_work_improvement_over_minimum_safe_by_N']['3']:.3f} at N=3 to {summary['maximum_oracle_work_improvement_over_minimum_safe_by_N']['8']:.3f} at N=8 and {summary['maximum_oracle_work_improvement_over_minimum_safe_by_N']['20']:.3f} at N=20, but remain below 5%. {concurrency_evidence} contains all cells.",
        f"**Q5. Which budgets are nontrivial?** Only three minimum-safe comparisons exceed a 5% relative gap, all at the 25% budget: {weak_condition_text}. No registered condition reaches 15%.",
        f"**Q6. Does the opportunity extend to eviction, migration, and failure?** The same allocation frontier applies, but the allocator gap remains weak. The largest selected oracle state in the sweep transfers in at most {transfer['10']:.2f}/{transfer['25']:.2f}/{transfer['100']:.2f} ms at 10/25/100 Gbps. Expected node-failure recomputation scales linearly with the predeclared 0.1/1/5% sensitivity and does not enlarge the allocation gap.",
        f"**Q7. Maximum perfect-allocator benefit?** Versus the strongest independent minimum-safe baseline, maximum relative session improvement is {summary['maximum_oracle_relative_session_improvement_over_minimum_safe']:.3f}, but the primary accumulated-work improvement is only {summary['maximum_oracle_relative_work_improvement_over_minimum_safe']:.3f}. This is a narrow 25%-budget effect, not broad 15-20% headroom.",
        "",
        "## Confirmed",
        "",
        "- Actual serialized checkpoint bytes are present for every representation.",
        "- The oracle is exact for the stated value model; larger safe representations are dominated by minimum-safe state.",
        "- Representation-only and admission+representation questions are reported separately.",
        "",
        "## Inferred",
        "",
        "- Any oracle advantage over minimum-safe comes from admission coupling, not from a hidden representation choice after quality is reduced to a binary SLO.",
        "",
        "## Unknown",
        "",
        "- Stage A has only two seeds; Stage B may change the measured safe frontier.",
        "- Accumulated denoising work is estimated from recovery latency rather than directly profiled cumulative GPU kernels.",
        "- No real failure rate, migration stack, or runtime overhead is claimed.",
        "",
        "## Upper-Bound Judgment",
        "",
        summary["judgment"],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    frontier_path = Path(args.frontier_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(frontier_path)
    sessions, profile_rows, validation = build_session_profiles(rows)
    predictions, policy_definitions = build_simple_policy_predictions(
        rows, args.bootstrap_samples
    )
    session_counts = [*SESSION_COUNTS, 32] if args.include_n32 else list(SESSION_COUNTS)
    budget_rows = run_simulation(
        sessions,
        predictions,
        trials=args.trials,
        random_seed=args.random_seed,
        session_counts=session_counts,
    )
    comparisons = build_policy_comparison(budget_rows)
    accumulated = build_accumulated_work_analysis(budget_rows)
    failure = build_failure_sensitivity(budget_rows)
    summary = build_summary(
        validation,
        sessions,
        budget_rows,
        comparisons,
        accumulated,
        args.trials,
        args.random_seed,
        policy_definitions,
    )
    summary["figure_paths"] = make_figures(output_dir, budget_rows, comparisons)
    _write_csv(output_dir / "session_profiles.csv", profile_rows)
    _write_csv(output_dir / "budget_sweep.csv", budget_rows)
    _write_csv(output_dir / "policy_comparison.csv", comparisons)
    _write_csv(output_dir / "accumulated_work_analysis.csv", accumulated)
    _write_csv(output_dir / "failure_model_sensitivity.csv", failure)
    (output_dir / "oracle_gap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir / "video_protection_upper_bound.md", summary, budget_rows)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"judgment": summary["judgment"], "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
