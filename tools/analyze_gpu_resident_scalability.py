#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GPU-resident paused-state scalability experiments and summarize break-even conditions."
    )
    parser.add_argument("--batch-csv", type=Path, required=True, help="Path to exp2_gpu_resident_scalability.batch.csv")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a derived summary JSON. Defaults to <batch>.analysis.json",
    )
    parser.add_argument(
        "--latency-degradation-ratio",
        type=float,
        default=1.05,
        help="Treat gpu_resident as degraded if mean foreground latency exceeds the baseline by this ratio.",
    )
    parser.add_argument(
        "--low-pressure-ratio",
        type=float,
        default=0.10,
        help="GPU footprint ratio threshold below which a condition is labeled low pressure.",
    )
    parser.add_argument(
        "--high-pressure-ratio",
        type=float,
        default=0.40,
        help="GPU footprint ratio threshold at or above which a condition is labeled high pressure.",
    )
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["policy"],
                int(row["paused_request_count"]),
                float(row["pause_duration_sec"]),
                row["foreground_level"],
            )
        ].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (item[0][3], item[0][2], item[0][1], item[0][0])):
        policy, paused_count, pause_sec, foreground_level = key
        def avg(name: str) -> float | None:
            vals = [_to_float(row[name]) for row in group_rows]
            vals = [val for val in vals if val is not None]
            return mean(vals) if vals else None

        aggregates.append(
            {
                "policy": policy,
                "paused_request_count": paused_count,
                "pause_duration_sec": pause_sec,
                "foreground_level": foreground_level,
                "runs": len(group_rows),
                "foreground_admission_rate": avg("foreground_admission_rate"),
                "foreground_batch_utilization": avg("foreground_batch_utilization"),
                "foreground_mean_latency_sec": avg("foreground_mean_latency_sec"),
                "foreground_mean_admission_latency_sec": avg("foreground_mean_admission_latency_sec"),
                "paused_mean_resume_or_restart_latency_sec": avg("paused_mean_resume_or_restart_latency_sec"),
                "paused_mean_output_ssim": avg("paused_mean_output_ssim"),
                "paused_mean_checkpoint_bytes": avg("paused_mean_checkpoint_bytes"),
                "post_pause_gpu_footprint_bytes": avg("post_pause_gpu_footprint_bytes"),
                "post_pause_gpu_footprint_ratio": avg("post_pause_gpu_footprint_ratio"),
                "paused_mean_recomputed_steps": avg("paused_mean_recomputed_steps"),
                "paused_mean_wasted_compute_sec": avg("paused_mean_wasted_compute_sec"),
            }
        )
    return aggregates


def _pressure_region(footprint_ratio: float | None, low_threshold: float, high_threshold: float) -> str:
    if footprint_ratio is None:
        return "unknown"
    if footprint_ratio < low_threshold:
        return "low"
    if footprint_ratio < high_threshold:
        return "moderate"
    return "high"


def _find_gpu_resident_break_even(
    aggregates: list[dict[str, Any]],
    *,
    latency_degradation_ratio: float,
    low_pressure_ratio: float,
    high_pressure_ratio: float,
) -> list[dict[str, Any]]:
    by_context: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["policy"] == "gpu_resident":
            by_context[(row["pause_duration_sec"], row["foreground_level"])].append(row)

    breakpoints: list[dict[str, Any]] = []
    for context, rows in sorted(by_context.items()):
        rows = sorted(rows, key=lambda row: row["paused_request_count"])
        baseline = rows[0] if rows else None
        if baseline is None:
            continue
        baseline_latency = baseline["foreground_mean_latency_sec"]
        for row in rows:
            degraded = False
            reasons: list[str] = []
            if row["foreground_admission_rate"] is not None and row["foreground_admission_rate"] < 1.0:
                degraded = True
                reasons.append("admission<1.0")
            if row["foreground_batch_utilization"] is not None and row["foreground_batch_utilization"] < 1.0:
                degraded = True
                reasons.append("batch_utilization<1.0")
            if (
                baseline_latency is not None
                and row["foreground_mean_latency_sec"] is not None
                and row["foreground_mean_latency_sec"] > baseline_latency * latency_degradation_ratio
            ):
                degraded = True
                reasons.append(f"latency>{latency_degradation_ratio:.2f}x baseline")

            if degraded:
                breakpoints.append(
                    {
                        "pause_duration_sec": context[0],
                        "foreground_level": context[1],
                        "paused_request_count": row["paused_request_count"],
                        "post_pause_gpu_footprint_bytes": row["post_pause_gpu_footprint_bytes"],
                        "post_pause_gpu_footprint_ratio": row["post_pause_gpu_footprint_ratio"],
                        "pressure_region": _pressure_region(
                            row["post_pause_gpu_footprint_ratio"], low_pressure_ratio, high_pressure_ratio
                        ),
                        "foreground_admission_rate": row["foreground_admission_rate"],
                        "foreground_batch_utilization": row["foreground_batch_utilization"],
                        "foreground_mean_latency_sec": row["foreground_mean_latency_sec"],
                        "reasons": reasons,
                    }
                )
                break
    return breakpoints


def _compare_policies(
    aggregates: list[dict[str, Any]],
    *,
    low_pressure_ratio: float,
    high_pressure_ratio: float,
) -> list[dict[str, Any]]:
    comparisons: dict[tuple[int, float, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregates:
        comparisons[
            (
                row["paused_request_count"],
                row["pause_duration_sec"],
                row["foreground_level"],
            )
        ][row["policy"]] = row

    rows: list[dict[str, Any]] = []
    for key, policies in sorted(comparisons.items()):
        paused_count, pause_sec, foreground_level = key
        gpu_row = policies.get("gpu_resident")
        if gpu_row is None:
            continue
        cpu_lossless = policies.get("cpu_lossless")
        cpu_value_aware = policies.get("cpu_value_aware")
        restart = policies.get("restart")
        rows.append(
            {
                "paused_request_count": paused_count,
                "pause_duration_sec": pause_sec,
                "foreground_level": foreground_level,
                "pressure_region": _pressure_region(
                    gpu_row["post_pause_gpu_footprint_ratio"], low_pressure_ratio, high_pressure_ratio
                ),
                "gpu_resident_footprint_ratio": gpu_row["post_pause_gpu_footprint_ratio"],
                "gpu_resident_admission_rate": gpu_row["foreground_admission_rate"],
                "gpu_resident_batch_utilization": gpu_row["foreground_batch_utilization"],
                "gpu_resident_latency_sec": gpu_row["foreground_mean_latency_sec"],
                "cpu_lossless_latency_sec": None if cpu_lossless is None else cpu_lossless["foreground_mean_latency_sec"],
                "cpu_value_aware_latency_sec": None if cpu_value_aware is None else cpu_value_aware["foreground_mean_latency_sec"],
                "restart_paused_wasted_compute_sec": None if restart is None else restart["paused_mean_wasted_compute_sec"],
                "cpu_value_aware_checkpoint_bytes": None
                if cpu_value_aware is None
                else cpu_value_aware["paused_mean_checkpoint_bytes"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.batch_csv)
    aggregates = _aggregate(rows)
    break_even = _find_gpu_resident_break_even(
        aggregates,
        latency_degradation_ratio=args.latency_degradation_ratio,
        low_pressure_ratio=args.low_pressure_ratio,
        high_pressure_ratio=args.high_pressure_ratio,
    )
    policy_comparison = _compare_policies(
        aggregates,
        low_pressure_ratio=args.low_pressure_ratio,
        high_pressure_ratio=args.high_pressure_ratio,
    )

    summary = {
        "batch_csv": str(args.batch_csv),
        "num_batch_rows": len(rows),
        "num_aggregate_rows": len(aggregates),
        "latency_degradation_ratio": args.latency_degradation_ratio,
        "low_pressure_ratio": args.low_pressure_ratio,
        "high_pressure_ratio": args.high_pressure_ratio,
        "gpu_resident_break_even": break_even,
        "gpu_resident_break_even_found": bool(break_even),
        "policy_comparison": policy_comparison,
    }

    output_json = args.output_json or args.batch_csv.with_suffix(".analysis.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
