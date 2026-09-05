#!/usr/bin/env python3
"""Cheap public-trace screening for spot recovery and GPU backfilling.

This script intentionally has no dependency on vLLM or a GPU.  It analyzes
only documented public traces and keeps interval-censoring and trace coverage
visible in every decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import tarfile
import tempfile
import urllib.request
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SPOT_COMMIT = "221da1ab9bdfe7966da296265d7694d52dc0c5a6"
ALIBABA_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
SPOT_BASE = f"https://raw.githubusercontent.com/skypilot-org/spot-traces/{SPOT_COMMIT}"
SPOT_FILES = {
    "us-east-1c_v100_1.json": "8405520cdc1f8327997709972c7fb7df63e28d320416223a65f94bc7115bb8c9",
    "us-east-1f_v100_1.json": "05dd39a548ed2714d23c19188b5385051547cd3650d3327c2bee90be92893dbc",
    "us-west-2b_v100_1.json": "6e9b7f7fec8be121ce94cc0e0e9d9a14dd57a5dcf6911388f49641f47a2bb064",
    "us-west-2c_v100_1.json": "68ddceb31103227f62ec2ef3e8459e47810eb42b573a7b8e6d501885e073cd63",
}
ALIBABA_SOURCES = {
    "pai_machine_spec.tar.gz": {
        "sha256": "cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d",
        "url": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz",
    },
    "pai_task_table.tar.gz": {
        "sha256": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
        "url": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz",
    },
    "pai_instance_table.tar.gz": {
        "sha256": "1bf1e423a7ce3f8d086699801c362fd56a7182abdb234139e5ebbed97995ca06",
        "url": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz",
    },
}
REQUEST_DURATIONS = (120.0, 230.0, 300.0, 600.0)
COLD_OVERHEADS = (0.0, 2.0, 5.0, 10.0, 30.0)
HOLE_THRESHOLDS = (1.0, 2.0, 5.7, 10.0, 30.0, 60.0)
QUANTUM_SECONDS = 5.7
CHECKPOINT_SECONDS = 0.004

TASK_HEADER = (
    "job_name", "task_name", "inst_num", "status", "start_time",
    "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type",
)
INSTANCE_HEADER = (
    "job_name", "task_name", "inst_name", "worker_name", "inst_id",
    "status", "start_time", "end_time", "machine",
)
MACHINE_HEADER = ("machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) == expected_sha256:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 << 20)
    actual = _sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {destination.name}: {actual}")
    partial.replace(destination)


def download_sources(source_dir: Path, include_alibaba: bool) -> None:
    spot_dir = source_dir / "spot"
    for name, digest in SPOT_FILES.items():
        url = f"{SPOT_BASE}/preemption/1-node/aws-04-22-2023/{name}"
        _download(url, spot_dir / name, digest)
    if include_alibaba:
        ali_dir = source_dir / "alibaba_v2020"
        for name, metadata in ALIBABA_SOURCES.items():
            _download(metadata["url"], ali_dir / name, metadata["sha256"])


def _floor_quantum(value: float, quantum: float = QUANTUM_SECONDS) -> float:
    return math.floor(max(0.0, value) / quantum + 1e-12) * quantum


def _integral_floor_quantum(lo: float, hi: float, quantum: float) -> float:
    """Integral of floor(x/q)*q over [lo, hi]."""
    if hi <= lo:
        return 0.0
    cursor = max(0.0, lo)
    end = max(0.0, hi)
    total = 0.0
    while cursor < end:
        k = math.floor(cursor / quantum + 1e-12)
        boundary = min(end, (k + 1) * quantum)
        if boundary <= cursor + 1e-12:
            boundary = min(end, cursor + quantum)
        total += (boundary - cursor) * k * quantum
        cursor = boundary
    return total


@dataclass(frozen=True)
class Exposure:
    probability_lower: float
    probability_midpoint: float
    probability_upper: float
    restart_lost_lower: float
    restart_lost_midpoint: float
    restart_lost_upper: float
    recoverable_lower: float
    recoverable_midpoint: float
    recoverable_upper: float


def first_interruption_exposure(values: Sequence[int], gap: float, start: int, duration: float) -> Exposure:
    """Bounds for a first 1->0 transition after a running start sample."""
    if values[start] != 1:
        raise ValueError("Request start must be a running observation")
    transition: tuple[float, float] | None = None
    max_index = min(len(values) - 1, int(math.ceil(start + duration / gap)) + 1)
    for index in range(start + 1, max_index + 1):
        if values[index - 1] == 1 and values[index] == 0:
            transition = ((index - 1 - start) * gap, (index - start) * gap)
            break
    if transition is None or transition[0] >= duration:
        return Exposure(*(0.0,) * 9)
    lo, hi = transition
    clipped_hi = min(hi, duration)
    if hi <= duration:
        p_lo = p_hi = 1.0
        lost_lo, lost_hi = lo, hi
        rec_lo, rec_hi = _floor_quantum(lo), _floor_quantum(hi)
    else:
        p_lo, p_hi = 0.0, 1.0
        lost_lo, lost_hi = 0.0, duration
        rec_lo, rec_hi = 0.0, _floor_quantum(duration)
    p_mid = (clipped_hi - lo) / (hi - lo)
    lost_mid = (clipped_hi * clipped_hi - lo * lo) / (2.0 * (hi - lo))
    rec_mid = _integral_floor_quantum(lo, clipped_hi, QUANTUM_SECONDS) / (hi - lo)
    return Exposure(p_lo, p_mid, p_hi, lost_lo, lost_mid, lost_hi, rec_lo, rec_mid, rec_hi)


def _mean_exposures(items: Sequence[Exposure]) -> Exposure:
    if not items:
        raise ValueError("No valid request start positions")
    return Exposure(*(statistics.fmean(getattr(x, field) for x in items) for field in Exposure.__dataclass_fields__))


def _availability_segments(values: Sequence[int], gap: float) -> tuple[list[float], list[tuple[float, float, int]]]:
    end = (len(values) - 1) * gap
    boundaries = [0.0]
    states = [int(values[0])]
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            boundaries.append((index - 0.5) * gap)
            states.append(int(values[index]))
    boundaries.append(end)
    segments = [(boundaries[i], boundaries[i + 1], states[i]) for i in range(len(states))]
    return boundaries, segments


def _segments_after(segments: Sequence[tuple[float, float, int]], start_time: float) -> Iterator[tuple[float, float, int]]:
    starts = [segment[0] for segment in segments]
    index = max(0, bisect_right(starts, start_time) - 1)
    for seg_start, seg_end, state in segments[index:]:
        yield max(start_time, seg_start), seg_end, state


def _simulate_continuation(
    segments: Sequence[tuple[float, float, int]],
    start_time: float,
    duration: float,
    cold_overhead: float,
    resume: bool,
) -> dict[str, float | int | bool]:
    progress = 0.0
    interruptions = 0
    wasted = 0.0
    seen_down = False
    for seg_start, seg_end, state in _segments_after(segments, start_time):
        if seg_end <= seg_start:
            continue
        if state == 0:
            seen_down = True
            continue
        budget = seg_end - seg_start
        if seen_down:
            if budget <= cold_overhead:
                continue
            budget -= cold_overhead
            seg_start += cold_overhead
            seen_down = False
        if not resume:
            if budget >= duration:
                return {"completed": True, "wall_seconds": seg_start + duration - start_time,
                        "interruptions": interruptions, "wasted_gpu_seconds": wasted}
            if seg_end < segments[-1][1]:
                wasted += budget
                interruptions += 1
                progress = 0.0
                seen_down = True
            continue

        # Each completed denoising quantum is durable after a 4 ms checkpoint.
        cursor = seg_start
        while budget > 0:
            remaining = duration - progress
            compute = min(QUANTUM_SECONDS, remaining)
            if budget < compute:
                wasted += budget
                budget = 0.0
                break
            budget -= compute
            cursor += compute
            if compute >= remaining - 1e-12:
                return {"completed": True, "wall_seconds": cursor - start_time,
                        "interruptions": interruptions, "wasted_gpu_seconds": wasted}
            if budget < CHECKPOINT_SECONDS:
                wasted += compute + budget
                budget = 0.0
                break
            budget -= CHECKPOINT_SECONDS
            cursor += CHECKPOINT_SECONDS
            progress += compute
        if seg_end < segments[-1][1]:
            interruptions += 1
            seen_down = True
    return {"completed": False, "wall_seconds": math.nan, "interruptions": interruptions,
            "wasted_gpu_seconds": wasted}


def _load_spot(path: Path) -> tuple[list[int], int]:
    payload = json.loads(path.read_text())
    if set(payload) - {"metadata", "data", "prices"}:
        raise ValueError(f"Unexpected keys in {path.name}: {sorted(payload)}")
    gap = int(payload["metadata"]["gap_seconds"])
    values = payload["data"]
    if gap <= 0 or len(values) < 2 or any(value not in (0, 1) for value in values):
        raise ValueError(f"Invalid binary preemption trace: {path}")
    return [int(value) for value in values], gap


def analyze_spot(source_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    pooled: dict[float, list[Exposure]] = defaultdict(list)
    continuation_pool: defaultdict[tuple[float, float, str], list[dict[str, float | int | bool]]] = defaultdict(list)
    trace_data: list[tuple[str, list[int], int, list[tuple[float, float, int]]]] = []

    for name, expected_hash in SPOT_FILES.items():
        path = source_dir / "spot" / name
        if not path.exists() or _sha256(path) != expected_hash:
            raise RuntimeError(f"Missing or invalid SkyPilot source: {path}")
        values, gap = _load_spot(path)
        zone = name.split("_v100", 1)[0]
        _, segments = _availability_segments(values, gap)
        trace_data.append((zone, values, gap, segments))
        transitions = sum(a == 1 and b == 0 for a, b in zip(values, values[1:]))
        provenance.append({
            "dataset": "skypilot_spot_traces", "provider": "AWS", "gpu_type": "V100",
            "zone": zone, "file": str(path), "source_url": f"{SPOT_BASE}/preemption/1-node/aws-04-22-2023/{name}",
            "commit": SPOT_COMMIT, "sha256": expected_hash, "records": len(values),
            "relative_start_seconds": 0, "relative_end_seconds": (len(values) - 1) * gap,
            "sampling_interval_seconds": gap, "implicit_missing_intervals": 0,
            "duplicate_timestamps": 0, "running_observations": sum(values),
            "unavailable_observations": len(values) - sum(values), "running_to_unavailable_transitions": transitions,
            "calendar_anchor": "directory label 2023-04-22; exact timestamp/timezone not published",
        })
        for duration in REQUEST_DURATIONS:
            latest_start = (len(values) - 1) * gap - duration
            exposures = [
                first_interruption_exposure(values, gap, index, duration)
                for index in range(len(values))
                if values[index] == 1 and index * gap <= latest_start
            ]
            pooled[duration].extend(exposures)
            avg = _mean_exposures(exposures)
            row = {"analysis": "first_interruption", "scope": zone, "request_duration_seconds": duration,
                   "valid_start_positions": len(exposures), **asdict(avg)}
            row["recoverable_fraction_lower"] = avg.recoverable_lower / duration
            row["recoverable_fraction_midpoint"] = avg.recoverable_midpoint / duration
            row["recoverable_fraction_upper"] = avg.recoverable_upper / duration
            checkpoints = math.floor(duration / QUANTUM_SECONDS)
            row["periodic_checkpoint_overhead_seconds"] = checkpoints * CHECKPOINT_SECONDS
            row["net_recoverable_midpoint_seconds"] = avg.recoverable_midpoint - checkpoints * CHECKPOINT_SECONDS
            rows.append(row)

    pooled_rows: list[dict[str, Any]] = []
    for duration in REQUEST_DURATIONS:
        avg = _mean_exposures(pooled[duration])
        row = {"analysis": "first_interruption", "scope": "all_zones_pooled",
               "request_duration_seconds": duration, "valid_start_positions": len(pooled[duration]), **asdict(avg)}
        row["recoverable_fraction_lower"] = avg.recoverable_lower / duration
        row["recoverable_fraction_midpoint"] = avg.recoverable_midpoint / duration
        row["recoverable_fraction_upper"] = avg.recoverable_upper / duration
        checkpoints = math.floor(duration / QUANTUM_SECONDS)
        row["periodic_checkpoint_overhead_seconds"] = checkpoints * CHECKPOINT_SECONDS
        row["net_recoverable_midpoint_seconds"] = avg.recoverable_midpoint - checkpoints * CHECKPOINT_SECONDS
        rows.append(row)
        pooled_rows.append(row)

    for zone, values, gap, segments in trace_data:
        end = (len(values) - 1) * gap
        for duration in REQUEST_DURATIONS:
            starts = [index * gap for index, value in enumerate(values) if value == 1 and index * gap + duration <= end]
            for cold in COLD_OVERHEADS:
                for policy, resume in (("restart_from_zero", False), ("exact_step_resume", True)):
                    outcomes = [_simulate_continuation(segments, start, duration, cold, resume) for start in starts]
                    continuation_pool[(duration, cold, policy)].extend(outcomes)
                    completed = [item for item in outcomes if item["completed"]]
                    rows.append({
                        "analysis": "continuation_midpoint", "scope": zone,
                        "request_duration_seconds": duration, "cold_resume_seconds": cold,
                        "policy": policy, "valid_start_positions": len(starts),
                        "completion_fraction_within_trace": len(completed) / len(starts) if starts else math.nan,
                        "mean_wall_seconds_completed": statistics.fmean(float(x["wall_seconds"]) for x in completed) if completed else math.nan,
                        "mean_interruptions": statistics.fmean(int(x["interruptions"]) for x in outcomes) if outcomes else math.nan,
                        "mean_wasted_gpu_seconds": statistics.fmean(float(x["wasted_gpu_seconds"]) for x in outcomes) if outcomes else math.nan,
                        "event_time_model": "transitions at sample-interval midpoint (explicit simulation assumption)",
                    })

    pooled_continuation: list[dict[str, Any]] = []
    for (duration, cold, policy), outcomes in sorted(continuation_pool.items()):
        completed = [item for item in outcomes if item["completed"]]
        row = {
            "analysis": "continuation_midpoint", "scope": "all_zones_pooled",
            "request_duration_seconds": duration, "cold_resume_seconds": cold,
            "policy": policy, "valid_start_positions": len(outcomes),
            "completion_fraction_within_trace": len(completed) / len(outcomes) if outcomes else math.nan,
            "mean_wall_seconds_completed": statistics.fmean(float(x["wall_seconds"]) for x in completed) if completed else math.nan,
            "mean_interruptions": statistics.fmean(int(x["interruptions"]) for x in outcomes) if outcomes else math.nan,
            "mean_wasted_gpu_seconds": statistics.fmean(float(x["wasted_gpu_seconds"]) for x in outcomes) if outcomes else math.nan,
            "event_time_model": "transitions at sample-interval midpoint (explicit simulation assumption)",
        }
        rows.append(row)
        pooled_continuation.append(row)

    primary = next(row for row in pooled_rows if row["request_duration_seconds"] == 230.0)
    strong_other = sum(row["recoverable_fraction_lower"] >= 0.02 for row in pooled_rows)
    if primary["recoverable_fraction_lower"] >= 0.05 and strong_other >= 3:
        decision = "STRONG OPPORTUNITY"
    elif all(row["recoverable_fraction_upper"] < 0.01 for row in pooled_rows):
        decision = "NO-GO"
    else:
        decision = "WEAK OPPORTUNITY"
    summary = {
        "decision": decision,
        "primary_duration_seconds": 230,
        "primary_all_zone_pooled": primary,
        "duration_rows_all_zone_pooled": pooled_rows,
        "continuation_rows_all_zone_pooled": pooled_continuation,
        "trace_semantics": "actual single-node instance preemption experiment, not launch-probe availability",
        "sampling_caveat": "transition time interval-censored at 32 seconds; midpoint assumes uniform event time",
        "checkpoint_model": "state saved after every 5.7-second denoising quantum at 0.004 seconds per save",
        "decision_rule_source": "DATA_FEASIBILITY.md, frozen before numerical analysis",
    }
    return rows, summary, provenance


def _tar_rows(path: Path, header: Sequence[str]) -> Iterator[dict[str, str]]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and not member.name.endswith(".header")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one table in {path}, found {[m.name for m in members]}")
        binary = archive.extractfile(members[0])
        if binary is None:
            raise RuntimeError(f"Cannot read {members[0].name}")
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        reader = csv.reader(text)
        first = next(reader, None)
        if first is None:
            return
        if tuple(first) == tuple(header):
            pass
        else:
            yield dict(zip(header, first, strict=True))
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise RuntimeError(f"Schema mismatch in {path}: {len(values)} columns")
            yield dict(zip(header, values, strict=True))


def _finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_machine_specs(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in _tar_rows(path, MACHINE_HEADER):
        capacity = _finite_float(row["cap_gpu"])
        if capacity is None or capacity <= 0 or abs(capacity - round(capacity)) > 1e-9:
            continue
        result[row["machine"]] = int(round(capacity))
    return result


def _load_task_gpu(path: Path) -> tuple[dict[tuple[str, str], float | None], Counter[str]]:
    result: dict[tuple[str, str], float | None] = {}
    stats: Counter[str] = Counter()
    for row in _tar_rows(path, TASK_HEADER):
        key = (row["job_name"], row["task_name"])
        gpu = _finite_float(row["plan_gpu"])
        if key in result and result[key] != gpu:
            result[key] = None
            stats["conflicting_duplicate_task_keys"] += 1
        else:
            result[key] = gpu
        stats["task_rows"] += 1
        stats["missing_plan_gpu"] += gpu is None
    return result, stats


def _build_event_db(
    instance_path: Path,
    task_gpu: dict[tuple[str, str], float | None],
    db_path: Path,
) -> Counter[str]:
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE events(machine TEXT, time REAL, delta_units INTEGER, delta_invalid INTEGER)")
    stats: Counter[str] = Counter()
    batch: list[tuple[str, float, int, int]] = []
    try:
        trace_start = math.inf
        trace_end = -math.inf
        for row in _tar_rows(instance_path, INSTANCE_HEADER):
            start = _finite_float(row["start_time"])
            end = _finite_float(row["end_time"])
            if start is not None:
                trace_start = min(trace_start, start)
                trace_end = max(trace_end, start)
            if end is not None:
                trace_start = min(trace_start, end)
                trace_end = max(trace_end, end)
        if not math.isfinite(trace_start) or not math.isfinite(trace_end):
            raise RuntimeError("Instance trace has no finite lifecycle timestamps")
        stats["trace_start_seconds"] = int(trace_start)
        stats["trace_end_seconds"] = int(trace_end)

        for row in _tar_rows(instance_path, INSTANCE_HEADER):
            stats["instance_rows"] += 1
            machine = row["machine"]
            start = _finite_float(row["start_time"])
            end = _finite_float(row["end_time"])
            gpu = task_gpu.get((row["job_name"], row["task_name"]))
            if not machine:
                stats["invalid_lifecycle_rows"] += 1
                continue
            if start is not None and end is not None and end == start:
                stats["zero_duration_rows"] += 1
                continue
            if start is not None and end is not None and end < start:
                stats["invalid_lifecycle_rows"] += 1
                continue
            lifecycle_uncertain = start is None or end is None
            if lifecycle_uncertain:
                stats["censored_lifecycle_rows"] += 1
                start = trace_start if start is None else start
                end = trace_end if end is None else end
                if end <= start:
                    continue
            if gpu is not None and gpu <= 0:
                stats["cpu_only_instances"] += 1
                continue
            units = 0
            invalid = 0
            if lifecycle_uncertain:
                invalid = 1
                stats["uncertain_lifecycle_gpu_instances"] += 1
            elif gpu is None:
                invalid = 1
                stats["unknown_gpu_instances"] += 1
            else:
                gpu_units = gpu / 100.0
                if gpu_units < 1.0 or abs(gpu_units - round(gpu_units)) > 1e-9:
                    invalid = 1
                    stats["fractional_gpu_instances"] += 1
                else:
                    units = int(round(gpu_units))
                    stats["integral_gpu_instances"] += 1
            batch.extend(((machine, start, units, invalid), (machine, end, -units, -invalid)))
            if len(batch) >= 200_000:
                connection.executemany("INSERT INTO events VALUES (?,?,?,?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO events VALUES (?,?,?,?)", batch)
        connection.execute("CREATE INDEX events_machine_time ON events(machine, time)")
        connection.commit()
    finally:
        connection.close()
    return stats


@dataclass
class HoleAnalysis:
    holes: list[dict[str, Any]]
    counters: dict[str, float]


def _extract_holes(db_path: Path, capacities: dict[str, int]) -> HoleAnalysis:
    connection = sqlite3.connect(db_path)
    holes: list[dict[str, Any]] = []
    counts: defaultdict[str, float] = defaultdict(float)
    try:
        machines = [row[0] for row in connection.execute("SELECT DISTINCT machine FROM events ORDER BY machine")]
        for machine in machines:
            capacity = capacities.get(machine)
            if capacity is None:
                counts["machines_missing_capacity"] += 1
                continue
            grouped: list[tuple[float, int, int]] = []
            for time_value, units, invalid in connection.execute(
                "SELECT time, SUM(delta_units), SUM(delta_invalid) FROM events WHERE machine=? GROUP BY time ORDER BY time",
                (machine,),
            ):
                grouped.append((float(time_value), int(units), int(invalid)))
            if len(grouped) < 2:
                counts["machines_insufficient_events"] += 1
                continue
            used = 0
            invalid_active = 0
            open_slots: dict[int, float] = {}
            counts["machines_analyzed"] += 1
            for index, (time_value, delta_units, delta_invalid) in enumerate(grouped[:-1]):
                used += delta_units
                invalid_active += delta_invalid
                next_time = grouped[index + 1][0]
                duration = next_time - time_value
                if duration <= 0:
                    raise RuntimeError("Non-positive grouped event interval")
                counts["total_machine_capacity_seconds"] += capacity * duration
                valid = invalid_active == 0 and 0 <= used <= capacity
                if not valid:
                    reason = "invalid_reservation" if invalid_active else "over_capacity"
                    counts[f"excluded_{reason}_capacity_seconds"] += capacity * duration
                    free = 0
                else:
                    free = capacity - used
                    counts["valid_machine_capacity_seconds"] += capacity * duration
                    counts["stranded_gpu_seconds"] += free * duration

                for slot in range(1, capacity + 1):
                    is_free = valid and free >= slot
                    if is_free and slot not in open_slots:
                        open_slots[slot] = time_value
                    elif not is_free and slot in open_slots:
                        start = open_slots.pop(slot)
                        if time_value > start:
                            holes.append({"machine": machine, "reservation_slot": slot,
                                          "start_time": start, "end_time": time_value,
                                          "duration_seconds": time_value - start})
            final_time = grouped[-1][0]
            for slot, start in open_slots.items():
                if final_time > start:
                    holes.append({"machine": machine, "reservation_slot": slot,
                                  "start_time": start, "end_time": final_time,
                                  "duration_seconds": final_time - start})
    finally:
        connection.close()
    return HoleAnalysis(holes=holes, counters=dict(counts))


def _feasibility_rows() -> list[dict[str, Any]]:
    return [
        {"dataset": "cluster-trace-gpu-v2026", "time_resolution": "hourly",
         "allocation_events": "hourly summaries", "placement_identity": "server/hour",
         "contiguous_free_windows": "no", "schedulable_capacity": "no",
         "suitable_5_7s": "no", "reason": "hourly buckets cannot establish second-scale continuity"},
        {"dataset": "cluster-trace-v2026-spot-gpu", "time_resolution": "seconds",
         "allocation_events": "job submit/duration only", "placement_identity": "none for jobs",
         "contiguous_free_windows": "no", "schedulable_capacity": "no",
         "suitable_5_7s": "no", "reason": "no job-to-node assignment or actual allocation/free events"},
        {"dataset": "cluster-trace-v2026-GenAI (GenTD26)", "time_resolution": "fine sampled metrics",
         "allocation_events": "no physical allocation/free events", "placement_identity": "container only",
         "contiguous_free_windows": "no", "schedulable_capacity": "no",
         "suitable_5_7s": "no", "reason": "low utilization is not schedulable free GPU capacity"},
        {"dataset": "cluster-trace-gpu-v2023", "time_resolution": "seconds",
         "allocation_events": "pod lifecycle", "placement_identity": "no observed node assignment",
         "contiguous_free_windows": "no without simulated placement", "schedulable_capacity": "no",
         "suitable_5_7s": "no", "reason": "placement simulation would invent the tested hole distribution"},
        {"dataset": "cluster-trace-gpu-v2020", "time_resolution": "seconds",
         "allocation_events": "instance launch/completion", "placement_identity": "machine",
         "contiguous_free_windows": "yes, restricted integral-reservation subset",
         "schedulable_capacity": "machine-level reservation capacity",
         "suitable_5_7s": "conditional", "reason": "no physical GPU ID; fractional/unknown intervals excluded"},
    ]


def analyze_backfill(source_dir: Path, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ali_dir = source_dir / "alibaba_v2020"
    provenance: list[dict[str, Any]] = []
    for name, metadata in ALIBABA_SOURCES.items():
        path = ali_dir / name
        if not path.exists() or _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Missing or invalid Alibaba source: {path}")
        provenance.append({"dataset": "alibaba_cluster_trace_gpu_v2020", "file": str(path),
                           "source_url": metadata["url"], "commit": ALIBABA_COMMIT,
                           "sha256": metadata["sha256"], "compressed_bytes": path.stat().st_size})

    capacities = _load_machine_specs(ali_dir / "pai_machine_spec.tar.gz")
    task_gpu, task_stats = _load_task_gpu(ali_dir / "pai_task_table.tar.gz")
    db_path = output_dir / "backfill" / "reservation_events.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    instance_stats = _build_event_db(ali_dir / "pai_instance_table.tar.gz", task_gpu, db_path)
    analysis = _extract_holes(db_path, capacities)
    db_path.unlink()
    holes = analysis.holes
    _write_csv(output_dir / "backfill" / "backfill_holes_integral_reservations.csv", holes)

    total_stranded = sum(float(hole["duration_seconds"]) for hole in holes)
    if abs(total_stranded - analysis.counters.get("stranded_gpu_seconds", 0.0)) > max(1.0, total_stranded * 1e-9):
        raise RuntimeError("Hole durations do not reproduce stranded GPU-seconds")
    threshold_rows: list[dict[str, Any]] = []
    for threshold in HOLE_THRESHOLDS:
        contained = sum(float(hole["duration_seconds"]) for hole in holes if float(hole["duration_seconds"]) >= threshold)
        threshold_rows.append({"threshold_seconds": threshold, "stranded_gpu_seconds_in_holes": contained,
                               "fraction_of_stranded_gpu_seconds": contained / total_stranded if total_stranded else math.nan})
    usable_rows: list[dict[str, Any]] = []
    for cold in COLD_OVERHEADS:
        usable = sum(
            _floor_quantum(max(0.0, float(hole["duration_seconds"]) - cold - CHECKPOINT_SECONDS))
            for hole in holes
        )
        usable_rows.append({"cold_overhead_seconds": cold, "quantum_seconds": QUANTUM_SECONDS,
                            "usable_gpu_seconds": usable,
                            "usable_fraction_of_stranded_gpu_seconds": usable / total_stranded if total_stranded else math.nan})
    _write_csv(output_dir / "backfill" / "hole_threshold_summary.csv", threshold_rows)
    _write_csv(output_dir / "backfill" / "usable_capacity_summary.csv", usable_rows)

    total_exposure = analysis.counters.get("total_machine_capacity_seconds", 0.0)
    valid_exposure = analysis.counters.get("valid_machine_capacity_seconds", 0.0)
    coverage = valid_exposure / total_exposure if total_exposure else 0.0
    warm_fraction = next(row["usable_fraction_of_stranded_gpu_seconds"] for row in usable_rows if row["cold_overhead_seconds"] == 0)
    contained_5_7 = next(row["fraction_of_stranded_gpu_seconds"] for row in threshold_rows if row["threshold_seconds"] == 5.7)
    if not holes or coverage < 0.5:
        decision = "TRACE-LIMITED"
    elif contained_5_7 >= 0.25 and warm_fraction >= 0.10:
        decision = "STRONG OPPORTUNITY"
    elif warm_fraction < 0.01:
        decision = "NO-GO"
    else:
        decision = "WEAK OPPORTUNITY"
    durations = [float(hole["duration_seconds"]) for hole in holes]
    summary = {
        "decision": decision,
        "scope": "machine-level integral GPU reservation intervals only",
        "physical_gpu_identity_available": False,
        "valid_machine_capacity_exposure_fraction": coverage,
        "hole_count": len(holes),
        "total_stranded_gpu_seconds": total_stranded,
        "hole_duration_seconds": {
            "median": statistics.median(durations) if durations else None,
            "mean": statistics.fmean(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "thresholds": threshold_rows,
        "usable_capacity": usable_rows,
        "accounting": analysis.counters,
        "task_table": dict(task_stats),
        "instance_table": dict(instance_stats),
        "machine_specs_loaded": len(capacities),
        "caveats": [
            "physical GPU assignment is absent; reservation slots are machine-level fungible-capacity bands",
            "fractional and unknown GPU-reservation intervals are excluded",
            "machine observation is bounded by first/last GPU allocation event; edge idle time is excluded",
            "work is counted only in complete 5.7-second quanta",
            "delayed admission is not evaluated and remains the strongest next baseline",
        ],
        "decision_rule_source": "DATA_FEASIBILITY.md, frozen before numerical analysis",
    }
    if decision == "TRACE-LIMITED":
        summary["reason"] = (
            "Only {:.8%} of internally observed machine-capacity time is valid "
            "under the preregistered integral-reservation restriction; this is below 50%."
        ).format(coverage)
    for item in provenance:
        name = Path(item["file"]).name
        if name == "pai_machine_spec.tar.gz":
            item["records_used"] = len(capacities)
        elif name == "pai_task_table.tar.gz":
            item["records"] = task_stats["task_rows"]
            item["missing_plan_gpu_records"] = task_stats["missing_plan_gpu"]
        elif name == "pai_instance_table.tar.gz":
            item["records"] = instance_stats["instance_rows"]
            item["relative_start_seconds"] = instance_stats["trace_start_seconds"]
            item["relative_end_seconds"] = instance_stats["trace_end_seconds"]
            item["censored_lifecycle_records"] = instance_stats["censored_lifecycle_rows"]
    return summary, provenance


def _svg_line_chart(path: Path, title: str, x_label: str, y_label: str,
                    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]]) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 55, 75
    points = [point for _, values, _ in series for point in values]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if y_max <= y_min:
        y_max = y_min + 1.0
    sx = lambda x: left + (x - x_min) / (x_max - x_min or 1.0) * (width - left - right)
    sy = lambda y: height - bottom - (y - y_min) / (y_max - y_min) * (height - top - bottom)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>']
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>',
                  f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3g}</text>']
    for x in sorted(set(xs)):
        lines.append(f'<text x="{sx(x):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x:g}</text>')
    for index, (label, values, color) in enumerate(series):
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in values:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        lines += [f'<line x1="{width-220}" y1="{70+index*22}" x2="{width-190}" y2="{70+index*22}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{width-180}" y="{74+index*22}" font-family="sans-serif" font-size="12">{label}</text>']
    lines += [f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
              f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{y_label}</text>',
              '</svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_figures(output_dir: Path, spot_summary: dict[str, Any], backfill_summary: dict[str, Any] | None) -> None:
    rows = spot_summary["duration_rows_all_zone_pooled"]
    _svg_line_chart(
        output_dir / "figures" / "spot_interruption_exposure.svg",
        "Spot interruption exposure (all AWS V100 zones)", "Request duration (s)", "Probability",
        [("lower", [(r["request_duration_seconds"], r["probability_lower"]) for r in rows], "#1f6f78"),
         ("midpoint assumption", [(r["request_duration_seconds"], r["probability_midpoint"]) for r in rows], "#d97706"),
         ("upper", [(r["request_duration_seconds"], r["probability_upper"]) for r in rows], "#9f1239")],
    )
    _svg_line_chart(
        output_dir / "figures" / "spot_recoverable_work.svg",
        "Step-checkpoint-recoverable work", "Request duration (s)", "Fraction of nominal GPU work",
        [("lower", [(r["request_duration_seconds"], r["recoverable_fraction_lower"]) for r in rows], "#1f6f78"),
         ("midpoint assumption", [(r["request_duration_seconds"], r["recoverable_fraction_midpoint"]) for r in rows], "#d97706"),
         ("upper", [(r["request_duration_seconds"], r["recoverable_fraction_upper"]) for r in rows], "#9f1239")],
    )
    if backfill_summary and backfill_summary["decision"] != "TRACE-LIMITED":
        threshold_rows = backfill_summary["thresholds"]
        _svg_line_chart(
            output_dir / "figures" / "stranded_capacity_by_hole_duration.svg",
            "Stranded reservation capacity in sufficiently long holes", "Minimum hole duration (s)", "Fraction of stranded GPU-seconds",
            [("integral-reservation subset", [(r["threshold_seconds"], r["fraction_of_stranded_gpu_seconds"]) for r in threshold_rows], "#1f6f78")],
        )
        hole_path = output_dir / "backfill" / "backfill_holes_integral_reservations.csv"
        durations: list[float] = []
        with hole_path.open(newline="") as handle:
            durations = sorted(float(row["duration_seconds"]) for row in csv.DictReader(handle))
        if durations:
            sample_indices = sorted(set(round(i * (len(durations) - 1) / 100) for i in range(101)))
            cdf = [(durations[i], (i + 1) / len(durations)) for i in sample_indices]
            _svg_line_chart(output_dir / "figures" / "hole_duration_cdf.svg", "Hole-duration CDF",
                            "Hole duration (s)", "Fraction of holes", [("CDF", cdf, "#1f6f78")])


def _summary_markdown(spot: dict[str, Any], backfill: dict[str, Any]) -> str:
    primary = spot["primary_all_zone_pooled"]
    threshold = next((x for x in backfill.get("thresholds", []) if x["threshold_seconds"] == 5.7), None)
    warm = next((x for x in backfill.get("usable_capacity", []) if x["cold_overhead_seconds"] == 0), None)
    continuation = spot["continuation_rows_all_zone_pooled"]
    resume_0 = next(x for x in continuation if x["request_duration_seconds"] == 230 and x["cold_resume_seconds"] == 0 and x["policy"] == "exact_step_resume")
    restart_0 = next(x for x in continuation if x["request_duration_seconds"] == 230 and x["cold_resume_seconds"] == 0 and x["policy"] == "restart_from_zero")
    resume_30 = next(x for x in continuation if x["request_duration_seconds"] == 230 and x["cold_resume_seconds"] == 30 and x["policy"] == "exact_step_resume")
    restart_30 = next(x for x in continuation if x["request_duration_seconds"] == 230 and x["cold_resume_seconds"] == 30 and x["policy"] == "restart_from_zero")
    recommendation = "ADVANCE NEITHER"
    spot_advance = spot["decision"] == "STRONG OPPORTUNITY"
    backfill_advance = backfill["decision"] == "STRONG OPPORTUNITY"
    if spot_advance and backfill_advance:
        recommendation = "ADVANCE BOTH"
    elif spot_advance:
        recommendation = "ADVANCE D"
    elif backfill_advance:
        recommendation = "ADVANCE A"
    lines = [
        "# Round 3 Cheap Fatal Screening", "", "## D - Generative Spot Computing",
        f"Decision: {spot['decision']}", "",
        f"- At 230 s, interruption probability is {primary['probability_lower']:.4f}-{primary['probability_upper']:.4f}; the uniform-within-interval midpoint estimate is {primary['probability_midpoint']:.4f}.",
        f"- Expected restart loss is {primary['restart_lost_lower']:.2f}-{primary['restart_lost_upper']:.2f} GPU-s/request; midpoint {primary['restart_lost_midpoint']:.2f}.",
        f"- Step-checkpoint-recoverable work is {primary['recoverable_fraction_lower']:.3%}-{primary['recoverable_fraction_upper']:.3%} of nominal work; midpoint {primary['recoverable_fraction_midpoint']:.3%}.",
        f"- In the midpoint continuation simulation, resume reduces mean 230 s completion time by {restart_0['mean_wall_seconds_completed'] - resume_0['mean_wall_seconds_completed']:.2f} s at 0 s cold overhead and {restart_30['mean_wall_seconds_completed'] - resume_30['mean_wall_seconds_completed']:.2f} s at 30 s cold overhead.",
        f"- The trace resolution is 32 s; midpoint numbers are assumptions, not observed event timestamps.", "",
        "## A - Generative Backfilling", f"Decision: {backfill['decision']}", "",
    ]
    if backfill["decision"] != "TRACE-LIMITED" and threshold and warm:
        lines += [
            f"- Valid integral-reservation exposure covers {backfill['valid_machine_capacity_exposure_fraction']:.2%} of internally observed machine-capacity time.",
            f"- {threshold['fraction_of_stranded_gpu_seconds']:.2%} of stranded reservation GPU-seconds lie in machine-level holes at least 5.7 s long.",
            f"- {warm['usable_fraction_of_stranded_gpu_seconds']:.2%} remains usable in complete 5.7 s quanta in the warm case.",
            f"- Physical GPU identities are absent; conclusions apply only to machine-level fungible reservation capacity.",
        ]
    else:
        lines += [
            f"- The public trace did not support a valid hole analysis: {backfill.get('reason', 'see backfill_summary.json').rstrip('.')}.",
            "- Fractional reservations, unknown GPU requests, censored lifecycles, and missing physical GPU placement prevent a representative 5.7-second free-window distribution.",
            "- No hole-duration or stranded-capacity figure is reported from the non-representative residual subset.",
        ]
    lines += ["", "## Recommendation", "", recommendation, "", "## Validity Boundary", "",
              "Measured trace values are separated from midpoint/cold-start simulation assumptions in the CSV and JSON outputs. No GPU experiment was run. Low utilization was never treated as free capacity. Delayed admission was not evaluated.", ""]
    return "\n".join(lines)


def _self_test() -> None:
    values = [1, 1, 0, 0]
    exposure = first_interruption_exposure(values, 32, 0, 40)
    assert exposure.probability_lower == 0.0
    assert exposure.probability_upper == 1.0
    assert abs(exposure.probability_midpoint - 0.25) < 1e-12
    exact = first_interruption_exposure(values, 32, 0, 80)
    assert exact.probability_lower == exact.probability_upper == 1.0
    boundaries, segments = _availability_segments([1, 1, 0, 1, 1], 10)
    assert boundaries == [0.0, 15.0, 25.0, 40.0]
    assert segments[1][2] == 0
    with tempfile.TemporaryDirectory() as temp:
        db = Path(temp) / "events.sqlite"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE events(machine TEXT, time REAL, delta_units INTEGER, delta_invalid INTEGER)")
        con.executemany("INSERT INTO events VALUES (?,?,?,?)", [
            ("m", 0, 1, 0), ("m", 10, -1, 0), ("m", 20, 1, 0), ("m", 30, -1, 0),
        ])
        con.commit(); con.close()
        holes = _extract_holes(db, {"m": 1}).holes
        assert [h["duration_seconds"] for h in holes] == [10.0]
    print("self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_serving_round3_screening"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    output_dir: Path = args.output_dir.resolve()
    source_dir = (args.source_dir or output_dir / "sources").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feasibility = output_dir / "DATA_FEASIBILITY.md"
    if not feasibility.exists():
        raise RuntimeError("DATA_FEASIBILITY.md must exist before numerical analysis")
    if args.download:
        download_sources(source_dir, include_alibaba=not args.skip_backfill)

    _write_csv(output_dir / "backfill_trace_feasibility.csv", _feasibility_rows())
    spot_rows, spot_summary, provenance = analyze_spot(source_dir, output_dir)
    _write_csv(output_dir / "spot_screening.csv", spot_rows)
    _write_json(output_dir / "spot_summary.json", spot_summary)

    if args.skip_backfill:
        backfill_summary = {"decision": "TRACE-LIMITED", "reason": "Alibaba numerical analysis explicitly skipped"}
    else:
        backfill_summary, ali_provenance = analyze_backfill(source_dir, output_dir)
        provenance.extend(ali_provenance)
    _write_json(output_dir / "backfill_summary.json", backfill_summary)

    provenance_payload = {
        "analysis_script": str(Path(__file__).resolve()), "analysis_script_sha256": _sha256(Path(__file__)),
        "python": sys.version, "cwd": os.getcwd(), "sources": provenance,
        "measured_system_parameters": {
            "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "resolution": "480x832",
            "frames": 33, "steps": 40, "trajectory_state_bytes_approx": 1_800_000,
            "checkpoint_seconds": CHECKPOINT_SECONDS, "quantum_seconds": QUANTUM_SECONDS,
            "typical_denoising_seconds": 230, "active_expert_weights_gb_approx": 26.6,
            "cold_load_observed_seconds": "1.8-2.8",
        },
        "simulation_assumptions": {
            "spot_transition_midpoint": "uniform event time within each 32-second transition interval",
            "spot_resume": "checkpoint after each complete 5.7-second quantum",
            "backfill": "machine-level integral reservations are fungible slots; complete quanta only",
        },
        "documentation": [
            {"name": "SkyPilot spot-traces README", "url": "https://github.com/skypilot-org/spot-traces",
             "use": "format and separation of availability probes from preemption traces"},
            {"name": "Can't Be Late, NSDI 2024", "url": "https://www.usenix.org/system/files/nsdi24-wu-zhanghao.pdf",
             "use": "real-preemption collection semantics"},
            {"name": "Alibaba cluster-trace-gpu-v2020 README",
             "url": "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md",
             "use": "table schemas, resource units, machine placement, and published checksums"},
            {"name": "Alibaba clusterdata repository", "url": "https://github.com/alibaba/clusterdata",
             "use": "candidate dataset inventory"},
        ],
    }
    _write_json(output_dir / "trace_provenance.json", provenance_payload)
    _write_figures(output_dir, spot_summary, backfill_summary)
    (output_dir / "round3_summary.md").write_text(_summary_markdown(spot_summary, backfill_summary))
    print(json.dumps({"files_created_under": str(output_dir), "spot_decision": spot_summary["decision"],
                      "backfill_decision": backfill_summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
