#!/usr/bin/env python3
"""Kill test for equal-byte, differently shaped checkpoint shard loss.

This experiment changes only the serialized ordering of a Wan checkpoint
latent. It deliberately does not implement a runtime, scheduler, or erasure
coding mechanism.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight
from experiments import video_denoising_error_correction_killtest as correction
from experiments import video_propagation_aware_checkpoint_killtest as propagation
from experiments import video_state_protection_killtest as protection


MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LAYOUTS = (
    "spatial_contiguous",
    "temporal_contiguous",
    "channel_contiguous",
    "random_element",
    "interleaved_striped",
)
CONTIGUOUS_LAYOUTS = ("spatial_contiguous", "temporal_contiguous")
DISTRIBUTED_LAYOUTS = ("random_element", "interleaved_striped")

LOSS_FIELDS = [
    "stage", "analysis_type", "prompt_id", "motion_category", "prompt", "seed",
    "checkpoint_step", "layout", "shard_count", "lost_shards", "loss_fraction",
    "fill_method", "raw_latent_bytes", "payload_bytes", "metadata_bytes",
    "total_checkpoint_bytes", "missing_payload_bytes", "missing_byte_fraction",
    "payload_sha256", "metadata_sha256", "encode_prepare_latency_ms",
    "storage_write_latency_ms", "storage_read_latency_ms",
    "decode_reconstruction_latency_ms", "resume_latency_ms", "initial_latent_mse",
    "initial_normalized_l2", "initial_cosine_similarity", "iso_error_target",
    "iso_error_scale", "final_latent_mse", "final_normalized_l2",
    "final_cosine_similarity", "final_error_amplification", "video_mse",
    "spatial_quality", "temporal_shape_quality", "temporal_dynamic_quality",
    "semantic_quality", "semantic_quality_relative", "artifact_dir",
    "recovered_video_path", "trajectory_metadata_path",
    "missing_bytes", "missing_fraction_actual", "initial_mse", "initial_cosine",
    "final_latent_error", "temporal_quality", "final_video_mse",
    "serialization_throughput_gbps", "reassembly_throughput_gbps",
    "bytes_per_shard",
    "layout_prepare_latency_ms", "gather_copy_latency_ms",
    "metadata_prepare_latency_ms",
]


@dataclass(frozen=True)
class LayoutPlan:
    layout: str
    shape: tuple[int, ...]
    shard_count: int
    random_seed: int
    storage_order: torch.Tensor
    shard_ranges: tuple[tuple[int, int], ...]
    block_elements: int | None = None

    @property
    def elements_per_shard(self) -> int:
        return self.shard_ranges[0][1] - self.shard_ranges[0][0]


@dataclass(frozen=True)
class SerializedLayout:
    plan: LayoutPlan
    shards: tuple[bytes, ...]
    metadata_raw: bytes
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    encode_ms: float
    layout_prepare_ms: float
    gather_copy_ms: float
    metadata_prepare_ms: float
    write_ms: float
    payload_sha256: str
    metadata_sha256: str
    artifact_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--prompt-set",
        default="experiments/video_propagation_aware_checkpoint_prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/video_error_shaped_checkpoint_killtest/smoke",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "loss_fraction", "progress", "content"),
        default="smoke",
    )
    parser.add_argument("--prior-summary")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layout-seed", type=int, default=20260827)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--sample-solver", choices=("euler",), default="euler")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--exact-resume-tolerance", type=float, default=1e-6)
    parser.add_argument("--iso-byte-tolerance", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
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
        "sample_solver": "euler",
        "enable_cpu_offload": True,
        "enable_layerwise_offload": False,
        "iso_byte_tolerance": 0.01,
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Kill-test configuration changed: {actual} != {expected}")


def _stage_scope(
    args: argparse.Namespace,
    entries: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[int], list[int]]:
    if args.stage == "smoke":
        return entries[:1], [20], [4]
    if args.stage == "loss_fraction":
        return entries[:1], [20], [8, 4, 2]
    if args.stage == "progress":
        return entries[:1], [10, 20, 30], [4]
    categories = {"static_low_motion", "camera_motion", "fast_complex_motion"}
    selected = [entry for entry in entries if entry["motion_category"] in categories]
    return selected[:4], [20], [4]


def _check_stage_gate(args: argparse.Namespace) -> None:
    if args.stage == "smoke":
        return
    if not args.prior_summary:
        raise ValueError(f"--stage {args.stage} requires --prior-summary")
    summary = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))
    if summary.get("eligible_next_stage") != args.stage:
        raise ValueError(
            f"Prior result does not permit {args.stage}: "
            f"{summary.get('eligible_next_stage')!r}"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shard_ranges(size: int, count: int) -> tuple[tuple[int, int], ...]:
    if count <= 0 or size % count:
        raise ValueError(f"{size} elements cannot be divided into {count} equal shards")
    width = size // count
    return tuple((index * width, (index + 1) * width) for index in range(count))


def _rectangle(height: int, width: int, cells: int) -> tuple[int, int]:
    candidates = [
        (h, cells // h)
        for h in range(1, height + 1)
        if cells % h == 0 and cells // h <= width
    ]
    if not candidates:
        raise ValueError(f"Cannot form a {cells}-cell rectangle inside {height}x{width}")
    return min(candidates, key=lambda pair: abs(pair[0] / pair[1] - height / width))


def _complete_order(first_shard: torch.Tensor, size: int) -> torch.Tensor:
    first_shard = first_shard.to(torch.int64).reshape(-1)
    if first_shard.unique().numel() != first_shard.numel():
        raise AssertionError("A layout shard contains duplicate source elements")
    selected = torch.zeros(size, dtype=torch.bool)
    selected[first_shard] = True
    return torch.cat((first_shard, torch.arange(size, dtype=torch.int64)[~selected]))


def _block_interleaved_order(shape: tuple[int, ...], shard_count: int) -> tuple[torch.Tensor, int]:
    coordinates = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    canonical = coordinates.permute(0, 2, 3, 4, 1).reshape(-1)
    target = canonical.numel() // shard_count
    block = max(value for value in range(1, min(4, target) + 1) if target % value == 0)
    blocks = canonical.reshape(-1, block)
    groups = blocks.reshape(-1, shard_count, block)
    group_indices = torch.arange(groups.shape[0], dtype=torch.int64)
    per_shard = [
        groups[group_indices, (shard + group_indices) % shard_count].reshape(-1)
        for shard in range(shard_count)
    ]
    if any(indices.numel() != target for indices in per_shard):
        raise AssertionError("Striped layout did not produce equal shards")
    return torch.cat(per_shard), block


def build_layout_plan(
    shape: tuple[int, ...],
    layout: str,
    shard_count: int,
    *,
    random_seed: int,
) -> LayoutPlan:
    if len(shape) != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {shape}")
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout}")
    size = math.prod(shape)
    ranges = _shard_ranges(size, shard_count)
    target = size // shard_count
    coordinates = torch.arange(size, dtype=torch.int64).reshape(shape)
    block_elements = None
    if layout == "spatial_contiguous":
        batch, channels, temporal, height, width = shape
        if target % (batch * channels * temporal):
            raise ValueError("Spatial shard cannot have an exact equal-byte rectangle")
        rectangle = _rectangle(height, width, target // (batch * channels * temporal))
        first = coordinates[..., : rectangle[0], : rectangle[1]].reshape(-1)
        order = _complete_order(first, size)
    elif layout == "temporal_contiguous":
        order_by_time = coordinates.permute(0, 2, 1, 3, 4).reshape(-1)
        order = _complete_order(order_by_time[:target], size)
    elif layout == "channel_contiguous":
        order = torch.arange(size, dtype=torch.int64)
    elif layout == "random_element":
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        order = torch.randperm(size, generator=generator)
    else:
        order, block_elements = _block_interleaved_order(shape, shard_count)
    if order.numel() != size or order.unique().numel() != size:
        raise AssertionError(f"{layout} is not a permutation")
    return LayoutPlan(
        layout=layout,
        shape=shape,
        shard_count=shard_count,
        random_seed=random_seed,
        storage_order=order,
        shard_ranges=ranges,
        block_elements=block_elements,
    )


def encode_layout_bytes(latent: torch.Tensor, plan: LayoutPlan) -> tuple[bytes, ...]:
    source = latent.detach().float().cpu().contiguous()
    if tuple(source.shape) != plan.shape:
        raise ValueError(f"Shape mismatch: {tuple(source.shape)} != {plan.shape}")
    storage = source.reshape(-1)[plan.storage_order].numpy()
    raw = storage.tobytes()
    return tuple(raw[start * 4 : end * 4] for start, end in plan.shard_ranges)


def reassemble_layout(
    shards: tuple[bytes | None, ...],
    plan: LayoutPlan,
    *,
    fill_method: str = "zero",
) -> torch.Tensor:
    if fill_method != "zero":
        raise ValueError("The preregistered primary fill method is zero")
    if len(shards) != plan.shard_count:
        raise ValueError("Shard count mismatch")
    storage = torch.empty(math.prod(plan.shape), dtype=torch.float32)
    for shard, (start, end) in zip(shards, plan.shard_ranges, strict=True):
        if shard is None:
            storage[start:end] = 0.0
            continue
        expected = (end - start) * 4
        if len(shard) != expected:
            raise ValueError(f"Shard has {len(shard)} bytes, expected {expected}")
        storage[start:end] = torch.from_numpy(np.frombuffer(shard, dtype=np.float32).copy())
    source_flat = torch.empty_like(storage)
    source_flat[plan.storage_order] = storage
    return source_flat.reshape(plan.shape).contiguous()


def missing_source_indices(plan: LayoutPlan, lost_shards: tuple[int, ...] = (0,)) -> torch.Tensor:
    chunks = [
        plan.storage_order[plan.shard_ranges[index][0] : plan.shard_ranges[index][1]]
        for index in lost_shards
    ]
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)


def _metadata(plan: LayoutPlan, latent: torch.Tensor) -> dict[str, Any]:
    return {
        "format": "error_shaped_fp32_shards_v1",
        "layout": plan.layout,
        "shape": list(plan.shape),
        "dtype": "float32",
        "shard_count": plan.shard_count,
        "random_seed": plan.random_seed,
        "block_elements": plan.block_elements,
        "shards": [
            {"shard_id": index, "elements": end - start, "payload_bytes": (end - start) * 4}
            for index, (start, end) in enumerate(plan.shard_ranges)
        ],
        "source_sha256": _sha256(latent.detach().float().cpu().contiguous().numpy().tobytes()),
    }


def layout_plan_from_metadata(metadata_raw: bytes) -> LayoutPlan:
    metadata = json.loads(metadata_raw.decode("utf-8"))
    if metadata.get("format") != "error_shaped_fp32_shards_v1":
        raise ValueError(f"Unsupported checkpoint format: {metadata.get('format')!r}")
    if metadata.get("dtype") != "float32":
        raise ValueError(f"Unsupported checkpoint dtype: {metadata.get('dtype')!r}")
    plan = build_layout_plan(
        tuple(int(value) for value in metadata["shape"]),
        str(metadata["layout"]),
        int(metadata["shard_count"]),
        random_seed=int(metadata["random_seed"]),
    )
    expected_shards = [
        {
            "shard_id": index,
            "elements": end - start,
            "payload_bytes": (end - start) * 4,
        }
        for index, (start, end) in enumerate(plan.shard_ranges)
    ]
    if metadata.get("shards") != expected_shards:
        raise ValueError("Serialized shard metadata does not match the deterministic layout")
    if metadata.get("block_elements") != plan.block_elements:
        raise ValueError("Serialized block size does not match the deterministic layout")
    return plan


def serialize_layout(
    latent: torch.Tensor,
    plan: LayoutPlan,
    artifact_dir: Path,
    *,
    layout_prepare_ms: float = 0.0,
) -> SerializedLayout:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shards = encode_layout_bytes(latent, plan)
    gather_copy_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    metadata_raw = json.dumps(
        _metadata(plan, latent), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata_prepare_ms = (time.perf_counter() - started) * 1000.0
    encode_ms = layout_prepare_ms + gather_copy_ms + metadata_prepare_ms
    started = time.perf_counter()
    for index, shard in enumerate(shards):
        propagation._atomic_bytes(artifact_dir / f"shard_{index:02d}.bin", shard)
    propagation._atomic_bytes(artifact_dir / "checkpoint.metadata.json", metadata_raw)
    write_ms = (time.perf_counter() - started) * 1000.0
    payload = b"".join(shards)
    return SerializedLayout(
        plan=plan,
        shards=shards,
        metadata_raw=metadata_raw,
        payload_bytes=len(payload),
        metadata_bytes=len(metadata_raw),
        total_bytes=len(payload) + len(metadata_raw),
        encode_ms=encode_ms,
        layout_prepare_ms=layout_prepare_ms,
        gather_copy_ms=gather_copy_ms,
        metadata_prepare_ms=metadata_prepare_ms,
        write_ms=write_ms,
        payload_sha256=_sha256(payload),
        metadata_sha256=_sha256(metadata_raw),
        artifact_dir=artifact_dir,
    )


def load_damaged_layout(
    encoded: SerializedLayout,
    *,
    lost_shards: tuple[int, ...] = (0,),
) -> tuple[torch.Tensor, float, float]:
    started = time.perf_counter()
    metadata_raw = (encoded.artifact_dir / "checkpoint.metadata.json").read_bytes()
    plan = layout_plan_from_metadata(metadata_raw)
    shards: list[bytes | None] = []
    for index in range(plan.shard_count):
        if index in lost_shards:
            shards.append(None)
        else:
            shards.append((encoded.artifact_dir / f"shard_{index:02d}.bin").read_bytes())
    read_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    restored = reassemble_layout(tuple(shards), plan, fill_method="zero")
    decode_ms = (time.perf_counter() - started) * 1000.0
    return restored, read_ms, decode_ms


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid_result_row(row: dict[str, str], total_steps: int) -> bool:
    try:
        video_path = Path(row["recovered_video_path"])
        metadata_path = Path(row["trajectory_metadata_path"])
        serialized_dir = Path(row["artifact_dir"])
        shard_paths = sorted(serialized_dir.glob("shard_*.bin"))
        checkpoint_metadata = serialized_dir / "checkpoint.metadata.json"
        checkpoint_step = int(row["checkpoint_step"])
        if (
            not video_path.exists()
            or not metadata_path.exists()
            or not checkpoint_metadata.exists()
            or len(shard_paths) != int(row["shard_count"])
        ):
            return False
        payload = b"".join(path.read_bytes() for path in shard_paths)
        metadata_raw = checkpoint_metadata.read_bytes()
        if (
            len(payload) != int(row["payload_bytes"])
            or len(metadata_raw) != int(row["metadata_bytes"])
            or _sha256(payload) != row["payload_sha256"]
            or _sha256(metadata_raw) != row["metadata_sha256"]
        ):
            return False
        video = np.load(video_path, allow_pickle=False, mmap_mode="r")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = correction._metadata_records(metadata)
        return video.size > 0 and set(records) == {0, total_steps - checkpoint_step} and all(
            Path(record["latent_path"]).exists() for record in records.values()
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str, str]:
    return (
        str(row["prompt_id"]),
        int(row["checkpoint_step"]),
        int(row["shard_count"]),
        str(row["layout"]),
        str(row["analysis_type"]),
    )


def _quality(
    video: np.ndarray,
    exact_video: np.ndarray,
    prompt: str,
    semantic: Any | None,
    exact_semantic: float,
) -> dict[str, float]:
    return correction._quality_metrics(video, exact_video, prompt, semantic, exact_semantic)


def _result_row(
    *,
    args: argparse.Namespace,
    entry: dict[str, str],
    seed: int,
    checkpoint_step: int,
    shard_count: int,
    layout: str,
    analysis_type: str,
    source: torch.Tensor,
    candidate: torch.Tensor,
    encoded: SerializedLayout,
    read_ms: float,
    decode_ms: float,
    resume_ms: float,
    video: np.ndarray,
    metadata: dict[str, Any],
    exact_video: np.ndarray,
    exact_final: torch.Tensor,
    semantic: Any | None,
    exact_semantic: float,
    iso_error_target: float | None,
    iso_error_scale: float,
    artifact_dir: Path,
    label: str,
) -> dict[str, Any]:
    initial = correction.latent_error(source, candidate)
    final = correction.latent_error(exact_final, propagation._final_latent(metadata))
    quality = _quality(video, exact_video, entry["prompt"], semantic, exact_semantic)
    missing_bytes = encoded.plan.elements_per_shard * 4
    return {
        "stage": args.stage,
        "analysis_type": analysis_type,
        "prompt_id": entry["prompt_id"],
        "motion_category": entry["motion_category"],
        "prompt": entry["prompt"],
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "layout": layout,
        "shard_count": shard_count,
        "lost_shards": "0",
        "loss_fraction": 1.0 / shard_count,
        "fill_method": "zero" if analysis_type == "iso_byte" else "scaled_zero_fill_control",
        "raw_latent_bytes": source.numel() * 4,
        "payload_bytes": encoded.payload_bytes,
        "metadata_bytes": encoded.metadata_bytes,
        "total_checkpoint_bytes": encoded.total_bytes,
        "missing_payload_bytes": missing_bytes,
        "missing_byte_fraction": missing_bytes / encoded.payload_bytes,
        "payload_sha256": encoded.payload_sha256,
        "metadata_sha256": encoded.metadata_sha256,
        "encode_prepare_latency_ms": encoded.encode_ms,
        "storage_write_latency_ms": encoded.write_ms,
        "storage_read_latency_ms": read_ms,
        "decode_reconstruction_latency_ms": decode_ms,
        "resume_latency_ms": resume_ms,
        "initial_latent_mse": initial["mse"],
        "initial_normalized_l2": initial["normalized_l2"],
        "initial_cosine_similarity": initial["cosine_similarity"],
        "iso_error_target": iso_error_target,
        "iso_error_scale": iso_error_scale,
        "final_latent_mse": final["mse"],
        "final_normalized_l2": final["normalized_l2"],
        "final_cosine_similarity": final["cosine_similarity"],
        "final_error_amplification": final["normalized_l2"]
        / max(initial["normalized_l2"], 1e-12),
        "video_mse": quality["video_mse_vs_uninterrupted"],
        "spatial_quality": quality["spatial_quality"],
        "temporal_shape_quality": quality["temporal_shape_quality"],
        "temporal_dynamic_quality": quality["temporal_dynamic_quality"],
        "semantic_quality": quality["semantic_quality"],
        "semantic_quality_relative": quality["semantic_quality_vs_uninterrupted"],
        "artifact_dir": str(encoded.artifact_dir),
        "recovered_video_path": str(artifact_dir / f"{label}.npy"),
        "trajectory_metadata_path": str(artifact_dir / f"{label}_trajectory_probe.json"),
        "missing_bytes": missing_bytes,
        "missing_fraction_actual": missing_bytes / encoded.payload_bytes,
        "initial_mse": initial["mse"],
        "initial_cosine": initial["cosine_similarity"],
        "final_latent_error": final["normalized_l2"],
        "temporal_quality": quality["temporal_dynamic_quality"],
        "final_video_mse": quality["video_mse_vs_uninterrupted"],
        "serialization_throughput_gbps": encoded.payload_bytes
        / max(encoded.encode_ms / 1000.0, 1e-12)
        / 1e9,
        "reassembly_throughput_gbps": encoded.payload_bytes
        / max(decode_ms / 1000.0, 1e-12)
        / 1e9,
        "bytes_per_shard": encoded.plan.elements_per_shard * 4,
        "layout_prepare_latency_ms": encoded.layout_prepare_ms,
        "gather_copy_latency_ms": encoded.gather_copy_ms,
        "metadata_prepare_latency_ms": encoded.metadata_prepare_ms,
    }


def _comparison_rows(rows: list[dict[str, Any]], analysis_type: str) -> list[dict[str, Any]]:
    output = []
    cells = sorted(
        {
            (str(row["prompt_id"]), int(row["checkpoint_step"]), int(row["shard_count"]))
            for row in rows
            if row["analysis_type"] == analysis_type
        }
    )
    for prompt_id, step, shard_count in cells:
        cell = {
            str(row["layout"]): row
            for row in rows
            if row["analysis_type"] == analysis_type
            and row["prompt_id"] == prompt_id
            and int(row["checkpoint_step"]) == step
            and int(row["shard_count"]) == shard_count
        }
        missing = [int(row["missing_payload_bytes"]) for row in cell.values()]
        mismatch = (max(missing) - min(missing)) / max(max(missing), 1) if missing else math.nan
        for layout, row in sorted(cell.items()):
            contiguous_dynamic = max(
                (
                    float(cell[name]["temporal_dynamic_quality"])
                    for name in CONTIGUOUS_LAYOUTS
                    if name in cell
                ),
                default=float("nan"),
            )
            output.append(
                {
                    "analysis_type": analysis_type,
                    "prompt_id": prompt_id,
                    "checkpoint_step": step,
                    "shard_count": shard_count,
                    "loss_fraction": 1.0 / shard_count,
                    "layout": layout,
                    "missing_payload_bytes": row["missing_payload_bytes"],
                    "iso_byte_mismatch": mismatch,
                    "within_iso_byte_tolerance": mismatch <= 0.01,
                    "initial_normalized_l2": row["initial_normalized_l2"],
                    "final_normalized_l2": row["final_normalized_l2"],
                    "spatial_quality": row["spatial_quality"],
                    "temporal_dynamic_quality": row["temporal_dynamic_quality"],
                    "semantic_quality_relative": row["semantic_quality_relative"],
                    "dynamic_delta_vs_best_contiguous": float(row["temporal_dynamic_quality"])
                    - contiguous_dynamic,
                }
            )
    return output


def _cell_signal(rows: list[dict[str, Any]], analysis_type: str) -> dict[str, float | bool]:
    cell = {str(row["layout"]): row for row in rows if row["analysis_type"] == analysis_type}
    if not all(name in cell for name in (*CONTIGUOUS_LAYOUTS, *DISTRIBUTED_LAYOUTS)):
        return {
            "complete": False,
            "dynamic_advantage": float("nan"),
            "spatial_floor": float("nan"),
            "random_vs_interleaved_dynamic_gap": float("nan"),
        }
    contiguous_best = max(
        float(cell[name]["temporal_dynamic_quality"])
        for name in CONTIGUOUS_LAYOUTS
    )
    distributed_worst = min(
        float(cell[name]["temporal_dynamic_quality"])
        for name in DISTRIBUTED_LAYOUTS
    )
    return {
        "complete": True,
        "dynamic_advantage": distributed_worst - contiguous_best,
        "spatial_floor": min(float(cell[name]["spatial_quality"]) for name in DISTRIBUTED_LAYOUTS),
        "random_vs_interleaved_dynamic_gap": abs(
            float(cell["random_element"]["temporal_dynamic_quality"])
            - float(cell["interleaved_striped"]["temporal_dynamic_quality"])
        ),
    }


def _plot(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    primary = [row for row in rows if row["analysis_type"] == "iso_byte"]
    if not primary:
        return []
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(row["layout"]) for row in primary]
    axis.bar(labels, [float(row["temporal_dynamic_quality"]) for row in primary])
    axis.set(ylabel="Temporal dynamic quality", title="Equal missing payload bytes")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "layout_quality_comparison.pdf"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for layout in LAYOUTS:
        selected = sorted(
            (row for row in primary if row["layout"] == layout),
            key=lambda row: float(row["missing_byte_fraction"]),
        )
        if selected:
            axis.plot(
                [float(row["missing_byte_fraction"]) for row in selected],
                [float(row["temporal_dynamic_quality"]) for row in selected],
                marker="o",
                label=layout,
            )
    axis.set(xlabel="Missing payload fraction", ylabel="Temporal dynamic quality")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = output_dir / "missing_fraction_quality_frontier.pdf"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for layout in LAYOUTS:
        selected = [row for row in primary if row["layout"] == layout]
        axis.scatter(
            [float(row["initial_latent_mse"]) for row in selected],
            [float(row["temporal_dynamic_quality"]) for row in selected],
            label=layout,
        )
    axis.set(xlabel="Initial latent MSE", ylabel="Temporal dynamic quality")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path = output_dir / "initial_mse_vs_final_quality.pdf"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, len(LAYOUTS), figsize=(12.0, 2.8), sharey=True)
    rng = np.random.default_rng(20260827)
    diagrams = {
        "spatial_contiguous": np.pad(np.ones((8, 8)), ((0, 8), (0, 8))),
        "temporal_contiguous": np.pad(np.ones((4, 16)), ((0, 12), (0, 0))),
        "channel_contiguous": np.pad(np.ones((16, 4)), ((0, 0), (0, 12))),
        "random_element": (rng.random((16, 16)) < 0.25).astype(float),
        "interleaved_striped": (np.arange(256).reshape(16, 16) % 4 == 0).astype(float),
    }
    for axis, layout in zip(axes, LAYOUTS, strict=True):
        axis.imshow(diagrams[layout], cmap="Reds", vmin=0, vmax=1)
        axis.set_title(layout, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    path = output_dir / "layout_diagram.pdf"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    iso_byte_advantage = float(summary["iso_byte_signal"]["dynamic_advantage"])
    iso_error_advantage = float(summary["iso_error_signal"]["dynamic_advantage"])
    layout_gap = float(
        summary["iso_byte_signal"]["random_vs_interleaved_dynamic_gap"]
    )
    material_geometry = iso_byte_advantage >= 0.03
    mse_control_survives = iso_error_advantage >= 0.01
    interleaving_tracks_random = layout_gap <= 0.03
    lines = [
        "# Error-Shaped Checkpoint Kill Test",
        "",
        "## Configuration",
        "",
        f"- Stage: `{summary['stage']}`",
        "- Solver: exact-resume `euler`.",
        "- Primary reconstruction: zero-fill one actual FP32 shard.",
        "- Primary comparison: equal missing payload bytes (<=1% mismatch).",
        "- Iso-error is a separate synthetic control and is not a storage scheme.",
        "",
        "## CONFIRMED",
        "",
        "- Exact-resume maximum normalized latent error: "
        f"`{summary['max_exact_resume_error']:.6g}`.",
        f"- Completed recovery rows: `{summary['recovery_row_count']}`.",
        f"- Iso-byte distributed-vs-contiguous temporal advantage: "
        f"`{summary['iso_byte_signal']['dynamic_advantage']:.4f}`.",
        f"- Iso-error distributed-vs-contiguous temporal advantage: "
        f"`{summary['iso_error_signal']['dynamic_advantage']:.4f}`.",
        "- Random-vs-striped temporal-quality gap: "
        f"`{summary['iso_byte_signal']['random_vs_interleaved_dynamic_gap']:.4f}`.",
        "- Initial-MSE vs temporal-quality Pearson correlation: "
        f"`{summary['initial_mse_vs_temporal_quality_pearson']:.4f}`.",
        "- Maximum layout preparation/serialization time: "
        f"`{summary['maximum_encode_prepare_latency_ms']:.3f} ms`.",
        "- Maximum read/reassembly time: "
        f"`{summary['maximum_read_plus_decode_latency_ms']:.3f} ms`.",
        "",
        "## INFERRED",
        "",
        "- A positive distributed-layout advantage is consistent with error geometry "
        "affecting recovery.",
        "- The iso-error control removes initial error magnitude, but remains a synthetic control.",
        "",
        "## UNKNOWN",
        "",
        "- Progress and content generality remain unknown unless explicitly unlocked "
        "by the prior stage.",
        "- Real erasure coding benefit remains unknown and is not implemented in this kill test.",
        "",
        "## Research Questions",
        "",
        f"- Q1: {'YES' if material_geometry else 'NO'} in the measured cells; "
        f"equal-byte temporal-quality delta is `{iso_byte_advantage:.4f}`.",
        f"- Q2: {'YES' if material_geometry else 'NO'} under the preregistered "
        "distributed-vs-best-contiguous comparison.",
        f"- Q3: {'YES' if interleaving_tracks_random else 'NO'} at a 0.03 tolerance; "
        f"random-vs-striped gap is `{layout_gap:.4f}`.",
        f"- Q4: {'YES' if mse_control_survives else 'NO'}; iso-error temporal-quality "
        f"delta is `{iso_error_advantage:.4f}`.",
        "- Q5: "
        + (
            "Measured in `loss_fraction_frontier.csv`."
            if summary["stage"] in {"loss_fraction", "progress", "content"}
            else "UNKNOWN until the gated loss-fraction stage runs."
        ),
        "- Q6: "
        + (
            "Measured in the per-step cells."
            if summary["stage"] in {"progress", "content"}
            else "UNKNOWN until the gated progress stage runs."
        ),
        "- Q7: See `layout_overhead.csv`; serving suitability is not inferred "
        "from smoke alone.",
        "- Q8: UNKNOWN; equal-overhead erasure coding is gated and not implemented.",
        "",
        "## Judgment",
        "",
        summary["judgment"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _check_stage_gate(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "0,"}:
        raise EnvironmentError("Run with CUDA_VISIBLE_DEVICES=0 on an exclusive GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EnvironmentError("Exactly one CUDA device must be visible")

    prompt_path, prompt_hash, all_entries = correction._read_prompts(args.prompt_set)
    entries, checkpoint_steps, shard_counts = _stage_scope(args, all_entries)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "research_question": (
            "does equal-byte checkpoint loss geometry change exact-Euler recovery quality"
        ),
        "stage": args.stage,
        "prompt_set": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "prompt_ids": [entry["prompt_id"] for entry in entries],
        "checkpoint_steps": checkpoint_steps,
        "shard_counts": shard_counts,
        "layouts": list(LAYOUTS),
        "lost_shards": [0],
        "primary_fill_method": "zero",
        "primary_comparison": "iso missing FP32 payload bytes",
        "iso_byte_tolerance": 0.01,
        "iso_error_control": (
            "scale each zero-fill error direction to the minimum normalized L2 in its cell"
        ),
        "config": vars(args),
        "go_thresholds": {
            "strong_iso_byte_dynamic_advantage": 0.10,
            "strong_iso_error_dynamic_advantage": 0.05,
            "strong_distributed_spatial_floor": 0.80,
            "conditional_iso_byte_dynamic_advantage": 0.03,
            "conditional_iso_error_dynamic_advantage": 0.01,
            "conditional_distributed_spatial_floor": 0.70,
            "expanded_stage_required_cell_fraction": "2/3",
        },
    }
    config_path = output_dir / "layout_config.json"
    if (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8")) != preregistration
    ):
        raise ValueError(f"Existing preregistration differs: {config_path}")
    correction._atomic_json(config_path, preregistration)

    rows_path = output_dir / "loss_geometry_rows.csv"
    rows: list[dict[str, Any]] = _read_csv(rows_path) if args.resume else []
    if args.resume:
        rows = [row for row in rows if _valid_result_row(row, args.num_inference_steps)]
        correction._atomic_csv(rows_path, LOSS_FIELDS, rows)
    completed = {_row_key(row) for row in rows}

    semantic = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )
    if semantic is not None and semantic.disabled_reason is not None:
        raise RuntimeError(
            "Semantic metric is required unless --disable-semantic-metric is explicit: "
            f"{semantic.disabled_reason}"
        )
    omni = protection._make_omni(args)
    exact_errors: list[float] = []
    try:
        for prompt_index, entry in enumerate(entries):
            seed = args.seed + prompt_index
            artifact_dir = output_dir / "artifacts" / f"{entry['prompt_id']}_seed{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            capture_steps = sorted(set(checkpoint_steps + [args.num_inference_steps]))
            baseline_label = f"{entry['prompt_id']}_seed{seed}_uninterrupted_euler"
            baseline_video, baseline_meta, _ = propagation._load_or_run(
                omni,
                args,
                entry,
                seed,
                None,
                0,
                artifact_dir,
                baseline_label,
                capture_steps,
            )
            baseline_records = correction._metadata_records(baseline_meta)
            for checkpoint_step in checkpoint_steps:
                source = torch.load(
                    baseline_records[checkpoint_step]["latent_path"], map_location="cpu"
                ).float().contiguous()
                baseline_final = torch.load(
                    baseline_records[args.num_inference_steps]["latent_path"], map_location="cpu"
                ).float()
                exact_label = (
                    f"{entry['prompt_id']}_seed{seed}_step{checkpoint_step:02d}_exact_euler"
                )
                exact_video, exact_meta, _ = propagation._load_or_run(
                    omni,
                    args,
                    entry,
                    seed,
                    source,
                    checkpoint_step,
                    artifact_dir,
                    exact_label,
                )
                exact_final = propagation._final_latent(exact_meta)
                exact_error = correction.latent_error(baseline_final, exact_final)
                exact_video_delta = (
                    baseline_video.astype(np.float32) - exact_video.astype(np.float32)
                )
                exact_video_mse = float(np.mean(np.square(exact_video_delta)))
                if (
                    exact_error["normalized_l2"] > args.exact_resume_tolerance
                    or exact_video_mse > args.exact_resume_tolerance
                ):
                    raise RuntimeError(
                        f"Exact Euler resume failed at step {checkpoint_step}: "
                        f"latent={exact_error['normalized_l2']}, video_mse={exact_video_mse}"
                    )
                exact_errors.append(exact_error["normalized_l2"])
                exact_semantic = (
                    semantic.score_video(entry["prompt"], exact_video)
                    if semantic is not None
                    else float("nan")
                )

                for shard_count in shard_counts:
                    encoded_by_layout: dict[str, SerializedLayout] = {}
                    damaged_by_layout: dict[str, tuple[torch.Tensor, float, float]] = {}
                    initial_errors: dict[str, dict[str, float]] = {}
                    for layout in LAYOUTS:
                        layout_started = time.perf_counter()
                        plan = build_layout_plan(
                            tuple(source.shape),
                            layout,
                            shard_count,
                            random_seed=args.layout_seed,
                        )
                        layout_prepare_ms = (time.perf_counter() - layout_started) * 1000.0
                        encoded = serialize_layout(
                            source,
                            plan,
                            artifact_dir
                            / f"serialized_step{checkpoint_step:02d}_k{shard_count}_{layout}",
                            layout_prepare_ms=layout_prepare_ms,
                        )
                        if not torch.equal(
                            reassemble_layout(tuple(encoded.shards), plan), source
                        ):
                            raise RuntimeError(f"Zero-loss reassembly failed for {layout}")
                        damaged, read_ms, decode_ms = load_damaged_layout(encoded)
                        encoded_by_layout[layout] = encoded
                        damaged_by_layout[layout] = (damaged, read_ms, decode_ms)
                        initial_errors[layout] = correction.latent_error(source, damaged)

                    missing_bytes = {
                        layout: encoded.plan.elements_per_shard * 4
                        for layout, encoded in encoded_by_layout.items()
                    }
                    byte_mismatch = (
                        max(missing_bytes.values()) - min(missing_bytes.values())
                    ) / max(max(missing_bytes.values()), 1)
                    if byte_mismatch > args.iso_byte_tolerance:
                        raise RuntimeError(
                            f"Iso-byte mismatch {byte_mismatch:.4%} exceeds tolerance"
                        )
                    iso_error_target = min(
                        metrics["normalized_l2"] for metrics in initial_errors.values()
                    )

                    for analysis_type in ("iso_byte", "iso_error"):
                        for layout in LAYOUTS:
                            key = (
                                entry["prompt_id"],
                                checkpoint_step,
                                shard_count,
                                layout,
                                analysis_type,
                            )
                            if key in completed:
                                continue
                            encoded = encoded_by_layout[layout]
                            damaged, read_ms, decode_ms = damaged_by_layout[layout]
                            if analysis_type == "iso_byte":
                                candidate = damaged
                                scale = 1.0
                                target: float | None = None
                            else:
                                candidate, scale = correction.calibrate_direction(
                                    source,
                                    damaged - source,
                                    iso_error_target,
                                )
                                target = iso_error_target
                            label = (
                                f"loss_step{checkpoint_step:02d}_k{shard_count}_{layout}_"
                                f"{analysis_type}"
                            )
                            video, metadata, resume_ms = propagation._load_or_run(
                                omni,
                                args,
                                entry,
                                seed,
                                candidate,
                                checkpoint_step,
                                artifact_dir,
                                label,
                            )
                            row = _result_row(
                                args=args,
                                entry=entry,
                                seed=seed,
                                checkpoint_step=checkpoint_step,
                                shard_count=shard_count,
                                layout=layout,
                                analysis_type=analysis_type,
                                source=source,
                                candidate=candidate,
                                encoded=encoded,
                                read_ms=read_ms,
                                decode_ms=decode_ms,
                                resume_ms=resume_ms,
                                video=video,
                                metadata=metadata,
                                exact_video=exact_video,
                                exact_final=exact_final,
                                semantic=semantic,
                                exact_semantic=exact_semantic,
                                iso_error_target=target,
                                iso_error_scale=scale,
                                artifact_dir=artifact_dir,
                                label=label,
                            )
                            rows.append(row)
                            completed.add(key)
                            correction._atomic_csv(rows_path, LOSS_FIELDS, rows)
                            print(
                                f"[error-shaped] step={checkpoint_step} k={shard_count} "
                                f"layout={layout} control={analysis_type} "
                                f"missing={row['missing_payload_bytes']} "
                                f"dynamic={row['temporal_dynamic_quality']:.4f}",
                                flush=True,
                            )
    finally:
        if hasattr(omni, "shutdown"):
            omni.shutdown()

    iso_byte = _comparison_rows(rows, "iso_byte")
    iso_error = _comparison_rows(rows, "iso_error")
    comparison_fields = list(iso_byte[0]) if iso_byte else [
        "analysis_type", "prompt_id", "checkpoint_step", "shard_count", "layout"
    ]
    correction._atomic_csv(output_dir / "iso_byte_comparison.csv", comparison_fields, iso_byte)
    error_fields = list(iso_error[0]) if iso_error else comparison_fields
    correction._atomic_csv(output_dir / "iso_error_comparison.csv", error_fields, iso_error)
    primary_rows = [row for row in rows if row["analysis_type"] == "iso_byte"]
    correction._atomic_csv(output_dir / "loss_fraction_frontier.csv", LOSS_FIELDS, primary_rows)
    overhead_fields = [
        "prompt_id", "checkpoint_step", "shard_count", "layout", "payload_bytes",
        "metadata_bytes", "total_checkpoint_bytes", "encode_prepare_latency_ms",
        "storage_write_latency_ms", "storage_read_latency_ms",
        "decode_reconstruction_latency_ms",
        "layout_prepare_latency_ms", "gather_copy_latency_ms",
        "metadata_prepare_latency_ms", "serialization_throughput_gbps",
        "reassembly_throughput_gbps",
    ]
    overhead = [
        {field: row[field] for field in overhead_fields}
        for row in primary_rows
    ]
    correction._atomic_csv(output_dir / "layout_overhead.csv", overhead_fields, overhead)

    cell_keys = sorted(
        {
            (str(row["prompt_id"]), int(row["checkpoint_step"]), int(row["shard_count"]))
            for row in rows
        }
    )
    signals = []
    for prompt_id, step, shard_count in cell_keys:
        cell = [
            row for row in rows
            if row["prompt_id"] == prompt_id
            and int(row["checkpoint_step"]) == step
            and int(row["shard_count"]) == shard_count
        ]
        signals.append(
            {
                "prompt_id": prompt_id,
                "checkpoint_step": step,
                "shard_count": shard_count,
                "iso_byte": _cell_signal(cell, "iso_byte"),
                "iso_error": _cell_signal(cell, "iso_error"),
            }
        )
    strong = [
        signal
        for signal in signals
        if signal["iso_byte"]["dynamic_advantage"] >= 0.10
        and signal["iso_error"]["dynamic_advantage"] >= 0.05
        and signal["iso_byte"]["spatial_floor"] >= 0.80
    ]
    conditional = [
        signal
        for signal in signals
        if signal["iso_byte"]["dynamic_advantage"] >= 0.03
        and signal["iso_error"]["dynamic_advantage"] >= 0.01
        and signal["iso_byte"]["spatial_floor"] >= 0.70
    ]
    required = 1 if args.stage == "smoke" else math.ceil(len(signals) * 2 / 3)
    if len(strong) >= required:
        judgment = "STRONG GO"
        eligible = {
            "smoke": "loss_fraction",
            "loss_fraction": "progress",
            "progress": "content",
            "content": None,
        }[args.stage]
    elif len(conditional) >= required:
        judgment = "CONDITIONAL GO"
        eligible = {
            "smoke": "loss_fraction",
            "loss_fraction": "progress",
            "progress": "content",
            "content": None,
        }[args.stage]
    else:
        judgment = "NO-GO"
        eligible = None
    def aggregate_signal(analysis_type: str) -> dict[str, float | bool]:
        selected = [signal[analysis_type] for signal in signals]
        if not selected:
            return _cell_signal([], analysis_type)
        return {
            "complete": all(bool(signal["complete"]) for signal in selected),
            "dynamic_advantage": min(
                float(signal["dynamic_advantage"]) for signal in selected
            ),
            "spatial_floor": min(float(signal["spatial_floor"]) for signal in selected),
            "random_vs_interleaved_dynamic_gap": max(
                float(signal["random_vs_interleaved_dynamic_gap"])
                for signal in selected
            ),
        }

    aggregate_iso_byte = aggregate_signal("iso_byte")
    aggregate_iso_error = aggregate_signal("iso_error")
    mse_values = np.asarray(
        [float(row["initial_latent_mse"]) for row in primary_rows], dtype=np.float64
    )
    dynamic_values = np.asarray(
        [float(row["temporal_dynamic_quality"]) for row in primary_rows], dtype=np.float64
    )
    mse_dynamic_correlation = (
        float(np.corrcoef(mse_values, dynamic_values)[0, 1])
        if len(mse_values) >= 2
        and float(np.std(mse_values)) > 0.0
        and float(np.std(dynamic_values)) > 0.0
        else float("nan")
    )
    summary = {
        "stage": args.stage,
        "prompt_ids": [entry["prompt_id"] for entry in entries],
        "checkpoint_steps": checkpoint_steps,
        "shard_counts": shard_counts,
        "max_exact_resume_error": max(exact_errors, default=float("nan")),
        "recovery_row_count": len(rows),
        "cell_signals": signals,
        "iso_byte_signal": aggregate_iso_byte,
        "iso_error_signal": aggregate_iso_error,
        "strong_cell_count": len(strong),
        "conditional_cell_count": len(conditional),
        "required_cell_count": required,
        "initial_mse_vs_temporal_quality_pearson": mse_dynamic_correlation,
        "maximum_encode_prepare_latency_ms": max(
            (float(row["encode_prepare_latency_ms"]) for row in primary_rows),
            default=float("nan"),
        ),
        "maximum_read_plus_decode_latency_ms": max(
            (
                float(row["storage_read_latency_ms"])
                + float(row["decode_reconstruction_latency_ms"])
                for row in primary_rows
            ),
            default=float("nan"),
        ),
        "judgment": judgment,
        "eligible_next_stage": eligible,
        "figures": _plot(output_dir, rows),
    }
    correction._atomic_json(output_dir / "error_shaped_checkpoint_summary.json", summary)
    _write_report(output_dir / "video_error_shaped_checkpoint_killtest.md", summary)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
