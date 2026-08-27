#!/usr/bin/env python3
"""Oracle kill test for propagation-aware Wan checkpoint encoding.

This experiment quantizes only a serialized checkpoint latent.  The oracle
may use measured future propagation sensitivity; no online policy is built.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight
from experiments import video_denoising_error_correction_killtest as correction
from experiments import video_state_protection_killtest as protection


MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
BITS = (2, 4, 8, 16)
BUDGET_FRACTIONS = (0.0625, 0.125, 0.25, 0.5)
QUALITY_TARGETS = (0.95, 0.975, 0.99)
SENSITIVITY_OBJECTIVES = (
    "final_latent_amplification",
    "spatial_quality_drop",
    "temporal_dynamic_quality_drop",
    "semantic_quality_drop",
)

SENSITIVITY_FIELDS = [
    "stage", "prompt_id", "motion_category", "seed", "checkpoint_step",
    "group_family", "group_configuration", "group_id", "group_elements",
    "target_mse", "target_normalized_l2", "initial_mse", "initial_normalized_l2",
    "relative_mse_mismatch", "relative_error_mismatch", "final_latent_mse",
    "final_latent_normalized_l2",
    "final_latent_amplification", "spatial_quality", "spatial_quality_drop",
    "temporal_dynamic_quality", "temporal_dynamic_quality_drop",
    "semantic_quality", "semantic_quality_relative", "semantic_quality_drop",
    "video_mse", "artifact_path", "trajectory_metadata_path",
]

SENSITIVITY_SUMMARY_FIELDS = [
    "prompt_id", "checkpoint_step", "group_family", "group_configuration",
    "metric", "rank", "group_id", "sensitivity", "cumulative_fraction",
]

ENCODING_FIELDS = [
    "stage", "prompt_id", "seed", "checkpoint_step", "encoder", "objective",
    "budget_fraction", "budget_bytes", "allocation", "payload_bytes", "metadata_bytes",
    "total_checkpoint_bytes", "full_checkpoint_bytes", "byte_fraction_vs_full",
    "compression_ratio_vs_full", "encode_latency_ms", "decode_latency_ms",
    "checkpoint_latent_mse", "checkpoint_latent_normalized_l2",
    "final_latent_mse", "final_latent_normalized_l2", "final_error_amplification",
    "spatial_quality", "temporal_dynamic_quality", "semantic_quality",
    "semantic_quality_relative", "joint_quality", "video_mse",
    "payload_sha256", "metadata_sha256", "artifact_dir",
    "recovered_video_path", "trajectory_metadata_path",
]

ISO_BYTE_FIELDS = ENCODING_FIELDS + ["frontier_method", "frontier_budget_bytes"]


@dataclass(frozen=True)
class Group:
    group_id: str
    indices: torch.Tensor


@dataclass(frozen=True)
class EncodedCheckpoint:
    restored: torch.Tensor
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    encode_ms: float
    decode_ms: float
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
        default="results/video_propagation_aware_checkpoint_killtest",
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
    parser.add_argument("--sample-solver", choices=("euler",), default="euler")
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--channel-group-counts", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--coder-channel-groups", type=int, default=8)
    parser.add_argument("--probe-error-tolerance", type=float, default=0.05)
    parser.add_argument("--exact-resume-tolerance", type=float, default=1e-6)
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
    }
    actual = {key: getattr(args, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Kill-test configuration changed: {actual} != {expected}")
    if args.checkpoint_step not in {10, 20, 30}:
        raise ValueError("--checkpoint-step must be 10, 20, or 30")
    if args.stage == "smoke" and args.checkpoint_step != 20:
        raise ValueError("Smoke is preregistered at checkpoint step 20")
    if args.coder_channel_groups not in args.channel_group_counts:
        raise ValueError("Coder group count must be included in channel probe group counts")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid_recovery_artifacts(
    video_path: Path,
    metadata_path: Path,
    remaining_steps: int,
) -> bool:
    if not video_path.exists() or not metadata_path.exists():
        return False
    try:
        video = np.load(video_path, allow_pickle=False, mmap_mode="r")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = correction._metadata_records(metadata)
        return video.size > 0 and set(records) == {0, remaining_steps} and all(
            Path(record["latent_path"]).exists() for record in records.values()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _valid_sensitivity_row(row: dict[str, str], num_inference_steps: int) -> bool:
    remaining = num_inference_steps - int(row["checkpoint_step"])
    return _valid_recovery_artifacts(
        Path(row["artifact_path"]),
        Path(row["trajectory_metadata_path"]),
        remaining,
    )


def _valid_encoding_row(row: dict[str, str], num_inference_steps: int) -> bool:
    artifact_dir = Path(row["artifact_dir"])
    payload_path = artifact_dir / "checkpoint.payload.bin"
    metadata_path = artifact_dir / "checkpoint.metadata.json"
    if not payload_path.exists() or not metadata_path.exists():
        return False
    payload = payload_path.read_bytes()
    metadata = metadata_path.read_bytes()
    if (
        len(payload) != int(row["payload_bytes"])
        or len(metadata) != int(row["metadata_bytes"])
        or _sha256_bytes(payload) != row["payload_sha256"]
        or _sha256_bytes(metadata) != row["metadata_sha256"]
    ):
        return False
    remaining = num_inference_steps - int(row["checkpoint_step"])
    return _valid_recovery_artifacts(
        Path(row["recovered_video_path"]),
        Path(row["trajectory_metadata_path"]),
        remaining,
    )


def _stage_scope(args: argparse.Namespace, entries: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[int]]:
    if args.stage == "smoke":
        return entries[:1], [20]
    if args.stage == "progress":
        return entries[:1], [10, 20, 30]
    categories = {"static_low_motion", "camera_motion", "fast_complex_motion"}
    selected = [entry for entry in entries if entry.get("motion_category") in categories]
    return (selected[:3] if len(selected) >= 3 else entries[:3]), [10, 20, 30]


def _check_stage_gate(args: argparse.Namespace) -> None:
    if args.stage == "smoke":
        return
    if not args.prior_summary:
        raise ValueError(f"--stage {args.stage} requires --prior-summary")
    summary = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))
    expected = "progress" if args.stage == "progress" else "content"
    if summary.get("eligible_next_stage") != expected:
        raise ValueError(f"Prior result does not permit {args.stage}: {summary.get('eligible_next_stage')!r}")


def _group_ranges(size: int, count: int) -> list[tuple[int, int]]:
    if count <= 0 or count > size:
        raise ValueError(f"Invalid group count {count} for dimension {size}")
    boundaries = [round(index * size / count) for index in range(count + 1)]
    return [(boundaries[index], boundaries[index + 1]) for index in range(count)]


def channel_groups(latent: torch.Tensor, count: int) -> list[Group]:
    _, channels, temporal, height, width = latent.shape
    groups = []
    for index, (start, end) in enumerate(_group_ranges(channels, count)):
        coordinates = torch.arange(latent.numel()).reshape(latent.shape)
        groups.append(Group(f"channel_{index:02d}", coordinates[:, start:end].reshape(-1)))
    return groups


def tile_groups(latent: torch.Tensor) -> list[Group]:
    _, _, temporal, height, width = latent.shape
    coordinates = torch.arange(latent.numel()).reshape(latent.shape)
    groups = []
    index = 0
    for t0, t1 in _group_ranges(temporal, 2):
        for h0, h1 in _group_ranges(height, 2):
            for w0, w1 in _group_ranges(width, 2):
                groups.append(
                    Group(
                        f"tile_{index:02d}",
                        coordinates[:, :, t0:t1, h0:h1, w0:w1].reshape(-1),
                    )
                )
                index += 1
    return groups


def _frequency_directions(
    residual: torch.Tensor,
    family: str,
) -> list[tuple[str, torch.Tensor, int]]:
    if family == "temporal_frequency":
        spectrum = torch.fft.rfft(residual, dim=2, norm="ortho")
        bins = spectrum.shape[2]
        output = []
        for index, (start, end) in enumerate(_group_ranges(bins, 3)):
            selected = torch.zeros_like(spectrum)
            selected[:, :, start:end] = spectrum[:, :, start:end]
            direction = torch.fft.irfft(selected, n=residual.shape[2], dim=2, norm="ortho").real
            coefficient_count = int(spectrum[:, :, start:end].numel() * 2)
            output.append(
                (f"temporal_frequency_{index:02d}", direction, coefficient_count)
            )
        return output
    if family != "spatial_frequency":
        raise ValueError(family)
    spectrum = torch.fft.rfft2(residual, dim=(-2, -1), norm="ortho")
    fy = torch.fft.fftfreq(residual.shape[-2]).abs().reshape(-1, 1)
    fx = torch.fft.rfftfreq(residual.shape[-1]).abs().reshape(1, -1)
    radius = torch.sqrt(fy.square() + fx.square())
    maximum = float(radius.max().item())
    output = []
    for index, (low, high) in enumerate(((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.000001))):
        mask = ((radius / max(maximum, 1e-12) >= low) & (radius / max(maximum, 1e-12) < high))
        selected = spectrum * mask.to(spectrum.dtype)
        direction = torch.fft.irfft2(selected, s=residual.shape[-2:], dim=(-2, -1), norm="ortho").real
        coefficient_count = int(
            torch.count_nonzero(mask).item()
            * residual.shape[0]
            * residual.shape[1]
            * residual.shape[2]
            * 2
        )
        output.append(
            (f"spatial_frequency_{index:02d}", direction, coefficient_count)
        )
    return output


def build_probe_directions(
    latent: torch.Tensor,
    channel_counts: Iterable[int],
) -> list[tuple[str, str, str, int, torch.Tensor]]:
    source = latent.detach().float().cpu().contiguous()
    residual = correction.symmetric_quantize_dequantize(source, 8) - source
    flat_residual = residual.reshape(-1)
    directions = []
    for count in channel_counts:
        for group in channel_groups(source, count):
            direction = torch.zeros_like(source).reshape(-1)
            direction[group.indices] = flat_residual[group.indices]
            directions.append(
                (
                    "channel",
                    f"channel_{count}",
                    group.group_id,
                    group.indices.numel(),
                    direction.reshape_as(source),
                )
            )
    for group in tile_groups(source):
        direction = torch.zeros_like(source).reshape(-1)
        direction[group.indices] = flat_residual[group.indices]
        directions.append(("tile", "tile_2x2x2", group.group_id, group.indices.numel(), direction.reshape_as(source)))
    for family in ("temporal_frequency", "spatial_frequency"):
        for group_id, direction, elements in _frequency_directions(residual, family):
            directions.append((family, f"{family}_3band", group_id, elements, direction))
    return directions


def _sampling(
    args: argparse.Namespace,
    *,
    seed: int,
    latent: torch.Tensor | None,
    checkpoint_step: int,
    artifact_dir: Path,
    label: str,
    baseline_capture_steps: list[int] | None = None,
) -> Any:
    if latent is None:
        sampling = protection._build_probe_sampling_params(
            args,
            seed=seed,
            artifact_dir=artifact_dir,
            request_label=label,
            checkpoint_steps=baseline_capture_steps or [checkpoint_step, args.num_inference_steps],
        )
    else:
        sampling = protection._build_resume_sampling_params(
            args,
            seed=seed,
            checkpoint_step=checkpoint_step,
            latents=latent,
        )
        remaining = args.num_inference_steps - checkpoint_step
        sampling.extra_args = {
            "flow_shift": args.flow_shift,
            "sample_solver": "euler",
            "trajectory_probe": {
                "artifact_dir": str(artifact_dir),
                "request_label": label,
                "capture_steps": [0, remaining],
                "fps": args.fps,
                "save_decoded": False,
                "save_latents": True,
                "save_mp4": False,
            },
        }
    sampling.extra_args = dict(sampling.extra_args or {})
    sampling.extra_args["sample_solver"] = "euler"
    return sampling


def _run(
    omni: Any,
    args: argparse.Namespace,
    entry: dict[str, str],
    seed: int,
    latent: torch.Tensor | None,
    checkpoint_step: int,
    artifact_dir: Path,
    label: str,
    baseline_capture_steps: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any], float]:
    sampling = _sampling(
        args,
        seed=seed,
        latent=latent,
        checkpoint_step=checkpoint_step,
        artifact_dir=artifact_dir,
        label=label,
        baseline_capture_steps=baseline_capture_steps,
    )
    started = time.perf_counter()
    outputs = omni.generate({"prompt": entry["prompt"]}, sampling)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    video, output = protection._normalize_output_video(outputs)
    metadata_path = output.custom_output.get("trajectory_probe_metadata_path")
    if not metadata_path:
        raise ValueError(f"Missing trajectory metadata for {label}")
    return video, json.loads(Path(metadata_path).read_text(encoding="utf-8")), elapsed_ms


def _load_or_run(
    omni: Any,
    args: argparse.Namespace,
    entry: dict[str, str],
    seed: int,
    latent: torch.Tensor | None,
    checkpoint_step: int,
    artifact_dir: Path,
    label: str,
    baseline_capture_steps: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any], float]:
    remaining = args.num_inference_steps if latent is None else args.num_inference_steps - checkpoint_step
    expected = baseline_capture_steps if latent is None else [0, remaining]
    metadata_path = artifact_dir / f"{label}_trajectory_probe.json"
    video_path = artifact_dir / f"{label}.npy"
    cached = correction._valid_probe(metadata_path, remaining) if expected == list(range(remaining + 1)) else None
    if args.resume and cached is None and metadata_path.exists() and video_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = correction._metadata_records(metadata)
            if set(records) == set(expected or []):
                cached = metadata
        except (OSError, ValueError, json.JSONDecodeError):
            cached = None
    if cached is not None and video_path.exists():
        return np.load(video_path, allow_pickle=False), cached, 0.0
    video, metadata, elapsed = _run(
        omni, args, entry, seed, latent, checkpoint_step, artifact_dir, label, baseline_capture_steps
    )
    np.save(video_path, video, allow_pickle=False)
    return video, metadata, elapsed


def _final_latent(metadata: dict[str, Any]) -> torch.Tensor:
    records = correction._metadata_records(metadata)
    final = records[max(records)]
    return torch.load(final["latent_path"], map_location="cpu")


def _quality(
    video: np.ndarray,
    exact_video: np.ndarray,
    prompt: str,
    semantic: Any | None,
    exact_semantic: float,
) -> dict[str, float]:
    return correction._quality_metrics(video, exact_video, prompt, semantic, exact_semantic)


def _calibrated_probe(
    source: torch.Tensor,
    direction: torch.Tensor,
    target_normalized_l2: float,
    target_mse: float,
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    candidate, _ = correction.calibrate_direction(
        source,
        direction,
        target_normalized_l2,
    )
    metrics = correction.latent_error(source, candidate)
    error_mismatch = abs(metrics["normalized_l2"] - target_normalized_l2) / max(
        target_normalized_l2,
        1e-12,
    )
    mse_mismatch = abs(metrics["mse"] - target_mse) / max(target_mse, 1e-12)
    if max(error_mismatch, mse_mismatch) > tolerance:
        raise ValueError(
            f"Probe mismatch exceeds {tolerance:.3%}: "
            f"normalized-L2={error_mismatch:.3%}, MSE={mse_mismatch:.3%}"
        )
    metrics["relative_error_mismatch"] = error_mismatch
    metrics["relative_mse_mismatch"] = mse_mismatch
    return candidate, metrics


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    maximum = (1 << bits) - 1
    if np.any(values > maximum):
        raise ValueError(f"Value exceeds {bits}-bit range")
    if bits == 8:
        return values.tobytes()
    per_byte = 8 // bits
    padded = np.pad(values, (0, (-len(values)) % per_byte))
    packed = np.zeros(len(padded) // per_byte, dtype=np.uint8)
    for offset in range(per_byte):
        packed |= padded[offset::per_byte] << (offset * bits)
    return packed.tobytes()


def _unpack_unsigned(data: bytes, bits: int, count: int) -> np.ndarray:
    packed = np.frombuffer(data, dtype=np.uint8)
    if bits == 8:
        return packed[:count].copy()
    per_byte = 8 // bits
    output = np.empty(len(packed) * per_byte, dtype=np.uint8)
    mask = (1 << bits) - 1
    for offset in range(per_byte):
        output[offset::per_byte] = (packed >> (offset * bits)) & mask
    return output[:count]


def _quantize_values(values: torch.Tensor, bits: int) -> tuple[np.ndarray, float, torch.Tensor]:
    values = values.detach().float().cpu().reshape(-1)
    if bits == 16:
        array = values.to(torch.float16).numpy()
        return array.view(np.uint8), 1.0, torch.from_numpy(array.copy()).float()
    qmax = (1 << (bits - 1)) - 1
    maximum = float(values.abs().max().item())
    scale = maximum / qmax if maximum else 1.0
    quantized = torch.round(values / scale).clamp(-qmax, qmax).to(torch.int16)
    unsigned = (quantized + qmax).numpy().astype(np.uint8)
    restored = quantized.float() * scale
    return unsigned, scale, restored


def _metadata_for_allocation(
    shape: tuple[int, ...],
    groups: list[Group],
    allocation: tuple[int, ...],
    scales: list[float],
) -> dict[str, Any]:
    records = []
    offset = 0
    for group, bits, scale in zip(groups, allocation, scales, strict=True):
        count = int(group.indices.numel())
        nbytes = count * 2 if bits == 16 else math.ceil(count * bits / 8)
        records.append(
            {
                "group_id": group.group_id,
                "bits": bits,
                "count": count,
                "offset": offset,
                "nbytes": nbytes,
                "scale": float(scale),
            }
        )
        offset += nbytes
    return {
        "format": "grouped_symmetric_v1",
        "partition": "equal_channel",
        "shape": list(shape),
        "group_count": len(groups),
        "groups": records,
    }


def _encode_grouped(
    latent: torch.Tensor,
    groups: list[Group],
    allocation: tuple[int, ...],
    artifact_dir: Path,
) -> EncodedCheckpoint:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = latent.detach().float().cpu().contiguous()
    flat = source.reshape(-1)
    payload_parts = []
    scales = []
    started = time.perf_counter()
    for group, bits in zip(groups, allocation, strict=True):
        unsigned, scale, _ = _quantize_values(flat[group.indices], bits)
        payload_parts.append(unsigned.tobytes() if bits == 16 else _pack_unsigned(unsigned, bits))
        scales.append(scale)
    metadata = _metadata_for_allocation(tuple(source.shape), groups, allocation, scales)
    metadata_raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"".join(payload_parts)
    encode_ms = (time.perf_counter() - started) * 1000.0
    _atomic_bytes(artifact_dir / "checkpoint.payload.bin", payload)
    _atomic_bytes(artifact_dir / "checkpoint.metadata.json", metadata_raw)

    started = time.perf_counter()
    restored_flat = torch.empty_like(flat)
    decoded_groups = channel_groups(source, int(metadata["group_count"]))
    for record, group in zip(metadata["groups"], decoded_groups, strict=True):
        chunk = payload[record["offset"] : record["offset"] + record["nbytes"]]
        count = int(record["count"])
        bits = int(record["bits"])
        if bits == 16:
            restored_values = torch.from_numpy(np.frombuffer(chunk, dtype=np.float16).copy()).float()
        else:
            qmax = (1 << (bits - 1)) - 1
            unsigned = _unpack_unsigned(chunk, bits, count).astype(np.int16)
            restored_values = torch.from_numpy(unsigned - qmax).float() * float(record["scale"])
        restored_flat[group.indices] = restored_values
    restored = restored_flat.reshape(source.shape).contiguous()
    decode_ms = (time.perf_counter() - started) * 1000.0
    return EncodedCheckpoint(
        restored=restored,
        payload_bytes=len(payload),
        metadata_bytes=len(metadata_raw),
        total_bytes=len(payload) + len(metadata_raw),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        payload_sha256=_sha256_bytes(payload),
        metadata_sha256=_sha256_bytes(metadata_raw),
        artifact_dir=artifact_dir,
    )


def _allocation_size(latent: torch.Tensor, groups: list[Group], allocation: tuple[int, ...]) -> int:
    if all(bits == 16 for bits in allocation):
        metadata = json.dumps(
            {"format": "fp16", "shape": list(latent.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return latent.numel() * 2 + len(metadata)
    flat = latent.detach().float().cpu().reshape(-1)
    scales = []
    payload = 0
    for group, bits in zip(groups, allocation, strict=True):
        _, scale, _ = _quantize_values(flat[group.indices], bits)
        scales.append(scale)
        payload += group.indices.numel() * 2 if bits == 16 else math.ceil(group.indices.numel() * bits / 8)
    metadata = _metadata_for_allocation(tuple(latent.shape), groups, allocation, scales)
    metadata_size = len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return int(payload + metadata_size)


def _allocation_options(
    latent: torch.Tensor,
    groups: list[Group],
) -> tuple[list[dict[int, float]], list[dict[int, int]], list[dict[int, float]]]:
    flat = latent.detach().float().cpu().reshape(-1)
    errors = []
    payload_sizes = []
    scales = []
    for group in groups:
        source = flat[group.indices]
        group_errors = {}
        group_sizes = {}
        group_scales = {}
        for bits in BITS:
            _, scale, restored = _quantize_values(source, bits)
            group_errors[bits] = float(torch.sum((restored - source).square()).item())
            group_sizes[bits] = source.numel() * 2 if bits == 16 else math.ceil(source.numel() * bits / 8)
            group_scales[bits] = scale
        errors.append(group_errors)
        payload_sizes.append(group_sizes)
        scales.append(group_scales)
    return errors, payload_sizes, scales


def _group_quantization_errors(latent: torch.Tensor, groups: list[Group]) -> list[dict[int, float]]:
    return _allocation_options(latent, groups)[0]


def choose_allocation(
    latent: torch.Tensor,
    groups: list[Group],
    budget_bytes: int,
    group_errors: list[dict[int, float]],
    sensitivities: list[float],
    *,
    uniform_if_equal: bool,
) -> tuple[int, ...] | None:
    if len(groups) != len(sensitivities):
        raise ValueError("Group/sensitivity length mismatch")
    _, payload_sizes, scales = _allocation_options(latent, groups)

    def allocation_size(allocation: tuple[int, ...]) -> int:
        if all(bits == 16 for bits in allocation):
            return _allocation_size(latent, groups, allocation)
        payload = sum(payload_sizes[index][bits] for index, bits in enumerate(allocation))
        selected_scales = [scales[index][bits] for index, bits in enumerate(allocation)]
        metadata = _metadata_for_allocation(tuple(latent.shape), groups, allocation, selected_scales)
        metadata_size = len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return payload + metadata_size

    if uniform_if_equal and max(sensitivities) - min(sensitivities) <= 1e-12:
        feasible = [
            (bits,) * len(groups)
            for bits in BITS
            if allocation_size((bits,) * len(groups)) <= budget_bytes
        ]
        return max(feasible, key=lambda item: item[0]) if feasible else None
    best = None
    best_score = math.inf
    for allocation in itertools.product(BITS, repeat=len(groups)):
        if allocation_size(allocation) > budget_bytes:
            continue
        score = sum(
            group_errors[index][bits] * max(float(sensitivities[index]), 1e-12)
            for index, bits in enumerate(allocation)
        )
        if score < best_score - 1e-18 or (abs(score - best_score) <= 1e-18 and (best is None or allocation > best)):
            best = allocation
            best_score = score
    return best


def _encode_full(latent: torch.Tensor, artifact_dir: Path) -> EncodedCheckpoint:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = latent.detach().float().cpu().contiguous()
    started = time.perf_counter()
    payload = source.numpy().tobytes()
    metadata_raw = json.dumps(
        {"format": "fp32", "shape": list(source.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encode_ms = (time.perf_counter() - started) * 1000.0
    _atomic_bytes(artifact_dir / "checkpoint.payload.bin", payload)
    _atomic_bytes(artifact_dir / "checkpoint.metadata.json", metadata_raw)
    started = time.perf_counter()
    restored = torch.from_numpy(np.frombuffer(payload, dtype=np.float32).copy()).reshape(source.shape)
    decode_ms = (time.perf_counter() - started) * 1000.0
    return EncodedCheckpoint(
        restored=restored,
        payload_bytes=len(payload),
        metadata_bytes=len(metadata_raw),
        total_bytes=len(payload) + len(metadata_raw),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        payload_sha256=_sha256_bytes(payload),
        metadata_sha256=_sha256_bytes(metadata_raw),
        artifact_dir=artifact_dir,
    )


def _encode_fp16(latent: torch.Tensor, artifact_dir: Path) -> EncodedCheckpoint:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = latent.detach().float().cpu().contiguous()
    started = time.perf_counter()
    payload = source.to(torch.float16).numpy().tobytes()
    metadata_raw = json.dumps(
        {"format": "fp16", "shape": list(source.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encode_ms = (time.perf_counter() - started) * 1000.0
    _atomic_bytes(artifact_dir / "checkpoint.payload.bin", payload)
    _atomic_bytes(artifact_dir / "checkpoint.metadata.json", metadata_raw)
    started = time.perf_counter()
    restored = torch.from_numpy(np.frombuffer(payload, dtype=np.float16).copy()).float()
    restored = restored.reshape(source.shape)
    decode_ms = (time.perf_counter() - started) * 1000.0
    return EncodedCheckpoint(
        restored=restored,
        payload_bytes=len(payload),
        metadata_bytes=len(metadata_raw),
        total_bytes=len(payload) + len(metadata_raw),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        payload_sha256=_sha256_bytes(payload),
        metadata_sha256=_sha256_bytes(metadata_raw),
        artifact_dir=artifact_dir,
    )


def _summarize_sensitivity(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    concentration: dict[str, Any] = {}
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["prompt_id"]),
                int(row["checkpoint_step"]),
                str(row["group_family"]),
                str(row["group_configuration"]),
            ),
            [],
        ).append(row)
    for (prompt_id, checkpoint_step, family, configuration), selected in sorted(grouped.items()):
        concentration_key = f"{prompt_id}|step{checkpoint_step}|{configuration}"
        concentration[concentration_key] = {}
        for metric in SENSITIVITY_OBJECTIVES:
            ranked_rows = sorted(selected, key=lambda row: max(float(row[metric]), 0.0), reverse=True)
            ranked = [max(float(row[metric]), 0.0) for row in ranked_rows]
            total = sum(ranked)
            fractions = {}
            for fraction in (0.10, 0.25, 0.50):
                count = max(1, math.ceil(len(ranked) * fraction))
                fractions[f"top_{int(fraction * 100)}pct"] = sum(ranked[:count]) / total if total else 0.0
            concentration[concentration_key][metric] = fractions
            cumulative = 0.0
            for rank, (row, value) in enumerate(zip(ranked_rows, ranked, strict=True), start=1):
                cumulative += value
                summaries.append(
                    {
                        "prompt_id": prompt_id,
                        "checkpoint_step": checkpoint_step,
                        "group_family": family,
                        "group_configuration": configuration,
                        "metric": metric,
                        "rank": rank,
                        "group_id": row["group_id"],
                        "sensitivity": value,
                        "cumulative_fraction": cumulative / total if total else 0.0,
                    }
                )
    return summaries, concentration


def build_iso_byte_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    cells = sorted({(str(row["prompt_id"]), int(row["checkpoint_step"])) for row in rows})
    methods = {
        "uniform_quantization": {"full", "fp16", "uniform_int8", "uniform_int4", "uniform_int2"},
        "mse_aware_mixed": {"mse_aware_mixed"},
        "propagation_oracle": {"propagation_oracle"},
    }
    for prompt_id, checkpoint_step in cells:
        cell = [
            row for row in rows
            if row["prompt_id"] == prompt_id and int(row["checkpoint_step"]) == checkpoint_step
        ]
        ceilings = sorted(
            {
                int(row["total_checkpoint_bytes"])
                for row in cell
                if row["encoder"] in methods["uniform_quantization"]
            }
        )
        for method, encoders in methods.items():
            previous_quality = -math.inf
            for ceiling in ceilings:
                feasible = [
                    row for row in cell
                    if row["encoder"] in encoders and int(row["total_checkpoint_bytes"]) <= ceiling
                ]
                if not feasible:
                    continue
                best = max(
                    feasible,
                    key=lambda row: (
                        float(row["joint_quality"]),
                        -int(row["total_checkpoint_bytes"]),
                    ),
                )
                quality = float(best["joint_quality"])
                if quality + 1e-12 < previous_quality:
                    raise AssertionError("Iso-byte frontier quality decreased with a larger budget")
                previous_quality = max(previous_quality, quality)
                output.append(
                    {
                        **best,
                        "frontier_method": method,
                        "frontier_budget_bytes": ceiling,
                    }
                )
    return output


def _plot(output_dir: Path, sensitivity_rows: list[dict[str, Any]], encoding_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    channel = [row for row in sensitivity_rows if row["group_configuration"] == "channel_8"]
    if channel:
        values = sorted((float(row["temporal_dynamic_quality_drop"]) for row in channel), reverse=True)
        figure, axis = plt.subplots(figsize=(6.5, 4.0))
        axis.plot(range(1, len(values) + 1), values, marker="o")
        axis.set(xlabel="Channel-group sensitivity rank", ylabel="Temporal quality drop")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / "sensitivity_rank_curve.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path))
    if encoding_rows:
        figure, axis = plt.subplots(figsize=(7.0, 4.3))
        for encoder in sorted({str(row["encoder"]) for row in encoding_rows}):
            selected = sorted(
                (row for row in encoding_rows if row["encoder"] == encoder),
                key=lambda row: float(row["byte_fraction_vs_full"]),
            )
            axis.plot(
                [float(row["byte_fraction_vs_full"]) for row in selected],
                [float(row["joint_quality"]) for row in selected],
                marker="o",
                label=encoder,
            )
        axis.set(xlabel="Checkpoint bytes / FP32 bytes", ylabel="Joint recovery quality")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        path = output_dir / "iso_byte_quality_frontier.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path))
        figure, axis = plt.subplots(figsize=(6.5, 4.2))
        axis.scatter(
            [float(row["checkpoint_latent_mse"]) for row in encoding_rows],
            [float(row["joint_quality"]) for row in encoding_rows],
            alpha=0.7,
        )
        axis.set(xlabel="Checkpoint latent MSE", ylabel="Joint recovery quality")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / "checkpoint_mse_vs_final_quality.pdf"
        figure.savefig(path)
        plt.close(figure)
        paths.append(str(path))
        oracle = [row for row in encoding_rows if row["encoder"] == "propagation_oracle"]
        if oracle:
            allocations = np.asarray(
                [[int(value) for value in str(row["allocation"]).split("-")] for row in oracle]
            )
            figure, axis = plt.subplots(figsize=(7.0, max(3.0, len(oracle) * 0.22)))
            image = axis.imshow(allocations, aspect="auto", cmap="viridis", vmin=2, vmax=16)
            axis.set(xlabel="Channel group", ylabel="Oracle candidate")
            figure.colorbar(image, ax=axis, label="Bits")
            figure.tight_layout()
            path = output_dir / "bit_allocation_map.pdf"
            figure.savefig(path)
            plt.close(figure)
            paths.append(str(path))
    return paths


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Propagation-Aware Checkpoint Oracle Kill Test",
        "",
        "## Configuration",
        "",
        f"- Stage: `{summary['stage']}`",
        f"- Solver: `euler`",
        f"- Prompt count: `{summary['prompt_count']}`",
        f"- Checkpoint steps: `{summary['checkpoint_steps']}`",
        "- Exact resume is required before sensitivity probing.",
        "",
        "## CONFIRMED",
        "",
        f"- Exact-resume maximum latent error: `{summary['max_exact_resume_error']:.6g}`.",
        f"- Sensitivity probes completed: `{summary['sensitivity_probe_count']}`.",
        f"- Encoded checkpoints evaluated: `{summary['encoding_count']}`.",
        f"- Best oracle reduction relative to INT8 at a shared target: `{summary['best_oracle_reduction_vs_int8']}`.",
        f"- Maximum top-25% sensitivity concentration: `{summary['maximum_top25_sensitivity_concentration']:.3f}`.",
        f"- Maximum oracle quality gain over MSE-aware coding: "
        f"`{summary['maximum_oracle_joint_quality_gain_vs_mse_aware']:.4f}`.",
        f"- Checkpoint-MSE vs recovery-quality Pearson correlation: "
        f"`{summary['checkpoint_mse_vs_joint_quality_pearson']:.4f}`.",
        "",
        "## INFERRED",
        "",
        "- Propagation-aware headroom is inferred only from the non-deployable future-aware oracle.",
        "- Group sensitivity is not yet an online-predictable signal.",
        "",
        "## UNKNOWN",
        "",
        "- Ranking stability across seeds remains unknown in the smoke stage.",
        "- Progress and content generality remain gated follow-ups.",
        "",
        "## Research Questions",
        "",
        "- Q1/Q2: See `sensitivity_concentration.json` and the ranked sensitivity table.",
        "- Q3: MSE correlation and the oracle-vs-MSE-aware budget comparison are reported above.",
        "- Q4/Q5: See `quality_target_results` and `int8_quality_matched_results` in the summary JSON.",
        "- Q6: Progress persistence is UNKNOWN until a STRONG GO smoke result permits expansion.",
        "- Q7: Online profiler work is permitted only by a STRONG GO judgment.",
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
    entries, checkpoint_steps = _stage_scope(args, all_entries)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "research_question": "oracle propagation-aware checkpoint coding headroom",
        "stage": args.stage,
        "prompt_set": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "prompt_ids": [entry["prompt_id"] for entry in entries],
        "checkpoint_steps": checkpoint_steps,
        "config": vars(args),
        "quality_targets": QUALITY_TARGETS,
        "budget_fractions": BUDGET_FRACTIONS,
        "budget_definition": "actual serialized uniform INT2/INT4/INT8/FP16 bytes, including metadata",
        "bit_choices": BITS,
        "sensitivity_map_storage": "not stored; oracle profiling cost is intentionally excluded",
        "joint_quality_rule": "minimum of spatial, temporal_dynamic, semantic relative quality",
        "go_thresholds": {
            "strong_oracle_reduction_vs_int8_at_95_quality": 2.0,
            "strong_top25_sensitivity_concentration": 0.60,
            "strong_oracle_joint_quality_gain_vs_mse_aware": 0.02,
            "conditional_oracle_reduction_vs_int8": 1.20,
            "conditional_oracle_joint_quality_gain_vs_mse_aware": 0.005,
            "expanded_stage_required_cell_fraction": "2/3",
        },
    }
    prereg_path = output_dir / "oracle_preregistration.json"
    if prereg_path.exists() and json.loads(prereg_path.read_text(encoding="utf-8")) != preregistration:
        raise ValueError(f"Existing preregistration differs: {prereg_path}")
    correction._atomic_json(prereg_path, preregistration)

    sensitivity_path = output_dir / "sensitivity_probe_rows.csv"
    encoding_path = output_dir / "checkpoint_encoding_rows.csv"
    sensitivity_rows: list[dict[str, Any]] = _read_csv(sensitivity_path) if args.resume else []
    encoding_rows: list[dict[str, Any]] = _read_csv(encoding_path) if args.resume else []
    if args.resume:
        sensitivity_rows = [
            row for row in sensitivity_rows
            if _valid_sensitivity_row(row, args.num_inference_steps)
        ]
        encoding_rows = [
            row for row in encoding_rows
            if _valid_encoding_row(row, args.num_inference_steps)
        ]
        correction._atomic_csv(sensitivity_path, SENSITIVITY_FIELDS, sensitivity_rows)
        correction._atomic_csv(encoding_path, ENCODING_FIELDS, encoding_rows)
    completed_probes = {
        (row["prompt_id"], int(row["checkpoint_step"]), row["group_configuration"], row["group_id"])
        for row in sensitivity_rows
    }
    completed_encodings = {
        (row["prompt_id"], int(row["checkpoint_step"]), row["encoder"], row["objective"], row["allocation"])
        for row in encoding_rows
    }
    semantic = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )
    omni = protection._make_omni(args)
    exact_errors = []
    try:
        for prompt_index, entry in enumerate(entries):
            seed = args.seed + prompt_index
            artifact_dir = output_dir / "artifacts" / f"{entry['prompt_id']}_seed{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            capture_steps = sorted(set(checkpoint_steps + [args.num_inference_steps]))
            baseline_label = f"{entry['prompt_id']}_seed{seed}_uninterrupted_euler"
            baseline_video, baseline_meta, _ = _load_or_run(
                omni, args, entry, seed, None, 0, artifact_dir, baseline_label, capture_steps
            )
            baseline_records = correction._metadata_records(baseline_meta)
            baseline_semantic = semantic.score_video(entry["prompt"], baseline_video) if semantic else float("nan")
            for checkpoint_step in checkpoint_steps:
                source = torch.load(baseline_records[checkpoint_step]["latent_path"], map_location="cpu").float()
                exact_label = f"{entry['prompt_id']}_seed{seed}_step{checkpoint_step:02d}_exact_euler"
                exact_video, exact_meta, _ = _load_or_run(
                    omni, args, entry, seed, source, checkpoint_step, artifact_dir, exact_label
                )
                exact_final = _final_latent(exact_meta)
                baseline_final = torch.load(
                    baseline_records[args.num_inference_steps]["latent_path"],
                    map_location="cpu",
                )
                exact_error = correction.latent_error(baseline_final, exact_final)
                exact_quality = _quality(exact_video, baseline_video, entry["prompt"], semantic, baseline_semantic)
                if (
                    exact_error["normalized_l2"] > args.exact_resume_tolerance
                    or exact_quality["video_mse_vs_uninterrupted"] > args.exact_resume_tolerance
                ):
                    raise RuntimeError(
                        f"Exact Euler resume failed at step {checkpoint_step}: "
                        f"latent={exact_error['normalized_l2']}, "
                        f"video_mse={exact_quality['video_mse_vs_uninterrupted']}"
                    )
                exact_errors.append(exact_error["normalized_l2"])

                uniform_int8 = correction.symmetric_quantize_dequantize(source, 8)
                target_metrics = correction.latent_error(source, uniform_int8)
                target = target_metrics["normalized_l2"]
                directions = build_probe_directions(source, args.channel_group_counts)
                for family, configuration, group_id, elements, direction in directions:
                    key = (entry["prompt_id"], checkpoint_step, configuration, group_id)
                    if key in completed_probes:
                        continue
                    candidate, initial = _calibrated_probe(
                        source,
                        direction,
                        target,
                        target_metrics["mse"],
                        args.probe_error_tolerance,
                    )
                    label = f"probe_step{checkpoint_step:02d}_{configuration}_{group_id}"
                    video, metadata, _ = _load_or_run(
                        omni, args, entry, seed, candidate, checkpoint_step, artifact_dir, label
                    )
                    final_error = correction.latent_error(exact_final, _final_latent(metadata))
                    quality = _quality(video, exact_video, entry["prompt"], semantic, baseline_semantic)
                    row = {
                        "stage": args.stage,
                        "prompt_id": entry["prompt_id"],
                        "motion_category": entry["motion_category"],
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "group_family": family,
                        "group_configuration": configuration,
                        "group_id": group_id,
                        "group_elements": elements,
                        "target_mse": target_metrics["mse"],
                        "target_normalized_l2": target,
                        "initial_mse": initial["mse"],
                        "initial_normalized_l2": initial["normalized_l2"],
                        "relative_mse_mismatch": initial["relative_mse_mismatch"],
                        "relative_error_mismatch": initial["relative_error_mismatch"],
                        "final_latent_mse": final_error["mse"],
                        "final_latent_normalized_l2": final_error["normalized_l2"],
                        "final_latent_amplification": final_error["normalized_l2"]
                        / max(initial["normalized_l2"], 1e-12),
                        "spatial_quality": quality["spatial_quality"],
                        "spatial_quality_drop": max(0.0, 1.0 - quality["spatial_quality"]),
                        "temporal_dynamic_quality": quality["temporal_dynamic_quality"],
                        "temporal_dynamic_quality_drop": max(0.0, 1.0 - quality["temporal_dynamic_quality"]),
                        "semantic_quality": quality["semantic_quality"],
                        "semantic_quality_relative": quality["semantic_quality_vs_uninterrupted"],
                        "semantic_quality_drop": max(0.0, 1.0 - quality["semantic_quality_vs_uninterrupted"]),
                        "video_mse": quality["video_mse_vs_uninterrupted"],
                        "artifact_path": str(artifact_dir / f"{label}.npy"),
                        "trajectory_metadata_path": str(artifact_dir / f"{label}_trajectory_probe.json"),
                    }
                    sensitivity_rows.append(row)
                    completed_probes.add(key)
                    correction._atomic_csv(sensitivity_path, SENSITIVITY_FIELDS, sensitivity_rows)
                    print(
                        f"[propagation-probe] step={checkpoint_step} group={configuration}/{group_id} "
                        f"amp={row['final_latent_amplification']:.3f} "
                        f"dynamic_drop={row['temporal_dynamic_quality_drop']:.4f}",
                        flush=True,
                    )

                selected = [
                    row for row in sensitivity_rows
                    if row["prompt_id"] == entry["prompt_id"]
                    and int(row["checkpoint_step"]) == checkpoint_step
                    and row["group_configuration"] == f"channel_{args.coder_channel_groups}"
                ]
                selected.sort(key=lambda row: row["group_id"])
                groups = channel_groups(source, args.coder_channel_groups)
                if len(selected) != len(groups):
                    raise RuntimeError("Incomplete channel sensitivity map")
                errors = _group_quantization_errors(source, groups)
                full_encoded = _encode_full(
                    source,
                    artifact_dir / f"encoded_step{checkpoint_step:02d}_full",
                )
                fp16_encoded = _encode_fp16(
                    source,
                    artifact_dir / f"encoded_step{checkpoint_step:02d}_fp16",
                )
                full_bytes = full_encoded.total_bytes
                candidates: list[tuple[str, str, float, int, tuple[int, ...]]] = [
                    ("full", "uniform", 1.0, full_bytes, (32,) * len(groups))
                ]
                for bits in BITS:
                    uniform_allocation = (bits,) * len(groups)
                    uniform_bytes = (
                        fp16_encoded.total_bytes
                        if bits == 16
                        else _allocation_size(source, groups, uniform_allocation)
                    )
                    candidates.append(
                        (
                            f"uniform_int{bits}" if bits != 16 else "fp16",
                            "uniform",
                            bits / 32,
                            uniform_bytes,
                            uniform_allocation,
                        )
                    )
                for budget_fraction in BUDGET_FRACTIONS:
                    reference_bits = {0.0625: 2, 0.125: 4, 0.25: 8, 0.5: 16}[budget_fraction]
                    budget = (
                        fp16_encoded.total_bytes
                        if reference_bits == 16
                        else _allocation_size(
                            source,
                            groups,
                            (reference_bits,) * len(groups),
                        )
                    )
                    mse_allocation = choose_allocation(
                        source, groups, budget, errors, [1.0] * len(groups), uniform_if_equal=False
                    )
                    if mse_allocation is not None:
                        candidates.append(
                            (
                                "mse_aware_mixed",
                                "checkpoint_mse",
                                budget_fraction,
                                budget,
                                mse_allocation,
                            )
                        )
                    for objective in SENSITIVITY_OBJECTIVES:
                        values = [float(row[objective]) for row in selected]
                        allocation = choose_allocation(
                            source, groups, budget, errors, values, uniform_if_equal=True
                        )
                        if allocation is not None:
                            candidates.append(("propagation_oracle", objective, budget_fraction, budget, allocation))

                allocation_cache: dict[
                    tuple[int, ...],
                    tuple[
                        EncodedCheckpoint,
                        np.ndarray,
                        dict[str, Any],
                        dict[str, float],
                        dict[str, float],
                    ],
                ] = {}
                for encoder, objective, budget_fraction, budget_bytes, allocation in candidates:
                    allocation_text = "-".join(map(str, allocation))
                    label = f"encoded_step{checkpoint_step:02d}_{allocation_text}"
                    key = (entry["prompt_id"], checkpoint_step, encoder, objective, allocation_text)
                    if key in completed_encodings:
                        continue
                    if allocation not in allocation_cache:
                        if all(bits == 32 for bits in allocation):
                            encoded = full_encoded
                        elif all(bits == 16 for bits in allocation):
                            encoded = fp16_encoded
                        else:
                            encoded_dir = artifact_dir / f"encoded_step{checkpoint_step:02d}_{allocation_text}"
                            encoded = _encode_grouped(source, groups, allocation, encoded_dir)
                        video, metadata, _ = _load_or_run(
                            omni, args, entry, seed, encoded.restored, checkpoint_step, artifact_dir, label
                        )
                        final_error = correction.latent_error(exact_final, _final_latent(metadata))
                        quality = _quality(video, exact_video, entry["prompt"], semantic, baseline_semantic)
                        checkpoint_error = correction.latent_error(source, encoded.restored)
                        allocation_cache[allocation] = (
                            encoded,
                            video,
                            metadata,
                            final_error,
                            {**quality, **checkpoint_error},
                        )
                    encoded, _, _, final_error, metrics = allocation_cache[allocation]
                    if encoder == "full" and (
                        float(metrics["normalized_l2"]) > args.exact_resume_tolerance
                        or float(final_error["normalized_l2"]) > args.exact_resume_tolerance
                        or float(metrics["video_mse_vs_uninterrupted"])
                        > args.exact_resume_tolerance
                    ):
                        raise RuntimeError("Serialized FULL checkpoint did not resume exactly")
                    semantic_relative = float(metrics["semantic_quality_vs_uninterrupted"])
                    joint_components = [float(metrics["spatial_quality"]), float(metrics["temporal_dynamic_quality"])]
                    if math.isfinite(semantic_relative):
                        joint_components.append(semantic_relative)
                    row = {
                        "stage": args.stage,
                        "prompt_id": entry["prompt_id"],
                        "seed": seed,
                        "checkpoint_step": checkpoint_step,
                        "encoder": encoder,
                        "objective": objective,
                        "budget_fraction": budget_fraction,
                        "budget_bytes": budget_bytes,
                        "allocation": allocation_text,
                        "payload_bytes": encoded.payload_bytes,
                        "metadata_bytes": encoded.metadata_bytes,
                        "total_checkpoint_bytes": encoded.total_bytes,
                        "full_checkpoint_bytes": full_bytes,
                        "byte_fraction_vs_full": encoded.total_bytes / full_bytes,
                        "compression_ratio_vs_full": full_bytes / encoded.total_bytes,
                        "encode_latency_ms": encoded.encode_ms,
                        "decode_latency_ms": encoded.decode_ms,
                        "checkpoint_latent_mse": metrics["mse"],
                        "checkpoint_latent_normalized_l2": metrics["normalized_l2"],
                        "final_latent_mse": final_error["mse"],
                        "final_latent_normalized_l2": final_error["normalized_l2"],
                        "final_error_amplification": final_error["normalized_l2"]
                        / max(metrics["normalized_l2"], 1e-12),
                        "spatial_quality": metrics["spatial_quality"],
                        "temporal_dynamic_quality": metrics["temporal_dynamic_quality"],
                        "semantic_quality": metrics["semantic_quality"],
                        "semantic_quality_relative": semantic_relative,
                        "joint_quality": min(joint_components),
                        "video_mse": metrics["video_mse_vs_uninterrupted"],
                        "payload_sha256": encoded.payload_sha256,
                        "metadata_sha256": encoded.metadata_sha256,
                        "artifact_dir": str(encoded.artifact_dir),
                        "recovered_video_path": str(artifact_dir / f"{label}.npy"),
                        "trajectory_metadata_path": str(
                            artifact_dir / f"{label}_trajectory_probe.json"
                        ),
                    }
                    encoding_rows.append(row)
                    completed_encodings.add(key)
                    correction._atomic_csv(encoding_path, ENCODING_FIELDS, encoding_rows)
                    print(
                        f"[propagation-encoding] step={checkpoint_step} encoder={encoder}/{objective} "
                        f"bytes={encoded.total_bytes} joint={row['joint_quality']:.4f}",
                        flush=True,
                    )
    finally:
        if hasattr(omni, "shutdown"):
            omni.shutdown()

    sensitivity_summary, concentration = _summarize_sensitivity(sensitivity_rows)
    correction._atomic_csv(
        output_dir / "sensitivity_summary.csv",
        SENSITIVITY_SUMMARY_FIELDS,
        sensitivity_summary,
    )
    correction._atomic_json(output_dir / "sensitivity_concentration.json", concentration)
    iso_frontier_rows = build_iso_byte_frontier(encoding_rows)
    correction._atomic_csv(output_dir / "iso_byte_frontier.csv", ISO_BYTE_FIELDS, iso_frontier_rows)
    unique_encoding_rows = list(
        {
            (row["prompt_id"], int(row["checkpoint_step"]), row["allocation"]): row
            for row in encoding_rows
        }.values()
    )
    correction._atomic_csv(
        output_dir / "mse_vs_recovery_quality.csv",
        ENCODING_FIELDS,
        unique_encoding_rows,
    )

    target_rows = []
    best_reduction = 0.0
    cells = sorted({(str(row["prompt_id"]), int(row["checkpoint_step"])) for row in encoding_rows})
    for prompt_id, checkpoint_step in cells:
        cell_rows = [
            row for row in encoding_rows
            if row["prompt_id"] == prompt_id and int(row["checkpoint_step"]) == checkpoint_step
        ]
        for target in QUALITY_TARGETS:
            uniform = [
                row for row in cell_rows
                if row["encoder"] == "uniform_int8" and float(row["joint_quality"]) >= target
            ]
            oracle = [
                row for row in cell_rows
                if row["encoder"] == "propagation_oracle" and float(row["joint_quality"]) >= target
            ]
            uniform_bytes = min((int(row["total_checkpoint_bytes"]) for row in uniform), default=None)
            oracle_bytes = min((int(row["total_checkpoint_bytes"]) for row in oracle), default=None)
            reduction = uniform_bytes / oracle_bytes if uniform_bytes and oracle_bytes else None
            if reduction is not None:
                best_reduction = max(best_reduction, reduction)
            target_rows.append(
                {
                    "prompt_id": prompt_id,
                    "checkpoint_step": checkpoint_step,
                    "target": target,
                    "uniform_int8_bytes": uniform_bytes,
                    "propagation_oracle_bytes": oracle_bytes,
                    "oracle_reduction_vs_int8": reduction,
                }
            )
    int8_rows = [row for row in encoding_rows if row["encoder"] == "uniform_int8"]
    quality_matched = []
    for int8_row in int8_rows:
        int8_quality = float(int8_row["joint_quality"])
        oracle = [
            row for row in encoding_rows
            if row["encoder"] == "propagation_oracle"
            and row["prompt_id"] == int8_row["prompt_id"]
            and int(row["checkpoint_step"]) == int(int8_row["checkpoint_step"])
            and float(row["joint_quality"]) >= int8_quality
        ]
        oracle_bytes = min((int(row["total_checkpoint_bytes"]) for row in oracle), default=None)
        reduction = int(int8_row["total_checkpoint_bytes"]) / oracle_bytes if oracle_bytes else None
        high_fidelity_oracle = [
            row for row in oracle
            if float(row["joint_quality"]) >= 0.95
        ]
        high_fidelity_oracle_bytes = min(
            (int(row["total_checkpoint_bytes"]) for row in high_fidelity_oracle),
            default=None,
        )
        high_fidelity_reduction = (
            int(int8_row["total_checkpoint_bytes"]) / high_fidelity_oracle_bytes
            if high_fidelity_oracle_bytes
            else None
        )
        if reduction is not None:
            best_reduction = max(best_reduction, reduction)
        quality_matched.append(
            {
                "prompt_id": int8_row["prompt_id"],
                "checkpoint_step": int(int8_row["checkpoint_step"]),
                "int8_joint_quality": int8_quality,
                "int8_bytes": int(int8_row["total_checkpoint_bytes"]),
                "oracle_bytes_at_int8_quality": oracle_bytes,
                "oracle_reduction_vs_int8": reduction,
                "oracle_bytes_at_95_quality": high_fidelity_oracle_bytes,
                "oracle_95_quality_reduction_vs_int8": high_fidelity_reduction,
            }
        )
    budget_deltas = []
    for prompt_id, checkpoint_step in cells:
        for budget in BUDGET_FRACTIONS:
            mse_rows = [
                row for row in encoding_rows
                if row["encoder"] == "mse_aware_mixed"
                and row["prompt_id"] == prompt_id
                and int(row["checkpoint_step"]) == checkpoint_step
                and float(row["budget_fraction"]) == budget
            ]
            oracle_rows = [
                row for row in encoding_rows
                if row["encoder"] == "propagation_oracle"
                and row["prompt_id"] == prompt_id
                and int(row["checkpoint_step"]) == checkpoint_step
                and float(row["budget_fraction"]) == budget
            ]
            if mse_rows and oracle_rows:
                budget_deltas.append(
                    {
                        "prompt_id": prompt_id,
                        "checkpoint_step": checkpoint_step,
                        "budget_fraction": budget,
                        "mse_aware_joint_quality": max(float(row["joint_quality"]) for row in mse_rows),
                        "oracle_joint_quality": max(float(row["joint_quality"]) for row in oracle_rows),
                        "oracle_minus_mse_aware": max(float(row["joint_quality"]) for row in oracle_rows)
                        - max(float(row["joint_quality"]) for row in mse_rows),
                    }
                )
    max_oracle_mse_delta = max(
        (row["oracle_minus_mse_aware"] for row in budget_deltas),
        default=0.0,
    )
    unique_allocations = {}
    for row in encoding_rows:
        unique_allocations.setdefault(
            (row["prompt_id"], int(row["checkpoint_step"]), row["allocation"]),
            row,
        )
    mse_values = np.asarray([float(row["checkpoint_latent_mse"]) for row in unique_allocations.values()])
    quality_values = np.asarray([float(row["joint_quality"]) for row in unique_allocations.values()])
    mse_quality_correlation = (
        float(np.corrcoef(mse_values, quality_values)[0, 1])
        if len(mse_values) >= 2 and np.std(mse_values) > 0 and np.std(quality_values) > 0
        else float("nan")
    )
    max_top25 = max(
        (
            metric["top_25pct"]
            for key, configuration in concentration.items()
            if key.endswith(f"channel_{args.coder_channel_groups}")
            for metric in configuration.values()
        ),
        default=0.0,
    )
    strong_cells = 0
    conditional_cells = 0
    for prompt_id, checkpoint_step in cells:
        cell_reductions = [
            float(row["oracle_reduction_vs_int8"])
            for row in quality_matched
            if row["prompt_id"] == prompt_id
            and int(row["checkpoint_step"]) == checkpoint_step
            and row["oracle_reduction_vs_int8"] is not None
        ]
        high_fidelity_reductions = [
            float(row["oracle_95_quality_reduction_vs_int8"])
            for row in quality_matched
            if row["prompt_id"] == prompt_id
            and int(row["checkpoint_step"]) == checkpoint_step
            and row["oracle_95_quality_reduction_vs_int8"] is not None
        ]
        cell_deltas = [
            float(row["oracle_minus_mse_aware"])
            for row in budget_deltas
            if row["prompt_id"] == prompt_id and int(row["checkpoint_step"]) == checkpoint_step
        ]
        prefix = f"{prompt_id}|step{checkpoint_step}|"
        cell_top25 = max(
            (
                metric["top_25pct"]
                for key, configuration in concentration.items()
                if key.startswith(prefix)
                and key.endswith(f"channel_{args.coder_channel_groups}")
                for metric in configuration.values()
            ),
            default=0.0,
        )
        cell_reduction = max(cell_reductions, default=0.0)
        high_fidelity_reduction = max(high_fidelity_reductions, default=0.0)
        cell_delta = max(cell_deltas, default=0.0)
        strong_cells += int(
            high_fidelity_reduction >= 2.0
            and cell_top25 >= 0.60
            and cell_delta >= 0.02
        )
        conditional_cells += int(cell_reduction >= 1.20 and cell_delta >= 0.005)
    required_strong_cells = 1 if args.stage == "smoke" else math.ceil(len(cells) * 2 / 3)
    if strong_cells >= required_strong_cells:
        judgment = "STRONG GO"
        eligible = "progress" if args.stage == "smoke" else ("content" if args.stage == "progress" else None)
    elif conditional_cells >= required_strong_cells:
        judgment = "CONDITIONAL GO"
        eligible = None
    else:
        judgment = "NO-GO"
        eligible = None
    figures = _plot(output_dir, sensitivity_rows, encoding_rows)
    summary = {
        "stage": args.stage,
        "prompt_count": len(entries),
        "checkpoint_steps": checkpoint_steps,
        "max_exact_resume_error": max(exact_errors, default=float("nan")),
        "sensitivity_probe_count": len(sensitivity_rows),
        "encoding_count": len(encoding_rows),
        "quality_target_results": target_rows,
        "int8_quality_matched_results": quality_matched,
        "oracle_vs_mse_aware_by_budget": budget_deltas,
        "checkpoint_mse_vs_joint_quality_pearson": mse_quality_correlation,
        "maximum_top25_sensitivity_concentration": max_top25,
        "maximum_oracle_joint_quality_gain_vs_mse_aware": max_oracle_mse_delta,
        "best_oracle_reduction_vs_int8": best_reduction,
        "strong_cells": strong_cells,
        "conditional_cells": conditional_cells,
        "required_strong_cells": required_strong_cells,
        "figure_paths": figures,
        "judgment": judgment,
        "eligible_next_stage": eligible,
    }
    correction._atomic_json(output_dir / "oracle_checkpoint_summary.json", summary)
    _write_report(output_dir / "video_propagation_aware_checkpoint_killtest.md", summary)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
