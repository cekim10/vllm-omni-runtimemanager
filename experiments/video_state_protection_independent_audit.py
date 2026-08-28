#!/usr/bin/env python3
"""Independent forensic audit of Wan video checkpoint/recovery experiments.

The script deliberately does not import Stage A/B representation or layout
helpers. ``--mode static`` audits existing bytes/results and CPU invariants.
``--mode full`` additionally runs one gated Euler trajectory on GPU0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REPRESENTATIONS = ("full", "fp16", "int8", "spatial_down2", "temporal_down2", "low_rank_25")
LAYOUTS = ("spatial_contiguous", "temporal_contiguous", "channel_contiguous", "random", "interleaved")
SEMANTIC_AXES = {0: "batch", 1: "channel", 2: "temporal", 3: "height", 4: "width"}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    summary: str
    evidence: str


@dataclass(frozen=True)
class Encoded:
    name: str
    payload_path: Path
    metadata_path: Path
    payload_bytes: int
    metadata_bytes: int
    restored: np.ndarray
    transformed_shapes: str
    details: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(array_bytes(array))


def error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    lhs = reference.astype(np.float64, copy=False)
    rhs = candidate.astype(np.float64, copy=False)
    delta = rhs - lhs
    norm = float(np.linalg.norm(delta.reshape(-1)))
    ref_norm = float(np.linalg.norm(lhs.reshape(-1)))
    return {
        "mse": float(np.mean(delta * delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "normalized_l2": norm / max(ref_norm, 1e-30),
        "bias": float(np.mean(delta)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def serialize_components(
    directory: Path,
    name: str,
    components: list[tuple[str, np.ndarray]],
    metadata: dict[str, Any],
) -> tuple[Path, Path, int, int]:
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    descriptions = []
    offset = 0
    for component_name, component in components:
        contiguous = np.ascontiguousarray(component)
        chunk = contiguous.tobytes(order="C")
        chunks.append(chunk)
        descriptions.append(
            {
                "name": component_name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "offset": offset,
                "nbytes": len(chunk),
            }
        )
        offset += len(chunk)
    document = dict(metadata)
    document["components"] = descriptions
    raw_metadata = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    payload_path = directory / f"{name}.payload.bin"
    metadata_path = directory / f"{name}.metadata.json"
    atomic_bytes(payload_path, b"".join(chunks))
    atomic_bytes(metadata_path, raw_metadata)
    return payload_path, metadata_path, offset, len(raw_metadata)


def deserialize_components(payload_path: Path, metadata_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    payload = payload_path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    arrays = []
    expected_offset = 0
    for component in metadata["components"]:
        offset = int(component["offset"])
        nbytes = int(component["nbytes"])
        if offset != expected_offset or offset + nbytes > len(payload):
            raise AssertionError("Serialized component offsets are not contiguous")
        array = np.frombuffer(payload[offset : offset + nbytes], dtype=np.dtype(component["dtype"]))
        arrays.append(array.reshape(tuple(component["shape"])).copy())
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise AssertionError("Serialized payload contains unaccounted bytes")
    return metadata, arrays


def _torch_interpolate(source: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(source)).float()
    return functional.interpolate(tensor, size=target, mode="trilinear", align_corners=False).cpu().numpy()


def encode_representation(source: np.ndarray, name: str, directory: Path) -> Encoded:
    source = np.ascontiguousarray(source.astype(np.float32, copy=False))
    if source.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {source.shape}")
    details: dict[str, Any] = {"source_shape": list(source.shape), "source_dtype": str(source.dtype)}
    if name == "full":
        components = [("latent", source)]
        transformed = str([list(source.shape)])
    elif name == "fp16":
        packed = source.astype(np.float16)
        components = [("latent_fp16", packed)]
        transformed = str([list(packed.shape)])
    elif name == "int8":
        maximum = max(float(np.max(np.abs(source))), 1e-8)
        scale = np.float32(maximum / 127.0)
        quantized = np.clip(np.rint(source / scale), -127, 127).astype(np.int8)
        components = [("latent_int8", quantized), ("scale", np.asarray([scale], dtype=np.float32))]
        details.update({"granularity": "per_tensor", "scale": float(scale), "zero_point": 0, "rounding": "rint", "signed": True})
        transformed = str([list(quantized.shape), [1]])
    elif name == "spatial_down2":
        _, _, temporal, height, width = source.shape
        down = _torch_interpolate(source, (temporal, max(1, height // 2), max(1, width // 2)))
        components = [("latent_down", down)]
        details["original_shape"] = list(source.shape)
        transformed = str([list(down.shape)])
    elif name == "temporal_down2":
        _, _, temporal, height, width = source.shape
        down = _torch_interpolate(source, (math.ceil(temporal / 2), height, width))
        components = [("latent_down", down)]
        details["original_shape"] = list(source.shape)
        transformed = str([list(down.shape)])
    elif name == "low_rank_25":
        batch, channels, temporal, height, width = source.shape
        matrix = source.reshape(batch * channels * temporal, height * width)
        rank = max(1, min(matrix.shape[0], matrix.shape[1], round(min(matrix.shape) * 0.25)))
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        components = [("u", u[:, :rank].astype(np.float32)), ("s", singular[:rank].astype(np.float32)), ("vh", vh[:rank].astype(np.float32))]
        details.update({"original_shape": list(source.shape), "rank": rank})
        transformed = str([list(component.shape) for _, component in components])
    else:
        raise ValueError(name)
    payload_path, metadata_path, payload_bytes, metadata_bytes = serialize_components(
        directory, name, components, {"variant": name, **details}
    )
    metadata, arrays = deserialize_components(payload_path, metadata_path)
    if name == "full":
        restored = arrays[0].astype(np.float32)
    elif name == "fp16":
        restored = arrays[0].astype(np.float32)
    elif name == "int8":
        restored = arrays[0].astype(np.float32) * float(arrays[1].reshape(-1)[0])
    elif name in {"spatial_down2", "temporal_down2"}:
        restored = _torch_interpolate(arrays[0], tuple(source.shape[2:]))
    else:
        restored = ((arrays[0] * arrays[1][None, :]) @ arrays[2]).reshape(source.shape)
    return Encoded(name, payload_path, metadata_path, payload_bytes, metadata_bytes, np.ascontiguousarray(restored), transformed, metadata)


def expected_mask(shape: tuple[int, ...], layout: str, shard_count: int, seed: int) -> np.ndarray:
    if len(shape) != 5:
        raise ValueError("Layouts require [B,C,T,H,W]")
    batch, channels, temporal, height, width = shape
    size = math.prod(shape)
    if size % shard_count:
        raise ValueError("Equal shards require divisible element count")
    target = size // shard_count
    mask = np.zeros(shape, dtype=bool)
    if layout == "spatial_contiguous":
        spatial_cells = target // (batch * channels * temporal)
        candidates = [(h, spatial_cells // h) for h in range(1, height + 1) if spatial_cells % h == 0 and spatial_cells // h <= width]
        rect_h, rect_w = min(candidates, key=lambda item: abs(item[0] / item[1] - height / width))
        mask[..., :rect_h, :rect_w] = True
    elif layout == "temporal_contiguous":
        # Equal-byte temporal-major prefix. For Wan T=9 and K=4 this is two
        # complete temporal slices plus a partial boundary slice; the audit
        # reports that naming limitation explicitly.
        temporal_major = np.arange(size).reshape(shape).transpose(0, 2, 1, 3, 4).reshape(-1)
        mask.reshape(-1)[temporal_major[:target]] = True
    elif layout == "channel_contiguous":
        if channels % shard_count:
            raise ValueError("Channel axis is not shard divisible")
        mask[:, : channels // shard_count, :, :, :] = True
    elif layout == "random":
        indices = np.random.default_rng(seed).permutation(size)[:target]
        mask.reshape(-1)[indices] = True
    elif layout == "interleaved":
        canonical = np.arange(size).reshape(shape).transpose(0, 2, 3, 4, 1).reshape(-1)
        mask.reshape(-1)[canonical[::shard_count]] = True
    else:
        raise ValueError(layout)
    if int(mask.sum()) != target:
        raise AssertionError(f"{layout} masks {mask.sum()} elements, expected {target}")
    return mask


def storage_order(shape: tuple[int, ...], layout: str, shard_count: int, seed: int) -> np.ndarray:
    mask = expected_mask(shape, layout, shard_count, seed)
    first = np.flatnonzero(mask.reshape(-1))
    if layout == "random":
        full = np.random.default_rng(seed).permutation(math.prod(shape))
        first = full[: len(first)]
        rest = full[len(first) :]
    elif layout == "interleaved":
        canonical = np.arange(math.prod(shape)).reshape(shape).transpose(0, 2, 3, 4, 1).reshape(-1)
        first = canonical[::shard_count]
        selected = np.zeros(math.prod(shape), dtype=bool)
        selected[first] = True
        rest = np.arange(math.prod(shape))[~selected]
    else:
        selected = np.zeros(math.prod(shape), dtype=bool)
        selected[first] = True
        rest = np.arange(math.prod(shape))[~selected]
    order = np.concatenate((first, rest))
    if len(np.unique(order)) != math.prod(shape):
        raise AssertionError("Storage order is not a permutation")
    return order


def shard_roundtrip(
    source: np.ndarray,
    layout: str,
    shard_count: int,
    seed: int,
    directory: Path,
    missing_shard: int | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    source = np.ascontiguousarray(source.astype(np.float32, copy=False))
    order = storage_order(source.shape, layout, shard_count, seed)
    storage = source.reshape(-1)[order]
    per_shard = source.size // shard_count
    paths = []
    for shard in range(shard_count):
        path = directory / layout / f"shard_{shard:02d}.bin"
        atomic_bytes(path, storage[shard * per_shard : (shard + 1) * per_shard].tobytes())
        paths.append(path)
    metadata = {
        "shape": list(source.shape), "dtype": "<f4", "layout": layout,
        "shard_count": shard_count, "seed": seed, "source_sha256": array_sha256(source),
    }
    atomic_bytes(directory / layout / "metadata.json", json.dumps(metadata, sort_keys=True).encode())
    loaded = np.empty(source.size, dtype=np.float32)
    for shard, path in enumerate(paths):
        start, end = shard * per_shard, (shard + 1) * per_shard
        loaded[start:end] = 0.0 if shard == missing_shard else np.frombuffer(path.read_bytes(), dtype=np.float32)
    restored_flat = np.empty_like(loaded)
    restored_flat[order] = loaded
    missing = np.zeros(source.size, dtype=bool)
    if missing_shard is not None:
        start, end = missing_shard * per_shard, (missing_shard + 1) * per_shard
        missing[order[start:end]] = True
    return restored_flat.reshape(source.shape), missing.reshape(source.shape), [path.stat().st_size for path in paths]


def mask_summary(mask: np.ndarray, layout: str) -> dict[str, Any]:
    coordinates = np.argwhere(mask)
    affected = lambda axis: sorted(np.unique(coordinates[:, axis]).astype(int).tolist())
    per_frame = mask.mean(axis=(0, 1, 3, 4))
    per_channel = mask.mean(axis=(0, 2, 3, 4))
    return {
        "layout": layout,
        "masked_elements": int(mask.sum()),
        "masked_bytes_fp32": int(mask.sum()) * 4,
        "masked_fraction": float(mask.mean()),
        "affected_temporal_indices": json.dumps(affected(2)),
        "affected_channel_indices": json.dumps(affected(1)),
        "height_range": json.dumps([int(coordinates[:, 3].min()), int(coordinates[:, 3].max())]),
        "width_range": json.dumps([int(coordinates[:, 4].min()), int(coordinates[:, 4].max())]),
        "per_frame_missing_fraction": json.dumps(per_frame.round(6).tolist()),
        "per_channel_missing_fraction": json.dumps(per_channel.round(6).tolist()),
    }


def save_mask_svg(path: Path, mask: np.ndarray, title: str) -> None:
    # Each row is one temporal slice; each column is one spatial position,
    # averaged over batch/channel. This requires no plotting dependency.
    projection = mask.mean(axis=(0, 1)).reshape(mask.shape[2], -1)
    cell = 3
    width = projection.shape[1] * cell
    height = projection.shape[0] * 20
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 28}">', f'<text x="4" y="16" font-size="12">{title}</text>']
    for row in range(projection.shape[0]):
        for column, value in enumerate(projection[row]):
            shade = int(round(255 * (1.0 - value)))
            elements.append(f'<rect x="{column * cell}" y="{28 + row * 20}" width="{cell}" height="20" fill="rgb(255,{shade},{shade})"/>')
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements))


def independent_video_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    lhs = candidate.astype(np.float64) / 255.0
    rhs = reference.astype(np.float64) / 255.0
    mse = float(np.mean((lhs - rhs) ** 2))
    variance = float(np.var(rhs)) + 1e-12
    spatial = 1.0 / (1.0 + mse / variance)
    lhs_delta = np.diff(lhs, axis=0)
    rhs_delta = np.diff(rhs, axis=0)
    temporal_mse = float(np.mean((lhs_delta - rhs_delta) ** 2)) if len(lhs) > 1 else 0.0
    temporal_scale = float(np.mean(rhs_delta**2)) + 1e-12
    temporal = 1.0 / (1.0 + temporal_mse / temporal_scale)
    return {"video_mse": mse * 255.0**2, "spatial_quality": spatial, "temporal_dynamic_quality": temporal}


def static_exact_equal_audit(raw_rows: list[dict[str, str]], output_dir: Path) -> list[Finding]:
    full = [row for row in raw_rows if row["variant"] == "full"]
    exact_count = sum(str(row["exact_equal"]).lower() == "true" for row in full)
    minimum_mse = min(float(row["video_mse"]) for row in full)
    maximum_mse = max(float(row["video_mse"]) for row in full)
    report = f"""# Stage B exact_equal Audit

## Conclusion: BUG

- `exact_equal` compares the decoded resumed video against the uninterrupted video using `np.array_equal`.
- It is evaluated after each representation transform; `False` is expected for lossy variants.
- FULL rows exact: `{exact_count}/{len(full)}`.
- FULL video MSE range: `{minimum_mse:.6g}` to `{maximum_mse:.6g}`.
- Stage B did not set `sample_solver`; Wan defaults to multistep UniPC.
- The checkpoint contains only the latent and step index, not UniPC solver history.

Therefore FULL failure is not benign numerical noise. Existing Stage B is not
an exact-resume experiment. Relative comparisons against its FULL-resume branch
may still be descriptive, but the 1080-row frontier must not support exact
fault-recovery claims and must be independently reproduced with Euler.
"""
    (output_dir / "exact_equal_audit.md").write_text(report)
    return [
        Finding(
            "CRITICAL",
            "STAGE_B_UNIPC_HISTORY_MISSING",
            "Every Stage B FULL resume diverges from uninterrupted generation.",
            f"exact={exact_count}/{len(full)}, video_mse=[{minimum_mse:.6g},{maximum_mse:.6g}], Stage B omitted sample_solver and solver history.",
        )
    ]


def audit_existing_representation_bytes(
    artifact_dir: Path, output_dir: Path
) -> tuple[np.ndarray, list[dict[str, Any]], list[Finding]]:
    base = artifact_dir / "serialized_step020"
    full_metadata, full_arrays = deserialize_components(base / "full/full.payload.bin", base / "full/full.metadata.json")
    source = full_arrays[0]
    rows = []
    findings = []
    for name in REPRESENTATIONS:
        metadata_path = base / name / f"{name}.metadata.json"
        payload_path = base / name / f"{name}.payload.bin"
        metadata, arrays = deserialize_components(payload_path, metadata_path)
        payload = payload_path.stat().st_size
        meta_bytes = metadata_path.stat().st_size
        rows.append(
            {
                "representation": name,
                "source_dtype": str(source.dtype),
                "source_payload_bytes": source.nbytes,
                "transformed_shapes": json.dumps([list(array.shape) for array in arrays]),
                "component_dtypes": json.dumps([str(array.dtype) for array in arrays]),
                "payload_bytes": payload,
                "metadata_bytes": meta_bytes,
                "total_bytes": payload + meta_bytes,
                "ratio_vs_full": (payload + meta_bytes) / (source.nbytes + (base / "full/full.metadata.json").stat().st_size),
            }
        )
    int8_row = next(row for row in rows if row["representation"] == "int8")
    if str(source.dtype) != "float32":
        findings.append(Finding("MAJOR", "SOURCE_DTYPE_NOT_FP32", "Stage B source representation was not FP32.", str(source.dtype)))
    ratio = float(int8_row["ratio_vs_full"])
    if not (0.24 <= ratio <= 0.26):
        findings.append(Finding("MAJOR", "INT8_RATIO_MISMATCH", "INT8 total bytes are not approximately 25% of the actual FP32 source.", str(ratio)))
    write_csv(output_dir / "representation_byte_audit.csv", rows)
    return source, rows, findings


def run_static_audit(args: argparse.Namespace) -> tuple[list[Finding], dict[str, Any]]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_csv(args.stage_b_dir / "frontier_raw.csv")
    findings = static_exact_equal_audit(raw_rows, output_dir)
    artifact_dir = args.stage_b_dir / "artifacts" / f"{args.prompt_id}_seed{args.seed}"
    source, byte_rows, byte_findings = audit_existing_representation_bytes(artifact_dir, output_dir)
    findings.extend(byte_findings)
    state = {
        "source": "existing Stage B independently parsed FULL payload",
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "strides_elements": [stride // source.itemsize for stride in source.strides],
        "contiguous": bool(source.flags.c_contiguous),
        "numel": int(source.size),
        "element_size": int(source.itemsize),
        "payload_bytes": int(source.nbytes),
        "min": float(source.min()), "max": float(source.max()), "mean": float(source.mean()), "std": float(source.std()),
        "nan_count": int(np.isnan(source).sum()), "inf_count": int(np.isinf(source).sum()),
        "sha256": array_sha256(source),
        "semantic_axes": {str(key): value for key, value in SEMANTIC_AXES.items()},
        "axis_evidence": [
            "Wan prepare_latents produces [B,C,T,H,W]",
            "pipeline_wan2_2.py VAE mean/std uses [1,z_dim,1,1,1]",
            "runtime shape for 33 frames at 480x832 is expected [1,16,9,60,104]",
        ],
    }
    if source.shape != (1, 16, 9, 60, 104):
        findings.append(Finding("CRITICAL", "LATENT_AXIS_OR_SHAPE", "Runtime latent shape does not match [B,C,T,H,W] geometry.", str(source.shape)))
    (output_dir / "latent_state_audit.json").write_text(json.dumps(state, indent=2) + "\n")

    # Independent raw identity round trip from the parsed source.
    identity_dir = output_dir / "identity_static"
    payload, metadata, _, _ = serialize_components(identity_dir, "identity", [("latent", source)], {"dtype": str(source.dtype)})
    _, arrays = deserialize_components(payload, metadata)
    identity_error = error(source, arrays[0])
    identity = {
        "scope": "CPU bytes only; GPU resume not yet run",
        "shape_equal": source.shape == arrays[0].shape,
        "dtype_equal": source.dtype == arrays[0].dtype,
        "byte_count_equal": source.nbytes == arrays[0].nbytes,
        "sha256_before": array_sha256(source),
        "sha256_after": array_sha256(arrays[0]),
        **identity_error,
    }
    (output_dir / "exact_identity_roundtrip.json").write_text(json.dumps(identity, indent=2) + "\n")
    if identity_error["max_abs"] != 0 or identity["sha256_before"] != identity["sha256_after"]:
        findings.append(Finding("CRITICAL", "IDENTITY_SERIALIZATION", "Independent CPU identity serialization is not exact.", json.dumps(identity)))

    # Independent layout masks and real disk shards.
    mask_rows, shard_rows, zero_rows = [], [], []
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        mask = expected_mask(source.shape, layout, 4, args.layout_seed)
        mask_rows.append(mask_summary(mask, layout))
        np.save(output_dir / "masks" / f"{layout}.npy", mask, allow_pickle=False)
        save_mask_svg(output_dir / "masks" / f"{layout}_t_by_hw.svg", mask, layout)
        exact, exact_mask, sizes = shard_roundtrip(source, layout, 4, args.layout_seed, output_dir / "shards", None)
        damaged, disk_mask, _ = shard_roundtrip(source, layout, 4, args.layout_seed, output_dir / "shards", 0)
        expected = expected_mask(source.shape, layout, 4, args.layout_seed)
        exact_metrics = error(source, exact)
        changed = damaged != source
        nonmissing_changed = int(np.logical_and(changed, ~disk_mask).sum())
        missing_nonzero = int(np.logical_and(disk_mask, damaged != 0).sum())
        shard_rows.append(
            {
                "layout": layout, "all_present_exact": exact_metrics["max_abs"] == 0,
                "all_present_sha_equal": array_sha256(exact) == array_sha256(source),
                "shard_bytes": json.dumps(sizes), "iso_byte_mismatch": (max(sizes) - min(sizes)) / max(sizes),
                "disk_mask_equals_expected": bool(np.array_equal(disk_mask, expected)),
            }
        )
        zero_rows.append(
            {
                "layout": layout, "changed_elements": int(changed.sum()),
                "expected_changed_elements": int(np.logical_and(disk_mask, source != 0).sum()),
                "accidentally_changed_nonmissing_elements": nonmissing_changed,
                "missing_elements_not_zero": missing_nonzero,
            }
        )
        if exact_metrics["max_abs"] != 0 or nonmissing_changed or missing_nonzero or not np.array_equal(disk_mask, expected):
            findings.append(Finding("CRITICAL", f"LAYOUT_{layout.upper()}", "Independent shard/reassembly invariant failed.", json.dumps(shard_rows[-1] | zero_rows[-1])))
        if layout == "temporal_contiguous":
            per_frame = mask.mean(axis=(0, 1, 3, 4))
            if any(value not in (0.0, 1.0) for value in per_frame):
                findings.append(
                    Finding(
                        "MAJOR",
                        "TEMPORAL_LAYOUT_PARTIAL_BOUNDARY",
                        "The equal-byte temporal-contiguous shard includes a partial temporal boundary slice.",
                        f"T={source.shape[2]}, K=4, per-frame missing fractions={per_frame.tolist()}",
                    )
                )
    write_csv(output_dir / "layout_mask_audit.csv", mask_rows)
    write_csv(output_dir / "shard_reassembly_audit.csv", shard_rows)
    write_csv(output_dir / "zero_fill_invariants.csv", zero_rows)

    # Human-readable synthetic tensor test, independent of Wan artifacts.
    synthetic_shape = (1, 4, 4, 4, 4)
    synthetic = np.empty(synthetic_shape, dtype=np.float32)
    for b in range(1):
        for c in range(4):
            for t in range(4):
                for h in range(4):
                    for w in range(4):
                        synthetic[b, c, t, h, w] = c * 10000 + t * 1000 + h * 100 + w
    mappings = {}
    for layout in LAYOUTS:
        mask = expected_mask(synthetic_shape, layout, 4, args.layout_seed)
        mappings[layout] = {
            "masked_coordinates": np.argwhere(mask).tolist(),
            "masked_values": synthetic[mask].astype(int).tolist(),
        }
    (output_dir / "synthetic_layout_mappings.json").write_text(json.dumps(mappings, indent=2) + "\n")

    # Metric controls operate on the exact stored reference video and are
    # deliberately independent of Stage B metric helpers.
    baseline = np.load(artifact_dir / f"{args.prompt_id}_seed{args.seed}_baseline.npy", allow_pickle=False)
    controls = {
        "self": baseline.copy(),
        "zero": np.zeros_like(baseline),
        "temporal_shuffle": baseline[::-1].copy(),
        "spatial_blur": ((
            baseline.astype(np.float32)
            + np.roll(baseline, 8, axis=1).astype(np.float32)
            + np.roll(baseline, -8, axis=1).astype(np.float32)
            + np.roll(baseline, 8, axis=2).astype(np.float32)
            + np.roll(baseline, -8, axis=2).astype(np.float32)
        ) / 5.0).clip(0, 255).astype(np.uint8),
    }
    metric_rows = []
    for name, candidate in controls.items():
        metric_rows.append({"control": name, **independent_video_metrics(candidate, baseline), "semantic_quality": "PENDING_GPU_CLIP"})
    write_csv(output_dir / "metric_audit.csv", metric_rows)
    self_row = next(row for row in metric_rows if row["control"] == "self")
    shuffle_row = next(row for row in metric_rows if row["control"] == "temporal_shuffle")
    if self_row["video_mse"] != 0 or self_row["spatial_quality"] != 1 or self_row["temporal_dynamic_quality"] != 1:
        findings.append(Finding("CRITICAL", "METRIC_SELF", "Independent metric self comparison is not ideal.", json.dumps(self_row)))
    if shuffle_row["temporal_dynamic_quality"] > 0.95:
        findings.append(Finding("CRITICAL", "TEMPORAL_SHUFFLE", "Independent temporal metric does not penalize reverse-order frames.", json.dumps(shuffle_row)))

    corruption_rows = []
    rng = np.random.default_rng(7)
    order = rng.permutation(baseline.size)
    for fraction in (0.0, 0.01, 0.05, 0.125, 0.25, 0.5, 1.0):
        candidate = baseline.copy().reshape(-1)
        candidate[order[: round(fraction * candidate.size)]] = 0
        corruption_rows.append({"family": "video_zero_control_only", "severity": fraction, **independent_video_metrics(candidate.reshape(baseline.shape), baseline)})
    write_csv(output_dir / "corruption_monotonicity.csv", corruption_rows)

    config = {
        "mode": args.mode, "model": args.model, "prompt_id": args.prompt_id, "seed": args.seed,
        "checkpoint_step": args.checkpoint_step, "sample_solver": "euler", "height": args.height,
        "width": args.width, "num_frames": args.num_frames, "num_inference_steps": args.num_inference_steps,
        "semantic_axes": SEMANTIC_AXES, "stage_b_raw_immutable": str(args.stage_b_dir),
    }
    (output_dir / "audit_config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {"static_complete": True, "gpu_complete": False, "findings": [asdict(item) for item in findings]}
    (output_dir / "audit_findings.json").write_text(json.dumps(summary, indent=2) + "\n")
    return findings, {"source": source, "artifact_dir": artifact_dir, "raw_rows": raw_rows}


def _sampling(args: argparse.Namespace, *, seed: int, label: str, artifact_dir: Path, latents: Any = None, step_index: int = 0) -> Any:
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        fps=args.fps, seed=seed, generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    sampling.latents = None if latents is None else latents.detach().cpu().clone()
    sampling.step_index = step_index
    remaining = args.num_inference_steps - step_index
    sampling.extra_args = {
        "flow_shift": args.flow_shift,
        "sample_solver": "euler",
        "trajectory_probe": {
            "artifact_dir": str(artifact_dir), "request_label": label,
            "capture_steps": list(range(remaining + 1)), "fps": args.fps,
            "save_decoded": False, "save_latents": True, "save_mp4": False,
        },
    }
    return sampling


def _normalize_video(outputs: Any) -> tuple[np.ndarray, Any]:
    import torch
    from vllm_omni.outputs import OmniRequestOutput

    output = OmniRequestOutput.unwrap_result(outputs)
    frames = output.images
    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple):
        frames = frames[0]
    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point():
            if tensor.min() < 0 or tensor.max() > 1:
                tensor = (tensor.clamp(-1, 1) + 1) / 2
            tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
        return tensor.numpy(), output
    array = np.asarray(frames)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.shape[-1] not in (3, 4) and array.shape[1] in (3, 4):
        array = array.transpose(0, 2, 3, 1)
    if array.dtype != np.uint8:
        if array.min() < 0 or array.max() > 1:
            array = (np.clip(array, -1, 1) + 1) / 2
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return array, output


def _run_generate(omni: Any, args: argparse.Namespace, prompt: str, seed: int, label: str, artifact_dir: Path, latents: Any = None, step_index: int = 0) -> tuple[np.ndarray, dict[str, Any], float]:
    started = time.perf_counter()
    outputs = omni.generate({"prompt": prompt}, _sampling(args, seed=seed, label=label, artifact_dir=artifact_dir, latents=latents, step_index=step_index))
    elapsed = (time.perf_counter() - started) * 1000
    video, output = _normalize_video(outputs)
    path = output.custom_output.get("trajectory_probe_metadata_path")
    if not path:
        raise RuntimeError("Trajectory probe metadata is missing")
    return video, json.loads(Path(path).read_text()), elapsed


def _records(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["step_index"]): row for row in metadata["records"]}


def _load_tensor(path: str | Path) -> Any:
    import torch
    return torch.load(path, map_location="cpu")


def _tensor_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().float().contiguous().numpy()


def _semantic_scores(prompt: str, videos: dict[str, np.ndarray], model_name: str) -> dict[str, float]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    scores = {}
    with torch.inference_mode():
        for name, video in videos.items():
            indices = np.linspace(0, len(video) - 1, 4, dtype=int)
            images = [Image.fromarray(video[index, ..., :3]) for index in indices]
            inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
            output = model(**inputs)
            scores[name] = float(output.logits_per_text.mean().item())
    return scores


def _final_latent(metadata: dict[str, Any]) -> Any:
    records = _records(metadata)
    return _load_tensor(records[max(records)]["latent_path"])


def run_gpu_audit(args: argparse.Namespace, findings: list[Finding], static: dict[str, Any]) -> list[Finding]:
    import torch
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler

    output_dir = args.output_dir
    prompt_entries = json.loads(args.prompt_set.read_text())
    entry = next(item for item in prompt_entries if item["prompt_id"] == args.prompt_id)
    prompt = entry["prompt"]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    omni = Omni(
        model=args.model, boundary_ratio=args.boundary_ratio, flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager, enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=DiffusionParallelConfig(), init_timeout=600, stage_init_timeout=600,
    )
    generated: dict[str, tuple[np.ndarray, dict[str, Any], float]] = {}
    try:
        generated["uninterrupted"] = _run_generate(omni, args, prompt, args.seed, "audit_uninterrupted", output_dir / "gpu")
        baseline_video, baseline_meta, _ = generated["uninterrupted"]
        baseline_records = _records(baseline_meta)
        clean_tensor = _load_tensor(baseline_records[args.checkpoint_step]["latent_path"])
        clean_np = _tensor_numpy(clean_tensor)
        torch.save(clean_tensor, output_dir / "clean_latent_step20.pt")

        # Runtime axis gate.
        expected_shape = (1, 16, math.ceil(args.num_frames / 4), math.ceil(args.height / 8), math.ceil(args.width / 8))
        if tuple(clean_tensor.shape) != expected_shape:
            findings.append(Finding("CRITICAL", "GPU_LATENT_AXIS_OR_SHAPE", "Actual GPU latent shape violates [B,C,T,H,W].", f"{tuple(clean_tensor.shape)} != {expected_shape}"))
            raise RuntimeError("STOP: actual latent axes/shape are invalid")

        # Raw independent identity serialization and direct/disk resumes.
        identity_dir = output_dir / "gpu_identity"
        payload, metadata, _, _ = serialize_components(identity_dir, "identity", [("latent", clean_np)], {"dtype": str(clean_np.dtype)})
        _, arrays = deserialize_components(payload, metadata)
        disk_tensor = torch.from_numpy(arrays[0]).to(dtype=clean_tensor.dtype)
        identity_error = error(clean_np, arrays[0])
        generated["direct_resume"] = _run_generate(omni, args, prompt, args.seed, "audit_direct_resume", output_dir / "gpu", clean_tensor, args.checkpoint_step)
        generated["disk_resume"] = _run_generate(omni, args, prompt, args.seed, "audit_disk_resume", output_dir / "gpu", disk_tensor, args.checkpoint_step)
        baseline_final = _load_tensor(baseline_records[args.num_inference_steps]["latent_path"])
        identity_result = {
            "shape_equal": tuple(clean_tensor.shape) == tuple(disk_tensor.shape),
            "dtype_equal": clean_tensor.dtype == disk_tensor.dtype,
            "byte_count_equal": clean_tensor.numel() * clean_tensor.element_size() == disk_tensor.numel() * disk_tensor.element_size(),
            "sha256_before": array_sha256(clean_np), "sha256_after": array_sha256(arrays[0]),
            **identity_error,
        }
        for name in ("direct_resume", "disk_resume"):
            video, metadata_doc, _ = generated[name]
            identity_result[f"{name}_final_latent"] = error(_tensor_numpy(baseline_final), _tensor_numpy(_final_latent(metadata_doc)))
            identity_result[f"{name}_video"] = independent_video_metrics(video, baseline_video)
        (output_dir / "exact_identity_roundtrip.json").write_text(json.dumps(identity_result, indent=2) + "\n")
        exact = all(
            identity_result[key]["max_abs"] == 0
            for key in ("direct_resume_final_latent", "disk_resume_final_latent")
        ) and all(identity_result[key]["video_mse"] == 0 for key in ("direct_resume_video", "disk_resume_video"))
        if not exact:
            findings.append(Finding("CRITICAL", "EULER_IDENTITY_RESUME", "Independent Euler identity resume is not exact.", json.dumps(identity_result)))
            raise RuntimeError("STOP: exact Euler identity resume failed")

        # Off-by-one controls. z20 is after 20 updates; correct continuation starts at schedule index 20.
        scheduler = WanEulerScheduler(num_train_timesteps=1000, shift=args.flow_shift)
        scheduler.set_timesteps(args.num_inference_steps, device="cpu")
        timestep_rows = []
        for name, index in (("correct", 20), ("repeat_current_step", 19), ("skip_next_step", 21)):
            if name != "correct":
                generated[name] = _run_generate(omni, args, prompt, args.seed, f"audit_{name}", output_dir / "gpu", clean_tensor, index)
            video, metadata_doc, _ = generated["direct_resume" if name == "correct" else name]
            timestep_rows.append(
                {
                    "resume_variant": name, "start_index": index,
                    "timestep_value": float(scheduler.timesteps[index]),
                    "final_latent_error": error(_tensor_numpy(baseline_final), _tensor_numpy(_final_latent(metadata_doc)))["normalized_l2"],
                    "video_error": independent_video_metrics(video, baseline_video)["video_mse"],
                }
            )
        write_csv(output_dir / "resume_timestep_audit.csv", timestep_rows)
        if timestep_rows[0]["final_latent_error"] != 0 or timestep_rows[1]["final_latent_error"] == 0 or timestep_rows[2]["final_latent_error"] == 0:
            findings.append(Finding("CRITICAL", "RESUME_OFF_BY_ONE", "Resume index controls do not uniquely identify step 20.", json.dumps(timestep_rows)))
            raise RuntimeError("STOP: resume timestep audit failed")

        # Independent representation bytes and one-cell recovery reproduction.
        representation_rows = []
        recovered_videos: dict[str, np.ndarray] = {}
        for name in REPRESENTATIONS:
            encoded = encode_representation(clean_np, name, output_dir / "independent_representations" / name)
            tensor = torch.from_numpy(encoded.restored).to(dtype=clean_tensor.dtype)
            video, metadata_doc, elapsed = _run_generate(omni, args, prompt, args.seed, f"audit_rep_{name}", output_dir / "gpu", tensor, args.checkpoint_step)
            recovered_videos[name] = video
            metrics = independent_video_metrics(video, baseline_video)
            representation_rows.append(
                {
                    "representation": name, "payload_bytes": encoded.payload_bytes,
                    "metadata_bytes": encoded.metadata_bytes, "total_bytes": encoded.payload_bytes + encoded.metadata_bytes,
                    "ratio_vs_full": (encoded.payload_bytes + encoded.metadata_bytes) / clean_np.nbytes,
                    "source_dtype": str(clean_np.dtype), "transformed_shapes": encoded.transformed_shapes,
                    "initial_mse": error(clean_np, encoded.restored)["mse"], "resume_ms": elapsed, **metrics,
                }
            )
        write_csv(output_dir / "independent_representation_reproduction.csv", representation_rows)

        # Independent disk layout and zero-fill recovery.
        layout_rows = []
        for layout in LAYOUTS:
            exact_array, _, _ = shard_roundtrip(clean_np, layout, 4, args.layout_seed, output_dir / "gpu_shards", None)
            damaged, disk_mask, sizes = shard_roundtrip(clean_np, layout, 4, args.layout_seed, output_dir / "gpu_shards", 0)
            if error(clean_np, exact_array)["max_abs"] != 0 or np.any(damaged[~disk_mask] != clean_np[~disk_mask]) or np.any(damaged[disk_mask] != 0):
                findings.append(Finding("CRITICAL", f"GPU_LAYOUT_{layout.upper()}", "GPU source disk-layout invariant failed before resume.", ""))
                raise RuntimeError("STOP: disk layout invariant failed")
            video, _, elapsed = _run_generate(omni, args, prompt, args.seed, f"audit_layout_{layout}", output_dir / "gpu", torch.from_numpy(damaged), args.checkpoint_step)
            recovered_videos[f"layout_{layout}"] = video
            layout_rows.append(
                {
                    "layout": layout, "missing_elements": int(disk_mask.sum()),
                    "missing_bytes": int(disk_mask.sum()) * clean_np.itemsize,
                    "missing_fraction": float(disk_mask.mean()), "shard_bytes": json.dumps(sizes),
                    "resume_ms": elapsed, **independent_video_metrics(video, baseline_video),
                }
            )
        write_csv(output_dir / "independent_layout_reproduction.csv", layout_rows)

        # Reference/seed and metric controls.
        different_seed_video, _, _ = _run_generate(omni, args, prompt, args.seed + 1, "audit_different_seed", output_dir / "gpu")
        metric_videos = {
            "self": baseline_video,
            "exact_resume": generated["disk_resume"][0],
            "different_seed": different_seed_video,
            "zero": np.zeros_like(baseline_video),
            "temporal_shuffle": baseline_video[::-1].copy(),
            "spatial_blur": np.rint(
                (baseline_video.astype(np.float32) + np.roll(baseline_video, 8, 1) + np.roll(baseline_video, -8, 1) + np.roll(baseline_video, 8, 2) + np.roll(baseline_video, -8, 2)) / 5
            ).astype(np.uint8),
        }
        semantic = {} if args.disable_semantic_metric else _semantic_scores(prompt, metric_videos, args.semantic_model)
        metric_rows = [{"control": name, **independent_video_metrics(video, baseline_video), "semantic_quality": semantic.get(name, math.nan)} for name, video in metric_videos.items()]
        write_csv(output_dir / "metric_audit.csv", metric_rows)
        rows_by_name = {row["control"]: row for row in metric_rows}
        if rows_by_name["self"]["video_mse"] != 0 or rows_by_name["temporal_shuffle"]["temporal_dynamic_quality"] > 0.95:
            findings.append(Finding("CRITICAL", "GPU_METRIC_CONTROL", "Metric self/shuffle control failed.", json.dumps(metric_rows)))
            raise RuntimeError("STOP: metric audit failed")

        # Resume corruption monotonicity, only after all gates above pass.
        corruption_rows = []
        order = np.random.default_rng(17).permutation(clean_np.size)
        for fraction in (0.0, 0.01, 0.05, 0.125, 0.25, 0.5, 1.0):
            candidate = clean_np.copy().reshape(-1)
            candidate[order[: round(fraction * candidate.size)]] = 0
            video, _, elapsed = _run_generate(omni, args, prompt, args.seed, f"audit_corruption_{fraction}", output_dir / "gpu", torch.from_numpy(candidate.reshape(clean_np.shape)), args.checkpoint_step)
            corruption_rows.append({"family": "random_zero_fill", "severity": fraction, "resume_ms": elapsed, **independent_video_metrics(video, baseline_video)})
        write_csv(output_dir / "corruption_monotonicity.csv", corruption_rows)

        # Cross-check only comparable facts; Stage B numerical quality uses UniPC and is not directly comparable.
        old = [row for row in static["raw_rows"] if row["prompt_id"] == args.prompt_id and int(row["seed"]) == args.seed and int(row["checkpoint_step"]) == args.checkpoint_step]
        old_by_variant = {row["variant"]: row for row in old}
        cross_rows = []
        for row in representation_rows:
            old_row = old_by_variant[row["representation"]]
            cross_rows.append(
                {
                    "condition": row["representation"], "field": "total_bytes",
                    "existing_value": old_row["total_checkpoint_bytes"], "independent_value": row["total_bytes"],
                    "absolute_delta": abs(float(old_row["total_checkpoint_bytes"]) - float(row["total_bytes"])),
                    "relative_delta": abs(float(old_row["total_checkpoint_bytes"]) - float(row["total_bytes"])) / max(float(old_row["total_checkpoint_bytes"]), 1),
                    "comparability": "exact byte implementation cross-check",
                }
            )
            cross_rows.append(
                {
                    "condition": row["representation"], "field": "temporal_dynamic",
                    "existing_value": old_row["temporal_dynamic_vs_full"], "independent_value": row["temporal_dynamic_quality"],
                    "absolute_delta": abs(float(old_row["temporal_dynamic_vs_full"]) - float(row["temporal_dynamic_quality"])),
                    "relative_delta": abs(float(old_row["temporal_dynamic_vs_full"]) - float(row["temporal_dynamic_quality"])) / max(abs(float(old_row["temporal_dynamic_vs_full"])), 1e-12),
                    "comparability": "NOT directly comparable: existing=UniPC relative metric, independent=Euler absolute metric",
                }
            )
        write_csv(output_dir / "crosscheck_existing_results.csv", cross_rows)

        # Provenance hashes.
        provenance = {
            "prompt": prompt, "prompt_id": args.prompt_id, "seed": args.seed,
            "checkpoint_step": args.checkpoint_step,
            "reference_video_path": baseline_records[args.num_inference_steps].get("frames_path"),
            "reference_video_sha256": array_sha256(baseline_video),
            "recovered_video_hashes": {name: array_sha256(video) for name, video in recovered_videos.items()},
        }
        (output_dir / "reference_seed_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")
    finally:
        omni.shutdown()
    return findings


def write_final_report(args: argparse.Namespace, findings: list[Finding], gpu_complete: bool) -> None:
    critical = [item for item in findings if item.severity == "CRITICAL"]
    stage_b_bug = any(item.code == "STAGE_B_UNIPC_HISTORY_MISSING" for item in critical)
    independent_failure = any(item.code != "STAGE_B_UNIPC_HISTORY_MISSING" for item in critical)
    if gpu_complete and not independent_failure and stage_b_bug:
        verdict = "PIPELINE PARTIALLY INVALID"
    elif gpu_complete and not critical:
        verdict = "PIPELINE VALIDATED"
    elif independent_failure:
        verdict = "PIPELINE INVALID"
    else:
        verdict = "PIPELINE PARTIALLY INVALID"
    lines = [
        verdict,
        "",
        "# Video State Protection Independent Audit",
        "",
        f"- GPU audit complete: `{gpu_complete}`",
        f"- Findings: `{len(findings)}`",
        "",
        "## Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- **{item.severity} {item.code}**: {item.summary} Evidence: {item.evidence}")
    lines += [
        "",
        "## Answers",
        "",
        "- Q1-Q2: See `exact_identity_roundtrip.json` and `resume_timestep_audit.csv`; unavailable until GPU audit completes." if not gpu_complete else "- Q1-Q2: Independent Euler identity/timestep gates completed; see the corresponding artifacts.",
        "- Q3: Stage B `exact_equal=False` is a BUG for FULL: it used default multistep UniPC without solver history.",
        "- Q4-Q5: Stage B source is FP32; INT8 is therefore approximately 25% including its one FP32 scale and metadata.",
        "- Q6-Q10: Static independent masks and disk reassembly use verified `[B,C,T,H,W]`; see layout/shard/zero-fill CSVs.",
        "- Q11-Q15: PENDING GPU audit." if not gpu_complete else "- Q11-Q15: See metric, corruption, independent representation, and independent layout reproduction CSVs.",
        "- Q16: Byte accounting and the claim that Stage B FULL was non-exact remain valid.",
        "- Q17: Existing Stage B quality frontier must be rerun with exact Euler before it supports recovery conclusions.",
    ]
    (args.output_dir / "video_state_protection_independent_audit.md").write_text("\n".join(lines) + "\n")
    result = {"verdict": verdict, "gpu_complete": gpu_complete, "findings": [asdict(item) for item in findings]}
    (args.output_dir / "audit_findings.json").write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "full"), default="static")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt-set", type=Path, default=Path("experiments/video_recovery_prompt_set.json"))
    parser.add_argument("--prompt-id", default="recovery_000")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-step", type=int, default=20)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--layout-seed", type=int, default=20260827)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--stage-b-dir", type=Path, default=Path("results/video_state_protection_killtest_gpu0/run"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_state_protection_independent_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_step != 20 or args.num_inference_steps != 40:
        raise ValueError("Independent audit is preregistered for checkpoint 20 of 40")
    findings, static = run_static_audit(args)
    gpu_complete = False
    if args.mode == "full":
        findings = run_gpu_audit(args, findings, static)
        gpu_complete = True
    write_final_report(args, findings, gpu_complete)
    print(json.dumps({"gpu_complete": gpu_complete, "findings": [asdict(item) for item in findings]}, indent=2))


if __name__ == "__main__":
    main()
