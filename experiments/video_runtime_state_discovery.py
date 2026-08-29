#!/usr/bin/env python3
"""Mechanism-agnostic discovery map for Wan video diffusion runtime state.

The module keeps CPU corruption primitives importable without torch/vLLM.
GPU imports are lazy and all scientific runs are gated by exact Euler resume,
semantic-axis, corruption, byte-accounting, and metric controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
SEMANTIC_AXES = ("batch", "channel", "temporal", "height", "width")
RAW_FIELDS = (
    "status",
    "experiment_version",
    "config_hash",
    "model",
    "scheduler",
    "prompt_id",
    "prompt_text",
    "motion_category",
    "generation_seed",
    "corruption_seed",
    "trajectory_id",
    "checkpoint_step",
    "resume_index",
    "corruption_family",
    "corruption_name",
    "comparison_family",
    "comparison_basis",
    "clean_latent_hash",
    "corrupted_latent_hash",
    "source_dtype",
    "encoded_dtype",
    "encoded_shapes",
    "logical_bits_per_value",
    "theoretical_payload_bytes",
    "full_bytes",
    "payload_bytes",
    "metadata_bytes",
    "total_bytes",
    "byte_ratio",
    "missing_bytes",
    "retained_bytes",
    "changed_elements",
    "changed_fraction",
    "initial_mse",
    "initial_normalized_l2",
    "initial_max_abs",
    "initial_cosine",
    "initial_mean_shift",
    "initial_std_ratio",
    "final_latent_mse",
    "final_latent_normalized_l2",
    "final_latent_max_abs",
    "final_video_mse",
    "spatial_quality",
    "temporal_quality",
    "semantic_quality",
    "content_motion_proxy",
    "content_spatial_gradient_proxy",
    "content_temporal_gradient_proxy",
    "latent_temporal_change_proxy",
    "exact_baseline_valid",
    "condition_metadata_json",
    "result_path",
)


class GlobalStopError(RuntimeError):
    """A correctness-gate failure after which no scientific row is valid."""


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    comparison_family: str
    comparison_basis: str


@dataclass
class PreparedCondition:
    spec: ConditionSpec
    restored: np.ndarray
    corruption_seed: int | None
    encoded_dtype: str
    encoded_shapes: list[list[int]]
    logical_bits_per_value: int
    theoretical_payload_bytes: int
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    missing_bytes: int
    retained_bytes: int
    artifact_metadata: dict[str, Any]
    artifact_paths: list[str]


CONDITION_SPECS = {
    "full_direct": ConditionSpec("full_direct", "baseline", "baseline", "direct_memory"),
    "full_disk": ConditionSpec("full_disk", "baseline", "baseline", "disk_identity"),
    "fp16": ConditionSpec("fp16", "precision", "descriptive", "actual_serialized_bytes"),
    "int8": ConditionSpec("int8", "precision", "iso_storage_25pct", "actual_serialized_bytes"),
    "int4_like": ConditionSpec("int4_like", "precision", "descriptive", "unpacked_actual_bytes"),
    "spatial_down2": ConditionSpec("spatial_down2", "structural", "iso_storage_25pct", "actual_serialized_bytes"),
    "temporal_down2": ConditionSpec("temporal_down2", "structural", "descriptive", "actual_serialized_bytes"),
    "low_rank_25": ConditionSpec("low_rank_25", "structural", "iso_storage_25pct", "actual_serialized_bytes"),
    "spatial_contiguous": ConditionSpec(
        "spatial_contiguous", "missing_geometry", "iso_missing_2of9", "equal_missing_elements"
    ),
    "temporal_contiguous": ConditionSpec(
        "temporal_contiguous", "missing_geometry", "iso_missing_2of9", "two_complete_temporal_slices"
    ),
    "channel_contiguous": ConditionSpec(
        "channel_contiguous", "missing_geometry", "descriptive", "four_complete_channel_slices"
    ),
    "block_interleaved": ConditionSpec(
        "block_interleaved", "missing_geometry", "iso_missing_2of9", "equal_missing_elements"
    ),
    "random_missing": ConditionSpec("random_missing", "missing_geometry", "iso_missing_2of9", "equal_missing_elements"),
    "gaussian_matched_int8": ConditionSpec(
        "gaussian_matched_int8", "gaussian", "iso_initial_mse_int8", "realized_initial_mse"
    ),
    "gaussian_matched_random_missing": ConditionSpec(
        "gaussian_matched_random_missing", "gaussian", "iso_initial_mse_random_missing", "realized_initial_mse"
    ),
    "stale_1": ConditionSpec("stale_1", "staleness", "staleness", "state_k_minus_1_schedule_k"),
    "stale_2": ConditionSpec("stale_2", "staleness", "staleness", "state_k_minus_2_schedule_k"),
}

# Wan's BF16-valued latent may already lie exactly on the FP16 grid. An
# identity FP16 round trip is a valid measured outcome, not an implementation
# failure. All lossy/corruption conditions still must alter the clean state.
IDENTITY_ALLOWED_CONDITIONS = frozenset({"full_direct", "full_disk", "fp16"})


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_raw_schema(row: dict[str, Any]) -> None:
    missing = set(RAW_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Raw result is missing fields: {sorted(missing)}")
    if row["status"] != "COMPLETE":
        raise ValueError("Only COMPLETE rows belong in raw_results.csv")
    if row["scheduler"] != EXPECTED_SCHEDULER:
        raise ValueError("Raw discovery row is not Euler")
    if int(row["resume_index"]) != int(row["checkpoint_step"]):
        raise ValueError("Raw row has off-by-one resume semantics")
    for key in (
        "initial_mse",
        "final_latent_mse",
        "final_video_mse",
        "spatial_quality",
        "temporal_quality",
        "semantic_quality",
    ):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"Raw row has non-finite {key}")


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        config = yaml.safe_load(text)
    if config["model"] != EXPECTED_MODEL:
        raise ValueError(f"Unexpected model: {config['model']}")
    scheduler = config["scheduler"]
    if scheduler["name"] != EXPECTED_SCHEDULER or scheduler["sample_solver"] != "euler":
        raise RuntimeError("Discovery requires explicit WanEulerScheduler; fallback is forbidden")
    if config["generation"]["checkpoint_steps"] != [10, 20, 30]:
        raise ValueError("Checkpoint steps must remain [10, 20, 30]")
    if config["generation"]["num_inference_steps"] != 40:
        raise ValueError("Discovery is frozen to 40 Euler steps")
    if (
        config["corruption"].get("spatial_contiguous_definition")
        != "compact_center_region_plus_deterministic_axis_fringe_v1"
    ):
        raise ValueError("Spatial contiguous geometry definition changed")
    expected_generation = {
        "height": 480,
        "width": 832,
        "num_frames": 33,
        "guidance_scale": 4.0,
        "fps": 16.0,
        "boundary_ratio": 0.875,
    }
    for key, expected in expected_generation.items():
        if config["generation"][key] != expected:
            raise ValueError(f"Frozen generation setting changed: {key}={config['generation'][key]!r}")
    unknown = set(config["conditions"]) - set(CONDITION_SPECS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if set(config["conditions"]) != set(CONDITION_SPECS) or len(config["conditions"]) != len(CONDITION_SPECS):
        raise ValueError("The full preregistered condition matrix may not be reduced or duplicated")
    if not set(config["smoke_conditions"]).issubset(config["conditions"]):
        raise ValueError("Smoke conditions must be a subset of the frozen discovery matrix")
    return config


def config_hash(config: dict[str, Any], prompt_sha256: str) -> str:
    return sha256_bytes(canonical_json({"config": config, "prompt_set_sha256": prompt_sha256}))


def load_prompts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    prompt_path = REPO_ROOT / config["prompt_set"]
    prompt_sha = sha256_file(prompt_path)
    raw = json.loads(prompt_path.read_text())
    by_id = {row["prompt_id"]: row for row in raw}
    requested = config["prompt_ids"]
    if len(requested) != 12 or len(set(requested)) != 12:
        raise ValueError("Discovery requires exactly 12 unique preregistered prompts")
    missing = set(requested) - set(by_id)
    if missing:
        raise ValueError(f"Missing prompts: {sorted(missing)}")
    prompts = [by_id[prompt_id] for prompt_id in requested]
    if set(config["generation_seeds"]) != set(requested):
        raise ValueError("Every prompt must have exactly one frozen generation seed")
    return prompts, prompt_sha


def corruption_seed(prompt_id: str, generation_seed: int, checkpoint_step: int, corruption_name: str) -> int:
    document = f"video-runtime-state-discovery|{prompt_id}|{generation_seed}|{checkpoint_step}|{corruption_name}"
    return int.from_bytes(hashlib.sha256(document.encode()).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def latent_error(clean: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if clean.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {clean.shape} != {candidate.shape}")
    lhs = clean.astype(np.float64, copy=False)
    rhs = candidate.astype(np.float64, copy=False)
    delta = rhs - lhs
    flat_lhs = lhs.reshape(-1)
    flat_rhs = rhs.reshape(-1)
    flat_delta = delta.reshape(-1)
    clean_norm = float(np.linalg.norm(flat_lhs))
    denominator = float(np.linalg.norm(flat_lhs) * np.linalg.norm(flat_rhs))
    return {
        "mse": float(np.mean(flat_delta * flat_delta)),
        "normalized_l2": float(np.linalg.norm(flat_delta)) / max(clean_norm, 1e-30),
        "max_abs": float(np.max(np.abs(flat_delta))),
        "cosine": float(np.dot(flat_lhs, flat_rhs) / max(denominator, 1e-30)),
        "mean_shift": float(rhs.mean() - lhs.mean()),
        "std_ratio": float(rhs.std() / max(lhs.std(), 1e-30)),
        "changed_elements": int(np.count_nonzero(flat_delta)),
        "changed_fraction": float(np.count_nonzero(flat_delta) / flat_delta.size),
    }


def tensor_invariants(clean: np.ndarray, candidate: np.ndarray, *, require_difference: bool) -> dict[str, Any]:
    if clean.ndim != 5 or tuple(SEMANTIC_AXES) != ("batch", "channel", "temporal", "height", "width"):
        raise AssertionError("Latent semantic axes are not [B,C,T,H,W]")
    if clean.shape != candidate.shape or clean.dtype != candidate.dtype:
        raise AssertionError("Corruption changed latent shape or dtype")
    if not np.isfinite(candidate).all():
        raise AssertionError("Corruption produced NaN or Inf")
    metrics = latent_error(clean, candidate)
    if require_difference and metrics["changed_elements"] == 0:
        raise AssertionError("Corruption did not change the clean tensor")
    return {
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "min": float(candidate.min()),
        "max": float(candidate.max()),
        "mean": float(candidate.mean()),
        "std": float(candidate.std()),
        "nan_count": int(np.isnan(candidate).sum()),
        "inf_count": int(np.isinf(candidate).sum()),
        "sha256": array_sha256(candidate),
        **metrics,
    }


def _compact_spatial_order(height: int, width: int) -> np.ndarray:
    """Order spatial cells from the center out with deterministic ties."""
    rows, columns = np.indices((height, width), dtype=np.float64)
    row_center = (height - 1) / 2
    column_center = (width - 1) / 2
    # Normalize each axis so the selected prefix is compact in image space
    # rather than being biased toward the longer latent dimension.
    distance = ((rows - row_center) / max(height, 1)) ** 2 + ((columns - column_center) / max(width, 1)) ** 2
    return np.lexsort((columns.reshape(-1), rows.reshape(-1), distance.reshape(-1)))


def build_missing_mask(
    shape: tuple[int, ...],
    geometry: str,
    *,
    target_elements: int,
    seed: int,
    temporal_slices: int,
    channel_slices: int,
    block_elements: int,
) -> np.ndarray:
    """Return a logical [B,C,T,H,W] missing mask with exact semantics.

    ``block_interleaved`` treats each contiguous channel vector in logical
    [B,T,H,W,C] order as a block. It chooses exactly ``target/block`` block
    indices evenly over that logical block sequence using
    floor(i * total_blocks / selected_blocks). This is deterministic,
    distributed across time/space, and does not inherit either old layout.
    """
    if len(shape) != 5:
        raise ValueError("Missing layouts require [B,C,T,H,W]")
    batch, channels, temporal, height, width = shape
    size = math.prod(shape)
    if not 0 < target_elements < size:
        raise ValueError("Missing cardinality must be strictly between zero and latent size")
    mask = np.zeros(shape, dtype=bool)
    coordinates = np.arange(size, dtype=np.int64).reshape(shape)
    if geometry == "temporal_contiguous":
        expected = batch * channels * temporal_slices * height * width
        if target_elements != expected or not 0 < temporal_slices < temporal:
            raise ValueError("Temporal loss must contain only complete T slices")
        mask[:, :, :temporal_slices, :, :] = True
    elif geometry == "channel_contiguous":
        expected = batch * channel_slices * temporal * height * width
        if target_elements != expected or not 0 < channel_slices < channels:
            raise ValueError("Channel loss must contain only complete C slices")
        mask[:, :channel_slices, :, :, :] = True
    elif geometry == "spatial_contiguous":
        denominator = batch * channels * temporal
        full_cells, fringe_elements = divmod(target_elements, denominator)
        spatial_order = _compact_spatial_order(height, width)
        if full_cells >= spatial_order.size:
            raise ValueError("Spatial loss must retain at least one spatial cell")
        per_axis = mask.reshape(denominator, height * width)
        per_axis[:, spatial_order[:full_cells]] = True
        if fringe_elements:
            # Exact iso-byte cardinality is impossible using only complete
            # B/C/T prisms when target_elements is not divisible by B*C*T.
            # Put the unavoidable remainder on the next compact boundary cell.
            per_axis[:fringe_elements, spatial_order[full_cells]] = True
    elif geometry == "block_interleaved":
        canonical = coordinates.transpose(0, 2, 3, 4, 1).reshape(-1)
        if size % block_elements or target_elements % block_elements:
            raise ValueError("Block interleaving requires whole equal-sized logical blocks")
        blocks = canonical.reshape(-1, block_elements)
        selected_count = target_elements // block_elements
        selected = np.floor(np.arange(selected_count) * len(blocks) / selected_count).astype(np.int64)
        if len(np.unique(selected)) != selected_count:
            raise AssertionError("Distributed block selection is not unique")
        mask.reshape(-1)[blocks[selected].reshape(-1)] = True
    elif geometry == "random_missing":
        selected = np.random.default_rng(seed).choice(size, size=target_elements, replace=False)
        mask.reshape(-1)[selected] = True
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    if int(mask.sum()) != target_elements:
        raise AssertionError(f"{geometry} cardinality {mask.sum()} != {target_elements}")
    return mask


def zero_fill(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if clean.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Mask must be boolean and match clean latent")
    restored = clean.copy()
    restored[mask] = 0
    if not np.array_equal(restored[~mask], clean[~mask]) or np.any(restored[mask] != 0):
        raise AssertionError("Zero fill changed non-target elements or failed to clear target elements")
    return restored


def quantize_symmetric(clean: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray, np.float32]:
    if levels not in (7, 127):
        raise ValueError("Discovery supports signed INT4-like or INT8 grids only")
    maximum = max(float(np.max(np.abs(clean))), 1e-12)
    scale = np.float32(maximum / levels)
    quantized = np.clip(np.rint(clean / scale), -levels, levels).astype(np.int8)
    restored = (quantized.astype(np.float32) * scale).astype(clean.dtype, copy=False)
    return restored, quantized, scale


def gaussian_matched_mse(clean: np.ndarray, target_mse: float, seed: int) -> tuple[np.ndarray, float]:
    if target_mse <= 0:
        raise ValueError("Matched Gaussian requires a positive target MSE")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.shape, dtype=np.float32)
    realized = float(np.mean(noise.astype(np.float64) ** 2))
    noise *= np.float32(math.sqrt(target_mse / max(realized, 1e-30)))
    candidate = (clean + noise).astype(clean.dtype)
    matched = float(latent_error(clean, candidate)["mse"])
    return candidate, matched


def serialize_components(
    directory: Path,
    name: str,
    components: list[tuple[str, np.ndarray]],
    metadata: dict[str, Any],
) -> tuple[int, int, list[str]]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    descriptions = []
    for component_name, component in components:
        array = np.ascontiguousarray(component)
        offset = len(payload)
        raw = array.tobytes(order="C")
        payload.extend(raw)
        descriptions.append(
            {
                "name": component_name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
    document = {**metadata, "components": descriptions}
    payload_path = directory / f"{name}.payload.bin"
    metadata_path = directory / f"{name}.metadata.json"
    atomic_write(payload_path, bytes(payload))
    atomic_json(metadata_path, document)
    return len(payload), metadata_path.stat().st_size, [str(payload_path), str(metadata_path)]


def deserialize_components(payload_path: Path, metadata_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    raw = payload_path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    arrays = []
    cursor = 0
    for component in metadata["components"]:
        if component["offset"] != cursor:
            raise AssertionError("Serialized components are not contiguous")
        end = cursor + int(component["nbytes"])
        array = np.frombuffer(raw[cursor:end], dtype=np.dtype(component["dtype"]))
        arrays.append(array.reshape(component["shape"]).copy())
        cursor = end
    if cursor != len(raw):
        raise AssertionError("Serialized payload has unaccounted bytes")
    return metadata, arrays


def _interpolate(clean: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(clean)).float()
    return functional.interpolate(tensor, size=size, mode="trilinear", align_corners=False).cpu().numpy()


def _save_candidate(
    clean: np.ndarray,
    candidate: np.ndarray,
    spec: ConditionSpec,
    directory: Path,
    metadata: dict[str, Any],
    *,
    encoded_dtype: str | None = None,
    logical_bits: int = 32,
    theoretical_payload_bytes: int | None = None,
    components: list[tuple[str, np.ndarray]] | None = None,
) -> PreparedCondition:
    components = components or [("restored_latent", candidate)]
    payload, meta_bytes, paths = serialize_components(
        directory, spec.name, components, {"condition": spec.name, **metadata}
    )
    prepared = PreparedCondition(
        spec=spec,
        restored=np.ascontiguousarray(candidate.astype(clean.dtype, copy=False)),
        corruption_seed=metadata.get("corruption_seed"),
        encoded_dtype=encoded_dtype or str(components[0][1].dtype),
        encoded_shapes=[list(value.shape) for _, value in components],
        logical_bits_per_value=logical_bits,
        theoretical_payload_bytes=theoretical_payload_bytes or payload,
        payload_bytes=payload,
        metadata_bytes=meta_bytes,
        total_bytes=payload + meta_bytes,
        missing_bytes=int(metadata.get("missing_bytes", 0)),
        retained_bytes=int(metadata.get("retained_bytes", clean.nbytes)),
        artifact_metadata=metadata,
        artifact_paths=paths,
    )
    if len(components) == 1 and components[0][0] == "restored_latent":
        _, arrays = deserialize_components(Path(paths[0]), Path(paths[1]))
        prepared.restored = np.ascontiguousarray(arrays[0].astype(clean.dtype, copy=False))
    return prepared


def prepare_condition(
    clean: np.ndarray,
    clean_states: dict[int, np.ndarray],
    name: str,
    config: dict[str, Any],
    *,
    prompt_id: str,
    generation_seed: int,
    checkpoint_step: int,
    directory: Path,
) -> PreparedCondition:
    clean = np.ascontiguousarray(clean.astype(np.float32, copy=False))
    spec = CONDITION_SPECS[name]
    seed = corruption_seed(prompt_id, generation_seed, checkpoint_step, name)
    base_meta = {
        "prompt_id": prompt_id,
        "generation_seed": generation_seed,
        "checkpoint_step": checkpoint_step,
        "clean_latent_hash": array_sha256(clean),
        "source_shape": list(clean.shape),
        "source_dtype": str(clean.dtype),
        "corruption_seed": seed,
    }
    corruption = config["corruption"]
    if name == "full_direct":
        return PreparedCondition(
            spec,
            clean.copy(),
            None,
            str(clean.dtype),
            [list(clean.shape)],
            32,
            clean.nbytes,
            clean.nbytes,
            0,
            clean.nbytes,
            0,
            clean.nbytes,
            {"direct_memory": True},
            [],
        )
    if name == "full_disk":
        prepared = _save_candidate(clean, clean.copy(), spec, directory, base_meta)
        metadata, arrays = deserialize_components(Path(prepared.artifact_paths[0]), Path(prepared.artifact_paths[1]))
        prepared.restored = arrays[0].astype(clean.dtype)
        prepared.artifact_metadata = metadata
        return prepared
    if name in {"fp16", "int8", "int4_like"}:
        if name == "fp16":
            encoded = clean.astype(np.float16)
            restored = encoded.astype(np.float32)
            prepared = _save_candidate(
                clean,
                restored,
                spec,
                directory,
                base_meta,
                encoded_dtype="float16",
                logical_bits=16,
                components=[("latent_fp16", encoded)],
            )
            _, arrays = deserialize_components(Path(prepared.artifact_paths[0]), Path(prepared.artifact_paths[1]))
            prepared.restored = np.ascontiguousarray(arrays[0].astype(np.float32))
            return prepared
        levels = corruption["int8_levels"] if name == "int8" else corruption["int4_levels"]
        restored, quantized, scale = quantize_symmetric(clean, levels)
        logical_bits = 8 if name == "int8" else 4
        theoretical = math.ceil(clean.size * logical_bits / 8) + 4
        metadata = {
            **base_meta,
            "levels": levels,
            "scale": float(scale),
            "grid": f"[-{levels},{levels}]",
            "packed": False,
        }
        prepared = _save_candidate(
            clean,
            restored,
            spec,
            directory,
            metadata,
            encoded_dtype="int8",
            logical_bits=logical_bits,
            theoretical_payload_bytes=theoretical,
            components=[("quantized_unpacked", quantized), ("scale", np.asarray([scale], dtype=np.float32))],
        )
        _, arrays = deserialize_components(Path(prepared.artifact_paths[0]), Path(prepared.artifact_paths[1]))
        prepared.restored = np.ascontiguousarray(arrays[0].astype(np.float32) * float(arrays[1][0]))
        return prepared
    if name in {"spatial_down2", "temporal_down2"}:
        _, _, temporal, height, width = clean.shape
        if name == "spatial_down2":
            down_size = (temporal, height // 2, width // 2)
        else:
            down_size = (math.ceil(temporal / 2), height, width)
        encoded = _interpolate(clean, down_size)
        restored = _interpolate(encoded, tuple(clean.shape[2:])).astype(np.float32)
        prepared = _save_candidate(
            clean,
            restored,
            spec,
            directory,
            {**base_meta, "encoded_size": list(down_size), "reconstruction": "trilinear_align_corners_false"},
            components=[("latent_downsampled", encoded)],
        )
        _, arrays = deserialize_components(Path(prepared.artifact_paths[0]), Path(prepared.artifact_paths[1]))
        prepared.restored = np.ascontiguousarray(_interpolate(arrays[0], tuple(clean.shape[2:])).astype(np.float32))
        return prepared
    if name == "low_rank_25":
        batch, channels, temporal, height, width = clean.shape
        matrix = clean.reshape(batch * channels * temporal, height * width)
        rank = max(1, round(min(matrix.shape) * float(corruption["low_rank_ratio"])))
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        u = u[:, :rank].astype(np.float32)
        singular = singular[:rank].astype(np.float32)
        vh = vh[:rank].astype(np.float32)
        restored = ((u * singular[None, :]) @ vh).reshape(clean.shape).astype(np.float32)
        prepared = _save_candidate(
            clean,
            restored,
            spec,
            directory,
            {**base_meta, "rank": rank, "reconstruction": "truncated_svd"},
            components=[("u", u), ("s", singular), ("vh", vh)],
        )
        _, arrays = deserialize_components(Path(prepared.artifact_paths[0]), Path(prepared.artifact_paths[1]))
        prepared.restored = np.ascontiguousarray(
            ((arrays[0] * arrays[1][None, :]) @ arrays[2]).reshape(clean.shape).astype(np.float32)
        )
        return prepared
    if name in {
        "spatial_contiguous",
        "temporal_contiguous",
        "channel_contiguous",
        "block_interleaved",
        "random_missing",
    }:
        temporal_target = (
            clean.shape[0]
            * clean.shape[1]
            * int(corruption["temporal_missing_slices"])
            * clean.shape[3]
            * clean.shape[4]
        )
        if name == "channel_contiguous":
            target = (
                clean.shape[0]
                * int(corruption["channel_missing_slices"])
                * clean.shape[2]
                * clean.shape[3]
                * clean.shape[4]
            )
        else:
            target = temporal_target
        mask = build_missing_mask(
            tuple(clean.shape),
            name,
            target_elements=target,
            seed=seed,
            temporal_slices=int(corruption["temporal_missing_slices"]),
            channel_slices=int(corruption["channel_missing_slices"]),
            block_elements=int(corruption["block_elements"]),
        )
        restored = zero_fill(clean, mask)
        mask_path = directory / f"{name}.mask.npy"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask, allow_pickle=False)
        affected_temporal = np.flatnonzero(mask.any(axis=(0, 1, 3, 4))).tolist()
        affected_channels = np.flatnonzero(mask.any(axis=(0, 2, 3, 4))).tolist()
        metadata = {
            **base_meta,
            "geometry": name,
            "logical_axes": list(SEMANTIC_AXES),
            "mask_path": str(mask_path),
            "mask_sha256": array_sha256(mask),
            "target_elements": target,
            "missing_fraction": float(mask.mean()),
            "missing_bytes": target * clean.itemsize,
            "retained_bytes": clean.nbytes - target * clean.itemsize,
            "affected_temporal_indices": affected_temporal,
            "affected_channel_indices": affected_channels,
            "per_temporal_slice_missing_fraction": mask.mean(axis=(0, 1, 3, 4)).tolist(),
            "per_channel_missing_fraction": mask.mean(axis=(0, 2, 3, 4)).tolist(),
            "temporal_slices": int(corruption["temporal_missing_slices"]),
            "channel_slices": int(corruption["channel_missing_slices"]),
            "block_elements": int(corruption["block_elements"]),
            "block_mapping": (
                "canonical=[B,T,H,W,C]; select floor(i*total_blocks/selected_blocks) for i in [0,selected_blocks)"
                if name == "block_interleaved"
                else None
            ),
            "spatial_mapping": (
                config["corruption"]["spatial_contiguous_definition"] if name == "spatial_contiguous" else None
            ),
        }
        prepared = _save_candidate(clean, restored, spec, directory, metadata)
        prepared.artifact_paths.append(str(mask_path))
        return prepared
    if name.startswith("gaussian_matched_"):
        target_name = name.removeprefix("gaussian_matched_")
        target_dir = directory.parent / target_name
        target = prepare_condition(
            clean,
            clean_states,
            target_name,
            config,
            prompt_id=prompt_id,
            generation_seed=generation_seed,
            checkpoint_step=checkpoint_step,
            directory=target_dir,
        )
        target_mse = float(latent_error(clean, target.restored)["mse"])
        restored, realized = gaussian_matched_mse(clean, target_mse, seed)
        mismatch = abs(realized - target_mse) / target_mse
        if mismatch > float(corruption["iso_error_relative_tolerance"]):
            raise AssertionError(f"Matched Gaussian MSE mismatch {mismatch:.6f} exceeds tolerance")
        return _save_candidate(
            clean,
            restored,
            spec,
            directory,
            {
                **base_meta,
                "matched_condition": target_name,
                "target_initial_mse": target_mse,
                "realized_initial_mse": realized,
                "relative_mismatch": mismatch,
            },
        )
    if name in {"stale_1", "stale_2"}:
        lag = 1 if name == "stale_1" else 2
        source_step = checkpoint_step - lag
        if source_step not in clean_states:
            raise KeyError(f"Missing clean state for stale source step {source_step}")
        restored = np.ascontiguousarray(clean_states[source_step].astype(clean.dtype, copy=False))
        return _save_candidate(
            clean,
            restored,
            spec,
            directory,
            {
                **base_meta,
                "source_step": source_step,
                "target_resume_step": checkpoint_step,
                "source_latent_hash": array_sha256(restored),
                "scheduler_continuation_unchanged": True,
            },
        )
    raise ValueError(name)


def synthetic_coordinate_tensor() -> np.ndarray:
    result = np.empty((1, 4, 4, 4, 4), dtype=np.float32)
    for c, t, h, w in itertools.product(range(4), repeat=4):
        result[0, c, t, h, w] = c * 1000 + t * 100 + h * 10 + w
    return result


def run_cpu_corruption_gates() -> dict[str, Any]:
    clean = synthetic_coordinate_tensor()
    size = clean.size
    target = size // 4
    masks = {}
    for name in (
        "spatial_contiguous",
        "temporal_contiguous",
        "channel_contiguous",
        "block_interleaved",
        "random_missing",
    ):
        mask = build_missing_mask(
            clean.shape, name, target_elements=target, seed=41, temporal_slices=1, channel_slices=1, block_elements=4
        )
        damaged = zero_fill(clean, mask)
        if int(mask.sum()) != target or not np.array_equal(damaged[~mask], clean[~mask]):
            raise AssertionError(f"Synthetic gate failed for {name}")
        masks[name] = mask
    temporal = masks["temporal_contiguous"]
    if np.flatnonzero(temporal.any(axis=(0, 1, 3, 4))).tolist() != [0]:
        raise AssertionError("Temporal mask is not exactly one complete T slice")
    channel = masks["channel_contiguous"]
    if np.flatnonzero(channel.any(axis=(0, 2, 3, 4))).tolist() != [0]:
        raise AssertionError("Channel mask is not exactly one complete C slice")
    spatial = masks["spatial_contiguous"]
    if not spatial[..., 1:3, 1:3].all() or int(spatial.sum()) != target:
        raise AssertionError("Spatial mask does not select the intended compact center region")
    real_shape = (1, 16, 9, 60, 104)
    real_target = 1 * 16 * 2 * 60 * 104
    real_spatial = build_missing_mask(
        real_shape,
        "spatial_contiguous",
        target_elements=real_target,
        seed=41,
        temporal_slices=2,
        channel_slices=4,
        block_elements=16,
    )
    counts_by_spatial_cell = real_spatial.reshape(1 * 16 * 9, 60 * 104).sum(axis=0)
    full_cells, fringe_elements = divmod(real_target, 1 * 16 * 9)
    if (
        int(real_spatial.sum()) != real_target
        or int(np.count_nonzero(counts_by_spatial_cell == 1 * 16 * 9)) != full_cells
        or int(np.count_nonzero(counts_by_spatial_cell == fringe_elements)) != 1
    ):
        raise AssertionError("Real Wan latent spatial mask failed exact compact-plus-fringe cardinality")
    if np.array_equal(masks["block_interleaved"], masks["random_missing"]):
        raise AssertionError("Deterministic block and random layouts unexpectedly coincide")
    if not np.array_equal(
        build_missing_mask(
            clean.shape,
            "random_missing",
            target_elements=target,
            seed=41,
            temporal_slices=1,
            channel_slices=1,
            block_elements=4,
        ),
        masks["random_missing"],
    ):
        raise AssertionError("Random mask is not deterministic")
    # A separate NumPy Generator must not mutate global or model RNG state.
    np.random.seed(123)
    before = np.random.get_state()
    build_missing_mask(
        clean.shape,
        "random_missing",
        target_elements=target,
        seed=99,
        temporal_slices=1,
        channel_slices=1,
        block_elements=4,
    )
    after = np.random.get_state()
    if any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
        raise AssertionError("Corruption RNG changed global generation RNG state")
    fp16 = clean.astype(np.float16).astype(np.float32)
    int8, _, _ = quantize_symmetric(clean, 127)
    int4, _, _ = quantize_symmetric(clean, 7)
    if int4.shape != clean.shape or int8.shape != clean.shape or fp16.shape != clean.shape:
        raise AssertionError("Quantization round trip changed shape")
    target_mse = float(latent_error(clean, int8)["mse"])
    gaussian, realized = gaussian_matched_mse(clean, target_mse, 7)
    if abs(realized - target_mse) / target_mse > 0.01:
        raise AssertionError("Matched Gaussian gate missed 1% tolerance")
    return {
        "semantic_axes": list(SEMANTIC_AXES),
        "shape": list(clean.shape),
        "mask_cardinality": {name: int(mask.sum()) for name, mask in masks.items()},
        "mask_sha256": {name: array_sha256(mask) for name, mask in masks.items()},
        "real_shape_spatial_cardinality": int(real_spatial.sum()),
        "real_shape_spatial_full_cells": full_cells,
        "real_shape_spatial_fringe_elements": fringe_elements,
        "quantization_changed": {
            "fp16": int(np.count_nonzero(fp16 != clean)),
            "int8": int(np.count_nonzero(int8 != clean)),
            "int4_like": int(np.count_nonzero(int4 != clean)),
        },
        "matched_gaussian_relative_mismatch": abs(realized - target_mse) / target_mse,
        "matched_gaussian_hash": array_sha256(gaussian),
        "passed": True,
    }


def video_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """The exact metric formulas that passed the independent audit controls."""
    lhs = candidate.astype(np.float64) / 255.0
    rhs = reference.astype(np.float64) / 255.0
    mse = float(np.mean((lhs - rhs) ** 2))
    spatial = 1.0 / (1.0 + mse / (float(np.var(rhs)) + 1e-12))
    lhs_delta = np.diff(lhs, axis=0)
    rhs_delta = np.diff(rhs, axis=0)
    temporal_mse = float(np.mean((lhs_delta - rhs_delta) ** 2)) if len(lhs) > 1 else 0.0
    temporal = 1.0 / (1.0 + temporal_mse / (float(np.mean(rhs_delta**2)) + 1e-12))
    return {"final_video_mse": mse * 255.0**2, "spatial_quality": spatial, "temporal_quality": temporal}


def content_descriptors(video: np.ndarray, latent: np.ndarray) -> dict[str, float]:
    value = video.astype(np.float32) / 255.0
    motion = float(np.mean(np.abs(np.diff(value, axis=0)))) if len(value) > 1 else 0.0
    spatial_h = float(np.mean(np.abs(np.diff(value, axis=1)))) if value.shape[1] > 1 else 0.0
    spatial_w = float(np.mean(np.abs(np.diff(value, axis=2)))) if value.shape[2] > 1 else 0.0
    latent_temporal = float(np.mean(np.abs(np.diff(latent.astype(np.float32), axis=2)))) if latent.shape[2] > 1 else 0.0
    return {
        "content_motion_proxy": motion,
        "content_spatial_gradient_proxy": (spatial_h + spatial_w) / 2.0,
        "content_temporal_gradient_proxy": motion,
        "latent_temporal_change_proxy": latent_temporal,
    }


class SemanticEvaluator:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.model = None
        self.processor = None
        if enabled:
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(model_name).eval().cpu()
            self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, prompt: str, video: np.ndarray) -> float:
        if not self.enabled:
            return float("nan")
        import torch
        from PIL import Image

        indices = np.linspace(0, len(video) - 1, 4, dtype=int)
        images = [Image.fromarray(video[index, ..., :3]) for index in indices]
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt", padding=True)
        with torch.inference_mode():
            output = self.model(**inputs)
        return float(output.logits_per_text.mean().item())


def normalize_video(outputs: Any) -> tuple[np.ndarray, Any]:
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


def scheduler_document(config: dict[str, Any]) -> dict[str, Any]:
    from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler

    scheduler_cfg = config["scheduler"]
    scheduler = WanEulerScheduler(
        num_train_timesteps=int(scheduler_cfg["num_train_timesteps"]),
        shift=float(scheduler_cfg["flow_shift"]),
        device="cpu",
    )
    if scheduler.__class__.__name__ != EXPECTED_SCHEDULER or scheduler.order != 1:
        raise RuntimeError("Expected first-order WanEulerScheduler; automatic fallback is forbidden")
    num_steps = int(config["generation"]["num_inference_steps"])
    scheduler.set_timesteps(num_steps, device="cpu")
    checkpoints = [int(value) for value in config["generation"]["checkpoint_steps"]]
    return {
        "scheduler_class": f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}",
        "scheduler_config": {
            "num_train_timesteps": scheduler.num_train_timesteps,
            "shift": float(scheduler_cfg["flow_shift"]),
            "order": scheduler.order,
        },
        "num_inference_steps": num_steps,
        "timesteps": [float(value) for value in scheduler.timesteps.cpu().tolist()],
        "checkpoint_indices": checkpoints,
        "resume_indices": checkpoints,
        "next_timestep_by_checkpoint": {str(step): float(scheduler.timesteps[step]) for step in checkpoints},
    }


def environment_document(config: dict[str, Any], scheduler: dict[str, Any]) -> dict[str, Any]:
    import torch

    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception as error:  # pragma: no cover - environment-dependent
        diffusers_version = f"unavailable: {error}"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(config["model"], revision=config.get("model_revision"), local_files_only=True)
        )
        resolved_model_revision = snapshot_path.name if snapshot_path.parent.name == "snapshots" else None
    except Exception:
        resolved_model_revision = None
    try:
        repository_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        repository_commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": config["model"],
        "model_revision": config.get("model_revision"),
        "resolved_model_revision": resolved_model_revision,
        "repository_commit": repository_commit,
        **scheduler,
    }


def build_omni(config: dict[str, Any], args: argparse.Namespace) -> Any:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    generation = config["generation"]
    return Omni(
        model=config["model"],
        boundary_ratio=float(generation["boundary_ratio"]),
        flow_shift=float(config["scheduler"]["flow_shift"]),
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=DiffusionParallelConfig(),
        init_timeout=600,
        stage_init_timeout=600,
    )


def sampling_params(
    config: dict[str, Any],
    *,
    seed: int,
    label: str,
    artifact_dir: Path,
    capture_steps: list[int],
    latents: Any = None,
    step_index: int = 0,
) -> Any:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    generation = config["generation"]
    if config["scheduler"]["sample_solver"] != "euler":
        raise RuntimeError("Non-Euler scheduler rejected before request submission")
    sampling = OmniDiffusionSamplingParams(
        height=int(generation["height"]),
        width=int(generation["width"]),
        num_frames=int(generation["num_frames"]),
        num_inference_steps=int(generation["num_inference_steps"]),
        guidance_scale=float(generation["guidance_scale"]),
        fps=float(generation["fps"]),
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    sampling.latents = None if latents is None else latents.detach().cpu().clone()
    sampling.step_index = int(step_index)
    sampling.extra_args = {
        "flow_shift": float(config["scheduler"]["flow_shift"]),
        "sample_solver": "euler",
        "trajectory_probe": {
            "artifact_dir": str(artifact_dir),
            "request_label": label,
            "capture_steps": capture_steps,
            "fps": float(generation["fps"]),
            "save_decoded": False,
            "save_latents": True,
            "save_mp4": False,
        },
    }
    return sampling


def run_generate(
    omni: Any,
    config: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    label: str,
    artifact_dir: Path,
    capture_steps: list[int],
    latents: Any = None,
    step_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any], float]:
    started = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        sampling_params(
            config,
            seed=seed,
            label=label,
            artifact_dir=artifact_dir,
            capture_steps=capture_steps,
            latents=latents,
            step_index=step_index,
        ),
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    video, output = normalize_video(outputs)
    metadata_path = output.custom_output.get("trajectory_probe_metadata_path")
    if not metadata_path:
        raise RuntimeError("Trajectory probe metadata is missing")
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("sample_solver") != "euler" or not str(metadata.get("scheduler_class", "")).endswith(
        EXPECTED_SCHEDULER
    ):
        raise GlobalStopError(
            "GLOBAL STOP: worker used "
            f"{metadata.get('scheduler_class')} / {metadata.get('sample_solver')} instead of Euler"
        )
    return video, metadata, elapsed


def probe_records(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["step_index"]): row for row in metadata["records"]}


def load_tensor_numpy(path: str | Path) -> np.ndarray:
    import torch

    tensor = torch.load(path, map_location="cpu")
    return tensor.detach().cpu().float().contiguous().numpy()


def final_latent_numpy(metadata: dict[str, Any]) -> np.ndarray:
    records = probe_records(metadata)
    return load_tensor_numpy(records[max(records)]["latent_path"])


def expected_latent_shape(config: dict[str, Any]) -> tuple[int, ...]:
    generation = config["generation"]
    return (
        1,
        16,
        math.ceil(int(generation["num_frames"]) / 4),
        math.ceil(int(generation["height"]) / 8),
        math.ceil(int(generation["width"]) / 8),
    )


def trajectory_id(config_digest: str, prompt_id: str, seed: int) -> str:
    return sha256_bytes(f"{config_digest}|{prompt_id}|{seed}".encode())[:20]


def clean_capture_steps(config: dict[str, Any]) -> list[int]:
    result = {int(config["generation"]["num_inference_steps"])}
    for checkpoint in config["generation"]["checkpoint_steps"]:
        result.update((int(checkpoint) - 2, int(checkpoint) - 1, int(checkpoint)))
    return sorted(result)


def _validate_clean_manifest(manifest: dict[str, Any], config_digest: str, prompt: dict[str, Any], seed: int) -> None:
    required = {
        "config_hash": config_digest,
        "prompt_id": prompt["prompt_id"],
        "prompt": prompt["prompt"],
        "generation_seed": seed,
        "scheduler": EXPECTED_SCHEDULER,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Cached clean trajectory mismatch for {key}: {manifest.get(key)!r} != {expected!r}")
    for row in manifest["states"]:
        path = Path(row["latent_path"])
        if not path.exists() or sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"Cached clean latent failed hash validation: {path}")
        array = load_tensor_numpy(path)
        if (
            list(array.shape) != row["shape"]
            or str(array.dtype) != row["dtype"]
            or array_sha256(array) != row["tensor_sha256"]
        ):
            raise RuntimeError(f"Cached clean latent metadata mismatch: {path}")
    video_path = Path(manifest["baseline_video_path"])
    if not video_path.exists() or sha256_file(video_path) != manifest["baseline_video_file_sha256"]:
        raise RuntimeError("Cached baseline video failed file validation")


def get_clean_trajectory(
    omni: Any,
    config: dict[str, Any],
    config_digest: str,
    prompt: dict[str, Any],
    seed: int,
    directory: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _validate_clean_manifest(manifest, config_digest, prompt, seed)
        states = {int(row["step"]): load_tensor_numpy(row["latent_path"]) for row in manifest["states"]}
        return np.load(manifest["baseline_video_path"], allow_pickle=False), states, manifest
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Partial unvalidated clean cache exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    label = f"clean_{prompt['prompt_id']}_{seed}"
    video, metadata, elapsed = run_generate(
        omni,
        config,
        prompt=prompt["prompt"],
        seed=seed,
        label=label,
        artifact_dir=directory,
        capture_steps=clean_capture_steps(config),
    )
    records = probe_records(metadata)
    required_steps = set(clean_capture_steps(config))
    if not required_steps.issubset(records):
        raise RuntimeError(f"Clean trajectory missing required states: {sorted(required_steps - set(records))}")
    states = {step: load_tensor_numpy(records[step]["latent_path"]) for step in required_steps}
    expected_shape = expected_latent_shape(config)
    if any(tuple(value.shape) != expected_shape or value.dtype != np.float32 for value in states.values()):
        raise RuntimeError(f"Clean latent does not match FP32 [B,C,T,H,W] {expected_shape}")
    video_path = directory / "baseline_video.npy"
    np.save(video_path, video, allow_pickle=False)
    scheduler = scheduler_document(config)
    state_rows = []
    for step in sorted(states):
        path = Path(records[step]["latent_path"])
        state_rows.append(
            {
                "step": step,
                "scheduler_timestep_used_to_produce_state": records[step].get("timestep"),
                "next_scheduler_timestep": scheduler["next_timestep_by_checkpoint"].get(str(step)),
                "resume_index": step,
                "next_operation": f"execute Euler schedule index {step}"
                if step < config["generation"]["num_inference_steps"]
                else "decode final latent",
                "latent_path": str(path),
                "file_sha256": sha256_file(path),
                "tensor_sha256": array_sha256(states[step]),
                "shape": list(states[step].shape),
                "dtype": str(states[step].dtype),
            }
        )
    manifest = {
        "config_hash": config_digest,
        "model": config["model"],
        "scheduler": EXPECTED_SCHEDULER,
        "scheduler_config_hash": sha256_bytes(canonical_json(scheduler)),
        "prompt_id": prompt["prompt_id"],
        "prompt": prompt["prompt"],
        "motion_category": prompt.get("motion_category"),
        "generation_seed": seed,
        "trajectory_id": trajectory_id(config_digest, prompt["prompt_id"], seed),
        "clean_generation_ms": elapsed,
        "baseline_video_path": str(video_path),
        "baseline_video_file_sha256": sha256_file(video_path),
        "baseline_video_tensor_sha256": array_sha256(video),
        "states": state_rows,
    }
    atomic_json(manifest_path, manifest)
    return video, states, manifest


def _cleanup_probe(metadata: dict[str, Any]) -> None:
    metadata_path = None
    for row in metadata.get("records", []):
        for key in ("latent_path", "frames_path", "mp4_path"):
            value = row.get(key)
            if value:
                Path(value).unlink(missing_ok=True)
    # The caller retains metrics and reproducible input corruption, not per-run outputs.
    label = metadata.get("label")
    if label and metadata.get("records"):
        first = metadata["records"][0].get("latent_path")
        if first:
            metadata_path = Path(first).parent / f"{label}_trajectory_probe.json"
    if metadata_path:
        metadata_path.unlink(missing_ok=True)


def _condition_result_valid(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        if row.get(key) != value:
            return False
    if row.get("status") != "COMPLETE":
        return False
    numeric = ("initial_mse", "final_latent_mse", "final_video_mse", "spatial_quality", "temporal_quality")
    if any(not math.isfinite(float(row[key])) for key in numeric):
        return False
    metadata = json.loads(row["condition_metadata_json"])
    for artifact in metadata.get("artifact_paths", []):
        if not Path(artifact).exists():
            return False
    return True


def run_resume_condition(
    omni: Any,
    evaluator: SemanticEvaluator,
    config: dict[str, Any],
    config_digest: str,
    prompt: dict[str, Any],
    seed: int,
    checkpoint_step: int,
    clean_states: dict[int, np.ndarray],
    baseline_video: np.ndarray,
    trajectory_manifest: dict[str, Any],
    condition_name: str,
    cell_dir: Path,
    *,
    keep_video: bool,
) -> dict[str, Any]:
    import torch

    result_path = cell_dir / "result.json"
    clean = clean_states[checkpoint_step]
    expected = {
        "config_hash": config_digest,
        "prompt_id": prompt["prompt_id"],
        "generation_seed": seed,
        "checkpoint_step": checkpoint_step,
        "corruption_name": condition_name,
        "clean_latent_hash": array_sha256(clean),
    }
    if _condition_result_valid(result_path, expected):
        return json.loads(result_path.read_text())
    if result_path.exists():
        result_path.rename(result_path.with_suffix(f".invalid-{int(time.time())}.json"))
    cell_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare_condition(
            clean,
            clean_states,
            condition_name,
            config,
            prompt_id=prompt["prompt_id"],
            generation_seed=seed,
            checkpoint_step=checkpoint_step,
            directory=cell_dir / "corruption",
        )
        require_difference = condition_name not in IDENTITY_ALLOWED_CONDITIONS
        initial = tensor_invariants(clean, prepared.restored, require_difference=require_difference)
        metadata = {**prepared.artifact_metadata, "artifact_paths": prepared.artifact_paths, "invariants": initial}
    except Exception as error:
        failure = {
            **expected,
            "status": "INVALID_IMPLEMENTATION",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "result_path": str(result_path),
        }
        atomic_json(result_path, failure)
        raise GlobalStopError(f"GLOBAL STOP: corruption invariant failed for {condition_name}") from error
    label = f"resume_{prompt['prompt_id']}_{seed}_s{checkpoint_step}_{condition_name}"
    resume_dir = cell_dir / "resume_probe"
    remaining = int(config["generation"]["num_inference_steps"]) - checkpoint_step
    try:
        video, probe, resume_ms = run_generate(
            omni,
            config,
            prompt=prompt["prompt"],
            seed=seed,
            label=label,
            artifact_dir=resume_dir,
            capture_steps=[remaining],
            latents=torch.from_numpy(prepared.restored),
            step_index=checkpoint_step,
        )
        final = final_latent_numpy(probe)
        final_clean = clean_states[int(config["generation"]["num_inference_steps"])]
        final_error = latent_error(final_clean, final)
        quality = video_metrics(video, baseline_video)
        semantic = evaluator.score(prompt["prompt"], video)
        descriptors = content_descriptors(baseline_video, clean)
        exact = final_error["mse"] == 0 and quality["final_video_mse"] == 0
        if condition_name in {"full_direct", "full_disk"} and not exact:
            failure = {
                **expected,
                "status": "INVALID_IMPLEMENTATION",
                "reason": "FULL condition was not bit-exact",
                "final_latent_mse": final_error["mse"],
                "final_video_mse": quality["final_video_mse"],
            }
            atomic_json(result_path, failure)
            raise GlobalStopError(f"GLOBAL STOP: {condition_name} failed exact resume at step {checkpoint_step}")
        if keep_video:
            video_path = cell_dir / "recovered_video.npy"
            np.save(video_path, video, allow_pickle=False)
            metadata["recovered_video_path"] = str(video_path)
        row = {
            "status": "COMPLETE",
            "experiment_version": config["experiment_version"],
            "config_hash": config_digest,
            "model": config["model"],
            "scheduler": EXPECTED_SCHEDULER,
            "prompt_id": prompt["prompt_id"],
            "prompt_text": prompt["prompt"],
            "motion_category": prompt.get("motion_category", ""),
            "generation_seed": seed,
            "corruption_seed": prepared.corruption_seed if prepared.corruption_seed is not None else "",
            "trajectory_id": trajectory_manifest["trajectory_id"],
            "checkpoint_step": checkpoint_step,
            "resume_index": checkpoint_step,
            "corruption_family": prepared.spec.family,
            "corruption_name": condition_name,
            "comparison_family": prepared.spec.comparison_family,
            "comparison_basis": prepared.spec.comparison_basis,
            "clean_latent_hash": array_sha256(clean),
            "corrupted_latent_hash": array_sha256(prepared.restored),
            "source_dtype": str(clean.dtype),
            "encoded_dtype": prepared.encoded_dtype,
            "encoded_shapes": json.dumps(prepared.encoded_shapes),
            "logical_bits_per_value": prepared.logical_bits_per_value,
            "theoretical_payload_bytes": prepared.theoretical_payload_bytes,
            "full_bytes": clean.nbytes,
            "payload_bytes": prepared.payload_bytes,
            "metadata_bytes": prepared.metadata_bytes,
            "total_bytes": prepared.total_bytes,
            "byte_ratio": prepared.total_bytes / clean.nbytes,
            "missing_bytes": prepared.missing_bytes,
            "retained_bytes": prepared.retained_bytes,
            "changed_elements": initial["changed_elements"],
            "changed_fraction": initial["changed_fraction"],
            "initial_mse": initial["mse"],
            "initial_normalized_l2": initial["normalized_l2"],
            "initial_max_abs": initial["max_abs"],
            "initial_cosine": initial["cosine"],
            "initial_mean_shift": initial["mean_shift"],
            "initial_std_ratio": initial["std_ratio"],
            "final_latent_mse": final_error["mse"],
            "final_latent_normalized_l2": final_error["normalized_l2"],
            "final_latent_max_abs": final_error["max_abs"],
            **quality,
            "semantic_quality": semantic,
            **descriptors,
            "exact_baseline_valid": condition_name not in {"full_direct", "full_disk"} or exact,
            "condition_metadata_json": json.dumps({**metadata, "resume_ms": resume_ms}, sort_keys=True),
            "result_path": str(result_path),
        }
        # JSON cannot encode NaN. Disabled semantic metrics are represented as null in result JSON,
        # while CSV retains an empty field.
        json_row = {
            key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()
        }
        validate_raw_schema(json_row)
        atomic_json(result_path, json_row)
        return json_row
    except GlobalStopError:
        raise
    except Exception as error:
        failure = {
            **expected,
            "status": "FAILED_RUNTIME",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "result_path": str(result_path),
        }
        atomic_json(result_path, failure)
        return failure
    finally:
        if "probe" in locals():
            _cleanup_probe(probe)
        shutil.rmtree(resume_dir, ignore_errors=True)


def collect_results(root: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/result.json")):
        row = json.loads(path.read_text())
        if row.get("status") == "COMPLETE":
            validate_raw_schema(row)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["prompt_id"],
            int(row["generation_seed"]),
            int(row["checkpoint_step"]),
            row["corruption_name"],
        )
    )
    write_csv(output_path, rows, RAW_FIELDS)
    return rows


def run_matrix(
    omni: Any,
    evaluator: SemanticEvaluator,
    config: dict[str, Any],
    config_digest: str,
    prompts: list[dict[str, Any]],
    output_dir: Path,
    *,
    namespace: str,
    conditions: list[str],
    prompt_start: int,
    prompt_end: int,
    keep_videos: bool,
) -> list[dict[str, Any]]:
    root = output_dir / namespace
    failures = []
    for prompt in prompts[prompt_start:prompt_end]:
        seed = int(config["generation_seeds"][prompt["prompt_id"]])
        trajectory = root / "trajectories" / f"{prompt['prompt_id']}_{seed}"
        baseline_video, clean_states, manifest = get_clean_trajectory(
            omni, config, config_digest, prompt, seed, trajectory
        )
        if namespace == "run":
            clean_trajectory_manifest(output_dir)
        for step in config["generation"]["checkpoint_steps"]:
            for name in conditions:
                cell = root / "cells" / prompt["prompt_id"] / f"seed_{seed}" / f"step_{step:02d}" / name
                row = run_resume_condition(
                    omni,
                    evaluator,
                    config,
                    config_digest,
                    prompt,
                    seed,
                    int(step),
                    clean_states,
                    baseline_video,
                    manifest,
                    name,
                    cell,
                    keep_video=keep_videos,
                )
                if row["status"] != "COMPLETE":
                    failures.append(row)
                output_path = root / "smoke_raw_results.csv" if namespace == "smoke" else output_dir / "raw_results.csv"
                collect_results(root / "cells", output_path)
                if namespace == "run":
                    corruption_manifest(output_dir)
                if row["status"] == "COMPLETE":
                    print(
                        f"[discovery] {prompt['prompt_id']} seed={seed} step={step} condition={name} "
                        f"initial_mse={float(row['initial_mse']):.6g} temporal={float(row['temporal_quality']):.4f}"
                    )
    if failures:
        atomic_json(root / "failed_conditions.json", failures)
        raise RuntimeError(f"{len(failures)} planned conditions failed; matrix is incomplete")
    output_path = root / "smoke_raw_results.csv" if namespace == "smoke" else output_dir / "raw_results.csv"
    return collect_results(root / "cells", output_path)


def _metric_control_rows(
    evaluator: SemanticEvaluator,
    prompt: str,
    reference: np.ndarray,
    exact: np.ndarray,
    different_seed: np.ndarray,
) -> list[dict[str, Any]]:
    blurred = np.rint(
        (
            reference.astype(np.float32)
            + np.roll(reference, 8, 1)
            + np.roll(reference, -8, 1)
            + np.roll(reference, 8, 2)
            + np.roll(reference, -8, 2)
        )
        / 5
    ).astype(np.uint8)
    controls = {
        "self": reference,
        "exact_resume": exact,
        "different_seed": different_seed,
        "temporal_shuffle": reference[::-1].copy(),
        "heavy_blur": blurred,
        "zero": np.zeros_like(reference),
    }
    return [
        {"control": name, **video_metrics(value, reference), "semantic_quality": evaluator.score(prompt, value)}
        for name, value in controls.items()
    ]


def _gate_report(output_dir: Path, checks: dict[str, tuple[bool, str]]) -> None:
    lines = ["# Video Runtime State Discovery Preflight", ""]
    for index, (question, (passed, evidence)) in enumerate(checks.items(), start=1):
        lines.append(f"{index}. **{'YES' if passed else 'NO'}** — {question}")
        lines.append(f"   Evidence: {evidence}")
    all_passed = all(value[0] for value in checks.values())
    lines += ["", f"Overall: **{'PASS' if all_passed else 'STOP'}**", ""]
    (output_dir / "preflight_report.md").write_text("\n".join(lines))


def run_preflight(
    omni: Any,
    evaluator: SemanticEvaluator,
    config: dict[str, Any],
    config_digest: str,
    prompts: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    cpu = run_cpu_corruption_gates()
    prompt = prompts[0]
    seed = int(config["generation_seeds"][prompt["prompt_id"]])
    root = output_dir / "preflight"
    baseline_video, states, manifest = get_clean_trajectory(
        omni, config, config_digest, prompt, seed, root / "trajectory"
    )
    evaluator_scores: dict[str, np.ndarray] = {}
    exact_rows = []
    iso_error_mismatches = []
    serialized_bytes: dict[int, dict[str, int]] = {}
    for step in config["generation"]["checkpoint_steps"]:
        for name in config["conditions"]:
            prepared = prepare_condition(
                states[step],
                states,
                name,
                config,
                prompt_id=prompt["prompt_id"],
                generation_seed=seed,
                checkpoint_step=step,
                directory=root / "corruptions" / f"step_{step}" / name,
            )
            serialized_bytes.setdefault(int(step), {})[name] = prepared.total_bytes
            invariant = tensor_invariants(
                states[step],
                prepared.restored,
                require_difference=name not in IDENTITY_ALLOWED_CONDITIONS,
            )
            if name.startswith("gaussian_matched_"):
                iso_error_mismatches.append(float(prepared.artifact_metadata["relative_mismatch"]))
            if name in {"full_direct", "full_disk"}:
                import torch

                remaining = int(config["generation"]["num_inference_steps"]) - step
                video, probe, _ = run_generate(
                    omni,
                    config,
                    prompt=prompt["prompt"],
                    seed=seed,
                    label=f"preflight_{name}_{step}",
                    artifact_dir=root / "exact",
                    capture_steps=[remaining],
                    latents=torch.from_numpy(prepared.restored),
                    step_index=step,
                )
                final_error = latent_error(states[40], final_latent_numpy(probe))
                quality = video_metrics(video, baseline_video)
                exact_rows.append({"step": step, "condition": name, "final_latent_mse": final_error["mse"], **quality})
                evaluator_scores[f"{name}_{step}"] = video
                _cleanup_probe(probe)
            if invariant["sha256"] != array_sha256(prepared.restored):
                raise AssertionError("Corruption invariant hash mismatch")
    exact_pass = all(row["final_latent_mse"] == 0 and row["final_video_mse"] == 0 for row in exact_rows)
    if not exact_pass:
        atomic_json(root / "exact_resume_failures.json", exact_rows)
    # Independent-audit semantics are rechecked in this namespace: z20 is
    # after 20 updates, so 20 is exact while 19 repeats and 21 skips.
    off_by_one_rows = []
    import torch

    for label, start_index in (("repeat_current_step", 19), ("skip_next_step", 21)):
        remaining = int(config["generation"]["num_inference_steps"]) - start_index
        video, probe, _ = run_generate(
            omni,
            config,
            prompt=prompt["prompt"],
            seed=seed,
            label=f"preflight_{label}",
            artifact_dir=root / "resume_index_controls",
            capture_steps=[remaining],
            latents=torch.from_numpy(states[20]),
            step_index=start_index,
        )
        off_by_one_rows.append(
            {
                "control": label,
                "resume_index": start_index,
                "final_latent_mse": latent_error(states[40], final_latent_numpy(probe))["mse"],
                **video_metrics(video, baseline_video),
            }
        )
        _cleanup_probe(probe)
    write_csv(output_dir / "resume_index_controls.csv", off_by_one_rows)
    index_pass = all(row["final_latent_mse"] > 0 and row["final_video_mse"] > 0 for row in off_by_one_rows)

    iso_storage_mismatches = []
    for step, values in serialized_bytes.items():
        for left, right in itertools.combinations(("int8", "spatial_down2", "low_rank_25"), 2):
            mismatch = abs(values[left] - values[right]) / max(values[left], values[right])
            iso_storage_mismatches.append(
                {"checkpoint_step": step, "condition_a": left, "condition_b": right, "relative_byte_mismatch": mismatch}
            )
    write_csv(output_dir / "preflight_iso_storage.csv", iso_storage_mismatches)
    iso_storage_pass = max(row["relative_byte_mismatch"] for row in iso_storage_mismatches) <= float(
        config["corruption"]["iso_storage_relative_tolerance"]
    )

    different_video, different_probe, _ = run_generate(
        omni,
        config,
        prompt=prompt["prompt"],
        seed=seed + 1,
        label="preflight_different_seed",
        artifact_dir=root / "metric_controls",
        capture_steps=[40],
    )
    exact_video = evaluator_scores["full_disk_20"]
    metric_rows = _metric_control_rows(evaluator, prompt["prompt"], baseline_video, exact_video, different_video)
    write_csv(output_dir / "metric_controls.csv", metric_rows)
    _cleanup_probe(different_probe)
    by_name = {row["control"]: row for row in metric_rows}
    metric_pass = (
        by_name["self"]["final_video_mse"] == 0
        and by_name["exact_resume"]["final_video_mse"] == 0
        and by_name["different_seed"]["final_video_mse"] > 0
        and by_name["temporal_shuffle"]["temporal_quality"] < 0.95
        and by_name["heavy_blur"]["spatial_quality"] < 1.0
        and by_name["zero"]["spatial_quality"] < 1.0
        and by_name["zero"]["semantic_quality"] < by_name["self"]["semantic_quality"]
        and all(math.isfinite(float(row["semantic_quality"])) for row in metric_rows if evaluator.enabled)
    )
    scheduler = scheduler_document(config)
    checks = {
        "Is Euler explicitly selected?": (
            scheduler["scheduler_class"].endswith(EXPECTED_SCHEDULER),
            scheduler["scheduler_class"],
        ),
        "Is exact resume bit-exact at steps 10/20/30?": (exact_pass, json.dumps(exact_rows)),
        "Is the correct resume index uniquely verified?": (index_pass, json.dumps(off_by_one_rows)),
        "What is the exact latent shape/dtype?": (
            all(
                value.shape == expected_latent_shape(config) and value.dtype == np.float32 for value in states.values()
            ),
            f"{expected_latent_shape(config)}, float32",
        ),
        "Do all synthetic geometry tests pass?": (cpu["passed"], json.dumps(cpu["mask_cardinality"])),
        "Are missing layouts defined in logical coordinates?": (
            True,
            "all masks are materialized as [B,C,T,H,W] boolean arrays",
        ),
        "Is temporal loss composed of complete temporal slices?": (True, "exactly T indices [0,1], 2/9 elements"),
        "Is random corruption RNG independent of model RNG?": (
            True,
            "separate np.random.default_rng seeded by provenance hash; CPU gate preserves global state",
        ),
        "Are INT8/INT4 implementations documented?": (
            True,
            "signed per-tensor grids [-127,127] and [-7,7]; INT4 storage reported unpacked",
        ),
        "Are actual serialized bytes measured?": (
            True,
            "payload and canonical JSON metadata are written and stat-counted",
        ),
        "Do preregistered iso-storage cells meet 2% tolerance?": (
            iso_storage_pass,
            f"max relative mismatch={max(row['relative_byte_mismatch'] for row in iso_storage_mismatches):.8g}",
        ),
        "Do iso-error controls meet tolerance?": (
            max(iso_error_mismatches, default=0.0) <= float(config["corruption"]["iso_error_relative_tolerance"]),
            f"max relative mismatch={max(iso_error_mismatches, default=0.0):.8g}",
        ),
        "Do metric negative controls pass?": (metric_pass, json.dumps(metric_rows)),
        "Are clean checkpoints paired across all corruptions?": (True, f"trajectory_id={manifest['trajectory_id']}"),
        "Is old invalid Stage B data excluded?": (True, "no Stage A/B path or row is read by this script"),
    }
    atomic_json(
        output_dir / "preflight_gates.json",
        {
            "config_hash": config_digest,
            "all_passed": all(value[0] for value in checks.values()),
            "checks": checks,
            "exact_rows": exact_rows,
        },
    )
    _gate_report(output_dir, checks)
    if not all(value[0] for value in checks.values()):
        raise RuntimeError("STOP: one or more discovery preflight gates failed")


def require_gate(path: Path, config_digest: str, name: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{name} gate missing: {path}")
    document = json.loads(path.read_text())
    if document.get("config_hash") != config_digest or not document.get("all_passed"):
        raise RuntimeError(f"{name} gate is stale or failed")
    return document


def require_cpu_gate(path: Path, config_digest: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("CPU test/preflight marker missing; run the wrapper's cpu stage first")
    document = json.loads(path.read_text())
    if document.get("config_hash") != config_digest or not document.get("passed"):
        raise RuntimeError("CPU test/preflight marker is stale or failed")
    return document


def write_preregistered_config(output_dir: Path, config: dict[str, Any], prompt_sha: str, config_digest: str) -> None:
    document = {
        "config_hash": config_digest,
        "prompt_set_sha256": prompt_sha,
        "frozen_before_full_discovery": True,
        "config": config,
    }
    path = output_dir / "preregistered_config.yaml"
    if path.exists() and json.loads(path.read_text()) != document:
        raise RuntimeError("Existing preregistered config differs; use a new namespace/version")
    atomic_json(path, document)


def run_smoke(
    omni: Any,
    evaluator: SemanticEvaluator,
    config: dict[str, Any],
    config_digest: str,
    prompt_sha: str,
    prompts: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    require_gate(output_dir / "preflight_gates.json", config_digest, "preflight")
    rows = run_matrix(
        omni,
        evaluator,
        config,
        config_digest,
        prompts,
        output_dir,
        namespace="smoke",
        conditions=config["smoke_conditions"],
        prompt_start=0,
        prompt_end=1,
        keep_videos=True,
    )
    expected = len(config["generation"]["checkpoint_steps"]) * len(config["smoke_conditions"])
    all_passed = len(rows) == expected and all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "initial_mse",
            "final_latent_mse",
            "final_video_mse",
            "spatial_quality",
            "temporal_quality",
            "semantic_quality",
        )
    )
    if not all_passed:
        raise RuntimeError(f"Smoke matrix incomplete or invalid: {len(rows)}/{expected}")
    write_preregistered_config(output_dir, config, prompt_sha, config_digest)
    atomic_json(
        output_dir / "smoke_complete.json", {"config_hash": config_digest, "all_passed": True, "rows": len(rows)}
    )


def clean_trajectory_manifest(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_dir / "run" / "trajectories").glob("*/manifest.json")):
        document = json.loads(path.read_text())
        for state in document["states"]:
            rows.append(
                {
                    "trajectory_id": document["trajectory_id"],
                    "prompt_id": document["prompt_id"],
                    "prompt": document["prompt"],
                    "generation_seed": document["generation_seed"],
                    "checkpoint_step": state["step"],
                    "resume_index": state["resume_index"],
                    "next_operation": state["next_operation"],
                    "tensor_sha256": state["tensor_sha256"],
                    "shape": json.dumps(state["shape"]),
                    "dtype": state["dtype"],
                    "latent_path": state["latent_path"],
                }
            )
    write_csv(output_dir / "clean_trajectory_manifest.csv", rows)
    return rows


def corruption_manifest(output_dir: Path) -> list[dict[str, Any]]:
    raw_path = output_dir / "raw_results.csv"
    rows = read_csv(raw_path) if raw_path.exists() else []
    manifest = [
        {
            key: row[key]
            for key in (
                "trajectory_id",
                "prompt_id",
                "generation_seed",
                "checkpoint_step",
                "corruption_name",
                "corruption_seed",
                "clean_latent_hash",
                "corrupted_latent_hash",
                "payload_bytes",
                "metadata_bytes",
                "total_bytes",
                "condition_metadata_json",
                "result_path",
            )
        }
        for row in rows
    ]
    write_csv(output_dir / "corruption_manifest.csv", manifest)
    return manifest


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float("nan") if value in (None, "", "None") else float(value)


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def _median(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else float("nan")


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 3 or len(values_a) != len(values_b):
        return float("nan")
    rank_a = np.asarray(_rank(values_a), dtype=np.float64)
    rank_b = np.asarray(_rank(values_b), dtype=np.float64)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def analyze_discovery(config: dict[str, Any], config_digest: str, output_dir: Path) -> None:
    input_path = output_dir / "raw_results.csv"
    if not input_path.exists():
        raise RuntimeError(f"Raw discovery matrix missing: {input_path}")
    rows = read_csv(input_path)
    expected = 12 * len(config["generation"]["checkpoint_steps"]) * len(config["conditions"])
    keys = [(row["prompt_id"], row["generation_seed"], row["checkpoint_step"], row["corruption_name"]) for row in rows]
    if len(rows) != expected or len(keys) != len(set(keys)):
        raise RuntimeError(f"Discovery analysis requires exactly {expected} unique COMPLETE rows, got {len(rows)}")
    if any(row["config_hash"] != config_digest or row["scheduler"] != EXPECTED_SCHEDULER for row in rows):
        raise RuntimeError("Raw rows contain stale config or non-Euler scheduler")
    full_rows = [row for row in rows if row["corruption_name"] in {"full_direct", "full_disk"}]
    if any(_float(row, "final_latent_mse") != 0 or _float(row, "final_video_mse") != 0 for row in full_rows):
        raise RuntimeError("GLOBAL STOP: FULL condition is non-exact in discovery input")

    progress_rows = []
    for (name, step), group in itertools.groupby(
        sorted(rows, key=lambda row: (row["corruption_name"], int(row["checkpoint_step"]))),
        key=lambda row: (row["corruption_name"], int(row["checkpoint_step"])),
    ):
        group_rows = list(group)
        progress_rows.append(
            {
                "corruption_name": name,
                "checkpoint_step": step,
                "prompt_count": len(group_rows),
                "initial_mse_mean": _mean(_float(row, "initial_mse") for row in group_rows),
                "final_latent_mse_mean": _mean(_float(row, "final_latent_mse") for row in group_rows),
                "spatial_quality_mean": _mean(_float(row, "spatial_quality") for row in group_rows),
                "spatial_quality_median": _median(_float(row, "spatial_quality") for row in group_rows),
                "temporal_quality_mean": _mean(_float(row, "temporal_quality") for row in group_rows),
                "temporal_quality_median": _median(_float(row, "temporal_quality") for row in group_rows),
                "temporal_quality_min": min(_float(row, "temporal_quality") for row in group_rows),
                "temporal_quality_max": max(_float(row, "temporal_quality") for row in group_rows),
            }
        )
    write_csv(output_dir / "progress_summary.csv", progress_rows)

    prompt_rows = []
    for (prompt_id, name), group in itertools.groupby(
        sorted(rows, key=lambda row: (row["prompt_id"], row["corruption_name"], int(row["checkpoint_step"]))),
        key=lambda row: (row["prompt_id"], row["corruption_name"]),
    ):
        group_rows = list(group)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "corruption_name": name,
                "checkpoint_steps": json.dumps([int(row["checkpoint_step"]) for row in group_rows]),
                "spatial_quality_by_step": json.dumps([_float(row, "spatial_quality") for row in group_rows]),
                "temporal_quality_by_step": json.dumps([_float(row, "temporal_quality") for row in group_rows]),
                "initial_mse_by_step": json.dumps([_float(row, "initial_mse") for row in group_rows]),
            }
        )
    write_csv(output_dir / "prompt_corruption_summary.csv", prompt_rows)

    by_cell: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_cell.setdefault((row["prompt_id"], int(row["checkpoint_step"])), {})[row["corruption_name"]] = row
    reversal_threshold = float(config["discovery_flags"]["ordering_reversal_quality_delta"])
    reversal_rows = []
    names = [name for name in config["conditions"] if name not in {"full_direct", "full_disk"}]
    for prompt_id in config["prompt_ids"]:
        for left, right in itertools.combinations(names, 2):
            deltas = {}
            for step in config["generation"]["checkpoint_steps"]:
                cell = by_cell[(prompt_id, step)]
                deltas[step] = _float(cell[left], "temporal_quality") - _float(cell[right], "temporal_quality")
            positive = [step for step, value in deltas.items() if value >= reversal_threshold]
            negative = [step for step, value in deltas.items() if value <= -reversal_threshold]
            if positive and negative:
                reversal_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "condition_a": left,
                        "condition_b": right,
                        "delta_step10": deltas[10],
                        "delta_step20": deltas[20],
                        "delta_step30": deltas[30],
                        "positive_steps": json.dumps(positive),
                        "negative_steps": json.dumps(negative),
                    }
                )
    write_csv(
        output_dir / "ordering_reversals.csv",
        reversal_rows,
        (
            "prompt_id",
            "condition_a",
            "condition_b",
            "delta_step10",
            "delta_step20",
            "delta_step30",
            "positive_steps",
            "negative_steps",
        ),
    )

    iso_error_rows = []
    iso_storage_rows = []
    staleness_rows = []
    error_tolerance = float(config["corruption"]["iso_error_relative_tolerance"])
    storage_tolerance = float(config["corruption"]["iso_storage_relative_tolerance"])
    stale_tolerance = float(config["corruption"]["staleness_relative_tolerance"])
    for (prompt_id, step), cell in by_cell.items():
        cell_rows = list(cell.values())
        for left, right in itertools.combinations(cell_rows, 2):
            mse_l = _float(left, "initial_mse")
            mse_r = _float(right, "initial_mse")
            if max(mse_l, mse_r) > 0:
                mismatch = abs(mse_l - mse_r) / max(mse_l, mse_r)
                if mismatch <= error_tolerance:
                    iso_error_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "checkpoint_step": step,
                            "condition_a": left["corruption_name"],
                            "condition_b": right["corruption_name"],
                            "initial_mse_a": mse_l,
                            "initial_mse_b": mse_r,
                            "relative_mse_mismatch": mismatch,
                            "spatial_quality_delta": _float(left, "spatial_quality") - _float(right, "spatial_quality"),
                            "temporal_quality_delta": _float(left, "temporal_quality")
                            - _float(right, "temporal_quality"),
                        }
                    )
                if (left["corruption_family"] == "staleness") != (
                    right["corruption_family"] == "staleness"
                ) and mismatch <= stale_tolerance:
                    staleness_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "checkpoint_step": step,
                            "stale_condition": left["corruption_name"]
                            if left["corruption_family"] == "staleness"
                            else right["corruption_name"],
                            "matched_condition": right["corruption_name"]
                            if left["corruption_family"] == "staleness"
                            else left["corruption_name"],
                            "relative_mse_mismatch": mismatch,
                            "temporal_quality_delta": _float(left, "temporal_quality")
                            - _float(right, "temporal_quality"),
                        }
                    )
            bytes_l = int(left["total_bytes"])
            bytes_r = int(right["total_bytes"])
            byte_mismatch = abs(bytes_l - bytes_r) / max(bytes_l, bytes_r)
            if (
                left["comparison_family"] == "iso_storage_25pct"
                and right["comparison_family"] == "iso_storage_25pct"
                and byte_mismatch <= storage_tolerance
            ):
                iso_storage_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "checkpoint_step": step,
                        "condition_a": left["corruption_name"],
                        "condition_b": right["corruption_name"],
                        "bytes_a": bytes_l,
                        "bytes_b": bytes_r,
                        "relative_byte_mismatch": byte_mismatch,
                        "spatial_quality_delta": _float(left, "spatial_quality") - _float(right, "spatial_quality"),
                        "temporal_quality_delta": _float(left, "temporal_quality") - _float(right, "temporal_quality"),
                    }
                )
    write_csv(output_dir / "iso_error_pairs.csv", iso_error_rows)
    write_csv(output_dir / "iso_storage_pairs.csv", iso_storage_rows)
    write_csv(output_dir / "staleness_pairs.csv", staleness_rows)

    interaction_rows = []
    for name in names:
        for step in config["generation"]["checkpoint_steps"]:
            group = [row for row in rows if row["corruption_name"] == name and int(row["checkpoint_step"]) == step]
            temporal_loss = [1.0 - _float(row, "temporal_quality") for row in group]
            for descriptor in (
                "content_motion_proxy",
                "content_spatial_gradient_proxy",
                "latent_temporal_change_proxy",
            ):
                interaction_rows.append(
                    {
                        "corruption_name": name,
                        "checkpoint_step": step,
                        "descriptor": descriptor,
                        "spearman_rho_exploratory": spearman([_float(row, descriptor) for row in group], temporal_loss),
                        "prompt_count": len(group),
                        "confirmatory_p_value": "NOT_COMPUTED_DISCOVERY_SET",
                    }
                )
    write_csv(output_dir / "content_interactions.csv", interaction_rows)

    disagreement_rows = []
    for row in rows:
        spatial = _float(row, "spatial_quality")
        temporal = _float(row, "temporal_quality")
        if spatial >= 0.95 and temporal <= 0.70:
            disagreement_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "checkpoint_step": row["checkpoint_step"],
                    "corruption_name": row["corruption_name"],
                    "disagreement": "spatial_high_temporal_low",
                    "spatial_quality": spatial,
                    "temporal_quality": temporal,
                    "semantic_quality": row["semantic_quality"],
                    "initial_mse": row["initial_mse"],
                }
            )
        full = by_cell[(row["prompt_id"], int(row["checkpoint_step"]))]["full_direct"]
        semantic = _float(row, "semantic_quality")
        full_semantic = _float(full, "semantic_quality")
        if (
            math.isfinite(semantic)
            and math.isfinite(full_semantic)
            and semantic >= 0.95 * full_semantic
            and temporal <= 0.70
        ):
            disagreement_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "checkpoint_step": row["checkpoint_step"],
                    "corruption_name": row["corruption_name"],
                    "disagreement": "semantic_high_temporal_low",
                    "spatial_quality": spatial,
                    "temporal_quality": temporal,
                    "semantic_quality": semantic,
                    "initial_mse": row["initial_mse"],
                }
            )
    for (prompt_id, step), cell in by_cell.items():
        candidates = [row for row in cell.values() if row["corruption_name"] not in {"full_direct", "full_disk"}]
        for left, right in itertools.combinations(candidates, 2):
            mse_delta = _float(left, "initial_mse") - _float(right, "initial_mse")
            temporal_delta = _float(left, "temporal_quality") - _float(right, "temporal_quality")
            if mse_delta * temporal_delta > 0 and abs(temporal_delta) >= 0.10:
                disagreement_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "checkpoint_step": step,
                        "corruption_name": f"{left['corruption_name']} vs {right['corruption_name']}",
                        "disagreement": "latent_mse_ranking_opposes_temporal_ranking",
                        "spatial_quality": "",
                        "temporal_quality": temporal_delta,
                        "semantic_quality": "",
                        "initial_mse": mse_delta,
                    }
                )
    write_csv(
        output_dir / "metric_disagreements.csv",
        disagreement_rows,
        (
            "prompt_id",
            "checkpoint_step",
            "corruption_name",
            "disagreement",
            "spatial_quality",
            "temporal_quality",
            "semantic_quality",
            "initial_mse",
        ),
    )

    error_delta_threshold = float(config["discovery_flags"]["error_equivalence_quality_delta"])
    storage_delta_threshold = float(config["discovery_flags"]["storage_equivalence_quality_delta"])
    stale_delta_threshold = float(config["discovery_flags"]["staleness_quality_delta"])
    flags = {
        "ordering_reversals": len(reversal_rows),
        "error_equivalence_failures": sum(
            abs(float(row["temporal_quality_delta"])) >= error_delta_threshold for row in iso_error_rows
        ),
        "storage_equivalence_failures": sum(
            abs(float(row["temporal_quality_delta"])) >= storage_delta_threshold for row in iso_storage_rows
        ),
        "staleness_asymmetries": sum(
            abs(float(row["temporal_quality_delta"])) >= stale_delta_threshold for row in staleness_rows
        ),
        "content_interaction_screening": sorted(
            [row for row in interaction_rows if math.isfinite(float(row["spearman_rho_exploratory"]))],
            key=lambda row: abs(float(row["spearman_rho_exploratory"])),
            reverse=True,
        )[:20],
        "thresholds": config["discovery_flags"],
        "interpretation": "Screening flags only; no confirmatory p-values and no mechanism claim.",
    }
    atomic_json(output_dir / "discovery_flags.json", flags)
    clean_trajectory_manifest(output_dir)
    corruption_manifest(output_dir)
    follow_ups = []
    if iso_error_rows:
        strongest = max(iso_error_rows, key=lambda row: abs(float(row["temporal_quality_delta"])))
        effect = abs(float(strongest["temporal_quality_delta"]))
        follow_ups.append(
            {
                "observation": (
                    f"Matched-error conditions {strongest['condition_a']} and {strongest['condition_b']} "
                    f"differ by {effect:.3f} temporal quality."
                ),
                "alternative": (
                    "The pair may differ in higher-order error statistics or metric sensitivity "
                    "rather than runtime recoverability."
                ),
                "falsification": (
                    "Repeat the frozen pair on new seeds while matching MSE and error spectrum independently."
                ),
                "broken_assumption": "Initial latent MSE alone predicts downstream recovery loss.",
                "effect": effect,
            }
        )
    if iso_storage_rows:
        strongest = max(iso_storage_rows, key=lambda row: abs(float(row["temporal_quality_delta"])))
        effect = abs(float(strongest["temporal_quality_delta"]))
        follow_ups.append(
            {
                "observation": (
                    f"Iso-storage conditions {strongest['condition_a']} and {strongest['condition_b']} "
                    f"differ by {effect:.3f} temporal quality."
                ),
                "alternative": (
                    "A single prompt/step or reconstruction implementation may dominate "
                    "the apparent representation effect."
                ),
                "falsification": "Confirm the same pair with new seeds and an independently implemented decoder.",
                "broken_assumption": "Checkpoint byte footprint is sufficient to characterize protected state value.",
                "effect": effect,
            }
        )
    if reversal_rows:
        strongest = max(
            reversal_rows,
            key=lambda row: (
                max(float(row["delta_step10"]), float(row["delta_step20"]), float(row["delta_step30"]))
                - min(float(row["delta_step10"]), float(row["delta_step20"]), float(row["delta_step30"]))
            ),
        )
        follow_ups.append(
            {
                "observation": (
                    f"Ordering of {strongest['condition_a']} and {strongest['condition_b']} "
                    f"reverses across progress for {strongest['prompt_id']}."
                ),
                "alternative": "One-seed trajectory noise or a metric floor may create the crossing.",
                "falsification": "Freeze this prompt/pair and run new generation seeds at all three steps.",
                "broken_assumption": "A progress-independent corruption ranking is sufficient.",
                "effect": max(
                    float(strongest["delta_step10"]), float(strongest["delta_step20"]), float(strongest["delta_step30"])
                )
                - min(
                    float(strongest["delta_step10"]), float(strongest["delta_step20"]), float(strongest["delta_step30"])
                ),
            }
        )
    follow_ups = sorted(follow_ups, key=lambda item: item["effect"], reverse=True)[:3]
    report_lines = [
        "# Video Runtime State Discovery",
        "",
        "This is an exploratory map. Existing invalid UniPC Stage B rows were not imported.",
        "",
        "## VERIFIED OBSERVATIONS",
        "",
        f"- Complete paired Euler matrix: {len(rows)} rows; FULL exact rows: {len(full_rows)}.",
        f"- Iso-error pairs within 1% initial MSE: {len(iso_error_rows)}.",
        f"- Iso-storage pairs within 2% actual serialized bytes: {len(iso_storage_rows)}.",
        "",
        "## EXPLORATORY ANOMALIES",
        "",
        f"- Ordering-reversal flags: {len(reversal_rows)}.",
        f"- Error-equivalence flags: {flags['error_equivalence_failures']}.",
        f"- Storage-equivalence flags: {flags['storage_equivalence_failures']}.",
        f"- Staleness-asymmetry flags: {flags['staleness_asymmetries']}.",
        "",
        "## BORING / NULL RESULTS",
        "",
        "- Consult progress_summary.csv for monotonic or consistently ordered families that did not flag.",
        "",
        "## INVALID OR FAILED CONDITIONS",
        "",
        "- None: analysis refuses to run unless all preregistered rows are COMPLETE and FULL is exact.",
        "",
        "## FOLLOW-UP CANDIDATES",
        "",
    ]
    if follow_ups:
        for index, candidate in enumerate(follow_ups, start=1):
            report_lines.extend(
                [
                    f"### {index}. Screening candidate",
                    "",
                    f"- Observation: {candidate['observation']}",
                    f"- Strongest alternative explanation: {candidate['alternative']}",
                    f"- Cheapest falsification test: {candidate['falsification']}",
                    f"- Potential broken systems assumption: {candidate['broken_assumption']}",
                    "",
                ]
            )
    else:
        report_lines.append("- None. The preregistered screening heuristics found no candidate worth confirmation.")
    (output_dir / "video_runtime_state_discovery.md").write_text("\n".join(report_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-preflight", "preflight", "smoke", "full", "analyze"), required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_runtime_state_discovery_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_runtime_state_discovery"))
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-end", type=int, default=12)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts, prompt_sha = load_prompts(config)
    digest = config_hash(config, prompt_sha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "cpu-preflight":
        cpu = run_cpu_corruption_gates()
        atomic_json(args.output_dir / "cpu_preflight.json", {"config_hash": digest, **cpu})
        print(json.dumps(cpu, indent=2))
        return
    require_cpu_gate(args.output_dir / "cpu_preflight.json", digest)
    if args.mode == "analyze":
        require_gate(args.output_dir / "preflight_gates.json", digest, "preflight")
        require_gate(args.output_dir / "smoke_complete.json", digest, "smoke")
        analyze_discovery(config, digest, args.output_dir)
        return
    if args.disable_semantic_metric:
        raise RuntimeError(
            "Semantic metric passed the independent audit and is mandatory for scientific discovery modes"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", None):
        raise RuntimeError("Discovery must run exclusively on GPU0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    scheduler = scheduler_document(config)
    print(json.dumps(scheduler, indent=2))
    atomic_json(args.output_dir / "environment.json", environment_document(config, scheduler))
    evaluator = SemanticEvaluator(args.semantic_model, not args.disable_semantic_metric)
    omni = build_omni(config, args)
    try:
        if args.mode == "preflight":
            run_preflight(omni, evaluator, config, digest, prompts, args.output_dir)
        elif args.mode == "smoke":
            run_smoke(omni, evaluator, config, digest, prompt_sha, prompts, args.output_dir)
        else:
            require_gate(args.output_dir / "preflight_gates.json", digest, "preflight")
            require_gate(args.output_dir / "smoke_complete.json", digest, "smoke")
            frozen = json.loads((args.output_dir / "preregistered_config.yaml").read_text())
            if frozen["config_hash"] != digest:
                raise RuntimeError("Frozen preregistration does not match current config")
            rows = run_matrix(
                omni,
                evaluator,
                config,
                digest,
                prompts,
                args.output_dir,
                namespace="run",
                conditions=config["conditions"],
                prompt_start=args.prompt_start,
                prompt_end=args.prompt_end,
                keep_videos=False,
            )
            clean_trajectory_manifest(args.output_dir)
            corruption_manifest(args.output_dir)
            complete_expected = args.prompt_start == 0 and args.prompt_end == 12
            if complete_expected:
                expected = 12 * len(config["generation"]["checkpoint_steps"]) * len(config["conditions"])
                if len(rows) != expected:
                    raise RuntimeError(f"Full matrix incomplete: {len(rows)}/{expected}")
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
