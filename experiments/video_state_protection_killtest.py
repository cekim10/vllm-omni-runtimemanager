#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import temporal_dimension_killtest_preflight as preflight


DEFAULT_VARIANTS = [
    "full",
    "fp16",
    "int8",
    "spatial_down2",
    "temporal_down2",
    "low_rank_25",
]
QUALITY_TARGETS = [0.95, 0.975, 0.99]
PAIRWISE_ISO_STORAGE_PAIRS = [
    ("int8", "spatial_down2"),
    ("int8", "low_rank_25"),
    ("spatial_down2", "low_rank_25"),
    ("fp16", "temporal_down2"),
]
REPRESENTATION_ORDER = [
    "int8",
    "spatial_down2",
    "low_rank_25",
    "temporal_down2",
    "fp16",
    "full",
]

FRONTIER_RAW_FIELDS = [
    "prompt_set_path",
    "prompt_set_sha256",
    "model",
    "prompt_id",
    "category",
    "prompt",
    "seed",
    "seed_index",
    "content_complexity_score",
    "checkpoint_step",
    "total_steps",
    "progress_fraction",
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
    "resume_latency_ms",
    "recovery_total_ms",
    "exact_equal",
    "video_mse",
    "max_abs_diff",
    "spatial_metric_abs",
    "temporal_shape_composite_abs",
    "temporal_dynamic_composite_abs",
    "temporal_metric_1_abs",
    "temporal_metric_2_abs",
    "motion_metric_abs",
    "motion_energy_ratio_abs",
    "flow_magnitude_cosine_abs",
    "flow_magnitude_ratio_abs",
    "flow_direction_cosine_abs",
    "flicker_similarity_abs",
    "semantic_metric_abs",
    "spatial_vs_full",
    "temporal_shape_vs_full",
    "temporal_dynamic_vs_full",
    "semantic_vs_full",
    "artifact_path",
    "serialized_artifact_dir",
    "serialized_payload_sha256",
    "serialized_metadata_sha256",
    "variant_metadata_json",
]

CHECKPOINT_SIZE_FIELDS = [
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

ISO_STORAGE_FIELDS = [
    "checkpoint_step",
    "pair",
    "variant_a",
    "variant_b",
    "num_rows",
    "mean_bytes_a",
    "mean_bytes_b",
    "mean_relative_byte_mismatch",
    "dynamic_delta_mean",
    "dynamic_delta_median",
    "dynamic_delta_std",
    "dynamic_delta_ci_low",
    "dynamic_delta_ci_high",
    "spatial_delta_mean",
    "spatial_delta_ci_low",
    "spatial_delta_ci_high",
    "semantic_delta_mean",
    "semantic_delta_ci_low",
    "semantic_delta_ci_high",
    "resolved_at_seed_count",
    "resolution_seed_count",
    "needs_extension_to_15",
]

MIN_SAFE_FIELDS = [
    "prompt_id",
    "category",
    "checkpoint_step",
    "quality_target",
    "selected_representation",
    "selected_total_checkpoint_bytes",
    "selected_compression_ratio_vs_full",
    "selection_rule",
    "safe_candidate_count",
    "seed_count",
    "selected_dynamic_mean",
    "selected_dynamic_ci_low",
    "selected_spatial_mean",
    "selected_spatial_ci_low",
    "selected_semantic_mean",
    "selected_semantic_ci_low",
]

SEPARABILITY_FIELDS = [
    "quality_target",
    "policy",
    "num_sessions",
    "representation_accuracy",
    "quality_violation_rate",
    "mean_excess_checkpoint_bytes",
    "median_excess_checkpoint_bytes",
    "mean_oracle_gap_bytes",
]

INTERACTION_FIELDS = [
    "quality_target",
    "prompt_id_a",
    "prompt_id_b",
    "category_a",
    "category_b",
    "step_10_order",
    "step_20_order",
    "step_30_order",
    "has_crossing",
]

BUDGET_FIELDS = [
    "quality_target",
    "session_count",
    "budget_fraction_of_full",
    "trial_index",
    "policy",
    "selected_sessions",
    "safe_selected_sessions",
    "quality_violation_rate",
    "bytes_used",
    "wasted_bytes_above_min_required",
    "oracle_gap_sessions",
]


@dataclass
class PromptProvenance:
    resolved_path: Path
    sha256: str
    prompt_ids: list[str]
    categories: list[str]
    entries: list[dict[str, Any]]


@dataclass
class SerializedRepresentation:
    variant: str
    restored_latent: torch.Tensor
    raw_latent_bytes: int
    encoded_payload_bytes: int
    metadata_bytes: int
    total_checkpoint_bytes: int
    encode_prepare_latency_ms: float
    storage_write_latency_ms: float
    load_read_latency_ms: float
    decode_reconstruction_latency_ms: float
    artifact_dir: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict kill test for video-diffusion state protection.")
    parser.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--prompt-set", default="experiments/video_recovery_prompt_set.json")
    parser.add_argument("--output-dir", default="results/video_state_protection_killtest")
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--prompt-start", type=int, default=0, help="Inclusive prompt index to execute.")
    parser.add_argument("--prompt-end", type=int, help="Exclusive prompt index to execute; defaults to --num-prompts.")
    parser.add_argument("--seed-start", type=int, default=0, help="Inclusive seed index to execute.")
    parser.add_argument("--seed-end", type=int, help="Exclusive seed index to execute; defaults to --num-seeds.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[10, 20, 30])
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--quality-targets", type=float, nargs="*", default=QUALITY_TARGETS)
    parser.add_argument("--iso-storage-tolerance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--budget-session-counts", type=int, nargs="*", default=[3, 8, 20])
    parser.add_argument("--budget-fractions", type=float, nargs="*", default=[0.25, 0.50, 0.75])
    parser.add_argument("--budget-trials", type=int, default=200)
    parser.add_argument("--complexity-bins", type=int, default=3)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--strict-prompt-provenance", action="store_true", default=True)
    parser.add_argument("--disable-semantic-metric", action="store_true")
    parser.add_argument("--semantic-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-frame-count", type=int, default=4)
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split prompt/seed sessions deterministically across this many workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index used with --num-shards.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete prompt/seed sessions already present in frontier_raw.csv.",
    )
    parser.add_argument(
        "--screening-stage",
        action="store_true",
        help="Apply the preregistered 12-prompt x 2-seed hard-kill screening rule.",
    )
    parser.add_argument(
        "--merge-shard-dirs",
        nargs="+",
        help="Merge completed shard output directories and run analysis without loading the model.",
    )
    return parser.parse_args()


def _mean_abs_frame_diff(video: np.ndarray) -> float:
    if len(video) <= 1:
        return 0.0
    diffs = np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32))
    return float(np.mean(diffs) / 255.0)


def _strict_prompt_provenance(path_str: str, expected_count: int) -> PromptProvenance:
    path = Path(path_str)
    candidate_paths = [path]
    if not path.is_absolute():
        candidate_paths.append(REPO_ROOT / path)
        candidate_paths.append(REPO_ROOT / "experiments" / path.name)
    resolved_path = next((candidate.resolve() for candidate in candidate_paths if candidate.exists()), None)
    if resolved_path is None:
        raise FileNotFoundError(f"Prompt set not found: {path_str}")
    raw_text = resolved_path.read_text(encoding="utf-8")
    entries = json.loads(raw_text)
    if not isinstance(entries, list):
        raise ValueError(f"Prompt set must be a list: {resolved_path}")
    entries = entries[:expected_count]
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} prompts, found {len(entries)} in {resolved_path}")
    prompt_ids = [str(entry["prompt_id"]) for entry in entries]
    if any(prompt_id.startswith("preflight_") for prompt_id in prompt_ids):
        raise ValueError(
            f"Prompt provenance check failed: found preflight-style prompt IDs in {resolved_path}: {prompt_ids}"
        )
    categories = [str(entry["motion_category"]) for entry in entries]
    sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return PromptProvenance(
        resolved_path=resolved_path,
        sha256=sha256,
        prompt_ids=prompt_ids,
        categories=categories,
        entries=entries,
    )


def _verify_requested_configuration(args: argparse.Namespace, provenance: PromptProvenance) -> dict[str, Any]:
    checks = {
        "prompt_set_path": str(provenance.resolved_path),
        "prompt_set_sha256": provenance.sha256,
        "prompt_ids": provenance.prompt_ids,
        "all_prompt_ids_are_recovery_ids": all(prompt_id.startswith("recovery_") for prompt_id in provenance.prompt_ids),
        "num_prompts": len(provenance.entries),
        "checkpoint_steps": list(args.checkpoint_steps),
        "num_inference_steps": int(args.num_inference_steps),
        "resolution": [int(args.height), int(args.width)],
        "num_frames": int(args.num_frames),
        "model": args.model,
        "variants": list(args.variants),
        "runs_all_representations": set(args.variants) >= {
            "full",
            "fp16",
            "int8",
            "spatial_down2",
            "temporal_down2",
            "low_rank_25",
        },
    }
    expected = {
        "checkpoint_steps": [10, 20, 30],
        "num_inference_steps": 40,
        "resolution": [480, 832],
        "num_frames": 33,
        "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    }
    if checks["checkpoint_steps"] != expected["checkpoint_steps"]:
        raise ValueError(f"Checkpoint steps mismatch: expected {expected['checkpoint_steps']} got {checks['checkpoint_steps']}")
    if checks["num_inference_steps"] != expected["num_inference_steps"]:
        raise ValueError(
            f"num_inference_steps mismatch: expected {expected['num_inference_steps']} got {checks['num_inference_steps']}"
        )
    if checks["resolution"] != expected["resolution"]:
        raise ValueError(f"Resolution mismatch: expected {expected['resolution']} got {checks['resolution']}")
    if checks["num_frames"] != expected["num_frames"]:
        raise ValueError(f"num_frames mismatch: expected {expected['num_frames']} got {checks['num_frames']}")
    if checks["model"] != expected["model"]:
        raise ValueError(f"Model mismatch: expected {expected['model']} got {checks['model']}")
    if not checks["runs_all_representations"]:
        raise ValueError(f"Missing required variants; got {args.variants}")
    return checks


def _write_preregistered_config(output_dir: Path, args: argparse.Namespace, provenance: PromptProvenance) -> dict[str, Any]:
    path = output_dir / "preregistered_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_provenance = existing.get("prompt_provenance", {})
        if existing_provenance.get("sha256") != provenance.sha256:
            raise ValueError(f"Existing preregistration uses a different prompt set: {path}")
        existing_config = existing.get("experiment_config", {})
        immutable_fields = (
            "model",
            "seed_base",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "checkpoint_steps",
            "guidance_scale",
            "fps",
            "flow_shift",
            "boundary_ratio",
            "enable_cpu_offload",
            "enable_layerwise_offload",
            "enforce_eager",
            "variants",
            "quality_targets",
            "iso_storage_tolerance",
            "semantic_model",
            "semantic_frame_count",
            "semantic_device",
            "disable_semantic_metric",
            "num_shards",
            "shard_index",
        )
        changed = [
            field
            for field in immutable_fields
            if field in existing_config and existing_config[field] != getattr(args, field)
        ]
        if changed:
            raise ValueError(f"Resume configuration differs from preregistration for fields: {changed}")
        return existing

    prereg = {
        "stopping_rule": {
            "screening_num_prompts": 12,
            "screening_num_seeds": 2,
            "screening_stop_only_if_all_conditions_hold": {
                "single_representation_safe_cell_fraction_gte": 0.90,
                "progress_only_representation_accuracy_gte": 0.90,
                "max_abs_iso_storage_quality_delta_lt": 0.05,
            },
            "screening_continuation_rule": (
                "If any screening stop condition is false, execute all primary cells at 5 seeds. "
                "Do not select cells based on favorable screening results."
            ),
            "primary_num_prompts": 12,
            "primary_num_seeds": 5,
            "extension_num_seeds": 15,
            "extend_only_predeclared_iso_storage_cells": True,
            "iso_storage_resolution_rule": (
                "A pairwise iso-storage cell is resolved at n=5 if the 95% bootstrap CI of the paired "
                "temporal_dynamic difference excludes 0. Otherwise it is marked ambiguous and eligible "
                "for extension to n=15; stop at 15 seeds."
            ),
        },
        "quality_targets": QUALITY_TARGETS,
        "quality_target_rule": (
            "A representation satisfies a target only if spatial_vs_full, temporal_dynamic_vs_full, "
            "and semantic_vs_full are all >= target."
        ),
        "iso_storage_tolerance": args.iso_storage_tolerance,
        "content_complexity_score": "mean absolute frame-to-frame pixel difference of the baseline final video",
        "prompt_provenance": {
            "resolved_path": str(provenance.resolved_path),
            "sha256": provenance.sha256,
            "prompt_ids": provenance.prompt_ids,
        },
        "experiment_config": vars(args),
        "execution_environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    path.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    return prereg


def _make_sampling_params(args: argparse.Namespace, seed: int) -> Any:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def _build_probe_sampling_params(
    args: argparse.Namespace,
    *,
    seed: int,
    artifact_dir: Path,
    request_label: str,
    checkpoint_steps: list[int],
) -> Any:
    sampling = _make_sampling_params(args, seed)
    sampling.extra_args = {
        "trajectory_probe": {
            "artifact_dir": str(artifact_dir),
            "request_label": request_label,
            "capture_steps": checkpoint_steps,
            "fps": args.fps,
            "save_decoded": False,
            "save_latents": True,
            "save_mp4": False,
        },
        "flow_shift": args.flow_shift,
    }
    return sampling


def _build_resume_sampling_params(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_step: int,
    latents: torch.Tensor,
) -> Any:
    sampling = _make_sampling_params(args, seed)
    sampling.latents = latents.detach().cpu().clone()
    sampling.step_index = checkpoint_step
    sampling.extra_args = {"flow_shift": args.flow_shift}
    return sampling


def _make_omni(args: argparse.Namespace) -> Any:
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    return Omni(
        model=args.model,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        parallel_config=DiffusionParallelConfig(),
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
    )


def _normalize_output_video(outputs: Any) -> tuple[np.ndarray, Any]:
    from vllm_omni.outputs import OmniRequestOutput

    output = OmniRequestOutput.unwrap_result(outputs)
    video = preflight._normalize_frames(output)
    return video, output


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    stats = []
    for _ in range(samples):
        choice = rng.choice(arr, size=len(arr), replace=True)
        stats.append(float(np.mean(choice)))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def _paired_bootstrap_ci(deltas: list[float], samples: int, seed: int) -> tuple[float, float]:
    return _bootstrap_ci(deltas, samples=samples, seed=seed)


def _metric_stats(values: list[float], samples: int, seed: int) -> dict[str, float]:
    ci_low, ci_high = _bootstrap_ci(values, samples=samples, seed=seed)
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "median": float(np.median(values)) if values else float("nan"),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _serialize_components(
    artifact_dir: Path,
    variant: str,
    components: list[tuple[str, np.ndarray]],
    metadata: dict[str, Any],
) -> tuple[int, int, float, float]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifact_dir / f"{variant}.payload.bin"
    metadata_path = artifact_dir / f"{variant}.metadata.json"

    payload_bytes = 0
    offsets = []
    chunks: list[bytes] = []
    offset = 0
    for name, array in components:
        array_c = np.ascontiguousarray(array)
        chunk = array_c.tobytes(order="C")
        size = len(chunk)
        payload_bytes += size
        offsets.append(
            {
                "name": name,
                "dtype": array_c.dtype.str,
                "shape": list(array_c.shape),
                "offset": offset,
                "nbytes": size,
            }
        )
        chunks.append(chunk)
        offset += size

    meta = dict(metadata)
    meta["components"] = offsets
    meta_bytes_raw = json.dumps(meta, sort_keys=True).encode("utf-8")

    write_start = time.perf_counter()
    metadata_path.write_bytes(meta_bytes_raw)
    with payload_path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
    write_ms = (time.perf_counter() - write_start) * 1000.0

    return payload_bytes, len(meta_bytes_raw), write_ms, len(meta_bytes_raw) + payload_bytes


def _read_serialized_components(artifact_dir: Path, variant: str) -> tuple[dict[str, Any], bytes, float]:
    payload_path = artifact_dir / f"{variant}.payload.bin"
    metadata_path = artifact_dir / f"{variant}.metadata.json"
    read_start = time.perf_counter()
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = payload_path.read_bytes()
    read_ms = (time.perf_counter() - read_start) * 1000.0
    return meta, payload, read_ms


def _extract_array(meta_component: dict[str, Any], payload: bytes) -> np.ndarray:
    start = int(meta_component["offset"])
    end = start + int(meta_component["nbytes"])
    dtype = np.dtype(meta_component["dtype"])
    shape = tuple(int(dim) for dim in meta_component["shape"])
    array = np.frombuffer(payload[start:end], dtype=dtype)
    return array.reshape(shape).copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_serialized_representation(
    meta: dict[str, Any], payload: bytes, variant: str
) -> torch.Tensor:
    if meta.get("variant") != variant:
        raise ValueError(f"Serialized variant mismatch: expected {variant}, got {meta.get('variant')}")
    components = meta.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Serialized metadata has no components")
    expected_offset = 0
    for component in components:
        offset = int(component["offset"])
        nbytes = int(component["nbytes"])
        if offset != expected_offset or nbytes <= 0 or offset + nbytes > len(payload):
            raise ValueError("Serialized component layout is invalid")
        _extract_array(component, payload)
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise ValueError("Serialized payload has unaccounted bytes")

    if variant in {"full", "fp16"}:
        restored = torch.from_numpy(_extract_array(components[0], payload)).to(torch.float32)
    elif variant == "int8":
        q = _extract_array(components[0], payload).astype(np.float32)
        scale = float(_extract_array(components[1], payload).reshape(-1)[0])
        restored = torch.from_numpy(q * scale).to(torch.float32)
    elif variant in {"spatial_down2", "temporal_down2"}:
        loaded = torch.from_numpy(_extract_array(components[0], payload)).to(torch.float32)
        restored = F.interpolate(
            loaded,
            size=tuple(int(value) for value in meta["original_shape"][2:]),
            mode="trilinear",
            align_corners=False,
        ).to(torch.float32)
    elif variant == "low_rank_25":
        u_arr = torch.from_numpy(_extract_array(components[0], payload)).to(torch.float32)
        s_arr = torch.from_numpy(_extract_array(components[1], payload)).to(torch.float32)
        vh_arr = torch.from_numpy(_extract_array(components[2], payload)).to(torch.float32)
        restored = ((u_arr * s_arr.unsqueeze(0)) @ vh_arr).reshape(
            tuple(int(value) for value in meta["original_shape"])
        )
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    if not torch.isfinite(restored).all():
        raise ValueError("Decoded latent contains non-finite values")
    return restored.to(torch.float32).contiguous()


def _serialized_restored_shape(meta: dict[str, Any], variant: str) -> tuple[int, ...]:
    if variant in {"full", "fp16", "int8"}:
        return tuple(int(value) for value in meta["components"][0]["shape"])
    return tuple(int(value) for value in meta["original_shape"])


def _full_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.nelement() * tensor.element_size())


def _encode_representation(latent: torch.Tensor, variant: str, artifact_dir: Path) -> SerializedRepresentation:
    raw_latent = latent.detach().to(torch.float32).cpu().contiguous()
    raw_latent_bytes = _full_bytes(raw_latent)

    encode_start = time.perf_counter()
    if variant == "full":
        components = [("latent", raw_latent.numpy())]
        meta = {"variant": "full"}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        restored = torch.from_numpy(_extract_array(meta_loaded["components"][0], payload)).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    elif variant == "fp16":
        packed = raw_latent.to(torch.float16).numpy()
        components = [("latent_fp16", packed)]
        meta = {"variant": "fp16"}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        restored = torch.from_numpy(_extract_array(meta_loaded["components"][0], payload)).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    elif variant == "int8":
        scale = raw_latent.abs().max().clamp_min(1e-8) / 127.0
        quantized = torch.clamp(torch.round(raw_latent / scale), -127, 127).to(torch.int8)
        components = [("latent_int8", quantized.numpy()), ("scale", np.asarray([float(scale)], dtype=np.float32))]
        meta = {"variant": "int8"}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        q = _extract_array(meta_loaded["components"][0], payload).astype(np.float32)
        s = float(_extract_array(meta_loaded["components"][1], payload).reshape(-1)[0])
        restored = torch.from_numpy(q * s).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    elif variant == "spatial_down2":
        tensor = raw_latent
        _, _, t_dim, h_dim, w_dim = tensor.shape
        down = F.interpolate(
            tensor,
            size=(t_dim, max(1, h_dim // 2), max(1, w_dim // 2)),
            mode="trilinear",
            align_corners=False,
        ).contiguous()
        components = [("latent_down", down.numpy())]
        meta = {"variant": "spatial_down2", "original_shape": list(tensor.shape)}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        loaded = torch.from_numpy(_extract_array(meta_loaded["components"][0], payload)).to(torch.float32)
        restored = F.interpolate(
            loaded,
            size=tuple(int(x) for x in meta_loaded["original_shape"][2:]),
            mode="trilinear",
            align_corners=False,
        ).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    elif variant == "temporal_down2":
        tensor = raw_latent
        _, _, t_dim, h_dim, w_dim = tensor.shape
        down = F.interpolate(
            tensor,
            size=(max(1, math.ceil(t_dim / 2)), h_dim, w_dim),
            mode="trilinear",
            align_corners=False,
        ).contiguous()
        components = [("latent_down", down.numpy())]
        meta = {"variant": "temporal_down2", "original_shape": list(tensor.shape)}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        loaded = torch.from_numpy(_extract_array(meta_loaded["components"][0], payload)).to(torch.float32)
        restored = F.interpolate(
            loaded,
            size=tuple(int(x) for x in meta_loaded["original_shape"][2:]),
            mode="trilinear",
            align_corners=False,
        ).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    elif variant == "low_rank_25":
        tensor = raw_latent
        bsz, channels, t_dim, h_dim, w_dim = tensor.shape
        matrix = tensor.reshape(bsz * channels * t_dim, h_dim * w_dim)
        rank = max(1, min(min(matrix.shape), int(round(min(matrix.shape) * 0.25))))
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        u_k = u[:, :rank].contiguous()
        s_k = s[:rank].contiguous()
        vh_k = vh[:rank, :].contiguous()
        components = [("u", u_k.numpy()), ("s", s_k.numpy()), ("vh", vh_k.numpy())]
        meta = {"variant": "low_rank_25", "original_shape": list(tensor.shape), "rank": rank}
        prep_ms = (time.perf_counter() - encode_start) * 1000.0
        payload_bytes, meta_bytes, write_ms, total_bytes = _serialize_components(artifact_dir, variant, components, meta)
        meta_loaded, payload, read_ms = _read_serialized_components(artifact_dir, variant)
        decode_start = time.perf_counter()
        u_arr = torch.from_numpy(_extract_array(meta_loaded["components"][0], payload)).to(torch.float32)
        s_arr = torch.from_numpy(_extract_array(meta_loaded["components"][1], payload)).to(torch.float32)
        vh_arr = torch.from_numpy(_extract_array(meta_loaded["components"][2], payload)).to(torch.float32)
        reconstructed = (u_arr * s_arr.unsqueeze(0)) @ vh_arr
        restored = reconstructed.reshape(tuple(int(x) for x in meta_loaded["original_shape"])).to(torch.float32)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return SerializedRepresentation(
        variant=variant,
        restored_latent=restored.contiguous(),
        raw_latent_bytes=raw_latent_bytes,
        encoded_payload_bytes=payload_bytes,
        metadata_bytes=meta_bytes,
        total_checkpoint_bytes=total_bytes,
        encode_prepare_latency_ms=prep_ms,
        storage_write_latency_ms=write_ms,
        load_read_latency_ms=read_ms,
        decode_reconstruction_latency_ms=decode_ms,
        artifact_dir=artifact_dir,
        metadata=meta_loaded,
    )


def _load_probe_records_by_step(probe_meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(record["step_index"]): record for record in probe_meta.get("records", [])}


def _run_single_baseline(
    omni: Any,
    args: argparse.Namespace,
    prompt_entry: dict[str, Any],
    seed: int,
    seed_index: int,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], float, float]:
    prompt_text = str(prompt_entry["prompt"])
    request_label = f"{prompt_entry['prompt_id']}_seed{seed}"
    sampling = _build_probe_sampling_params(
        args,
        seed=seed,
        artifact_dir=artifact_dir,
        request_label=request_label,
        checkpoint_steps=list(args.checkpoint_steps),
    )
    start = time.perf_counter()
    outputs = omni.generate({"prompt": prompt_text}, sampling)
    latency_ms = (time.perf_counter() - start) * 1000.0
    video, output = _normalize_output_video(outputs)
    probe_meta_path = output.custom_output.get("trajectory_probe_metadata_path")
    if not probe_meta_path:
        raise ValueError(f"Missing trajectory probe metadata path for {request_label}")
    probe_meta = json.loads(Path(probe_meta_path).read_text(encoding="utf-8"))
    if not str(prompt_entry["prompt_id"]).startswith("recovery_"):
        raise ValueError(f"Unexpected prompt_id provenance: {prompt_entry['prompt_id']}")
    if probe_meta.get("label") != request_label:
        raise ValueError(f"Probe label mismatch: expected {request_label}, got {probe_meta.get('label')}")
    final_path = artifact_dir / f"{request_label}_baseline.mp4"
    preflight._save_video(final_path, video, fps=args.fps)
    frames_path = artifact_dir / f"{request_label}_baseline.npy"
    frames_tmp_path = frames_path.with_suffix(".npy.tmp")
    with frames_tmp_path.open("wb") as handle:
        np.save(handle, video, allow_pickle=False)
    frames_tmp_path.replace(frames_path)
    return video, probe_meta, latency_ms, _mean_abs_frame_diff(video)


def _load_cached_baseline(
    args: argparse.Namespace,
    prompt_entry: dict[str, Any],
    seed: int,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], float, float] | None:
    request_label = f"{prompt_entry['prompt_id']}_seed{seed}"
    frames_path = artifact_dir / f"{request_label}_baseline.npy"
    probe_meta_path = artifact_dir / f"{request_label}_trajectory_probe.json"
    if not frames_path.exists() or not probe_meta_path.exists():
        return None
    try:
        video = np.load(frames_path, allow_pickle=False)
        probe_meta = json.loads(probe_meta_path.read_text(encoding="utf-8"))
        records = _load_probe_records_by_step(probe_meta)
        if probe_meta.get("label") != request_label:
            return None
        if video.ndim != 4 or video.shape[0] != int(args.num_frames):
            return None
        if not np.isfinite(video).all():
            return None
        for checkpoint_step in args.checkpoint_steps:
            record = records.get(int(checkpoint_step))
            if record is None or not record.get("latent_path") or not Path(record["latent_path"]).exists():
                return None
            latent = torch.load(record["latent_path"], map_location="cpu")
            if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or not torch.isfinite(latent).all():
                return None
        return video, probe_meta, 0.0, _mean_abs_frame_diff(video)
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _validate_session_sources(
    args: argparse.Namespace,
    provenance: PromptProvenance,
    rows: list[dict[str, Any]],
    artifact_root: Path,
) -> set[tuple[str, int]]:
    entries = {str(entry["prompt_id"]): entry for entry in provenance.entries}
    rows_by_session = _group_rows(rows, "prompt_id", "seed")
    valid = set()
    for prompt_id, seed_value in sorted(rows_by_session):
        seed = int(seed_value)
        artifact_dir = artifact_root / f"{prompt_id}_seed{seed}"
        cached = _load_cached_baseline(args, entries[prompt_id], seed, artifact_dir)
        if cached is None:
            continue
        _, probe_meta, _, _ = cached
        records = _load_probe_records_by_step(probe_meta)
        source_shapes = {
            step: tuple(torch.load(records[step]["latent_path"], map_location="cpu").shape)
            for step in {int(row["checkpoint_step"]) for row in rows_by_session[(prompt_id, seed_value)]}
        }
        try:
            shapes_match = all(
                _serialized_restored_shape(
                    json.loads(row["variant_metadata_json"]), str(row["variant"])
                )
                == source_shapes[int(row["checkpoint_step"])]
                for row in rows_by_session[(prompt_id, seed_value)]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            shapes_match = False
        if shapes_match:
            valid.add((prompt_id, seed))
    return valid


def _compute_vs_full(rows: list[dict[str, Any]]) -> None:
    by_session: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if row["variant"] == "full":
            by_session[(row["prompt_id"], int(row["seed"]), int(row["checkpoint_step"]))] = row
    for row in rows:
        full_row = by_session[(row["prompt_id"], int(row["seed"]), int(row["checkpoint_step"]))]
        row["spatial_vs_full"] = preflight._safe_relative(float(row["spatial_metric_abs"]), float(full_row["spatial_metric_abs"]))
        row["temporal_shape_vs_full"] = preflight._safe_relative(
            float(row["temporal_shape_composite_abs"]), float(full_row["temporal_shape_composite_abs"])
        )
        row["temporal_dynamic_vs_full"] = preflight._safe_relative(
            float(row["temporal_dynamic_composite_abs"]), float(full_row["temporal_dynamic_composite_abs"])
        )
        row["semantic_vs_full"] = preflight._safe_relative(
            float(row["semantic_metric_abs"]), float(full_row["semantic_metric_abs"])
        )


def _safe_for_target(row: dict[str, Any], target: float) -> bool:
    return (
        float(row["spatial_vs_full"]) >= target
        and float(row["temporal_dynamic_vs_full"]) >= target
        and float(row["semantic_vs_full"]) >= target
    )


def _group_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _aggregate_frontier_summary(rows: list[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grouped = _group_rows(rows, "checkpoint_step", "variant")
    for (step, variant), group in grouped.items():
        prefix = f"step_{step}_{variant}"
        summary[prefix] = {
            "num_rows": len(group),
            "total_checkpoint_bytes": _metric_stats(
                [float(row["total_checkpoint_bytes"]) for row in group], bootstrap_samples, bootstrap_seed
            ),
            "temporal_dynamic_vs_full": _metric_stats(
                [float(row["temporal_dynamic_vs_full"]) for row in group], bootstrap_samples, bootstrap_seed
            ),
            "spatial_vs_full": _metric_stats(
                [float(row["spatial_vs_full"]) for row in group], bootstrap_samples, bootstrap_seed
            ),
            "semantic_vs_full": _metric_stats(
                [float(row["semantic_vs_full"]) for row in group], bootstrap_samples, bootstrap_seed
            ),
        }
    return summary


def _build_checkpoint_size_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({field: row[field] for field in CHECKPOINT_SIZE_FIELDS})
    return out


def _build_iso_storage_rows(
    rows: list[dict[str, Any]],
    tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    seed_counts = {
        prompt_id: len({int(row["seed_index"]) for row in rows if row["prompt_id"] == prompt_id})
        for prompt_id in {row["prompt_id"] for row in rows}
    }
    if len(set(seed_counts.values())) != 1:
        raise ValueError(f"Inconsistent seed counts across prompts: {seed_counts}")
    resolution_seed_count = next(iter(seed_counts.values()), 0)
    by_key = _group_rows(rows, "prompt_id", "seed", "checkpoint_step", "variant")
    output_rows = []
    for step in sorted({int(row["checkpoint_step"]) for row in rows}):
        for variant_a, variant_b in PAIRWISE_ISO_STORAGE_PAIRS:
            deltas_dynamic: list[float] = []
            deltas_spatial: list[float] = []
            deltas_semantic: list[float] = []
            byte_mismatches: list[float] = []
            bytes_a: list[float] = []
            bytes_b: list[float] = []
            for prompt_id, seed in sorted({(row["prompt_id"], int(row["seed"])) for row in rows if int(row["checkpoint_step"]) == step}):
                row_a_list = by_key.get((prompt_id, seed, step, variant_a))
                row_b_list = by_key.get((prompt_id, seed, step, variant_b))
                if not row_a_list or not row_b_list:
                    continue
                row_a = row_a_list[0]
                row_b = row_b_list[0]
                a_bytes = float(row_a["total_checkpoint_bytes"])
                b_bytes = float(row_b["total_checkpoint_bytes"])
                mismatch = abs(a_bytes - b_bytes) / max(a_bytes, b_bytes)
                if mismatch > tolerance:
                    continue
                byte_mismatches.append(mismatch)
                bytes_a.append(a_bytes)
                bytes_b.append(b_bytes)
                deltas_dynamic.append(float(row_a["temporal_dynamic_vs_full"]) - float(row_b["temporal_dynamic_vs_full"]))
                deltas_spatial.append(float(row_a["spatial_vs_full"]) - float(row_b["spatial_vs_full"]))
                deltas_semantic.append(float(row_a["semantic_vs_full"]) - float(row_b["semantic_vs_full"]))
            if not deltas_dynamic:
                output_rows.append(
                    {
                        "checkpoint_step": step,
                        "pair": f"{variant_a}__vs__{variant_b}",
                        "variant_a": variant_a,
                        "variant_b": variant_b,
                        "num_rows": 0,
                        "mean_bytes_a": float("nan"),
                        "mean_bytes_b": float("nan"),
                        "mean_relative_byte_mismatch": float("nan"),
                        "dynamic_delta_mean": float("nan"),
                        "dynamic_delta_median": float("nan"),
                        "dynamic_delta_std": float("nan"),
                        "dynamic_delta_ci_low": float("nan"),
                        "dynamic_delta_ci_high": float("nan"),
                        "spatial_delta_mean": float("nan"),
                        "spatial_delta_ci_low": float("nan"),
                        "spatial_delta_ci_high": float("nan"),
                        "semantic_delta_mean": float("nan"),
                        "semantic_delta_ci_low": float("nan"),
                        "semantic_delta_ci_high": float("nan"),
                        "resolved_at_seed_count": False,
                        "resolution_seed_count": resolution_seed_count,
                        "needs_extension_to_15": False,
                    }
                )
                continue
            dlow, dhigh = _paired_bootstrap_ci(deltas_dynamic, bootstrap_samples, bootstrap_seed)
            slow, shigh = _paired_bootstrap_ci(deltas_spatial, bootstrap_samples, bootstrap_seed + 1)
            semlow, semhigh = _paired_bootstrap_ci(deltas_semantic, bootstrap_samples, bootstrap_seed + 2)
            resolved = not (dlow <= 0.0 <= dhigh)
            output_rows.append(
                {
                    "checkpoint_step": step,
                    "pair": f"{variant_a}__vs__{variant_b}",
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "num_rows": len(deltas_dynamic),
                    "mean_bytes_a": float(np.mean(bytes_a)),
                    "mean_bytes_b": float(np.mean(bytes_b)),
                    "mean_relative_byte_mismatch": float(np.mean(byte_mismatches)),
                    "dynamic_delta_mean": float(np.mean(deltas_dynamic)),
                    "dynamic_delta_median": float(np.median(deltas_dynamic)),
                    "dynamic_delta_std": float(np.std(deltas_dynamic, ddof=1)) if len(deltas_dynamic) > 1 else 0.0,
                    "dynamic_delta_ci_low": dlow,
                    "dynamic_delta_ci_high": dhigh,
                    "spatial_delta_mean": float(np.mean(deltas_spatial)),
                    "spatial_delta_ci_low": slow,
                    "spatial_delta_ci_high": shigh,
                    "semantic_delta_mean": float(np.mean(deltas_semantic)),
                    "semantic_delta_ci_low": semlow,
                    "semantic_delta_ci_high": semhigh,
                    "resolved_at_seed_count": resolved,
                    "resolution_seed_count": resolution_seed_count,
                    "needs_extension_to_15": not resolved,
                }
            )
    return output_rows


def _build_minimum_safe_rows(
    rows: list[dict[str, Any]],
    targets: list[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    output = []
    grouped = _group_rows(rows, "prompt_id", "category", "checkpoint_step", "variant")
    prompts = sorted({(row["prompt_id"], row["category"]) for row in rows})
    steps = sorted({int(row["checkpoint_step"]) for row in rows})
    for prompt_id, category in prompts:
        for step in steps:
            for target in targets:
                candidate_rows = []
                full_bytes_mean = float("nan")
                for variant in DEFAULT_VARIANTS:
                    group = grouped.get((prompt_id, category, step, variant))
                    if not group:
                        continue
                    dyn_vals = [float(row["temporal_dynamic_vs_full"]) for row in group]
                    spat_vals = [float(row["spatial_vs_full"]) for row in group]
                    sem_vals = [float(row["semantic_vs_full"]) for row in group]
                    dyn_stats = _metric_stats(dyn_vals, bootstrap_samples, bootstrap_seed)
                    spat_stats = _metric_stats(spat_vals, bootstrap_samples, bootstrap_seed + 1)
                    sem_stats = _metric_stats(sem_vals, bootstrap_samples, bootstrap_seed + 2)
                    bytes_mean = float(np.mean([float(row["total_checkpoint_bytes"]) for row in group]))
                    safe = dyn_stats["ci_low"] >= target and spat_stats["ci_low"] >= target and sem_stats["ci_low"] >= target
                    candidate_rows.append(
                        {
                            "variant": variant,
                            "bytes_mean": bytes_mean,
                            "safe": safe,
                            "dyn_stats": dyn_stats,
                            "spat_stats": spat_stats,
                            "sem_stats": sem_stats,
                        }
                    )
                    if variant == "full":
                        full_bytes_mean = bytes_mean
                safe_candidates = [candidate for candidate in candidate_rows if candidate["safe"]]
                if safe_candidates:
                    selected = min(safe_candidates, key=lambda candidate: candidate["bytes_mean"])
                    selected_rep = selected["variant"]
                    selected_bytes = selected["bytes_mean"]
                    selected_dyn = selected["dyn_stats"]
                    selected_spat = selected["spat_stats"]
                    selected_sem = selected["sem_stats"]
                else:
                    selected_rep = "none"
                    selected_bytes = float("nan")
                    selected_dyn = {"mean": float("nan"), "ci_low": float("nan")}
                    selected_spat = {"mean": float("nan"), "ci_low": float("nan")}
                    selected_sem = {"mean": float("nan"), "ci_low": float("nan")}
                output.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "checkpoint_step": step,
                        "quality_target": target,
                        "selected_representation": selected_rep,
                        "selected_total_checkpoint_bytes": selected_bytes,
                        "selected_compression_ratio_vs_full": preflight._safe_relative(selected_bytes, full_bytes_mean)
                        if selected_rep != "none"
                        else float("nan"),
                        "selection_rule": "lower_95ci_all_metrics>=target",
                        "safe_candidate_count": len(safe_candidates),
                        "seed_count": len(grouped.get((prompt_id, category, step, "full"), [])),
                        "selected_dynamic_mean": selected_dyn["mean"],
                        "selected_dynamic_ci_low": selected_dyn["ci_low"],
                        "selected_spatial_mean": selected_spat["mean"],
                        "selected_spatial_ci_low": selected_spat["ci_low"],
                        "selected_semantic_mean": selected_sem["mean"],
                        "selected_semantic_ci_low": selected_sem["ci_low"],
                    }
                )
    return output


def _oracle_session_table(rows: list[dict[str, Any]], targets: list[float]) -> list[dict[str, Any]]:
    by_session = _group_rows(rows, "prompt_id", "category", "seed", "seed_index", "checkpoint_step")
    output = []
    for key, group in by_session.items():
        prompt_id, category, seed, seed_index, checkpoint_step = key
        content_complexity = float(group[0]["content_complexity_score"])
        for target in targets:
            safe_rows = [row for row in group if _safe_for_target(row, target)]
            oracle_row = min(safe_rows, key=lambda row: float(row["total_checkpoint_bytes"])) if safe_rows else None
            output.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "seed": int(seed),
                    "seed_index": int(seed_index),
                    "checkpoint_step": int(checkpoint_step),
                    "progress_fraction": float(group[0]["progress_fraction"]),
                    "content_complexity_score": content_complexity,
                    "quality_target": float(target),
                    "oracle_variant": oracle_row["variant"] if oracle_row else "none",
                    "oracle_bytes": float(oracle_row["total_checkpoint_bytes"]) if oracle_row else float("nan"),
                    "rows": group,
                }
            )
    return output


def _complexity_bin(value: float, bin_edges: list[float]) -> int:
    for idx, edge in enumerate(bin_edges[1:], start=1):
        if value <= edge:
            return idx - 1
    return len(bin_edges) - 2


def _fit_simple_policies(oracle_sessions: list[dict[str, Any]], complexity_bins: int) -> dict[str, Any]:
    feasible = [session for session in oracle_sessions if not math.isnan(float(session["oracle_bytes"]))]
    if not feasible:
        return {}
    complexity_values = sorted(float(session["content_complexity_score"]) for session in feasible)
    quantiles = [float(np.quantile(complexity_values, q)) for q in np.linspace(0.0, 1.0, complexity_bins + 1)]
    quantiles[-1] = quantiles[-1] + 1e-9

    progress_only: dict[tuple[float, int], float] = {}
    content_only: dict[tuple[float, int], float] = {}
    additive_lookup: dict[tuple[float, int, int], float] = {}
    target_groups = defaultdict(list)
    for session in feasible:
        target_groups[float(session["quality_target"])].append(session)

    linear_models: dict[float, tuple[float, float, float]] = {}
    for target, group in target_groups.items():
        for step in sorted({int(session["checkpoint_step"]) for session in group}):
            vals = [float(session["oracle_bytes"]) for session in group if int(session["checkpoint_step"]) == step]
            progress_only[(target, step)] = float(np.median(vals))
        for bin_idx in range(complexity_bins):
            vals = [
                float(session["oracle_bytes"])
                for session in group
                if _complexity_bin(float(session["content_complexity_score"]), quantiles) == bin_idx
            ]
            if vals:
                content_only[(target, bin_idx)] = float(np.median(vals))
        for step in sorted({int(session["checkpoint_step"]) for session in group}):
            for bin_idx in range(complexity_bins):
                vals = [
                    float(session["oracle_bytes"])
                    for session in group
                    if int(session["checkpoint_step"]) == step
                    and _complexity_bin(float(session["content_complexity_score"]), quantiles) == bin_idx
                ]
                if vals:
                    additive_lookup[(target, step, bin_idx)] = float(np.median(vals))
        x = np.asarray(
            [
                [1.0, float(session["progress_fraction"]), float(session["content_complexity_score"])]
                for session in group
            ],
            dtype=np.float64,
        )
        y = np.asarray([float(session["oracle_bytes"]) for session in group], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        linear_models[target] = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))

    return {
        "quantiles": quantiles,
        "progress_only": progress_only,
        "content_only": content_only,
        "additive_lookup": additive_lookup,
        "linear_models": linear_models,
    }


def _pick_representation_by_predicted_bytes(rows: list[dict[str, Any]], predicted_bytes: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (float(row["total_checkpoint_bytes"]), REPRESENTATION_ORDER.index(row["variant"])))
    for row in ordered:
        if float(row["total_checkpoint_bytes"]) >= predicted_bytes:
            return row
    return ordered[-1]


def _evaluate_simple_policies(
    oracle_sessions: list[dict[str, Any]],
    policy_fit: dict[str, Any],
    targets: list[float],
) -> list[dict[str, Any]]:
    if not policy_fit:
        return []
    rows = []
    quantiles = policy_fit["quantiles"]
    for target in targets:
        target_sessions = [session for session in oracle_sessions if float(session["quality_target"]) == float(target)]
        if not target_sessions:
            continue
        evaluations: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = defaultdict(list)
        for session in target_sessions:
            rep_rows = session["rows"]
            oracle_variant = session["oracle_variant"]
            if math.isnan(float(session["oracle_bytes"])):
                continue
            step = int(session["checkpoint_step"])
            complexity = float(session["content_complexity_score"])
            bin_idx = _complexity_bin(complexity, quantiles)
            # progress-only
            predicted_bytes = policy_fit["progress_only"][(target, step)]
            chosen = _pick_representation_by_predicted_bytes(rep_rows, predicted_bytes)
            evaluations["progress_only"].append((session, chosen))
            # content-only
            predicted_bytes = policy_fit["content_only"].get((target, bin_idx), policy_fit["progress_only"][(target, step)])
            chosen = _pick_representation_by_predicted_bytes(rep_rows, predicted_bytes)
            evaluations["content_only"].append((session, chosen))
            # additive lookup
            predicted_bytes = policy_fit["additive_lookup"].get(
                (target, step, bin_idx),
                policy_fit["progress_only"][(target, step)],
            )
            chosen = _pick_representation_by_predicted_bytes(rep_rows, predicted_bytes)
            evaluations["additive_lookup"].append((session, chosen))
            # linear
            bias, beta_progress, beta_complexity = policy_fit["linear_models"][target]
            predicted_bytes = bias + beta_progress * float(session["progress_fraction"]) + beta_complexity * complexity
            predicted_bytes = max(predicted_bytes, min(float(row["total_checkpoint_bytes"]) for row in rep_rows))
            chosen = _pick_representation_by_predicted_bytes(rep_rows, predicted_bytes)
            evaluations["linear_progress_complexity"].append((session, chosen))

        for policy_name, pairs in evaluations.items():
            if not pairs:
                continue
            accuracies = []
            violations = []
            excess_bytes = []
            oracle_gaps = []
            for session, chosen in pairs:
                oracle_bytes = float(session["oracle_bytes"])
                accuracies.append(1.0 if chosen["variant"] == session["oracle_variant"] else 0.0)
                safe = _safe_for_target(chosen, float(session["quality_target"]))
                violations.append(0.0 if safe else 1.0)
                chosen_bytes = float(chosen["total_checkpoint_bytes"])
                excess_bytes.append(max(0.0, chosen_bytes - oracle_bytes))
                oracle_gaps.append(chosen_bytes - oracle_bytes)
            rows.append(
                {
                    "quality_target": target,
                    "policy": policy_name,
                    "num_sessions": len(pairs),
                    "representation_accuracy": float(np.mean(accuracies)),
                    "quality_violation_rate": float(np.mean(violations)),
                    "mean_excess_checkpoint_bytes": float(np.mean(excess_bytes)),
                    "median_excess_checkpoint_bytes": float(np.median(excess_bytes)),
                    "mean_oracle_gap_bytes": float(np.mean(oracle_gaps)),
                }
            )
    return rows


def _interaction_crossings(min_safe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target_prompt = _group_rows(min_safe_rows, "quality_target", "prompt_id")
    prompt_meta = {
        row["prompt_id"]: row["category"]
        for row in min_safe_rows
    }
    output = []
    targets = sorted({float(row["quality_target"]) for row in min_safe_rows})
    prompt_ids = sorted({row["prompt_id"] for row in min_safe_rows})
    for target in targets:
        rows_for_target = {
            prompt_id: {int(row["checkpoint_step"]): row for row in by_target_prompt[(target, prompt_id)]}
            for prompt_id in prompt_ids
        }
        for idx, prompt_a in enumerate(prompt_ids):
            for prompt_b in prompt_ids[idx + 1 :]:
                step_orders = {}
                for step in [10, 20, 30]:
                    row_a = rows_for_target[prompt_a].get(step)
                    row_b = rows_for_target[prompt_b].get(step)
                    a_bytes = float(row_a["selected_total_checkpoint_bytes"]) if row_a and row_a["selected_representation"] != "none" else float("inf")
                    b_bytes = float(row_b["selected_total_checkpoint_bytes"]) if row_b and row_b["selected_representation"] != "none" else float("inf")
                    if a_bytes < b_bytes:
                        step_orders[step] = "a_lt_b"
                    elif a_bytes > b_bytes:
                        step_orders[step] = "a_gt_b"
                    else:
                        step_orders[step] = "equal"
                order_values = [step_orders[step] for step in [10, 20, 30]]
                non_equal = [value for value in order_values if value != "equal"]
                has_crossing = len(set(non_equal)) > 1
                output.append(
                    {
                        "quality_target": target,
                        "prompt_id_a": prompt_a,
                        "prompt_id_b": prompt_b,
                        "category_a": prompt_meta[prompt_a],
                        "category_b": prompt_meta[prompt_b],
                        "step_10_order": step_orders[10],
                        "step_20_order": step_orders[20],
                        "step_30_order": step_orders[30],
                        "has_crossing": has_crossing,
                    }
                )
    return output


def _budget_simulation(
    oracle_sessions: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    policy_fit: dict[str, Any],
    budget_session_counts: list[int],
    budget_fractions: list[float],
    budget_trials: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    if not oracle_sessions:
        return []
    best_simple_policy = None
    if separability_rows:
        best_simple_policy = min(
            separability_rows,
            key=lambda row: (float(row["quality_violation_rate"]), float(row["mean_excess_checkpoint_bytes"])),
        )["policy"]
    rng = random.Random(bootstrap_seed)
    by_target = defaultdict(list)
    for session in oracle_sessions:
        if not math.isnan(float(session["oracle_bytes"])):
            by_target[float(session["quality_target"])].append(session)
    output = []
    quantiles = policy_fit.get("quantiles", [])
    for target, sessions in by_target.items():
        for session_count in budget_session_counts:
            for trial_idx in range(budget_trials):
                sampled = [rng.choice(sessions) for _ in range(session_count)]
                full_total = sum(float(next(row for row in session["rows"] if row["variant"] == "full")["total_checkpoint_bytes"]) for session in sampled)
                for budget_fraction in budget_fractions:
                    budget = full_total * budget_fraction
                    for policy in ["oracle", "full", "uniform_int8", "progress_only", best_simple_policy]:
                        if policy is None:
                            continue
                        chosen_rows = []
                        for session in sampled:
                            if policy == "oracle":
                                chosen = next((row for row in session["rows"] if row["variant"] == session["oracle_variant"]), None)
                            elif policy == "full":
                                chosen = next(row for row in session["rows"] if row["variant"] == "full")
                            elif policy == "uniform_int8":
                                chosen = next(row for row in session["rows"] if row["variant"] == "int8")
                            elif policy == "progress_only":
                                predicted = policy_fit["progress_only"][(target, int(session["checkpoint_step"]))]
                                chosen = _pick_representation_by_predicted_bytes(session["rows"], predicted)
                            elif policy == "content_only":
                                bin_idx = _complexity_bin(float(session["content_complexity_score"]), quantiles)
                                predicted = policy_fit["content_only"].get((target, bin_idx), policy_fit["progress_only"][(target, int(session["checkpoint_step"]))])
                                chosen = _pick_representation_by_predicted_bytes(session["rows"], predicted)
                            elif policy == "additive_lookup":
                                bin_idx = _complexity_bin(float(session["content_complexity_score"]), quantiles)
                                predicted = policy_fit["additive_lookup"].get(
                                    (target, int(session["checkpoint_step"]), bin_idx),
                                    policy_fit["progress_only"][(target, int(session["checkpoint_step"]))],
                                )
                                chosen = _pick_representation_by_predicted_bytes(session["rows"], predicted)
                            elif policy == "linear_progress_complexity":
                                bias, beta_progress, beta_complexity = policy_fit["linear_models"][target]
                                predicted = bias + beta_progress * float(session["progress_fraction"]) + beta_complexity * float(session["content_complexity_score"])
                                chosen = _pick_representation_by_predicted_bytes(session["rows"], predicted)
                            else:
                                raise ValueError(f"Unknown policy {policy}")
                            if chosen is not None:
                                chosen_rows.append((session, chosen))
                        chosen_rows.sort(key=lambda item: float(item[1]["total_checkpoint_bytes"]))
                        selected = []
                        used = 0.0
                        for session, chosen in chosen_rows:
                            bytes_used = float(chosen["total_checkpoint_bytes"])
                            if used + bytes_used <= budget:
                                selected.append((session, chosen))
                                used += bytes_used
                        safe_selected = [item for item in selected if _safe_for_target(item[1], target)]
                        violation_rate = 1.0 - (len(safe_selected) / len(selected)) if selected else 0.0
                        wasted = sum(
                            max(0.0, float(chosen["total_checkpoint_bytes"]) - float(session["oracle_bytes"]))
                            for session, chosen in selected
                        )
                        oracle_safe_count = None
                        if policy != "oracle":
                            oracle_candidates = []
                            for session in sampled:
                                chosen = next((row for row in session["rows"] if row["variant"] == session["oracle_variant"]), None)
                                if chosen is not None:
                                    oracle_candidates.append((session, chosen))
                            oracle_candidates.sort(key=lambda item: float(item[1]["total_checkpoint_bytes"]))
                            oracle_selected = []
                            oracle_used = 0.0
                            for session, chosen in oracle_candidates:
                                bytes_used = float(chosen["total_checkpoint_bytes"])
                                if oracle_used + bytes_used <= budget:
                                    oracle_selected.append((session, chosen))
                                    oracle_used += bytes_used
                            oracle_safe_count = len(oracle_selected)
                        output.append(
                            {
                                "quality_target": target,
                                "session_count": session_count,
                                "budget_fraction_of_full": budget_fraction,
                                "trial_index": trial_idx,
                                "policy": policy,
                                "selected_sessions": len(selected),
                                "safe_selected_sessions": len(safe_selected),
                                "quality_violation_rate": violation_rate,
                                "bytes_used": used,
                                "wasted_bytes_above_min_required": wasted,
                                "oracle_gap_sessions": 0 if oracle_safe_count is None else max(0, oracle_safe_count - len(safe_selected)),
                            }
                        )
    return output


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    _write_csv(temp_path, fieldnames, rows)
    temp_path.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["prompt_id"]),
        int(row["seed"]),
        int(row["checkpoint_step"]),
        str(row["variant"]),
    )


def _validate_existing_frontier_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    provenance: PromptProvenance,
) -> list[dict[str, Any]]:
    valid_prompt_ids = set(provenance.prompt_ids)
    valid_steps = {int(step) for step in args.checkpoint_steps}
    valid_variants = set(args.variants)
    prompt_indexes = {prompt_id: index for index, prompt_id in enumerate(provenance.prompt_ids)}
    unique: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("prompt_set_sha256") != provenance.sha256 or row.get("model") != args.model:
            raise ValueError("Existing frontier_raw.csv was produced by a different prompt set or model")
        key = _row_key(row)
        prompt_id, seed, step, variant = key
        if prompt_id not in valid_prompt_ids or step not in valid_steps or variant not in valid_variants:
            continue
        artifact_path = Path(str(row.get("artifact_path", "")))
        serialized_dir = Path(str(row.get("serialized_artifact_dir", "")))
        payload_path = serialized_dir / f"{variant}.payload.bin"
        metadata_path = serialized_dir / f"{variant}.metadata.json"
        try:
            seed_index = int(row["seed_index"])
            expected_seed = int(args.seed_base) + prompt_indexes[prompt_id] * 1000 + seed_index
            payload_size = payload_path.stat().st_size
            metadata_size = metadata_path.stat().st_size
            valid_numeric_fields = (
                seed == expected_seed
                and seed_index >= 0
                and int(row["total_steps"]) == int(args.num_inference_steps)
                and math.isclose(
                    float(row["progress_fraction"]), step / int(args.num_inference_steps)
                )
                and int(row["encoded_payload_bytes"]) == payload_size
                and int(row["metadata_bytes"]) == metadata_size
                and int(row["total_checkpoint_bytes"]) == payload_size + metadata_size
                and int(row["total_checkpoint_bytes"]) > 0
                and float(row["resume_latency_ms"]) > 0.0
                and math.isfinite(float(row["spatial_metric_abs"]))
                and math.isfinite(float(row["temporal_dynamic_composite_abs"]))
                and math.isfinite(float(row["semantic_metric_abs"]))
                and math.isfinite(float(row["spatial_vs_full"]))
                and math.isfinite(float(row["temporal_dynamic_vs_full"]))
                and math.isfinite(float(row["semantic_vs_full"]))
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded_metadata = json.loads(row["variant_metadata_json"])
            if metadata != recorded_metadata:
                valid_numeric_fields = False
            payload = payload_path.read_bytes()
            restored = _decode_serialized_representation(metadata, payload, variant)
            if _full_bytes(restored) != int(row["raw_latent_bytes"]):
                valid_numeric_fields = False
            payload_hash = _sha256_file(payload_path)
            metadata_hash = _sha256_file(metadata_path)
            recorded_payload_hash = str(row.get("serialized_payload_sha256", ""))
            recorded_metadata_hash = str(row.get("serialized_metadata_sha256", ""))
            if recorded_payload_hash and recorded_payload_hash != payload_hash:
                valid_numeric_fields = False
            if recorded_metadata_hash and recorded_metadata_hash != metadata_hash:
                valid_numeric_fields = False
            row["serialized_payload_sha256"] = payload_hash
            row["serialized_metadata_sha256"] = metadata_hash
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
            valid_numeric_fields = False
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size <= 0
            or not payload_path.is_file()
            or not metadata_path.is_file()
            or not valid_numeric_fields
        ):
            continue
        if key in unique:
            raise ValueError(f"Duplicate frontier row in resume data: {key}")
        unique[key] = row
    return list(unique.values())


def _session_belongs_to_shard(prompt_idx: int, seed_index: int, num_shards: int, shard_index: int) -> bool:
    # Assignment remains stable when Stage A grows from 2 to 5 seeds and stays
    # balanced across prompt/seed combinations.
    return (prompt_idx + seed_index) % num_shards == shard_index


def _expected_session_keys(
    provenance: PromptProvenance,
    *,
    seed_count: int,
    seed_base: int,
) -> set[tuple[str, int]]:
    return {
        (str(prompt_entry["prompt_id"]), seed_base + prompt_idx * 1000 + seed_index)
        for prompt_idx, prompt_entry in enumerate(provenance.entries)
        for seed_index in range(seed_count)
    }


def _completed_session_keys(
    rows: list[dict[str, Any]],
    *,
    checkpoint_steps: list[int],
    variants: list[str],
) -> set[tuple[str, int]]:
    expected_cells = {(int(step), variant) for step in checkpoint_steps for variant in variants}
    cells_by_session: dict[tuple[str, int], set[tuple[int, str]]] = defaultdict(set)
    for row in rows:
        cells_by_session[(str(row["prompt_id"]), int(row["seed"]))].add(
            (int(row["checkpoint_step"]), str(row["variant"]))
        )
    return {session for session, cells in cells_by_session.items() if cells == expected_cells}


def _maybe_make_figures(output_dir: Path, frontier_rows: list[dict[str, Any]], min_safe_rows: list[dict[str, Any]], budget_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_paths = []
    # Figure 1
    fig1 = output_dir / "figure1_frontier_progress.png"
    plt.figure(figsize=(8, 5))
    for step in [10, 20, 30]:
        step_rows = [row for row in frontier_rows if int(row["checkpoint_step"]) == step and row["variant"] in DEFAULT_VARIANTS]
        x = [float(row["total_checkpoint_bytes"]) for row in step_rows]
        y = [float(row["temporal_dynamic_vs_full"]) for row in step_rows]
        plt.scatter(x, y, label=f"step {step}", alpha=0.6)
    plt.xlabel("Total checkpoint bytes")
    plt.ylabel("Temporal dynamic vs full")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()
    figure_paths.append(str(fig1))

    # Figure 2
    fig2 = output_dir / "figure2_min_safe_heatmap.png"
    targets = QUALITY_TARGETS
    prompts = sorted({row["prompt_id"] for row in min_safe_rows})
    steps = [10, 20, 30]
    rep_to_idx = {rep: idx for idx, rep in enumerate(["none"] + REPRESENTATION_ORDER)}
    heat = np.full((len(prompts), len(steps)), np.nan, dtype=np.float32)
    target = QUALITY_TARGETS[0]
    for i, prompt_id in enumerate(prompts):
        for j, step in enumerate(steps):
            match = next(
                (
                    row
                    for row in min_safe_rows
                    if row["prompt_id"] == prompt_id and int(row["checkpoint_step"]) == step and float(row["quality_target"]) == target
                ),
                None,
            )
            if match:
                heat[i, j] = rep_to_idx.get(match["selected_representation"], np.nan)
    plt.figure(figsize=(6, max(4, len(prompts) * 0.25)))
    plt.imshow(heat, aspect="auto")
    plt.yticks(range(len(prompts)), prompts, fontsize=6)
    plt.xticks(range(len(steps)), [str(step) for step in steps])
    plt.colorbar()
    plt.title(f"Min safe representation @ target {target}")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()
    figure_paths.append(str(fig2))

    # Figure 3
    fig3 = output_dir / "figure3_budget_oracle_gap.png"
    plt.figure(figsize=(8, 5))
    grouped = _group_rows(budget_rows, "session_count", "policy")
    for session_count in sorted({int(row["session_count"]) for row in budget_rows}):
        for policy in sorted({row["policy"] for row in budget_rows}):
            group = grouped.get((session_count, policy), [])
            if not group:
                continue
            xs = sorted({float(row["budget_fraction_of_full"]) for row in group})
            ys = []
            for x in xs:
                subset = [row for row in group if float(row["budget_fraction_of_full"]) == x]
                ys.append(float(np.mean([float(row["safe_selected_sessions"]) for row in subset])))
            plt.plot(xs, ys, label=f"N={session_count} {policy}")
    plt.xlabel("Budget fraction of full-state total")
    plt.ylabel("Mean safe selected sessions")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(fig3)
    plt.close()
    figure_paths.append(str(fig3))

    # Figure 4
    fig4 = output_dir / "figure4_iso_storage.png"
    plt.figure(figsize=(8, 5))
    for pair in sorted({f"{row['variant_a']} vs {row['variant_b']}" for row in frontier_rows[:0]}):
        del pair
    iso_rows_path = output_dir / "iso_storage_frontier.csv"
    if iso_rows_path.exists():
        iso_rows = list(csv.DictReader(iso_rows_path.open()))
        pairs = sorted({row["pair"] for row in iso_rows if int(row["num_rows"]) > 0})
        for pair in pairs:
            pair_rows = [row for row in iso_rows if row["pair"] == pair and int(row["num_rows"]) > 0]
            xs = [int(row["checkpoint_step"]) for row in pair_rows]
            ys = [float(row["dynamic_delta_mean"]) for row in pair_rows]
            plt.plot(xs, ys, marker="o", label=pair)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Checkpoint step")
    plt.ylabel("Dynamic delta (A - B)")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(fig4)
    plt.close()
    figure_paths.append(str(fig4))
    return figure_paths


def _final_judgment(
    iso_rows: list[dict[str, Any]],
    min_safe_rows: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    resolved_iso = [row for row in iso_rows if int(row["num_rows"]) > 0]
    strong_iso = any(abs(float(row["dynamic_delta_mean"])) >= 0.1 for row in resolved_iso)
    target_rows = [row for row in min_safe_rows if float(row["quality_target"]) == 0.95]
    hetero_count = len({row["selected_representation"] for row in target_rows if row["selected_representation"] != "none"})
    progress_change = 0
    by_prompt = _group_rows(target_rows, "prompt_id")
    for prompt_rows in by_prompt.values():
        reps = [row["selected_representation"] for row in sorted(prompt_rows, key=lambda row: int(row["checkpoint_step"]))]
        if len(set(reps)) > 1:
            progress_change += 1
    simple_gap = 0.0
    if budget_rows:
        oracle = [row for row in budget_rows if row["policy"] == "oracle"]
        non_oracle = [row for row in budget_rows if row["policy"] != "oracle"]
        if oracle and non_oracle:
            oracle_mean = float(np.mean([float(row["safe_selected_sessions"]) for row in oracle]))
            best_simple_mean = max(
                float(np.mean([float(row["safe_selected_sessions"]) for row in non_oracle if row["policy"] == policy]))
                for policy in {row["policy"] for row in non_oracle}
            )
            simple_gap = (oracle_mean - best_simple_mean) / max(oracle_mean, 1e-9)
    details = {
        "strong_iso_storage_effect": strong_iso,
        "heterogeneous_min_safe_representations": hetero_count,
        "prompts_with_progress_dependent_Rstar": progress_change,
        "relative_oracle_gap_vs_best_simple": simple_gap,
    }
    if strong_iso and hetero_count >= 3 and progress_change >= 3 and simple_gap >= 0.15:
        return "STRONG GO", details
    if strong_iso and (hetero_count >= 2 or progress_change >= 2):
        return "CONDITIONAL GO", details
    return "NO-GO", details


def _screening_judgment(
    frontier_rows: list[dict[str, Any]],
    iso_rows: list[dict[str, Any]],
    min_safe_rows: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[str, dict[str, Any]]:
    target_rows = [row for row in min_safe_rows if math.isclose(float(row["quality_target"]), 0.95)]
    total_cells = len({(str(row["prompt_id"]), int(row["checkpoint_step"])) for row in target_rows})
    safe_cells_by_representation: Counter[str] = Counter()
    grouped = _group_rows(frontier_rows, "prompt_id", "checkpoint_step", "variant")
    for (_, _, variant), rows in grouped.items():
        if variant == "full":
            continue
        metric_lows = [
            _metric_stats(
                [float(row[field]) for row in rows],
                samples=bootstrap_samples,
                seed=bootstrap_seed + field_idx,
            )["ci_low"]
            for field_idx, field in enumerate(("spatial_vs_full", "temporal_dynamic_vs_full", "semantic_vs_full"))
        ]
        if all(value >= 0.95 for value in metric_lows):
            safe_cells_by_representation[str(variant)] += 1
    most_common_representation = "none"
    most_common_fraction = 0.0
    if safe_cells_by_representation and total_cells:
        most_common_representation, count = safe_cells_by_representation.most_common(1)[0]
        most_common_fraction = count / total_cells

    progress_rows = [
        row
        for row in separability_rows
        if row["policy"] == "progress_only" and math.isclose(float(row["quality_target"]), 0.95)
    ]
    progress_accuracy = (
        float(progress_rows[0]["representation_accuracy"]) if progress_rows else 0.0
    )
    iso_quality_deltas = []
    for row in iso_rows:
        if int(row["num_rows"]) <= 0:
            continue
        iso_quality_deltas.extend(
            abs(float(row[field]))
            for field in ("dynamic_delta_mean", "spatial_delta_mean", "semantic_delta_mean")
            if math.isfinite(float(row[field]))
        )
    max_abs_iso_quality_delta = max(iso_quality_deltas, default=0.0)

    stop_conditions = {
        "single_representation_safe_fraction_gte_0_90": most_common_fraction >= 0.90,
        "progress_only_accuracy_gte_0_90": progress_accuracy >= 0.90,
        "max_abs_iso_storage_quality_delta_lt_0_05": max_abs_iso_quality_delta < 0.05,
    }
    details = {
        "most_common_safe_representation": most_common_representation,
        "most_common_safe_representation_fraction": most_common_fraction,
        "progress_only_representation_accuracy": progress_accuracy,
        "max_abs_iso_storage_quality_delta": max_abs_iso_quality_delta,
        "stop_conditions": stop_conditions,
    }
    if all(stop_conditions.values()):
        return "SCREENING NO-GO", details
    return "CONTINUE TO 5 SEEDS", details


def _write_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    executed_prompt_count: int,
    executed_seed_count: int,
    prereg: dict[str, Any],
    provenance: PromptProvenance,
    config_checks: dict[str, Any],
    frontier_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    iso_rows: list[dict[str, Any]],
    min_safe_rows: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    judgment: str,
    judgment_details: dict[str, Any],
    figure_paths: list[str],
) -> None:
    lines = [
        "# Video State Protection Kill Test",
        "",
        "## Experimental Configuration",
        "",
        f"- Model: `{args.model}`",
        f"- Prompt set: `{provenance.resolved_path}`",
        f"- Prompt set SHA256: `{provenance.sha256}`",
        f"- Prompt IDs: `{', '.join(provenance.prompt_ids)}`",
        f"- Checkpoint steps: `{args.checkpoint_steps}`",
        f"- Inference steps: `{args.num_inference_steps}`",
        f"- Resolution: `{args.height}x{args.width}`",
        f"- Frames: `{args.num_frames}`",
        f"- Variants: `{' '.join(args.variants)}`",
        f"- Requested prompts: `{args.num_prompts}`",
        f"- Executed prompts: `{executed_prompt_count}`",
        f"- Prompt execution range: `[{args.prompt_start}, {args.prompt_end if args.prompt_end is not None else args.num_prompts})`",
        f"- Requested seeds: `{args.num_seeds}`",
        f"- Executed seeds: `{executed_seed_count}`",
        f"- Seed execution range: `[{args.seed_start}, {args.seed_end if args.seed_end is not None else args.num_seeds})`",
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES')}`",
        f"- Smoke-only mode: `{args.smoke_only}`",
        "",
        "## Verified Prompt Provenance",
        "",
        f"- Recovery prompt provenance verified: `{config_checks['all_prompt_ids_are_recovery_ids']}`",
        f"- Prompt count verified: `{config_checks['num_prompts']}`",
        f"- Checkpoint steps verified: `{config_checks['checkpoint_steps']}`",
        "",
        "## Preregistered Stopping Rule",
        "",
        f"- Screening seeds: `{prereg['stopping_rule']['screening_num_seeds']}`",
        f"- Screening hard-stop conditions: `{json.dumps(prereg['stopping_rule']['screening_stop_only_if_all_conditions_hold'], sort_keys=True)}`",
        f"- Screening continuation: {prereg['stopping_rule']['screening_continuation_rule']}",
        f"- Primary seeds: `{prereg['stopping_rule']['primary_num_seeds']}`",
        f"- Extension seeds: `{prereg['stopping_rule']['extension_num_seeds']}`",
        f"- Iso-storage CI rule: {prereg['stopping_rule']['iso_storage_resolution_rule']}",
        f"- Iso-storage tolerance: `{prereg['iso_storage_tolerance']}`",
        "",
        "## Exact Checkpoint Sizes",
        "",
    ]
    for row in sorted(checkpoint_rows, key=lambda item: (int(item["checkpoint_step"]), item["variant"]))[:12]:
        lines.append(
            f"- step `{row['checkpoint_step']}` / `{row['variant']}`:"
            f" payload=`{int(row['encoded_payload_bytes'])}`"
            f", metadata=`{int(row['metadata_bytes'])}`"
            f", total=`{int(row['total_checkpoint_bytes'])}`"
            f", encode_ms=`{float(row['encode_prepare_latency_ms']):.2f}`"
            f", decode_ms=`{float(row['decode_reconstruction_latency_ms']):.2f}`"
        )
    lines.extend(["", "## Iso-Storage Comparison", ""])
    for row in iso_rows:
        if int(row["num_rows"]) == 0:
            continue
        lines.append(
            f"- step `{row['checkpoint_step']}` / `{row['pair']}`:"
            f" mismatch=`{float(row['mean_relative_byte_mismatch']):.4f}`"
            f", dynamic_delta=`{float(row['dynamic_delta_mean']):.4f}`"
            f", CI=[`{float(row['dynamic_delta_ci_low']):.4f}`, `{float(row['dynamic_delta_ci_high']):.4f}`]"
            f", resolved=`{row['resolved_at_seed_count']}` at seeds=`{row['resolution_seed_count']}`"
        )
    lines.extend(["", "## Minimum Safe Representation", ""])
    sample_rows = [row for row in min_safe_rows if float(row["quality_target"]) == 0.95][:12]
    for row in sample_rows:
        lines.append(
            f"- `{row['prompt_id']}` step `{row['checkpoint_step']}` target `{row['quality_target']}`:"
            f" `{row['selected_representation']}`"
        )
    lines.extend(["", "## Separability Analysis", ""])
    for row in separability_rows:
        lines.append(
            f"- target `{row['quality_target']}` / `{row['policy']}`:"
            f" accuracy=`{float(row['representation_accuracy']):.3f}`"
            f", violation=`{float(row['quality_violation_rate']):.3f}`"
            f", excess_bytes=`{float(row['mean_excess_checkpoint_bytes']):.1f}`"
        )
    lines.extend(["", "## Interaction Crossings", ""])
    crossing_rate = (
        float(np.mean([1.0 if row["has_crossing"] else 0.0 for row in interaction_rows])) if interaction_rows else float("nan")
    )
    lines.append(f"- Crossing rate across prompt pairs: `{crossing_rate:.3f}`")
    lines.extend(["", "## Budget Simulation", ""])
    for policy in sorted({row["policy"] for row in budget_rows}):
        subset = [row for row in budget_rows if row["policy"] == policy]
        lines.append(
            f"- `{policy}`: mean safe selected sessions=`{float(np.mean([float(row['safe_selected_sessions']) for row in subset])):.3f}`, "
            f"mean violation=`{float(np.mean([float(row['quality_violation_rate']) for row in subset])):.3f}`"
        )
    lines.extend(["", "## CONFIRMED", ""])
    lines.append("- Prompt provenance was verified against the recovery prompt set file.")
    lines.append("- Checkpoint size accounting used deterministic serialization with explicit payload and metadata bytes.")
    lines.append("- Recovery quality was measured separately for spatial, temporal/dynamic, and semantic metrics.")
    lines.extend(["", "## INFERRED", ""])
    lines.append("- If iso-storage gaps and minimum-safe representation heterogeneity persist, a finite-budget representation-allocation problem may exist.")
    lines.extend(["", "## UNKNOWN", ""])
    lines.append("- Whether the effect remains strong under the full 12 prompts x 5 seeds execution on the intended server if this report is generated from a smoke or partial run.")
    lines.extend(["", "## Final Judgment", ""])
    lines.append(f"- `{judgment}`")
    lines.append(f"- Evidence: `{json.dumps(judgment_details, sort_keys=True)}`")
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for figure_path in figure_paths:
            lines.append(f"- `{figure_path}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _analyze_and_write_results(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    provenance: PromptProvenance,
    config_checks: dict[str, Any],
    prereg: dict[str, Any],
    frontier_rows: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    _compute_vs_full(frontier_rows)
    frontier_rows.sort(key=_row_key)
    checkpoint_rows = _build_checkpoint_size_rows(frontier_rows)
    iso_rows = _build_iso_storage_rows(
        frontier_rows,
        tolerance=float(args.iso_storage_tolerance),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    min_safe_rows = _build_minimum_safe_rows(
        frontier_rows,
        targets=[float(target) for target in args.quality_targets],
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    oracle_sessions = _oracle_session_table(frontier_rows, [float(target) for target in args.quality_targets])
    policy_fit = _fit_simple_policies(oracle_sessions, complexity_bins=int(args.complexity_bins))
    separability_rows = _evaluate_simple_policies(
        oracle_sessions, policy_fit, [float(target) for target in args.quality_targets]
    )
    interaction_rows = _interaction_crossings(min_safe_rows)
    budget_rows = _budget_simulation(
        oracle_sessions,
        separability_rows,
        policy_fit,
        budget_session_counts=[int(x) for x in args.budget_session_counts],
        budget_fractions=[float(x) for x in args.budget_fractions],
        budget_trials=int(args.budget_trials),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    frontier_summary = _aggregate_frontier_summary(
        frontier_rows, int(args.bootstrap_samples), int(args.bootstrap_seed)
    )

    session_keys = {(str(row["prompt_id"]), int(row["seed"])) for row in frontier_rows}
    executed_prompt_count = len({prompt_id for prompt_id, _ in session_keys})
    executed_seed_count = len({int(row["seed_index"]) for row in frontier_rows})
    if mode == "smoke":
        judgment = "SMOKE-PASS"
        judgment_details = {
            "completed_sessions": len(session_keys),
            "frontier_rows": len(frontier_rows),
            "note": "Smoke run verifies prompt provenance, checkpoint serialization, reconstruction, and resume paths only.",
        }
    elif mode == "shard":
        judgment = "SHARD-COMPLETE"
        judgment_details = {
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "completed_sessions": len(session_keys),
            "note": "Merge every shard before interpreting screening or GO/NO-GO results.",
        }
    elif mode == "partial":
        judgment = "RANGE-COMPLETE"
        judgment_details = {
            "completed_sessions": len(session_keys),
            "note": "Run the remaining prompt/seed ranges in the same output directory before interpreting results.",
        }
    elif mode == "screening":
        judgment, judgment_details = _screening_judgment(
            frontier_rows,
            iso_rows,
            min_safe_rows,
            separability_rows,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    else:
        judgment = "ANALYSIS-PENDING"
        judgment_details = {
            "completed_sessions": len(session_keys),
            "note": (
                "The embedded legacy analysis is descriptive only. Run the preregistered fixed-state "
                "noise-floor experiment and experiments/video_state_protection_analysis.py before "
                "issuing STRONG GO, CONDITIONAL GO, or NO-GO."
            ),
            "invalidated_legacy_outputs": [
                "iso_storage_frontier.csv confidence intervals",
                "minimum_safe_representation.csv n=5 resolution labels",
                "separability_analysis.csv",
                "interaction_crossings.csv",
                "budget_simulation.csv",
            ],
        }

    _write_csv_atomic(output_dir / "frontier_raw.csv", FRONTIER_RAW_FIELDS, frontier_rows)
    _write_csv(output_dir / "checkpoint_sizes.csv", CHECKPOINT_SIZE_FIELDS, checkpoint_rows)
    _write_csv(output_dir / "iso_storage_frontier.csv", ISO_STORAGE_FIELDS, iso_rows)
    _write_csv(output_dir / "minimum_safe_representation.csv", MIN_SAFE_FIELDS, min_safe_rows)
    _write_csv(output_dir / "separability_analysis.csv", SEPARABILITY_FIELDS, separability_rows)
    _write_csv(output_dir / "interaction_crossings.csv", INTERACTION_FIELDS, interaction_rows)
    _write_csv(output_dir / "budget_simulation.csv", BUDGET_FIELDS, budget_rows)

    figure_paths = _maybe_make_figures(output_dir, frontier_rows, min_safe_rows, budget_rows)
    summary = {
        "prompt_provenance": {
            "resolved_path": str(provenance.resolved_path),
            "sha256": provenance.sha256,
            "prompt_ids": provenance.prompt_ids,
            "categories": provenance.categories,
        },
        "config_checks": config_checks,
        "preregistered_config_path": str(output_dir / "preregistered_config.json"),
        "frontier_summary": frontier_summary,
        "judgment": judgment,
        "judgment_details": judgment_details,
        "analysis_validity": (
            "screening_only" if mode == "screening" else "descriptive_legacy_not_for_final_claims"
        ),
        "execution_mode": mode,
        "smoke_only": bool(args.smoke_only),
        "executed_prompt_count": executed_prompt_count,
        "executed_seed_count": executed_seed_count,
        "completed_sessions": len(session_keys),
        "figure_paths": figure_paths,
        "artifact_paths": {
            "frontier_raw_csv": str(output_dir / "frontier_raw.csv"),
            "checkpoint_sizes_csv": str(output_dir / "checkpoint_sizes.csv"),
            "iso_storage_frontier_csv": str(output_dir / "iso_storage_frontier.csv"),
            "minimum_safe_representation_csv": str(output_dir / "minimum_safe_representation.csv"),
            "separability_analysis_csv": str(output_dir / "separability_analysis.csv"),
            "interaction_crossings_csv": str(output_dir / "interaction_crossings.csv"),
            "budget_simulation_csv": str(output_dir / "budget_simulation.csv"),
        },
    }
    (output_dir / "frontier_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(
        output_dir / "video_state_protection_killtest.md",
        args=args,
        executed_prompt_count=executed_prompt_count,
        executed_seed_count=executed_seed_count,
        prereg=prereg,
        provenance=provenance,
        config_checks=config_checks,
        frontier_rows=frontier_rows,
        checkpoint_rows=checkpoint_rows,
        iso_rows=iso_rows,
        min_safe_rows=min_safe_rows,
        separability_rows=separability_rows,
        interaction_rows=interaction_rows,
        budget_rows=budget_rows,
        judgment=judgment,
        judgment_details=judgment_details,
        figure_paths=figure_paths,
    )
    return summary


def run_killtest(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_artifact_dir = output_dir / "artifacts"
    frontier_artifact_dir.mkdir(parents=True, exist_ok=True)

    prompt_count = 1 if args.smoke_only else args.num_prompts
    seed_count = 1 if args.smoke_only else args.num_seeds
    prompt_start = 0 if args.smoke_only else int(args.prompt_start)
    prompt_end = 1 if args.smoke_only else int(args.prompt_end if args.prompt_end is not None else prompt_count)
    seed_start = 0 if args.smoke_only else int(args.seed_start)
    seed_end = 1 if args.smoke_only else int(args.seed_end if args.seed_end is not None else seed_count)
    provenance = _strict_prompt_provenance(args.prompt_set, prompt_count)
    config_checks = _verify_requested_configuration(args, provenance)
    prereg = _write_preregistered_config(output_dir, args, provenance)

    raw_csv_path = output_dir / "frontier_raw.csv"
    if raw_csv_path.exists() and not args.resume:
        raise FileExistsError(f"{raw_csv_path} already exists; pass --resume or use a new output directory")
    frontier_rows = _validate_existing_frontier_rows(
        _read_csv_rows(raw_csv_path) if args.resume else [],
        args=args,
        provenance=provenance,
    )
    represented_sources = {(str(row["prompt_id"]), int(row["seed"])) for row in frontier_rows}
    valid_represented_sources = _validate_session_sources(
        args, provenance, frontier_rows, frontier_artifact_dir
    )
    invalid_represented_sources = represented_sources - valid_represented_sources
    if invalid_represented_sources:
        frontier_rows = [
            row
            for row in frontier_rows
            if (str(row["prompt_id"]), int(row["seed"])) not in invalid_represented_sources
        ]
        print(
            f"[killtest] invalidated {len(invalid_represented_sources)} resumed session(s) "
            "with missing/corrupt source trajectory artifacts",
            flush=True,
        )
    frontier_rows.sort(key=_row_key)
    if args.resume:
        _write_csv_atomic(raw_csv_path, FRONTIER_RAW_FIELDS, frontier_rows)

    omni = _make_omni(args)
    semantic_evaluator = None if args.disable_semantic_metric else preflight.SemanticMetricEvaluator(
        model_name=args.semantic_model,
        device=args.semantic_device,
        frame_count=args.semantic_frame_count,
    )

    try:
        for prompt_idx, prompt_entry in enumerate(provenance.entries):
            if prompt_idx < prompt_start or prompt_idx >= prompt_end:
                continue
            for seed_index in range(seed_count):
                if seed_index < seed_start or seed_index >= seed_end:
                    continue
                if not _session_belongs_to_shard(
                    prompt_idx, seed_index, int(args.num_shards), int(args.shard_index)
                ):
                    continue
                seed = args.seed_base + prompt_idx * 1000 + seed_index
                prompt_id = str(prompt_entry["prompt_id"])
                category = str(prompt_entry["motion_category"])
                prompt_text = str(prompt_entry["prompt"])
                expected_session_cells = {
                    (prompt_id, seed, int(step), variant)
                    for step in args.checkpoint_steps
                    for variant in args.variants
                }
                existing_keys = {_row_key(row) for row in frontier_rows}
                if expected_session_cells <= existing_keys:
                    print(f"[killtest] skip complete prompt_id={prompt_id} seed={seed}", flush=True)
                    continue

                artifact_dir = frontier_artifact_dir / f"{prompt_id}_seed{seed}"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                cached_baseline = _load_cached_baseline(args, prompt_entry, seed, artifact_dir) if args.resume else None
                if cached_baseline is None:
                    baseline_video, probe_meta, _, complexity_score = _run_single_baseline(
                        omni, args, prompt_entry, seed, seed_index, artifact_dir
                    )
                else:
                    baseline_video, probe_meta, _, complexity_score = cached_baseline
                    print(f"[killtest] reuse trajectory prompt_id={prompt_id} seed={seed}", flush=True)
                probe_records = _load_probe_records_by_step(probe_meta)

                for checkpoint_step in args.checkpoint_steps:
                    checkpoint_record = probe_records.get(int(checkpoint_step))
                    if checkpoint_record is None:
                        raise ValueError(f"Missing checkpoint step {checkpoint_step} for {prompt_id} seed {seed}")
                    latent_path = checkpoint_record.get("latent_path")
                    if not latent_path:
                        raise ValueError(f"Missing latent path for {prompt_id} seed {seed} step {checkpoint_step}")
                    exact_latent = torch.load(latent_path, map_location="cpu")
                    checkpoint_cpu_copy_ms = float(checkpoint_record.get("latent_cpu_copy_ms") or 0.0)
                    checkpoint_save_ms = float(checkpoint_record.get("latent_save_ms") or 0.0)
                    checkpoint_protection_ms = checkpoint_cpu_copy_ms + checkpoint_save_ms
                    full_key = (prompt_id, seed, int(checkpoint_step), "full")
                    full_row = next((row for row in frontier_rows if _row_key(row) == full_key), None)
                    full_total_bytes = int(full_row["total_checkpoint_bytes"]) if full_row is not None else None

                    ordered_variants = sorted(args.variants, key=lambda variant: variant != "full")
                    for variant in ordered_variants:
                        row_key = (prompt_id, seed, int(checkpoint_step), variant)
                        if row_key in {_row_key(row) for row in frontier_rows}:
                            print(
                                f"[killtest] skip complete prompt_id={prompt_id} seed={seed} "
                                f"step={checkpoint_step} variant={variant}",
                                flush=True,
                            )
                            continue
                        serialized_dir = artifact_dir / f"serialized_step{int(checkpoint_step):03d}" / variant
                        encoded = _encode_representation(exact_latent, variant, serialized_dir)
                        if variant == "full":
                            full_total_bytes = encoded.total_checkpoint_bytes
                        if full_total_bytes is None:
                            raise ValueError("full representation must complete before compressed variants")
                        variant_sampling = _build_resume_sampling_params(
                            args,
                            seed=seed,
                            checkpoint_step=int(checkpoint_step),
                            latents=encoded.restored_latent,
                        )
                        resume_start = time.perf_counter()
                        outputs = omni.generate({"prompt": prompt_text}, variant_sampling)
                        resume_latency_ms = (time.perf_counter() - resume_start) * 1000.0
                        recovery_total_ms = encoded.load_read_latency_ms + encoded.decode_reconstruction_latency_ms + resume_latency_ms
                        resumed_video, _ = _normalize_output_video(outputs)
                        artifact_path = artifact_dir / f"step{int(checkpoint_step):02d}_{variant}.mp4"
                        preflight._save_video(artifact_path, resumed_video, fps=args.fps)

                        temporal = preflight._temporal_metrics(resumed_video, baseline_video)
                        spatial_abs = preflight._spatial_metric(resumed_video, baseline_video)
                        semantic_abs = (
                            semantic_evaluator.score_video(prompt_text, resumed_video)
                            if semantic_evaluator is not None
                            else float("nan")
                        )
                        video_metrics = {
                            "exact_equal": bool(np.array_equal(resumed_video, baseline_video)),
                            "video_mse": float(np.mean((resumed_video.astype(np.float32) - baseline_video.astype(np.float32)) ** 2)),
                            "max_abs_diff": float(np.max(np.abs(resumed_video.astype(np.float32) - baseline_video.astype(np.float32)))),
                        }
                        frontier_rows.append(
                            {
                                "prompt_set_path": str(provenance.resolved_path),
                                "prompt_set_sha256": provenance.sha256,
                                "model": args.model,
                                "prompt_id": prompt_id,
                                "category": category,
                                "prompt": prompt_text,
                                "seed": seed,
                                "seed_index": seed_index,
                                "content_complexity_score": complexity_score,
                                "checkpoint_step": int(checkpoint_step),
                                "total_steps": int(args.num_inference_steps),
                                "progress_fraction": float(checkpoint_step / args.num_inference_steps),
                                "variant": variant,
                                "raw_latent_bytes": encoded.raw_latent_bytes,
                                "encoded_payload_bytes": encoded.encoded_payload_bytes,
                                "metadata_bytes": encoded.metadata_bytes,
                                "total_checkpoint_bytes": encoded.total_checkpoint_bytes,
                                "compression_ratio_vs_full": float(encoded.total_checkpoint_bytes / max(full_total_bytes, 1)),
                                "checkpoint_cpu_copy_ms": checkpoint_cpu_copy_ms,
                                "checkpoint_save_ms": checkpoint_save_ms,
                                "checkpoint_protection_ms": checkpoint_protection_ms,
                                "encode_prepare_latency_ms": encoded.encode_prepare_latency_ms,
                                "storage_write_latency_ms": encoded.storage_write_latency_ms,
                                "load_read_latency_ms": encoded.load_read_latency_ms,
                                "decode_reconstruction_latency_ms": encoded.decode_reconstruction_latency_ms,
                                "resume_latency_ms": resume_latency_ms,
                                "recovery_total_ms": recovery_total_ms,
                                "exact_equal": video_metrics["exact_equal"],
                                "video_mse": video_metrics["video_mse"],
                                "max_abs_diff": video_metrics["max_abs_diff"],
                                "spatial_metric_abs": spatial_abs,
                                "temporal_shape_composite_abs": temporal["temporal_shape_composite"],
                                "temporal_dynamic_composite_abs": temporal["temporal_dynamic_composite"],
                                "temporal_metric_1_abs": temporal["temporal_metric_1"],
                                "temporal_metric_2_abs": temporal["temporal_metric_2"],
                                "motion_metric_abs": temporal["motion_metric"],
                                "motion_energy_ratio_abs": temporal["motion_energy_ratio"],
                                "flow_magnitude_cosine_abs": temporal["flow_magnitude_cosine"],
                                "flow_magnitude_ratio_abs": temporal["flow_magnitude_ratio"],
                                "flow_direction_cosine_abs": temporal["flow_direction_cosine"],
                                "flicker_similarity_abs": temporal["flicker_similarity"],
                                "semantic_metric_abs": semantic_abs,
                                "spatial_vs_full": float("nan"),
                                "temporal_shape_vs_full": float("nan"),
                                "temporal_dynamic_vs_full": float("nan"),
                                "semantic_vs_full": float("nan"),
                                "artifact_path": str(artifact_path),
                                "serialized_artifact_dir": str(encoded.artifact_dir),
                                "serialized_payload_sha256": _sha256_file(
                                    encoded.artifact_dir / f"{variant}.payload.bin"
                                ),
                                "serialized_metadata_sha256": _sha256_file(
                                    encoded.artifact_dir / f"{variant}.metadata.json"
                                ),
                                "variant_metadata_json": json.dumps(encoded.metadata, sort_keys=True),
                            }
                        )
                        _compute_vs_full(frontier_rows)
                        frontier_rows.sort(key=_row_key)
                        _write_csv_atomic(raw_csv_path, FRONTIER_RAW_FIELDS, frontier_rows)
                        print(
                            f"[killtest] prompt_id={prompt_id} seed={seed} step={checkpoint_step} variant={variant} "
                            f"bytes={encoded.total_checkpoint_bytes} dynamic_abs={temporal['temporal_dynamic_composite']:.4f}",
                            flush=True,
                        )
                completed_keys = {_row_key(row) for row in frontier_rows}
                if not expected_session_cells <= completed_keys:
                    raise RuntimeError(f"Session did not complete all cells: prompt_id={prompt_id} seed={seed}")
                done_path = artifact_dir / "DONE.json"
                done_tmp_path = done_path.with_suffix(".json.tmp")
                done_tmp_path.write_text(
                    json.dumps(
                        {
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "checkpoint_steps": list(args.checkpoint_steps),
                            "variants": list(args.variants),
                            "frontier_rows": len(expected_session_cells),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                done_tmp_path.replace(done_path)
    finally:
        omni.close()

    assigned_sessions = {
        (str(prompt_entry["prompt_id"]), args.seed_base + prompt_idx * 1000 + seed_index)
        for prompt_idx, prompt_entry in enumerate(provenance.entries)
        if prompt_start <= prompt_idx < prompt_end
        for seed_index in range(seed_start, seed_end)
        if _session_belongs_to_shard(prompt_idx, seed_index, int(args.num_shards), int(args.shard_index))
    }
    completed_sessions = _completed_session_keys(
        frontier_rows,
        checkpoint_steps=[int(step) for step in args.checkpoint_steps],
        variants=list(args.variants),
    )
    missing_sessions = sorted(assigned_sessions - completed_sessions)
    if missing_sessions:
        raise RuntimeError(f"Shard finished with incomplete sessions: {missing_sessions}")

    global_expected_sessions = _expected_session_keys(
        provenance,
        seed_count=seed_count,
        seed_base=int(args.seed_base),
    )
    global_complete = global_expected_sessions <= completed_sessions
    if args.smoke_only:
        mode = "smoke"
    elif not global_complete:
        mode = "shard" if int(args.num_shards) > 1 else "partial"
    elif int(args.num_shards) > 1:
        mode = "shard"
    elif args.screening_stage:
        mode = "screening"
    else:
        mode = "full"
    return _analyze_and_write_results(
        args=args,
        output_dir=output_dir,
        provenance=provenance,
        config_checks=config_checks,
        prereg=prereg,
        frontier_rows=frontier_rows,
        mode=mode,
    )


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _strict_prompt_provenance(args.prompt_set, int(args.num_prompts))
    config_checks = _verify_requested_configuration(args, provenance)
    prereg = _write_preregistered_config(output_dir, args, provenance)

    merged_by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for shard_dir_str in args.merge_shard_dirs:
        shard_dir = Path(shard_dir_str)
        shard_csv = shard_dir / "frontier_raw.csv"
        if not shard_csv.exists():
            raise FileNotFoundError(f"Missing shard result: {shard_csv}")
        rows = _validate_existing_frontier_rows(
            _read_csv_rows(shard_csv),
            args=args,
            provenance=provenance,
        )
        for row in rows:
            key = _row_key(row)
            if key in merged_by_key:
                raise ValueError(f"Shard outputs overlap at {key}")
            merged_by_key[key] = row

    frontier_rows = list(merged_by_key.values())
    expected_sessions = _expected_session_keys(
        provenance,
        seed_count=int(args.num_seeds),
        seed_base=int(args.seed_base),
    )
    completed_sessions = _completed_session_keys(
        frontier_rows,
        checkpoint_steps=[int(step) for step in args.checkpoint_steps],
        variants=list(args.variants),
    )
    missing_sessions = sorted(expected_sessions - completed_sessions)
    extra_sessions = sorted(completed_sessions - expected_sessions)
    expected_rows = len(expected_sessions) * len(args.checkpoint_steps) * len(args.variants)
    if missing_sessions or extra_sessions or len(frontier_rows) != expected_rows:
        raise ValueError(
            "Shard merge is incomplete: "
            f"expected_rows={expected_rows}, actual_rows={len(frontier_rows)}, "
            f"missing_sessions={missing_sessions}, extra_sessions={extra_sessions}"
        )

    mode = "screening" if args.screening_stage else "full"
    return _analyze_and_write_results(
        args=args,
        output_dir=output_dir,
        provenance=provenance,
        config_checks=config_checks,
        prereg=prereg,
        frontier_rows=frontier_rows,
        mode=mode,
    )


def main() -> None:
    args = parse_args()
    if any(step < 0 or step >= args.num_inference_steps for step in args.checkpoint_steps):
        raise ValueError(f"checkpoint step out of range for {args.num_inference_steps}: {args.checkpoint_steps}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    prompt_end = args.prompt_end if args.prompt_end is not None else args.num_prompts
    seed_end = args.seed_end if args.seed_end is not None else args.num_seeds
    if not (0 <= args.prompt_start < prompt_end <= args.num_prompts):
        raise ValueError("prompt range must satisfy 0 <= prompt-start < prompt-end <= num-prompts")
    if not (0 <= args.seed_start < seed_end <= args.num_seeds):
        raise ValueError("seed range must satisfy 0 <= seed-start < seed-end <= num-seeds")
    if args.screening_stage and args.num_seeds != 2:
        raise ValueError("--screening-stage requires --num-seeds 2")
    if args.merge_shard_dirs and (args.resume or args.smoke_only or args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("--merge-shard-dirs cannot be combined with run, resume, smoke, or shard options")
    summary = merge_shards(args) if args.merge_shard_dirs else run_killtest(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
