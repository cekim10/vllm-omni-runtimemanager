#!/usr/bin/env python3
"""Preregistered FP16 replay and fixed-BF16-MSE concentration kill test.

This experiment reads validated v3 artifacts but never writes to the v3
namespace.  It deliberately contains no scheduler, placement, or policy
mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_runtime_state_discovery as v3  # noqa: E402


EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_RUNTIME_DTYPE = "torch.bfloat16"
PRIMARY_OPERATOR = "sparse_additive_gaussian"
SECONDARY_OPERATOR = "sparse_multiplicative_replacement"
DECISION_INPUT_FIELDS = frozenset(
    {"prompt_id", "target_name", "active_fraction", "frame_ssim_mean", "relative_mse_mismatch"}
)
RAW_FIELDS = (
    "status",
    "experiment_version",
    "config_hash",
    "provenance_hash",
    "source_raw_sha256",
    "model",
    "scheduler",
    "prompt_id",
    "prompt_text",
    "difficulty",
    "generation_seed",
    "checkpoint_step",
    "resume_index",
    "target_name",
    "target_mse",
    "operator_family",
    "active_fraction",
    "intended_active_fraction",
    "realized_runtime_active_fraction",
    "active_elements",
    "realized_nonzero_elements",
    "total_elements",
    "random_seed",
    "support_seed",
    "perturbation_value_seed",
    "realized_probe_mse",
    "realized_runtime_bf16_mse",
    "relative_mse_mismatch",
    "error_mean",
    "error_std",
    "error_abs_mean",
    "error_abs_max",
    "error_l1",
    "error_l2",
    "error_linf",
    "linf_error",
    "error_kurtosis",
    "error_skewness",
    "zero_fraction_of_error",
    "p50_abs_error",
    "p90_abs_error",
    "p95_abs_error",
    "p99_abs_error",
    "p999_abs_error",
    "clean_abs_max",
    "restored_abs_max",
    "restored_to_clean_absmax_ratio",
    "active_error_rms",
    "exceeds_clean_dynamic_range",
    "replacement_alpha",
    "replacement_alpha_class",
    "runtime_input_hash",
    "clean_checkpoint_hash",
    "final_latent_mse",
    "video_mse",
    "video_psnr",
    "frame_ssim_mean",
    "temporal_delta_mse",
    "temporal_delta_agreement",
    "prompt_clip_score",
    "resume_ms",
    "recovered_final_latent_sha256",
    "recovered_video_sha256",
    "recovered_final_latent_path",
    "recovered_video_path",
    "result_path",
)


class GlobalStopError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    v3.atomic_json(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    v3.write_csv(path, rows, fields)


def read_csv(path: Path) -> list[dict[str, str]]:
    return v3.read_csv(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["model"] != EXPECTED_MODEL:
        raise ValueError("The model is frozen to Wan2.2 T2V A14B")
    scheduler = config["scheduler"]
    if scheduler["name"] != EXPECTED_SCHEDULER or scheduler["sample_solver"] != "euler":
        raise ValueError("The kill test requires explicit WanEulerScheduler")
    if config["generation"]["checkpoint_steps"] != [10, 20, 30]:
        raise ValueError("Source checkpoint metadata must remain [10, 20, 30]")
    concentration = config["concentration"]
    if "preferred_runtime_mse_relative_tolerance" in concentration:
        raise ValueError("Dead preferred MSE tolerance is forbidden")
    if concentration["runtime_mse_relative_tolerance"] != 0.01:
        raise ValueError("The single authoritative runtime-MSE tolerance must remain 0.01")
    if concentration["checkpoint_step"] != 20:
        raise ValueError("Concentration checkpoint must remain step 20")
    if concentration["additive_support_fractions"] != [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        raise ValueError("Primary support sweep changed")
    if concentration["replacement_support_fractions"] != [1.0, 0.2, 0.05]:
        raise ValueError("Secondary support sweep changed")
    if len(concentration["selected_prompts"]) != 9:
        raise ValueError("Exactly nine prompts are preregistered")
    if [row["name"] for row in concentration["targets"]] != ["small", "large"]:
        raise ValueError("Both preregistered MSE regimes are required")
    analysis = config["analysis"]
    frozen = {
        "primary_metric": "frame_ssim_mean",
        "go_endpoint_ssim_difference": 0.10,
        "no_go_endpoint_ssim_difference": 0.05,
        "go_direction_prompt_count": 7,
        "go_monotonic_prompt_count": 6,
        "go_spearman_rho": 0.7,
    }
    for key, expected in frozen.items():
        if analysis[key] != expected:
            raise ValueError(f"Frozen analysis threshold changed: {key}")
    if set(analysis["descriptive_only_metrics"]) & DECISION_INPUT_FIELDS:
        raise ValueError("Descriptive temporal metrics entered decision inputs")
    if set(analysis["auxiliary_only_metrics"]) & DECISION_INPUT_FIELDS:
        raise ValueError("Auxiliary CLIP entered decision inputs")
    expected_primary = 9 * 2 * 7
    expected_secondary = 9 * 2 * 3
    if concentration["primary_condition_count"] != expected_primary:
        raise ValueError("Primary matrix count changed")
    if concentration["secondary_condition_count"] != expected_secondary:
        raise ValueError("Secondary matrix count changed")
    if concentration["full_condition_count"] != expected_primary + expected_secondary:
        raise ValueError("Full matrix count changed")
    return config


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_provenance(config_path: Path, source_raw: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    pipeline = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"
    scheduler = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py"
    status = _git_value("status", "--short") or ""
    diff = _git_value("diff", "--", str(script.relative_to(REPO_ROOT)), str(config_path.relative_to(REPO_ROOT))) or ""
    source_provenance_path = source_raw.parent / "run_provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text())
    document = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "relevant_diff_sha256": sha256_bytes(diff.encode()),
        "experiment_script_sha256": sha256_file(script),
        "config_sha256": sha256_file(config_path),
        "pipeline_wan2_2_sha256": sha256_file(pipeline),
        "scheduler_sha256": sha256_file(scheduler),
        "source_raw_sha256": sha256_file(source_raw),
        "source_config_hash": source_raw.parent.joinpath("preregistered_config.yaml").exists()
        and json.loads(source_raw.parent.joinpath("preregistered_config.yaml").read_text()).get("config_hash"),
        "source_provenance_hash": source_provenance.get("provenance_hash"),
        "source_prompt_set_sha256": source_provenance.get("prompt_set_sha256"),
    }
    document["provenance_hash"] = sha256_bytes(canonical_json(document))
    return document


def environment_document(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for module_name in ("torch", "diffusers", "transformers", "skimage"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            versions[module_name] = f"unavailable: {error}"
    gpu_model = None
    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
    except Exception:
        pass
    source_environment_path = _resolve(config["source_v3"]["root"]) / "environment.json"
    source_environment = json.loads(source_environment_path.read_text())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "cuda_version": cuda_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_model": gpu_model,
        "model": config["model"],
        "requested_model_revision": config.get("model_revision"),
        "source_resolved_model_revision": source_environment.get("resolved_model_revision"),
        "source_v3_result_root": config["source_v3"]["root"],
        "source_raw_results_sha256": config["source_v3"]["raw_results_sha256"],
        "source_config_hash": config["source_v3"]["config_hash"],
        "source_provenance_hash": config["source_v3"]["provenance_hash"],
        "provenance": provenance,
    }


def assert_provenance_matches(path: Path, current: dict[str, Any]) -> None:
    if not path.exists():
        raise GlobalStopError(f"GLOBAL STOP: missing prerequisite provenance {path}")
    stored = json.loads(path.read_text())
    if stored != current:
        raise GlobalStopError("GLOBAL STOP: stale code/config/source provenance cannot authorize this mode")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def source_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    source = config["source_v3"]
    root = _resolve(source["root"])
    raw = root / "raw_results.csv"
    if sha256_file(raw) != source["raw_results_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source v3 raw_results.csv hash mismatch")
    rows = read_csv(raw)
    if len(rows) != int(source["expected_raw_rows"]):
        raise GlobalStopError("GLOBAL STOP: source v3 row count mismatch")
    for row in rows:
        if row["status"] != "COMPLETE" or row["scheduler"] != source["scheduler"]:
            raise GlobalStopError("GLOBAL STOP: invalid source v3 result row")
        if row["config_hash"] != source["config_hash"] or row["provenance_hash"] != source["provenance_hash"]:
            raise GlobalStopError("GLOBAL STOP: source v3 config/provenance mismatch")
        if row["runtime_dtype"] != source["runtime_dtype"]:
            raise GlobalStopError("GLOBAL STOP: source v3 runtime dtype mismatch")
    return rows


def derive_targets(rows: list[dict[str, str]]) -> dict[str, float]:
    mapping = {"small": "int8", "large": "random_missing"}
    return {
        name: statistics.fmean(
            float(row["initial_mse_runtime_dtype"])
            for row in rows
            if row["corruption_name"] == condition
        )
        for name, condition in mapping.items()
    }


def derive_unique_fp16_anomaly_from_v3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the sole non-exact FP16 result from authoritative v3 fields."""
    fp16_rows = [row for row in rows if row.get("corruption_name") == "fp16"]
    nonexact = [
        row
        for row in fp16_rows
        if not (
            float(row["final_latent_mse"]) == 0.0
            and float(row["video_mse"]) == 0.0
        )
    ]
    if len(nonexact) != 1:
        raise GlobalStopError(
            "GLOBAL STOP: trusted v3 must contain exactly one non-exact FP16 row; "
            f"found {len(nonexact)}"
        )
    row = nonexact[0]
    return {
        "prompt_id": row["prompt_id"],
        "generation_seed": int(row["generation_seed"]),
        "checkpoint_step": int(row["checkpoint_step"]),
        "frame_ssim_mean": float(row["frame_ssim_mean"]),
        "clean_latent_hash": row["clean_latent_hash"],
        "corrupted_latent_hash": row.get("corrupted_latent_hash"),
        "final_latent_exact": float(row["final_latent_mse"]) == 0.0,
        "video_exact": float(row["video_mse"]) == 0.0,
        "final_latent_mse": float(row["final_latent_mse"]),
        "video_mse": float(row["video_mse"]),
    }


def validate_fp16_config_matches_unique_v3_anomaly(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    derived = derive_unique_fp16_anomaly_from_v3(rows)
    replay = config["fp16_replay"]
    expected = {
        "prompt_id": replay["prompt_id"],
        "generation_seed": int(replay["generation_seed"]),
        "checkpoint_step": int(replay["checkpoint_step"]),
        "frame_ssim_mean": float(replay["original_frame_ssim_mean"]),
    }
    actual = {key: derived[key] for key in expected}
    if actual != expected:
        raise GlobalStopError(
            "GLOBAL STOP: configured FP16 replay cell does not match the unique trusted-v3 anomaly; "
            f"configured={expected}, derived={actual}"
        )
    return derived


def select_prompts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            row
            for row in rows
            if row["corruption_name"] == "int8" and int(row["checkpoint_step"]) == 20
        ),
        key=lambda row: (float(row["frame_ssim_mean"]), row["prompt_id"]),
    )
    if len(ranked) != 12:
        raise GlobalStopError("GLOBAL STOP: expected 12 source step-20 INT8 rows")
    median = statistics.median(float(row["frame_ssim_mean"]) for row in ranked)
    hard = ranked[:3]
    easy = ranked[-3:]
    used = {row["prompt_id"] for row in hard + easy}
    middle = sorted(
        (row for row in ranked if row["prompt_id"] not in used),
        key=lambda row: (abs(float(row["frame_ssim_mean"]) - median), row["prompt_id"]),
    )[:3]
    selected = []
    for difficulty, group in (("hard", hard), ("middle", middle), ("easy", easy)):
        selected.extend(
            {
                "difficulty": difficulty,
                "prompt_id": row["prompt_id"],
                "generation_seed": int(row["generation_seed"]),
                "source_ssim": float(row["frame_ssim_mean"]),
            }
            for row in group
        )
    return selected


def validate_frozen_source_derivations(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    targets = {row["name"]: float(row["mse"]) for row in config["concentration"]["targets"]}
    if targets != derive_targets(rows):
        raise GlobalStopError("GLOBAL STOP: frozen MSE targets do not match source-v3 derivation")
    if config["concentration"]["selected_prompts"] != select_prompts(rows):
        raise GlobalStopError("GLOBAL STOP: frozen prompt selection does not match source-v3 ranking")
    validate_fp16_config_matches_unique_v3_anomaly(config, rows)


def _trajectory_manifest(config: dict[str, Any], prompt_id: str, seed: int) -> tuple[Path, dict[str, Any]]:
    root = _resolve(config["source_v3"]["root"])
    path = root / "run" / "trajectories" / f"{prompt_id}_{seed}" / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["prompt_id"] != prompt_id or int(manifest["generation_seed"]) != seed:
        raise GlobalStopError("GLOBAL STOP: source trajectory identity mismatch")
    if manifest["config_hash"] != config["source_v3"]["config_hash"]:
        raise GlobalStopError("GLOBAL STOP: source trajectory config mismatch")
    if manifest["provenance_hash"] != config["source_v3"]["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: source trajectory provenance mismatch")
    return path, manifest


@dataclass(frozen=True)
class SourceTrajectory:
    prompt_id: str
    prompt: str
    difficulty: str
    seed: int
    checkpoint_step: int
    clean: np.ndarray
    final_latent: np.ndarray
    video: np.ndarray
    clean_hash: str
    manifest_path: Path
    checkpoint_path: Path


def load_v3_checkpoint_numpy(path: Path, state: dict[str, Any]) -> np.ndarray:
    """Load the validated single-storage FP32 probe, with a torch-free audit path."""
    if state.get("probe_dtype") != "float32" or state.get("dtype") != "float32":
        raise GlobalStopError("GLOBAL STOP: unsupported v3 checkpoint probe dtype")
    expected_bytes = int(state["probe_payload_bytes"])
    expected_shape = tuple(int(value) for value in state["shape"])
    with zipfile.ZipFile(path) as archive:
        storage_names = [
            name
            for name in archive.namelist()
            if "/data/" in name and "/.data/" not in name
        ]
        if len(storage_names) != 1:
            raise GlobalStopError("GLOBAL STOP: v3 checkpoint is not a single contiguous storage")
        raw = archive.read(storage_names[0])
    if len(raw) != expected_bytes or expected_bytes != math.prod(expected_shape) * 4:
        raise GlobalStopError("GLOBAL STOP: v3 checkpoint storage size does not match manifest")
    return np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()


def load_source_trajectory(config: dict[str, Any], prompt_spec: dict[str, Any], step: int) -> SourceTrajectory:
    prompt_id = prompt_spec["prompt_id"]
    seed = int(prompt_spec["generation_seed"])
    manifest_path, manifest = _trajectory_manifest(config, prompt_id, seed)
    by_step = {int(row["step"]): row for row in manifest["states"]}
    if step not in by_step or 40 not in by_step:
        raise GlobalStopError("GLOBAL STOP: source trajectory lacks required checkpoint/final state")
    state = by_step[step]
    checkpoint_path = _resolve(state["latent_path"])
    if sha256_file(checkpoint_path) != state["file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source clean checkpoint file hash mismatch")
    clean = load_v3_checkpoint_numpy(checkpoint_path, state)
    if v3.array_sha256(clean) != state["tensor_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source clean checkpoint tensor hash mismatch")
    if state["runtime_dtype"] != EXPECTED_RUNTIME_DTYPE or int(state["runtime_element_size_bytes"]) != 2:
        raise GlobalStopError("GLOBAL STOP: source checkpoint is not validated BF16 runtime state")
    final_path = _resolve(by_step[40]["latent_path"])
    if sha256_file(final_path) != by_step[40]["file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source final latent file hash mismatch")
    final_latent = load_v3_checkpoint_numpy(final_path, by_step[40])
    video_path = _resolve(manifest["baseline_video_path"])
    if sha256_file(video_path) != manifest["baseline_video_file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source baseline video file hash mismatch")
    video = np.load(video_path, allow_pickle=False)
    if v3.array_sha256(video) != manifest["baseline_video_tensor_sha256"]:
        raise GlobalStopError("GLOBAL STOP: source baseline video tensor hash mismatch")
    return SourceTrajectory(
        prompt_id=prompt_id,
        prompt=manifest["prompt"],
        difficulty=prompt_spec.get("difficulty", "anomaly"),
        seed=seed,
        checkpoint_step=step,
        clean=clean,
        final_latent=final_latent,
        video=video,
        clean_hash=state["tensor_sha256"],
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
    )


def deterministic_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def condition_random_seeds(
    config_digest: str,
    prompt_id: str,
    generation_seed: int,
    checkpoint_step: int,
    target_name: str,
    active_fraction: float,
    operator_family: str,
) -> tuple[int, int]:
    support_seed = deterministic_seed(
        config_digest,
        prompt_id,
        generation_seed,
        checkpoint_step,
        target_name,
        active_fraction,
        "support_coordinates",
    )
    value_seed = deterministic_seed(
        config_digest,
        prompt_id,
        generation_seed,
        checkpoint_step,
        target_name,
        active_fraction,
        operator_family,
        "perturbation_values",
    )
    return support_seed, value_seed


def encode_runtime_bf16(array: np.ndarray) -> np.ndarray:
    """Encode float32 as BF16 with IEEE round-to-nearest-even.

    The server CPU tests independently compare this path with
    ``torch.Tensor.to(torch.bfloat16)`` before any GPU mode is authorized.
    """
    value = np.ascontiguousarray(array, dtype=np.float32)
    bits = value.view(np.uint32)
    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    special_nan = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding
    encoded = (rounded >> np.uint32(16)).astype(np.uint16)
    # Preserve NaN as a quiet BF16 NaN rather than allowing rounding to Inf.
    if np.any(special_nan):
        encoded = encoded.copy()
        encoded[special_nan] |= np.uint16(0x0040)
    return encoded


def decode_runtime_bf16(encoded: np.ndarray) -> np.ndarray:
    if encoded.dtype != np.uint16:
        raise ValueError("BF16 encoded values must use uint16 storage")
    bits = np.ascontiguousarray(encoded).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def cast_runtime_bf16(array: np.ndarray) -> np.ndarray:
    return decode_runtime_bf16(encode_runtime_bf16(array))


def runtime_bf16_mse(clean: np.ndarray, candidate: np.ndarray) -> float:
    lhs = cast_runtime_bf16(clean).astype(np.float64)
    rhs = cast_runtime_bf16(candidate).astype(np.float64)
    return float(np.mean((rhs - lhs) ** 2))


def support_count(total: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("Support fraction must be in (0, 1]")
    return max(1, min(total, int(round(total * fraction))))


def select_support(total: int, fraction: float, seed: int) -> np.ndarray:
    count = support_count(total, fraction)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def _runtime_candidate(clean: np.ndarray, selected: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat = clean.astype(np.float32, copy=True).reshape(-1)
    flat[selected] = values.astype(np.float32, copy=False)
    encoded = encode_runtime_bf16(flat.reshape(clean.shape))
    return decode_runtime_bf16(encoded)


def _search_scale(
    clean: np.ndarray,
    selected: np.ndarray,
    target_mse: float,
    value_at_scale: Any,
    *,
    relative_tolerance: float,
    max_iterations: int = 100,
) -> tuple[np.ndarray, float, float]:
    if target_mse <= 0:
        raise ValueError("Target MSE must be positive")

    def evaluate(scale: float) -> tuple[np.ndarray, float]:
        candidate = _runtime_candidate(clean, selected, value_at_scale(scale))
        mse = runtime_bf16_mse(clean, candidate)
        return candidate, mse

    low = 0.0
    high = 1.0
    best_candidate, best_mse = evaluate(low)
    best_scale = low
    for _ in range(80):
        candidate, realized = evaluate(high)
        if abs(realized - target_mse) < abs(best_mse - target_mse):
            best_candidate, best_mse, best_scale = candidate, realized, high
        if realized >= target_mse:
            break
        high *= 2.0
    else:
        raise RuntimeError("Runtime-BF16 MSE matcher failed to bracket the target")
    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        candidate, realized = evaluate(midpoint)
        if abs(realized - target_mse) < abs(best_mse - target_mse):
            best_candidate, best_mse, best_scale = candidate, realized, midpoint
        mismatch = abs(realized - target_mse) / target_mse
        if mismatch <= relative_tolerance:
            return candidate, realized, midpoint
        if realized < target_mse:
            low = midpoint
        else:
            high = midpoint
    mismatch = abs(best_mse - target_mse) / target_mse
    if not math.isfinite(best_mse) or mismatch > relative_tolerance:
        raise RuntimeError(
            "Runtime-BF16 MSE matcher failed closed: "
            f"target={target_mse}, realized={best_mse}, mismatch={mismatch}, "
            f"tolerance={relative_tolerance}"
        )
    return best_candidate, best_mse, best_scale


def construct_fixed_mse_error(
    clean: np.ndarray,
    *,
    target_mse: float,
    active_fraction: float,
    operator_family: str,
    support_seed: int,
    perturbation_value_seed: int,
    relative_tolerance: float,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime_clean = cast_runtime_bf16(clean)
    flat = runtime_clean.reshape(-1)
    selected = select_support(flat.size, active_fraction, support_seed)
    rng = np.random.default_rng(perturbation_value_seed)
    if operator_family == PRIMARY_OPERATOR:
        direction = rng.standard_normal(selected.size).astype(np.float32)
        if int(np.count_nonzero(direction)) != selected.size:
            raise AssertionError("Additive pre-cast support is not exact")
        base = flat[selected].copy()

        def value_at_scale(scale: float) -> np.ndarray:
            return base + direction * np.float32(scale)

    elif operator_family == SECONDARY_OPERATOR:
        base = flat[selected].copy()
        if not np.any(base):
            raise RuntimeError("Multiplicative replacement is infeasible on all-zero support")

        def value_at_scale(scale: float) -> np.ndarray:
            return base * np.float32(1.0 - scale)

    else:
        raise ValueError(f"Unknown operator family: {operator_family}")
    candidate, runtime_mse, scale = _search_scale(
        runtime_clean,
        selected,
        target_mse,
        value_at_scale,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    clean_bits = encode_runtime_bf16(runtime_clean).reshape(-1)
    candidate_bits = encode_runtime_bf16(candidate).reshape(-1)
    intended = np.zeros(flat.size, dtype=bool)
    intended[selected] = True
    if not np.array_equal(clean_bits[~intended], candidate_bits[~intended]):
        raise AssertionError("Unchanged coordinates are not bit-identical")
    realized_mask = clean_bits != candidate_bits
    probe_mse = float(np.mean((candidate.astype(np.float64) - clean.astype(np.float64)) ** 2))
    mismatch = abs(runtime_mse - target_mse) / target_mse
    if mismatch > relative_tolerance:
        raise RuntimeError("Runtime-BF16 MSE mismatch exceeded preregistered tolerance")
    support_exact = int(selected.size) == support_count(flat.size, active_fraction)
    complement_exact = bool(np.array_equal(clean_bits[~intended], candidate_bits[~intended]))
    if not support_exact or not complement_exact:
        raise AssertionError("Recorded support construction invariants failed")
    realized_nonzero = int(realized_mask.sum())
    replacement_alpha = 1.0 - scale if operator_family == SECONDARY_OPERATOR else None
    details = {
        "target_mse": target_mse,
        "realized_probe_mse": probe_mse,
        "realized_runtime_bf16_mse": runtime_mse,
        "relative_mse_mismatch": mismatch,
        "active_fraction": active_fraction,
        "intended_active_fraction": active_fraction,
        "realized_runtime_active_fraction": realized_nonzero / flat.size,
        "active_elements": int(selected.size),
        "realized_nonzero_elements": realized_nonzero,
        "total_elements": int(flat.size),
        "operator_family": operator_family,
        "random_seed": perturbation_value_seed,
        "support_seed": support_seed,
        "perturbation_value_seed": perturbation_value_seed,
        "solved_scale": scale,
        "replacement_alpha": replacement_alpha,
        "replacement_alpha_class": classify_replacement_alpha(replacement_alpha),
        "selected_indices_sha256": sha256_bytes(selected.astype(np.int64).tobytes()),
        "runtime_input_hash": sha256_bytes(candidate_bits.tobytes()),
        "intended_support_count_exact": support_exact,
        "unchanged_coordinates_bit_exact": complement_exact,
    }
    return candidate, {**details, **error_descriptors(runtime_clean, candidate)}


def classify_replacement_alpha(alpha: float | None) -> str:
    if alpha is None:
        return "not_applicable"
    if alpha == 0:
        return "exact_zero_fill"
    if alpha < 0:
        return "sign_inverting_multiplicative_perturbation"
    if alpha <= 1:
        return "attenuation_or_replacement_like"
    return "amplification"


def error_descriptors(clean: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    error = candidate.astype(np.float64).reshape(-1) - clean.astype(np.float64).reshape(-1)
    absolute = np.abs(error)
    mean = float(error.mean())
    std = float(error.std())
    centered = error - mean
    skewness = float(np.mean(centered**3) / std**3) if std > 0 else 0.0
    kurtosis = float(np.mean(centered**4) / std**4 - 3.0) if std > 0 else 0.0
    quantiles = np.quantile(absolute, [0.5, 0.9, 0.95, 0.99, 0.999])
    nonzero = absolute[absolute > 0]
    clean_abs_max = float(np.max(np.abs(clean.astype(np.float64))))
    restored_abs_max = float(np.max(np.abs(candidate.astype(np.float64))))
    return {
        "error_mean": mean,
        "error_std": std,
        "error_abs_mean": float(absolute.mean()),
        "error_abs_max": float(absolute.max()),
        "error_l1": float(absolute.sum()),
        "error_l2": float(np.linalg.norm(error)),
        "error_linf": float(absolute.max()),
        "linf_error": float(absolute.max()),
        "error_kurtosis": kurtosis,
        "error_skewness": skewness,
        "zero_fraction_of_error": float(np.count_nonzero(error == 0) / error.size),
        "p50_abs_error": float(quantiles[0]),
        "p90_abs_error": float(quantiles[1]),
        "p95_abs_error": float(quantiles[2]),
        "p99_abs_error": float(quantiles[3]),
        "p999_abs_error": float(quantiles[4]),
        "clean_abs_max": clean_abs_max,
        "restored_abs_max": restored_abs_max,
        "restored_to_clean_absmax_ratio": restored_abs_max / max(clean_abs_max, 1e-30),
        "active_error_rms": float(math.sqrt(float(np.mean(nonzero**2)))) if nonzero.size else 0.0,
        "exceeds_clean_dynamic_range": restored_abs_max > clean_abs_max,
    }


def validate_real_v3_construction_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct all 180 inputs on retained z20 states without GPU execution."""
    concentration = config["concentration"]
    tolerance = float(concentration["runtime_mse_relative_tolerance"])
    digest = config_hash(config)
    rows: list[dict[str, Any]] = []
    support_seeds: dict[int, tuple[str, str, float]] = {}
    for prompt_spec in concentration["selected_prompts"]:
        source = load_source_trajectory(config, prompt_spec, 20)
        for target in concentration["targets"]:
            for operator, fractions in (
                (PRIMARY_OPERATOR, concentration["additive_support_fractions"]),
                (SECONDARY_OPERATOR, concentration["replacement_support_fractions"]),
            ):
                for fraction in fractions:
                    support_seed, value_seed = condition_random_seeds(
                        digest,
                        source.prompt_id,
                        source.seed,
                        20,
                        target["name"],
                        float(fraction),
                        operator,
                    )
                    seed_identity = (source.prompt_id, target["name"], float(fraction))
                    previous = support_seeds.setdefault(support_seed, seed_identity)
                    if previous != seed_identity:
                        raise GlobalStopError("GLOBAL STOP: support seed collision across scientific cells")
                    _, details = construct_fixed_mse_error(
                        source.clean,
                        target_mse=float(target["mse"]),
                        active_fraction=float(fraction),
                        operator_family=operator,
                        support_seed=support_seed,
                        perturbation_value_seed=value_seed,
                        relative_tolerance=tolerance,
                    )
                    rows.append(
                        {
                            "prompt_id": source.prompt_id,
                            "target_name": target["name"],
                            **details,
                        }
                    )
    if len(rows) != int(concentration["full_condition_count"]):
        raise GlobalStopError("GLOBAL STOP: real construction matrix is incomplete")
    if any(float(row["relative_mse_mismatch"]) > tolerance for row in rows):
        raise GlobalStopError("GLOBAL STOP: real construction matrix exceeds MSE tolerance")
    paired: dict[tuple[str, str, float], dict[str, str]] = {}
    for row in rows:
        key = (row["prompt_id"], row["target_name"], float(row["active_fraction"]))
        paired.setdefault(key, {})[row["operator_family"]] = row["selected_indices_sha256"]
    for key, operator_hashes in paired.items():
        if set(operator_hashes) == {PRIMARY_OPERATOR, SECONDARY_OPERATOR} and len(set(operator_hashes.values())) != 1:
            raise GlobalStopError(f"GLOBAL STOP: paired operators use different supports for {key}")
    validate_realized_support_ordering(rows)
    return rows


def validate_realized_support_ordering(rows: list[dict[str, Any]]) -> None:
    primary = [row for row in rows if row["operator_family"] == PRIMARY_OPERATOR]
    groups = sorted({(row["prompt_id"], row["target_name"]) for row in primary})
    for group in groups:
        cells = sorted(
            (row for row in primary if (row["prompt_id"], row["target_name"]) == group),
            key=lambda row: float(row["intended_active_fraction"]),
        )
        intended = [float(row["intended_active_fraction"]) for row in cells]
        realized = [float(row["realized_runtime_active_fraction"]) for row in cells]
        if any(right <= left for left, right in zip(intended, intended[1:])):
            raise GlobalStopError(f"GLOBAL STOP: intended support ordering is not strict for {group}")
        if any(right <= left for left, right in zip(realized, realized[1:])):
            raise GlobalStopError(
                f"GLOBAL STOP: realized BF16 support ordering differs from intended ordering for {group}"
            )


def paired_operator_supports_match(rows: list[dict[str, Any]]) -> bool:
    groups: dict[tuple[str, str, float], dict[str, str]] = {}
    for row in rows:
        key = (row["prompt_id"], row["target_name"], float(row["active_fraction"]))
        groups.setdefault(key, {})[row["operator_family"]] = row["selected_indices_sha256"]
    comparable = [values for values in groups.values() if set(values) == {PRIMARY_OPERATOR, SECONDARY_OPERATOR}]
    return bool(comparable) and all(len(set(values.values())) == 1 for values in comparable)


def classify_fp16_replays(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    fp16 = sorted((row for row in rows if row["replay_kind"] == "fp16"), key=lambda row: row["replay_index"])
    full = sorted((row for row in rows if row["replay_kind"] == "full_direct"), key=lambda row: row["replay_index"])
    expected = config["fp16_replay"]
    if len(fp16) != int(expected["repeats"]) or len(full) != int(expected["full_direct_repeats"]):
        raise ValueError("FP16 replay matrix is incomplete")
    if len({row["input_runtime_bf16_sha256"] for row in fp16}) != 1:
        raise GlobalStopError("GLOBAL STOP: FP16 repeats did not use identical runtime input bits")
    if not all(bool(row["exact_final_latent"]) and bool(row["exact_video"]) for row in full):
        classification = "A5_FULL_DIRECT_INVALID"
    elif all(bool(row["exact_final_latent"]) and bool(row["exact_video"]) for row in fp16):
        classification = "A4_ORIGINAL_ANOMALY_NOT_REPRODUCED"
    elif len({row["recovered_final_latent_sha256"] for row in fp16}) > 1:
        classification = "A2_DOWNSTREAM_EXECUTION_NONDETERMINISM"
    elif len({row["recovered_video_sha256"] for row in fp16}) > 1 or len(
        {float(row["frame_ssim_mean"]) for row in fp16}
    ) > 1:
        classification = "A3_DECODER_ARTIFACT_OR_METRIC_INCONSISTENCY"
    else:
        original = float(expected["original_frame_ssim_mean"])
        tolerance = float(expected["a1_ssim_absolute_tolerance"])
        similar = all(abs(float(row["frame_ssim_mean"]) - original) <= tolerance for row in fp16)
        all_nonexact = all(not bool(row["exact_final_latent"]) for row in fp16)
        classification = (
            "A1_DETERMINISTIC_SENSITIVITY_CANDIDATE"
            if similar and all_nonexact
            else "UNRESOLVED_REPLAY_PATTERN"
        )
    return {
        "classification": classification,
        "fp16_repeat_count": len(fp16),
        "full_direct_repeat_count": len(full),
        "unique_fp16_input_runtime_hashes": sorted({row["input_runtime_bf16_sha256"] for row in fp16}),
        "unique_fp16_final_latent_hashes": sorted({row["recovered_final_latent_sha256"] for row in fp16}),
        "unique_fp16_video_hashes": sorted({row["recovered_video_sha256"] for row in fp16}),
        "fp16_ssim_mean": statistics.fmean(float(row["frame_ssim_mean"]) for row in fp16),
        "full_controls_exact": all(bool(row["exact_final_latent"]) and bool(row["exact_video"]) for row in full),
    }


def expected_scientific_keys(config: dict[str, Any], *, smoke: bool = False) -> set[tuple[str, str, str, float]]:
    concentration = config["concentration"]
    prompts = concentration["selected_prompts"][:1] if smoke else concentration["selected_prompts"]
    additive = (
        concentration["smoke_additive_support_fractions"]
        if smoke
        else concentration["additive_support_fractions"]
    )
    replacement = (
        concentration["smoke_replacement_support_fractions"]
        if smoke
        else concentration["replacement_support_fractions"]
    )
    keys = set()
    for prompt in prompts:
        for target in concentration["targets"]:
            keys.update((prompt["prompt_id"], target["name"], PRIMARY_OPERATOR, float(value)) for value in additive)
            keys.update(
                (prompt["prompt_id"], target["name"], SECONDARY_OPERATOR, float(value))
                for value in replacement
            )
    return keys


def validate_expected_keys(rows: list[dict[str, Any]], expected: set[tuple[str, str, str, float]]) -> None:
    keys = [
        (row["prompt_id"], row["target_name"], row["operator_family"], float(row["active_fraction"]))
        for row in rows
    ]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    actual = set(keys)
    if duplicates or actual != expected:
        raise GlobalStopError(
            f"GLOBAL STOP: scientific key mismatch missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)} duplicates={duplicates}"
        )


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left, right = _ranks(x), _ranks(y)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def analyze_primary(rows: list[dict[str, Any]], config: dict[str, Any], *, controls_passed: bool) -> dict[str, Any]:
    primary = [row for row in rows if row["operator_family"] == PRIMARY_OPERATOR]
    expected_primary = {
        key for key in expected_scientific_keys(config) if key[2] == PRIMARY_OPERATOR
    }
    actual_primary = {
        (row["prompt_id"], row["target_name"], row["operator_family"], float(row["active_fraction"]))
        for row in primary
    }
    if actual_primary != expected_primary:
        raise ValueError("Primary analysis matrix is incomplete")
    analysis = config["analysis"]
    dense = float(analysis["endpoint_dense_fraction"])
    concentrated = float(analysis["endpoint_concentrated_fraction"])
    target_summaries = []
    prompt_rows = []
    for target in (row["name"] for row in config["concentration"]["targets"]):
        endpoint_differences = []
        monotonic_count = 0
        direction_count = 0
        for prompt in (row["prompt_id"] for row in config["concentration"]["selected_prompts"]):
            cells = sorted(
                (row for row in primary if row["target_name"] == target and row["prompt_id"] == prompt),
                key=lambda row: float(row["active_fraction"]),
            )
            by_fraction = {float(row["active_fraction"]): float(row["frame_ssim_mean"]) for row in cells}
            endpoint = by_fraction[dense] - by_fraction[concentrated]
            rho = spearman_rho(
                [math.log(float(row["active_fraction"])) for row in cells],
                [float(row["frame_ssim_mean"]) for row in cells],
            )
            realized_rho = spearman_rho(
                [
                    math.log(float(row.get("realized_runtime_active_fraction", row["active_fraction"])))
                    for row in cells
                ],
                [float(row["frame_ssim_mean"]) for row in cells],
            )
            x = np.asarray([math.log(float(row["active_fraction"])) for row in cells])
            y = np.asarray([float(row["frame_ssim_mean"]) for row in cells])
            slope = float(np.polyfit(x, y, 1)[0])
            endpoint_differences.append(endpoint)
            direction_count += int(endpoint > 0)
            monotonic_count += int(rho >= float(analysis["go_spearman_rho"]))
            prompt_rows.append(
                {
                    "target_name": target,
                    "prompt_id": prompt,
                    "dense_minus_concentrated_ssim": endpoint,
                    "spearman_rho": rho,
                    "realized_support_spearman_rho": realized_rho,
                    "linear_log_fraction_slope": slope,
                }
            )
        target_summaries.append(
            {
                "target_name": target,
                "mean_dense_minus_concentrated_ssim": statistics.fmean(endpoint_differences),
                "median_dense_minus_concentrated_ssim": statistics.median(endpoint_differences),
                "direction_prompt_count": direction_count,
                "monotonic_prompt_count": monotonic_count,
            }
        )
    all_mse_matched = all(
        float(row["relative_mse_mismatch"])
        <= float(config["concentration"]["runtime_mse_relative_tolerance"])
        for row in primary
    )
    go = any(
        row["mean_dense_minus_concentrated_ssim"] >= float(analysis["go_endpoint_ssim_difference"])
        and row["direction_prompt_count"] >= int(analysis["go_direction_prompt_count"])
        and row["monotonic_prompt_count"] >= int(analysis["go_monotonic_prompt_count"])
        for row in target_summaries
    ) and all_mse_matched and controls_passed
    no_go = all(
        row["mean_dense_minus_concentrated_ssim"]
        < float(analysis["no_go_endpoint_ssim_difference"])
        for row in target_summaries
    )
    if go:
        decision = "GO_TO_INDEPENDENT_CONFIRMATION"
    elif no_go:
        decision = "NO_GO"
    else:
        decision = "WEAK_INCONCLUSIVE"
    return {
        "decision": decision,
        "controls_passed": controls_passed,
        "all_primary_mse_matched": all_mse_matched,
        "target_summaries": sorted(target_summaries, key=lambda row: row["target_name"]),
        "prompt_effects": sorted(prompt_rows, key=lambda row: (row["target_name"], row["prompt_id"])),
        "decision_input_fields": sorted(DECISION_INPUT_FIELDS),
        "secondary_operator_used_for_decision": False,
    }


def analysis_bytes(rows: list[dict[str, Any]], config: dict[str, Any], *, controls_passed: bool = True) -> bytes:
    return canonical_json(analyze_primary(rows, config, controls_passed=controls_passed))


def _source_prompt_spec(config: dict[str, Any], prompt_id: str, seed: int) -> dict[str, Any]:
    for row in config["concentration"]["selected_prompts"]:
        if row["prompt_id"] == prompt_id and int(row["generation_seed"]) == seed:
            return row
    return {"prompt_id": prompt_id, "generation_seed": seed, "difficulty": "anomaly"}


def _fp16_source_candidate(config: dict[str, Any]) -> tuple[SourceTrajectory, np.ndarray, dict[str, Any]]:
    replay = config["fp16_replay"]
    anomaly = validate_fp16_config_matches_unique_v3_anomaly(config, source_rows(config))
    source = load_source_trajectory(
        config,
        _source_prompt_spec(config, replay["prompt_id"], int(replay["generation_seed"])),
        int(replay["checkpoint_step"]),
    )
    root = _resolve(config["source_v3"]["root"])
    directory = (
        root
        / "run/cells"
        / source.prompt_id
        / f"seed_{source.seed}"
        / f"step_{source.checkpoint_step}"
        / "fp16"
    )
    result = json.loads((directory / "result.json").read_text())
    metadata, arrays = v3.deserialize_components(
        directory / "corruption/fp16.payload.bin",
        directory / "corruption/fp16.metadata.json",
    )
    if len(arrays) != 1 or arrays[0].dtype != np.float16:
        raise GlobalStopError("GLOBAL STOP: original FP16 artifact encoding changed")
    candidate = arrays[0].astype(np.float32)
    if metadata["clean_latent_hash"] != source.clean_hash or result["clean_latent_hash"] != source.clean_hash:
        raise GlobalStopError("GLOBAL STOP: original FP16 artifact is paired with another checkpoint")
    if v3.array_sha256(candidate) != result["corrupted_latent_hash"]:
        raise GlobalStopError("GLOBAL STOP: original FP16 restored probe hash mismatch")
    if result["corrupted_latent_hash"] != anomaly["corrupted_latent_hash"]:
        raise GlobalStopError("GLOBAL STOP: reconstructed FP16 artifact differs from derived anomaly")
    runtime_hash = sha256_bytes(encode_runtime_bf16(candidate).tobytes())
    return source, candidate, {
        "source_result": result,
        "source_metadata": metadata,
        "original_v3_fp16_probe_sha256": result["corrupted_latent_hash"],
        "frozen_reconstructed_fp16_runtime_sha256": runtime_hash,
    }


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "tensor_sha256": v3.array_sha256(value),
    }


def _validate_saved_array(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.exists() or sha256_file(path) != record["file_sha256"]:
        return False
    return v3.array_sha256(np.load(path, allow_pickle=False)) == record["tensor_sha256"]


def _shutdown(omni: Any) -> None:
    shutdown = getattr(omni, "shutdown", None)
    if callable(shutdown):
        shutdown()


def run_resume(
    omni: Any,
    config: dict[str, Any],
    source: SourceTrajectory,
    runtime_candidate: np.ndarray,
    *,
    step_index: int,
    label: str,
    directory: Path,
) -> dict[str, Any]:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    remaining = int(config["generation"]["num_inference_steps"]) - step_index
    video, metadata, elapsed = v3.run_generate(
        omni,
        config,
        prompt=source.prompt,
        seed=source.seed,
        label=label,
        artifact_dir=directory / "trajectory_probe",
        capture_steps=[remaining],
        latents=torch.from_numpy(np.ascontiguousarray(runtime_candidate)),
        step_index=step_index,
    )
    final_latent = v3.final_latent_numpy(metadata)
    video_record = _save_array(directory / "recovered_video.npy", video)
    latent_record = _save_array(directory / "recovered_final_latent.npy", final_latent)
    quality = v3.video_metrics(video, source.video)
    final_error = v3.latent_error(source.final_latent, final_latent)
    scheduler_class = str(metadata.get("scheduler_class", ""))
    if metadata.get("sample_solver") != "euler" or not scheduler_class.endswith(EXPECTED_SCHEDULER):
        raise GlobalStopError("GLOBAL STOP: non-Euler resume detected")
    document = {
        "scheduler_class": scheduler_class,
        "sample_solver": metadata["sample_solver"],
        "resume_index": step_index,
        "resume_ms": elapsed,
        "final_latent_mse": final_error["mse"],
        "exact_final_latent": bool(np.array_equal(final_latent, source.final_latent)),
        "exact_video": bool(np.array_equal(video, source.video)),
        "recovered_final_latent_sha256": latent_record["tensor_sha256"],
        "recovered_video_sha256": video_record["tensor_sha256"],
        "recovered_final_latent_artifact": latent_record,
        "recovered_video_artifact": video_record,
        "trajectory_probe_metadata_path": metadata.get("metadata_path"),
        **quality,
    }
    if not _validate_saved_array(video_record) or not _validate_saved_array(latent_record):
        raise GlobalStopError("GLOBAL STOP: recovered scientific artifact retention failed")
    return document


def _result_valid(
    path: Path,
    provenance: dict[str, Any],
    source_raw_sha: str,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
        if row["status"] != "COMPLETE":
            raise GlobalStopError(f"GLOBAL STOP: incomplete result exists at {path}")
        if row["provenance_hash"] != provenance["provenance_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: stale result provenance at {path}")
        if row["source_raw_sha256"] != source_raw_sha:
            raise GlobalStopError(f"GLOBAL STOP: stale source-v3 hash at {path}")
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise GlobalStopError(f"GLOBAL STOP: result identity mismatch at {path}")
        if not _validate_saved_array(row["recovered_video_artifact"]):
            raise GlobalStopError(f"GLOBAL STOP: recovered video artifact invalid at {path}")
        if not _validate_saved_array(row["recovered_final_latent_artifact"]):
            raise GlobalStopError(f"GLOBAL STOP: recovered latent artifact invalid at {path}")
        return row
    except GlobalStopError:
        raise
    except Exception as error:
        raise GlobalStopError(f"GLOBAL STOP: malformed completed result at {path}: {error}") from error


def _metric_control_result(reference: np.ndarray) -> dict[str, Any]:
    exact = v3.video_metrics(reference, reference)
    mild = reference.copy()
    mild.reshape(-1)[:: max(1, mild.size // 1000)] //= 2
    zero = np.zeros_like(reference)
    reversed_video = reference[::-1].copy()
    mild_metrics = v3.video_metrics(mild, reference)
    zero_metrics = v3.video_metrics(zero, reference)
    reverse_metrics = v3.video_metrics(reversed_video, reference)
    passed = (
        exact["video_mse"] == 0
        and exact["frame_ssim_mean"] == 1
        and exact["temporal_delta_mse"] == 0
        and mild_metrics["frame_ssim_mean"] > zero_metrics["frame_ssim_mean"]
        and zero_metrics["frame_ssim_mean"] < 0.95
        and reverse_metrics["temporal_delta_mse"] > exact["temporal_delta_mse"]
    )
    return {
        "passed": passed,
        "exact": exact,
        "mild": mild_metrics,
        "zero": zero_metrics,
        "temporal_reverse": reverse_metrics,
    }


def _gate(name: str, passed: bool, evidence: Any, expected: str, artifacts: Iterable[str | Path] = ()) -> dict[str, Any]:
    return v3.gate_record(name, passed, evidence, artifacts, expected)


def _write_gates(path: Path, gates: list[dict[str, Any]]) -> None:
    all_passed = v3.validate_gate_records(gates)
    atomic_json(path, {"all_passed": all_passed, "gates": gates})
    if not all_passed:
        failures = [gate["name"] for gate in gates if gate["required"] and gate["status"] != "PASS"]
        raise GlobalStopError(f"GLOBAL STOP: required gates failed: {failures}")


def run_cpu_mode(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = source_rows(config)
    validate_frozen_source_derivations(config, rows)
    anomaly = validate_fp16_config_matches_unique_v3_anomaly(config, rows)
    real_matrix_rows = validate_real_v3_construction_matrix(config)
    source_raw = _resolve(config["source_v3"]["root"]) / "raw_results.csv"
    provenance = build_provenance(config_path, source_raw)
    synthetic = np.linspace(-2, 2, 4096, dtype=np.float32).reshape(1, 4, 4, 16, 16)
    matcher_rows = []
    for target in config["concentration"]["targets"]:
        for fraction in (1.0, 0.2, 0.05):
            support_seed, value_seed = condition_random_seeds(
                config_hash(config), "synthetic_cpu", 0, 20, target["name"], fraction, PRIMARY_OPERATOR
            )
            candidate, details = construct_fixed_mse_error(
                synthetic,
                target_mse=float(target["mse"]),
                active_fraction=fraction,
                operator_family=PRIMARY_OPERATOR,
                support_seed=support_seed,
                perturbation_value_seed=value_seed,
                relative_tolerance=float(config["concentration"]["runtime_mse_relative_tolerance"]),
            )
            if candidate.shape != synthetic.shape:
                raise AssertionError("CPU matcher changed shape")
            matcher_rows.append({"target": target["name"], **details})
    gates = [
        _gate("G1 Euler scheduler only", config["scheduler"]["name"] == EXPECTED_SCHEDULER, config["scheduler"], EXPECTED_SCHEDULER),
        _gate("G2 source BF16 runtime state", all(row["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE for row in rows), {"rows": len(rows)}, EXPECTED_RUNTIME_DTYPE),
        _gate("G7 source-v3 raw and derivations", sha256_file(source_raw) == config["source_v3"]["raw_results_sha256"] and config["concentration"]["selected_prompts"] == select_prompts(rows) and {row["name"]: row["mse"] for row in config["concentration"]["targets"]} == derive_targets(rows), {"raw_sha256": sha256_file(source_raw), "selected_prompts": select_prompts(rows), "targets": derive_targets(rows)}, "frozen source hash, prompt ranking, and targets match"),
        _gate("fp16_config_matches_unique_v3_anomaly", anomaly["final_latent_exact"] is False and anomaly["video_exact"] is False, anomaly, "configured replay identity is the sole non-exact v3 FP16 row"),
        _gate("G8 paired support construction", all(row["intended_support_count_exact"] and row["unchanged_coordinates_bit_exact"] for row in real_matrix_rows), {"cells": len(real_matrix_rows), "unique_support_seeds": len({row["support_seed"] for row in real_matrix_rows})}, "all constructions enforce exact intended support and paired operators share support hashes"),
        _gate("G9 runtime-BF16 MSE matcher", all(row["relative_mse_mismatch"] <= float(config["concentration"]["runtime_mse_relative_tolerance"]) for row in real_matrix_rows), {"cells": len(real_matrix_rows), "max_mismatch": max(row["relative_mse_mismatch"] for row in real_matrix_rows)}, "all 180 retained-v3 constructions meet the single authoritative <=1% tolerance"),
        _gate("G12 temporal metrics descriptive only", not (set(config["analysis"]["descriptive_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "temporal metrics absent from decision inputs"),
        _gate("G13 CLIP auxiliary only", not (set(config["analysis"]["auxiliary_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "CLIP absent from decision inputs"),
        _gate("G14 frozen exact key sets", len(expected_scientific_keys(config)) == 180 and len(expected_scientific_keys(config, smoke=True)) == 8, {"full": len(expected_scientific_keys(config)), "smoke": len(expected_scientific_keys(config, smoke=True))}, "180 full and 8 smoke keys"),
        _gate("G15 frozen provenance", provenance["source_raw_sha256"] == config["source_v3"]["raw_results_sha256"], provenance, "current code/config/source hashes recorded"),
        _gate("G16 no v2 or UniPC reuse", "v2" not in str(source_raw) and all(row["scheduler"] == EXPECTED_SCHEDULER for row in rows), str(source_raw), "only v3-corrected Euler source"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "preregistered_config.json", config)
    atomic_json(output_dir / "run_provenance.json", provenance)
    atomic_json(output_dir / "environment.json", environment_document(config, provenance))
    atomic_json(output_dir / "cpu_matcher_checks.json", matcher_rows)
    atomic_json(output_dir / "real_v3_construction_checks.json", real_matrix_rows)
    _write_gates(output_dir / "cpu_gates.json", gates)
    return {"mode": "cpu", "all_passed": True, "provenance_hash": provenance["provenance_hash"]}


def require_mode_gate(output_dir: Path, name: str, provenance: dict[str, Any]) -> None:
    assert_provenance_matches(output_dir / "run_provenance.json", provenance)
    path = output_dir / name
    if not path.exists() or not json.loads(path.read_text()).get("all_passed"):
        raise GlobalStopError(f"GLOBAL STOP: prerequisite gate did not pass: {path}")


def run_preflight(
    config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    source_raw = _resolve(config["source_v3"]["root"]) / "raw_results.csv"
    provenance = build_provenance(config_path, source_raw)
    require_mode_gate(output_dir, "cpu_gates.json", provenance)
    trusted_rows = source_rows(config)
    anomaly = validate_fp16_config_matches_unique_v3_anomaly(config, trusted_rows)
    prompt_spec = config["concentration"]["selected_prompts"][0]
    source = load_source_trajectory(config, prompt_spec, 20)
    scheduler = v3.scheduler_document(config)
    metric_control = _metric_control_result(source.video)
    matcher_checks = []
    for target in config["concentration"]["targets"]:
        for operator in (PRIMARY_OPERATOR, SECONDARY_OPERATOR):
            support_seed, value_seed = condition_random_seeds(
                config_hash(config), source.prompt_id, source.seed, 20, target["name"], 0.2, operator
            )
            _, details = construct_fixed_mse_error(
                source.clean,
                target_mse=float(target["mse"]),
                active_fraction=0.2,
                operator_family=operator,
                support_seed=support_seed,
                perturbation_value_seed=value_seed,
                relative_tolerance=float(config["concentration"]["runtime_mse_relative_tolerance"]),
            )
            matcher_checks.append({"target_name": target["name"], **details})
    omni = v3.build_omni(config, args)
    try:
        direct = run_resume(omni, config, source, source.clean, step_index=20, label="preflight_full_direct", directory=output_dir / "preflight/full_direct")
        roundtrip_path = output_dir / "preflight/full_disk/source_checkpoint.npy"
        roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(roundtrip_path, source.clean, allow_pickle=False)
        disk_clean = np.load(roundtrip_path, allow_pickle=False)
        disk = run_resume(omni, config, source, disk_clean, step_index=20, label="preflight_full_disk", directory=output_dir / "preflight/full_disk")
        controls = []
        for index in (19, 21):
            controls.append(
                {
                    "resume_index": index,
                    **run_resume(
                        omni,
                        config,
                        source,
                        source.clean,
                        step_index=index,
                        label=f"preflight_off_by_one_{index}",
                        directory=output_dir / f"preflight/off_by_one_{index}",
                    ),
                }
            )
    finally:
        _shutdown(omni)
    paired_support_pass = all(
        len(
            {
                row["selected_indices_sha256"]
                for row in matcher_checks
                if row["target_name"] == target["name"]
            }
        )
        == 1
        for target in config["concentration"]["targets"]
    )
    exact_direct = direct["exact_final_latent"] and direct["exact_video"]
    exact_disk = disk["exact_final_latent"] and disk["exact_video"]
    off_by_one = all(not row["exact_final_latent"] and not row["exact_video"] for row in controls)
    retained = all(
        _validate_saved_array(record)
        for result in (direct, disk, *controls)
        for record in (result["recovered_video_artifact"], result["recovered_final_latent_artifact"])
    )
    gates = [
        _gate("G1 Euler scheduler only", scheduler["scheduler_class"].endswith(EXPECTED_SCHEDULER), scheduler, EXPECTED_SCHEDULER),
        _gate("G2 BF16 runtime state captured", v3.array_sha256(source.clean) == source.clean_hash, {"clean_hash": source.clean_hash, "dtype": EXPECTED_RUNTIME_DTYPE}, "validated v3 BF16 boundary state"),
        _gate("G3 FULL direct exact", exact_direct, direct, "final latent and video exact"),
        _gate("G4 FULL loaded checkpoint exact", exact_disk, disk, "round-tripped source checkpoint exact"),
        _gate("G5 checkpoint step 20 semantics", direct["resume_index"] == 20 and off_by_one, controls, "20 exact; 19 and 21 non-exact"),
        _gate("G6 off-by-one controls non-exact", off_by_one, controls, "both controls non-exact"),
        _gate("G7 source clean hash", v3.array_sha256(source.clean) == source.clean_hash, {"path": source.checkpoint_path, "hash": source.clean_hash}, "loaded tensor matches v3 manifest"),
        _gate("fp16_config_matches_unique_v3_anomaly", anomaly["final_latent_exact"] is False and anomaly["video_exact"] is False, anomaly, "configured replay identity is the sole non-exact v3 FP16 row"),
        _gate("G8 intended-coordinate isolation", paired_support_pass and all(row["intended_support_count_exact"] and row["unchanged_coordinates_bit_exact"] for row in matcher_checks), matcher_checks, "exact intended support, bit-exact complement, and identical paired operator support"),
        _gate("G9 runtime-BF16 MSE matching", all(row["relative_mse_mismatch"] <= float(config["concentration"]["runtime_mse_relative_tolerance"]) for row in matcher_checks), matcher_checks, "all preflight matches <=1%"),
        _gate("G10 artifacts retained and hashed", retained, [direct, disk], "all final videos and latents validate"),
        _gate("G11 SSIM negative controls", metric_control["passed"], metric_control, "exact, zero, mild, reversal controls pass"),
        _gate("G12 temporal metrics descriptive only", not (set(config["analysis"]["descriptive_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "temporal fields absent from decisions"),
        _gate("G13 CLIP auxiliary only", not (set(config["analysis"]["auxiliary_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "CLIP absent from decisions"),
        _gate("G14 expected scientific key set", len(expected_scientific_keys(config)) == 180, len(expected_scientific_keys(config)), "exactly 180 keys"),
        _gate("G15 provenance frozen", provenance["source_raw_sha256"] == config["source_v3"]["raw_results_sha256"], provenance, "code/config/source provenance matches CPU gate"),
        _gate("G16 no v2/UniPC reuse", "v2" not in str(source.manifest_path) and scheduler["scheduler_class"].endswith(EXPECTED_SCHEDULER), str(source.manifest_path), "v3-corrected Euler source only"),
    ]
    atomic_json(output_dir / "metric_controls.json", metric_control)
    atomic_json(output_dir / "preflight_results.json", {"full_direct": direct, "full_disk": disk, "off_by_one": controls})
    _write_gates(output_dir / "preflight_gates.json", gates)
    return {"mode": "preflight", "all_passed": True}


def run_fp16_replay(
    config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    source_raw = _resolve(config["source_v3"]["root"]) / "raw_results.csv"
    provenance = build_provenance(config_path, source_raw)
    require_mode_gate(output_dir, "preflight_gates.json", provenance)
    source, fp16_candidate, original = _fp16_source_candidate(config)
    frozen_runtime_hash = original["frozen_reconstructed_fp16_runtime_sha256"]
    rows = []
    omni = v3.build_omni(config, args)
    try:
        for kind, count, candidate in (
            # Preserve the original v3 FP16-restored FP32 probe while the
            # separately recorded hash proves the model-consumed BF16 bits.
            ("fp16", int(config["fp16_replay"]["repeats"]), fp16_candidate),
            ("full_direct", int(config["fp16_replay"]["full_direct_repeats"]), source.clean),
        ):
            for replay_index in range(count):
                result_path = output_dir / "fp16_replay/cells" / kind / f"repeat_{replay_index:02d}/result.json"
                existing = _result_valid(
                    result_path,
                    provenance,
                    config["source_v3"]["raw_results_sha256"],
                    expected_identity={
                        "replay_kind": kind,
                        "replay_index": replay_index,
                        "prompt_id": source.prompt_id,
                        "checkpoint_step": source.checkpoint_step,
                    },
                )
                if existing is not None:
                    rows.append(existing)
                    continue
                runtime_hash = sha256_bytes(encode_runtime_bf16(candidate).tobytes())
                if kind == "fp16" and runtime_hash != frozen_runtime_hash:
                    raise GlobalStopError("GLOBAL STOP: FP16 replay input differs from frozen reconstructed bits")
                result = run_resume(
                    omni,
                    config,
                    source,
                    candidate,
                    step_index=source.checkpoint_step,
                    label=f"fp16_replay_{kind}_{replay_index}",
                    directory=result_path.parent / "scientific_artifacts",
                )
                row = {
                    "status": "COMPLETE",
                    "experiment_version": config["experiment_version"],
                    "config_hash": config_hash(config),
                    "provenance_hash": provenance["provenance_hash"],
                    "source_raw_sha256": config["source_v3"]["raw_results_sha256"],
                    "replay_kind": kind,
                    "replay_index": replay_index,
                    "prompt_id": source.prompt_id,
                    "generation_seed": source.seed,
                    "checkpoint_step": source.checkpoint_step,
                    "input_probe_latent_sha256": v3.array_sha256(candidate),
                    "input_runtime_bf16_sha256": runtime_hash,
                    "original_v3_fp16_probe_sha256": original["original_v3_fp16_probe_sha256"],
                    "frozen_reconstructed_fp16_runtime_sha256": frozen_runtime_hash,
                    "clean_checkpoint_latent_sha256": source.clean_hash,
                    "clean_final_latent_sha256": v3.array_sha256(source.final_latent),
                    "clean_reference_video_sha256": v3.array_sha256(source.video),
                    "next_scheduler_timestep": v3.scheduler_document(config)["next_timestep_by_checkpoint"][str(source.checkpoint_step)],
                    "result_path": str(result_path),
                    **result,
                }
                atomic_json(result_path, row)
                rows.append(row)
    finally:
        _shutdown(omni)
    rows.sort(key=lambda row: (row["replay_kind"], int(row["replay_index"])))
    summary = classify_fp16_replays(rows, config)
    if summary["classification"] == "A5_FULL_DIRECT_INVALID":
        raise GlobalStopError("GLOBAL STOP: FULL-direct replay control is not exact")
    write_csv(output_dir / "fp16_replay_raw.csv", rows)
    atomic_json(output_dir / "fp16_replay_summary.json", summary)
    report = [
        "# FP16 Exact Replay",
        "",
        f"Classification: **{summary['classification']}**",
        "",
        f"FP16 repeats: {summary['fp16_repeat_count']}",
        f"FULL-direct repeats: {summary['full_direct_repeat_count']}",
        f"Mean FP16 SSIM: {summary['fp16_ssim_mean']:.6f}",
        "",
        "The trusted v3 row did not record a runtime-BF16 input hash. The frozen runtime hash is reconstructed from the independently verified FP16 probe artifact.",
        "",
        "This is a neutral replay classification, not a mechanism claim.",
    ]
    (output_dir / "fp16_replay_report.md").write_text("\n".join(report) + "\n")
    return summary


def _condition_path(
    output_dir: Path, mode: str, prompt_id: str, target: str, operator: str, fraction: float
) -> Path:
    tag = f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return output_dir / mode / "cells" / prompt_id / target / operator / f"support_{tag}" / "result.json"


def _condition_row(
    config: dict[str, Any],
    provenance: dict[str, Any],
    source: SourceTrajectory,
    target: dict[str, Any],
    operator: str,
    fraction: float,
    details: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "experiment_version": config["experiment_version"],
        "config_hash": config_hash(config),
        "provenance_hash": provenance["provenance_hash"],
        "source_raw_sha256": config["source_v3"]["raw_results_sha256"],
        "model": config["model"],
        "scheduler": EXPECTED_SCHEDULER,
        "prompt_id": source.prompt_id,
        "prompt_text": source.prompt,
        "difficulty": source.difficulty,
        "generation_seed": source.seed,
        "checkpoint_step": source.checkpoint_step,
        "resume_index": source.checkpoint_step,
        "target_name": target["name"],
        "clean_checkpoint_hash": source.clean_hash,
        "prompt_clip_score": "",
        "recovered_final_latent_path": result["recovered_final_latent_artifact"]["path"],
        "recovered_video_path": result["recovered_video_artifact"]["path"],
        "result_path": str(result_path),
        **details,
        **result,
    }


def run_concentration(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    smoke: bool,
) -> dict[str, Any]:
    source_raw = _resolve(config["source_v3"]["root"]) / "raw_results.csv"
    provenance = build_provenance(config_path, source_raw)
    require_mode_gate(output_dir, "preflight_gates.json", provenance)
    replay_summary_path = output_dir / "fp16_replay_summary.json"
    if not replay_summary_path.exists():
        raise GlobalStopError("GLOBAL STOP: FP16 replay must be inspected before concentration")
    replay_summary = json.loads(replay_summary_path.read_text())
    if replay_summary["classification"] == "A5_FULL_DIRECT_INVALID":
        raise GlobalStopError("GLOBAL STOP: FP16 replay invalidated the environment")
    concentration = config["concentration"]
    prompt_specs = concentration["selected_prompts"][:1] if smoke else concentration["selected_prompts"]
    additive = concentration["smoke_additive_support_fractions"] if smoke else concentration["additive_support_fractions"]
    replacement = concentration["smoke_replacement_support_fractions"] if smoke else concentration["replacement_support_fractions"]
    mode = "concentration_smoke" if smoke else "full"
    rows: list[dict[str, Any]] = []
    omni = v3.build_omni(config, args)
    try:
        for prompt_spec in prompt_specs:
            source = load_source_trajectory(config, prompt_spec, 20)
            for target in concentration["targets"]:
                for operator, fractions in ((PRIMARY_OPERATOR, additive), (SECONDARY_OPERATOR, replacement)):
                    for fraction in fractions:
                        result_path = _condition_path(output_dir, mode, source.prompt_id, target["name"], operator, float(fraction))
                        existing = _result_valid(
                            result_path,
                            provenance,
                            config["source_v3"]["raw_results_sha256"],
                            expected_identity={
                                "prompt_id": source.prompt_id,
                                "target_name": target["name"],
                                "operator_family": operator,
                                "active_fraction": float(fraction),
                                "checkpoint_step": 20,
                            },
                        )
                        if existing is not None:
                            rows.append(existing)
                            continue
                        support_seed, value_seed = condition_random_seeds(
                            config_hash(config),
                            source.prompt_id,
                            source.seed,
                            20,
                            target["name"],
                            float(fraction),
                            operator,
                        )
                        candidate, details = construct_fixed_mse_error(
                            source.clean,
                            target_mse=float(target["mse"]),
                            active_fraction=float(fraction),
                            operator_family=operator,
                            support_seed=support_seed,
                            perturbation_value_seed=value_seed,
                            relative_tolerance=float(concentration["runtime_mse_relative_tolerance"]),
                        )
                        result = run_resume(
                            omni,
                            config,
                            source,
                            candidate,
                            step_index=20,
                            label=f"{mode}_{source.prompt_id}_{target['name']}_{operator}_{fraction}",
                            directory=result_path.parent / "scientific_artifacts",
                        )
                        row = _condition_row(
                            config, provenance, source, target, operator, float(fraction), details, result, result_path
                        )
                        atomic_json(result_path, row)
                        rows.append(row)
                        print(
                            f"[error-shape] {source.prompt_id} target={target['name']} "
                            f"operator={operator} support={fraction:g} "
                            f"mismatch={details['relative_mse_mismatch']:.4g} "
                            f"ssim={result['frame_ssim_mean']:.4f}",
                            flush=True,
                        )
    finally:
        _shutdown(omni)
    expected = expected_scientific_keys(config, smoke=smoke)
    validate_expected_keys(rows, expected)
    validate_realized_support_ordering(rows)
    rows.sort(key=lambda row: (row["prompt_id"], row["target_name"], row["operator_family"], float(row["active_fraction"])))
    output_csv = output_dir / ("concentration_smoke_raw.csv" if smoke else "concentration_raw.csv")
    write_csv(output_csv, rows, RAW_FIELDS)
    all_mse = all(float(row["relative_mse_mismatch"]) <= float(concentration["runtime_mse_relative_tolerance"]) for row in rows)
    artifacts = all(
        _validate_saved_array(row["recovered_video_artifact"])
        and _validate_saved_array(row["recovered_final_latent_artifact"])
        for row in rows
    )
    gates = [
        _gate("G1 Euler scheduler only", all(row["scheduler_class"].endswith(EXPECTED_SCHEDULER) for row in rows), {"rows": len(rows)}, EXPECTED_SCHEDULER),
        _gate("G5 step-20 resume semantics", all(int(row["resume_index"]) == 20 for row in rows), {"rows": len(rows)}, "all resumes start at 20"),
        _gate("G7 source clean checkpoint hashes", all(row["clean_checkpoint_hash"] for row in rows), {"unique": sorted({row["clean_checkpoint_hash"] for row in rows})}, "all hashes validated on source load"),
        _gate("G8 intended-coordinate isolation", paired_operator_supports_match(rows) and all(row["intended_support_count_exact"] and row["unchanged_coordinates_bit_exact"] for row in rows), {"rows": len(rows), "paired_operator_supports_match": paired_operator_supports_match(rows)}, "exact intended support, bit-exact complement, and identical paired operator support"),
        _gate("G9 runtime-BF16 MSE matching", all_mse, {"max_mismatch": max(float(row["relative_mse_mismatch"]) for row in rows)}, "all matches <=1%"),
        _gate("G10 artifacts retained and hashed", artifacts, {"rows": len(rows)}, "video and final latent for every row"),
        _gate("G12 temporal metrics descriptive only", not (set(config["analysis"]["descriptive_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "temporal fields absent from decisions"),
        _gate("G13 CLIP auxiliary only", not (set(config["analysis"]["auxiliary_only_metrics"]) & DECISION_INPUT_FIELDS), sorted(DECISION_INPUT_FIELDS), "CLIP absent from decisions"),
        _gate("G14 exact expected key set", len(rows) == len(expected), {"actual": len(rows), "expected": len(expected)}, "exact set equality already validated"),
        _gate("G15 provenance frozen", all(row["provenance_hash"] == provenance["provenance_hash"] for row in rows), provenance, "all rows current provenance"),
        _gate("G16 no v2/UniPC reuse", all(row["scheduler"] == EXPECTED_SCHEDULER for row in rows), {"source": config["source_v3"]["root"]}, "v3-corrected Euler only"),
    ]
    gate_path = output_dir / ("concentration_smoke_gates.json" if smoke else "full_gates.json")
    _write_gates(gate_path, gates)
    if smoke:
        summary = {"mode": mode, "row_count": len(rows), "all_passed": True}
        atomic_json(output_dir / "concentration_smoke_summary.json", summary)
        return summary
    metric_controls = json.loads((output_dir / "metric_controls.json").read_text())
    analysis_result = analyze_primary(rows, config, controls_passed=bool(metric_controls["passed"]))
    secondary_rows = []
    lookup = {
        (row["prompt_id"], row["target_name"], float(row["active_fraction"]), row["operator_family"]): row
        for row in rows
    }
    for prompt in concentration["selected_prompts"]:
        for target in concentration["targets"]:
            for fraction in concentration["replacement_support_fractions"]:
                additive_row = lookup[(prompt["prompt_id"], target["name"], float(fraction), PRIMARY_OPERATOR)]
                replacement_row = lookup[(prompt["prompt_id"], target["name"], float(fraction), SECONDARY_OPERATOR)]
                secondary_rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "target_name": target["name"],
                        "active_fraction": fraction,
                        "replacement_minus_additive_ssim": float(replacement_row["frame_ssim_mean"]) - float(additive_row["frame_ssim_mean"]),
                    }
                )
    write_csv(output_dir / "concentration_primary_prompt_effects.csv", analysis_result["prompt_effects"])
    support_summary = []
    for target in concentration["targets"]:
        for fraction in concentration["additive_support_fractions"]:
            cells = [
                row
                for row in rows
                if row["operator_family"] == PRIMARY_OPERATOR
                and row["target_name"] == target["name"]
                and float(row["active_fraction"]) == float(fraction)
            ]
            support_summary.append(
                {
                    "target_name": target["name"],
                    "intended_active_fraction": fraction,
                    "mean_realized_runtime_active_fraction": statistics.fmean(
                        float(row["realized_runtime_active_fraction"]) for row in cells
                    ),
                    "mean_frame_ssim": statistics.fmean(float(row["frame_ssim_mean"]) for row in cells),
                    "dynamic_range_exceedance_count": sum(
                        bool(row["exceeds_clean_dynamic_range"]) for row in cells
                    ),
                }
            )
    write_csv(output_dir / "concentration_support_summary.csv", support_summary)
    write_csv(output_dir / "replacement_operator_exploratory.csv", secondary_rows)
    atomic_json(output_dir / "concentration_summary.json", analysis_result)
    report = [
        "# Video Runtime Error-Shape Kill Test",
        "",
        f"Decision: **{analysis_result['decision']}**",
        "",
        "Primary endpoint is frame_ssim_mean. Temporal-delta metrics are descriptive and CLIP is auxiliary.",
        "The secondary replacement operator does not participate in this decision.",
        "Intended support is the preregistered construction parameter; realized runtime-BF16 support is reported separately and used only as a robustness descriptor.",
        "Dynamic-range exceedance and replacement-alpha class are descriptive only and never alter inclusion or GO/NO-GO.",
        "LARGE-regime results test concentration at fixed total realized MSE; they must not be interpreted as isolating sparsity alone.",
        "",
        "## Target summaries",
        "",
    ]
    for row in analysis_result["target_summaries"]:
        report.append(
            f"- {row['target_name']}: mean endpoint={row['mean_dense_minus_concentrated_ssim']:.6f}, "
            f"direction={row['direction_prompt_count']}/9, monotonic={row['monotonic_prompt_count']}/9"
        )
    (output_dir / "video_runtime_error_shape_killtest.md").write_text("\n".join(report) + "\n")
    return analysis_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "preflight", "fp16-replay", "concentration-smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/video_runtime_error_shape_killtest_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/video_runtime_error_shape_killtest")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    if output_dir == _resolve(config["source_v3"]["root"]):
        raise GlobalStopError("GLOBAL STOP: output directory may not be the validated v3 namespace")
    if args.mode == "cpu":
        result = run_cpu_mode(config, config_path, output_dir)
    elif args.mode == "preflight":
        result = run_preflight(config, config_path, output_dir, args)
    elif args.mode == "fp16-replay":
        result = run_fp16_replay(config, config_path, output_dir, args)
    elif args.mode == "concentration-smoke":
        result = run_concentration(config, config_path, output_dir, args, smoke=True)
    else:
        require_mode_gate(
            output_dir,
            "concentration_smoke_gates.json",
            build_provenance(config_path, _resolve(config["source_v3"]["root"]) / "raw_results.csv"),
        )
        result = run_concentration(config, config_path, output_dir, args, smoke=False)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
