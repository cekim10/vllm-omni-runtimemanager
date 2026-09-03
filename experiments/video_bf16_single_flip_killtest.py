#!/usr/bin/env python3
"""Exp 0: single-coordinate adjacent-BF16 perturbation map at the FP16 anchor.

Anchor: the unique non-exact trusted-v3 FP16 cell (re-derived at run time).
Question: can ONE runtime-BF16 coordinate, moved by exactly ONE adjacent
representable BF16 value, produce deterministic downstream divergence?

This script reads validated v3 / error-shape artifacts but never writes into
their namespaces.  It contains no scheduler, placement, or policy mechanism
and implements no Experiment A/B/C.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_runtime_error_shape_killtest as base  # noqa: E402
from experiments import video_runtime_state_discovery as v3  # noqa: E402

GlobalStopError = base.GlobalStopError

EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_RUNTIME_DTYPE = "torch.bfloat16"
EXPECTED_SCHEDULER_MODULE = "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler"
EXPECTED_SCHEDULER_CLASS = f"{EXPECTED_SCHEDULER_MODULE}.WanEulerScheduler"
CANDIDATE_STORAGE_DTYPE = "<f4"  # BF16 values persisted as float32 with zero low-order bits
EXPERIMENT_VERSION = "video-bf16-single-flip-killtest-v1"
PERTURBATION_FAMILY = "single_coordinate_adjacent_bf16"
ALLOWED_MODES = ("cpu", "preflight", "smoke", "analyze-smoke")
DIRECTIONS = ("down", "up")
DIRECTION_SIGN = {"down": -1, "up": +1}
# Fields the decision code may read. final_latent_mse and video_mse enter ONLY as replay-equality
# conditions (identical across the three runs of a triggered row); no threshold is ever applied to them.
DECISION_INPUT_FIELDS = frozenset(
    {
        "coordinate_flat_index",
        "direction",
        "replay_id",
        "frame_ssim_mean",
        "final_latent_mse",
        "video_mse",
        "runtime_candidate_identity_sha256_v1",
        "recovered_final_latent_identity_sha256_v1",
        "recovered_video_identity_sha256_v1",
    }
)
REPLAY_EQUALITY_FIELDS = (
    "runtime_candidate_identity_sha256_v1",
    "recovered_final_latent_identity_sha256_v1",
    "recovered_video_identity_sha256_v1",
    "frame_ssim_mean",
    "final_latent_mse",
    "video_mse",
)
TENSOR_IDENTITY_FORMAT = "single-flip-tensor-identity-v1"
# One frozen comparison policy for every recomputed floating accounting/metric field.
ACCOUNTING_RELATIVE_TOLERANCE = 1e-12
# Config keys that would hard-code a derived count; any of them is a design violation.
FORBIDDEN_COUNT_KEYS = frozenset(
    {
        "eligible_count",
        "eligible_coordinate_count",
        "expected_eligible_count",
        "primary_row_count",
        "expected_primary_rows",
        "historical_changed_count",
        "expected_historical_coordinates",
        "eligible_coordinates",
        "historical_coordinates",
        "K",
    }
)
COMMON_GATE_NAMES = (
    "G1 trusted source hashes exact",
    "G3 anchor is the unique non-exact v3 FP16 row",
    "G4 clean anchor state hash and BF16 representability",
    "G5 historical FP16 delta re-derived",
    "G6 eligible set derived and contains historical support",
    "G7 adjacent-BF16 primitive exhaustively verified",
    "G8 single-coordinate isolation for every construction",
    "G9 BF16 state accounting",
    "G10 frozen primary key set",
    "G11 no hard-coded derived counts",
    "G15 decision inputs restricted to preregistered fields",
    "G19 provenance frozen",
    "G20 no automatic expansion",
)
REQUIRED_GATE_NAMES = {
    "cpu_gates.json": frozenset(COMMON_GATE_NAMES + ("G2 Euler scheduler only (config)",)),
    "preflight_gates.json": frozenset(
        COMMON_GATE_NAMES
        + (
            "G2 Euler scheduler only (runtime)",
            "G12 at least three FULL-direct controls bit-exact",
            "G13 resume index equals anchor step",
            "G14 SSIM negative controls",
            "G17 control artifacts retained and hashed",
            "G18 anchor manifest unchanged since CPU mode",
        )
    ),
    "smoke_gates.json": frozenset(
        COMMON_GATE_NAMES
        + (
            "G2 Euler scheduler only (runtime)",
            "G12 smoke FULL-direct control bit-exact",
            "G13 resume index equals anchor step",
            "G16 replay semantics honoured",
            "G17 artifacts retained and hashed",
            "G18 executed runtime input hashes equal frozen hashes",
            "G21 finite frame SSIM for all rows",
        )
    ),
}
ROW_CLASSES = (
    "ABSORBED_EXACT",
    "BENIGN",
    "INTERMEDIATE",
    "CATASTROPHIC_DETERMINISTIC",
    "CATASTROPHIC_NONDETERMINISTIC",
    "TRIGGERED_INCOMPLETE",
)
RAW_FIELDS = (
    "status",
    "experiment_version",
    "config_hash",
    "provenance_hash",
    "source_raw_sha256",
    "model",
    "scheduler",
    "scheduler_class",
    "prompt_id",
    "prompt_text",
    "generation_seed",
    "checkpoint_step",
    "resume_index",
    "clean_checkpoint_hash",
    "anchor_manifest_sha256",
    "condition_id",
    "perturbation_family",
    "coordinate_flat_index",
    "coordinate_multi_index",
    "direction",
    "replay_id",
    "clean_value",
    "perturbed_value",
    "delta",
    "abs_clean_value",
    "clean_bf16_bits_hex",
    "perturbed_bf16_bits_hex",
    "requested_direction",
    "resume_timestep",
    "scheduler_config_json",
    "runtime_dtype",
    "latent_shape_json",
    "clean_state_identity_sha256_v1",
    "runtime_candidate_identity_sha256_v1",
    "recovered_final_latent_identity_sha256_v1",
    "recovered_video_identity_sha256_v1",
    "changed_coordinate_count",
    "realized_nonzero_elements",
    "total_elements",
    "realized_l2",
    "realized_mse",
    "realized_linf",
    "historical_fp16_support_member",
    "runtime_input_hash",
    "final_latent_mse",
    "exact_final_latent",
    "exact_video",
    "video_mse",
    "video_psnr",
    "frame_ssim_mean",
    "temporal_delta_mse",
    "temporal_delta_agreement",
    "prompt_clip_score",
    "resume_ms",
    "recovered_final_latent_sha256",
    "recovered_video_sha256",
    "result_path",
)

canonical_json = base.canonical_json
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
atomic_json = base.atomic_json
write_csv = base.write_csv
read_csv = base.read_csv
_resolve = base._resolve
_validate_saved_array = base._validate_saved_array
_shutdown = base._shutdown


# --------------------------------------------------------------------------- canonical tensor identity


def tensor_identity_sha256_v1(array: np.ndarray) -> str:
    """Versioned canonical tensor identity: format tag, little-endian dtype, ndim, exact shape, raw bytes.

    Unlike the historical byte-only hash, tensors with identical bytes but different
    dtype or shape receive different identities.
    """
    array = np.asarray(array)
    dtype = np.dtype(array.dtype).newbyteorder("<")
    data = np.ascontiguousarray(array.astype(dtype, copy=False))
    header = canonical_json(
        {
            "format": TENSOR_IDENTITY_FORMAT,
            "dtype": dtype.str,
            "ndim": int(array.ndim),
            "shape": [int(value) for value in array.shape],
            "nbytes": int(data.nbytes),
        }
    )
    return sha256_bytes(header + b"\x00" + data.tobytes())


def _identity_fields(array: np.ndarray) -> dict[str, Any]:
    return {
        "tensor_identity_sha256_v1": tensor_identity_sha256_v1(array),
        "identity_format": TENSOR_IDENTITY_FORMAT,
        "dtype": np.dtype(array.dtype).newbyteorder("<").str,
        "shape": [int(value) for value in array.shape],
    }


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    """Persist an array and record file hash, legacy byte hash, and canonical identity."""
    record = base._save_array(path, value)
    return {**record, **_identity_fields(value)}


def _identity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add the canonical identity to an artifact record produced by trusted base code."""
    path = _resolve(record["path"])
    if not path.exists() or sha256_file(path) != record["file_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: artifact file hash mismatch while recording identity: {path}")
    array = np.load(path, allow_pickle=False)
    if v3.array_sha256(array) != record["tensor_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: artifact byte hash mismatch while recording identity: {path}")
    return {**record, **_identity_fields(array)}


def _accounting_matches(declared: Any, recomputed: float) -> bool:
    """Frozen numerical policy for recomputed floating fields (relative 1e-12; zero must be exact)."""
    try:
        declared_value = float(declared)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(declared_value) and math.isfinite(recomputed)):
        return False
    if recomputed == 0.0:
        return declared_value == 0.0
    return abs(declared_value - recomputed) <= ACCOUNTING_RELATIVE_TOLERANCE * abs(recomputed)


def scheduler_timesteps_numpy(config: dict[str, Any]) -> list[float]:
    """Reproduce the pinned WanEulerScheduler schedule without torch (mirrors the trusted stability kill test)."""
    num_steps = int(config["generation"]["num_inference_steps"])
    train_steps = int(config["scheduler"]["num_train_timesteps"])
    shift = float(config["scheduler"]["flow_shift"])
    original = np.linspace(train_steps, 0, num_steps + 1, dtype=np.float32)
    sigma = original / np.float32(train_steps)
    shifted = np.float32(shift) * sigma / (np.float32(1.0) + np.float32(shift - 1.0) * sigma)
    return [float(value) for value in (shifted[:-1] * np.float32(train_steps))]


def anchor_resume_timestep(config: dict[str, Any], checkpoint_step: int) -> float:
    return scheduler_timesteps_numpy(config)[int(checkpoint_step)]


def frozen_scheduler_identity(config: dict[str, Any], checkpoint_step: int) -> dict[str, Any]:
    """Scheduler identity frozen on CPU from source, never from a runtime result file."""
    source_path = REPO_ROOT / (EXPECTED_SCHEDULER_MODULE.replace(".", "/") + ".py")
    if not source_path.exists() or f"class {EXPECTED_SCHEDULER}:" not in source_path.read_text():
        raise GlobalStopError("GLOBAL STOP: frozen scheduler class is not defined at the expected module path")
    return {
        "scheduler_class": EXPECTED_SCHEDULER_CLASS,
        "scheduler_name": EXPECTED_SCHEDULER,
        "scheduler_source_sha256": sha256_file(source_path),
        "scheduler_config": config["scheduler"],
        "checkpoint_step": int(checkpoint_step),
        "resume_index": int(checkpoint_step),
        "resume_timestep": anchor_resume_timestep(config, checkpoint_step),
        "num_inference_steps": int(config["generation"]["num_inference_steps"]),
    }


# --------------------------------------------------------------------------- config


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["experiment_version"] != EXPERIMENT_VERSION:
        raise ValueError("Experiment version changed")
    if config["model"] != EXPECTED_MODEL:
        raise ValueError("Model changed")
    scheduler = config["scheduler"]
    if scheduler["name"] != EXPECTED_SCHEDULER or scheduler["sample_solver"] != "euler":
        raise ValueError("Explicit Euler scheduler is required")
    generation = config["generation"]
    if int(generation["num_inference_steps"]) != 40 or generation["checkpoint_steps"] != [10, 20, 30]:
        raise ValueError("Generation schedule must match the trusted v3 source")
    anchor = config["anchor"]
    if (anchor["prompt_id"], int(anchor["generation_seed"]), int(anchor["checkpoint_step"])) != (
        "recovery_008",
        9234,
        10,
    ):
        raise ValueError("Anchor identity changed")
    perturbation = config["perturbation"]
    frozen_perturbation = {
        "family": PERTURBATION_FAMILY,
        "eligibility_abs_threshold": 1e-05,
        "secondary_report_abs_threshold": 1e-04,
        "directions": list(DIRECTIONS),
        "adjacent_steps": 1,
    }
    for key, expected in frozen_perturbation.items():
        if perturbation[key] != expected:
            raise ValueError(f"Frozen perturbation value changed: {key}")
    forbidden = FORBIDDEN_COUNT_KEYS & (set(config) | set(perturbation) | set(anchor) | set(config["analysis"]))
    if forbidden:
        raise ValueError(f"Hard-coded derived counts are forbidden in config: {sorted(forbidden)}")
    controls = config["controls"]
    if int(controls["preflight_full_direct_repeats"]) < 3 or int(controls["smoke_full_direct_repeats"]) < 1:
        raise ValueError("At least three preflight and one smoke FULL-direct controls are required")
    replay = config["replay"]
    if replay["trigger_frame_ssim_below"] != 0.95 or int(replay["total_runs_per_triggered_row"]) != 3:
        raise ValueError("Frozen replay semantics changed")
    if replay["non_triggered_rows_run_once"] is not True:
        raise ValueError("Non-triggered rows must run exactly once")
    if list(replay["replay_equality_fields"]) != list(REPLAY_EQUALITY_FIELDS):
        raise ValueError("Frozen replay equality fields changed")
    if replay["tensor_identity_format"] != TENSOR_IDENTITY_FORMAT:
        raise ValueError("Tensor identity format changed")
    if set(config["analysis"]["replay_equality_only_fields"]) != {"final_latent_mse", "video_mse"}:
        raise ValueError("Replay-equality-only fields changed")
    analysis = config["analysis"]
    frozen_analysis = {
        "primary_metric": "frame_ssim_mean",
        "catastrophic_frame_ssim_below": 0.95,
        "no_go_frame_ssim_at_least": 0.99,
    }
    for key, expected in frozen_analysis.items():
        if analysis[key] != expected:
            raise ValueError(f"Frozen analysis value changed: {key}")
    if analysis["catastrophic_frame_ssim_below"] != replay["trigger_frame_ssim_below"]:
        raise ValueError("Replay trigger and catastrophic threshold must be the same preregistered value")
    if set(analysis["decision_input_fields"]) != DECISION_INPUT_FIELDS:
        raise ValueError("Configured decision inputs differ from preregistered fields")
    excluded = set(analysis["descriptive_only_metrics"]) | set(analysis["auxiliary_only_metrics"])
    if excluded & DECISION_INPUT_FIELDS:
        raise ValueError("Descriptive/auxiliary metric entered decision inputs")
    if config["allowed_modes"] != list(ALLOWED_MODES):
        raise ValueError("Allowed modes changed; no automatic expansion is permitted")
    if config["trusted_v3"]["runtime_dtype"] != EXPECTED_RUNTIME_DTYPE:
        raise ValueError("Trusted runtime dtype must be BF16")
    return config


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def validate_source_hashes(config: dict[str, Any]) -> dict[str, Any]:
    trusted = config["trusted_v3"]
    error = config["validated_error_shape_source"]
    checks = {
        "trusted_v3_raw": (_resolve(trusted["root"]) / "raw_results.csv", trusted["raw_results_sha256"]),
        "trusted_v3_config_file": (_resolve(trusted["config_file"]), trusted["config_file_sha256"]),
        "trusted_v3_provenance_file": (_resolve(trusted["provenance_file"]), trusted["provenance_file_sha256"]),
        "error_shape_config": (_resolve(error["config_file"]), error["config_file_sha256"]),
        "error_shape_matcher": (_resolve(error["matcher_script"]), error["matcher_script_sha256"]),
    }
    evidence: dict[str, Any] = {}
    for name, (path, expected) in checks.items():
        actual = sha256_file(path) if path.exists() else None
        evidence[name] = {"path": str(path), "expected": expected, "actual": actual}
        if actual != expected:
            raise GlobalStopError(f"GLOBAL STOP: trusted source hash mismatch for {name}: {path}")
    provenance = json.loads(_resolve(trusted["provenance_file"]).read_text())
    if provenance.get("provenance_hash") != trusted["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 provenance hash mismatch")
    stored_config = json.loads(_resolve(trusted["config_file"]).read_text())
    if stored_config.get("config_hash") != trusted["config_hash"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 config hash mismatch")
    return evidence


def base_config(config: dict[str, Any]) -> dict[str, Any]:
    """Load the hash-verified error-shape config used for anchor reconstruction."""
    validate_source_hashes(config)
    loaded = base.load_config(_resolve(config["validated_error_shape_source"]["config_file"]))
    if _resolve(loaded["source_v3"]["root"]).resolve() != _resolve(config["trusted_v3"]["root"]).resolve():
        raise GlobalStopError("GLOBAL STOP: error-shape source root differs from trusted v3 root")
    if loaded["source_v3"]["raw_results_sha256"] != config["trusted_v3"]["raw_results_sha256"]:
        raise GlobalStopError("GLOBAL STOP: error-shape source hash differs from trusted v3 hash")
    anchor = config["anchor"]
    replay = loaded["fp16_replay"]
    if (
        replay["prompt_id"] != anchor["prompt_id"]
        or int(replay["generation_seed"]) != int(anchor["generation_seed"])
        or int(replay["checkpoint_step"]) != int(anchor["checkpoint_step"])
        or float(replay["original_frame_ssim_mean"]) != float(anchor["original_frame_ssim_mean"])
    ):
        raise GlobalStopError("GLOBAL STOP: configured anchor differs from validated FP16 replay identity")
    for key in ("model", "model_revision", "scheduler", "generation"):
        if loaded[key] != config[key]:
            raise GlobalStopError(f"GLOBAL STOP: generation setting {key} differs from validated source")
    return loaded


def validate_output_namespace(config: dict[str, Any], output_dir: Path) -> None:
    output = output_dir.resolve()
    protected = [
        _resolve(config["trusted_v3"]["root"]).resolve(),
        _resolve(config["validated_error_shape_source"]["root"]).resolve(),
        _resolve("results/video_checkpoint_stability_killtest").resolve(),
    ]
    if any(output == root or root in output.parents for root in protected):
        raise GlobalStopError("GLOBAL STOP: output namespace overlaps a trusted result namespace")


# --------------------------------------------------------------------------- provenance


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _relevant_paths(paths: list[Path]) -> list[str]:
    return [str(path.resolve().relative_to(REPO_ROOT)) for path in paths]


def _relevant_diff(paths: list[Path]) -> str:
    return _git_value("diff", "--", *_relevant_paths(paths)) or ""


def _relevant_status(paths: list[Path]) -> list[str]:
    """git status restricted to the files that define this experiment; unrelated repo state is excluded."""
    status = _git_value("status", "--short", "--", *_relevant_paths(paths)) or ""
    return status.splitlines()


def trusted_resolved_model_revision(config: dict[str, Any]) -> str:
    environment = json.loads((_resolve(config["trusted_v3"]["root"]) / "environment.json").read_text())
    revision = environment.get("resolved_model_revision")
    if not isinstance(revision, str) or not revision:
        raise GlobalStopError("GLOBAL STOP: trusted v3 environment does not record a resolved model revision")
    if environment.get("model") != config["model"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 environment model differs from configured model")
    return revision


def build_provenance(config_path: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    runner = REPO_ROOT / "experiments/run_video_bf16_single_flip_killtest_gpu0.sh"
    pipeline = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"
    scheduler = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py"
    error_script = REPO_ROOT / "experiments/video_runtime_error_shape_killtest.py"
    error_config = REPO_ROOT / "experiments/video_runtime_error_shape_killtest_config.yaml"
    v3_script = REPO_ROOT / "experiments/video_runtime_state_discovery.py"
    files = [script, config_path, runner, pipeline, scheduler, error_script, error_config, v3_script]
    status = _relevant_status(files)
    document = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty_relevant": bool(status),
        "git_status_relevant": status,
        "relevant_paths": _relevant_paths(files),
        "relevant_diff_sha256": sha256_bytes(_relevant_diff(files).encode()),
        "experiment_script_sha256": sha256_file(script),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner) if runner.exists() else None,
        "pipeline_wan2_2_sha256": sha256_file(pipeline),
        "scheduler_sha256": sha256_file(scheduler),
        "error_shape_script_sha256": sha256_file(error_script),
        "error_shape_config_sha256": sha256_file(error_config),
        "v3_script_sha256": sha256_file(v3_script),
        "trusted_v3_raw_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/raw_results.csv")
        ),
        "trusted_v3_config_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/preregistered_config.yaml")
        ),
        "trusted_v3_provenance_file_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/run_provenance.json")
        ),
        "trusted_v3_environment_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/environment.json")
        ),
        "trusted_v3_resolved_model_revision": json.loads(
            _resolve("results/video_runtime_state_discovery_v3_corrected/environment.json").read_text()
        ).get("resolved_model_revision"),
    }
    document["provenance_hash"] = sha256_bytes(canonical_json(document))
    return document


def environment_document(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for module_name in ("torch", "diffusers", "transformers", "skimage", "numpy"):
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
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "cuda_version": cuda_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_model": gpu_model,
        "model": config["model"],
        "requested_model_revision": config.get("model_revision"),
        "pinned_model_revision_from_trusted_v3": trusted_resolved_model_revision(config),
        "experiment_version": config["experiment_version"],
        "trusted_v3_root": config["trusted_v3"]["root"],
        "provenance": provenance,
    }


def assert_provenance_matches(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists() or json.loads(path.read_text()) != expected:
        raise GlobalStopError("GLOBAL STOP: prior mode provenance differs from current content")


def _gate_document_hash(gates: list[dict[str, Any]], provenance_hash: str, manifest_sha256: str) -> str:
    return sha256_bytes(
        canonical_json({"gates": gates, "provenance_hash": provenance_hash, "anchor_manifest_sha256": manifest_sha256})
    )


def require_mode_gate(output_dir: Path, name: str, provenance: dict[str, Any], manifest_sha256: str) -> None:
    """Fail closed unless the prerequisite gate file is complete, all-PASS, and bound to provenance and manifest."""
    assert_provenance_matches(output_dir / "run_provenance.json", provenance)
    path = output_dir / name
    if name not in REQUIRED_GATE_NAMES:
        raise GlobalStopError(f"GLOBAL STOP: unknown prerequisite gate file {name}")
    if not path.exists():
        raise GlobalStopError(f"GLOBAL STOP: prerequisite gate did not pass: {path}")
    try:
        document = json.loads(path.read_text())
        gates = document["gates"]
        names = {gate["name"] for gate in gates}
        if names != REQUIRED_GATE_NAMES[name] or len(names) != len(gates):
            raise GlobalStopError(f"GLOBAL STOP: prerequisite gate file lacks the required gate set: {path}")
        if document.get("provenance_hash") != provenance["provenance_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: prerequisite gates were produced under other provenance: {path}")
        if document.get("anchor_manifest_sha256") != manifest_sha256:
            raise GlobalStopError(f"GLOBAL STOP: prerequisite gates are bound to another anchor manifest: {path}")
        if document.get("gates_sha256") != _gate_document_hash(gates, provenance["provenance_hash"], manifest_sha256):
            raise GlobalStopError(f"GLOBAL STOP: prerequisite gate content hash mismatch: {path}")
        if not v3.validate_gate_records(gates) or document.get("all_passed") is not True:
            raise GlobalStopError(f"GLOBAL STOP: prerequisite gate did not pass: {path}")
    except GlobalStopError:
        raise
    except Exception as error:
        raise GlobalStopError(f"GLOBAL STOP: malformed prerequisite gate file {path}: {error}") from error


def _gate(
    name: str,
    passed: bool | None,
    evidence: Any,
    expected: str,
    artifacts: Iterable[str | Path] = (),
    *,
    required: bool = True,
) -> dict[str, Any]:
    return v3.gate_record(name, passed, evidence, artifacts, expected, required=required)


def _write_gates(path: Path, gates: list[dict[str, Any]], provenance: dict[str, Any], manifest_sha256: str) -> None:
    all_passed = v3.validate_gate_records(gates)
    names = [gate["name"] for gate in gates]
    if path.name in REQUIRED_GATE_NAMES and (set(names) != REQUIRED_GATE_NAMES[path.name] or len(set(names)) != len(names)):
        raise GlobalStopError(f"GLOBAL STOP: gate list for {path.name} differs from the preregistered gate set")
    atomic_json(
        path,
        {
            "all_passed": all_passed,
            "gates": gates,
            "provenance_hash": provenance["provenance_hash"],
            "anchor_manifest_sha256": manifest_sha256,
            "gates_sha256": _gate_document_hash(gates, provenance["provenance_hash"], manifest_sha256),
        },
    )
    if not all_passed:
        failed = [row["name"] for row in gates if row["required"] and row["status"] != "PASS"]
        raise GlobalStopError(f"GLOBAL STOP: required gates failed: {failed}")


# --------------------------------------------------------------------------- BF16 bit arithmetic

BF16_EXPONENT_MASK = 0x7F80
BF16_SIGN_MASK = 0x8000
BF16_MAGNITUDE_MASK = 0x7FFF


def bf16_bits_to_float32(bits: np.ndarray | int) -> np.ndarray:
    """Exact widening of BF16 bit patterns to float32 (upper 16 bits)."""
    array = np.atleast_1d(np.asarray(bits, dtype=np.uint16))
    return (array.astype(np.uint32) << np.uint32(16)).view(np.float32)


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Runtime-BF16 bit patterns (RNE, identical to torch .to(bfloat16))."""
    return base.encode_runtime_bf16(np.asarray(values, dtype=np.float32))


def is_finite_bf16_bits(bits: np.ndarray | int) -> np.ndarray:
    array = np.asarray(bits, dtype=np.uint16)
    return (array & np.uint16(BF16_EXPONENT_MASK)) != np.uint16(BF16_EXPONENT_MASK)


def adjacent_bf16_bits(bits: int, direction: str) -> int:
    """Bit pattern of the immediately adjacent finite BF16 value in value order.

    ``up`` is the next larger representable value, ``down`` the next smaller
    one.  Both zeros are treated as the single value 0.  Transitions that would
    leave the finite range raise.  The primitive is verified exhaustively by
    :func:`verify_adjacent_bf16_by_enumeration`.
    """
    if direction not in DIRECTION_SIGN:
        raise ValueError(f"Unknown direction {direction!r}")
    pattern = int(bits)
    if not 0 <= pattern <= 0xFFFF:
        raise ValueError("BF16 bit pattern out of range")
    if not bool(is_finite_bf16_bits(pattern)):
        raise ValueError("Adjacent value of a non-finite BF16 pattern is undefined")
    sign = (pattern >> 15) & 1
    magnitude = pattern & BF16_MAGNITUDE_MASK
    up = DIRECTION_SIGN[direction] > 0
    if magnitude == 0:
        result = 0x0001 if up else 0x8001
    elif sign == 0:
        result = magnitude + 1 if up else magnitude - 1
    else:
        if up:
            result = 0x0000 if magnitude == 1 else (BF16_SIGN_MASK | (magnitude - 1))
        else:
            result = BF16_SIGN_MASK | (magnitude + 1)
    if not bool(is_finite_bf16_bits(result)):
        raise ValueError("Adjacent BF16 transition leaves the finite range")
    return int(result)


def adjacent_bf16_value(value: float, direction: str) -> tuple[np.float32, int, int]:
    """Return (adjacent float32 value, clean bits, adjacent bits) for a BF16-representable value."""
    clean = np.asarray([value], dtype=np.float32)
    bits = int(float32_to_bf16_bits(clean)[0])
    if float(bf16_bits_to_float32(bits)[0]) != float(clean[0]):
        raise ValueError("Value is not BF16-representable; refusing to perturb a rounded value")
    neighbour = adjacent_bf16_bits(bits, direction)
    return bf16_bits_to_float32(neighbour)[0], bits, neighbour


def verify_adjacent_bf16_by_enumeration() -> dict[str, Any]:
    """Independent check: neighbours must be successors/predecessors in sorted value order.

    All 65,536 patterns are enumerated; the finite ones are sorted by value
    (both zeros collapse to one value).  The primitive is compared against the
    sorted neighbour for every finite pattern in both directions.  A second,
    independent cross-check against ``torch.nextafter`` on bfloat16 is
    reported when the local torch build supports it.
    """
    all_bits = np.arange(65536, dtype=np.uint16)
    finite = all_bits[is_finite_bf16_bits(all_bits)]
    values = bf16_bits_to_float32(finite).astype(np.float64)
    unique_values = np.unique(values)
    position = {float(value): index for index, value in enumerate(unique_values)}
    up_mismatch = down_mismatch = 0
    up_checked = down_checked = 0
    boundary_rejections = 0
    for pattern in finite.tolist():
        value = float(bf16_bits_to_float32(pattern)[0])
        index = position[value]
        for direction, offset in (("up", 1), ("down", -1)):
            target = index + offset
            if 0 <= target < len(unique_values):
                got = float(bf16_bits_to_float32(adjacent_bf16_bits(pattern, direction))[0])
                if direction == "up":
                    up_checked += 1
                    up_mismatch += got != float(unique_values[target])
                else:
                    down_checked += 1
                    down_mismatch += got != float(unique_values[target])
            else:
                try:
                    adjacent_bf16_bits(pattern, direction)
                except ValueError:
                    boundary_rejections += 1
    torch_agreement: bool | None = None
    torch_error: str | None = None
    try:
        import torch

        sample = torch.tensor(
            [1e-6, -3.2e-7, 1.0, -1.0, 0.0, -2.4e-6, 6.0e-6, 65504.0, -1e-30, 3.0e-39],
            dtype=torch.bfloat16,
        )
        inf = torch.tensor(float("inf"), dtype=torch.bfloat16)
        expected_up = torch.nextafter(sample, inf).float().tolist()
        expected_down = torch.nextafter(sample, -inf).float().tolist()
        ours_up = [float(adjacent_bf16_value(v, "up")[0]) for v in sample.float().tolist()]
        ours_down = [float(adjacent_bf16_value(v, "down")[0]) for v in sample.float().tolist()]
        torch_agreement = ours_up == expected_up and ours_down == expected_down
    except Exception as error:  # pragma: no cover - depends on the local torch build
        torch_error = f"{type(error).__name__}: {error}"
    passed = (
        up_mismatch == 0
        and down_mismatch == 0
        and up_checked == len(unique_values) - 1 + (len(finite) - len(unique_values))
        and down_checked == up_checked
        and boundary_rejections == 2  # +max up and -max down; zeros are interior values
        and torch_agreement in (True, None)
    )
    return {
        "passed": bool(passed),
        "finite_patterns": int(len(finite)),
        "unique_finite_values": int(len(unique_values)),
        "up_checked": int(up_checked),
        "down_checked": int(down_checked),
        "up_mismatches": int(up_mismatch),
        "down_mismatches": int(down_mismatch),
        "boundary_rejections": int(boundary_rejections),
        "torch_nextafter_agreement": torch_agreement,
        "torch_nextafter_error": torch_error,
    }


# --------------------------------------------------------------------------- anchor derivations


def assert_bf16_representable(clean: np.ndarray) -> np.ndarray:
    """The trusted clean state is FP32 storage of BF16 runtime values; enforce that."""
    runtime = base.cast_runtime_bf16(clean)
    if runtime.shape != clean.shape or not np.array_equal(runtime.view(np.uint32), clean.view(np.uint32)):
        raise GlobalStopError("GLOBAL STOP: anchor clean state is not bit-exact BF16-representable")
    return float32_to_bf16_bits(clean).reshape(-1)


def derive_eligible_coordinates(clean: np.ndarray, threshold: float) -> np.ndarray:
    """Flat indices with |runtime-BF16 value| < threshold, ascending."""
    runtime = base.cast_runtime_bf16(clean).reshape(-1)
    eligible = np.flatnonzero(np.abs(runtime) < np.float32(threshold))
    return np.sort(eligible).astype(np.int64)


def count_below(clean: np.ndarray, threshold: float) -> int:
    runtime = base.cast_runtime_bf16(clean).reshape(-1)
    return int(np.count_nonzero(np.abs(runtime) < np.float32(threshold)))


def adjacent_step_count(clean_bits: int, target_bits: int, limit: int = 4096) -> tuple[int, str]:
    """Number of adjacent-BF16 steps from clean to target and the direction walked."""
    clean_value = float(bf16_bits_to_float32(clean_bits)[0])
    target_value = float(bf16_bits_to_float32(target_bits)[0])
    if clean_value == target_value:
        return 0, "none"
    direction = "up" if target_value > clean_value else "down"
    current = clean_bits
    steps = 0
    while float(bf16_bits_to_float32(current)[0]) != target_value:
        current = adjacent_bf16_bits(current, direction)
        steps += 1
        if steps > limit:
            raise GlobalStopError("GLOBAL STOP: historical delta is not reachable by adjacent steps within limit")
    return steps, direction


def derive_historical_delta(clean: np.ndarray, fp16_candidate: np.ndarray) -> dict[str, Any]:
    """Re-derive e* = runtime_bf16(fp16 probe) - runtime_bf16(clean) coordinate by coordinate."""
    if clean.shape != fp16_candidate.shape:
        raise GlobalStopError("GLOBAL STOP: historical FP16 probe shape differs from clean state")
    clean_bits = float32_to_bf16_bits(clean).reshape(-1)
    probe_bits = float32_to_bf16_bits(fp16_candidate).reshape(-1)
    changed = np.flatnonzero(clean_bits != probe_bits)
    clean_runtime = bf16_bits_to_float32(clean_bits)
    probe_runtime = bf16_bits_to_float32(probe_bits)
    delta = probe_runtime.astype(np.float64) - clean_runtime.astype(np.float64)
    coordinates = []
    for flat in changed.tolist():
        steps, direction = adjacent_step_count(int(clean_bits[flat]), int(probe_bits[flat]))
        coordinates.append(
            {
                "coordinate_flat_index": int(flat),
                "coordinate_multi_index": [int(v) for v in np.unravel_index(flat, clean.shape)],
                "clean_value": float(clean_runtime[flat]),
                "perturbed_value": float(probe_runtime[flat]),
                "delta": float(delta[flat]),
                "abs_clean_value": float(abs(clean_runtime[flat])),
                "clean_bf16_bits_hex": f"0x{int(clean_bits[flat]):04x}",
                "perturbed_bf16_bits_hex": f"0x{int(probe_bits[flat]):04x}",
                "adjacent_steps": int(steps),
                "direction": direction,
            }
        )
    return {
        "changed_coordinate_count": int(len(changed)),
        "changed_coordinates": coordinates,
        "delta_l2": float(np.sqrt(np.sum(delta**2))),
        "delta_linf": float(np.max(np.abs(delta))) if len(changed) else 0.0,
        "single_adjacent_step_count": int(sum(row["adjacent_steps"] == 1 for row in coordinates)),
        "clean_runtime_sha256": sha256_bytes(clean_bits.tobytes()),
        "probe_runtime_sha256": sha256_bytes(probe_bits.tobytes()),
    }


def condition_id(flat_index: int, direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}")
    return f"c{int(flat_index):07d}_{direction}"


def expected_primary_keys(eligible: np.ndarray) -> list[tuple[int, str]]:
    keys = [(int(flat), direction) for flat in eligible.tolist() for direction in DIRECTIONS]
    if len(set(keys)) != len(keys):
        raise GlobalStopError("GLOBAL STOP: duplicate primary keys")
    return keys


def build_single_flip_state(clean: np.ndarray, flat_index: int, direction: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the perturbed FP32 state whose runtime-BF16 image differs from clean in exactly one coordinate."""
    clean_bits = assert_bf16_representable(clean)
    if not 0 <= int(flat_index) < clean_bits.size:
        raise ValueError("Coordinate index out of range")
    original = int(clean_bits[flat_index])
    neighbour = adjacent_bf16_bits(original, direction)
    perturbed_bits = clean_bits.copy()
    perturbed_bits[flat_index] = np.uint16(neighbour)
    state = bf16_bits_to_float32(perturbed_bits).reshape(clean.shape).astype(np.float32, copy=True)
    # Hard isolation asserts.  Every one of these is a construction failure, not a result.
    runtime_bits = float32_to_bf16_bits(state).reshape(-1)
    changed = np.flatnonzero(runtime_bits != clean_bits)
    if changed.size != 1 or int(changed[0]) != int(flat_index):
        raise GlobalStopError("GLOBAL STOP: single-flip state changed a coordinate set other than the requested one")
    if int(runtime_bits[flat_index]) != neighbour or not np.array_equal(runtime_bits, perturbed_bits):
        raise GlobalStopError("GLOBAL STOP: single-flip state was rounded away from the adjacent BF16 value")
    clean_value = float(bf16_bits_to_float32(original)[0])
    perturbed_value = float(bf16_bits_to_float32(neighbour)[0])
    if not ((perturbed_value > clean_value) == (direction == "up") and perturbed_value != clean_value):
        raise GlobalStopError("GLOBAL STOP: adjacent value moved in the wrong direction")
    # Realized error accounting, measured on the actual state difference (no rescaling is ever applied).
    difference = state.astype(np.float64) - clean.astype(np.float64)
    realized_nonzero = int(np.count_nonzero(difference))
    realized_linf = float(np.max(np.abs(difference)))
    realized_l2 = float(np.sqrt(np.sum(difference**2)))
    realized_mse = float(np.mean(difference**2))
    if realized_nonzero != 1 or realized_linf != abs(perturbed_value - clean_value) or realized_l2 != realized_linf:
        raise GlobalStopError("GLOBAL STOP: realized perturbation accounting disagrees with the single flip")
    record = {
        "condition_id": condition_id(flat_index, direction),
        "perturbation_family": PERTURBATION_FAMILY,
        "coordinate_flat_index": int(flat_index),
        "coordinate_multi_index": [int(v) for v in np.unravel_index(int(flat_index), clean.shape)],
        "direction": direction,
        "clean_value": clean_value,
        "perturbed_value": perturbed_value,
        "delta": perturbed_value - clean_value,
        "abs_clean_value": abs(clean_value),
        "clean_bf16_bits_hex": f"0x{original:04x}",
        "perturbed_bf16_bits_hex": f"0x{neighbour:04x}",
        "changed_coordinate_count": int(changed.size),
        "realized_nonzero_elements": realized_nonzero,
        "total_elements": int(clean.size),
        "realized_l2": realized_l2,
        "realized_mse": realized_mse,
        "realized_linf": realized_linf,
        "runtime_input_hash": sha256_bytes(runtime_bits.tobytes()),
        "state_fp32_sha256": v3.array_sha256(state),
        # Frozen expected candidate: the ONLY authority a GPU row may be validated against.
        "expected_candidate_tensor_identity_sha256_v1": tensor_identity_sha256_v1(state),
        "expected_candidate_shape": [int(v) for v in state.shape],
        "expected_candidate_dtype": CANDIDATE_STORAGE_DTYPE,
        "expected_candidate_runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "expected_candidate_raw_bf16_bytes_sha256": sha256_bytes(runtime_bits.tobytes()),
        "expected_changed_flat_index": int(flat_index),
        "expected_clean_bf16_bits": original,
        "expected_perturbed_bf16_bits": neighbour,
    }
    if np.dtype(state.dtype).newbyteorder("<").str != CANDIDATE_STORAGE_DTYPE:
        raise GlobalStopError("GLOBAL STOP: candidate storage dtype drifted")
    return state, record


def build_anchor_manifest(
    config: dict[str, Any],
    source: base.SourceTrajectory,
    meta: dict[str, Any],
    eligible: np.ndarray,
    historical: dict[str, Any],
    constructions: list[dict[str, Any]],
) -> dict[str, Any]:
    perturbation = config["perturbation"]
    historical_set = {row["coordinate_flat_index"] for row in historical["changed_coordinates"]}
    manifest = {
        "experiment_version": config["experiment_version"],
        "config_hash": config_hash(config),
        "anchor": {
            "prompt_id": source.prompt_id,
            "generation_seed": source.seed,
            "checkpoint_step": source.checkpoint_step,
            "prompt": source.prompt,
            "clean_checkpoint_hash": source.clean_hash,
            "clean_runtime_bf16_sha256": historical["clean_runtime_sha256"],
            "checkpoint_path": str(source.checkpoint_path),
            "manifest_path": str(source.manifest_path),
            "state_shape": list(source.clean.shape),
            "runtime_numel": int(source.clean.size),
            "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "runtime_full_bytes": int(source.clean.size) * 2,
            "clean_state_identity_sha256_v1": tensor_identity_sha256_v1(source.clean),
            "clean_final_latent_identity_sha256_v1": tensor_identity_sha256_v1(source.final_latent),
            "clean_video_identity_sha256_v1": tensor_identity_sha256_v1(source.video),
            "resume_index": source.checkpoint_step,
            "resume_timestep": anchor_resume_timestep(config, source.checkpoint_step),
            "scheduler_config": config["scheduler"],
            "scheduler_class": EXPECTED_SCHEDULER_CLASS,
            "scheduler_identity": frozen_scheduler_identity(config, source.checkpoint_step),
            "model": config["model"],
            "pinned_model_revision_from_trusted_v3": trusted_resolved_model_revision(config),
            "original_v3_fp16_probe_sha256": meta["original_v3_fp16_probe_sha256"],
            "frozen_reconstructed_fp16_runtime_sha256": meta["frozen_reconstructed_fp16_runtime_sha256"],
        },
        "eligibility": {
            "abs_threshold": float(perturbation["eligibility_abs_threshold"]),
            "eligible_count": int(eligible.size),
            "eligible_flat_indices": [int(v) for v in eligible.tolist()],
            "secondary_report_abs_threshold": float(perturbation["secondary_report_abs_threshold"]),
            "count_below_secondary_threshold": count_below(
                source.clean, float(perturbation["secondary_report_abs_threshold"])
            ),
            "historical_support_subset_of_eligible": historical_set <= set(int(v) for v in eligible.tolist()),
        },
        "historical_fp16_delta": historical,
        "directions": list(DIRECTIONS),
        "primary_row_count": int(eligible.size) * len(DIRECTIONS),
        "constructions": constructions,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def reconstruct_anchor(config: dict[str, Any]) -> tuple[base.SourceTrajectory, np.ndarray, dict[str, Any]]:
    """Reconstruct the anchor clean state and the historical FP16 probe via the validated path."""
    loaded = base_config(config)
    anomaly = base.derive_unique_fp16_anomaly_from_v3(base.source_rows(loaded))
    source, candidate, meta = base._fp16_source_candidate(loaded)
    anchor = config["anchor"]
    identity = (anchor["prompt_id"], int(anchor["generation_seed"]), int(anchor["checkpoint_step"]))
    if (source.prompt_id, source.seed, source.checkpoint_step) != identity:
        raise GlobalStopError("GLOBAL STOP: reconstructed anchor identity differs from configured anchor")
    if (anomaly["prompt_id"], anomaly["generation_seed"], anomaly["checkpoint_step"]) != identity:
        raise GlobalStopError("GLOBAL STOP: derived unique v3 FP16 anomaly differs from configured anchor")
    if anomaly["final_latent_exact"] or anomaly["video_exact"]:
        raise GlobalStopError("GLOBAL STOP: derived anomaly row is exact; anchor premise failed")
    meta = {**meta, "derived_unique_anomaly": anomaly}
    assert_bf16_representable(source.clean)
    return source, candidate, meta


def derive_all(config: dict[str, Any]) -> dict[str, Any]:
    source, candidate, meta = reconstruct_anchor(config)
    perturbation = config["perturbation"]
    eligible = derive_eligible_coordinates(source.clean, float(perturbation["eligibility_abs_threshold"]))
    if eligible.size == 0:
        raise GlobalStopError("GLOBAL STOP: eligible set is empty; the preregistered threshold selects nothing")
    historical = derive_historical_delta(source.clean, candidate)
    constructions = []
    for flat, direction in expected_primary_keys(eligible):
        _, record = build_single_flip_state(source.clean, flat, direction)
        record["historical_fp16_support_member"] = flat in {
            row["coordinate_flat_index"] for row in historical["changed_coordinates"]
        }
        constructions.append(record)
    manifest = build_anchor_manifest(config, source, meta, eligible, historical, constructions)
    return {
        "source": source,
        "candidate": candidate,
        "meta": meta,
        "eligible": eligible,
        "historical": historical,
        "constructions": constructions,
        "manifest": manifest,
    }


# --------------------------------------------------------------------------- analysis


def classify_row_group(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Classify one (coordinate, direction) key from its executed replay rows."""
    trigger = float(config["replay"]["trigger_frame_ssim_below"])
    total = int(config["replay"]["total_runs_per_triggered_row"])
    benign_floor = float(config["analysis"]["no_go_frame_ssim_at_least"])
    ordered = sorted(rows, key=lambda row: int(row["replay_id"]))
    if [int(row["replay_id"]) for row in ordered] != list(range(len(ordered))):
        raise GlobalStopError("GLOBAL STOP: replay ids are not contiguous from zero")
    ssims = [float(row["frame_ssim_mean"]) for row in ordered]
    if any(not math.isfinite(value) for value in ssims):
        raise GlobalStopError("GLOBAL STOP: non-finite frame SSIM in decision inputs")
    primary_ssim = ssims[0]
    triggered = primary_ssim < trigger
    if not triggered and len(ordered) != 1:
        raise GlobalStopError("GLOBAL STOP: non-triggered row executed more than once")
    latent_hashes = {row["recovered_final_latent_identity_sha256_v1"] for row in ordered}
    video_hashes = {row["recovered_video_identity_sha256_v1"] for row in ordered}
    all_catastrophic = all(value < trigger for value in ssims)
    identical_metrics = len(set(ssims)) == 1
    # Replay determinism: every equality field identical across all runs of the row.
    equality = {field: len({repr(row[field]) for row in ordered}) == 1 for field in REPLAY_EQUALITY_FIELDS}
    bit_deterministic = all(equality.values())
    if triggered and len(ordered) < total:
        row_class = "TRIGGERED_INCOMPLETE"
    elif triggered and all_catastrophic and bit_deterministic:
        row_class = "CATASTROPHIC_DETERMINISTIC"
    elif triggered:
        row_class = "CATASTROPHIC_NONDETERMINISTIC"
    elif primary_ssim >= benign_floor:
        row_class = "BENIGN"
    else:
        row_class = "INTERMEDIATE"
    return {
        "coordinate_flat_index": int(ordered[0]["coordinate_flat_index"]),
        "direction": ordered[0]["direction"],
        "run_count": len(ordered),
        "primary_frame_ssim_mean": primary_ssim,
        "min_frame_ssim_mean": min(ssims),
        "max_frame_ssim_mean": max(ssims),
        "triggered": triggered,
        "all_runs_catastrophic": all_catastrophic,
        "bit_deterministic_across_runs": bit_deterministic,
        "identical_frame_ssim_across_runs": identical_metrics,
        "replay_equality_by_field": equality,
        "distinct_final_latent_hashes": len(latent_hashes),
        "distinct_video_hashes": len(video_hashes),
        "row_class": row_class,
    }


def analyze_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    expected_keys: list[tuple[int, str]],
    *,
    controls_passed: bool,
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["coordinate_flat_index"]), str(row["direction"]))
        grouped.setdefault(key, []).append(row)
    expected = list(expected_keys)
    if set(grouped) != set(expected) or len(set(expected)) != len(expected):
        missing = sorted(set(expected) - set(grouped))
        extra = sorted(set(grouped) - set(expected))
        raise GlobalStopError(f"GLOBAL STOP: executed key set differs from frozen keys; missing={missing} extra={extra}")
    summaries = [classify_row_group(grouped[key], config) for key in expected]
    classes = {name: sum(row["row_class"] == name for row in summaries) for name in ROW_CLASSES}
    all_rows_benign = all(
        float(row["frame_ssim_mean"]) >= float(config["analysis"]["no_go_frame_ssim_at_least"]) for row in rows
    )
    incomplete = classes["TRIGGERED_INCOMPLETE"] > 0
    if not controls_passed:
        decision = "WEAK_INCONCLUSIVE"
        reason = "FULL-direct controls or correctness gates did not pass"
    elif incomplete:
        decision = "WEAK_INCONCLUSIVE"
        reason = "a triggered row lacks its preregistered replays"
    elif classes["CATASTROPHIC_DETERMINISTIC"] > 0:
        decision = "GO_TO_LOCAL_BRANCH_MAP"
        reason = "at least one single adjacent-BF16 flip reproduced deterministic catastrophic divergence"
    elif all_rows_benign:
        decision = "NO_GO"
        reason = "every single adjacent-BF16 flip stayed at or above the preregistered benign floor"
    else:
        decision = "WEAK_INCONCLUSIVE"
        reason = "intermediate or non-deterministic outcomes without a deterministic catastrophic row"
    by_direction = {
        direction: {
            "count": sum(row["direction"] == direction for row in summaries),
            "catastrophic_deterministic": sum(
                row["direction"] == direction and row["row_class"] == "CATASTROPHIC_DETERMINISTIC" for row in summaries
            ),
            "min_primary_ssim": min(
                (row["primary_frame_ssim_mean"] for row in summaries if row["direction"] == direction), default=None
            ),
        }
        for direction in DIRECTIONS
    }
    return {
        "decision": decision,
        "decision_reason": reason,
        "controls_passed": bool(controls_passed),
        "decision_input_fields": sorted(DECISION_INPUT_FIELDS),
        "expected_key_count": len(expected),
        "executed_row_count": len(rows),
        "row_class_counts": classes,
        "all_rows_at_or_above_no_go_floor": all_rows_benign,
        "min_primary_frame_ssim_mean": min(row["primary_frame_ssim_mean"] for row in summaries),
        "by_direction_descriptive": by_direction,
        "row_summaries": summaries,
    }


def decision_source_field_audit() -> dict[str, Any]:
    """AST sweep: decision code may only read the preregistered decision fields from rows."""
    forbidden = set()
    for name in ("analyze_rows", "classify_row_group"):
        tree = ast.parse(inspect.getsource(globals()[name]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in {
                    "temporal_delta_mse",
                    "temporal_delta_agreement",
                    "prompt_clip_score",
                    "video_psnr",
                    "recovered_final_latent_sha256",
                    "recovered_video_sha256",
                    "exact_final_latent",
                    "exact_video",
                    "historical_fp16_support_member",
                    "abs_clean_value",
                }:
                    forbidden.add(node.value)
    return {"passed": not forbidden, "forbidden_fields_referenced": sorted(forbidden)}


# --------------------------------------------------------------------------- modes


def _scientific_row(
    config: dict[str, Any],
    provenance: dict[str, Any],
    source: base.SourceTrajectory,
    manifest: dict[str, Any],
    record: dict[str, Any],
    replay_id: int,
    candidate_record: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    manifest_sha256 = manifest["manifest_sha256"]
    latent_record = _identity_record(result["recovered_final_latent_artifact"])
    video_record = _identity_record(result["recovered_video_artifact"])
    if "tensor_identity_sha256_v1" not in candidate_record:
        raise GlobalStopError("GLOBAL STOP: candidate record lacks canonical identity")
    return {
        "status": "COMPLETE",
        "experiment_version": config["experiment_version"],
        "config_hash": config_hash(config),
        "provenance_hash": provenance["provenance_hash"],
        "source_raw_sha256": config["trusted_v3"]["raw_results_sha256"],
        "model": config["model"],
        "scheduler": EXPECTED_SCHEDULER,
        "scheduler_class": result["scheduler_class"],
        "prompt_id": source.prompt_id,
        "prompt_text": source.prompt,
        "generation_seed": source.seed,
        "checkpoint_step": source.checkpoint_step,
        "resume_index": result["resume_index"],
        "resume_timestep": manifest["anchor"]["resume_timestep"],
        "scheduler_config_json": canonical_json(config["scheduler"]).decode(),
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "latent_shape_json": json.dumps([int(v) for v in source.clean.shape]),
        "clean_checkpoint_hash": source.clean_hash,
        "clean_state_identity_sha256_v1": manifest["anchor"]["clean_state_identity_sha256_v1"],
        "anchor_manifest_sha256": manifest_sha256,
        "condition_id": record["condition_id"],
        "perturbation_family": PERTURBATION_FAMILY,
        "coordinate_flat_index": record["coordinate_flat_index"],
        "coordinate_multi_index": json.dumps(record["coordinate_multi_index"]),
        "direction": record["direction"],
        "requested_direction": record["direction"],
        "replay_id": replay_id,
        "clean_value": record["clean_value"],
        "perturbed_value": record["perturbed_value"],
        "delta": record["delta"],
        "abs_clean_value": record["abs_clean_value"],
        "clean_bf16_bits_hex": record["clean_bf16_bits_hex"],
        "perturbed_bf16_bits_hex": record["perturbed_bf16_bits_hex"],
        "changed_coordinate_count": record["changed_coordinate_count"],
        "realized_nonzero_elements": record["realized_nonzero_elements"],
        "total_elements": record["total_elements"],
        "realized_l2": record["realized_l2"],
        "realized_mse": record["realized_mse"],
        "realized_linf": record["realized_linf"],
        "historical_fp16_support_member": record["historical_fp16_support_member"],
        "runtime_input_hash": record["runtime_input_hash"],
        "final_latent_mse": result["final_latent_mse"],
        "exact_final_latent": result["exact_final_latent"],
        "exact_video": result["exact_video"],
        "video_mse": result["video_mse"],
        "video_psnr": result["video_psnr"],
        "frame_ssim_mean": result["frame_ssim_mean"],
        "temporal_delta_mse": result["temporal_delta_mse"],
        "temporal_delta_agreement": result["temporal_delta_agreement"],
        "prompt_clip_score": "",
        "resume_ms": result["resume_ms"],
        "runtime_candidate_artifact": candidate_record,
        "recovered_final_latent_artifact": latent_record,
        "recovered_video_artifact": video_record,
        "runtime_candidate_identity_sha256_v1": candidate_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_identity_sha256_v1": latent_record["tensor_identity_sha256_v1"],
        "recovered_video_identity_sha256_v1": video_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_sha256": result["recovered_final_latent_sha256"],
        "recovered_video_sha256": result["recovered_video_sha256"],
        "result_path": str(result_path),
    }


def _result_valid(
    path: Path, provenance: dict[str, Any], expected_identity: dict[str, Any], expected_candidate: dict[str, Any]
) -> dict[str, Any] | None:
    """Accept a persisted row only if it is bound to the frozen identity AND the frozen expected candidate."""
    for field in EXPECTED_CANDIDATE_FIELDS:
        if field not in expected_candidate:
            raise GlobalStopError(f"GLOBAL STOP: expected candidate lacks frozen field {field}")
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
        if row["status"] != "COMPLETE":
            raise GlobalStopError(f"GLOBAL STOP: incomplete result exists: {path}")
        if row["provenance_hash"] != provenance["provenance_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: stale result provenance: {path}")
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise GlobalStopError(f"GLOBAL STOP: result identity mismatch: {path}")
        arrays = {}
        for field in ("runtime_candidate_artifact", "recovered_final_latent_artifact", "recovered_video_artifact"):
            arrays[field] = _load_verified_array(row[field])
            if arrays[field] is None:
                raise GlobalStopError(f"GLOBAL STOP: invalid retained artifact {field}: {path}")
        # The persisted candidate must equal the frozen expected construction (authority: CPU manifest) ...
        candidate = arrays["runtime_candidate_artifact"]
        _check_candidate_against_frozen(candidate, expected_candidate, str(path))
        if int(row["coordinate_flat_index"]) != int(expected_candidate["expected_changed_flat_index"]):
            raise GlobalStopError(f"GLOBAL STOP: row coordinate differs from the frozen expected candidate: {path}")
        # ... AND equal the row's own declaration; a row may never redefine its expected identity.
        if row["runtime_candidate_identity_sha256_v1"] != expected_candidate["expected_candidate_tensor_identity_sha256_v1"]:
            raise GlobalStopError(f"GLOBAL STOP: row-declared candidate identity differs from the frozen expected candidate: {path}")
        if sha256_bytes(float32_to_bf16_bits(candidate).tobytes()) != row["runtime_input_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: persisted runtime candidate does not hash to runtime_input_hash: {path}")
        if tensor_identity_sha256_v1(candidate) != row["runtime_candidate_identity_sha256_v1"]:
            raise GlobalStopError(f"GLOBAL STOP: runtime candidate identity is not derived from the persisted artifact: {path}")
        latent = arrays["recovered_final_latent_artifact"]
        if tensor_identity_sha256_v1(latent) != row["recovered_final_latent_identity_sha256_v1"]:
            raise GlobalStopError(f"GLOBAL STOP: recovered latent identity is not derived from the persisted artifact: {path}")
        if v3.array_sha256(latent) != row["recovered_final_latent_sha256"]:
            raise GlobalStopError(f"GLOBAL STOP: recovered latent hash is not derived from the persisted artifact: {path}")
        video = arrays["recovered_video_artifact"]
        if tensor_identity_sha256_v1(video) != row["recovered_video_identity_sha256_v1"]:
            raise GlobalStopError(f"GLOBAL STOP: recovered video identity is not derived from the persisted artifact: {path}")
        if v3.array_sha256(video) != row["recovered_video_sha256"]:
            raise GlobalStopError(f"GLOBAL STOP: recovered video hash is not derived from the persisted artifact: {path}")
        return row
    except GlobalStopError:
        raise
    except Exception as error:
        raise GlobalStopError(f"GLOBAL STOP: malformed result {path}: {error}") from error


def _load_verified_array(record: dict[str, Any]) -> np.ndarray | None:
    """Reload an artifact and require file hash, byte hash, canonical identity, dtype and shape to agree."""
    path = _resolve(record["path"])
    if not path.exists() or sha256_file(path) != record["file_sha256"]:
        return None
    array = np.load(path, allow_pickle=False)
    if v3.array_sha256(array) != record["tensor_sha256"]:
        return None
    identity = _identity_fields(array)
    for key in ("tensor_identity_sha256_v1", "identity_format", "dtype", "shape"):
        if record.get(key) != identity[key]:
            return None
    return array


def verify_row_against_source(row: dict[str, Any], source: base.SourceTrajectory, expected: dict[str, Any]) -> dict[str, Any]:
    """Recompute every decision-relevant quantity of a persisted row from its artifacts and the clean anchor."""
    where = str(row.get("result_path"))
    candidate = _load_verified_array(row["runtime_candidate_artifact"])
    latent = _load_verified_array(row["recovered_final_latent_artifact"])
    video = _load_verified_array(row["recovered_video_artifact"])
    if candidate is None or latent is None or video is None:
        raise GlobalStopError(f"GLOBAL STOP: row artifacts cannot be reloaded: {where}")
    # Shape / dtype / BF16 identity BEFORE any flattening.
    if candidate.shape != source.clean.shape:
        raise GlobalStopError(f"GLOBAL STOP: candidate shape {list(candidate.shape)} differs from clean shape {list(source.clean.shape)}: {where}")
    if candidate.dtype != source.clean.dtype or np.dtype(candidate.dtype).newbyteorder("<").str != CANDIDATE_STORAGE_DTYPE:
        raise GlobalStopError(f"GLOBAL STOP: candidate dtype {candidate.dtype} differs from clean dtype {source.clean.dtype}: {where}")
    if not np.array_equal(base.cast_runtime_bf16(candidate).view(np.uint32), candidate.view(np.uint32)):
        raise GlobalStopError(f"GLOBAL STOP: candidate is not bit-exact {EXPECTED_RUNTIME_DTYPE}: {where}")
    _check_candidate_against_frozen(candidate, expected, where)
    clean_bits = assert_bf16_representable(source.clean)
    candidate_bits = float32_to_bf16_bits(candidate).reshape(-1)
    flat = int(row["coordinate_flat_index"])
    if flat != int(expected["expected_changed_flat_index"]):
        raise GlobalStopError(f"GLOBAL STOP: row coordinate differs from the frozen expected coordinate: {where}")
    if row.get("requested_direction") != row.get("direction"):
        raise GlobalStopError(f"GLOBAL STOP: requested and recorded direction differ: {where}")
    changed = np.flatnonzero(candidate_bits != clean_bits)
    if changed.size != 1:
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate changes {int(changed.size)} coordinates, not exactly one: {where}")
    if int(changed[0]) != flat:
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate changes coordinate {int(changed[0])} instead of {flat}: {where}")
    if int(clean_bits[flat]) != int(expected["expected_clean_bf16_bits"]):
        raise GlobalStopError(f"GLOBAL STOP: clean BF16 bits at the coordinate differ from the frozen expectation: {where}")
    if int(candidate_bits[flat]) != int(expected["expected_perturbed_bf16_bits"]):
        raise GlobalStopError(f"GLOBAL STOP: changed BF16 bits differ from the frozen expected adjacent BF16 bits: {where}")
    if int(candidate_bits[flat]) != adjacent_bf16_bits(int(clean_bits[flat]), str(row["direction"])):
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate is not the claimed adjacent BF16 value: {where}")
    others = np.ones(clean_bits.size, dtype=bool)
    others[flat] = False
    if not np.array_equal(candidate_bits[others], clean_bits[others]):
        raise GlobalStopError(f"GLOBAL STOP: unchanged coordinates are not bit-exact: {where}")
    # Independent realized-perturbation accounting from the two verified BF16 states only.
    difference = bf16_bits_to_float32(candidate_bits).astype(np.float64) - bf16_bits_to_float32(clean_bits).astype(np.float64)
    accounting = {
        "realized_nonzero_elements": int(np.count_nonzero(difference)),
        "realized_l2": float(np.sqrt(np.sum(difference**2))),
        "realized_mse": float(np.mean(difference**2)),
        "realized_linf": float(np.max(np.abs(difference))),
    }
    if accounting["realized_nonzero_elements"] != 1:
        raise GlobalStopError(f"GLOBAL STOP: recomputed perturbation touches {accounting['realized_nonzero_elements']} coordinates: {row.get('result_path')}")
    if row.get("realized_nonzero_elements") != 1 or row.get("total_elements") != int(clean_bits.size):
        raise GlobalStopError(f"GLOBAL STOP: declared perturbation accounting counts are false: {row.get('result_path')}")
    bad_accounting = {
        key: (row.get(key), accounting[key])
        for key in ("realized_l2", "realized_mse", "realized_linf")
        if not _accounting_matches(row.get(key), accounting[key])
    }
    if bad_accounting:
        raise GlobalStopError(f"GLOBAL STOP: declared realized accounting differs from recomputation {bad_accounting}: {row.get('result_path')}")
    metrics = v3.video_metrics(video, source.video)
    latent_error = v3.latent_error(source.final_latent, latent)
    recomputed_exact = {
        "exact_final_latent": bool(np.array_equal(latent, source.final_latent)),
        "exact_video": bool(np.array_equal(video, source.video)),
        "runtime_candidate_identity_sha256_v1": tensor_identity_sha256_v1(candidate),
        "recovered_final_latent_identity_sha256_v1": tensor_identity_sha256_v1(latent),
        "recovered_video_identity_sha256_v1": tensor_identity_sha256_v1(video),
        "recovered_final_latent_sha256": v3.array_sha256(latent),
        "recovered_video_sha256": v3.array_sha256(video),
    }
    recomputed_float = {
        "frame_ssim_mean": float(metrics["frame_ssim_mean"]),
        "video_mse": float(metrics["video_mse"]),
        "final_latent_mse": float(latent_error["mse"]),
    }
    mismatched = {key: (row.get(key), value) for key, value in recomputed_exact.items() if row.get(key) != value}
    mismatched.update({key: (row.get(key), value) for key, value in recomputed_float.items() if not _accounting_matches(row.get(key), value)})
    if mismatched:
        raise GlobalStopError(f"GLOBAL STOP: stored metrics differ from artifact recomputation {mismatched}: {row.get('result_path')}")
    return {**recomputed_exact, **recomputed_float, **accounting}


def _verify_observed_preflight(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """preflight_results.json is observed evidence only; it must agree with the frozen anchor manifest."""
    path = output_dir / "preflight_results.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: preflight results are missing")
    observed = json.loads(path.read_text())
    anchor = manifest["anchor"]
    if observed.get("anchor_manifest_sha256") != manifest["manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: observed preflight results are bound to another anchor manifest")
    if str(observed["scheduler"]["scheduler_class"]) != anchor["scheduler_class"]:
        raise GlobalStopError("GLOBAL STOP: observed preflight scheduler class differs from the frozen anchor")
    if not math.isclose(float(observed["scheduler"]["timesteps"][anchor["resume_index"]]), float(anchor["resume_timestep"]), rel_tol=0.0, abs_tol=1e-6):
        raise GlobalStopError("GLOBAL STOP: observed preflight timestep differs from the frozen anchor")
    for control in observed["full_direct_controls"]:
        if (
            str(control["scheduler_class"]) != anchor["scheduler_class"]
            or int(control["resume_index"]) != anchor["resume_index"]
            or control["clean_state_identity_sha256_v1"] != anchor["clean_state_identity_sha256_v1"]
            or control["recovered_final_latent_identity_sha256_v1"] != anchor["clean_final_latent_identity_sha256_v1"]
            or control["recovered_video_identity_sha256_v1"] != anchor["clean_video_identity_sha256_v1"]
        ):
            raise GlobalStopError("GLOBAL STOP: observed preflight control disagrees with the frozen anchor")
    return observed


def _row_identity(
    config: dict[str, Any],
    source: base.SourceTrajectory,
    manifest: dict[str, Any],
    record: dict[str, Any],
    replay_id: int,
) -> dict[str, Any]:
    """Every field a persisted row must carry to be accepted as this anchor's row (frozen manifest is authoritative)."""
    anchor = manifest["anchor"]
    return {
        "experiment_version": config["experiment_version"],
        "config_hash": config_hash(config),
        "source_raw_sha256": config["trusted_v3"]["raw_results_sha256"],
        "model": config["model"],
        "scheduler": EXPECTED_SCHEDULER,
        "scheduler_class": anchor["scheduler_class"],
        "scheduler_config_json": canonical_json(anchor["scheduler_config"]).decode(),
        "prompt_id": source.prompt_id,
        "generation_seed": source.seed,
        "checkpoint_step": source.checkpoint_step,
        "resume_index": source.checkpoint_step,
        "resume_timestep": anchor["resume_timestep"],
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "latent_shape_json": json.dumps([int(v) for v in source.clean.shape]),
        "clean_checkpoint_hash": source.clean_hash,
        "clean_state_identity_sha256_v1": anchor["clean_state_identity_sha256_v1"],
        "anchor_manifest_sha256": manifest["manifest_sha256"],
        "condition_id": record["condition_id"],
        "perturbation_family": PERTURBATION_FAMILY,
        "coordinate_flat_index": int(record["coordinate_flat_index"]),
        "direction": record["direction"],
        "requested_direction": record["direction"],
        "replay_id": replay_id,
        "runtime_input_hash": record["runtime_input_hash"],
        "changed_coordinate_count": 1,
    }


EXPECTED_CANDIDATE_FIELDS = (
    "expected_candidate_tensor_identity_sha256_v1",
    "expected_candidate_shape",
    "expected_candidate_dtype",
    "expected_candidate_runtime_dtype",
    "expected_candidate_raw_bf16_bytes_sha256",
    "expected_changed_flat_index",
    "expected_clean_bf16_bits",
    "expected_perturbed_bf16_bits",
)


def _frozen_constructions(output_dir: Path, derived: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-key expected candidates from the immutable CPU manifest; must equal the fresh re-derivation."""
    stored = _load_manifest(output_dir)
    frozen = {row["condition_id"]: row for row in stored["constructions"]}
    fresh = {row["condition_id"]: row for row in derived["constructions"]}
    if set(frozen) != set(fresh):
        raise GlobalStopError("GLOBAL STOP: frozen construction keys differ from re-derived keys")
    for condition, expected in frozen.items():
        for field in EXPECTED_CANDIDATE_FIELDS:
            if field not in expected or expected[field] != fresh[condition][field]:
                raise GlobalStopError(f"GLOBAL STOP: frozen expected candidate {field} differs from re-derivation for {condition}")
    keys_document = json.loads((output_dir / "expected_primary_keys.json").read_text())
    for key in keys_document["keys"]:
        expected = frozen[key["condition_id"]]
        for field in EXPECTED_CANDIDATE_FIELDS:
            if key.get(field) != expected[field]:
                raise GlobalStopError(f"GLOBAL STOP: expected_primary_keys.json disagrees with the anchor manifest for {key['condition_id']}")
    return frozen


def _check_candidate_against_frozen(candidate: np.ndarray, expected: dict[str, Any], where: str) -> None:
    """Shape, dtype and canonical identity of a persisted candidate must equal the frozen construction."""
    if [int(v) for v in candidate.shape] != list(expected["expected_candidate_shape"]):
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate shape differs from the frozen expected candidate: {where}")
    if np.dtype(candidate.dtype).newbyteorder("<").str != expected["expected_candidate_dtype"]:
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate dtype differs from the frozen expected candidate: {where}")
    if not np.array_equal(base.cast_runtime_bf16(candidate).view(np.uint32), candidate.view(np.uint32)):
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate is not bit-exact {EXPECTED_RUNTIME_DTYPE}: {where}")
    if tensor_identity_sha256_v1(candidate) != expected["expected_candidate_tensor_identity_sha256_v1"]:
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate identity differs from the frozen expected candidate: {where}")
    if sha256_bytes(float32_to_bf16_bits(candidate).tobytes()) != expected["expected_candidate_raw_bf16_bytes_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: persisted candidate BF16 bytes differ from the frozen expected candidate: {where}")


def _row_directory(output_dir: Path, record: dict[str, Any], replay_id: int) -> Path:
    return output_dir / "smoke/rows" / record["condition_id"] / f"replay_{replay_id:02d}"


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "anchor_manifest.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: anchor manifest from CPU mode is missing")
    return json.loads(path.read_text())


def _check_manifest_unchanged(derived: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    stored = _load_manifest(output_dir)
    if derived["manifest"] != stored:
        raise GlobalStopError("GLOBAL STOP: anchor manifest changed after CPU mode")
    return stored


def _common_gates(config: dict[str, Any], derived: dict[str, Any], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = derived["manifest"]
    historical = derived["historical"]
    source = derived["source"]
    eligible = derived["eligible"]
    constructions = derived["constructions"]
    audit = decision_source_field_audit()
    return [
        _gate("G1 trusted source hashes exact", True, validate_source_hashes(config), "v3 raw/config/provenance and error-shape config/script hashes match"),
        _gate("G3 anchor is the unique non-exact v3 FP16 row", (derived["meta"]["derived_unique_anomaly"]["prompt_id"], derived["meta"]["derived_unique_anomaly"]["generation_seed"], derived["meta"]["derived_unique_anomaly"]["checkpoint_step"]) == (source.prompt_id, source.seed, source.checkpoint_step) and not derived["meta"]["derived_unique_anomaly"]["final_latent_exact"] and not derived["meta"]["derived_unique_anomaly"]["video_exact"], derived["meta"]["derived_unique_anomaly"], "configured anchor equals the derived unique anomaly and that row is non-exact"),
        _gate("G4 clean anchor state hash and BF16 representability", v3.array_sha256(source.clean) == source.clean_hash and bool(np.array_equal(base.cast_runtime_bf16(source.clean), source.clean)), {"clean_hash": source.clean_hash, "runtime_sha256": historical["clean_runtime_sha256"]}, "v3 manifest hash matches; state is bit-exact BF16"),
        _gate("G5 historical FP16 delta re-derived", historical["changed_coordinate_count"] > 0 and historical["probe_runtime_sha256"] == derived["meta"]["frozen_reconstructed_fp16_runtime_sha256"], {"changed": historical["changed_coordinate_count"], "l2": historical["delta_l2"], "coords": [row["coordinate_flat_index"] for row in historical["changed_coordinates"]]}, "derived from the retained payload, not from any list"),
        _gate("G6 eligible set derived and contains historical support", eligible.size > 0 and manifest["eligibility"]["historical_support_subset_of_eligible"], manifest["eligibility"], "|z|<1e-5 derived at run time; historical support is a subset"),
        _gate("G7 adjacent-BF16 primitive exhaustively verified", derived["adjacent_verification"]["passed"], derived["adjacent_verification"], "0 mismatches against sorted enumeration of all finite BF16 values"),
        _gate("G8 single-coordinate isolation for every construction", all(row["changed_coordinate_count"] == 1 and all(field in row for field in EXPECTED_CANDIDATE_FIELDS) and row["expected_changed_flat_index"] == row["coordinate_flat_index"] for row in constructions) and len(constructions) == len(expected_primary_keys(eligible)), {"constructions": len(constructions), "unique_input_hashes": len({row["runtime_input_hash"] for row in constructions})}, "exactly one BF16 coordinate changes per state; all input hashes distinct"),
        _gate("G9 BF16 state accounting", manifest["anchor"]["runtime_full_bytes"] == manifest["anchor"]["runtime_numel"] * 2 and manifest["anchor"]["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE, manifest["anchor"], "numel x 2 bytes"),
        _gate("G10 frozen primary key set", manifest["primary_row_count"] == len(expected_primary_keys(eligible)) and len({row["runtime_input_hash"] for row in constructions}) == manifest["primary_row_count"], {"primary_row_count": manifest["primary_row_count"], "manifest_sha256": manifest["manifest_sha256"]}, "K x 2 distinct keys"),
        _gate("G11 no hard-coded derived counts", not (FORBIDDEN_COUNT_KEYS & (set(config) | set(config["perturbation"]) | set(config["anchor"]))), sorted(FORBIDDEN_COUNT_KEYS), "K derived from clean state only"),
        _gate("G15 decision inputs restricted to preregistered fields", audit["passed"] and set(config["analysis"]["decision_input_fields"]) == DECISION_INPUT_FIELDS, audit, "temporal/CLIP/MSE/exactness/membership cannot enter the decision"),
        _gate("G19 provenance frozen", provenance["trusted_v3_raw_sha256"] == config["trusted_v3"]["raw_results_sha256"] and provenance["trusted_v3_resolved_model_revision"] == trusted_resolved_model_revision(config), provenance, "code/config/source hashes and pinned model revision recorded"),
        _gate("G20 no automatic expansion", config["allowed_modes"] == list(ALLOWED_MODES) and config["replay"]["non_triggered_rows_run_once"] is True, config["allowed_modes"], "four modes; non-triggered rows run once"),
    ]


def run_cpu_mode(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    derived = derive_all(config)
    derived["adjacent_verification"] = verify_adjacent_bf16_by_enumeration()
    manifest = derived["manifest"]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "preregistered_config.json", config)
    atomic_json(output_dir / "run_provenance.json", provenance)
    atomic_json(output_dir / "environment.json", environment_document(config, provenance))
    atomic_json(output_dir / "anchor_manifest.json", manifest)
    atomic_json(output_dir / "eligible_coordinates.json", manifest["eligibility"])
    atomic_json(output_dir / "historical_fp16_delta.json", derived["historical"])
    atomic_json(
        output_dir / "expected_primary_keys.json",
        {
            "primary_row_count": manifest["primary_row_count"],
            "keys": [
                {
                    "coordinate_flat_index": row["coordinate_flat_index"],
                    "requested_direction": row["direction"],
                    "direction": row["direction"],
                    "condition_id": row["condition_id"],
                    **{field: row[field] for field in EXPECTED_CANDIDATE_FIELDS},
                }
                for row in derived["constructions"]
            ],
            "runtime_input_hashes": {row["condition_id"]: row["runtime_input_hash"] for row in derived["constructions"]},
        },
    )
    atomic_json(output_dir / "adjacent_bf16_verification.json", derived["adjacent_verification"])
    gates = [
        *_common_gates(config, derived, provenance),
        _gate("G2 Euler scheduler only (config)", config["scheduler"]["name"] == EXPECTED_SCHEDULER and config["scheduler"]["sample_solver"] == "euler", config["scheduler"], EXPECTED_SCHEDULER),
    ]
    _write_gates(output_dir / "cpu_gates.json", gates, provenance, manifest["manifest_sha256"])
    return {
        "mode": "cpu",
        "all_passed": True,
        "eligible_count": int(derived["eligible"].size),
        "primary_row_count": manifest["primary_row_count"],
        "historical_changed_count": derived["historical"]["changed_coordinate_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "provenance_hash": provenance["provenance_hash"],
    }


def _full_direct(omni: Any, config: dict[str, Any], source: base.SourceTrajectory, label: str, directory: Path) -> dict[str, Any]:
    result = base.run_resume(
        omni, config, source, source.clean, step_index=source.checkpoint_step, label=label, directory=directory
    )
    if not (result["exact_final_latent"] and result["exact_video"]):
        raise GlobalStopError(f"GLOBAL STOP: FULL-direct control {label} is not bit-exact")
    if int(result["resume_index"]) != source.checkpoint_step or str(result["scheduler_class"]) != EXPECTED_SCHEDULER_CLASS:
        raise GlobalStopError(f"GLOBAL STOP: FULL-direct control {label} resumed with wrong index or scheduler")
    latent_record = _identity_record(result["recovered_final_latent_artifact"])
    video_record = _identity_record(result["recovered_video_artifact"])
    if latent_record["tensor_identity_sha256_v1"] != tensor_identity_sha256_v1(source.final_latent):
        raise GlobalStopError(f"GLOBAL STOP: FULL-direct control {label} latent identity differs from the anchor")
    if video_record["tensor_identity_sha256_v1"] != tensor_identity_sha256_v1(source.video):
        raise GlobalStopError(f"GLOBAL STOP: FULL-direct control {label} video identity differs from the anchor")
    return {
        **result,
        "recovered_final_latent_artifact": latent_record,
        "recovered_video_artifact": video_record,
        "recovered_final_latent_identity_sha256_v1": latent_record["tensor_identity_sha256_v1"],
        "recovered_video_identity_sha256_v1": video_record["tensor_identity_sha256_v1"],
        "prompt_id": source.prompt_id,
        "generation_seed": source.seed,
        "checkpoint_step": source.checkpoint_step,
        "resume_timestep": anchor_resume_timestep(config, source.checkpoint_step),
        "clean_state_identity_sha256_v1": tensor_identity_sha256_v1(source.clean),
    }


def run_preflight(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "cpu_gates.json", provenance, _load_manifest(output_dir)["manifest_sha256"])
    derived = derive_all(config)
    derived["adjacent_verification"] = verify_adjacent_bf16_by_enumeration()
    stored = _check_manifest_unchanged(derived, output_dir)
    source = derived["source"]
    scheduler = v3.scheduler_document(config)
    runtime_timestep = float(scheduler["timesteps"][source.checkpoint_step])
    if not math.isclose(runtime_timestep, float(stored["anchor"]["resume_timestep"]), rel_tol=0.0, abs_tol=1e-6):
        raise GlobalStopError("GLOBAL STOP: runtime resume timestep differs from the frozen anchor timestep")
    if str(scheduler["scheduler_class"]) != stored["anchor"]["scheduler_class"]:
        raise GlobalStopError("GLOBAL STOP: runtime scheduler class differs from the frozen anchor scheduler identity")
    metric_control = base._metric_control_result(source.video)
    repeats = int(config["controls"]["preflight_full_direct_repeats"])
    controls = []
    omni = v3.build_omni(config, args)
    try:
        for index in range(repeats):
            controls.append(
                _full_direct(
                    omni, config, source, f"singleflip_preflight_full_direct_{index}", output_dir / f"preflight/full_direct_{index:02d}"
                )
            )
    finally:
        _shutdown(omni)
    all_exact = all(row["exact_final_latent"] and row["exact_video"] for row in controls)
    retained = all(
        _validate_saved_array(record)
        for row in controls
        for record in (row["recovered_video_artifact"], row["recovered_final_latent_artifact"])
    )
    gates = [
        *_common_gates(config, derived, provenance),
        _gate("G2 Euler scheduler only (runtime)", scheduler["scheduler_class"] == stored["anchor"]["scheduler_class"] and all(row["scheduler_class"] == stored["anchor"]["scheduler_class"] for row in controls), {"runtime": scheduler["scheduler_class"], "frozen": stored["anchor"]["scheduler_class"]}, EXPECTED_SCHEDULER_CLASS),
        _gate("G12 at least three FULL-direct controls bit-exact", all_exact and len(controls) >= 3, [{k: row[k] for k in ("exact_final_latent", "exact_video", "frame_ssim_mean", "recovered_final_latent_identity_sha256_v1", "resume_timestep")} for row in controls], "every control exact"),
        _gate("G13 resume index equals anchor step", all(int(row["resume_index"]) == source.checkpoint_step for row in controls) and math.isclose(runtime_timestep, float(stored["anchor"]["resume_timestep"]), rel_tol=0.0, abs_tol=1e-6), {"resume_indices": [row["resume_index"] for row in controls], "runtime_timestep": runtime_timestep, "frozen_timestep": stored["anchor"]["resume_timestep"]}, f"resume index {source.checkpoint_step} at the frozen timestep"),
        _gate("G14 SSIM negative controls", metric_control["passed"], metric_control, "exact, zero, mild, reversal controls pass"),
        _gate("G17 control artifacts retained and hashed", retained, [row["recovered_final_latent_artifact"] for row in controls], "all retained"),
        _gate("G18 anchor manifest unchanged since CPU mode", derived["manifest"] == stored, stored["manifest_sha256"], "unchanged"),
    ]
    atomic_json(output_dir / "metric_controls.json", metric_control)
    atomic_json(
        output_dir / "preflight_results.json",
        {"full_direct_controls": controls, "scheduler": scheduler, "anchor_manifest_sha256": stored["manifest_sha256"], "role": "observed evidence; the anchor manifest is authoritative"},
    )
    _write_gates(output_dir / "preflight_gates.json", gates, provenance, stored["manifest_sha256"])
    return {"mode": "preflight", "all_passed": True, "full_direct_controls": len(controls)}


def run_smoke(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "preflight_gates.json", provenance, _load_manifest(output_dir)["manifest_sha256"])
    derived = derive_all(config)
    derived["adjacent_verification"] = verify_adjacent_bf16_by_enumeration()
    stored = _check_manifest_unchanged(derived, output_dir)
    source = derived["source"]
    frozen_hashes = json.loads((output_dir / "expected_primary_keys.json").read_text())["runtime_input_hashes"]
    trigger = float(config["replay"]["trigger_frame_ssim_below"])
    total_runs = int(config["replay"]["total_runs_per_triggered_row"])
    by_condition = {row["condition_id"]: row for row in derived["constructions"]}
    frozen_constructions = _frozen_constructions(output_dir, derived)
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    omni = v3.build_omni(config, args)
    try:
        for index in range(int(config["controls"]["smoke_full_direct_repeats"])):
            controls.append(
                _full_direct(omni, config, source, f"singleflip_smoke_full_direct_{index}", output_dir / f"smoke/full_direct_{index:02d}")
            )
        for flat, direction in expected_primary_keys(derived["eligible"]):
            state, record = build_single_flip_state(source.clean, flat, direction)
            record["historical_fp16_support_member"] = by_condition[record["condition_id"]]["historical_fp16_support_member"]
            if record["runtime_input_hash"] != frozen_hashes[record["condition_id"]]:
                raise GlobalStopError(f"GLOBAL STOP: runtime input hash drifted for {record['condition_id']}")
            group: list[dict[str, Any]] = []
            for replay_id in range(total_runs):
                directory = _row_directory(output_dir, record, replay_id)
                result_path = directory / "result.json"
                identity = _row_identity(config, source, stored, record, replay_id)
                row = _result_valid(result_path, provenance, identity, frozen_constructions[record["condition_id"]])
                if row is None:
                    candidate_record = _save_array(directory / "runtime_candidate.npy", state)
                    result = base.run_resume(
                        omni,
                        config,
                        source,
                        state,
                        step_index=source.checkpoint_step,
                        label=f"singleflip_{record['condition_id']}_r{replay_id}",
                        directory=directory,
                    )
                    row = _scientific_row(config, provenance, source, stored, record, replay_id, candidate_record, result, result_path)
                    if any(row.get(key) != value for key, value in identity.items()):
                        raise GlobalStopError(f"GLOBAL STOP: freshly generated row violates the frozen identity: {result_path}")
                    atomic_json(result_path, row)
                group.append(row)
                # Replay semantics: only a primary run below the trigger earns two more runs.
                if float(group[0]["frame_ssim_mean"]) >= trigger:
                    break
            rows.extend(group)
    finally:
        _shutdown(omni)
    write_csv(output_dir / "smoke_raw_results.csv", rows, RAW_FIELDS)
    executed_keys = {(int(row["coordinate_flat_index"]), row["direction"]) for row in rows}
    expected = set(expected_primary_keys(derived["eligible"]))
    replay_ok = True
    for key in expected:
        group = [row for row in rows if (int(row["coordinate_flat_index"]), row["direction"]) == key]
        primary = next(row for row in group if int(row["replay_id"]) == 0)
        needed = total_runs if float(primary["frame_ssim_mean"]) < trigger else 1
        replay_ok &= len(group) == needed and sorted(int(row["replay_id"]) for row in group) == list(range(needed))
    retained = all(
        _validate_saved_array(row[field])
        for row in rows
        for field in ("runtime_candidate_artifact", "recovered_final_latent_artifact", "recovered_video_artifact")
    )
    gates = [
        *_common_gates(config, derived, provenance),
        _gate("G2 Euler scheduler only (runtime)", all(str(row["scheduler_class"]) == stored["anchor"]["scheduler_class"] for row in rows + controls), sorted({str(row["scheduler_class"]) for row in rows + controls}), EXPECTED_SCHEDULER_CLASS),
        _gate("G12 smoke FULL-direct control bit-exact", all(row["exact_final_latent"] and row["exact_video"] for row in controls) and len(controls) >= 1, [row["recovered_final_latent_identity_sha256_v1"] for row in controls], "exact"),
        _gate("G13 resume index equals anchor step", all(int(row["resume_index"]) == source.checkpoint_step for row in rows + controls), sorted({int(row["resume_index"]) for row in rows}), f"resume index {source.checkpoint_step}"),
        _gate("G16 replay semantics honoured", bool(replay_ok) and executed_keys == expected, {"executed_keys": len(executed_keys), "rows": len(rows)}, "triggered rows have exactly 3 runs; others exactly 1"),
        _gate("G17 artifacts retained and hashed", retained, len(rows), "all retained"),
        _gate("G18 executed runtime input hashes equal frozen hashes", all(row["runtime_input_hash"] == frozen_hashes[row["condition_id"]] for row in rows), len(frozen_hashes), "no drift"),
        _gate("G21 finite frame SSIM for all rows", all(math.isfinite(float(row["frame_ssim_mean"])) for row in rows), len(rows), "finite"),
    ]
    atomic_json(output_dir / "smoke_controls.json", controls)
    _write_gates(output_dir / "smoke_gates.json", gates, provenance, stored["manifest_sha256"])
    return {"mode": "smoke", "all_passed": True, "rows": len(rows), "keys": len(executed_keys)}


def _load_smoke_results(config: dict[str, Any], provenance: dict[str, Any], output_dir: Path, derived: dict[str, Any]) -> list[dict[str, Any]]:
    trigger = float(config["replay"]["trigger_frame_ssim_below"])
    total_runs = int(config["replay"]["total_runs_per_triggered_row"])
    frozen_hashes = json.loads((output_dir / "expected_primary_keys.json").read_text())["runtime_input_hashes"]
    stored = _load_manifest(output_dir)
    source = derived["source"]
    frozen_constructions = _frozen_constructions(output_dir, derived)
    rows = []
    for record in derived["constructions"]:
        if frozen_hashes[record["condition_id"]] != record["runtime_input_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: frozen runtime input hash drifted for {record['condition_id']}")
        expected = frozen_constructions[record["condition_id"]]
        group: list[dict[str, Any]] = []
        for replay_id in range(total_runs):
            path = _row_directory(output_dir, record, replay_id) / "result.json"
            row = _result_valid(path, provenance, _row_identity(config, source, stored, record, replay_id), expected)
            if row is None:
                raise GlobalStopError(f"GLOBAL STOP: missing completed smoke result: {path}")
            verify_row_against_source(row, source, expected)
            group.append(row)
            if float(group[0]["frame_ssim_mean"]) >= trigger:
                break
        extra = _row_directory(output_dir, record, len(group)) / "result.json"
        if extra.exists():
            raise GlobalStopError(f"GLOBAL STOP: unexpected extra replay exists: {extra}")
        rows.extend(group)
    return rows


def run_analyze_smoke(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "smoke_gates.json", provenance, _load_manifest(output_dir)["manifest_sha256"])
    derived = derive_all(config)
    stored = _check_manifest_unchanged(derived, output_dir)
    _verify_observed_preflight(output_dir, stored)
    rows = _load_smoke_results(config, provenance, output_dir, derived)
    smoke_gates = json.loads((output_dir / "smoke_gates.json").read_text())
    preflight_gates = json.loads((output_dir / "preflight_gates.json").read_text())
    controls_passed = bool(smoke_gates["all_passed"]) and bool(preflight_gates["all_passed"])
    result = analyze_rows(rows, config, expected_primary_keys(derived["eligible"]), controls_passed=controls_passed)
    historical = {row["coordinate_flat_index"] for row in derived["historical"]["changed_coordinates"]}
    for summary in result["row_summaries"]:
        summary["historical_fp16_support_member_descriptive"] = summary["coordinate_flat_index"] in historical
    write_csv(output_dir / "row_class_summary.csv", result["row_summaries"])
    atomic_json(output_dir / "single_flip_summary.json", result)
    report = [
        "# Exp 0: Single-Coordinate Adjacent-BF16 Perturbation Map",
        "",
        f"Decision: **{result['decision']}**",
        f"Reason: {result['decision_reason']}",
        "",
        f"Anchor: {derived['source'].prompt_id} / seed {derived['source'].seed} / step {derived['source'].checkpoint_step}",
        f"Eligible coordinates (|z| < {config['perturbation']['eligibility_abs_threshold']}): {int(derived['eligible'].size)}",
        f"Primary keys: {result['expected_key_count']}; executed rows (with replays): {result['executed_row_count']}",
        f"Row classes: {json.dumps(result['row_class_counts'], sort_keys=True)}",
        f"Minimum primary frame SSIM: {result['min_primary_frame_ssim_mean']:.6f}",
        "",
        "The decision reads only coordinate identity, direction, replay id, frame SSIM, and recovered hashes.",
        "Temporal metrics, CLIP, latent/video MSE, exactness flags, and historical membership are descriptive only.",
        "A GO here authorizes a local branch map only; it establishes no mechanism.",
    ]
    (output_dir / "video_bf16_single_flip_killtest.md").write_text("\n".join(report) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=ALLOWED_MODES, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/video_bf16_single_flip_killtest_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/video_bf16_single_flip_killtest")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    if args.mode == "cpu":
        result = run_cpu_mode(config, config_path, output_dir)
    elif args.mode == "preflight":
        result = run_preflight(config, config_path, output_dir, args)
    elif args.mode == "smoke":
        result = run_smoke(config, config_path, output_dir, args)
    else:
        result = run_analyze_smoke(config, config_path, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
