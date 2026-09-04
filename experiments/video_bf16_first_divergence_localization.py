#!/usr/bin/env python3
"""Portable first-divergence trace localization for the audited Wan BF16 anchor.

The experiment records raw pairwise evolution only.  It deliberately has no
mechanism, threshold search, or automatic phase-2/phase-3 expansion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_bf16_single_flip_killtest as single_flip
from experiments import video_runtime_state_discovery as v3

GlobalStopError = single_flip.GlobalStopError
EXPERIMENT_VERSION = "video-bf16-first-divergence-localization-v2"
EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_RUNTIME_DTYPE = "torch.bfloat16"
EXPECTED_INPUT_STORAGE_DTYPE = "<f4"
EXPECTED_INPUT_RUNTIME_DTYPE = "torch.float32"
# The runtime computes the shifted schedule in float32 on the accelerator; individual
# timesteps may differ from the CPU derivation in the last float32 bit (~6e-5 near 1000).
# Consecutive timesteps are > 2 apart (smallest gap at the end of the schedule), so mapping is unambiguous under this tolerance
# combined with a nearest-frozen-timestep uniqueness requirement.
TIMESTEP_MATCH_ABS_TOL = 1e-3
TRAJECTORIES = ("CLEAN", "PLUS1", "HISTORICAL_PLUS14")
MODES = ("cpu", "preflight", "phase1", "analyze-phase1", "phase2", "analyze-phase2", "phase3", "analyze-phase3")
IDENTITY_FORMAT = single_flip.TENSOR_IDENTITY_FORMAT
CPU_REQUIRED_GATES = (
    "G1 trusted source hashes unchanged",
    "G2 exact frozen anchor identity",
    "G3 exact CLEAN tensor identity",
    "G4 exact PLUS1 expected tensor identity",
    "G5 exact HISTORICAL_PLUS14 tensor identity",
    "G6 PLUS1 exactly one changed coordinate",
    "G7 historical support exactly six coordinates",
    "G8 coordinate 516515 historical distance exactly +14 adjacent BF16 steps",
    "G9 Euler scheduler/resume semantics exact",
    "G10 model/scheduler/timestep provenance frozen",
    "G23 phase-2 step explicitly frozen before execution",
    "G24 no automatic phase-3 expansion",
    "G25 no FP32 threshold search",
    "G26 no trusted namespace mutation",
    "G27 provenance/config/manifest hash-bound",
)
PREFLIGHT_REQUIRED_GATES = (
    "G11 CLEAN repeated final determinism",
    "G12 PLUS1 repeated final determinism",
    "G13 historical repeated final determinism",
    "G14 PLUS1 final == historical final",
    "G15 CLEAN final != PLUS1 final",
    "G15a CLEAN final equals trusted clean",
    "G15b PLUS1 final equals trusted PLUS1",
    "G15c historical final equals trusted historical",
)
PHASE2_AVAILABLE_BOUNDARIES = (
    "latent_entering_step",
    "transformer_input",
    "guidance_combined_output",
    "scheduler_input",
    "scheduler_output",
)
PHASE2_UNAVAILABLE_BOUNDARIES = ("transformer_raw_output",)
PHASE2_REQUIRED_GATES = tuple(f"P2-G{number}" for number in range(1, 26))
PHASE3_REQUIRED_GATES = tuple(f"P3-G{number}" for number in range(1, 32))
PHASE3_BRANCHES = ("positive", "negative")
PHASE3_BLOCK_COUNT = 40
# Wan2.2-T2V-A14B transformer: 40 heads x 128 = inner dim 5120; verified at runtime against the
# loaded model config and recorded in the trace, so a different checkpoint fails closed.
PHASE3_NUM_HEADS = 40
PHASE3_HEAD_DIM = 128
PHASE3_PHASE1_COMMIT = "0742c718ed942a752be5e03ab24cd578395e9e89"
PHASE3_PHASE2_COMMIT = "10fe61f3e986787ed0bb9cd8ecddb2cbf97043a8"
PHASE3_FIXED_BOUNDARIES = (
    "transformer_entry",
    "pre_block_hidden_state",
    *(f"after_block_{index:03d}" for index in range(PHASE3_BLOCK_COUNT)),
    "transformer_pre_output",
    "raw_transformer_output",
)
PHASE2_BOUNDARY_SEMANTICS = {
    "latent_entering_step": {
        "tensor": "latent state entering the selected resumed denoising update",
        "guidance_position": "before guidance",
        "consumed_by_next_operation": "used to construct transformer_input and retained as scheduler_input",
    },
    "transformer_input": {
        "tensor": "latent_model_input after conversion to the selected Wan transformer dtype",
        "guidance_position": "before guidance",
        "consumed_by_next_operation": "consumed by the positive and, when enabled, negative transformer calls",
    },
    "guidance_combined_output": {
        "tensor": "noise prediction returned after classifier-free guidance combination",
        "guidance_position": "after guidance",
        "consumed_by_next_operation": "consumed by WanEulerScheduler as model_output",
    },
    "scheduler_input": {
        "tensor": "pre-update latent state passed to WanEulerScheduler as sample",
        "guidance_position": "after guidance and before scheduler update",
        "consumed_by_next_operation": "consumed by WanEulerScheduler as sample",
    },
    "scheduler_output": {
        "tensor": "updated latent state returned by WanEulerScheduler",
        "guidance_position": "after guidance and scheduler update",
        "consumed_by_next_operation": "consumed by the next denoising update or final VAE decode",
    },
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


SOURCE_SCOPE_PREFIXES = ("experiments/", "tests/", "vllm_omni/")


def git_state() -> dict[str, Any]:
    """Source-relevant git state only.

    Every modification of a tracked file counts. Untracked files count only
    under the source prefixes that can affect execution or validation, so a
    local virtual environment or scratch file cannot change provenance or
    block GPU work, while a stray experiment/test/runtime file still does.
    """
    status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()

    def relevant(row: str) -> bool:
        path = row[3:].replace("\\", "/")
        if row.startswith("??"):
            return path.startswith(SOURCE_SCOPE_PREFIXES)
        return True

    source_dirty = sorted(row for row in status if relevant(row))
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(source_dirty),
        "source_dirty_entries": source_dirty,
        "source_scope_prefixes": list(SOURCE_SCOPE_PREFIXES),
        "dirty_policy": "tracked modifications anywhere; untracked files only under source scope prefixes",
    }


def identity(array: np.ndarray) -> str:
    return single_flip.tensor_identity_sha256_v1(np.asarray(array))


def relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical_json(value))
    temp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["experiment_version"] != EXPERIMENT_VERSION or tuple(config["allowed_modes"]) != MODES:
        raise GlobalStopError("GLOBAL STOP: experiment version or modes changed")
    if config["model"] != EXPECTED_MODEL:
        raise GlobalStopError("GLOBAL STOP: model changed")
    if config["scheduler"]["name"] != EXPECTED_SCHEDULER or config["scheduler"]["sample_solver"] != "euler":
        raise GlobalStopError("GLOBAL STOP: Euler scheduler is required")
    if config["generation"] != {"height": 480, "width": 832, "num_frames": 33, "num_inference_steps": 40, "guidance_scale": 4.0, "fps": 16.0, "boundary_ratio": 0.875}:
        raise GlobalStopError("GLOBAL STOP: trusted generation configuration changed")
    anchor = config["anchor"]
    if (anchor["prompt_id"], int(anchor["generation_seed"]), int(anchor["checkpoint_step"]), int(anchor["critical_flat_index"])) != ("recovery_008", 9234, 10, 516515):
        raise GlobalStopError("GLOBAL STOP: frozen anchor changed")
    if int(config["controls"]["repeats_per_trajectory"]) < 3:
        raise GlobalStopError("GLOBAL STOP: at least three deterministic control repeats are required")
    validate_phase3_config(config)
    validate_phase2_config(config)
    return config


def validate_phase2_config(config: dict[str, Any]) -> None:
    phase2 = config.get("phase2")
    if not isinstance(phase2, dict):
        raise GlobalStopError("GLOBAL STOP: phase 2 configuration is absent")
    expected = {
        "selected_step": 10,
        "entry_phase1_boundary": "input",
        "exit_phase1_boundary": "after_step_001",
        "allowed_boundaries": [
            "latent_entering_step",
            "transformer_input",
            "transformer_raw_output",
            "guidance_combined_output",
            "scheduler_input",
            "scheduler_output",
        ],
        "available_boundaries": list(PHASE2_AVAILABLE_BOUNDARIES),
        "unavailable_boundaries": list(PHASE2_UNAVAILABLE_BOUNDARIES),
    }
    if phase2 != expected:
        raise GlobalStopError("GLOBAL STOP: Phase-2 selected step or operation-boundary freeze changed")
    trusted_root = config.get("trusted_phase1_root")
    if trusted_root != "results/video_bf16_first_divergence_localization":
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 root changed")


def validate_phase3_config(config: dict[str, Any]) -> None:
    expected = {
        "enabled": True,
        "selected_step": 10,
        "phase2_entry_boundary": "transformer_input",
        "phase2_exit_boundary": "guidance_combined_output",
        "transformer_role": "high_noise_transformer",
        "cfg_execution": "sequential_positive_then_negative",
        "branches": list(PHASE3_BRANCHES),
        "expected_global_block_count": PHASE3_BLOCK_COUNT,
        "expected_start_layer": 0,
        "expected_end_layer": PHASE3_BLOCK_COUNT,
        "phase4_enabled": False,
        "auto_expand": False,
    }
    if config.get("phase3") != expected:
        raise GlobalStopError("GLOBAL STOP: frozen Phase-3 target or execution structure changed")
    if config.get("trusted_phase1_commit") != PHASE3_PHASE1_COMMIT:
        raise GlobalStopError("GLOBAL STOP: Phase-1-producing commit changed")
    if config.get("trusted_phase2_commit") != PHASE3_PHASE2_COMMIT:
        raise GlobalStopError("GLOBAL STOP: Phase-2-producing commit changed")


def validate_output_namespace(config: dict[str, Any], output_dir: Path) -> None:
    protected = ("video_runtime_state_discovery", "video_bf16_single_flip", "video_runtime_error_shape", "video_checkpoint_stability")
    if output_dir.name != Path(config["output_root"]).name or any(item in str(output_dir) for item in protected):
        raise GlobalStopError("GLOBAL STOP: output namespace is not the isolated localization namespace")


def provenance(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    files = [
        "experiments/video_bf16_first_divergence_localization.py",
        "experiments/video_bf16_first_divergence_localization_config.yaml",
        str(config["trusted_single_flip_config"]),
        "experiments/video_bf16_single_flip_killtest.py",
        "experiments/video_runtime_state_discovery.py",
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
        "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
        "experiments/run_video_bf16_first_divergence_localization_gpu0.sh",
        "tests/diffusion/test_video_bf16_first_divergence_localization.py",
    ]
    hashes = {item: sha256_file(REPO_ROOT / item) for item in files}
    record = {
        "config_sha256": sha256_file(config_path),
        "experiment_script_sha256": hashes["experiments/video_bf16_first_divergence_localization.py"],
        "pipeline_sha256": hashes["vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"],
        "runner_sha256": hashes["experiments/run_video_bf16_first_divergence_localization_gpu0.sh"],
        "test_sha256": hashes["tests/diffusion/test_video_bf16_first_divergence_localization.py"],
        "files": hashes,
        "identity_format": IDENTITY_FORMAT,
        **git_state(),
    }
    return {**record, "provenance_hash": sha256_bytes(canonical_json(record))}


def require_committed_source(prov: dict[str, Any]) -> None:
    if prov.get("source_dirty_entries"):
        raise GlobalStopError(
            "GLOBAL STOP: GPU work requires a committed source revision; dirty source entries: "
            + repr(prov["source_dirty_entries"])
        )


def save_tensor(root: Path, relative: str, array: np.ndarray, *, runtime_semantics: str = EXPECTED_RUNTIME_DTYPE) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    value = np.load(path, allow_pickle=False)
    return {
        "relative_path": relative_path(root, path), "file_sha256": sha256_file(path),
        "canonical_identity": identity(value), "storage_dtype": np.dtype(value.dtype).newbyteorder("<").str,
        "shape": [int(x) for x in value.shape], "nbytes": int(value.nbytes),
        "runtime_dtype_semantics": runtime_semantics,
    }


def load_tensor(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise GlobalStopError("GLOBAL STOP: artifact path is not a safe relative path")
    path = root / relative
    if not path.exists() or sha256_file(path) != record["file_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: artifact file validation failed: {relative}")
    value = np.load(path, allow_pickle=False)
    if identity(value) != record["canonical_identity"] or [int(x) for x in value.shape] != record["shape"] or np.dtype(value.dtype).newbyteorder("<").str != record["storage_dtype"]:
        raise GlobalStopError(f"GLOBAL STOP: artifact canonical identity validation failed: {relative}")
    if not np.isfinite(value).all():
        raise GlobalStopError(f"GLOBAL STOP: non-finite tensor artifact: {relative}")
    return value


CFG_RECONSTRUCTION_RULE = (
    "vllm_omni.diffusion.distributed.cfg_parallel.CFGParallelMixin.combine_cfg_noise with cfg_normalize=False: "
    "comb = n + true_cfg_scale * (p - n), evaluated eagerly on bfloat16 tensors as three separate ops, "
    "each computed in float32 op-math and rounded to bfloat16 (round-to-nearest-even): "
    "d = bf16(p - n); m = bf16(float32(scale) * d); comb = bf16(n + m)"
)


def reconstruct_cfg_combined_bits(positive_bits: np.ndarray, negative_bits: np.ndarray, guidance_scale: float) -> np.ndarray:
    """Canonical CPU emulation of the production CFG combination on BF16 tensors.

    Derived from ``CFGParallelMixin.combine_cfg_noise`` (sequential path used by
    Wan2.2 T2V with ``cfg_normalize=False``)::

        comb = n + true_cfg_scale * (p - n)

    PyTorch evaluates this eagerly as three elementwise kernels on bfloat16
    tensors; every kernel upcasts its bfloat16 inputs to float32 op-math and
    rounds its result to bfloat16 (RNE). A Python float scale does not promote
    the dtype. The emulation below therefore rounds after each of the three
    operations, in the same order, with float32 intermediate arithmetic. The
    result is returned as BF16 bit patterns for bit-exact comparison.
    """
    if positive_bits.shape != negative_bits.shape:
        raise GlobalStopError("GLOBAL STOP: CFG operand shapes differ")
    p = single_flip.bf16_bits_to_float32(np.ascontiguousarray(positive_bits, dtype=np.uint16)).reshape(positive_bits.shape)
    n = single_flip.bf16_bits_to_float32(np.ascontiguousarray(negative_bits, dtype=np.uint16)).reshape(negative_bits.shape)
    cast = single_flip.base.cast_runtime_bf16
    difference = cast((p - n).astype(np.float32))
    scaled = cast((np.float32(guidance_scale) * difference).astype(np.float32))
    combined = cast((n + scaled).astype(np.float32))
    return single_flip.float32_to_bf16_bits(combined).reshape(positive_bits.shape)


def phase3_cfg_guidance_scale(config: dict[str, Any]) -> float:
    """The guidance scale bound to the traced invocation: the high-noise (transformer) stage uses guidance_low,
    which for a scalar request guidance equals config generation.guidance_scale."""
    scale = config["generation"]["guidance_scale"]
    if isinstance(scale, (list, tuple)):
        raise GlobalStopError("GLOBAL STOP: Phase-3 is frozen for a scalar guidance scale")
    return float(scale)


def phase3_runtime_identity(bits: np.ndarray, shape: list[int]) -> str:
    bits = np.ascontiguousarray(bits, dtype=np.uint16)
    header = canonical_json(
        {
            "version": "phase3-runtime-tensor-identity-v1",
            "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "shape": [int(item) for item in shape],
        }
    )
    return sha256_bytes(header + b"\0" + bits.tobytes(order="C"))


def phase3_artifact_record(root: Path, path: Path, *, shape: list[int]) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise GlobalStopError("GLOBAL STOP: Phase-3 artifact is outside the result root")
    bits = np.load(path, allow_pickle=False)
    if bits.dtype != np.uint16 or bits.size != math.prod(shape):
        raise GlobalStopError("GLOBAL STOP: Phase-3 BF16 artifact encoding/shape is invalid")
    bits = np.ascontiguousarray(bits.reshape(shape))
    widened = single_flip.bf16_bits_to_float32(bits).reshape(shape)
    return {
        "relative_path": relative_path(root, path),
        "file_sha256": sha256_file(path),
        "artifact_encoding": "bf16_bits_v1",
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "shape": [int(item) for item in shape],
        "runtime_canonical_identity": phase3_runtime_identity(bits, shape),
        "comparison_canonical_identity": identity(widened),
        "nbytes": int(bits.nbytes),
    }


def load_phase3_artifact(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise GlobalStopError("GLOBAL STOP: Phase-3 artifact path is not a safe relative path")
    path = root / relative
    if not path.exists() or sha256_file(path) != record.get("file_sha256"):
        raise GlobalStopError("GLOBAL STOP: Phase-3 artifact file validation failed")
    bits = np.load(path, allow_pickle=False)
    shape = [int(item) for item in record.get("shape", [])]
    if (
        bits.dtype != np.uint16
        or bits.size != math.prod(shape)
        or record.get("artifact_encoding") != "bf16_bits_v1"
        or record.get("runtime_dtype") != EXPECTED_RUNTIME_DTYPE
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-3 artifact dtype/shape semantics changed")
    bits = np.ascontiguousarray(bits.reshape(shape))
    widened = single_flip.bf16_bits_to_float32(bits).reshape(shape)
    if (
        phase3_runtime_identity(bits, shape) != record.get("runtime_canonical_identity")
        or identity(widened) != record.get("comparison_canonical_identity")
        or int(bits.nbytes) != int(record.get("nbytes", -1))
        or not np.isfinite(widened).all()
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-3 artifact canonical identity validation failed")
    return widened


def boundary_keys(remaining: int) -> list[str]:
    return ["input", *[f"after_step_{step:03d}" for step in range(1, remaining + 1)]]


def timestep_match_policy() -> dict[str, Any]:
    return {
        "abs_tol": TIMESTEP_MATCH_ABS_TOL,
        "rule": "observed runtime timestep must lie within abs_tol of the frozen CPU-derived timestep AND the frozen timestep nearest to the observed value must be the expected one",
    }


def timestep_matches(observed: Any, expected: float, frozen_schedule: list[float]) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value) or abs(value - expected) > TIMESTEP_MATCH_ABS_TOL:
        return False
    nearest = min(frozen_schedule, key=lambda item: abs(item - value))
    return nearest == expected


def boundary_specifications(config: dict[str, Any], checkpoint_step: int) -> list[dict[str, Any]]:
    timesteps = single_flip.scheduler_timesteps_numpy(config)
    remaining = int(config["generation"]["num_inference_steps"]) - checkpoint_step
    specs = [{
        "boundary": "input",
        "resumed_update_index": None,
        "absolute_diffusion_step_index": checkpoint_step,
        "scheduler_timestep": timesteps[checkpoint_step],
        "expected_storage_dtype": EXPECTED_INPUT_STORAGE_DTYPE,
        "expected_runtime_dtype": EXPECTED_INPUT_RUNTIME_DTYPE,
        "runtime_dtype_semantics": "float32 resume input containing BF16-exact values",
    }]
    for completed_updates in range(1, remaining + 1):
        specs.append({
            "boundary": f"after_step_{completed_updates:03d}",
            "resumed_update_index": completed_updates - 1,
            "absolute_diffusion_step_index": checkpoint_step + completed_updates,
            "scheduler_timestep": timesteps[checkpoint_step + completed_updates - 1],
            "expected_storage_dtype": "<f4",
            "expected_runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "runtime_dtype_semantics": "BF16 scheduler output widened losslessly to float32 storage",
        })
    return specs


def early_late_cutoff(boundary_count: int) -> dict[str, Any]:
    remaining_updates = boundary_count - 1
    first_late_index = math.ceil(remaining_updates * 0.75)
    return {
        "rule": "persistent merge is EARLY iff its boundary index is before ceil(0.75 * remaining_updates)",
        "remaining_updates": remaining_updates,
        "first_late_boundary_index": first_late_index,
        "last_early_boundary_index": first_late_index - 1,
        "first_late_boundary": f"after_step_{first_late_index:03d}",
    }


def phase2_selection_mapping(
    config: dict[str, Any], manifest: dict[str, Any], selected_step: int
) -> dict[str, Any]:
    validate_phase2_config(config)
    if selected_step != 10:
        raise GlobalStopError("GLOBAL STOP: Phase-2 selected_step must remain frozen at 10")
    checkpoint_step = int(manifest["anchor"]["checkpoint_step"])
    final_step = int(config["generation"]["num_inference_steps"])
    if not checkpoint_step <= selected_step < final_step:
        raise GlobalStopError("GLOBAL STOP: phase-2 selected_step is outside the resumed denoising range")
    resumed_update_index = selected_step - checkpoint_step
    entry_boundary = "input" if resumed_update_index == 0 else f"after_step_{resumed_update_index:03d}"
    specs = {row["boundary"]: row for row in manifest["boundary_specifications"]}
    entry_spec = specs.get(entry_boundary)
    if entry_spec is None or int(entry_spec["absolute_diffusion_step_index"]) != selected_step:
        raise GlobalStopError("GLOBAL STOP: phase-2 selected_step does not map to one frozen Phase-1 boundary")
    exit_boundary = f"after_step_{resumed_update_index + 1:03d}"
    exit_spec = specs.get(exit_boundary)
    if (
        entry_boundary != config["phase2"]["entry_phase1_boundary"]
        or exit_boundary != config["phase2"]["exit_phase1_boundary"]
        or exit_spec is None
        or int(exit_spec["absolute_diffusion_step_index"]) != selected_step + 1
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-2 entry/exit mapping differs from the trusted Phase-1 plan")
    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {
        "selected_absolute_diffusion_step_index": selected_step,
        "selected_resumed_update_index": resumed_update_index,
        "phase1_entry_boundary": entry_boundary,
        "phase1_entry_boundary_specification": entry_spec,
        "phase1_exit_boundary": exit_boundary,
        "phase1_exit_boundary_specification": exit_spec,
        "selected_scheduler_timestep": timesteps[selected_step],
    }


def phase2_freeze(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_step": 10,
        "entry_phase1_boundary": "input",
        "exit_phase1_boundary": "after_step_001",
        "selected_scheduler_timestep": single_flip.scheduler_timesteps_numpy(config)[10],
        "available_boundaries": list(PHASE2_AVAILABLE_BOUNDARIES),
        "unavailable_boundaries": list(PHASE2_UNAVAILABLE_BOUNDARIES),
        "boundary_semantics": PHASE2_BOUNDARY_SEMANTICS,
    }


def resolve_preserved_path(recorded: str | Path) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    text = str(path).replace("\\", "/")
    marker = "/results/"
    if marker in text:
        candidate = REPO_ROOT / ("results/" + text.split(marker, 1)[1])
        if candidate.exists():
            return candidate.resolve()
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise GlobalStopError(f"GLOBAL STOP: preserved trusted artifact is unavailable: {recorded}")


def trusted_array_record(path: Path, *, source_path: Path) -> dict[str, Any]:
    if not source_path.is_absolute():
        source_path = (REPO_ROOT / source_path).resolve()
    value = np.load(path, allow_pickle=False)
    return {
        "source_relative_path": str(source_path.relative_to(REPO_ROOT)),
        "source_file_sha256": sha256_file(source_path),
        "artifact_relative_path": str(path.relative_to(REPO_ROOT)),
        "artifact_file_sha256": sha256_file(path),
        "canonical_identity": identity(value),
        "storage_dtype": np.dtype(value.dtype).newbyteorder("<").str,
        "shape": [int(item) for item in value.shape],
    }


def exact_historical_invariants(historical: dict[str, Any]) -> dict[str, Any]:
    """Cross-host-stable identity of the historical FP16 delta: bits, coordinates, hashes; no float reductions."""
    return {
        "changed_coordinate_count": historical.get("changed_coordinate_count"),
        "coordinates": [
            {
                "coordinate_flat_index": row.get("coordinate_flat_index"),
                "clean_bf16_bits_hex": row.get("clean_bf16_bits_hex"),
                "perturbed_bf16_bits_hex": row.get("perturbed_bf16_bits_hex"),
                "adjacent_steps": row.get("adjacent_steps"),
                "direction": row.get("direction"),
            }
            for row in historical.get("changed_coordinates", [])
        ],
        "clean_runtime_sha256": historical.get("clean_runtime_sha256"),
        "probe_runtime_sha256": historical.get("probe_runtime_sha256"),
    }


def exact_construction_invariants(construction: dict[str, Any]) -> dict[str, Any]:
    """Cross-host-stable identity of the PLUS1 construction; float accounting values stay descriptive."""
    keys = (
        "condition_id", "perturbation_family", "coordinate_flat_index", "direction", "clean_bf16_bits_hex",
        "perturbed_bf16_bits_hex", "changed_coordinate_count", "realized_nonzero_elements", "total_elements",
        "runtime_input_hash", "state_fp32_sha256", "expected_candidate_tensor_identity_sha256_v1",
        "expected_candidate_shape", "expected_candidate_dtype", "expected_candidate_raw_bf16_bytes_sha256",
        "expected_changed_flat_index", "expected_clean_bf16_bits", "expected_perturbed_bf16_bits",
    )
    return {key: construction.get(key) for key in keys}


def exact_pair_invariants(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    """Cross-host-stable pair binding; intentionally contains no reductions."""
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise GlobalStopError("GLOBAL STOP: exact pair tensors differ in shape or dtype")
    lhs = np.ascontiguousarray(lhs)
    rhs = np.ascontiguousarray(rhs)
    differing = int(np.count_nonzero(lhs.view(np.uint32) != rhs.view(np.uint32)))
    return {
        "lhs_canonical_identity": identity(lhs),
        "rhs_canonical_identity": identity(rhs),
        "shape": [int(item) for item in lhs.shape],
        "storage_dtype": np.dtype(lhs.dtype).newbyteorder("<").str,
        "nbytes": int(lhs.nbytes),
        "differing_element_count": differing,
        "bit_exact": differing == 0,
    }


def derive_trusted_phase1(config: dict[str, Any]) -> dict[str, Any]:
    root = (REPO_ROOT / config["trusted_phase1_root"]).resolve()
    trace_path = root / "phase1/trace_manifest.json"
    analysis_path = root / "phase1/phase1_analysis.json"
    if not trace_path.exists() or not analysis_path.exists():
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 trace/analysis is unavailable")
    trace = json.loads(trace_path.read_text())
    recorded_analysis = json.loads(analysis_path.read_text())
    expected_boundaries = boundary_keys(
        int(config["generation"]["num_inference_steps"]) - int(config["anchor"]["checkpoint_step"])
    )
    if trace.get("expected_boundaries") != expected_boundaries:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 boundary plan changed")
    recomputed = analyze_trace(root, trace)
    if (
        recomputed["outcome"] != "EARLY_EXACT_MERGE"
        or recomputed["plus1_historical_event"]
        != {"classification": "PERSISTENT_EXACT_MERGE", "first_boundary": "after_step_001"}
        or recorded_analysis.get("outcome") != recomputed["outcome"]
        or recorded_analysis.get("plus1_historical_event") != recomputed["plus1_historical_event"]
    ):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 scientific result changed")
    frozen: dict[str, Any] = {}
    for name in TRAJECTORIES:
        rows = {row["boundary"]: row for row in trace["traces"][name]}
        if set(rows) != set(expected_boundaries):
            raise GlobalStopError("GLOBAL STOP: trusted Phase-1 trajectory key set changed")
        entry = load_tensor(root, rows["input"]["artifact"])
        exit_state = load_tensor(root, rows["after_step_001"]["artifact"])
        frozen[name] = {
            "entry_artifact": rows["input"]["artifact"],
            "entry_identity": identity(entry),
            "exit_artifact": rows["after_step_001"]["artifact"],
            "exit_identity": identity(exit_state),
        }
    input_clean_plus = exact_pair_invariants(
        load_tensor(root, frozen["CLEAN"]["entry_artifact"]),
        load_tensor(root, frozen["PLUS1"]["entry_artifact"]),
    )
    exit_clean_plus = exact_pair_invariants(
        load_tensor(root, frozen["CLEAN"]["exit_artifact"]),
        load_tensor(root, frozen["PLUS1"]["exit_artifact"]),
    )
    exit_plus_historical = exact_pair_invariants(
        load_tensor(root, frozen["PLUS1"]["exit_artifact"]),
        load_tensor(root, frozen["HISTORICAL_PLUS14"]["exit_artifact"]),
    )
    if (
        input_clean_plus["differing_element_count"] != 1
        or exit_clean_plus["differing_element_count"] != 41639
        or not exit_plus_historical["bit_exact"]
    ):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 anchor observations changed")
    return {
        "source_root_relative_path": str(root.relative_to(REPO_ROOT)),
        "trace_manifest_sha256": sha256_file(trace_path),
        "source_provenance_hash": trace.get("provenance_hash"),
        "source_manifest_sha256": trace.get("manifest_sha256"),
        "entry_boundary": "input",
        "exit_boundary": "after_step_001",
        "trajectories": frozen,
        "trusted_input_clean_vs_plus1": input_clean_plus,
        "trusted_exit_clean_vs_plus1": exit_clean_plus,
        "trusted_exit_plus1_vs_historical": exit_plus_historical,
    }


def float64_bit_pattern(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def derive_trusted_phase2(config: dict[str, Any]) -> dict[str, Any]:
    root = (REPO_ROOT / config["trusted_phase1_root"]).resolve()
    provenance_path = root / "provenance.json"
    phase2_manifest_path = root / "phase2/phase2_manifest.json"
    phase2_analysis_path = root / "phase2/phase2_analysis.json"
    phase2_gates_path = root / "phase2/phase2_gates.json"
    for path in (provenance_path, phase2_manifest_path, phase2_analysis_path, phase2_gates_path):
        if not path.exists():
            raise GlobalStopError(f"GLOBAL STOP: trusted Phase-2 artifact is missing: {path}")
    source_provenance = json.loads(provenance_path.read_text())
    if source_provenance.get("git_commit") != PHASE3_PHASE2_COMMIT or source_provenance.get("git_dirty"):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 producing revision is not the frozen clean commit")
    phase2_manifest = json.loads(phase2_manifest_path.read_text())
    phase2_analysis = json.loads(phase2_analysis_path.read_text())
    gate_rows = json.loads(phase2_gates_path.read_text()).get("gates", [])
    if (
        [row.get("name") for row in gate_rows] != list(PHASE2_REQUIRED_GATES)
        or any(row.get("required") is not True or row.get("status") != "PASS" for row in gate_rows)
    ):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 gate set is not 25/25 PASS")
    if phase2_manifest.get("selected_step") != 10:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 selected step changed")
    mapping = phase2_manifest.get("selection_mapping", {})
    timestep = mapping.get("selected_scheduler_timestep")
    if (
        mapping.get("phase1_entry_boundary") != "input"
        or mapping.get("phase1_exit_boundary") != "after_step_001"
        or float64_bit_pattern(timestep) != float64_bit_pattern(single_flip.scheduler_timesteps_numpy(config)[10])
    ):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 mapping changed")
    if phase2_analysis.get("plus1_historical_exact_merge", {}).get("classification") != "MERGED_AT_GUIDANCE_OUTPUT":
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 exact classification changed")

    trajectories: dict[str, Any] = {}
    exact_rows: list[dict[str, Any]] = []
    for name in TRAJECTORIES:
        rows = {row["boundary"]: row for row in phase2_manifest["traces"][name]}
        if set(rows) != set(PHASE2_AVAILABLE_BOUNDARIES):
            raise GlobalStopError("GLOBAL STOP: trusted Phase-2 boundary set changed")
        entry = load_tensor(root, rows["transformer_input"]["artifact"])
        exit_state = load_tensor(root, rows["guidance_combined_output"]["artifact"])
        final_latent = load_tensor(root, phase2_manifest["final_latents"][name])
        final_video = load_tensor(root, phase2_manifest["final_videos"][name])
        trajectories[name] = {
            "entry_artifact": rows["transformer_input"]["artifact"],
            "entry_identity": identity(entry),
            "entry_shape": [int(item) for item in entry.shape],
            "entry_storage_dtype": np.dtype(entry.dtype).newbyteorder("<").str,
            "exit_artifact": rows["guidance_combined_output"]["artifact"],
            "exit_identity": identity(exit_state),
            "exit_shape": [int(item) for item in exit_state.shape],
            "exit_storage_dtype": np.dtype(exit_state.dtype).newbyteorder("<").str,
            "final_latent_identity": identity(final_latent),
            "final_video_identity": identity(final_video),
        }
    for boundary in PHASE2_AVAILABLE_BOUNDARIES:
        left_row = next(row for row in phase2_manifest["traces"]["PLUS1"] if row["boundary"] == boundary)
        right_row = next(row for row in phase2_manifest["traces"]["HISTORICAL_PLUS14"] if row["boundary"] == boundary)
        exact_rows.append({
            "boundary": boundary,
            "pair": "PLUS1_VS_HISTORICAL_PLUS14",
            **exact_pair_invariants(
                load_tensor(root, left_row["artifact"]),
                load_tensor(root, right_row["artifact"]),
            ),
        })
    recomputed_event = _phase2_merge_event(exact_rows)
    if recomputed_event["classification"] != "MERGED_AT_GUIDANCE_OUTPUT":
        raise GlobalStopError("GLOBAL STOP: Phase-2 exact event does not reproduce from artifacts")
    return {
        "source_root_relative_path": str(root.relative_to(REPO_ROOT)),
        "producing_commit": PHASE3_PHASE2_COMMIT,
        "source_provenance_hash": phase2_manifest.get("provenance_hash"),
        "phase2_manifest_sha256": sha256_file(phase2_manifest_path),
        "selected_step": 10,
        "selected_scheduler_timestep_bits": float64_bit_pattern(timestep),
        "entry_boundary": "transformer_input",
        "exit_boundary": "guidance_combined_output",
        "exact_classification": recomputed_event["classification"],
        "trajectories": trajectories,
    }


def phase3_freeze(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_step": 10,
        "selected_scheduler_timestep_bits": float64_bit_pattern(single_flip.scheduler_timesteps_numpy(config)[10]),
        "phase2_entry_boundary": "transformer_input",
        "phase2_exit_boundary": "guidance_combined_output",
        "transformer_role": "high_noise_transformer",
        "cfg_execution": "sequential_positive_then_negative",
        "branches": list(PHASE3_BRANCHES),
        "global_block_count": PHASE3_BLOCK_COUNT,
        "start_layer": 0,
        "end_layer": PHASE3_BLOCK_COUNT,
        "block_order": list(range(PHASE3_BLOCK_COUNT)),
        "branch_boundaries": list(PHASE3_FIXED_BOUNDARIES),
        "expected_shapes": _phase3_expected_shapes(config),
        "same_semantics_support_boundaries": [
            "pre_block_hidden_state",
            *(f"after_block_{index:03d}" for index in range(PHASE3_BLOCK_COUNT)),
        ],
        "expected_architecture": {
            "model_class": "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.WanTransformer3DModel",
            "configured_num_layers": PHASE3_BLOCK_COUNT,
            "start_layer": 0,
            "end_layer": PHASE3_BLOCK_COUNT,
            "module_list_length": PHASE3_BLOCK_COUNT,
            "executed_block_order": list(range(PHASE3_BLOCK_COUNT)),
            "num_attention_heads": PHASE3_NUM_HEADS,
            "attention_head_dim": PHASE3_HEAD_DIM,
            "inner_dim": PHASE3_NUM_HEADS * PHASE3_HEAD_DIM,
            "patch_size": [1, 2, 2],
        },
        "expected_artifact_budget": _phase3_artifact_budget(config),
        "cfg_guidance_scale": phase3_cfg_guidance_scale(config),
        "cfg_guidance_scale_bits": float64_bit_pattern(phase3_cfg_guidance_scale(config)),
        "cfg_reconstruction_rule": CFG_RECONSTRUCTION_RULE,
        "intermediate_authentication_limit": (
            "Block-boundary tensors between transformer_entry and raw_transformer_output have no CPU-side external "
            "reference. Their authenticity rests on the committed production write path of the same run, the frozen "
            "block order, the traced-final-equals-trusted-final controls, the Phase-2 entry/exit identity controls, and "
            "the raw-operand CFG reconstruction gate that binds both raw branch outputs to the persisted combined output."
        ),
        "architecture": {
            "pre_block_processing": "patch_embedding -> flatten/transpose -> SP shard point -> condition embedding",
            "block_loop": "self.blocks[self.start_layer:self.end_layer], each exactly once in ascending global index",
            "post_block_processing": "norm_out -> proj_out -> reshape/permute/unpatchify",
            "block_output_aliasing": "no in-place loop assignment observed; each returned hidden_states becomes the next block input",
        },
        "phase4_enabled": False,
        "auto_expand": False,
        "phase1_producing_commit": PHASE3_PHASE1_COMMIT,
        "phase2_producing_commit": PHASE3_PHASE2_COMMIT,
    }


def derive_anchor(config: dict[str, Any]) -> dict[str, Any]:
    trusted_cfg = single_flip.load_config(REPO_ROOT / config["trusted_single_flip_config"])
    derived = single_flip.derive_all(trusted_cfg)
    source, historical = derived["source"], derived["candidate"]
    plus1, plus_record = single_flip.build_single_flip_state(source.clean, int(config["anchor"]["critical_flat_index"]), "up")
    historical_delta = single_flip.derive_historical_delta(source.clean, historical)
    critical = next((row for row in historical_delta["changed_coordinates"] if row["coordinate_flat_index"] == int(config["anchor"]["critical_flat_index"])), None)
    if historical_delta["changed_coordinate_count"] != 6 or critical is None or critical["adjacent_steps"] != 14 or critical["direction"] != "up":
        raise GlobalStopError("GLOBAL STOP: historical anchor does not have the frozen six-coordinate/+14 construction")
    for value in (source.clean, plus1, historical):
        if value.shape != source.clean.shape or not np.array_equal(single_flip.base.cast_runtime_bf16(value), value):
            raise GlobalStopError("GLOBAL STOP: trajectory runtime BF16 construction invalid")
    if plus_record["changed_coordinate_count"] != 1 or plus_record["coordinate_flat_index"] != 516515:
        raise GlobalStopError("GLOBAL STOP: PLUS1 is not the expected single coordinate state")
    trusted_rows = REPO_ROOT / trusted_cfg["output_root"] / "smoke_raw_results.csv"
    if not trusted_rows.exists():
        raise GlobalStopError("GLOBAL STOP: trusted PLUS1 smoke result is missing")
    matches = [
        row for row in csv.DictReader(trusted_rows.open())
        if int(row["coordinate_flat_index"]) == 516515 and row["direction"] == "up" and int(row["replay_id"]) == 0
    ]
    if len(matches) != 1:
        raise GlobalStopError("GLOBAL STOP: trusted PLUS1 smoke identity is ambiguous")
    plus_result_path = resolve_preserved_path(matches[0]["result_path"])
    plus_result = json.loads(plus_result_path.read_text())
    plus_latent = trusted_array_record(resolve_preserved_path(plus_result["recovered_final_latent_artifact"]["path"]), source_path=plus_result_path)
    plus_video = trusted_array_record(resolve_preserved_path(plus_result["recovered_video_artifact"]["path"]), source_path=plus_result_path)
    if plus_latent["canonical_identity"] != matches[0]["recovered_final_latent_identity_sha256_v1"] or plus_video["canonical_identity"] != matches[0]["recovered_video_identity_sha256_v1"]:
        raise GlobalStopError("GLOBAL STOP: trusted PLUS1 row does not match its preserved artifacts")

    historical_result_path = resolve_preserved_path(derived["meta"]["source_result"]["result_path"])
    historical_result = json.loads(historical_result_path.read_text())
    historical_latent = trusted_array_record(resolve_preserved_path(historical_result["recovered_final_latent_path"]), source_path=historical_result_path)
    historical_video = trusted_array_record(resolve_preserved_path(historical_result["recovered_video_path"]), source_path=historical_result_path)
    trusted_finals = {
        "CLEAN": {
            "final_latent_identity": identity(source.final_latent),
            "video_identity": identity(source.video),
            "source_manifest_relative_path": str(source.manifest_path.relative_to(REPO_ROOT)),
            "source_manifest_file_sha256": sha256_file(source.manifest_path),
        },
        "PLUS1": {"final_latent": plus_latent, "video": plus_video},
        "HISTORICAL_PLUS14": {"final_latent": historical_latent, "video": historical_video},
    }
    return {
        "source": source,
        "clean": source.clean,
        "plus1": plus1,
        "historical": historical,
        "plus_record": plus_record,
        "historical_delta": historical_delta,
        "trusted_config": trusted_cfg,
        "trusted_finals": trusted_finals,
        "trusted_phase1": derive_trusted_phase1(config),
        "trusted_phase2": derive_trusted_phase2(config),
    }


def anchor_manifest(root: Path, config: dict[str, Any], data: dict[str, Any], prov: dict[str, Any]) -> dict[str, Any]:
    source = data["source"]
    artifacts = {
        "CLEAN": save_tensor(root, "inputs/clean_runtime_state.npy", data["clean"]),
        "PLUS1": save_tensor(root, "inputs/plus1_runtime_state.npy", data["plus1"]),
        "HISTORICAL_PLUS14": save_tensor(root, "inputs/historical_plus14_runtime_state.npy", data["historical"]),
    }
    bits = single_flip.float32_to_bf16_bits(data["clean"]).reshape(-1)
    flat = int(config["anchor"]["critical_flat_index"])
    clean_bit = int(bits[flat])
    boundary_specs = boundary_specifications(config, source.checkpoint_step)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION, "provenance_hash": prov["provenance_hash"],
        "anchor": {
            "prompt_id": source.prompt_id, "prompt": source.prompt, "prompt_sha256": sha256_bytes(source.prompt.encode()),
            "generation_seed": source.seed, "checkpoint_step": source.checkpoint_step, "resume_index": source.checkpoint_step,
            "resume_timestep": single_flip.anchor_resume_timestep(data["trusted_config"], source.checkpoint_step),
            "scheduler_class": single_flip.EXPECTED_SCHEDULER_CLASS, "scheduler_config": config["scheduler"],
            "model": config["model"], "runtime_dtype": EXPECTED_RUNTIME_DTYPE, "shape": [int(x) for x in source.clean.shape],
            "clean_canonical_identity": artifacts["CLEAN"]["canonical_identity"],
        },
        "critical_coordinate": {
            "flat_index": flat, "multi_index": [int(x) for x in np.unravel_index(flat, source.clean.shape)],
            "clean_bits_hex": f"0x{clean_bit:04x}", "clean_value": float(single_flip.bf16_bits_to_float32(clean_bit)[0]),
            "adjacent_down_bits_hex": f"0x{single_flip.adjacent_bf16_bits(clean_bit, 'down'):04x}",
            "adjacent_up_bits_hex": f"0x{single_flip.adjacent_bf16_bits(clean_bit, 'up'):04x}",
        },
        "trajectories": {"CLEAN": artifacts["CLEAN"], "PLUS1": {**artifacts["PLUS1"], "construction": data["plus_record"]}, "HISTORICAL_PLUS14": {**artifacts["HISTORICAL_PLUS14"], "historical_delta": data["historical_delta"]}},
        "trusted_final_identities": data["trusted_finals"],
        "trusted_phase1": data["trusted_phase1"],
        "trusted_phase2": data["trusted_phase2"],
        "expected_boundaries": [row["boundary"] for row in boundary_specs],
        "boundary_specifications": boundary_specs,
        "early_late_cutoff": early_late_cutoff(len(boundary_specs)),
        "timestep_match_policy": timestep_match_policy(),
        "phase2_freeze": phase2_freeze(config),
        "phase3_freeze": phase3_freeze(config),
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def gate(name: str, passed: bool | None, evidence: Any, *, required: bool) -> dict[str, Any]:
    status = "NOT_TESTED" if passed is None else ("PASS" if passed else "FAIL")
    return {"name": name, "status": status, "required": required, "evidence": evidence}


def validate_gate_document(
    document: dict[str, Any],
    required_names: tuple[str, ...],
    *,
    provenance_hash: str,
    manifest_sha256: str,
) -> None:
    rows = document.get("gates")
    if not isinstance(rows, list):
        raise GlobalStopError("GLOBAL STOP: gate document does not contain a gate list")
    names = [row.get("name") for row in rows]
    if len(names) != len(set(names)):
        raise GlobalStopError("GLOBAL STOP: gate document contains duplicate gate names")
    missing = set(required_names) - set(names)
    if missing:
        raise GlobalStopError(f"GLOBAL STOP: gate document is missing required gates: {sorted(missing)}")
    required_rows = [row for row in rows if row.get("name") in required_names]
    if any(row.get("required") is not True or row.get("status") != "PASS" for row in required_rows):
        raise GlobalStopError("GLOBAL STOP: required gate is FAIL, NOT_TESTED, or non-required")
    if document.get("provenance_hash") != provenance_hash or document.get("manifest_sha256") != manifest_sha256:
        raise GlobalStopError("GLOBAL STOP: gate provenance/manifest binding mismatch")


def config_phase2_is_frozen(manifest: dict[str, Any]) -> bool:
    phase2 = manifest.get("phase2_freeze", {})
    return (
        phase2.get("selected_step") == 10
        and phase2.get("entry_phase1_boundary") == "input"
        and phase2.get("exit_phase1_boundary") == "after_step_001"
        and phase2.get("available_boundaries") == list(PHASE2_AVAILABLE_BOUNDARIES)
        and phase2.get("unavailable_boundaries") == list(PHASE2_UNAVAILABLE_BOUNDARIES)
        and phase2.get("boundary_semantics") == PHASE2_BOUNDARY_SEMANTICS
    )


def write_cpu_gates(root: Path, manifest: dict[str, Any], data: dict[str, Any], prov: dict[str, Any]) -> None:
    hist = data["historical_delta"]
    gates = [
        gate("G1 trusted source hashes unchanged", True, single_flip.validate_source_hashes(data["trusted_config"]), required=True),
        gate("G2 exact frozen anchor identity", True, manifest["anchor"], required=True),
        gate("G3 exact CLEAN tensor identity", True, manifest["trajectories"]["CLEAN"], required=True),
        gate("G4 exact PLUS1 expected tensor identity", True, manifest["trajectories"]["PLUS1"]["construction"], required=True),
        gate("G5 exact HISTORICAL_PLUS14 tensor identity", True, manifest["trajectories"]["HISTORICAL_PLUS14"], required=True),
        gate("G6 PLUS1 exactly one changed coordinate", data["plus_record"]["changed_coordinate_count"] == 1, data["plus_record"], required=True),
        gate("G7 historical support exactly six coordinates", hist["changed_coordinate_count"] == 6, hist["changed_coordinates"], required=True),
        gate("G8 coordinate 516515 historical distance exactly +14 adjacent BF16 steps", next(x for x in hist["changed_coordinates"] if x["coordinate_flat_index"] == 516515)["adjacent_steps"] == 14, hist, required=True),
        gate("G9 Euler scheduler/resume semantics exact", True, manifest["anchor"], required=True),
        gate("G10 model/scheduler/timestep provenance frozen", True, {"provenance": prov, "boundaries": manifest["boundary_specifications"]}, required=True),
        *[gate(f"G{number} GPU scientific gate", None, "GPU work not run in CPU mode", required=False) for number in range(11, 23)],
        gate("G23 phase-2 step explicitly frozen before execution", config_phase2_is_frozen(manifest), manifest["phase2_freeze"], required=True),
        gate("G24 no automatic phase-3 expansion", True, False, required=True),
        gate("G25 no FP32 threshold search", True, "no threshold-search mode exists", required=True),
        gate("G26 no trusted namespace mutation", True, str(root), required=True),
        gate("G27 provenance/config/manifest hash-bound", True, manifest["manifest_sha256"], required=True),
    ]
    document = {"cpu_all_passed": True, "gpu_all_passed": False, "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    validate_gate_document(document, CPU_REQUIRED_GATES, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    atomic_json(root / "cpu_gates.json", document)


def metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise GlobalStopError("GLOBAL STOP: pairwise tensors differ in shape or storage dtype")
    diff = lhs.astype(np.float64) - rhs.astype(np.float64)
    differing = int(np.count_nonzero(lhs.view(np.uint32) != rhs.view(np.uint32)))
    l2 = float(np.sqrt(np.sum(diff**2)))
    rhs_l2 = float(np.sqrt(np.sum(rhs.astype(np.float64) ** 2)))
    return {"bit_exact": bool(differing == 0), "differing_element_count": differing, "differing_fraction": differing / lhs.size, "max_abs_diff": float(np.max(np.abs(diff))), "mean_abs_diff": float(np.mean(np.abs(diff))), "mse": float(np.mean(diff**2)), "l2": l2, "relative_l2": l2 / max(rhs_l2, 1e-300), "lhs_canonical_identity": identity(lhs), "rhs_canonical_identity": identity(rhs)}


def analyze_trace(root: Path, trace_manifest: dict[str, Any]) -> dict[str, Any]:
    expected = trace_manifest["expected_boundaries"]
    expected_specs = trace_manifest.get("boundary_specifications")
    traces = trace_manifest["traces"]
    if set(traces) != set(TRAJECTORIES):
        raise GlobalStopError("GLOBAL STOP: trace trajectory set is incomplete")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for trajectory in TRAJECTORIES:
        rows = traces[trajectory]
        names = [row["boundary"] for row in rows]
        if names != expected or len(set(names)) != len(names):
            raise GlobalStopError("GLOBAL STOP: denoising boundary set is incomplete or duplicated")
        if expected_specs is not None:
            for row, spec in zip(rows, expected_specs, strict=True):
                for key in ("boundary", "resumed_update_index", "absolute_diffusion_step_index", "scheduler_timestep"):
                    if row.get(key) != spec[key]:
                        raise GlobalStopError(f"GLOBAL STOP: trace boundary mapping differs at {spec['boundary']}:{key}")
        loaded[trajectory] = {row["boundary"]: load_tensor(root, row["artifact"]) for row in rows}
    pairings = (("CLEAN", "PLUS1"), ("CLEAN", "HISTORICAL_PLUS14"), ("PLUS1", "HISTORICAL_PLUS14"))
    rows = [{"boundary": boundary, "pair": f"{left}_VS_{right}", **metrics(loaded[left][boundary], loaded[right][boundary])} for boundary in expected for left, right in pairings]
    plus_hist = [row for row in rows if row["pair"] == "PLUS1_VS_HISTORICAL_PLUS14"]
    exact_indices = [index for index, row in enumerate(plus_hist) if row["bit_exact"]]
    # A match at the final boundary alone is not an internal merge. It is a
    # distinct preregistered outcome even though it is trivially persistent.
    persistent = next(
        (
            index
            for index in exact_indices
            if index < len(expected) - 1 and all(item["bit_exact"] for item in plus_hist[index:])
        ),
        None,
    )
    final_exact = bool(plus_hist[-1]["bit_exact"])
    cutoff = trace_manifest.get("early_late_cutoff", early_late_cutoff(len(expected)))
    first_late_index = int(cutoff["first_late_boundary_index"])
    if persistent is not None:
        outcome = "EARLY_EXACT_MERGE" if persistent < first_late_index else "LATE_EXACT_MERGE"
        event = {"classification": "PERSISTENT_EXACT_MERGE", "first_boundary": expected[persistent]}
    elif final_exact and not any(index < len(expected) - 1 for index in exact_indices):
        outcome, event = "FINAL_ONLY_MATCH", {"classification": "FINAL_ONLY_MATCH"}
    elif exact_indices:
        outcome, event = "TRACE_INVALID", {"classification": "TRANSIENT_EXACT_MATCH", "boundaries": [expected[i] for i in exact_indices]}
    elif final_exact:
        outcome, event = "FINAL_ONLY_MATCH", {"classification": "FINAL_ONLY_MATCH"}
    else:
        outcome, event = "NO_EXACT_MERGE", {"classification": "NO_EXACT_MERGE"}
    return {"outcome": outcome, "plus1_historical_event": event, "pairwise_rows": rows, "expected_boundaries": expected}


def run_cpu(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    validate_phase2_config(config)
    validate_output_namespace(config, root)
    prov, data = provenance(config_path, config), derive_anchor(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = anchor_manifest(root, config, data, prov)
    atomic_json(root / "preregistered_config.json", config)
    atomic_json(root / "provenance.json", prov)
    atomic_json(root / "anchor_manifest.json", manifest)
    atomic_json(root / "boundary_plan.json", {"specifications": manifest["boundary_specifications"], "early_late_cutoff": manifest["early_late_cutoff"]})
    write_cpu_gates(root, manifest, data, prov)
    return {"mode": "cpu", "cpu_all_passed": True, "gpu_all_passed": False, "manifest_sha256": manifest["manifest_sha256"], "boundary_count": len(manifest["expected_boundaries"])}


def require_cpu(root: Path, config_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_phase3_config(config)
    validate_phase2_config(config)
    manifest_path, gate_path = root / "anchor_manifest.json", root / "cpu_gates.json"
    if not manifest_path.exists() or not gate_path.exists():
        raise GlobalStopError("GLOBAL STOP: CPU manifest/gates missing")
    manifest, gates = json.loads(manifest_path.read_text()), json.loads(gate_path.read_text())
    prov = provenance(config_path, config)
    if manifest.get("provenance_hash") != prov["provenance_hash"] or gates.get("provenance_hash") != prov["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: stale CPU provenance")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest); unhashed.pop("manifest_sha256", None)
    if recorded_manifest_hash != sha256_bytes(canonical_json(unhashed)) or gates.get("manifest_sha256") != recorded_manifest_hash:
        raise GlobalStopError("GLOBAL STOP: CPU anchor manifest hash mismatch")
    validate_gate_document(
        gates, CPU_REQUIRED_GATES,
        provenance_hash=prov["provenance_hash"], manifest_sha256=recorded_manifest_hash,
    )
    # A self-consistent edited manifest is not authoritative. Reconstruct the
    # frozen source and compare every scientific input to the derived anchor.
    data = derive_anchor(config)
    source = data["source"]
    anchor = manifest.get("anchor", {})
    expected_anchor = {
        "prompt_id": source.prompt_id,
        "generation_seed": source.seed,
        "checkpoint_step": source.checkpoint_step,
        "resume_index": source.checkpoint_step,
        "resume_timestep": single_flip.anchor_resume_timestep(data["trusted_config"], source.checkpoint_step),
        "scheduler_class": single_flip.EXPECTED_SCHEDULER_CLASS,
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "shape": [int(x) for x in source.clean.shape],
    }
    if any(anchor.get(key) != value for key, value in expected_anchor.items()):
        raise GlobalStopError("GLOBAL STOP: manifest anchor does not equal the independently derived source")
    expected_specs = boundary_specifications(config, source.checkpoint_step)
    if manifest.get("boundary_specifications") != expected_specs or manifest.get("expected_boundaries") != [row["boundary"] for row in expected_specs]:
        raise GlobalStopError("GLOBAL STOP: manifest boundary mapping differs from the frozen scheduler/config")
    if manifest.get("early_late_cutoff") != early_late_cutoff(len(expected_specs)):
        raise GlobalStopError("GLOBAL STOP: manifest early/late cutoff differs from the frozen rule")
    if manifest.get("timestep_match_policy") != timestep_match_policy():
        raise GlobalStopError("GLOBAL STOP: manifest timestep match policy differs from the frozen rule")
    expected_states = {
        "CLEAN": data["clean"],
        "PLUS1": data["plus1"],
        "HISTORICAL_PLUS14": data["historical"],
    }
    for name, expected_state in expected_states.items():
        record = manifest.get("trajectories", {}).get(name)
        if record is None:
            raise GlobalStopError(f"GLOBAL STOP: manifest is missing {name} input")
        actual_state = load_tensor(root, record)
        if not np.array_equal(actual_state, expected_state) or record.get("canonical_identity") != identity(expected_state):
            raise GlobalStopError(f"GLOBAL STOP: manifest {name} input differs from frozen construction")
    historical = manifest["trajectories"]["HISTORICAL_PLUS14"].get("historical_delta", {})
    if historical.get("changed_coordinate_count") != 6 or exact_historical_invariants(historical) != exact_historical_invariants(data["historical_delta"]):
        raise GlobalStopError("GLOBAL STOP: manifest historical support differs from frozen construction")
    construction = manifest["trajectories"]["PLUS1"].get("construction", {})
    if exact_construction_invariants(construction) != exact_construction_invariants(data["plus_record"]):
        raise GlobalStopError("GLOBAL STOP: manifest PLUS1 construction differs from frozen construction")
    if manifest.get("trusted_final_identities") != data["trusted_finals"]:
        raise GlobalStopError("GLOBAL STOP: manifest trusted final identities differ from frozen artifacts")
    if manifest.get("trusted_phase1") != data["trusted_phase1"]:
        raise GlobalStopError("GLOBAL STOP: manifest trusted Phase-1 binding differs from preserved artifacts")
    if manifest.get("trusted_phase2") != data["trusted_phase2"]:
        raise GlobalStopError("GLOBAL STOP: manifest trusted Phase-2 binding differs from preserved artifacts")
    if manifest.get("phase2_freeze") != phase2_freeze(config):
        raise GlobalStopError("GLOBAL STOP: manifest Phase-2 freeze differs from config/scheduler")
    if not config_phase2_is_frozen(manifest):
        raise GlobalStopError("GLOBAL STOP: manifest Phase-2 freeze differs from the preregistered configuration")
    if manifest.get("phase3_freeze") != phase3_freeze(config):
        raise GlobalStopError("GLOBAL STOP: manifest Phase-3 freeze differs from config/source architecture")
    expected_mapping = phase2_selection_mapping(config, manifest, 10)
    if manifest["phase2_freeze"].get("selected_scheduler_timestep") != expected_mapping["selected_scheduler_timestep"]:
        raise GlobalStopError("GLOBAL STOP: manifest Phase-2 scheduler timestep changed")
    return manifest, prov


def validate_probe_records(records: list[dict[str, Any]], expected_specs: list[dict[str, Any]]) -> None:
    """Bind every persisted probe record to the frozen boundary specification.

    Production semantics: the resume input is a float32 tensor holding
    BF16-exact values; every scheduler output is BF16. Any deviation, in
    either direction, is a change of execution semantics and fails closed.
    """
    if [int(row.get("step_index", -1)) for row in records] != list(range(len(expected_specs))):
        raise GlobalStopError("GLOBAL STOP: trajectory probe did not persist every requested boundary")
    frozen_schedule = sorted({float(spec["scheduler_timestep"]) for spec in expected_specs})
    for record, expected in zip(records, expected_specs, strict=True):
        if record.get("runtime_dtype") != expected["expected_runtime_dtype"]:
            raise GlobalStopError(
                f"GLOBAL STOP: {expected['boundary']} runtime dtype {record.get('runtime_dtype')} "
                f"differs from production semantics {expected['expected_runtime_dtype']}"
            )
        if not timestep_matches(record.get("timestep"), float(expected["scheduler_timestep"]), frozen_schedule):
            raise GlobalStopError(
                f"GLOBAL STOP: {expected['boundary']} scheduler timestep mapping is ambiguous "
                f"(observed {record.get('timestep')}, frozen {expected['scheduler_timestep']})"
            )
        if not record.get("latent_path"):
            raise GlobalStopError(f"GLOBAL STOP: {expected['boundary']} has no persisted latent")


def _run_trace(omni: Any, config: dict[str, Any], source: Any, state: np.ndarray, name: str, root: Path) -> dict[str, Any]:
    """Run one Euler resume and persist every scheduler-update latent losslessly."""
    import torch

    remaining = int(config["generation"]["num_inference_steps"]) - int(source.checkpoint_step)
    capture = list(range(remaining + 1))
    run_dir = root / "phase1" / name.lower()
    # The FP32 tensor contains exact BF16 values and is the production resume
    # input. It is not recast and no persisted probe is ever fed back.
    video, metadata, _ = v3.run_generate(
        omni, config, prompt=source.prompt, seed=source.seed, label=name.lower(), artifact_dir=run_dir / "probe",
        capture_steps=capture, latents=torch.from_numpy(np.ascontiguousarray(state)), step_index=source.checkpoint_step,
    )
    if metadata.get("sample_solver") != "euler" or str(metadata.get("scheduler_class", "")) != single_flip.EXPECTED_SCHEDULER_CLASS:
        raise GlobalStopError("GLOBAL STOP: phase-1 runtime was not the frozen Euler scheduler")
    records = metadata.get("records", [])
    expected_specs = boundary_specifications(config, source.checkpoint_step)
    validate_probe_records(records, expected_specs)
    rows = []
    for record, expected in zip(records, expected_specs, strict=True):
        source_path = Path(record["latent_path"])
        latent = torch.load(source_path, map_location="cpu").detach().cpu().float().numpy()
        storage_dtype = np.dtype(latent.dtype).newbyteorder("<").str
        if storage_dtype != expected["expected_storage_dtype"]:
            raise GlobalStopError(f"GLOBAL STOP: {expected['boundary']} persisted storage dtype changed")
        rows.append({
            **expected,
            "probe_step_index": int(record["step_index"]),
            "observed_scheduler_timestep": record["timestep"],
            "observed_runtime_dtype": record["runtime_dtype"],
            "observed_storage_dtype": storage_dtype,
            "artifact": save_tensor(
                root, f"phase1/artifacts/{name}/{expected['boundary']}.npy", latent,
                runtime_semantics=expected["runtime_dtype_semantics"],
            ),
        })
    video_artifact = save_tensor(root, f"phase1/artifacts/{name}/final_video.npy", video, runtime_semantics="uint8 decoded video")
    return {"trajectory": name, "boundaries": rows, "final_latent_artifact": rows[-1]["artifact"], "final_video_artifact": video_artifact}


def _build_omni(config: dict[str, Any], args: argparse.Namespace) -> Any:
    return v3.build_omni(config, args)


def _preflight_document(root: Path, manifest: dict[str, Any], prov: dict[str, Any], controls: dict[str, list[dict[str, Any]]], source: Any) -> dict[str, Any]:
    final_ids = {name: [row["final_latent_identity"] for row in values] for name, values in controls.items()}
    video_ids = {name: [row["video_identity"] for row in values] for name, values in controls.items()}
    stable = {name: len(set(final_ids[name])) == len(set(video_ids[name])) == 1 for name in TRAJECTORIES}
    plus_hist_equal = final_ids["PLUS1"][0] == final_ids["HISTORICAL_PLUS14"][0] and video_ids["PLUS1"][0] == video_ids["HISTORICAL_PLUS14"][0]
    clean_differs = final_ids["CLEAN"][0] != final_ids["PLUS1"][0] and video_ids["CLEAN"][0] != video_ids["PLUS1"][0]
    clean_trusted = final_ids["CLEAN"][0] == identity(source.final_latent) and video_ids["CLEAN"][0] == identity(source.video)
    trusted = manifest["trusted_final_identities"]
    expected_plus = trusted["PLUS1"]
    expected_historical = trusted["HISTORICAL_PLUS14"]
    plus_trusted = final_ids["PLUS1"][0] == expected_plus["final_latent"]["canonical_identity"] and video_ids["PLUS1"][0] == expected_plus["video"]["canonical_identity"]
    historical_trusted = final_ids["HISTORICAL_PLUS14"][0] == expected_historical["final_latent"]["canonical_identity"] and video_ids["HISTORICAL_PLUS14"][0] == expected_historical["video"]["canonical_identity"]
    gates = [
        gate("G11 CLEAN repeated final determinism", stable["CLEAN"], final_ids["CLEAN"], required=True),
        gate("G12 PLUS1 repeated final determinism", stable["PLUS1"], final_ids["PLUS1"], required=True),
        gate("G13 historical repeated final determinism", stable["HISTORICAL_PLUS14"], final_ids["HISTORICAL_PLUS14"], required=True),
        gate("G14 PLUS1 final == historical final", plus_hist_equal, {"plus": expected_plus, "historical": expected_historical}, required=True),
        gate("G15 CLEAN final != PLUS1 final", clean_differs, {"clean": final_ids["CLEAN"], "plus": final_ids["PLUS1"]}, required=True),
        gate("G15a CLEAN final equals trusted clean", clean_trusted, trusted["CLEAN"], required=True),
        gate("G15b PLUS1 final equals trusted PLUS1", plus_trusted, expected_plus, required=True),
        gate("G15c historical final equals trusted historical", historical_trusted, expected_historical, required=True),
    ]
    document = {"gpu_all_passed": True, "gates": gates, "controls": controls, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    validate_gate_document(document, PREFLIGHT_REQUIRED_GATES, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    atomic_json(root / "preflight" / "preflight_gates.json", document)
    return document


def run_preflight(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config)
    require_committed_source(prov)
    data = derive_anchor(config); source = data["source"]
    candidates = {"CLEAN": data["clean"], "PLUS1": data["plus1"], "HISTORICAL_PLUS14": data["historical"]}
    omni = _build_omni(config, args); controls: dict[str, list[dict[str, Any]]] = {name: [] for name in TRAJECTORIES}
    try:
        for name, state in candidates.items():
            for replay in range(int(config["controls"]["repeats_per_trajectory"])):
                result = single_flip.base.run_resume(omni, config, source, state, step_index=source.checkpoint_step, label=f"{name.lower()}_control_{replay}", directory=root / "preflight" / name / f"repeat_{replay}")
                latent_record = tensor_record_from_file(root, Path(result["recovered_final_latent_artifact"]["path"]))
                video_record = tensor_record_from_file(root, Path(result["recovered_video_artifact"]["path"]), runtime_semantics="uint8 decoded video")
                latent = load_tensor(root, latent_record)
                video = load_tensor(root, video_record)
                controls[name].append({
                    "repeat_id": replay,
                    "final_latent_identity": identity(latent), "video_identity": identity(video),
                    "final_latent_artifact": latent_record, "video_artifact": video_record,
                    "exact_vs_clean": bool(np.array_equal(latent, source.final_latent) and np.array_equal(video, source.video)),
                })
        doc = _preflight_document(root, manifest, prov, controls, source)
        return {"mode": "preflight", **doc}
    finally:
        single_flip._shutdown(omni)


def tensor_record_from_file(root: Path, path: Path, *, runtime_semantics: str = EXPECTED_RUNTIME_DTYPE) -> dict[str, Any]:
    """Root-relative, identity-bearing record for an array already persisted under ``root`` by trusted base code."""
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise GlobalStopError(f"GLOBAL STOP: preflight artifact is outside the result root: {path}")
    value = np.load(path, allow_pickle=False)
    return {
        "relative_path": relative_path(root, path), "file_sha256": sha256_file(path),
        "canonical_identity": identity(value), "storage_dtype": np.dtype(value.dtype).newbyteorder("<").str,
        "shape": [int(x) for x in value.shape], "nbytes": int(value.nbytes),
        "runtime_dtype_semantics": runtime_semantics,
    }


def rederive_preflight_controls(root: Path, manifest: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every preflight scientific fact from the persisted repeat artifacts; PASS booleans are not trusted."""
    repeats = 3
    controls = document.get("controls")
    if not isinstance(controls, dict) or set(controls) != set(TRAJECTORIES):
        raise GlobalStopError("GLOBAL STOP: preflight controls do not cover exactly the three trajectories")
    identities: dict[str, list[tuple[str, str]]] = {}
    for name in TRAJECTORIES:
        rows = controls[name]
        if not isinstance(rows, list) or sorted(int(row.get("repeat_id", -1)) for row in rows) != list(range(repeats)):
            raise GlobalStopError(f"GLOBAL STOP: preflight {name} repeat key set is not exactly repeats 0..{repeats - 1}")
        identities[name] = []
        for row in sorted(rows, key=lambda item: int(item["repeat_id"])):
            latent = load_tensor(root, row["final_latent_artifact"])
            video = load_tensor(root, row["video_artifact"])
            identities[name].append((identity(latent), identity(video)))
    trusted = {name: _trusted_final_pair(manifest, name) for name in TRAJECTORIES}
    facts = {
        "G11 CLEAN repeated final determinism": len(set(identities["CLEAN"])) == 1,
        "G12 PLUS1 repeated final determinism": len(set(identities["PLUS1"])) == 1,
        "G13 historical repeated final determinism": len(set(identities["HISTORICAL_PLUS14"])) == 1,
        "G14 PLUS1 final == historical final": identities["PLUS1"][0] == identities["HISTORICAL_PLUS14"][0],
        "G15 CLEAN final != PLUS1 final": identities["CLEAN"][0][0] != identities["PLUS1"][0][0] and identities["CLEAN"][0][1] != identities["PLUS1"][0][1],
        "G15a CLEAN final equals trusted clean": identities["CLEAN"][0] == trusted["CLEAN"],
        "G15b PLUS1 final equals trusted PLUS1": identities["PLUS1"][0] == trusted["PLUS1"],
        "G15c historical final equals trusted historical": identities["HISTORICAL_PLUS14"][0] == trusted["HISTORICAL_PLUS14"],
    }
    if set(facts) != set(PREFLIGHT_REQUIRED_GATES):
        raise GlobalStopError("GLOBAL STOP: preflight fact set does not match the required gate set")
    failed = [name for name, passed in facts.items() if not passed]
    if failed:
        raise GlobalStopError(f"GLOBAL STOP: preflight controls re-derived from artifacts fail: {failed}")
    return {"identities": identities, "facts": facts}


def require_preflight(root: Path, manifest: dict[str, Any], prov: dict[str, Any]) -> None:
    """Authorize GPU science only from persisted repeat artifacts bound to the current provenance and manifest."""
    path = root / "preflight" / "preflight_gates.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: preflight gates are missing")
    value = json.loads(path.read_text())
    validate_gate_document(
        value, PREFLIGHT_REQUIRED_GATES,
        provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"],
    )
    rederived = rederive_preflight_controls(root, manifest, value)
    declared = {row["name"]: row.get("status") == "PASS" for row in value["gates"]}
    if any(declared.get(name) is not True for name in rederived["facts"]):
        raise GlobalStopError("GLOBAL STOP: preflight gate document disagrees with re-derived artifact facts")


def run_phase1(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    data = derive_anchor(config); source = data["source"]
    omni = _build_omni(config, args)
    try:
        trace_results = {
            name: _run_trace(omni, config, source, state, name, root)
            for name, state in {"CLEAN": data["clean"], "PLUS1": data["plus1"], "HISTORICAL_PLUS14": data["historical"]}.items()
        }
        traces = {name: result["boundaries"] for name, result in trace_results.items()}
        document = {
            "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"],
            "expected_boundaries": manifest["expected_boundaries"],
            "boundary_specifications": manifest["boundary_specifications"],
            "early_late_cutoff": manifest["early_late_cutoff"],
            "traces": traces,
            "final_latents": {name: result["final_latent_artifact"] for name, result in trace_results.items()},
            "final_videos": {name: result["final_video_artifact"] for name, result in trace_results.items()},
        }
        atomic_json(root / "phase1" / "trace_manifest.json", document)
        return {"mode": "phase1", "trajectory_count": len(traces), "boundary_count": len(manifest["expected_boundaries"])}
    finally:
        single_flip._shutdown(omni)


def _trusted_final_pair(manifest: dict[str, Any], name: str) -> tuple[str, str]:
    trusted = manifest["trusted_final_identities"][name]
    if name == "CLEAN":
        return trusted["final_latent_identity"], trusted["video_identity"]
    return trusted["final_latent"]["canonical_identity"], trusted["video"]["canonical_identity"]


def validate_traced_finals(root: Path, manifest: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    mismatches: list[str] = []
    for name in TRAJECTORIES:
        latent = load_tensor(root, trace["final_latents"][name])
        video = load_tensor(root, trace["final_videos"][name])
        latent_id, video_id = identity(latent), identity(video)
        expected_latent, expected_video = _trusted_final_pair(manifest, name)
        evidence[name] = {
            "traced_final_latent_identity": latent_id,
            "trusted_final_latent_identity": expected_latent,
            "traced_video_identity": video_id,
            "trusted_video_identity": expected_video,
            "matches": latent_id == expected_latent and video_id == expected_video,
        }
        if not evidence[name]["matches"]:
            mismatches.append(name)
    if mismatches:
        failure = {"classification": "TRACE_ALTERS_EXECUTION", "mismatched_trajectories": mismatches, "evidence": evidence}
        atomic_json(root / "phase1" / "trace_alters_execution.json", failure)
        raise GlobalStopError(f"GLOBAL STOP: TRACE_ALTERS_EXECUTION: {mismatches}")
    return evidence


def run_analyze_phase1(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    path = root / "phase1" / "trace_manifest.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: phase-1 trace manifest is missing")
    trace = json.loads(path.read_text())
    if trace.get("provenance_hash") != prov["provenance_hash"] or trace.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: phase-1 trace provenance mismatch")
    if trace.get("boundary_specifications") != manifest["boundary_specifications"] or trace.get("early_late_cutoff") != manifest["early_late_cutoff"]:
        raise GlobalStopError("GLOBAL STOP: phase-1 boundary mapping/cutoff differs from CPU manifest")
    final_control_evidence = validate_traced_finals(root, manifest, trace)
    result = analyze_trace(root, trace)
    # G16--G22 are recomputed from Phase-1 artifacts, not copied from rows.
    expected = manifest["expected_boundaries"]
    traces = trace.get("traces", {})
    trajectory_sets_complete = set(traces) == set(TRAJECTORIES) and all(
        [row.get("boundary") for row in traces[name]] == expected for name in TRAJECTORIES
    )
    no_duplicates = all(
        len({row.get("boundary") for row in traces[name]}) == len(expected) for name in TRAJECTORIES
    )
    artifacts_valid = True
    finite = True
    try:
        for name in TRAJECTORIES:
            for row in traces[name]:
                artifact = load_tensor(root, row["artifact"])
                artifacts_valid &= artifact.shape == tuple(manifest["anchor"]["shape"])
                finite &= bool(np.isfinite(artifact).all())
            video = load_tensor(root, trace["final_videos"][name])
            artifacts_valid &= video.dtype == np.uint8
            finite &= bool(np.isfinite(video).all())
    except (KeyError, GlobalStopError):
        artifacts_valid = False
        finite = False
    pairwise_complete = len(result["pairwise_rows"]) == len(expected) * 3 and all(
        np.isfinite(float(row[key])) for row in result["pairwise_rows"]
        for key in ("mse", "l2", "relative_l2", "max_abs_diff", "mean_abs_diff")
    )
    relocation = all(not Path(row["artifact"]["relative_path"]).is_absolute() for name in TRAJECTORIES for row in traces[name])
    relocation &= all(not Path(trace["final_videos"][name]["relative_path"]).is_absolute() for name in TRAJECTORIES)
    gate_values = {
        "G16 expected denoising boundary key set complete": trajectory_sets_complete,
        "G17 no duplicate boundaries": no_duplicates,
        "G18 all trajectory artifacts exist": artifacts_valid,
        "G19 artifact identities recompute correctly": artifacts_valid,
        "G20 pairwise metrics recompute from artifacts": pairwise_complete,
        "G21 no NaN/Inf": finite and pairwise_complete,
        "G22 relative paths resolve under relocated result root": relocation,
    }
    if not all(gate_values.values()):
        failed = [name for name, passed in gate_values.items() if not passed]
        raise GlobalStopError(f"GLOBAL STOP: phase-1 artifact gate failed: {failed}")
    result["gates"] = gate_values
    result["traced_vs_trusted_final_controls"] = final_control_evidence
    atomic_json(root / "phase1" / "phase1_analysis.json", result)
    return {"mode": "analyze-phase1", "outcome": result["outcome"], "pairwise_rows": len(result["pairwise_rows"])}


def _phase2_runtime_dtype(boundary: str) -> str:
    return EXPECTED_INPUT_RUNTIME_DTYPE if boundary in ("latent_entering_step", "scheduler_input") else EXPECTED_RUNTIME_DTYPE


def _run_within_step_trace(omni: Any, config: dict[str, Any], source: Any, state: np.ndarray, name: str, root: Path, selected_step: int) -> dict[str, Any]:
    """Capture only actual Wan T2V operations for one frozen absolute step."""
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    local_step = selected_step - int(source.checkpoint_step)
    if not 0 <= local_step < int(config["generation"]["num_inference_steps"]) - int(source.checkpoint_step):
        raise GlobalStopError("GLOBAL STOP: phase-2 selected_step is outside the resumed denoising range")
    generation = config["generation"]
    sampling = OmniDiffusionSamplingParams(
        height=int(generation["height"]), width=int(generation["width"]), num_frames=int(generation["num_frames"]),
        num_inference_steps=int(generation["num_inference_steps"]), guidance_scale=float(generation["guidance_scale"]),
        fps=float(generation["fps"]), seed=source.seed, generator=torch.Generator(device="cpu").manual_seed(source.seed),
    )
    sampling.latents = torch.from_numpy(np.ascontiguousarray(state))
    sampling.step_index = int(source.checkpoint_step)
    sampling.extra_args = {
        "flow_shift": float(config["scheduler"]["flow_shift"]), "sample_solver": "euler",
        "trajectory_probe": {
            "artifact_dir": str(root / "phase2" / name.lower() / "final_probe"),
            "request_label": name.lower(),
            "capture_steps": [0, int(config["generation"]["num_inference_steps"]) - int(source.checkpoint_step)],
            "save_latents": True,
            "save_decoded": False,
            "save_mp4": False,
        },
        "within_step_probe": {
            "artifact_dir": str(root / "phase2" / name.lower() / "probe"), "request_label": name.lower(),
            "selected_local_step": local_step, "selected_absolute_step": selected_step,
        },
    }
    outputs = omni.generate({"prompt": source.prompt}, sampling)
    video, output = v3.normalize_video(outputs)
    probe = output.custom_output.get("within_step_probe")
    if not isinstance(probe, dict):
        raise GlobalStopError("GLOBAL STOP: within-step probe output is missing")
    allowed = list(PHASE2_AVAILABLE_BOUNDARIES)
    records = probe.get("records", [])
    if [row.get("boundary") for row in records] != allowed or probe.get("unavailable_boundaries") != list(PHASE2_UNAVAILABLE_BOUNDARIES):
        raise GlobalStopError("GLOBAL STOP: Phase-2 did not record the actual Wan boundary set")
    schedule = single_flip.scheduler_timesteps_numpy(config)
    expected_timestep = schedule[selected_step]
    if any(
        int(record.get("step_idx", -1)) != local_step
        or int(record.get("absolute_step", -1)) != selected_step
        or not timestep_matches(record.get("timestep"), expected_timestep, schedule)
        for record in records
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-2 probe step/timestep mapping differs from the frozen scheduler")
    result = []
    for record in records:
        expected_runtime_dtype = _phase2_runtime_dtype(record["boundary"])
        if record.get("runtime_dtype") != expected_runtime_dtype:
            raise GlobalStopError(f"GLOBAL STOP: Phase-2 {record['boundary']} runtime dtype differs from production semantics")
        tensor = torch.load(Path(record["latent_path"]), map_location="cpu").detach().cpu().float().numpy()
        storage_dtype = np.dtype(tensor.dtype).newbyteorder("<").str
        result.append({
            "boundary": record["boundary"], "absolute_step": int(record["absolute_step"]), "timestep": record["timestep"],
            "phase1_entry_boundary": "input" if local_step == 0 else f"after_step_{local_step:03d}",
            "runtime_dtype": record["runtime_dtype"], "storage_dtype": storage_dtype,
            "actual_shape": [int(item) for item in tensor.shape],
            **PHASE2_BOUNDARY_SEMANTICS[record["boundary"]],
            "artifact": save_tensor(root, f"phase2/artifacts/{name}/{record['boundary']}.npy", tensor, runtime_semantics=record["runtime_dtype"]),
        })
    trajectory_probe = output.custom_output.get("trajectory_probe_metadata")
    if not isinstance(trajectory_probe, dict):
        raise GlobalStopError("GLOBAL STOP: Phase-2 final-latent trajectory probe is missing")
    final_step_index = int(config["generation"]["num_inference_steps"]) - int(source.checkpoint_step)
    final_rows = [row for row in trajectory_probe.get("records", []) if int(row.get("step_index", -1)) == final_step_index]
    if len(final_rows) != 1 or not final_rows[0].get("latent_path"):
        raise GlobalStopError("GLOBAL STOP: Phase-2 final latent artifact is missing or ambiguous")
    final_latent = torch.load(Path(final_rows[0]["latent_path"]), map_location="cpu").detach().cpu().float().numpy()
    return {
        "trajectory": name,
        "boundaries": result,
        "final_latent_artifact": save_tensor(root, f"phase2/artifacts/{name}/final_latent.npy", final_latent),
        "final_video_artifact": save_tensor(root, f"phase2/artifacts/{name}/final_video.npy", video, runtime_semantics="uint8 decoded video"),
    }


def run_phase2(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    validate_phase2_config(config)
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    require_committed_source(prov)
    selected = config["phase2"]["selected_step"]
    selection_mapping = phase2_selection_mapping(config, manifest, int(selected))
    data = derive_anchor(config); omni = _build_omni(config, args)
    try:
        results = {name: _run_within_step_trace(omni, config, data["source"], state, name, root, int(selected)) for name, state in {"CLEAN": data["clean"], "PLUS1": data["plus1"], "HISTORICAL_PLUS14": data["historical"]}.items()}
        traces = {name: result["boundaries"] for name, result in results.items()}
        document = {
            "provenance_hash": prov["provenance_hash"],
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_step": int(selected),
            "selection_mapping": selection_mapping,
            "expected_boundaries": list(PHASE2_AVAILABLE_BOUNDARIES),
            "boundary_semantics": PHASE2_BOUNDARY_SEMANTICS,
            "traces": traces,
            "final_latents": {name: result["final_latent_artifact"] for name, result in results.items()},
            "final_videos": {name: result["final_video_artifact"] for name, result in results.items()},
            "unavailable_boundaries": list(PHASE2_UNAVAILABLE_BOUNDARIES),
        }
        atomic_json(root / "phase2" / "phase2_manifest.json", document)
        return {"mode": "phase2", "selected_step": int(selected), "boundaries": len(next(iter(traces.values())))}
    finally:
        single_flip._shutdown(omni)


def _phase2_merge_event(pairwise_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in pairwise_rows if row["pair"] == "PLUS1_VS_HISTORICAL_PLUS14"]
    by_boundary = {row["boundary"]: row for row in rows}
    if not by_boundary["scheduler_output"]["bit_exact"]:
        raise GlobalStopError("GLOBAL STOP: CONTRADICTS_PHASE1: PLUS1/HISTORICAL scheduler outputs differ")
    classifications = {
        "transformer_input": "MERGED_AT_TRANSFORMER_INPUT",
        "guidance_combined_output": "MERGED_AT_GUIDANCE_OUTPUT",
        "scheduler_input": "MERGED_AT_SCHEDULER_INPUT",
        "scheduler_output": "MERGED_AT_SCHEDULER_OUTPUT",
    }
    exact = [boundary for boundary in PHASE2_AVAILABLE_BOUNDARIES if by_boundary[boundary]["bit_exact"]]
    if exact and exact[0] == "latent_entering_step":
        raise GlobalStopError("GLOBAL STOP: Phase-2 entry contradicts the distinct trusted Phase-1 inputs")
    boundary = exact[0]
    return {
        "classification": classifications[boundary],
        "first_bit_exact_boundary": boundary,
        "exact_boundaries": exact,
        "allowed_claim": f"PLUS1 and HISTORICAL first become bit-exact at boundary {boundary}.",
    }


def _clean_plus_spread(pairwise_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [row for row in pairwise_rows if row["pair"] == "CLEAN_VS_PLUS1"]
    progression: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in source:
        item = {key: row[key] for key in (
            "boundary", "differing_element_count", "differing_fraction", "max_abs_diff", "mean_abs_diff", "mse", "l2", "relative_l2",
            "lhs_canonical_identity", "rhs_canonical_identity",
        )}
        if previous is None:
            item.update({"delta_differing_count": None, "ratio_differing_count": None, "delta_relative_l2": None, "ratio_relative_l2": None})
        else:
            item.update({
                "delta_differing_count": row["differing_element_count"] - previous["differing_element_count"],
                "ratio_differing_count": row["differing_element_count"] / previous["differing_element_count"] if previous["differing_element_count"] else None,
                "delta_relative_l2": row["relative_l2"] - previous["relative_l2"],
                "ratio_relative_l2": row["relative_l2"] / previous["relative_l2"] if previous["relative_l2"] else None,
            })
        progression.append(item)
        previous = row
    increases = [row for row in progression[1:] if row["delta_differing_count"] is not None]
    largest = max(row["delta_differing_count"] for row in increases)
    tied = [row["boundary"] for row in increases if row["delta_differing_count"] == largest]
    return progression, {
        "statistic": "largest absolute increase in differing_element_count between consecutive available Phase-2 boundaries",
        "largest_increase": largest,
        "boundaries": tied,
        "descriptive_only": True,
    }


def _load_trusted_phase1_boundary(root: Path, manifest: dict[str, Any], name: str, which: str) -> np.ndarray:
    trusted = manifest["trusted_phase1"]
    trace_path = root / "phase1/trace_manifest.json"
    if not trace_path.exists():
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 trace is missing from the result root")
    if sha256_file(trace_path) != trusted["trace_manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 source file hash changed")
    return load_tensor(root, trusted["trajectories"][name][f"{which}_artifact"])


def _p2_gate(number: int, description: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {**gate(f"P2-G{number}", passed, evidence, required=True), "description": description}


def analyze_phase2_artifacts(root: Path, config: dict[str, Any], manifest: dict[str, Any], prov: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    validate_phase2_config(config)
    require_committed_source(prov)
    selected = int(config["phase2"]["selected_step"])
    mapping = phase2_selection_mapping(config, manifest, selected)
    if trace.get("provenance_hash") != prov["provenance_hash"] or trace.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: Phase-2 trace provenance mismatch")
    if int(trace.get("selected_step", -1)) != selected or trace.get("selection_mapping") != mapping:
        raise GlobalStopError("GLOBAL STOP: Phase-2 selected-step binding mismatch")
    expected = list(PHASE2_AVAILABLE_BOUNDARIES)
    if trace.get("expected_boundaries") != expected or trace.get("unavailable_boundaries") != list(PHASE2_UNAVAILABLE_BOUNDARIES):
        raise GlobalStopError("GLOBAL STOP: Phase-2 real/unavailable boundary declaration changed")
    if trace.get("boundary_semantics") != PHASE2_BOUNDARY_SEMANTICS:
        raise GlobalStopError("GLOBAL STOP: Phase-2 boundary semantics changed")
    traces = trace.get("traces")
    if not isinstance(traces, dict) or set(traces) != set(TRAJECTORIES):
        raise GlobalStopError("GLOBAL STOP: Phase-2 trajectory set is incomplete")
    schedule = single_flip.scheduler_timesteps_numpy(config)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    normalized_rows: dict[str, list[dict[str, Any]]] = {}
    artifact_evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for name in TRAJECTORIES:
        rows = traces[name]
        names = [row.get("boundary") for row in rows]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise GlobalStopError("GLOBAL STOP: Phase-2 operation boundary set is missing, duplicate, or unexpected")
        by_name = {row["boundary"]: row for row in rows}
        normalized_rows[name] = [by_name[boundary] for boundary in expected]
        loaded[name] = {}
        artifact_evidence[name] = {}
        for boundary, row in zip(expected, normalized_rows[name], strict=True):
            if (
                int(row.get("absolute_step", -1)) != selected
                or row.get("phase1_entry_boundary") != mapping["phase1_entry_boundary"]
                or not timestep_matches(row.get("timestep"), mapping["selected_scheduler_timestep"], schedule)
                or row.get("runtime_dtype") != _phase2_runtime_dtype(boundary)
                or row.get("storage_dtype") != "<f4"
                or {key: row.get(key) for key in PHASE2_BOUNDARY_SEMANTICS[boundary]} != PHASE2_BOUNDARY_SEMANTICS[boundary]
            ):
                raise GlobalStopError(f"GLOBAL STOP: Phase-2 boundary metadata changed at {name}:{boundary}")
            value = load_tensor(root, row["artifact"])
            if row.get("actual_shape") != [int(item) for item in value.shape]:
                raise GlobalStopError("GLOBAL STOP: Phase-2 boundary shape metadata does not match artifact")
            loaded[name][boundary] = value
            artifact_evidence[name][boundary] = {
                "recorded_identity": row["artifact"]["canonical_identity"],
                "recomputed_identity": identity(value),
                "valid": row["artifact"]["canonical_identity"] == identity(value),
            }
    pairings = (("CLEAN", "PLUS1"), ("CLEAN", "HISTORICAL_PLUS14"), ("PLUS1", "HISTORICAL_PLUS14"))
    pairwise_rows = [
        {"boundary": boundary, "pair": f"{left}_VS_{right}", **metrics(loaded[left][boundary], loaded[right][boundary])}
        for boundary in expected for left, right in pairings
    ]
    expected_pair_identities = {
        (boundary, f"{left}_VS_{right}"): (
            identity(loaded[left][boundary]), identity(loaded[right][boundary])
        )
        for boundary in expected for left, right in pairings
    }
    pairwise_recomputed = all(
        (row["lhs_canonical_identity"], row["rhs_canonical_identity"])
        == expected_pair_identities[(row["boundary"], row["pair"])]
        for row in pairwise_rows
    )
    if pairwise_rows[0]["differing_element_count"] != 1:
        raise GlobalStopError("GLOBAL STOP: Phase-2 CLEAN/PLUS1 entry does not reproduce one changed coordinate")

    entry_matches: dict[str, bool] = {}
    exit_matches: dict[str, bool] = {}
    input_matches: dict[str, bool] = {}
    for name in TRAJECTORIES:
        phase1_entry = _load_trusted_phase1_boundary(root, manifest, name, "entry")
        phase1_exit = _load_trusted_phase1_boundary(root, manifest, name, "exit")
        entry_matches[name] = identity(loaded[name]["latent_entering_step"]) == identity(phase1_entry)
        exit_matches[name] = identity(loaded[name]["scheduler_output"]) == identity(phase1_exit)
        input_matches[name] = identity(loaded[name]["latent_entering_step"]) == manifest["trajectories"][name]["canonical_identity"]
    if not all(entry_matches.values()) or not all(exit_matches.values()):
        failure = {"classification": "PHASE1_PHASE2_BOUNDARY_MISMATCH", "entry_matches": entry_matches, "exit_matches": exit_matches}
        atomic_json(root / "phase2/phase1_phase2_boundary_mismatch.json", failure)
        raise GlobalStopError("GLOBAL STOP: PHASE1_PHASE2_BOUNDARY_MISMATCH")

    final_matches: dict[str, dict[str, Any]] = {}
    for name in TRAJECTORIES:
        latent = load_tensor(root, trace["final_latents"][name])
        video = load_tensor(root, trace["final_videos"][name])
        expected_latent, expected_video = _trusted_final_pair(manifest, name)
        final_matches[name] = {
            "final_latent_identity": identity(latent),
            "trusted_final_latent_identity": expected_latent,
            "final_video_identity": identity(video),
            "trusted_final_video_identity": expected_video,
            "matches": identity(latent) == expected_latent and identity(video) == expected_video,
            "artifacts_valid": (
                trace["final_latents"][name]["canonical_identity"] == identity(latent)
                and trace["final_videos"][name]["canonical_identity"] == identity(video)
            ),
        }
    if not all(row["matches"] for row in final_matches.values()):
        failure = {"classification": "PHASE2_TRACE_ALTERS_EXECUTION", "evidence": final_matches}
        atomic_json(root / "phase2/phase2_trace_alters_execution.json", failure)
        raise GlobalStopError("GLOBAL STOP: PHASE2_TRACE_ALTERS_EXECUTION")

    merge_event = _phase2_merge_event(pairwise_rows)
    spread, largest_support_increase = _clean_plus_spread(pairwise_rows)
    cross_event = {
        "merge_boundary": merge_event["first_bit_exact_boundary"],
        "largest_support_increase_boundaries": largest_support_increase["boundaries"],
        "same_boundary": merge_event["first_bit_exact_boundary"] in largest_support_increase["boundaries"],
        "descriptive_only": True,
    }
    finite = all(
        np.isfinite(float(row[key])) for row in pairwise_rows
        for key in ("differing_fraction", "max_abs_diff", "mean_abs_diff", "mse", "l2", "relative_l2")
    )
    relative_paths = all(
        not Path(row["artifact"]["relative_path"]).is_absolute()
        for name in TRAJECTORIES for row in normalized_rows[name]
    ) and all(
        not Path(trace[group][name]["relative_path"]).is_absolute()
        for group in ("final_latents", "final_videos") for name in TRAJECTORIES
    )
    plus_hist_output = metrics(loaded["PLUS1"]["scheduler_output"], loaded["HISTORICAL_PLUS14"]["scheduler_output"])
    final_plus = load_tensor(root, trace["final_latents"]["PLUS1"])
    final_hist = load_tensor(root, trace["final_latents"]["HISTORICAL_PLUS14"])
    final_clean = load_tensor(root, trace["final_latents"]["CLEAN"])
    final_plus_video = load_tensor(root, trace["final_videos"]["PLUS1"])
    final_hist_video = load_tensor(root, trace["final_videos"]["HISTORICAL_PLUS14"])
    final_clean_video = load_tensor(root, trace["final_videos"]["CLEAN"])
    all_artifacts_valid = (
        all(
            evidence["valid"]
            for trajectory in artifact_evidence.values()
            for evidence in trajectory.values()
        )
        and all(row["artifacts_valid"] for row in final_matches.values())
    )
    plus_hist_final_equal = (
        identity(final_plus) == identity(final_hist)
        and identity(final_plus_video) == identity(final_hist_video)
    )
    clean_plus_final_different = (
        identity(final_clean) != identity(final_plus)
        and identity(final_clean_video) != identity(final_plus_video)
    )
    gates = [
        _p2_gate(1, "committed source / clean provenance", not prov.get("source_dirty_entries"), prov),
        _p2_gate(2, "exact selected_step == 10", selected == 10, selected),
        _p2_gate(3, "exact Phase1 input to Phase2 entry mapping", mapping["phase1_entry_boundary"] == "input", mapping),
        _p2_gate(4, "exact Phase1 after_step_001 to Phase2 exit mapping", mapping["phase1_exit_boundary"] == "after_step_001", mapping),
        _p2_gate(5, "exact scheduler timestep", all(timestep_matches(row["timestep"], mapping["selected_scheduler_timestep"], schedule) for rows in normalized_rows.values() for row in rows), mapping),
        _p2_gate(6, "exact CLEAN input identity", input_matches["CLEAN"], entry_matches),
        _p2_gate(7, "exact PLUS1 input identity", input_matches["PLUS1"], entry_matches),
        _p2_gate(8, "exact HISTORICAL input identity", input_matches["HISTORICAL_PLUS14"], entry_matches),
        _p2_gate(9, "all expected real operation boundaries present", all([row["boundary"] for row in rows] == expected for rows in normalized_rows.values()), expected),
        _p2_gate(10, "no unexpected/duplicate boundaries", all(len({row["boundary"] for row in rows}) == len(expected) for rows in normalized_rows.values()), expected),
        _p2_gate(11, "persisted artifact identities valid", all_artifacts_valid, artifact_evidence),
        _p2_gate(12, "pairwise metrics recomputed from artifacts", pairwise_recomputed and len(pairwise_rows) == len(expected) * 3, {"row_count": len(pairwise_rows), "identity_binding_valid": pairwise_recomputed}),
        _p2_gate(13, "CLEAN Phase2 final equals trusted CLEAN", final_matches["CLEAN"]["matches"], final_matches["CLEAN"]),
        _p2_gate(14, "PLUS1 Phase2 final equals trusted PLUS1", final_matches["PLUS1"]["matches"], final_matches["PLUS1"]),
        _p2_gate(15, "HISTORICAL Phase2 final equals trusted historical", final_matches["HISTORICAL_PLUS14"]["matches"], final_matches["HISTORICAL_PLUS14"]),
        _p2_gate(16, "PLUS1 Phase2 final equals HISTORICAL Phase2 final", plus_hist_final_equal, {"latent_equal": identity(final_plus) == identity(final_hist), "video_equal": identity(final_plus_video) == identity(final_hist_video)}),
        _p2_gate(17, "CLEAN Phase2 final differs from PLUS1 Phase2 final", clean_plus_final_different, {"latent_differs": identity(final_clean) != identity(final_plus), "video_differs": identity(final_clean_video) != identity(final_plus_video)}),
        _p2_gate(18, "Phase2 entries equal Phase1 inputs", all(entry_matches.values()), entry_matches),
        _p2_gate(19, "Phase2 scheduler outputs equal Phase1 after_step_001", all(exit_matches.values()), exit_matches),
        _p2_gate(20, "PLUS1 scheduler_output equals HISTORICAL scheduler_output", plus_hist_output["bit_exact"], plus_hist_output),
        _p2_gate(21, "no NaN/Inf", finite, None),
        _p2_gate(22, "relocation/path resolution valid", relative_paths, None),
        _p2_gate(23, "Phase3 not automatically triggered", config["phase3"]["auto_expand"] is False, config["phase3"]),
        _p2_gate(24, "no forbidden expansion modes", tuple(config["allowed_modes"]) == MODES and "fp32-search" not in MODES, list(MODES)),
        _p2_gate(25, "provenance/config/manifest/selected-step binding valid", trace["provenance_hash"] == prov["provenance_hash"] and trace["manifest_sha256"] == manifest["manifest_sha256"] and trace["selection_mapping"] == mapping, mapping),
    ]
    gate_document = {"gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    atomic_json(root / "phase2/phase2_gates.json", gate_document)
    validate_gate_document(gate_document, PHASE2_REQUIRED_GATES, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {
        "selected_step": selected,
        "selection_mapping": mapping,
        "available_boundaries": expected,
        "unavailable_boundaries": list(PHASE2_UNAVAILABLE_BOUNDARIES),
        "boundary_semantics": PHASE2_BOUNDARY_SEMANTICS,
        "pairwise_rows": pairwise_rows,
        "plus1_historical_exact_merge": merge_event,
        "clean_plus1_spread": spread,
        "largest_support_increase": largest_support_increase,
        "cross_event_observation": cross_event,
        "phase1_phase2_crosscheck": {"entry_matches": entry_matches, "exit_matches": exit_matches},
        "traced_vs_trusted_final_controls": final_matches,
        "gates": gates,
    }


def run_analyze_phase2(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    validate_phase2_config(config)
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    path = root / "phase2" / "phase2_manifest.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: Phase-2 trace manifest is missing")
    result = analyze_phase2_artifacts(root, config, manifest, prov, json.loads(path.read_text()))
    atomic_json(root / "phase2" / "phase2_analysis.json", result)
    return {"mode": "analyze-phase2", "selected_step": result["selected_step"], "pairwise_rows": len(result["pairwise_rows"]), "classification": result["plus1_historical_exact_merge"]["classification"]}


def _phase3_expected_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    generation = config["generation"]
    latent_shape = [
        1,
        16,
        math.ceil(int(generation["num_frames"]) / 4),
        math.ceil(int(generation["height"]) / 8),
        math.ceil(int(generation["width"]) / 8),
    ]
    token_count = latent_shape[2] * (latent_shape[3] // 2) * (latent_shape[4] // 2)
    return {
        "latent": latent_shape,
        "block_hidden": [1, token_count, PHASE3_NUM_HEADS * PHASE3_HEAD_DIM],
    }


def _phase3_artifact_budget(config: dict[str, Any]) -> dict[str, Any]:
    """Frozen, bounded artifact volume: every persisted tensor is enumerated up front."""
    shapes = _phase3_expected_shapes(config)
    hidden_bytes = 2 * math.prod(shapes["block_hidden"])
    latent_bytes = 2 * math.prod(shapes["latent"])
    hidden_boundaries = len(PHASE3_FIXED_BOUNDARIES) - 2  # all but transformer_entry / raw_transformer_output
    per_branch = hidden_boundaries * hidden_bytes + 2 * latent_bytes
    per_trajectory = len(PHASE3_BRANCHES) * per_branch + latent_bytes  # + cfg_combined_output
    return {
        "hidden_tensor_bytes_bf16": hidden_bytes,
        "latent_tensor_bytes_bf16": latent_bytes,
        "tensors_per_branch": len(PHASE3_FIXED_BOUNDARIES),
        "bytes_per_trajectory": per_trajectory,
        "bytes_total_three_trajectories": len(TRAJECTORIES) * per_trajectory,
        "note": "block tensors persisted as raw BF16 bit patterns (lossless, 2 bytes/element); no decoded videos except the three final controls",
    }


def _run_phase3_trace(
    omni: Any,
    config: dict[str, Any],
    source: Any,
    state: np.ndarray,
    name: str,
    root: Path,
) -> dict[str, Any]:
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    selected_step = 10
    local_step = selected_step - int(source.checkpoint_step)
    generation = config["generation"]
    sampling = OmniDiffusionSamplingParams(
        height=int(generation["height"]),
        width=int(generation["width"]),
        num_frames=int(generation["num_frames"]),
        num_inference_steps=int(generation["num_inference_steps"]),
        guidance_scale=float(generation["guidance_scale"]),
        fps=float(generation["fps"]),
        seed=source.seed,
        generator=torch.Generator(device="cpu").manual_seed(source.seed),
    )
    sampling.latents = torch.from_numpy(np.ascontiguousarray(state))
    sampling.step_index = int(source.checkpoint_step)
    sampling.extra_args = {
        "flow_shift": float(config["scheduler"]["flow_shift"]),
        "sample_solver": "euler",
        "trajectory_probe": {
            "artifact_dir": str(root / "phase3" / name.lower() / "final_probe"),
            "request_label": name.lower(),
            "capture_steps": [0, int(generation["num_inference_steps"]) - int(source.checkpoint_step)],
            "save_latents": True,
            "save_decoded": False,
            "save_mp4": False,
        },
        "phase3_block_probe": {
            "artifact_dir": str(root / "phase3" / name.lower() / "block_probe"),
            "request_label": name.lower(),
            "selected_local_step": local_step,
            "selected_absolute_step": selected_step,
        },
    }
    outputs = omni.generate({"prompt": source.prompt}, sampling)
    video, output = v3.normalize_video(outputs)
    probe = output.custom_output.get("phase3_block_probe")
    if not isinstance(probe, dict):
        raise GlobalStopError("GLOBAL STOP: Phase-3 block probe output is missing")
    if (
        int(probe.get("selected_local_step", -1)) != local_step
        or int(probe.get("selected_absolute_step", -1)) != selected_step
        or probe.get("cfg_execution") != "sequential_positive_then_negative"
        or int(probe.get("pipeline_parallel_world_size", -1)) != 1
        or int(probe.get("cfg_parallel_world_size", -1)) != 1
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-3 transformer invocation is not the frozen single-GPU sequential CFG path")
    expected_architecture = phase3_freeze(config)["expected_architecture"]
    branches: dict[str, Any] = {}
    for invocation_index, branch in enumerate(PHASE3_BRANCHES):
        branch_probe = probe.get("branches", {}).get(branch)
        if not isinstance(branch_probe, dict):
            raise GlobalStopError(f"GLOBAL STOP: Phase-3 {branch} branch trace is missing")
        architecture = branch_probe.get("architecture", {})
        if any(architecture.get(key) != value for key, value in expected_architecture.items()):
            raise GlobalStopError("GLOBAL STOP: actual Wan transformer architecture differs from the frozen Phase-3 architecture")
        records = branch_probe.get("records", [])
        if [row.get("boundary") for row in records] != list(PHASE3_FIXED_BOUNDARIES):
            raise GlobalStopError(f"GLOBAL STOP: Phase-3 {branch} block boundary order changed")
        converted = []
        for row in records:
            if row.get("branch") != branch or int(row.get("invocation_index", -1)) != invocation_index:
                raise GlobalStopError("GLOBAL STOP: Phase-3 CFG branch identity/order changed")
            if row.get("runtime_dtype") != EXPECTED_RUNTIME_DTYPE or row.get("artifact_encoding") != "bf16_bits_v1":
                raise GlobalStopError("GLOBAL STOP: Phase-3 runtime dtype or artifact encoding changed")
            converted.append({
                "boundary": row["boundary"],
                "block_index": row.get("block_index"),
                "branch": branch,
                "invocation_index": invocation_index,
                "runtime_dtype": row["runtime_dtype"],
                "shape": [int(item) for item in row["shape"]],
                "artifact": phase3_artifact_record(root, Path(row["artifact_path"]), shape=row["shape"]),
            })
        branches[branch] = {"architecture": architecture, "records": converted}
    cfg = probe.get("cfg_combined_output")
    if not isinstance(cfg, dict) or cfg.get("runtime_dtype") != EXPECTED_RUNTIME_DTYPE:
        raise GlobalStopError("GLOBAL STOP: Phase-3 CFG-combined output is missing or has wrong dtype")
    frozen_scale = phase3_cfg_guidance_scale(config)
    if float64_bit_pattern(cfg.get("guidance_scale", float("nan"))) != float64_bit_pattern(frozen_scale) or cfg.get("cfg_normalize") is not False:
        raise GlobalStopError("GLOBAL STOP: runtime CFG guidance scale/normalization differs from the frozen Phase-3 configuration")
    cfg_record = {
        "boundary": "cfg_combined_output",
        "absolute_step": int(cfg["absolute_step"]),
        "local_step": int(cfg["local_step"]),
        "timestep_bits": float64_bit_pattern(cfg["timestep"]),
        "runtime_dtype": cfg["runtime_dtype"],
        "shape": [int(item) for item in cfg["shape"]],
        "guidance_scale": frozen_scale,
        "guidance_scale_bits": float64_bit_pattern(frozen_scale),
        "cfg_normalize": False,
        "artifact": phase3_artifact_record(root, Path(cfg["artifact_path"]), shape=cfg["shape"]),
    }
    # Fail fast on the GPU host: the persisted raw branch outputs must reconstruct the persisted combined output.
    raw_bits = {}
    for branch in PHASE3_BRANCHES:
        raw_row = next(row for row in branches[branch]["records"] if row["boundary"] == "raw_transformer_output")
        raw_bits[branch] = np.load(root / raw_row["artifact"]["relative_path"], allow_pickle=False).reshape(cfg_record["shape"])
    persisted_cfg_bits = np.load(root / cfg_record["artifact"]["relative_path"], allow_pickle=False).reshape(cfg_record["shape"])
    if not np.array_equal(reconstruct_cfg_combined_bits(raw_bits["positive"], raw_bits["negative"], frozen_scale), persisted_cfg_bits):
        raise GlobalStopError(f"GLOBAL STOP: CFG_OPERAND_RECONSTRUCTION_MISMATCH for {name} at trace time")
    trajectory_probe = output.custom_output.get("trajectory_probe_metadata")
    if not isinstance(trajectory_probe, dict):
        raise GlobalStopError("GLOBAL STOP: Phase-3 final-latent trajectory probe is missing")
    final_step_index = int(generation["num_inference_steps"]) - int(source.checkpoint_step)
    final_rows = [row for row in trajectory_probe.get("records", []) if int(row.get("step_index", -1)) == final_step_index]
    if len(final_rows) != 1 or not final_rows[0].get("latent_path"):
        raise GlobalStopError("GLOBAL STOP: Phase-3 final latent is missing or ambiguous")
    final_latent = torch.load(Path(final_rows[0]["latent_path"]), map_location="cpu").detach().cpu().float().numpy()
    return {
        "trajectory": name,
        "branches": branches,
        "cfg_combined_output": cfg_record,
        "final_latent": save_tensor(root, f"phase3/artifacts/{name}/final_latent.npy", final_latent),
        "final_video": save_tensor(root, f"phase3/artifacts/{name}/final_video.npy", video, runtime_semantics="uint8 decoded video"),
    }


def run_phase3(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    validate_phase3_config(config)
    manifest, prov = require_cpu(root, config_path, config)
    require_preflight(root, manifest, prov)
    require_committed_source(prov)
    data = derive_anchor(config)
    omni = _build_omni(config, args)
    try:
        results = {
            name: _run_phase3_trace(omni, config, data["source"], state, name, root)
            for name, state in {
                "CLEAN": data["clean"],
                "PLUS1": data["plus1"],
                "HISTORICAL_PLUS14": data["historical"],
            }.items()
        }
        document = {
            "provenance_hash": prov["provenance_hash"],
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_step": 10,
            "selected_scheduler_timestep_bits": float64_bit_pattern(single_flip.scheduler_timesteps_numpy(config)[10]),
            "phase3_freeze": manifest["phase3_freeze"],
            "trajectories": results,
        }
        atomic_json(root / "phase3" / "phase3_manifest.json", document)
        return {"mode": "phase3", "trajectory_count": len(results), "selected_step": 10}
    finally:
        single_flip._shutdown(omni)


def _load_trusted_phase2_boundary(root: Path, manifest: dict[str, Any], name: str, which: str) -> np.ndarray:
    trusted = manifest["trusted_phase2"]
    source_root = (REPO_ROOT / trusted["source_root_relative_path"]).resolve()
    path = source_root / "phase2/phase2_manifest.json"
    if not path.exists() or sha256_file(path) != trusted["phase2_manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-2 manifest is missing or changed")
    return load_tensor(source_root, trusted["trajectories"][name][f"{which}_artifact"])


def _p3_gate(number: int, description: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return gate(f"P3-G{number}", passed, evidence, required=True)


def _persistent_merge_event(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [bool(row["bit_exact"]) for row in rows]
    transient = [
        rows[index]["boundary"]
        for index, is_exact in enumerate(exact)
        if is_exact and not all(exact[index:])
    ]
    persistent = next(
        (index for index, is_exact in enumerate(exact) if is_exact and all(exact[index:])),
        None,
    )
    if persistent is None:
        classification = "TRANSIENT_EXACT_EQUALITY" if transient else "NO_INTERNAL_EXACT_MERGE"
        boundary = None
    else:
        boundary = rows[persistent]["boundary"]
        if boundary == "transformer_entry":
            classification = "MERGED_AT_TRANSFORMER_ENTRY"
        elif boundary == "pre_block_hidden_state":
            classification = "MERGED_BEFORE_BLOCK_000"
        elif boundary.startswith("after_block_"):
            classification = f"MERGED_AFTER_BLOCK_{int(boundary.rsplit('_', 1)[1]):03d}"
        elif boundary == "transformer_pre_output":
            classification = "MERGED_AT_TRANSFORMER_PRE_OUTPUT"
        elif boundary == "raw_transformer_output":
            classification = "MERGED_AT_RAW_TRANSFORMER_OUTPUT"
        else:
            raise GlobalStopError(f"GLOBAL STOP: unavailable Phase-3 merge boundary: {boundary}")
    return {
        "classification": classification,
        "first_persistent_exact_boundary": boundary,
        "transient_exact_boundaries": transient,
        "exact_boundaries": [row["boundary"] for row in rows if row["bit_exact"]],
    }


def _phase3_spread(
    pairwise_rows: list[dict[str, Any]],
    branch: str,
    same_semantics: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = {
        row["boundary"]: row
        for row in pairwise_rows
        if row["branch"] == branch and row["pair"] == "CLEAN_VS_PLUS1"
    }
    progression: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for boundary in PHASE3_FIXED_BOUNDARIES:
        row = source[boundary]
        item = {key: row[key] for key in (
            "branch", "boundary", "differing_element_count", "differing_fraction",
            "max_abs_diff", "mean_abs_diff", "mse", "l2", "relative_l2",
            "lhs_canonical_identity", "rhs_canonical_identity",
        )}
        if boundary not in same_semantics or previous is None:
            item.update({
                "delta_differing_element_count": None,
                "ratio_differing_element_count": None,
                "delta_relative_l2": None,
                "ratio_relative_l2": None,
            })
        else:
            item.update({
                "delta_differing_element_count": row["differing_element_count"] - previous["differing_element_count"],
                "ratio_differing_element_count": row["differing_element_count"] / previous["differing_element_count"] if previous["differing_element_count"] else None,
                "delta_relative_l2": row["relative_l2"] - previous["relative_l2"],
                "ratio_relative_l2": row["relative_l2"] / previous["relative_l2"] if previous["relative_l2"] else None,
            })
        progression.append(item)
        previous = row if boundary in same_semantics else None
    comparable = [row for row in progression if row["delta_differing_element_count"] is not None]
    largest = max(row["delta_differing_element_count"] for row in comparable)
    return progression, {
        "statistic": "largest absolute increase in differing_element_count between consecutive comparable block-hidden boundaries",
        "largest_increase": largest,
        "boundaries": [row["boundary"] for row in comparable if row["delta_differing_element_count"] == largest],
        "comparable_boundaries": same_semantics,
        "descriptive_only": True,
    }


def _binding_contains_float_reduction(value: Any, key: str = "") -> bool:
    forbidden = ("relative_l2", "mse", "mean_abs", "max_abs", "l2")
    if any(item in key.lower() for item in forbidden):
        return True
    if isinstance(value, dict):
        return any(_binding_contains_float_reduction(item, str(name)) for name, item in value.items())
    if isinstance(value, list):
        return any(_binding_contains_float_reduction(item, key) for item in value)
    return False


def _phase3_cfg_classification(raw_exact: dict[str, bool]) -> str:
    """Where the PLUS1/HISTORICAL exact merge sits relative to CFG combination.

    Uses only exact equality of the persisted raw branch outputs and of the
    persisted CFG-combined output (the latter is required to be exact).
    """
    positive, negative = bool(raw_exact["positive"]), bool(raw_exact["negative"])
    if positive and negative:
        return "MERGED_WITHIN_BOTH_TRANSFORMER_BRANCHES"
    if positive and not negative:
        return "MERGED_WITHIN_POSITIVE_BRANCH_ONLY_CFG_COMBINATION_EXACT"
    if negative and not positive:
        return "MERGED_WITHIN_NEGATIVE_BRANCH_ONLY_CFG_COMBINATION_EXACT"
    return "MERGED_AT_CFG_COMBINATION"


def analyze_phase3_artifacts(
    root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    prov: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Recompute every Phase-3 quantity from persisted artifacts.

    Memory discipline: block-hidden tensors are ~144 MB in BF16 and ~288 MB
    widened; the three trajectories of one boundary are held at a time and
    released before the next boundary. Only identities, exact flags and
    descriptive metrics are retained.
    """
    validate_phase3_config(config)
    require_committed_source(prov)
    freeze = phase3_freeze(config)
    if (
        trace.get("provenance_hash") != prov["provenance_hash"]
        or trace.get("manifest_sha256") != manifest["manifest_sha256"]
        or trace.get("selected_step") != 10
        or trace.get("selected_scheduler_timestep_bits") != freeze["selected_scheduler_timestep_bits"]
        or trace.get("phase3_freeze") != freeze
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-3 provenance/target binding mismatch")
    trajectories = trace.get("trajectories")
    if not isinstance(trajectories, dict) or set(trajectories) != set(TRAJECTORIES):
        raise GlobalStopError("GLOBAL STOP: Phase-3 trajectory set is incomplete or unexpected")

    expected_shapes = freeze["expected_shapes"]
    expected_boundaries = list(PHASE3_FIXED_BOUNDARIES)
    same_semantics = freeze["same_semantics_support_boundaries"]
    expected_architecture = freeze["expected_architecture"]
    normalized: dict[str, Any] = {}
    architecture_valid = True
    branch_valid = True
    relative_paths = True
    # ---- structural validation of every record before any tensor is loaded
    for name in TRAJECTORIES:
        result = trajectories[name]
        if set(result.get("branches", {})) != set(PHASE3_BRANCHES):
            raise GlobalStopError("GLOBAL STOP: Phase-3 CFG branch set changed")
        normalized[name] = {"branches": {}}
        for invocation_index, branch in enumerate(PHASE3_BRANCHES):
            branch_result = result["branches"][branch]
            architecture = branch_result.get("architecture", {})
            architecture_valid &= all(architecture.get(key) == value for key, value in expected_architecture.items())
            rows = branch_result.get("records", [])
            names = [row.get("boundary") for row in rows]
            if set(names) != set(expected_boundaries) or len(names) != len(set(names)):
                raise GlobalStopError("GLOBAL STOP: Phase-3 block boundary set is missing, duplicate, or unexpected")
            by_boundary = {row["boundary"]: row for row in rows}
            ordered = [by_boundary[boundary] for boundary in expected_boundaries]
            for boundary, row in zip(expected_boundaries, ordered, strict=True):
                expected_block_index = int(boundary.rsplit("_", 1)[1]) if boundary.startswith("after_block_") else None
                branch_valid &= row.get("branch") == branch and row.get("invocation_index") == invocation_index
                if row.get("block_index") != expected_block_index:
                    raise GlobalStopError("GLOBAL STOP: Phase-3 block execution order/index changed")
                relative_paths &= not Path(row["artifact"]["relative_path"]).is_absolute()
            normalized[name]["branches"][branch] = {"architecture": architecture, "records": ordered}
        cfg_row = result.get("cfg_combined_output", {})
        if (
            cfg_row.get("boundary") != "cfg_combined_output"
            or cfg_row.get("absolute_step") != 10
            or cfg_row.get("local_step") != 0
            or cfg_row.get("timestep_bits") != freeze["selected_scheduler_timestep_bits"]
            or cfg_row.get("runtime_dtype") != EXPECTED_RUNTIME_DTYPE
            or cfg_row.get("shape") != expected_shapes["latent"]
        ):
            raise GlobalStopError("GLOBAL STOP: Phase-3 CFG output metadata changed")
        relative_paths &= not Path(cfg_row["artifact"]["relative_path"]).is_absolute()
        normalized[name]["cfg_combined_output"] = cfg_row
        relative_paths &= (
            not Path(trajectories[name]["final_latent"]["relative_path"]).is_absolute()
            and not Path(trajectories[name]["final_video"]["relative_path"]).is_absolute()
        )

    # ---- trusted Phase-2 boundaries and traced finals (small tensors)
    trusted_entry_identity = {name: identity(_load_trusted_phase2_boundary(root, manifest, name, "entry")) for name in TRAJECTORIES}
    trusted_exit_identity = {name: identity(_load_trusted_phase2_boundary(root, manifest, name, "exit")) for name in TRAJECTORIES}
    cfg_loaded = {name: load_phase3_artifact(root, normalized[name]["cfg_combined_output"]["artifact"]) for name in TRAJECTORIES}
    phase2_exit_matches = {name: identity(cfg_loaded[name]) == trusted_exit_identity[name] for name in TRAJECTORIES}
    final_matches: dict[str, dict[str, bool]] = {}
    for name in TRAJECTORIES:
        final_latent = load_tensor(root, trajectories[name]["final_latent"])
        final_video = load_tensor(root, trajectories[name]["final_video"])
        trusted_latent, trusted_video = _trusted_final_pair(manifest, name)
        final_matches[name] = {"latent": identity(final_latent) == trusted_latent, "video": identity(final_video) == trusted_video}

    # ---- streamed pairwise recomputation, one boundary (three trajectories) at a time
    pairings = (("CLEAN", "PLUS1"), ("CLEAN", "HISTORICAL_PLUS14"), ("PLUS1", "HISTORICAL_PLUS14"))
    pairwise_rows: list[dict[str, Any]] = []
    artifact_valid = True
    shape_dtype_valid = True
    phase2_entry_matches: dict[str, dict[str, bool]] = {name: {} for name in TRAJECTORIES}
    for branch in PHASE3_BRANCHES:
        for boundary in expected_boundaries:
            expected_shape = expected_shapes["latent"] if boundary in ("transformer_entry", "raw_transformer_output") else expected_shapes["block_hidden"]
            loaded: dict[str, np.ndarray] = {}
            for name in TRAJECTORIES:
                row = normalized[name]["branches"][branch]["records"][expected_boundaries.index(boundary)]
                value = load_phase3_artifact(root, row["artifact"])
                shape_dtype_valid &= (
                    row.get("runtime_dtype") == EXPECTED_RUNTIME_DTYPE
                    and row.get("shape") == expected_shape
                    and list(value.shape) == expected_shape
                )
                artifact_valid &= row["artifact"]["comparison_canonical_identity"] == identity(value)
                if boundary == "transformer_entry":
                    phase2_entry_matches[name][branch] = identity(value) == trusted_entry_identity[name]
                loaded[name] = value
            for left, right in pairings:
                pairwise_rows.append({"branch": branch, "boundary": boundary, "pair": f"{left}_VS_{right}", **metrics(loaded[left], loaded[right])})
            del loaded
    cfg_pairwise_rows = [
        {"branch": "cfg_combined", "boundary": "cfg_combined_output", "pair": f"{left}_VS_{right}", **metrics(cfg_loaded[left], cfg_loaded[right])}
        for left, right in pairings
    ]
    del cfg_loaded

    if not all(all(value.values()) for value in phase2_entry_matches.values()) or not all(phase2_exit_matches.values()):
        failure = {"classification": "PHASE2_PHASE3_BOUNDARY_MISMATCH", "entry": phase2_entry_matches, "exit": phase2_exit_matches}
        atomic_json(root / "phase3/phase2_phase3_boundary_mismatch.json", failure)
        raise GlobalStopError("GLOBAL STOP: PHASE2_PHASE3_BOUNDARY_MISMATCH")
    if not all(all(value.values()) for value in final_matches.values()):
        failure = {"classification": "PHASE3_TRACE_ALTERS_EXECUTION", "final_matches": final_matches}
        atomic_json(root / "phase3/phase3_trace_alters_execution.json", failure)
        raise GlobalStopError("GLOBAL STOP: PHASE3_TRACE_ALTERS_EXECUTION")

    def row_of(branch: str, boundary: str, pair: str) -> dict[str, Any]:
        return next(row for row in pairwise_rows if row["branch"] == branch and row["boundary"] == boundary and row["pair"] == pair)

    entry_one_difference = {branch: row_of(branch, "transformer_entry", "CLEAN_VS_PLUS1")["differing_element_count"] == 1 for branch in PHASE3_BRANCHES}
    branch_events: dict[str, Any] = {}
    propagation: dict[str, Any] = {}
    largest_support: dict[str, Any] = {}
    for branch in PHASE3_BRANCHES:
        plus_hist = [row for row in pairwise_rows if row["branch"] == branch and row["pair"] == "PLUS1_VS_HISTORICAL_PLUS14"]
        branch_events[branch] = _persistent_merge_event(plus_hist)
        propagation[branch], largest_support[branch] = _phase3_spread(pairwise_rows, branch, same_semantics)
    cfg_plus_hist = next(row for row in cfg_pairwise_rows if row["pair"] == "PLUS1_VS_HISTORICAL_PLUS14")
    raw_exact = {branch: bool(row_of(branch, "raw_transformer_output", "PLUS1_VS_HISTORICAL_PLUS14")["bit_exact"]) for branch in PHASE3_BRANCHES}
    if not cfg_plus_hist["bit_exact"]:
        raise GlobalStopError("GLOBAL STOP: Phase-3 CFG output contradicts trusted Phase-2 exact merge")
    # ---- M1 hard gate: persisted raw branch outputs must reconstruct the persisted combined output bit-exactly.
    frozen_scale = phase3_cfg_guidance_scale(config)
    reconstruction: dict[str, Any] = {}
    for name in TRAJECTORIES:
        cfg_row = normalized[name]["cfg_combined_output"]
        if (
            float64_bit_pattern(cfg_row.get("guidance_scale", float("nan"))) != freeze["cfg_guidance_scale_bits"]
            or cfg_row.get("guidance_scale_bits") != freeze["cfg_guidance_scale_bits"]
            or cfg_row.get("cfg_normalize") is not False
        ):
            raise GlobalStopError(f"GLOBAL STOP: CFG_OPERAND_RECONSTRUCTION_MISMATCH: guidance scale/normalization binding differs for {name}")
        raw_bits = {}
        for branch in PHASE3_BRANCHES:
            row = normalized[name]["branches"][branch]["records"][expected_boundaries.index("raw_transformer_output")]
            widened = load_phase3_artifact(root, row["artifact"])
            raw_bits[branch] = single_flip.float32_to_bf16_bits(widened).reshape(widened.shape)
        cfg_widened = load_phase3_artifact(root, cfg_row["artifact"])
        persisted_bits = single_flip.float32_to_bf16_bits(cfg_widened).reshape(cfg_widened.shape)
        reconstructed_bits = reconstruct_cfg_combined_bits(raw_bits["positive"], raw_bits["negative"], frozen_scale)
        swapped_bits = reconstruct_cfg_combined_bits(raw_bits["negative"], raw_bits["positive"], frozen_scale)
        reconstruction[name] = {
            "bit_exact": bool(np.array_equal(reconstructed_bits, persisted_bits)),
            "differing_element_count": int(np.count_nonzero(reconstructed_bits != persisted_bits)),
            "swapped_operands_bit_exact": bool(np.array_equal(swapped_bits, persisted_bits)),
            "positive_equals_negative": bool(np.array_equal(raw_bits["positive"], raw_bits["negative"])),
            "reconstructed_identity": phase3_runtime_identity(reconstructed_bits, list(reconstructed_bits.shape)),
            "persisted_identity": phase3_runtime_identity(persisted_bits, list(persisted_bits.shape)),
            "guidance_scale_bits": freeze["cfg_guidance_scale_bits"],
            "rule": CFG_RECONSTRUCTION_RULE,
        }
        del raw_bits, cfg_widened
    if not all(row["bit_exact"] for row in reconstruction.values()):
        failure = {"classification": "CFG_OPERAND_RECONSTRUCTION_MISMATCH", "reconstruction": reconstruction}
        atomic_json(root / "phase3/cfg_operand_reconstruction_mismatch.json", failure)
        raise GlobalStopError(f"GLOBAL STOP: CFG_OPERAND_RECONSTRUCTION_MISMATCH: {[n for n, r in reconstruction.items() if not r['bit_exact']]}")
    # Branch semantics are established by the arithmetic relationship, not by labels: a swap must not also reconstruct
    # unless the two raw outputs are identical (in which case the branch distinction is vacuous and reported as such).
    branch_semantics_bound = all(r["positive_equals_negative"] or not r["swapped_operands_bit_exact"] for r in reconstruction.values())
    cfg_classification = _phase3_cfg_classification(raw_exact)
    cross_event = {
        branch: {
            "first_persistent_merge_boundary": branch_events[branch]["first_persistent_exact_boundary"],
            "largest_support_increase_boundaries": largest_support[branch]["boundaries"],
            "same_boundary": branch_events[branch]["first_persistent_exact_boundary"] in largest_support[branch]["boundaries"],
            "descriptive_only": True,
        }
        for branch in PHASE3_BRANCHES
    }
    finite = all(
        np.isfinite(float(row[key]))
        for row in [*pairwise_rows, *cfg_pairwise_rows]
        for key in ("differing_fraction", "max_abs_diff", "mean_abs_diff", "mse", "l2", "relative_l2")
    )
    pairwise_valid = len(pairwise_rows) == len(PHASE3_BRANCHES) * len(expected_boundaries) * len(pairings)
    no_float_binding = not _binding_contains_float_reduction(manifest["trusted_phase1"]) and not _binding_contains_float_reduction(manifest["trusted_phase2"])
    gates = [
        _p3_gate(1, "committed source / clean provenance", not prov.get("source_dirty_entries"), prov),
        _p3_gate(2, "selected_step exactly 10", trace["selected_step"] == 10, trace["selected_step"]),
        _p3_gate(3, "selected transformer invocation uniquely identified", branch_valid and architecture_valid and set(PHASE3_BRANCHES) == {"positive", "negative"}, {"branches": list(PHASE3_BRANCHES), "cfg": freeze["cfg_execution"], "architecture": expected_architecture}),
        _p3_gate(4, "exact scheduler timestep", trace["selected_scheduler_timestep_bits"] == freeze["selected_scheduler_timestep_bits"], trace["selected_scheduler_timestep_bits"]),
        *[_p3_gate(5 + index, f"Phase3 {name} entry equals Phase2 transformer_input", all(phase2_entry_matches[name].values()), phase2_entry_matches[name]) for index, name in enumerate(TRAJECTORIES)],
        *[_p3_gate(8 + index, f"Phase3 {name} exit equals Phase2 guidance_combined_output", phase2_exit_matches[name], phase2_exit_matches[name]) for index, name in enumerate(TRAJECTORIES)],
        _p3_gate(11, "exact block count/order matches frozen manifest", architecture_valid, expected_architecture),
        _p3_gate(12, "no missing expected block boundary", True, expected_boundaries),
        _p3_gate(13, "no duplicate block boundary", True, expected_boundaries),
        _p3_gate(14, "no unexpected block boundary", True, expected_boundaries),
        _p3_gate(15, "CFG branch identity valid", branch_valid, list(PHASE3_BRANCHES)),
        _p3_gate(16, "tensor shape/dtype semantics valid", shape_dtype_valid, expected_shapes),
        _p3_gate(17, "canonical artifact identities valid", artifact_valid, "all Phase-3 artifacts independently loaded and re-identified"),
        _p3_gate(18, "pairwise metrics recomputed from artifacts", pairwise_valid, {"rows": len(pairwise_rows)}),
        *[_p3_gate(19 + index, f"{name} traced final equals trusted final", all(final_matches[name].values()), final_matches[name]) for index, name in enumerate(TRAJECTORIES)],
        _p3_gate(22, "PLUS1/HIST Phase3 exit bit-exact", cfg_plus_hist["bit_exact"], cfg_plus_hist),
        _p3_gate(23, "CLEAN/PLUS1 Phase3 entry exactly one differing coordinate", all(entry_one_difference.values()), entry_one_difference),
        _p3_gate(24, "no NaN/Inf", finite, None),
        _p3_gate(25, "relocation/path resolution valid", relative_paths, None),
        _p3_gate(26, "no float-derived metric in provenance binding", no_float_binding, {"trusted_phase1_keys": sorted(manifest["trusted_phase1"]), "trusted_phase2_keys": sorted(manifest["trusted_phase2"])}),
        _p3_gate(27, "Phase4 disabled", not freeze["phase4_enabled"], freeze["phase4_enabled"]),
        _p3_gate(28, "no forbidden expansion mode", not freeze["auto_expand"] and tuple(config["allowed_modes"]) == MODES, list(MODES)),
        _p3_gate(29, "Phase1-producing commit frozen", freeze["phase1_producing_commit"] == PHASE3_PHASE1_COMMIT, freeze["phase1_producing_commit"]),
        _p3_gate(30, "Phase2-producing commit frozen", freeze["phase2_producing_commit"] == PHASE3_PHASE2_COMMIT, freeze["phase2_producing_commit"]),
        _p3_gate(31, "raw CFG operands reconstruct combined output bit-exactly (per trajectory, branch-ordered)", all(row["bit_exact"] for row in reconstruction.values()) and branch_semantics_bound, reconstruction),
    ]
    gate_document = {"gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    atomic_json(root / "phase3/phase3_gates.json", gate_document)
    validate_gate_document(gate_document, PHASE3_REQUIRED_GATES, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {
        "selected_step": 10,
        "selected_scheduler_timestep_bits": freeze["selected_scheduler_timestep_bits"],
        "architecture": freeze["architecture"],
        "observed_architecture": {name: {branch: normalized[name]["branches"][branch]["architecture"] for branch in PHASE3_BRANCHES} for name in TRAJECTORIES},
        "cfg_execution": freeze["cfg_execution"],
        "branch_events": branch_events,
        "cfg_event": {"classification": cfg_classification, "raw_branch_outputs_bit_exact": raw_exact, "cfg_combined_bit_exact": True},
        "cfg_operand_reconstruction": reconstruction,
        "intermediate_authentication_limit": freeze["intermediate_authentication_limit"],
        "pairwise_rows": pairwise_rows,
        "cfg_pairwise_rows": cfg_pairwise_rows,
        "clean_plus1_propagation": propagation,
        "largest_support_increase": largest_support,
        "cross_event_observation": cross_event,
        "phase2_phase3_crosscheck": {"entry": phase2_entry_matches, "exit": phase2_exit_matches},
        "traced_vs_trusted_final_controls": final_matches,
        "gates": gates,
    }


def run_analyze_phase3(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config)
    require_preflight(root, manifest, prov)
    path = root / "phase3/phase3_manifest.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: Phase-3 trace manifest is missing")
    result = analyze_phase3_artifacts(root, config, manifest, prov, json.loads(path.read_text()))
    atomic_json(root / "phase3/phase3_analysis.json", result)
    return {
        "mode": "analyze-phase3",
        "selected_step": 10,
        "branch_events": result["branch_events"],
        "cfg_event": result["cfg_event"],
    }


def unavailable_gpu(mode: str) -> None:
    raise GlobalStopError(f"GLOBAL STOP: {mode} requires optional instrumentation that is intentionally not auto-enabled.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_bf16_first_divergence_localization_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_bf16_first_divergence_localization"))
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    if args.mode == "cpu":
        print(json.dumps(run_cpu(config, args.config, args.output_dir), indent=2, sort_keys=True)); return
    if args.mode == "preflight":
        print(json.dumps(run_preflight(config, args.config, args.output_dir, args), indent=2, sort_keys=True)); return
    if args.mode == "phase1":
        print(json.dumps(run_phase1(config, args.config, args.output_dir, args), indent=2, sort_keys=True)); return
    if args.mode == "analyze-phase1":
        print(json.dumps(run_analyze_phase1(config, args.config, args.output_dir), indent=2, sort_keys=True)); return
    require_cpu(args.output_dir, args.config, config)
    if args.mode == "phase2":
        print(json.dumps(run_phase2(config, args.config, args.output_dir, args), indent=2, sort_keys=True)); return
    if args.mode == "analyze-phase2":
        print(json.dumps(run_analyze_phase2(config, args.config, args.output_dir), indent=2, sort_keys=True)); return
    if args.mode == "phase3":
        print(json.dumps(run_phase3(config, args.config, args.output_dir, args), indent=2, sort_keys=True)); return
    if args.mode == "analyze-phase3":
        print(json.dumps(run_analyze_phase3(config, args.config, args.output_dir), indent=2, sort_keys=True)); return
    unavailable_gpu(args.mode)


if __name__ == "__main__":
    main()
