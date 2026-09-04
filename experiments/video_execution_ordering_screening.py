#!/usr/bin/env python3
"""Execution-Ordering Screening (preregistered, fail-closed).

Question
--------
Does reconstruction-error ordering reliably predict one-step execution-error ordering?

For every trusted v3 CLEAN runtime state (12 prompts x steps {10, 20, 30}) ten serialized
representation candidates are constructed on CPU (BF16, FP16, INT8 x4 granularities,
INT4 x4 granularities), decoded back, cast to the BF16 runtime encoding, and executed for
exactly ONE scheduler update on the GPU. The primary comparison is

    X = relative L2 (runtime-BF16 candidate state  vs runtime-BF16 CLEAN state)
    Y = relative L2 (candidate next-step latent    vs CLEAN next-step latent)

restricted to same-tier candidate pairs whose measured X values lie within a factor of two.
A "meaningful reversal" is X(A) < X(B) and Y(A) >= 2 Y(B) and Y(A) - Y(B) >= 1e-4.
CONTINUE iff meaningful reversals occur in >= 4 distinct prompts AND >= 2 distinct steps
AND >= 2 distinct candidate-pair types; otherwise NO_GO.

The one-step execution path is a new pipeline option (``execution_step_limit``); it is
validated before any screening run by (a) reproducing the Phase-1 ``after_step_001``
artifacts for the recovery_008 CLEAN/PLUS1/HISTORICAL_PLUS14 inputs bit-exactly and
(b) reproducing every trusted v3 state k from state k-1 bit-exactly. Any failure is a
GLOBAL STOP: SCREENING INVALID.

No oracle, no storage headroom, no final-video quality, no localization, no new
perturbation families. Only CPU modes may be run by the auditor; GPU modes are run by the
experimenter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_bf16_first_divergence_localization as loc  # noqa: E402
from experiments import video_bf16_single_flip_killtest as single_flip  # noqa: E402
from experiments import video_runtime_error_shape_killtest as base  # noqa: E402
from experiments import video_runtime_state_discovery as v3  # noqa: E402

EXPERIMENT_VERSION = "video-execution-ordering-screening-v1"
MODES = ("cpu", "validate", "screening", "analyze")
GPU_MODES = ("validate", "screening")
EXPECTED_RUNTIME_DTYPE = "torch.bfloat16"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_SHAPE = (1, 16, 9, 60, 104)
CHECKPOINT_STEPS = (10, 20, 30)
PRIMARY_TIERS = ("int8", "int4")
CONTROL_TIER = "control"
CANDIDATE_NAMES = (
    "bf16",
    "fp16",
    "int8_per_tensor",
    "int8_group4",
    "int8_group8",
    "int8_group16",
    "int4_per_tensor",
    "int4_group4",
    "int4_group8",
    "int4_group16",
)
X_RATIO_MAX = 2.0
Y_RATIO_MIN = 2.0
Y_ABS_FLOOR = 1e-4
MIN_DISTINCT_PROMPTS = 4
MIN_DISTINCT_STEPS = 2
MIN_DISTINCT_PAIR_TYPES = 2
# Cross-host float64 summation noise observed on identical artifacts (Phase 3: <= 4e-14 relative).
# Used ONLY to compare a recomputed descriptive float against its persisted copy; never as a
# scientific tolerance. Every decision quantity is recomputed on the analysing host.
FLOAT_RECOMPUTE_REL_TOL = 1e-9
IDENTITY_FORMAT = single_flip.TENSOR_IDENTITY_FORMAT
ANCHOR_TRAJECTORIES = ("CLEAN", "PLUS1", "HISTORICAL_PLUS14")

CPU_REQUIRED_GATES = tuple(f"S-C{index}" for index in range(1, 13))
VALIDATE_REQUIRED_GATES = tuple(f"S-V{index}" for index in range(1, 8))
ANALYZE_REQUIRED_GATES = tuple(f"S-A{index}" for index in range(1, 17))

GlobalStopError = loc.GlobalStopError
canonical_json = loc.canonical_json
sha256_bytes = loc.sha256_bytes
sha256_file = loc.sha256_file
identity = loc.identity
gate = loc.gate
validate_gate_document = loc.validate_gate_document
atomic_json = loc.atomic_json


# --------------------------------------------------------------------------------------
# configuration / provenance
# --------------------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("experiment_version") != EXPERIMENT_VERSION or tuple(config.get("allowed_modes", ())) != MODES:
        raise GlobalStopError("GLOBAL STOP: experiment version or modes changed")
    if tuple(config.get("checkpoint_steps", ())) != CHECKPOINT_STEPS:
        raise GlobalStopError("GLOBAL STOP: checkpoint steps are frozen to (10, 20, 30)")
    names = tuple(row["name"] for row in config.get("candidates", ()))
    if names != CANDIDATE_NAMES:
        raise GlobalStopError("GLOBAL STOP: candidate set is frozen")
    for row in config["candidates"]:
        expected_tier = CONTROL_TIER if row["name"] in ("bf16", "fp16") else row["name"].split("_", 1)[0]
        if row["tier"] != expected_tier:
            raise GlobalStopError("GLOBAL STOP: candidate tier labelling changed")
        if row["tier"] in PRIMARY_TIERS and (row["bits"] != int(row["tier"][3:]) or row["group_count"] not in (1, 4, 8, 16)):
            raise GlobalStopError("GLOBAL STOP: quantizer bits/group configuration changed")
    primary = config["primary"]
    if (
        tuple(primary["primary_tiers"]) != PRIMARY_TIERS
        or primary["pair_eligibility"] != {"same_tier": True, "x_ratio_max": X_RATIO_MAX, "both_x_positive": True}
        or primary["meaningful_reversal"]["y_ratio_min"] != Y_RATIO_MIN
        or primary["meaningful_reversal"]["y_abs_floor"] != Y_ABS_FLOOR
        or primary["meaningful_reversal"]["x_strictly_less"] is not True
        or primary["continue_rule"]
        != {
            "min_distinct_prompts": MIN_DISTINCT_PROMPTS,
            "min_distinct_steps": MIN_DISTINCT_STEPS,
            "min_distinct_pair_types": MIN_DISTINCT_PAIR_TYPES,
        }
    ):
        raise GlobalStopError("GLOBAL STOP: preregistered reversal/decision rule changed")
    generation = config["generation"]
    if int(generation["num_inference_steps"]) != 40 or config["scheduler"]["sample_solver"] != "euler":
        raise GlobalStopError("GLOBAL STOP: generation schedule is frozen to 40 Euler steps")
    if config["trusted_v3"]["runtime_dtype"] != EXPECTED_RUNTIME_DTYPE or config["trusted_v3"]["scheduler"] != EXPECTED_SCHEDULER:
        raise GlobalStopError("GLOBAL STOP: trusted v3 runtime semantics changed")
    if len(config["trusted_v3"]["trajectory_ids"]) != 12 or sorted(config["trusted_v3"]["required_states"]) != [9, 10, 19, 20, 29, 30]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 prompt set / required states changed")
    return config


PROVENANCE_FILES = (
    "experiments/video_execution_ordering_screening.py",
    "experiments/video_execution_ordering_screening_config.yaml",
    "experiments/run_video_execution_ordering_screening_gpu0.sh",
    "tests/diffusion/test_video_execution_ordering_screening.py",
    "experiments/video_bf16_first_divergence_localization.py",
    "experiments/video_bf16_single_flip_killtest.py",
    "experiments/video_runtime_error_shape_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)


def provenance(config_path: Path) -> dict[str, Any]:
    hashes = {item: sha256_file(REPO_ROOT / item) for item in PROVENANCE_FILES}
    record = {
        "config_sha256": sha256_file(config_path),
        "experiment_script_sha256": hashes["experiments/video_execution_ordering_screening.py"],
        "pipeline_sha256": hashes["vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"],
        "files": hashes,
        "identity_format": IDENTITY_FORMAT,
        **loc.git_state(),
    }
    return {**record, "provenance_hash": sha256_bytes(canonical_json(record))}


def require_committed_source(prov: dict[str, Any]) -> None:
    loc.require_committed_source(prov)


def relative_path(root: Path, path: Path) -> str:
    return loc.relative_path(root, path)


def save_array(root: Path, relative: str, array: np.ndarray) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    value = np.load(path, allow_pickle=False)
    if value.shape != array.shape or value.dtype != array.dtype or not np.array_equal(value, array):
        raise GlobalStopError(f"GLOBAL STOP: artifact retention failed for {relative}")
    return {
        "relative_path": relative,
        "file_sha256": sha256_file(path),
        "canonical_identity": identity(value),
        "shape": [int(item) for item in value.shape],
        "storage_dtype": np.dtype(value.dtype).newbyteorder("<").str,
        "nbytes": int(value.nbytes),
    }


def load_bound_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / record["relative_path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise GlobalStopError("GLOBAL STOP: artifact path escapes the result root")
    if sha256_file(path) != record["file_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: artifact file hash mismatch for {record['relative_path']}")
    value = np.load(path, allow_pickle=False)
    if identity(value) != record["canonical_identity"] or list(value.shape) != list(record["shape"]):
        raise GlobalStopError(f"GLOBAL STOP: artifact identity mismatch for {record['relative_path']}")
    return value


# --------------------------------------------------------------------------------------
# trusted sources
# --------------------------------------------------------------------------------------
def trusted_v3_root(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config["trusted_v3"]["root"]


def validate_trusted_v3_pins(config: dict[str, Any]) -> dict[str, Any]:
    trusted = config["trusted_v3"]
    config_file = REPO_ROOT / trusted["config_file"]
    provenance_file = REPO_ROOT / trusted["provenance_file"]
    if sha256_file(config_file) != trusted["config_file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 config file hash changed")
    if sha256_file(provenance_file) != trusted["provenance_file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 provenance file hash changed")
    return {"config_file_sha256": trusted["config_file_sha256"], "provenance_file_sha256": trusted["provenance_file_sha256"]}


def load_v3_manifest(config: dict[str, Any], trajectory_id: str) -> tuple[Path, dict[str, Any]]:
    path = trusted_v3_root(config) / "run" / "trajectories" / trajectory_id / "manifest.json"
    manifest = json.loads(path.read_text())
    trusted = config["trusted_v3"]
    if manifest.get("config_hash") != trusted["config_hash"] or manifest.get("provenance_hash") != trusted["provenance_hash"]:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 manifest binding mismatch for {trajectory_id}")
    if f"{manifest['prompt_id']}_{int(manifest['generation_seed'])}" != trajectory_id:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 trajectory identity mismatch for {trajectory_id}")
    if manifest.get("scheduler") != EXPECTED_SCHEDULER or manifest.get("model") != config["model"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 scheduler/model mismatch")
    return path, manifest


def v3_state_record(manifest: dict[str, Any], step: int) -> dict[str, Any]:
    rows = [row for row in manifest["states"] if int(row["step"]) == int(step)]
    if len(rows) != 1:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 state {step} missing or duplicated")
    row = rows[0]
    if row.get("runtime_dtype") != EXPECTED_RUNTIME_DTYPE or int(row.get("runtime_element_size_bytes", 0)) != 2:
        raise GlobalStopError("GLOBAL STOP: trusted v3 state is not validated BF16 runtime state")
    if tuple(int(item) for item in row["shape"]) != EXPECTED_SHAPE:
        raise GlobalStopError("GLOBAL STOP: trusted v3 state shape changed")
    return row


def load_v3_state(row: dict[str, Any]) -> np.ndarray:
    path = REPO_ROOT / row["latent_path"]
    if sha256_file(path) != row["file_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 state file hash mismatch: {row['latent_path']}")
    array = base.load_v3_checkpoint_numpy(path, row)
    if v3.array_sha256(array) != row["tensor_sha256"]:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 state tensor hash mismatch: {row['latent_path']}")
    if not np.array_equal(base.cast_runtime_bf16(array), array):
        raise GlobalStopError("GLOBAL STOP: trusted v3 state is not BF16-exact")
    return array


def load_cell_clean(config: dict[str, Any], cell: dict[str, Any]) -> np.ndarray:
    """Trusted v3 CLEAN state for a frozen cell, re-verified against the frozen identity."""
    _, v3_manifest = load_v3_manifest(config, cell["trajectory_id"])
    clean = load_v3_state(v3_state_record(v3_manifest, int(cell["step"])))
    if identity(clean) != cell["clean_state"]["canonical_identity"]:
        raise GlobalStopError(f"GLOBAL STOP: trusted v3 clean state changed for {cell['prompt_id']}/step{int(cell['step']):03d}")
    return clean


def schedule(config: dict[str, Any]) -> list[float]:
    return single_flip.scheduler_timesteps_numpy(config)


def trusted_phase1(config: dict[str, Any]) -> dict[str, Any]:
    trusted = config["trusted_phase1"]
    root = REPO_ROOT / trusted["root"]
    anchor_path = root / trusted["anchor_manifest"]
    trace_path = root / trusted["trace_manifest"]
    if sha256_file(anchor_path) != trusted["anchor_manifest_sha256"] or sha256_file(trace_path) != trusted["trace_manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 manifest hashes changed")
    anchor = json.loads(anchor_path.read_text())
    trace = json.loads(trace_path.read_text())
    if tuple(trusted["trajectories"]) != ANCHOR_TRAJECTORIES:
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 trajectory set changed")
    anchor_info = anchor.get("anchor", {})
    if (
        anchor_info.get("prompt_id") != trusted["anchor_prompt_id"]
        or int(anchor_info.get("generation_seed", -1)) != int(trusted["anchor_generation_seed"])
        or int(anchor_info.get("checkpoint_step", -1)) != int(trusted["anchor_checkpoint_step"])
    ):
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 anchor identity mismatch")
    inputs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for name in ANCHOR_TRAJECTORIES:
        record = anchor["trajectories"][name]
        if record.get("runtime_dtype_semantics") != EXPECTED_RUNTIME_DTYPE:
            raise GlobalStopError("GLOBAL STOP: trusted Phase-1 input semantics changed")
        inputs[name] = {key: record[key] for key in ("relative_path", "file_sha256", "canonical_identity", "shape")}
        rows = [row for row in trace["traces"][name] if row.get("boundary") == trusted["validation_boundary"]]
        if len(rows) != 1 or int(rows[0].get("absolute_diffusion_step_index", -1)) != int(trusted["anchor_checkpoint_step"]) + 1:
            raise GlobalStopError("GLOBAL STOP: trusted Phase-1 validation boundary missing or mis-indexed")
        artifact = rows[0]["artifact"]
        outputs[name] = {
            "relative_path": artifact["relative_path"],
            "file_sha256": artifact["file_sha256"],
            "canonical_identity": artifact["canonical_identity"],
            "shape": artifact["shape"],
            "scheduler_timestep": rows[0]["scheduler_timestep"],
        }
    return {
        "root": trusted["root"],
        "anchor_manifest_sha256": trusted["anchor_manifest_sha256"],
        "trace_manifest_sha256": trusted["trace_manifest_sha256"],
        "trace_provenance_hash": trace["provenance_hash"],
        "prompt_id": trusted["anchor_prompt_id"],
        "generation_seed": int(trusted["anchor_generation_seed"]),
        "checkpoint_step": int(trusted["anchor_checkpoint_step"]),
        "prompt": anchor_info["prompt"],
        "resume_timestep": anchor_info["resume_timestep"],
        "validation_boundary": trusted["validation_boundary"],
        "inputs": inputs,
        "expected_outputs": outputs,
    }


def load_phase1_array(config: dict[str, Any], record: dict[str, Any]) -> np.ndarray:
    root = REPO_ROOT / config["trusted_phase1"]["root"]
    return load_bound_array(root, record)


# --------------------------------------------------------------------------------------
# serialized candidate representations (numpy re-implementation of grouped_symmetric_v1)
# --------------------------------------------------------------------------------------
def group_ranges(size: int, count: int) -> list[tuple[int, int]]:
    if count <= 0 or count > size:
        raise GlobalStopError(f"GLOBAL STOP: invalid group count {count} for dimension {size}")
    boundaries = [round(index * size / count) for index in range(count + 1)]
    return [(boundaries[index], boundaries[index + 1]) for index in range(count)]


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.ascontiguousarray(values, dtype=np.uint8)
    if bits == 8:
        return values.tobytes()
    per_byte = 8 // bits
    padded = np.zeros(math.ceil(values.size / per_byte) * per_byte, dtype=np.uint8)
    padded[: values.size] = values
    packed = np.zeros(padded.size // per_byte, dtype=np.uint8)
    for offset in range(per_byte):
        packed |= (padded[offset::per_byte] & ((1 << bits) - 1)).astype(np.uint8) << np.uint8(offset * bits)
    return packed.tobytes()


def unpack_unsigned(payload: bytes, bits: int, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return packed[:count].copy()
    per_byte = 8 // bits
    output = np.empty(packed.size * per_byte, dtype=np.uint8)
    mask = (1 << bits) - 1
    for offset in range(per_byte):
        output[offset::per_byte] = (packed >> np.uint8(offset * bits)) & np.uint8(mask)
    return output[:count].copy()


def quantize_values(values: np.ndarray, bits: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Symmetric per-group quantization; mirrors the trusted torch reference exactly.

    scale = max|x| / qmax (python float); q = round_half_even(x / float32(scale)) clamped to
    [-qmax, qmax]; restored = float32(q) * float32(scale).
    """
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    qmax = (1 << (bits - 1)) - 1
    maximum = float(np.abs(values).max()) if values.size else 0.0
    scale = maximum / qmax if maximum else 1.0
    quantized = np.clip(np.rint(values / np.float32(scale)), -qmax, qmax).astype(np.int16)
    unsigned = (quantized + qmax).astype(np.uint8)
    restored = quantized.astype(np.float32) * np.float32(scale)
    return unsigned, scale, restored


def encode_grouped(clean: np.ndarray, bits: int, group_count: int) -> tuple[bytes, dict[str, Any]]:
    channels = clean.shape[1]
    parts: list[bytes] = []
    records: list[dict[str, Any]] = []
    offset = 0
    for index, (start, end) in enumerate(group_ranges(channels, group_count)):
        values = clean[:, start:end].reshape(-1)
        unsigned, scale, _ = quantize_values(values, bits)
        chunk = pack_unsigned(unsigned, bits)
        records.append(
            {
                "group_id": f"channel_{index:02d}",
                "channel_range": [start, end],
                "bits": bits,
                "count": int(values.size),
                "offset": offset,
                "nbytes": len(chunk),
                "scale": float(scale),
            }
        )
        offset += len(chunk)
        parts.append(chunk)
    metadata = {
        "format": "grouped_symmetric_v1",
        "partition": "equal_channel",
        "shape": [int(item) for item in clean.shape],
        "group_count": group_count,
        "groups": records,
    }
    return b"".join(parts), metadata


def decode_grouped(payload: bytes, metadata: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(item) for item in metadata["shape"])
    restored = np.empty(shape, dtype=np.float32)
    ranges = group_ranges(shape[1], int(metadata["group_count"]))
    if len(ranges) != len(metadata["groups"]):
        raise GlobalStopError("GLOBAL STOP: grouped payload metadata is inconsistent")
    for (start, end), record in zip(ranges, metadata["groups"], strict=True):
        if list(record["channel_range"]) != [start, end]:
            raise GlobalStopError("GLOBAL STOP: grouped payload channel ranges changed")
        bits = int(record["bits"])
        qmax = (1 << (bits - 1)) - 1
        chunk = payload[record["offset"] : record["offset"] + record["nbytes"]]
        unsigned = unpack_unsigned(chunk, bits, int(record["count"])).astype(np.int16)
        values = (unsigned - qmax).astype(np.float32) * np.float32(float(record["scale"]))
        restored[:, start:end] = values.reshape(restored[:, start:end].shape)
    return restored


def decode_candidate(payload: bytes, metadata: dict[str, Any]) -> np.ndarray:
    """Decode any persisted candidate payload back to float32 storage."""
    shape = tuple(int(item) for item in metadata["shape"])
    if metadata["format"] == "bf16_bits":
        return np.ascontiguousarray(base.decode_runtime_bf16(np.frombuffer(payload, dtype=np.uint16).reshape(shape)), dtype=np.float32)
    if metadata["format"] == "fp16":
        return np.frombuffer(payload, dtype=np.float16).reshape(shape).astype(np.float32)
    if metadata["format"] == "grouped_symmetric_v1":
        return decode_grouped(payload, metadata)
    raise GlobalStopError(f"GLOBAL STOP: unknown candidate format {metadata.get('format')!r}")


def encode_candidate(clean: np.ndarray, spec: dict[str, Any]) -> tuple[bytes, dict[str, Any], np.ndarray]:
    """Serialize `clean` (float32 storage of BF16-exact values) and decode it back from the bytes."""
    name = spec["name"]
    if name == "bf16":
        payload = base.encode_runtime_bf16(clean).tobytes()
        metadata = {"format": "bf16_bits", "shape": [int(item) for item in clean.shape]}
        restored = base.decode_runtime_bf16(np.frombuffer(payload, dtype=np.uint16).reshape(clean.shape))
    elif name == "fp16":
        payload = clean.astype(np.float16).tobytes()
        metadata = {"format": "fp16", "shape": [int(item) for item in clean.shape]}
        restored = np.frombuffer(payload, dtype=np.float16).reshape(clean.shape).astype(np.float32)
    else:
        payload, metadata = encode_grouped(clean, int(spec["bits"]), int(spec["group_count"]))
        restored = decode_grouped(payload, metadata)
    return payload, metadata, np.ascontiguousarray(restored, dtype=np.float32)


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    difference = np.asarray(candidate, dtype=np.float64) - reference64
    denominator = float(np.sqrt(np.sum(reference64 * reference64)))
    if denominator == 0.0:
        raise GlobalStopError("GLOBAL STOP: reference tensor has zero norm; relative L2 undefined")
    return float(np.sqrt(np.sum(difference * difference)) / denominator)


def prediction_path_relative_l2(clean: np.ndarray, reference_next: np.ndarray, candidate: np.ndarray, candidate_next: np.ndarray) -> float | None:
    """Descriptive only: relative L2 of the change in the scheduler UPDATE (next - input) between
    candidate and CLEAN, i.e. the part of Y that does not come from the residual latent path.
    None when the CLEAN update has zero norm."""
    reference_update = np.asarray(reference_next, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    candidate_update = np.asarray(candidate_next, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    denominator = float(np.sqrt(np.sum(reference_update * reference_update)))
    if denominator == 0.0:
        return None
    difference = candidate_update - reference_update
    return float(np.sqrt(np.sum(difference * difference)) / denominator)


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    difference = candidate64 - reference64
    changed = int(np.count_nonzero(np.asarray(reference) != np.asarray(candidate)))
    return {
        "relative_l2": relative_l2(reference, candidate),
        "mse": float(np.mean(difference * difference)),
        "max_abs_diff": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "changed_element_count": changed,
        "changed_element_fraction": changed / int(np.asarray(reference).size),
        "bit_exact": changed == 0,
    }


# --------------------------------------------------------------------------------------
# CPU mode: construct every candidate, freeze the manifest
# --------------------------------------------------------------------------------------
def cell_key(prompt_id: str, step: int) -> str:
    return f"{prompt_id}/step{int(step):03d}"


def build_manifest(config: dict[str, Any], config_path: Path, root: Path, prov: dict[str, Any]) -> dict[str, Any]:
    pins = validate_trusted_v3_pins(config)
    frozen_schedule = schedule(config)
    phase1 = trusted_phase1(config)
    anchor_expected = frozen_schedule[phase1["checkpoint_step"]]
    if not loc.timestep_matches(phase1["resume_timestep"], anchor_expected, frozen_schedule):
        raise GlobalStopError("GLOBAL STOP: Phase-1 anchor resume timestep does not match the frozen schedule")
    for name, record in phase1["expected_outputs"].items():
        if not loc.timestep_matches(record["scheduler_timestep"], anchor_expected, frozen_schedule):
            raise GlobalStopError(f"GLOBAL STOP: Phase-1 {name} validation boundary timestep does not match the frozen schedule")
    specs = {row["name"]: row for row in config["candidates"]}
    cells: dict[str, Any] = {}
    prompts: dict[str, Any] = {}
    for trajectory_id in config["trusted_v3"]["trajectory_ids"]:
        manifest_path, manifest = load_v3_manifest(config, trajectory_id)
        prompt_id = manifest["prompt_id"]
        seed = int(manifest["generation_seed"])
        prompts[prompt_id] = {
            "trajectory_id": trajectory_id,
            "generation_seed": seed,
            "prompt": manifest["prompt"],
            "motion_category": manifest.get("motion_category"),
            "manifest_relative_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
        }
        for step in CHECKPOINT_STEPS:
            previous_row = v3_state_record(manifest, step - 1)
            state_row = v3_state_record(manifest, step)
            expected_timestep = frozen_schedule[step]
            if not loc.timestep_matches(state_row.get("next_scheduler_timestep"), expected_timestep, frozen_schedule):
                raise GlobalStopError(f"GLOBAL STOP: trusted v3 state {trajectory_id}/{step} next timestep does not match the frozen schedule")
            clean = load_v3_state(state_row)
            load_v3_state(previous_row)  # hash + BF16-exactness of the transition source
            candidates: dict[str, Any] = {}
            for name in CANDIDATE_NAMES:
                spec = specs[name]
                payload, metadata, restored = encode_candidate(clean, spec)
                runtime = base.cast_runtime_bf16(restored)
                if not np.array_equal(base.cast_runtime_bf16(runtime), runtime):
                    raise GlobalStopError("GLOBAL STOP: runtime candidate is not BF16-exact")
                relative_dir = f"candidates/{prompt_id}/step{step:03d}"
                payload_path = root / relative_dir / f"{name}.payload.bin"
                metadata_path = root / relative_dir / f"{name}.metadata.json"
                payload_path.parent.mkdir(parents=True, exist_ok=True)
                payload_path.write_bytes(payload)
                metadata_raw = canonical_json(metadata)
                metadata_path.write_bytes(metadata_raw)
                # decode again from the persisted bytes to prove the artifact, not memory, defines the candidate
                reloaded = decode_candidate(payload_path.read_bytes(), json.loads(metadata_path.read_text()))
                if reloaded.shape != restored.shape or reloaded.tobytes() != restored.tobytes():
                    raise GlobalStopError("GLOBAL STOP: persisted payload does not decode to the constructed candidate")
                state_record = save_array(root, f"{relative_dir}/{name}.runtime_state.npy", runtime)
                candidates[name] = {
                    "name": name,
                    "tier": spec["tier"],
                    "format": metadata["format"],
                    "bits": spec["bits"],
                    "group_count": spec.get("group_count"),
                    "payload_relative_path": f"{relative_dir}/{name}.payload.bin",
                    "payload_sha256": sha256_bytes(payload),
                    "payload_bytes": len(payload),
                    "metadata_relative_path": f"{relative_dir}/{name}.metadata.json",
                    "metadata_sha256": sha256_bytes(metadata_raw),
                    "metadata_bytes": len(metadata_raw),
                    "serialized_bytes": len(payload) + len(metadata_raw),
                    "byte_fraction_vs_bf16_runtime": (len(payload) + len(metadata_raw)) / (2 * clean.size),
                    "restored_equals_runtime": bool(np.array_equal(restored, runtime)),
                    "runtime_state": state_record,
                    "x": error_metrics(clean, runtime),
                }
            cells[cell_key(prompt_id, step)] = {
                "prompt_id": prompt_id,
                "trajectory_id": trajectory_id,
                "generation_seed": seed,
                "step": step,
                "absolute_step_executed": step,
                "expected_scheduler_timestep": expected_timestep,
                "clean_state": {
                    "latent_path": state_row["latent_path"],
                    "file_sha256": state_row["file_sha256"],
                    "tensor_sha256": state_row["tensor_sha256"],
                    "canonical_identity": identity(clean),
                },
                "transition_source_state": {
                    "step": step - 1,
                    "latent_path": previous_row["latent_path"],
                    "file_sha256": previous_row["file_sha256"],
                    "tensor_sha256": previous_row["tensor_sha256"],
                    "expected_scheduler_timestep": frozen_schedule[step - 1],
                },
                "candidates": candidates,
            }
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_sha256": prov["config_sha256"],
        "provenance_hash": prov["provenance_hash"],
        "frozen_rules": {
            "primary_tiers": list(PRIMARY_TIERS),
            "x_ratio_max": X_RATIO_MAX,
            "y_ratio_min": Y_RATIO_MIN,
            "y_abs_floor": Y_ABS_FLOOR,
            "min_distinct_prompts": MIN_DISTINCT_PROMPTS,
            "min_distinct_steps": MIN_DISTINCT_STEPS,
            "min_distinct_pair_types": MIN_DISTINCT_PAIR_TYPES,
            "x_definition": config["primary"]["x"],
            "y_definition": config["primary"]["y"],
        },
        "execution": {
            "mode": "one_step",
            "execution_step_limit": 1,
            "skip_vae_decode": True,
            "scheduler_class": single_flip.EXPECTED_SCHEDULER_CLASS,
            "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "frozen_schedule": frozen_schedule,
            "timestep_match_abs_tol": loc.TIMESTEP_MATCH_ABS_TOL,
        },
        "trusted_v3": {**{key: config["trusted_v3"][key] for key in ("root", "config_hash", "provenance_hash")}, **pins},
        "trusted_phase1": phase1,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "candidate_names": list(CANDIDATE_NAMES),
        "prompts": prompts,
        "cells": cells,
        "expected_counts": {
            "cells": len(cells),
            "candidate_runs": len(cells) * len(CANDIDATE_NAMES),
            "reference_runs": len(cells),
            "validation_anchor_runs": len(ANCHOR_TRAJECTORIES),
            "validation_transition_runs": len(cells),
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def run_cpu(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    prov = provenance(config_path)
    manifest = build_manifest(config, config_path, root, prov)
    cells = manifest["cells"]
    all_candidates = [row for cell in cells.values() for row in cell["candidates"].values()]
    bf16_exact = all(row["x"]["bit_exact"] for row in all_candidates if row["name"] == "bf16")
    expected_bytes = {
        "bf16": 2 * math.prod(EXPECTED_SHAPE),
        "fp16": 2 * math.prod(EXPECTED_SHAPE),
        "int8": math.prod(EXPECTED_SHAPE),
        "int4": None,
    }

    def payload_ok(row: dict[str, Any]) -> bool:
        if row["tier"] == CONTROL_TIER:
            return row["payload_bytes"] == expected_bytes[row["name"]]
        if row["tier"] == "int8":
            return row["payload_bytes"] == expected_bytes["int8"]
        ranges = group_ranges(EXPECTED_SHAPE[1], int(row["group_count"]))
        per_group = math.prod(EXPECTED_SHAPE) // EXPECTED_SHAPE[1]
        return row["payload_bytes"] == sum(math.ceil((end - start) * per_group * 4 / 8) for start, end in ranges)

    x_positive_by_tier = {
        tier: all(row["x"]["relative_l2"] > 0.0 for row in all_candidates if row["tier"] == tier)
        for tier in PRIMARY_TIERS
    }
    gates = [
        gate("S-C1 config version/modes/candidates/rules frozen", True, {"experiment_version": EXPERIMENT_VERSION, "rules": manifest["frozen_rules"]}, required=True),
        gate("S-C2 trusted v3 pins verified (config/provenance files, 12 manifests)", len(manifest["prompts"]) == 12, {"prompts": sorted(manifest["prompts"])}, required=True),
        gate("S-C3 trusted Phase-1 manifests verified and anchor inputs/outputs bound", True, {key: manifest["trusted_phase1"][key] for key in ("trace_manifest_sha256", "anchor_manifest_sha256", "trace_provenance_hash")}, required=True),
        gate("S-C4 every cell has 10 candidates", all(len(cell["candidates"]) == len(CANDIDATE_NAMES) for cell in cells.values()) and len(cells) == 36, {"cells": len(cells)}, required=True),
        gate("S-C5 clean states BF16-exact and bf16 candidates bit-exact to clean", bf16_exact, {"bf16_candidates_bit_exact_to_clean": bf16_exact, "clean_bf16_exactness": "GLOBAL STOP during construction otherwise"}, required=True),
        gate("S-C6 payload byte accounting matches format", all(payload_ok(row) for row in all_candidates), {"expected_bytes": expected_bytes}, required=True),
        gate("S-C7 every candidate decodes from its persisted bytes", True, "checked during construction (GLOBAL STOP otherwise)", required=True),
        gate("S-C8 frozen schedule matches trusted v3 and Phase-1 timesteps", True, {"steps": {step: manifest["execution"]["frozen_schedule"][step] for step in CHECKPOINT_STEPS}}, required=True),
        gate("S-C9 every primary-tier candidate has positive X", all(x_positive_by_tier.values()), x_positive_by_tier, required=True),
        gate("S-C10 no float-derived quantity is used as a binding", True, "bindings use sha256/identities only; X values are recomputed at analysis", required=True),
        gate("S-C11 manifest hash-bound to provenance/config", True, {"manifest_sha256": manifest["manifest_sha256"], "provenance_hash": prov["provenance_hash"]}, required=True),
        gate("S-C12 excluded scope not present", True, config["excluded"], required=True),
    ]
    document = {"mode": "cpu", "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    atomic_json(root / "provenance.json", prov)
    atomic_json(root / "preregistered_config.json", config)
    atomic_json(root / "screening_manifest.json", manifest)
    atomic_json(root / "cpu_gates.json", document)
    validate_gate_document(document, _cpu_gate_names(gates), provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {
        "mode": "cpu",
        "cells": len(cells),
        "candidate_runs": manifest["expected_counts"]["candidate_runs"],
        "manifest_sha256": manifest["manifest_sha256"],
        "all_passed": all(row["status"] == "PASS" for row in gates),
    }


def _cpu_gate_names(gates: list[dict[str, Any]]) -> tuple[str, ...]:
    names = tuple(row["name"] for row in gates)
    if tuple(name.split(" ", 1)[0] for name in names) != CPU_REQUIRED_GATES:
        raise GlobalStopError("GLOBAL STOP: CPU gate set changed")
    return names


def load_frozen(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prov = json.loads((root / "provenance.json").read_text())
    manifest = json.loads((root / "screening_manifest.json").read_text())
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if recorded != sha256_bytes(canonical_json(unhashed)):
        raise GlobalStopError("GLOBAL STOP: screening manifest hash mismatch")
    if manifest.get("provenance_hash") != prov.get("provenance_hash"):
        raise GlobalStopError("GLOBAL STOP: screening manifest is not bound to the stored provenance")
    current = provenance(config_path)
    if current["provenance_hash"] != prov["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: current source/config provenance differs from the frozen manifest provenance")
    cpu = json.loads((root / "cpu_gates.json").read_text())
    validate_gate_document(cpu, tuple(row["name"] for row in cpu["gates"]), provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    if tuple(name.split(" ", 1)[0] for name in (row["name"] for row in cpu["gates"])) != CPU_REQUIRED_GATES:
        raise GlobalStopError("GLOBAL STOP: CPU gate document does not carry the required gate set")
    return prov, manifest


# --------------------------------------------------------------------------------------
# GPU execution helpers (never executed by the auditor)
# --------------------------------------------------------------------------------------
def one_step_sampling_params(config: dict[str, Any], *, seed: int, label: str, artifact_dir: Path, latents: Any, step_index: int) -> Any:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    generation = config["generation"]
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
    sampling.latents = latents.detach().cpu().clone()
    sampling.step_index = int(step_index)
    sampling.extra_args = {
        "flow_shift": float(config["scheduler"]["flow_shift"]),
        "sample_solver": "euler",
        "execution_step_limit": 1,
        "skip_vae_decode": True,
        "trajectory_probe": {
            "artifact_dir": str(artifact_dir),
            "request_label": label,
            "capture_steps": [0, 1],
            "fps": float(generation["fps"]),
            "save_decoded": False,
            "save_latents": True,
            "save_mp4": False,
        },
    }
    return sampling


def run_one_step(
    omni: Any,
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    label: str,
    artifact_dir: Path,
    input_state: np.ndarray,
    step_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Execute exactly one scheduler update from `input_state` at absolute `step_index`.

    Returns (observed_input_state, next_state, semantics) where both arrays are float32
    storage of the BF16 runtime values persisted by the trajectory probe.
    """
    import torch

    from vllm_omni.outputs import OmniRequestOutput

    started = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        one_step_sampling_params(
            config,
            seed=seed,
            label=label,
            artifact_dir=artifact_dir,
            latents=torch.from_numpy(np.ascontiguousarray(input_state, dtype=np.float32)),
            step_index=step_index,
        ),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output = OmniRequestOutput.unwrap_result(outputs)
    custom = output.custom_output or {}
    control = custom.get("execution_control")
    metadata_path = custom.get("trajectory_probe_metadata_path")
    if not isinstance(control, dict) or not metadata_path:
        raise GlobalStopError("GLOBAL STOP: bounded execution metadata missing from the worker output")
    metadata = json.loads(Path(metadata_path).read_text())
    records = {int(row["step_index"]): row for row in metadata["records"]}
    if set(records) != {0, 1}:
        raise GlobalStopError("GLOBAL STOP: one-step probe did not record exactly steps 0 and 1")
    observed_input = v3.load_tensor_numpy(records[0]["latent_path"])
    next_state = v3.load_tensor_numpy(records[1]["latent_path"])
    frozen_schedule = manifest["execution"]["frozen_schedule"]
    semantics = {
        "elapsed_ms": elapsed_ms,
        "execution_step_limit": control.get("execution_step_limit"),
        "executed_local_steps": control.get("executed_local_steps"),
        "resume_step_index": control.get("resume_step_index"),
        "vae_decode_skipped": control.get("vae_decode_skipped"),
        "probe_num_steps": metadata.get("num_steps"),
        "scheduler_class": metadata.get("scheduler_class"),
        "sample_solver": metadata.get("sample_solver"),
        "runtime_dtype": records[1].get("runtime_dtype"),
        "observed_timestep": records[1].get("timestep"),
        "expected_timestep": frozen_schedule[step_index],
        "trajectory_probe_metadata_path": str(metadata_path),
    }
    semantics["valid"] = bool(
        semantics["execution_step_limit"] == 1
        and semantics["executed_local_steps"] == 1
        and semantics["resume_step_index"] == step_index
        and semantics["vae_decode_skipped"] is True
        and semantics["probe_num_steps"] == 1
        and str(semantics["scheduler_class"]).endswith(EXPECTED_SCHEDULER)
        and semantics["sample_solver"] == "euler"
        and semantics["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE
        and loc.timestep_matches(semantics["observed_timestep"], semantics["expected_timestep"], frozen_schedule)
        and observed_input.shape == EXPECTED_SHAPE
        and next_state.shape == EXPECTED_SHAPE
        and bool(np.array_equal(observed_input, np.asarray(input_state, dtype=np.float32)))
    )
    return observed_input, next_state, semantics


def build_omni(config: dict[str, Any], args: argparse.Namespace) -> Any:
    return v3.build_omni(config, args)


def run_validate(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    prov, manifest = load_frozen(root, config_path)
    require_committed_source(prov)
    phase1 = manifest["trusted_phase1"]
    frozen_schedule = manifest["execution"]["frozen_schedule"]
    omni = build_omni(config, args)
    anchor_rows: dict[str, Any] = {}
    transition_rows: dict[str, Any] = {}
    try:
        anchor_prompt = manifest["prompts"][phase1["prompt_id"]]
        if int(anchor_prompt["generation_seed"]) != int(phase1["generation_seed"]) or anchor_prompt["prompt"] != phase1["prompt"]:
            raise GlobalStopError("GLOBAL STOP: Phase-1 anchor prompt/seed does not match the trusted v3 prompt")
        for name in ANCHOR_TRAJECTORIES:
            input_state = load_phase1_array(config, phase1["inputs"][name])
            expected = load_phase1_array(config, phase1["expected_outputs"][name])
            observed_input, next_state, semantics = run_one_step(
                omni,
                config,
                manifest,
                prompt=anchor_prompt["prompt"],
                seed=int(phase1["generation_seed"]),
                label=f"validate_anchor_{name.lower()}",
                artifact_dir=root / "validate" / "anchor" / name / "trajectory_probe",
                input_state=input_state,
                step_index=int(phase1["checkpoint_step"]),
            )
            record = save_array(root, f"validate/anchor/{name}/next_state.npy", next_state)
            anchor_rows[name] = {
                "input_identity": identity(input_state),
                "expected_identity": phase1["expected_outputs"][name]["canonical_identity"],
                "observed_identity": record["canonical_identity"],
                "bit_exact": bool(np.array_equal(next_state, expected)),
                "artifact": record,
                "semantics": semantics,
            }
        for key, cell in manifest["cells"].items():
            source_row = cell["transition_source_state"]
            _, target_manifest = load_v3_manifest(config, cell["trajectory_id"])
            source_record = v3_state_record(target_manifest, int(source_row["step"]))
            if source_record["file_sha256"] != source_row["file_sha256"] or source_record["tensor_sha256"] != source_row["tensor_sha256"]:
                raise GlobalStopError("GLOBAL STOP: trusted v3 transition source changed since the manifest was frozen")
            source = load_v3_state(source_record)
            target = load_v3_state(v3_state_record(target_manifest, cell["step"]))
            if identity(target) != cell["clean_state"]["canonical_identity"]:
                raise GlobalStopError("GLOBAL STOP: trusted v3 clean state identity changed since the manifest was frozen")
            prompt = manifest["prompts"][cell["prompt_id"]]
            observed_input, next_state, semantics = run_one_step(
                omni,
                config,
                manifest,
                prompt=prompt["prompt"],
                seed=int(cell["generation_seed"]),
                label=f"validate_transition_{cell['prompt_id']}_step{cell['step']:03d}",
                artifact_dir=root / "validate" / "transitions" / cell["prompt_id"] / f"step{cell['step']:03d}" / "trajectory_probe",
                input_state=source,
                step_index=int(source_row["step"]),
            )
            record = save_array(root, f"validate/transitions/{cell['prompt_id']}/step{cell['step']:03d}/next_state.npy", next_state)
            transition_rows[key] = {
                "source_step": source_row["step"],
                "target_step": cell["step"],
                "target_identity": cell["clean_state"]["canonical_identity"],
                "observed_identity": record["canonical_identity"],
                "bit_exact": bool(np.array_equal(next_state, target)),
                "artifact": record,
                "semantics": semantics,
            }
    finally:
        base._shutdown(omni)
    all_semantics_valid = all(row["semantics"]["valid"] for row in anchor_rows.values()) and all(row["semantics"]["valid"] for row in transition_rows.values())
    gates = [
        gate("S-V1 Phase-1 inputs/outputs re-identified from trusted artifacts", True, {name: phase1["expected_outputs"][name]["canonical_identity"] for name in ANCHOR_TRAJECTORIES}, required=True),
        gate("S-V2 one-step CLEAN == Phase-1 after_step_001 CLEAN", anchor_rows["CLEAN"]["bit_exact"], anchor_rows["CLEAN"], required=True),
        gate("S-V3 one-step PLUS1 == Phase-1 after_step_001 PLUS1", anchor_rows["PLUS1"]["bit_exact"], anchor_rows["PLUS1"], required=True),
        gate("S-V4 one-step HISTORICAL_PLUS14 == Phase-1 after_step_001 HISTORICAL_PLUS14", anchor_rows["HISTORICAL_PLUS14"]["bit_exact"], anchor_rows["HISTORICAL_PLUS14"], required=True),
        gate("S-V5 one-step from trusted v3 state k-1 reproduces state k for all 36 cells", len(transition_rows) == len(manifest["cells"]) and all(row["bit_exact"] for row in transition_rows.values()), {key: row["bit_exact"] for key, row in transition_rows.items()}, required=True),
        gate("S-V6 bounded-execution semantics valid on every validation run", all_semantics_valid, {"anchor": {k: v["semantics"] for k, v in anchor_rows.items()}, "transitions": {k: v["semantics"] for k, v in transition_rows.items()}}, required=True),
        gate("S-V7 executed timesteps match the frozen schedule", all(loc.timestep_matches(row["semantics"]["observed_timestep"], row["semantics"]["expected_timestep"], frozen_schedule) for row in [*anchor_rows.values(), *transition_rows.values()]), {"schedule_steps": {step: frozen_schedule[step] for step in CHECKPOINT_STEPS}}, required=True),
    ]
    document = {"mode": "validate", "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "anchor": anchor_rows, "transitions": transition_rows}
    atomic_json(root / "validate" / "validate_gates.json", document)
    if any(row["status"] != "PASS" for row in gates):
        atomic_json(root / "validate" / "SCREENING_INVALID.json", {"reason": "one-step execution path failed validation", "failed": [row["name"] for row in gates if row["status"] != "PASS"]})
        raise GlobalStopError("GLOBAL STOP: SCREENING INVALID - one-step execution path failed validation")
    validate_gate_document(document, tuple(row["name"] for row in gates), provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {"mode": "validate", "anchor_bit_exact": {k: v["bit_exact"] for k, v in anchor_rows.items()}, "transitions_bit_exact": sum(1 for row in transition_rows.values() if row["bit_exact"]), "all_passed": True}


def require_validation(root: Path, prov: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    path = root / "validate" / "validate_gates.json"
    if not path.exists() or (root / "validate" / "SCREENING_INVALID.json").exists():
        raise GlobalStopError("GLOBAL STOP: validation gates are missing or the screening was marked INVALID")
    document = json.loads(path.read_text())
    names = tuple(row["name"] for row in document.get("gates", []))
    if tuple(name.split(" ", 1)[0] for name in names) != VALIDATE_REQUIRED_GATES:
        raise GlobalStopError("GLOBAL STOP: validation gate set changed")
    validate_gate_document(document, names, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    # re-derive the validation booleans from persisted artifacts; never trust the stored flags
    for name, row in document["anchor"].items():
        observed = load_bound_array(root, row["artifact"])
        if identity(observed) != manifest["trusted_phase1"]["expected_outputs"][name]["canonical_identity"]:
            raise GlobalStopError(f"GLOBAL STOP: persisted validation artifact for {name} is not bit-exact with Phase-1")
    for key, row in document["transitions"].items():
        observed = load_bound_array(root, row["artifact"])
        if identity(observed) != manifest["cells"][key]["clean_state"]["canonical_identity"]:
            raise GlobalStopError(f"GLOBAL STOP: persisted transition artifact for {key} is not bit-exact with trusted v3")
    return document


def run_screening(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    prov, manifest = load_frozen(root, config_path)
    require_committed_source(prov)
    require_validation(root, prov, manifest)
    omni = build_omni(config, args)
    trace: dict[str, Any] = {"provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "cells": {}}
    try:
        for key, cell in manifest["cells"].items():
            prompt = manifest["prompts"][cell["prompt_id"]]
            clean = load_cell_clean(config, cell)
            cell_dir = f"screening/{cell['prompt_id']}/step{cell['step']:03d}"
            _, reference_next, reference_semantics = run_one_step(
                omni, config, manifest,
                prompt=prompt["prompt"], seed=int(cell["generation_seed"]),
                label=f"reference_{cell['prompt_id']}_step{cell['step']:03d}",
                artifact_dir=root / cell_dir / "reference" / "trajectory_probe",
                input_state=clean, step_index=int(cell["step"]),
            )
            if not reference_semantics["valid"]:
                raise GlobalStopError(f"GLOBAL STOP: reference one-step semantics invalid for {key}")
            reference_record = save_array(root, f"{cell_dir}/reference/next_state.npy", reference_next)
            runs: dict[str, Any] = {}
            for name in CANDIDATE_NAMES:
                candidate = cell["candidates"][name]
                state = load_bound_array(root, candidate["runtime_state"])
                _, next_state, semantics = run_one_step(
                    omni, config, manifest,
                    prompt=prompt["prompt"], seed=int(cell["generation_seed"]),
                    label=f"{name}_{cell['prompt_id']}_step{cell['step']:03d}",
                    artifact_dir=root / cell_dir / name / "trajectory_probe",
                    input_state=state, step_index=int(cell["step"]),
                )
                if not semantics["valid"]:
                    raise GlobalStopError(f"GLOBAL STOP: candidate one-step semantics invalid for {key}/{name}")
                record = save_array(root, f"{cell_dir}/{name}/next_state.npy", next_state)
                runs[name] = {"next_state": record, "semantics": semantics}
                if name == "bf16" and not np.array_equal(next_state, reference_next):
                    raise GlobalStopError(f"GLOBAL STOP: CLEAN one-step repeat is not deterministic for {key}")
            trace["cells"][key] = {"reference": {"next_state": reference_record, "semantics": reference_semantics}, "runs": runs}
            atomic_json(root / "screening" / "trace.json", trace)
    finally:
        base._shutdown(omni)
    atomic_json(root / "screening" / "trace.json", trace)
    return {"mode": "screening", "cells": len(trace["cells"]), "runs": sum(len(cell["runs"]) + 1 for cell in trace["cells"].values())}


# --------------------------------------------------------------------------------------
# analysis (CPU): recompute everything from persisted artifacts
# --------------------------------------------------------------------------------------
def rules_frozen(rules: dict[str, Any]) -> bool:
    return (
        rules.get("primary_tiers") == list(PRIMARY_TIERS)
        and rules.get("x_ratio_max") == X_RATIO_MAX
        and rules.get("y_ratio_min") == Y_RATIO_MIN
        and rules.get("y_abs_floor") == Y_ABS_FLOOR
        and rules.get("min_distinct_prompts") == MIN_DISTINCT_PROMPTS
        and rules.get("min_distinct_steps") == MIN_DISTINCT_STEPS
        and rules.get("min_distinct_pair_types") == MIN_DISTINCT_PAIR_TYPES
    )


def pair_type(tier: str, first: str, second: str) -> str:
    left, right = sorted((first, second))
    return f"{tier}:{left}|{right}"


def evaluate_pair(x_a: float, y_a: float, x_b: float, y_b: float) -> dict[str, Any]:
    """Preregistered eligibility + meaningful-reversal test for one same-tier pair (A, B)."""
    eligible = x_a > 0.0 and x_b > 0.0 and max(x_a, x_b) / min(x_a, x_b) <= X_RATIO_MAX
    if not eligible or x_a == x_b:
        return {"eligible": eligible, "orderable": eligible and x_a != x_b, "reversal": False, "meaningful": False}
    low, high = ((x_a, y_a), (x_b, y_b)) if x_a < x_b else ((x_b, y_b), (x_a, y_a))
    reversal = low[1] > high[1]
    meaningful = reversal and low[1] >= Y_RATIO_MIN * high[1] and (low[1] - high[1]) >= Y_ABS_FLOOR
    return {"eligible": True, "orderable": True, "reversal": reversal, "meaningful": meaningful, "y_lower_x": low[1], "y_higher_x": high[1]}


def decide(meaningful_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Preregistered CONTINUE / NO_GO rule over meaningful same-tier reversals only."""
    distinct_prompts = sorted({row["prompt_id"] for row in meaningful_rows})
    distinct_steps = sorted({int(row["step"]) for row in meaningful_rows})
    distinct_pair_types = sorted({row["pair_type"] for row in meaningful_rows})
    continue_ok = (
        len(distinct_prompts) >= MIN_DISTINCT_PROMPTS
        and len(distinct_steps) >= MIN_DISTINCT_STEPS
        and len(distinct_pair_types) >= MIN_DISTINCT_PAIR_TYPES
    )
    return {
        "decision": "CONTINUE" if continue_ok else "NO_GO",
        "distinct_prompts": distinct_prompts,
        "distinct_steps": distinct_steps,
        "distinct_pair_types": distinct_pair_types,
    }


def analyze_artifacts(root: Path, config: dict[str, Any], prov: dict[str, Any], manifest: dict[str, Any], trace: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if trace.get("provenance_hash") != prov["provenance_hash"] or trace.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: screening trace is not bound to the frozen manifest/provenance")
    if set(trace.get("cells", {})) != set(manifest["cells"]):
        raise GlobalStopError("GLOBAL STOP: screening trace cell set differs from the manifest")
    frozen_schedule = manifest["execution"]["frozen_schedule"]
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cross_tier_rows: list[dict[str, Any]] = []
    determinism: dict[str, bool] = {}
    semantics_valid = True
    x_recompute_ok = True
    finite = True
    specs = {row["name"]: row for row in config["candidates"]}
    for key, cell in manifest["cells"].items():
        trace_cell = trace["cells"][key]
        if set(trace_cell["runs"]) != set(CANDIDATE_NAMES):
            raise GlobalStopError(f"GLOBAL STOP: candidate run set incomplete for {key}")
        clean = load_cell_clean(config, cell)
        reference_next = load_bound_array(root, trace_cell["reference"]["next_state"])
        finite &= bool(np.all(np.isfinite(reference_next)))
        semantics_valid &= bool(trace_cell["reference"]["semantics"].get("valid")) and loc.timestep_matches(trace_cell["reference"]["semantics"].get("observed_timestep"), frozen_schedule[cell["step"]], frozen_schedule)
        cell_rows: dict[str, dict[str, Any]] = {}
        for name in CANDIDATE_NAMES:
            candidate = cell["candidates"][name]
            state = load_bound_array(root, candidate["runtime_state"])
            if not np.array_equal(base.cast_runtime_bf16(state), state):
                raise GlobalStopError(f"GLOBAL STOP: persisted runtime candidate is not BF16-exact for {key}/{name}")
            x = error_metrics(clean, state)
            recorded = candidate["x"]
            x_recompute_ok &= (
                x["changed_element_count"] == recorded["changed_element_count"]
                and x["bit_exact"] == recorded["bit_exact"]
                and abs(x["relative_l2"] - recorded["relative_l2"]) <= FLOAT_RECOMPUTE_REL_TOL * max(abs(recorded["relative_l2"]), 1e-300)
            )
            run = trace_cell["runs"][name]
            next_state = load_bound_array(root, run["next_state"])
            finite &= bool(np.all(np.isfinite(next_state)))
            semantics_valid &= bool(run["semantics"].get("valid")) and loc.timestep_matches(run["semantics"].get("observed_timestep"), frozen_schedule[cell["step"]], frozen_schedule)
            y = error_metrics(reference_next, next_state)
            if name == "bf16":
                determinism[key] = bool(np.array_equal(next_state, reference_next))
            row = {
                "cell": key,
                "prompt_id": cell["prompt_id"],
                "step": cell["step"],
                "candidate": name,
                "tier": specs[name]["tier"],
                "serialized_bytes": candidate["serialized_bytes"],
                "byte_fraction_vs_bf16_runtime": candidate["byte_fraction_vs_bf16_runtime"],
                "x_relative_l2": x["relative_l2"],
                "x_mse": x["mse"],
                "x_changed_element_fraction": x["changed_element_fraction"],
                "x_bit_exact": x["bit_exact"],
                "y_relative_l2": y["relative_l2"],
                "y_mse": y["mse"],
                "y_changed_element_fraction": y["changed_element_fraction"],
                "y_bit_exact": y["bit_exact"],
                "descriptive_prediction_path_relative_l2": prediction_path_relative_l2(clean, reference_next, state, next_state),
                "candidate_state_identity": candidate["runtime_state"]["canonical_identity"],
                "next_state_identity": run["next_state"]["canonical_identity"],
                "reference_next_identity": trace_cell["reference"]["next_state"]["canonical_identity"],
            }
            rows.append(row)
            cell_rows[name] = row
        for tier in PRIMARY_TIERS:
            members = [name for name in CANDIDATE_NAMES if specs[name]["tier"] == tier]
            for first, second in combinations(members, 2):
                a, b = cell_rows[first], cell_rows[second]
                verdict = evaluate_pair(a["x_relative_l2"], a["y_relative_l2"], b["x_relative_l2"], b["y_relative_l2"])
                pair_rows.append({"cell": key, "prompt_id": cell["prompt_id"], "step": cell["step"], "tier": tier, "pair_type": pair_type(tier, first, second), "a": first, "b": second, "x_a": a["x_relative_l2"], "x_b": b["x_relative_l2"], "y_a": a["y_relative_l2"], "y_b": b["y_relative_l2"], **verdict})
        for first in [name for name in CANDIDATE_NAMES if specs[name]["tier"] == "int8"]:
            for second in [name for name in CANDIDATE_NAMES if specs[name]["tier"] == "int4"]:
                a, b = cell_rows[first], cell_rows[second]
                verdict = evaluate_pair(a["x_relative_l2"], a["y_relative_l2"], b["x_relative_l2"], b["y_relative_l2"])
                cross_tier_rows.append({"cell": key, "pair_type": pair_type("cross", first, second), "x_a": a["x_relative_l2"], "x_b": b["x_relative_l2"], "y_a": a["y_relative_l2"], "y_b": b["y_relative_l2"], **verdict})
    meaningful = [row for row in pair_rows if row["meaningful"]]
    eligible = [row for row in pair_rows if row["eligible"]]
    orderable = [row for row in pair_rows if row["orderable"]]
    verdict = decide(meaningful)
    decision = verdict["decision"]
    distinct_prompts = verdict["distinct_prompts"]
    distinct_steps = verdict["distinct_steps"]
    distinct_pair_types = verdict["distinct_pair_types"]
    controls = {
        "bf16_next_state_bit_exact_all_cells": all(row["y_bit_exact"] for row in rows if row["candidate"] == "bf16"),
        "fp16_cells_with_x_zero": sum(1 for row in rows if row["candidate"] == "fp16" and row["x_bit_exact"]),
        "fp16_cells_with_y_bit_exact": sum(1 for row in rows if row["candidate"] == "fp16" and row["y_bit_exact"]),
        "fp16_cells_with_x_positive_and_y_bit_exact": sum(1 for row in rows if row["candidate"] == "fp16" and not row["x_bit_exact"] and row["y_bit_exact"]),
    }
    per_tier = {
        tier: {
            "pairs": sum(1 for row in pair_rows if row["tier"] == tier),
            "eligible": sum(1 for row in eligible if row["tier"] == tier),
            "orderable": sum(1 for row in orderable if row["tier"] == tier),
            "reversals_any": sum(1 for row in orderable if row["tier"] == tier and row["reversal"]),
            "meaningful_reversals": sum(1 for row in meaningful if row["tier"] == tier),
        }
        for tier in PRIMARY_TIERS
    }
    gates = [
        gate("S-A1 committed source / clean provenance", not prov.get("source_dirty_entries"), {"git_commit": prov.get("git_commit")}, required=True),
        gate("S-A2 validation gates re-derived from persisted artifacts", True, {"anchor": list(validation["anchor"]), "transitions": len(validation["transitions"])}, required=True),
        gate("S-A3 all 36 cells x 10 candidates + 36 references present", len(rows) == 360 and len(manifest["cells"]) == 36, {"rows": len(rows)}, required=True),
        gate("S-A4 every artifact re-identified (file sha256 + canonical identity)", True, "GLOBAL STOP otherwise", required=True),
        gate("S-A5 X recomputed from persisted candidate states agrees with the frozen manifest", x_recompute_ok, {"float_recompute_rel_tol": FLOAT_RECOMPUTE_REL_TOL, "note": "integers/bit-exact flags must match exactly; decision uses recomputed values"}, required=True),
        gate("S-A6 runtime candidates and clean states BF16-exact", True, "GLOBAL STOP otherwise", required=True),
        gate("S-A7 CLEAN one-step determinism (bf16 candidate run == reference run) in every cell", all(determinism.values()) and len(determinism) == 36, determinism, required=True),
        gate("S-A8 bounded-execution semantics valid on every run", semantics_valid, {"execution_step_limit": 1, "skip_vae_decode": True}, required=True),
        gate("S-A9 no NaN/Inf in any next-state artifact", finite, None, required=True),
        gate("S-A10 preregistered rule constants unchanged", rules_frozen(manifest["frozen_rules"]), manifest["frozen_rules"], required=True),
        gate("S-A11 eligibility restricted to same-tier primary pairs", all(row["tier"] in PRIMARY_TIERS for row in pair_rows), {"pairs": len(pair_rows)}, required=True),
        gate("S-A12 controls never enter eligibility or the decision", all(row["a"] not in ("bf16", "fp16") and row["b"] not in ("bf16", "fp16") for row in pair_rows), None, required=True),
        gate("S-A13 prompt set and steps frozen", sorted(manifest["prompts"]) == sorted({row["prompt_id"] for row in rows}) and sorted({row["step"] for row in rows}) == list(CHECKPOINT_STEPS), {"prompts": len(manifest["prompts"])}, required=True),
        gate("S-A14 decision derived only from meaningful same-tier reversals", True, {"distinct_prompts": distinct_prompts, "distinct_steps": distinct_steps, "distinct_pair_types": distinct_pair_types}, required=True),
        gate("S-A15 no oracle / quality / localization quantities computed", True, config["excluded"], required=True),
        gate("S-A16 provenance/manifest hash-bound", True, {"manifest_sha256": manifest["manifest_sha256"], "provenance_hash": prov["provenance_hash"]}, required=True),
    ]
    return {
        "decision": decision,
        "decision_rule": {"min_distinct_prompts": MIN_DISTINCT_PROMPTS, "min_distinct_steps": MIN_DISTINCT_STEPS, "min_distinct_pair_types": MIN_DISTINCT_PAIR_TYPES},
        "meaningful_reversals": meaningful,
        "meaningful_reversal_summary": {"count": len(meaningful), "distinct_prompts": distinct_prompts, "distinct_steps": distinct_steps, "distinct_pair_types": distinct_pair_types},
        "descriptive": {
            "per_tier": per_tier,
            "eligible_pairs": len(eligible),
            "orderable_pairs": len(orderable),
            "reversals_any": sum(1 for row in orderable if row["reversal"]),
            "violation_rate_over_orderable": (sum(1 for row in orderable if row["reversal"]) / len(orderable)) if orderable else None,
            "meaningful_rate_over_orderable": (len(meaningful) / len(orderable)) if orderable else None,
            "cross_tier_int8_vs_int4": {"pairs": len(cross_tier_rows), "eligible": sum(1 for row in cross_tier_rows if row["eligible"]), "meaningful_reversals": sum(1 for row in cross_tier_rows if row["meaningful"])},
            "controls": controls,
        },
        "rows": rows,
        "pair_rows": pair_rows,
        "cross_tier_rows": cross_tier_rows,
        "gates": gates,
    }


def run_analyze(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    prov, manifest = load_frozen(root, config_path)
    validation = require_validation(root, prov, manifest)
    trace_path = root / "screening" / "trace.json"
    if not trace_path.exists():
        raise GlobalStopError("GLOBAL STOP: screening trace missing")
    trace = json.loads(trace_path.read_text())
    result = analyze_artifacts(root, config, prov, manifest, trace, validation)
    gates = result["gates"]
    document = {"mode": "analyze", "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "decision": result["decision"]}
    atomic_json(root / "analysis.json", {**result, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]})
    atomic_json(root / "gates.json", document)
    base.write_csv(root / "rows.csv", result["rows"])
    base.write_csv(root / "pair_rows.csv", result["pair_rows"])
    names = tuple(row["name"] for row in gates)
    if tuple(name.split(" ", 1)[0] for name in names) != ANALYZE_REQUIRED_GATES:
        raise GlobalStopError("GLOBAL STOP: analysis gate set changed")
    validate_gate_document(document, names, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {"mode": "analyze", "decision": result["decision"], "summary": result["meaningful_reversal_summary"], "descriptive": result["descriptive"]}


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_execution_ordering_screening_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = (args.output_dir or Path(config["output_root"])).resolve()
    if args.mode == "cpu":
        result = run_cpu(config, args.config, root)
    elif args.mode == "validate":
        result = run_validate(config, args.config, root, args)
    elif args.mode == "screening":
        result = run_screening(config, args.config, root, args)
    else:
        result = run_analyze(config, args.config, root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
