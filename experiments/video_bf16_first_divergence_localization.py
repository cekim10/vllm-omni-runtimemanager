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
EXPERIMENT_VERSION = "video-bf16-first-divergence-localization-v1"
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
    if config["phase3"] != {"enabled": False, "auto_expand": False}:
        raise GlobalStopError("GLOBAL STOP: phase 3 must stay disabled and non-automatic")
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
    input_clean_plus = metrics(
        load_tensor(root, frozen["CLEAN"]["entry_artifact"]),
        load_tensor(root, frozen["PLUS1"]["entry_artifact"]),
    )
    exit_clean_plus = metrics(
        load_tensor(root, frozen["CLEAN"]["exit_artifact"]),
        load_tensor(root, frozen["PLUS1"]["exit_artifact"]),
    )
    exit_plus_historical = metrics(
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
        "phase1_analysis_sha256": sha256_file(analysis_path),
        "source_provenance_hash": trace.get("provenance_hash"),
        "source_manifest_sha256": trace.get("manifest_sha256"),
        "entry_boundary": "input",
        "exit_boundary": "after_step_001",
        "trajectories": frozen,
        "trusted_input_clean_vs_plus1": input_clean_plus,
        "trusted_exit_clean_vs_plus1": exit_clean_plus,
        "trusted_exit_plus1_vs_historical": exit_plus_historical,
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
    return {"source": source, "clean": source.clean, "plus1": plus1, "historical": historical, "plus_record": plus_record, "historical_delta": historical_delta, "trusted_config": trusted_cfg, "trusted_finals": trusted_finals, "trusted_phase1": derive_trusted_phase1(config)}


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
        "expected_boundaries": [row["boundary"] for row in boundary_specs],
        "boundary_specifications": boundary_specs,
        "early_late_cutoff": early_late_cutoff(len(boundary_specs)),
        "timestep_match_policy": timestep_match_policy(),
        "phase2_freeze": phase2_freeze(config),
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
    if historical.get("changed_coordinate_count") != 6 or historical != data["historical_delta"]:
        raise GlobalStopError("GLOBAL STOP: manifest historical support differs from frozen construction")
    construction = manifest["trajectories"]["PLUS1"].get("construction", {})
    if construction != data["plus_record"]:
        raise GlobalStopError("GLOBAL STOP: manifest PLUS1 construction differs from frozen construction")
    if manifest.get("trusted_final_identities") != data["trusted_finals"]:
        raise GlobalStopError("GLOBAL STOP: manifest trusted final identities differ from frozen artifacts")
    if manifest.get("trusted_phase1") != data["trusted_phase1"]:
        raise GlobalStopError("GLOBAL STOP: manifest trusted Phase-1 binding differs from preserved artifacts")
    if manifest.get("phase2_freeze") != phase2_freeze(config):
        raise GlobalStopError("GLOBAL STOP: manifest Phase-2 freeze differs from config/scheduler")
    if not config_phase2_is_frozen(manifest):
        raise GlobalStopError("GLOBAL STOP: manifest Phase-2 freeze differs from the preregistered configuration")
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
                latent = np.load(result["recovered_final_latent_artifact"]["path"], allow_pickle=False)
                video = np.load(result["recovered_video_artifact"]["path"], allow_pickle=False)
                controls[name].append({
                    "final_latent_identity": identity(latent), "video_identity": identity(video),
                    "exact_vs_clean": bool(np.array_equal(latent, source.final_latent) and np.array_equal(video, source.video)),
                })
        doc = _preflight_document(root, manifest, prov, controls, source)
        return {"mode": "preflight", **doc}
    finally:
        single_flip._shutdown(omni)


def require_preflight(root: Path, manifest: dict[str, Any], prov: dict[str, Any]) -> None:
    path = root / "preflight" / "preflight_gates.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: preflight gates are missing")
    value = json.loads(path.read_text())
    validate_gate_document(
        value, PREFLIGHT_REQUIRED_GATES,
        provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"],
    )


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
    analysis_path = root / "phase1/phase1_analysis.json"
    if not trace_path.exists() or not analysis_path.exists():
        raise GlobalStopError("GLOBAL STOP: trusted Phase-1 files are missing from the result root")
    if sha256_file(trace_path) != trusted["trace_manifest_sha256"] or sha256_file(analysis_path) != trusted["phase1_analysis_sha256"]:
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
        _p2_gate(23, "Phase3 not automatically triggered", config["phase3"] == {"enabled": False, "auto_expand": False}, config["phase3"]),
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
    if args.mode in ("phase3", "analyze-phase3") and not config["phase3"]["enabled"]:
        raise GlobalStopError("GLOBAL STOP: phase 3 is disabled; no automatic block expansion is permitted")
    unavailable_gpu(args.mode)


if __name__ == "__main__":
    main()
