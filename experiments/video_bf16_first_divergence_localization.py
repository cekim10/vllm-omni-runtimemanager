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
    return config


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
    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {
        "selected_absolute_diffusion_step_index": selected_step,
        "selected_resumed_update_index": resumed_update_index,
        "phase1_entry_boundary": entry_boundary,
        "phase1_entry_boundary_specification": entry_spec,
        "selected_scheduler_timestep": timesteps[selected_step],
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
    return {"source": source, "clean": source.clean, "plus1": plus1, "historical": historical, "plus_record": plus_record, "historical_delta": historical_delta, "trusted_config": trusted_cfg, "trusted_finals": trusted_finals}


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
        "expected_boundaries": [row["boundary"] for row in boundary_specs],
        "boundary_specifications": boundary_specs,
        "early_late_cutoff": early_late_cutoff(len(boundary_specs)),
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
        gate("G23 phase-2 step explicitly frozen before execution", None, "phase2.selected_step is null; phase2 must fail closed", required=False),
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
    return manifest, prov


def validate_probe_records(records: list[dict[str, Any]], expected_specs: list[dict[str, Any]]) -> None:
    """Bind every persisted probe record to the frozen boundary specification.

    Production semantics: the resume input is a float32 tensor holding
    BF16-exact values; every scheduler output is BF16. Any deviation, in
    either direction, is a change of execution semantics and fails closed.
    """
    if [int(row.get("step_index", -1)) for row in records] != list(range(len(expected_specs))):
        raise GlobalStopError("GLOBAL STOP: trajectory probe did not persist every requested boundary")
    for record, expected in zip(records, expected_specs, strict=True):
        if record.get("runtime_dtype") != expected["expected_runtime_dtype"]:
            raise GlobalStopError(
                f"GLOBAL STOP: {expected['boundary']} runtime dtype {record.get('runtime_dtype')} "
                f"differs from production semantics {expected['expected_runtime_dtype']}"
            )
        if record.get("timestep") != expected["scheduler_timestep"]:
            raise GlobalStopError(f"GLOBAL STOP: {expected['boundary']} scheduler timestep mapping is ambiguous")
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


def _run_within_step_trace(omni: Any, config: dict[str, Any], source: Any, state: np.ndarray, name: str, root: Path, selected_step: int) -> list[dict[str, Any]]:
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
        "within_step_probe": {
            "artifact_dir": str(root / "phase2" / name.lower() / "probe"), "request_label": name.lower(),
            "selected_local_step": local_step, "selected_absolute_step": selected_step,
        },
    }
    outputs = omni.generate({"prompt": source.prompt}, sampling)
    _, output = v3.normalize_video(outputs)
    probe = output.custom_output.get("within_step_probe")
    if not isinstance(probe, dict):
        raise GlobalStopError("GLOBAL STOP: within-step probe output is missing")
    allowed = ["latent_entering_step", "transformer_input", "guidance_combined_output", "scheduler_input", "scheduler_output"]
    records = probe.get("records", [])
    if [row.get("boundary") for row in records] != allowed or probe.get("unavailable_boundaries") != ["transformer_raw_output"]:
        raise GlobalStopError("GLOBAL STOP: Phase-2 did not record the actual Wan boundary set")
    expected_timestep = single_flip.scheduler_timesteps_numpy(config)[selected_step]
    if any(
        int(record.get("step_idx", -1)) != local_step
        or int(record.get("absolute_step", -1)) != selected_step
        or record.get("timestep") != expected_timestep
        for record in records
    ):
        raise GlobalStopError("GLOBAL STOP: Phase-2 probe step/timestep mapping differs from the frozen scheduler")
    result = []
    for record in records:
        expected_runtime_dtype = (
            EXPECTED_INPUT_RUNTIME_DTYPE
            if local_step == 0 and record["boundary"] in ("latent_entering_step", "scheduler_input")
            else EXPECTED_RUNTIME_DTYPE
        )
        if record.get("runtime_dtype") != expected_runtime_dtype:
            raise GlobalStopError(f"GLOBAL STOP: Phase-2 {record['boundary']} runtime dtype differs from production semantics")
        tensor = torch.load(Path(record["latent_path"]), map_location="cpu").detach().cpu().float().numpy()
        storage_dtype = np.dtype(tensor.dtype).newbyteorder("<").str
        result.append({
            "boundary": record["boundary"], "absolute_step": int(record["absolute_step"]), "timestep": record["timestep"],
            "phase1_entry_boundary": "input" if local_step == 0 else f"after_step_{local_step:03d}",
            "runtime_dtype": record["runtime_dtype"], "storage_dtype": storage_dtype,
            "artifact": save_tensor(root, f"phase2/artifacts/{name}/{record['boundary']}.npy", tensor, runtime_semantics=record["runtime_dtype"]),
        })
    return result


def run_phase2(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    selected = config["phase2"]["selected_step"]
    if selected is None:
        raise GlobalStopError("GLOBAL STOP: phase 2 requires an explicitly frozen selected_step")
    selection_mapping = phase2_selection_mapping(config, manifest, int(selected))
    data = derive_anchor(config); omni = _build_omni(config, args)
    try:
        traces = {name: _run_within_step_trace(omni, config, data["source"], state, name, root, int(selected)) for name, state in {"CLEAN": data["clean"], "PLUS1": data["plus1"], "HISTORICAL_PLUS14": data["historical"]}.items()}
        document = {"provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "selected_step": int(selected), "selection_mapping": selection_mapping, "traces": traces, "unavailable_boundaries": ["transformer_raw_output"]}
        atomic_json(root / "phase2" / "phase2_manifest.json", document)
        return {"mode": "phase2", "selected_step": int(selected), "boundaries": len(next(iter(traces.values())))}
    finally:
        single_flip._shutdown(omni)


def run_analyze_phase2(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    manifest, prov = require_cpu(root, config_path, config); require_preflight(root, manifest, prov)
    path = root / "phase2" / "phase2_manifest.json"
    if not path.exists():
        raise GlobalStopError("GLOBAL STOP: phase-2 trace manifest is missing")
    trace = json.loads(path.read_text())
    if trace.get("provenance_hash") != prov["provenance_hash"] or trace.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise GlobalStopError("GLOBAL STOP: phase-2 trace provenance mismatch")
    selected = config["phase2"]["selected_step"]
    if selected is None or int(trace.get("selected_step", -1)) != int(selected):
        raise GlobalStopError("GLOBAL STOP: phase-2 trace selected_step differs from the explicitly frozen config step")
    if trace.get("selection_mapping") != phase2_selection_mapping(config, manifest, int(selected)):
        raise GlobalStopError("GLOBAL STOP: phase-2 selection mapping differs from the frozen Phase-1 boundary mapping")
    expected = ["latent_entering_step", "transformer_input", "guidance_combined_output", "scheduler_input", "scheduler_output"]
    adapted = {"expected_boundaries": expected, "traces": trace.get("traces", {})}
    result = analyze_trace(root, adapted)
    result["selected_step"] = trace["selected_step"]
    result["unavailable_boundaries"] = trace["unavailable_boundaries"]
    atomic_json(root / "phase2" / "phase2_analysis.json", result)
    return {"mode": "analyze-phase2", "selected_step": result["selected_step"], "pairwise_rows": len(result["pairwise_rows"])}


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
