#!/usr/bin/env python3
"""Confirmatory kill test for the "Forkable Generative Execution" seed (Wan2.2, exact Euler).

Independent replication across THREE NEW seeds of the structure observed in the exploratory
seed-1234 smoke (which stays frozen as INCONCLUSIVE and is never combined with this run):

    OLD -> NEW conditioning forks stay NEW-responsive after 25-37.5% OLD prefix reuse
    (k=10, 15) and later forks (k=20, 25) shift toward OLD commitment.

Everything scientific is reused from the audited smoke implementation
(``experiments/video_trajectory_fork_killtest.py``): exact Euler resume, baseline runs,
same-conditioning exactness controls, fork execution, CLIP-style descriptive scoring.
This module adds only: the frozen preregistration, three-seed orchestration, the
baseline-informativeness rule, the five-level label scale, blinded qualitative review,
and the frozen GO / NO-GO / INCONCLUSIVE / INVALID classifier.

Phases (each refuses to run unless the previous one is frozen and valid):
    cpu        write + hash preregistration, provenance, expected key set (no GPU)
    baselines  Phase 1: OLD/NEW full baselines for all 3 seeds, baseline judgment template
    controls   Phase 2: OLD->OLD exactness at k=10,15,20,25 for informative seeds
    forks      Phase 3: OLD->NEW at k=10,15,20,25 for informative seeds
    blind      Phase 4: anonymised fork samples + label template
    analyze    Phase 5: frozen classifier, exactly once

No mechanism, scheduler, correction, blending, or speedup claim is made or implied.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_trajectory_fork_killtest as smoke  # noqa: E402

EXPERIMENT_VERSION = "video-trajectory-fork-confirmatory-v1"
NAMESPACE = "video_trajectory_fork_confirmatory"
DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_confirmatory_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE
MODES = ("cpu", "baselines", "controls", "forks", "blind", "analyze")
SEEDS = (2345, 3456, 4567)
EXPLORATORY_SEED = 1234
SWITCHES = (10, 15, 20, 25)
OLD_PROMPT = "A red sports car driving on a snowy road, cinematic video"
NEW_PROMPT = "A blue sports car driving on a snowy road, cinematic video"
OLD_CONCEPT = "red sports car"
NEW_CONCEPT = "blue sports car"
LABELS = ("NEW", "MIXED_NEW_DOMINANT", "MIXED_BALANCED", "MIXED_OLD_DOMINANT", "OLD")
ORDINAL = {"NEW": 2, "MIXED_NEW_DOMINANT": 1, "MIXED_BALANCED": 0, "MIXED_OLD_DOMINANT": -1, "OLD": -2}
EARLY_NEW_LABELS = ("NEW", "MIXED_NEW_DOMINANT")
FRAME_POSITIONS = ("first", "q25", "middle", "q75", "last")
BASELINE_CAPTURE = [0, *SWITCHES, 40]
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_confirmatory_config.yaml",
    "experiments/run_video_trajectory_fork_confirmatory_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = smoke.FORBIDDEN_OUTPUT_PARTS + ("video_trajectory_fork_killtest",)

GateError = smoke.GateError
canonical_json = smoke.canonical_json
sha256_bytes = smoke.sha256_bytes
sha256_file = smoke.sha256_file
array_sha256 = smoke.array_sha256
atomic_json = smoke.atomic_json
read_csv = smoke.read_csv


# --------------------------------------------------------------------------------------
# configuration / isolation
# --------------------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("experiment_version changed")
    if config["model"] != smoke.EXPECTED_MODEL:
        raise ValueError(f"Model must remain {smoke.EXPECTED_MODEL}")
    scheduler = config["scheduler"]
    if scheduler["name"] != smoke.EXPECTED_SCHEDULER or scheduler["sample_solver"] != "euler" or float(scheduler["flow_shift"]) != 12.0:
        raise ValueError("The trusted Wan Euler scheduler configuration is frozen")
    expected_generation = {"height": 480, "width": 832, "num_frames": 33, "num_inference_steps": 40, "guidance_scale": 4.0, "fps": 16.0, "boundary_ratio": 0.875}
    for key, expected in expected_generation.items():
        if config["generation"][key] != expected:
            raise ValueError(f"Frozen generation field changed: {key}={config['generation'][key]!r}")
    if tuple(int(value) for value in config["seeds"]) != SEEDS:
        raise ValueError(f"Confirmatory seeds are frozen to {SEEDS}")
    if EXPLORATORY_SEED in config["seeds"]:
        raise ValueError("The exploratory seed 1234 must not be reused")
    if tuple(int(value) for value in config["switch_steps"]) != SWITCHES:
        raise ValueError(f"Switch points are frozen to {SWITCHES}")
    prompts = config["prompts"]
    if prompts["old"] != OLD_PROMPT or prompts["new"] != NEW_PROMPT:
        raise ValueError("Prompt pair changed")
    if prompts["old_concept"] != OLD_CONCEPT or prompts["new_concept"] != NEW_CONCEPT:
        raise ValueError("Concept strings changed")
    if tuple(config["labels"]["scale"]) != LABELS or {k: int(v) for k, v in config["labels"]["ordinal"].items()} != ORDINAL:
        raise ValueError("Qualitative label scale / ordinal coding changed")
    gate = config["gate"]
    if tuple(gate["early_new_labels"]) != EARLY_NEW_LABELS or gate["early_new_steps"] != [10, 15] or gate["late_shift_reference_step"] != 15 or gate["late_shift_candidate_steps"] != [20, 25]:
        raise ValueError("EARLY_NEW / LATE_SHIFT definitions changed")
    if gate["go_min_frontier_seeds"] != 2 or gate["go_min_early_new_seeds"] != 2 or gate["no_go_min_early_new_failures"] != 2 or gate["invalid_min_uninformative_baselines"] != 2:
        raise ValueError("GO / NO-GO / INVALID thresholds changed")
    metric = config["concept_metric"]
    if metric["model"] != "openai/clip-vit-base-patch32" or int(metric["frame_count"]) != 8 or float(metric["sign_only_threshold"]) != 0.0:
        raise ValueError("Descriptive concept metric configuration changed")
    if tuple(config["frame_positions"]) != FRAME_POSITIONS:
        raise ValueError("Deterministic frame positions changed")
    if config.get("blinded_review") is not True:
        raise ValueError("Blinded qualitative review is preregistered")


def validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT / "results")
    except ValueError as error:
        raise ValueError("Output must be under this repository's results directory") from error
    if not relative.parts or relative.parts[0] != NAMESPACE:
        raise ValueError(f"Output must use the isolated {NAMESPACE} namespace")
    if any(part in str(relative) for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError("Trusted prior-result namespace cannot be used")


def seed_family(config: dict[str, Any], seed: int) -> dict[str, Any]:
    prompts = config["prompts"]
    return {
        "id": f"red_to_blue_seed{int(seed)}",
        "severity": "confirmatory",
        "seed": int(seed),
        "old_concept": prompts["old_concept"],
        "new_concept": prompts["new_concept"],
        "old_prompt": prompts["old"],
        "new_prompt": prompts["new"],
    }


def seed_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Per-seed view consumed by the reused smoke machinery (which reads config['seed'])."""
    adapted = json.loads(json.dumps(config))
    adapted["seed"] = int(seed)
    adapted["generation"]["switch_steps"] = list(smoke.EXPECTED_SWITCHES)  # scheduler plan adapter only
    return adapted


# --------------------------------------------------------------------------------------
# expected key set
# --------------------------------------------------------------------------------------
def expected_keys(informative_seeds: tuple[int, ...] | list[int] | None = None) -> set[tuple[int, str, int]]:
    """(seed, trajectory_type, switch_step). Baselines for every seed; controls/forks only for informative seeds."""
    keys: set[tuple[int, str, int]] = set()
    for seed in SEEDS:
        keys.add((seed, "old_baseline", -1))
        keys.add((seed, "new_baseline", -1))
    for seed in (SEEDS if informative_seeds is None else informative_seeds):
        for step in SWITCHES:
            keys.add((seed, "same_condition", step))
            keys.add((seed, "fork_new", step))
    return keys


def row_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["seed"]), str(row["trajectory_type"]), int(row["switch_step"])


def validate_key_set(rows: list[dict[str, Any]], expected: set[tuple[int, str, int]]) -> None:
    keys = [row_key(row) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    actual = set(keys)
    if duplicates or actual != expected:
        raise GateError(f"Result key-set mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}, duplicates={duplicates}")


# --------------------------------------------------------------------------------------
# provenance / preregistration
# --------------------------------------------------------------------------------------
def build_provenance(config_path: Path) -> dict[str, Any]:
    paths = [REPO_ROOT / value for value in TRUSTED_SOURCE_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise GateError(f"Provenance inputs missing: {missing}")
    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        commit, status = "UNKNOWN", ["git status unavailable"]
    relevant = [line for line in status if any(name in line for name in TRUSTED_SOURCE_FILES)]
    document = {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status": status,
        "relevant_git_status": relevant,
        "source_sha256": hashes,
        "config_sha256": sha256_file(config_path),
    }
    basis = {key: document[key] for key in ("git_commit", "relevant_git_status", "source_sha256", "config_sha256")}
    document["provenance_hash"] = sha256_bytes(canonical_json(basis))
    return document


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan = smoke.scheduler_plan(seed_config(config, SEEDS[0]))
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "title": "Video Trajectory Fork Confirmatory Kill Test",
        "hypothesis": (
            "Across independent seeds, OLD->NEW conditioning forks of Wan2.2 exact-Euler trajectories stay "
            "NEW-responsive after 25-37.5% OLD prefix reuse (k=10,15) and shift toward OLD commitment at later forks (k=20,25)."
        ),
        "prior_exploratory_result": {"seed": EXPLORATORY_SEED, "decision": "INCONCLUSIVE", "rule": "original smoke rule; frozen; never combined with this run"},
        "model": config["model"],
        "scheduler": config["scheduler"],
        "generation": config["generation"],
        "prompts": config["prompts"],
        "seeds": list(SEEDS),
        "seed_policy": "three new seeds fixed before any GPU execution; no replacement, no seed shopping; an uninformative baseline pair stays in the denominator",
        "switch_steps": list(SWITCHES),
        "prefix_reuse_fraction": {str(step): step / 40.0 for step in SWITCHES},
        "labels": {"scale": list(LABELS), "ordinal": ORDINAL, "definitions": config["labels"]["definitions"]},
        "frame_positions": list(FRAME_POSITIONS),
        "blinded_review": True,
        "baseline_informativeness": config["baseline_informativeness"],
        "gate": config["gate"],
        "metric_disagreement": config["metric_disagreement"],
        "concept_metric": config["concept_metric"],
        "controls": config["controls"],
        "expected_keys_max": sorted([list(key) for key in expected_keys()]),
        "expected_trajectories_max": len(expected_keys()),
        "scheduler_plan": plan,
        "expert_by_switch": {str(step): smoke.expert_metadata(seed_config(config, SEEDS[0]), plan, step) for step in SWITCHES},
        "source_commit": provenance["git_commit"],
        "provenance_hash": provenance["provenance_hash"],
        "config_sha256": provenance["config_sha256"],
        "claim_boundary": "No scheduler, serving mechanism, caching, correction, speedup, or cross-model claim; CLIP scores are descriptive and uncalibrated.",
    }


def preregistration_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "preregistration.json", output_dir / "preregistration.sha256"


def require_preregistration(output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    prereg_path, sha_path = preregistration_paths(output_dir)
    if not prereg_path.exists() or not sha_path.exists():
        raise GateError("Run the cpu phase first; preregistration is missing")
    recorded = sha_path.read_text().split()[0]
    if sha256_file(prereg_path) != recorded:
        raise GateError("preregistration.json was modified after it was hashed")
    document = json.loads(prereg_path.read_text())
    if document.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Code/config provenance changed after preregistration; use a fresh output namespace")
    frozen = json.loads((output_dir / "provenance.json").read_text())
    if frozen.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("provenance.json does not match the current source")
    return document


def run_cpu(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    prereg_path, sha_path = preregistration_paths(output_dir)
    if prereg_path.exists():
        existing = json.loads(prereg_path.read_text())
        if existing.get("provenance_hash") != provenance["provenance_hash"]:
            raise GateError("Output namespace already holds a preregistration from different code/config; use a fresh namespace")
        if (output_dir / "baseline_results.csv").exists() or (output_dir / "raw_results.csv").exists():
            return {"mode": "cpu", "status": "ALREADY_FROZEN", "preregistration_sha256": sha_path.read_text().split()[0]}
    document = build_preregistration(config, provenance)
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(prereg_path, document)
    digest = sha256_file(prereg_path)
    sha_path.write_text(f"{digest}  preregistration.json\n")
    return {"mode": "cpu", "status": "FROZEN", "preregistration_sha256": digest, "expected_trajectories_max": document["expected_trajectories_max"], "seeds": list(SEEDS), "switch_steps": list(SWITCHES)}


# --------------------------------------------------------------------------------------
# frames / csv helpers
# --------------------------------------------------------------------------------------
def frame_indices(frame_count: int) -> dict[str, int]:
    return {"first": 0, "q25": (frame_count - 1) // 4, "middle": frame_count // 2, "q75": (3 * (frame_count - 1)) // 4, "last": frame_count - 1}


def save_frames(video: np.ndarray, directory: Path, stem: str) -> dict[str, str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, index in frame_indices(len(video)).items():
        path = directory / f"{stem}_{name}.png"
        Image.fromarray(video[index, ..., :3]).save(path)
        out[name] = str(path)
    return out


CSV_FIELDS = tuple(smoke.RAW_FIELDS) + ("confirmatory_phase",)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))
    temporary.replace(path)


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in existing + incoming:
        key = row_key(row)
        previous = merged.get(key)
        if previous is not None:
            for field in ("provenance_hash", "final_latent_hash", "video_hash", "resume_input_hash"):
                if str(previous.get(field)) != str(row.get(field)):
                    raise GateError(f"Conflicting completed rows for {key}")
            continue
        merged[key] = row
    return sorted(merged.values(), key=row_key)


# --------------------------------------------------------------------------------------
# Phase 1: baselines + informativeness
# --------------------------------------------------------------------------------------
def run_baselines(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    initial_identity: dict[str, Any] = {}
    try:
        for seed in SEEDS:
            run_config = seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_row, old_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, family, which="old", capture_steps=list(BASELINE_CAPTURE))
            new_row, new_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, family, which="new", capture_steps=[0, 40])
            old_initial = smoke._probe_array(smoke._record_by_step(old_meta, 0))
            new_initial = smoke._probe_array(smoke._record_by_step(new_meta, 0))
            identical = array_sha256(old_initial) == array_sha256(new_initial)
            initial_identity[str(seed)] = {"identical_initial_latent": identical, "initial_latent_hash": array_sha256(old_initial)}
            for row in (old_row, new_row):
                row["confirmatory_phase"] = "baseline"
                video = np.load(row["final_video_npy"], allow_pickle=False)
                save_frames(video, output_dir / "qualitative" / family["id"], row["trajectory_type"])
            rows.extend((old_row, new_row))
    finally:
        omni.shutdown()
    validate_key_set(rows, {key for key in expected_keys() if key[1] in ("old_baseline", "new_baseline")})
    write_csv(output_dir / "baseline_results.csv", rows)
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows) if (output_dir / "raw_results.csv").exists() else rows)
    automated = {}
    for seed in SEEDS:
        old = next(r for r in rows if row_key(r) == (seed, "old_baseline", -1))
        new = next(r for r in rows if row_key(r) == (seed, "new_baseline", -1))
        automated[str(seed)] = {
            "old_new_minus_old": float(old["new_minus_old"]),
            "new_new_minus_old": float(new["new_minus_old"]),
            "automated_sign_ok": float(old["new_minus_old"]) < 0.0 < float(new["new_minus_old"]),
            "ssim_old_vs_new": None,
        }
    atomic_json(output_dir / "baseline_automated_evidence.json", {"provenance_hash": provenance["provenance_hash"], "seeds": automated, "initial_identity": initial_identity})
    template_path = output_dir / "baseline_judgment_template.json"
    if not template_path.exists():
        atomic_json(template_path, {
            "provenance_hash": provenance["provenance_hash"],
            "fork_outcomes_examined": False,
            "instructions": "Inspect ONLY old_baseline/new_baseline videos and the five fixed frames per seed. OLD must clearly show a RED sports car, NEW a BLUE sports car, and the pair must be clearly different. Do not open any fork output.",
            "seeds": {str(seed): {"old_matches_expected": None, "new_matches_expected": None, "clearly_different": None, "notes": ""} for seed in SEEDS},
        })
    return {"mode": "baselines", "rows": len(rows), "initial_identity": initial_identity, "automated": automated, "next": "fill baseline_judgment.json from the template, then run controls"}


def evaluate_baseline_informativeness(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    """Frozen rule: informative iff manual booleans all true AND automated CLIP signs agree (old<0<new).
    A manual/automated disagreement is recorded and the seed is BASELINE_UNINFORMATIVE (conservative)."""
    judgment_path = output_dir / "baseline_judgment.json"
    if not judgment_path.exists():
        raise GateError("Fill baseline_judgment.json from baseline_judgment_template.json before running controls")
    judgment = json.loads(judgment_path.read_text())
    if judgment.get("provenance_hash") != provenance_hash:
        raise GateError("Baseline judgment provenance mismatch")
    if judgment.get("fork_outcomes_examined") is not False:
        raise GateError("Baseline judgment must be frozen before any fork outcome is examined")
    automated = json.loads((output_dir / "baseline_automated_evidence.json").read_text())
    if automated.get("provenance_hash") != provenance_hash:
        raise GateError("Baseline automated evidence provenance mismatch")
    result: dict[str, Any] = {}
    for seed in SEEDS:
        entry = judgment.get("seeds", {}).get(str(seed))
        if not isinstance(entry, dict):
            raise GateError(f"Baseline judgment missing seed {seed}")
        flags = {key: entry.get(key) for key in ("old_matches_expected", "new_matches_expected", "clearly_different")}
        if any(not isinstance(value, bool) for value in flags.values()):
            raise GateError(f"Baseline judgment for seed {seed} must contain explicit booleans")
        manual_ok = all(flags.values())
        auto = automated["seeds"][str(seed)]
        auto_ok = bool(auto["automated_sign_ok"])
        identical = bool(automated["initial_identity"][str(seed)]["identical_initial_latent"])
        informative = manual_ok and auto_ok and identical
        result[str(seed)] = {
            "manual": flags,
            "manual_ok": manual_ok,
            "automated_sign_ok": auto_ok,
            "identical_initial_latent": identical,
            "metric_disagreement": manual_ok != auto_ok,
            "status": "BASELINE_INFORMATIVE" if informative else "BASELINE_UNINFORMATIVE",
            "reason": None if informative else ("manual" if not manual_ok else "automated_sign" if not auto_ok else "initial_latent"),
        }
    informative_seeds = [seed for seed in SEEDS if result[str(seed)]["status"] == "BASELINE_INFORMATIVE"]
    return {"seeds": result, "informative_seeds": informative_seeds, "uninformative_count": len(SEEDS) - len(informative_seeds)}


# --------------------------------------------------------------------------------------
# Phase 2: hard controls
# --------------------------------------------------------------------------------------
def _old_records(output_dir: Path, family_id: str) -> dict[int, dict[str, Any]]:
    metadata = json.loads((output_dir / "rows" / family_id / "old_baseline_metadata.json").read_text())["worker_trajectory_probe"]
    return {int(row["step_index"]): row for row in metadata["records"]}


def run_controls(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    informativeness = evaluate_baseline_informativeness(output_dir, provenance["provenance_hash"])
    preflight: dict[str, Any] = {"provenance_hash": provenance["provenance_hash"], "baseline_informativeness": informativeness, "seeds": {}}
    if informativeness["uninformative_count"] >= int(config["gate"]["invalid_min_uninformative_baselines"]):
        preflight["status"] = "INVALID_BASELINES"
        preflight["reason"] = f"{informativeness['uninformative_count']} of {len(SEEDS)} baseline pairs are uninformative; forks are not run"
        atomic_json(output_dir / "preflight.json", preflight)
        raise GateError(preflight["reason"])
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in informativeness["informative_seeds"]:
            run_config = seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, family["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, family["id"], "new_baseline")
            records = _old_records(output_dir, family["id"])
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            seed_rows = []
            for step in SWITCHES:
                row = smoke._execute_one(
                    omni, evaluator, run_config, provenance, scheduler, output_dir, family,
                    label=f"same_condition_k{step:02d}", trajectory_type="same_condition", switch_step=step,
                    prompt=family["old_prompt"], input_tensor=smoke._load_probe_tensor(records[step]["latent_path"]),
                    initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=old_ref,
                )
                row["confirmatory_phase"] = "control"
                seed_rows.append(row)
            rows.extend(seed_rows)
            preflight["seeds"][str(seed)] = {
                "same_condition_exact_all_switches": all(r["control_exact"] is True for r in seed_rows),
                "old_checkpoint_hashes": {str(step): array_sha256(smoke._probe_array(records[step])) for step in SWITCHES},
                "resume_inputs_match_checkpoints": all(r["resume_input_hash"] == array_sha256(smoke._probe_array(records[int(r["switch_step"])])) for r in seed_rows),
                "scheduler_euler": all(str(r["scheduler_class"]).endswith(smoke.EXPECTED_SCHEDULER) for r in seed_rows),
                "expert_at_switch": {str(int(r["switch_step"])): r["expert_at_switch"] for r in seed_rows},
            }
    finally:
        omni.shutdown()
    validate_key_set(rows, {key for key in expected_keys(informativeness["informative_seeds"]) if key[1] == "same_condition"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    c1 = all(entry["same_condition_exact_all_switches"] and entry["resume_inputs_match_checkpoints"] and entry["scheduler_euler"] for entry in preflight["seeds"].values())
    preflight["status"] = "PASS" if c1 else "INVALID"
    preflight["same_condition_exact_all_informative_seeds"] = c1
    atomic_json(output_dir / "preflight.json", preflight)
    if not c1:
        raise GateError("C1 same-conditioning exactness failed; INVALID / STOP")
    return {"mode": "controls", "status": preflight["status"], "informative_seeds": informativeness["informative_seeds"], "rows": len(rows)}


def require_controls(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if not path.exists():
        raise GateError("Controls have not completed")
    preflight = json.loads(path.read_text())
    if preflight.get("provenance_hash") != provenance_hash or preflight.get("status") != "PASS":
        raise GateError(f"Control gate not passed (status={preflight.get('status')})")
    informative = preflight["baseline_informativeness"]["informative_seeds"]
    if set(map(str, informative)) != set(preflight["seeds"]):
        raise GateError("Control gate covers a different seed set than the informative baselines")
    for seed, entry in preflight["seeds"].items():
        if not (entry["same_condition_exact_all_switches"] and entry["resume_inputs_match_checkpoints"] and entry["scheduler_euler"]):
            raise GateError(f"Control gate failed for seed {seed}")
    return preflight


# --------------------------------------------------------------------------------------
# Phase 3: confirmatory forks
# --------------------------------------------------------------------------------------
def run_forks(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(seed) for seed in preflight["baseline_informativeness"]["informative_seeds"]]
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in informative:
            run_config = seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, family["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, family["id"], "new_baseline")
            records = _old_records(output_dir, family["id"])
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            for step in SWITCHES:
                expected_checkpoint = preflight["seeds"][str(seed)]["old_checkpoint_hashes"][str(step)]
                checkpoint = smoke._load_probe_tensor(records[step]["latent_path"])
                if array_sha256(checkpoint.float().numpy()) != expected_checkpoint:
                    raise GateError(f"OLD checkpoint for seed {seed} k={step} changed since controls")
                row = smoke._execute_one(
                    omni, evaluator, run_config, provenance, scheduler, output_dir, family,
                    label=f"fork_new_k{step:02d}", trajectory_type="fork_new", switch_step=step,
                    prompt=family["new_prompt"], input_tensor=checkpoint, initial_hash=initial_hash,
                    old_reference=old_ref, new_reference=new_ref, exact_reference=None,
                )
                row["confirmatory_phase"] = "fork"
                rows.append(row)
    finally:
        omni.shutdown()
    validate_key_set(rows, {key for key in expected_keys(informative) if key[1] == "fork_new"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    return {"mode": "forks", "rows": len(rows), "informative_seeds": informative, "next": "run blind, label blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 4: blinded review
# --------------------------------------------------------------------------------------
def blinded_assignment(fork_keys: list[tuple[int, int]], preregistration_sha256: str) -> dict[str, tuple[int, int]]:
    """Deterministic anonymisation: sample ids are a shuffle seeded from the preregistration hash."""
    ordered = sorted(fork_keys)
    rng = random.Random(int(preregistration_sha256[:16], 16))
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    ids = [f"sample_{index:02d}" for index in range(1, len(shuffled) + 1)]
    return dict(zip(ids, shuffled, strict=True))


def run_blind(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(seed) for seed in preflight["baseline_informativeness"]["informative_seeds"]]
    rows = [row for row in read_csv(output_dir / "raw_results.csv") if row["trajectory_type"] == "fork_new"]
    validate_key_set(rows, {key for key in expected_keys(informative) if key[1] == "fork_new"})
    if (output_dir / "blinded_labels.json").exists():
        raise GateError("blinded_labels.json already exists; the blinded set must not be regenerated after labelling")
    prereg_sha = (output_dir / "preregistration.sha256").read_text().split()[0]
    mapping = blinded_assignment([(int(r["seed"]), int(r["switch_step"])) for r in rows], prereg_sha)
    by_key = {(int(r["seed"]), int(r["switch_step"])): r for r in rows}
    blind_dir = output_dir / "qualitative" / "blinded"
    blind_dir.mkdir(parents=True, exist_ok=True)
    manifest_samples = {}
    for sample_id, (seed, step) in mapping.items():
        row = by_key[(seed, step)]
        video = np.load(row["final_video_npy"], allow_pickle=False)
        frames = save_frames(video, blind_dir, sample_id)
        mp4 = blind_dir / f"{sample_id}.mp4"
        shutil.copyfile(row["final_video_mp4"], mp4)
        manifest_samples[sample_id] = {"mp4": str(mp4), "frames": frames, "video_hash": row["video_hash"]}
    sealed = {"provenance_hash": provenance["provenance_hash"], "mapping": {k: {"seed": v[0], "switch_step": v[1]} for k, v in mapping.items()}}
    atomic_json(output_dir / "blinded_mapping.sealed.json", sealed)
    atomic_json(output_dir / "blinded_manifest.json", {
        "provenance_hash": provenance["provenance_hash"],
        "sealed_mapping_sha256": sha256_file(output_dir / "blinded_mapping.sealed.json"),
        "instructions": "Label each sample from its five fixed frames and full video without opening blinded_mapping.sealed.json. Judge the whole video, not a favourable frame.",
        "labels": list(LABELS),
        "definitions": config["labels"]["definitions"],
        "samples": manifest_samples,
    })
    if not (output_dir / "blinded_labels_template.json").exists():
        atomic_json(output_dir / "blinded_labels_template.json", {
            "provenance_hash": provenance["provenance_hash"],
            "mapping_revealed_before_labelling": False,
            "labels": {sample_id: {"label": None, "notes": ""} for sample_id in sorted(mapping)},
        })
    return {"mode": "blind", "samples": len(mapping), "next": "fill blinded_labels.json (labels only), then analyze"}


# --------------------------------------------------------------------------------------
# Phase 5: frozen classifier
# --------------------------------------------------------------------------------------
def early_new(labels: dict[int, str]) -> bool:
    return all(labels[step] in EARLY_NEW_LABELS for step in (10, 15))


def late_shift(labels: dict[int, str]) -> bool:
    return any(ORDINAL[labels[step]] < ORDINAL[labels[15]] for step in (20, 25))


def frontier_replication(labels: dict[int, str]) -> bool:
    return early_new(labels) and late_shift(labels)


def monotone_non_increasing(labels: dict[int, str]) -> bool:
    scores = [ORDINAL[labels[step]] for step in SWITCHES]
    return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def metric_disagreement(label: str, new_minus_old: float) -> bool:
    if label in ("NEW", "MIXED_NEW_DOMINANT"):
        return new_minus_old < 0.0
    if label in ("OLD", "MIXED_OLD_DOMINANT"):
        return new_minus_old > 0.0
    return False


def classify(
    seed_labels: dict[int, dict[int, str]],
    seed_margins: dict[int, dict[int, float]],
    informative_seeds: list[int],
    controls_pass: bool,
) -> dict[str, Any]:
    """Frozen confirmatory gate over the three preregistered seeds.

    Uninformative seeds and seeds with any METRIC_DISAGREEMENT never count as successes and never count as
    interpretable failures; they stay in the denominator and push toward INCONCLUSIVE.
    """
    per_seed: dict[str, Any] = {}
    frontier = early = failures = 0
    if len(SEEDS) - len(informative_seeds) >= 2:
        decision = "INVALID"
        rationale = f"{len(SEEDS) - len(informative_seeds)} of {len(SEEDS)} baseline pairs uninformative"
    elif not controls_pass:
        decision, rationale = "INVALID", "Hard controls failed"
    else:
        decision = rationale = None
    for seed in SEEDS:
        if seed not in informative_seeds:
            per_seed[str(seed)] = {"status": "BASELINE_UNINFORMATIVE", "counts_as_success": False, "counts_as_failure": False}
            continue
        labels = seed_labels[seed]
        if set(labels) != set(SWITCHES) or any(label not in LABELS for label in labels.values()):
            raise GateError(f"Seed {seed} labels incomplete or invalid: {labels}")
        margins = seed_margins[seed]
        disagreements = [step for step in SWITCHES if metric_disagreement(labels[step], float(margins[step]))]
        e, l = early_new(labels), late_shift(labels)
        f = e and l
        clean = not disagreements
        per_seed[str(seed)] = {
            "status": "LABELLED",
            "labels": {str(step): labels[step] for step in SWITCHES},
            "ordinal": {str(step): ORDINAL[labels[step]] for step in SWITCHES},
            "conditioning_dominance_sequence": [ORDINAL[labels[step]] for step in SWITCHES],
            "monotone_non_increasing": monotone_non_increasing(labels),
            "EARLY_NEW": e,
            "LATE_SHIFT": l,
            "FRONTIER_REPLICATION": f,
            "metric_disagreement_steps": disagreements,
            "counts_as_success": clean and f,
            "counts_as_early_new": clean and e,
            "counts_as_failure": clean and not e,
        }
        frontier += int(clean and f)
        early += int(clean and e)
        failures += int(clean and not e)
    if decision is None:
        if frontier >= 2 and early >= 2:
            decision, rationale = "GO", f"{frontier}/3 seeds replicate the frontier and {early}/3 satisfy EARLY_NEW"
        elif failures >= 2:
            decision, rationale = "NO-GO", f"{failures}/3 seeds fail EARLY_NEW for interpretable reasons"
        else:
            reasons = []
            if frontier < 2:
                reasons.append(f"only {frontier}/3 seeds replicate the frontier")
            if any(v.get("metric_disagreement_steps") for v in per_seed.values()):
                reasons.append("metric/qualitative disagreement on at least one seed")
            if any(v["status"] == "BASELINE_UNINFORMATIVE" for v in per_seed.values()):
                reasons.append("one baseline pair uninformative")
            decision, rationale = "INCONCLUSIVE", "; ".join(reasons) or "evidence insufficient under the frozen rule"
    return {
        "decision": decision,
        "rationale": rationale,
        "per_seed": per_seed,
        "counts": {"baseline_informative": len(informative_seeds), "EARLY_NEW": early, "LATE_SHIFT": sum(1 for v in per_seed.values() if v.get("LATE_SHIFT")), "FRONTIER_REPLICATION": frontier, "denominator": len(SEEDS)},
        "ROUND4_ELIGIBLE": decision == "GO",
    }


def run_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen classifier runs exactly once")
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(seed) for seed in preflight["baseline_informativeness"]["informative_seeds"]]
    labels_path = output_dir / "blinded_labels.json"
    if not labels_path.exists():
        raise GateError("Fill blinded_labels.json from blinded_labels_template.json before analysis")
    labels_doc = json.loads(labels_path.read_text())
    if labels_doc.get("provenance_hash") != provenance["provenance_hash"] or labels_doc.get("mapping_revealed_before_labelling") is not False:
        raise GateError("Blinded labels provenance mismatch or mapping was revealed before labelling")
    manifest = json.loads((output_dir / "blinded_manifest.json").read_text())
    sealed_path = output_dir / "blinded_mapping.sealed.json"
    if sha256_file(sealed_path) != manifest["sealed_mapping_sha256"]:
        raise GateError("Sealed blinded mapping was modified")
    mapping = {k: (int(v["seed"]), int(v["switch_step"])) for k, v in json.loads(sealed_path.read_text())["mapping"].items()}
    if set(labels_doc.get("labels", {})) != set(mapping):
        raise GateError("Blinded labels do not cover exactly the blinded sample set")
    rows = read_csv(output_dir / "raw_results.csv")
    for row in rows:
        smoke._validate_result_artifacts(row, provenance["provenance_hash"])
    validate_key_set(rows, expected_keys(informative))
    forks = {(int(r["seed"]), int(r["switch_step"])): r for r in rows if r["trajectory_type"] == "fork_new"}
    seed_labels: dict[int, dict[int, str]] = {seed: {} for seed in informative}
    seed_margins: dict[int, dict[int, float]] = {seed: {} for seed in informative}
    qualitative = {}
    for sample_id, (seed, step) in mapping.items():
        entry = labels_doc["labels"][sample_id]
        label = entry.get("label") if isinstance(entry, dict) else None
        if label not in LABELS:
            raise GateError(f"Invalid or missing label for {sample_id}: {label!r}")
        row = forks[(seed, step)]
        if manifest["samples"][sample_id]["video_hash"] != row["video_hash"]:
            raise GateError(f"Blinded sample {sample_id} does not match the persisted fork video")
        seed_labels[seed][step] = label
        seed_margins[seed][step] = float(row["new_minus_old"])
        qualitative[f"{seed}:{step}"] = {"sample_id": sample_id, "label": label, "notes": entry.get("notes", "")}
    atomic_json(output_dir / "unblinded_mapping.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {k: {"seed": v[0], "switch_step": v[1]} for k, v in mapping.items()}, "revealed_after_labels": True})
    atomic_json(output_dir / "qualitative_judgment.json", {"provenance_hash": provenance["provenance_hash"], "blinded": True, "labels": qualitative})
    result = classify(seed_labels, seed_margins, informative, controls_pass=preflight["status"] == "PASS")
    table = []
    for seed in informative:
        for step in SWITCHES:
            row = forks[(seed, step)]
            table.append({
                "seed": seed, "k": step, "reuse_pct": 100.0 * step / 40.0, "label": seed_labels[seed][step], "ordinal": ORDINAL[seed_labels[seed][step]],
                "clip_new_minus_old": float(row["new_minus_old"]), "ssim_old": float(row["ssim_to_old"]), "ssim_new": float(row["ssim_to_new"]),
                "latent_mse_old": float(row["final_latent_mse_to_old"]), "latent_mse_new": float(row["final_latent_mse_to_new"]), "wall_s": float(row["wall_time_s"]),
                "expert_at_switch": row["expert_at_switch"], "remaining_high_noise_steps": row["remaining_high_noise_steps"], "remaining_low_noise_steps": row["remaining_low_noise_steps"],
            })
    summary = {
        "decision": result["decision"],
        "decision_rationale": result["rationale"],
        "ROUND4_ELIGIBLE": result["ROUND4_ELIGIBLE"],
        "hypothesis": prereg["hypothesis"],
        "prior_exploratory_result": prereg["prior_exploratory_result"],
        "seeds": list(SEEDS),
        "switch_steps": list(SWITCHES),
        "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0],
        "provenance_hash": provenance["provenance_hash"],
        "controls": {"preflight": preflight, "baseline_informativeness": preflight["baseline_informativeness"]},
        "per_seed": result["per_seed"],
        "counts": result["counts"],
        "fork_table": table,
        "claim_boundary": prereg["claim_boundary"],
        "trajectories_executed": len(rows),
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print_console_summary(output_dir, provenance, summary)
    return summary


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = ["# Video Trajectory Fork Confirmatory Kill Test", "", "## Decision", "", summary["decision"], "", "## Frozen hypothesis", "", summary["hypothesis"], "", "## Prior exploratory result", "",
             f"Previous seed {EXPLORATORY_SEED} smoke was INCONCLUSIVE under its original rule. It is not combined with the three new seeds for the confirmatory gate.", "", "## Controls", ""]
    pre = summary["controls"]["preflight"]
    lines.append(f"- Control status: {pre['status']}")
    for seed, entry in pre["baseline_informativeness"]["seeds"].items():
        lines.append(f"- Seed {seed}: {entry['status']}" + (f" ({entry['reason']})" if entry["reason"] else "") + f"; automated sign ok={entry['automated_sign_ok']}, identical initial latent={entry['identical_initial_latent']}")
    for seed, entry in pre["seeds"].items():
        lines.append(f"- Seed {seed}: same-conditioning exact at all k={entry['same_condition_exact_all_switches']}, resume inputs match checkpoints={entry['resume_inputs_match_checkpoints']}, Euler={entry['scheduler_euler']}")
    lines += ["", "## Independent seed results", "", "| seed | k | reuse % | qualitative label | ordinal | CLIP new-old | SSIM OLD | SSIM NEW | expert |", "|---|---|---|---|---|---|---|---|---|"]
    for row in summary["fork_table"]:
        lines.append(f"| {row['seed']} | {row['k']} | {row['reuse_pct']:.1f} | {row['label']} | {row['ordinal']:+d} | {row['clip_new_minus_old']:+.4f} | {row['ssim_old']:.3f} | {row['ssim_new']:.3f} | {row['expert_at_switch']} |")
    lines += ["", "| seed | EARLY_NEW | LATE_SHIFT | FRONTIER_REPLICATION | monotone non-increasing? | metric disagreement |", "|---|---|---|---|---|---|"]
    for seed, entry in summary["per_seed"].items():
        if entry["status"] != "LABELLED":
            lines.append(f"| {seed} | - | - | - | - | {entry['status']} |")
        else:
            lines.append(f"| {seed} | {entry['EARLY_NEW']} | {entry['LATE_SHIFT']} | {entry['FRONTIER_REPLICATION']} | {entry['monotone_non_increasing']} | {entry['metric_disagreement_steps'] or 'none'} |")
    c = summary["counts"]
    lines += ["", f"- baseline informative: {c['baseline_informative']} / 3", f"- EARLY_NEW: {c['EARLY_NEW']} / 3", f"- LATE_SHIFT: {c['LATE_SHIFT']} / 3", f"- FRONTIER_REPLICATION: {c['FRONTIER_REPLICATION']} / 3", "", "## Decision rationale", "", summary["decision_rationale"], "", "## Scientific interpretation", ""]
    if summary["decision"] == "GO":
        lines.append("Across at least two independent confirmatory seeds, Wan2.2 trajectories retained new-conditioning responsiveness after 25-37.5% of denoising had been computed under the old conditioning, while later forks shifted toward old-conditioning dominance. No claim is made about other models, serving speedup, optimal rollback, or a commitment mechanism.")
        lines.append("")
        lines.append("ROUND4_ELIGIBLE = true (no further GPU experiments were run; awaiting explicit approval).")
    elif summary["decision"] == "NO-GO":
        lines.append("The exploratory seed-specific frontier did not replicate sufficiently. The forkable-execution direction stops here; no rescue with more seeds, prompts, models, or mechanisms.")
        lines.append("")
        lines.append("ROUND4_ELIGIBLE = false")
    else:
        lines.append(f"No decision could be reached under the frozen rule: {summary['decision_rationale']}. The result is not automatically expanded.")
        lines.append("")
        lines.append("ROUND4_ELIGIBLE = false")
    lines += ["", "## Caveats", "", "- Prefix reuse is theoretical compute reuse, not measured serving speedup.", "- CLIP-style scores are descriptive and uncalibrated.", "- Expert regime (high-noise transformer for k<=25) is recorded per row; k=25 leaves 1 high-noise step, so any transition near k=25 is confounded with the Wan2.2 expert boundary.", f"- Qualitative labels were {'blinded' if (output_dir / 'unblinded_mapping.json').exists() else 'NOT blinded'}.", ""]
    (output_dir / "video_trajectory_fork_confirmatory.md").write_text("\n".join(lines))


def print_console_summary(output_dir: Path, provenance: dict[str, Any], summary: dict[str, Any]) -> None:
    c = summary["counts"]
    print(f"git_commit={provenance['git_commit']}")
    print(f"git_dirty={provenance['git_dirty']} relevant_status={provenance['relevant_git_status']}")
    print(f"preregistration_sha256={summary['preregistration_sha256']}")
    print(f"seeds={summary['seeds']} switch_steps={summary['switch_steps']}")
    print(f"old_prompt={OLD_PROMPT!r}\nnew_prompt={NEW_PROMPT!r}")
    print(f"gpu_trajectories_executed={summary['trajectories_executed']}")
    print(f"baseline_informative={c['baseline_informative']}/3 same_condition_exact={summary['controls']['preflight']['status']}")
    print(f"EARLY_NEW={c['EARLY_NEW']}/3 LATE_SHIFT={c['LATE_SHIFT']}/3 FRONTIER_REPLICATION={c['FRONTIER_REPLICATION']}/3")
    print(f"decision={summary['decision']} ROUND4_ELIGIBLE={summary['ROUND4_ELIGIBLE']}")
    print(f"files_created={[str(output_dir / name) for name in ('preregistration.json', 'preregistration.sha256', 'provenance.json', 'preflight.json', 'baseline_results.csv', 'raw_results.csv', 'blinded_manifest.json', 'blinded_labels.json', 'unblinded_mapping.json', 'qualitative_judgment.json', 'summary.json', 'video_trajectory_fork_confirmatory.md')]}")
    print("caveats=prefix fraction is not realized speedup; CLIP is descriptive; exploratory seed 1234 is not in the confirmatory denominator; k=25 borders the Wan2.2 expert switch")


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    validate_output_path(output_dir)
    if args.mode == "cpu":
        result = run_cpu(config, args.config, output_dir)
    elif args.mode == "baselines":
        result = run_baselines(config, args.config, output_dir, args)
    elif args.mode == "controls":
        result = run_controls(config, args.config, output_dir, args)
    elif args.mode == "forks":
        result = run_forks(config, args.config, output_dir, args)
    elif args.mode == "blind":
        result = run_blind(config, args.config, output_dir)
    else:
        result = run_analyze(config, args.config, output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ("fork_table", "per_seed", "controls")}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
