#!/usr/bin/env python3
"""Round 4B: static-K oracle vs trajectory-specific oracle gap (fatal screening, preregistered).

Question: for the fixed red->blue edit, how much compute does a request-level policy that is
allowed to pick the best static fork depth K (post hoc, for this exact edit) lose relative to a
policy that knows each trajectory's own latest safe fork depth?

Design (all frozen before GPU work):
    10 new seeds x K in {12, 15, 18, 21, 24}; OLD/NEW baselines per seed; one OLD->OLD
    exactness spot control per informative seed at K=18; OLD->NEW forks at every K;
    blinded five-level labels (same scale as the confirmatory experiment).

    success(i, K)  := label in {NEW, MIXED_NEW_DOMINANT}
    K_i*           := largest grid K such that every grid K' <= K succeeds (monotone cut; 0 if none)
    K_static(tau)  := largest grid K whose success rate over informative seeds >= tau (0 if none)
    G              := mean_i (K_i* - K_static(1.0)) / 40        [fraction of a full generation]

    NO-GO if G <= 0.09 (the measured Round 4A probe-pair cost), GO if G >= 0.15, otherwise WEAK;
    INVALID if a hard control fails or fewer than 8 seeds are baseline-informative.

No probe is used. No mechanism, scheduler, or speedup claim is made. The confirmatory and
Round 4A results are read only for context and never enter the gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_trajectory_fork_confirmatory as conf  # noqa: E402
from experiments import video_trajectory_fork_killtest as smoke  # noqa: E402

EXPERIMENT_VERSION = "video-trajectory-fork-oracle-gap-killtest-v1"
NAMESPACE = "video_trajectory_fork_oracle_gap_killtest"
DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_oracle_gap_killtest_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE
MODES = ("cpu", "baselines", "controls", "forks", "blind", "analyze")
SEEDS = (5678, 6789, 7890, 8901, 9012, 10123, 11234, 12345, 13456, 14567)
EXCLUDED_SEEDS = (1234, 2345, 3456, 4567)
KS = (12, 15, 18, 21, 24)
CONTROL_K = 18
TOTAL_STEPS = 40
OLD_PROMPT = conf.OLD_PROMPT
NEW_PROMPT = conf.NEW_PROMPT
LABELS = conf.LABELS
ORDINAL = conf.ORDINAL
SUCCESS_LABELS = ("NEW", "MIXED_NEW_DOMINANT")
TAU_PRIMARY = 1.0
TAU_DESCRIPTIVE = 0.9
PROBE_PAIR_COST_FRACTION = 0.09  # measured in Round 4A (mean OLD+NEW one-step probe wall / full generation)
NO_GO_MAX_GAP = 0.09
GO_MIN_GAP = 0.15
MIN_INFORMATIVE_SEEDS = 8
BASELINE_CAPTURE = [0, *KS, TOTAL_STEPS]
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_oracle_gap_killtest.py",
    "experiments/video_trajectory_fork_oracle_gap_killtest_config.yaml",
    "experiments/run_video_trajectory_fork_oracle_gap_killtest_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_oracle_gap_killtest.py",
    "experiments/video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = conf.FORBIDDEN_OUTPUT_PARTS + ("video_trajectory_fork_confirmatory", "video_trajectory_fork_probe_killtest")
GateError = smoke.GateError
canonical_json = smoke.canonical_json
sha256_bytes = smoke.sha256_bytes
sha256_file = smoke.sha256_file
array_sha256 = smoke.array_sha256
atomic_json = smoke.atomic_json
read_csv = smoke.read_csv
CSV_FIELDS = tuple(smoke.RAW_FIELDS) + ("round4b_phase",)


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
    if tuple(int(v) for v in config["seeds"]) != SEEDS:
        raise ValueError(f"Seeds are frozen to {SEEDS}")
    if any(seed in EXCLUDED_SEEDS for seed in config["seeds"]):
        raise ValueError("Exploratory/confirmatory seeds must not be reused")
    if tuple(int(v) for v in config["k_grid"]) != KS or int(config["control_k"]) != CONTROL_K:
        raise ValueError("K grid / control K are frozen")
    if config["prompts"]["old"] != OLD_PROMPT or config["prompts"]["new"] != NEW_PROMPT:
        raise ValueError("Prompt pair changed")
    if tuple(config["labels"]["scale"]) != LABELS or {k: int(v) for k, v in config["labels"]["ordinal"].items()} != ORDINAL:
        raise ValueError("Label scale / ordinal changed")
    oracle = config["oracle"]
    if tuple(oracle["success_labels"]) != SUCCESS_LABELS or float(oracle["tau_primary"]) != TAU_PRIMARY or float(oracle["tau_descriptive"]) != TAU_DESCRIPTIVE:
        raise ValueError("success set / tau frozen")
    gate = config["gate"]
    if float(gate["no_go_max_gap"]) != NO_GO_MAX_GAP or float(gate["go_min_gap"]) != GO_MIN_GAP or float(gate["probe_pair_cost_fraction"]) != PROBE_PAIR_COST_FRACTION or int(gate["min_informative_seeds"]) != MIN_INFORMATIVE_SEEDS:
        raise ValueError("gap thresholds / informative minimum frozen")
    if config.get("blinded_review") is not True or tuple(config["frame_positions"]) != conf.FRAME_POSITIONS:
        raise ValueError("blinded review and frame positions are frozen")


def validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT / "results")
    except ValueError as error:
        raise ValueError("Output must be under results/") from error
    if not relative.parts or relative.parts[0] != NAMESPACE:
        raise ValueError(f"Output must use the isolated {NAMESPACE} namespace")
    if any(part in str(relative) for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError("Trusted prior-result namespace cannot be used")


def seed_family(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {**conf.seed_family(config, seed), "id": f"red_to_blue_seed{int(seed)}", "severity": "round4b"}


# --------------------------------------------------------------------------------------
# key set
# --------------------------------------------------------------------------------------
def expected_keys(informative: list[int] | tuple[int, ...] | None = None) -> set[tuple[int, str, int]]:
    keys: set[tuple[int, str, int]] = set()
    for seed in SEEDS:
        keys.add((seed, "old_baseline", -1))
        keys.add((seed, "new_baseline", -1))
    for seed in (SEEDS if informative is None else informative):
        keys.add((seed, "same_condition", CONTROL_K))
        for k in KS:
            keys.add((seed, "fork_new", k))
    return keys


row_key = conf.row_key
validate_key_set = conf.validate_key_set
merge_rows = conf.merge_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# provenance / preregistration
# --------------------------------------------------------------------------------------
def build_provenance(config_path: Path) -> dict[str, Any]:
    paths = [REPO_ROOT / v for v in TRUSTED_SOURCE_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise GateError(f"Provenance inputs missing: {missing}")
    hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        commit, status = "UNKNOWN", ["git status unavailable"]
    relevant = [line for line in status if any(name in line for name in TRUSTED_SOURCE_FILES)]
    document = {"git_commit": commit, "git_dirty": bool(status), "git_status": status, "relevant_git_status": relevant, "source_sha256": hashes, "config_sha256": sha256_file(config_path)}
    document["provenance_hash"] = sha256_bytes(canonical_json({k: document[k] for k in ("git_commit", "relevant_git_status", "source_sha256", "config_sha256")}))
    return document


def context_pins() -> dict[str, Any]:
    """Read-only pins of the two frozen predecessor results (context only; never used by the gate)."""
    pins: dict[str, Any] = {}
    for name in ("video_trajectory_fork_confirmatory", "video_trajectory_fork_probe_killtest"):
        root = REPO_ROOT / "results" / name
        entry: dict[str, Any] = {}
        for fname in ("preregistration.sha256", "summary.json"):
            path = root / fname
            entry[fname] = sha256_file(path) if path.exists() else None
        if (root / "summary.json").exists():
            entry["decision"] = json.loads((root / "summary.json").read_text()).get("decision")
        pins[name] = entry
    return pins


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan = smoke.scheduler_plan(conf.seed_config(config, SEEDS[0]))
    schedule = [float(v) for v in plan["timesteps"]]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "title": "Round 4B - Static-K Oracle vs Trajectory-Specific Oracle Gap",
        "question": "For the fixed red->blue edit, how much compute does a request-level policy with the best post-hoc static fork depth lose relative to a policy that knows each trajectory's latest safe fork depth?",
        "context_only": context_pins(),
        "prompts": config["prompts"],
        "seeds": list(SEEDS),
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "k_grid": list(KS),
        "control_k": CONTROL_K,
        "prefix_reuse_fraction": {str(k): k / TOTAL_STEPS for k in KS},
        "schedule_timesteps_by_k": {str(k): schedule[k] for k in KS},
        "expert_by_k": {str(k): smoke.expert_metadata(conf.seed_config(config, SEEDS[0]), plan, k) for k in KS},
        "labels": config["labels"],
        "oracle": config["oracle"],
        "gate": config["gate"],
        "baseline_informativeness": config["baseline_informativeness"],
        "controls": config["controls"],
        "blinded_review": True,
        "frame_positions": list(conf.FRAME_POSITIONS),
        "expected_keys_max": sorted([list(k) for k in expected_keys()]),
        "expected_trajectories_max": len(expected_keys()),
        "scheduler_plan": plan,
        "model": config["model"],
        "scheduler": config["scheduler"],
        "generation": config["generation"],
        "source_commit": provenance["git_commit"],
        "provenance_hash": provenance["provenance_hash"],
        "config_sha256": provenance["config_sha256"],
        "claim_boundary": config["claim_boundary"],
    }


def require_preregistration(output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    prereg_path, sha_path = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if not prereg_path.exists() or not sha_path.exists():
        raise GateError("Run the cpu phase first; preregistration is missing")
    if sha256_file(prereg_path) != sha_path.read_text().split()[0]:
        raise GateError("preregistration.json was modified after it was hashed")
    document = json.loads(prereg_path.read_text())
    frozen = json.loads((output_dir / "provenance.json").read_text())
    if document.get("provenance_hash") != provenance["provenance_hash"] or frozen.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Code/config provenance changed after preregistration; use a fresh output namespace")
    if provenance["relevant_git_status"]:
        raise GateError(f"Scientific source files are not committed: {provenance['relevant_git_status']}")
    return document


def run_cpu(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    prereg_path, sha_path = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
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
    return {"mode": "cpu", "status": "FROZEN", "preregistration_sha256": digest, "expected_trajectories_max": document["expected_trajectories_max"], "relevant_git_status": provenance["relevant_git_status"]}


# --------------------------------------------------------------------------------------
# Phase 1: baselines
# --------------------------------------------------------------------------------------
def run_baselines(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    initial_identity: dict[str, Any] = {}
    try:
        for seed in SEEDS:
            run_config = conf.seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_row, old_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, family, which="old", capture_steps=list(BASELINE_CAPTURE))
            new_row, new_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, family, which="new", capture_steps=[0, TOTAL_STEPS])
            old_initial = smoke._probe_array(smoke._record_by_step(old_meta, 0))
            new_initial = smoke._probe_array(smoke._record_by_step(new_meta, 0))
            initial_identity[str(seed)] = {"identical_initial_latent": array_sha256(old_initial) == array_sha256(new_initial), "initial_latent_hash": array_sha256(old_initial)}
            for row in (old_row, new_row):
                row["round4b_phase"] = "baseline"
                conf.save_frames(np.load(row["final_video_npy"], allow_pickle=False), output_dir / "qualitative" / family["id"], row["trajectory_type"])
            rows.extend((old_row, new_row))
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys() if k[1] in ("old_baseline", "new_baseline")})
    write_csv(output_dir / "baseline_results.csv", rows)
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows) if (output_dir / "raw_results.csv").exists() else rows)
    automated = {}
    for seed in SEEDS:
        old = next(r for r in rows if row_key(r) == (seed, "old_baseline", -1))
        new = next(r for r in rows if row_key(r) == (seed, "new_baseline", -1))
        automated[str(seed)] = {"old_new_minus_old": float(old["new_minus_old"]), "new_new_minus_old": float(new["new_minus_old"]), "automated_sign_ok": float(old["new_minus_old"]) < 0.0 < float(new["new_minus_old"])}
    atomic_json(output_dir / "baseline_automated_evidence.json", {"provenance_hash": provenance["provenance_hash"], "seeds": automated, "initial_identity": initial_identity})
    template = output_dir / "baseline_judgment_template.json"
    if not template.exists():
        atomic_json(template, {"provenance_hash": provenance["provenance_hash"], "fork_outcomes_examined": False,
                               "instructions": "Inspect ONLY old_baseline/new_baseline videos and the five fixed frames per seed. OLD must clearly show a RED sports car, NEW a BLUE sports car, and the pair must be clearly different.",
                               "seeds": {str(seed): {"old_matches_expected": None, "new_matches_expected": None, "clearly_different": None, "notes": ""} for seed in SEEDS}})
    return {"mode": "baselines", "rows": len(rows), "automated": automated, "next": "fill baseline_judgment.json, then run controls"}


def evaluate_baseline_informativeness(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    judgment_path = output_dir / "baseline_judgment.json"
    if not judgment_path.exists():
        raise GateError("Fill baseline_judgment.json from baseline_judgment_template.json before running controls")
    judgment = json.loads(judgment_path.read_text())
    if judgment.get("provenance_hash") != provenance_hash or judgment.get("fork_outcomes_examined") is not False:
        raise GateError("Baseline judgment provenance mismatch or fork outcomes examined before freezing")
    automated = json.loads((output_dir / "baseline_automated_evidence.json").read_text())
    if automated.get("provenance_hash") != provenance_hash:
        raise GateError("Baseline automated evidence provenance mismatch")
    result: dict[str, Any] = {}
    for seed in SEEDS:
        entry = judgment.get("seeds", {}).get(str(seed))
        if not isinstance(entry, dict):
            raise GateError(f"Baseline judgment missing seed {seed}")
        flags = {key: entry.get(key) for key in ("old_matches_expected", "new_matches_expected", "clearly_different")}
        if any(not isinstance(v, bool) for v in flags.values()):
            raise GateError(f"Baseline judgment for seed {seed} must contain explicit booleans")
        manual_ok = all(flags.values())
        auto_ok = bool(automated["seeds"][str(seed)]["automated_sign_ok"])
        identical = bool(automated["initial_identity"][str(seed)]["identical_initial_latent"])
        informative = manual_ok and auto_ok and identical
        result[str(seed)] = {"manual": flags, "manual_ok": manual_ok, "automated_sign_ok": auto_ok, "identical_initial_latent": identical, "metric_disagreement": manual_ok != auto_ok,
                             "status": "BASELINE_INFORMATIVE" if informative else "BASELINE_UNINFORMATIVE", "reason": None if informative else ("manual" if not manual_ok else "automated_sign" if not auto_ok else "initial_latent")}
    informative_seeds = [seed for seed in SEEDS if result[str(seed)]["status"] == "BASELINE_INFORMATIVE"]
    return {"seeds": result, "informative_seeds": informative_seeds, "uninformative_count": len(SEEDS) - len(informative_seeds)}


# --------------------------------------------------------------------------------------
# Phase 2: spot exactness control
# --------------------------------------------------------------------------------------
def _old_records(output_dir: Path, family_id: str) -> dict[int, dict[str, Any]]:
    metadata = json.loads((output_dir / "rows" / family_id / "old_baseline_metadata.json").read_text())["worker_trajectory_probe"]
    return {int(r["step_index"]): r for r in metadata["records"]}


def run_controls(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    informativeness = evaluate_baseline_informativeness(output_dir, provenance["provenance_hash"])
    preflight: dict[str, Any] = {"provenance_hash": provenance["provenance_hash"], "baseline_informativeness": informativeness, "seeds": {}}
    if len(informativeness["informative_seeds"]) < MIN_INFORMATIVE_SEEDS:
        preflight["status"] = "INVALID_BASELINES"
        preflight["reason"] = f"only {len(informativeness['informative_seeds'])} of {len(SEEDS)} baseline pairs informative (minimum {MIN_INFORMATIVE_SEEDS})"
        atomic_json(output_dir / "preflight.json", preflight)
        raise GateError(preflight["reason"])
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in informativeness["informative_seeds"]:
            run_config = conf.seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, family["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, family["id"], "new_baseline")
            records = _old_records(output_dir, family["id"])
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, family,
                                     label=f"same_condition_k{CONTROL_K:02d}", trajectory_type="same_condition", switch_step=CONTROL_K,
                                     prompt=family["old_prompt"], input_tensor=smoke._load_probe_tensor(records[CONTROL_K]["latent_path"]),
                                     initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=old_ref)
            row["round4b_phase"] = "control"
            rows.append(row)
            preflight["seeds"][str(seed)] = {
                "same_condition_exact_k18": row["control_exact"] is True,
                "old_checkpoint_hashes": {str(k): array_sha256(smoke._probe_array(records[k])) for k in KS},
                "resume_input_matches_checkpoint": row["resume_input_hash"] == array_sha256(smoke._probe_array(records[CONTROL_K])),
                "scheduler_euler": str(row["scheduler_class"]).endswith(smoke.EXPECTED_SCHEDULER),
            }
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys(informativeness["informative_seeds"]) if k[1] == "same_condition"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    c1 = all(e["same_condition_exact_k18"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"] for e in preflight["seeds"].values())
    preflight["status"] = "PASS" if c1 else "INVALID"
    atomic_json(output_dir / "preflight.json", preflight)
    if not c1:
        raise GateError("C1 spot exactness control failed; INVALID / STOP")
    return {"mode": "controls", "status": "PASS", "informative_seeds": informativeness["informative_seeds"], "rows": len(rows)}


def require_controls(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if not path.exists():
        raise GateError("Controls have not completed")
    preflight = json.loads(path.read_text())
    if preflight.get("provenance_hash") != provenance_hash or preflight.get("status") != "PASS":
        raise GateError(f"Control gate not passed (status={preflight.get('status')})")
    informative = preflight["baseline_informativeness"]["informative_seeds"]
    if set(map(str, informative)) != set(preflight["seeds"]) or len(informative) < MIN_INFORMATIVE_SEEDS:
        raise GateError("Control gate covers a different or insufficient seed set")
    for seed, e in preflight["seeds"].items():
        if not (e["same_condition_exact_k18"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"]):
            raise GateError(f"Control gate failed for seed {seed}")
    return preflight


# --------------------------------------------------------------------------------------
# Phase 3: forks
# --------------------------------------------------------------------------------------
def run_forks(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"]]
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in informative:
            run_config = conf.seed_config(config, seed)
            family = seed_family(config, seed)
            evaluator = smoke.ConceptEvaluator(run_config, family)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, family["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, family["id"], "new_baseline")
            records = _old_records(output_dir, family["id"])
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            for k in KS:
                checkpoint = smoke._load_probe_tensor(records[k]["latent_path"])
                if array_sha256(checkpoint.float().numpy()) != preflight["seeds"][str(seed)]["old_checkpoint_hashes"][str(k)]:
                    raise GateError(f"OLD checkpoint for seed {seed} k={k} changed since controls")
                row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, family,
                                         label=f"fork_new_k{k:02d}", trajectory_type="fork_new", switch_step=k, prompt=family["new_prompt"],
                                         input_tensor=checkpoint, initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=None)
                row["round4b_phase"] = "fork"
                rows.append(row)
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys(informative) if k[1] == "fork_new"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    return {"mode": "forks", "rows": len(rows), "informative_seeds": informative, "next": "run blind, label blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 4: blinding (reuses the confirmatory assignment function)
# --------------------------------------------------------------------------------------
def run_blind(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"]]
    rows = [r for r in read_csv(output_dir / "raw_results.csv") if r["trajectory_type"] == "fork_new"]
    validate_key_set(rows, {k for k in expected_keys(informative) if k[1] == "fork_new"})
    if (output_dir / "blinded_labels.json").exists():
        raise GateError("blinded_labels.json already exists; the blinded set must not be regenerated after labelling")
    prereg_sha = (output_dir / "preregistration.sha256").read_text().split()[0]
    mapping = conf.blinded_assignment([(int(r["seed"]), int(r["switch_step"])) for r in rows], prereg_sha)
    by_key = {(int(r["seed"]), int(r["switch_step"])): r for r in rows}
    blind_dir = output_dir / "qualitative" / "blinded"
    blind_dir.mkdir(parents=True, exist_ok=True)
    samples = {}
    for sample_id, (seed, k) in mapping.items():
        row = by_key[(seed, k)]
        frames = conf.save_frames(np.load(row["final_video_npy"], allow_pickle=False), blind_dir, sample_id)
        mp4 = blind_dir / f"{sample_id}.mp4"
        shutil.copyfile(row["final_video_mp4"], mp4)
        samples[sample_id] = {"mp4": str(mp4), "frames": frames, "video_hash": row["video_hash"]}
    atomic_json(output_dir / "blinded_mapping.sealed.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"seed": v[0], "switch_step": v[1]} for s, v in mapping.items()}})
    atomic_json(output_dir / "blinded_manifest.json", {"provenance_hash": provenance["provenance_hash"], "sealed_mapping_sha256": sha256_file(output_dir / "blinded_mapping.sealed.json"),
                                                        "instructions": "Label each sample from its five fixed frames and full video without opening blinded_mapping.sealed.json. Judge the whole video.",
                                                        "labels": list(LABELS), "definitions": config["labels"]["definitions"], "samples": samples})
    if not (output_dir / "blinded_labels_template.json").exists():
        atomic_json(output_dir / "blinded_labels_template.json", {"provenance_hash": provenance["provenance_hash"], "mapping_revealed_before_labelling": False, "labels": {s: {"label": None, "notes": ""} for s in sorted(mapping)}})
    return {"mode": "blind", "samples": len(mapping), "next": "fill blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 5: frozen oracle-gap analysis
# --------------------------------------------------------------------------------------
def success(label: str) -> bool:
    if label not in LABELS:
        raise GateError(f"Label outside the frozen scale: {label!r}")
    return label in SUCCESS_LABELS


def trajectory_oracle_k(labels_by_k: dict[int, str]) -> dict[str, Any]:
    """Largest grid K such that every grid K' <= K succeeds (monotone cut); 0 if the first grid point fails."""
    k_star = 0
    violations = 0
    failed_before = False
    for k in KS:
        ok = success(labels_by_k[k])
        if ok and failed_before:
            violations += 1
        if ok and not failed_before:
            k_star = k
        if not ok:
            failed_before = True
    return {"k_star": k_star, "monotonicity_violations": violations, "success_by_k": {str(k): success(labels_by_k[k]) for k in KS}}


def static_oracle_k(labels: dict[int, dict[int, str]], tau: float) -> dict[str, Any]:
    """Largest grid K whose success rate over the given seeds is >= tau; 0 if none."""
    n = len(labels)
    rates = {k: sum(1 for seed in labels if success(labels[seed][k])) / n for k in KS}
    candidates = [k for k in KS if rates[k] >= tau - 1e-12]
    k_static = max(candidates) if candidates else 0
    return {"k_static": k_static, "tau": tau, "success_rate_by_k": {str(k): rates[k] for k in KS}, "failing_seeds_at_k_static": [seed for seed in labels if k_static and not success(labels[seed][k_static])]}


def oracle_gap(labels: dict[int, dict[int, str]], tau: float) -> dict[str, Any]:
    static = static_oracle_k(labels, tau)
    per_seed = {seed: trajectory_oracle_k(labels[seed]) for seed in labels}
    k_static = static["k_static"]
    diffs = [per_seed[seed]["k_star"] - k_static for seed in labels]
    gap = float(np.mean(diffs)) / TOTAL_STEPS
    return {
        "tau": tau,
        "k_static": k_static,
        "static": static,
        "k_star_by_seed": {str(seed): per_seed[seed]["k_star"] for seed in labels},
        "monotonicity_violations_total": sum(per_seed[seed]["monotonicity_violations"] for seed in labels),
        "mean_k_star": float(np.mean([per_seed[seed]["k_star"] for seed in labels])),
        "gap_fraction_of_generation": gap,
        "gap_relative_to_static_recompute": (float(np.mean(diffs)) / (TOTAL_STEPS - k_static)) if k_static < TOTAL_STEPS else None,
        "seeds_with_k_star_above_static": sum(1 for d in diffs if d > 0),
        "seeds_with_k_star_below_static": sum(1 for d in diffs if d < 0),
        "diffs_in_steps": diffs,
    }


def decide(*, valid: bool, invalid_reason: str | None, gap: float) -> dict[str, Any]:
    if not valid:
        return {"decision": "INVALID", "rationale": invalid_reason or "hard control failed", "MECHANISM_STAGE_ELIGIBLE": False}
    if gap <= NO_GO_MAX_GAP:
        return {"decision": "NO-GO", "rationale": f"oracle gap G={gap:.4f} <= {NO_GO_MAX_GAP} (measured probe-pair cost); trajectory information cannot pay for itself", "MECHANISM_STAGE_ELIGIBLE": False}
    if gap >= GO_MIN_GAP:
        return {"decision": "GO", "rationale": f"oracle gap G={gap:.4f} >= {GO_MIN_GAP}", "MECHANISM_STAGE_ELIGIBLE": True}
    return {"decision": "WEAK", "rationale": f"oracle gap G={gap:.4f} in ({NO_GO_MAX_GAP}, {GO_MIN_GAP})", "MECHANISM_STAGE_ELIGIBLE": False}


def run_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"]]
    labels_path = output_dir / "blinded_labels.json"
    if not labels_path.exists():
        raise GateError("Fill blinded_labels.json before analysis")
    labels_doc = json.loads(labels_path.read_text())
    if labels_doc.get("provenance_hash") != provenance["provenance_hash"] or labels_doc.get("mapping_revealed_before_labelling") is not False:
        raise GateError("Blinded labels provenance mismatch or mapping revealed before labelling")
    manifest = json.loads((output_dir / "blinded_manifest.json").read_text())
    sealed_path = output_dir / "blinded_mapping.sealed.json"
    if sha256_file(sealed_path) != manifest["sealed_mapping_sha256"]:
        raise GateError("Sealed blinded mapping was modified")
    mapping = {s: (int(v["seed"]), int(v["switch_step"])) for s, v in json.loads(sealed_path.read_text())["mapping"].items()}
    if set(labels_doc.get("labels", {})) != set(mapping):
        raise GateError("Blinded labels do not cover exactly the blinded sample set")
    rows = read_csv(output_dir / "raw_results.csv")
    for row in rows:
        smoke._validate_result_artifacts(row, provenance["provenance_hash"])
    validate_key_set(rows, expected_keys(informative))
    forks = {(int(r["seed"]), int(r["switch_step"])): r for r in rows if r["trajectory_type"] == "fork_new"}
    labels: dict[int, dict[int, str]] = {seed: {} for seed in informative}
    qualitative = {}
    for sample_id, (seed, k) in mapping.items():
        entry = labels_doc["labels"][sample_id]
        label = entry.get("label") if isinstance(entry, dict) else None
        if label not in LABELS:
            raise GateError(f"Invalid or missing label for {sample_id}: {label!r}")
        if manifest["samples"][sample_id]["video_hash"] != forks[(seed, k)]["video_hash"]:
            raise GateError(f"Blinded sample {sample_id} does not match the persisted fork video")
        labels[seed][k] = label
        qualitative[f"{seed}:{k}"] = {"sample_id": sample_id, "label": label, "notes": entry.get("notes", "")}
    if any(set(labels[seed]) != set(KS) for seed in informative):
        raise GateError("Label matrix incomplete")
    atomic_json(output_dir / "unblinded_mapping.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"seed": v[0], "switch_step": v[1]} for s, v in mapping.items()}, "revealed_after_labels": True})
    atomic_json(output_dir / "qualitative_judgment.json", {"provenance_hash": provenance["provenance_hash"], "blinded": True, "labels": qualitative})
    primary = oracle_gap(labels, TAU_PRIMARY)
    descriptive = oracle_gap(labels, TAU_DESCRIPTIVE)
    # descriptive: sensitivity to the single earliest-committing trajectory
    k_stars = {seed: primary["k_star_by_seed"][str(seed)] for seed in informative}
    loo = None
    if len(informative) > 1:
        weakest = min(informative, key=lambda s: k_stars[s])
        loo = {"dropped_seed": weakest, **{k: v for k, v in oracle_gap({s: labels[s] for s in informative if s != weakest}, TAU_PRIMARY).items() if k in ("k_static", "gap_fraction_of_generation", "mean_k_star")}}
    verdict = decide(valid=preflight["status"] == "PASS" and len(informative) >= MIN_INFORMATIVE_SEEDS, invalid_reason=None, gap=primary["gap_fraction_of_generation"])
    table = []
    for seed in informative:
        for k in KS:
            r = forks[(seed, k)]
            table.append({"seed": seed, "k": k, "reuse_pct": 100.0 * k / TOTAL_STEPS, "label": labels[seed][k], "ordinal": ORDINAL[labels[seed][k]], "success": success(labels[seed][k]),
                          "clip_new_minus_old": float(r["new_minus_old"]), "ssim_old": float(r["ssim_to_old"]), "ssim_new": float(r["ssim_to_new"]), "wall_s": float(r["wall_time_s"]), "expert": r["expert_at_switch"]})
    summary = {
        **verdict, "G": primary["gap_fraction_of_generation"], "tau_primary": TAU_PRIMARY,
        "primary": primary, "descriptive_tau_0_9": descriptive, "leave_out_earliest_committing_seed": loo,
        "label_matrix": {str(s): {str(k): labels[s][k] for k in KS} for s in informative},
        "informative_seeds": informative, "uninformative_seeds": [s for s in SEEDS if s not in informative],
        "controls": {"preflight_status": preflight["status"], "baseline_informativeness": preflight["baseline_informativeness"]},
        "fork_table": table, "trajectories_executed": len(rows),
        "thresholds": {"no_go_max_gap": NO_GO_MAX_GAP, "go_min_gap": GO_MIN_GAP, "probe_pair_cost_fraction": PROBE_PAIR_COST_FRACTION},
        "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0], "provenance_hash": provenance["provenance_hash"],
        "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print_console(summary, provenance)
    return {k: v for k, v in summary.items() if k not in ("fork_table", "primary", "descriptive_tau_0_9")}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    p = s["primary"]; d = s["descriptive_tau_0_9"]
    L = ["# Round 4B - Static-K Oracle vs Trajectory-Specific Oracle Gap", "", "## Decision", "", s["decision"], "", "## Question", "",
         "For the fixed red->blue edit, how much compute does a request-level policy with the best post-hoc static fork depth lose relative to a policy that knows each trajectory's latest safe fork depth?", "",
         "## Frozen definitions", "", f"- success := label in {list(SUCCESS_LABELS)}; K_i* := largest grid K with every grid K' <= K successful; K_static(tau) := largest grid K with success rate >= tau; G := mean(K_i* - K_static(1.0)) / 40.",
         f"- Gate: NO-GO if G <= {NO_GO_MAX_GAP} (Round 4A measured probe-pair cost), GO if G >= {GO_MIN_GAP}, otherwise WEAK; INVALID below {MIN_INFORMATIVE_SEEDS} informative seeds or on control failure.", "",
         "## Controls", "", f"- Preflight: {s['controls']['preflight_status']}; informative seeds {len(s['informative_seeds'])}/{len(SEEDS)}; uninformative: {s['uninformative_seeds']}", "",
         "## Label matrix (blinded)", "", "| seed | " + " | ".join(f"k{k}" for k in KS) + " | K_i* |", "|---|" + "---|" * (len(KS) + 1)]
    for seed in s["informative_seeds"]:
        L.append(f"| {seed} | " + " | ".join(s["label_matrix"][str(seed)][str(k)] for k in KS) + f" | {p['k_star_by_seed'][str(seed)]} |")
    L += ["", "## Oracle comparison (primary, tau = 1.0)", "", f"- success rate by K: {p['static']['success_rate_by_k']}", f"- K_static = {p['k_static']}", f"- mean K_i* = {p['mean_k_star']:.2f}",
          f"- G = {p['gap_fraction_of_generation']:.4f} of a full generation; relative to static recompute = {p['gap_relative_to_static_recompute']}", f"- seeds with K_i* above/below K_static: {p['seeds_with_k_star_above_static']} / {p['seeds_with_k_star_below_static']}", f"- monotonicity violations: {p['monotonicity_violations_total']}", "",
          "## Descriptive (tau = 0.9)", "", f"- K_static(0.9) = {d['k_static']} (failing seeds at that K: {d['static']['failing_seeds_at_k_static']}); G(0.9) = {d['gap_fraction_of_generation']:.4f}", "",
          "## Descriptive: leave out the earliest-committing seed", "", f"- {s['leave_out_earliest_committing_seed']}", "", "## Fork table", "", "| seed | k | reuse % | label | success | CLIP new-old | SSIM OLD | SSIM NEW |", "|---|---|---|---|---|---|---|---|"]
    for t in s["fork_table"]:
        L.append(f"| {t['seed']} | {t['k']} | {t['reuse_pct']:.1f} | {t['label']} | {t['success']} | {t['clip_new_minus_old']:+.4f} | {t['ssim_old']:.3f} | {t['ssim_new']:.3f} |")
    L += ["", "## Decision rationale", "", s["rationale"], "", "## Claim boundary", "", s["claim_boundary"], "", f"MECHANISM_STAGE_ELIGIBLE = {str(s['MECHANISM_STAGE_ELIGIBLE']).lower()}", ""]
    (output_dir / "video_trajectory_fork_oracle_gap_killtest.md").write_text("\n".join(L))


def print_console(s: dict[str, Any], provenance: dict[str, Any]) -> None:
    print(f"git_commit={provenance['git_commit']} relevant_dirty={provenance['relevant_git_status']}")
    print(f"preregistration_sha256={s['preregistration_sha256']}")
    print(f"informative_seeds={len(s['informative_seeds'])}/{len(SEEDS)} K_static={s['primary']['k_static']} mean_K_star={s['primary']['mean_k_star']:.2f} G={s['G']:.4f}")
    print(f"decision={s['decision']} MECHANISM_STAGE_ELIGIBLE={s['MECHANISM_STAGE_ELIGIBLE']}")


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
    dispatch = {"cpu": lambda: run_cpu(config, args.config, output_dir), "baselines": lambda: run_baselines(config, args.config, output_dir, args),
                "controls": lambda: run_controls(config, args.config, output_dir, args), "forks": lambda: run_forks(config, args.config, output_dir, args),
                "blind": lambda: run_blind(config, args.config, output_dir), "analyze": lambda: run_analyze(config, args.config, output_dir)}
    print(json.dumps(dispatch[args.mode](), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
