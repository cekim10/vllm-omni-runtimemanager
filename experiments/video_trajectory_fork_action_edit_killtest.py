#!/usr/bin/env python3
"""Round 4C-0 extension: the preregistered third edit class (action) - final STOP / INVEST decision.

Round 4C-0 ended EXTEND_ONE_EDIT (object WEAK, background STRONG at n=3). Its frozen rule allows exactly one
extra edit class, "action", with the same design, after which the branch is decided for good:

    action EDIT_STRONG  -> INVEST_4C1
    otherwise           -> STOP_DIFFUSION_REUSE      (no grid change, no edit swap, no seed addition)

Design (frozen): OLD "driving forward, static camera" -> NEW "parked motionless, static camera"; the same four
seeds and K grid {4, 8, 12, 16, 20, 24}; OLD->OLD exactness control at K=8; forks for ALL seeds.
Baseline informativeness is decided by BLINDED HUMAN labels: the OLD and NEW baselines are mixed into the
blinded pool with the 24 forks; a seed is informative iff its OLD baseline is labelled OLD, its NEW baseline
NEW, and the two share the identical initial latent. No CLIP or motion metric acts as a validity gate; a
whole-frame motion energy is computed as a frozen descriptive secondary only.
"""
from __future__ import annotations

import argparse
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
from experiments import video_trajectory_fork_multi_edit_killtest as me  # noqa: E402

EXPERIMENT_VERSION = "video-trajectory-fork-action-edit-screening-v1"
NAMESPACE = "video_trajectory_fork_action_edit_killtest"
DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_action_edit_killtest_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE
ROUND4C0_DIR = REPO_ROOT / "results" / me.NAMESPACE
MODES = ("cpu", "baselines", "controls", "forks", "blind", "analyze")
EDIT = "action"
SEEDS = me.SEEDS
KS = me.KS
CONTROL_K = me.CONTROL_K
TOTAL_STEPS = 40
OLD_PROMPT = "A red sports car driving forward on a snowy road, static camera, cinematic video"
NEW_PROMPT = "A red sports car parked motionless on a snowy road, static camera, cinematic video"
OLD_CONCEPT = "a car driving"
NEW_CONCEPT = "a parked car"
LABELS = conf.LABELS
ORDINAL = conf.ORDINAL
SUCCESS_LABELS = me.SUCCESS_LABELS
STRONG_MIN_GAP = me.STRONG_MIN_GAP
DISPERSION_MIN_RANGE = me.DISPERSION_MIN_RANGE
MIN_INFORMATIVE_SEEDS = me.MIN_INFORMATIVE_SEEDS
BASELINE_CAPTURE = [0, *KS, TOTAL_STEPS]
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_action_edit_killtest.py",
    "experiments/video_trajectory_fork_action_edit_killtest_config.yaml",
    "experiments/run_video_trajectory_fork_action_edit_killtest_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_action_edit_killtest.py",
    "experiments/video_trajectory_fork_multi_edit_killtest.py",
    "experiments/video_trajectory_fork_oracle_gap_killtest.py",
    "experiments/video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = me.FORBIDDEN_OUTPUT_PARTS + (me.NAMESPACE,)
GateError = smoke.GateError
canonical_json = smoke.canonical_json
sha256_bytes = smoke.sha256_bytes
sha256_file = smoke.sha256_file
array_sha256 = smoke.array_sha256
atomic_json = smoke.atomic_json
read_csv = smoke.read_csv
CSV_FIELDS = me.CSV_FIELDS
row_key = me.row_key
validate_key_set = me.validate_key_set
merge_rows = me.merge_rows
write_csv = me.write_csv
Key = me.Key


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
    if tuple(int(v) for v in config["k_grid"]) != KS or int(config["control_k"]) != CONTROL_K:
        raise ValueError("K grid / control K are frozen")
    prompts = config["prompts"]
    if prompts["old"] != OLD_PROMPT or prompts["new"] != NEW_PROMPT or prompts["old_concept"] != OLD_CONCEPT or prompts["new_concept"] != NEW_CONCEPT:
        raise ValueError("Action prompt pair / concept pair changed")
    if "static camera" not in prompts["old"] or "static camera" not in prompts["new"]:
        raise ValueError("Both prompts must carry the static-camera control")
    if config["edit"] != EDIT:
        raise ValueError("Edit class is frozen to action")
    labels = config["labels"]
    if tuple(labels["scale"]) != LABELS or {k: int(v) for k, v in labels["ordinal"].items()} != ORDINAL or set(labels["definitions"]) != set(LABELS):
        raise ValueError("Label scale / ordinal / definitions frozen")
    if tuple(config["oracle"]["success_labels"]) != SUCCESS_LABELS:
        raise ValueError("success set frozen")
    gate = config["gate"]
    if float(gate["strong_min_gap"]) != STRONG_MIN_GAP or int(gate["dispersion_min_range"]) != DISPERSION_MIN_RANGE or int(gate["min_informative_seeds"]) != MIN_INFORMATIVE_SEEDS:
        raise ValueError("gap threshold / dispersion / informative minimum frozen")
    inf = config["baseline_informativeness"]
    if inf["method"] != "blinded_human" or inf["old_baseline_required_label"] != "OLD" or inf["new_baseline_required_label"] != "NEW" or inf.get("automated_validity_gate") is not None:
        raise ValueError("Baseline informativeness is frozen to blinded human labels with no automated validity gate")
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


def family(seed: int) -> dict[str, Any]:
    return {"id": f"{EDIT}_seed{int(seed)}", "severity": "round4c_action", "seed": int(seed), "edit": EDIT, "old_prompt": OLD_PROMPT, "new_prompt": NEW_PROMPT, "old_concept": OLD_CONCEPT, "new_concept": NEW_CONCEPT}


def expected_keys(fork_seeds: list[int] | tuple[int, ...] | None = None) -> set[Key]:
    keys: set[Key] = set()
    for seed in SEEDS:
        keys.add((EDIT, seed, "old_baseline", -1))
        keys.add((EDIT, seed, "new_baseline", -1))
        keys.add((EDIT, seed, "same_condition", CONTROL_K))
    for seed in (SEEDS if fork_seeds is None else fork_seeds):
        for k in KS:
            keys.add((EDIT, seed, "fork_new", k))
    return keys


def _tag(row: dict[str, Any], phase: str) -> dict[str, Any]:
    row["edit"] = EDIT
    row["round4c_phase"] = phase
    return row


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


def round4c0_reference() -> dict[str, Any]:
    """Frozen Round 4C-0 pins. The final decision combines the pinned 4C-0 edit statuses with the action status."""
    for name in ("preregistration.sha256", "summary.json"):
        if not (ROUND4C0_DIR / name).exists():
            raise GateError(f"Round 4C-0 reference missing: {ROUND4C0_DIR / name}")
    summary = json.loads((ROUND4C0_DIR / "summary.json").read_text())
    if summary.get("decision") != "EXTEND_ONE_EDIT":
        raise GateError(f"Round 4C-0 decision is {summary.get('decision')!r}; the action extension is only allowed after EXTEND_ONE_EDIT")
    return {
        "namespace": me.NAMESPACE,
        "preregistration_sha256": (ROUND4C0_DIR / "preregistration.sha256").read_text().split()[0],
        "summary_sha256": sha256_file(ROUND4C0_DIR / "summary.json"),
        "decision": summary["decision"],
        "edit_status": summary["edit_status"],
        "gap_retry_by_edit": {e: summary["edits"][e]["gap_retry"] for e in summary["edits"]},
        "k_star_by_edit": {e: summary["edits"][e]["k_star_by_seed"] for e in summary["edits"]},
    }


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan = smoke.scheduler_plan(conf.seed_config(config, SEEDS[0]))
    schedule = [float(v) for v in plan["timesteps"]]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "title": "Round 4C-0 extension - action edit (final STOP / INVEST decision for the trajectory-fork branch)",
        "question": "For the action edit (driving forward -> parked motionless, static camera) on the same four seeds, do per-trajectory K* disperse and does the trajectory oracle beat the best detect-and-retry static depth by >= 0.15 of a generation?",
        "statistical_claim": "none; screening only (n <= 4 seeds)",
        "round4c0_reference": round4c0_reference(),
        "round4b_context": me.r4b.context_pins(),
        "edit": EDIT,
        "prompts": config["prompts"],
        "seeds": list(SEEDS),
        "k_grid": list(KS),
        "control_k": CONTROL_K,
        "prefix_reuse_fraction": {str(k): k / TOTAL_STEPS for k in KS},
        "schedule_timesteps_by_k": {str(k): schedule[k] for k in KS},
        "expert_by_k": {str(k): smoke.expert_metadata(conf.seed_config(config, SEEDS[0]), plan, k) for k in KS},
        "labels": config["labels"],
        "oracle": config["oracle"],
        "gate": config["gate"],
        "baseline_informativeness": config["baseline_informativeness"],
        "motion_metric_descriptive": config["motion_metric_descriptive"],
        "controls": config["controls"],
        "blinded_review": True,
        "blinded_pool": "8 baselines (OLD/NEW x 4 seeds) + 24 forks, one pool, sample ids from the preregistration hash",
        "frame_positions": list(conf.FRAME_POSITIONS),
        "expected_keys": sorted([list(k) for k in expected_keys()]),
        "expected_trajectories": len(expected_keys()),
        "scheduler_plan": plan,
        "model": config["model"],
        "scheduler": config["scheduler"],
        "generation": config["generation"],
        "source_commit": provenance["git_commit"],
        "provenance_hash": provenance["provenance_hash"],
        "config_sha256": provenance["config_sha256"],
        "final_decision_rule": config["gate"]["final_decision_rule"],
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
        if (output_dir / "raw_results.csv").exists():
            return {"mode": "cpu", "status": "ALREADY_FROZEN", "preregistration_sha256": sha_path.read_text().split()[0]}
    document = build_preregistration(config, provenance)
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(prereg_path, document)
    digest = sha256_file(prereg_path)
    sha_path.write_text(f"{digest}  preregistration.json\n")
    return {"mode": "cpu", "status": "FROZEN", "preregistration_sha256": digest, "expected_trajectories": document["expected_trajectories"], "relevant_git_status": provenance["relevant_git_status"]}


# --------------------------------------------------------------------------------------
# Phase 1: baselines (no manual judgment step; informativeness is decided blind at analyze)
# --------------------------------------------------------------------------------------
def run_baselines(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    try:
        for seed in SEEDS:
            run_config = conf.seed_config(config, seed)
            fam = family(seed)
            evaluator = smoke.ConceptEvaluator(run_config, fam)
            scheduler = prereg["scheduler_plan"]
            old_row, old_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, fam, which="old", capture_steps=list(BASELINE_CAPTURE))
            new_row, new_meta = smoke._run_baseline(omni, evaluator, run_config, provenance, scheduler, output_dir, fam, which="new", capture_steps=[0, TOTAL_STEPS])
            old_initial = array_sha256(smoke._probe_array(smoke._record_by_step(old_meta, 0)))
            new_initial = array_sha256(smoke._probe_array(smoke._record_by_step(new_meta, 0)))
            identity[str(seed)] = {"identical_initial_latent": old_initial == new_initial, "initial_latent_hash": old_initial,
                                   "clip_descriptive": {"old_new_minus_old": float(old_row["new_minus_old"]), "new_new_minus_old": float(new_row["new_minus_old"])}}
            rows.extend((_tag(old_row, "baseline"), _tag(new_row, "baseline")))
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys() if k[2] in ("old_baseline", "new_baseline")})
    write_csv(output_dir / "baseline_results.csv", rows)
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows) if (output_dir / "raw_results.csv").exists() else rows)
    atomic_json(output_dir / "baseline_automated_evidence.json", {"provenance_hash": provenance["provenance_hash"], "seeds": identity,
                                                                    "note": "identical initial latent is the only automated informativeness component; CLIP values are descriptive and never gate"})
    return {"mode": "baselines", "rows": len(rows), "identity": {s: v["identical_initial_latent"] for s, v in identity.items()}, "next": "run controls (no manual baseline judgment; baselines are labelled blind later)"}


# --------------------------------------------------------------------------------------
# Phase 2: exactness control for every seed
# --------------------------------------------------------------------------------------
def _old_records(output_dir: Path, seed: int) -> dict[int, dict[str, Any]]:
    metadata = json.loads((output_dir / "rows" / family(seed)["id"] / "old_baseline_metadata.json").read_text())["worker_trajectory_probe"]
    return {int(r["step_index"]): r for r in metadata["records"]}


def run_controls(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    evidence = json.loads((output_dir / "baseline_automated_evidence.json").read_text())
    if evidence.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Baseline evidence provenance mismatch")
    preflight: dict[str, Any] = {"provenance_hash": provenance["provenance_hash"], "identical_initial_latent": {s: v["identical_initial_latent"] for s, v in evidence["seeds"].items()}, "seeds": {}}
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            run_config = conf.seed_config(config, seed)
            fam = family(seed)
            evaluator = smoke.ConceptEvaluator(run_config, fam)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, fam["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, fam["id"], "new_baseline")
            records = _old_records(output_dir, seed)
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, fam,
                                     label=f"same_condition_k{CONTROL_K:02d}", trajectory_type="same_condition", switch_step=CONTROL_K,
                                     prompt=OLD_PROMPT, input_tensor=smoke._load_probe_tensor(records[CONTROL_K]["latent_path"]),
                                     initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=old_ref)
            rows.append(_tag(row, "control"))
            preflight["seeds"][str(seed)] = {
                "same_condition_exact_k08": row["control_exact"] is True,
                "old_checkpoint_hashes": {str(k): array_sha256(smoke._probe_array(records[k])) for k in KS},
                "resume_input_matches_checkpoint": row["resume_input_hash"] == array_sha256(smoke._probe_array(records[CONTROL_K])),
                "scheduler_euler": str(row["scheduler_class"]).endswith(smoke.EXPECTED_SCHEDULER),
            }
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys() if k[2] == "same_condition"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    c1 = all(e["same_condition_exact_k08"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"] for e in preflight["seeds"].values())
    preflight["status"] = "PASS" if c1 else "INVALID"
    atomic_json(output_dir / "preflight.json", preflight)
    if not c1:
        raise GateError("C1 exactness control failed; INVALID / STOP")
    return {"mode": "controls", "status": "PASS", "rows": len(rows)}


def require_controls(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if not path.exists():
        raise GateError("Controls have not completed")
    preflight = json.loads(path.read_text())
    if preflight.get("provenance_hash") != provenance_hash or preflight.get("status") != "PASS" or set(preflight["seeds"]) != {str(s) for s in SEEDS}:
        raise GateError(f"Control gate not passed (status={preflight.get('status')})")
    for seed, e in preflight["seeds"].items():
        if not (e["same_condition_exact_k08"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"]):
            raise GateError(f"Control gate failed for seed {seed}")
    return preflight


# --------------------------------------------------------------------------------------
# Phase 3: forks for every seed (informativeness is decided afterwards, blind)
# --------------------------------------------------------------------------------------
def run_forks(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            run_config = conf.seed_config(config, seed)
            fam = family(seed)
            evaluator = smoke.ConceptEvaluator(run_config, fam)
            scheduler = prereg["scheduler_plan"]
            old_ref = smoke._load_saved_baseline(output_dir, fam["id"], "old_baseline")
            new_ref = smoke._load_saved_baseline(output_dir, fam["id"], "new_baseline")
            records = _old_records(output_dir, seed)
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            for k in KS:
                checkpoint = smoke._load_probe_tensor(records[k]["latent_path"])
                if array_sha256(checkpoint.float().numpy()) != preflight["seeds"][str(seed)]["old_checkpoint_hashes"][str(k)]:
                    raise GateError(f"OLD checkpoint for seed {seed} k={k} changed since controls")
                row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, fam,
                                         label=f"fork_new_k{k:02d}", trajectory_type="fork_new", switch_step=k, prompt=NEW_PROMPT,
                                         input_tensor=checkpoint, initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=None)
                rows.append(_tag(row, "fork"))
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys() if k[2] == "fork_new"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    return {"mode": "forks", "rows": len(rows), "next": "run blind, label blinded_labels.json (32 samples incl. baselines), then analyze"}


# --------------------------------------------------------------------------------------
# Phase 4: one blinded pool of baselines + forks
# --------------------------------------------------------------------------------------
def pool_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["trajectory_type"]), int(row["seed"]), int(row["switch_step"])


def run_blind(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    require_preregistration(output_dir, provenance)
    require_controls(output_dir, provenance["provenance_hash"])
    rows = [r for r in read_csv(output_dir / "raw_results.csv") if r["trajectory_type"] in ("old_baseline", "new_baseline", "fork_new")]
    validate_key_set(rows, {k for k in expected_keys() if k[2] != "same_condition"})
    if (output_dir / "blinded_labels.json").exists():
        raise GateError("blinded_labels.json already exists; the blinded set must not be regenerated after labelling")
    prereg_sha = (output_dir / "preregistration.sha256").read_text().split()[0]
    keys = [pool_key(r) for r in rows]
    mapping = conf.blinded_assignment(keys, prereg_sha)  # type: ignore[arg-type]
    by_key = {pool_key(r): r for r in rows}
    blind_dir = output_dir / "qualitative" / "blinded"
    blind_dir.mkdir(parents=True, exist_ok=True)
    samples = {}
    for sample_id, key in mapping.items():
        row = by_key[key]  # type: ignore[index]
        frames = conf.save_frames(np.load(row["final_video_npy"], allow_pickle=False), blind_dir, sample_id)
        mp4 = blind_dir / f"{sample_id}.mp4"
        shutil.copyfile(row["final_video_mp4"], mp4)
        samples[sample_id] = {"mp4": str(mp4), "frames": frames, "video_hash": row["video_hash"]}  # trajectory type deliberately not recorded here
    atomic_json(output_dir / "blinded_mapping.sealed.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"trajectory_type": v[0], "seed": v[1], "switch_step": v[2]} for s, v in mapping.items()}})
    atomic_json(output_dir / "blinded_manifest.json", {"provenance_hash": provenance["provenance_hash"], "sealed_mapping_sha256": sha256_file(output_dir / "blinded_mapping.sealed.json"),
                                                        "instructions": "Label every sample on the action scale from its five fixed frames and full video without opening blinded_mapping.sealed.json. Baselines and forks are mixed; seed, K and type are hidden.",
                                                        "labels": list(LABELS), "definitions": config["labels"]["definitions"], "samples": samples})
    if not (output_dir / "blinded_labels_template.json").exists():
        atomic_json(output_dir / "blinded_labels_template.json", {"provenance_hash": provenance["provenance_hash"], "mapping_revealed_before_labelling": False, "labels": {s: {"label": None, "notes": ""} for s in sorted(mapping)}})
    return {"mode": "blind", "samples": len(mapping), "next": "fill blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 5: frozen analysis
# --------------------------------------------------------------------------------------
def motion_energy(video: np.ndarray) -> float:
    """Frozen descriptive secondary: mean absolute grayscale frame difference over the whole frame, in [0, 1]."""
    gray = video[..., :3].astype(np.float32).mean(axis=-1) / 255.0
    if len(gray) < 2:
        return 0.0
    return float(np.abs(np.diff(gray, axis=0)).mean())


def informativeness_from_labels(baseline_labels: dict[int, dict[str, str]], identity: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for seed in SEEDS:
        old_ok = baseline_labels[seed]["old_baseline"] == "OLD"
        new_ok = baseline_labels[seed]["new_baseline"] == "NEW"
        identical = bool(identity[str(seed)]["identical_initial_latent"])
        out[str(seed)] = {"old_baseline_label": baseline_labels[seed]["old_baseline"], "new_baseline_label": baseline_labels[seed]["new_baseline"], "identical_initial_latent": identical,
                          "status": "BASELINE_INFORMATIVE" if old_ok and new_ok and identical else "BASELINE_UNINFORMATIVE",
                          "reason": None if old_ok and new_ok and identical else ("old_label" if not old_ok else "new_label" if not new_ok else "initial_latent")}
    informative = [s for s in SEEDS if out[str(s)]["status"] == "BASELINE_INFORMATIVE"]
    return {"seeds": out, "informative_seeds": informative, "uninformative_count": len(SEEDS) - len(informative)}


def final_decision(action_status: str, round4c0_status: dict[str, str]) -> dict[str, Any]:
    strong = [e for e, s in {**round4c0_status, EDIT: action_status}.items() if s == "EDIT_STRONG"]
    if action_status == "EDIT_INVALID":
        return {"decision": "INVALID", "rationale": f"action edit INVALID; 4C-0 statuses {round4c0_status}", "BRANCH_CLOSED": False, "strong_edits": strong}
    if action_status == "EDIT_STRONG":
        return {"decision": "INVEST_4C1", "rationale": f"action EDIT_STRONG; strong edits {strong} of {list(round4c0_status) + [EDIT]}", "BRANCH_CLOSED": False, "strong_edits": strong}
    return {"decision": "STOP_DIFFUSION_REUSE", "rationale": f"action {action_status}; only {strong} strong; the trajectory-fork branch and diffusion intermediate-state reuse close", "BRANCH_CLOSED": True, "strong_edits": strong}


def run_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    preflight = require_controls(output_dir, provenance["provenance_hash"])
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
    mapping = {s: (str(v["trajectory_type"]), int(v["seed"]), int(v["switch_step"])) for s, v in json.loads(sealed_path.read_text())["mapping"].items()}
    if set(labels_doc.get("labels", {})) != set(mapping):
        raise GateError("Blinded labels do not cover exactly the blinded sample set")
    rows = read_csv(output_dir / "raw_results.csv")
    for row in rows:
        smoke._validate_result_artifacts(row, provenance["provenance_hash"])
    validate_key_set(rows, expected_keys())
    by_pool = {pool_key(r): r for r in rows}
    baseline_labels: dict[int, dict[str, str]] = {s: {} for s in SEEDS}
    fork_labels: dict[int, dict[int, str]] = {s: {} for s in SEEDS}
    qualitative = {}
    for sample_id, (ttype, seed, k) in mapping.items():
        entry = labels_doc["labels"][sample_id]
        label = entry.get("label") if isinstance(entry, dict) else None
        if label not in LABELS:
            raise GateError(f"Invalid or missing label for {sample_id}: {label!r}")
        if manifest["samples"][sample_id]["video_hash"] != by_pool[(ttype, seed, k)]["video_hash"]:
            raise GateError(f"Blinded sample {sample_id} does not match the persisted video")
        if ttype == "fork_new":
            fork_labels[seed][k] = label
        else:
            baseline_labels[seed][ttype] = label
        qualitative[f"{ttype}:{seed}:{k}"] = {"sample_id": sample_id, "label": label, "notes": entry.get("notes", "")}
    if any(set(fork_labels[s]) != set(KS) or set(baseline_labels[s]) != {"old_baseline", "new_baseline"} for s in SEEDS):
        raise GateError("Label matrix incomplete")
    atomic_json(output_dir / "unblinded_mapping.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"trajectory_type": v[0], "seed": v[1], "switch_step": v[2]} for s, v in mapping.items()}, "revealed_after_labels": True})
    atomic_json(output_dir / "qualitative_judgment.json", {"provenance_hash": provenance["provenance_hash"], "blinded": True, "labels": qualitative})

    evidence = json.loads((output_dir / "baseline_automated_evidence.json").read_text())
    informativeness = informativeness_from_labels(baseline_labels, evidence["seeds"])
    informative = informativeness["informative_seeds"]
    valid = preflight["status"] == "PASS" and len(informative) >= MIN_INFORMATIVE_SEEDS
    labels_inf = {s: fork_labels[s] for s in informative}
    gap = me.edit_gap(labels_inf) if informative else {"n_informative": 0, "k_star_by_seed": {}, "k_star_distribution": {}, "k_star_range": 0, "k_star_distinct": 0, "dispersion": False, "all_k_star_at_grid_top": False,
                                                       "monotonicity_violations_total": 0, "static_retry": {"p_fail_by_k": {}, "cost_by_k": {}, "best_k": None, "best_cost": None}, "c_traj": None, "gap_retry": None, "tau_frontier_descriptive": {}}
    status = me.classify_edit(gap, valid=valid) if informative else "EDIT_INVALID"
    verdict = final_decision(status, prereg["round4c0_reference"]["edit_status"])
    # descriptive secondaries: motion energy and CLIP per sample
    motion = {}
    for r in rows:
        if r["trajectory_type"] == "same_condition":
            continue
        key = pool_key(r)
        lab = fork_labels[key[1]][key[2]] if key[0] == "fork_new" else baseline_labels[key[1]][key[0]]
        motion[f"{key[0]}:{key[1]}:{key[2]}"] = {"motion_energy": motion_energy(np.load(r["final_video_npy"], allow_pickle=False)), "clip_new_minus_old": float(r["new_minus_old"]), "label": lab, "ordinal": ORDINAL[lab], "success": lab in SUCCESS_LABELS}
    ords = [v["ordinal"] for v in motion.values()]
    motion_summary = {
        "spearman_motion_vs_ordinal": me.spearman([v["motion_energy"] for v in motion.values()], ords),
        "spearman_clip_vs_ordinal": me.spearman([v["clip_new_minus_old"] for v in motion.values()], ords),
        "motion_by_label": {lab: [round(v["motion_energy"], 5) for v in motion.values() if v["label"] == lab] for lab in LABELS},
        "motion_separates_success": (max(v["motion_energy"] for v in motion.values() if v["success"]) < min(v["motion_energy"] for v in motion.values() if not v["success"])) if any(v["success"] for v in motion.values()) and any(not v["success"] for v in motion.values()) else None,
        "clip_separates_success": (min(v["clip_new_minus_old"] for v in motion.values() if v["success"]) > max(v["clip_new_minus_old"] for v in motion.values() if not v["success"])) if any(v["success"] for v in motion.values()) and any(not v["success"] for v in motion.values()) else None,
        "note": "descriptive only; never a validity gate; motion is whole-frame mean |frame difference|, lower should mean more parked",
    }
    summary = {
        **verdict, "action_status": status, "round4c0_edit_status": prereg["round4c0_reference"]["edit_status"], "edit": {**gap, "status": status, "label_matrix": {str(s): {str(k): fork_labels[s][k] for k in KS} for s in informative},
                                                                                                                             "all_seed_label_matrix": {str(s): {str(k): fork_labels[s][k] for k in KS} for s in SEEDS},
                                                                                                                             "label_counts": {lab: sum(1 for s in informative for k in KS if fork_labels[s][k] == lab) for lab in LABELS}},
        "baseline_informativeness": informativeness, "informative_seeds": informative, "descriptive_secondaries": motion_summary, "per_sample_descriptive": motion,
        "controls": {"preflight_status": preflight["status"]}, "trajectories_executed": len(rows),
        "thresholds": {"strong_min_gap": STRONG_MIN_GAP, "dispersion_min_range": DISPERSION_MIN_RANGE, "min_informative_seeds": MIN_INFORMATIVE_SEEDS},
        "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0], "provenance_hash": provenance["provenance_hash"], "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print(f"git_commit={provenance['git_commit']} relevant_dirty={provenance['relevant_git_status']}")
    print(f"preregistration_sha256={summary['preregistration_sha256']}")
    print(f"informative={informative} K*={gap['k_star_by_seed']} static_best_K={gap['static_retry']['best_k']} c_traj={gap['c_traj']} gap_retry={gap['gap_retry']} -> {status}")
    print(f"decision={summary['decision']} BRANCH_CLOSED={summary['BRANCH_CLOSED']}")
    return {k: v for k, v in summary.items() if k not in ("edit", "per_sample_descriptive", "descriptive_secondaries")}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    e = s["edit"]
    L = ["# Round 4C-0 extension - action edit (final decision for the trajectory-fork branch)", "", "## Decision", "", s["decision"], "", s["rationale"], "",
         "Screening only: n <= 4 seeds, no statistical claim.", "",
         "## Frozen definitions", "", f"- OLD: {OLD_PROMPT}", f"- NEW: {NEW_PROMPT}", f"- success := label in {list(SUCCESS_LABELS)}; K_i* := monotone cut on grid {list(KS)}; DISPERSION := >= 2 distinct K_i* and range >= {DISPERSION_MIN_RANGE}; gap_retry := (min_K C_static - C_traj) / 40 with a perfect free checker.",
         f"- EDIT_STRONG iff DISPERSION and gap_retry >= {STRONG_MIN_GAP}; EDIT_INVALID below {MIN_INFORMATIVE_SEEDS} informative seeds or C1 failure.", "- Final: INVEST_4C1 iff action EDIT_STRONG; otherwise STOP_DIFFUSION_REUSE.",
         "- Baseline informativeness: blinded human labels (OLD baseline must be OLD, NEW baseline must be NEW) + identical initial latent; no automated gate.", "",
         "## Baseline informativeness (blind)", "", "| seed | OLD baseline label | NEW baseline label | identical init | status |", "|---|---|---|---|---|"]
    for seed, v in s["baseline_informativeness"]["seeds"].items():
        L.append(f"| {seed} | {v['old_baseline_label']} | {v['new_baseline_label']} | {v['identical_initial_latent']} | {v['status']} |")
    L += ["", f"## Action edit - {e['status']}", "", "| seed | " + " | ".join(f"k{k}" for k in KS) + " | K_i* | informative |", "|---|" + "---|" * (len(KS) + 2)]
    for seed, row in e["all_seed_label_matrix"].items():
        inf = seed in e["label_matrix"]
        L.append(f"| {seed} | " + " | ".join(row[str(k)] for k in KS) + f" | {e['k_star_by_seed'].get(seed, '-')} | {inf} |")
    st = e["static_retry"]
    L += ["", f"- K* distribution: {e['k_star_distribution']}; distinct {e['k_star_distinct']}, range {e['k_star_range']}, dispersion {e['dispersion']}; label counts {e['label_counts']}",
          f"- p_fail by K: {st['p_fail_by_k']}", f"- static retry cost by K: {st['cost_by_k']} -> best K = {st['best_k']} at {st['best_cost']}", f"- trajectory oracle cost: {e['c_traj']}; gap_retry = {e['gap_retry']}",
          f"- descriptive tau frontier: {e['tau_frontier_descriptive']}", f"- monotonicity violations: {e['monotonicity_violations_total']}", "",
          "## Descriptive secondaries (never gates)", "", f"- {json.dumps({k: v for k, v in s['descriptive_secondaries'].items() if k != 'motion_by_label'})}", f"- motion by label: {s['descriptive_secondaries']['motion_by_label']}", "",
          "## Round 4C-0 context (pinned)", "", f"- edit statuses: {s['round4c0_edit_status']}; strong edits overall: {s['strong_edits']}", "",
          "## Claim boundary", "", s["claim_boundary"], "", f"BRANCH_CLOSED = {str(s['BRANCH_CLOSED']).lower()}", ""]
    (output_dir / "video_trajectory_fork_action_edit_killtest.md").write_text("\n".join(L))


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
