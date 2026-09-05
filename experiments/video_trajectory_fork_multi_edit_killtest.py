#!/usr/bin/env python3
"""Round 4C-0: replication screening - is the colour-edit oracle gap a one-edit peculiarity? (preregistered)

Two structural edits (object: sports car -> motorcycle; background: snowy road -> desert road) fork from the
same OLD trajectories (OLD prompt and the first four Round 4B seeds in frozen order). For each edit and each
informative seed the OLD checkpoint at K in {4, 8, 12, 16, 20, 24} is resumed under the NEW prompt; the 48
fork outputs are labelled blind on the five-level scale with edit-specific definitions.

This is a SCREENING experiment: n = 4 seeds per edit supports no statistical claim. Per edit (frozen):
    success(i, K)  := label in {NEW, MIXED_NEW_DOMINANT}
    K_i*           := largest grid K such that every grid K' <= K succeeds (0 if K=4 fails)
    DISPERSION     := at least two distinct K_i* values and max K_i* - min K_i* >= 8 (two grid steps)
    C_traj         := mean_i (40 - K_i*)
    C_static(K)    := (40 - K) + p_fail(K) * 40 for K in grid u {0}   (perfect, free checker; full retry)
    gap_retry      := (min_K C_static(K) - C_traj) / 40
    EDIT_STRONG iff DISPERSION and gap_retry >= 0.15; EDIT_INVALID if fewer than 3 informative seeds or C0/C1 fail;
    otherwise EDIT_WEAK.
Overall: INVEST_4C1 iff both edits STRONG; EXTEND_ONE_EDIT iff exactly one STRONG (one preregistered extra
edit class, then final); STOP_DIFFUSION_REUSE iff neither is STRONG; INVALID if any edit INVALID.

No probe, no mechanism, no speedup claim. Round 4B (colour) is a context pin and a C0 determinism reference only.
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
from experiments import video_trajectory_fork_oracle_gap_killtest as r4b  # noqa: E402

EXPERIMENT_VERSION = "video-trajectory-fork-multi-edit-screening-v1"
NAMESPACE = "video_trajectory_fork_multi_edit_killtest"
DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_multi_edit_killtest_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE
ROUND4B_DIR = REPO_ROOT / "results" / r4b.NAMESPACE
MODES = ("cpu", "baselines", "controls", "forks", "blind", "analyze")
SEEDS = r4b.SEEDS[:4]  # first four Round 4B seeds in frozen order: 5678, 6789, 7890, 8901
KS = (4, 8, 12, 16, 20, 24)
CONTROL_K = 8
TOTAL_STEPS = 40
OLD_PROMPT = conf.OLD_PROMPT
EDITS = ("object", "background")
EDIT_SPEC = {
    "object": {"new_prompt": "A red motorcycle driving on a snowy road, cinematic video", "old_concept": "sports car", "new_concept": "motorcycle"},
    "background": {"new_prompt": "A red sports car driving on a desert road, cinematic video", "old_concept": "snowy road", "new_concept": "desert road"},
}
LABELS = conf.LABELS
ORDINAL = conf.ORDINAL
SUCCESS_LABELS = ("NEW", "MIXED_NEW_DOMINANT")
STRONG_MIN_GAP = 0.15
DISPERSION_MIN_RANGE = 8
MIN_INFORMATIVE_SEEDS = 3
ALLOWED_FAILURES_DESCRIPTIVE = (1, 0)
EXTENSION_EDIT_CLASS = "action"  # the only edit class allowed for an EXTEND_ONE_EDIT follow-up
BASELINE_CAPTURE = [0, *KS, TOTAL_STEPS]
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_multi_edit_killtest.py",
    "experiments/video_trajectory_fork_multi_edit_killtest_config.yaml",
    "experiments/run_video_trajectory_fork_multi_edit_killtest_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_multi_edit_killtest.py",
    "experiments/video_trajectory_fork_oracle_gap_killtest.py",
    "experiments/video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = r4b.FORBIDDEN_OUTPUT_PARTS + (r4b.NAMESPACE,)
GateError = smoke.GateError
canonical_json = smoke.canonical_json
sha256_bytes = smoke.sha256_bytes
sha256_file = smoke.sha256_file
array_sha256 = smoke.array_sha256
atomic_json = smoke.atomic_json
read_csv = smoke.read_csv
CSV_FIELDS = tuple(smoke.RAW_FIELDS) + ("edit", "round4c_phase")
Key = tuple[str, int, str, int]


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
        raise ValueError(f"Seeds are frozen to the first four Round 4B seeds {SEEDS}")
    if tuple(int(v) for v in config["k_grid"]) != KS or int(config["control_k"]) != CONTROL_K:
        raise ValueError("K grid / control K are frozen")
    if config["prompts"]["old"] != OLD_PROMPT:
        raise ValueError("OLD prompt changed")
    if tuple(config["edits"]) != EDITS:
        raise ValueError("Edit set is frozen")
    for edit in EDITS:
        spec = config["prompts"][edit]
        if spec["new"] != EDIT_SPEC[edit]["new_prompt"] or spec["old_concept"] != EDIT_SPEC[edit]["old_concept"] or spec["new_concept"] != EDIT_SPEC[edit]["new_concept"]:
            raise ValueError(f"Edit prompt / concept pair changed: {edit}")
        if set(config["labels"]["definitions"][edit]) != set(LABELS):
            raise ValueError(f"Label definitions incomplete for {edit}")
    if tuple(config["labels"]["scale"]) != LABELS or {k: int(v) for k, v in config["labels"]["ordinal"].items()} != ORDINAL:
        raise ValueError("Label scale / ordinal changed")
    oracle = config["oracle"]
    if tuple(oracle["success_labels"]) != SUCCESS_LABELS or tuple(int(v) for v in oracle["allowed_failures_descriptive"]) != ALLOWED_FAILURES_DESCRIPTIVE:
        raise ValueError("success set / descriptive frontier frozen")
    gate = config["gate"]
    if float(gate["strong_min_gap"]) != STRONG_MIN_GAP or int(gate["dispersion_min_range"]) != DISPERSION_MIN_RANGE or int(gate["min_informative_seeds"]) != MIN_INFORMATIVE_SEEDS or gate["extension_edit_class"] != EXTENSION_EDIT_CLASS:
        raise ValueError("gap threshold / dispersion / informative minimum / extension class frozen")
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


def old_family(seed: int) -> dict[str, Any]:
    """Shared OLD trajectory family. The concept pair only feeds the row-level CLIP score of the OLD baseline;
    informativeness rescoring uses each edit's own pair."""
    spec = EDIT_SPEC[EDITS[0]]
    return {"id": f"old_seed{int(seed)}", "severity": "round4c", "seed": int(seed), "old_prompt": OLD_PROMPT, "new_prompt": OLD_PROMPT,
            "old_concept": spec["old_concept"], "new_concept": spec["new_concept"]}


def edit_family(edit: str, seed: int) -> dict[str, Any]:
    spec = EDIT_SPEC[edit]
    return {"id": f"{edit}_seed{int(seed)}", "severity": "round4c", "seed": int(seed), "edit": edit, "old_prompt": OLD_PROMPT,
            "new_prompt": spec["new_prompt"], "old_concept": spec["old_concept"], "new_concept": spec["new_concept"]}


def edit_of_row(row: dict[str, Any]) -> str:
    family = str(row["prompt_family"])
    return family.split("_seed")[0]


# --------------------------------------------------------------------------------------
# key set
# --------------------------------------------------------------------------------------
def expected_keys(informative: dict[str, list[int]] | None = None) -> set[Key]:
    keys: set[Key] = set()
    for seed in SEEDS:
        keys.add(("old", seed, "old_baseline", -1))
        for edit in EDITS:
            keys.add((edit, seed, "new_baseline", -1))
    inf = {edit: list(SEEDS) for edit in EDITS} if informative is None else {edit: [int(s) for s in informative[edit]] for edit in EDITS}
    control_seeds = sorted({s for edit in EDITS for s in inf[edit]})
    for seed in control_seeds:
        keys.add(("old", seed, "same_condition", CONTROL_K))
    for edit in EDITS:
        for seed in inf[edit]:
            for k in KS:
                keys.add((edit, seed, "fork_new", k))
    return keys


def row_key(row: dict[str, Any]) -> Key:
    return edit_of_row(row), int(row["seed"]), str(row["trajectory_type"]), int(row["switch_step"])


def validate_key_set(rows: list[dict[str, Any]], expected: set[Key]) -> None:
    keys = [row_key(r) for r in rows]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    actual = set(keys)
    if duplicates or actual != expected:
        raise GateError(f"Result key-set mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}, duplicates={duplicates}")


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[Key, dict[str, Any]] = {}
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))
    tmp.replace(path)


def _tag(row: dict[str, Any], phase: str) -> dict[str, Any]:
    row["edit"] = edit_of_row(row)
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


def round4b_reference() -> dict[str, Any]:
    """Frozen Round 4B pins: preregistration/summary hashes, OLD baseline hashes (C0 reference), colour K* (descriptive)."""
    for name in ("preregistration.sha256", "summary.json"):
        if not (ROUND4B_DIR / name).exists():
            raise GateError(f"Round 4B reference missing: {ROUND4B_DIR / name}")
    summary = json.loads((ROUND4B_DIR / "summary.json").read_text())
    old_hashes = {}
    for seed in SEEDS:
        row = json.loads((ROUND4B_DIR / "rows" / f"red_to_blue_seed{seed}" / "old_baseline.json").read_text())
        old_hashes[str(seed)] = {"final_latent_hash": row["final_latent_hash"], "video_hash": row["video_hash"], "initial_latent_hash": row["initial_latent_hash"]}
    return {
        "namespace": r4b.NAMESPACE,
        "preregistration_sha256": (ROUND4B_DIR / "preregistration.sha256").read_text().split()[0],
        "summary_sha256": sha256_file(ROUND4B_DIR / "summary.json"),
        "decision": summary["decision"],
        "G": summary["G"],
        "colour_k_star_by_seed": summary["primary"]["k_star_by_seed"],
        "old_baseline_hashes": old_hashes,
    }


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan = smoke.scheduler_plan(conf.seed_config(config, SEEDS[0]))
    schedule = [float(v) for v in plan["timesteps"]]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "title": "Round 4C-0 - Replication screening: is the colour-edit oracle gap a one-edit peculiarity?",
        "question": "For an object edit and a background edit forking from the same OLD trajectories (4 seeds), do per-trajectory K* disperse and does the trajectory oracle beat the best detect-and-retry static depth by >= 0.15 of a generation?",
        "statistical_claim": "none; screening only (n = 4 seeds per edit)",
        "round4b_reference": round4b_reference(),
        "context_only": r4b.context_pins(),
        "prompts": config["prompts"],
        "edits": list(EDITS),
        "seeds": list(SEEDS),
        "seed_reuse_rationale": "First four Round 4B seeds in frozen order (not selected on any outcome). Same OLD prompt, so the OLD trajectories are deterministic re-runs (C0 checks bit-exactness against Round 4B) and colour K* is known for a within-trajectory descriptive comparison.",
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
# Phase 1: baselines (shared OLD + one NEW per edit)
# --------------------------------------------------------------------------------------
def run_baselines(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    reference = prereg["round4b_reference"]["old_baseline_hashes"]
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    try:
        for seed in SEEDS:
            run_config = conf.seed_config(config, seed)
            scheduler = prereg["scheduler_plan"]
            fam_old = old_family(seed)
            evaluators = {edit: smoke.ConceptEvaluator(run_config, edit_family(edit, seed)) for edit in EDITS}
            old_row, old_meta = smoke._run_baseline(omni, evaluators[EDITS[0]], run_config, provenance, scheduler, output_dir, fam_old, which="old", capture_steps=list(BASELINE_CAPTURE))
            old_video = np.load(old_row["final_video_npy"], allow_pickle=False)
            old_initial = array_sha256(smoke._probe_array(smoke._record_by_step(old_meta, 0)))
            conf.save_frames(old_video, output_dir / "qualitative" / fam_old["id"], "old_baseline")
            rows.append(_tag(old_row, "baseline"))
            ref = reference[str(seed)]
            seed_evidence: dict[str, Any] = {
                "c0_old_bit_exact_vs_round4b": old_row["final_latent_hash"] == ref["final_latent_hash"] and old_row["video_hash"] == ref["video_hash"],
                "initial_latent_hash": old_initial,
                "initial_matches_round4b": old_initial == ref["initial_latent_hash"],
                "edits": {},
            }
            for edit in EDITS:
                fam = edit_family(edit, seed)
                new_row, new_meta = smoke._run_baseline(omni, evaluators[edit], run_config, provenance, scheduler, output_dir, fam, which="new", capture_steps=[0, TOTAL_STEPS])
                new_initial = array_sha256(smoke._probe_array(smoke._record_by_step(new_meta, 0)))
                conf.save_frames(np.load(new_row["final_video_npy"], allow_pickle=False), output_dir / "qualitative" / fam["id"], "new_baseline")
                rows.append(_tag(new_row, "baseline"))
                old_clip = evaluators[edit].score(old_video)["new_minus_old"]
                seed_evidence["edits"][edit] = {
                    "old_new_minus_old": float(old_clip),
                    "new_new_minus_old": float(new_row["new_minus_old"]),
                    "automated_sign_ok": float(old_clip) < 0.0 < float(new_row["new_minus_old"]),
                    "identical_initial_latent": new_initial == old_initial,
                }
            evidence[str(seed)] = seed_evidence
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys() if k[2] in ("old_baseline", "new_baseline")})
    write_csv(output_dir / "baseline_results.csv", rows)
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows) if (output_dir / "raw_results.csv").exists() else rows)
    atomic_json(output_dir / "baseline_automated_evidence.json", {"provenance_hash": provenance["provenance_hash"], "seeds": evidence})
    template = output_dir / "baseline_judgment_template.json"
    if not template.exists():
        atomic_json(template, {
            "provenance_hash": provenance["provenance_hash"], "fork_outcomes_examined": False,
            "instructions": "Inspect ONLY baseline videos/frames. old_matches_expected: OLD clearly shows a red sports car on a snowy road. Per edit: new_matches_expected (object: clearly a red motorcycle on a snowy road; background: clearly a red sports car on a desert road) and clearly_different from OLD.",
            "seeds": {str(seed): {"old_matches_expected": None, "edits": {edit: {"new_matches_expected": None, "clearly_different": None, "notes": ""} for edit in EDITS}} for seed in SEEDS},
        })
    c0 = {s: e["c0_old_bit_exact_vs_round4b"] for s, e in evidence.items()}
    return {"mode": "baselines", "rows": len(rows), "c0_bit_exact": c0, "automated": {s: {e: v["automated_sign_ok"] for e, v in ev["edits"].items()} for s, ev in evidence.items()}, "next": "fill baseline_judgment.json, then run controls"}


def evaluate_baseline_informativeness(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    judgment_path = output_dir / "baseline_judgment.json"
    if not judgment_path.exists():
        raise GateError("Fill baseline_judgment.json from baseline_judgment_template.json before running controls")
    judgment = json.loads(judgment_path.read_text())
    if judgment.get("provenance_hash") != provenance_hash or judgment.get("fork_outcomes_examined") is not False:
        raise GateError("Baseline judgment provenance mismatch or fork outcomes examined before freezing")
    evidence = json.loads((output_dir / "baseline_automated_evidence.json").read_text())
    if evidence.get("provenance_hash") != provenance_hash:
        raise GateError("Baseline automated evidence provenance mismatch")
    per_pair: dict[str, dict[str, Any]] = {edit: {} for edit in EDITS}
    c0_failures = []
    for seed in SEEDS:
        ev = evidence["seeds"][str(seed)]
        if not ev["c0_old_bit_exact_vs_round4b"]:
            c0_failures.append(seed)
        entry = judgment.get("seeds", {}).get(str(seed))
        if not isinstance(entry, dict) or not isinstance(entry.get("old_matches_expected"), bool):
            raise GateError(f"Baseline judgment for seed {seed} must contain an explicit old_matches_expected boolean")
        for edit in EDITS:
            e = entry.get("edits", {}).get(edit)
            if not isinstance(e, dict) or any(not isinstance(e.get(k), bool) for k in ("new_matches_expected", "clearly_different")):
                raise GateError(f"Baseline judgment for seed {seed}/{edit} must contain explicit booleans")
            manual = {"old_matches_expected": entry["old_matches_expected"], "new_matches_expected": e["new_matches_expected"], "clearly_different": e["clearly_different"]}
            manual_ok = all(manual.values())
            auto_ok = bool(ev["edits"][edit]["automated_sign_ok"])
            identical = bool(ev["edits"][edit]["identical_initial_latent"])
            informative = manual_ok and auto_ok and identical
            per_pair[edit][str(seed)] = {"manual": manual, "manual_ok": manual_ok, "automated_sign_ok": auto_ok, "identical_initial_latent": identical, "metric_disagreement": manual_ok != auto_ok,
                                         "status": "BASELINE_INFORMATIVE" if informative else "BASELINE_UNINFORMATIVE",
                                         "reason": None if informative else ("manual" if not manual_ok else "automated_sign" if not auto_ok else "initial_latent")}
    informative_seeds = {edit: [s for s in SEEDS if per_pair[edit][str(s)]["status"] == "BASELINE_INFORMATIVE"] for edit in EDITS}
    return {"pairs": per_pair, "informative_seeds": informative_seeds, "uninformative_count": {edit: len(SEEDS) - len(informative_seeds[edit]) for edit in EDITS}, "c0_failures": c0_failures}


# --------------------------------------------------------------------------------------
# Phase 2: spot exactness control (shared OLD -> OLD at K=8)
# --------------------------------------------------------------------------------------
def _old_records(output_dir: Path, seed: int) -> dict[int, dict[str, Any]]:
    metadata = json.loads((output_dir / "rows" / old_family(seed)["id"] / "old_baseline_metadata.json").read_text())["worker_trajectory_probe"]
    return {int(r["step_index"]): r for r in metadata["records"]}


def _old_reference(output_dir: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return smoke._load_saved_baseline(output_dir, old_family(seed)["id"], "old_baseline")


def run_controls(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    informativeness = evaluate_baseline_informativeness(output_dir, provenance["provenance_hash"])
    preflight: dict[str, Any] = {"provenance_hash": provenance["provenance_hash"], "baseline_informativeness": informativeness, "seeds": {}}
    short = {edit: len(informativeness["informative_seeds"][edit]) for edit in EDITS if len(informativeness["informative_seeds"][edit]) < MIN_INFORMATIVE_SEEDS}
    if short or informativeness["c0_failures"]:
        preflight["status"] = "INVALID_BASELINES"
        preflight["reason"] = f"informative seeds below {MIN_INFORMATIVE_SEEDS}: {short}; C0 failures: {informativeness['c0_failures']}"
        atomic_json(output_dir / "preflight.json", preflight)
        raise GateError(preflight["reason"])
    control_seeds = sorted({s for edit in EDITS for s in informativeness["informative_seeds"][edit]})
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for seed in control_seeds:
            run_config = conf.seed_config(config, seed)
            fam = old_family(seed)
            evaluator = smoke.ConceptEvaluator(run_config, fam)
            scheduler = prereg["scheduler_plan"]
            old_ref = _old_reference(output_dir, seed)
            records = _old_records(output_dir, seed)
            initial_hash = array_sha256(smoke._probe_array(records[0]))
            row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, fam,
                                     label=f"same_condition_k{CONTROL_K:02d}", trajectory_type="same_condition", switch_step=CONTROL_K,
                                     prompt=OLD_PROMPT, input_tensor=smoke._load_probe_tensor(records[CONTROL_K]["latent_path"]),
                                     initial_hash=initial_hash, old_reference=old_ref, new_reference=None, exact_reference=old_ref)
            rows.append(_tag(row, "control"))
            preflight["seeds"][str(seed)] = {
                "same_condition_exact_k08": row["control_exact"] is True,
                "old_checkpoint_hashes": {str(k): array_sha256(smoke._probe_array(records[k])) for k in KS},
                "resume_input_matches_checkpoint": row["resume_input_hash"] == array_sha256(smoke._probe_array(records[CONTROL_K])),
                "scheduler_euler": str(row["scheduler_class"]).endswith(smoke.EXPECTED_SCHEDULER),
            }
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys(informativeness["informative_seeds"]) if k[2] == "same_condition"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    c1 = all(e["same_condition_exact_k08"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"] for e in preflight["seeds"].values())
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
    control_seeds = {str(s) for edit in EDITS for s in informative[edit]}
    if control_seeds != set(preflight["seeds"]) or any(len(informative[edit]) < MIN_INFORMATIVE_SEEDS for edit in EDITS) or preflight["baseline_informativeness"]["c0_failures"]:
        raise GateError("Control gate covers a different or insufficient seed set")
    for seed, e in preflight["seeds"].items():
        if not (e["same_condition_exact_k08"] and e["resume_input_matches_checkpoint"] and e["scheduler_euler"]):
            raise GateError(f"Control gate failed for seed {seed}")
    return preflight


# --------------------------------------------------------------------------------------
# Phase 3: forks
# --------------------------------------------------------------------------------------
def run_forks(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = {edit: [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"][edit]] for edit in EDITS}
    omni = smoke._build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for edit in EDITS:
            for seed in informative[edit]:
                run_config = conf.seed_config(config, seed)
                fam = edit_family(edit, seed)
                evaluator = smoke.ConceptEvaluator(run_config, fam)
                scheduler = prereg["scheduler_plan"]
                old_ref = _old_reference(output_dir, seed)
                new_ref = smoke._load_saved_baseline(output_dir, fam["id"], "new_baseline")
                records = _old_records(output_dir, seed)
                initial_hash = array_sha256(smoke._probe_array(records[0]))
                for k in KS:
                    checkpoint = smoke._load_probe_tensor(records[k]["latent_path"])
                    if array_sha256(checkpoint.float().numpy()) != preflight["seeds"][str(seed)]["old_checkpoint_hashes"][str(k)]:
                        raise GateError(f"OLD checkpoint for seed {seed} k={k} changed since controls")
                    row = smoke._execute_one(omni, evaluator, run_config, provenance, scheduler, output_dir, fam,
                                             label=f"fork_new_k{k:02d}", trajectory_type="fork_new", switch_step=k, prompt=fam["new_prompt"],
                                             input_tensor=checkpoint, initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref, exact_reference=None)
                    rows.append(_tag(row, "fork"))
    finally:
        omni.shutdown()
    validate_key_set(rows, {k for k in expected_keys(informative) if k[2] == "fork_new"})
    write_csv(output_dir / "raw_results.csv", merge_rows(read_csv(output_dir / "raw_results.csv"), rows))
    return {"mode": "forks", "rows": len(rows), "informative_seeds": informative, "next": "run blind, label blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 4: blinding (both edits in one pool)
# --------------------------------------------------------------------------------------
def run_blind(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    require_preregistration(output_dir, provenance)
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = {edit: [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"][edit]] for edit in EDITS}
    rows = [r for r in read_csv(output_dir / "raw_results.csv") if r["trajectory_type"] == "fork_new"]
    validate_key_set(rows, {k for k in expected_keys(informative) if k[2] == "fork_new"})
    if (output_dir / "blinded_labels.json").exists():
        raise GateError("blinded_labels.json already exists; the blinded set must not be regenerated after labelling")
    prereg_sha = (output_dir / "preregistration.sha256").read_text().split()[0]
    keys = [(edit_of_row(r), int(r["seed"]), int(r["switch_step"])) for r in rows]
    mapping = conf.blinded_assignment(keys, prereg_sha)  # type: ignore[arg-type]
    by_key = {(edit_of_row(r), int(r["seed"]), int(r["switch_step"])): r for r in rows}
    blind_dir = output_dir / "qualitative" / "blinded"
    blind_dir.mkdir(parents=True, exist_ok=True)
    samples = {}
    for sample_id, key in mapping.items():
        row = by_key[key]  # type: ignore[index]
        frames = conf.save_frames(np.load(row["final_video_npy"], allow_pickle=False), blind_dir, sample_id)
        mp4 = blind_dir / f"{sample_id}.mp4"
        shutil.copyfile(row["final_video_mp4"], mp4)
        samples[sample_id] = {"mp4": str(mp4), "frames": frames, "video_hash": row["video_hash"], "edit": key[0]}
    atomic_json(output_dir / "blinded_mapping.sealed.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"edit": v[0], "seed": v[1], "switch_step": v[2]} for s, v in mapping.items()}})
    atomic_json(output_dir / "blinded_manifest.json", {"provenance_hash": provenance["provenance_hash"], "sealed_mapping_sha256": sha256_file(output_dir / "blinded_mapping.sealed.json"),
                                                        "instructions": "Label each sample from its five fixed frames and full video without opening blinded_mapping.sealed.json. The edit class is visible from the video and is listed per sample; seed and K are hidden. Use the definitions of that edit.",
                                                        "labels": list(LABELS), "definitions": config["labels"]["definitions"], "samples": samples})
    if not (output_dir / "blinded_labels_template.json").exists():
        atomic_json(output_dir / "blinded_labels_template.json", {"provenance_hash": provenance["provenance_hash"], "mapping_revealed_before_labelling": False, "labels": {s: {"label": None, "notes": ""} for s in sorted(mapping)}})
    return {"mode": "blind", "samples": len(mapping), "per_edit": {edit: sum(1 for v in mapping.values() if v[0] == edit) for edit in EDITS}, "next": "fill blinded_labels.json, then analyze"}


# --------------------------------------------------------------------------------------
# Phase 5: frozen analysis
# --------------------------------------------------------------------------------------
success = r4b.success


def trajectory_oracle_k(labels_by_k: dict[int, str]) -> dict[str, Any]:
    k_star, violations, failed_before = 0, 0, False
    for k in KS:
        ok = success(labels_by_k[k])
        if ok and failed_before:
            violations += 1
        if ok and not failed_before:
            k_star = k
        if not ok:
            failed_before = True
    return {"k_star": k_star, "monotonicity_violations": violations}


def retry_static_costs(labels: dict[int, dict[int, str]]) -> dict[str, Any]:
    """Expected steps per request for a static depth with a perfect free checker and full retry on failure."""
    n = len(labels)
    p_fail = {0: 0.0, **{k: sum(1 for s in labels if not success(labels[s][k])) / n for k in KS}}
    cost = {k: (TOTAL_STEPS - k) + p_fail[k] * TOTAL_STEPS for k in (0, *KS)}
    best_k = min(cost, key=lambda k: (cost[k], -k))
    return {"p_fail_by_k": {str(k): p_fail[k] for k in (0, *KS)}, "cost_by_k": {str(k): cost[k] for k in (0, *KS)}, "best_k": best_k, "best_cost": cost[best_k]}


def edit_gap(labels: dict[int, dict[int, str]]) -> dict[str, Any]:
    per_seed = {s: trajectory_oracle_k(labels[s]) for s in labels}
    k_stars = {s: per_seed[s]["k_star"] for s in labels}
    static = retry_static_costs(labels)
    c_traj = float(np.mean([TOTAL_STEPS - k for k in k_stars.values()]))
    n = len(labels)
    frontier = {}
    for f in ALLOWED_FAILURES_DESCRIPTIVE:
        cands = [k for k in KS if sum(1 for s in labels if not success(labels[s][k])) <= f]
        ks = max(cands) if cands else 0
        frontier[str(f)] = {"allowed_failures": f, "nominal_reliability": 1 - f / n, "k_static": ks, "gap_fraction": float(np.mean([k_stars[s] - ks for s in labels])) / TOTAL_STEPS}
    return {
        "n_informative": n,
        "k_star_by_seed": {str(s): k_stars[s] for s in labels},
        "k_star_distribution": {str(k): sum(1 for v in k_stars.values() if v == k) for k in (0, *KS)},
        "k_star_range": max(k_stars.values()) - min(k_stars.values()),
        "k_star_distinct": len(set(k_stars.values())),
        "dispersion": len(set(k_stars.values())) >= 2 and (max(k_stars.values()) - min(k_stars.values())) >= DISPERSION_MIN_RANGE,
        "all_k_star_at_grid_top": all(v == KS[-1] for v in k_stars.values()),
        "monotonicity_violations_total": sum(per_seed[s]["monotonicity_violations"] for s in labels),
        "static_retry": static,
        "c_traj": c_traj,
        "gap_retry": (static["best_cost"] - c_traj) / TOTAL_STEPS,
        "tau_frontier_descriptive": frontier,
    }


def classify_edit(gap: dict[str, Any], *, valid: bool) -> str:
    if not valid:
        return "EDIT_INVALID"
    if gap["dispersion"] and gap["gap_retry"] >= STRONG_MIN_GAP:
        return "EDIT_STRONG"
    return "EDIT_WEAK"


def decide(edit_status: dict[str, str]) -> dict[str, Any]:
    strong = sum(1 for v in edit_status.values() if v == "EDIT_STRONG")
    if any(v == "EDIT_INVALID" for v in edit_status.values()):
        return {"decision": "INVALID", "rationale": f"edit status {edit_status}; no scientific decision", "NEXT": "report only; a re-run needs a fresh namespace and an explicit decision"}
    if strong == len(edit_status):
        return {"decision": "INVEST_4C1", "rationale": f"both edits EDIT_STRONG: {edit_status}", "NEXT": "design Round 4C-1 (more seeds per edit; automated evaluator only if CLIP separated labels in both edits here)"}
    if strong == 1:
        return {"decision": "EXTEND_ONE_EDIT", "rationale": f"exactly one edit EDIT_STRONG: {edit_status}", "NEXT": f"one preregistered extra edit class ({EXTENSION_EDIT_CLASS}), same design; then final STOP or INVEST"}
    return {"decision": "STOP_DIFFUSION_REUSE", "rationale": f"no edit EDIT_STRONG: {edit_status}; the trajectory-fork line and diffusion intermediate-state reuse close", "NEXT": "no further GPU work on diffusion reuse"}


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = np.array(ranks(x)), np.array(ranks(y))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def run_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    preflight = require_controls(output_dir, provenance["provenance_hash"])
    informative = {edit: [int(s) for s in preflight["baseline_informativeness"]["informative_seeds"][edit]] for edit in EDITS}
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
    mapping = {s: (str(v["edit"]), int(v["seed"]), int(v["switch_step"])) for s, v in json.loads(sealed_path.read_text())["mapping"].items()}
    if set(labels_doc.get("labels", {})) != set(mapping):
        raise GateError("Blinded labels do not cover exactly the blinded sample set")
    rows = read_csv(output_dir / "raw_results.csv")
    for row in rows:
        smoke._validate_result_artifacts(row, provenance["provenance_hash"])
    validate_key_set(rows, expected_keys(informative))
    forks = {(edit_of_row(r), int(r["seed"]), int(r["switch_step"])): r for r in rows if r["trajectory_type"] == "fork_new"}
    labels: dict[str, dict[int, dict[int, str]]] = {edit: {s: {} for s in informative[edit]} for edit in EDITS}
    qualitative = {}
    for sample_id, (edit, seed, k) in mapping.items():
        entry = labels_doc["labels"][sample_id]
        label = entry.get("label") if isinstance(entry, dict) else None
        if label not in LABELS:
            raise GateError(f"Invalid or missing label for {sample_id}: {label!r}")
        if manifest["samples"][sample_id]["video_hash"] != forks[(edit, seed, k)]["video_hash"]:
            raise GateError(f"Blinded sample {sample_id} does not match the persisted fork video")
        labels[edit][seed][k] = label
        qualitative[f"{edit}:{seed}:{k}"] = {"sample_id": sample_id, "label": label, "notes": entry.get("notes", "")}
    if any(set(labels[edit][s]) != set(KS) for edit in EDITS for s in informative[edit]):
        raise GateError("Label matrix incomplete")
    atomic_json(output_dir / "unblinded_mapping.json", {"provenance_hash": provenance["provenance_hash"], "mapping": {s: {"edit": v[0], "seed": v[1], "switch_step": v[2]} for s, v in mapping.items()}, "revealed_after_labels": True})
    atomic_json(output_dir / "qualitative_judgment.json", {"provenance_hash": provenance["provenance_hash"], "blinded": True, "labels": qualitative})

    per_edit: dict[str, Any] = {}
    status: dict[str, str] = {}
    for edit in EDITS:
        gap = edit_gap(labels[edit])
        valid = preflight["status"] == "PASS" and len(informative[edit]) >= MIN_INFORMATIVE_SEEDS
        status[edit] = classify_edit(gap, valid=valid)
        clip_succ = [float(forks[(edit, s, k)]["new_minus_old"]) for s in informative[edit] for k in KS if success(labels[edit][s][k])]
        clip_fail = [float(forks[(edit, s, k)]["new_minus_old"]) for s in informative[edit] for k in KS if not success(labels[edit][s][k])]
        per_edit[edit] = {
            **gap, "status": status[edit],
            "label_matrix": {str(s): {str(k): labels[edit][s][k] for k in KS} for s in informative[edit]},
            "label_counts": {lab: sum(1 for s in informative[edit] for k in KS if labels[edit][s][k] == lab) for lab in LABELS},
            "clip_separability": {"min_success": min(clip_succ) if clip_succ else None, "max_failure": max(clip_fail) if clip_fail else None,
                                  "separated": (min(clip_succ) > max(clip_fail)) if clip_succ and clip_fail else None},
            "fork_table": [{"seed": s, "k": k, "label": labels[edit][s][k], "success": success(labels[edit][s][k]), "clip_new_minus_old": float(forks[(edit, s, k)]["new_minus_old"]),
                            "ssim_old": float(forks[(edit, s, k)]["ssim_to_old"]), "ssim_new": float(forks[(edit, s, k)]["ssim_to_new"]), "wall_s": float(forks[(edit, s, k)]["wall_time_s"])} for s in informative[edit] for k in KS],
        }
    colour = {int(s): int(v) for s, v in prereg["round4b_reference"]["colour_k_star_by_seed"].items()}
    shared = sorted(set(informative["object"]) & set(informative["background"]))
    within = {
        "seeds": shared,
        "colour_k_star_round4b": {str(s): colour.get(s) for s in shared},
        "object_k_star": {str(s): per_edit["object"]["k_star_by_seed"][str(s)] for s in shared},
        "background_k_star": {str(s): per_edit["background"]["k_star_by_seed"][str(s)] for s in shared},
        "spearman_object_vs_background": spearman([per_edit["object"]["k_star_by_seed"][str(s)] for s in shared], [per_edit["background"]["k_star_by_seed"][str(s)] for s in shared]),
        "spearman_object_vs_colour": spearman([per_edit["object"]["k_star_by_seed"][str(s)] for s in shared], [colour[s] for s in shared]),
        "spearman_background_vs_colour": spearman([per_edit["background"]["k_star_by_seed"][str(s)] for s in shared], [colour[s] for s in shared]),
        "note": "descriptive only; n <= 4 with ties, anecdotal; colour K* comes from the frozen Round 4B summary and uses a different K grid",
    }
    verdict = decide(status)
    summary = {
        **verdict, "edit_status": status, "edits": per_edit, "within_trajectory_descriptive": within,
        "informative_seeds": informative, "controls": {"preflight_status": preflight["status"], "c0_failures": preflight["baseline_informativeness"]["c0_failures"], "baseline_informativeness": preflight["baseline_informativeness"]},
        "trajectories_executed": len(rows), "thresholds": {"strong_min_gap": STRONG_MIN_GAP, "dispersion_min_range": DISPERSION_MIN_RANGE, "min_informative_seeds": MIN_INFORMATIVE_SEEDS},
        "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0], "provenance_hash": provenance["provenance_hash"], "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print_console(summary, provenance)
    return {k: v for k, v in summary.items() if k not in ("edits", "within_trajectory_descriptive", "controls")}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    L = ["# Round 4C-0 - Replication screening: is the colour-edit oracle gap a one-edit peculiarity?", "", "## Decision", "", s["decision"], "", s["rationale"], "", f"Next: {s['NEXT']}", "",
         "Screening only: n = 4 seeds per edit, no statistical claim.", "",
         "## Frozen definitions", "", f"- success := label in {list(SUCCESS_LABELS)}; K_i* := monotone cut on grid {list(KS)}; DISPERSION := >= 2 distinct K_i* and range >= {DISPERSION_MIN_RANGE}; C_traj := mean(40 - K_i*); C_static(K) := (40 - K) + p_fail(K) * 40 with a perfect free checker; gap_retry := (min_K C_static - C_traj) / 40.",
         f"- Per edit: EDIT_STRONG iff DISPERSION and gap_retry >= {STRONG_MIN_GAP}; EDIT_INVALID below {MIN_INFORMATIVE_SEEDS} informative seeds or C0/C1 failure; otherwise EDIT_WEAK.",
         f"- Overall: INVEST_4C1 iff both STRONG; EXTEND_ONE_EDIT ({EXTENSION_EDIT_CLASS}) iff exactly one STRONG; STOP_DIFFUSION_REUSE iff none; INVALID if any edit INVALID.", "",
         "## Controls", "", f"- Preflight {s['controls']['preflight_status']}; C0 failures {s['controls']['c0_failures']}; informative seeds " + ", ".join(f"{e}: {len(s['informative_seeds'][e])}" for e in EDITS), ""]
    for edit in EDITS:
        e = s["edits"][edit]
        L += [f"## Edit: {edit} - {e['status']}", "", "| seed | " + " | ".join(f"k{k}" for k in KS) + " | K_i* |", "|---|" + "---|" * (len(KS) + 1)]
        for seed, row in e["label_matrix"].items():
            L.append(f"| {seed} | " + " | ".join(row[str(k)] for k in KS) + f" | {e['k_star_by_seed'][seed]} |")
        st = e["static_retry"]
        L += ["", f"- K* distribution: {e['k_star_distribution']}; distinct {e['k_star_distinct']}, range {e['k_star_range']}, dispersion {e['dispersion']}, all at grid top {e['all_k_star_at_grid_top']}; label counts: {e['label_counts']}",
              f"- p_fail by K: {st['p_fail_by_k']}", f"- static retry cost by K: {st['cost_by_k']} -> best K = {st['best_k']} at {st['best_cost']:.2f} steps",
              f"- trajectory oracle cost: {e['c_traj']:.2f} steps; gap_retry = {e['gap_retry']:+.4f}",
              f"- descriptive tau frontier (allowed failures -> K_static, gap): " + "; ".join(f"{f['allowed_failures']} ({f['nominal_reliability']:.0%}) -> {f['k_static']}, {f['gap_fraction']:+.4f}" for f in e["tau_frontier_descriptive"].values()),
              f"- monotonicity violations: {e['monotonicity_violations_total']}; CLIP separability: {e['clip_separability']}", ""]
    w = s["within_trajectory_descriptive"]
    L += ["## Within-trajectory comparison (descriptive)", "", f"- shared seeds {w['seeds']}", f"- colour K* (Round 4B): {w['colour_k_star_round4b']}", f"- object K*: {w['object_k_star']}", f"- background K*: {w['background_k_star']}",
          f"- Spearman object~background {w['spearman_object_vs_background']}, object~colour {w['spearman_object_vs_colour']}, background~colour {w['spearman_background_vs_colour']}", f"- {w['note']}", "",
          "## Claim boundary", "", s["claim_boundary"], ""]
    (output_dir / "video_trajectory_fork_multi_edit_killtest.md").write_text("\n".join(L))


def print_console(s: dict[str, Any], provenance: dict[str, Any]) -> None:
    print(f"git_commit={provenance['git_commit']} relevant_dirty={provenance['relevant_git_status']}")
    print(f"preregistration_sha256={s['preregistration_sha256']}")
    for edit in EDITS:
        e = s["edits"][edit]
        print(f"{edit}: n={e['n_informative']} K*={e['k_star_by_seed']} static_best_K={e['static_retry']['best_k']} cost={e['static_retry']['best_cost']:.2f} c_traj={e['c_traj']:.2f} gap_retry={e['gap_retry']:+.4f} -> {e['status']}")
    print(f"decision={s['decision']} next={s['NEXT']}")


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
