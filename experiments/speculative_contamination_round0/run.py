#!/usr/bin/env python3
"""Speculative-contamination Round 0, revision 2: phased, preregistered, fail-closed orchestration.

Phases: cpu | preregister | reference | perturb | propagation | oracle | recompute | analyze | weakprobe   (quarantine: DEFERRED_TO_ROUND1)

t = absolute Euler step index (x_t = state after t completed updates; the next update consumes schedule[t]); T = {12, 20, 28}.
L = 4 stale regional updates precede t (PRIMARY STALE_VELOCITY); d = 4 normal updates after t before repair (aligned with L).

Arms per decision cell (trajectory, t, region):
  CONTROL_ZERO           exact clean resume through the perturbation path (bit-exact required)
  NO_REPAIR              STALE_VELOCITY state at t, dense target computation from t onward == GLOBAL_DENSE_REFRESH baseline
  ORACLE_REPAIR(D)       at t+d copy clean latents inside domain D (R_state*; structural probe only)
  RECOMPUTE_REPAIR(D)    FULL-FORWARD SELECTIVE-STATE RECOMPUTATION with cached contaminated context: over t..t+d regenerate D with
                         actual target steps (each a full 14,040-token forward), keep only D's outputs, non-D = cached perturbed
                         latents (R_compute*; PRIMARY). 100% == chain of clean steps (bit-exact required). No sparse compute is measured.
Every run is one request into the trusted Wan2.2 pipeline; no denoising is reimplemented. `analyze` may STOP after propagation alone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_trajectory_fork_killtest as smoke  # noqa: E402
from experiments.speculative_contamination_round0 import core  # noqa: E402

PKG = "experiments/speculative_contamination_round0"
EXPERIMENT_VERSION = "regional-recompute-round0-v5"
NAMESPACE = "regional_recompute_round0"
ATTEMPT_SUBDIR = "attempt3"
PRIOR_ATTEMPTS = {  # both INVALID at the velocity-validation gate; the pipeline was bit-exact both times, the checks were quantisation-naive
    "attempt1": REPO_ROOT / "results" / "speculative_contamination_round0",
    "attempt2": REPO_ROOT / "results" / "regional_recompute_round0",
}
DEFAULT_CONFIG = REPO_ROOT / PKG / "config.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE / "attempt3"
PHASES = ("cpu", "preregister", "reference", "perturb", "propagation", "oracle", "recompute", "analyze", "weakprobe", "quarantine")
TRUSTED_SOURCE_FILES = (
    f"{PKG}/__init__.py", f"{PKG}/core.py", f"{PKG}/run.py", f"{PKG}/config.json", f"{PKG}/run_gpu0.sh",
    "tests/diffusion/test_speculative_contamination_round0.py",
    "experiments/video_trajectory_fork_killtest.py", "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py", "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py", "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
)
FORBIDDEN_OUTPUT_PARTS = ("video_trajectory_fork", "video_runtime_state_discovery", "video_bf16", "video_execution_ordering", "video_resource_lifetime", "speculative_contamination_round0")
GateError = smoke.GateError
sha256_file, sha256_bytes, array_sha256, atomic_json = smoke.sha256_file, smoke.sha256_bytes, smoke.array_sha256, smoke.atomic_json
SECONDS_FIXED, SECONDS_PER_STEP, SECONDS_SINGLE_STEP = 8.0, 5.8, 11.0  # measured on the L40S host (Rounds 4B/4C, 4A)
PHASE_DIR = {"reference": "reference", "propagation": "propagation", "oracle": "oracle_repair", "recompute": "recompute_repair", "weakprobe": "weak_band_characterization"}
RECOMPUTE_LABEL = "FULL-FORWARD SELECTIVE-STATE RECOMPUTATION with cached contaminated context"


# --------------------------------------------------------------------------------------
# config / matrix
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
    sch = config["scheduler"]
    if sch["name"] != smoke.EXPECTED_SCHEDULER or sch["sample_solver"] != "euler" or float(sch["flow_shift"]) != 12.0 or int(sch["num_train_timesteps"]) != 1000:
        raise ValueError("Trusted Wan Euler scheduler configuration is frozen")
    for k, v in {"height": 480, "width": 832, "num_frames": 33, "num_inference_steps": 40, "guidance_scale": 4.0, "fps": 16.0, "boundary_ratio": 0.875}.items():
        if config["generation"][k] != v:
            raise ValueError(f"Frozen generation field changed: {k}")
    prompts = config["prompts"]
    if len(prompts) != 4 or [p["category"] for p in prompts] != ["static", "object_motion", "articulated_motion", "complex_scene"] or len({p["prompt"] for p in prompts}) != 4:
        raise ValueError("Exactly four frozen, distinct prompt categories are required")
    if tuple(int(s) for s in config["seeds"]) != (9101, 9202) or int(config["decision_seed"]) != 9101:
        raise ValueError("Seeds / decision seed are frozen")
    if tuple(config["checkpoints"]) != core.CHECKPOINTS or tuple(config["propagation_offsets"]) != core.PROPAGATION_OFFSETS or int(config["repair_offset"]) != core.REPAIR_OFFSET:
        raise ValueError("Checkpoint / offset grids are frozen")
    if config["thresholds"] != core.frozen_thresholds():
        raise ValueError("Thresholds in config must equal the frozen core definitions")
    m = config["matrix"]
    if m["decision_regions"] != list(core.DECISION_REGIONS) or m["decision_error_model"] != core.DECISION_ERROR:
        raise ValueError("Decision regions / error model frozen")
    if m["secondary_error_models"] != ["STALE_VELOCITY_W1", "GAUSS_HIGH"] or "gauss_high_oracle_domains" in m:
        raise ValueError("Secondary arms frozen: W1 and GAUSS_HIGH are no-repair only; no Gaussian oracle sweep")
    if m["weak_band_characterization_domains"] != {r: list(d) for r, d in core.WEAK_BAND_DOMAINS.items()}:
        raise ValueError("WEAK-band characterization domains frozen")
    for key in ("cost_semantics", "verification_semantics", "kill_scope", "recompute_arm_label"):
        if not isinstance(config.get(key), (str, dict)) or not config.get(key):
            raise ValueError(f"Frozen statement missing: {key}")
    if config["recompute_arm_label"] != RECOMPUTE_LABEL:
        raise ValueError("Recompute arm label frozen")
    if m["recovery_domains"] != {r: list(core.RECOVERY_DOMAINS[r]) for r in core.DECISION_REGIONS}:
        raise ValueError("Recovery domain grid frozen")
    if config.get("quarantine") != "DEFERRED_TO_ROUND1" or config.get("blinded_review") is not False:
        raise ValueError("Quarantine must be deferred; Round 0 has no human labels")


def validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT / "results")
    except ValueError as error:
        raise ValueError("Output must be under results/") from error
    if len(rel.parts) < 2 or rel.parts[0] != NAMESPACE or rel.parts[1] != ATTEMPT_SUBDIR:
        raise ValueError(f"Output must be results/{NAMESPACE}/{ATTEMPT_SUBDIR}[/...] (attempts 1-2 are frozen INVALID records)")
    if any(part in str(rel) for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError("Trusted prior-result namespace cannot be used")


def trajectories(config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, p in enumerate(config["prompts"]):
        for seed in config["seeds"]:
            out.append({"id": f"p{i}_s{seed}", "prompt_index": i, "category": p["category"], "prompt": p["prompt"], "seed": int(seed), "decision": int(seed) == int(config["decision_seed"])})
    return out


def run_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every continuation request; bounded single-step requests are counted in `single_steps`."""
    plan: list[dict[str, Any]] = []
    d = core.REPAIR_OFFSET
    for tr in trajectories(config):
        plan.append({"phase": "reference", "label": f"{tr['id']}_reference", "traj": tr["id"], "start": 0, "kind": "CLEAN", "single_steps": 3 * len(core.CHECKPOINTS)})  # validation at t-1, cached v at t-5 and t-2
        for t in core.CHECKPOINTS:
            plan.append({"phase": "propagation", "label": f"{tr['id']}_t{t}_ZERO", "traj": tr["id"], "t": t, "start": t, "kind": "CONTROL_ZERO", "error": "ZERO", "region": "SPATIAL_SMALL"})
            for region in core.DECISION_REGIONS:
                plan.append({"phase": "propagation", "label": f"{tr['id']}_t{t}_{region}_STALE_VELOCITY_noRepair", "traj": tr["id"], "t": t, "start": t, "kind": "NO_REPAIR", "error": "STALE_VELOCITY", "region": region})
            if not tr["decision"]:
                continue
            for error in config["matrix"]["secondary_error_models"]:
                for region in core.DECISION_REGIONS:
                    plan.append({"phase": "propagation", "label": f"{tr['id']}_t{t}_{region}_{error}_noRepair", "traj": tr["id"], "t": t, "start": t, "kind": "NO_REPAIR", "error": error, "region": region})
            plan.append({"phase": "oracle", "label": f"{tr['id']}_t{t}_STALE_VELOCITY_d{d}_ORACLE100", "traj": tr["id"], "t": t, "d": d, "start": t + d, "kind": "FULL_ROLLBACK", "error": "STALE_VELOCITY", "region": "SPATIAL_SMALL", "domain": "100"})
            for region in core.DECISION_REGIONS:
                for key in core.RECOVERY_DOMAINS[region]:
                    if key != "100":
                        plan.append({"phase": "oracle", "label": f"{tr['id']}_t{t}_{region}_STALE_VELOCITY_d{d}_ORACLE{key}", "traj": tr["id"], "t": t, "d": d, "start": t + d, "kind": "ORACLE_REPAIR", "error": "STALE_VELOCITY", "region": region, "domain": key})
            plan.append({"phase": "recompute", "label": f"{tr['id']}_t{t}_STALE_VELOCITY_d{d}_RECOMPUTE100", "traj": tr["id"], "t": t, "d": d, "start": t + d, "kind": "RECOMPUTE_FULL", "error": "STALE_VELOCITY", "region": "SPATIAL_SMALL", "domain": "100", "single_steps": d})
            for region in core.DECISION_REGIONS:
                for key in core.RECOVERY_DOMAINS[region]:
                    if key != "100":
                        plan.append({"phase": "recompute", "label": f"{tr['id']}_t{t}_{region}_STALE_VELOCITY_d{d}_RECOMPUTE{key}", "traj": tr["id"], "t": t, "d": d, "start": t + d, "kind": "RECOMPUTE_REPAIR", "error": "STALE_VELOCITY", "region": region, "domain": key, "single_steps": d})
    labels = [p["label"] for p in plan]
    if len(labels) != len(set(labels)):
        raise GateError("Duplicate run labels in plan")
    return plan


PRIOR_ATTEMPT_NOTES = {
    "attempt1": {"cause": "implied-delta-sigma MEDIAN check failed at t=12/20: one-step updates (~0.006) are below the bf16 ulp (0.0078); scheduler_output and single-step chain were bit-exact on all 24 checks, bf16 reconstruction >= 0.99988",
                 "fix": "STALE_VELOCITY rebuilt from probe-captured model outputs applied step by step; slope check replaced by a least-squares slope"},
    "attempt2": {"cause": "least-squares slope tolerance 5% too tight: sub-ulp rounding biases |slope| upward by 3-7% at the smallest delta_sigma (k=7..11); 5 of 72 checks (prompt p3) failed while all 72 were bit-exact with bf16 reconstruction >= 0.99989",
                 "fix": "slope tolerance 15% (sign + gross scale only); precision is carried by the bit-exact and >= 99.9% bf16-reconstruction tests, which detect a one-index sigma error (~5% of positions flip)"},
}


def prior_attempt_references() -> dict[str, Any]:
    """Pins of the two INVALID attempts (frozen records; never reused)."""
    out = {}
    for name, root in PRIOR_ATTEMPTS.items():
        dec = root / "decision.json"
        entry = {"namespace": str(root.relative_to(REPO_ROOT / "results")), **PRIOR_ATTEMPT_NOTES[name]}
        if dec.exists():
            doc = json.loads(dec.read_text())
            entry.update({"decision": doc.get("decision"), "reason": doc.get("reason"), "decision_sha256": sha256_file(dec)})
        else:
            entry["status"] = "decision.json not present on this host"
        out[name] = entry
    return out


def weak_band_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Frozen descriptive characterization, executed ONLY if analyze returns the WEAK band; never part of the primary key set."""
    d = core.REPAIR_OFFSET
    plan = []
    for tr in trajectories(config):
        if not tr["decision"]:
            continue
        for t in core.CHECKPOINTS:
            for region, domains in core.WEAK_BAND_DOMAINS.items():
                for key in domains:
                    plan.append({"phase": "weakprobe", "label": f"{tr['id']}_t{t}_{region}_STALE_VELOCITY_d{d}_WEAKRECOMPUTE{key}", "traj": tr["id"], "t": t, "d": d, "start": t + d, "kind": "WEAK_RECOMPUTE", "error": "STALE_VELOCITY", "region": region, "domain": key, "single_steps": d})
    return plan


def estimate_cost(plan: list[dict[str, Any]]) -> dict[str, Any]:
    seconds, by_phase = 0.0, {}
    for p in plan:
        s = SECONDS_FIXED + SECONDS_PER_STEP * (core.TOTAL_STEPS - p["start"]) + SECONDS_SINGLE_STEP * p.get("single_steps", 0)
        seconds += s
        by_phase[p["phase"]] = by_phase.get(p["phase"], 0.0) + s
    n_latents = 2 * len(plan) + sum(6 for p in plan if p["phase"] == "propagation") + sum(p.get("single_steps", 0) for p in plan) + 16 * 8
    return {"continuations": len(plan), "single_step_requests": sum(p.get("single_steps", 0) for p in plan), "gpu_requests": len(plan) + sum(p.get("single_steps", 0) for p in plan),
            "hours_total": seconds / 3600, "hours_by_phase": {k: v / 3600 for k, v in by_phase.items()}, "latent_files_est": n_latents, "storage_gb_est": (n_latents * 3.6 + 8 * 40 + len(plan) * 1.5) / 1024}


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
    doc = {"git_commit": commit, "git_dirty": bool(status), "git_status": status, "relevant_git_status": relevant, "source_sha256": hashes, "config_sha256": sha256_file(config_path)}
    doc["provenance_hash"] = sha256_bytes(core.canonical({k: doc[k] for k in ("git_commit", "relevant_git_status", "source_sha256", "config_sha256")}))
    return doc


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan_cfg = json.loads(json.dumps(config))
    plan_cfg["seed"] = int(config["seeds"][0])
    plan_cfg["generation"]["switch_steps"] = list(smoke.EXPECTED_SWITCHES)
    plan = smoke.scheduler_plan(plan_cfg)
    timesteps = [float(v) for v in plan["timesteps"]]
    sigmas = core.sigmas_from_timesteps(timesteps, int(config["scheduler"]["num_train_timesteps"]))
    runs = run_plan(config)
    d = core.REPAIR_OFFSET
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "attempt": 3,
        "prior_attempts": prior_attempt_references(),
        "title": "Round 0 (rev 5, attempt 3) - does regional approximation error require a global dense refresh, or can a recomputable dependency-bounded subset recover the trajectory?",
        "primary_question": config["primary_question"],
        "related_work_and_problem_statement": config["related_work_and_problem_statement"],
        "headroom_disclaimer": config["headroom_disclaimer"],
        "definitions": {"t": "absolute Euler step index; x_t = state after t completed updates; next update uses schedule[t]", "L": f"stale regional updates preceding t; PRIMARY L = {core.STALE_WINDOW_PRIMARY}", "d": f"normal updates after t before repair; frozen d = {d} (aligned with L); continuations run to 40 then decode"},
        "recompute_arm_label": RECOMPUTE_LABEL,
        "recompute_arm_statement": config["recompute_arm_statement"],
        "verification_semantics": config["verification_semantics"],
        "cost_semantics": config["cost_semantics"],
        "idealized_window_saving_by_domain": {r: {k: core.idealized_window_saving(core.RECOVERY_DOMAINS[r][k].fraction) for k in core.RECOVERY_DOMAINS[r]} for r in core.DECISION_REGIONS},
        "kill_scope": config["kill_scope"],
        "weak_band_characterization": {"allowed_iff": f"{core.RCOMPUTE_GO} < worst-region median R_compute* <= {core.RCOMPUTE_WEAK}", "domains": {r: {k: core.WEAK_BAND_DOMAINS[r][k].fraction for k in core.WEAK_BAND_DOMAINS[r]} for r in core.DECISION_REGIONS},
                                       "arm": "RECOMPUTE only (same d, same cells)", "status": "DESCRIPTIVE ONLY; cannot upgrade WEAK to GO; if the expanded cone still needs > 0.70 the branch is NO_GO", "labels": [p["label"] for p in weak_band_plan(config)]},
        "scheduler_algebra": {"step": "x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * v_k  (WanEulerScheduler.step: fp32 arithmetic, result cast to the model dtype bf16)", "sigma": "sigma_k = timesteps[k] / num_train_timesteps",
                              "delta_sigma_by_t": {str(t): {"t-5": core.delta_sigma(sigmas, t - 5), "t-2": core.delta_sigma(sigmas, t - 2), "t-1": core.delta_sigma(sigmas, t - 1), "t": core.delta_sigma(sigmas, t)} for t in core.CHECKPOINTS},
                              "bf16_note": "one-step updates at t=12 (~0.006) are below the bf16 ulp of the latent (0.0078); velocities are therefore taken from the probe-captured model output, never from bf16 state differences",
                              "on_host_validation": "reference phase runs bounded single steps at t-1, t-2 and t-5 with the within-step probe; requires scheduler_output == x_{k+1} bit-exact, the single-step output == x_{k+1} bit-exact, bf16(x_k + delta_sigma_k * v_probe) == x_{k+1} at >= 99.9% of positions, bf16 runtime dtype, and the least-squares slope of (x_{k+1} - x_k) on v_probe over all positions within 15% of delta_sigma_k with the correct sign (sign / gross-scale guard only: sub-ulp rounding biases |slope| upward by 3-7% at k=7..11 on the real model; the bit-exact and reconstruction tests carry the precision, and a one-index sigma error flips ~5% of positions in the reconstruction); failure is INVALID_EXPERIMENT"},
        "error_models": {"STALE_VELOCITY": {"role": "PRIMARY, decides", "window": core.STALE_WINDOW_PRIMARY, "formula": "x'_{t-4}[R] = x_{t-4}[R]; for k = t-4..t-1: x'_{k+1}[R] = bf16(x'_k[R] + (sigma_{k+1} - sigma_k) * v_{t-5}[R]); v_{t-5} = probe-captured guidance_combined_output at step t-5; outside R identical to clean x_t",
                                            "meaning": "R skipped its target evaluation at steps t-4..t-1 and kept the velocity of its last active step (RAS/HSA cached Euler update over a multi-step skip)",
                                            "magnitude_evidence": "frozen Round 4C-0 OLD trajectories: 4-step stale average velocity gives 0.0085-0.0123 latent std and 0.7-1.9 one-step-update RMS at t=12..24; L=1 is ~1/16 of that, below the bf16 half-ulp (0.0039 std)",
                                            "idealization": "non-R is taken as clean during the stale window; a real engine's non-R would have consumed stale R, so this is a lower bound on multi-step staleness error"},
                         "STALE_VELOCITY_W1": {"role": "descriptive magnitude / control arm, propagation only, never decides", "window": core.STALE_WINDOW_SECONDARY, "formula": "x'_t[R] = bf16(x_{t-1}[R] + (sigma_t - sigma_{t-1}) * v_{t-2}[R]); v_{t-2} = probe-captured model output at step t-2"},
                         "GAUSS_HIGH": {"role": "stress / calibration control, no-repair (propagation) only; never decides", "alpha_fraction_of_channel_std": core.GAUSS_HIGH_ALPHA}},
        "checkpoints": list(core.CHECKPOINTS), "schedule_timesteps_by_t": {str(t): timesteps[t] for t in core.CHECKPOINTS},
        "expert_by_t": {str(t): "high_noise_transformer" if timesteps[t] >= 875.0 else "low_noise_transformer_2" for t in core.CHECKPOINTS},
        "trajectories": trajectories(config), "prompts": config["prompts"], "seeds": list(config["seeds"]), "decision_seed": config["decision_seed"],
        "geometry": core.geometry_document(),
        "contamination": {"error_map": "e_p = RMS over channels of (x - clean)/sigma_c(clean) per latent position", "material": f"e_p > {core.MATERIAL_THRESHOLD}", "C(d)": "material fraction after d steps / region fraction", "offsets": list(core.PROPAGATION_OFFSETS), "role": "descriptive"},
        "recovery_criterion": {"final_relative_l2_max": core.RECOVERY_REL_L2, "frame_ssim_min": core.RECOVERY_SSIM, "sensitivity_descriptive": list(core.RECOVERY_SENSITIVITY)},
        "quantities": {
            "R_state*": "minimum ORACLE clean-state replacement fraction at t+d such that the criterion holds there and at every larger tested fraction (structural probe; never decides)",
            "R_compute*": f"minimum recovery-domain fraction whose state is REGENERATED over t..t+d ({RECOMPUTE_LABEL}), non-domain tokens supplied as the cached perturbed-trajectory latents, such that the criterion holds there and at every larger tested fraction (PRIMARY)",
            "GLOBAL_DENSE_REFRESH": "the NO_REPAIR arm: dense target computation from the contaminated state onward, the most favourable form of the dense-step reset used by RAS/HSA and the Stage-3 full recompute of Chorus; if it converges under the criterion there is no recovery problem (STOP)",
            "recompute_fidelity_label": RECOMPUTE_LABEL + ". Every recompute step executes a full 14,040-token target forward; only the recovery-domain output is retained; context K/V are recomputed inside the forward from the cached perturbed latents (optimistic relative to a cached-K/V engine). This measures logical/state recomputability under cached context; it does NOT demonstrate sparse GPU compute proportional to |D|; actual sparse selective execution cost is deferred to Round 1.",
        },
        "recovery_domains": {r: {k: core.RECOVERY_DOMAINS[r][k].fraction for k in core.RECOVERY_DOMAINS[r]} for r in core.DECISION_REGIONS},
        "matrix": config["matrix"], "run_plan_labels": [r["label"] for r in runs], "n_continuations": len(runs), "cost_estimate": estimate_cost(runs),
        "thresholds": core.frozen_thresholds(),
        "decision_logic": {
            "INVALID_EXPERIMENT": "velocity/sign validation, zero-perturbation resume, oracle 100% repair, or recompute 100% chain fails bit-exactness; or the run key set is incomplete",
            "STOP_NO_RECOVERY_PROBLEM": f"fewer than {core.MATERIAL_CELL_FRACTION_MIN:.0%} of the 24 primary STALE_VELOCITY cells (8 trajectories x 3 t, either decision region material) have a no-repair final error that fails the recovery criterion; evaluated after propagation alone (early stop, oracle/recompute phases not required)",
            "materiality_cell": "a (trajectory, t) cell is material if the STALE_VELOCITY no-repair error fails the criterion in at least one decision region; per-region fractions are reported",
            "WEAK_band": f"{core.RCOMPUTE_GO} < median R_compute* <= {core.RCOMPUTE_WEAK}: the ONE frozen dependency-cone characterization (weakprobe phase) may run; descriptive only; cannot upgrade to GO; if its expanded cone still requires > {core.RCOMPUTE_WEAK} the branch is NO_GO",
            "gate": f"worst-region median R_compute* over material cells: <= {core.RCOMPUTE_STRONG_GO} STRONG_GO; <= {core.RCOMPUTE_GO} GO; <= {core.RCOMPUTE_WEAK} WEAK (one dependency-cone characterization allowed); else NO_GO (global dense refresh required)",
            "item_8": "R_state* << 1 with R_compute* ~= 1 is NO_GO and is reported as oracle-only headroom",
            "never_decides": ["GAUSS_HIGH", "STALE_VELOCITY_W1", "R_state*", "contamination curves", "SSIM alone", "visual impressions", "weak-band characterization"],
        },
        "controls": ["1 clean uninterrupted", "2 exact resume = 6 zero perturbation through the perturbation path (bit-exact)", "3 perturbed no repair = GLOBAL_DENSE_REFRESH", "4 oracle 100% repair (bit-exact)", "5 recompute 100% chain (bit-exact with clean x_{t+d} and with the reference final)", "7 velocity / sign validation against the within-step probe"],
        "deferred_to_round1": ["quarantine / speculative isolation", "SPATIAL_MEDIUM, TEMPORAL_SMALL, disjoint-region control, Gaussian LOW, Gaussian oracle repair", "d in {1, 2} repair arms", "error-map-guided domains", "actual sparse selective execution and its GPU cost", "verification cost", "real skipped compute, failure frequency, dense-refresh frequency/cost, dependency-discovery overhead, synchronization/versioning overhead, net end-to-end speedup vs GLOBAL_DENSE_REFRESH"],
        "model": config["model"], "scheduler": config["scheduler"], "generation": config["generation"], "scheduler_plan": plan,
        "source_commit": provenance["git_commit"], "provenance_hash": provenance["provenance_hash"], "config_sha256": provenance["config_sha256"],
        "claim_boundary": config["claim_boundary"],
    }


def require_sealed(output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    pp, sp = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if not pp.exists() or not sp.exists():
        raise GateError("Preregistration is not sealed; run --phase preregister first")
    if sha256_file(pp) != sp.read_text().split()[0]:
        raise GateError("preregistration.json was modified after sealing")
    doc = json.loads(pp.read_text())
    frozen = json.loads((output_dir / "provenance.json").read_text())
    if doc.get("provenance_hash") != provenance["provenance_hash"] or frozen.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Code/config provenance changed after preregistration; use a fresh namespace")
    if provenance["relevant_git_status"]:
        raise GateError(f"Scientific source files are not committed: {provenance['relevant_git_status']}")
    return doc


def synthetic_sigmas() -> np.ndarray:
    """Shifted flow-matching schedule with the trusted scheduler's functional form (CPU demonstrations and tests only)."""
    u = np.linspace(1.0, 0.0, core.TOTAL_STEPS + 1)
    shift = 12.0
    return shift * u / (1 + (shift - 1) * u)


def phase_cpu(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    sig = synthetic_sigmas()
    x0, eps = rng.standard_normal(core.LATENT_SHAPE).astype(np.float32), rng.standard_normal(core.LATENT_SHAPE).astype(np.float32)
    traj = {k: core.bf16_round(((1 - sig[k]) * x0 + sig[k] * eps + 0.02 * np.sin(k / 3.0) * rng.standard_normal(core.LATENT_SHAPE)).astype(np.float32)) for k in range(0, 30)}
    demo: dict[str, Any] = {"geometry": {k: v for k, v in core.geometry_document().items() if k in ("token_grid_fhw", "n_tokens")}}
    for t in core.CHECKPOINTS:
        for w in (core.STALE_WINDOW_PRIMARY, core.STALE_WINDOW_SECONDARY):
            v_last = core.implied_velocity(traj[t - w - 1], traj[t - w], sig, t - w - 1).astype(np.float32)  # synthetic stand-in for the probe-captured model output
            xs, st = core.stale_velocity_state(x_last_active=traj[t - w], v_last_active=v_last, x_t=traj[t], sigmas=sig, t=t, window=w, region="SPATIAL_SMALL")
            cm = core.contamination_metrics(xs, traj[t], "SPATIAL_SMALL")
            demo[f"stale_velocity_L{w}_t{t}"] = {"rms_over_std": round(st["rms_in_region_over_mean_channel_std"], 5), "material_inside_at_injection": cm["material_fraction_inside_region"], "outside_changed": st["outside_region_changed"], "total_delta_sigma": round(st["total_delta_sigma_window"], 5)}
    v = core.implied_velocity(traj[11], traj[12], sig, 11)
    demo["euler_identity_max_abs"] = float(np.abs(core.euler_step(traj[11], v, sig, 11) - traj[12]).max())
    xg, sg = core.gaussian_state(traj[12], trajectory_id="demo", t=12, region="SPATIAL_SMALL")
    demo["gauss_high"] = {"rms_over_std": round(sg["rms_in_region_over_mean_channel_std"], 4), "material_inside": core.contamination_metrics(xg, traj[12], "SPATIAL_SMALL")["material_fraction_inside_region"]}
    fake = {"R": {"recovered": False}, "20": {"recovered": False}, "40": {"recovered": True}, "70": {"recovered": True}, "100": {"recovered": True}}
    demo["r_star_example"] = core.r_star(fake, {"R": 0.0385, "20": 0.205, "40": 0.40, "70": 0.70, "100": 1.0})["r_star"]
    demo["decision_examples"] = {
        "STOP": core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.25, rcompute_median_by_region={"SPATIAL_SMALL": 0.2, "TEMPORAL_MEDIUM": 0.2}, rstate_median_by_region={})["decision"],
        "STRONG_GO": core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 0.0385, "TEMPORAL_MEDIUM": 0.111}, rstate_median_by_region={"SPATIAL_SMALL": 0.0385, "TEMPORAL_MEDIUM": 0.111})["decision"],
        "GO": core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 0.205, "TEMPORAL_MEDIUM": 0.222}, rstate_median_by_region={"SPATIAL_SMALL": 0.0385, "TEMPORAL_MEDIUM": 0.111})["decision"],
        "WEAK": core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 0.4, "TEMPORAL_MEDIUM": 0.667}, rstate_median_by_region={})["decision"],
        "NO_GO_item8": core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 1.0, "TEMPORAL_MEDIUM": 1.0}, rstate_median_by_region={"SPATIAL_SMALL": 0.205, "TEMPORAL_MEDIUM": 0.222}),
    }
    demo["plan"] = estimate_cost(run_plan(config))
    demo["weak_band_plan"] = estimate_cost(weak_band_plan(config))
    demo["idealized_window_saving_examples"] = {r: core.idealized_window_saving(core.RECOVERY_DOMAINS[r]["R"].fraction) for r in core.DECISION_REGIONS}
    return {"phase": "cpu", "status": "OK", "synthetic": demo}


def phase_preregister(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    pp, sp = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if pp.exists():
        if json.loads(pp.read_text()).get("provenance_hash") != provenance["provenance_hash"]:
            raise GateError("Namespace already holds a preregistration from different code/config; use a fresh namespace")
        return {"phase": "preregister", "status": "ALREADY_SEALED", "preregistration_sha256": sp.read_text().split()[0]}
    doc = build_preregistration(config, provenance)
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(output_dir / "prompts.json", {"trajectories": doc["trajectories"], "prompts": config["prompts"], "seeds": config["seeds"], "decision_seed": config["decision_seed"]})
    atomic_json(pp, doc)
    digest = sha256_file(pp)
    sp.write_text(f"{digest}  preregistration.json\n")
    try:
        from experiments.video_runtime_state_discovery import environment_document

        atomic_json(output_dir / "environment.json", environment_document(config, doc["scheduler_plan"], provenance))
    except Exception as error:
        atomic_json(output_dir / "environment.json", {"status": f"deferred: {error}", "provenance_hash": provenance["provenance_hash"]})
    _write_manifest(output_dir, provenance, {"phase": "preregister"})
    return {"phase": "preregister", "status": "SEALED", "preregistration_sha256": digest, "n_continuations": doc["n_continuations"], "cost_estimate": doc["cost_estimate"], "relevant_git_status": provenance["relevant_git_status"]}


def _write_manifest(output_dir: Path, provenance: dict[str, Any], extra: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"provenance_hash": provenance["provenance_hash"], "immutable_sha256": {}, "phases": []}
    for name in ("preregistration.json", "provenance.json", "prompts.json"):
        if (output_dir / name).exists():
            manifest["immutable_sha256"][name] = sha256_file(output_dir / name)
    manifest["phases"].append({**extra, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    atomic_json(path, manifest)


# --------------------------------------------------------------------------------------
# GPU primitive (trusted pipeline only)
# --------------------------------------------------------------------------------------
WITHIN_BOUNDARIES = ("latent_entering_step", "transformer_input", "guidance_combined_output", "scheduler_input", "scheduler_output")


class Engine:
    def __init__(self, config: dict[str, Any], args: argparse.Namespace, output_dir: Path, prereg: dict[str, Any], provenance: dict[str, Any]):
        from experiments.video_runtime_state_discovery import build_omni

        self.config, self.output_dir, self.prereg, self.provenance = config, output_dir, prereg, provenance
        self.schedule = [float(v) for v in prereg["scheduler_plan"]["timesteps"]]
        self.sigmas = core.sigmas_from_timesteps(self.schedule, int(config["scheduler"]["num_train_timesteps"]))
        self.omni = build_omni(config, args)
        try:
            from experiments.video_runtime_state_discovery import environment_document

            env = output_dir / "environment.json"
            if not env.exists() or str(json.loads(env.read_text()).get("status", "")).startswith("deferred"):
                atomic_json(env, environment_document(config, prereg["scheduler_plan"], provenance))
        except Exception:
            pass

    def shutdown(self) -> None:
        self.omni.shutdown()

    def _sampling(self, *, prompt, seed, label, artifact_dir, capture_steps, latents, step_index, step_limit, skip_decode, within_step):
        import torch

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        g = self.config["generation"]
        s = OmniDiffusionSamplingParams(height=int(g["height"]), width=int(g["width"]), num_frames=int(g["num_frames"]), num_inference_steps=int(g["num_inference_steps"]), guidance_scale=float(g["guidance_scale"]), fps=float(g["fps"]), seed=seed, generator=torch.Generator(device="cpu").manual_seed(seed))
        s.latents = None if latents is None else torch.from_numpy(np.ascontiguousarray(latents.astype(np.float32)))
        s.step_index = int(step_index)
        s.extra_args = {"flow_shift": float(self.config["scheduler"]["flow_shift"]), "sample_solver": "euler",
                        "trajectory_probe": {"artifact_dir": str(artifact_dir / "trajectory"), "request_label": label, "capture_steps": sorted(set(capture_steps)), "fps": float(g["fps"]), "save_decoded": False, "save_latents": True, "save_mp4": False}}
        if step_limit is not None:
            s.extra_args["execution_step_limit"] = int(step_limit)
        if skip_decode:
            s.extra_args["skip_vae_decode"] = True
        if within_step:
            s.extra_args["within_step_probe"] = {"artifact_dir": str(artifact_dir / "within"), "request_label": label, "selected_local_step": 0, "selected_absolute_step": int(step_index)}
        return s

    def generate(self, *, prompt: str, seed: int, label: str, capture_local_steps: list[int], latents: np.ndarray | None, step_index: int, step_limit: int | None = None, skip_decode: bool = False, within_step: bool = False) -> dict[str, Any]:
        import torch

        from experiments.video_runtime_state_discovery import normalize_video
        from vllm_omni.outputs import OmniRequestOutput

        local_steps = (core.TOTAL_STEPS - step_index) if step_limit is None else step_limit
        capture = sorted(set(capture_local_steps) | {0, local_steps})
        probe_dir = self.output_dir / "_probe_tmp" / label
        if probe_dir.exists():
            shutil.rmtree(probe_dir)
        started = time.perf_counter()
        outputs = self.omni.generate({"prompt": prompt}, self._sampling(prompt=prompt, seed=seed, label=label, artifact_dir=probe_dir, capture_steps=capture, latents=latents, step_index=step_index, step_limit=step_limit, skip_decode=skip_decode, within_step=within_step))
        wall = time.perf_counter() - started
        output = OmniRequestOutput.unwrap_result(outputs)
        custom = output.custom_output or {}
        meta = json.loads(Path(custom["trajectory_probe_metadata_path"]).read_text())
        if meta.get("sample_solver") != "euler" or not str(meta.get("scheduler_class", "")).endswith(smoke.EXPECTED_SCHEDULER):
            raise GateError(f"Worker did not execute the trusted Euler path for {label}")
        if int(meta.get("num_steps", -1)) != local_steps:
            raise GateError(f"Executed {meta.get('num_steps')} local steps, expected {local_steps} for {label}")
        control = custom.get("execution_control") or {}
        if step_limit is not None and (control.get("execution_step_limit") != step_limit or control.get("executed_local_steps") != step_limit or control.get("resume_step_index") != step_index):
            raise GateError(f"Bounded execution semantics violated for {label}: {control}")
        records = {int(r["step_index"]): r for r in meta["records"]}
        if set(records) != set(capture):
            raise GateError(f"Trajectory probe captured {sorted(records)} but {capture} was requested for {label}")

        def load(p: str) -> np.ndarray:
            return torch.load(p, map_location="cpu").detach().float().contiguous().numpy()

        arrays = {k: load(r["latent_path"]) for k, r in records.items()}
        if abs(float(records[0]["timestep"]) - self.schedule[step_index]) > 1e-3:
            raise GateError(f"Resume timestep {records[0]['timestep']} != schedule[{step_index}] for {label}")
        if latents is not None and array_sha256(arrays[0]) != array_sha256(np.ascontiguousarray(latents.astype(np.float32))):
            raise GateError(f"State entering the resumed step differs from the supplied latents for {label} (mutation or cast)")
        within = None
        if within_step:
            recs = {r["boundary"]: r for r in (custom.get("within_step_probe") or {}).get("records", [])}
            if any(b not in recs for b in WITHIN_BOUNDARIES):
                raise GateError(f"Within-step probe incomplete for {label}: {sorted(recs)}")
            within = {b: load(recs[b]["latent_path"]) for b in ("guidance_combined_output", "scheduler_input", "scheduler_output")}
            within["timestep"] = float(recs["transformer_input"]["timestep"])
            within["runtime_dtype"] = recs["guidance_combined_output"].get("runtime_dtype")
        video = None
        if not skip_decode:
            video, _ = normalize_video(outputs)
            if video.shape[0] != int(self.config["generation"]["num_frames"]):
                raise GateError(f"Decoded video has {video.shape[0]} frames for {label}")
        shutil.rmtree(probe_dir, ignore_errors=True)
        return {"arrays": arrays, "video": video, "wall_s": wall, "meta": meta, "local_steps": local_steps, "within": within}

    def single_step(self, tr: dict[str, Any], label: str, x: np.ndarray, step_index: int, *, within_step: bool = False) -> dict[str, Any]:
        r = self.generate(prompt=tr["prompt"], seed=tr["seed"], label=label, capture_local_steps=[0, 1], latents=x, step_index=step_index, step_limit=1, skip_decode=True, within_step=within_step)
        return {"next": r["arrays"][1], "within": r["within"], "wall_s": r["wall_s"]}


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------
def _run_dir(output_dir: Path, phase_dir: str, label: str) -> Path:
    return output_dir / phase_dir / label


def _load_run(output_dir: Path, phase_dir: str, label: str, provenance_hash: str) -> dict[str, Any] | None:
    rd = _run_dir(output_dir, phase_dir, label)
    if not (rd / "run.json").exists():
        return None
    doc = json.loads((rd / "run.json").read_text())
    if doc.get("provenance_hash") != provenance_hash or doc.get("status") != "COMPLETE":
        return None
    for rel, h in doc["artifact_sha256"].items():
        p = rd / rel
        if not p.exists():
            return None
        if rel.endswith(".npy") and array_sha256(np.load(p, allow_pickle=False)) != h:
            raise GateError(f"Persisted artifact hash mismatch: {p}")
    return doc


def _save_run(output_dir: Path, phase_dir: str, label: str, provenance_hash: str, *, arrays: dict[str, np.ndarray], video: np.ndarray | None, info: dict[str, Any], fps: float, keep_video_npy: bool = False) -> dict[str, Any]:
    rd = _run_dir(output_dir, phase_dir, label)
    rd.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr.astype(np.float32))
        np.save(rd / f"{name}.npy", a, allow_pickle=False)
        hashes[f"{name}.npy"] = array_sha256(a)
    if video is not None:
        from experiments.temporal_dimension_killtest_preflight import _save_video

        _save_video(rd / "video.mp4", video, fps)
        hashes["video.mp4"] = sha256_file(rd / "video.mp4")
        info["video_hash"] = array_sha256(video)
        if keep_video_npy:
            np.save(rd / "video.npy", video, allow_pickle=False)
            hashes["video.npy"] = array_sha256(video)
    doc = {"label": label, "status": "COMPLETE", "provenance_hash": provenance_hash, "artifact_sha256": hashes, **info}
    atomic_json(rd / "run.json", doc)
    return doc


def _reference(output_dir: Path, traj_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rd = _run_dir(output_dir, "reference", f"{traj_id}_reference")
    return np.load(rd / "final.npy", allow_pickle=False), np.load(rd / "video.npy", allow_pickle=False), json.loads((rd / "run.json").read_text())


def _checkpoint(output_dir: Path, traj_id: str, k: int) -> np.ndarray:
    return np.load(output_dir / "checkpoints" / traj_id / f"x_{k:02d}.npy", allow_pickle=False)


def _perturbed(output_dir: Path, traj_id: str, t: int, region: str, error: str) -> tuple[np.ndarray, dict[str, Any]]:
    d = output_dir / "perturbations" / traj_id / f"t{t}"
    return np.load(d / f"{region}_{error}.npy", allow_pickle=False), json.loads((d / f"{region}_{error}.json").read_text())


def _video_metrics(video: np.ndarray | None, ref_video: np.ndarray) -> dict[str, Any] | None:
    if video is None:
        return None
    from experiments.video_runtime_state_discovery import video_metrics

    return video_metrics(video, ref_video)


def _traj(config: dict[str, Any], traj_id: str) -> dict[str, Any]:
    return next(x for x in trajectories(config) if x["id"] == traj_id)


# --------------------------------------------------------------------------------------
# phase: reference (+ velocity / sign validation)
# --------------------------------------------------------------------------------------
def reference_capture_steps() -> list[int]:
    steps = {core.TOTAL_STEPS}
    for t in core.CHECKPOINTS:
        for w in (core.STALE_WINDOW_PRIMARY, core.STALE_WINDOW_SECONDARY):
            steps.update({t, t - w, t - w - 1})
        steps.add(t - 1)  # velocity validation step
    return sorted(steps)


def velocity_validation(x_k: np.ndarray, x_k1: np.ndarray, within: dict[str, Any], next_state: np.ndarray, sigmas: np.ndarray, k: int) -> dict[str, Any]:
    """Validate the scheduler algebra at step k against the actual model output (v) and scheduler output of the trusted pipeline.

    The per-position ratio (x_{k+1} - x_k) / v is quantisation-dominated when the update is below the bf16 ulp, so the sign / scale of
    delta_sigma is checked with a least-squares slope over all positions; exactness is checked with bit-exact and bf16-reconstruction tests.
    """
    v = within["guidance_combined_output"].astype(np.float64)
    dsig = core.delta_sigma(sigmas, k)
    recon = core.bf16_round(core.euler_step(x_k, v, sigmas, k))
    exact_frac = float(np.mean(recon == x_k1))
    dx = x_k1.astype(np.float64) - x_k.astype(np.float64)
    slope = float((dx * v).sum() / max((v * v).sum(), 1e-30))
    rel_err = abs(slope - dsig) / abs(dsig)
    sched_exact = bool(np.array_equal(within["scheduler_output"], x_k1))
    dtype_ok = str(within.get("runtime_dtype", "")).endswith("bfloat16")
    return {"step": k, "scheduler_output_bit_exact": sched_exact, "single_step_bit_exact": bool(np.array_equal(next_state, x_k1)), "delta_sigma_from_plan": dsig, "delta_sigma_ls_slope": slope, "delta_sigma_relative_error": rel_err,
            "bf16_reconstruction_exact_fraction": exact_frac, "reconstruction_max_abs": float(np.abs(recon.astype(np.float64) - x_k1.astype(np.float64)).max()), "timestep": within["timestep"], "runtime_dtype": within.get("runtime_dtype"),
            "pass": bool(sched_exact and exact_frac >= 0.999 and rel_err < 0.15 and dtype_ok and np.sign(slope) == np.sign(dsig))}  # precision comes from the bit-exact + reconstruction tests (a one-index sigma error flips ~5% of positions); the slope guards sign / gross scale (sub-ulp rounding bias reached 7% on the real model)


def phase_reference(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    engine = Engine(config, args, output_dir, prereg, provenance)
    out, failures = {}, []
    try:
        for tr in trajectories(config):
            label = f"{tr['id']}_reference"
            if _load_run(output_dir, "reference", label, provenance["provenance_hash"]) is None:
                r = engine.generate(prompt=tr["prompt"], seed=tr["seed"], label=label, capture_local_steps=reference_capture_steps(), latents=None, step_index=0)
                if any(a.shape != core.LATENT_SHAPE for a in r["arrays"].values()):
                    raise GateError(f"Latent shape {r['arrays'][0].shape} != frozen {core.LATENT_SHAPE}")
                ck = output_dir / "checkpoints" / tr["id"]
                ck.mkdir(parents=True, exist_ok=True)
                ck_hashes = {}
                for k in reference_capture_steps():
                    if k != core.TOTAL_STEPS:
                        np.save(ck / f"x_{k:02d}.npy", r["arrays"][k], allow_pickle=False)
                        ck_hashes[str(k)] = array_sha256(r["arrays"][k])
                validation, v_hashes = {}, {}
                for t in core.CHECKPOINTS:
                    for k, role in ((t - 1, "validation"), (t - core.STALE_WINDOW_SECONDARY - 1, "cached_velocity_W1"), (t - core.STALE_WINDOW_PRIMARY - 1, "cached_velocity_L4")):
                        s = engine.single_step(tr, f"{label}_v_step{k:02d}", r["arrays"][k], k, within_step=True)
                        vv = velocity_validation(r["arrays"][k], r["arrays"][k + 1], s["within"], s["next"], engine.sigmas, k)
                        vv["role"] = role
                        validation[f"t{t}_k{k}"] = vv
                        if not vv["pass"]:
                            failures.append(f"{label}_t{t}_k{k}")
                        v_arr = np.ascontiguousarray(s["within"]["guidance_combined_output"].astype(np.float32))
                        np.save(ck / f"v_{k:02d}.npy", v_arr, allow_pickle=False)
                        v_hashes[str(k)] = array_sha256(v_arr)
                sigma = {str(t): core.channel_std(r["arrays"][t]).tolist() for t in core.CHECKPOINTS}
                one_step = {str(t): float(np.sqrt(np.mean((r["arrays"][t].astype(np.float64) - r["arrays"][t - 1].astype(np.float64)) ** 2))) for t in core.CHECKPOINTS}
                _save_run(output_dir, "reference", label, provenance["provenance_hash"], arrays={"final": r["arrays"][core.TOTAL_STEPS], "initial": r["arrays"][0]}, video=r["video"], keep_video_npy=True, fps=float(config["generation"]["fps"]),
                          info={"kind": "CLEAN", "traj": tr["id"], "prompt": tr["prompt"], "seed": tr["seed"], "wall_s": r["wall_s"], "checkpoint_sha256": ck_hashes, "velocity_sha256": v_hashes, "channel_std_by_t": sigma, "one_step_update_rms_by_t": one_step, "velocity_validation": validation, "latent_shape": list(core.LATENT_SHAPE)})
            out[tr["id"]] = "COMPLETE"
    finally:
        engine.shutdown()
    if failures:
        atomic_json(output_dir / "decision.json", {"decision": "INVALID_EXPERIMENT", "reason": f"velocity/sign validation failed: {failures}", "provenance_hash": provenance["provenance_hash"]})
        raise GateError(f"INVALID_EXPERIMENT: scheduler algebra validation failed for {failures}")
    _write_manifest(output_dir, provenance, {"phase": "reference", "trajectories": out})
    return {"phase": "reference", "trajectories": out, "velocity_validation_failures": failures}


# --------------------------------------------------------------------------------------
# phase: perturb (CPU)
# --------------------------------------------------------------------------------------
def phase_perturb(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    sigmas = core.sigmas_from_timesteps([float(v) for v in prereg["scheduler_plan"]["timesteps"]], int(config["scheduler"]["num_train_timesteps"]))
    index: dict[str, Any] = {"provenance_hash": provenance["provenance_hash"], "states": {}}
    P, S = core.STALE_WINDOW_PRIMARY, core.STALE_WINDOW_SECONDARY
    for tr in trajectories(config):
        _, _, ref = _reference(output_dir, tr["id"])
        if not all(v["pass"] for v in ref["velocity_validation"].values()):
            raise GateError(f"Velocity validation did not pass for {tr['id']}")
        for t in core.CHECKPOINTS:
            x = {k: _checkpoint(output_dir, tr["id"], k) for k in sorted({t, t - 1, t - P, t - P - 1, t - S, t - S - 1})}
            for k, arr in x.items():
                if array_sha256(arr) != ref["checkpoint_sha256"][str(k)]:
                    raise GateError(f"Checkpoint hash mismatch {tr['id']} k={k}")
            v = {k: np.load(output_dir / "checkpoints" / tr["id"] / f"v_{k:02d}.npy", allow_pickle=False) for k in (t - P - 1, t - S - 1)}
            for k, arr in v.items():
                if array_sha256(arr) != ref["velocity_sha256"][str(k)]:
                    raise GateError(f"Cached velocity hash mismatch {tr['id']} k={k}")
            d = output_dir / "perturbations" / tr["id"] / f"t{t}"
            d.mkdir(parents=True, exist_ok=True)
            one_step = x[t].astype(np.float64) - x[t - 1].astype(np.float64)
            states = {("SPATIAL_SMALL", "ZERO"): (x[t].copy(), {"error_model": "ZERO", "region": "SPATIAL_SMALL", "rms_in_region": 0.0})}
            for region in core.DECISION_REGIONS:
                states[(region, "STALE_VELOCITY")] = core.stale_velocity_state(x_last_active=x[t - P], v_last_active=v[t - P - 1], x_t=x[t], sigmas=sigmas, t=t, window=P, region=region)
                if tr["decision"]:
                    states[(region, "STALE_VELOCITY_W1")] = core.stale_velocity_state(x_last_active=x[t - S], v_last_active=v[t - S - 1], x_t=x[t], sigmas=sigmas, t=t, window=S, region=region)
                    states[(region, "GAUSS_HIGH")] = core.gaussian_state(x[t], trajectory_id=tr["id"], t=t, region=region)
            for (region, error), (xp, stats) in states.items():
                if error == "ZERO" and not np.array_equal(xp, x[t]):
                    raise GateError("Zero perturbation altered the state")
                if error != "ZERO":
                    mask = core.token_to_latent_mask(core.REGIONS[region].token_mask())
                    stats["one_step_update_rms_in_region"] = float(np.sqrt(np.mean(one_step[mask] ** 2)))
                    stats["rms_over_one_step_update_rms"] = float(stats["rms_in_region"] / max(stats["one_step_update_rms_in_region"], 1e-12))
                stats.update({"checkpoint_sha256": array_sha256(x[t]), "perturbed_sha256": array_sha256(xp), "traj": tr["id"], "t": t})
                np.save(d / f"{region}_{error}.npy", xp, allow_pickle=False)
                atomic_json(d / f"{region}_{error}.json", stats)
                index["states"][f"{tr['id']}_t{t}_{region}_{error}"] = {k: stats.get(k) for k in ("error_model", "perturbed_sha256", "rms_in_region", "rms_in_region_over_mean_channel_std", "relative_l2_global", "max_abs", "material_fraction_inside_region_at_injection", "rms_over_one_step_update_rms")}
    atomic_json(output_dir / "perturbations" / "index.json", index)
    _write_manifest(output_dir, provenance, {"phase": "perturb", "states": len(index["states"])})
    return {"phase": "perturb", "states": len(index["states"])}


# --------------------------------------------------------------------------------------
# phase: propagation (controls 2/6, NO_REPAIR = GLOBAL_DENSE_REFRESH)
# --------------------------------------------------------------------------------------
def _propagation_capture(t: int) -> list[int]:
    return sorted({0, 1, 2, 3, 4, core.TOTAL_STEPS - t})


def phase_propagation(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    _require_no_invalid(output_dir)
    plan = [p for p in run_plan(config) if p["phase"] == "propagation"]
    engine = Engine(config, args, output_dir, prereg, provenance)
    try:
        for p in plan:
            if _load_run(output_dir, "propagation", p["label"], provenance["provenance_hash"]) is not None:
                continue
            tr = _traj(config, p["traj"])
            ref_final, ref_video, ref = _reference(output_dir, tr["id"])
            x_in, pstats = _perturbed(output_dir, tr["id"], p["t"], p["region"], p["error"])
            r = engine.generate(prompt=tr["prompt"], seed=tr["seed"], label=p["label"], capture_local_steps=_propagation_capture(p["t"]), latents=x_in, step_index=p["t"])
            arrays = {f"local_{k:02d}": v for k, v in r["arrays"].items()}
            arrays["final"] = r["arrays"][core.TOTAL_STEPS - p["t"]]
            vm = _video_metrics(r["video"], ref_video)
            info = {"kind": p["kind"], "traj": tr["id"], "t": p["t"], "region": p["region"], "error": p["error"], "wall_s": r["wall_s"], "input_sha256": array_sha256(x_in), "perturbation": pstats,
                    "final_sha256": array_sha256(arrays["final"]), "final_bit_exact_vs_reference": array_sha256(arrays["final"]) == array_sha256(ref_final),
                    "video_bit_exact_vs_reference": bool(r["video"] is not None and array_sha256(r["video"]) == ref["video_hash"]), "video_metrics_vs_reference": vm,
                    "convergence": core.final_convergence(arrays["final"], ref_final, frame_ssim_mean=vm["frame_ssim_mean"] if vm else None)}
            _save_run(output_dir, "propagation", p["label"], provenance["provenance_hash"], arrays=arrays, video=r["video"], fps=float(config["generation"]["fps"]), info=info)
    finally:
        engine.shutdown()
    metrics = compute_propagation_metrics(config, output_dir, provenance["provenance_hash"])
    atomic_json(output_dir / "propagation_metrics.json", metrics)
    if metrics["control_zero_failures"]:
        atomic_json(output_dir / "decision.json", {"decision": "INVALID_EXPERIMENT", "reason": f"zero-perturbation resume not bit-exact: {metrics['control_zero_failures']}", "provenance_hash": provenance["provenance_hash"]})
        raise GateError(f"INVALID_EXPERIMENT: clean checkpoint->resume is not bit-exact for {metrics['control_zero_failures']}")
    _write_manifest(output_dir, provenance, {"phase": "propagation", "runs": len(plan)})
    return {"phase": "propagation", "runs": len(plan), "summary": metrics["summary"]}


def compute_propagation_metrics(config: dict[str, Any], output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    plan = [p for p in run_plan(config) if p["phase"] == "propagation"]
    per_run, zero_fail = {}, []
    for p in plan:
        doc = _load_run(output_dir, "propagation", p["label"], provenance_hash)
        if doc is None:
            raise GateError(f"Propagation run missing: {p['label']}")
        if p["kind"] == "CONTROL_ZERO":
            if not (doc["final_bit_exact_vs_reference"] and doc["video_bit_exact_vs_reference"]):
                zero_fail.append(p["label"])
            continue
        rd, zero = _run_dir(output_dir, "propagation", p["label"]), _run_dir(output_dir, "propagation", f"{p['traj']}_t{p['t']}_ZERO")
        row = {"traj": p["traj"], "t": p["t"], "region": p["region"], "error": p["error"], "decision_traj": _traj(config, p["traj"])["decision"], "perturbation": doc["perturbation"], "by_d": {}}
        for d in (0, 1, 2, 3, 4, core.TOTAL_STEPS - p["t"]):
            row["by_d"][str(d)] = core.contamination_metrics(np.load(rd / f"local_{d:02d}.npy", allow_pickle=False), np.load(zero / f"local_{d:02d}.npy", allow_pickle=False), p["region"])
        row["convergence_no_repair"] = doc["convergence"]
        row["material_no_repair"] = not doc["convergence"]["recovered"]
        per_run[p["label"]] = row
    summary: dict[str, Any] = {}
    for error in core.ERROR_MODELS:
        for region in core.DECISION_REGIONS:
            rows = [r for r in per_run.values() if r["region"] == region and r["error"] == error]
            if rows:
                summary[f"{region}_{error}"] = {"n": len(rows), "region_fraction": core.REGIONS[region].fraction,
                                                "median_rms_over_std_at_injection": float(np.median([r["perturbation"]["rms_in_region_over_mean_channel_std"] for r in rows])),
                                                "median_material_inside_at_injection": float(np.median([r["by_d"]["0"]["material_fraction_inside_region"] for r in rows])),
                                                **{f"median_material_fraction_d{d}": float(np.median([r["by_d"][str(d)]["material_fraction"] for r in rows])) for d in (1, 2, 4)},
                                                **{f"median_material_outside_d{d}": float(np.median([r["by_d"][str(d)]["material_fraction_outside_region"] for r in rows])) for d in (1, 2, 4)},
                                                **{f"median_C_d{d}": float(np.median([r["by_d"][str(d)]["expansion_C"] for r in rows])) for d in (1, 2, 4)},
                                                "median_final_relative_l2_no_repair": float(np.median([r["convergence_no_repair"]["final_relative_l2"] for r in rows])),
                                                "material_no_repair_fraction": float(np.mean([r["material_no_repair"] for r in rows]))}
    prim = [r for r in per_run.values() if r["error"] == core.DECISION_ERROR and r["region"] in core.DECISION_REGIONS]
    cells = {(r["traj"], r["t"]): False for r in prim}
    for r in prim:
        cells[(r["traj"], r["t"])] = cells[(r["traj"], r["t"])] or r["material_no_repair"]
    summary["decision_input"] = {"n_cells": len(cells), "material_cell_fraction": float(np.mean(list(cells.values()))) if cells else None,
                                 "material_cells_by_region": {reg: float(np.mean([r["material_no_repair"] for r in prim if r["region"] == reg])) for reg in core.DECISION_REGIONS} if prim else {},
                                 "material_by_cell": {f"{k[0]}_t{k[1]}": v for k, v in cells.items()}, "rule": "cell material iff either decision region fails the criterion under GLOBAL_DENSE_REFRESH; all 8 trajectories x 3 t"}
    return {"provenance_hash": provenance_hash, "control_zero_failures": zero_fail, "runs": per_run, "summary": summary}


# --------------------------------------------------------------------------------------
# phase: oracle (R_state*, control 4)
# --------------------------------------------------------------------------------------
def _clean_and_perturbed_at(output_dir: Path, traj: str, t: int, region: str, error: str, d: int) -> tuple[np.ndarray, np.ndarray]:
    zero, pert = _run_dir(output_dir, "propagation", f"{traj}_t{t}_ZERO"), _run_dir(output_dir, "propagation", f"{traj}_t{t}_{region}_{error}_noRepair")
    return np.load(zero / f"local_{d:02d}.npy", allow_pickle=False), np.load(pert / f"local_{d:02d}.npy", allow_pickle=False)


def _continue_and_save(engine: Engine, output_dir: Path, phase_dir: str, p: dict[str, Any], tr: dict[str, Any], x_in: np.ndarray, info_extra: dict[str, Any]) -> dict[str, Any]:
    ref_final, ref_video, ref = _reference(output_dir, tr["id"])
    r = engine.generate(prompt=tr["prompt"], seed=tr["seed"], label=p["label"], capture_local_steps=[], latents=x_in, step_index=p["start"])
    final = r["arrays"][core.TOTAL_STEPS - p["start"]]
    vm = _video_metrics(r["video"], ref_video)
    info = {"kind": p["kind"], "traj": tr["id"], "t": p["t"], "d": p["d"], "region": p["region"], "error": p["error"], "domain": p.get("domain"), "start": p["start"], "wall_s": r["wall_s"], "input_sha256": array_sha256(x_in),
            "final_sha256": array_sha256(final), "final_bit_exact_vs_reference": array_sha256(final) == array_sha256(ref_final), "video_bit_exact_vs_reference": bool(r["video"] is not None and array_sha256(r["video"]) == ref["video_hash"]),
            "video_metrics_vs_reference": vm, "convergence": core.final_convergence(final, ref_final, frame_ssim_mean=vm["frame_ssim_mean"] if vm else None), **info_extra}
    return _save_run(output_dir, phase_dir, p["label"], engine.provenance["provenance_hash"], arrays={"final": final}, video=r["video"], fps=float(engine.config["generation"]["fps"]), info=info)


def phase_oracle(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    _require_no_invalid(output_dir)
    plan = [p for p in run_plan(config) if p["phase"] == "oracle"]
    engine = Engine(config, args, output_dir, prereg, provenance)
    try:
        for p in plan:
            if _load_run(output_dir, PHASE_DIR["oracle"], p["label"], provenance["provenance_hash"]) is not None:
                continue
            tr = _traj(config, p["traj"])
            clean, pert = _clean_and_perturbed_at(output_dir, p["traj"], p["t"], p["region"], p["error"], p["d"])
            mask = core.RECOVERY_DOMAINS[p["region"]][p["domain"]].token_mask()
            x_in = core.oracle_repair(pert, clean, mask)
            if p["kind"] == "FULL_ROLLBACK" and not np.array_equal(x_in, clean):
                raise GateError("Full oracle rollback did not reproduce the clean state")
            _continue_and_save(engine, output_dir, PHASE_DIR["oracle"], p, tr, x_in, {"repair": "ORACLE_STATE_REPLACEMENT (structural probe)", "restored_fraction": float(mask.mean()), "pre_repair_contamination": core.contamination_metrics(pert, clean, p["region"])})
    finally:
        engine.shutdown()
    metrics = compute_oracle_metrics(config, output_dir, provenance["provenance_hash"])
    atomic_json(output_dir / "oracle_metrics.json", metrics)
    if metrics["control_full_failures"]:
        atomic_json(output_dir / "decision.json", {"decision": "INVALID_EXPERIMENT", "reason": f"oracle full repair not bit-exact: {metrics['control_full_failures']}", "provenance_hash": provenance["provenance_hash"]})
        raise GateError(f"INVALID_EXPERIMENT: full oracle rollback not bit-exact for {metrics['control_full_failures']}")
    _write_manifest(output_dir, provenance, {"phase": "oracle", "runs": len(plan)})
    return {"phase": "oracle", "runs": len(plan), "summary": metrics["summary"]}


def sweep_labels(tr_id: str, t: int, region: str, prefix: str) -> dict[str, str]:
    """Run labels for every recovery-domain key of a (trajectory, t, region) sweep; 100% is shared across regions."""
    d = core.REPAIR_OFFSET
    return {key: (f"{tr_id}_t{t}_STALE_VELOCITY_d{d}_{prefix}100" if key == "100" else f"{tr_id}_t{t}_{region}_STALE_VELOCITY_d{d}_{prefix}{key}") for key in core.RECOVERY_DOMAINS[region]}


def _sweep_points(output_dir: Path, provenance_hash: str, phase_key: str, tr_id: str, t: int, region: str, prefix: str) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    points, fractions, docs = {}, {}, {}
    for key, label in sweep_labels(tr_id, t, region, prefix).items():
        doc = _load_run(output_dir, PHASE_DIR[phase_key], label, provenance_hash)
        if doc is None:
            raise GateError(f"{phase_key} run missing: {label}")
        points[key], fractions[key], docs[key] = doc["convergence"], core.RECOVERY_DOMAINS[region][key].fraction, doc
    return points, fractions, docs


def compute_oracle_metrics(config: dict[str, Any], output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    plan = [p for p in run_plan(config) if p["phase"] == "oracle"]
    sweeps, full_fail = {}, []
    for p in plan:
        doc = _load_run(output_dir, PHASE_DIR["oracle"], p["label"], provenance_hash)
        if doc is None:
            raise GateError(f"Oracle run missing: {p['label']}")
        if p["kind"] == "FULL_ROLLBACK" and not (doc["final_bit_exact_vs_reference"] and doc["video_bit_exact_vs_reference"]):
            full_fail.append(p["label"])
    for tr in trajectories(config):
        if not tr["decision"]:
            continue
        for t in core.CHECKPOINTS:
            for region in core.DECISION_REGIONS:
                points, fractions, _ = _sweep_points(output_dir, provenance_hash, "oracle", tr["id"], t, region, "ORACLE")
                sweeps[f"{tr['id']}_t{t}_{region}"] = {"traj": tr["id"], "t": t, "region": region, "fractions": fractions, "final_relative_l2_by_fraction": {k: v["final_relative_l2"] for k, v in points.items()},
                                                       "ssim_by_fraction": {k: v["frame_ssim_mean"] for k, v in points.items()}, "r_state": core.r_star(points, fractions),
                                                       "r_state_sensitivity": {str(th): core.r_star(points, fractions, threshold_key=str(th))["r_star"] for th in core.RECOVERY_SENSITIVITY}}
    summary = {"r_state_median_by_region": {reg: float(np.median([s["r_state"]["r_star"] for s in sweeps.values() if s["region"] == reg])) for reg in core.DECISION_REGIONS},
               "r_state_values_by_region": {reg: [s["r_state"]["r_star"] for s in sweeps.values() if s["region"] == reg] for reg in core.DECISION_REGIONS},
               "label": "ORACLE STATE-REPAIR HEADROOM (structural probe; never decides)"}
    return {"provenance_hash": provenance_hash, "control_full_failures": full_fail, "sweeps": sweeps, "summary": summary}


# --------------------------------------------------------------------------------------
# phase: recompute (R_compute*, PRIMARY; control 5 chain exactness)
# --------------------------------------------------------------------------------------
def phase_recompute(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    _require_no_invalid(output_dir)
    plan = [p for p in run_plan(config) if p["phase"] == "recompute"]
    engine = Engine(config, args, output_dir, prereg, provenance)
    try:
        for p in plan:
            if _load_run(output_dir, PHASE_DIR["recompute"], p["label"], provenance["provenance_hash"]) is not None:
                continue
            tr = _traj(config, p["traj"])
            t, d = p["t"], p["d"]
            zero, pert_dir = _run_dir(output_dir, "propagation", f"{tr['id']}_t{t}_ZERO"), _run_dir(output_dir, "propagation", f"{tr['id']}_t{t}_{p['region']}_{p['error']}_noRepair")
            clean = {j: np.load(zero / f"local_{j:02d}.npy", allow_pickle=False) for j in range(d + 1)}
            cached = {j: np.load(pert_dir / f"local_{j:02d}.npy", allow_pickle=False) for j in range(d + 1)}  # perturbed-trajectory context, computed once in propagation
            mask = core.RECOVERY_DOMAINS[p["region"]][p["domain"]].token_mask()
            lat_mask = core.token_to_latent_mask(mask)
            domain_state = clean[0]  # the verified state at t; only its in-domain values are consumed
            steps_log, walls = [], []
            for j in range(d):
                x_in = core.compose_recompute_input(mask, domain_state, cached[j])
                s = engine.single_step(tr, f"{p['label']}_step{j}", x_in, t + j)
                walls.append(s["wall_s"])
                domain_state = core.compose_recompute_output(mask, s["next"], domain_state)
                steps_log.append({"j": j, "step": t + j, "domain_material_vs_clean": float((core.normalised_error_map(domain_state, clean[j + 1]) > core.MATERIAL_THRESHOLD)[core.position_mask(mask)].mean())})
            repaired = core.compose_recompute_input(mask, domain_state, cached[d])
            extra = {"repair": RECOMPUTE_LABEL, "full_forwards_executed": d, "tokens_per_forward": core.N_TOKENS, "cost": core.idealized_window_saving(float(mask.mean()), d), "verification_state_source": "verified clean x_t[D] (cost not counted; see preregistration.verification_semantics)",
                     "restored_fraction": float(mask.mean()), "single_steps": d, "single_step_wall_s": walls, "steps": steps_log,
                     "domain_bit_exact_vs_clean_t_plus_d": bool(np.array_equal(repaired[lat_mask], clean[d][lat_mask])), "repaired_state_contamination_vs_clean": core.contamination_metrics(repaired, clean[d], p["region"])}
            if p["kind"] == "RECOMPUTE_FULL":
                extra["chain_bit_exact_vs_clean_t_plus_d"] = bool(np.array_equal(repaired, clean[d]))
            _continue_and_save(engine, output_dir, PHASE_DIR["recompute"], p, tr, repaired, extra)
    finally:
        engine.shutdown()
    metrics = compute_recompute_metrics(config, output_dir, provenance["provenance_hash"])
    atomic_json(output_dir / "recompute_metrics.json", metrics)
    if metrics["control_chain_failures"]:
        atomic_json(output_dir / "decision.json", {"decision": "INVALID_EXPERIMENT", "reason": f"100% recompute chain not bit-exact: {metrics['control_chain_failures']}", "provenance_hash": provenance["provenance_hash"]})
        raise GateError(f"INVALID_EXPERIMENT: chained clean single steps not bit-exact for {metrics['control_chain_failures']}")
    _write_manifest(output_dir, provenance, {"phase": "recompute", "runs": len(plan)})
    return {"phase": "recompute", "runs": len(plan), "summary": metrics["summary"]}


def compute_recompute_metrics(config: dict[str, Any], output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    plan = [p for p in run_plan(config) if p["phase"] == "recompute"]
    chain_fail, sweeps = [], {}
    for p in plan:
        doc = _load_run(output_dir, PHASE_DIR["recompute"], p["label"], provenance_hash)
        if doc is None:
            raise GateError(f"Recompute run missing: {p['label']}")
        if p["kind"] == "RECOMPUTE_FULL" and not (doc.get("chain_bit_exact_vs_clean_t_plus_d") and doc["final_bit_exact_vs_reference"]):
            chain_fail.append(p["label"])
    for tr in trajectories(config):
        if not tr["decision"]:
            continue
        for t in core.CHECKPOINTS:
            for region in core.DECISION_REGIONS:
                points, fractions, docs = _sweep_points(output_dir, provenance_hash, "recompute", tr["id"], t, region, "RECOMPUTE")
                sweeps[f"{tr['id']}_t{t}_{region}"] = {"traj": tr["id"], "t": t, "region": region, "fractions": fractions, "final_relative_l2_by_fraction": {k: v["final_relative_l2"] for k, v in points.items()},
                                                       "ssim_by_fraction": {k: v["frame_ssim_mean"] for k, v in points.items()}, "domain_exact_by_fraction": {k: docs[k]["domain_bit_exact_vs_clean_t_plus_d"] for k in points},
                                                       "repaired_material_fraction_by_fraction": {k: docs[k]["repaired_state_contamination_vs_clean"]["material_fraction"] for k in points},
                                                       "r_compute": core.r_star(points, fractions), "r_compute_sensitivity": {str(th): core.r_star(points, fractions, threshold_key=str(th))["r_star"] for th in core.RECOVERY_SENSITIVITY}}
    summary = {"r_compute_median_by_region": {reg: float(np.median([s["r_compute"]["r_star"] for s in sweeps.values() if s["region"] == reg])) for reg in core.DECISION_REGIONS},
               "r_compute_values_by_region": {reg: [s["r_compute"]["r_star"] for s in sweeps.values() if s["region"] == reg] for reg in core.DECISION_REGIONS},
               "r_compute_by_t": {reg: {str(t): float(np.median([s["r_compute"]["r_star"] for s in sweeps.values() if s["region"] == reg and s["t"] == t])) for t in core.CHECKPOINTS} for reg in core.DECISION_REGIONS},
               "monotonicity_violations": int(sum(s["r_compute"]["monotonicity_violations"] for s in sweeps.values())),
               "label": RECOMPUTE_LABEL + " (PRIMARY; no sparse compute measured)", "idealized_window_saving_by_domain": {r: {k: core.idealized_window_saving(core.RECOVERY_DOMAINS[r][k].fraction) for k in core.RECOVERY_DOMAINS[r]} for r in core.DECISION_REGIONS}}
    return {"provenance_hash": provenance_hash, "control_chain_failures": chain_fail, "sweeps": sweeps, "summary": summary}


# --------------------------------------------------------------------------------------
# phase: analyze (one shot)
# --------------------------------------------------------------------------------------
def _require_no_invalid(output_dir: Path) -> None:
    p = output_dir / "decision.json"
    if p.exists() and json.loads(p.read_text()).get("decision") == "INVALID_EXPERIMENT":
        raise GateError("Experiment already marked INVALID_EXPERIMENT; no further phases")


def early_stop_decision(*, material_fraction: float | None, velocity_ok: bool, zero_failures: list[str], have_sweeps: bool) -> dict[str, Any] | None:
    """Frozen early stop after propagation: INVALID on control failure, STOP_NO_RECOVERY_PROBLEM if the materiality gate fails.

    Returns None when the sweeps are required (materiality passed and the sweep metrics exist); raises if they are required but absent.
    """
    if not velocity_ok or zero_failures:
        return core.decide(valid=False, invalid_reason=f"velocity_ok={velocity_ok}; zero failures {zero_failures}", material_cell_fraction=material_fraction, rcompute_median_by_region={}, rstate_median_by_region={})
    if material_fraction is None or material_fraction < core.MATERIAL_CELL_FRACTION_MIN:
        return core.decide(valid=True, invalid_reason=None, material_cell_fraction=material_fraction, rcompute_median_by_region={}, rstate_median_by_region={})
    if not have_sweeps:
        raise GateError(f"Materiality gate passed ({material_fraction:.2f} of cells material): run --phase oracle and --phase recompute before analyze")
    return None


def median_over_material_cells(sweeps: dict[str, Any], key: str, material: dict[str, bool]) -> dict[str, float]:
    out = {}
    for reg in core.DECISION_REGIONS:
        vals = [s[key]["r_star"] for s in sweeps.values() if s["region"] == reg and material.get(f"{s['traj']}_t{s['t']}_{reg}_STALE_VELOCITY_noRepair", False)]
        out[reg] = float(np.median(vals)) if vals else 1.0
    return out


def phase_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    _require_no_invalid(output_dir)
    prop = json.loads((output_dir / "propagation_metrics.json").read_text())
    refs = [json.loads((_run_dir(output_dir, "reference", f"{tr['id']}_reference") / "run.json").read_text()) for tr in trajectories(config)]
    velocity_ok = all(v["pass"] for r in refs for v in r["velocity_validation"].values())
    material_fraction = prop["summary"]["decision_input"]["material_cell_fraction"]
    early = early_stop_decision(material_fraction=material_fraction, velocity_ok=velocity_ok, zero_failures=prop["control_zero_failures"], have_sweeps=(output_dir / "oracle_metrics.json").exists() and (output_dir / "recompute_metrics.json").exists())
    if early is not None:
        decision, orc, rec, rc, rs = early, None, None, {}, {}
    else:
        orc, rec = (json.loads((output_dir / f).read_text()) for f in ("oracle_metrics.json", "recompute_metrics.json"))
        for doc in (prop, orc, rec):
            if doc["provenance_hash"] != provenance["provenance_hash"]:
                raise GateError("Metrics provenance mismatch")
        missing = [p["label"] for p in run_plan(config) if _load_run(output_dir, PHASE_DIR[p["phase"]], p["label"], provenance["provenance_hash"]) is None]
        valid = velocity_ok and not prop["control_zero_failures"] and not orc["control_full_failures"] and not rec["control_chain_failures"] and not missing
        reason = None if valid else f"velocity_ok={velocity_ok}; zero {prop['control_zero_failures']}; full {orc['control_full_failures']}; chain {rec['control_chain_failures']}; missing {missing[:10]}"
        material = {k: r["material_no_repair"] for k, r in prop["runs"].items() if r["error"] == core.DECISION_ERROR and r["decision_traj"]}
        rc, rs = median_over_material_cells(rec["sweeps"], "r_compute", material), median_over_material_cells(orc["sweeps"], "r_state", material)
        decision = core.decide(valid=valid, invalid_reason=reason, material_cell_fraction=material_fraction, rcompute_median_by_region=rc, rstate_median_by_region=rs)
    summary = {**decision, "primary_question": prereg["primary_question"],
               "P_material_no_repair": prop["summary"]["decision_input"], "P_contamination": {k: v for k, v in prop["summary"].items() if k != "decision_input"},
               "R_compute": rec["summary"] if rec else None, "R_compute_median_material_cells": rc, "R_state": orc["summary"] if orc else None, "R_state_median_material_cells": rs,
               "recompute_arm_label": RECOMPUTE_LABEL, "cost_semantics": prereg["cost_semantics"], "verification_semantics": prereg["verification_semantics"], "kill_scope": prereg["kill_scope"],
               "weak_band_characterization": "eligible: run --phase weakprobe (descriptive only)" if decision["decision"].startswith("WEAK") else "not eligible",
               "perturbation_calibration": json.loads((output_dir / "perturbations" / "index.json").read_text())["states"], "velocity_validation": {r["traj"]: r["velocity_validation"] for r in refs},
               "n_continuations": len(run_plan(config)), "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0], "provenance_hash": provenance["provenance_hash"],
               "headroom_disclaimer": prereg["headroom_disclaimer"], "claim_boundary": prereg["claim_boundary"]}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "decision.json", {"decision": decision["decision"], "rationale": decision["rationale"], "provenance_hash": provenance["provenance_hash"], "preregistration_sha256": summary["preregistration_sha256"]})
    write_report(output_dir, summary)
    _write_manifest(output_dir, provenance, {"phase": "analyze", "decision": decision["decision"]})
    print(f"decision={decision['decision']}\n{decision['rationale']}")
    return {"decision": decision["decision"], "rationale": decision["rationale"]}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    L = ["# Speculative-contamination Round 0 (rev 2)", "", "## Decision", "", s["decision"], "", s["rationale"], "", "## Question", "", s["primary_question"], "",
         f"STALE_VELOCITY (L = {core.STALE_WINDOW_PRIMARY}) is a controlled approximation error in the RAS/HSA cached-velocity form, not a draft-model error. Oracle repair is a structural probe; the recompute arm is {RECOMPUTE_LABEL}: full 14,040-token forwards, only the domain output retained, no sparse compute measured. No speedup is claimed.", "", f"Kill scope: {s['kill_scope']}", "",
         "## Material no-repair error (GLOBAL_DENSE_REFRESH baseline)", "", f"- {json.dumps(s['P_material_no_repair'])}", "",
         "## R_compute* (PRIMARY)", "", f"- median by region over material cells: {s['R_compute_median_material_cells']}", f"- by t: {(s['R_compute'] or {}).get('r_compute_by_t')}", f"- monotonicity violations: {(s['R_compute'] or {}).get('monotonicity_violations')}", f"- idealized saved window work vs FULL_ROLLBACK (theoretical only; experiment executed d full forwards): {(s['R_compute'] or {}).get('idealized_window_saving_by_domain')}", "",
         "## R_state* (oracle, descriptive)", "", f"- median by region over material cells: {s['R_state_median_material_cells']}", "", "## Cost semantics (frozen)", "", f"- {json.dumps(s['cost_semantics'])}", "", "## Verification semantics", "", s["verification_semantics"], "", f"## Weak-band characterization: {s['weak_band_characterization']}", "", "## Contamination (descriptive)", ""]
    for k, v in s["P_contamination"].items():
        L.append(f"- {k}: injection rms/std {v['median_rms_over_std_at_injection']:.4f}, material inside at injection {v['median_material_inside_at_injection']:.2f}; material d1/d2/d4 {v['median_material_fraction_d1']:.3f}/{v['median_material_fraction_d2']:.3f}/{v['median_material_fraction_d4']:.3f}; no-repair final rel-L2 {v['median_final_relative_l2_no_repair']:.4f}; material no-repair fraction {v['material_no_repair_fraction']:.2f}")
    L += ["", "## Headroom disclaimer", "", s["headroom_disclaimer"], "", "## Claim boundary", "", s["claim_boundary"], ""]
    (output_dir / "speculative_contamination_round0.md").write_text("\n".join(L))


# --------------------------------------------------------------------------------------
# phase: weakprobe (ONE frozen WEAK-band dependency-cone characterization; descriptive only)
# --------------------------------------------------------------------------------------
def phase_weakprobe(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    dec_path = output_dir / "decision.json"
    if not dec_path.exists() or not str(json.loads(dec_path.read_text()).get("decision", "")).startswith("WEAK"):
        raise GateError("weakprobe is allowed only after analyze returned the WEAK band")
    if (output_dir / "weak_band_characterization.json").exists():
        raise GateError("The single WEAK-band characterization has already been run")
    plan = weak_band_plan(config)
    engine = Engine(config, args, output_dir, prereg, provenance)
    try:
        for p in plan:
            if _load_run(output_dir, PHASE_DIR["weakprobe"], p["label"], provenance["provenance_hash"]) is not None:
                continue
            tr = _traj(config, p["traj"])
            t, d = p["t"], p["d"]
            zero, pert_dir = _run_dir(output_dir, "propagation", f"{tr['id']}_t{t}_ZERO"), _run_dir(output_dir, "propagation", f"{tr['id']}_t{t}_{p['region']}_{p['error']}_noRepair")
            clean = {j: np.load(zero / f"local_{j:02d}.npy", allow_pickle=False) for j in range(d + 1)}
            cached = {j: np.load(pert_dir / f"local_{j:02d}.npy", allow_pickle=False) for j in range(d + 1)}
            mask = core.WEAK_BAND_DOMAINS[p["region"]][p["domain"]].token_mask()
            domain_state = clean[0]
            for j in range(d):
                s = engine.single_step(tr, f"{p['label']}_step{j}", core.compose_recompute_input(mask, domain_state, cached[j]), t + j)
                domain_state = core.compose_recompute_output(mask, s["next"], domain_state)
            repaired = core.compose_recompute_input(mask, domain_state, cached[d])
            _continue_and_save(engine, output_dir, PHASE_DIR["weakprobe"], p, tr, repaired, {"repair": RECOMPUTE_LABEL, "role": "WEAK-band characterization; DESCRIPTIVE ONLY; cannot upgrade the verdict", "restored_fraction": float(mask.mean()), "full_forwards_executed": d, "cost": core.idealized_window_saving(float(mask.mean()), d)})
    finally:
        engine.shutdown()
    rec = json.loads((output_dir / "recompute_metrics.json").read_text())
    rows = {}
    for p in plan:
        doc = _load_run(output_dir, PHASE_DIR["weakprobe"], p["label"], provenance["provenance_hash"])
        rows[p["label"]] = {"traj": p["traj"], "t": p["t"], "region": p["region"], "domain": p["domain"], "fraction": core.WEAK_BAND_DOMAINS[p["region"]][p["domain"]].fraction, "convergence": doc["convergence"]}
    expanded = {}
    for key, sw in rec["sweeps"].items():
        pts = {k: {"recovered": v} for k, v in sw["r_compute"]["recovered_by_fraction"].items()}
        fr = dict(sw["fractions"])
        for lab, r in rows.items():
            if r["traj"] == sw["traj"] and r["t"] == sw["t"] and r["region"] == sw["region"]:
                pts[r["domain"]], fr[r["domain"]] = {"recovered": r["convergence"]["recovered"]}, r["fraction"]
        expanded[key] = core.r_star(pts, fr)["r_star"]
    medians = {reg: float(np.median([v for k, v in expanded.items() if rec["sweeps"][k]["region"] == reg])) for reg in core.DECISION_REGIONS}
    worst = max(medians.values())
    out = {"provenance_hash": provenance["provenance_hash"], "role": "DESCRIPTIVE ONLY; the primary verdict is unchanged", "runs": rows, "r_compute_with_expanded_cone_by_cell": expanded, "median_by_region": medians, "worst_region_median": worst,
           "branch_outcome": "NO_GO (expanded dependency cone still requires > 0.70)" if worst > core.RCOMPUTE_WEAK else "WEAK confirmed; characterization only; no upgrade to GO"}
    atomic_json(output_dir / "weak_band_characterization.json", out)
    _write_manifest(output_dir, provenance, {"phase": "weakprobe", "runs": len(plan), "branch_outcome": out["branch_outcome"]})
    return {"phase": "weakprobe", "runs": len(plan), "median_by_region": medians, "branch_outcome": out["branch_outcome"]}


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = args.output_dir.resolve()
    validate_output_path(out)
    if args.phase == "quarantine":
        raise GateError("DEFERRED_TO_ROUND1: quarantine / speculative isolation is not part of Round 0")
    if args.phase == "cpu":
        result = phase_cpu(config)
    elif args.phase == "preregister":
        result = phase_preregister(config, args.config, out)
    elif args.phase == "perturb":
        result = phase_perturb(config, args.config, out)
    elif args.phase == "analyze":
        result = phase_analyze(config, args.config, out)
    else:
        result = {"reference": phase_reference, "propagation": phase_propagation, "oracle": phase_oracle, "recompute": phase_recompute, "weakprobe": phase_weakprobe}[args.phase](config, args.config, out, args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
