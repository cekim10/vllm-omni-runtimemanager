"""Numerical Compatibility Round 0 - frozen science (pure numpy / JSON; no torch, no GPU).

Question: does an intermediate Wan2.2 denoising state created under execution configuration A remain the same
sample when denoising is continued under configuration X?  Observable: H_{A->X,t} = F_X^{t:40}(x_t^A) versus F_A, F_X.

Everything that decides is here so it can be exercised by CPU truth-table tests before any GPU run.  The GPU runner
(run.py) only produces arrays/hashes and calls these functions; it never re-implements a rule.

Boundary indexing: x_k is the latent ENTERING step k (k = 0..39); x_40 is the final latent.  x_0 is the initial noise drawn on
the CPU generator before any forward, hence identical across configurations for the same seed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------------------------------
# frozen constants (rev 0.2f)
# --------------------------------------------------------------------------------------------------------------------
TOTAL_STEPS = 40
N_BOUNDARIES = TOTAL_STEPS + 1
LATENT_SHAPE = (1, 16, 9, 60, 104)
CHECKPOINTS_T = (12, 20, 28)
HYBRID_DECISION_T = 12
EXACT_RESUME_ANCHOR_T = 12
RESUME_PLUMBING_ANCHOR_T = 0

STATE_REL_L2_MAX = 0.02  # STATE_DIFFERENT iff final latent rel-L2 > this
OUTPUT_SSIM_MIN = 0.95  # OUTPUT_DIFFERENT iff mean frame SSIM < this
MATERIAL_POSITION_THRESHOLD = 0.02  # channel-normalised per-position RMS error (descriptive per-step curve)
MATERIAL_FRACTION_STEP_THRESHOLD = 0.02  # k_material: first boundary with > 2% material positions
AMPLIFICATION_FACTOR_H1 = 10.0  # descriptive hypothesis H1
KSTAR_OFFSET_H1 = 4

CORE_AXES = ("EAGER", "OFFLOAD", "ATTN_SDPA", "ATTN_FLASHINFER", "ATTN_CUDNN")
NOT_MEASURED_AXES = {
    "BATCH": "Wan2.2 pipeline does not implement SupportsStepExecution; step_execution=True is rejected and max_num_seqs>1 is forced to 1; batch invariance is covered by vLLM-Omni #5512 Part 2",
    "PARALLEL": "host has 2x L40S but GPU 1 is allocated to another user; only GPU 0 is available for Round 0",
}
CANONICAL = "CANONICAL"
MIN_CORE_AXES_MEASURED = 2
N_TRAJECTORIES_REQUIRED = 3

WITHIN_REPRODUCIBLE, WITHIN_NONDETERMINISTIC = "WITHIN_REPRODUCIBLE", "WITHIN_NONDETERMINISTIC"
FULL_TRAJECTORY_BIT_INVARIANT, FULL_SAMPLE_EQUIVALENT, FULL_SAMPLE_DIFFERENT, FULL_MIXED = "FULL_TRAJECTORY_BIT_INVARIANT", "FULL_SAMPLE_EQUIVALENT", "FULL_SAMPLE_DIFFERENT", "FULL_MIXED"
HYBRID_INCOMPATIBLE, HYBRID_COMPATIBLE_BOTH, HYBRID_REPLAY_LIKE, HYBRID_FRESH_LIKE, HYBRID_MIXED = "HYBRID_INCOMPATIBLE", "HYBRID_COMPATIBLE_BOTH", "HYBRID_REPLAY_LIKE", "HYBRID_FRESH_LIKE", "HYBRID_MIXED"
CROSS_ENGINE_REPRODUCIBLE, CROSS_ENGINE_NONREPRODUCIBLE, CROSS_ENGINE_NOT_RUN = "CROSS_ENGINE_REPRODUCIBLE", "CROSS_ENGINE_NONREPRODUCIBLE", "CROSS_ENGINE_NOT_RUN"

GO_STRONG, GO, WEAK, NO_GO_PORTABLE, NO_GO_INVARIANT, NO_GO, INVALID = ("GO_STRONG_REGIME_CROSSING", "GO_STATE_NOT_PORTABLE", "WEAK", "NO_GO_STATE_PORTABLE", "NO_GO_INVARIANT_SINGLE_GPU", "NO_GO", "INVALID_EXPERIMENT")

# Declared execution configurations.  rev 0.2e: on the L40S host the platform default resolves to FLASH_ATTN (FA3 provider,
# fa3_fwd_interface / flash_attn_interface), NOT SDPA: the first canonical run of rev0_2 stopped fail-closed on exactly this mismatch.  `expect` holds the runtime-observable fingerprint fields that MUST be recorded with these
# values for a run to count for its configuration (silent fallbacks make the run INVALID for the axis, never re-labelled).
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    CANONICAL: {"enforce_eager": False, "enable_cpu_offload": True, "attention_backend_env": None,
                "expect": {"inductor_active": True, "offload_hooks_present": True, "attention_backend_resolved": "FLASH_ATTN"}},
    "EAGER": {"enforce_eager": True, "enable_cpu_offload": True, "attention_backend_env": None,
              "expect": {"inductor_active": False, "offload_hooks_present": True, "attention_backend_resolved": "FLASH_ATTN"}},
    "OFFLOAD": {"enforce_eager": False, "enable_cpu_offload": False, "attention_backend_env": None,
                "expect": {"inductor_active": True, "offload_hooks_present": False, "attention_backend_resolved": "FLASH_ATTN"}},
    "ATTN_SDPA": {"enforce_eager": False, "enable_cpu_offload": True, "attention_backend_env": "TORCH_SDPA",
                  "expect": {"inductor_active": True, "offload_hooks_present": True, "attention_backend_resolved": "SDPA"}},
    "ATTN_FLASHINFER": {"enforce_eager": False, "enable_cpu_offload": True, "attention_backend_env": "FLASHINFER_ATTN",
                        "expect": {"inductor_active": True, "offload_hooks_present": True, "attention_backend_resolved": "FLASHINFER_ATTN"}},
    "ATTN_CUDNN": {"enforce_eager": False, "enable_cpu_offload": True, "attention_backend_env": "CUDNN_ATTN",
                   "expect": {"inductor_active": True, "offload_hooks_present": True, "attention_backend_resolved": "CUDNN_ATTN"}},
}
# Environment fields that must be identical across ALL configurations (they are not axes); a difference is INVALID.
FINGERPRINT_ENVIRONMENT_FIELDS = ("torch", "cuda", "cudnn", "gpu_model", "model_revision", "dtype", "cublas_workspace_config", "tf32_matmul", "tf32_cudnn", "flashinfer_version", "flash_attn_version", "flash_attn_provider", "parallel")
FINGERPRINT_FIELDS = ("enforce_eager", "inductor_active", "compiled_block_count", "enable_cpu_offload", "offload_hooks_present", "offload_hook_names",
                      "attention_backend_resolved", "attention_backend_names_all", "step_execution", "max_num_seqs", "batch_slot") + FINGERPRINT_ENVIRONMENT_FIELDS

RESULT_RANKING = (
    f"{FULL_SAMPLE_EQUIVALENT} + {HYBRID_INCOMPATIBLE}: neither regime changes the sample when used consistently; incompatibility arises only when state crosses regimes (strongest)",
    f"{FULL_SAMPLE_DIFFERENT} + {HYBRID_INCOMPATIBLE}",
    "everything else",
)

# wall-time model measured on the L40S host (rev 4.1 runs): full run 236 s; resume(t) ~ 55 + 4.5 * (40 - t) s; engine ~300 s
SECONDS_FULL, SECONDS_FIXED, SECONDS_PER_STEP, SECONDS_ENGINE = 236.0, 55.0, 4.5, 300.0


# --------------------------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------------------------
def rel_l2(x: np.ndarray, ref: np.ndarray) -> float:
    d = x.astype(np.float64) - ref.astype(np.float64)
    return float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(ref.astype(np.float64).ravel()), 1e-12))


def channel_std(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[1], -1).astype(np.float64).std(axis=1)


def normalised_rms_map(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    sigma = np.maximum(channel_std(ref), 1e-12)
    d = (x.astype(np.float64) - ref.astype(np.float64))[0] / sigma[:, None, None, None]
    return np.sqrt(np.mean(d * d, axis=0))


def material_fraction(x: np.ndarray, ref: np.ndarray, threshold: float = MATERIAL_POSITION_THRESHOLD) -> float:
    return float((normalised_rms_map(x, ref) > threshold).mean())


def compare_final(*, rel: float, ssim: float | None, bit_exact: bool, trajectory_bit_exact: bool | None = None) -> dict[str, Any]:
    """Frozen vocabulary for one final-state comparison.  ssim None (video not decoded) counts as OUTPUT_SAME only when bit-exact.

    trajectory_bit_exact (rev 0.2d): all 41 latent boundaries AND the video bit-exact; None when the boundaries are not both available.
    """
    if trajectory_bit_exact and not bit_exact:
        raise ValueError("trajectory-wide bit-exactness implies final bit-exactness")
    state_different = bool(rel > STATE_REL_L2_MAX)
    if ssim is None:
        output_different = not bit_exact
    else:
        output_different = bool(ssim < OUTPUT_SSIM_MIN)
    if bit_exact and (state_different or output_different):
        raise ValueError("bit-exact states cannot be STATE_/OUTPUT_DIFFERENT")
    return {"rel_l2": float(rel), "frame_ssim_mean": None if ssim is None else float(ssim), "bit_exact": bool(bit_exact),
            "state_different": state_different, "output_different": output_different,
            "strict_compatible": bool(not state_different and not output_different),
            "different_sample": bool(state_different and output_different), "trajectory_bit_exact": trajectory_bit_exact}


def divergence_curve(hashes_x: list[str], hashes_ref: list[str], rel_by_k: list[float] | None, material_by_k: list[float] | None) -> dict[str, Any]:
    """k* (first boundary with any bit difference) and k_material (first boundary with > 2% material positions)."""
    if len(hashes_x) != N_BOUNDARIES or len(hashes_ref) != N_BOUNDARIES:
        raise ValueError("expected hashes at all 41 boundaries")
    diff = [a != b for a, b in zip(hashes_x, hashes_ref)]
    k_star = next((k for k, d in enumerate(diff) if d), None)
    k_material = None
    if material_by_k is not None:
        k_material = next((k for k, m in enumerate(material_by_k) if m > MATERIAL_FRACTION_STEP_THRESHOLD), None)
    return {"k_star": k_star, "k_material": k_material, "bit_exact_all": k_star is None,
            "rel_l2_by_k": None if rel_by_k is None else [float(v) for v in rel_by_k],
            "material_fraction_by_k": None if material_by_k is None else [float(v) for v in material_by_k]}


def amplification_h1(rel_by_k: list[float], k_star: int | None) -> dict[str, Any]:
    """Descriptive H1: final rel-L2 >= 10 x rel-L2 at boundary k*+4."""
    if k_star is None:
        return {"applicable": False, "reason": "bit-exact"}
    base_k = min(k_star + KSTAR_OFFSET_H1, TOTAL_STEPS)
    base = float(rel_by_k[base_k])
    final = float(rel_by_k[TOTAL_STEPS])
    ratio = None if base <= 0 else final / base
    return {"applicable": True, "base_boundary": base_k, "rel_l2_base": base, "rel_l2_final": final, "ratio": ratio,
            "h1_holds": None if ratio is None else bool(ratio >= AMPLIFICATION_FACTOR_H1)}


# --------------------------------------------------------------------------------------------------------------------
# classification (each over the trajectory set; 3/3 required for every class)
# --------------------------------------------------------------------------------------------------------------------
def _require_all(per_traj: dict[str, Any]) -> None:
    if len(per_traj) != N_TRAJECTORIES_REQUIRED:
        raise ValueError(f"classification requires exactly {N_TRAJECTORIES_REQUIRED} trajectories, got {sorted(per_traj)}")


def within_class(per_traj: dict[str, dict[str, Any]]) -> str:
    """per_traj[traj] = compare of F_X run 2 vs run 1 (final latent) plus video_bit_exact."""
    _require_all(per_traj)
    ok = all(c["bit_exact"] and c.get("video_bit_exact", True) for c in per_traj.values())
    return WITHIN_REPRODUCIBLE if ok else WITHIN_NONDETERMINISTIC


def full_relation(per_traj: dict[str, dict[str, Any]]) -> str:
    """per_traj[traj] = compare_final(F_X run 1 vs F_A run 1) with trajectory_bit_exact set.

    rev 0.2d: FULL_TRAJECTORY_BIT_INVARIANT requires ALL 41 latent boundaries and the video to be bit-exact in 3/3.  A final-only
    bit-exact run whose intermediate states differ is NOT invariant (it is FULL_SAMPLE_EQUIVALENT and enters D).
    """
    _require_all(per_traj)
    vals = list(per_traj.values())
    if any(c.get("trajectory_bit_exact") is None for c in vals):
        raise ValueError("full_relation requires trajectory_bit_exact (all-boundary hashes) for every trajectory")
    if all(c["trajectory_bit_exact"] for c in vals):
        return FULL_TRAJECTORY_BIT_INVARIANT
    if all(c["strict_compatible"] for c in vals):
        return FULL_SAMPLE_EQUIVALENT
    if all(not c["strict_compatible"] for c in vals):
        return FULL_SAMPLE_DIFFERENT
    return FULL_MIXED


def hybrid_class(vs_a: dict[str, dict[str, Any]], vs_x: dict[str, dict[str, Any]]) -> str:
    """H_{A->X,t} compared with F_A (vs_a) and with F_X (vs_x), per trajectory.

    Precedence (rev 0.2c): compatible with both -> HYBRID_COMPATIBLE_BOTH (no incompatibility); incompatible with both 3/3 ->
    HYBRID_INCOMPATIBLE; compatible with exactly one side 3/3 -> REPLAY_LIKE / FRESH_LIKE; otherwise MIXED.
    """
    _require_all(vs_a)
    _require_all(vs_x)
    if set(vs_a) != set(vs_x):
        raise ValueError("hybrid comparisons must cover the same trajectories")
    ca = {k: bool(vs_a[k]["strict_compatible"]) for k in vs_a}
    cx = {k: bool(vs_x[k]["strict_compatible"]) for k in vs_x}
    if all(ca.values()) and all(cx.values()):
        return HYBRID_COMPATIBLE_BOTH
    if not any(ca.values()) and not any(cx.values()):
        return HYBRID_INCOMPATIBLE
    if all(ca.values()):
        return HYBRID_REPLAY_LIKE
    if all(cx.values()):
        return HYBRID_FRESH_LIKE
    return HYBRID_MIXED


def t_portable(vs_a_by_t: dict[int, dict[str, dict[str, Any]]]) -> int | None:
    """Earliest t in CHECKPOINTS_T at which H_{A->X,t} is STRICT_COMPATIBLE with F_A in every trajectory (descriptive)."""
    for t in CHECKPOINTS_T:
        per = vs_a_by_t.get(t)
        if per is None:
            continue
        _require_all(per)
        if all(c["strict_compatible"] for c in per.values()):
            return t
    return None


def cross_engine_class(fresh_vs_run1: dict[str, Any] | None) -> str:
    if fresh_vs_run1 is None:
        return CROSS_ENGINE_NOT_RUN
    return CROSS_ENGINE_REPRODUCIBLE if fresh_vs_run1["bit_exact"] else CROSS_ENGINE_NONREPRODUCIBLE


def axis_in_decision_set(within: str, full: str) -> bool:
    """D = within-reproducible axes whose regime change is numerically visible somewhere along the trajectory (not trajectory-bit-invariant)."""
    return within == WITHIN_REPRODUCIBLE and full != FULL_TRAJECTORY_BIT_INVARIANT


# --------------------------------------------------------------------------------------------------------------------
# anchors / validity
# --------------------------------------------------------------------------------------------------------------------
def anchor_failures(*, x0_bit_exact_by_axis: dict[str, dict[str, bool]], canonical_within: str, exact_resume_bit_exact: dict[str, bool],
                    resume_plumbing_bit_exact_by_axis: dict[str, dict[str, bool]], fingerprint_failures: list[str], n_core_measured: int, gate_errors: list[str]) -> list[str]:
    fails: list[str] = []
    for axis, per in x0_bit_exact_by_axis.items():
        bad = [tr for tr, ok in per.items() if not ok]
        if bad:
            fails.append(f"anchor0: x_0 differs from canonical under {axis} for {bad}")
    if canonical_within != WITHIN_REPRODUCIBLE:
        fails.append("anchor1: canonical F_A not WITHIN_REPRODUCIBLE")
    bad = [tr for tr, ok in exact_resume_bit_exact.items() if not ok]
    if bad or len(exact_resume_bit_exact) != N_TRAJECTORIES_REQUIRED:
        fails.append(f"anchor2: H_(A->A,{EXACT_RESUME_ANCHOR_T}) not bit-exact with F_A for {bad or 'missing trajectories'}")
    for axis, per in resume_plumbing_bit_exact_by_axis.items():
        bad = [tr for tr, ok in per.items() if not ok]
        if bad or len(per) != N_TRAJECTORIES_REQUIRED:
            fails.append(f"anchor3: H_(A->{axis},0) not bit-exact with F_{axis} run 1 for {bad or 'missing trajectories'}")
    fails.extend(f"fingerprint: {f}" for f in fingerprint_failures)
    if n_core_measured < MIN_CORE_AXES_MEASURED:
        fails.append(f"fewer than {MIN_CORE_AXES_MEASURED} core axes measured ({n_core_measured})")
    fails.extend(f"gate: {g}" for g in gate_errors)
    return fails


def fingerprint_check(configuration: str, recorded: dict[str, Any]) -> list[str]:
    """Declared-vs-recorded check for one run.  Returns a list of mismatches (empty = pass)."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {configuration}")
    cfg = CONFIGURATIONS[configuration]
    diffs = []
    if recorded.get("enforce_eager") != cfg["enforce_eager"]:
        diffs.append(f"enforce_eager recorded {recorded.get('enforce_eager')} declared {cfg['enforce_eager']}")
    if recorded.get("enable_cpu_offload") != cfg["enable_cpu_offload"]:
        diffs.append(f"enable_cpu_offload recorded {recorded.get('enable_cpu_offload')} declared {cfg['enable_cpu_offload']}")
    for field, expected in cfg["expect"].items():
        if recorded.get(field) != expected:
            diffs.append(f"{field} recorded {recorded.get(field)!r} expected {expected!r}")
    if recorded.get("step_execution") not in (False, None) or (recorded.get("max_num_seqs") not in (1, None)):
        diffs.append("step_execution/max_num_seqs not the single-request regime")
    return diffs


def environment_consistency(recorded_by_run: dict[str, dict[str, Any]]) -> list[str]:
    """Non-axis environment fields must agree across every run (torch/cuda/cudnn/gpu/model revision/TF32/...)."""
    diffs = []
    runs = list(recorded_by_run.items())
    if not runs:
        return ["no fingerprints recorded"]
    _, ref = runs[0]
    for label, rec in runs[1:]:
        for f in FINGERPRINT_ENVIRONMENT_FIELDS:
            if rec.get(f) != ref.get(f):
                diffs.append(f"{label}: {f} = {rec.get(f)!r} != {ref.get(f)!r} ({runs[0][0]})")
    return diffs


# --------------------------------------------------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------------------------------------------------
def decide(*, invalid_reasons: list[str], axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """axes[name] = {"measured": bool, "within": str, "full": str, "hybrid_t12": str | None}.  BATCH/PARALLEL never appear here."""
    if invalid_reasons:
        return {"decision": INVALID, "rationale": "; ".join(invalid_reasons), "decision_set_D": [], "tier_axes": {}}
    for name in axes:
        if name not in CORE_AXES:
            raise ValueError(f"non-core axis {name} cannot enter the decision")
    measured = {n: a for n, a in axes.items() if a.get("measured")}
    D = sorted(n for n, a in measured.items() if axis_in_decision_set(a["within"], a["full"]))
    incompatible = sorted(n for n in D if measured[n].get("hybrid_t12") == HYBRID_INCOMPATIBLE)
    strong = sorted(n for n in incompatible if measured[n]["full"] == FULL_SAMPLE_EQUIVALENT)
    all_invariant = bool(measured) and all(a["full"] == FULL_TRAJECTORY_BIT_INVARIANT and a["within"] == WITHIN_REPRODUCIBLE for a in measured.values())
    if len(D) >= 2 and strong:
        decision, why = GO_STRONG, f"|D|={len(D)} {D}; regime-crossing incompatibility with sample-equivalent full runs on {strong}"
    elif len(D) >= 2 and incompatible:
        decision, why = GO, f"|D|={len(D)} {D}; hybrid incompatible on {incompatible}"
    elif len(D) == 1 and incompatible:
        decision, why = WEAK, f"|D|=1 {D}; the single numerically visible axis is hybrid-incompatible (phenomenon present, generality not shown)"
    elif len(D) >= 1 and not incompatible:
        decision, why = NO_GO_PORTABLE, f"|D|={len(D)} {D}; regimes change numerics but no state crossing produced a third semantics (no HYBRID_INCOMPATIBLE) -> intermediate state is portable for this round's axes"
    elif not D and all_invariant:
        decision, why = NO_GO_INVARIANT, "every measured core axis is FULL_TRAJECTORY_BIT_INVARIANT; the problem, if any, lies in axes this round cannot vary (parallel layout, batching)"
    else:
        classes = ", ".join(f"{n}: {a['within']}/{a['full']}/{a.get('hybrid_t12')}" for n, a in measured.items())
        decision, why = NO_GO, f"|D|={len(D)} {D}; classes {{{classes}}}"
    return {"decision": decision, "rationale": why, "decision_set_D": D, "tier_axes": {"hybrid_incompatible": incompatible, "regime_crossing": strong},
            "claim": ("on this model and host, intermediate state created under phi_0 and continued under at least one other single-GPU configuration is strictly incompatible with both the replay and the fresh run"
                      if decision in (GO_STRONG, GO) else "no state-portability claim")}


# --------------------------------------------------------------------------------------------------------------------
# run plan and cost
# --------------------------------------------------------------------------------------------------------------------
def run_plan(trajectories: list[dict[str, Any]], fresh_trajectory: str) -> list[dict[str, Any]]:
    """Every GPU run of Round 0 with its phase, configuration, kind and resume point.  Deterministic order."""
    if len(trajectories) != N_TRAJECTORIES_REQUIRED:
        raise ValueError("Round 0 uses exactly three trajectories")
    plan: list[dict[str, Any]] = []
    for tr in trajectories:
        for rep in (1, 2):
            plan.append({"phase": "canonical", "config": CANONICAL, "kind": "FULL", "traj": tr["id"], "rep": rep, "t": 0, "label": f"{tr['id']}_CANONICAL_F{rep}", "keep_all_latents": rep == 1})
    for tr in trajectories:
        plan.append({"phase": "canonical", "config": CANONICAL, "kind": "HYBRID", "traj": tr["id"], "rep": 1, "t": EXACT_RESUME_ANCHOR_T, "label": f"{tr['id']}_CANONICAL_H{EXACT_RESUME_ANCHOR_T}", "keep_all_latents": True, "role": "exact_resume_anchor"})
    for axis in CORE_AXES:
        for tr in trajectories:
            for rep in (1, 2):
                plan.append({"phase": "axis", "config": axis, "kind": "FULL", "traj": tr["id"], "rep": rep, "t": 0, "label": f"{tr['id']}_{axis}_F{rep}", "keep_all_latents": rep == 1})
            for t in (RESUME_PLUMBING_ANCHOR_T,) + CHECKPOINTS_T:
                plan.append({"phase": "axis", "config": axis, "kind": "HYBRID", "traj": tr["id"], "rep": 1, "t": t, "label": f"{tr['id']}_{axis}_H{t}", "keep_all_latents": True,
                             "role": "resume_plumbing_anchor" if t == RESUME_PLUMBING_ANCHOR_T else "hybrid"})
    for cfg in (CANONICAL,) + CORE_AXES:
        plan.append({"phase": "freshcheck", "config": cfg, "kind": "FULL", "traj": fresh_trajectory, "rep": 3, "t": 0, "label": f"{fresh_trajectory}_{cfg}_FRESH", "keep_all_latents": False, "role": "fresh_engine_control"})
    return plan


def estimate_cost(plan: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = 0.0
    n_full = n_resume = 0
    for p in plan:
        if p["kind"] == "FULL":
            n_full += 1
            seconds += SECONDS_FULL
        else:
            n_resume += 1
            seconds += SECONDS_FIXED + SECONDS_PER_STEP * (TOTAL_STEPS - p["t"])
    engines = len({(p["phase"], p["config"]) for p in plan})
    seconds += engines * SECONDS_ENGINE
    all_latent_runs = sum(1 for p in plan if p["keep_all_latents"])
    return {"runs": len(plan), "full_runs": n_full, "resume_runs": n_resume, "engine_constructions": engines, "hours_total": seconds / 3600.0,
            "storage_gb_est": (all_latent_runs * 41 * 3.6 + len(plan) * 12) / 1024.0}


def frozen_design() -> dict[str, Any]:
    return {"total_steps": TOTAL_STEPS, "boundary_indexing": "x_k = latent entering step k (k=0..39); x_40 final; x_0 initial noise before any forward",
            "checkpoints_t": list(CHECKPOINTS_T), "hybrid_decision_t": HYBRID_DECISION_T, "exact_resume_anchor_t": EXACT_RESUME_ANCHOR_T, "resume_plumbing_anchor_t": RESUME_PLUMBING_ANCHOR_T,
            "state_rel_l2_max": STATE_REL_L2_MAX, "output_ssim_min": OUTPUT_SSIM_MIN, "material_position_threshold": MATERIAL_POSITION_THRESHOLD,
            "material_fraction_step_threshold": MATERIAL_FRACTION_STEP_THRESHOLD, "core_axes": list(CORE_AXES), "not_measured_axes": dict(NOT_MEASURED_AXES),
            "configurations": {k: {kk: vv for kk, vv in v.items()} for k, v in CONFIGURATIONS.items()}, "fingerprint_fields": list(FINGERPRINT_FIELDS),
            "environment_fields_must_match": list(FINGERPRINT_ENVIRONMENT_FIELDS), "min_core_axes_measured": MIN_CORE_AXES_MEASURED, "trajectories_required": N_TRAJECTORIES_REQUIRED,
            "classes": {"within": [WITHIN_REPRODUCIBLE, WITHIN_NONDETERMINISTIC], "full": [FULL_TRAJECTORY_BIT_INVARIANT, FULL_SAMPLE_EQUIVALENT, FULL_SAMPLE_DIFFERENT, FULL_MIXED],
                        "hybrid": [HYBRID_INCOMPATIBLE, HYBRID_COMPATIBLE_BOTH, HYBRID_REPLAY_LIKE, HYBRID_FRESH_LIKE, HYBRID_MIXED], "cross_engine": [CROSS_ENGINE_REPRODUCIBLE, CROSS_ENGINE_NONREPRODUCIBLE, CROSS_ENGINE_NOT_RUN]},
            "decision_set_D": "core axes that are WITHIN_REPRODUCIBLE and not FULL_TRAJECTORY_BIT_INVARIANT (all 41 boundaries + video bit-exact in 3/3)",
            "decision": {GO_STRONG: "|D| >= 2 and >= 1 axis in D is FULL_SAMPLE_EQUIVALENT and HYBRID_INCOMPATIBLE at t=12",
                         GO: "|D| >= 2 and >= 1 axis in D is HYBRID_INCOMPATIBLE at t=12",
                         WEAK: "|D| == 1 and that axis is HYBRID_INCOMPATIBLE; no preregistered follow-up",
                         NO_GO_PORTABLE: "|D| >= 1 and no axis in D is HYBRID_INCOMPATIBLE (numerics change, state portability holds)",
                         NO_GO_INVARIANT: "|D| == 0 and every measured core axis FULL_TRAJECTORY_BIT_INVARIANT", NO_GO: "otherwise", INVALID: "any anchor / fingerprint / environment / gate failure"},
            "result_ranking": list(RESULT_RANKING), "hypotheses_non_gating": {
                "H1": f"final rel-L2 vs F_A >= {AMPLIFICATION_FACTOR_H1}x rel-L2 at boundary k*+{KSTAR_OFFSET_H1} for every axis in D",
                "H2": "t_portable exists (<= 28) for >= 1 axis in D; predicted to FAIL", "H3": "k* = 1 for every axis in D", "H4": "final divergence p3 > p2 > p0",
                "H5": "every configuration CROSS_ENGINE_REPRODUCIBLE"},
            "wall_time_model_s": {"full": SECONDS_FULL, "resume": f"{SECONDS_FIXED} + {SECONDS_PER_STEP} * (40 - t)", "engine": SECONDS_ENGINE}}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
