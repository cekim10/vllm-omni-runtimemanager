"""CPU truth-table tests for Numerical Compatibility Round 0 (no torch, no GPU).

Corner cases required by the preregistration audit:
  A/X bit-identical                     -> not in D
  A/X non-bit-exact, sample-equivalent  -> in D
  A/X material                          -> in D
  X within-nondeterministic             -> cannot support a compatibility claim
  hybrid replay-like                    -> no state incompatibility
  hybrid fresh-like                     -> no state incompatibility
  hybrid different from both            -> state incompatibility
  1 incompatible axis                   -> lower-tier result (WEAK)
  >= 2 in D + required hybrid evidence  -> GO; sample-equivalent full runs -> GO_STRONG
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.numerical_compatibility_round0 import core

TRAJ = ("p0_s9101", "p2_s9101", "p3_s9101")
REPO = Path(__file__).resolve().parents[2]


def cmp(*, rel: float, ssim: float | None, bit_exact: bool = False, traj: bool | None = None) -> dict:
    if traj is None:
        traj = bit_exact
    return core.compare_final(rel=rel, ssim=ssim, bit_exact=bit_exact, trajectory_bit_exact=traj)


def per(c: dict) -> dict:
    return {t: dict(c) for t in TRAJ}


EXACT = cmp(rel=0.0, ssim=1.0, bit_exact=True)
EQUIV = cmp(rel=0.004, ssim=0.992)          # not bit-exact, strictly compatible
DIFF = cmp(rel=0.15, ssim=0.86)             # different sample
STATE_ONLY = cmp(rel=0.025, ssim=0.998)     # state different, output same -> strict incompatible but NOT "different sample"
FINAL_ONLY_EXACT = cmp(rel=0.0, ssim=1.0, bit_exact=True, traj=False)  # final latent + video bit-exact but intermediates differ


# ------------------------------------------------------------------ vocabulary
def test_vocabulary_flags():
    assert EXACT["strict_compatible"] and not EXACT["state_different"] and not EXACT["output_different"]
    assert EQUIV["strict_compatible"] and not EQUIV["different_sample"]
    assert DIFF["state_different"] and DIFF["output_different"] and DIFF["different_sample"] and not DIFF["strict_compatible"]
    assert STATE_ONLY["state_different"] and not STATE_ONLY["output_different"] and not STATE_ONLY["strict_compatible"] and not STATE_ONLY["different_sample"]
    with pytest.raises(ValueError):
        core.compare_final(rel=0.5, ssim=0.5, bit_exact=True)
    no_video = core.compare_final(rel=0.0, ssim=None, bit_exact=True)
    assert no_video["strict_compatible"]
    no_video_diff = core.compare_final(rel=0.001, ssim=None, bit_exact=False)
    assert no_video_diff["output_different"] and not no_video_diff["strict_compatible"]


def test_thresholds_are_the_frozen_rev41_values():
    assert core.STATE_REL_L2_MAX == 0.02 and core.OUTPUT_SSIM_MIN == 0.95 and core.MATERIAL_POSITION_THRESHOLD == 0.02
    assert core.CHECKPOINTS_T == (12, 20, 28) and core.HYBRID_DECISION_T == 12 and core.TOTAL_STEPS == 40


# ------------------------------------------------------------------ metrics
def test_rel_l2_and_material_fraction():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(core.LATENT_SHAPE).astype(np.float32)
    assert core.rel_l2(ref, ref) == 0.0
    assert core.material_fraction(ref, ref) == 0.0
    x = ref.copy()
    x[0, :, 0, 0, 0] += 1.0  # one position far off
    assert core.material_fraction(x, ref) == pytest.approx(1.0 / np.prod(core.LATENT_SHAPE[2:]))
    assert 0 < core.rel_l2(x, ref) < 0.01


def test_divergence_curve_indexing():
    ref = [f"h{k}" for k in range(41)]
    same = core.divergence_curve(ref, list(ref), None, None)
    assert same["k_star"] is None and same["bit_exact_all"]
    x = list(ref)
    x[1] = "different"  # first forward changes x_1; x_0 identical by construction
    mat = [0.0] * 41
    mat[7] = 0.03
    d = core.divergence_curve(x, ref, [0.0] * 41, mat)
    assert d["k_star"] == 1 and d["k_material"] == 7 and not d["bit_exact_all"]
    with pytest.raises(ValueError):
        core.divergence_curve(ref[:-1], ref, None, None)


def test_amplification_h1():
    rel = [0.0] * 41
    rel[1] = 1e-4
    rel[5] = 1e-3
    rel[40] = 0.05
    h = core.amplification_h1(rel, 1)
    assert h["base_boundary"] == 5 and h["ratio"] == pytest.approx(50.0) and h["h1_holds"]
    assert core.amplification_h1(rel, None)["applicable"] is False


# ------------------------------------------------------------------ axis classes (truth table)
def test_within_class():
    assert core.within_class(per({**EXACT, "video_bit_exact": True})) == core.WITHIN_REPRODUCIBLE
    bad = per({**EXACT, "video_bit_exact": True})
    bad["p3_s9101"] = {**EQUIV, "video_bit_exact": False}
    assert core.within_class(bad) == core.WITHIN_NONDETERMINISTIC
    with pytest.raises(ValueError):
        core.within_class({"p0_s9101": EXACT})


def test_full_relation_truth_table():
    assert core.full_relation(per(EXACT)) == core.FULL_TRAJECTORY_BIT_INVARIANT
    # rev 0.2d blocking fix: final-only bit-exactness with differing intermediates is NOT invariance -> enters D
    assert core.full_relation(per(FINAL_ONLY_EXACT)) == core.FULL_SAMPLE_EQUIVALENT
    assert core.axis_in_decision_set(core.WITHIN_REPRODUCIBLE, core.full_relation(per(FINAL_ONLY_EXACT)))
    with pytest.raises(ValueError, match="trajectory_bit_exact"):
        core.full_relation(per({**EXACT, "trajectory_bit_exact": None}))
    with pytest.raises(ValueError):
        core.compare_final(rel=0.1, ssim=0.5, bit_exact=False, trajectory_bit_exact=True)
    assert core.full_relation(per(EQUIV)) == core.FULL_SAMPLE_EQUIVALENT
    assert core.full_relation(per(DIFF)) == core.FULL_SAMPLE_DIFFERENT
    assert core.full_relation(per(STATE_ONLY)) == core.FULL_SAMPLE_DIFFERENT  # strict incompatible in 3/3
    mixed = per(DIFF)
    mixed["p0_s9101"] = EQUIV
    assert core.full_relation(mixed) == core.FULL_MIXED


def test_decision_set_membership():
    assert not core.axis_in_decision_set(core.WITHIN_REPRODUCIBLE, core.FULL_TRAJECTORY_BIT_INVARIANT)   # bit-identical -> not in D
    assert core.axis_in_decision_set(core.WITHIN_REPRODUCIBLE, core.FULL_SAMPLE_EQUIVALENT)   # non-bit-exact, equivalent -> in D
    assert core.axis_in_decision_set(core.WITHIN_REPRODUCIBLE, core.FULL_SAMPLE_DIFFERENT)    # material -> in D
    assert core.axis_in_decision_set(core.WITHIN_REPRODUCIBLE, core.FULL_MIXED)
    for full in (core.FULL_TRAJECTORY_BIT_INVARIANT, core.FULL_SAMPLE_EQUIVALENT, core.FULL_SAMPLE_DIFFERENT):
        assert not core.axis_in_decision_set(core.WITHIN_NONDETERMINISTIC, full)              # nondeterministic -> never


def test_hybrid_class_truth_table():
    assert core.hybrid_class(per(DIFF), per(DIFF)) == core.HYBRID_INCOMPATIBLE
    assert core.hybrid_class(per(EQUIV), per(DIFF)) == core.HYBRID_REPLAY_LIKE
    assert core.hybrid_class(per(DIFF), per(EQUIV)) == core.HYBRID_FRESH_LIKE
    assert core.hybrid_class(per(EQUIV), per(EQUIV)) == core.HYBRID_COMPATIBLE_BOTH
    assert core.hybrid_class(per(STATE_ONLY), per(STATE_ONLY)) == core.HYBRID_INCOMPATIBLE  # strict criterion, not "different sample"
    mixed = per(DIFF)
    mixed["p2_s9101"] = EQUIV
    assert core.hybrid_class(mixed, per(DIFF)) == core.HYBRID_MIXED
    with pytest.raises(ValueError):
        core.hybrid_class(per(DIFF), {"p0_s9101": DIFF, "p2_s9101": DIFF, "other": DIFF})


def test_t_portable():
    assert core.t_portable({12: per(DIFF), 20: per(DIFF), 28: per(DIFF)}) is None
    assert core.t_portable({12: per(DIFF), 20: per(EQUIV), 28: per(EQUIV)}) == 20
    partial = per(EQUIV)
    partial["p3_s9101"] = DIFF
    assert core.t_portable({12: per(DIFF), 20: partial, 28: per(EQUIV)}) == 28


def test_cross_engine_class():
    assert core.cross_engine_class(None) == core.CROSS_ENGINE_NOT_RUN
    assert core.cross_engine_class(EXACT) == core.CROSS_ENGINE_REPRODUCIBLE
    assert core.cross_engine_class(EQUIV) == core.CROSS_ENGINE_NONREPRODUCIBLE


# ------------------------------------------------------------------ decision truth table
def axis(within=core.WITHIN_REPRODUCIBLE, full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE, measured=True):
    return {"measured": measured, "within": within, "full": full, "hybrid_t12": hybrid}


def test_decision_strong_go_regime_crossing():
    d = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(full=core.FULL_SAMPLE_EQUIVALENT, hybrid=core.HYBRID_INCOMPATIBLE),   # F_A ~ F_X but H incompatible with both
        "ATTN_CUDNN": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(measured=False, within=None, full=None, hybrid=None)})
    assert d["decision"] == core.GO_STRONG
    assert d["decision_set_D"] == ["ATTN_CUDNN", "EAGER"] and d["tier_axes"]["regime_crossing"] == ["EAGER"]


def test_decision_go_state_not_portable():
    d = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE),
        "ATTN_CUDNN": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_REPLAY_LIKE),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(full=core.FULL_MIXED, hybrid=core.HYBRID_MIXED)})
    assert d["decision"] == core.GO and set(d["decision_set_D"]) == {"EAGER", "ATTN_CUDNN", "ATTN_FLASHINFER"}


def test_decision_weak_single_incompatible_axis():
    d = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE),
        "ATTN_CUDNN": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH)})
    assert d["decision"] == core.WEAK


def test_decision_no_go_state_portable_when_numerics_differ_but_no_crossing_incompatibility():
    # rev 0.2d: several numerically visible axes but no third semantics -> negative result, not WEAK
    d = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_REPLAY_LIKE),
        "ATTN_CUDNN": axis(full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_FRESH_LIKE),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH)})
    assert d["decision"] == core.NO_GO_PORTABLE
    one = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(full=core.FULL_SAMPLE_EQUIVALENT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_CUDNN": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH)})
    assert one["decision"] == core.NO_GO_PORTABLE


def test_decision_no_go_invariant_single_gpu():
    d = core.decide(invalid_reasons=[], axes={n: axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH) for n in core.CORE_AXES})
    assert d["decision"] == core.NO_GO_INVARIANT


def test_decision_no_go_when_only_nondeterministic_axes_differ():
    d = core.decide(invalid_reasons=[], axes={
        "EAGER": axis(within=core.WITHIN_NONDETERMINISTIC, full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE),  # cannot support a claim
        "ATTN_CUDNN": axis(within=core.WITHIN_NONDETERMINISTIC, full=core.FULL_SAMPLE_DIFFERENT, hybrid=core.HYBRID_INCOMPATIBLE),
        "OFFLOAD": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH),
        "ATTN_FLASHINFER": axis(full=core.FULL_TRAJECTORY_BIT_INVARIANT, hybrid=core.HYBRID_COMPATIBLE_BOTH)})
    assert d["decision"] == core.NO_GO and d["decision_set_D"] == []


def test_decision_invalid_dominates_and_non_core_rejected():
    d = core.decide(invalid_reasons=["anchor2: ..."], axes={"EAGER": axis()})
    assert d["decision"] == core.INVALID
    with pytest.raises(ValueError):
        core.decide(invalid_reasons=[], axes={"BATCH": axis()})


# ------------------------------------------------------------------ anchors / fingerprints
def test_anchor_failures_enumerated():
    ok3 = {t: True for t in TRAJ}
    assert core.anchor_failures(x0_bit_exact_by_axis={"EAGER": ok3}, canonical_within=core.WITHIN_REPRODUCIBLE, exact_resume_bit_exact=ok3,
                                resume_plumbing_bit_exact_by_axis={"EAGER": ok3}, fingerprint_failures=[], n_core_measured=4, gate_errors=[]) == []
    bad = dict(ok3)
    bad["p3_s9101"] = False
    fails = core.anchor_failures(x0_bit_exact_by_axis={"EAGER": bad}, canonical_within=core.WITHIN_NONDETERMINISTIC, exact_resume_bit_exact=bad,
                                 resume_plumbing_bit_exact_by_axis={"OFFLOAD": bad}, fingerprint_failures=["EAGER run p0: inductor_active recorded True expected False"], n_core_measured=1, gate_errors=["probe mismatch"])
    assert len(fails) == 7 and any(f.startswith("anchor0") for f in fails) and any("fewer than 2" in f for f in fails)


def test_fingerprint_declared_vs_recorded():
    rec = {"enforce_eager": False, "enable_cpu_offload": True, "inductor_active": True, "offload_hooks_present": True, "attention_backend_resolved": "FLASH_ATTN", "step_execution": False, "max_num_seqs": 1}
    assert core.fingerprint_check("CANONICAL", rec) == []
    assert core.fingerprint_check("EAGER", {**rec, "enforce_eager": True, "inductor_active": False}) == []
    silent_compile = core.fingerprint_check("EAGER", {**rec, "enforce_eager": True, "inductor_active": True})
    assert silent_compile and "inductor_active" in silent_compile[0]
    silent_attn_fallback = core.fingerprint_check("ATTN_FLASHINFER", rec)  # requested FLASHINFER but FLASH_ATTN resolved
    assert core.fingerprint_check("CANONICAL", {**rec, "attention_backend_resolved": "SDPA"})  # the rev0_2 stop, mirrored
    assert any("attention_backend_resolved" in d for d in silent_attn_fallback)
    assert core.fingerprint_check("OFFLOAD", {**rec, "enable_cpu_offload": False, "offload_hooks_present": False}) == []
    assert core.fingerprint_check("ATTN_SDPA", {**rec, "attention_backend_resolved": "SDPA"}) == [] and core.fingerprint_check("ATTN_SDPA", rec)
    assert core.fingerprint_check("OFFLOAD", {**rec, "enable_cpu_offload": False, "offload_hooks_present": True})
    with pytest.raises(ValueError):
        core.fingerprint_check("BATCH", rec)


def test_environment_consistency():
    base = {f: f"v_{f}" for f in core.FINGERPRINT_ENVIRONMENT_FIELDS}
    assert core.environment_consistency({"a": base, "b": dict(base)}) == []
    other = dict(base)
    other["torch"] = "2.12"
    assert core.environment_consistency({"a": base, "b": other}) == ["b: torch = '2.12' != 'v_torch' (a)"]


# ------------------------------------------------------------------ plan / cost / design freeze
def test_run_plan_counts_and_cost():
    trajs = [{"id": t} for t in TRAJ]
    plan = core.run_plan(trajs, "p0_s9101")
    labels = [p["label"] for p in plan]
    assert len(labels) == len(set(labels)) == 105
    full = [p for p in plan if p["kind"] == "FULL"]
    resume = [p for p in plan if p["kind"] == "HYBRID"]
    assert len(full) == 42 and len(resume) == 63
    assert sum(1 for p in plan if p["phase"] == "canonical") == 9
    assert sum(1 for p in plan if p["phase"] == "axis") == 90 and sum(1 for p in plan if p["phase"] == "freshcheck") == 6
    assert {p["t"] for p in resume} == {0, 12, 20, 28}
    cost = core.estimate_cost(plan)
    assert cost["engine_constructions"] == 12  # canonical, 5 axes, 6 fresh processes
    assert 6.0 < cost["hours_total"] < 7.5 and cost["storage_gb_est"] < 14
    with pytest.raises(ValueError):
        core.run_plan(trajs[:2], "p0_s9101")


def test_frozen_design_matches_config_draft():
    cfg = json.loads((REPO / "experiments/numerical_compatibility_round0/config.json").read_text())
    design = core.frozen_design()
    assert cfg["thresholds"]["same_sample_rel_l2_max"] == design["state_rel_l2_max"]
    assert cfg["thresholds"]["same_sample_frame_ssim_min"] == design["output_ssim_min"]
    assert cfg["checkpoints_t"] == design["checkpoints_t"]
    assert [t["id"] for t in cfg["trajectories"]] == list(TRAJ)
    assert set(cfg["axes"]) == set(core.CORE_AXES) | set(core.NOT_MEASURED_AXES)
    for ax in core.CORE_AXES:
        assert cfg["axes"][ax]["class"] == "core"
    for ax in core.NOT_MEASURED_AXES:
        assert cfg["axes"][ax]["class"].startswith("NOT_MEASURED")
    assert json.loads(core.canonical(design)) == design


# ------------------------------------------------------------------ runner contracts (CPU only)
import shutil  # noqa: E402

from experiments.numerical_compatibility_round0 import run as rn  # noqa: E402

CONFIG_PATH = REPO / "experiments/numerical_compatibility_round0/config.json"


def _config():
    return rn.load_config(CONFIG_PATH)


def _synthetic_scheduler_plan(config):
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40, "timesteps": timesteps,
            "checkpoint_indices": list(rn.smoke.EXPECTED_SWITCHES), "resume_indices": list(rn.smoke.EXPECTED_SWITCHES), "next_timestep_by_checkpoint": {str(s): timesteps[s] for s in rn.smoke.EXPECTED_SWITCHES}}


def test_config_freeze_and_mutations():
    cfg = _config()
    assert cfg["version"] == rn.EXPERIMENT_VERSION
    for mutate, msg in ((lambda c: c["thresholds"].__setitem__("same_sample_rel_l2_max", 0.05), "thresholds"),
                        (lambda c: c.__setitem__("checkpoints_t", [12, 20]), "checkpoints"),
                        (lambda c: c["axes"].pop("ATTN_CUDNN"), "axis set"),
                        (lambda c: c["axes"]["BATCH"].__setitem__("class", "core"), "NOT_MEASURED"),
                        (lambda c: c["trajectories"].pop(), "three"),
                        (lambda c: c["generation"].__setitem__("num_inference_steps", 30), "40 inference")):
        c = json.loads(json.dumps(cfg))
        mutate(c)
        with pytest.raises(ValueError, match=msg):
            rn.validate_config(c)


def test_output_namespace_rules():
    rn.validate_output_path(rn.DEFAULT_OUTPUT)
    for bad in (REPO / "results/regional_recompute_round0/rev4_1", REPO / "results/speculative_contamination_round0", REPO / "results/numerical_compatibility_round0/other", REPO / "elsewhere"):
        with pytest.raises(rn.GateError):
            rn.validate_output_path(bad)


def test_plan_matches_core_and_configurations():
    cfg = _config()
    plan = rn.plan_for(cfg)
    assert len(plan) == 105 and all(p["config"] in core.CONFIGURATIONS for p in plan)
    assert rn.fresh_trajectory(cfg) == "p0_s9101"
    axes = {p["config"] for p in plan if p["phase"] == "axis"}
    assert axes == set(core.CORE_AXES) and "BATCH" not in axes and "PARALLEL" not in axes
    # configurations change exactly one intended field relative to canonical
    canon = core.CONFIGURATIONS[core.CANONICAL]
    for ax in core.CORE_AXES:
        spec = core.CONFIGURATIONS[ax]
        changed = [k for k in ("enforce_eager", "enable_cpu_offload", "attention_backend_env") if spec[k] != canon[k]]
        assert changed and len(changed) == 1, (ax, changed)


def test_phase_cpu_runs():
    res = rn.phase_cpu(_config())
    assert res["runs"] == 105 and 6.0 < res["cost_estimate"]["hours_total"] < 7.5 and set(res["not_measured"]) == {"BATCH", "PARALLEL"}


def test_preregistration_seal_and_tamper(tmp_path, monkeypatch):
    cfg = _config()
    try:
        rn.smoke.scheduler_plan({**json.loads(json.dumps(cfg)), "seed": 9101, "generation": {**cfg["generation"], "switch_steps": list(rn.smoke.EXPECTED_SWITCHES)}})
    except ModuleNotFoundError:
        monkeypatch.setattr(rn.smoke, "scheduler_plan", _synthetic_scheduler_plan)
    out = rn.REPO_ROOT / "results" / rn.NAMESPACE / rn.ATTEMPT_SUBDIR / "_pytest_tmp"
    shutil.rmtree(out, ignore_errors=True)
    try:
        res = rn.phase_preregister(cfg, CONFIG_PATH, out)
        assert res["status"] == "SEALED" and res["n_runs"] == 105
        pre = json.loads((out / "preregistration.json").read_text())
        assert pre["revision"] == "0.2f" and "rev0_2_instrument_mismatch" in pre["prior_runs"] and pre["design"]["core_axes"] == list(core.CORE_AXES) and set(pre["design"]["not_measured_axes"]) == {"BATCH", "PARALLEL"}
        assert pre["design"]["decision_set_D"].startswith("core axes that are WITHIN_REPRODUCIBLE")
        assert core.GO_STRONG in pre["design"]["decision"] and "predicted to FAIL" in pre["design"]["hypotheses_non_gating"]["H2"]
        assert pre["motivation_evidence"]["status"].startswith("synthetic")
        assert len(pre["run_plan_labels"]) == 105 and "p2_s9101_ATTN_SDPA_H28" in pre["run_plan_labels"] and "p0_s9101_EAGER_H12" in pre["run_plan_labels"] and "p0_s9101_ATTN_CUDNN_FRESH" in pre["run_plan_labels"]
        assert pre["gpu_execution"].startswith("requires separate explicit user approval")
        assert rn.phase_preregister(cfg, CONFIG_PATH, out)["status"] == "ALREADY_SEALED"
        prov = rn.build_provenance(CONFIG_PATH)
        # require_sealed refuses when scientific files are uncommitted OR when the document is tampered
        doc = json.loads((out / "preregistration.json").read_text())
        doc["design"]["state_rel_l2_max"] = 0.5
        (out / "preregistration.json").write_text(json.dumps(doc))
        with pytest.raises(rn.GateError, match="modified"):
            rn.require_sealed(out, prov)
        with pytest.raises(rn.GateError, match="not sealed"):
            rn.require_sealed(out / "fresh", prov)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_engine_rejects_unknown_configuration_before_touching_gpu():
    with pytest.raises(rn.GateError, match="unknown configuration"):
        rn.Engine(_config(), "BATCH", Path("/nonexistent"), {"scheduler_plan": {"timesteps": [0.0] * 41}}, {"provenance_hash": "x"})


def test_report_writer(tmp_path):
    s = {"experiment_version": "v", "decision": {"decision": core.GO_STRONG, "rationale": "r"}, "invalid_reasons": [],
         "compatibility_matrix": {"EAGER": {"within": core.WITHIN_REPRODUCIBLE, "full": core.FULL_SAMPLE_EQUIVALENT, "hybrid_by_t": {"12": core.HYBRID_INCOMPATIBLE, "20": core.HYBRID_MIXED, "28": core.HYBRID_REPLAY_LIKE}, "t_portable": 28, "in_decision_set": True, "cross_engine": core.CROSS_ENGINE_REPRODUCIBLE}},
         "not_measured_axes": core.NOT_MEASURED_AXES, "canonical": {"within": core.WITHIN_REPRODUCIBLE, "exact_resume_anchor_bit_exact": {t: True for t in TRAJ}, "cross_engine": core.CROSS_ENGINE_REPRODUCIBLE, "cross_session_vs_rev4_1": {t: True for t in TRAJ}},
         "kill_scope": "k", "claim_boundary": "c"}
    rn.write_report(tmp_path, s)
    text = (tmp_path / "numerical_compatibility_round0.md").read_text()
    assert core.GO_STRONG in text and "| EAGER |" in text and "HYBRID_INCOMPATIBLE / HYBRID_MIXED / HYBRID_REPLAY_LIKE" in text


# ------------------------------------------------------------------ synthetic end-to-end analyze (wiring of labels, pairs, anchors, decision)
SMALL = (1, 2, 2, 8, 8)
FRAMES = (4, 16, 16, 3)


def _fp(cfg_name: str) -> dict:
    spec = core.CONFIGURATIONS[cfg_name]
    env = {"torch": "2.11", "cuda": "12.8", "cudnn": 91900, "gpu_model": "L40S", "model_revision": "rev", "dtype": "torch.bfloat16", "cublas_workspace_config": None, "tf32_matmul": False, "tf32_cudnn": True,
           "flashinfer_version": "0.4", "flash_attn_version": None, "flash_attn_provider": "fa3_fwd_interface", "parallel": {"tp": 1, "sp": 1, "ulysses": 1, "ring": 1, "cfg": 1, "pp": 1}}
    return {"enforce_eager": spec["enforce_eager"], "enable_cpu_offload": spec["enable_cpu_offload"], **spec["expect"], "step_execution": False, "max_num_seqs": 1, "batch_slot": 0, "compiled_block_count": 40 if spec["expect"]["inductor_active"] else 0, **env}


def _write_run(out: Path, item: dict, arrays: dict, video: np.ndarray, H: str) -> None:
    rd = out / rn.PHASE_DIR[item["phase"]] / item["label"]
    rd.mkdir(parents=True, exist_ok=True)
    sha = {str(k): rn.array_sha256(np.ascontiguousarray(arrays[k].astype(np.float32))) for k in sorted(arrays)}
    hashes = {}
    keep = sorted(arrays) if item["keep_all_latents"] else [item["t"], core.TOTAL_STEPS]
    for k in keep:
        np.save(rd / f"x_{k:02d}.npy", np.ascontiguousarray(arrays[k].astype(np.float32)), allow_pickle=False)
        hashes[f"x_{k:02d}.npy"] = sha[str(k)]
    vh = rn.array_sha256(video)
    if item["rep"] == 1 and item["phase"] != "freshcheck":
        np.save(rd / "video.npy", video, allow_pickle=False)
        hashes["video.npy"] = vh
    doc = {"label": item["label"], "status": "COMPLETE", "provenance_hash": H, "config": item["config"], "kind": item["kind"], "traj": item["traj"], "rep": item["rep"], "t": item["t"], "role": item.get("role", "primary"),
           "boundary_sha256": sha, "final_sha256": sha[str(core.TOTAL_STEPS)], "video_sha256": vh, "artifact_sha256": hashes, "fingerprint": _fp(item["config"]), "fingerprint_mismatches": [], "runtime_dtype": "torch.bfloat16", "wall_s": 1.0}
    (rd / "run.json").write_text(json.dumps(doc))


def _synthetic_world(out: Path, H: str, scenario: dict[str, str]) -> None:
    """scenario[axis] in {'invariant', 'equivalent_incompatible', 'different_incompatible', 'different_replay', 'final_only_exact'}."""
    cfg = _config()
    plan = rn.plan_for(cfg)
    rng = np.random.default_rng(1)
    canon = {tr: {k: rng.standard_normal(SMALL).astype(np.float32) for k in range(41)} for tr in TRAJ}
    canon_video = {tr: rng.integers(0, 255, FRAMES, dtype=np.uint8) for tr in TRAJ}

    def full(tr: str, ax: str) -> tuple[dict, np.ndarray]:
        s = scenario.get(ax, "invariant")
        if ax == core.CANONICAL or s == "invariant":
            return dict(canon[tr]), canon_video[tr]
        r = np.random.default_rng(hash((tr, ax)) % 2**32)
        if s == "final_only_exact":  # intermediates differ, final latent and video identical -> must NOT count as invariant
            return {k: (canon[tr][k] if k in (0, 40) else canon[tr][k] + 1e-3 * r.standard_normal(SMALL).astype(np.float32)) for k in range(41)}, canon_video[tr]
        if s == "equivalent_incompatible":
            return {k: (canon[tr][k] if k == 0 else canon[tr][k] + 1e-4 * r.standard_normal(SMALL).astype(np.float32)) for k in range(41)}, canon_video[tr]
        return {k: (canon[tr][k] if k == 0 else canon[tr][k] + 0.5 * r.standard_normal(SMALL).astype(np.float32)) for k in range(41)}, r.integers(0, 255, FRAMES, dtype=np.uint8)

    def hybrid(tr: str, ax: str, t: int, fx: dict, vx: np.ndarray) -> tuple[dict, np.ndarray]:
        s = scenario.get(ax, "invariant")
        if t == 0:
            return {k: fx[k] for k in range(41)}, vx  # plumbing anchor == F_X run 1
        if ax == core.CANONICAL or s in ("invariant", "different_replay", "final_only_exact"):
            return {k: canon[tr][k] for k in range(t, 41)}, canon_video[tr]  # replay-like / compatible-both
        r = np.random.default_rng((hash((tr, ax, t)) + 7) % 2**32)
        return {k: (canon[tr][k] if k == t else canon[tr][k] + 0.7 * r.standard_normal(SMALL).astype(np.float32)) for k in range(t, 41)}, r.integers(0, 255, FRAMES, dtype=np.uint8)

    for item in plan:
        fx, vx = full(item["traj"], item["config"])
        if item["kind"] == "FULL":
            _write_run(out, item, fx, vx, H)
        else:
            arrays, video = hybrid(item["traj"], item["config"], item["t"], fx, vx)
            _write_run(out, item, arrays, video, H)


def test_analyze_end_to_end_synthetic(monkeypatch):
    out = rn.REPO_ROOT / "results" / rn.NAMESPACE / rn.ATTEMPT_SUBDIR / "_pytest_analyze"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    H = "synthetic-provenance"
    monkeypatch.setattr(rn, "build_provenance", lambda p: {"provenance_hash": H, "git_commit": "x", "relevant_git_status": []})
    monkeypatch.setattr(rn, "require_sealed", lambda o, p: {"scheduler_plan": {"timesteps": [0.0] * 41}})
    (out / "preregistration.sha256").write_text("deadbeef  preregistration.json\n")
    try:
        _synthetic_world(out, H, {"EAGER": "different_incompatible", "OFFLOAD": "invariant", "ATTN_FLASHINFER": "equivalent_incompatible", "ATTN_CUDNN": "different_replay"})
        res = rn.phase_analyze(_config(), CONFIG_PATH, out)
        assert res["decision"] == core.GO_STRONG, res
        s = json.loads((out / "summary.json").read_text())
        m = s["compatibility_matrix"]
        assert m["OFFLOAD"]["full"] == core.FULL_TRAJECTORY_BIT_INVARIANT and not m["OFFLOAD"]["in_decision_set"] and m["OFFLOAD"]["hybrid_by_t"]["12"] == core.HYBRID_COMPATIBLE_BOTH
        assert m["ATTN_FLASHINFER"]["full"] == core.FULL_SAMPLE_EQUIVALENT and m["ATTN_FLASHINFER"]["hybrid_by_t"]["12"] == core.HYBRID_INCOMPATIBLE
        assert m["EAGER"]["full"] == core.FULL_SAMPLE_DIFFERENT and m["EAGER"]["hybrid_by_t"]["12"] == core.HYBRID_INCOMPATIBLE and m["EAGER"]["t_portable"] is None
        assert m["ATTN_CUDNN"]["full"] == core.FULL_SAMPLE_DIFFERENT and m["ATTN_CUDNN"]["hybrid_by_t"]["12"] == core.HYBRID_REPLAY_LIKE and m["ATTN_CUDNN"]["t_portable"] == 12
        assert all(m[ax]["cross_engine"] == core.CROSS_ENGINE_REPRODUCIBLE for ax in core.CORE_AXES) and all(m[ax]["within"] == core.WITHIN_REPRODUCIBLE for ax in core.CORE_AXES)
        assert s["decision"]["decision_set_D"] == ["ATTN_CUDNN", "ATTN_FLASHINFER", "EAGER"] and s["decision"]["tier_axes"]["regime_crossing"] == ["ATTN_FLASHINFER"]
        assert s["canonical"]["within"] == core.WITHIN_REPRODUCIBLE and all(s["canonical"]["exact_resume_anchor_bit_exact"].values())
        assert all(all(v.values()) for v in s["x0_bit_exact_by_axis"].values()) and all(all(v.values()) for v in s["resume_plumbing_by_axis"].values())
        assert s["hypotheses"]["H3"]["EAGER"] == {t: True for t in TRAJ}
        assert (out / "decision.json").exists() and (out / "numerical_compatibility_round0.md").exists()
        with pytest.raises(rn.GateError, match="exactly once"):
            rn.phase_analyze(_config(), CONFIG_PATH, out)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_analyze_invalid_on_fingerprint_mismatch_and_broken_anchor(monkeypatch):
    out = rn.REPO_ROOT / "results" / rn.NAMESPACE / rn.ATTEMPT_SUBDIR / "_pytest_analyze_invalid"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    H = "synthetic-provenance"
    monkeypatch.setattr(rn, "build_provenance", lambda p: {"provenance_hash": H, "git_commit": "x", "relevant_git_status": []})
    monkeypatch.setattr(rn, "require_sealed", lambda o, p: {"scheduler_plan": {"timesteps": [0.0] * 41}})
    (out / "preregistration.sha256").write_text("deadbeef  preregistration.json\n")
    try:
        _synthetic_world(out, H, {"EAGER": "different_incompatible", "OFFLOAD": "invariant", "ATTN_FLASHINFER": "different_incompatible", "ATTN_CUDNN": "invariant"})
        # silent attention fallback recorded on one ATTN_FLASHINFER run
        p = out / "axis" / "p2_s9101_ATTN_FLASHINFER_F1" / "run.json"
        d = json.loads(p.read_text()); d["fingerprint_mismatches"] = ["attention_backend_resolved recorded 'SDPA' expected 'FLASHINFER_ATTN'"]; p.write_text(json.dumps(d))
        # broken exact-resume anchor
        p = out / "canonical" / "p0_s9101_CANONICAL_H12" / "run.json"
        d = json.loads(p.read_text()); d["final_sha256"] = "0" * 64; p.write_text(json.dumps(d))
        res = rn.phase_analyze(_config(), CONFIG_PATH, out)
        assert res["decision"] == core.INVALID
        assert any(r.startswith("anchor2") for r in res["invalid_reasons"]) and any("fingerprint" in r for r in res["invalid_reasons"])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_analyze_final_only_exact_axis_enters_D_and_portable_is_no_go(monkeypatch):
    out = rn.REPO_ROOT / "results" / rn.NAMESPACE / rn.ATTEMPT_SUBDIR / "_pytest_analyze_portable"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    H = "synthetic-provenance"
    monkeypatch.setattr(rn, "build_provenance", lambda p: {"provenance_hash": H, "git_commit": "x", "relevant_git_status": []})
    monkeypatch.setattr(rn, "require_sealed", lambda o, p: {"scheduler_plan": {"timesteps": [0.0] * 41}})
    (out / "preregistration.sha256").write_text("deadbeef  preregistration.json\n")
    try:
        _synthetic_world(out, H, {"EAGER": "final_only_exact", "OFFLOAD": "invariant", "ATTN_FLASHINFER": "different_replay", "ATTN_CUDNN": "invariant"})
        res = rn.phase_analyze(_config(), CONFIG_PATH, out)
        s = json.loads((out / "summary.json").read_text())
        m = s["compatibility_matrix"]
        assert m["EAGER"]["full"] == core.FULL_SAMPLE_EQUIVALENT and m["EAGER"]["in_decision_set"]  # final-only equality is not invariance
        assert m["EAGER"]["hybrid_by_t"]["12"] == core.HYBRID_COMPATIBLE_BOTH
        assert m["ATTN_FLASHINFER"]["full"] == core.FULL_SAMPLE_DIFFERENT and m["ATTN_FLASHINFER"]["hybrid_by_t"]["12"] == core.HYBRID_REPLAY_LIKE
        assert set(s["decision"]["decision_set_D"]) == {"EAGER", "ATTN_FLASHINFER"}
        assert res["decision"] == core.NO_GO_PORTABLE
    finally:
        shutil.rmtree(out, ignore_errors=True)
