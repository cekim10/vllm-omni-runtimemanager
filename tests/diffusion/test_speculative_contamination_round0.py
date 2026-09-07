"""CPU / synthetic contracts for the speculative-contamination Round-0 experiment, revision 3 (L = 4 / d = 4). No GPU, no model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments.speculative_contamination_round0 import core
from experiments.speculative_contamination_round0 import run as r0

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/speculative_contamination_round0/config.json")


def _config():
    return r0.load_config(CONFIG_PATH)


def _latent(seed=0):
    return core.bf16_round(np.random.default_rng(seed).standard_normal(core.LATENT_SHAPE).astype(np.float32))


def _trajectory(seed=0, n=30):
    rng = np.random.default_rng(seed)
    sig = r0.synthetic_sigmas()
    x0, eps = rng.standard_normal(core.LATENT_SHAPE).astype(np.float32), rng.standard_normal(core.LATENT_SHAPE).astype(np.float32)
    return sig, {k: core.bf16_round(((1 - sig[k]) * x0 + sig[k] * eps + 0.02 * np.sin(k / 3.0) * rng.standard_normal(core.LATENT_SHAPE)).astype(np.float32)) for k in range(n)}


# ------------------------------------------------------------------ freeze
def test_frozen_config_and_constants():
    c = _config()
    assert tuple(c["checkpoints"]) == (12, 20, 28) and tuple(c["seeds"]) == (9101, 9202) and c["decision_seed"] == 9101
    assert core.STALE_WINDOW_PRIMARY == 4 and core.STALE_WINDOW_SECONDARY == 1 and core.REPAIR_OFFSET == 4 and c["repair_offset"] == 4
    assert core.ERROR_MODELS == ("STALE_VELOCITY", "STALE_VELOCITY_W1", "GAUSS_HIGH") and c["matrix"]["secondary_error_models"] == ["STALE_VELOCITY_W1", "GAUSS_HIGH"]
    assert "gauss_high_oracle_domains" not in c["matrix"]
    assert c["thresholds"] == core.frozen_thresholds() and c["quarantine"] == "DEFERRED_TO_ROUND1"
    assert (core.RCOMPUTE_STRONG_GO, core.RCOMPUTE_GO, core.RCOMPUTE_WEAK) == (0.20, 0.40, 0.70) and core.MATERIAL_CELL_FRACTION_MIN == 0.5
    assert c["recompute_arm_label"] == r0.RECOMPUTE_LABEL == "FULL-FORWARD SELECTIVE-STATE RECOMPUTATION with cached contaminated context"
    for phrase, key in (("full 14,040-token target forward", "recompute_arm_statement"), ("NOT demonstrate sparse GPU compute", "recompute_arm_statement"), ("NOT counted", "verification_semantics"), ("L = 4", "kill_scope"), ("does NOT claim", "related_work_and_problem_statement"), ("does NOT establish", "headroom_disclaimer"), ("L = 4", "primary_question")):
        assert phrase in c[key], (phrase, key)
    assert set(c["cost_semantics"]) == {"GLOBAL_DENSE_REFRESH", "FULL_ROLLBACK", "SELECTIVE_STATE_RECOMPUTE(D)", "ceiling_note"} and "NO measured sparse-compute speedup" in c["cost_semantics"]["SELECTIVE_STATE_RECOMPUTE(D)"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [9101, 9203]),
        lambda c: c.__setitem__("decision_seed", 9202),
        lambda c: c.__setitem__("checkpoints", [12, 20, 30]),
        lambda c: c.__setitem__("repair_offset", 2),
        lambda c: c["prompts"].pop(),
        lambda c: c["thresholds"].__setitem__("stale_window_primary", 1),
        lambda c: c["thresholds"].__setitem__("rcompute_go", 0.5),
        lambda c: c["thresholds"].__setitem__("material_cell_fraction_min", 0.1),
        lambda c: c["matrix"].__setitem__("decision_error_model", "GAUSS_HIGH"),
        lambda c: c["matrix"].__setitem__("secondary_error_models", ["STALE_VELOCITY_W4", "GAUSS_HIGH"]),
        lambda c: c["matrix"].__setitem__("gauss_high_oracle_domains", ["R"]),
        lambda c: c["matrix"]["weak_band_characterization_domains"]["SPATIAL_SMALL"].append("65"),
        lambda c: c["matrix"]["recovery_domains"]["SPATIAL_SMALL"].append("10"),
        lambda c: c.__setitem__("recompute_arm_label", "approximate selective recomputation"),
        lambda c: c.pop("verification_semantics"),
        lambda c: c.pop("kill_scope"),
        lambda c: c.__setitem__("quarantine", "RUN"),
        lambda c: c["scheduler"].__setitem__("sample_solver", "unipc"),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    c = json.loads(CONFIG_PATH.read_text())
    mutate(c)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        r0.load_config(p)


def test_output_namespace_isolation():
    r0.validate_output_path(r0.REPO_ROOT / "results" / r0.NAMESPACE)
    for bad in ("video_trajectory_fork_confirmatory", "other"):
        with pytest.raises(ValueError):
            r0.validate_output_path(r0.REPO_ROOT / "results" / bad)


# ------------------------------------------------------------------ geometry
def test_token_geometry_recovery_and_weak_domains():
    g = core.geometry_document()
    assert core.TOKEN_GRID == (9, 30, 52) and core.N_TOKENS == 14040
    sd = {k: v["fraction"] for k, v in g["recovery_domains"]["SPATIAL_SMALL"].items()}
    assert set(sd) == {"R", "20", "40", "70", "100"} and abs(sd["20"] - 0.205) < 0.001 and abs(sd["40"] - 0.40) < 0.001 and abs(sd["70"] - 0.70) < 0.001
    td = {k: v["fraction"] for k, v in g["recovery_domains"]["TEMPORAL_MEDIUM"].items()}
    assert set(td) == {"R", "22", "44", "67", "100"}
    wd = g["weak_band_characterization_domains"]
    assert abs(wd["SPATIAL_SMALL"]["50"]["fraction"] - 0.50) < 0.001 and abs(wd["SPATIAL_SMALL"]["60"]["fraction"] - 0.60) < 0.001 and abs(wd["TEMPORAL_MEDIUM"]["56"]["fraction"] - 5 / 9) < 1e-9
    # weak domains are nested strictly between the 40 and 70 grid points (validated inside geometry_document)
    m40, m50, m60, m70 = (core.RECOVERY_DOMAINS["SPATIAL_SMALL"]["40"].token_mask(), core.WEAK_BAND_DOMAINS["SPATIAL_SMALL"]["50"].token_mask(), core.WEAK_BAND_DOMAINS["SPATIAL_SMALL"]["60"].token_mask(), core.RECOVERY_DOMAINS["SPATIAL_SMALL"]["70"].token_mask())
    assert (m40 & ~m50).sum() == 0 and (m50 & ~m60).sum() == 0 and (m60 & ~m70).sum() == 0


# ------------------------------------------------------------------ scheduler algebra / STALE_VELOCITY
def test_scheduler_algebra_identities():
    sig, traj = _trajectory()
    assert np.all(np.diff(sig) < 0)
    for k in (7, 11, 19, 27):
        assert core.delta_sigma(sig, k) < 0
        v = core.implied_velocity(traj[k], traj[k + 1], sig, k)
        assert np.abs(core.euler_step(traj[k], v, sig, k) - traj[k + 1]).max() < 1e-5


def test_stale_velocity_primary_window_matches_hand_derivation():
    sig, traj = _trajectory()
    t, L = 12, core.STALE_WINDOW_PRIMARY
    xs, st = core.stale_velocity_state(x_last_active_prev=traj[t - L - 1], x_last_active=traj[t - L], x_t=traj[t], sigmas=sig, t=t, window=L, region="SPATIAL_SMALL")
    mask = core.token_to_latent_mask(core.REGIONS["SPATIAL_SMALL"].token_mask())
    hand = traj[t - L].astype(np.float64) + (sig[t] - sig[t - L]) * (traj[t - L].astype(np.float64) - traj[t - L - 1].astype(np.float64)) / (sig[t - L] - sig[t - L - 1])
    assert np.array_equal(xs[mask], core.bf16_round(hand.astype(np.float32))[mask]) and np.array_equal(xs[~mask], traj[t][~mask])
    assert st["error_model"] == "STALE_VELOCITY" and st["window"] == 4 and st["last_active_step"] == t - 5 and st["total_delta_sigma_window"] == pytest.approx(sig[t] - sig[t - 4])
    # L = 1 is the descriptive arm and is smaller than L = 4 on the same trajectory (error grows with the stale window)
    x1, s1 = core.stale_velocity_state(x_last_active_prev=traj[t - 2], x_last_active=traj[t - 1], x_t=traj[t], sigmas=sig, t=t, window=1, region="SPATIAL_SMALL")
    assert s1["error_model"] == "STALE_VELOCITY_W1" and s1["rms_in_region"] < st["rms_in_region"]
    # no look-ahead: only x_{t-5}, x_{t-4} and the schedule enter R; x_t enters only outside R
    x_alt = dict(traj)
    x_alt[t] = traj[t] + np.float32(1.0)
    xs2, _ = core.stale_velocity_state(x_last_active_prev=traj[t - 5], x_last_active=traj[t - 4], x_t=x_alt[t], sigmas=sig, t=t, window=4, region="SPATIAL_SMALL")
    assert np.array_equal(xs2[mask], xs[mask])
    with pytest.raises(ValueError):
        core.stale_velocity_state(x_last_active_prev=traj[0], x_last_active=traj[1], x_t=traj[4], sigmas=sig, t=4, window=4, region="SPATIAL_SMALL")


def test_velocity_validation_pass_and_fail():
    sig, traj = _trajectory()
    t = 20
    v = core.implied_velocity(traj[t - 1], traj[t], sig, t - 1).astype(np.float32)
    within = {"guidance_combined_output": v, "scheduler_input": traj[t - 1], "scheduler_output": traj[t], "timestep": float(sig[t - 1] * 1000), "runtime_dtype": "torch.bfloat16"}
    assert r0.velocity_validation(traj[t - 1], traj[t], within, traj[t], sig, t)["pass"]
    assert not r0.velocity_validation(traj[t - 1], traj[t], {**within, "guidance_combined_output": -v}, traj[t], sig, t)["pass"]
    assert not r0.velocity_validation(traj[t - 1], traj[t], {**within, "scheduler_output": traj[t - 1]}, traj[t], sig, t)["pass"]


def test_gaussian_stress_control_is_local_and_deterministic():
    x = _latent()
    a, sa = core.gaussian_state(x, trajectory_id="p0", t=12, region="TEMPORAL_MEDIUM")
    b, _ = core.gaussian_state(x, trajectory_id="p0", t=12, region="TEMPORAL_MEDIUM")
    assert np.array_equal(a, b) and abs(sa["rms_in_region_over_mean_channel_std"] - 0.30) < 0.02 and not sa["outside_region_changed"]


# ------------------------------------------------------------------ contamination + repair operators
def test_contamination_and_repair_operators():
    x = _latent()
    xp, _ = core.gaussian_state(x, trajectory_id="a", t=12, region="SPATIAL_SMALL")
    m = core.contamination_metrics(xp, x, "SPATIAL_SMALL")
    assert m["material_fraction_inside_region"] == 1.0 and m["material_fraction_outside_region"] == 0.0
    sigma = core.channel_std(x)[None, :, None, None, None]
    assert core.contamination_metrics(x + (0.005 * sigma * np.random.default_rng(3).standard_normal(core.LATENT_SHAPE)).astype(np.float32), x, "SPATIAL_SMALL")["material_fraction"] == 0.0
    R = core.REGIONS["SPATIAL_SMALL"].token_mask()
    assert np.array_equal(core.oracle_repair(xp, x, R), x)
    D = core.RECOVERY_DOMAINS["SPATIAL_SMALL"]["40"].token_mask()
    dom, ctx = _latent(1), _latent(2)
    lm = core.token_to_latent_mask(D)
    inp = core.compose_recompute_input(D, dom, ctx)
    assert np.array_equal(inp[lm], dom[lm]) and np.array_equal(inp[~lm], ctx[~lm])
    out = core.compose_recompute_output(D, _latent(4), dom)
    assert np.array_equal(out[~lm], dom[~lm])
    assert np.array_equal(core.compose_recompute_input(core.FULL_DOMAIN.token_mask(), dom, ctx), dom)


def test_idealized_window_saving_is_reported_separately_from_executed_work():
    s = core.idealized_window_saving(core.RECOVERY_DOMAINS["SPATIAL_SMALL"]["20"].fraction)
    assert s["full_rollback_forwards"] == 4 and s["experimental_forwards_executed"] == 4 and s["idealized_selective_forwards"] == pytest.approx(4 * 0.2051, abs=1e-3) and s["idealized_saved_forwards"] == pytest.approx(4 * (1 - 0.2051), abs=1e-3)
    assert core.idealized_window_saving(1.0)["idealized_saved_forwards"] == 0.0


# ------------------------------------------------------------------ convergence, R*, decision, early stop
def test_convergence_r_star_and_decision_bands():
    x = _latent(5)
    assert core.final_convergence(x, x, frame_ssim_mean=1.0)["recovered"] and not core.final_convergence(x, x, frame_ssim_mean=0.9)["recovered"]
    fr = {"R": 0.0385, "20": 0.205, "40": 0.40, "70": 0.70, "100": 1.0}
    sweep = {"R": {"recovered": False}, "20": {"recovered": True}, "40": {"recovered": True}, "70": {"recovered": True}, "100": {"recovered": True}}
    assert core.r_star(sweep, fr)["r_star"] == 0.205
    sweep["40"]["recovered"] = False
    assert core.r_star(sweep, fr) == {**core.r_star(sweep, fr), "r_star": 0.70, "monotonicity_violations": 1}
    small = {"SPATIAL_SMALL": 0.0385, "TEMPORAL_MEDIUM": 0.111}
    assert core.decide(valid=False, invalid_reason="x", material_cell_fraction=1.0, rcompute_median_by_region=small, rstate_median_by_region=small)["decision"] == "INVALID_EXPERIMENT"
    assert core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.49, rcompute_median_by_region=small, rstate_median_by_region=small)["decision"] == "STOP_NO_RECOVERY_PROBLEM"
    assert core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.5, rcompute_median_by_region={"SPATIAL_SMALL": 0.20, "TEMPORAL_MEDIUM": 0.111}, rstate_median_by_region=small)["decision"] == "STRONG_GO_DEPENDENCY_BOUNDED_RECOMPUTE"
    assert core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 0.205, "TEMPORAL_MEDIUM": 0.40}, rstate_median_by_region={})["decision"] == "GO_DEPENDENCY_BOUNDED_RECOMPUTE"
    assert core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 0.401, "TEMPORAL_MEDIUM": 0.222}, rstate_median_by_region={})["decision"].startswith("WEAK")
    d8 = core.decide(valid=True, invalid_reason=None, material_cell_fraction=0.9, rcompute_median_by_region={"SPATIAL_SMALL": 1.0, "TEMPORAL_MEDIUM": 0.667}, rstate_median_by_region={"SPATIAL_SMALL": 0.205, "TEMPORAL_MEDIUM": 0.222})
    assert d8["decision"] == "NO_GO_GLOBAL_DENSE_REFRESH_REQUIRED" and d8["oracle_only_headroom"] is True


def test_early_stop_logic():
    stop = r0.early_stop_decision(material_fraction=0.25, velocity_ok=True, zero_failures=[], have_sweeps=False)
    assert stop["decision"] == "STOP_NO_RECOVERY_PROBLEM"
    inval = r0.early_stop_decision(material_fraction=0.9, velocity_ok=False, zero_failures=[], have_sweeps=True)
    assert inval["decision"] == "INVALID_EXPERIMENT"
    assert r0.early_stop_decision(material_fraction=0.9, velocity_ok=True, zero_failures=["p0_s9101_t12_ZERO"], have_sweeps=True)["decision"] == "INVALID_EXPERIMENT"
    assert r0.early_stop_decision(material_fraction=0.9, velocity_ok=True, zero_failures=[], have_sweeps=True) is None
    with pytest.raises(r0.GateError, match="run --phase oracle"):
        r0.early_stop_decision(material_fraction=0.9, velocity_ok=True, zero_failures=[], have_sweeps=False)
    assert r0.early_stop_decision(material_fraction=None, velocity_ok=True, zero_failures=[], have_sweeps=False)["decision"] == "STOP_NO_RECOVERY_PROBLEM"


# ------------------------------------------------------------------ plan / cost
def test_run_plan_counts_labels_and_cost():
    c = _config()
    plan = r0.run_plan(c)
    by = {}
    for p in plan:
        by[p["phase"]] = by.get(p["phase"], 0) + 1
    # reference 8; propagation 8x3x(ZERO+2 STALE) + 4x3x(2 W1 + 2 GAUSS); oracle 4x3x(FULL + 4 + 4); recompute 4x3x(FULL + 4 + 4)
    assert by == {"reference": 8, "propagation": 72 + 48, "oracle": 4 * 3 * 9, "recompute": 4 * 3 * 9}
    assert len(plan) == 8 + 120 + 108 + 108 == 344 and len({p["label"] for p in plan}) == 344
    labels = {p["label"] for p in plan}
    assert "p0_s9101_t20_SPATIAL_SMALL_STALE_VELOCITY_d4_RECOMPUTE40" in labels and "p0_s9101_t20_STALE_VELOCITY_d4_RECOMPUTE100" in labels and "p0_s9101_t20_STALE_VELOCITY_d4_ORACLE100" in labels
    assert "p3_s9202_t28_TEMPORAL_MEDIUM_STALE_VELOCITY_noRepair" in labels and "p1_s9101_t12_SPATIAL_SMALL_STALE_VELOCITY_W1_noRepair" in labels
    assert not any("GAUSS" in p["label"] and p["phase"] != "propagation" for p in plan)
    assert not any(p["traj"].endswith("9202") for p in plan if p["phase"] in ("oracle", "recompute"))
    assert all(p["start"] == p["t"] + 4 for p in plan if p["phase"] in ("oracle", "recompute")) and all(p.get("single_steps") == 4 for p in plan if p["phase"] == "recompute")
    est = r0.estimate_cost(plan)
    assert est["continuations"] == 344 and est["single_step_requests"] == 8 * 3 + 108 * 4 == 456 and 10 < est["hours_total"] < 15 and est["storage_gb_est"] < 10
    weak = r0.weak_band_plan(c)
    assert len(weak) == 4 * 3 * 3 and all(p["kind"] == "WEAK_RECOMPUTE" for p in weak) and not (set(p["label"] for p in weak) & labels)
    assert sorted(r0.reference_capture_steps()) == [7, 8, 10, 11, 12, 15, 16, 18, 19, 20, 23, 24, 26, 27, 28, 40]


# ------------------------------------------------------------------ preregistration
def _synthetic_scheduler_plan(config):
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40, "timesteps": timesteps,
            "checkpoint_indices": list(r0.smoke.EXPECTED_SWITCHES), "resume_indices": list(r0.smoke.EXPECTED_SWITCHES), "next_timestep_by_checkpoint": {str(s): timesteps[s] for s in r0.smoke.EXPECTED_SWITCHES}}


def test_preregistration_seal_and_gpu_refusal(tmp_path, monkeypatch):
    config = _config()
    try:
        r0.smoke.scheduler_plan({**json.loads(json.dumps(config)), "seed": 9101, "generation": {**config["generation"], "switch_steps": list(r0.smoke.EXPECTED_SWITCHES)}})
    except ModuleNotFoundError:
        monkeypatch.setattr(r0.smoke, "scheduler_plan", _synthetic_scheduler_plan)
    out = r0.REPO_ROOT / "results" / r0.NAMESPACE / "_pytest_tmp"
    shutil.rmtree(out, ignore_errors=True)
    try:
        res = r0.phase_preregister(config, CONFIG_PATH, out)
        assert res["status"] == "SEALED" and res["n_continuations"] == 344
        prereg = json.loads((out / "preregistration.json").read_text())
        assert prereg["definitions"]["L"].endswith("4") and prereg["error_models"]["STALE_VELOCITY"]["window"] == 4 and prereg["error_models"]["STALE_VELOCITY_W1"]["role"].startswith("descriptive")
        assert prereg["recompute_arm_label"] == r0.RECOMPUTE_LABEL and "full 14,040-token" in prereg["recompute_arm_statement"] and "NOT counted" in prereg["verification_semantics"]
        assert set(prereg["cost_semantics"]) >= {"GLOBAL_DENSE_REFRESH", "FULL_ROLLBACK", "SELECTIVE_STATE_RECOMPUTE(D)"} and "L = 4" in prereg["kill_scope"]
        assert prereg["weak_band_characterization"]["domains"] == {"SPATIAL_SMALL": {"50": 0.5, "60": 0.6}, "TEMPORAL_MEDIUM": {"56": 5 / 9}} and len(prereg["weak_band_characterization"]["labels"]) == 36
        assert "early stop" in prereg["decision_logic"]["STOP_NO_RECOVERY_PROBLEM"] and "weak-band characterization" in prereg["decision_logic"]["never_decides"]
        assert prereg["idealized_window_saving_by_domain"]["SPATIAL_SMALL"]["20"]["experimental_forwards_executed"] == 4
        prov = r0.build_provenance(CONFIG_PATH)
        path = out / "preregistration.json"
        doc = json.loads(path.read_text()); doc["thresholds"]["rcompute_go"] = 0.9
        path.write_text(json.dumps(doc))
        with pytest.raises(r0.GateError, match="modified"):
            r0.require_sealed(out, prov)
        with pytest.raises(r0.GateError, match="not sealed"):
            r0.require_sealed(out / "fresh", prov)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_cpu_phase_demonstration_runs():
    demo = r0.phase_cpu(_config())["synthetic"]
    assert demo["euler_identity_max_abs"] < 1e-5
    for t in (12, 20, 28):
        assert not demo[f"stale_velocity_L4_t{t}"]["outside_changed"] and demo[f"stale_velocity_L1_t{t}"]["rms_over_std"] < demo[f"stale_velocity_L4_t{t}"]["rms_over_std"]
    assert demo["decision_examples"]["STOP"] == "STOP_NO_RECOVERY_PROBLEM" and demo["decision_examples"]["NO_GO_item8"]["oracle_only_headroom"] is True
    assert demo["plan"]["continuations"] == 344 and demo["weak_band_plan"]["continuations"] == 36
