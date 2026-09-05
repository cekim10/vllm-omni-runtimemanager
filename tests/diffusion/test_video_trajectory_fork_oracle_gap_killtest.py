"""CPU contracts for Round 4B (static-K oracle vs trajectory-specific oracle gap). No GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import video_trajectory_fork_oracle_gap_killtest as og

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_trajectory_fork_oracle_gap_killtest_config.yaml")


def _config():
    return og.load_config(CONFIG_PATH)


def _matrix(rows: dict[int, str]) -> dict[int, dict[int, str]]:
    """rows: seed -> 5-char string over {N,D,B,O,R} for K=12,15,18,21,24."""
    code = {"N": "NEW", "D": "MIXED_NEW_DOMINANT", "B": "MIXED_BALANCED", "O": "MIXED_OLD_DOMINANT", "R": "OLD"}
    return {seed: {k: code[c] for k, c in zip(og.KS, s, strict=True)} for seed, s in rows.items()}


# ------------------------------------------------------------------ freeze
def test_frozen_design():
    config = _config()
    assert tuple(config["seeds"]) == og.SEEDS and len(og.SEEDS) == 10 and len(set(og.SEEDS)) == 10
    assert not set(og.SEEDS) & set(og.EXCLUDED_SEEDS) and set(og.EXCLUDED_SEEDS) == {1234, 2345, 3456, 4567}
    assert og.KS == (12, 15, 18, 21, 24) and og.CONTROL_K == 18 and og.CONTROL_K in og.KS
    assert og.SUCCESS_LABELS == ("NEW", "MIXED_NEW_DOMINANT") and og.TAU_PRIMARY == 1.0 and og.TAU_DESCRIPTIVE == 0.9
    assert og.NO_GO_MAX_GAP == 0.09 == og.PROBE_PAIR_COST_FRACTION and og.GO_MIN_GAP == 0.15 and og.MIN_INFORMATIVE_SEEDS == 8
    assert og.BASELINE_CAPTURE == [0, 12, 15, 18, 21, 24, 40]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [1234, 6789, 7890, 8901, 9012, 10123, 11234, 12345, 13456, 14567]),
        lambda c: c.__setitem__("seeds", [5678, 6789, 7890, 8901, 9012, 10123, 11234, 12345, 13456]),
        lambda c: c.__setitem__("k_grid", [12, 15, 18, 21, 24, 27]),
        lambda c: c.__setitem__("control_k", 12),
        lambda c: c["oracle"].__setitem__("success_labels", ["NEW", "MIXED_NEW_DOMINANT", "MIXED_BALANCED"]),
        lambda c: c["oracle"].__setitem__("tau_primary", 0.9),
        lambda c: c["gate"].__setitem__("no_go_max_gap", 0.05),
        lambda c: c["gate"].__setitem__("go_min_gap", 0.10),
        lambda c: c["gate"].__setitem__("min_informative_seeds", 5),
        lambda c: c["prompts"].__setitem__("new", "A green sports car driving on a snowy road, cinematic video"),
        lambda c: c["labels"]["ordinal"].__setitem__("MIXED_BALANCED", 1),
        lambda c: c["generation"].__setitem__("guidance_scale", 5.0),
        lambda c: c["scheduler"].__setitem__("sample_solver", "unipc"),
        lambda c: c.__setitem__("blinded_review", False),
        lambda c: c.__setitem__("experiment_version", "video-trajectory-fork-oracle-gap-killtest-v2"),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        og.load_config(path)


def test_output_namespace_isolation():
    og.validate_output_path(og.REPO_ROOT / "results" / og.NAMESPACE)
    og.validate_output_path(og.REPO_ROOT / "results" / og.NAMESPACE / "rerun")
    for bad in ("video_trajectory_fork_confirmatory", "video_trajectory_fork_probe_killtest", "video_trajectory_fork_killtest", "video_runtime_state_discovery_v3_corrected", "other"):
        with pytest.raises(ValueError):
            og.validate_output_path(og.REPO_ROOT / "results" / bad)
    with pytest.raises(ValueError):
        og.validate_output_path(og.REPO_ROOT / "results" / og.NAMESPACE / "video_trajectory_fork_confirmatory")


# ------------------------------------------------------------------ key set
def test_expected_key_set():
    full = og.expected_keys()
    assert len(full) == 10 * (2 + 1 + 5) == 80
    assert (5678, "same_condition", 18) in full and (5678, "same_condition", 12) not in full
    partial = og.expected_keys([5678, 6789, 7890, 8901, 9012, 10123, 11234, 12345])
    assert len(partial) == 20 + 8 * 6
    assert (13456, "old_baseline", -1) in partial and (13456, "fork_new", 12) not in partial
    rows = [{"seed": s, "trajectory_type": t, "switch_step": k} for (s, t, k) in full]
    og.validate_key_set(rows, full)
    with pytest.raises(og.GateError, match="duplicates"):
        og.validate_key_set(rows + [rows[0]], full)
    with pytest.raises(og.GateError, match="missing"):
        og.validate_key_set(rows[:-1], full)


# ------------------------------------------------------------------ oracle definitions
def test_success_set_and_unknown_label():
    assert og.success("NEW") and og.success("MIXED_NEW_DOMINANT")
    assert not og.success("MIXED_BALANCED") and not og.success("MIXED_OLD_DOMINANT") and not og.success("OLD")
    with pytest.raises(og.GateError):
        og.success("new")


@pytest.mark.parametrize(
    "seq,k_star,violations",
    [
        ("NNNNN", 24, 0),
        ("NNDBR", 18, 0),
        ("NBNNN", 12, 3),  # non-monotone: success after failure never rescues K*
        ("BNNNN", 0, 4),
        ("RRRRR", 0, 0),
        ("DDDDB", 21, 0),
        ("NNBRO", 15, 0),
    ],
)
def test_trajectory_oracle_monotone_cut(seq, k_star, violations):
    result = og.trajectory_oracle_k(_matrix({1: seq})[1])
    assert result["k_star"] == k_star and result["monotonicity_violations"] == violations


def test_static_oracle_tau():
    m = _matrix({1: "NNNNB", 2: "NNNBB", 3: "NNNNN", 4: "NNNNN", 5: "NNNNN", 6: "NNNNN", 7: "NNNNN", 8: "NNNNN", 9: "NNNNN", 10: "NNNNN"})
    strict = og.static_oracle_k(m, 1.0)
    assert strict["k_static"] == 18 and strict["success_rate_by_k"]["21"] == 0.9 and strict["success_rate_by_k"]["24"] == 0.8
    loose = og.static_oracle_k(m, 0.9)
    assert loose["k_static"] == 21 and loose["failing_seeds_at_k_static"] == [2]
    assert og.static_oracle_k(_matrix({1: "BBBBB", 2: "NNNNN"}), 1.0)["k_static"] == 0


def test_gap_formula_and_descriptives():
    # every seed identical -> static oracle captures everything -> G = 0
    same = _matrix({s: "NNNBB" for s in range(1, 9)})
    g = og.oracle_gap(same, 1.0)
    assert g["k_static"] == 18 and g["gap_fraction_of_generation"] == 0.0 and g["seeds_with_k_star_above_static"] == 0
    # spread: K* = 24 x4, 12 x4 -> K_static = 12, mean diff = 6 -> G = 0.15
    spread = _matrix({**{s: "NNNNN" for s in range(1, 5)}, **{s: "NBBBB" for s in range(5, 9)}})
    g = og.oracle_gap(spread, 1.0)
    assert g["k_static"] == 12 and g["mean_k_star"] == 18.0
    assert g["gap_fraction_of_generation"] == pytest.approx(6 / 40) and g["gap_relative_to_static_recompute"] == pytest.approx(6 / 28)
    assert g["seeds_with_k_star_above_static"] == 4 and g["seeds_with_k_star_below_static"] == 0
    # non-monotone seed can sit below K_static and the gap term goes negative (kept, not clipped)
    nm = _matrix({**{s: "NNNNN" for s in range(1, 8)}, 8: "NBNNN"})
    g = og.oracle_gap(nm, 1.0)
    assert g["k_static"] == 24 and g["k_star_by_seed"]["8"] == 12 and g["seeds_with_k_star_below_static"] == 1
    assert g["gap_fraction_of_generation"] == pytest.approx(-12 / 8 / 40) and g["monotonicity_violations_total"] == 3


def test_decision_thresholds_and_boundaries():
    assert og.decide(valid=True, invalid_reason=None, gap=0.0)["decision"] == "NO-GO"
    assert og.decide(valid=True, invalid_reason=None, gap=0.09)["decision"] == "NO-GO"
    assert og.decide(valid=True, invalid_reason=None, gap=0.0901)["decision"] == "WEAK"
    assert og.decide(valid=True, invalid_reason=None, gap=0.1499)["decision"] == "WEAK"
    go = og.decide(valid=True, invalid_reason=None, gap=0.15)
    assert go["decision"] == "GO" and go["MECHANISM_STAGE_ELIGIBLE"] is True
    assert og.decide(valid=True, invalid_reason=None, gap=-0.05)["decision"] == "NO-GO"
    invalid = og.decide(valid=False, invalid_reason="7 informative", gap=0.5)
    assert invalid["decision"] == "INVALID" and invalid["MECHANISM_STAGE_ELIGIBLE"] is False
    for d in (og.decide(valid=True, invalid_reason=None, gap=0.12), invalid):
        assert d["MECHANISM_STAGE_ELIGIBLE"] is False


def test_preregistered_examples_end_to_end():
    # NO-GO example: all 10 seeds succeed through 21 and fail at 24 (a static policy at 21 is optimal)
    no_go = _matrix({s: "NNNNB" for s in og.SEEDS})
    g = og.oracle_gap(no_go, 1.0)
    assert og.decide(valid=True, invalid_reason=None, gap=g["gap_fraction_of_generation"])["decision"] == "NO-GO"
    # GO example: half the seeds tolerate K=24, half only K=12
    go = _matrix({**{s: "NNNNN" for s in og.SEEDS[:5]}, **{s: "NBBBB" for s in og.SEEDS[5:]}})
    g = og.oracle_gap(go, 1.0)
    assert g["gap_fraction_of_generation"] == pytest.approx(0.15)
    assert og.decide(valid=True, invalid_reason=None, gap=g["gap_fraction_of_generation"])["decision"] == "GO"
    # single outlier seed: 9 seeds tolerate 24, one fails at 12 -> K_static 0, G = (9*24)/10/40 = 0.54 primary,
    # but tau=0.9 descriptive shows K_static(0.9)=24 with gap 0 (reported, does not change the decision)
    outlier = _matrix({**{s: "NNNNN" for s in og.SEEDS[:9]}, og.SEEDS[9]: "BBBBB"})
    assert og.oracle_gap(outlier, 1.0)["k_static"] == 0 and og.oracle_gap(outlier, 1.0)["gap_fraction_of_generation"] == pytest.approx(0.54)
    d = og.oracle_gap(outlier, 0.9)
    assert d["k_static"] == 24 and d["gap_fraction_of_generation"] == pytest.approx(-24 / 10 / 40)


# ------------------------------------------------------------------ blinding
def test_blinded_assignment_covers_all_forks():
    keys = [(s, k) for s in og.SEEDS[:8] for k in og.KS]
    mapping = og.conf.blinded_assignment(keys, "ab" * 32)
    assert len(mapping) == 40 and sorted(mapping.values()) == sorted(keys)
    assert mapping == og.conf.blinded_assignment(list(reversed(keys)), "ab" * 32)
    assert mapping != og.conf.blinded_assignment(keys, "cd" * 32)


# ------------------------------------------------------------------ preregistration
def _synthetic_scheduler_plan(config):
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40,
            "timesteps": timesteps, "checkpoint_indices": list(og.smoke.EXPECTED_SWITCHES), "resume_indices": list(og.smoke.EXPECTED_SWITCHES),
            "next_timestep_by_checkpoint": {str(step): timesteps[step] for step in og.smoke.EXPECTED_SWITCHES}}


def test_preregistration_immutability(tmp_path, monkeypatch):
    config = _config()
    out = og.REPO_ROOT / "results" / og.NAMESPACE / "_pytest_tmp"
    try:
        og.smoke.scheduler_plan(og.conf.seed_config(config, og.SEEDS[0]))
    except ModuleNotFoundError:
        from experiments import video_runtime_state_discovery as v3

        monkeypatch.setattr(og.smoke, "scheduler_plan", _synthetic_scheduler_plan)
        monkeypatch.setattr(og.smoke, "expert_metadata", lambda cfg, plan, step: v3.expert_region_metadata(cfg, plan, step))
    if out.exists():
        shutil.rmtree(out)
    try:
        result = og.run_cpu(config, CONFIG_PATH, out)
        assert result["status"] == "FROZEN" and result["expected_trajectories_max"] == 80
        prov = og.build_provenance(CONFIG_PATH)
        prereg = og.require_preregistration(out, prov) if not prov["relevant_git_status"] else json.loads((out / "preregistration.json").read_text())
        assert prereg["seeds"] == list(og.SEEDS) and prereg["k_grid"] == [12, 15, 18, 21, 24] and prereg["control_k"] == 18
        assert prereg["prefix_reuse_fraction"]["24"] == 0.6 and prereg["gate"]["no_go_max_gap"] == 0.09
        assert all(prereg["expert_by_k"][str(k)]["current_expert"] == "high_noise_transformer" for k in og.KS)  # every grid K resumes inside the high-noise expert
        assert set(prereg["context_only"]) == {"video_trajectory_fork_confirmatory", "video_trajectory_fork_probe_killtest"}
        path = out / "preregistration.json"
        doc = json.loads(path.read_text()); doc["gate"]["go_min_gap"] = 0.05
        path.write_text(json.dumps(doc))
        with pytest.raises(og.GateError, match="modified"):
            og.require_preregistration(out, prov)
        path.write_text(json.dumps(json.loads(path.read_text())))
        with pytest.raises(og.GateError):
            og.require_preregistration(out, {**prov, "provenance_hash": "0" * 64, "relevant_git_status": []})
        # rerunning cpu on the same namespace with the same provenance is idempotent
        again = og.run_cpu(config, CONFIG_PATH, out)
        assert again["status"] == "FROZEN" and again["preregistration_sha256"] == result["preregistration_sha256"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_seed_family_uses_round4b_severity():
    fam = og.seed_family(_config(), 5678)
    assert fam["id"] == "red_to_blue_seed5678" and fam["severity"] == "round4b" and fam["new_prompt"] == og.NEW_PROMPT
