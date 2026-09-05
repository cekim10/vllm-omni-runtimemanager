"""CPU contracts for Round 4C-0 (two-edit replication screening). No GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import video_trajectory_fork_multi_edit_killtest as me

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_trajectory_fork_multi_edit_killtest_config.yaml")
CODE = {"N": "NEW", "D": "MIXED_NEW_DOMINANT", "B": "MIXED_BALANCED", "O": "MIXED_OLD_DOMINANT", "R": "OLD"}


def _config():
    return me.load_config(CONFIG_PATH)


def _matrix(rows: dict[int, str]) -> dict[int, dict[int, str]]:
    """seed -> 6-char string over {N,D,B,O,R} for K = 4, 8, 12, 16, 20, 24."""
    return {seed: {k: CODE[c] for k, c in zip(me.KS, s, strict=True)} for seed, s in rows.items()}


# ------------------------------------------------------------------ freeze
def test_frozen_design():
    config = _config()
    assert me.SEEDS == (5678, 6789, 7890, 8901) == tuple(config["seeds"]) and me.SEEDS == me.r4b.SEEDS[:4]
    assert me.KS == (4, 8, 12, 16, 20, 24) and me.CONTROL_K == 8 and me.CONTROL_K in me.KS
    assert me.EDITS == ("object", "background") and "style" not in me.EDITS
    assert me.SUCCESS_LABELS == ("NEW", "MIXED_NEW_DOMINANT")
    assert me.STRONG_MIN_GAP == 0.15 and me.DISPERSION_MIN_RANGE == 8 and me.MIN_INFORMATIVE_SEEDS == 3 and me.EXTENSION_EDIT_CLASS == "action"
    assert me.BASELINE_CAPTURE == [0, 4, 8, 12, 16, 20, 24, 40]
    assert me.EDIT_SPEC["object"]["new_prompt"].startswith("A red motorcycle") and me.EDIT_SPEC["background"]["new_prompt"].endswith("desert road, cinematic video")
    for edit in me.EDITS:
        assert set(config["labels"]["definitions"][edit]) == set(me.LABELS)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [5678, 6789, 7890, 9012]),
        lambda c: c.__setitem__("seeds", [5678, 6789, 7890, 8901, 9012]),
        lambda c: c.__setitem__("k_grid", [4, 8, 12, 16]),
        lambda c: c.__setitem__("control_k", 12),
        lambda c: c.__setitem__("edits", ["object", "background", "style"]),
        lambda c: c["prompts"]["object"].__setitem__("new", "A red bicycle driving on a snowy road, cinematic video"),
        lambda c: c["prompts"]["background"].__setitem__("new_concept", "beach"),
        lambda c: c["prompts"].__setitem__("old", "A blue sports car driving on a snowy road, cinematic video"),
        lambda c: c["oracle"].__setitem__("success_labels", ["NEW", "MIXED_NEW_DOMINANT", "MIXED_BALANCED"]),
        lambda c: c["gate"].__setitem__("strong_min_gap", 0.10),
        lambda c: c["gate"].__setitem__("dispersion_min_range", 4),
        lambda c: c["gate"].__setitem__("min_informative_seeds", 2),
        lambda c: c["gate"].__setitem__("extension_edit_class", "style"),
        lambda c: c["labels"]["definitions"]["object"].pop("MIXED_BALANCED"),
        lambda c: c["generation"].__setitem__("guidance_scale", 5.0),
        lambda c: c["scheduler"].__setitem__("sample_solver", "unipc"),
        lambda c: c.__setitem__("blinded_review", False),
        lambda c: c.__setitem__("experiment_version", "video-trajectory-fork-multi-edit-screening-v2"),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        me.load_config(path)


def test_output_namespace_isolation():
    me.validate_output_path(me.REPO_ROOT / "results" / me.NAMESPACE)
    for bad in ("video_trajectory_fork_oracle_gap_killtest", "video_trajectory_fork_confirmatory", "video_trajectory_fork_probe_killtest", "video_trajectory_fork_killtest", "other"):
        with pytest.raises(ValueError):
            me.validate_output_path(me.REPO_ROOT / "results" / bad)
    with pytest.raises(ValueError):
        me.validate_output_path(me.REPO_ROOT / "results" / me.NAMESPACE / "video_trajectory_fork_oracle_gap_killtest")


# ------------------------------------------------------------------ families / keys
def test_families_and_edit_of_row():
    old = me.old_family(5678)
    assert old["id"] == "old_seed5678" and old["old_prompt"] == old["new_prompt"] == me.OLD_PROMPT
    obj = me.edit_family("object", 5678)
    assert obj["id"] == "object_seed5678" and obj["new_prompt"] == me.EDIT_SPEC["object"]["new_prompt"] and obj["old_concept"] == "sports car"
    assert me.edit_of_row({"prompt_family": "background_seed6789"}) == "background" and me.edit_of_row({"prompt_family": "old_seed6789"}) == "old"


def test_expected_key_set():
    full = me.expected_keys()
    # 4 OLD + 8 NEW baselines + 4 controls + 2 x 4 x 6 forks
    assert len(full) == 4 + 8 + 4 + 48 == 64
    assert ("old", 5678, "same_condition", 8) in full and ("object", 5678, "same_condition", 8) not in full
    partial = me.expected_keys({"object": [5678, 6789, 7890], "background": [5678, 6789, 7890, 8901]})
    assert len(partial) == 12 + 4 + 3 * 6 + 4 * 6
    assert ("object", 8901, "fork_new", 4) not in partial and ("background", 8901, "fork_new", 4) in partial and ("object", 8901, "new_baseline", -1) in partial
    only_bg = me.expected_keys({"object": [], "background": [5678]})
    assert ("old", 5678, "same_condition", 8) in only_bg and ("old", 6789, "same_condition", 8) not in only_bg
    rows = [{"prompt_family": f"{e}_seed{s}", "seed": s, "trajectory_type": t, "switch_step": k} for (e, s, t, k) in full]
    me.validate_key_set(rows, full)
    with pytest.raises(me.GateError, match="duplicates"):
        me.validate_key_set(rows + [rows[0]], full)
    with pytest.raises(me.GateError, match="missing"):
        me.validate_key_set(rows[:-1], full)


# ------------------------------------------------------------------ oracle definitions
@pytest.mark.parametrize("seq,k_star,viol", [("NNNNNN", 24, 0), ("NNDBRR", 12, 0), ("NBNNNN", 4, 4), ("BNNNNN", 0, 5), ("RRRRRR", 0, 0), ("NNNNBR", 16, 0)])
def test_trajectory_oracle(seq, k_star, viol):
    r = me.trajectory_oracle_k(_matrix({1: seq})[1])
    assert r["k_star"] == k_star and r["monotonicity_violations"] == viol


def test_retry_static_costs():
    # K* = 4, 8, 16, 16 (the user's "clear dispersion" example)
    m = _matrix({1: "NBBBBB", 2: "NNBBBB", 3: "NNNNBB", 4: "NNNNBB"})
    st = me.retry_static_costs(m)
    assert st["p_fail_by_k"] == {"0": 0.0, "4": 0.0, "8": 0.25, "12": 0.5, "16": 0.5, "20": 1.0, "24": 1.0}
    assert st["cost_by_k"]["0"] == 40 and st["cost_by_k"]["4"] == 36 and st["cost_by_k"]["8"] == 42 and st["cost_by_k"]["16"] == 44
    assert st["best_k"] == 4 and st["best_cost"] == 36
    g = me.edit_gap(m)
    assert g["k_star_by_seed"] == {"1": 4, "2": 8, "3": 16, "4": 16} and g["c_traj"] == 29.0
    assert g["gap_retry"] == pytest.approx(7 / 40) and g["dispersion"] is True and g["k_star_range"] == 12
    assert me.classify_edit(g, valid=True) == "EDIT_STRONG"


def test_no_dispersion_is_weak_even_with_reuse():
    # K* = 8, 8, 8, 8 -> static K=8 is perfect, gap 0
    m = _matrix({s: "NNBBBB" for s in range(1, 5)})
    g = me.edit_gap(m)
    assert g["static_retry"]["best_k"] == 8 and g["gap_retry"] == 0.0 and g["dispersion"] is False
    assert me.classify_edit(g, valid=True) == "EDIT_WEAK"
    # all at grid top: no dispersion, flagged
    top = me.edit_gap(_matrix({s: "NNNNNN" for s in range(1, 5)}))
    assert top["all_k_star_at_grid_top"] is True and me.classify_edit(top, valid=True) == "EDIT_WEAK"
    # dispersion but small gap (range 8, one seed early): K* = 8,16,16,16
    m2 = _matrix({1: "NNBBBB", 2: "NNNNBB", 3: "NNNNBB", 4: "NNNNBB"})
    g2 = me.edit_gap(m2)
    # static K=16: 24 + 0.25*40 = 34; K=8: 32 + 0 = 32 -> best 8 at 32; c_traj = mean(32,24,24,24) = 26 -> gap 6/40 = 0.15
    assert g2["static_retry"]["best_k"] == 8 and g2["gap_retry"] == pytest.approx(0.15) and me.classify_edit(g2, valid=True) == "EDIT_STRONG"
    # tie-break: range exactly 8 but gap below threshold: K* = 8, 8, 8, 16 -> static 8 at 32, c_traj 30 -> 0.05
    g3 = me.edit_gap(_matrix({1: "NNBBBB", 2: "NNBBBB", 3: "NNBBBB", 4: "NNNNBB"}))
    assert g3["dispersion"] is True and g3["gap_retry"] == pytest.approx(0.05) and me.classify_edit(g3, valid=True) == "EDIT_WEAK"
    # dispersion range 4 only is not dispersion
    g4 = me.edit_gap(_matrix({1: "NNBBBB", 2: "NNNBBB", 3: "NNNBBB", 4: "NNNBBB"}))
    assert g4["k_star_range"] == 4 and g4["dispersion"] is False
    assert me.classify_edit(g4, valid=False) == "EDIT_INVALID"


def test_round4b_colour_matrix_under_screening_rule():
    # Round 4B colour labels restricted to a hypothetical 4-seed subset must be reproducible by the same functions
    # (K grid differs, so this only checks the retry-cost arithmetic on the 4B pattern: K* = 12, 12, 18->16-ish is not mappable);
    # use a synthetic grid-aligned analogue: K* = 12, 12, 16, 0
    g = me.edit_gap(_matrix({1: "NNNBBB", 2: "NNNBBB", 3: "NNNNBB", 4: "BBBBBB"}))
    assert g["static_retry"]["p_fail_by_k"]["4"] == 0.25 and g["static_retry"]["best_k"] == 12  # 28 + 0.25*40 = 38 < 40
    assert g["c_traj"] == 30.0 and g["gap_retry"] == pytest.approx(8 / 40)


def test_descriptive_tau_frontier():
    g = me.edit_gap(_matrix({1: "NNNBBB", 2: "NNNBBB", 3: "NNNNBB", 4: "BBBBBB"}))
    fr = g["tau_frontier_descriptive"]
    assert fr["0"]["k_static"] == 0 and fr["0"]["nominal_reliability"] == 1.0
    assert fr["1"]["k_static"] == 12 and fr["1"]["nominal_reliability"] == 0.75


def test_decision_table():
    assert me.decide({"object": "EDIT_STRONG", "background": "EDIT_STRONG"})["decision"] == "INVEST_4C1"
    d = me.decide({"object": "EDIT_STRONG", "background": "EDIT_WEAK"})
    assert d["decision"] == "EXTEND_ONE_EDIT" and "action" in d["NEXT"]
    assert me.decide({"object": "EDIT_WEAK", "background": "EDIT_WEAK"})["decision"] == "STOP_DIFFUSION_REUSE"
    assert me.decide({"object": "EDIT_INVALID", "background": "EDIT_STRONG"})["decision"] == "INVALID"


def test_spearman_with_ties():
    assert me.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert me.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert me.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert me.spearman([1, 2], [1, 2]) is None
    assert -1.0 <= me.spearman([12, 12, 18, 0], [4, 8, 16, 16]) <= 1.0


# ------------------------------------------------------------------ blinding
def test_blinded_assignment_over_edit_keys():
    keys = [(e, s, k) for e in me.EDITS for s in me.SEEDS for k in me.KS]
    mapping = me.conf.blinded_assignment(keys, "ab" * 32)
    assert len(mapping) == 48 and sorted(mapping.values()) == sorted(keys)
    assert mapping == me.conf.blinded_assignment(list(reversed(keys)), "ab" * 32)
    assert mapping != me.conf.blinded_assignment(keys, "cd" * 32)


# ------------------------------------------------------------------ preregistration
def _synthetic_scheduler_plan(config):
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40,
            "timesteps": timesteps, "checkpoint_indices": list(me.smoke.EXPECTED_SWITCHES), "resume_indices": list(me.smoke.EXPECTED_SWITCHES),
            "next_timestep_by_checkpoint": {str(step): timesteps[step] for step in me.smoke.EXPECTED_SWITCHES}}


@pytest.mark.skipif(not (me.ROUND4B_DIR / "summary.json").exists(), reason="Round 4B results not present")
def test_preregistration_immutability(tmp_path, monkeypatch):
    config = _config()
    out = me.REPO_ROOT / "results" / me.NAMESPACE / "_pytest_tmp"
    try:
        me.smoke.scheduler_plan(me.conf.seed_config(config, me.SEEDS[0]))
    except ModuleNotFoundError:
        from experiments import video_runtime_state_discovery as v3

        monkeypatch.setattr(me.smoke, "scheduler_plan", _synthetic_scheduler_plan)
        monkeypatch.setattr(me.smoke, "expert_metadata", lambda cfg, plan, step: v3.expert_region_metadata(cfg, plan, step))
    if out.exists():
        shutil.rmtree(out)
    try:
        result = me.run_cpu(config, CONFIG_PATH, out)
        assert result["status"] == "FROZEN" and result["expected_trajectories_max"] == 64
        prereg = json.loads((out / "preregistration.json").read_text())
        assert prereg["seeds"] == [5678, 6789, 7890, 8901] and prereg["k_grid"] == [4, 8, 12, 16, 20, 24] and prereg["statistical_claim"].startswith("none")
        ref = prereg["round4b_reference"]
        assert ref["decision"] == "GO" and set(ref["old_baseline_hashes"]) == {str(s) for s in me.SEEDS} and set(ref["colour_k_star_by_seed"]) >= {str(s) for s in me.SEEDS}
        assert all(prereg["expert_by_k"][str(k)]["current_expert"] == "high_noise_transformer" for k in me.KS)
        prov = me.build_provenance(CONFIG_PATH)
        path = out / "preregistration.json"
        doc = json.loads(path.read_text()); doc["gate"]["strong_min_gap"] = 0.05
        path.write_text(json.dumps(doc))
        with pytest.raises(me.GateError, match="modified"):
            me.require_preregistration(out, prov)
        path.write_text(json.dumps(json.loads(path.read_text())))
        with pytest.raises(me.GateError):
            me.require_preregistration(out, {**prov, "provenance_hash": "0" * 64, "relevant_git_status": []})
        again = me.run_cpu(config, CONFIG_PATH, out)
        assert again["status"] == "FROZEN" and again["preregistration_sha256"] == result["preregistration_sha256"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
