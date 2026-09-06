"""CPU contracts for the Round 4C-0 action-edit extension. No GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments import video_trajectory_fork_action_edit_killtest as ae

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_trajectory_fork_action_edit_killtest_config.yaml")
CODE = {"N": "NEW", "D": "MIXED_NEW_DOMINANT", "B": "MIXED_BALANCED", "O": "MIXED_OLD_DOMINANT", "R": "OLD"}


def _config():
    return ae.load_config(CONFIG_PATH)


def _matrix(rows):
    return {seed: {k: CODE[c] for k, c in zip(ae.KS, s, strict=True)} for seed, s in rows.items()}


def test_frozen_design():
    config = _config()
    assert ae.SEEDS == (5678, 6789, 7890, 8901) and ae.KS == (4, 8, 12, 16, 20, 24) and ae.CONTROL_K == 8
    assert "static camera" in ae.OLD_PROMPT and "static camera" in ae.NEW_PROMPT
    assert ae.OLD_PROMPT.startswith("A red sports car driving forward") and "parked motionless" in ae.NEW_PROMPT
    assert config["baseline_informativeness"]["method"] == "blinded_human" and config["baseline_informativeness"]["automated_validity_gate"] is None
    assert ae.STRONG_MIN_GAP == 0.15 and ae.DISPERSION_MIN_RANGE == 8 and ae.MIN_INFORMATIVE_SEEDS == 3
    assert "no K-grid change" in config["gate"]["final_decision_rule"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [5678, 6789, 7890, 9012]),
        lambda c: c.__setitem__("k_grid", [4, 8, 12, 16]),
        lambda c: c["prompts"].__setitem__("new", "A red sports car parked motionless on a snowy road, cinematic video"),
        lambda c: c["prompts"].__setitem__("old", "A red sports car driving forward on a snowy road, cinematic video"),
        lambda c: c["prompts"].__setitem__("new", "A red sports car drifting on a snowy road, static camera, cinematic video"),
        lambda c: c.__setitem__("edit", "style"),
        lambda c: c["baseline_informativeness"].__setitem__("method", "clip"),
        lambda c: c["baseline_informativeness"].__setitem__("automated_validity_gate", "motion_energy"),
        lambda c: c["baseline_informativeness"].__setitem__("new_baseline_required_label", "MIXED_NEW_DOMINANT"),
        lambda c: c["gate"].__setitem__("strong_min_gap", 0.10),
        lambda c: c["gate"].__setitem__("min_informative_seeds", 2),
        lambda c: c["oracle"].__setitem__("success_labels", ["NEW"]),
        lambda c: c["labels"]["definitions"].pop("OLD"),
        lambda c: c["generation"].__setitem__("num_inference_steps", 30),
        lambda c: c.__setitem__("blinded_review", False),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        ae.load_config(path)


def test_output_namespace_isolation():
    ae.validate_output_path(ae.REPO_ROOT / "results" / ae.NAMESPACE)
    for bad in ("video_trajectory_fork_multi_edit_killtest", "video_trajectory_fork_oracle_gap_killtest", "video_trajectory_fork_confirmatory", "other"):
        with pytest.raises(ValueError):
            ae.validate_output_path(ae.REPO_ROOT / "results" / bad)


def test_expected_keys():
    full = ae.expected_keys()
    assert len(full) == 4 * (2 + 1 + 6) == 36
    assert ("action", 5678, "same_condition", 8) in full and ("action", 5678, "fork_new", 24) in full
    rows = [{"prompt_family": f"action_seed{s}", "seed": s, "trajectory_type": t, "switch_step": k} for (_, s, t, k) in full]
    ae.validate_key_set(rows, full)
    assert ae.family(6789)["id"] == "action_seed6789" and ae.family(6789)["old_prompt"] == ae.OLD_PROMPT


def test_informativeness_from_blind_labels():
    identity = {str(s): {"identical_initial_latent": True} for s in ae.SEEDS}
    identity["8901"]["identical_initial_latent"] = False
    base = {5678: {"old_baseline": "OLD", "new_baseline": "NEW"}, 6789: {"old_baseline": "MIXED_OLD_DOMINANT", "new_baseline": "NEW"},
            7890: {"old_baseline": "OLD", "new_baseline": "MIXED_NEW_DOMINANT"}, 8901: {"old_baseline": "OLD", "new_baseline": "NEW"}}
    inf = ae.informativeness_from_labels(base, identity)
    assert inf["informative_seeds"] == [5678]
    assert inf["seeds"]["6789"]["reason"] == "old_label" and inf["seeds"]["7890"]["reason"] == "new_label" and inf["seeds"]["8901"]["reason"] == "initial_latent"


def test_final_decision_rule():
    ctx = {"object": "EDIT_WEAK", "background": "EDIT_STRONG"}
    d = ae.final_decision("EDIT_STRONG", ctx)
    assert d["decision"] == "INVEST_4C1" and sorted(d["strong_edits"]) == ["action", "background"] and d["BRANCH_CLOSED"] is False
    d = ae.final_decision("EDIT_WEAK", ctx)
    assert d["decision"] == "STOP_DIFFUSION_REUSE" and d["BRANCH_CLOSED"] is True
    d = ae.final_decision("EDIT_INVALID", ctx)
    assert d["decision"] == "INVALID" and d["BRANCH_CLOSED"] is False


def test_edit_gap_reuse_from_multi_edit_module():
    # K* = 4, 12, 24 (the 4C-0 background pattern) -> STRONG; K* = 4,4,4,8 (object pattern) -> WEAK
    strong = ae.me.edit_gap(_matrix({1: "NBBBBB", 2: "NNNBBB", 3: "NNNNNN"}))
    assert strong["gap_retry"] == pytest.approx((36 - (36 + 28 + 16) / 3) / 40) and ae.me.classify_edit(strong, valid=True) == "EDIT_STRONG"
    weak = ae.me.edit_gap(_matrix({1: "NRRRRR", 2: "NRRRRR", 3: "NBBRRR", 4: "NNBBRR"}))
    assert ae.me.classify_edit(weak, valid=True) == "EDIT_WEAK" and weak["gap_retry"] == pytest.approx(0.025)


def test_motion_energy_descriptive():
    still = np.zeros((33, 8, 8, 3), dtype=np.uint8)
    assert ae.motion_energy(still) == 0.0
    moving = np.zeros((33, 8, 8, 3), dtype=np.uint8)
    moving[1::2] = 255
    assert ae.motion_energy(moving) == pytest.approx(1.0)
    assert ae.motion_energy(np.zeros((1, 8, 8, 3), dtype=np.uint8)) == 0.0


def test_blinded_pool_includes_baselines():
    rows = [("old_baseline", s, -1) for s in ae.SEEDS] + [("new_baseline", s, -1) for s in ae.SEEDS] + [("fork_new", s, k) for s in ae.SEEDS for k in ae.KS]
    mapping = ae.conf.blinded_assignment(rows, "ab" * 32)
    assert len(mapping) == 32 and sorted(mapping.values()) == sorted(rows)
    assert mapping == ae.conf.blinded_assignment(list(reversed(rows)), "ab" * 32)


def _synthetic_scheduler_plan(config):
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40,
            "timesteps": timesteps, "checkpoint_indices": list(ae.smoke.EXPECTED_SWITCHES), "resume_indices": list(ae.smoke.EXPECTED_SWITCHES),
            "next_timestep_by_checkpoint": {str(step): timesteps[step] for step in ae.smoke.EXPECTED_SWITCHES}}


@pytest.mark.skipif(not (ae.ROUND4C0_DIR / "summary.json").exists(), reason="Round 4C-0 results not present")
def test_preregistration_immutability(tmp_path, monkeypatch):
    config = _config()
    out = ae.REPO_ROOT / "results" / ae.NAMESPACE / "_pytest_tmp"
    try:
        ae.smoke.scheduler_plan(ae.conf.seed_config(config, ae.SEEDS[0]))
    except ModuleNotFoundError:
        from experiments import video_runtime_state_discovery as v3

        monkeypatch.setattr(ae.smoke, "scheduler_plan", _synthetic_scheduler_plan)
        monkeypatch.setattr(ae.smoke, "expert_metadata", lambda cfg, plan, step: v3.expert_region_metadata(cfg, plan, step))
    if out.exists():
        shutil.rmtree(out)
    try:
        result = ae.run_cpu(config, CONFIG_PATH, out)
        assert result["status"] == "FROZEN" and result["expected_trajectories"] == 36
        prereg = json.loads((out / "preregistration.json").read_text())
        ref = prereg["round4c0_reference"]
        assert ref["decision"] == "EXTEND_ONE_EDIT" and ref["edit_status"] == {"object": "EDIT_WEAK", "background": "EDIT_STRONG"}
        assert prereg["baseline_informativeness"]["method"] == "blinded_human" and "INVEST_4C1 iff action" in prereg["final_decision_rule"]
        prov = ae.build_provenance(CONFIG_PATH)
        path = out / "preregistration.json"
        doc = json.loads(path.read_text()); doc["gate"]["strong_min_gap"] = 0.05
        path.write_text(json.dumps(doc))
        with pytest.raises(ae.GateError, match="modified"):
            ae.require_preregistration(out, prov)
        path.write_text(json.dumps(json.loads(path.read_text())))
        with pytest.raises(ae.GateError):
            ae.require_preregistration(out, {**prov, "provenance_hash": "0" * 64, "relevant_git_status": []})
    finally:
        shutil.rmtree(out, ignore_errors=True)
