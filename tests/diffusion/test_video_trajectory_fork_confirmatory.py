"""CPU contracts for the confirmatory trajectory-fork kill test (no GPU)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments import video_trajectory_fork_confirmatory as conf

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_trajectory_fork_confirmatory_config.yaml")


def _config():
    return conf.load_config(CONFIG_PATH)


# ------------------------------------------------------------------ freeze
def test_frozen_seeds_switches_prompts():
    config = _config()
    assert tuple(config["seeds"]) == (2345, 3456, 4567) and 1234 not in config["seeds"]
    assert tuple(config["switch_steps"]) == (10, 15, 20, 25)
    assert config["prompts"]["old"].startswith("A red sports car") and config["prompts"]["new"].startswith("A blue sports car")
    assert conf.ORDINAL == {"NEW": 2, "MIXED_NEW_DOMINANT": 1, "MIXED_BALANCED": 0, "MIXED_OLD_DOMINANT": -1, "OLD": -2}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [1234, 3456, 4567]),
        lambda c: c.__setitem__("seeds", [2345, 3456]),
        lambda c: c.__setitem__("switch_steps", [5, 10, 15, 20, 25]),
        lambda c: c["prompts"].__setitem__("new", "A green sports car driving on a snowy road, cinematic video"),
        lambda c: c["labels"]["ordinal"].__setitem__("MIXED_BALANCED", 1),
        lambda c: c["gate"].__setitem__("go_min_frontier_seeds", 1),
        lambda c: c["gate"].__setitem__("early_new_labels", ["NEW", "MIXED_NEW_DOMINANT", "MIXED_BALANCED"]),
        lambda c: c["generation"].__setitem__("guidance_scale", 5.0),
        lambda c: c["scheduler"].__setitem__("sample_solver", "unipc"),
        lambda c: c.__setitem__("blinded_review", False),
        lambda c: c["concept_metric"].__setitem__("sign_only_threshold", 0.01),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        conf.load_config(path)


def test_output_namespace_isolation():
    conf.validate_output_path(conf.REPO_ROOT / "results" / "video_trajectory_fork_confirmatory")
    for bad in ("video_trajectory_fork_killtest", "video_runtime_state_discovery_v3_corrected", "video_bf16_single_flip_killtest", "something_else"):
        with pytest.raises(ValueError):
            conf.validate_output_path(conf.REPO_ROOT / "results" / bad)


# ------------------------------------------------------------------ key set
def test_expected_key_set_and_duplicates():
    full = conf.expected_keys()
    assert len(full) == 30 and len({k for k in full if k[1] in ("old_baseline", "new_baseline")}) == 6
    partial = conf.expected_keys([2345, 4567])
    assert len(partial) == 6 + 2 * 8
    rows = [{"seed": s, "trajectory_type": t, "switch_step": k} for s, t, k in sorted(full)]
    conf.validate_key_set(rows, full)
    with pytest.raises(conf.GateError, match="duplicates"):
        conf.validate_key_set(rows + [rows[0]], full)
    with pytest.raises(conf.GateError, match="missing"):
        conf.validate_key_set(rows[:-1], full)
    extra = rows + [{"seed": 2345, "trajectory_type": "fork_new", "switch_step": 30}]
    with pytest.raises(conf.GateError, match="unexpected"):
        conf.validate_key_set(extra, full)


# ------------------------------------------------------------------ gate primitives
def _labels(*values):
    return dict(zip((10, 15, 20, 25), values, strict=True))


def test_early_new_late_shift_frontier():
    assert conf.early_new(_labels("NEW", "NEW", "OLD", "OLD"))
    assert conf.early_new(_labels("MIXED_NEW_DOMINANT", "NEW", "OLD", "OLD"))
    assert not conf.early_new(_labels("NEW", "MIXED_BALANCED", "OLD", "OLD"))
    assert conf.late_shift(_labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "NEW"))
    assert conf.late_shift(_labels("NEW", "NEW", "NEW", "OLD"))
    assert conf.late_shift(_labels("NEW", "MIXED_NEW_DOMINANT", "NEW", "MIXED_OLD_DOMINANT"))
    assert not conf.late_shift(_labels("NEW", "NEW", "NEW", "NEW"))
    assert not conf.late_shift(_labels("NEW", "MIXED_BALANCED", "NEW", "MIXED_NEW_DOMINANT"))  # later points are MORE new
    assert conf.frontier_replication(_labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"))
    assert not conf.frontier_replication(_labels("NEW", "NEW", "NEW", "NEW"))
    assert conf.monotone_non_increasing(_labels("NEW", "NEW", "MIXED_BALANCED", "OLD"))
    assert not conf.monotone_non_increasing(_labels("NEW", "MIXED_BALANCED", "NEW", "OLD"))


def test_metric_disagreement_rule():
    assert conf.metric_disagreement("NEW", -0.01) and conf.metric_disagreement("MIXED_NEW_DOMINANT", -0.01)
    assert conf.metric_disagreement("OLD", 0.01) and conf.metric_disagreement("MIXED_OLD_DOMINANT", 0.01)
    assert not conf.metric_disagreement("NEW", 0.05) and not conf.metric_disagreement("OLD", -0.05)
    assert not conf.metric_disagreement("MIXED_BALANCED", 0.05) and not conf.metric_disagreement("MIXED_BALANCED", -0.05)
    assert not conf.metric_disagreement("NEW", 0.0) and not conf.metric_disagreement("OLD", 0.0)


def _margins_for(labels):
    return {k: (0.05 if conf.ORDINAL[v] > 0 else -0.05 if conf.ORDINAL[v] < 0 else 0.0) for k, v in labels.items()}


def _classify(seed_labels, informative=(2345, 3456, 4567), controls=True, margins=None):
    margins = margins or {s: _margins_for(l) for s, l in seed_labels.items()}
    return conf.classify(seed_labels, margins, list(informative), controls)


def test_classifier_go_example_from_preregistration():
    result = _classify({
        2345: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"),
        3456: _labels("NEW", "NEW", "MIXED_BALANCED", "OLD"),
        4567: _labels("NEW", "NEW", "NEW", "NEW"),
    })
    assert result["decision"] == "GO" and result["ROUND4_ELIGIBLE"] is True
    assert result["counts"] == {"baseline_informative": 3, "EARLY_NEW": 3, "LATE_SHIFT": 2, "FRONTIER_REPLICATION": 2, "denominator": 3}


def test_classifier_no_go_example_from_preregistration():
    result = _classify({
        2345: _labels("NEW", "OLD", "OLD", "OLD"),
        3456: _labels("MIXED_OLD_DOMINANT", "OLD", "OLD", "OLD"),
        4567: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"),
    })
    assert result["decision"] == "NO-GO" and result["ROUND4_ELIGIBLE"] is False
    assert result["counts"]["EARLY_NEW"] == 1 and result["counts"]["FRONTIER_REPLICATION"] == 1


def test_classifier_inconclusive_cases():
    one_frontier = _classify({
        2345: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"),
        3456: _labels("NEW", "NEW", "NEW", "NEW"),
        4567: _labels("NEW", "NEW", "NEW", "NEW"),
    })
    assert one_frontier["decision"] == "INCONCLUSIVE"  # early responsiveness replicates, no later shift
    mixed = _classify({
        2345: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"),
        3456: _labels("NEW", "OLD", "OLD", "OLD"),
        4567: _labels("NEW", "NEW", "NEW", "NEW"),
    })
    assert mixed["decision"] == "INCONCLUSIVE"  # 1 frontier, 1 failure, 1 no-shift
    # exploratory seed-1234 pattern mapped onto the new scale would be: NEW NEW MIXED_NEW_DOMINANT OLD -> frontier
    assert conf.frontier_replication(_labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"))


def test_classifier_uninformative_and_invalid_handling():
    labels = {2345: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"), 3456: _labels("NEW", "NEW", "MIXED_BALANCED", "OLD")}
    two_of_three = _classify(labels, informative=(2345, 3456))
    assert two_of_three["decision"] == "GO" and two_of_three["per_seed"]["4567"]["status"] == "BASELINE_UNINFORMATIVE"
    one_success = _classify({2345: labels[2345], 3456: _labels("NEW", "NEW", "NEW", "NEW")}, informative=(2345, 3456))
    assert one_success["decision"] == "INCONCLUSIVE"
    invalid = _classify({2345: labels[2345]}, informative=(2345,))
    assert invalid["decision"] == "INVALID"
    controls_failed = _classify(labels | {4567: _labels("NEW", "NEW", "NEW", "OLD")}, controls=False)
    assert controls_failed["decision"] == "INVALID"


def test_metric_disagreement_blocks_success_and_failure_counting():
    labels = {
        2345: _labels("NEW", "NEW", "MIXED_NEW_DOMINANT", "OLD"),
        3456: _labels("NEW", "NEW", "MIXED_BALANCED", "OLD"),
        4567: _labels("OLD", "OLD", "OLD", "OLD"),
    }
    margins = {s: _margins_for(l) for s, l in labels.items()}
    margins[3456][10] = -0.02  # labelled NEW but CLIP favours OLD -> disagreement
    result = _classify(labels, margins=margins)
    assert result["per_seed"]["3456"]["metric_disagreement_steps"] == [10]
    assert result["per_seed"]["3456"]["counts_as_success"] is False
    assert result["decision"] == "INCONCLUSIVE"  # only one clean frontier seed remains
    margins2 = {s: _margins_for(l) for s, l in labels.items()}
    margins2[4567][10] = 0.03  # labelled OLD but CLIP favours NEW -> disagreement, not an interpretable failure
    labels2 = copy.deepcopy(labels); labels2[2345] = _labels("OLD", "OLD", "OLD", "OLD")
    margins2[2345] = _margins_for(labels2[2345])
    result2 = _classify(labels2, margins=margins2)
    assert result2["per_seed"]["4567"]["counts_as_failure"] is False and result2["decision"] == "INCONCLUSIVE"


def test_classifier_rejects_incomplete_labels():
    with pytest.raises(conf.GateError, match="incomplete"):
        _classify({2345: {10: "NEW", 15: "NEW", 20: "NEW"}, 3456: _labels("NEW", "NEW", "NEW", "OLD"), 4567: _labels("NEW", "NEW", "NEW", "OLD")})


# ------------------------------------------------------------------ blinding / preregistration
def test_blinded_assignment_is_deterministic_bijection():
    keys = [(s, k) for s in (2345, 3456, 4567) for k in (10, 15, 20, 25)]
    a = conf.blinded_assignment(keys, "ab" * 32)
    b = conf.blinded_assignment(list(reversed(keys)), "ab" * 32)
    assert a == b and sorted(a.values()) == sorted(keys) and len(a) == 12
    assert list(a) == [f"sample_{i:02d}" for i in range(1, 13)]
    assert conf.blinded_assignment(keys, "cd" * 32) != a  # a different preregistration hash gives a different shuffle
    assert list(a.values()) != sorted(keys)  # actually shuffled


def test_frame_indices_are_deterministic():
    assert conf.frame_indices(33) == {"first": 0, "q25": 8, "middle": 16, "q75": 24, "last": 32}


def _synthetic_scheduler_plan(config):
    """Torch-free stand-in used only when the trusted WanEulerScheduler cannot be imported locally."""
    from experiments import video_bf16_single_flip_killtest as single_flip

    timesteps = single_flip.scheduler_timesteps_numpy(config)
    return {"scheduler_class": "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler", "num_inference_steps": 40,
            "timesteps": timesteps, "checkpoint_indices": list(conf.smoke.EXPECTED_SWITCHES), "resume_indices": list(conf.smoke.EXPECTED_SWITCHES),
            "next_timestep_by_checkpoint": {str(step): timesteps[step] for step in conf.smoke.EXPECTED_SWITCHES}}


def test_preregistration_immutability(tmp_path, monkeypatch):
    config = _config()
    out = conf.REPO_ROOT / "results" / "video_trajectory_fork_confirmatory" / "_pytest_tmp"
    try:
        conf.smoke.scheduler_plan(conf.seed_config(config, 2345))
    except ModuleNotFoundError:
        from experiments import video_runtime_state_discovery as v3

        monkeypatch.setattr(conf.smoke, "scheduler_plan", _synthetic_scheduler_plan)
        monkeypatch.setattr(conf.smoke, "expert_metadata", lambda cfg, plan, step: v3.expert_region_metadata(cfg, plan, step))
    import shutil
    if out.exists():
        shutil.rmtree(out)
    try:
        result = conf.run_cpu(config, CONFIG_PATH, out)
        assert result["status"] == "FROZEN" and result["expected_trajectories_max"] == 30
        prov = conf.build_provenance(CONFIG_PATH)
        prereg = conf.require_preregistration(out, prov)
        assert prereg["seeds"] == [2345, 3456, 4567] and prereg["switch_steps"] == [10, 15, 20, 25]
        assert prereg["prior_exploratory_result"]["decision"] == "INCONCLUSIVE"
        # tamper -> rejected
        path = out / "preregistration.json"
        doc = json.loads(path.read_text()); doc["gate"]["go_min_frontier_seeds"] = 1
        path.write_text(json.dumps(doc))
        with pytest.raises(conf.GateError, match="modified"):
            conf.require_preregistration(out, prov)
        # different provenance -> rejected
        path.write_text(json.dumps(json.loads(path.read_text())))
        with pytest.raises(conf.GateError):
            conf.require_preregistration(out, {**prov, "provenance_hash": "0" * 64})
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_seed_family_and_config_adapter():
    config = _config()
    fam = conf.seed_family(config, 2345)
    assert fam["id"] == "red_to_blue_seed2345" and fam["old_prompt"] == conf.OLD_PROMPT and fam["new_prompt"] == conf.NEW_PROMPT
    adapted = conf.seed_config(config, 3456)
    assert adapted["seed"] == 3456 and adapted["generation"]["switch_steps"] == list(conf.smoke.EXPECTED_SWITCHES)
    assert config.get("seed") is None  # the shared config carries no single seed
