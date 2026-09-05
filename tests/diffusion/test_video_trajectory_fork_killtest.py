from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments import video_trajectory_fork_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

CONFIG = killtest.REPO_ROOT / "experiments/video_trajectory_fork_killtest_config.yaml"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _primary_rows(margins: dict[int, float] | None = None) -> list[dict]:
    margins = margins or {5: -0.1, 10: -0.1, 15: -0.1, 20: -0.1, 25: -0.1, 30: -0.1}
    return [
        {"trajectory_type": "fork_new", "switch_step": step, "new_minus_old": margins[step]}
        for step in killtest.EXPECTED_SWITCHES
        if step > 0
    ]


def test_frozen_configuration_and_prompt_change_isolation() -> None:
    config = _config()
    killtest.validate_config(config)
    family = killtest.prompt_family(config, killtest.PRIMARY_FAMILY)
    assert family["old_prompt"].replace("red", "blue") == family["new_prompt"]
    assert tuple(config["generation"]["switch_steps"]) == killtest.EXPECTED_SWITCHES
    assert config["generation"]["num_inference_steps"] == 40


def test_switch_step_index_and_k0_semantics() -> None:
    config = _config()
    assert 40 - 0 == 40
    assert 40 - 5 == 35
    assert killtest.expected_primary_keys() == {
        ("fork_new", 5), ("fork_new", 10), ("fork_new", 15),
        ("fork_new", 20), ("fork_new", 25), ("fork_new", 30),
    }
    assert ("fork_new", 0) in killtest.expected_control_keys()


def test_worker_probe_binds_resume_index_to_scheduler_timestep() -> None:
    scheduler = {
        "timesteps": [float(value) for value in range(40, 0, -1)],
    }
    metadata = {
        "sample_solver": "euler",
        "scheduler_class": "module.WanEulerScheduler",
        "num_steps": 30,
        "records": [
            {"step_index": 0, "timestep": 30.0},
            {"step_index": 30, "timestep": 1.0},
        ],
    }
    killtest.validate_worker_probe(metadata, scheduler, resume_index=10, local_steps=30)
    wrong = copy.deepcopy(metadata)
    wrong["records"][0]["timestep"] = 31.0
    with pytest.raises(killtest.GateError, match="wrong scheduler timestep"):
        killtest.validate_worker_probe(wrong, scheduler, resume_index=10, local_steps=30)


def test_expected_key_set_rejects_missing_duplicate_and_unexpected() -> None:
    rows = _primary_rows()
    killtest.validate_key_set(rows, killtest.expected_primary_keys())
    with pytest.raises(killtest.GateError, match="missing"):
        killtest.validate_key_set(rows[:-1], killtest.expected_primary_keys())
    with pytest.raises(killtest.GateError, match="duplicates"):
        killtest.validate_key_set(rows + [copy.deepcopy(rows[0])], killtest.expected_primary_keys())
    changed = copy.deepcopy(rows)
    changed[0]["switch_step"] = 7
    with pytest.raises(killtest.GateError, match="unexpected"):
        killtest.validate_key_set(changed, killtest.expected_primary_keys())


def test_result_merge_is_idempotent_and_rejects_conflicts() -> None:
    row = {
        "prompt_family": "red_to_blue", "trajectory_type": "fork_new", "switch_step": 5,
        "provenance_hash": "p", "final_latent_hash": "l", "video_hash": "v", "resume_input_hash": "i",
    }
    assert killtest.merge_result_rows([row], [copy.deepcopy(row)]) == [row]
    conflicting = dict(row, video_hash="other")
    with pytest.raises(killtest.GateError, match="Conflicting"):
        killtest.merge_result_rows([row], [conflicting])


def test_output_path_isolation(tmp_path: Path) -> None:
    valid = killtest.REPO_ROOT / "results/video_trajectory_fork_killtest/test"
    killtest.validate_output_path(valid)
    with pytest.raises(ValueError):
        killtest.validate_output_path(killtest.REPO_ROOT / "results/video_runtime_state_discovery_v3")
    with pytest.raises(ValueError):
        killtest.validate_output_path(tmp_path)


def test_array_identity_binds_shape_dtype_and_bytes() -> None:
    np = pytest.importorskip("numpy")
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert killtest.array_sha256(original) != killtest.array_sha256(original.reshape(2, 6))
    assert killtest.array_sha256(original) != killtest.array_sha256(original.astype(np.float64))


def test_result_artifact_hashes_are_recomputed(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    video = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    latent = np.zeros((1, 2), dtype=np.float32)
    video_path, latent_path = tmp_path / "video.npy", tmp_path / "latent.npy"
    mp4_path, metadata_path = tmp_path / "video.mp4", tmp_path / "metadata.json"
    np.save(video_path, video, allow_pickle=False)
    np.save(latent_path, latent, allow_pickle=False)
    mp4_path.write_bytes(b"video")
    metadata_path.write_text("{}")
    row = {
        "status": "COMPLETE", "provenance_hash": "p",
        "final_video_npy": str(video_path), "final_latent_npy": str(latent_path),
        "final_video_mp4": str(mp4_path), "condition_metadata_json": str(metadata_path),
        "video_hash": killtest.array_sha256(video), "final_latent_hash": killtest.array_sha256(latent),
    }
    killtest._validate_result_artifacts(row, "p")
    latent[0, 0] = 1
    np.save(latent_path, latent, allow_pickle=False)
    with pytest.raises(killtest.GateError, match="hash mismatch"):
        killtest._validate_result_artifacts(row, "p")


def test_frozen_old_checkpoint_hashes_reject_artifact_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    records = {}
    for step in killtest.EXPECTED_SWITCHES:
        path = tmp_path / f"step_{step}.pt"
        torch.save(torch.tensor([float(step)], dtype=torch.float32), path)
        records[step] = {"latent_path": str(path)}
    frozen = killtest.checkpoint_hashes(records)
    killtest.validate_checkpoint_hashes(records, frozen)
    torch.save(torch.tensor([999.0], dtype=torch.float32), records[10]["latent_path"])
    with pytest.raises(killtest.GateError, match="step"):
        killtest.validate_checkpoint_hashes(records, frozen)


def test_baseline_control_gate_blocks_primary() -> None:
    valid = {
        "provenance_hash": "p", "prompt_family": "red_to_blue",
        "old_matches_expected": True, "new_matches_expected": True,
        "clearly_different": True, "fork_outcomes_examined": False,
    }
    killtest.validate_baseline_judgment(valid, "p", "red_to_blue")
    invalid = dict(valid, clearly_different=False)
    with pytest.raises(killtest.GateError, match="C3 failed"):
        killtest.validate_baseline_judgment(invalid, "p", "red_to_blue")
    invalid = dict(valid, fork_outcomes_examined=True)
    with pytest.raises(killtest.GateError, match="before fork"):
        killtest.validate_baseline_judgment(invalid, "p", "red_to_blue")


def test_gpu_control_file_fails_closed(tmp_path: Path) -> None:
    incomplete = {
        "status": "PASS",
        "provenance_hash": "p",
        "scheduler_euler": True,
        "same_condition_exact_all_switches": False,
        "k0_new_baseline_exact": True,
        "identical_initial_latent": True,
        "control_row_count": len(killtest.expected_control_keys()),
    }
    (tmp_path / "preflight.json").write_text(json.dumps(incomplete))
    with pytest.raises(killtest.GateError, match="incomplete/failed"):
        killtest.require_controls(tmp_path, "p")


def test_decision_rules_and_metric_qualitative_disagreement() -> None:
    rows = _primary_rows()
    outcome, largest, _ = killtest.classify_primary(rows, {step: "old" for step in range(5, 31, 5)}, True)
    assert outcome == "CLEAR NO-GO" and largest is None

    margins = {5: 0.1, 10: 0.1, 15: 0.1, 20: 0.1, 25: -0.1, 30: -0.1}
    qualitative = {5: "new", 10: "new", 15: "new", 20: "new", 25: "old", 30: "old"}
    outcome, largest, _ = killtest.classify_primary(_primary_rows(margins), qualitative, True)
    assert outcome == "PROMISING" and largest == 20

    disagreement = dict(qualitative, **{})
    disagreement[10] = "old"
    outcome, _, _ = killtest.classify_primary(_primary_rows(margins), disagreement, True)
    assert outcome == "INCONCLUSIVE"

    outcome, _, _ = killtest.classify_primary(rows, None, True)
    assert outcome == "INCONCLUSIVE"
    outcome, _, _ = killtest.classify_primary(rows, {step: "old" for step in range(5, 31, 5)}, False)
    assert outcome == "INVALID"


def test_only_early_k5_success_is_inconclusive() -> None:
    margins = {5: 0.1, 10: -0.1, 15: -0.1, 20: -0.1, 25: -0.1, 30: -0.1}
    qualitative = {5: "new", 10: "mixed", 15: "old", 20: "old", 25: "old", 30: "old"}
    decision, largest, _ = killtest.classify_primary(_primary_rows(margins), qualitative, True)
    assert decision == "INCONCLUSIVE" and largest is None


def test_expansion_is_blocked_without_promising_summary(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"decision": "CLEAR NO-GO"}))
    with pytest.raises(killtest.GateError, match="forbidden"):
        killtest.run_expansion(_config(), object(), tmp_path)


def test_config_rejects_scientific_matrix_mutation() -> None:
    config = _config()
    config["generation"]["switch_steps"] = [0, 10, 20, 30]
    with pytest.raises(ValueError, match="Switch points"):
        killtest.validate_config(config)
