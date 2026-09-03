"""CPU-only contracts for the batch-composition reproducibility kill test."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments import video_batch_determinism_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_batch_determinism_killtest_config.yaml")


def _config() -> dict:
    return killtest.load_config(CONFIG_PATH)


def _manifests():
    config = _config()
    targets = killtest.trusted_targets(config)
    return config, {"targets": targets}, killtest.filler_manifest(targets, config), killtest.run_order(targets, config)


def _identities(target):
    tokens = {item["prompt_id"]: {"token_ids_hash": "tokens", "attention_mask_hash": "mask"} for item in target["targets"]}
    runtime = {
        "model_revision": "pinned", "runtime_dtype": "torch.bfloat16", "backend": "flash",
        "compile_state": "compiled", "scheduler_config_hash": "scheduler",
    }
    return tokens, runtime


def _validate(rows, config, target, fillers, order):
    tokens, runtime = _identities(target)
    return killtest.validate_rows(
        rows, config, target, fillers, order,
        token_identities=tokens, runtime_identity=runtime,
    )


def _row(config, target, fillers, order, target_id: str, mode: str, *, replay_id=0, ssim=1.0, exact=True):
    batch = config["batch_modes"][mode]
    fill = killtest.expected_filler_rows(target_id, mode, fillers)
    seed = next(x["generation_seed"] for x in target["targets"] if x["prompt_id"] == target_id)
    run_index = next(x["run_order_index"] for x in order["order"] if x["target_id"] == target_id and x["batch_mode"] == mode)
    noise = killtest.tensor_identity(killtest.make_noise(seed, config))
    return {
        "target_id": target_id, "target_seed": seed, "batch_mode": mode, "effective_batch_size": batch,
        "target_batch_index": 0, "filler_ids_json": json.dumps([x["prompt_id"] for x in fill]),
        "filler_seeds_json": json.dumps([x["generation_seed"] for x in fill]), "run_order_index": run_index,
        "replay_id": replay_id, "target_initial_noise_identity": noise, "target_initial_latent_identity": noise,
        "target_token_ids_hash": "tokens", "target_attention_mask_hash": "mask", "scheduler_class": killtest.EXPECTED_SCHEDULER_CLASS,
        "scheduler_config_hash": "scheduler", "model": killtest.EXPECTED_MODEL, "model_revision": "pinned",
        "runtime_dtype": "torch.bfloat16", "backend": "flash", "compile_state": "compiled",
        "final_latent_identity": f"latent-{target_id}-{mode}",
        "video_identity": f"video-{target_id}-{mode}",
        "final_latent_exact_vs_solo": exact, "video_exact_vs_solo": exact, "frame_ssim_mean": ssim,
        "video_mse": 0.0 if exact else 1.0 - ssim, "final_latent_mse": 0.0 if exact else 1.0,
        "strong_difference": ssim < .95, "target_output_artifact_json": "{}", "final_latent_artifact_json": "{}", "result_path": "row.json",
    }


def _matrix(config, target, fillers, order):
    return [_row(config, target, fillers, order, t["prompt_id"], mode) for t in target["targets"] for mode in killtest.TARGET_MODES]


def _replays(config, target, fillers, order, target_id, mode, ssim):
    return [_row(config, target, fillers, order, target_id, mode, replay_id=i, ssim=ssim, exact=False) for i in range(3)]


def test_cpu_manifest_is_exact_12_targets_36_keys_and_deterministic_fillers(tmp_path):
    config = _config(); config["output_namespace"] = "out"
    result = killtest.run_cpu(config, CONFIG_PATH, tmp_path / "out")
    assert result["primary_rows"] == 36 and len(result["targets"]) == 12
    keys = json.loads((tmp_path / "out/expected_keys.json").read_text())
    assert len(keys["keys"]) == 36 == len({(x["target_id"], x["batch_mode"]) for x in keys["keys"]})
    fillers = json.loads((tmp_path / "out/filler_manifest.json").read_text())
    for target_id, data in fillers["targets"].items():
        assert target_id not in [x["prompt_id"] for x in data["B4"] + data["B8"]]
        assert len(data["B4"]) == 3 and len(data["B8"]) == 7
    capability = json.loads((tmp_path / "out/runtime_batching_capability.json").read_text())
    assert capability["status"] == "UNSUPPORTED"
    assert not capability["pipeline_supports_step_execution_contract"]
    assert capability["runner_uses_single_request_forward"]


@pytest.mark.parametrize("field,value", [
    ("target_initial_noise_identity", "different"), ("target_batch_index", 1),
    ("filler_ids_json", "[]"), ("filler_seeds_json", "[]"),
    ("effective_batch_size", 99), ("scheduler_class", "UniPC"), ("model", "other"),
    ("target_token_ids_hash", "other-tokens"), ("target_attention_mask_hash", "other-mask"),
    ("model_revision", "other-revision"), ("runtime_dtype", "torch.float32"),
    ("backend", "other-backend"), ("compile_state", "eager"),
    ("scheduler_config_hash", "other-scheduler"),
])
def test_validator_rejects_target_and_batch_mutations(field, value):
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    row = next(item for item in rows if item["batch_mode"] == "B4") if field.startswith("filler_") else rows[0]
    row[field] = value
    with pytest.raises(killtest.GlobalStopError): _validate(rows, config, target, fillers, order)


def test_validator_rejects_missing_duplicate_wrong_order_and_nonfinite():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    with pytest.raises(killtest.GlobalStopError): _validate(rows[:-1], config, target, fillers, order)
    with pytest.raises(killtest.GlobalStopError): _validate(rows + [copy.deepcopy(rows[0])], config, target, fillers, order)
    rows[0]["frame_ssim_mean"] = float("nan")
    with pytest.raises(killtest.GlobalStopError): _validate(rows, config, target, fillers, order)


def test_validator_rejects_frozen_run_order_mutation():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    rows[0]["run_order_index"] = 35 - int(rows[0]["run_order_index"])
    with pytest.raises(killtest.GlobalStopError): _validate(rows, config, target, fillers, order)


def test_decision_boundaries_and_replay_requirement():
    config, target, fillers, order = _manifests()
    exact = _matrix(config, target, fillers, order)
    assert killtest.analyze(exact, config, gates_passed=True)["decision"] == "NO_GO"
    # One or two target effects cannot satisfy the breadth requirement.
    for count in (1, 2):
        rows = _matrix(config, target, fillers, order)
        for item in target["targets"][:count]:
            key = item["prompt_id"]; rows = [x for x in rows if not (x["target_id"] == key and x["batch_mode"] == "B4")] + _replays(config, target, fillers, order, key, "B4", .98)
        _validate(rows, config, target, fillers, order)
        assert killtest.analyze(rows, config, gates_passed=True)["decision"] == "WEAK_INCONCLUSIVE"
    rows = _matrix(config, target, fillers, order)
    for item in target["targets"][:3]:
        key = item["prompt_id"]; rows = [x for x in rows if not (x["target_id"] == key and x["batch_mode"] == "B4")] + _replays(config, target, fillers, order, key, "B4", .989999)
    _validate(rows, config, target, fillers, order)
    assert killtest.analyze(rows, config, gates_passed=True)["decision"] == "GO_SYSTEMS_RELEVANCE"


def test_three_material_rows_with_only_one_stable_replay_cannot_go():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    for index, item in enumerate(target["targets"][:3]):
        key = item["prompt_id"]
        rows = [x for x in rows if not (x["target_id"] == key and x["batch_mode"] == "B4")]
        replay = _replays(config, target, fillers, order, key, "B4", .98)
        if index:
            replay[-1]["video_identity"] += "-nondeterministic"
        rows.extend(replay)
    _validate(rows, config, target, fillers, order)
    assert killtest.analyze(rows, config, gates_passed=True)["decision"] == "WEAK_INCONCLUSIVE"


def test_numeric_only_boundaries_and_row_order_invariance():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    first = target["targets"][0]["prompt_id"]
    for row in rows:
        if row["target_id"] == first and row["batch_mode"] == "B8":
            row.update(final_latent_exact_vs_solo=False, video_exact_vs_solo=False, frame_ssim_mean=.99, video_mse=.01, final_latent_mse=.01)
    _validate(rows, config, target, fillers, order)
    original = killtest.analyze(rows, config, gates_passed=True)
    assert original["decision"] == "WEAK_NUMERICAL_ONLY"
    assert killtest.analyze(list(reversed(rows)), config, gates_passed=True) == original
    assert all(not x["strong_difference"] for x in original["pair_summary"])


def test_replay_input_mutation_is_not_deterministic_evidence():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    key = target["targets"][0]["prompt_id"]
    rows = [x for x in rows if not (x["target_id"] == key and x["batch_mode"] == "B4")] + _replays(config, target, fillers, order, key, "B4", .98)
    rows[-1]["filler_ids_json"] = "[]"
    with pytest.raises(killtest.GlobalStopError): _validate(rows, config, target, fillers, order)


def test_strong_single_target_and_auxiliary_annotation_cannot_rescue_go():
    config, target, fillers, order = _manifests(); rows = _matrix(config, target, fillers, order)
    key = target["targets"][0]["prompt_id"]
    rows = [x for x in rows if not (x["target_id"] == key and x["batch_mode"] == "B8")]
    rows.extend(_replays(config, target, fillers, order, key, "B8", .86))
    for row in rows:
        row["visual_plausibility_annotation"] = "visually plausible"
        row["strong_difference"] = True
    _validate(rows, config, target, fillers, order)
    assert killtest.analyze(rows, config, gates_passed=True)["decision"] == "WEAK_INCONCLUSIVE"


def test_output_namespace_cannot_overlap_trusted_experiments(tmp_path):
    config = _config()
    with pytest.raises(killtest.GlobalStopError):
        killtest.validate_output_dir(tmp_path / "video_runtime_state_discovery_v3", config)


def test_anchor_forensics_distinguishes_storage_and_runtime_semantics():
    data = killtest.anchor_forensics(_config())
    assert data["persisted_npy_storage_dtype"] == "float32"
    assert data["runtime_semantics_dtype"] == "torch.bfloat16"
    assert data["anchor"]["flat_index"] == 516515
