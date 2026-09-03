from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import video_bf16_single_flip_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments/video_bf16_single_flip_killtest_config.yaml"
TRUSTED_ROOT = REPO_ROOT / "results/video_runtime_state_discovery_v3_corrected"


def _config() -> dict:
    return killtest.load_config(CONFIG_PATH)


def _bits(value: float) -> int:
    return int(killtest.float32_to_bf16_bits(np.array([value], dtype=np.float32))[0])


def _value(bits: int) -> float:
    return float(killtest.bf16_bits_to_float32(bits)[0])


def _synthetic_clean() -> np.ndarray:
    rng = np.random.default_rng(7)
    clean = rng.normal(scale=0.5, size=(1, 4, 3, 8, 8)).astype(np.float32)
    flat = clean.reshape(-1)
    flat[[5, 77, 200, 401]] = np.array([3e-6, -4.5e-6, 2.4e-6, -3.2e-7], dtype=np.float32)
    flat[[9, 300]] = np.array([5e-5, -7e-5], dtype=np.float32)
    return killtest.base.cast_runtime_bf16(clean)


def _row(flat: int, direction: str, replay_id: int, ssim: float, latent: str = "L", video: str = "V") -> dict:
    return {
        "coordinate_flat_index": flat,
        "direction": direction,
        "requested_direction": direction,
        "replay_id": replay_id,
        "frame_ssim_mean": ssim,
        "runtime_candidate_identity_sha256_v1": f"C{flat}{direction}",
        "recovered_final_latent_identity_sha256_v1": latent,
        "recovered_video_identity_sha256_v1": video,
        "recovered_final_latent_sha256": "legacy",
        "recovered_video_sha256": "legacy",
        "temporal_delta_mse": 0.0,
        "temporal_delta_agreement": 1.0,
        "video_mse": 0.0,
        "final_latent_mse": 0.0,
        "prompt_clip_score": "",
        "historical_fp16_support_member": False,
    }


def _benign_rows(keys: list[tuple[int, str]]) -> list[dict]:
    return [_row(flat, direction, 0, 1.0) for flat, direction in keys]


KEYS = [(5, "down"), (5, "up"), (77, "down"), (77, "up")]


# ----------------------------------------------------------------------------- config


def test_config_loads_and_freezes_anchor():
    config = _config()
    assert (config["anchor"]["prompt_id"], config["anchor"]["generation_seed"], config["anchor"]["checkpoint_step"]) == (
        "recovery_008",
        9234,
        10,
    )
    assert config["perturbation"]["family"] == killtest.PERTURBATION_FAMILY


def test_config_rejects_hard_coded_counts(tmp_path):
    for key in ("eligible_count", "primary_row_count", "eligible_coordinates", "K"):
        config = json.loads(CONFIG_PATH.read_text())
        config["perturbation"][key] = 10
        path = tmp_path / f"{key}.yaml"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="Hard-coded"):
            killtest.load_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["anchor"].__setitem__("checkpoint_step", 20),
        lambda c: c["perturbation"].__setitem__("eligibility_abs_threshold", 1e-4),
        lambda c: c["perturbation"].__setitem__("adjacent_steps", 2),
        lambda c: c["replay"].__setitem__("total_runs_per_triggered_row", 2),
        lambda c: c["analysis"].__setitem__("no_go_frame_ssim_at_least", 0.98),
        lambda c: c["allowed_modes"].append("full"),
        lambda c: c["analysis"]["decision_input_fields"].append("temporal_delta_mse"),
        lambda c: c["controls"].__setitem__("preflight_full_direct_repeats", 2),
        lambda c: c["scheduler"].__setitem__("name", "UniPCMultistepScheduler"),
    ],
)
def test_config_frozen_values(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "mutated.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        killtest.load_config(path)


def test_decision_fields_exclude_descriptive_and_auxiliary():
    config = _config()
    excluded = set(config["analysis"]["descriptive_only_metrics"]) | set(config["analysis"]["auxiliary_only_metrics"])
    assert not (excluded & killtest.DECISION_INPUT_FIELDS)
    assert killtest.decision_source_field_audit()["passed"]


def test_output_namespace_rejects_trusted_roots():
    config = _config()
    for root in (
        config["trusted_v3"]["root"],
        config["validated_error_shape_source"]["root"],
        config["trusted_v3"]["root"] + "/sub",
    ):
        with pytest.raises(killtest.GlobalStopError):
            killtest.validate_output_namespace(config, REPO_ROOT / root)
    killtest.validate_output_namespace(config, REPO_ROOT / config["output_root"])


# ----------------------------------------------------------------------------- BF16 primitive


def test_adjacent_bf16_exhaustive_enumeration():
    report = killtest.verify_adjacent_bf16_by_enumeration()
    assert report["passed"], report
    assert report["up_mismatches"] == 0 and report["down_mismatches"] == 0
    assert report["finite_patterns"] == 65280 and report["unique_finite_values"] == 65279
    assert report["boundary_rejections"] == 2


def test_adjacent_bf16_zero_handling():
    assert killtest.adjacent_bf16_bits(0x0000, "up") == 0x0001
    assert killtest.adjacent_bf16_bits(0x8000, "up") == 0x0001
    assert killtest.adjacent_bf16_bits(0x0000, "down") == 0x8001
    assert killtest.adjacent_bf16_bits(0x8000, "down") == 0x8001
    assert killtest.adjacent_bf16_bits(0x0001, "down") == 0x0000
    assert killtest.adjacent_bf16_bits(0x8001, "up") == 0x0000


def test_adjacent_bf16_negative_direction_semantics():
    bits = _bits(-3.2e-7)
    up = killtest.adjacent_bf16_bits(bits, "up")
    down = killtest.adjacent_bf16_bits(bits, "down")
    assert _value(down) < _value(bits) < _value(up)
    assert (up & 0x7FFF) == (bits & 0x7FFF) - 1  # up on a negative value shrinks magnitude
    assert (down & 0x7FFF) == (bits & 0x7FFF) + 1


def test_adjacent_bf16_rejects_non_finite_and_overflow():
    with pytest.raises(ValueError):
        killtest.adjacent_bf16_bits(0x7F80, "up")  # +inf
    with pytest.raises(ValueError):
        killtest.adjacent_bf16_bits(0x7FC0, "down")  # nan
    with pytest.raises(ValueError):
        killtest.adjacent_bf16_bits(0x7F7F, "up")  # +max -> inf
    with pytest.raises(ValueError):
        killtest.adjacent_bf16_bits(0xFF7F, "down")  # -max -> -inf
    with pytest.raises(ValueError):
        killtest.adjacent_bf16_bits(0x0001, "sideways")


def test_adjacent_bf16_is_involutive_and_single_step():
    rng = np.random.default_rng(3)
    for value in rng.normal(scale=1e-5, size=200).astype(np.float32):
        runtime = float(killtest.base.cast_runtime_bf16(np.array([value], dtype=np.float32))[0])
        bits = _bits(runtime)
        for direction, inverse in (("up", "down"), ("down", "up")):
            neighbour = killtest.adjacent_bf16_bits(bits, direction)
            assert killtest.adjacent_bf16_bits(neighbour, inverse) == bits
            # No representable value lies strictly between: encoding the midpoint returns one of the pair.
            midpoint = np.array([(runtime + _value(neighbour)) / 2.0], dtype=np.float32)
            assert int(killtest.float32_to_bf16_bits(midpoint)[0]) in (bits, neighbour)


def test_adjacent_bf16_matches_torch_nextafter():
    torch = pytest.importorskip("torch")
    sample = torch.tensor([1e-6, -3.2e-7, 1.0, -1.0, 0.0, -2.4e-6, 6.0e-6], dtype=torch.bfloat16)
    inf = torch.tensor(float("inf"), dtype=torch.bfloat16)
    try:
        expected_up = torch.nextafter(sample, inf).float().tolist()
        expected_down = torch.nextafter(sample, -inf).float().tolist()
    except RuntimeError as error:  # pragma: no cover - torch build without bf16 nextafter
        pytest.skip(f"torch.nextafter bf16 unsupported: {error}")
    ours_up = [float(killtest.adjacent_bf16_value(v, "up")[0]) for v in sample.float().tolist()]
    ours_down = [float(killtest.adjacent_bf16_value(v, "down")[0]) for v in sample.float().tolist()]
    assert ours_up == expected_up and ours_down == expected_down


def test_adjacent_bf16_value_rejects_unrepresentable_input():
    with pytest.raises(ValueError, match="not BF16-representable"):
        killtest.adjacent_bf16_value(1.0000001, "up")


def test_bits_roundtrip_matches_runtime_encoding():
    values = np.linspace(-3, 3, 1001, dtype=np.float32)
    runtime = killtest.base.cast_runtime_bf16(values)
    bits = killtest.float32_to_bf16_bits(runtime)
    assert np.array_equal(killtest.bf16_bits_to_float32(bits), runtime)
    assert np.array_equal(bits, killtest.base.encode_runtime_bf16(values))


# ----------------------------------------------------------------------------- derivations


def test_eligible_set_derived_from_runtime_bf16_state():
    clean = _synthetic_clean()
    eligible = killtest.derive_eligible_coordinates(clean, 1e-5)
    expected = np.flatnonzero(np.abs(killtest.base.cast_runtime_bf16(clean).reshape(-1)) < np.float32(1e-5))
    assert np.array_equal(eligible, np.sort(expected))
    assert {5, 77, 200, 401} <= set(eligible.tolist())
    assert not ({9, 300} & set(eligible.tolist()))
    assert killtest.count_below(clean, 1e-4) >= eligible.size + 2


def test_eligible_set_sorted_unique_and_threshold_strict():
    clean = _synthetic_clean()
    flat = clean.reshape(-1)
    flat[13] = np.float32(_value(_bits(1e-5)))  # exactly a BF16 value >= threshold after rounding
    eligible = killtest.derive_eligible_coordinates(clean, 1e-5)
    assert list(eligible) == sorted(set(eligible.tolist()))
    assert (13 in eligible.tolist()) == (abs(float(flat[13])) < 1e-5)


def test_assert_bf16_representable_rejects_rounded_state():
    clean = _synthetic_clean()
    killtest.assert_bf16_representable(clean)
    tampered = clean.copy()
    tampered.reshape(-1)[0] = np.float32(1.0000001)
    with pytest.raises(killtest.GlobalStopError):
        killtest.assert_bf16_representable(tampered)


def test_build_single_flip_changes_exactly_one_coordinate():
    clean = _synthetic_clean()
    for flat, direction in ((5, "up"), (77, "down"), (401, "up"), (0, "down")):
        state, record = killtest.build_single_flip_state(clean, flat, direction)
        assert state.shape == clean.shape and state.dtype == np.float32
        changed = np.flatnonzero(killtest.float32_to_bf16_bits(state) != killtest.float32_to_bf16_bits(clean))
        assert changed.tolist() == [flat]
        assert record["changed_coordinate_count"] == 1
        assert record["coordinate_flat_index"] == flat and record["direction"] == direction
        assert (record["perturbed_value"] > record["clean_value"]) == (direction == "up")
        # Result is exactly BF16-representable: encoding does not move it.
        assert np.array_equal(killtest.base.cast_runtime_bf16(state), state)
        assert record["perturbed_value"] == _value(killtest.adjacent_bf16_bits(_bits(record["clean_value"]), direction))


def test_build_single_flip_input_hashes_distinct_and_deterministic():
    clean = _synthetic_clean()
    eligible = killtest.derive_eligible_coordinates(clean, 1e-5)
    records = [killtest.build_single_flip_state(clean, f, d)[1] for f, d in killtest.expected_primary_keys(eligible)]
    hashes = [row["runtime_input_hash"] for row in records]
    assert len(set(hashes)) == len(hashes)
    again = [killtest.build_single_flip_state(clean, f, d)[1]["runtime_input_hash"] for f, d in killtest.expected_primary_keys(eligible)]
    assert again == hashes


def test_build_single_flip_realized_error_accounting():
    clean = _synthetic_clean()
    for flat, direction in ((5, "up"), (77, "down")):
        state, record = killtest.build_single_flip_state(clean, flat, direction)
        difference = state.astype(np.float64) - clean.astype(np.float64)
        assert record["realized_nonzero_elements"] == 1 and record["total_elements"] == clean.size
        assert record["realized_linf"] == abs(record["delta"]) == float(np.max(np.abs(difference)))
        assert record["realized_l2"] == record["realized_linf"]
        assert record["realized_mse"] == record["delta"] ** 2 / clean.size
    for field in ("realized_l2", "realized_mse", "realized_linf", "realized_nonzero_elements", "total_elements"):
        assert field in killtest.RAW_FIELDS


def test_build_single_flip_rejects_out_of_range_and_unrepresentable():
    clean = _synthetic_clean()
    with pytest.raises(ValueError):
        killtest.build_single_flip_state(clean, clean.size, "up")
    tampered = clean.copy()
    tampered.reshape(-1)[3] = np.float32(1.0000001)
    with pytest.raises(killtest.GlobalStopError):
        killtest.build_single_flip_state(tampered, 5, "up")


def test_historical_delta_derivation_from_synthetic_pair():
    clean = _synthetic_clean()
    probe = clean.copy()
    flat = probe.reshape(-1)
    steps = {5: ("up", 1), 77: ("up", 4), 200: ("down", 2)}
    for index, (direction, count) in steps.items():
        bits = _bits(float(flat[index]))
        for _ in range(count):
            bits = killtest.adjacent_bf16_bits(bits, direction)
        flat[index] = np.float32(_value(bits))
    historical = killtest.derive_historical_delta(clean, probe)
    assert historical["changed_coordinate_count"] == 3
    derived = {row["coordinate_flat_index"]: (row["direction"], row["adjacent_steps"]) for row in historical["changed_coordinates"]}
    assert derived == steps
    assert historical["single_adjacent_step_count"] == 1
    assert historical["delta_l2"] > 0
    with pytest.raises(killtest.GlobalStopError):
        killtest.derive_historical_delta(clean, probe.reshape(1, 4, 3, 64))


def test_historical_delta_of_identical_states_is_empty():
    clean = _synthetic_clean()
    historical = killtest.derive_historical_delta(clean, clean.copy())
    assert historical["changed_coordinate_count"] == 0 and historical["delta_l2"] == 0.0


def test_expected_primary_keys_and_condition_ids():
    eligible = np.array([5, 77, 200], dtype=np.int64)
    keys = killtest.expected_primary_keys(eligible)
    assert keys == [(5, "down"), (5, "up"), (77, "down"), (77, "up"), (200, "down"), (200, "up")]
    assert killtest.condition_id(66153, "up") == "c0066153_up"
    with pytest.raises(ValueError):
        killtest.condition_id(1, "left")


# ----------------------------------------------------------------------------- decision logic


def test_classify_row_group_classes():
    config = _config()
    assert killtest.classify_row_group([_row(5, "up", 0, 0.995)], config)["row_class"] == "BENIGN"
    assert killtest.classify_row_group([_row(5, "up", 0, 0.97)], config)["row_class"] == "INTERMEDIATE"
    assert killtest.classify_row_group([_row(5, "up", 0, 0.95)], config)["row_class"] == "INTERMEDIATE"
    assert killtest.classify_row_group([_row(5, "up", 0, 0.9)], config)["row_class"] == "TRIGGERED_INCOMPLETE"
    deterministic = [_row(5, "up", r, 0.6) for r in range(3)]
    assert killtest.classify_row_group(deterministic, config)["row_class"] == "CATASTROPHIC_DETERMINISTIC"
    different_hash = [_row(5, "up", 0, 0.6), _row(5, "up", 1, 0.6), _row(5, "up", 2, 0.6, latent="L2")]
    assert killtest.classify_row_group(different_hash, config)["row_class"] == "CATASTROPHIC_NONDETERMINISTIC"
    inconsistent = [_row(5, "up", 0, 0.6), _row(5, "up", 1, 0.99), _row(5, "up", 2, 0.6)]
    assert killtest.classify_row_group(inconsistent, config)["row_class"] == "CATASTROPHIC_NONDETERMINISTIC"


def test_classify_requires_identical_metrics_across_replays():
    config = _config()
    same_hash_different_ssim = [_row(5, "down", 0, 0.80), _row(5, "down", 1, 0.81), _row(5, "down", 2, 0.82)]
    summary = killtest.classify_row_group(same_hash_different_ssim, config)
    assert summary["row_class"] == "CATASTROPHIC_NONDETERMINISTIC"
    assert summary["identical_frame_ssim_across_runs"] is False and summary["bit_deterministic_across_runs"] is False
    rows = _benign_rows(KEYS[1:]) + same_hash_different_ssim
    assert killtest.analyze_rows(rows, config, KEYS, controls_passed=True)["decision"] == "WEAK_INCONCLUSIVE"


def test_replay_equality_fields_govern_determinism_only_across_replays():
    config = _config()
    triple = [_row(5, "down", r, 0.5) for r in range(3)]
    assert killtest.classify_row_group(triple, config)["row_class"] == "CATASTROPHIC_DETERMINISTIC"
    for field in killtest.REPLAY_EQUALITY_FIELDS:
        rows = copy.deepcopy(triple)
        rows[1][field] = 0.123 if isinstance(rows[1][field], float) else "other"
        summary = killtest.classify_row_group(rows, config)
        assert summary["row_class"] == "CATASTROPHIC_NONDETERMINISTIC", field
        assert summary["replay_equality_by_field"][field] is False
    # On non-triggered rows MSE values cannot move the decision in either direction.
    benign = _benign_rows(KEYS)
    for row in benign:
        row["video_mse"] = 1e9
        row["final_latent_mse"] = 1e9
    assert killtest.analyze_rows(benign, config, KEYS, controls_passed=True)["decision"] == "NO_GO"
    assert list(config["replay"]["replay_equality_fields"]) == list(killtest.REPLAY_EQUALITY_FIELDS)


def test_classify_row_group_rejects_extra_runs_on_non_triggered_row():
    config = _config()
    with pytest.raises(killtest.GlobalStopError):
        killtest.classify_row_group([_row(5, "up", 0, 0.99), _row(5, "up", 1, 0.99)], config)
    with pytest.raises(killtest.GlobalStopError):
        killtest.classify_row_group([_row(5, "up", 1, 0.6)], config)
    with pytest.raises(killtest.GlobalStopError):
        killtest.classify_row_group([_row(5, "up", 0, float("nan"))], config)


def test_decision_no_go_when_all_rows_benign():
    result = killtest.analyze_rows(_benign_rows(KEYS), _config(), KEYS, controls_passed=True)
    assert result["decision"] == "NO_GO"
    assert result["row_class_counts"]["BENIGN"] == 4


def test_decision_go_on_single_deterministic_catastrophic_row():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", r, 0.5) for r in range(3)]
    result = killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)
    assert result["decision"] == "GO_TO_LOCAL_BRANCH_MAP"
    assert result["row_class_counts"]["CATASTROPHIC_DETERMINISTIC"] == 1


def test_decision_weak_when_catastrophic_but_non_deterministic():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.5), _row(5, "down", 1, 0.5, latent="other"), _row(5, "down", 2, 0.5)]
    assert killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)["decision"] == "WEAK_INCONCLUSIVE"
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.5), _row(5, "down", 1, 0.999), _row(5, "down", 2, 0.5)]
    assert killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)["decision"] == "WEAK_INCONCLUSIVE"


def test_decision_weak_on_intermediate_row_only():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.97)]
    result = killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)
    assert result["decision"] == "WEAK_INCONCLUSIVE"
    assert result["row_class_counts"]["INTERMEDIATE"] == 1


def test_decision_weak_when_controls_failed_even_with_go_pattern():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", r, 0.5) for r in range(3)]
    result = killtest.analyze_rows(rows, _config(), KEYS, controls_passed=False)
    assert result["decision"] == "WEAK_INCONCLUSIVE"
    assert result["row_class_counts"]["CATASTROPHIC_DETERMINISTIC"] == 1  # pattern present, controls veto
    assert killtest.analyze_rows(_benign_rows(KEYS), _config(), KEYS, controls_passed=False)["decision"] == "WEAK_INCONCLUSIVE"


def test_decision_weak_when_triggered_row_lacks_replays():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.5)]
    result = killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)
    assert result["decision"] == "WEAK_INCONCLUSIVE"
    assert result["row_class_counts"]["TRIGGERED_INCOMPLETE"] == 1


def test_decision_rejects_missing_or_extra_keys():
    config = _config()
    with pytest.raises(killtest.GlobalStopError, match="missing"):
        killtest.analyze_rows(_benign_rows(KEYS[:-1]), config, KEYS, controls_passed=True)
    with pytest.raises(killtest.GlobalStopError, match="extra"):
        killtest.analyze_rows(_benign_rows(KEYS + [(999, "up")]), config, KEYS, controls_passed=True)


def test_descriptive_metrics_cannot_rescue_or_sink_decision():
    config = _config()
    benign = _benign_rows(KEYS)
    for row in benign:
        row["video_mse"] = 1e9
        row["temporal_delta_mse"] = 1e9
        row["temporal_delta_agreement"] = 0.0
        row["final_latent_mse"] = 1e9
        row["prompt_clip_score"] = 0.0
        row["historical_fp16_support_member"] = True
    assert killtest.analyze_rows(benign, config, KEYS, controls_passed=True)["decision"] == "NO_GO"
    go_rows = _benign_rows(KEYS[1:]) + [_row(5, "down", r, 0.5) for r in range(3)]
    for row in go_rows:
        row["video_mse"] = 0.0
        row["temporal_delta_mse"] = 0.0
        row["final_latent_mse"] = 0.0
        row["prompt_clip_score"] = 1.0
    assert killtest.analyze_rows(go_rows, config, KEYS, controls_passed=True)["decision"] == "GO_TO_LOCAL_BRANCH_MAP"


def test_decision_threshold_boundaries():
    config = _config()
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.99)]
    assert killtest.analyze_rows(rows, config, KEYS, controls_passed=True)["decision"] == "NO_GO"
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", 0, 0.9899999)]
    assert killtest.analyze_rows(rows, config, KEYS, controls_passed=True)["decision"] == "WEAK_INCONCLUSIVE"
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", r, 0.9499999) for r in range(3)]
    assert killtest.analyze_rows(rows, config, KEYS, controls_passed=True)["decision"] == "GO_TO_LOCAL_BRANCH_MAP"


def test_decision_uses_only_preregistered_fields_ast_sweep():
    audit = killtest.decision_source_field_audit()
    assert audit["passed"] and audit["forbidden_fields_referenced"] == []
    assert set(_config()["analysis"]["decision_input_fields"]) == killtest.DECISION_INPUT_FIELDS


def test_decision_direction_descriptives_do_not_alter_decision():
    rows = _benign_rows(KEYS[1:]) + [_row(5, "down", r, 0.5) for r in range(3)]
    result = killtest.analyze_rows(rows, _config(), KEYS, controls_passed=True)
    assert result["by_direction_descriptive"]["down"]["catastrophic_deterministic"] == 1
    assert result["by_direction_descriptive"]["up"]["catastrophic_deterministic"] == 0


# ----------------------------------------------------------------------------- provenance / results


def test_provenance_is_deterministic_and_covers_trusted_sources():
    first = killtest.build_provenance(CONFIG_PATH)
    second = killtest.build_provenance(CONFIG_PATH)
    assert first == second
    for key in ("error_shape_script_sha256", "error_shape_config_sha256", "trusted_v3_raw_sha256", "runner_sha256", "scheduler_sha256"):
        assert first[key]
    config = _config()
    assert first["trusted_v3_raw_sha256"] == config["trusted_v3"]["raw_results_sha256"]
    assert first["error_shape_config_sha256"] == config["validated_error_shape_source"]["config_file_sha256"]
    # git status is restricted to the files that define the experiment.
    assert "git_status" not in first
    relevant = set(first["relevant_paths"])
    for line in first["git_status_relevant"]:
        assert line.split()[-1] in relevant, line
    assert isinstance(first["trusted_v3_resolved_model_revision"], str) and len(first["trusted_v3_resolved_model_revision"]) == 40


def _passing_gates(name: str) -> list[dict]:
    return [killtest._gate(gate, True, {"v": 1}, "x") for gate in sorted(killtest.REQUIRED_GATE_NAMES[name])]


def test_require_mode_gate_fails_closed(tmp_path):
    provenance = killtest.build_provenance(CONFIG_PATH)
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_mode_gate(tmp_path, "cpu_gates.json", provenance, "m" * 64)
    killtest.atomic_json(tmp_path / "run_provenance.json", provenance)
    with pytest.raises(killtest.GlobalStopError, match="did not pass"):
        killtest.require_mode_gate(tmp_path, "cpu_gates.json", provenance, "m" * 64)
    killtest._write_gates(tmp_path / "cpu_gates.json", _passing_gates("cpu_gates.json"), provenance, "m" * 64)
    killtest.require_mode_gate(tmp_path, "cpu_gates.json", provenance, "m" * 64)  # genuine file passes
    stale = dict(provenance, provenance_hash="0" * 64)
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_mode_gate(tmp_path, "cpu_gates.json", stale, "m" * 64)
    with pytest.raises(killtest.GlobalStopError, match="another anchor manifest"):
        killtest.require_mode_gate(tmp_path, "cpu_gates.json", provenance, "n" * 64)
    with pytest.raises(killtest.GlobalStopError, match="unknown"):
        killtest.require_mode_gate(tmp_path, "other_gates.json", provenance, "m" * 64)


def test_require_mode_gate_rejects_tampered_gate_files(tmp_path):
    provenance = killtest.build_provenance(CONFIG_PATH)
    killtest.atomic_json(tmp_path / "run_provenance.json", provenance)
    path = tmp_path / "preflight_gates.json"
    # The previously accepted forgery: all_passed without any gates.
    killtest.atomic_json(path, {"all_passed": True, "gates": []})
    with pytest.raises(killtest.GlobalStopError, match="required gate set"):
        killtest.require_mode_gate(tmp_path, "preflight_gates.json", provenance, "m" * 64)
    killtest._write_gates(path, _passing_gates("preflight_gates.json"), provenance, "m" * 64)
    genuine = json.loads(path.read_text())
    # Drop the FULL-direct control gate but keep all_passed.
    forged = copy.deepcopy(genuine)
    forged["gates"] = [g for g in forged["gates"] if not g["name"].startswith("G12")]
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="required gate set"):
        killtest.require_mode_gate(tmp_path, "preflight_gates.json", provenance, "m" * 64)
    # Flip a FAIL to PASS in place: content hash no longer matches.
    forged = copy.deepcopy(genuine)
    forged["gates"][0]["measured_evidence"] = {"v": 2}
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="content hash"):
        killtest.require_mode_gate(tmp_path, "preflight_gates.json", provenance, "m" * 64)
    # A genuine file from another provenance is rejected.
    forged = copy.deepcopy(genuine)
    forged["provenance_hash"] = "1" * 64
    forged["gates_sha256"] = killtest._gate_document_hash(forged["gates"], "1" * 64, "m" * 64)
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="other provenance"):
        killtest.require_mode_gate(tmp_path, "preflight_gates.json", provenance, "m" * 64)
    # A failed gate written honestly is never accepted even if all_passed is edited to true.
    gates = _passing_gates("preflight_gates.json")
    gates[3] = killtest._gate(gates[3]["name"], False, {}, "x")
    with pytest.raises(killtest.GlobalStopError):
        killtest._write_gates(path, gates, provenance, "m" * 64)
    forged = json.loads(path.read_text())
    forged["all_passed"] = True
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_mode_gate(tmp_path, "preflight_gates.json", provenance, "m" * 64)


EXPECTED_KEY = (5, "up")


def _expected_construction() -> dict:
    return killtest.build_single_flip_state(_synthetic_clean(), *EXPECTED_KEY)[1]


def _consistent_row(tmp_path: Path, tag: str = "a", latent: np.ndarray | None = None) -> dict:
    candidate = killtest.build_single_flip_state(_synthetic_clean(), *EXPECTED_KEY)[0]
    candidate_record = killtest._save_array(tmp_path / f"cand_{tag}.npy", candidate)
    if latent is None:
        latent = np.full((2, 2), float(ord(tag[0])), dtype=np.float32)
    video = np.full((2, 4, 4, 3), ord(tag[0]), dtype=np.uint8)
    latent_record = killtest._save_array(tmp_path / f"lat_{tag}.npy", latent)
    video_record = killtest._save_array(tmp_path / f"vid_{tag}.npy", video)
    return {
        "status": "COMPLETE",
        "provenance_hash": "abc",
        "condition_id": "c0000005_up",
        "coordinate_flat_index": 5,
        "direction": "up",
        "requested_direction": "up",
        "runtime_input_hash": killtest.sha256_bytes(killtest.float32_to_bf16_bits(candidate).tobytes()),
        "runtime_candidate_artifact": candidate_record,
        "recovered_final_latent_artifact": latent_record,
        "recovered_video_artifact": video_record,
        "runtime_candidate_identity_sha256_v1": candidate_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_identity_sha256_v1": latent_record["tensor_identity_sha256_v1"],
        "recovered_video_identity_sha256_v1": video_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_sha256": latent_record["tensor_sha256"],
        "recovered_video_sha256": video_record["tensor_sha256"],
    }


def test_result_valid_rejects_stale_and_mismatched_rows(tmp_path):
    provenance = {"provenance_hash": "abc"}
    path = tmp_path / "result.json"
    assert killtest._result_valid(path, provenance, {}, _expected_construction()) is None
    row = _consistent_row(tmp_path)
    killtest.atomic_json(path, row)
    assert killtest._result_valid(path, provenance, {"condition_id": "c0000005_up"}, _expected_construction()) == row
    with pytest.raises(killtest.GlobalStopError, match="identity"):
        killtest._result_valid(path, provenance, {"condition_id": "c0000005_down"}, _expected_construction())
    with pytest.raises(killtest.GlobalStopError, match="stale"):
        killtest._result_valid(path, {"provenance_hash": "zzz"}, {}, _expected_construction())
    killtest.atomic_json(path, dict(row, status="RUNNING"))
    with pytest.raises(killtest.GlobalStopError, match="incomplete"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    killtest.atomic_json(path, row)
    (tmp_path / "cand_a.npy").write_bytes(b"corrupt")
    with pytest.raises(killtest.GlobalStopError, match="artifact"):
        killtest._result_valid(path, provenance, {}, _expected_construction())


def test_result_valid_binds_top_level_hashes_to_persisted_artifacts(tmp_path):
    provenance = {"provenance_hash": "abc"}
    path = tmp_path / "result.json"
    row = _consistent_row(tmp_path, "a")
    other = _consistent_row(tmp_path, "b")
    # Copied top-level hashes with different real artifacts: previously accepted, now refused.
    forged = dict(row, recovered_final_latent_artifact=other["recovered_final_latent_artifact"])
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="latent identity is not derived"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    forged = dict(row, recovered_video_artifact=other["recovered_video_artifact"])
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="video identity is not derived"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    forged = dict(row, runtime_input_hash="0" * 64)
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="runtime_input_hash"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    forged = dict(row, recovered_video_sha256=other["recovered_video_sha256"])
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="video hash is not derived"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    forged = dict(row, recovered_final_latent_identity_sha256_v1=other["recovered_final_latent_identity_sha256_v1"])
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="latent identity is not derived"):
        killtest._result_valid(path, provenance, {}, _expected_construction())


# ----------------------------------------------------------------------------- canonical tensor identity


def test_tensor_identity_separates_shape_and_dtype_with_identical_bytes():
    flat = np.arange(4, dtype=np.float32)
    square = flat.reshape(2, 2)
    as_uint = flat.view(np.uint32)
    assert killtest.v3.array_sha256(flat) == killtest.v3.array_sha256(square) == killtest.v3.array_sha256(as_uint)  # legacy collides
    identities = {killtest.tensor_identity_sha256_v1(a) for a in (flat, square, as_uint)}
    assert len(identities) == 3
    assert killtest.tensor_identity_sha256_v1(square) == killtest.tensor_identity_sha256_v1(np.asfortranarray(square).copy(order="C"))
    assert killtest.tensor_identity_sha256_v1(square) == killtest.tensor_identity_sha256_v1(square.astype(">f4"))  # byte order canonicalised
    assert len(killtest.tensor_identity_sha256_v1(flat)) == 64


def test_result_valid_rejects_same_bytes_different_shape_or_dtype(tmp_path):
    provenance = {"provenance_hash": "abc"}
    path = tmp_path / "result.json"
    latent = np.arange(4, dtype=np.float32).reshape(2, 2)
    row = _consistent_row(tmp_path, "a", latent=latent)
    killtest.atomic_json(path, row)
    assert killtest._result_valid(path, provenance, {}, _expected_construction()) == row
    # A: identical raw bytes, shape (4,) persisted where (2,2) is declared.
    np.save(tmp_path / "lat_a.npy", latent.reshape(4), allow_pickle=False)
    with pytest.raises(killtest.GlobalStopError, match="artifact"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    # B: identical raw bytes, dtype uint32 persisted.
    np.save(tmp_path / "lat_a.npy", latent.view(np.uint32), allow_pickle=False)
    with pytest.raises(killtest.GlobalStopError, match="artifact"):
        killtest._result_valid(path, provenance, {}, _expected_construction())
    np.save(tmp_path / "lat_a.npy", latent, allow_pickle=False)
    assert killtest._result_valid(path, provenance, {}, _expected_construction()) == row
    # Same declared legacy hash, but declared shape / dtype changed in the record.
    for key, value in (("shape", [4]), ("dtype", "<u4")):
        forged = copy.deepcopy(row)
        forged["recovered_final_latent_artifact"][key] = value
        killtest.atomic_json(path, forged)
        with pytest.raises(killtest.GlobalStopError, match="artifact"):
            killtest._result_valid(path, provenance, {}, _expected_construction())
    # Same declared legacy hash with an artifact of another shape whose record is self-consistent: identity mismatch.
    forged = copy.deepcopy(row)
    other_record = killtest._save_array(tmp_path / "lat_flat.npy", latent.reshape(4))
    assert other_record["tensor_sha256"] == row["recovered_final_latent_sha256"]
    forged["recovered_final_latent_artifact"] = other_record
    killtest.atomic_json(path, forged)
    with pytest.raises(killtest.GlobalStopError, match="latent identity is not derived"):
        killtest._result_valid(path, provenance, {}, _expected_construction())


def test_accounting_policy_is_frozen():
    assert killtest.ACCOUNTING_RELATIVE_TOLERANCE == 1e-12
    assert killtest._accounting_matches(1.0, 1.0) and killtest._accounting_matches(1.0 + 1e-14, 1.0)
    assert not killtest._accounting_matches(1.0 + 1e-9, 1.0)
    assert killtest._accounting_matches(0.0, 0.0) and not killtest._accounting_matches(1e-300, 0.0)
    assert not killtest._accounting_matches(float("nan"), 1.0) and not killtest._accounting_matches("x", 1.0)


def test_scheduler_timesteps_match_trusted_stability_derivation():
    from experiments import video_checkpoint_stability_killtest as ck

    config = _config()
    ours = killtest.scheduler_timesteps_numpy(config)
    assert ours == ck.scheduler_timesteps_numpy(config)
    assert killtest.anchor_resume_timestep(config, 10) == 972.9729614257812


def test_gate_writer_fails_closed(tmp_path):
    provenance = {"provenance_hash": "p" * 64}
    gates = [killtest._gate("G0 demo", False, {"x": 1}, "expected")]
    with pytest.raises(killtest.GlobalStopError):
        killtest._write_gates(tmp_path / "gates.json", gates, provenance, "m" * 64)
    written = json.loads((tmp_path / "gates.json").read_text())
    assert written["all_passed"] is False and written["provenance_hash"] == "p" * 64
    assert written["anchor_manifest_sha256"] == "m" * 64
    assert written["gates_sha256"] == killtest._gate_document_hash(written["gates"], "p" * 64, "m" * 64)
    with pytest.raises(killtest.GlobalStopError, match="preregistered gate set"):
        killtest._write_gates(tmp_path / "cpu_gates.json", _passing_gates("smoke_gates.json"), provenance, "m" * 64)


def test_anchor_manifest_sha_is_deterministic():
    clean = _synthetic_clean()
    config = _config()

    class Source:
        prompt_id, seed, checkpoint_step, prompt = "p", 1, 10, "text"
        clean_hash, checkpoint_path, manifest_path = "h", Path("ck"), Path("mf")

    Source.clean = clean
    Source.final_latent = np.zeros_like(clean)
    Source.video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    eligible = killtest.derive_eligible_coordinates(clean, 1e-5)
    historical = killtest.derive_historical_delta(clean, clean.copy())
    meta = {"original_v3_fp16_probe_sha256": "a", "frozen_reconstructed_fp16_runtime_sha256": "b"}
    constructions = [killtest.build_single_flip_state(clean, f, d)[1] for f, d in killtest.expected_primary_keys(eligible)]
    first = killtest.build_anchor_manifest(config, Source, meta, eligible, historical, constructions)
    second = killtest.build_anchor_manifest(config, Source, meta, eligible, historical, copy.deepcopy(constructions))
    assert first == second and len(first["manifest_sha256"]) == 64
    assert first["primary_row_count"] == 2 * eligible.size
    assert first["eligibility"]["eligible_count"] == eligible.size


# ----------------------------------------------------------------------------- trusted-data integration (read-only)

needs_trusted = pytest.mark.skipif(not (TRUSTED_ROOT / "raw_results.csv").exists(), reason="trusted v3 artifacts absent")


@needs_trusted
def test_trusted_source_hashes_validate():
    evidence = killtest.validate_source_hashes(_config())
    assert all(row["actual"] == row["expected"] for row in evidence.values())


@needs_trusted
def test_anchor_reconstruction_and_derivations_on_trusted_data():
    config = _config()
    derived = killtest.derive_all(config)
    source = derived["source"]
    assert (source.prompt_id, source.seed, source.checkpoint_step) == ("recovery_008", 9234, 10)
    eligible = derived["eligible"]
    historical = derived["historical"]
    # Counts are derived, not asserted against a preset list; only structural facts are checked.
    assert eligible.size > 0
    assert historical["changed_coordinate_count"] > 0
    assert {row["coordinate_flat_index"] for row in historical["changed_coordinates"]} <= set(eligible.tolist())
    assert historical["probe_runtime_sha256"] == derived["meta"]["frozen_reconstructed_fp16_runtime_sha256"]
    assert len(derived["constructions"]) == 2 * eligible.size
    assert all(row["changed_coordinate_count"] == 1 for row in derived["constructions"])
    assert derived["manifest"]["anchor"]["runtime_full_bytes"] == source.clean.size * 2


@needs_trusted
def test_cpu_mode_writes_frozen_artifacts_and_gates(tmp_path):
    config = _config()
    result = killtest.run_cpu_mode(config, CONFIG_PATH, tmp_path / "out")
    assert result["all_passed"] and result["primary_row_count"] == 2 * result["eligible_count"]
    gates = json.loads((tmp_path / "out/cpu_gates.json").read_text())
    assert gates["all_passed"]
    names = {gate["name"].split()[0] for gate in gates["gates"]}
    assert {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G15", "G19", "G20"} <= names
    keys = json.loads((tmp_path / "out/expected_primary_keys.json").read_text())
    assert keys["primary_row_count"] == len(keys["keys"]) == len(keys["runtime_input_hashes"])
    manifest = json.loads((tmp_path / "out/anchor_manifest.json").read_text())
    assert manifest["eligibility"]["eligible_count"] == result["eligible_count"]
    # A second derivation must reproduce the manifest exactly (preflight/smoke depend on this).
    assert killtest.derive_all(config)["manifest"] == manifest


# ----------------------------------------------------------------------------- GPU-mode code paths with a faked resume (no GPU)


def _fake_pipeline(monkeypatch, catastrophic: set[str], *, nondeterministic: bool = False, clean_exact: bool = True):
    """Replace the generator with a deterministic stand-in; everything else runs for real."""
    counter = {"calls": 0}
    ssim_by_video_hash: dict[str, float] = {}

    def fake_build_omni(config, args):
        return object()

    def fake_scheduler_document(config):
        return {"scheduler_class": killtest.EXPECTED_SCHEDULER_CLASS, "timesteps": killtest.scheduler_timesteps_numpy(config)}

    def fake_video_metrics(candidate, reference):
        if np.array_equal(candidate, reference):
            ssim = 1.0
        else:
            ssim = ssim_by_video_hash[killtest.tensor_identity_sha256_v1(candidate)]
        return {"frame_ssim_mean": ssim, "video_mse": 0.0 if ssim == 1.0 else 1.0 - ssim, "video_psnr": "inf", "temporal_delta_mse": 0.0, "temporal_delta_agreement": 1.0}

    def fake_latent_error(clean, candidate):
        return {"mse": 0.0 if clean.shape == candidate.shape and np.array_equal(clean, candidate) else 1.0}

    def fake_run_resume(omni, config, source, runtime_candidate, *, step_index, label, directory):
        counter["calls"] += 1
        directory.mkdir(parents=True, exist_ok=True)
        runtime_hash = killtest.sha256_bytes(killtest.float32_to_bf16_bits(runtime_candidate).tobytes())
        clean_hash = killtest.sha256_bytes(killtest.float32_to_bf16_bits(source.clean).tobytes())
        exact = runtime_hash == clean_hash and clean_exact
        condition = next((c for c in catastrophic if c in label), None)
        if exact:
            ssim, tag = 1.0, "clean"
        elif condition is not None:
            ssim, tag = 0.5, f"cat-{condition}" + (f"-{counter['calls']}" if nondeterministic else "")
        else:
            ssim, tag = 0.999, f"benign-{runtime_hash[:8]}"
        if exact:
            latent, video = source.final_latent, source.video
        else:
            digest = np.frombuffer(bytes.fromhex(killtest.sha256_bytes(tag.encode())), dtype=np.uint8)
            latent = digest[:16].astype(np.float32).reshape(4, 4)
            video = digest[16:32].reshape(4, 4).copy()
        ssim_by_video_hash[killtest.tensor_identity_sha256_v1(video)] = ssim
        # Trusted base code produces records WITHOUT canonical identity; the experiment must add it.
        latent_record = killtest.base._save_array(directory / "recovered_final_latent.npy", latent)
        video_record = killtest.base._save_array(directory / "recovered_video.npy", video)
        return {
            "scheduler_class": killtest.EXPECTED_SCHEDULER_CLASS,
            "sample_solver": "euler",
            "resume_index": step_index,
            "resume_ms": 1.0,
            "final_latent_mse": fake_latent_error(source.final_latent, latent)["mse"],
            "exact_final_latent": exact,
            "exact_video": exact,
            "recovered_final_latent_sha256": latent_record["tensor_sha256"],
            "recovered_video_sha256": video_record["tensor_sha256"],
            "recovered_final_latent_artifact": latent_record,
            "recovered_video_artifact": video_record,
            "video_mse": fake_video_metrics(video, source.video)["video_mse"],
            "video_psnr": "inf",
            "frame_ssim_mean": ssim,
            "temporal_delta_mse": 0.0,
            "temporal_delta_agreement": 1.0,
        }

    monkeypatch.setattr(killtest.v3, "build_omni", fake_build_omni)
    monkeypatch.setattr(killtest.v3, "scheduler_document", fake_scheduler_document)
    monkeypatch.setattr(killtest.v3, "video_metrics", fake_video_metrics)
    monkeypatch.setattr(killtest.v3, "latent_error", fake_latent_error)
    monkeypatch.setattr(killtest.base, "_metric_control_result", lambda reference: {"passed": True, "faked": True})
    monkeypatch.setattr(killtest.base, "run_resume", fake_run_resume)
    return counter


class _Args:
    enable_cpu_offload = True
    enforce_eager = False


@needs_trusted
def test_pipeline_go_path_with_replays(monkeypatch, tmp_path):
    config = _config()
    out = tmp_path / "out"
    cpu = killtest.run_cpu_mode(config, CONFIG_PATH, out)
    first_key = json.loads((out / "expected_primary_keys.json").read_text())["keys"][0]["condition_id"]
    counter = _fake_pipeline(monkeypatch, {first_key})
    preflight = killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    assert preflight["full_direct_controls"] == 3
    smoke = killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    assert smoke["keys"] == cpu["primary_row_count"]
    assert smoke["rows"] == cpu["primary_row_count"] + 2  # one triggered row earned two replays
    assert counter["calls"] == 3 + 1 + cpu["primary_row_count"] + 2
    result = killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    assert result["decision"] == "GO_TO_LOCAL_BRANCH_MAP"
    assert result["row_class_counts"] == {**{c: 0 for c in killtest.ROW_CLASSES}, "BENIGN": cpu["primary_row_count"] - 1, "CATASTROPHIC_DETERMINISTIC": 1}
    # Resume safety: a second smoke call re-uses every completed row and runs nothing new.
    before = counter["calls"]
    killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    assert counter["calls"] == before + 1  # only the in-run FULL-direct control repeats
    rows = killtest.read_csv(out / "smoke_raw_results.csv")
    assert set(killtest.RAW_FIELDS) <= set(rows[0])
    assert all(row["changed_coordinate_count"] == "1" for row in rows)


@needs_trusted
def test_pipeline_no_go_and_nondeterministic_paths(monkeypatch, tmp_path):
    config = _config()
    out = tmp_path / "out"
    killtest.run_cpu_mode(config, CONFIG_PATH, out)
    _fake_pipeline(monkeypatch, set())
    killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "NO_GO"

    out2 = tmp_path / "out2"
    killtest.run_cpu_mode(config, CONFIG_PATH, out2)
    first_key = json.loads((out2 / "expected_primary_keys.json").read_text())["keys"][0]["condition_id"]
    _fake_pipeline(monkeypatch, {first_key}, nondeterministic=True)
    killtest.run_preflight(config, CONFIG_PATH, out2, _Args())
    killtest.run_smoke(config, CONFIG_PATH, out2, _Args())
    result = killtest.run_analyze_smoke(config, CONFIG_PATH, out2)
    assert result["decision"] == "WEAK_INCONCLUSIVE"
    assert result["row_class_counts"]["CATASTROPHIC_NONDETERMINISTIC"] == 1


@needs_trusted
def test_pipeline_global_stops(monkeypatch, tmp_path):
    config = _config()
    out = tmp_path / "out"
    with pytest.raises(killtest.GlobalStopError, match="missing|provenance"):
        killtest.run_preflight(config, CONFIG_PATH, out, _Args())  # no CPU mode yet: fail closed
    killtest.run_cpu_mode(config, CONFIG_PATH, out)
    _fake_pipeline(monkeypatch, set(), clean_exact=False)
    with pytest.raises(killtest.GlobalStopError, match="FULL-direct"):
        killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    with pytest.raises(killtest.GlobalStopError, match="prerequisite"):
        killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    _fake_pipeline(monkeypatch, set())
    killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    # Tamper with the frozen manifest: every later mode must refuse.
    manifest = json.loads((out / "anchor_manifest.json").read_text())
    manifest["eligibility"]["eligible_flat_indices"] = manifest["eligibility"]["eligible_flat_indices"][:-1]
    killtest.atomic_json(out / "anchor_manifest.json", manifest)
    with pytest.raises(killtest.GlobalStopError, match="manifest changed"):
        killtest.run_smoke(config, CONFIG_PATH, out, _Args())


@needs_trusted
def test_analysis_rejects_tampered_rows_and_gate_files(monkeypatch, tmp_path):
    config = _config()
    out = tmp_path / "out"
    killtest.run_cpu_mode(config, CONFIG_PATH, out)
    keys = json.loads((out / "expected_primary_keys.json").read_text())["keys"]
    catastrophic, benign = keys[0]["condition_id"], keys[1]["condition_id"]
    _fake_pipeline(monkeypatch, {catastrophic})
    killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "GO_TO_LOCAL_BRANCH_MAP"

    benign_path = out / "smoke/rows" / benign / "replay_00/result.json"
    cat_path = out / "smoke/rows" / catastrophic / "replay_00/result.json"
    genuine = benign_path.read_text()
    row = json.loads(genuine)

    def restore():
        benign_path.write_text(genuine)

    # Wrong anchor identity on an otherwise valid row (includes exploits C and D).
    for field, value in (
        ("prompt_id", "recovery_001"),
        ("generation_seed", 1),
        ("checkpoint_step", 20),
        ("resume_index", 11),
        ("resume_timestep", 923.076904296875),
        ("scheduler_class", "fake.UniPCMultistepScheduler"),
        ("scheduler_config_json", json.dumps({"name": "WanEulerScheduler", "sample_solver": "euler", "num_train_timesteps": 1000, "flow_shift": 5.0}, sort_keys=True, separators=(",", ":"))),
        ("model", "Wan-AI/Wan2.2-T2V-A14B"),
        ("runtime_dtype", "torch.float16"),
        ("latent_shape_json", json.dumps([1, 16, 9, 60, 104][::-1])),
        ("clean_state_identity_sha256_v1", "0" * 64),
        ("clean_checkpoint_hash", "0" * 64),
        ("scheduler", "UniPC"),
        ("requested_direction", "up" if row["direction"] == "down" else "down"),
    ):
        assert row[field] != value, field
        killtest.atomic_json(benign_path, dict(row, **{field: value}))
        with pytest.raises(killtest.GlobalStopError, match="identity"):
            killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    # Exploits E-H: each realized accounting field falsified alone; every other guard still passes.
    for field, value in (
        ("realized_nonzero_elements", 2),
        ("realized_l2", row["realized_l2"] * (1 + 1e-6)),
        ("realized_mse", row["realized_mse"] * 2),
        ("realized_linf", 0.0),
    ):
        killtest.atomic_json(benign_path, dict(row, **{field: value}))
        with pytest.raises(killtest.GlobalStopError, match="accounting"):
            killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    # Stored MSE values edited without touching artifacts: recomputation disagrees.
    for field in ("final_latent_mse", "video_mse"):
        killtest.atomic_json(benign_path, dict(row, **{field: row[field] + 0.25}))
        with pytest.raises(killtest.GlobalStopError, match="recomputation"):
            killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    # Row artifacts carry canonical identities and the row's identity fields are consistent with them.
    for artifact, field in (
        ("runtime_candidate_artifact", "runtime_candidate_identity_sha256_v1"),
        ("recovered_final_latent_artifact", "recovered_final_latent_identity_sha256_v1"),
        ("recovered_video_artifact", "recovered_video_identity_sha256_v1"),
    ):
        assert row[artifact]["tensor_identity_sha256_v1"] == row[field]
        assert row[artifact]["identity_format"] == killtest.TENSOR_IDENTITY_FORMAT
    assert row["resume_timestep"] == 972.9729614257812 and row["scheduler_class"] == killtest.EXPECTED_SCHEDULER_CLASS
    # Stored SSIM edited without touching artifacts: recomputation disagrees.
    killtest.atomic_json(benign_path, dict(row, frame_ssim_mean=0.5))
    with pytest.raises(killtest.GlobalStopError, match="recomputation"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    # Catastrophic video artifact grafted onto the benign row with consistent hashes: metric recomputation catches it.
    cat_row = json.loads(cat_path.read_text())
    grafted = dict(
        row,
        recovered_video_artifact=cat_row["recovered_video_artifact"],
        recovered_video_sha256=cat_row["recovered_video_sha256"],
        recovered_video_identity_sha256_v1=cat_row["recovered_video_identity_sha256_v1"],
    )
    killtest.atomic_json(benign_path, grafted)
    with pytest.raises(killtest.GlobalStopError, match="recomputation"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    # A candidate artifact that is not the claimed flip.
    other = json.loads((out / "smoke/rows" / keys[2]["condition_id"] / "replay_00/result.json").read_text())
    swapped = dict(row, runtime_candidate_artifact=other["runtime_candidate_artifact"], runtime_input_hash=other["runtime_input_hash"])
    killtest.atomic_json(benign_path, swapped)
    with pytest.raises(killtest.GlobalStopError, match="identity"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    restore()
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "GO_TO_LOCAL_BRANCH_MAP"
    # Gate-file forgery: smoke gates with a control gate removed but all_passed kept.
    gate_path = out / "smoke_gates.json"
    genuine_gates = gate_path.read_text()
    forged = json.loads(genuine_gates)
    forged["gates"] = [g for g in forged["gates"] if not g["name"].startswith("G12")]
    killtest.atomic_json(gate_path, forged)
    with pytest.raises(killtest.GlobalStopError, match="required gate set"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    killtest.atomic_json(gate_path, {"all_passed": True, "gates": []})
    with pytest.raises(killtest.GlobalStopError):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    gate_path.write_text(genuine_gates)
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "GO_TO_LOCAL_BRANCH_MAP"



# ----------------------------------------------------------------------------- frozen expected candidate binding


class _SyntheticSource:
    """Minimal anchor stand-in with real metric computation on small arrays."""

    prompt_id, seed, checkpoint_step, prompt, clean_hash = "p", 1, 10, "text", "h"
    checkpoint_path, manifest_path = Path("ck"), Path("mf")

    def __init__(self):
        self.clean = _synthetic_clean()
        self.final_latent = np.zeros_like(self.clean)
        self.video = np.zeros((2, 8, 8, 3), dtype=np.uint8)


def _source_row(tmp_path: Path, source: _SyntheticSource, flat: int = 5, direction: str = "up") -> tuple[dict, dict]:
    pytest.importorskip("skimage")
    state, expected = killtest.build_single_flip_state(source.clean, flat, direction)
    candidate_record = killtest._save_array(tmp_path / "cand.npy", state)
    latent_record = killtest._save_array(tmp_path / "lat.npy", source.final_latent + np.float32(0.5))
    video = source.video.copy()
    video[0, 0, 0, 0] = 200
    video_record = killtest._save_array(tmp_path / "vid.npy", video)
    metrics = killtest.v3.video_metrics(video, source.video)
    row = {
        "result_path": str(tmp_path / "result.json"),
        "coordinate_flat_index": flat,
        "direction": direction,
        "requested_direction": direction,
        "runtime_candidate_artifact": candidate_record,
        "recovered_final_latent_artifact": latent_record,
        "recovered_video_artifact": video_record,
        "runtime_candidate_identity_sha256_v1": candidate_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_identity_sha256_v1": latent_record["tensor_identity_sha256_v1"],
        "recovered_video_identity_sha256_v1": video_record["tensor_identity_sha256_v1"],
        "recovered_final_latent_sha256": latent_record["tensor_sha256"],
        "recovered_video_sha256": video_record["tensor_sha256"],
        "realized_nonzero_elements": 1,
        "total_elements": int(source.clean.size),
        "realized_l2": expected["realized_l2"],
        "realized_mse": expected["realized_mse"],
        "realized_linf": expected["realized_linf"],
        "exact_final_latent": False,
        "exact_video": False,
        "frame_ssim_mean": float(metrics["frame_ssim_mean"]),
        "video_mse": float(metrics["video_mse"]),
        "final_latent_mse": float(killtest.v3.latent_error(source.final_latent, source.final_latent + np.float32(0.5))["mse"]),
    }
    return row, expected


def _rewrite_candidate(row: dict, tmp_path: Path, array: np.ndarray) -> dict:
    """Exploit helper: persist a different candidate array, recompute its metadata, and update the row declaration."""
    record = killtest._save_array(tmp_path / "cand.npy", array)
    return dict(
        row,
        runtime_candidate_artifact=record,
        runtime_candidate_identity_sha256_v1=record["tensor_identity_sha256_v1"],
        runtime_input_hash=killtest.sha256_bytes(killtest.float32_to_bf16_bits(array).tobytes()),
    )


def test_source_verification_accepts_genuine_row_and_enforces_frozen_construction(tmp_path):
    source = _SyntheticSource()
    row, expected = _source_row(tmp_path, source)
    result = killtest.verify_row_against_source(row, source, expected)
    assert result["realized_nonzero_elements"] == 1
    genuine = np.load(tmp_path / "cand.npy")
    # Exploit: reshape with identical bytes, recompute metadata AND row-declared identity.
    reshaped = _rewrite_candidate(row, tmp_path, genuine.reshape(4, 3, 8, 8))
    with pytest.raises(killtest.GlobalStopError, match="candidate shape"):
        killtest.verify_row_against_source(reshaped, source, expected)
    # Same shape, reinterpretable bytes as uint32, metadata and declaration updated.
    reinterpreted = _rewrite_candidate(row, tmp_path, genuine.view(np.uint32))
    with pytest.raises(killtest.GlobalStopError, match="candidate dtype"):
        killtest.verify_row_against_source(reinterpreted, source, expected)
    # Wrong adjacent target bit at the correct coordinate (two steps instead of one), fully self-consistent row.
    two_steps = genuine.copy()
    bits = killtest.float32_to_bf16_bits(two_steps).reshape(-1)
    bits[5] = np.uint16(killtest.adjacent_bf16_bits(int(bits[5]), "up"))
    two_steps = killtest.bf16_bits_to_float32(bits).reshape(genuine.shape)
    wrong_bit = _rewrite_candidate(row, tmp_path, two_steps)
    with pytest.raises(killtest.GlobalStopError, match="frozen expected candidate"):
        killtest.verify_row_against_source(wrong_bit, source, expected)
    # Bit-level check is reachable when the attacker also controls the frozen identity fields:
    forged_expected = dict(expected, expected_candidate_tensor_identity_sha256_v1=wrong_bit["runtime_candidate_identity_sha256_v1"], expected_candidate_raw_bf16_bytes_sha256=wrong_bit["runtime_input_hash"])
    with pytest.raises(killtest.GlobalStopError, match="frozen expected adjacent BF16 bits"):
        killtest.verify_row_against_source(wrong_bit, source, forged_expected)
    # Correct changed value plus a hidden additional changed coordinate.
    hidden = genuine.copy()
    bits = killtest.float32_to_bf16_bits(hidden).reshape(-1)
    bits[77] = np.uint16(killtest.adjacent_bf16_bits(int(bits[77]), "down"))
    hidden = killtest.bf16_bits_to_float32(bits).reshape(genuine.shape)
    extra = _rewrite_candidate(row, tmp_path, hidden)
    with pytest.raises(killtest.GlobalStopError, match="frozen expected candidate"):
        killtest.verify_row_against_source(extra, source, expected)
    forged_expected = dict(expected, expected_candidate_tensor_identity_sha256_v1=extra["runtime_candidate_identity_sha256_v1"], expected_candidate_raw_bf16_bytes_sha256=extra["runtime_input_hash"])
    with pytest.raises(killtest.GlobalStopError, match="not exactly one"):
        killtest.verify_row_against_source(extra, source, forged_expected)
    # Restore the genuine artifact; the row passes again.
    killtest._save_array(tmp_path / "cand.npy", genuine)
    killtest.verify_row_against_source(row, source, expected)


def test_result_valid_binds_candidate_to_frozen_construction(tmp_path):
    provenance = {"provenance_hash": "abc"}
    path = tmp_path / "result.json"
    expected = _expected_construction()
    row = _consistent_row(tmp_path, "a")
    killtest.atomic_json(path, row)
    assert killtest._result_valid(path, provenance, {}, expected) == row
    genuine = np.load(tmp_path / "cand_a.npy")

    def rewrite(array: np.ndarray) -> dict:
        record = killtest._save_array(tmp_path / "cand_a.npy", array)
        return dict(row, runtime_candidate_artifact=record, runtime_candidate_identity_sha256_v1=record["tensor_identity_sha256_v1"],
                    runtime_input_hash=killtest.sha256_bytes(killtest.float32_to_bf16_bits(array).tobytes()))

    # 1-5: reshape with identical bytes, metadata and declared identity recomputed, everything else valid.
    killtest.atomic_json(path, rewrite(genuine.reshape(4, 3, 8, 8)))
    with pytest.raises(killtest.GlobalStopError, match="shape differs from the frozen expected candidate"):
        killtest._result_valid(path, provenance, {}, expected)
    # same shape, reinterpretable bytes as uint32
    killtest.atomic_json(path, rewrite(genuine.view(np.uint32)))
    with pytest.raises(killtest.GlobalStopError, match="dtype differs from the frozen expected candidate"):
        killtest._result_valid(path, provenance, {}, expected)
    # correct artifact, wrong row-declared identity
    killtest._save_array(tmp_path / "cand_a.npy", genuine)
    killtest.atomic_json(path, dict(row, runtime_candidate_identity_sha256_v1="0" * 64))
    with pytest.raises(killtest.GlobalStopError, match="row-declared candidate identity differs from the frozen"):
        killtest._result_valid(path, provenance, {}, expected)
    # row declaration updated to a different VALID tensor identity (another key's candidate) with that artifact persisted
    other_state = killtest.build_single_flip_state(_synthetic_clean(), 77, "down")[0]
    killtest.atomic_json(path, rewrite(other_state))
    with pytest.raises(killtest.GlobalStopError, match="identity differs from the frozen expected candidate"):
        killtest._result_valid(path, provenance, {}, expected)
    # wrong adjacent target bit at the correct coordinate
    bits = killtest.float32_to_bf16_bits(genuine).reshape(-1)
    bits[5] = np.uint16(killtest.adjacent_bf16_bits(int(bits[5]), "up"))
    killtest.atomic_json(path, rewrite(killtest.bf16_bits_to_float32(bits).reshape(genuine.shape)))
    with pytest.raises(killtest.GlobalStopError, match="identity differs from the frozen expected candidate"):
        killtest._result_valid(path, provenance, {}, expected)
    # a frozen record missing a field can never authorize a row
    with pytest.raises(killtest.GlobalStopError, match="lacks frozen field"):
        killtest._result_valid(path, provenance, {}, {k: v for k, v in expected.items() if k != "expected_candidate_shape"})
    killtest._save_array(tmp_path / "cand_a.npy", genuine)
    killtest.atomic_json(path, row)
    assert killtest._result_valid(path, provenance, {}, expected) == row


def test_frozen_scheduler_identity_and_expected_keys_document():
    config = _config()
    identity = killtest.frozen_scheduler_identity(config, 10)
    assert identity["scheduler_class"] == killtest.EXPECTED_SCHEDULER_CLASS
    assert identity["scheduler_class"].endswith("scheduling_wan_euler.WanEulerScheduler")
    assert identity["resume_timestep"] == 972.9729614257812 and identity["resume_index"] == identity["checkpoint_step"] == 10
    assert identity["scheduler_config"] == config["scheduler"]
    assert "_expected_scheduler_class" not in dir(killtest)


@needs_trusted
def test_pipeline_reshape_exploit_and_mutable_preflight_rejected(monkeypatch, tmp_path):
    config = _config()
    out = tmp_path / "out"
    cpu = killtest.run_cpu_mode(config, CONFIG_PATH, out)
    keys = json.loads((out / "expected_primary_keys.json").read_text())["keys"]
    for key in keys:
        for field in killtest.EXPECTED_CANDIDATE_FIELDS + ("coordinate_flat_index", "requested_direction"):
            assert field in key
    assert len(keys) == cpu["primary_row_count"]
    manifest = json.loads((out / "anchor_manifest.json").read_text())
    assert manifest["anchor"]["scheduler_class"] == killtest.EXPECTED_SCHEDULER_CLASS
    assert manifest["anchor"]["scheduler_identity"]["resume_timestep"] == 972.9729614257812
    assert json.loads((out / "cpu_gates.json").read_text())["anchor_manifest_sha256"] == manifest["manifest_sha256"]
    _fake_pipeline(monkeypatch, set())
    killtest.run_preflight(config, CONFIG_PATH, out, _Args())
    killtest.run_smoke(config, CONFIG_PATH, out, _Args())
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "NO_GO"
    target = keys[3]["condition_id"]
    row_path = out / "smoke/rows" / target / "replay_00/result.json"
    genuine_row = row_path.read_text()
    row = json.loads(genuine_row)
    candidate_path = Path(row["runtime_candidate_artifact"]["path"])
    genuine_candidate = np.load(candidate_path)
    # The exact remaining exploit: reshape, recompute metadata + canonical identity, update the row declaration.
    reshaped = genuine_candidate.reshape(16, 9, 60, 104)
    record = killtest._save_array(candidate_path, reshaped)
    forged = dict(row, runtime_candidate_artifact=record, runtime_candidate_identity_sha256_v1=record["tensor_identity_sha256_v1"])
    killtest.atomic_json(row_path, forged)
    with pytest.raises(killtest.GlobalStopError, match="shape differs from the frozen expected candidate"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    with pytest.raises(killtest.GlobalStopError, match="shape differs from the frozen expected candidate"):
        killtest.run_smoke(config, CONFIG_PATH, out, _Args())  # resume path refuses to reuse it too
    killtest._save_array(candidate_path, genuine_candidate)
    row_path.write_text(genuine_row)
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "NO_GO"
    # Mutable preflight_results.json is observed evidence only and cannot redefine the scheduler.
    preflight_path = out / "preflight_results.json"
    genuine_preflight = preflight_path.read_text()
    observed = json.loads(genuine_preflight)
    observed["scheduler"]["scheduler_class"] = "other.UniPCMultistepScheduler"
    killtest.atomic_json(preflight_path, observed)
    with pytest.raises(killtest.GlobalStopError, match="observed preflight scheduler class"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    observed = json.loads(genuine_preflight)
    observed["anchor_manifest_sha256"] = "0" * 64
    killtest.atomic_json(preflight_path, observed)
    with pytest.raises(killtest.GlobalStopError, match="another anchor manifest"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    preflight_path.write_text(genuine_preflight)
    # A gate file re-bound to another manifest hash is refused.
    gate_path = out / "smoke_gates.json"
    genuine_gates = gate_path.read_text()
    forged_gates = json.loads(genuine_gates)
    forged_gates["anchor_manifest_sha256"] = "0" * 64
    forged_gates["gates_sha256"] = killtest._gate_document_hash(forged_gates["gates"], forged_gates["provenance_hash"], "0" * 64)
    killtest.atomic_json(gate_path, forged_gates)
    with pytest.raises(killtest.GlobalStopError, match="another anchor manifest"):
        killtest.run_analyze_smoke(config, CONFIG_PATH, out)
    gate_path.write_text(genuine_gates)
    assert killtest.run_analyze_smoke(config, CONFIG_PATH, out)["decision"] == "NO_GO"
