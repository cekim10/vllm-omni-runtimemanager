from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from experiments import video_checkpoint_stability_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments/video_checkpoint_stability_killtest_config.yaml"


def _config() -> dict:
    return killtest.load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def trusted_rows() -> list[dict[str, str]]:
    return killtest.source_rows(_config())


@pytest.fixture(scope="module")
def trajectories(trusted_rows: list[dict[str, str]]) -> list[dict]:
    return killtest.derive_primary_trajectories(trusted_rows, _config())


def _latent() -> np.ndarray:
    value = np.linspace(-2.0, 2.0, 8192, dtype=np.float32).reshape(1, 4, 8, 16, 16)
    return killtest.base.cast_runtime_bf16(value)


def _decision_rows(
    config: dict,
    trajectories: list[dict],
    values: dict[str, list[float]] | None = None,
) -> list[dict]:
    seeds = killtest.derive_replicate_seeds(config)
    if values is None:
        values = {row["trajectory_id"]: [0.9] * 16 for row in trajectories}
    rows = []
    target = config["validated_error_shape_source"]["target_mse"]
    for trajectory in trajectories:
        identity = trajectory["trajectory_id"]
        for replicate_id, seed in enumerate(seeds):
            rows.append(
                {
                    "trajectory_id": identity,
                    "prompt_id": trajectory["prompt_id"],
                    "generation_seed": trajectory["generation_seed"],
                    "checkpoint_step": 20,
                    "expert_regime": killtest.EXPECTED_EXPERT,
                    "replicate_id": replicate_id,
                    "replicate_seed": seed,
                    "perturbation_family": killtest.PERTURBATION_FAMILY,
                    "frame_ssim_mean": values[identity][replicate_id],
                    "realized_runtime_bf16_mse": target,
                    "relative_mse_mismatch": 0.0,
                    "realized_runtime_active_fraction": 0.845,
                    "minimum_ssim": 0.0,
                    "temporal_delta_agreement": 0.5,
                    "prompt_clip_score": 0.5,
                    "fp16_anomaly": False,
                }
            )
    return rows


def _clear_effect_values(trajectories: list[dict]) -> dict[str, list[float]]:
    values = {row["trajectory_id"]: [0.85] * 16 for row in trajectories}
    # Mean remains 0.85, but four low outcomes create a deep centered tail.
    values[trajectories[0]["trajectory_id"]] = [0.50] * 4 + [0.9666666666666667] * 12
    return values


def _range_effect_values(trajectories: list[dict], depth: float) -> dict[str, list[float]]:
    values = {row["trajectory_id"]: [0.85] * 16 for row in trajectories}
    low = 0.85 - depth
    # Four lows and twelve highs at 0.85 give tail depth 0.75*depth.
    values[trajectories[0]["trajectory_id"]] = [low] * 4 + [0.85] * 12
    return values


def test_exactly_twelve_source_derived_prompts(trajectories: list[dict]) -> None:
    assert len(trajectories) == 12
    assert len({row["prompt_id"] for row in trajectories}) == 12


def test_one_trusted_seed_per_prompt(
    trusted_rows: list[dict[str, str]], trajectories: list[dict]
) -> None:
    for trajectory in trajectories:
        seeds = {
            int(row["generation_seed"])
            for row in trusted_rows
            if row["prompt_id"] == trajectory["prompt_id"]
        }
        assert seeds == {trajectory["generation_seed"]}


def test_primary_manifest_is_step20_only(trajectories: list[dict]) -> None:
    assert {row["checkpoint_step"] for row in trajectories} == {20}


def test_manifest_is_source_row_order_invariant(trusted_rows: list[dict[str, str]]) -> None:
    config = _config()
    expected = killtest.derive_primary_trajectories(trusted_rows, config)
    shuffled = copy.deepcopy(trusted_rows)
    random.Random(55).shuffle(shuffled)
    assert killtest.derive_primary_trajectories(shuffled, config) == expected


def test_source_hash_mutation_rejected() -> None:
    config = copy.deepcopy(_config())
    config["trusted_v3"]["raw_results_sha256"] = "0" * 64
    with pytest.raises(killtest.GlobalStopError, match="source content hash mismatch"):
        killtest.source_rows(config)


def test_expert_regime_is_derived_from_scheduler_logic(
    trusted_rows: list[dict[str, str]], trajectories: list[dict]
) -> None:
    metadata = killtest.derive_expert_metadata(_config(), 20)
    assert metadata["resume_timestep"] == pytest.approx(923.076904296875)
    assert metadata["expert_regime"] == "high_noise_transformer"
    assert metadata["remaining_high_noise_steps"] == 6
    assert metadata["remaining_low_noise_steps"] == 14
    assert {row["expert_regime"] for row in trajectories} == {metadata["expert_regime"]}


def test_step30_is_low_noise_and_cannot_enter_primary(
    trajectories: list[dict]
) -> None:
    assert killtest.derive_expert_metadata(_config(), 30)["expert_regime"] == "low_noise_transformer_2"
    rows = _decision_rows(_config(), trajectories)
    rows[0]["checkpoint_step"] = 30
    with pytest.raises(killtest.GlobalStopError, match="non-step20"):
        killtest.analyze_rows(rows, _config(), correctness_passed=True)


def test_shared_canonical_replicate_schedule(trajectories: list[dict]) -> None:
    config = _config()
    rows = _decision_rows(config, trajectories)
    killtest.validate_seed_schedule(rows, config)
    expected = set(enumerate(killtest.derive_replicate_seeds(config)))
    for trajectory in trajectories:
        actual = {
            (row["replicate_id"], row["replicate_seed"])
            for row in rows
            if row["trajectory_id"] == trajectory["trajectory_id"]
        }
        assert actual == expected


def test_replicate_seed_schedule_has_no_collisions() -> None:
    seeds = killtest.derive_replicate_seeds(_config())
    assert len(seeds) == len(set(seeds)) == 16


def test_dense_perturbation_is_deterministic() -> None:
    kwargs = {
        "target_mse": _config()["validated_error_shape_source"]["target_mse"],
        "replicate_seed": killtest.derive_replicate_seeds(_config())[0],
        "relative_tolerance": 0.01,
    }
    first, first_details = killtest.construct_dense_perturbation(_latent(), **kwargs)
    second, second_details = killtest.construct_dense_perturbation(_latent(), **kwargs)
    assert np.array_equal(first, second)
    assert first_details == second_details


def test_dense_additive_only_and_full_intended_support() -> None:
    _, details = killtest.construct_dense_perturbation(
        _latent(),
        target_mse=_config()["validated_error_shape_source"]["target_mse"],
        replicate_seed=3,
        relative_tolerance=0.01,
    )
    assert details["perturbation_family"] == killtest.PERTURBATION_FAMILY
    assert details["operator_family"] == killtest.PRIMARY_OPERATOR
    assert details["active_elements"] == details["total_elements"]
    assert 0 < details["realized_runtime_active_fraction"] <= 1


def test_bf16_matcher_converges() -> None:
    target = _config()["validated_error_shape_source"]["target_mse"]
    _, details = killtest.construct_dense_perturbation(
        _latent(), target_mse=target, replicate_seed=42, relative_tolerance=0.01
    )
    assert abs(details["realized_runtime_bf16_mse"] - target) / target <= 0.01


def test_bf16_matcher_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="failed closed"):
        killtest.construct_dense_perturbation(
            _latent(), target_mse=1e-20, replicate_seed=42,
            relative_tolerance=1e-12, max_iterations=1,
        )


def test_all_192_planned_perturbations_construct(
    trajectories: list[dict]
) -> None:
    config = _config()
    target = config["validated_error_shape_source"]["target_mse"]
    rows = []
    for trajectory in trajectories:
        source = killtest._source_trajectory(config, trajectory)
        for replicate_id, seed in enumerate(killtest.derive_replicate_seeds(config)):
            _, details = killtest.construct_dense_perturbation(
                source.clean, target_mse=target, replicate_seed=seed,
                relative_tolerance=0.01,
            )
            rows.append(
                {
                    "trajectory_id": trajectory["trajectory_id"],
                    "replicate_id": replicate_id,
                    "replicate_seed": seed,
                    **details,
                }
            )
    killtest.validate_expected_keys(rows, config, trajectories)
    killtest.validate_seed_schedule(rows, config)
    summary = killtest.construction_statistics(rows, config)
    assert len(rows) == 192
    assert summary["all_cells_mse_matched"]
    assert summary["trajectory_mean_mse_gate_passed"]
    assert summary["trajectory_mean_support_gate_passed"]


def test_expected_primary_key_set_has_192_rows(trajectories: list[dict]) -> None:
    assert len(killtest.expected_primary_keys(_config(), trajectories)) == 192


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_primary_key_validation_rejects_mutations(
    trajectories: list[dict], mutation: str
) -> None:
    config = _config()
    rows = _decision_rows(config, trajectories)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["trajectory_id"] = "unexpected"
    with pytest.raises(killtest.GlobalStopError, match="key mismatch"):
        killtest.validate_expected_keys(rows, config, trajectories)


def test_bottom4_uses_four_smallest_values() -> None:
    values = [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4] + [0.9] * 8
    assert killtest.lower_tail_mean(values, 4) == pytest.approx(0.25)
    assert killtest.lower_tail_mean(values, 4) != pytest.approx(np.mean(sorted(values)[-4:]))


def test_tail_depth_is_mean_minus_bottom4() -> None:
    values = [0.5] * 4 + [0.9] * 12
    assert killtest.tail_depth(values, 4) == pytest.approx(np.mean(values) - 0.5)
    assert killtest.tail_depth(values, 4) >= 0


def test_absolute_ssim_level_difference_does_not_create_go(
    trajectories: list[dict]
) -> None:
    values = {
        row["trajectory_id"]: [0.45 + index * 0.04] * 16
        for index, row in enumerate(trajectories)
    }
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert max(max(v) for v in values.values()) - min(min(v) for v in values.values()) > 0.2
    assert result["tail_depth_range"] == pytest.approx(0)
    assert result["decision"] == "NO_GO"


def test_same_means_different_tail_depths_can_go(trajectories: list[dict]) -> None:
    values = _clear_effect_values(trajectories)
    means = [np.mean(value) for value in values.values()]
    assert max(means) - min(means) < 1e-12
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["decision"] == "GO_TO_BROADER_STABILITY_MAP"


def test_different_means_same_zero_tail_depth_is_no_go(trajectories: list[dict]) -> None:
    values = {
        row["trajectory_id"]: [0.55 + index * 0.02] * 16
        for index, row in enumerate(trajectories)
    }
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["decision"] == "NO_GO"


def test_clear_tail_depth_effect_produces_go(trajectories: list[dict]) -> None:
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, _clear_effect_values(trajectories)),
        _config(), correctness_passed=True,
    )
    assert result["tail_depth_range"] >= 0.10
    assert result["maximum_trajectory_tail_depth"] >= 0.10
    assert result["loo_tail_depth_threshold_count"] == 16
    assert result["decision"] == "GO_TO_BROADER_STABILITY_MAP"


def test_flat_trajectories_produce_no_go(trajectories: list[dict]) -> None:
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories), _config(), correctness_passed=True
    )
    assert result["decision"] == "NO_GO"


def test_intermediate_effect_is_weak(trajectories: list[dict]) -> None:
    values = _range_effect_values(trajectories, 0.095)  # tail depth ~= 0.07125
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert 0.05 <= result["tail_depth_range"] < 0.10
    assert result["decision"] == "WEAK_INCONCLUSIVE"


def test_one_catastrophic_replicate_cannot_create_go(trajectories: list[dict]) -> None:
    values = {row["trajectory_id"]: [0.9] * 16 for row in trajectories}
    values[trajectories[0]["trajectory_id"]][0] = 0.0
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["tail_depth_range"] >= 0.10
    assert result["minimum_loo_tail_depth_range"] == pytest.approx(0)
    assert not result["single_replicate_robust"]
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_minimum_or_worst_case_cannot_rescue(trajectories: list[dict]) -> None:
    rows = _decision_rows(_config(), trajectories)
    rows[0]["minimum_ssim"] = -100.0
    result = killtest.analyze_rows(rows, _config(), correctness_passed=True)
    assert result["decision"] == "NO_GO"
    assert not result["minimum_ssim_used_for_decision"]


def test_large_drop_rate_alone_cannot_rescue(trajectories: list[dict]) -> None:
    values = {row["trajectory_id"]: [0.5] * 4 + [0.9] * 12 for row in trajectories}
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["max_large_drop_rate"] >= 0.25
    assert result["tail_depth_range"] == pytest.approx(0)
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"
    assert not result["large_drop_rate_used_for_decision"]


def test_fp16_anomaly_metadata_cannot_rescue(trajectories: list[dict]) -> None:
    rows = _decision_rows(_config(), trajectories)
    baseline = killtest.analyze_rows(copy.deepcopy(rows), _config(), correctness_passed=True)
    for row in rows:
        row["fp16_anomaly"] = True
    mutated = killtest.analyze_rows(rows, _config(), correctness_passed=True)
    assert baseline["decision"] == mutated["decision"] == "NO_GO"
    assert not mutated["fp16_anomaly_used_for_decision"]


@pytest.mark.parametrize("field", ["temporal_delta_agreement", "prompt_clip_score"])
def test_descriptive_metrics_cannot_rescue(
    trajectories: list[dict], field: str
) -> None:
    rows = _decision_rows(_config(), trajectories)
    baseline = killtest.analyze_rows(copy.deepcopy(rows), _config(), correctness_passed=True)
    for index, row in enumerate(rows):
        row[field] = float(index * 1000)
    mutated = killtest.analyze_rows(rows, _config(), correctness_passed=True)
    assert baseline["decision"] == mutated["decision"] == "NO_GO"


def test_loo_exactly_eleven_of_sixteen_does_not_go(
    trajectories: list[dict],
) -> None:
    first = [
        0.3673736069491059, 0.33197997526291806, 0.6234137265258678,
        0.9102203129401656, 0.6470683476080099, 0.3173492755502836,
        0.6136440993697114, 0.6565488663488234, 0.8879776826764184,
        0.6159485319927827, 0.8502534720971241, 0.3795633858564526,
        0.9197299282253829, 0.9079083224136489, 0.7164982226112085,
        0.6267035999982293,
    ]
    second = [
        0.43208350256332584, 0.39071286780524944, 0.5165615782934692,
        0.9043409311214179, 0.4295172066629801, 0.9499400800439547,
        0.8947858688526238, 0.5266054275216886, 0.4137328389857612,
        0.5435011064155579, 0.6210897088115566, 0.6701471108655673,
        0.5026959200206157, 0.5669404923786355, 0.7550496235328347,
        0.7284811785458416,
    ]
    values = {row["trajectory_id"]: list(second) for row in trajectories}
    values[trajectories[0]["trajectory_id"]] = first
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["loo_tail_depth_threshold_count"] == 11
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_loo_exactly_twelve_of_sixteen_can_go(
    trajectories: list[dict],
) -> None:
    values = {row["trajectory_id"]: [0.85] * 16 for row in trajectories}
    # d=0.135 gives full depth 0.10125. Removing a high leaves depth
    # 0.099 (12 cases); removing a low leaves 0.07425 (4 cases).
    values[trajectories[0]["trajectory_id"]] = [0.715] * 4 + [0.85] * 12
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, values), _config(), correctness_passed=True
    )
    assert result["loo_tail_depth_threshold_count"] == 12
    assert result["decision"] == "GO_TO_BROADER_STABILITY_MAP"


def test_loo_count_one_mutation_would_be_caught() -> None:
    ranges = [0.08] + [0.06] * 15
    assert sum(value >= 0.075 for value in ranges) == 1
    result = killtest.classify_decision(
        _config(), tail_depth_range=0.2, maximum_trajectory_tail_depth=0.2,
        loo_tail_depth_ranges=ranges, mse_confound_passed=True,
        support_confound_passed=True, correctness_passed=True,
    )
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_mse_gate_removal_is_caught(trajectories: list[dict]) -> None:
    rows = _decision_rows(_config(), trajectories, _clear_effect_values(trajectories))
    target = _config()["validated_error_shape_source"]["target_mse"]
    for row in rows:
        if row["trajectory_id"] == trajectories[0]["trajectory_id"]:
            row["realized_runtime_bf16_mse"] = target * 1.009
            row["relative_mse_mismatch"] = 0.009
    result = killtest.analyze_rows(rows, _config(), correctness_passed=True)
    assert not result["construction_confounds"]["trajectory_mean_mse_gate_passed"]
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_support_gate_removal_is_caught(trajectories: list[dict]) -> None:
    rows = _decision_rows(_config(), trajectories, _clear_effect_values(trajectories))
    for row in rows:
        if row["trajectory_id"] == trajectories[0]["trajectory_id"]:
            row["realized_runtime_active_fraction"] = 0.82
    result = killtest.analyze_rows(rows, _config(), correctness_passed=True)
    assert not result["construction_confounds"]["trajectory_mean_support_gate_passed"]
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_correctness_gate_removal_is_caught(trajectories: list[dict]) -> None:
    result = killtest.analyze_rows(
        _decision_rows(_config(), trajectories, _clear_effect_values(trajectories)),
        _config(), correctness_passed=False,
    )
    assert result["decision"] != "GO_TO_BROADER_STABILITY_MAP"


def test_analysis_is_row_order_invariant(trajectories: list[dict]) -> None:
    rows = _decision_rows(_config(), trajectories, _clear_effect_values(trajectories))
    baseline = killtest.analyze_rows(copy.deepcopy(rows), _config(), correctness_passed=True)
    random.Random(17).shuffle(rows)
    assert killtest.analyze_rows(rows, _config(), correctness_passed=True) == baseline


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("analysis", "go_tail_depth_range", 0.09),
        ("analysis", "loo_required_count", 1),
        ("analysis", "large_drop_threshold", 0.05),
        ("perturbation", "trajectory_mean_mse_range_relative_to_target_limit", 0.01),
    ],
)
def test_thresholds_are_frozen(
    tmp_path: Path, section: str, field: str, value: float
) -> None:
    config = copy.deepcopy(_config())
    config[section][field] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Frozen"):
        killtest.load_config(path)


def test_source_data_cannot_precompute_sixteen_sample_tail(
    trusted_rows: list[dict[str, str]], trajectories: list[dict]
) -> None:
    audit = killtest.source_falsifiability_audit(trusted_rows, trajectories, _config())
    assert audit["preregistered_question_not_already_answered_by_source_data"]
    assert not audit["within_trajectory_tail_depth_computable"]
    assert set(audit["source_samples_per_trajectory"].values()) == {1}
    assert not audit["absolute_ssim_range_used_for_decision"]


def test_old_six_checkpoint_selection_cannot_feed_primary() -> None:
    config = _config()
    assert "frozen_checkpoints" not in config["selection"]
    assert not hasattr(killtest, "derive_checkpoint_selection")
    assert config["matrix"]["trajectory_count"] == 12


def test_no_automatic_expansion_mode() -> None:
    assert killtest.ALLOWED_MODES == ("cpu", "preflight", "smoke", "analyze-smoke")
    assert "full" not in killtest.ALLOWED_MODES
    assert "broad" not in killtest.ALLOWED_MODES


@pytest.mark.parametrize(
    "protected",
    [
        "results/video_runtime_state_discovery_v3_corrected",
        "results/video_runtime_error_shape_killtest",
    ],
)
def test_trusted_namespace_cannot_be_output(protected: str) -> None:
    with pytest.raises(killtest.GlobalStopError, match="overlaps"):
        killtest.validate_output_namespace(_config(), REPO_ROOT / protected)


def test_invalid_unipc_namespace_rejected() -> None:
    config = copy.deepcopy(_config())
    config["trusted_v3"]["root"] = "results/old_unipc_v2"
    with pytest.raises(killtest.GlobalStopError, match="v2/UniPC"):
        killtest.validate_trusted_namespace(config)


def test_target_is_recomputed_from_pinned_source(trusted_rows: list[dict[str, str]]) -> None:
    target = killtest.derive_target(_config(), trusted_rows)
    assert target["target_mse"] == 0.00011270707699032352
    assert target["source_row_count"] == 36


def test_decision_fields_exclude_absolute_and_auxiliary_metrics() -> None:
    assert "frame_ssim_mean" in killtest.DECISION_INPUT_FIELDS
    assert "minimum_ssim" not in killtest.DECISION_INPUT_FIELDS
    assert "large_drop_rate" not in killtest.DECISION_INPUT_FIELDS
    assert "temporal_delta_agreement" not in killtest.DECISION_INPUT_FIELDS
    assert "prompt_clip_score" not in killtest.DECISION_INPUT_FIELDS
