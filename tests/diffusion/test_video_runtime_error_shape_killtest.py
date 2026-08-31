from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest

from experiments import video_runtime_error_shape_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments/video_runtime_error_shape_killtest_config.yaml"


def _config() -> dict:
    return killtest.load_config(CONFIG_PATH)


def _latent() -> np.ndarray:
    values = np.linspace(-2.0, 2.0, 8192, dtype=np.float32)
    return values.reshape(1, 4, 8, 16, 16)


@pytest.fixture(scope="module")
def real_construction_rows() -> list[dict]:
    return killtest.validate_real_v3_construction_matrix(_config())


def _construct(fraction: float, operator: str = killtest.PRIMARY_OPERATOR):
    return killtest.construct_fixed_mse_error(
        _latent(),
        target_mse=0.01,
        active_fraction=fraction,
        operator_family=operator,
        support_seed=12345,
        perturbation_value_seed=54321,
        relative_tolerance=0.01,
    )


def test_numpy_bf16_encoding_matches_torch_runtime_cast() -> None:
    torch = pytest.importorskip("torch")

    values = np.asarray(
        [-100.25, -1.00390625, -1e-8, -0.0, 0.0, 1e-8, 1.00390625, 100.25],
        dtype=np.float32,
    )
    expected = torch.from_numpy(values.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    assert np.array_equal(killtest.encode_runtime_bf16(values), expected)
    decoded = killtest.decode_runtime_bf16(expected)
    torch_decoded = torch.from_numpy(values.copy()).to(torch.bfloat16).float().numpy()
    assert np.array_equal(decoded, torch_decoded)


def test_support_cardinality_and_unchanged_runtime_bits() -> None:
    clean = _latent()
    candidate, details = _construct(0.2)
    expected = killtest.support_count(clean.size, 0.2)
    assert details["active_elements"] == expected
    selected = killtest.select_support(clean.size, 0.2, 12345)
    mask = np.zeros(clean.size, dtype=bool)
    mask[selected] = True
    clean_bits = killtest.encode_runtime_bf16(clean).reshape(-1)
    candidate_bits = killtest.encode_runtime_bf16(candidate).reshape(-1)
    assert np.array_equal(clean_bits[~mask], candidate_bits[~mask])
    assert np.count_nonzero(clean_bits != candidate_bits) == details["realized_nonzero_elements"]


def test_coordinate_selection_is_deterministic_and_seeded() -> None:
    first = killtest.select_support(1000, 0.1, 7)
    second = killtest.select_support(1000, 0.1, 7)
    third = killtest.select_support(1000, 0.1, 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert len(first) == 100


def test_paired_operators_share_support_but_additive_values_are_deterministic() -> None:
    config = _config()
    digest = killtest.config_hash(config)
    arguments = (digest, "recovery_009", 10234, 20, "small", 0.2)
    additive_seeds = killtest.condition_random_seeds(*arguments, killtest.PRIMARY_OPERATOR)
    replacement_seeds = killtest.condition_random_seeds(*arguments, killtest.SECONDARY_OPERATOR)
    assert additive_seeds[0] == replacement_seeds[0]
    assert additive_seeds[1] != replacement_seeds[1]
    additive_a, details_a = killtest.construct_fixed_mse_error(
        _latent(),
        target_mse=0.01,
        active_fraction=0.2,
        operator_family=killtest.PRIMARY_OPERATOR,
        support_seed=additive_seeds[0],
        perturbation_value_seed=additive_seeds[1],
        relative_tolerance=0.01,
    )
    additive_b, details_b = killtest.construct_fixed_mse_error(
        _latent(),
        target_mse=0.01,
        active_fraction=0.2,
        operator_family=killtest.PRIMARY_OPERATOR,
        support_seed=additive_seeds[0],
        perturbation_value_seed=additive_seeds[1],
        relative_tolerance=0.01,
    )
    _, replacement_details = killtest.construct_fixed_mse_error(
        _latent(),
        target_mse=0.01,
        active_fraction=0.2,
        operator_family=killtest.SECONDARY_OPERATOR,
        support_seed=replacement_seeds[0],
        perturbation_value_seed=replacement_seeds[1],
        relative_tolerance=0.01,
    )
    assert np.array_equal(additive_a, additive_b)
    assert details_a["runtime_input_hash"] == details_b["runtime_input_hash"]
    assert details_a["selected_indices_sha256"] == replacement_details["selected_indices_sha256"]


def test_support_seeds_have_no_cross_cell_collisions() -> None:
    config = _config()
    digest = killtest.config_hash(config)
    seeds = {}
    for prompt in config["concentration"]["selected_prompts"]:
        for target in config["concentration"]["targets"]:
            for fraction in config["concentration"]["additive_support_fractions"]:
                support_seed, _ = killtest.condition_random_seeds(
                    digest,
                    prompt["prompt_id"],
                    prompt["generation_seed"],
                    20,
                    target["name"],
                    fraction,
                    killtest.PRIMARY_OPERATOR,
                )
                identity = (prompt["prompt_id"], target["name"], fraction)
                assert support_seed not in seeds
                seeds[support_seed] = identity
    assert len(seeds) == 126


@pytest.mark.parametrize("operator", [killtest.PRIMARY_OPERATOR, killtest.SECONDARY_OPERATOR])
def test_runtime_bf16_mse_matching_converges(operator: str) -> None:
    clean = _latent()
    candidate, details = killtest.construct_fixed_mse_error(
        clean,
        target_mse=0.03,
        active_fraction=0.05,
        operator_family=operator,
        support_seed=91,
        perturbation_value_seed=92,
        relative_tolerance=0.01,
    )
    independently_realized = killtest.runtime_bf16_mse(clean, candidate)
    assert independently_realized == details["realized_runtime_bf16_mse"]
    assert details["relative_mse_mismatch"] <= 0.01


def test_runtime_bf16_mse_matcher_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="failed closed"):
        killtest.construct_fixed_mse_error(
            _latent(),
            target_mse=0.0123456789,
            active_fraction=0.01,
            operator_family=killtest.PRIMARY_OPERATOR,
            support_seed=4,
            perturbation_value_seed=5,
            relative_tolerance=1e-15,
            max_iterations=0,
        )


def test_lower_support_requires_larger_active_coordinate_amplitude() -> None:
    clean = _latent()
    dense, _ = _construct(1.0)
    sparse, _ = _construct(0.01)
    runtime_clean = killtest.cast_runtime_bf16(clean)
    dense_error = np.abs(dense.astype(np.float64) - runtime_clean)
    sparse_error = np.abs(sparse.astype(np.float64) - runtime_clean)
    dense_active_rms = np.linalg.norm(dense_error) / np.sqrt(np.count_nonzero(dense_error))
    sparse_active_rms = np.linalg.norm(sparse_error) / np.sqrt(np.count_nonzero(sparse_error))
    assert sparse_active_rms > dense_active_rms * 5


def _scientific_rows(config: dict, *, endpoint: float = 0.0) -> list[dict]:
    rows = []
    for prompt in config["concentration"]["selected_prompts"]:
        for target in config["concentration"]["targets"]:
            for fraction in config["concentration"]["additive_support_fractions"]:
                # Dense support is higher quality by `endpoint` than 5% support.
                score = 0.7 + endpoint * min(1.0, max(0.0, (fraction - 0.05) / 0.95))
                rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "target_name": target["name"],
                        "operator_family": killtest.PRIMARY_OPERATOR,
                        "active_fraction": fraction,
                        "frame_ssim_mean": score,
                        "relative_mse_mismatch": 0.001,
                        "temporal_delta_mse": random.random(),
                        "temporal_delta_agreement": random.random(),
                        "prompt_clip_score": random.random(),
                    }
                )
    return rows


def test_shuffled_analysis_is_byte_identical() -> None:
    config = _config()
    rows = _scientific_rows(config, endpoint=0.12)
    shuffled = rows.copy()
    random.Random(99).shuffle(shuffled)
    assert killtest.analysis_bytes(rows, config) == killtest.analysis_bytes(shuffled, config)


def test_signed_hypothesized_curve_goes_to_confirmation() -> None:
    config = _config()
    result = killtest.analyze_primary(_scientific_rows(config, endpoint=0.12), config, controls_passed=True)
    assert result["decision"] == "GO_TO_INDEPENDENT_CONFIRMATION"
    assert all(row["mean_dense_minus_concentrated_ssim"] > 0.10 for row in result["target_summaries"])
    assert all(row["direction_prompt_count"] == 9 for row in result["target_summaries"])
    assert all(row["monotonic_prompt_count"] == 9 for row in result["target_summaries"])


def test_strong_mirrored_curve_is_no_go_not_absolute_effect() -> None:
    config = _config()
    result = killtest.analyze_primary(_scientific_rows(config, endpoint=-0.12), config, controls_passed=True)
    assert result["decision"] == "NO_GO"
    assert all(row["mean_dense_minus_concentrated_ssim"] < -0.10 for row in result["target_summaries"])
    assert all(row["direction_prompt_count"] == 0 for row in result["target_summaries"])
    assert all(row["monotonic_prompt_count"] == 0 for row in result["target_summaries"])
    # An abs(endpoint) or abs(rho) mutation would incorrectly accept this case.
    assert all(abs(row["mean_dense_minus_concentrated_ssim"]) > 0.10 for row in result["target_summaries"])
    assert all(abs(row["spearman_rho"]) >= 0.7 for row in result["prompt_effects"])


def test_flat_primary_curve_is_no_go() -> None:
    result = killtest.analyze_primary(_scientific_rows(_config(), endpoint=0.0), _config(), controls_passed=True)
    assert result["decision"] == "NO_GO"
    assert all(row["mean_dense_minus_concentrated_ssim"] == 0 for row in result["target_summaries"])


def test_exact_expected_key_set_validation() -> None:
    config = _config()
    rows = _scientific_rows(config)
    # Add the secondary matrix to form all 180 exact keys.
    for prompt in config["concentration"]["selected_prompts"]:
        for target in config["concentration"]["targets"]:
            for fraction in config["concentration"]["replacement_support_fractions"]:
                rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "target_name": target["name"],
                        "operator_family": killtest.SECONDARY_OPERATOR,
                        "active_fraction": fraction,
                    }
                )
    killtest.validate_expected_keys(rows, killtest.expected_scientific_keys(config))
    with pytest.raises(killtest.GlobalStopError, match="scientific key mismatch"):
        killtest.validate_expected_keys(rows[:-1], killtest.expected_scientific_keys(config))


def test_source_v3_hash_mismatch_is_rejected() -> None:
    config = _config()
    broken = copy.deepcopy(config)
    broken["source_v3"]["raw_results_sha256"] = "0" * 64
    with pytest.raises(killtest.GlobalStopError, match="hash mismatch"):
        killtest.source_rows(broken)


def test_unique_fp16_anomaly_is_derived_from_trusted_v3() -> None:
    config = _config()
    rows = killtest.source_rows(config)
    anomaly = killtest.derive_unique_fp16_anomaly_from_v3(rows)
    assert anomaly == {
        "prompt_id": "recovery_008",
        "generation_seed": 9234,
        "checkpoint_step": 10,
        "frame_ssim_mean": 0.8658326932016058,
        "clean_latent_hash": "75e60265e81a20fe03e6b84be4901b19cdc788a0cce4013d412ef26ddf208019",
        "corrupted_latent_hash": "07ec2e63a6c415efca29bac4edaad2cede74f89871a95efb039fd4f3918a5289",
        "final_latent_exact": False,
        "video_exact": False,
        "final_latent_mse": 0.013858933865754236,
        "video_mse": 0.003784726131015719,
    }
    assert killtest.validate_fp16_config_matches_unique_v3_anomaly(config, rows) == anomaly


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_id", "recovery_009"),
        ("generation_seed", 1),
        ("checkpoint_step", 20),
        ("original_frame_ssim_mean", 0.9),
    ],
)
def test_mutated_fp16_replay_identity_is_rejected(field: str, value) -> None:
    config = _config()
    config["fp16_replay"][field] = value
    with pytest.raises(killtest.GlobalStopError, match="does not match"):
        killtest.validate_fp16_config_matches_unique_v3_anomaly(config, killtest.source_rows(_config()))


def test_zero_or_multiple_fp16_anomalies_are_rejected() -> None:
    rows = killtest.source_rows(_config())
    zero = copy.deepcopy(rows)
    anomaly = next(
        row
        for row in zero
        if row["corruption_name"] == "fp16"
        and row["prompt_id"] == "recovery_008"
        and int(row["checkpoint_step"]) == 10
    )
    anomaly["final_latent_mse"] = "0"
    anomaly["video_mse"] = "0"
    with pytest.raises(killtest.GlobalStopError, match="found 0"):
        killtest.derive_unique_fp16_anomaly_from_v3(zero)

    multiple = copy.deepcopy(rows)
    another = next(
        row
        for row in multiple
        if row["corruption_name"] == "fp16" and row["prompt_id"] != "recovery_008"
    )
    another["final_latent_mse"] = "0.01"
    with pytest.raises(killtest.GlobalStopError, match="found 2"):
        killtest.derive_unique_fp16_anomaly_from_v3(multiple)


def _replay_rows(config: dict, fp16: list[dict], *, full_exact: bool = True) -> list[dict]:
    rows = []
    for index, row in enumerate(fp16):
        rows.append(
            {
                "replay_kind": "fp16",
                "replay_index": index,
                "input_runtime_bf16_sha256": "same-input",
                "recovered_final_latent_sha256": row["latent"],
                "recovered_video_sha256": row["video"],
                "frame_ssim_mean": row["ssim"],
                "exact_final_latent": row["exact_latent"],
                "exact_video": row["exact_video"],
            }
        )
    for index in range(config["fp16_replay"]["full_direct_repeats"]):
        rows.append(
            {
                "replay_kind": "full_direct",
                "replay_index": index,
                "input_runtime_bf16_sha256": "full-input",
                "recovered_final_latent_sha256": "clean-latent",
                "recovered_video_sha256": "clean-video",
                "frame_ssim_mean": 1.0,
                "exact_final_latent": full_exact,
                "exact_video": full_exact,
            }
        )
    return rows


def _fp16_pattern(config: dict, **updates) -> list[dict]:
    base = {
        "latent": "same-bad-latent",
        "video": "same-bad-video",
        "ssim": config["fp16_replay"]["original_frame_ssim_mean"],
        "exact_latent": False,
        "exact_video": False,
    }
    return [{**base, **updates} for _ in range(config["fp16_replay"]["repeats"])]


def test_fp16_replay_classification_a1_to_a5() -> None:
    config = _config()
    a1 = _replay_rows(config, _fp16_pattern(config))
    assert killtest.classify_fp16_replays(a1, config)["classification"].startswith("A1_")

    a2_pattern = _fp16_pattern(config)
    a2_pattern[1]["latent"] = "different-latent"
    assert killtest.classify_fp16_replays(_replay_rows(config, a2_pattern), config)["classification"].startswith("A2_")

    a3_pattern = _fp16_pattern(config)
    a3_pattern[1]["video"] = "different-video"
    assert killtest.classify_fp16_replays(_replay_rows(config, a3_pattern), config)["classification"].startswith("A3_")

    a4 = _fp16_pattern(config, latent="clean", video="clean", ssim=1.0, exact_latent=True, exact_video=True)
    assert killtest.classify_fp16_replays(_replay_rows(config, a4), config)["classification"].startswith("A4_")

    assert killtest.classify_fp16_replays(_replay_rows(config, _fp16_pattern(config), full_exact=False), config)["classification"].startswith("A5_")


def test_temporal_and_clip_metrics_cannot_change_decision() -> None:
    config = _config()
    rows = _scientific_rows(config, endpoint=0.12)
    baseline = killtest.analysis_bytes(rows, config)
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["temporal_delta_mse"] = 1e20 + index
        row["temporal_delta_agreement"] = -1e20 - index
        row["prompt_clip_score"] = 1e30 if index % 2 else -1e30
    assert killtest.analysis_bytes(changed, config) == baseline
    assert not (set(config["analysis"]["descriptive_only_metrics"]) & killtest.DECISION_INPUT_FIELDS)
    assert not (set(config["analysis"]["auxiliary_only_metrics"]) & killtest.DECISION_INPUT_FIELDS)


def test_go_no_go_thresholds_are_frozen() -> None:
    config = _config()
    assert config["analysis"]["go_endpoint_ssim_difference"] == 0.10
    assert config["analysis"]["no_go_endpoint_ssim_difference"] == 0.05
    assert config["analysis"]["go_direction_prompt_count"] == 7
    assert config["analysis"]["go_monotonic_prompt_count"] == 6
    assert config["analysis"]["go_spearman_rho"] == 0.7
    broken = copy.deepcopy(config)
    broken["analysis"]["go_endpoint_ssim_difference"] = 0.09
    path = Path("/tmp/video_runtime_error_shape_bad_config.json")
    path.write_text(json.dumps(broken))
    try:
        with pytest.raises(ValueError, match="threshold changed"):
            killtest.load_config(path)
    finally:
        path.unlink(missing_ok=True)


def test_secondary_replacement_cannot_rescue_primary_no_go() -> None:
    config = _config()
    primary = _scientific_rows(config, endpoint=0.0)
    baseline = killtest.analyze_primary(primary, config, controls_passed=True)
    assert baseline["decision"] == "NO_GO"
    secondary = []
    for key in killtest.expected_scientific_keys(config):
        if key[2] == killtest.SECONDARY_OPERATOR:
            secondary.append(
                {
                    "prompt_id": key[0],
                    "target_name": key[1],
                    "operator_family": key[2],
                    "active_fraction": key[3],
                    "frame_ssim_mean": 1.0,
                    "relative_mse_mismatch": 0.0,
                }
            )
    with_secondary = killtest.analyze_primary(primary + secondary, config, controls_passed=True)
    assert with_secondary == baseline
    assert with_secondary["secondary_operator_used_for_decision"] is False


def test_single_authoritative_tolerance_and_all_real_cells(
    real_construction_rows: list[dict],
) -> None:
    config = _config()
    concentration = config["concentration"]
    assert concentration["runtime_mse_relative_tolerance"] == 0.01
    assert "preferred_runtime_mse_relative_tolerance" not in concentration
    assert len(real_construction_rows) == 180
    assert max(row["relative_mse_mismatch"] for row in real_construction_rows) <= 0.01
    assert killtest.paired_operator_supports_match(real_construction_rows)


def test_realized_support_fraction_and_ordering_on_real_checkpoints(
    real_construction_rows: list[dict],
) -> None:
    for row in real_construction_rows:
        assert row["realized_runtime_active_fraction"] == (
            row["realized_nonzero_elements"] / row["total_elements"]
        )
    killtest.validate_realized_support_ordering(real_construction_rows)
    small_dense = [
        row
        for row in real_construction_rows
        if row["operator_family"] == killtest.PRIMARY_OPERATOR
        and row["target_name"] == "small"
        and row["active_fraction"] == 1.0
    ]
    assert all(row["realized_runtime_active_fraction"] < 1.0 for row in small_dense)


def test_dynamic_range_descriptors_are_computed_and_descriptive_only() -> None:
    clean = np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    restored = np.asarray([[-3.0, -1.0, 0.0, 1.0, 4.0]], dtype=np.float32)
    descriptors = killtest.error_descriptors(clean, restored)
    assert descriptors["clean_abs_max"] == 2.0
    assert descriptors["restored_abs_max"] == 4.0
    assert descriptors["restored_to_clean_absmax_ratio"] == 2.0
    assert descriptors["linf_error"] == 2.0
    assert descriptors["active_error_rms"] == pytest.approx(np.sqrt(2.5))
    assert descriptors["exceeds_clean_dynamic_range"] is True

    config = _config()
    rows = _scientific_rows(config, endpoint=0.12)
    baseline = killtest.analysis_bytes(rows, config)
    for index, row in enumerate(rows):
        row["exceeds_clean_dynamic_range"] = bool(index % 2)
        row["restored_to_clean_absmax_ratio"] = 1000 + index
        row["active_error_rms"] = 100 + index
    assert killtest.analysis_bytes(rows, config) == baseline


@pytest.mark.parametrize(
    ("alpha", "classification"),
    [
        (None, "not_applicable"),
        (0.5, "attenuation_or_replacement_like"),
        (0.0, "exact_zero_fill"),
        (-0.1, "sign_inverting_multiplicative_perturbation"),
        (1.1, "amplification"),
    ],
)
def test_replacement_alpha_classification(alpha, classification: str) -> None:
    assert killtest.classify_replacement_alpha(alpha) == classification


def test_stale_provenance_gate_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    current = {"provenance_hash": "new", "script": "abc"}
    path.write_text(json.dumps(current))
    killtest.assert_provenance_matches(path, current)
    stale = {"provenance_hash": "old", "script": "abc"}
    with pytest.raises(killtest.GlobalStopError, match="stale"):
        killtest.assert_provenance_matches(path, stale)
