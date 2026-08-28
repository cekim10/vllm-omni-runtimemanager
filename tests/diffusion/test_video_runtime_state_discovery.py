from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.video_runtime_state_discovery import (
    CONDITION_SPECS,
    EXPECTED_SCHEDULER,
    RAW_FIELDS,
    analyze_discovery,
    array_sha256,
    build_missing_mask,
    corruption_seed,
    deserialize_components,
    gaussian_matched_mse,
    latent_error,
    load_config,
    prepare_condition,
    quantize_symmetric,
    run_cpu_corruption_gates,
    serialize_components,
    synthetic_coordinate_tensor,
    validate_raw_schema,
    write_csv,
    zero_fill,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _mask(name: str, seed: int = 7) -> np.ndarray:
    shape = synthetic_coordinate_tensor().shape
    return build_missing_mask(
        shape,
        name,
        target_elements=np.prod(shape) // 4,
        seed=seed,
        temporal_slices=1,
        channel_slices=1,
        block_elements=4,
    )


def test_semantic_axis_masks_and_cardinality() -> None:
    spatial = _mask("spatial_contiguous")
    temporal = _mask("temporal_contiguous")
    channel = _mask("channel_contiguous")
    assert int(spatial.sum()) == int(temporal.sum()) == int(channel.sum()) == 64
    assert spatial[..., :2, :2].all()
    assert not spatial[..., 2:, :].any()
    assert not spatial[..., :, 2:].any()
    assert np.flatnonzero(temporal.any(axis=(0, 1, 3, 4))).tolist() == [0]
    assert temporal[:, :, 0].all() and not temporal[:, :, 1:].any()
    assert np.flatnonzero(channel.any(axis=(0, 2, 3, 4))).tolist() == [0]
    assert channel[:, 0].all() and not channel[:, 1:].any()


def test_block_interleaving_is_distributed_and_exact() -> None:
    mask = _mask("block_interleaved")
    assert int(mask.sum()) == 64
    affected_t = np.flatnonzero(mask.any(axis=(0, 1, 3, 4)))
    affected_h = np.flatnonzero(mask.any(axis=(0, 1, 2, 4)))
    assert len(affected_t) > 1
    assert len(affected_h) > 1
    assert not np.array_equal(mask, _mask("spatial_contiguous"))


def test_random_mask_is_deterministic_and_rng_independent() -> None:
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    first = _mask("random_missing", 91)
    observed = np.random.random(4)
    second = _mask("random_missing", 91)
    third = _mask("random_missing", 92)
    assert np.array_equal(observed, expected)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert int(first.sum()) == 64


@pytest.mark.parametrize(
    "geometry",
    ("spatial_contiguous", "temporal_contiguous", "channel_contiguous", "block_interleaved", "random_missing"),
)
def test_zero_fill_changes_only_masked_elements(geometry: str) -> None:
    clean = synthetic_coordinate_tensor() + 1
    mask = _mask(geometry)
    damaged = zero_fill(clean, mask)
    assert np.all(damaged[mask] == 0)
    assert np.array_equal(damaged[~mask], clean[~mask])


def test_quantization_round_trip_uses_declared_grids() -> None:
    clean = np.linspace(-3.1, 2.9, 257, dtype=np.float32).reshape(1, 1, 1, 1, -1)
    int8, q8, scale8 = quantize_symmetric(clean, 127)
    int4, q4, scale4 = quantize_symmetric(clean, 7)
    assert q8.dtype == q4.dtype == np.int8
    assert q8.min() >= -127 and q8.max() <= 127
    assert q4.min() >= -7 and q4.max() <= 7
    assert np.array_equal(int8, q8.astype(np.float32) * scale8)
    assert np.array_equal(int4, q4.astype(np.float32) * scale4)
    assert latent_error(clean, int8)["mse"] < latent_error(clean, int4)["mse"]


def test_serialized_byte_accounting_round_trip(tmp_path) -> None:
    first = np.arange(64, dtype=np.int8)
    second = np.asarray([0.25], dtype=np.float32)
    payload, metadata, paths = serialize_components(
        tmp_path, "payload", [("q", first), ("scale", second)], {"grid": "test"}
    )
    document, arrays = deserialize_components(tmp_path / "payload.payload.bin", tmp_path / "payload.metadata.json")
    assert payload == first.nbytes + second.nbytes
    assert metadata == (tmp_path / "payload.metadata.json").stat().st_size
    assert len(paths) == 2 and document["grid"] == "test"
    assert np.array_equal(arrays[0], first) and np.array_equal(arrays[1], second)


def test_gaussian_rescaling_matches_realized_mse() -> None:
    clean = np.linspace(-1, 1, 1024, dtype=np.float32).reshape(1, 4, 4, 8, 8)
    candidate, realized = gaussian_matched_mse(clean, 0.0125, 99)
    assert abs(realized - 0.0125) / 0.0125 <= 0.01
    assert candidate.dtype == clean.dtype and candidate.shape == clean.shape


def test_stale_state_uses_requested_source_and_target(tmp_path) -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    states = {
        18: np.full((1, 4, 4, 4, 4), 18, dtype=np.float32),
        19: np.full((1, 4, 4, 4, 4), 19, dtype=np.float32),
        20: np.full((1, 4, 4, 4, 4), 20, dtype=np.float32),
    }
    stale = prepare_condition(
        states[20],
        states,
        "stale_1",
        config,
        prompt_id="recovery_000",
        generation_seed=1234,
        checkpoint_step=20,
        directory=tmp_path / "stale",
    )
    assert np.array_equal(stale.restored, states[19])
    assert stale.artifact_metadata["source_step"] == 19
    assert stale.artifact_metadata["target_resume_step"] == 20
    assert stale.artifact_metadata["source_latent_hash"] == array_sha256(states[19])


def test_corruption_seed_is_stable_and_condition_specific() -> None:
    value = corruption_seed("recovery_000", 1234, 20, "random_missing")
    assert value == corruption_seed("recovery_000", 1234, 20, "random_missing")
    assert value != corruption_seed("recovery_000", 1234, 20, "gaussian_matched_int8")


def test_raw_schema_rejects_off_by_one() -> None:
    row = {field: "" for field in RAW_FIELDS}
    row.update(
        {
            "status": "COMPLETE",
            "scheduler": EXPECTED_SCHEDULER,
            "checkpoint_step": 20,
            "resume_index": 20,
            "initial_mse": 0.1,
            "final_latent_mse": 0.2,
            "final_video_mse": 0.3,
            "spatial_quality": 0.9,
            "temporal_quality": 0.8,
            "semantic_quality": 0.7,
        }
    )
    validate_raw_schema(row)
    row["resume_index"] = 19
    with pytest.raises(ValueError, match="off-by-one"):
        validate_raw_schema(row)


def test_cpu_preflight_covers_all_condition_families() -> None:
    result = run_cpu_corruption_gates()
    assert result["passed"] is True
    assert set(result["mask_cardinality"]) == {
        "spatial_contiguous",
        "temporal_contiguous",
        "channel_contiguous",
        "block_interleaved",
        "random_missing",
    }
    assert set(CONDITION_SPECS) >= {"full_direct", "int4_like", "gaussian_matched_int8", "stale_2"}


def test_analysis_emits_all_required_tables_from_complete_mock_matrix(tmp_path) -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    digest = "mock-config"
    rows = []
    for prompt_index, prompt_id in enumerate(config["prompt_ids"]):
        for step in config["generation"]["checkpoint_steps"]:
            for condition_index, condition in enumerate(config["conditions"]):
                exact = condition in {"full_direct", "full_disk"}
                row = {field: "" for field in RAW_FIELDS}
                row.update(
                    {
                        "status": "COMPLETE",
                        "experiment_version": config["experiment_version"],
                        "config_hash": digest,
                        "model": config["model"],
                        "scheduler": EXPECTED_SCHEDULER,
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_id,
                        "motion_category": "mock",
                        "generation_seed": 1000 + prompt_index,
                        "trajectory_id": f"trajectory-{prompt_id}",
                        "checkpoint_step": step,
                        "resume_index": step,
                        "corruption_family": CONDITION_SPECS[condition].family,
                        "corruption_name": condition,
                        "comparison_family": CONDITION_SPECS[condition].comparison_family,
                        "comparison_basis": CONDITION_SPECS[condition].comparison_basis,
                        "initial_mse": 0.0 if exact else 0.01 * (condition_index + 1),
                        "final_latent_mse": 0.0 if exact else 0.02 * (condition_index + 1),
                        "final_video_mse": 0.0 if exact else condition_index + 1,
                        "spatial_quality": 1.0 if exact else 0.99 - condition_index * 0.01,
                        "temporal_quality": 1.0 if exact else 0.98 - condition_index * 0.02 + step * 0.001,
                        "semantic_quality": 30.0,
                        "content_motion_proxy": 0.01 * (prompt_index + 1),
                        "content_spatial_gradient_proxy": 0.02 * (prompt_index + 1),
                        "content_temporal_gradient_proxy": 0.01 * (prompt_index + 1),
                        "latent_temporal_change_proxy": 0.03 * (prompt_index + 1),
                        "total_bytes": 1000
                        if CONDITION_SPECS[condition].comparison_family == "iso_storage_25pct"
                        else 4000,
                        "condition_metadata_json": "{}",
                        "result_path": "mock",
                    }
                )
                rows.append(row)
    write_csv(tmp_path / "raw_results.csv", rows, RAW_FIELDS)
    analyze_discovery(config, digest, tmp_path)
    for filename in (
        "progress_summary.csv",
        "ordering_reversals.csv",
        "iso_error_pairs.csv",
        "iso_storage_pairs.csv",
        "staleness_pairs.csv",
        "content_interactions.csv",
        "metric_disagreements.csv",
        "discovery_flags.json",
        "video_runtime_state_discovery.md",
    ):
        assert (tmp_path / filename).exists(), filename
