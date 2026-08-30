from __future__ import annotations

import ast
import copy
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from experiments.video_runtime_state_discovery import (
    CONDITION_SPECS,
    DESIGNED_ISO_ERROR_PAIRS,
    EXPECTED_RUNTIME_DTYPE,
    EXPECTED_SCHEDULER,
    IDENTITY_ALLOWED_CONDITIONS,
    ISO_MISSING_CONDITIONS,
    PRIMARY_ISO_STORAGE_CONDITIONS,
    RAW_FIELDS,
    analyze_discovery,
    array_sha256,
    assert_provenance_matches,
    build_missing_mask,
    build_prompt_corruption_summary,
    comparison_role,
    corruption_seed,
    designed_iso_error_comparisons,
    deserialize_components,
    evaluate_metric_controls,
    expected_matrix_keys,
    expected_smoke_keys,
    _json_safe,
    gate_record,
    gaussian_matched_mse,
    gaussian_matched_runtime_mse,
    latent_error,
    load_config,
    prepare_condition,
    primary_iso_storage_byte_plan,
    quantize_symmetric,
    run_cpu_corruption_gates,
    runtime_dtype_mse,
    runtime_state_accounting,
    serialize_components,
    synthetic_coordinate_tensor,
    stale_minus_matched,
    validate_raw_schema,
    validate_expected_key_set,
    validate_clean_checkpoint_pairing,
    validate_primary_iso_storage_accounting,
    validate_gate_records,
    video_metrics,
    write_csv,
    zero_fill,
)
from experiments.video_runtime_state_discovery import _metric_control_rows

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
    assert spatial[..., 1:3, 1:3].all()
    assert not spatial[..., :1, :].any()
    assert not spatial[..., 3:, :].any()
    assert not spatial[..., :, :1].any()
    assert not spatial[..., :, 3:].any()
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


def test_all_preregistered_missing_geometries_have_equal_cardinality() -> None:
    assert ISO_MISSING_CONDITIONS == (
        "spatial_contiguous",
        "temporal_contiguous",
        "block_interleaved",
        "random_missing",
    )
    assert "channel_contiguous" not in ISO_MISSING_CONDITIONS
    assert {int(_mask(name).sum()) for name in ISO_MISSING_CONDITIONS} == {64}


def test_real_wan_iso_missing_cardinality_excludes_channel_control() -> None:
    shape = (1, 16, 9, 60, 104)
    target = 1 * 16 * 2 * 60 * 104
    masks = {
        name: build_missing_mask(
            shape,
            name,
            target_elements=target,
            seed=7,
            temporal_slices=2,
            channel_slices=4,
            block_elements=16,
        )
        for name in ISO_MISSING_CONDITIONS
    }
    channel_target = 1 * 4 * 9 * 60 * 104
    channel = build_missing_mask(
        shape,
        "channel_contiguous",
        target_elements=channel_target,
        seed=7,
        temporal_slices=2,
        channel_slices=4,
        block_elements=16,
    )
    assert target == 199_680
    assert {int(mask.sum()) for mask in masks.values()} == {199_680}
    assert int(channel.sum()) == 224_640


def test_spatial_mask_matches_real_wan_cardinality_with_one_boundary_fringe() -> None:
    shape = (1, 16, 9, 60, 104)
    target = 1 * 16 * 2 * 60 * 104
    mask = build_missing_mask(
        shape,
        "spatial_contiguous",
        target_elements=target,
        seed=7,
        temporal_slices=2,
        channel_slices=4,
        block_elements=16,
    )
    per_spatial_cell = mask.reshape(1 * 16 * 9, 60 * 104).sum(axis=0)
    full_cells, fringe_elements = divmod(target, 1 * 16 * 9)
    assert int(mask.sum()) == target
    assert np.count_nonzero(per_spatial_cell == 1 * 16 * 9) == full_cells
    assert np.count_nonzero(per_spatial_cell == fringe_elements) == 1
    assert np.count_nonzero(per_spatial_cell) == full_cells + 1


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


def test_only_fp16_and_exact_controls_allow_identity_round_trip() -> None:
    assert IDENTITY_ALLOWED_CONDITIONS == {"full_direct", "full_disk", "fp16"}
    assert {
        "int8",
        "int4_like",
        "spatial_width_down2_bf16",
        "random_missing",
        "stale_1",
    }.isdisjoint(IDENTITY_ALLOWED_CONDITIONS)


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
        runtime_state={
            "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "runtime_element_size_bytes": 2,
            "runtime_numel": states[20].size,
        },
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
            "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
            "runtime_element_size_bytes": 2,
            "runtime_numel": 100,
            "runtime_full_bytes": 200,
            "initial_mse_probe_dtype": 0.1,
            "initial_mse_runtime_dtype": 0.1,
            "final_latent_mse": 0.2,
            "video_mse": 0.3,
            "frame_ssim_mean": 0.9,
            "temporal_delta_mse": 0.2,
            "temporal_delta_agreement": 0.8,
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


def test_bf16_runtime_accounting_survives_fp32_probe_widening() -> None:
    probe = np.zeros((1, 4, 4, 8, 8), dtype=np.float32)
    accounting = runtime_state_accounting(
        runtime_dtype=EXPECTED_RUNTIME_DTYPE,
        runtime_element_size_bytes=2,
        runtime_numel=probe.size,
        probe=probe,
    )
    assert accounting["runtime_full_bytes"] == probe.size * 2
    assert accounting["probe_payload_bytes"] == probe.size * 4


def test_fp16_is_dtype_conversion_control_not_half_size_compression() -> None:
    assert comparison_role("fp16") == "dtype_conversion_control"
    runtime_numel = 4096
    fp16_payload_bytes = np.zeros(runtime_numel, dtype=np.float16).nbytes
    assert fp16_payload_bytes / (runtime_numel * 2) == 1.0


def test_primary_iso_storage_plan_is_bf16_relative_and_byte_matched() -> None:
    shape = (1, 16, 9, 60, 104)
    plan = primary_iso_storage_byte_plan(shape)
    assert tuple(plan) == PRIMARY_ISO_STORAGE_CONDITIONS
    runtime_full_bytes = math.prod(shape) * 2
    payloads = {name: int(plan[name]["payload_bytes"]) for name in plan}
    assert runtime_full_bytes == 1_797_120
    assert payloads == {
        "int8": 898_564,
        "spatial_width_down2_bf16": 898_560,
        "low_rank_byte_matched_bf16": 893_900,
    }
    assert plan["low_rank_byte_matched_bf16"]["parameters"]["rank"] == 70
    for left, right in itertools.combinations(PRIMARY_ISO_STORAGE_CONDITIONS, 2):
        mismatch = abs(payloads[left] - payloads[right]) / max(payloads[left], payloads[right])
        assert mismatch <= 0.02
    assert payloads["spatial_width_down2_bf16"] / runtime_full_bytes == 0.5
    assert payloads["low_rank_byte_matched_bf16"] / runtime_full_bytes == pytest.approx(
        0.497405, rel=1e-5
    )


def test_primary_iso_storage_parameters_depend_only_on_shape_and_bytes() -> None:
    first = primary_iso_storage_byte_plan((1, 16, 9, 60, 104))
    second = primary_iso_storage_byte_plan((1, 16, 9, 60, 104))
    assert first == second
    assert "quality" not in repr(first).lower()


def test_primary_iso_storage_gate_fails_closed() -> None:
    shape = (1, 16, 9, 60, 104)
    plan = primary_iso_storage_byte_plan(shape)
    values = {
        name: {
            "payload_bytes": int(plan[name]["payload_bytes"]),
            "total_bytes": int(plan[name]["payload_bytes"]) + 1024,
        }
        for name in PRIMARY_ISO_STORAGE_CONDITIONS
    }
    payload_rows, pair_rows = validate_primary_iso_storage_accounting(values, shape, 0.02)
    assert len(payload_rows) == 3 and len(pair_rows) == 3

    wrong_payload = copy.deepcopy(values)
    wrong_payload["spatial_width_down2_bf16"]["payload_bytes"] += 2
    with pytest.raises(RuntimeError, match="payload mismatch"):
        validate_primary_iso_storage_accounting(wrong_payload, shape, 0.02)

    wrong_total = copy.deepcopy(values)
    wrong_total["low_rank_byte_matched_bf16"]["total_bytes"] //= 2
    with pytest.raises(RuntimeError, match="serialized-byte mismatch"):
        validate_primary_iso_storage_accounting(wrong_total, shape, 0.02)


def test_primary_iso_storage_serialization_uses_declared_widths(tmp_path) -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    clean = np.linspace(-2, 2, 1024, dtype=np.float32).reshape(1, 4, 4, 8, 8)
    runtime_state = {
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "runtime_element_size_bytes": 2,
        "runtime_numel": clean.size,
    }
    plan = primary_iso_storage_byte_plan(tuple(clean.shape))
    for condition in PRIMARY_ISO_STORAGE_CONDITIONS:
        prepared = prepare_condition(
            clean,
            {20: clean},
            condition,
            config,
            runtime_state=runtime_state,
            prompt_id="recovery_000",
            generation_seed=1234,
            checkpoint_step=20,
            directory=tmp_path / condition,
        )
        assert prepared.payload_bytes == plan[condition]["payload_bytes"]
        expected_dtype = "int8" if condition == "int8" else "bfloat16"
        assert prepared.encoded_dtype == expected_dtype
        metadata = json.loads(Path(prepared.artifact_paths[1]).read_text())
        component_dtypes = {item["dtype"] for item in metadata["components"]}
        if condition == "int8":
            assert "|i1" in component_dtypes and "<f4" in component_dtypes
        else:
            assert component_dtypes == {"<u2"}


def test_probe_and_runtime_dtype_mse_are_separate() -> None:
    clean = np.asarray([1.001, -0.997, 0.0031, 9.999], dtype=np.float32).reshape(1, 1, 1, 1, 4)
    candidate = clean + np.asarray([0.0005, 0.0005, -0.0005, 0.0005], dtype=np.float32).reshape(clean.shape)
    probe_mse = float(latent_error(clean, candidate)["mse"])
    realized_mse = runtime_dtype_mse(clean, candidate, EXPECTED_RUNTIME_DTYPE)
    assert probe_mse != realized_mse


def test_runtime_dtype_gaussian_matcher_converges_and_fails_closed(monkeypatch) -> None:
    clean = np.linspace(-1, 1, 4096, dtype=np.float32).reshape(1, 4, 4, 16, 16)
    candidate, realized = gaussian_matched_runtime_mse(
        clean, 0.01, 123, EXPECTED_RUNTIME_DTYPE
    )
    assert candidate.shape == clean.shape
    assert abs(realized - 0.01) / 0.01 <= 1e-4

    import experiments.video_runtime_state_discovery as discovery

    monkeypatch.setattr(
        discovery,
        "runtime_dtype_mse",
        lambda clean, candidate, runtime_dtype: 0.02,
    )
    with pytest.raises(RuntimeError, match="failed to converge"):
        discovery.gaussian_matched_runtime_mse(
            clean, 0.01, 123, EXPECTED_RUNTIME_DTYPE
        )


def _quality_row(name: str, *, mse: float, ssim: float, temporal: float) -> dict[str, object]:
    return {
        "control": name,
        "video_mse": mse,
        "frame_ssim_mean": ssim,
        "temporal_delta_mse": 0.0 if name in {"self", "exact_resume"} else 1.0 - temporal,
        "temporal_delta_agreement": temporal,
        "prompt_clip_score": 0.0,
    }


def test_metric_controls_and_clip_independence() -> None:
    rows = [
        _quality_row("self", mse=0.0, ssim=1.0, temporal=1.0),
        _quality_row("exact_resume", mse=0.0, ssim=1.0, temporal=1.0),
        _quality_row("mild_corruption", mse=0.01, ssim=0.95, temporal=0.9),
        _quality_row("severe_corruption", mse=0.1, ssim=0.4, temporal=0.4),
        _quality_row("zero", mse=0.3, ssim=0.1, temporal=0.5),
        _quality_row("frozen_first_frame", mse=0.2, ssim=0.3, temporal=0.99),
        _quality_row("temporal_shuffle", mse=0.1, ssim=0.5, temporal=0.2),
    ]
    assert evaluate_metric_controls(rows)["passed"] is True
    for row in rows:
        row["prompt_clip_score"] = 10_000 if row["control"] == "zero" else -10_000
        row["temporal_delta_mse"] = -10_000
        row["temporal_delta_agreement"] = 10_000
    assert evaluate_metric_controls(rows)["passed"] is True


def test_temporal_delta_controls_are_descriptive_only() -> None:
    rows = [
        _quality_row("self", mse=0.0, ssim=1.0, temporal=-100.0),
        _quality_row("exact_resume", mse=0.0, ssim=1.0, temporal=-100.0),
        _quality_row("mild_corruption", mse=0.01, ssim=0.95, temporal=-100.0),
        _quality_row("severe_corruption", mse=0.1, ssim=0.4, temporal=100.0),
        _quality_row("zero", mse=0.3, ssim=0.1, temporal=100.0),
        _quality_row("frozen_first_frame", mse=0.2, ssim=0.3, temporal=100.0),
        _quality_row("temporal_shuffle", mse=0.1, ssim=0.5, temporal=100.0),
    ]
    result = evaluate_metric_controls(rows)
    assert result["passed"] is True
    assert all("temporal" not in name for name in result["required_invariants"])
    assert set(result["descriptive_temporal_controls"]) == {
        "self",
        "zero",
        "frozen_first_frame",
        "temporal_shuffle",
    }


def test_zero_frozen_and_shuffle_controls_are_measured_from_artifacts() -> None:
    class Evaluator:
        @staticmethod
        def score(prompt, video):
            return 0.0

    rng = np.random.default_rng(17)
    reference = rng.integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    rows = _metric_control_rows(Evaluator(), "prompt", reference, reference.copy(), reference[::-1].copy())
    by_name = {row["control"]: row for row in rows}
    assert {"zero", "frozen_first_frame", "temporal_shuffle"}.issubset(by_name)
    assert by_name["zero"]["video_mse"] > 0
    assert by_name["frozen_first_frame"]["frame_ssim_mean"] < 1
    assert math.isfinite(by_name["temporal_shuffle"]["temporal_delta_mse"])


def test_reference_metrics_exact_and_temporal_controls_are_recorded() -> None:
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    exact = video_metrics(reference, reference)
    shuffled = video_metrics(reference[::-1].copy(), reference)
    assert exact["video_mse"] == 0
    assert exact["frame_ssim_mean"] == 1
    assert exact["temporal_delta_mse"] == 0
    assert math.isfinite(shuffled["temporal_delta_mse"])
    assert math.isfinite(shuffled["temporal_delta_agreement"])


def test_temporal_delta_metrics_do_not_enter_active_analysis_decisions() -> None:
    source = Path("experiments/video_runtime_state_discovery.py").read_text()
    tree = ast.parse(source)
    analyze = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "analyze_discovery"
    )
    for node in ast.walk(analyze):
        if isinstance(node, ast.If):
            assert "temporal_delta" not in ast.unparse(node.test)
        if isinstance(node, ast.Lambda):
            assert "temporal_delta" not in ast.unparse(node.body)


def test_staleness_delta_sign_is_explicit_and_order_independent() -> None:
    stale = {"corruption_family": "staleness", "corruption_name": "aaa_stale", "frame_ssim_mean": 0.7}
    matched = {"corruption_family": "precision", "corruption_name": "zzz_match", "frame_ssim_mean": 0.9}
    assert stale_minus_matched(stale, matched)["stale_minus_matched"] == pytest.approx(-0.2)
    stale["corruption_name"], matched["corruption_name"] = "zzz_stale", "aaa_match"
    assert stale_minus_matched(stale, matched)["stale_minus_matched"] == pytest.approx(-0.2)


def test_only_preregistered_pairs_enter_primary_iso_error() -> None:
    cell = {}
    for name in {item for pair in DESIGNED_ISO_ERROR_PAIRS for item in pair}:
        family = "gaussian" if name.startswith("gaussian_") else "precision"
        cell[name] = {
            "corruption_name": name,
            "corruption_family": family,
            "initial_mse_probe_dtype": 0.1,
            "initial_mse_runtime_dtype": 0.1,
            "frame_ssim_mean": 0.9,
            "temporal_delta_agreement": 0.8,
        }
    rows = designed_iso_error_comparisons(cell, 0.01)
    assert {frozenset((row["condition_a"], row["condition_b"])) for row in rows} == {
        frozenset(pair) for pair in DESIGNED_ISO_ERROR_PAIRS
    }


def test_code_hash_mismatch_invalidates_gate() -> None:
    expected = {"provenance_hash": "new"}
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        assert_provenance_matches({"provenance_hash": "old"}, expected)


def test_expected_key_set_reports_missing_duplicate_and_unexpected() -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    expected = expected_matrix_keys(config)
    rows = [
        {
            "prompt_id": key[0],
            "generation_seed": key[1],
            "checkpoint_step": key[2],
            "corruption_name": key[3],
        }
        for key in expected
    ]
    assert validate_expected_key_set(rows, expected)["passed"] is True
    rows.append(rows[0].copy())
    rows.append({"prompt_id": "unexpected", "generation_seed": 1, "checkpoint_step": 10, "corruption_name": "int8"})
    report = validate_expected_key_set(rows, expected, raise_on_error=False)
    assert report["duplicate_keys"] and report["unexpected_keys"]


def test_smoke_completeness_uses_exact_expected_key_set() -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    prompt_id = config["prompt_ids"][0]
    expected = expected_smoke_keys(config, prompt_id)
    rows = [
        {
            "prompt_id": key[0],
            "generation_seed": key[1],
            "checkpoint_step": key[2],
            "corruption_name": key[3],
        }
        for key in expected
    ]
    assert validate_expected_key_set(rows, expected)["passed"] is True
    rows[-1] = rows[0].copy()
    report = validate_expected_key_set(rows, expected, raise_on_error=False)
    assert len(rows) == len(expected)
    assert report["passed"] is False
    assert report["missing_keys"] and report["duplicate_keys"]


def test_clean_checkpoint_pairing_rejects_one_mutated_hash() -> None:
    rows = [
        {
            "prompt_id": "recovery_000",
            "generation_seed": 1234,
            "checkpoint_step": 20,
            "corruption_name": name,
            "clean_latent_hash": "clean-hash",
        }
        for name in ("full_direct", "int8", "random_missing")
    ]
    assert validate_clean_checkpoint_pairing(rows)["passed"] is True
    mutated = copy.deepcopy(rows)
    mutated[-1]["clean_latent_hash"] = "mutated-clean-hash"
    with pytest.raises(RuntimeError, match="Clean checkpoint pairing mismatch"):
        validate_clean_checkpoint_pairing(mutated)


def test_by_step_summary_is_explicit_and_input_order_independent() -> None:
    rows = [
        {
            "prompt_id": "recovery_000",
            "motion_category": "static",
            "corruption_name": "int8",
            "checkpoint_step": step,
            "frame_ssim_mean": 0.90 + step / 1000,
            "temporal_delta_agreement": 0.80 + step / 1000,
            "initial_mse_runtime_dtype": step / 100,
        }
        for step in (10, 20, 30)
    ]
    expected = build_prompt_corruption_summary(rows)
    shuffled = rows.copy()
    random.Random(7).shuffle(shuffled)
    observed = build_prompt_corruption_summary(shuffled)
    assert observed == expected
    assert json.loads(observed[0]["checkpoint_steps"]) == [10, 20, 30]
    assert json.loads(observed[0]["frame_ssim_by_step"]) == [0.91, 0.92, 0.93]


def test_gate_evidence_is_json_serializable_after_gpu_preflight() -> None:
    """Gate evidence is written only after the whole GPU preflight has run.

    A TypeError at that point discards every generation, so evidence containing
    sets/Paths/numpy values must be normalized when the gate is constructed.
    """
    evidence = {
        10: {199_680},
        20: {"random_missing", "temporal_contiguous"},
        "path": Path("/tmp/mask.npy"),
        "numpy_scalar": np.int64(7),
        "numpy_array": np.arange(3),
        "nested": [{"inner": frozenset({3, 1, 2})}],
        "non_finite": float("inf"),
    }
    gate = gate_record("evidence", True, evidence, [Path("/tmp/a.csv")], "serializable")
    encoded = json.dumps(gate, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)["measured_evidence"]
    assert decoded["10"] == [199_680]
    assert decoded["20"] == ["random_missing", "temporal_contiguous"]
    assert decoded["path"] == "/tmp/mask.npy"
    assert decoded["numpy_scalar"] == 7
    assert decoded["numpy_array"] == [0, 1, 2]
    assert decoded["nested"] == [{"inner": [1, 2, 3]}]
    assert decoded["non_finite"] == "inf"


def test_g7_cardinality_evidence_shape_is_serializable() -> None:
    """Regression: G7 evidence previously carried dict[int, set] and killed a completed preflight."""
    expected_missing = {step: {199_680} for step in (10, 20, 30)}
    expected_names = {step: set(ISO_MISSING_CONDITIONS) for step in (10, 20, 30)}
    gate = gate_record(
        "G7 corruption cardinality",
        True,
        {
            "real_iso_missing_cardinality": {s: sorted(v) for s, v in expected_missing.items()},
            "real_iso_missing_conditions": {s: sorted(v) for s, v in expected_names.items()},
            "expected_cardinality": 199_680,
            "excluded_descriptive_conditions": ["channel_contiguous"],
        },
        [],
        "equal cardinality within the preregistered iso-missing family",
    )
    evidence = json.loads(json.dumps(gate, allow_nan=False))["measured_evidence"]
    assert evidence["real_iso_missing_cardinality"]["10"] == [199_680]
    assert evidence["real_iso_missing_conditions"]["30"] == sorted(ISO_MISSING_CONDITIONS)
    assert "channel_contiguous" not in evidence["real_iso_missing_conditions"]["10"]

    # The raw dict[int, set] form that previously reached json.dumps must also survive.
    raw_gate = gate_record(
        "G7 corruption cardinality",
        True,
        {
            "real_iso_missing_cardinality": expected_missing,
            "real_iso_missing_conditions": expected_names,
        },
        [],
        "equal cardinality within the preregistered iso-missing family",
    )
    raw = json.loads(json.dumps(raw_gate, allow_nan=False))["measured_evidence"]
    assert raw["real_iso_missing_cardinality"]["20"] == [199_680]
    assert raw["real_iso_missing_conditions"]["20"] == sorted(ISO_MISSING_CONDITIONS)


def test_json_safe_leaves_plain_values_untouched() -> None:
    plain = {"a": 1, "b": 2.5, "c": "s", "d": True, "e": None, "f": [1, {"g": 2}]}
    assert _json_safe(plain) == plain


def test_preflight_gate_schema_has_no_unmeasured_required_pass() -> None:
    measured = gate_record("measured", True, {"value": 1}, [], "value == 1")
    informational = gate_record("info", None, {"note": "descriptive"}, [], "none", required=False)
    assert measured["status"] == "PASS"
    assert informational["status"] == "INFORMATIONAL"
    assert validate_gate_records([measured, informational]) is True


def test_run_preflight_has_no_literal_true_scientific_gate() -> None:
    source = Path("experiments/video_runtime_state_discovery.py").read_text()
    tree = ast.parse(source)
    run_preflight = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_preflight"
    )
    iso_storage_gate = None
    for node in ast.walk(run_preflight):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "gate_record":
            assert len(node.args) >= 2
            assert not (isinstance(node.args[1], ast.Constant) and node.args[1].value is True)
            if (
                isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "Iso-storage byte match"
            ):
                iso_storage_gate = node
    assert iso_storage_gate is not None
    required_keywords = {keyword.arg: keyword.value for keyword in iso_storage_gate.keywords}
    assert "required" not in required_keywords


def _mock_v3_matrix_rows(config: dict, digest: str, provenance: dict) -> list[dict]:
    rows = []
    mse_by_condition = {name: 0.01 * (index + 1) for index, name in enumerate(config["conditions"])}
    mse_by_condition["gaussian_matched_int8"] = mse_by_condition["int8"]
    mse_by_condition["gaussian_matched_random_missing"] = mse_by_condition["random_missing"]
    for prompt_id in config["prompt_ids"]:
        seed = config["generation_seeds"][prompt_id]
        for step in config["generation"]["checkpoint_steps"]:
            for index, condition in enumerate(config["conditions"]):
                exact = condition in {"full_direct", "full_disk"}
                row = {field: "" for field in RAW_FIELDS}
                row.update(
                    {
                        "status": "COMPLETE",
                        "experiment_version": config["experiment_version"],
                        "config_hash": digest,
                        "provenance_hash": provenance["provenance_hash"],
                        "model": config["model"],
                        "scheduler": EXPECTED_SCHEDULER,
                        "prompt_id": prompt_id,
                        "motion_category": "mock",
                        "generation_seed": seed,
                        "checkpoint_step": step,
                        "resume_index": step,
                        "corruption_family": CONDITION_SPECS[condition].family,
                        "corruption_name": condition,
                        "comparison_family": CONDITION_SPECS[condition].comparison_family,
                        "comparison_basis": CONDITION_SPECS[condition].comparison_basis,
                        "comparison_role": comparison_role(condition),
                        "clean_latent_hash": f"clean-{prompt_id}-{seed}-{step}",
                        "initial_mse_probe_dtype": 0.0 if exact else mse_by_condition[condition],
                        "initial_mse_runtime_dtype": 0.0 if exact else mse_by_condition[condition],
                        "final_latent_mse": 0.0 if exact else 0.01,
                        "video_mse": 0.0 if exact else 0.01,
                        "frame_ssim_mean": 1.0 if exact else 0.9 - index * 0.001,
                        "temporal_delta_mse": 0.0 if exact else 0.01,
                        "temporal_delta_agreement": 1.0 if exact else 0.85 - index * 0.001,
                        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
                        "runtime_element_size_bytes": 2,
                        "runtime_numel": 2000,
                        "runtime_full_bytes": 4000,
                        "encoded_dtype": "int8"
                        if condition == "int8"
                        else "bfloat16"
                        if condition in {
                            "spatial_width_down2_bf16",
                            "low_rank_byte_matched_bf16",
                        }
                        else "",
                        "serialized_candidate_bytes": 4000
                        if condition == "fp16"
                        else 2000
                        if condition in PRIMARY_ISO_STORAGE_CONDITIONS
                        else 1000,
                        "candidate_ratio_vs_runtime_state": 1.0
                        if condition == "fp16"
                        else 0.5
                        if condition in PRIMARY_ISO_STORAGE_CONDITIONS
                        else 0.25,
                        "serialized_candidate_deployment_meaningful": True,
                        "checkpoint_scheduler_timestep": 900 - step,
                        "current_expert": "mock_expert",
                        "crosses_expert_boundary_after_resume": step < 30,
                        "condition_metadata_json": "{}",
                        "result_path": "mock",
                    }
                )
                rows.append(row)
    return rows


def test_v3_analysis_emits_corrected_tables_from_exact_key_matrix(tmp_path) -> None:
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    digest = "mock-config"
    provenance = {"provenance_hash": "mock-provenance"}
    rows = _mock_v3_matrix_rows(config, digest, provenance)
    write_csv(tmp_path / "raw_results.csv", rows, RAW_FIELDS)
    analyze_discovery(config, digest, provenance, tmp_path)
    for filename in (
        "matrix_completeness.json",
        "progress_summary.csv",
        "prompt_corruption_summary.csv",
        "iso_error_pairs.csv",
        "exploratory_incidental_iso_error_pairs.csv",
        "iso_storage_pairs.csv",
        "staleness_pairs.csv",
        "discovery_flags.json",
        "video_runtime_state_discovery.md",
    ):
        assert (tmp_path / filename).exists(), filename
    assert "temporal_delta" not in (tmp_path / "iso_error_pairs.csv").read_text().splitlines()[0]
    assert "temporal_delta" not in (tmp_path / "iso_storage_pairs.csv").read_text().splitlines()[0]

    mutated = copy.deepcopy(rows)
    mutated[-1]["clean_latent_hash"] = "mutated-clean-hash"
    write_csv(tmp_path / "raw_results.csv", mutated, RAW_FIELDS)
    with pytest.raises(RuntimeError, match="Clean checkpoint pairing mismatch"):
        analyze_discovery(config, digest, provenance, tmp_path)


def test_v2_scientific_analyzer_is_not_importable_from_v3_module() -> None:
    source = Path("experiments/video_runtime_state_discovery.py").read_text()
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    assert "_analyze_discovery_v2_legacy" not in source
    assert '"temporal_quality"' not in source
    assert '"spatial_quality"' not in source
    assert "spatial_down2" not in config["conditions"]
    assert "low_rank_25" not in config["conditions"]
    assert {
        name for name in config["conditions"] if CONDITION_SPECS[name].comparison_family == "iso_storage_runtime"
    } == set(PRIMARY_ISO_STORAGE_CONDITIONS)


def test_analysis_outputs_are_byte_identical_under_input_row_shuffle(tmp_path) -> None:
    """Every emitted analysis table must be independent of raw_results.csv row order."""
    config = load_config(Path("experiments/video_runtime_state_discovery_config.yaml"))
    digest = "mock-config"
    provenance = {"provenance_hash": "mock-provenance"}
    emitted = (
        "progress_summary.csv",
        "prompt_corruption_summary.csv",
        "iso_error_pairs.csv",
        "exploratory_incidental_iso_error_pairs.csv",
        "iso_storage_pairs.csv",
        "staleness_pairs.csv",
    )

    ordered_dir = tmp_path / "ordered"
    ordered_dir.mkdir()
    write_csv(ordered_dir / "raw_results.csv", _mock_v3_matrix_rows(config, digest, provenance), RAW_FIELDS)
    analyze_discovery(config, digest, provenance, ordered_dir)

    shuffled_rows = _mock_v3_matrix_rows(config, digest, provenance)
    random.Random(1234).shuffle(shuffled_rows)
    shuffled_dir = tmp_path / "shuffled"
    shuffled_dir.mkdir()
    write_csv(shuffled_dir / "raw_results.csv", shuffled_rows, RAW_FIELDS)
    analyze_discovery(config, digest, provenance, shuffled_dir)

    for filename in emitted:
        assert (ordered_dir / filename).read_text() == (shuffled_dir / filename).read_text(), filename
