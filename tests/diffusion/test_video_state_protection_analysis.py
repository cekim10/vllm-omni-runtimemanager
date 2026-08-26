from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from experiments.video_state_protection_analysis import (
    EXPECTED_MODEL,
    POLICIES,
    QUALITY_METRICS,
    VARIANTS,
    _resolve_threshold_noise_decision,
    build_global_policy_predictions,
    build_iso_storage,
    build_policy_predictions,
    load_analysis_config,
    paired_relative_values,
    run_budget_simulation,
    validate_analysis_config,
    validate_frontier,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _frontier_rows(prompt_count: int = 2, seed_count: int = 2) -> list[dict[str, str]]:
    rows = []
    sizes = {
        "spatial_down2": 100,
        "int8": 102,
        "low_rank_25": 104,
        "fp16": 200,
        "temporal_down2": 205,
        "full": 400,
    }
    for prompt_index in range(prompt_count):
        for seed_index in range(seed_count):
            for step in (10, 20, 30):
                for variant in VARIANTS:
                    base = 1.0
                    if variant == "spatial_down2":
                        base = 0.6 if prompt_index == 0 else 0.8
                    elif variant == "low_rank_25":
                        base = 0.5
                    row = {
                        "prompt_set_sha256": "fixture-prompt-hash",
                        "model": EXPECTED_MODEL,
                        "prompt_id": f"recovery_{prompt_index:03d}",
                        "category": "test",
                        "seed": str(1000 + prompt_index * 10 + seed_index),
                        "seed_index": str(seed_index),
                        "checkpoint_step": str(step),
                        "total_steps": "40",
                        "progress_fraction": str(step / 40),
                        "variant": variant,
                        "raw_latent_bytes": "400",
                        "encoded_payload_bytes": str(sizes[variant] - 4),
                        "metadata_bytes": "4",
                        "total_checkpoint_bytes": str(sizes[variant]),
                        "compression_ratio_vs_full": str(sizes[variant] / sizes["full"]),
                        "encode_prepare_latency_ms": "1.0",
                        "storage_write_latency_ms": "1.0",
                        "load_read_latency_ms": "1.0",
                        "decode_reconstruction_latency_ms": "1.0",
                        "content_complexity_score": str(prompt_index / max(prompt_count - 1, 1)),
                    }
                    for metric in QUALITY_METRICS:
                        row[metric] = str(base)
                    rows.append(row)
    return rows


def _policy_cells(prompt_count: int = 4) -> dict:
    cells = {}
    sizes = {
        "spatial_down2": 100.0,
        "int8": 102.0,
        "low_rank_25": 104.0,
        "fp16": 200.0,
        "temporal_down2": 205.0,
        "full": 400.0,
    }
    for prompt_index in range(prompt_count):
        prompt_id = f"recovery_{prompt_index:03d}"
        for step in (10, 20, 30):
            for target in (0.95, 0.975, 0.99):
                variants = {
                    variant: {
                        "bytes": size,
                        f"safe_{target}": variant in {"int8", "fp16", "full"},
                    }
                    for variant, size in sizes.items()
                }
                cells[(prompt_id, step, target)] = {
                    "row": {
                        "selected_representation": "int8",
                        "selected_total_checkpoint_bytes": sizes["int8"],
                    },
                    "variants": variants,
                    "content_complexity": prompt_index / max(prompt_count - 1, 1),
                }
    return cells


def test_n2_frontier_cannot_validate_as_n5() -> None:
    rows = _frontier_rows(seed_count=2)

    with pytest.raises(ValueError, match="cannot produce an n=5 report"):
        validate_frontier(rows, expected_prompts=2, expected_seeds=5)


def test_iso_storage_uses_prompt_means_as_population_units() -> None:
    rows = _frontier_rows(prompt_count=2, seed_count=2)

    output = build_iso_storage(
        rows,
        tolerance=0.05,
        samples=500,
        seed=7,
        expected_seeds=2,
    )
    cell = next(
        row
        for row in output
        if row["checkpoint_step"] == 10 and row["pair"] == "int8__vs__spatial_down2"
    )

    assert cell["prompt_count"] == 2
    assert cell["seed_count"] == 2
    assert cell["sample_unit"] == "prompt_mean_of_paired_seed_deltas"
    assert cell["dynamic_delta_mean"] == pytest.approx(0.3)
    assert cell["resolution_seed_count"] == 2
    assert "resolved_at_n5" not in cell


def test_simple_policies_use_leave_one_prompt_out_predictions() -> None:
    cells = _policy_cells(prompt_count=4)

    details, predictions = build_policy_predictions(
        cells,
        targets=[0.95, 0.975, 0.99],
        complexity_bins=3,
        max_training_violation=0.05,
    )

    assert len(details) == 4 * 3 * 3 * 3
    assert all(row["evaluation"] == "leave_one_prompt_out" for row in details)
    assert {
        policy for (_, _, _, policy) in predictions
    } == {"progress_only", "content_only", "simple_separable"}


def test_budget_simulator_emits_six_unique_policies_per_cell() -> None:
    cells = _policy_cells(prompt_count=4)
    predictions, definitions = build_global_policy_predictions(
        cells,
        targets=[0.95, 0.975, 0.99],
        complexity_bins=3,
        max_training_violation=0.05,
    )
    assert all(definition["deployable"] is False for definition in definitions)
    for target in (0.95, 0.975, 0.99):
        for step in (10, 20, 30):
            choices = {
                predictions[(f"recovery_{prompt_index:03d}", step, target, "progress_only")]
                for prompt_index in range(4)
            }
            assert len(choices) == 1

    rows = run_budget_simulation(
        cells,
        predictions,
        session_counts=[3],
        budget_fractions=[0.5],
        trials=1,
        seed=7,
    )

    grouped = {}
    for row in rows:
        key = (row["mixture"], row["session_count"], row["budget_fraction_of_all_full"], row["trial_index"])
        grouped.setdefault(key, []).append(row["policy"])
    assert grouped
    assert all(policies == POLICIES for policies in grouped.values())
    assert all(len(policies) == len(set(policies)) for policies in grouped.values())
    mixed = [row for row in rows if row["mixture"] == "mixed_targets"]
    assert mixed
    for row in mixed:
        session_ids = json.loads(row["sampled_session_ids_json"])
        assert row["sampled_unique_session_count"] == 3
        assert len(session_ids) == len(set(session_ids)) == 3


def test_identical_frontier_makes_global_simple_match_oracle() -> None:
    cells = _policy_cells(prompt_count=4)
    predictions, _ = build_global_policy_predictions(cells, [0.95, 0.975, 0.99], 3, 0.05)

    rows = run_budget_simulation(cells, predictions, [3], [0.5], 3, 7)

    simple = [row for row in rows if row["policy"] == "simple_separable"]
    assert simple
    assert all(row["absolute_oracle_gap_sessions"] == 0 for row in simple)


def test_nonseparable_frontier_can_create_global_simple_oracle_gap() -> None:
    cells = _policy_cells(prompt_count=4)
    requirement = {
        (0, 10): "int8", (0, 20): "full", (0, 30): "int8",
        (1, 10): "full", (1, 20): "int8", (1, 30): "full",
        (2, 10): "int8", (2, 20): "full", (2, 30): "int8",
        (3, 10): "full", (3, 20): "int8", (3, 30): "full",
    }
    for (prompt_id, step, target), cell in cells.items():
        prompt_index = int(prompt_id.rsplit("_", 1)[1])
        selected = requirement[(prompt_index, step)]
        selected_bytes = cell["variants"][selected]["bytes"]
        for info in cell["variants"].values():
            info[f"safe_{target}"] = info["bytes"] >= selected_bytes
        cell["row"]["selected_representation"] = selected
        cell["row"]["selected_total_checkpoint_bytes"] = selected_bytes
    predictions, _ = build_global_policy_predictions(cells, [0.95, 0.975, 0.99], 2, 0.0)

    rows = run_budget_simulation(cells, predictions, [3], [0.5], 20, 7)

    assert any(
        row["absolute_oracle_gap_sessions"] > 0
        for row in rows
        if row["policy"] == "simple_separable"
    )


def test_analysis_config_preregisters_all_judgment_thresholds() -> None:
    config = load_analysis_config(Path("experiments/video_state_protection_stage_b_analysis_config.yaml"))
    args = type(
        "Args",
        (),
        {
            "expected_prompts": 12,
            "expected_seeds": 5,
            "bootstrap_samples": 5000,
            "bootstrap_seed": 7,
            "iso_storage_tolerance": 0.05,
            "complexity_bins": 3,
            "simple_policy_max_training_violation": 0.05,
            "budget_trials": 200,
            "budget_session_counts": [3, 8, 20],
            "budget_fractions": [0.25, 0.5, 0.75],
        },
    )()

    validate_analysis_config(config, args)
    assert len(config["judgment"]) == 17


def test_resume_validation_decodes_and_hashes_serialized_payload(tmp_path: Path) -> None:
    from experiments import video_state_protection_killtest as killtest

    serialized_dir = tmp_path / "serialized" / "full"
    encoded = killtest._encode_representation(torch.randn(1, 2, 3, 4, 5), "full", serialized_dir)
    video_path = tmp_path / "recovered.mp4"
    video_path.write_bytes(b"validated-video-placeholder")
    provenance = killtest.PromptProvenance(
        resolved_path=tmp_path / "prompts.json",
        sha256="prompt-hash",
        prompt_ids=["recovery_000"],
        categories=["test"],
        entries=[{"prompt_id": "recovery_000", "motion_category": "test", "prompt": "test"}],
    )
    args = Namespace(
        checkpoint_steps=[10],
        variants=["full"],
        model="test-model",
        seed_base=1234,
        num_inference_steps=40,
    )
    row = {
        "prompt_set_sha256": provenance.sha256,
        "model": args.model,
        "prompt_id": "recovery_000",
        "seed": "1234",
        "seed_index": "0",
        "checkpoint_step": "10",
        "total_steps": "40",
        "progress_fraction": "0.25",
        "variant": "full",
        "raw_latent_bytes": str(encoded.raw_latent_bytes),
        "encoded_payload_bytes": str(encoded.encoded_payload_bytes),
        "metadata_bytes": str(encoded.metadata_bytes),
        "total_checkpoint_bytes": str(encoded.total_checkpoint_bytes),
        "resume_latency_ms": "1.0",
        "spatial_metric_abs": "1.0",
        "temporal_dynamic_composite_abs": "1.0",
        "semantic_metric_abs": "1.0",
        "spatial_vs_full": "1.0",
        "temporal_dynamic_vs_full": "1.0",
        "semantic_vs_full": "1.0",
        "artifact_path": str(video_path),
        "serialized_artifact_dir": str(serialized_dir),
        "variant_metadata_json": json.dumps(encoded.metadata, sort_keys=True),
    }

    valid = killtest._validate_existing_frontier_rows([row], args=args, provenance=provenance)
    assert len(valid) == 1
    assert len(valid[0]["serialized_payload_sha256"]) == 64
    assert len(valid[0]["serialized_metadata_sha256"]) == 64

    payload_path = serialized_dir / "full.payload.bin"
    payload_path.write_bytes(payload_path.read_bytes()[:-1])
    assert killtest._validate_existing_frontier_rows([row], args=args, provenance=provenance) == []


def test_noise_floor_requires_stable_threshold_decision() -> None:
    target = 0.99
    stats = {
        "int8": {"bytes": 100.0, f"safe_{target}": False},
        "fp16": {"bytes": 200.0, f"safe_{target}": True},
        "full": {"bytes": 400.0, f"safe_{target}": True},
    }
    noise = {}
    for variant, probability in (("int8", 0.0), ("fp16", 1.0), ("full", 1.0)):
        for metric in QUALITY_METRICS:
            noise[("recovery_000", 10, variant, metric)] = {
                "probability_ge_0_99": probability,
                "ordering_stable": True,
            }

    resolved = _resolve_threshold_noise_decision(
        "recovery_000", 10, target, "fp16", stats, noise
    )
    assert resolved["decision_above_noise_floor"] is True
    assert resolved["noise_floor_status"] == "threshold_stable"

    noise[("recovery_000", 10, "int8", QUALITY_METRICS[0])]["probability_ge_0_99"] = 0.4
    ambiguous = _resolve_threshold_noise_decision(
        "recovery_000", 10, target, "fp16", stats, noise
    )
    assert ambiguous["decision_above_noise_floor"] is False
    assert ambiguous["noise_floor_status"] == "threshold_ambiguous"


def test_full_representation_is_not_noise_resolved_without_measurement() -> None:
    target = 0.99
    stats = {
        "int8": {"bytes": 100.0, f"safe_{target}": False},
        "fp16": {"bytes": 200.0, f"safe_{target}": False},
        "full": {"bytes": 400.0, f"safe_{target}": True},
    }

    decision = _resolve_threshold_noise_decision(
        "recovery_000", 10, target, "full", stats, {}
    )

    assert decision["decision_above_noise_floor"] == ""
    assert decision["noise_floor_status"] == "unmeasured"


def test_noise_relative_quality_is_paired_by_repeat() -> None:
    values = [0.90, 1.10]
    full_values = [0.90, 1.10]

    assert paired_relative_values(values, full_values) == [1.0, 1.0]
