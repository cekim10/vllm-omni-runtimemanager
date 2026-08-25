import math

import pytest
import torch

from experiments.video_recovery_frontier import (
    SUMMARY_FIELDS,
    _aggregate_rows,
    _apply_variant,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("variant", "expected_ratio_upper_bound"),
    [
        ("full", 1.0),
        ("fp16", 0.51),
        ("int8", 0.26),
        ("int4", 0.14),
        ("spatial_down2", 0.30),
        ("temporal_down2", 0.55),
        ("low_rank_25", 0.80),
    ],
)
def test_apply_variant_preserves_shape_and_reduces_retained_bytes(
    variant: str,
    expected_ratio_upper_bound: float,
) -> None:
    latent = torch.linspace(-1.0, 1.0, steps=2 * 3 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 3, 4, 8, 8)

    perturbed = _apply_variant(latent, variant)

    assert perturbed.restored_latent.shape == latent.shape
    assert perturbed.restored_latent.dtype == latent.dtype
    assert torch.isfinite(perturbed.restored_latent).all()

    full_bytes = latent.nelement() * latent.element_size()
    retained_ratio = perturbed.retained_bytes / full_bytes
    assert retained_ratio <= expected_ratio_upper_bound
    assert perturbed.retained_bytes > 0
    assert perturbed.metadata["type"]


def test_apply_variant_unknown_raises() -> None:
    latent = torch.zeros((1, 2, 2, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="Unsupported variant"):
        _apply_variant(latent, "unknown")


def test_aggregate_rows_summarizes_per_step_and_variant() -> None:
    rows = [
        {
            "checkpoint_step": 10,
            "variant": "int8",
            "retained_ratio": 0.25,
            "resume_latency_ms": 100.0,
            "spatial_metric": 0.9,
            "temporal_shape_composite": 0.8,
            "temporal_dynamic_composite": 0.7,
            "semantic_metric": 0.95,
            "video_mse": 1.0,
            "exact_equal": False,
        },
        {
            "checkpoint_step": 10,
            "variant": "int8",
            "retained_ratio": 0.35,
            "resume_latency_ms": 140.0,
            "spatial_metric": 0.7,
            "temporal_shape_composite": 0.6,
            "temporal_dynamic_composite": 0.5,
            "semantic_metric": 0.85,
            "video_mse": 3.0,
            "exact_equal": True,
        },
    ]

    summary = _aggregate_rows(rows)

    assert len(summary) == 1
    row = summary[0]
    assert list(row.keys()) == SUMMARY_FIELDS
    assert row["checkpoint_step"] == 10
    assert row["variant"] == "int8"
    assert row["num_rows"] == 2
    assert math.isclose(row["mean_retained_ratio"], 0.30)
    assert math.isclose(row["mean_resume_latency_ms"], 120.0)
    assert math.isclose(row["mean_spatial_metric"], 0.8)
    assert math.isclose(row["mean_temporal_shape_composite"], 0.7)
    assert math.isclose(row["mean_temporal_dynamic_composite"], 0.6)
    assert math.isclose(row["mean_semantic_metric"], 0.9)
    assert math.isclose(row["mean_video_mse"], 2.0)
    assert math.isclose(row["exact_equal_rate"], 0.5)
