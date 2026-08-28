from __future__ import annotations

import numpy as np
import pytest

from experiments.video_state_protection_independent_audit import (
    LAYOUTS,
    error,
    expected_mask,
    independent_video_metrics,
    shard_roundtrip,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _coordinate_tensor() -> np.ndarray:
    result = np.empty((1, 4, 4, 4, 4), dtype=np.float32)
    for c in range(4):
        for t in range(4):
            for h in range(4):
                for w in range(4):
                    result[0, c, t, h, w] = c * 10000 + t * 1000 + h * 100 + w
    return result


def test_named_masks_target_semantic_axes() -> None:
    shape = _coordinate_tensor().shape
    temporal = expected_mask(shape, "temporal_contiguous", 4, 7)
    channel = expected_mask(shape, "channel_contiguous", 4, 7)
    spatial = expected_mask(shape, "spatial_contiguous", 4, 7)
    assert np.flatnonzero(temporal.any(axis=(0, 1, 3, 4))).tolist() == [0]
    assert np.flatnonzero(channel.any(axis=(0, 2, 3, 4))).tolist() == [0]
    assert spatial[..., :2, :2].all()
    assert not spatial[..., 2:, :].any()
    assert not spatial[..., :, 2:].any()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_disk_shards_are_exact_and_only_zero_missing_coordinates(tmp_path, layout: str) -> None:
    source = _coordinate_tensor()
    exact, _, sizes = shard_roundtrip(source, layout, 4, 7, tmp_path, None)
    damaged, missing, _ = shard_roundtrip(source, layout, 4, 7, tmp_path, 0)
    assert error(source, exact)["max_abs"] == 0
    assert len(set(sizes)) == 1
    assert np.array_equal(damaged[~missing], source[~missing])
    assert np.all(damaged[missing] == 0)


def test_independent_metric_controls() -> None:
    reference = np.arange(8 * 8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 8, 3)
    self_metrics = independent_video_metrics(reference, reference)
    shuffled = independent_video_metrics(reference[::-1].copy(), reference)
    zero = independent_video_metrics(np.zeros_like(reference), reference)
    assert self_metrics == {"video_mse": 0.0, "spatial_quality": 1.0, "temporal_dynamic_quality": 1.0}
    assert shuffled["temporal_dynamic_quality"] < 0.95
    assert zero["spatial_quality"] < 1.0
