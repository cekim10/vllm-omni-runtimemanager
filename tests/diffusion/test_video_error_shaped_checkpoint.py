from __future__ import annotations

import json

import pytest
import torch

from experiments import video_error_shaped_checkpoint_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _latent() -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 1 * 8 * 5 * 4 * 4).reshape(1, 8, 5, 4, 4)


@pytest.mark.parametrize("layout", killtest.LAYOUTS)
@pytest.mark.parametrize("shard_count", [2, 4, 8])
def test_layout_round_trip_is_exact_without_loss(layout: str, shard_count: int) -> None:
    latent = _latent()
    plan = killtest.build_layout_plan(
        tuple(latent.shape), layout, shard_count, random_seed=20260827
    )
    shards = killtest.encode_layout_bytes(latent, plan)
    restored = killtest.reassemble_layout(tuple(shards), plan)
    assert torch.equal(restored, latent.float())
    assert torch.equal(torch.sort(plan.storage_order).values, torch.arange(latent.numel()))


@pytest.mark.parametrize("shard_count", [2, 4, 8])
def test_all_layouts_lose_identical_payload_bytes(shard_count: int) -> None:
    latent = _latent()
    sizes = {}
    for layout in killtest.LAYOUTS:
        plan = killtest.build_layout_plan(
            tuple(latent.shape), layout, shard_count, random_seed=7
        )
        shards = killtest.encode_layout_bytes(latent, plan)
        sizes[layout] = len(shards[0])
    assert len(set(sizes.values())) == 1
    assert next(iter(sizes.values())) == latent.numel() * 4 // shard_count


@pytest.mark.parametrize("layout", killtest.LAYOUTS)
def test_missing_shard_matches_declared_source_geometry(layout: str) -> None:
    latent = _latent()
    plan = killtest.build_layout_plan(tuple(latent.shape), layout, 4, random_seed=19)
    shards = list(killtest.encode_layout_bytes(latent, plan))
    shards[0] = None
    restored = killtest.reassemble_layout(tuple(shards), plan)
    expected = latent.float().reshape(-1).clone()
    expected[killtest.missing_source_indices(plan)] = 0.0
    assert torch.equal(restored.reshape(-1), expected)


def test_random_and_striped_layouts_are_deterministic_and_distributed() -> None:
    shape = tuple(_latent().shape)
    random_a = killtest.build_layout_plan(shape, "random_element", 4, random_seed=31)
    random_b = killtest.build_layout_plan(shape, "random_element", 4, random_seed=31)
    random_c = killtest.build_layout_plan(shape, "random_element", 4, random_seed=32)
    striped_a = killtest.build_layout_plan(shape, "interleaved_striped", 4, random_seed=31)
    striped_b = killtest.build_layout_plan(shape, "interleaved_striped", 4, random_seed=99)
    assert torch.equal(random_a.storage_order, random_b.storage_order)
    assert not torch.equal(random_a.storage_order, random_c.storage_order)
    assert torch.equal(striped_a.storage_order, striped_b.storage_order)
    missing = killtest.missing_source_indices(striped_a)
    assert missing.max() - missing.min() > _latent().numel() // 2
    _, channels, temporal, height, width = shape
    channel_ids = (missing // (temporal * height * width)) % channels
    temporal_ids = (missing // (height * width)) % temporal
    assert channel_ids.unique().numel() == channels
    assert temporal_ids.unique().numel() == temporal


def test_same_storage_geometry_has_same_reconstruction() -> None:
    latent = _latent()
    original = killtest.build_layout_plan(
        tuple(latent.shape), "random_element", 4, random_seed=41
    )
    alias = killtest.LayoutPlan(
        layout="layout_alias",
        shape=original.shape,
        shard_count=original.shard_count,
        random_seed=original.random_seed,
        storage_order=original.storage_order.clone(),
        shard_ranges=original.shard_ranges,
        block_elements=original.block_elements,
    )
    shards = list(killtest.encode_layout_bytes(latent, original))
    shards[0] = None
    first = killtest.reassemble_layout(tuple(shards), original)
    second = killtest.reassemble_layout(tuple(shards), alias)
    assert torch.equal(first, second)


def test_serialized_layout_accounts_for_actual_shards(tmp_path) -> None:
    latent = _latent()
    plan = killtest.build_layout_plan(
        tuple(latent.shape), "interleaved_striped", 4, random_seed=5
    )
    encoded = killtest.serialize_layout(latent, plan, tmp_path)
    shard_paths = sorted(tmp_path.glob("shard_*.bin"))
    metadata_path = tmp_path / "checkpoint.metadata.json"
    assert encoded.payload_bytes == sum(path.stat().st_size for path in shard_paths)
    assert encoded.metadata_bytes == metadata_path.stat().st_size
    assert encoded.total_bytes == encoded.payload_bytes + encoded.metadata_bytes
    assert json.loads(metadata_path.read_text())["layout"] == "interleaved_striped"
    restored_plan = killtest.layout_plan_from_metadata(metadata_path.read_bytes())
    assert torch.equal(restored_plan.storage_order, plan.storage_order)
    restored, _, _ = killtest.load_damaged_layout(encoded, lost_shards=())
    assert torch.equal(restored, latent.float())


def test_iso_error_control_matches_normalized_l2() -> None:
    latent = _latent()
    candidates = []
    for layout in killtest.LAYOUTS:
        plan = killtest.build_layout_plan(tuple(latent.shape), layout, 4, random_seed=11)
        shards = list(killtest.encode_layout_bytes(latent, plan))
        shards[0] = None
        candidates.append(killtest.reassemble_layout(tuple(shards), plan))
    target = min(
        killtest.correction.latent_error(latent, candidate)["normalized_l2"]
        for candidate in candidates
    )
    calibrated = [
        killtest.correction.calibrate_direction(latent, candidate - latent, target)[0]
        for candidate in candidates
    ]
    errors = [
        killtest.correction.latent_error(latent, candidate)["normalized_l2"]
        for candidate in calibrated
    ]
    assert max(errors) - min(errors) < 1e-6


def test_geometry_signal_requires_both_distributed_layouts_to_win() -> None:
    rows = [
        {
            "analysis_type": "iso_byte",
            "layout": "spatial_contiguous",
            "temporal_dynamic_quality": 0.60,
            "spatial_quality": 0.80,
        },
        {
            "analysis_type": "iso_byte",
            "layout": "temporal_contiguous",
            "temporal_dynamic_quality": 0.65,
            "spatial_quality": 0.82,
        },
        {
            "analysis_type": "iso_byte",
            "layout": "random_element",
            "temporal_dynamic_quality": 0.82,
            "spatial_quality": 0.91,
        },
        {
            "analysis_type": "iso_byte",
            "layout": "interleaved_striped",
            "temporal_dynamic_quality": 0.80,
            "spatial_quality": 0.89,
        },
    ]
    signal = killtest._cell_signal(rows, "iso_byte")
    assert signal["dynamic_advantage"] == pytest.approx(0.15)
    assert signal["spatial_floor"] == pytest.approx(0.89)
    rows[-1]["temporal_dynamic_quality"] = 0.64
    assert killtest._cell_signal(rows, "iso_byte")["dynamic_advantage"] < 0
