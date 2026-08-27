from __future__ import annotations

import json

import pytest
import torch

from experiments import video_propagation_aware_checkpoint_killtest as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _latent() -> torch.Tensor:
    return torch.linspace(-2.0, 2.0, 1 * 8 * 3 * 4 * 4).reshape(1, 8, 3, 4, 4)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_bitpacking_round_trip(bits: int) -> None:
    maximum = (1 << bits) - 1
    values = torch.arange(37).remainder(maximum + 1).numpy().astype("uint8")
    packed = killtest._pack_unsigned(values, bits)
    restored = killtest._unpack_unsigned(packed, bits, len(values))
    assert restored.tolist() == values.tolist()


def test_grouped_serialization_accounts_for_actual_files(tmp_path) -> None:
    latent = _latent()
    groups = killtest.channel_groups(latent, 4)
    encoded = killtest._encode_grouped(latent, groups, (8, 4, 2, 16), tmp_path)

    payload = tmp_path / "checkpoint.payload.bin"
    metadata = tmp_path / "checkpoint.metadata.json"
    assert encoded.payload_bytes == payload.stat().st_size
    assert encoded.metadata_bytes == metadata.stat().st_size
    assert encoded.total_bytes == payload.stat().st_size + metadata.stat().st_size
    assert tuple(json.loads(metadata.read_text())["shape"]) == tuple(latent.shape)
    assert encoded.restored.shape == latent.shape


def test_full_serialization_is_exact_and_accounts_for_bytes(tmp_path) -> None:
    latent = _latent()
    encoded = killtest._encode_full(latent, tmp_path)
    assert torch.equal(encoded.restored, latent.float())
    assert encoded.payload_bytes == latent.numel() * 4
    assert encoded.total_bytes == sum(path.stat().st_size for path in tmp_path.iterdir())


def test_fp16_serialization_uses_dense_payload(tmp_path) -> None:
    latent = _latent()
    encoded = killtest._encode_fp16(latent, tmp_path)
    assert encoded.payload_bytes == latent.numel() * 2
    assert torch.equal(encoded.restored, latent.to(torch.float16).float())


def test_channel_partition_covers_every_element_once() -> None:
    latent = _latent()
    groups = killtest.channel_groups(latent, 8)
    indices = torch.cat([group.indices for group in groups])
    assert indices.numel() == latent.numel()
    assert torch.equal(torch.sort(indices).values, torch.arange(latent.numel()))


def test_uniform_allocation_matches_uniform_group_encoding(tmp_path) -> None:
    latent = _latent()
    groups = killtest.channel_groups(latent, 4)
    first = killtest._encode_grouped(latent, groups, (4, 4, 4, 4), tmp_path / "first")
    second = killtest._encode_grouped(latent, groups, (4,) * len(groups), tmp_path / "second")
    assert torch.equal(first.restored, second.restored)
    assert first.payload_bytes == second.payload_bytes


def test_equal_sensitivity_reduces_to_uniform_allocation() -> None:
    latent = _latent()
    groups = killtest.channel_groups(latent, 4)
    errors = killtest._group_quantization_errors(latent, groups)
    budget = killtest._allocation_size(latent, groups, (4, 4, 4, 4))
    allocation = killtest.choose_allocation(
        latent,
        groups,
        budget,
        errors,
        [1.0] * len(groups),
        uniform_if_equal=True,
    )
    assert allocation == (4, 4, 4, 4)


def test_larger_budget_does_not_worsen_predicted_oracle() -> None:
    latent = _latent()
    groups = killtest.channel_groups(latent, 4)
    errors = killtest._group_quantization_errors(latent, groups)
    sensitivities = [1.0, 2.0, 4.0, 8.0]
    allocations = []
    for bits in (2, 4, 8):
        budget = killtest._allocation_size(latent, groups, (bits,) * len(groups))
        allocation = killtest.choose_allocation(
            latent,
            groups,
            budget,
            errors,
            sensitivities,
            uniform_if_equal=False,
        )
        assert allocation is not None
        allocations.append(allocation)

    def score(allocation: tuple[int, ...]) -> float:
        return sum(errors[index][bits] * sensitivities[index] for index, bits in enumerate(allocation))

    scores = [score(allocation) for allocation in allocations]
    assert scores[1] <= scores[0] + 1e-12
    assert scores[2] <= scores[1] + 1e-12


def test_iso_byte_frontier_is_monotonic() -> None:
    rows = []
    for encoder, size, quality in (
        ("uniform_int2", 10, 0.70),
        ("uniform_int4", 20, 0.80),
        ("uniform_int8", 40, 0.90),
        ("fp16", 80, 0.99),
        ("propagation_oracle", 10, 0.75),
        ("propagation_oracle", 20, 0.88),
        ("propagation_oracle", 40, 0.87),
    ):
        rows.append(
            {
                "prompt_id": "p0",
                "checkpoint_step": 20,
                "encoder": encoder,
                "total_checkpoint_bytes": size,
                "joint_quality": quality,
                "allocation": encoder,
            }
        )
    frontier = killtest.build_iso_byte_frontier(rows)
    oracle = [row for row in frontier if row["frontier_method"] == "propagation_oracle"]
    qualities = [float(row["joint_quality"]) for row in oracle]
    assert qualities == sorted(qualities)


def test_probe_directions_cover_required_families() -> None:
    directions = killtest.build_probe_directions(_latent(), [4, 8])
    families = {item[0] for item in directions}
    assert families == {"channel", "tile", "temporal_frequency", "spatial_frequency"}
