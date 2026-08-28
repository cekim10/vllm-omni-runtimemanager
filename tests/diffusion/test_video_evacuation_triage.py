from __future__ import annotations

import pytest

from experiments.video_evacuation_triage_killtest import (
    CostProfile,
    Session,
    Shape,
    evacuation_cost_ms,
    evaluate_policies,
    latent_elements,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _session(session_id: int, value: float, cost: float) -> Session:
    return Session(
        session_id=session_id,
        prompt_id="recovery_000",
        checkpoint_step=20,
        shape=Shape("test", 480, 832, 33),
        shape_scale=1.0,
        value_ms=value,
        costs_ms={
            "full": cost * 4,
            "fp16": cost * 2,
            "int8": cost,
            "spatial_down2": cost,
            "temporal_down2": cost * 2,
            "low_rank_25": cost * 2,
        },
        safe_variants=frozenset({"full", "fp16", "int8"}),
    )


def test_reference_shape_matches_measured_wan_latent_elements() -> None:
    assert latent_elements(Shape("reference", 480, 832, 33)) == 898_560


def test_evacuation_cost_includes_scaled_copy_encode_and_transfer() -> None:
    profile = CostProfile("int8", payload_bytes=1_000_000, metadata_bytes=100, d2h_ms=1.0, encode_ms=2.0)
    cost = evacuation_cost_ms(profile, shape_scale=2.0, bandwidth_gbps=1.0, fixed_latency_ms=2.0)
    assert cost == pytest.approx(24.0008)


def test_fractional_oracle_upper_bounds_feasible_density_policy() -> None:
    results = {row.policy: row for row in evaluate_policies([_session(0, 10, 6), _session(1, 9, 6)], 10)}
    assert results["minimum_safe_value_density"].protected_work_ms == 10
    assert results["fractional_oracle_upper_bound"].protected_work_ms == pytest.approx(16)
    assert results["minimum_safe_value_density"].evacuation_time_ms <= 10


def test_all_policies_match_when_deadline_fits_every_session() -> None:
    results = {row.policy: row for row in evaluate_policies([_session(0, 10, 2), _session(1, 9, 2)], 100)}
    assert results["minimum_safe_value_density"].protected_work_ms == 19
    assert results["fractional_oracle_upper_bound"].protected_work_ms == 19
