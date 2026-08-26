from __future__ import annotations

import pytest

from experiments.video_protection_upper_bound import (
    Representation,
    Session,
    admission_evaluation,
    exact_knapsack_select,
    representation_only_evaluation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _session(index: int, int8_safe: bool = True) -> Session:
    quality = 1.0 if int8_safe else 0.9
    representations = {
        "int8": Representation("int8", 5, 0.25, quality, quality, quality),
        "fp16": Representation("fp16", 10, 0.5, 1.0, 1.0, 1.0),
        "full": Representation("full", 20, 1.0, 1.0, 1.0, 1.0),
    }
    return Session(
        session_id=f"s{index}",
        prompt_id=f"recovery_{index:03d}",
        category="test",
        seed=index,
        seed_index=0,
        checkpoint_step=10 * (index + 1),
        total_steps=40,
        content_complexity=float(index),
        representations=representations,
        per_step_gpu_ms=1.0,
        fixed_resume_overhead_ms=0.0,
        timing_method="fixture",
    )


def test_exact_knapsack_finds_non_greedy_optimum() -> None:
    sessions = [_session(index) for index in range(3)]
    costs = [6, 5, 4]
    values = [9.0, 8.0, 7.0]
    items = []
    for index, (session, cost, value) in enumerate(zip(sessions, costs, values, strict=True)):
        representation = Representation("fixture", cost, 1.0, 1.0, 1.0, 1.0)
        items.append((index, session, representation, value))

    selected = exact_knapsack_select(items, budget=9)

    assert {item[0] for item in selected} == {1, 2}
    assert sum(item[3] for item in selected) == 15.0


def test_identical_frontiers_make_independent_equal_oracle() -> None:
    sessions = [_session(index) for index in range(3)]
    priorities = [1, 1, 1]

    independent = admission_evaluation(
        sessions, 0.99, 10, "minimum_safe_independent", "equal_session", priorities, {}
    )
    oracle = admission_evaluation(
        sessions, 0.99, 10, "joint_oracle", "equal_session", priorities, {}
    )

    assert independent["protected_session_count"] == oracle["protected_session_count"] == 2
    assert independent["checkpoint_bytes_used"] == oracle["checkpoint_bytes_used"]


def test_full_budget_protects_all_valid_sessions() -> None:
    sessions = [_session(0, int8_safe=True), _session(1, int8_safe=False), _session(2, int8_safe=True)]
    priorities = [1, 2, 4]
    budget = sum(session.representations["full"].bytes for session in sessions)

    oracle = admission_evaluation(
        sessions, 0.99, budget, "joint_oracle", "accumulated_work", priorities, {}
    )
    representation_only = representation_only_evaluation(
        sessions, 0.99, budget, "joint_oracle", "accumulated_work", priorities, {}
    )

    assert oracle["protected_session_count"] == len(sessions)
    assert representation_only["all_sessions_feasible"] == 1.0
    assert representation_only["quality_violation_count"] == 0.0
