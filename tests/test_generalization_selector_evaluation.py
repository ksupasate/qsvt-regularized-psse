from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    select_resource_constrained_support,
    support_constraint_report,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_small_support_selection_is_deterministic_and_constrained() -> None:
    matrix = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 4.0]])
    constraints = SupportConstraints(k_budget=5, slot_budget=2, coverage_enabled=True)
    first = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    second = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    assert first.solver_status == "optimal"
    assert np.array_equal(first.support, second.support)
    assert support_constraint_report(matrix, first.support, constraints)["valid"]


def test_fixed_seed_random_objective_reproduces() -> None:
    matrix = np.ones((4, 4))
    constraints = SupportConstraints(k_budget=8, slot_budget=2, coverage_enabled=True)
    first_scores = np.random.default_rng(314159).random(matrix.shape)
    second_scores = np.random.default_rng(314159).random(matrix.shape)
    first = select_resource_constrained_support(matrix, first_scores, constraints)
    second = select_resource_constrained_support(matrix, second_scores, constraints)
    assert np.array_equal(first.support, second.support)


def test_infeasible_budget_is_retained_as_failure() -> None:
    matrix = np.eye(4)
    result = select_resource_constrained_support(
        matrix,
        np.ones_like(matrix),
        SupportConstraints(k_budget=3, slot_budget=1, coverage_enabled=True),
    )
    assert result.status == "failed"
    assert "coverage_constraint_infeasible" in result.failure_reason


