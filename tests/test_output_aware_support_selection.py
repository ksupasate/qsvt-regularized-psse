"""Constraint, determinism, random-baseline, and refinement tests."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    assign_slot_permutations,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    refine_support_one_swap,
    select_resource_constrained_support,
    support_constraint_report,
)


def _matrix() -> np.ndarray:
    return np.array(
        [
            [8.0, 0.8, 0.5, 0.2],
            [0.7, 7.0, 0.6, 0.3],
            [0.4, 0.9, 6.0, 0.5],
            [0.6, 0.2, 0.7, 5.0],
        ]
    )


def _training_tasks() -> list[RidgeTask]:
    functionals = {
        "coordinate_e0": np.array([1.0, 0.0, 0.0, 0.0]),
        "signed_difference_e0_minus_e1": np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2),
        "aggregate_e0_to_e3": np.ones(4) / 2.0,
    }
    tasks = []
    for seed in (1, 2):
        residual = np.random.default_rng(seed).normal(size=4)
        for functional_id, functional in functionals.items():
            tasks.append(
                RidgeTask(
                    task_id=f"training_{seed}_{functional_id}",
                    seed_id=seed,
                    split="training",
                    residual=residual,
                    functional_id=functional_id,
                    functional=functional,
                )
            )
    return tasks


def test_milp_support_obeys_cardinality_degree_and_coverage_constraints():
    matrix = _matrix()
    constraints = SupportConstraints(k_budget=8, slot_budget=2, coverage_enabled=True)
    result = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    assert result.status == "completed"
    assert result.solver_status == "optimal"
    assert result.solver_used == "scipy.optimize.milp_highs"
    assert result.optimality_gap is not None
    report = support_constraint_report(matrix, result.support, constraints)
    assert report["valid"]
    assert report["actual_nonzeros"] <= 8
    assert report["actual_max_row_degree"] <= 2
    assert report["actual_max_column_degree"] <= 2
    assert report["active_rows_covered"]
    assert report["active_columns_covered"]


def test_support_selection_is_deterministic_and_slot_decomposition_succeeds():
    matrix = _matrix()
    constraints = SupportConstraints(8, 2, True)
    first = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    second = select_resource_constrained_support(matrix, np.abs(matrix), constraints)
    assert np.array_equal(first.support, second.support)
    pattern = first.support.T
    assignment = assign_slot_permutations(pattern)
    validation = validate_slot_assignment(pattern, assignment)
    assert validation["valid"]
    assert assignment.slots <= constraints.slot_budget


def test_greedy_fallback_is_feasible_but_never_labeled_optimal():
    matrix = _matrix()
    constraints = SupportConstraints(8, 2, True)
    result = select_resource_constrained_support(
        matrix,
        np.abs(matrix),
        constraints,
        force_greedy_fallback=True,
    )
    assert result.status == "completed"
    assert result.fallback_used
    assert result.solver_status == "feasible_greedy"
    assert "optimal" not in result.solver_status
    assert result.optimality_gap is None
    assert support_constraint_report(matrix, result.support, constraints)["valid"]


def test_fixed_random_seed_reproduces_random_feasible_support():
    matrix = _matrix()
    constraints = SupportConstraints(8, 2, True)

    def selected(seed: int):
        rng = np.random.default_rng(seed)
        scores = rng.random(matrix.shape)
        return select_resource_constrained_support(matrix, scores, constraints).support

    assert np.array_equal(selected(991), selected(991))
    assert not np.array_equal(selected(991), selected(992))


def test_coverage_infeasibility_is_retained_with_status():
    matrix = _matrix()
    result = select_resource_constrained_support(
        matrix,
        np.abs(matrix),
        SupportConstraints(k_budget=3, slot_budget=2, coverage_enabled=True),
    )
    assert result.status == "failed"
    assert result.solver_status == "infeasible"
    assert "coverage_constraint_infeasible" in result.failure_reason


def test_exact_loss_refinement_is_deterministic_strict_and_constraint_preserving():
    matrix = _matrix()
    constraints = SupportConstraints(8, 2, True)
    initial = select_resource_constrained_support(
        matrix,
        np.arange(matrix.size, dtype=float).reshape(matrix.shape) + 1.0,
        constraints,
    ).support
    kwargs = {
        "alpha": 0.8,
        "y_floor": 1.0e-6,
        "objective": "mean_normalized_error",
        "max_iterations": 3,
        "improvement_tolerance": 1.0e-12,
    }
    first = refine_support_one_swap(
        matrix, initial, _training_tasks(), constraints, **kwargs
    )
    second = refine_support_one_swap(
        matrix, initial, _training_tasks(), constraints, **kwargs
    )
    assert np.array_equal(first.support, second.support)
    assert first.trace == second.trace
    assert first.iterations_accepted <= 3
    assert support_constraint_report(matrix, first.support, constraints)["valid"]
    accepted = [row for row in first.trace if row["action"] == "one_swap"]
    assert all(row["objective_after"] < row["objective_before"] for row in accepted)
    assert first.final_objective <= first.initial_objective
