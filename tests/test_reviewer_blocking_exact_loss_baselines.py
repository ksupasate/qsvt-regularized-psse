"""Workstream 2 tests: exact-loss greedy, oracle correctness, and data isolation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    RidgeTask,
    SupportConstraints,
    _ridge_filter_operator,
    support_constraint_report,
)
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    ExactLossEvaluator,
    exact_loss_greedy,
    exhaustive_oracle,
    near_oracle_multistart,
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


def _tasks(matrix: np.ndarray, split: str = "training") -> list[RidgeTask]:
    functionals = {
        "coordinate_e0": np.array([1.0, 0.0, 0.0, 0.0]),
        "aggregate": np.ones(4) / 2.0,
    }
    tasks = []
    for seed in (1, 2, 3):
        residual = np.random.default_rng(seed).normal(size=4)
        for name, functional in functionals.items():
            tasks.append(
                RidgeTask(
                    task_id=f"{split}_{seed}_{name}",
                    seed_id=seed,
                    split=split,
                    residual=residual,
                    functional_id=name,
                    functional=functional,
                )
            )
    return tasks


def _evaluator(matrix, tasks, alpha=1e-2, y_floor=1e-6) -> ExactLossEvaluator:
    return ExactLossEvaluator(matrix, tasks, alpha, y_floor)


def test_evaluator_matches_independent_normalized_loss():
    matrix = _matrix()
    tasks = _tasks(matrix)
    alpha = 1e-2
    evaluator = _evaluator(matrix, tasks, alpha)
    support = np.zeros_like(matrix, dtype=bool)
    for i in range(4):
        support[i, i] = True
    got = evaluator.normalized(support)
    sparse = np.where(support, matrix, 0.0)
    sparse_op = _ridge_filter_operator(sparse, alpha)
    expected = []
    for task in tasks:
        full = task.functional @ ridge_svd_solution(matrix, task.residual, alpha=alpha)
        approx = task.functional @ (sparse_op @ task.residual)
        expected.append(abs(approx - full) / max(abs(full), 1e-6))
    assert np.allclose(got, expected)


def test_exact_loss_greedy_respects_budgets_and_coverage():
    matrix = _matrix()
    evaluator = _evaluator(matrix, _tasks(matrix))
    constraints = SupportConstraints(k_budget=6, slot_budget=2, coverage_enabled=True)
    result = exact_loss_greedy(evaluator, constraints, objective="mean")
    assert result["status"] == "completed"
    report = support_constraint_report(matrix, result["support"], constraints)
    assert report["valid"]
    assert report["actual_nonzeros"] <= 6
    assert report["actual_max_row_degree"] <= 2
    assert report["actual_max_column_degree"] <= 2
    assert report["active_rows_covered"] and report["active_columns_covered"]


def test_exact_loss_greedy_is_deterministic():
    matrix = _matrix()
    constraints = SupportConstraints(6, 2, True)
    first = exact_loss_greedy(_evaluator(matrix, _tasks(matrix)), constraints, objective="mean")
    second = exact_loss_greedy(_evaluator(matrix, _tasks(matrix)), constraints, objective="mean")
    assert np.array_equal(first["support"], second["support"])
    assert first["final_loss"] == second["final_loss"]


def test_infeasible_budget_is_retained_not_raised():
    matrix = _matrix()
    evaluator = _evaluator(matrix, _tasks(matrix))
    # k below the 4-row/4-column coverage minimum is infeasible under coverage.
    constraints = SupportConstraints(2, 1, True)
    result = exact_loss_greedy(evaluator, constraints, objective="mean")
    assert result["status"] == "failed"
    assert result["support"] is None
    assert "coverage" in result["failure_reason"]


def test_exhaustive_oracle_matches_independent_bruteforce():
    matrix = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.5, 3.0, 1.0],
            [0.0, 0.7, 2.0],
        ]
    )
    tasks = _tasks_3(matrix)
    alpha = 1e-1
    evaluator = _evaluator(matrix, tasks, alpha)
    constraints = SupportConstraints(k_budget=5, slot_budget=2, coverage_enabled=True)
    oracle = exhaustive_oracle(evaluator, constraints, objective="mean", max_candidates=10_000)
    assert oracle["status"] == "completed"

    # Independent brute force over all subsets of candidate nonzeros.
    positions = [tuple(int(v) for v in p) for p in np.argwhere(matrix != 0.0)]
    active_rows = set(np.flatnonzero(np.any(matrix != 0.0, axis=1)).tolist())
    active_cols = set(np.flatnonzero(np.any(matrix != 0.0, axis=0)).tolist())
    best_loss = np.inf
    for size in range(1, constraints.k_budget + 1):
        for subset in itertools.combinations(positions, size):
            rows = [r for r, _ in subset]
            cols = [c for _, c in subset]
            if any(rows.count(r) > 2 for r in set(rows)):
                continue
            if any(cols.count(c) > 2 for c in set(cols)):
                continue
            if not active_rows.issubset(rows) or not active_cols.issubset(cols):
                continue
            support = np.zeros_like(matrix, dtype=bool)
            for r, c in subset:
                support[r, c] = True
            sparse_op = _ridge_filter_operator(np.where(support, matrix, 0.0), alpha)
            losses = []
            for task in tasks:
                full = task.functional @ ridge_svd_solution(matrix, task.residual, alpha=alpha)
                approx = task.functional @ (sparse_op @ task.residual)
                losses.append(abs(approx - full) / max(abs(full), 1e-6))
            best_loss = min(best_loss, float(np.mean(losses)))
    assert oracle["final_loss"] == pytest.approx(best_loss, rel=1e-9, abs=1e-12)


def test_oracle_is_no_worse_than_any_feasible_support():
    matrix = np.array([[4.0, 1.0, 0.0], [0.5, 3.0, 1.0], [0.0, 0.7, 2.0]])
    evaluator = _evaluator(matrix, _tasks_3(matrix), 1e-1)
    constraints = SupportConstraints(5, 2, True)
    oracle = exhaustive_oracle(evaluator, constraints, objective="worst", max_candidates=10_000)
    greedy = exact_loss_greedy(evaluator, constraints, objective="worst")
    assert oracle["final_loss"] <= greedy["final_loss"] + 1e-12


def test_near_oracle_multistart_dominates_every_seed():
    matrix = _matrix()
    tasks = _tasks(matrix)
    evaluator = _evaluator(matrix, tasks)
    constraints = SupportConstraints(6, 2, True)
    # Build a few feasible seeds via greedy from both objectives.
    seeds = [
        exact_loss_greedy(evaluator, constraints, objective="mean")["support"],
        exact_loss_greedy(evaluator, constraints, objective="worst")["support"],
    ]
    beam = {"final_loss": None, "beam_width": 4}
    near = near_oracle_multistart(
        evaluator, tasks, constraints, objective="mean", seed_supports=seeds, beam_diagnostic=beam
    )
    assert near["status"] == "completed"
    for seed in seeds:
        assert near["final_loss"] <= evaluator.loss(seed, "mean") + 1e-12


def _tasks_3(matrix: np.ndarray) -> list[RidgeTask]:
    functional = np.array([1.0, 0.0, 0.0])
    tasks = []
    for seed in (7, 11):
        residual = np.random.default_rng(seed).normal(size=3)
        tasks.append(
            RidgeTask(
                task_id=f"training_{seed}",
                seed_id=seed,
                split="training",
                residual=residual,
                functional_id="coordinate_e0",
                functional=functional,
            )
        )
    return tasks


def test_selection_uses_training_tasks_only():
    matrix = _matrix()
    training = _tasks(matrix, split="training")
    assert all(task.split == "training" for task in training)
    evaluator = _evaluator(matrix, training)
    assert all(task.split == "training" for task in evaluator.tasks)
