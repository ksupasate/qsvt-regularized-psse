"""Generalized selector orchestration parameterized by the output-aware score functional set.

Mirrors ``reviewer_blocking.exact_loss_baselines.produce_supports`` /
``joint_feasibility.select_support`` / ``resource_pareto.build_support`` but replaces the
hardcoded ``LEGACY_FUNCTIONALS`` score filter with an explicit ``score_functional_ids`` set, so
the sensitivity selectors are output-aware of the physical functionals actually being evaluated.
Every selector primitive (MILP, ridge leverage, output-aware scores, one-swap refinement,
exact-loss greedy, near-oracle beam / multi-start) is imported and reused unchanged; a
predeclared per-cell near-oracle compute ceiling is added for the larger block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    deterministic_ridge_leverage_scores,
    refine_support_one_swap,
    select_resource_constrained_support,
)
from robust_qsvt_se.qsvt.ridge_output_certificate import ridge_selected_output_gradient
from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import (
    ExactLossEvaluator,
    balanced_magnitude_score,
    exact_loss_greedy,
    near_oracle_beam,
    near_oracle_multistart,
)


@dataclass(slots=True)
class SensitivityScores:
    sensitivity_mean: np.ndarray
    sensitivity_worst_case: np.ndarray


def output_aware_sensitivity_scores(
    matrix: np.ndarray, tasks, *, alpha: float, epsilon: float = 1e-15
) -> SensitivityScores:
    """Aggregate output-aware entry scores over an arbitrary physical-functional task set.

    Reproduces the exact ``sensitivity_mean`` / ``sensitivity_worst_case`` aggregation of the
    protected ``compute_output_aware_entry_scores`` (same core gradient, same normalization) but
    omits its per-functional breakdown, which is hardcoded to the 3 legacy ``FUNCTIONAL_IDS`` and
    raises on physical functionals.  For legacy tasks the aggregates are byte-identical (asserted
    by a regression test), so selector semantics are preserved; here they become output-aware of
    the physical functionals actually evaluated.
    """

    if not tasks or any(task.split != "training" for task in tasks):
        raise ValueError("data leakage guard: entry scores require training tasks only")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("score normalization epsilon must be positive and finite")
    values = np.asarray(matrix, dtype=np.float64)
    normalized_rows: list[np.ndarray] = []
    for task in tasks:
        gradient = ridge_selected_output_gradient(values, task.residual, task.functional, alpha)
        removal = np.abs(values * gradient)
        normalized_rows.append(removal / (float(np.sum(removal)) + float(epsilon)))
    normalized = np.stack(normalized_rows, axis=0)
    return SensitivityScores(
        sensitivity_mean=np.mean(normalized, axis=0),
        sensitivity_worst_case=np.max(normalized, axis=0),
    )


def _score_tasks(training_tasks, score_functional_ids: tuple[str, ...]):
    wanted = set(score_functional_ids)
    return [task for task in training_tasks if task.functional_id in wanted]


def _milp_support(matrix: np.ndarray, score: np.ndarray, constraints: SupportConstraints):
    result = select_resource_constrained_support(matrix, score, constraints)
    return {
        "support": result.support,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "solver_used": result.solver_used,
        "optimality_gap": result.optimality_gap,
        "runtime_seconds": result.runtime_seconds,
        "milp_solves": 1,
    }


def estimate_near_oracle_loss_evals(
    matrix: np.ndarray, constraints: SupportConstraints, *, beam_width: int, max_steps: int
) -> int:
    """Deterministic upper-bound estimate of near-oracle beam loss evaluations (for the ceiling)."""

    nnz = int(np.count_nonzero(matrix))
    return int(beam_width * max_steps * nnz)


def produce_supports_generalized(
    design,
    training_tasks,
    constraints: SupportConstraints,
    *,
    evaluator: ExactLossEvaluator,
    score_functional_ids: tuple[str, ...],
    random_seed: int,
    beam_width: int,
    beam_max_steps: int,
    oracle_max_candidates: int,
    include_oracle: bool,
    near_oracle_max_loss_evals: int,
    selectors: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {selector: support-record} for one ``(k, s)`` budget on training data.

    Identical selector semantics to the frozen protocol, with the output-aware score set
    parameterized and a predeclared near-oracle compute ceiling.  ``selectors`` optionally
    restricts to a subset (e.g. Track B mean-objective required subset); ``None`` runs the full
    frozen suite.
    """

    matrix = design.matrix
    outcomes: dict[str, dict[str, Any]] = {}
    score_tasks = _score_tasks(training_tasks, score_functional_ids)

    magnitude = np.abs(matrix)
    _row_lev, _col_lev, leverage = deterministic_ridge_leverage_scores(matrix, alpha=design.alpha)
    scores = output_aware_sensitivity_scores(matrix, score_tasks, alpha=design.alpha)
    rng = np.random.default_rng(random_seed)
    random_scores = np.zeros_like(matrix)
    random_scores[matrix != 0.0] = rng.random(int(np.count_nonzero(matrix)))

    milp_bank = {
        "global_magnitude": magnitude,
        "balanced_magnitude": balanced_magnitude_score(matrix),
        "ridge_leverage": leverage,
        "random_feasible": random_scores,
        "sensitivity_initial_mean": scores.sensitivity_mean,
        "sensitivity_initial_worst_case": scores.sensitivity_worst_case,
    }
    wanted = None if selectors is None else set(selectors)

    def _want(name: str) -> bool:
        return wanted is None or name in wanted

    for name, score in milp_bank.items():
        if _want(name) or (
            # refined selectors depend on their initial support
            name == "sensitivity_initial_mean" and _want("sensitivity_refined_mean")
        ) or (
            name == "sensitivity_initial_worst_case" and _want("sensitivity_refined_worst_case")
        ):
            outcomes[name] = _milp_support(matrix, score, constraints)

    for initial_name, refined_name, objective in (
        ("sensitivity_initial_mean", "sensitivity_refined_mean", "mean_normalized_error"),
        (
            "sensitivity_initial_worst_case",
            "sensitivity_refined_worst_case",
            "worst_case_normalized_error",
        ),
    ):
        if not _want(refined_name):
            continue
        initial = outcomes.get(initial_name)
        if initial is None or initial["support"] is None:
            outcomes[refined_name] = {
                "support": None,
                "status": "failed",
                "failure_reason": "initial_support_unavailable",
                "runtime_seconds": 0.0,
                "milp_solves": 0,
            }
            continue
        started = time.perf_counter()
        refined = refine_support_one_swap(
            matrix,
            initial["support"],
            score_tasks,
            constraints,
            alpha=design.alpha,
            y_floor=evaluator.y_floor,
            objective=objective,
            max_iterations=8,
            improvement_tolerance=1e-12,
        )
        outcomes[refined_name] = {
            "support": refined.support,
            "status": "completed",
            "failure_reason": "",
            "runtime_seconds": time.perf_counter() - started,
            "refinement_iterations_accepted": refined.iterations_accepted,
            "milp_solves": 0,
        }

    for objective, name in (
        ("mean", "exact_loss_greedy_mean"),
        ("worst", "exact_loss_greedy_worst"),
    ):
        if _want(name):
            outcomes[name] = exact_loss_greedy(evaluator, constraints, objective=objective)

    if include_oracle:
        from robust_qsvt_se.reviewer_blocking.exact_loss_baselines import exhaustive_oracle

        for objective, name in (("mean", "oracle_mean"), ("worst", "oracle_worst")):
            if _want(name):
                outcomes[name] = exhaustive_oracle(
                    evaluator, constraints, objective=objective,
                    max_candidates=oracle_max_candidates,
                )

    # Near-oracle: multi-start one-swap over all feasible constructions + coverage-first beam,
    # subject to a predeclared per-cell compute ceiling on estimated beam loss evaluations.
    construction_seeds = [
        outcome["support"]
        for outcome in outcomes.values()
        if outcome.get("support") is not None and outcome.get("status") == "completed"
    ]
    estimated_evals = estimate_near_oracle_loss_evals(
        matrix, constraints, beam_width=beam_width, max_steps=beam_max_steps
    )
    for objective, name in (("mean", "near_oracle_mean"), ("worst", "near_oracle_worst")):
        if not _want(name):
            continue
        if estimated_evals > near_oracle_max_loss_evals:
            outcomes[name] = {
                "support": None,
                "status": "skipped_compute_ceiling",
                "failure_reason": (
                    f"estimated_beam_loss_evals {estimated_evals} exceeds ceiling "
                    f"{near_oracle_max_loss_evals}"
                ),
                "estimated_beam_loss_evals": estimated_evals,
                "runtime_seconds": 0.0,
                "milp_solves": 0,
                "algorithm": "near_oracle_multistart_local_search",
            }
            continue
        beam = near_oracle_beam(
            evaluator, constraints, objective=objective,
            beam_width=beam_width, max_steps=beam_max_steps,
        )
        seeds = list(construction_seeds)
        if beam.get("support") is not None:
            seeds.append(beam["support"])
        outcomes[name] = near_oracle_multistart(
            evaluator, training_tasks, constraints, objective=objective,
            seed_supports=seeds, beam_diagnostic=beam,
        )
    return outcomes


def select_support_generalized(
    design, selector: str, constraints: SupportConstraints, training_tasks,
    score_functional_ids: tuple[str, ...],
) -> np.ndarray | None:
    """MILP/refined single-selector support (joint-feasibility path), physical-functional scored."""

    matrix = design.matrix
    score_tasks = _score_tasks(training_tasks, score_functional_ids)
    if selector == "global_magnitude":
        return select_resource_constrained_support(matrix, np.abs(matrix), constraints).support
    if selector == "balanced_magnitude":
        return select_resource_constrained_support(
            matrix, balanced_magnitude_score(matrix), constraints
        ).support
    if selector == "ridge_leverage":
        _r, _c, leverage = deterministic_ridge_leverage_scores(matrix, alpha=design.alpha)
        return select_resource_constrained_support(matrix, leverage, constraints).support
    if selector in {"sensitivity_initial_mean", "sensitivity_refined_mean"}:
        scores = output_aware_sensitivity_scores(matrix, score_tasks, alpha=design.alpha)
        initial = select_resource_constrained_support(matrix, scores.sensitivity_mean, constraints)
        if selector == "sensitivity_initial_mean" or initial.support is None:
            return initial.support
        refined = refine_support_one_swap(
            matrix, initial.support, score_tasks, constraints,
            alpha=design.alpha, y_floor=1e-6, objective="mean_normalized_error",
            max_iterations=8, improvement_tolerance=1e-12,
        )
        return refined.support
    raise ValueError(f"unknown selector {selector}")
