"""Solve-based noise-propagation and posterior-variance support selectors.

The public API intentionally has no truth, held-out, or residual arguments.
Every objective is a deterministic function of the sparse matrix, alpha, and
unit functional bank.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    SupportConstraints,
    select_resource_constrained_support,
    support_constraint_report,
)

RiskKind = Literal["noise_propagation", "posterior_variance_reference"]
Aggregation = Literal["mean", "worst_case"]
NEGATIVE_TOLERANCE = 1e-11
UNIT_NORM_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class RiskEvaluation:
    objective: float
    per_functional: np.ndarray
    task_count: int
    unique_functional_count: int


@dataclass(frozen=True, slots=True)
class RemovalScoreResult:
    raw_scores: np.ndarray
    milp_scores: np.ndarray
    full_support_objective: float
    exact_solves: int
    task_count: int
    unique_functional_count: int
    translation: float


@dataclass(frozen=True, slots=True)
class RiskRefinementResult:
    support: np.ndarray
    initial_objective: float
    final_objective: float
    accepted_swaps: int
    termination_reason: str
    exact_solves: int
    runtime_seconds: float
    trace: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class RiskSupportResult:
    support: np.ndarray | None
    initial_support: np.ndarray | None
    status: str
    failure_reason: str
    risk_kind: RiskKind
    aggregation: Aggregation
    initial_objective: float | None
    final_objective: float | None
    full_support_objective: float | None
    accepted_swaps: int
    termination_reason: str
    exact_solves: int
    runtime_seconds: float
    solver_used: str
    solver_status: str
    fallback_used: bool
    raw_score_min: float | None
    raw_score_max: float | None
    negative_raw_score_count: int
    refinement_trace: tuple[dict[str, object], ...]


def _validated_matrix(matrix: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be a finite two-dimensional array")
    if values.shape[1] == 0 or values.shape[0] == 0:
        raise ValueError("matrix dimensions must be nonzero")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be positive and finite")
    return values


def _functional_matrix(functionals: Sequence[np.ndarray], dimension: int) -> np.ndarray:
    if not functionals:
        raise ValueError("at least one physical functional is required")
    rows = []
    for index, functional in enumerate(functionals):
        ell = np.asarray(functional, dtype=np.float64)
        if ell.shape != (dimension,) or not np.all(np.isfinite(ell)):
            raise ValueError(f"functional {index} has incompatible shape or nonfinite values")
        norm_error = abs(float(np.linalg.norm(ell)) - 1.0)
        if norm_error > UNIT_NORM_TOLERANCE:
            raise ValueError(f"functional {index} is not unit norm: error={norm_error}")
        rows.append(ell)
    return np.stack(rows, axis=0)


def _compress_functionals(functionals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique: list[np.ndarray] = []
    counts: list[int] = []
    positions: dict[str, int] = {}
    for ell in functionals:
        key = hashlib.sha256(np.ascontiguousarray(ell).tobytes()).hexdigest()
        if key in positions:
            counts[positions[key]] += 1
        else:
            positions[key] = len(unique)
            unique.append(ell)
            counts.append(1)
    return np.stack(unique, axis=0), np.asarray(counts, dtype=np.int64)


def _risk_values_solve(
    matrix: np.ndarray,
    alpha: float,
    unique_functionals: np.ndarray,
    risk_kind: RiskKind,
) -> np.ndarray:
    values = _validated_matrix(matrix, alpha)
    if risk_kind not in {"noise_propagation", "posterior_variance_reference"}:
        raise ValueError(f"unknown risk kind {risk_kind}")
    gram = values.T @ values
    regularized = gram + float(alpha) * np.eye(values.shape[1], dtype=np.float64)
    # One multi-right-hand-side solve evaluates all functionals.  Alpha > 0
    # preserves this path for rank-deficient sparse matrices.
    solved = np.linalg.solve(regularized, unique_functionals.T)
    if risk_kind == "noise_propagation":
        propagated = values @ solved
        risks = np.sum(propagated * propagated, axis=0)
    else:
        risks = np.sum(unique_functionals.T * solved, axis=0)
    risks = np.asarray(risks, dtype=np.float64)
    if not np.all(np.isfinite(risks)):
        raise np.linalg.LinAlgError("risk solve produced a nonfinite result")
    if float(np.min(risks, initial=0.0)) < -NEGATIVE_TOLERANCE:
        raise np.linalg.LinAlgError("risk solve produced a materially negative variance")
    return np.maximum(risks, 0.0)


def evaluate_risk(
    matrix: np.ndarray,
    alpha: float,
    functionals: Sequence[np.ndarray],
    *,
    risk_kind: RiskKind,
    aggregation: Aggregation,
) -> RiskEvaluation:
    values = _validated_matrix(matrix, alpha)
    full_bank = _functional_matrix(functionals, values.shape[1])
    unique, counts = _compress_functionals(full_bank)
    risks = _risk_values_solve(values, alpha, unique, risk_kind)
    if aggregation == "mean":
        objective = float(np.average(risks, weights=counts))
    elif aggregation == "worst_case":
        objective = float(np.max(risks))
    else:
        raise ValueError(f"unknown risk aggregation {aggregation}")
    expanded = np.repeat(risks, counts)
    return RiskEvaluation(
        objective=objective,
        per_functional=expanded,
        task_count=int(full_bank.shape[0]),
        unique_functional_count=int(unique.shape[0]),
    )


def noise_propagation_risk(matrix: np.ndarray, alpha: float, functional: np.ndarray) -> float:
    return evaluate_risk(
        matrix,
        alpha,
        [functional],
        risk_kind="noise_propagation",
        aggregation="mean",
    ).objective


def posterior_variance_reference(matrix: np.ndarray, alpha: float, functional: np.ndarray) -> float:
    return evaluate_risk(
        matrix,
        alpha,
        [functional],
        risk_kind="posterior_variance_reference",
        aggregation="mean",
    ).objective


def exact_single_removal_scores(
    matrix: np.ndarray,
    alpha: float,
    functionals: Sequence[np.ndarray],
    *,
    risk_kind: RiskKind,
    aggregation: Aggregation,
) -> RemovalScoreResult:
    values = _validated_matrix(matrix, alpha)
    candidate = values != 0.0
    full = evaluate_risk(values, alpha, functionals, risk_kind=risk_kind, aggregation=aggregation)
    raw = np.zeros_like(values)
    exact_solves = 1
    for row, column in np.argwhere(candidate):
        trial = values.copy()
        trial[row, column] = 0.0
        evaluated = evaluate_risk(
            trial, alpha, functionals, risk_kind=risk_kind, aggregation=aggregation
        )
        exact_solves += 1
        raw[row, column] = evaluated.objective - full.objective
    candidate_raw = raw[candidate]
    translation = -float(np.min(candidate_raw))
    milp = np.zeros_like(values)
    milp[candidate] = np.maximum(candidate_raw + translation, 0.0)
    return RemovalScoreResult(
        raw_scores=raw,
        milp_scores=milp,
        full_support_objective=full.objective,
        exact_solves=exact_solves,
        task_count=full.task_count,
        unique_functional_count=full.unique_functional_count,
        translation=translation,
    )


def refine_risk_support_one_swap(
    matrix: np.ndarray,
    initial_support: np.ndarray,
    alpha: float,
    functionals: Sequence[np.ndarray],
    constraints: SupportConstraints,
    *,
    risk_kind: RiskKind,
    aggregation: Aggregation,
    max_iterations: int,
    improvement_tolerance: float,
) -> RiskRefinementResult:
    started = time.perf_counter()
    values = _validated_matrix(matrix, alpha)
    support = np.asarray(initial_support, dtype=bool).copy()
    if not support_constraint_report(values, support, constraints)["valid"]:
        raise ValueError("initial support violates support constraints")
    if max_iterations < 0 or improvement_tolerance < 0.0:
        raise ValueError("refinement limits must be nonnegative")

    cache: dict[bytes, float] = {}

    def objective(candidate: np.ndarray) -> float:
        key = np.packbits(candidate, bitorder="little").tobytes()
        if key not in cache:
            sparse = np.where(candidate, values, 0.0)
            cache[key] = evaluate_risk(
                sparse,
                alpha,
                functionals,
                risk_kind=risk_kind,
                aggregation=aggregation,
            ).objective
        return cache[key]

    current = objective(support)
    initial = current
    trace: list[dict[str, object]] = [
        {
            "iteration": 0,
            "action": "initial",
            "objective_before": current,
            "objective_after": current,
            "accepted": True,
            "removed_flat_index": None,
            "added_flat_index": None,
        }
    ]
    accepted = 0
    termination = "maximum_iterations_reached"
    candidate_mask = values != 0.0
    for iteration in range(1, int(max_iterations) + 1):
        selected = np.flatnonzero(support.ravel())
        excluded = np.flatnonzero(candidate_mask.ravel() & ~support.ravel())
        best: tuple[float, int, int, np.ndarray] | None = None
        for removed in selected:
            for added in excluded:
                trial = support.copy()
                trial.ravel()[removed] = False
                trial.ravel()[added] = True
                if not support_constraint_report(values, trial, constraints)["valid"]:
                    continue
                trial_objective = objective(trial)
                key = (trial_objective, int(removed), int(added))
                if best is None or key < best[:3]:
                    best = (trial_objective, int(removed), int(added), trial)
        if best is None:
            termination = "no_feasible_swap"
            trace.append(
                {
                    "iteration": iteration,
                    "action": termination,
                    "objective_before": current,
                    "objective_after": current,
                    "accepted": False,
                    "removed_flat_index": None,
                    "added_flat_index": None,
                }
            )
            break
        improvement = current - best[0]
        if not improvement > float(improvement_tolerance):
            termination = "no_strict_improvement"
            trace.append(
                {
                    "iteration": iteration,
                    "action": termination,
                    "objective_before": current,
                    "objective_after": best[0],
                    "improvement": improvement,
                    "accepted": False,
                    "removed_flat_index": best[1],
                    "added_flat_index": best[2],
                }
            )
            break
        trace.append(
            {
                "iteration": iteration,
                "action": "one_swap",
                "objective_before": current,
                "objective_after": best[0],
                "improvement": improvement,
                "accepted": True,
                "removed_flat_index": best[1],
                "added_flat_index": best[2],
            }
        )
        current = best[0]
        support = best[3]
        accepted += 1
    return RiskRefinementResult(
        support=support,
        initial_objective=initial,
        final_objective=current,
        accepted_swaps=accepted,
        termination_reason=termination,
        exact_solves=len(cache),
        runtime_seconds=time.perf_counter() - started,
        trace=tuple(trace),
    )


def select_risk_support(
    matrix: np.ndarray,
    alpha: float,
    functionals: Sequence[np.ndarray],
    constraints: SupportConstraints,
    *,
    risk_kind: RiskKind,
    aggregation: Aggregation,
    refine: bool,
    max_refinement_iterations: int,
    improvement_tolerance: float,
    time_limit_seconds: float = 30.0,
    relative_mip_gap: float = 0.0,
    tie_epsilon_relative: float = 1e-12,
) -> RiskSupportResult:
    started = time.perf_counter()
    try:
        scores = exact_single_removal_scores(
            matrix,
            alpha,
            functionals,
            risk_kind=risk_kind,
            aggregation=aggregation,
        )
        selection = select_resource_constrained_support(
            matrix,
            scores.milp_scores,
            constraints,
            time_limit_seconds=time_limit_seconds,
            relative_mip_gap=relative_mip_gap,
            tie_epsilon_relative=tie_epsilon_relative,
        )
        if selection.support is None or selection.status != "completed":
            return RiskSupportResult(
                support=None,
                initial_support=None,
                status="failed",
                failure_reason=selection.failure_reason,
                risk_kind=risk_kind,
                aggregation=aggregation,
                initial_objective=None,
                final_objective=None,
                full_support_objective=scores.full_support_objective,
                accepted_swaps=0,
                termination_reason="initial_support_unavailable",
                exact_solves=scores.exact_solves,
                runtime_seconds=time.perf_counter() - started,
                solver_used=selection.solver_used,
                solver_status=selection.solver_status,
                fallback_used=selection.fallback_used,
                raw_score_min=float(np.min(scores.raw_scores[matrix != 0.0])),
                raw_score_max=float(np.max(scores.raw_scores[matrix != 0.0])),
                negative_raw_score_count=int(np.sum(scores.raw_scores[matrix != 0.0] < 0.0)),
                refinement_trace=(),
            )
        initial_support = selection.support.copy()
        initial_objective = evaluate_risk(
            np.where(initial_support, matrix, 0.0),
            alpha,
            functionals,
            risk_kind=risk_kind,
            aggregation=aggregation,
        ).objective
        exact_solves = scores.exact_solves + 1
        final_support = initial_support
        final_objective = initial_objective
        accepted = 0
        termination = "initial_only"
        trace: tuple[dict[str, object], ...] = ()
        if refine:
            refined = refine_risk_support_one_swap(
                matrix,
                initial_support,
                alpha,
                functionals,
                constraints,
                risk_kind=risk_kind,
                aggregation=aggregation,
                max_iterations=max_refinement_iterations,
                improvement_tolerance=improvement_tolerance,
            )
            final_support = refined.support
            final_objective = refined.final_objective
            accepted = refined.accepted_swaps
            termination = refined.termination_reason
            exact_solves += refined.exact_solves
            trace = refined.trace
        candidate_raw = scores.raw_scores[np.asarray(matrix) != 0.0]
        return RiskSupportResult(
            support=final_support,
            initial_support=initial_support,
            status="completed",
            failure_reason="",
            risk_kind=risk_kind,
            aggregation=aggregation,
            initial_objective=initial_objective,
            final_objective=final_objective,
            full_support_objective=scores.full_support_objective,
            accepted_swaps=accepted,
            termination_reason=termination,
            exact_solves=exact_solves,
            runtime_seconds=time.perf_counter() - started,
            solver_used=selection.solver_used,
            solver_status=selection.solver_status,
            fallback_used=selection.fallback_used,
            raw_score_min=float(np.min(candidate_raw)),
            raw_score_max=float(np.max(candidate_raw)),
            negative_raw_score_count=int(np.sum(candidate_raw < 0.0)),
            refinement_trace=trace,
        )
    except Exception as exc:
        return RiskSupportResult(
            support=None,
            initial_support=None,
            status="failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            risk_kind=risk_kind,
            aggregation=aggregation,
            initial_objective=None,
            final_objective=None,
            full_support_objective=None,
            accepted_swaps=0,
            termination_reason="exception",
            exact_solves=0,
            runtime_seconds=time.perf_counter() - started,
            solver_used="unavailable",
            solver_status="exception",
            fallback_used=False,
            raw_score_min=None,
            raw_score_max=None,
            negative_raw_score_count=0,
            refinement_trace=(),
        )
