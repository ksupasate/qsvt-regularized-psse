from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.qsvt.bounded_target_designs import (
    NormalizedProblem,
    _build_normalized_problem,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.weighted_singular_support import SingularSupport, compute_singular_support
from robust_qsvt_se.utils.io import ensure_directory

CODESIGNED_CLAIM = (
    "Co-designed QSVT-safe bounded-target family study on a selected IEEE14-derived "
    "subproblem. Each polynomial target is fitted with singular-support weighting "
    "and checked for QSVT boundedness (max|p| <= 1), Ridge-aligned direction, "
    "success amplitude, and residual under deployable amplitude/known-C scale "
    "recovery. Ridge/Tikhonov remains the reference filter; QSVT is an implementation "
    "pathway for the same regularized spectral filter, not a superior classical "
    "solver. No QSVT superiority over Ridge/Tikhonov, quantum speedup, quantum "
    "advantage, full IEEE-scale gate-level solving, or hardware execution is claimed."
)

QSVT_SAFE_TOLERANCE = 1.0e-6
SUCCESS_PROBABILITY_FLOOR = 1.0e-3
OVERSHOOT_FACTOR = 2.0
FEASIBLE_RATIO_MAX = 0.1
FEASIBLE_DIRECTION_MAX = 0.1

DEFAULT_DEGREES = (15, 25, 35, 51, 75, 101)
DEFAULT_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_TARGET_FAMILIES = (
    "weighted_support_ls",
    "direction_preserving",
    "residual_aware",
    "monotone_clipped",
    "success_amplitude_aware",
    "degree_scheduled",
)

DEPLOYABILITY_CLASSES = {
    "general_qsvt_safe",
    "instance_aware_qsvt_safe",
    "diagnostic_only",
    "not_qsvt_safe",
    "runtime_limited",
}

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "C_scale",
    "max_abs_polynomial_on_grid",
    "qsvt_safe",
    "success_probability_proxy",
    "weighted_filter_error",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "residual_known_C",
    "residual_amplitude_recovered_proxy",
    "residual_best_scalar_diagnostic",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "overshoot_detected",
    "instance_aware",
    "deployability_class",
    "dominant_limitation",
]

COEFFICIENT_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_family",
    "coefficient_index",
    "bounded_power_coefficient",
    "raw_power_coefficient",
]

BOUNDEDNESS_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_family",
    "C_scale",
    "max_abs_polynomial_on_grid",
    "qsvt_safe",
    "overshoot_detected",
    "success_probability_proxy",
    "deployability_class",
]

FIT_ERROR_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "weighted_filter_error",
    "residual_known_C",
    "residual_ratio_vs_no_update",
    "deployability_class",
]

DIRECTION_TRADEOFF_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_family",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "residual_amplitude_recovered_proxy",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge",
    "deployability_class",
]

SUCCESS_TRADEOFF_COLUMNS = [
    "case",
    "alpha",
    "degree",
    "target_family",
    "success_probability_proxy",
    "max_abs_polynomial_on_grid",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "deployability_class",
]


@dataclass(frozen=True, slots=True)
class CodesignedTarget:
    """A raw odd polynomial target plus its provenance before bounding."""

    target_family: str
    weighting_scheme: str
    raw_polynomial: Polynomial
    instance_aware: bool
    base_deployability: str


@dataclass(frozen=True, slots=True)
class CodesignedSolution:
    """A fully bounded co-designed target with its physical known-C update."""

    target_family: str
    weighting_scheme: str
    bounded_coefficients: np.ndarray
    raw_polynomial: Polynomial
    c_scale: float
    beta: float
    known_c_update: np.ndarray
    success_probability_proxy: float
    direction_error_vs_ridge: float
    deployability_class: str
    instance_aware: bool
    qsvt_safe: bool


def build_codesigned_solution(
    subproblem: SelectedSubproblem,
    *,
    alpha: float,
    degree: int,
    target_family: str,
    grid_size: int = 4096,
) -> CodesignedSolution | None:
    """Rebuild a co-designed bounded target and its physical known-C update.

    Reused by gate-level validation (which synthesizes phases from
    ``bounded_coefficients``) and the observable-first solver (which reads out
    functionals of ``known_c_update``).
    """

    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    try:
        problem = _build_normalized_problem(subproblem, alpha=float(alpha))
        support = compute_singular_support(H, r, alpha=float(alpha))
        target = _build_codesigned_target(
            family=str(target_family),
            problem=problem,
            support=support,
            degree=int(degree),
            alpha=float(alpha),
            grid_size=int(grid_size),
        )
    except Exception:
        return None
    if target is None:
        return None

    p_raw = target.raw_polynomial
    qsvt_grid = np.linspace(-1.0, 1.0, max(8193, 256 * int(degree) + 1), dtype=np.float64)
    raw_max_abs = float(np.max(np.abs(p_raw(qsvt_grid))))
    c_scale = max(1.0, raw_max_abs) * (1.0 + QSVT_SAFE_TOLERANCE)
    qsvt_safe = (raw_max_abs / c_scale) <= 1.0 + QSVT_SAFE_TOLERANCE
    bounded_coefficients = np.asarray(p_raw.coef, dtype=np.float64) / c_scale

    bounded_filter = p_raw(problem.singular_values_A) / c_scale
    bounded_output = problem.U @ (bounded_filter[:, None] * problem.Vh) @ problem.r
    known_c_update = rescale_bounded_target_to_original(
        bounded_output, beta=problem.beta, C=c_scale
    )
    r_norm = float(np.linalg.norm(problem.r))
    bounded_norm = float(np.linalg.norm(bounded_output))
    success_probability = float(min(1.0, (bounded_norm / r_norm) ** 2) if r_norm > 0.0 else 0.0)
    cos_angle = _cos_angle(known_c_update, problem.ridge_update)
    direction_error = _direction_error_from_cos(cos_angle)
    deployability_class = _deployability_class(target.base_deployability, qsvt_safe)
    return CodesignedSolution(
        target_family=target.target_family,
        weighting_scheme=target.weighting_scheme,
        bounded_coefficients=bounded_coefficients,
        raw_polynomial=p_raw,
        c_scale=float(c_scale),
        beta=float(problem.beta),
        known_c_update=known_c_update,
        success_probability_proxy=success_probability,
        direction_error_vs_ridge=float(direction_error),
        deployability_class=deployability_class,
        instance_aware=bool(target.instance_aware),
        qsvt_safe=bool(qsvt_safe),
    )


def run_qsvt_codesigned_bounded_target_study(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": list(DEFAULT_ALPHAS),
        "degrees": list(DEFAULT_DEGREES),
        "target_families": list(DEFAULT_TARGET_FAMILIES),
        "grid_size": 4096,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_codesigned_bounded_target_study",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    summary_rows, coefficient_rows = evaluate_codesigned_target_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        target_families=[str(value) for value in resolved["target_families"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        grid_size=int(resolved["grid_size"]),
    )
    artifacts = write_codesigned_target_outputs(
        output_dir, resolved, summary_rows, coefficient_rows
    )
    return {
        "output_dir": output_dir,
        "rows": summary_rows,
        "coefficient_rows": coefficient_rows,
        "artifacts": artifacts,
    }


def evaluate_codesigned_target_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    target_families: list[str],
    case: str = "ieee14",
    model: str = "ac_linearized",
    subproblem_id: str = "subproblem",
    grid_size: int = 4096,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        try:
            problem = _build_normalized_problem(subproblem, alpha=float(alpha))
            support = compute_singular_support(H, r, alpha=float(alpha))
        except Exception as exc:
            for degree in degrees:
                for family in target_families:
                    summary_rows.append(
                        _failed_row(
                            case=case,
                            model=model,
                            subproblem_id=subproblem_id,
                            alpha=float(alpha),
                            degree=int(degree),
                            target_family=family,
                            reason=f"normalization_failed:{type(exc).__name__}",
                        )
                    )
            continue
        for degree in degrees:
            for family in target_families:
                base = {
                    "case": case,
                    "model": model,
                    "subproblem_id": subproblem_id,
                    "alpha": float(alpha),
                    "degree": int(degree),
                }
                target = _build_codesigned_target(
                    family=family,
                    problem=problem,
                    support=support,
                    degree=int(degree),
                    alpha=float(alpha),
                    grid_size=int(grid_size),
                )
                if target is None:
                    summary_rows.append(
                        _failed_row(
                            **base, target_family=family, reason="target_construction_failed"
                        )
                    )
                    continue
                row, coeff_rows = _evaluate_codesigned_target(
                    problem=problem,
                    support=support,
                    target=target,
                    base=base,
                    degree=int(degree),
                    grid_size=int(grid_size),
                )
                summary_rows.append(row)
                coefficient_rows.extend(coeff_rows)
    return summary_rows, coefficient_rows


def _build_codesigned_target(
    *,
    family: str,
    problem: NormalizedProblem,
    support: SingularSupport,
    degree: int,
    alpha: float,
    grid_size: int,
) -> CodesignedTarget | None:
    try:
        if family == "weighted_support_ls":
            return _weighted_support_target(problem, support, degree, grid_size)
        if family == "direction_preserving":
            return _direction_preserving_target(problem, support, degree, grid_size)
        if family == "residual_aware":
            return _residual_aware_target(problem, support, degree, grid_size)
        if family == "monotone_clipped":
            return _monotone_clipped_target(problem, support, degree, grid_size, clip_factor=0.5)
        if family == "success_amplitude_aware":
            return _success_amplitude_target(problem, support, degree, grid_size)
        if family == "degree_scheduled":
            return _degree_scheduled_target(problem, support, degree, grid_size)
    except Exception:
        return None
    return None


def _fit_grid(problem: NormalizedProblem, degree: int, grid_size: int) -> np.ndarray:
    domain_min = problem.domain_min
    return np.linspace(domain_min, 1.0, max(int(grid_size), int(degree) + 2), dtype=np.float64)


def _support_values(problem: NormalizedProblem) -> np.ndarray:
    sv = problem.singular_values_A
    return sv[sv > 1.0e-14]


def _odd_polynomial_from_q(q_cheb: Chebyshev, degree: int) -> Polynomial:
    q_poly = q_cheb.convert(kind=Polynomial)
    coefficients = np.zeros(int(degree) + 1, dtype=np.float64)
    for index, coefficient in enumerate(q_poly.coef):
        target_index = 2 * index + 1
        if target_index <= int(degree):
            coefficients[target_index] = coefficient
    return Polynomial(coefficients)


def _weighted_q_fit(
    *,
    y_points: np.ndarray,
    q_target: np.ndarray,
    weights: np.ndarray,
    reduced_degree: int,
    y_min: float,
    y_max: float,
) -> Chebyshev:
    sqrt_weights = np.sqrt(np.maximum(weights, 0.0))
    return Chebyshev.fit(
        y_points,
        q_target,
        deg=int(reduced_degree),
        domain=[float(y_min), float(y_max)],
        w=sqrt_weights,
    )


def _weighted_support_target(
    problem: NormalizedProblem, support: SingularSupport, degree: int, grid_size: int
) -> CodesignedTarget:
    grid = _fit_grid(problem, degree, grid_size)
    sv = _support_values(problem)
    alpha_norm = problem.alpha_norm
    reduced_degree = (int(degree) - 1) // 2

    dense_y = grid**2
    support_y = sv**2
    base_dense = 1.0 / (dense_y + alpha_norm)
    base_support = 1.0 / (support_y + alpha_norm)

    combined = support.combined_weight[: sv.size]
    emphasis = float(max(dense_y.size, 1)) * 25.0
    y_points = np.concatenate([dense_y, support_y])
    q_target = np.concatenate([base_dense, base_support])
    weights = np.concatenate([np.full(dense_y.size, 1.0), combined * emphasis + 1.0e-9])
    q_cheb = _weighted_q_fit(
        y_points=y_points,
        q_target=q_target,
        weights=weights,
        reduced_degree=reduced_degree,
        y_min=float(grid[0] ** 2),
        y_max=1.0,
    )
    return CodesignedTarget(
        target_family="weighted_support_ls",
        weighting_scheme="combined_singular_support",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=True,
        base_deployability="instance_aware_qsvt_safe",
    )


def _direction_preserving_target(
    problem: NormalizedProblem, support: SingularSupport, degree: int, grid_size: int
) -> CodesignedTarget:
    grid = _fit_grid(problem, degree, grid_size)
    sv = _support_values(problem)
    alpha_norm = problem.alpha_norm
    reduced_degree = (int(degree) - 1) // 2

    g = problem.Vh @ problem.r
    projection = problem.U.T @ problem.ridge_update
    g = g[: sv.size]
    projection = projection[: sv.size]
    g_floor = 1.0e-12 * max(float(np.max(np.abs(g))), 1.0)
    base_support = 1.0 / (sv**2 + alpha_norm)
    q_support = np.empty(sv.size, dtype=np.float64)
    for index in range(sv.size):
        if abs(g[index]) <= g_floor:
            q_support[index] = base_support[index]
        else:
            f_value = projection[index] / g[index]
            q_support[index] = f_value / sv[index]

    dense_y = grid**2
    base_dense = 1.0 / (dense_y + alpha_norm)
    combined = support.combined_weight[: sv.size]
    emphasis = float(max(dense_y.size, 1)) * 25.0
    y_points = np.concatenate([dense_y, sv**2])
    q_target = np.concatenate([base_dense, q_support])
    weights = np.concatenate([np.full(dense_y.size, 1.0), combined * emphasis + 1.0e-9])
    q_cheb = _weighted_q_fit(
        y_points=y_points,
        q_target=q_target,
        weights=weights,
        reduced_degree=reduced_degree,
        y_min=float(grid[0] ** 2),
        y_max=1.0,
    )
    return CodesignedTarget(
        target_family="direction_preserving",
        weighting_scheme="ridge_direction",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=True,
        base_deployability="diagnostic_only",
    )


def _residual_aware_target(
    problem: NormalizedProblem, support: SingularSupport, degree: int, grid_size: int
) -> CodesignedTarget:
    grid = _fit_grid(problem, degree, grid_size)
    sv = _support_values(problem)
    alpha_norm = problem.alpha_norm
    reduced_degree = (int(degree) - 1) // 2

    dense_y = grid**2
    base_dense = 1.0 / (dense_y + alpha_norm)
    support_y = sv**2
    q_support = 1.0 / support_y  # p(sv) = 1/sv minimizes residual (pseudo-inverse)
    emphasis = float(max(dense_y.size, 1)) * 25.0
    y_points = np.concatenate([dense_y, support_y])
    q_target = np.concatenate([base_dense, q_support])
    weights = np.concatenate([np.full(dense_y.size, 1.0), np.full(sv.size, emphasis)])
    q_cheb = _weighted_q_fit(
        y_points=y_points,
        q_target=q_target,
        weights=weights,
        reduced_degree=reduced_degree,
        y_min=float(grid[0] ** 2),
        y_max=1.0,
    )
    return CodesignedTarget(
        target_family="residual_aware",
        weighting_scheme="residual_least_squares",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=False,
        base_deployability="general_qsvt_safe",
    )


def _clipped_q_fit(
    problem: NormalizedProblem,
    *,
    degree: int,
    grid_size: int,
    gain_cap: float,
    support_weights: np.ndarray | None = None,
) -> Chebyshev:
    grid = _fit_grid(problem, degree, grid_size)
    sv = _support_values(problem)
    alpha_norm = problem.alpha_norm
    reduced_degree = (int(degree) - 1) // 2
    dense_y = grid**2
    dense_x = grid
    raw_dense = dense_x / (dense_y + alpha_norm)
    capped_dense = np.minimum(raw_dense, float(gain_cap))
    q_dense = capped_dense / dense_x

    y_points = dense_y
    q_target = q_dense
    weights = np.full(dense_y.size, 1.0)
    if support_weights is not None:
        support_y = sv**2
        raw_support = sv / (support_y + alpha_norm)
        capped_support = np.minimum(raw_support, float(gain_cap))
        q_support = capped_support / sv
        emphasis = float(max(dense_y.size, 1)) * 25.0
        y_points = np.concatenate([dense_y, support_y])
        q_target = np.concatenate([q_dense, q_support])
        weights = np.concatenate(
            [np.full(dense_y.size, 1.0), support_weights[: sv.size] * emphasis + 1.0e-9]
        )
    return _weighted_q_fit(
        y_points=y_points,
        q_target=q_target,
        weights=weights,
        reduced_degree=reduced_degree,
        y_min=float(grid[0] ** 2),
        y_max=1.0,
    )


def _support_gain_cap(problem: NormalizedProblem, factor: float) -> float:
    sv = _support_values(problem)
    c_support = float(np.max(sv / (sv**2 + problem.alpha_norm)))
    return float(factor) * c_support


def _monotone_clipped_target(
    problem: NormalizedProblem,
    support: SingularSupport,
    degree: int,
    grid_size: int,
    *,
    clip_factor: float,
) -> CodesignedTarget:
    gain_cap = _support_gain_cap(problem, clip_factor)
    q_cheb = _clipped_q_fit(problem, degree=degree, grid_size=grid_size, gain_cap=gain_cap)
    return CodesignedTarget(
        target_family="monotone_clipped",
        weighting_scheme="gain_capped",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=False,
        base_deployability="general_qsvt_safe",
    )


def _success_amplitude_target(
    problem: NormalizedProblem, support: SingularSupport, degree: int, grid_size: int
) -> CodesignedTarget:
    # Cap small-singular-value gain to lift the postselected success amplitude while
    # keeping the high-weight directions accurately fitted.
    gain_cap = _support_gain_cap(problem, 0.35)
    q_cheb = _clipped_q_fit(
        problem,
        degree=degree,
        grid_size=grid_size,
        gain_cap=gain_cap,
        support_weights=support.combined_weight,
    )
    return CodesignedTarget(
        target_family="success_amplitude_aware",
        weighting_scheme="combined_singular_support_capped",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=True,
        base_deployability="instance_aware_qsvt_safe",
    )


def _degree_scheduled_target(
    problem: NormalizedProblem, support: SingularSupport, degree: int, grid_size: int
) -> CodesignedTarget:
    # Higher degree affords a steeper (less clipped) target without off-support overshoot.
    schedule = min(1.0, 0.25 + 0.0075 * float(degree))
    gain_cap = _support_gain_cap(problem, schedule)
    q_cheb = _clipped_q_fit(problem, degree=degree, grid_size=grid_size, gain_cap=gain_cap)
    return CodesignedTarget(
        target_family="degree_scheduled",
        weighting_scheme="scheduled_gain_cap",
        raw_polynomial=_odd_polynomial_from_q(q_cheb, degree),
        instance_aware=False,
        base_deployability="general_qsvt_safe",
    )


def _evaluate_codesigned_target(
    *,
    problem: NormalizedProblem,
    support: SingularSupport,
    target: CodesignedTarget,
    base: dict[str, Any],
    degree: int,
    grid_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p_raw = target.raw_polynomial
    sv = _support_values(problem)
    beta = problem.beta

    qsvt_grid = np.linspace(-1.0, 1.0, max(8193, 256 * int(degree) + 1), dtype=np.float64)
    raw_max_abs = float(np.max(np.abs(p_raw(qsvt_grid))))
    support_values = p_raw(sv)
    support_max_abs = float(np.max(np.abs(support_values))) if support_values.size else 0.0
    c_scale = max(1.0, raw_max_abs) * (1.0 + QSVT_SAFE_TOLERANCE)
    max_abs_on_grid = raw_max_abs / c_scale
    qsvt_safe = max_abs_on_grid <= 1.0 + QSVT_SAFE_TOLERANCE
    overshoot_detected = raw_max_abs > OVERSHOOT_FACTOR * max(support_max_abs, np.finfo(float).eps)

    bounded_filter = p_raw(problem.singular_values_A) / c_scale
    bounded_operator = problem.U @ (bounded_filter[:, None] * problem.Vh)
    bounded_output = bounded_operator @ problem.r
    bounded_norm = float(np.linalg.norm(bounded_output))
    r_norm = float(np.linalg.norm(problem.r))
    # Physically-faithful postselection probability: the gate circuit prepares the
    # NORMALIZED residual state |r/||r||>, so the success amplitude is the contracted
    # output norm relative to ||r||. This matches the dense gate circuit (which never
    # saturates) and resolves the earlier matrix-level min(1, ||output||^2) artifact.
    success_probability = float(min(1.0, (bounded_norm / r_norm) ** 2) if r_norm > 0.0 else 0.0)
    unit = (
        bounded_output / bounded_norm if bounded_norm > 1.0e-30 else np.zeros_like(bounded_output)
    )

    physical_known_c = rescale_bounded_target_to_original(bounded_output, beta=beta, C=c_scale)
    residual_known_c = _residual(problem.H, physical_known_c, problem.r)

    amplitude_magnitude = (c_scale / beta) * r_norm * math.sqrt(max(success_probability, 0.0))
    amplitude_update = amplitude_magnitude * unit
    residual_amplitude = _residual(problem.H, amplitude_update, problem.r)

    best_scalar = best_scalar_for_residual(problem.H, unit, problem.r)
    residual_best_scalar = _residual(problem.H, best_scalar * unit, problem.r)

    no_update = float(np.linalg.norm(problem.r))
    ridge_residual = problem.ridge_residual
    residual_ratio_no_update = residual_amplitude / no_update if no_update > 0.0 else float("nan")
    residual_ratio_ridge = (
        residual_amplitude / ridge_residual if ridge_residual > 0.0 else float("nan")
    )

    cos_angle = _cos_angle(physical_known_c, problem.ridge_update)
    direction_error = _direction_error_from_cos(cos_angle)

    physical_filter = support_values / beta
    weighted_filter_error = _weighted_filter_error(
        physical_filter, support.ridge_filter_value[: sv.size], support.combined_weight[: sv.size]
    )

    deployability_class = _deployability_class(target.base_deployability, qsvt_safe)
    dominant_limitation = _dominant_limitation(
        qsvt_safe=qsvt_safe,
        success_probability=success_probability,
        direction_error=direction_error,
        residual_ratio_no_update=residual_ratio_no_update,
        residual_best_scalar=residual_best_scalar,
        no_update=no_update,
        overshoot_detected=overshoot_detected,
    )

    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(base)
    row.update(
        {
            "target_family": target.target_family,
            "weighting_scheme": target.weighting_scheme,
            "C_scale": float(c_scale),
            "max_abs_polynomial_on_grid": float(max_abs_on_grid),
            "qsvt_safe": bool(qsvt_safe),
            "success_probability_proxy": float(success_probability),
            "weighted_filter_error": float(weighted_filter_error),
            "direction_error_vs_ridge": float(direction_error),
            "cos_angle_vs_ridge": float(cos_angle),
            "residual_known_C": float(residual_known_c),
            "residual_amplitude_recovered_proxy": float(residual_amplitude),
            "residual_best_scalar_diagnostic": float(residual_best_scalar),
            "residual_ratio_vs_no_update": float(residual_ratio_no_update),
            "residual_ratio_vs_ridge": float(residual_ratio_ridge),
            "overshoot_detected": bool(overshoot_detected),
            "instance_aware": bool(target.instance_aware),
            "deployability_class": deployability_class,
            "dominant_limitation": dominant_limitation,
        }
    )

    bounded_coefficients = np.asarray(p_raw.coef, dtype=np.float64) / c_scale
    raw_coefficients = np.asarray(p_raw.coef, dtype=np.float64)
    coeff_rows = [
        {
            "case": base["case"],
            "alpha": base["alpha"],
            "degree": base["degree"],
            "target_family": target.target_family,
            "coefficient_index": int(index),
            "bounded_power_coefficient": float(bounded_coefficients[index]),
            "raw_power_coefficient": float(raw_coefficients[index]),
        }
        for index in range(raw_coefficients.size)
        if abs(float(raw_coefficients[index])) > 0.0
    ]
    return row, coeff_rows


def _deployability_class(base_deployability: str, qsvt_safe: bool) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    return base_deployability


def _dominant_limitation(
    *,
    qsvt_safe: bool,
    success_probability: float,
    direction_error: float,
    residual_ratio_no_update: float,
    residual_best_scalar: float,
    no_update: float,
    overshoot_detected: bool,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    if success_probability < SUCCESS_PROBABILITY_FLOOR:
        return "low_success_amplitude"
    if not math.isfinite(direction_error) or direction_error > FEASIBLE_DIRECTION_MAX:
        return "direction_error"
    if math.isfinite(residual_ratio_no_update) and residual_ratio_no_update <= FEASIBLE_RATIO_MAX:
        return "none"
    best_scalar_ratio = residual_best_scalar / no_update if no_update > 0.0 else float("inf")
    if best_scalar_ratio <= FEASIBLE_RATIO_MAX:
        return "magnitude_scale_recovery"
    if overshoot_detected:
        return "polynomial_overshoot"
    return "polynomial_target_or_conditioning"


def write_codesigned_target_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(summary_rows, SUMMARY_COLUMNS)
    coefficient_frame = _frame_with_columns(coefficient_rows, COEFFICIENT_COLUMNS)
    boundedness_frame = summary_frame[BOUNDEDNESS_COLUMNS].copy()
    fit_error_frame = summary_frame[FIT_ERROR_COLUMNS].copy()
    direction_frame = summary_frame[DIRECTION_TRADEOFF_COLUMNS].copy()
    success_frame = summary_frame[SUCCESS_TRADEOFF_COLUMNS].copy()

    summary_path = output_dir / "codesigned_target_summary.csv"
    coefficient_path = output_dir / "polynomial_coefficients.csv"
    boundedness_path = output_dir / "boundedness_validation.csv"
    fit_error_path = output_dir / "weighted_support_fit_error.csv"
    direction_path = output_dir / "direction_residual_tradeoff.csv"
    success_path = output_dir / "success_amplitude_tradeoff.csv"
    interpretation_path = output_dir / "codesigned_target_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    coefficient_frame.to_csv(coefficient_path, index=False)
    boundedness_frame.to_csv(boundedness_path, index=False)
    fit_error_frame.to_csv(fit_error_path, index=False)
    direction_frame.to_csv(direction_path, index=False)
    success_frame.to_csv(success_path, index=False)
    interpretation_path.write_text(
        codesigned_target_interpretation(summary_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "codesigned_target_summary": str(summary_path),
            "polynomial_coefficients": str(coefficient_path),
            "boundedness_validation": str(boundedness_path),
            "weighted_support_fit_error": str(fit_error_path),
            "direction_residual_tradeoff": str(direction_path),
            "success_amplitude_tradeoff": str(success_path),
            "codesigned_target_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CODESIGNED_CLAIM,
    )
    return {
        "manifest": manifest,
        "codesigned_target_summary": summary_path,
        "polynomial_coefficients": coefficient_path,
        "boundedness_validation": boundedness_path,
        "weighted_support_fit_error": fit_error_path,
        "direction_residual_tradeoff": direction_path,
        "success_amplitude_tradeoff": success_path,
        "codesigned_target_interpretation": interpretation_path,
    }


def codesigned_target_interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n".join(["# Co-Designed Bounded-Target Study", "", CODESIGNED_CLAIM, ""])

    numeric = frame.copy()
    for column in (
        "residual_ratio_vs_no_update",
        "direction_error_vs_ridge",
        "success_probability_proxy",
        "weighted_filter_error",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    deployable = numeric[
        numeric["deployability_class"].isin({"general_qsvt_safe", "instance_aware_qsvt_safe"})
        & (numeric["qsvt_safe"] == True)  # noqa: E712
    ]
    diagnostic = numeric[numeric["deployability_class"] == "diagnostic_only"]

    best_deployable_ratio = (
        float(deployable["residual_ratio_vs_no_update"].min())
        if not deployable.empty
        else float("nan")
    )
    best_diagnostic_ratio = (
        float(diagnostic["residual_ratio_vs_no_update"].min())
        if not diagnostic.empty
        else float("nan")
    )
    best_direction = float(numeric["direction_error_vs_ridge"].min())

    # Does weighted support fitting improve direction error vs the unweighted clipped families?
    weighted = numeric[numeric["target_family"] == "weighted_support_ls"]
    clipped = numeric[numeric["target_family"] == "monotone_clipped"]
    weighted_dir = (
        float(weighted["direction_error_vs_ridge"].min()) if not weighted.empty else float("nan")
    )
    clipped_dir = (
        float(clipped["direction_error_vs_ridge"].min()) if not clipped.empty else float("nan")
    )
    weighting_helps = (
        "yes"
        if math.isfinite(weighted_dir)
        and math.isfinite(clipped_dir)
        and weighted_dir <= clipped_dir
        else "no"
    )

    any_safe_reduces = (
        "yes"
        if math.isfinite(best_deployable_ratio) and best_deployable_ratio <= FEASIBLE_RATIO_MAX
        else "no"
    )

    instance_best = (
        float(
            numeric[numeric["instance_aware"] == True]["residual_ratio_vs_no_update"].min()  # noqa: E712
        )
        if not numeric[numeric["instance_aware"] == True].empty  # noqa: E712
        else float("nan")
    )
    general_best = (
        float(
            numeric[numeric["instance_aware"] != True]["residual_ratio_vs_no_update"].min()  # noqa: E712
        )
        if not numeric[numeric["instance_aware"] != True].empty  # noqa: E712
        else float("nan")
    )
    instance_helps = (
        "yes"
        if math.isfinite(instance_best)
        and math.isfinite(general_best)
        and instance_best < general_best
        else "no"
    )

    scheduled = numeric[numeric["target_family"] == "degree_scheduled"]
    overshoot_rate = (
        float((scheduled["overshoot_detected"] == True).mean())  # noqa: E712
        if not scheduled.empty
        else float("nan")
    )

    success_family = numeric[numeric["target_family"] == "success_amplitude_aware"]
    success_best_prob = (
        float(success_family["success_probability_proxy"].max())
        if not success_family.empty
        else float("nan")
    )
    base_best_prob = float(numeric["success_probability_proxy"].max())

    best_row = (
        deployable.loc[deployable["residual_ratio_vs_no_update"].idxmin()]
        if not deployable.empty and math.isfinite(best_deployable_ratio)
        else None
    )
    if best_row is not None:
        best_ratio_value = float(best_row["residual_ratio_vs_no_update"])
        best_line = (
            f"target_family={best_row['target_family']}, alpha={float(best_row['alpha']):.3g}, "
            f"degree={int(best_row['degree'])}, ratio={best_ratio_value:.6g}, "
            f"direction_error={float(best_row['direction_error_vs_ridge']):.6g}, "
            f"success_proxy={float(best_row['success_probability_proxy']):.4g}"
        )
    else:
        best_line = "no QSVT-safe deployable target reached residual_ratio_vs_no_update <= 0.1"

    return "\n".join(
        [
            "# Co-Designed Bounded-Target Study",
            "",
            CODESIGNED_CLAIM,
            "",
            "## Counts",
            f"- Total target rows: {len(frame)}",
            f"- QSVT-safe deployable rows: {len(deployable)}",
            f"- Diagnostic-only rows: {len(diagnostic)}",
            "",
            "## Best QSVT-safe Deployable Target",
            f"- {best_line}",
            f"- Best deployable residual_ratio_vs_no_update: {best_deployable_ratio:.6g}",
            f"- Best diagnostic residual_ratio_vs_no_update: {best_diagnostic_ratio:.6g}",
            f"- Best direction_error_vs_ridge (any family): {best_direction:.6g}",
            "",
            "## Required Answers",
            f"1. Does weighted singular-support fitting improve direction error? {weighting_helps} "
            f"(weighted {weighted_dir:.4g} vs clipped {clipped_dir:.4g}).",
            f"2. Does any QSVT-safe target reduce residual under deployable scaling to <= 0.1? "
            f"{any_safe_reduces} (best deployable ratio {best_deployable_ratio:.6g}).",
            f"3. Does relaxing to instance-aware fitting help? {instance_helps} (instance "
            f"{instance_best:.6g} vs general {general_best:.6g}).",
            f"4. Does degree scheduling avoid high-degree overshoot? overshoot rate among "
            f"degree_scheduled rows = {overshoot_rate:.3g}.",
            f"5. Does success-amplitude-aware fitting improve deployability (success proxy)? best "
            f"success_amplitude_aware proxy {success_best_prob:.4g} vs best overall "
            f"{base_best_prob:.4g}.",
            "",
            "## Conclusion",
            "- The known-C-rescaled physical update is independent of the bounding constant C; "
            "C only sets the boundedness margin and success amplitude. Target SHAPE (weighting, "
            "clipping, degree schedule) is what moves the residual and direction. The deployable "
            "amplitude/known-C residual is reported as residual_ratio_vs_no_update; the "
            "best-scalar diagnostic residual is a non-deployable lower bound.",
            "",
        ]
    )


def _failed_row(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    alpha: float,
    degree: int,
    target_family: str,
    reason: str,
) -> dict[str, Any]:
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "alpha": float(alpha),
            "degree": int(degree),
            "target_family": target_family,
            "weighting_scheme": "n/a",
            "qsvt_safe": False,
            "overshoot_detected": False,
            "instance_aware": False,
            "deployability_class": "runtime_limited",
            "dominant_limitation": reason,
        }
    )
    return row


def _weighted_filter_error(
    physical_filter: np.ndarray, target_filter: np.ndarray, weights: np.ndarray
) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or physical_filter.size == 0:
        return float("nan")
    error = physical_filter - target_filter
    return float(np.sqrt(np.sum(weights * error**2) / weight_sum))


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    if not np.all(np.isfinite(update)):
        return float("nan")
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _cos_angle(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64)))
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def _direction_error_from_cos(cos_angle: float) -> float:
    if not math.isfinite(cos_angle):
        return float("nan")
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - cos_angle))))


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
