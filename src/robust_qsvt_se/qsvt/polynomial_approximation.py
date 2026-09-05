from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import linprog

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, required_case_name
from robust_qsvt_se.qsvt.polynomial import (
    fit_odd_regularized_polynomial,
    regularized_filter_on_normalized_domain,
)

APPROXIMATION_CAVEAT = (
    "Bounded polynomial approximation diagnostic for the QSVT-compatible target. "
    "This is not a quantum speedup claim, quantum advantage claim, full hardware "
    "execution, or evidence that QSVT outperforms Ridge/Tikhonov under the same alpha."
)
POLYNOMIAL_FALLBACK_CAVEAT = (
    "Polynomial fallback, not full QSP/QSVT phase synthesis. Passing or failing "
    "the configured tolerance is reported explicitly."
)
DEFAULT_ALPHA_GRID = [1.0e-4, 1.0e-2, 1.0]
DEFAULT_DEGREE_GRID = [15, 25, 35, 51, 71, 101, 151, 201]
DEFAULT_ADAPTIVE_DEGREES = [15, 25, 35, 51, 71, 101, 151, 201, 251, 301]
DEFAULT_TOLERANCES = [1.0e-2, 5.0e-3, 1.0e-3, 1.0e-4]
DEFAULT_METHODS = [
    "odd_chebyshev_reduced_y",
    "odd_chebyshev_ls",
    "odd_chebyshev_minimax_lp",
    "chebyshev_interpolation_positive",
]


@dataclass(frozen=True, slots=True)
class ApproximationContext:
    case_name: str
    matrix_source: str
    matrix_shape: str
    beta: float
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    domain_min: float
    domain_max: float
    source_note: str


@dataclass(frozen=True, slots=True)
class PolynomialApproximationResult:
    method: str
    degree: int
    parity: str
    approximation_values: np.ndarray
    target_values: np.ndarray
    bounded_target_values: np.ndarray
    bounded_approximation_values: np.ndarray
    pointwise_errors: np.ndarray
    evaluation_points: np.ndarray
    evaluation_kind: list[str]
    bounded_scaling_C: float
    max_unbounded_filter_value: float
    max_bounded_filter_value: float
    power_coefficients: np.ndarray
    chebyshev_coefficients: np.ndarray
    numerical_stability_status: str
    requires_optional_dependency: bool
    dependency_name_if_any: str
    caveat: str


def build_approximation_context(config: dict[str, Any] | None = None) -> ApproximationContext:
    resolved = dict(config or {})
    try:
        system, matrix_source = build_engineering_system(resolved)
        source_note = "primary matrix source"
    except Exception as exc:
        if not bool(resolved.get("fallback_to_synthetic", True)):
            raise
        fallback = dict(resolved)
        fallback["matrix_source"] = "synthetic"
        system, matrix_source = build_engineering_system(fallback)
        matrix_source = f"{matrix_source}_fallback"
        source_note = f"synthetic fallback after: {exc}"
    H_tilde = np.asarray(system.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H_tilde, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("approximation diagnostics require positive singular values")
    beta = float(np.max(positive))
    normalized = positive / beta
    return ApproximationContext(
        case_name=required_case_name(system),
        matrix_source=matrix_source,
        matrix_shape=f"{H_tilde.shape[0]}x{H_tilde.shape[1]}",
        beta=beta,
        singular_values=positive,
        normalized_singular_values=normalized,
        domain_min=max(float(np.min(normalized)), np.finfo(float).eps),
        domain_max=1.0,
        source_note=source_note,
    )


def evaluate_polynomial_approximation(
    *,
    context: ApproximationContext,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
) -> PolynomialApproximationResult:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    odd_degree = as_odd_degree(degree)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    grid = np.linspace(
        context.domain_min,
        context.domain_max,
        max(int(grid_size), odd_degree + 2),
        dtype=np.float64,
    )
    evaluation_points = np.concatenate([grid, context.normalized_singular_values])
    evaluation_kind = ["grid"] * grid.size + [
        "actual_singular_value"
    ] * context.normalized_singular_values.size
    target = regularized_filter_on_normalized_domain(
        evaluation_points,
        alpha=alpha,
        block_encoding_normalization=context.beta,
    )
    bounded_scaling = max(1.0, float(np.max(np.abs(target))))
    bounded_target = target / bounded_scaling
    fit_grid = np.unique(np.concatenate([grid, context.normalized_singular_values]))
    fit_target = regularized_filter_on_normalized_domain(
        fit_grid,
        alpha=alpha,
        block_encoding_normalization=context.beta,
    )
    fit_bounded_target = fit_target / bounded_scaling
    if method == "odd_chebyshev_reduced_y":
        approximation, parity, power_coeffs, cheb_coeffs, status = _fit_reduced_y(
            alpha=alpha,
            context=context,
            degree=odd_degree,
            grid_size=max(grid.size, odd_degree + 2),
            evaluation_points=evaluation_points,
            bounded_scaling=bounded_scaling,
        )
    elif method == "odd_chebyshev_ls":
        approximation, parity, power_coeffs, cheb_coeffs, status = _fit_odd_chebyshev_ls(
            fit_grid=fit_grid,
            fit_target=fit_bounded_target,
            evaluation_points=evaluation_points,
            degree=odd_degree,
        )
    elif method == "odd_chebyshev_minimax_lp":
        approximation, parity, power_coeffs, cheb_coeffs, status = _fit_odd_minimax_lp(
            fit_grid=fit_grid,
            fit_target=fit_bounded_target,
            evaluation_points=evaluation_points,
            degree=odd_degree,
        )
    elif method == "chebyshev_interpolation_positive":
        approximation, parity, power_coeffs, cheb_coeffs, status = _fit_positive_interpolation(
            alpha=alpha,
            context=context,
            degree=int(degree),
            evaluation_points=evaluation_points,
            bounded_scaling=bounded_scaling,
        )
    else:
        raise ValueError(f"unknown approximation method: {method}")

    bounded_approximation = np.asarray(approximation, dtype=np.float64)
    errors = np.abs(bounded_approximation - bounded_target)
    return PolynomialApproximationResult(
        method=method,
        degree=int(odd_degree if parity == "odd" else degree),
        parity=parity,
        approximation_values=bounded_approximation * bounded_scaling,
        target_values=target,
        bounded_target_values=bounded_target,
        bounded_approximation_values=bounded_approximation,
        pointwise_errors=errors,
        evaluation_points=evaluation_points,
        evaluation_kind=evaluation_kind,
        bounded_scaling_C=float(bounded_scaling),
        max_unbounded_filter_value=float(np.max(np.abs(target))),
        max_bounded_filter_value=float(np.max(np.abs(bounded_target))),
        power_coefficients=np.asarray(power_coeffs, dtype=np.float64),
        chebyshev_coefficients=np.asarray(cheb_coeffs, dtype=np.float64),
        numerical_stability_status=status,
        requires_optional_dependency=False,
        dependency_name_if_any="scipy.optimize.linprog" if method.endswith("_lp") else "",
        caveat=_method_caveat(method),
    )


def approximation_summary_row(
    *,
    context: ApproximationContext,
    alpha: float,
    result: PolynomialApproximationResult,
    tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    grid_errors = result.pointwise_errors[
        np.asarray(result.evaluation_kind, dtype=object) == "grid"
    ]
    singular_errors = result.pointwise_errors[
        np.asarray(result.evaluation_kind, dtype=object) == "actual_singular_value"
    ]
    max_error = float(np.max(result.pointwise_errors))
    return {
        "case_name": context.case_name,
        "matrix_source": context.matrix_source,
        "matrix_shape": context.matrix_shape,
        "alpha": float(alpha),
        "degree": int(result.degree),
        "sigma_min": float(np.min(context.singular_values)),
        "sigma_max": float(np.max(context.singular_values)),
        "bounded_scaling_C": result.bounded_scaling_C,
        "max_unbounded_filter_value": result.max_unbounded_filter_value,
        "max_bounded_filter_value": result.max_bounded_filter_value,
        "approximation_method": result.method,
        "method": result.method,
        "parity": result.parity,
        "max_pointwise_error": max_error,
        "mean_pointwise_error": float(np.mean(result.pointwise_errors)),
        "rms_pointwise_error": float(np.sqrt(np.mean(result.pointwise_errors**2))),
        "max_grid_pointwise_error": float(np.max(grid_errors)),
        "error_at_actual_singular_values_max": float(np.max(singular_errors)),
        "error_at_actual_singular_values_mean": float(np.mean(singular_errors)),
        "max_error_on_singular_values": float(np.max(singular_errors)),
        "mean_error_on_singular_values": float(np.mean(singular_errors)),
        "grid_size": int(np.count_nonzero(np.asarray(result.evaluation_kind) == "grid")),
        "query_count_estimate": int(2 * result.degree + 1),
        "passed_tol_1e_minus_2": bool(max_error <= 1.0e-2),
        "passed_tol_5e_minus_3": bool(max_error <= 5.0e-3),
        "passed_tol_1e_minus_3": bool(max_error <= 1.0e-3),
        "passed_tol_1e_minus_4": bool(max_error <= 1.0e-4),
        "passed_strict_1e3": bool(max_error <= 1.0e-3),
        "passed_1e_minus_3": bool(max_error <= 1.0e-3),
        "passed_configured_tolerance": bool(max_error <= tolerance),
        "numerical_stability_status": result.numerical_stability_status,
        "requires_optional_dependency": result.requires_optional_dependency,
        "dependency_name_if_any": result.dependency_name_if_any,
        "source_note": context.source_note,
        "caveat": result.caveat,
    }


def pointwise_rows(
    *,
    context: ApproximationContext,
    alpha: float,
    result: PolynomialApproximationResult,
    include_values: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (kind, sigma, target, approximation, error) in enumerate(
        zip(
            result.evaluation_kind,
            result.evaluation_points,
            result.bounded_target_values,
            result.bounded_approximation_values,
            result.pointwise_errors,
            strict=True,
        )
    ):
        row = {
            "case_name": context.case_name,
            "matrix_source": context.matrix_source,
            "alpha": float(alpha),
            "degree": int(result.degree),
            "method": result.method,
            "evaluation_index": int(index),
            "evaluation_kind": kind,
            "sigma_normalized": float(sigma),
            "pointwise_error": float(error),
        }
        if include_values:
            row.update(
                {
                    "target_bounded_value": float(target),
                    "approximation_bounded_value": float(approximation),
                    "target_value": float(target * result.bounded_scaling_C),
                    "approximation_value": float(approximation * result.bounded_scaling_C),
                }
            )
        rows.append(row)
    return rows


def as_odd_degree(degree: int) -> int:
    value = int(degree)
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _fit_reduced_y(
    *,
    alpha: float,
    context: ApproximationContext,
    degree: int,
    grid_size: int,
    evaluation_points: np.ndarray,
    bounded_scaling: float,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, str]:
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha,
        block_encoding_normalization=context.beta,
        degree=degree,
        domain_min=context.domain_min,
        domain_max=context.domain_max,
        grid_size=grid_size,
    )
    polynomial = Polynomial(np.asarray(approximation.power_coefficients, dtype=np.float64))
    values = polynomial(evaluation_points) / bounded_scaling
    cheb = polynomial.convert(kind=Chebyshev).coef
    return values, "odd", np.asarray(polynomial.coef, dtype=np.float64), cheb, "ok"


def _fit_odd_chebyshev_ls(
    *,
    fit_grid: np.ndarray,
    fit_target: np.ndarray,
    evaluation_points: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, str]:
    odds = _odd_indices(degree)
    fit_basis = chebvander(fit_grid, degree)[:, odds]
    coefficients, *_ = np.linalg.lstsq(fit_basis, fit_target, rcond=None)
    cheb_coefficients = np.zeros(degree + 1, dtype=np.float64)
    cheb_coefficients[odds] = coefficients
    values = chebvander(evaluation_points, degree)[:, odds] @ coefficients
    power, status = _safe_chebyshev_to_power_coefficients(cheb_coefficients)
    return values, "odd", power, cheb_coefficients, status


def _fit_odd_minimax_lp(
    *,
    fit_grid: np.ndarray,
    fit_target: np.ndarray,
    evaluation_points: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, str]:
    odds = _odd_indices(degree)
    fit_basis = chebvander(fit_grid, degree)[:, odds]
    objective = np.zeros(len(odds) + 1, dtype=np.float64)
    objective[-1] = 1.0
    ones = np.ones(fit_basis.shape[0], dtype=np.float64)
    constraints = np.vstack(
        [
            np.column_stack([fit_basis, -ones]),
            np.column_stack([-fit_basis, -ones]),
        ]
    )
    bounds = np.concatenate([fit_target, -fit_target])
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * len(odds) + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"minimax linear program failed: {result.message}")
    coefficients = np.asarray(result.x[:-1], dtype=np.float64)
    cheb_coefficients = np.zeros(degree + 1, dtype=np.float64)
    cheb_coefficients[odds] = coefficients
    values = chebvander(evaluation_points, degree)[:, odds] @ coefficients
    power, status = _safe_chebyshev_to_power_coefficients(cheb_coefficients)
    return values, "odd", power, cheb_coefficients, status


def _fit_positive_interpolation(
    *,
    alpha: float,
    context: ApproximationContext,
    degree: int,
    evaluation_points: np.ndarray,
    bounded_scaling: float,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, str]:
    def target_function(points: np.ndarray) -> np.ndarray:
        return (
            regularized_filter_on_normalized_domain(
                np.asarray(points, dtype=np.float64),
                alpha=alpha,
                block_encoding_normalization=context.beta,
            )
            / bounded_scaling
        )

    polynomial = Chebyshev.interpolate(
        target_function,
        deg=int(degree),
        domain=[context.domain_min, context.domain_max],
    )
    values = polynomial(evaluation_points)
    power, status = _safe_chebyshev_to_power_coefficients(
        np.asarray(polynomial.coef, dtype=np.float64)
    )
    return values, "any", power, np.asarray(polynomial.coef, dtype=np.float64), status


def _odd_indices(degree: int) -> list[int]:
    return [index for index in range(int(degree) + 1) if index % 2 == 1]


def _safe_chebyshev_to_power_coefficients(
    cheb_coefficients: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Convert to monomial coefficients while labelling unstable diagnostic conversions."""

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            power = (
                Chebyshev(np.asarray(cheb_coefficients, dtype=np.float64), domain=[-1.0, 1.0])
                .convert(kind=Polynomial)
                .coef
            )
    status = "ok"
    if captured or not np.all(np.isfinite(power)):
        status = "power_basis_overflow_diagnostic_only"
    return np.asarray(power, dtype=np.float64), status


def _method_caveat(method: str) -> str:
    if method == "chebyshev_interpolation_positive":
        return (
            "Positive-interval Chebyshev interpolation diagnostic only; it does not "
            "enforce QSVT odd parity. " + APPROXIMATION_CAVEAT
        )
    if method == "odd_chebyshev_minimax_lp":
        return (
            "Odd Chebyshev minimax linear-program diagnostic; polynomial fallback, "
            "not full phase synthesis. " + APPROXIMATION_CAVEAT
        )
    return POLYNOMIAL_FALLBACK_CAVEAT + " " + APPROXIMATION_CAVEAT
