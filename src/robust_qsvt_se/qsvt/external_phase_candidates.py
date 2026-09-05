from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander

from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain
from robust_qsvt_se.qsvt.polynomial_approximation import (
    ApproximationContext,
    build_approximation_context,
    evaluate_polynomial_approximation,
)

TARGET_TOLERANCE = 1.0e-3


@dataclass(frozen=True, slots=True)
class ExternalPhaseCandidate:
    candidate_name: str
    alpha: float
    degree: int
    native_basis: str
    method: str
    lambda_if_any: float | None
    chebyshev_coefficients: np.ndarray | None
    monomial_coefficients: np.ndarray | None
    full_domain_grid: np.ndarray
    full_domain_target: np.ndarray
    full_domain_polynomial: np.ndarray | None
    actual_singular_values: np.ndarray
    actual_singular_targets: np.ndarray
    actual_singular_polynomial: np.ndarray | None
    domain_min: float
    domain_max: float
    bounded_scaling_C: float
    native_max_error_full_domain: float
    native_max_error_actual_singular_values: float
    native_max_abs_value: float
    bounded_in_native_basis: bool
    parity_error: float
    monomial_dynamic_range: float
    monomial_bounded_after_conversion: bool
    supported_input_bases: tuple[str, ...]

    @property
    def supports_chebyshev(self) -> bool:
        return "chebyshev" in self.supported_input_bases

    @property
    def supports_monomial(self) -> bool:
        return "monomial" in self.supported_input_bases

    @property
    def supports_function_values(self) -> bool:
        return "function_values" in self.supported_input_bases


def build_external_phase_candidates(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    context = build_approximation_context(_context_config(resolved))
    candidates: list[ExternalPhaseCandidate] = []
    for degree in resolved["degrees"]:
        odd_degree = _as_odd_degree(int(degree))
        candidates.extend(_polynomial_candidates(context, resolved, odd_degree))
    candidates.append(
        _function_sample_candidate(
            context,
            resolved,
            _as_odd_degree(resolved["function_degree"]),
        )
    )
    return {"context": context, "candidates": candidates, "config": resolved}


def candidate_is_safe_for_backend(
    candidate: ExternalPhaseCandidate,
    *,
    backend_name: str,
    native_tolerance: float = TARGET_TOLERANCE,
    coefficient_dynamic_range_limit: float = 1.0e12,
) -> tuple[bool, str, str]:
    if backend_name == "pyqsp_sym_qsp":
        if not candidate.supports_chebyshev:
            return False, "skipped_unsupported_basis", "candidate lacks Chebyshev coefficients"
        if candidate.native_max_error_full_domain > native_tolerance:
            return False, "skipped_unstable_candidate", "native full-domain error exceeds 1e-3"
        if not candidate.bounded_in_native_basis:
            return False, "skipped_unstable_candidate", "native polynomial is not bounded"
        if candidate.parity_error > 1.0e-10:
            return False, "skipped_unstable_candidate", "candidate parity check failed"
        return True, "ready", ""
    if backend_name == "pennylane_poly_to_angles":
        if not candidate.supports_monomial or candidate.monomial_coefficients is None:
            return False, "skipped_unsupported_basis", "candidate lacks monomial coefficients"
        if candidate.native_max_error_full_domain > native_tolerance:
            return False, "skipped_unstable_candidate", "native full-domain error exceeds 1e-3"
        if not candidate.monomial_bounded_after_conversion:
            return False, "skipped_unstable_candidate", "monomial conversion is not bounded"
        if candidate.monomial_dynamic_range > coefficient_dynamic_range_limit:
            return (
                False,
                "skipped_unstable_candidate",
                "monomial coefficient dynamic range is unsafe",
            )
        return True, "ready", ""
    if backend_name == "local_optimization_qsp":
        if not candidate.supports_function_values:
            return False, "skipped_unsupported_basis", "candidate lacks function samples"
        return True, "ready", ""
    return False, "skipped_backend_unavailable", "backend unavailable or unsupported"


def _polynomial_candidates(
    context: ApproximationContext,
    config: dict[str, Any],
    degree: int,
) -> list[ExternalPhaseCandidate]:
    candidates: list[ExternalPhaseCandidate] = []
    for method in ["odd_chebyshev_minimax_lp", "odd_chebyshev_ls"]:
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(config["alpha"]),
            degree=degree,
            method=method,
            grid_size=int(config["fit_grid_size"]),
        )
        candidates.append(
            _candidate_from_chebyshev(
                context=context,
                config=config,
                degree=int(result.degree),
                method=method,
                lambda_if_any=None,
                chebyshev_coefficients=np.asarray(result.chebyshev_coefficients, dtype=np.float64),
            )
        )
    for penalty in config["lambdas"]:
        candidates.append(
            _candidate_from_chebyshev(
                context=context,
                config=config,
                degree=degree,
                method="coefficient_conditioned_chebyshev",
                lambda_if_any=float(penalty),
                chebyshev_coefficients=_fit_conditioned_chebyshev(
                    context=context,
                    alpha=float(config["alpha"]),
                    degree=degree,
                    grid_size=int(config["fit_grid_size"]),
                    penalty=float(penalty),
                ),
            )
        )
    return candidates


def _candidate_from_chebyshev(
    *,
    context: ApproximationContext,
    config: dict[str, Any],
    degree: int,
    method: str,
    lambda_if_any: float | None,
    chebyshev_coefficients: np.ndarray,
) -> ExternalPhaseCandidate:
    alpha = float(config["alpha"])
    cheb = _odd_padded(chebyshev_coefficients, degree)
    full_grid = _signed_domain_grid(context, int(config["validation_grid_size"]))
    actual = np.asarray(context.normalized_singular_values, dtype=np.float64)
    target_full, scale = _bounded_target(full_grid, context=context, alpha=alpha)
    actual_target, _ = _bounded_target(actual, context=context, alpha=alpha, scale=scale)
    polynomial = Chebyshev(cheb, domain=[-1.0, 1.0])
    full_poly = polynomial(full_grid)
    actual_poly = polynomial(actual)
    full_error = np.abs(full_poly - target_full)
    actual_error = np.abs(actual_poly - actual_target)
    unit_grid = np.linspace(-1.0, 1.0, int(config["boundedness_grid_size"]))
    native_max_abs = float(np.max(np.abs(polynomial(unit_grid))))
    monomial = polynomial.convert(kind=Polynomial).coef
    monomial_diag = _coefficient_diagnostics(monomial)
    monomial_values = Polynomial(monomial)(unit_grid)
    monomial_bounded = bool(
        np.max(np.abs(monomial_values)) <= 1.0 + float(config["bound_tolerance"])
    )
    supported_bases = ("chebyshev",)
    if (
        monomial_bounded
        and monomial_diag["coefficient_dynamic_range"]
        <= float(config["coefficient_dynamic_range_limit"])
        and float(np.max(np.abs(Polynomial(monomial)(full_grid) - full_poly)))
        <= float(config["conversion_error_tolerance"])
    ):
        supported_bases = ("chebyshev", "monomial")
    return ExternalPhaseCandidate(
        candidate_name=_candidate_name(method, degree, lambda_if_any),
        alpha=alpha,
        degree=degree,
        native_basis="chebyshev_T_low_to_high_on_unit_interval",
        method=method,
        lambda_if_any=lambda_if_any,
        chebyshev_coefficients=cheb,
        monomial_coefficients=np.asarray(monomial, dtype=np.float64),
        full_domain_grid=full_grid,
        full_domain_target=target_full,
        full_domain_polynomial=full_poly,
        actual_singular_values=actual,
        actual_singular_targets=actual_target,
        actual_singular_polynomial=actual_poly,
        domain_min=float(context.domain_min),
        domain_max=float(context.domain_max),
        bounded_scaling_C=scale,
        native_max_error_full_domain=float(np.max(full_error)),
        native_max_error_actual_singular_values=float(np.max(actual_error)),
        native_max_abs_value=native_max_abs,
        bounded_in_native_basis=bool(native_max_abs <= 1.0 + float(config["bound_tolerance"])),
        parity_error=float(_max_even_coefficient_abs(cheb)),
        monomial_dynamic_range=monomial_diag["coefficient_dynamic_range"],
        monomial_bounded_after_conversion=monomial_bounded,
        supported_input_bases=supported_bases,
    )


def _function_sample_candidate(
    context: ApproximationContext,
    config: dict[str, Any],
    degree: int,
) -> ExternalPhaseCandidate:
    alpha = float(config["alpha"])
    full_grid = _signed_domain_grid(context, int(config["validation_grid_size"]))
    actual = np.asarray(context.normalized_singular_values, dtype=np.float64)
    target_full, scale = _bounded_target(full_grid, context=context, alpha=alpha)
    actual_target, _ = _bounded_target(actual, context=context, alpha=alpha, scale=scale)
    return ExternalPhaseCandidate(
        candidate_name=f"function_sample_target_degree_{degree}",
        alpha=alpha,
        degree=degree,
        native_basis="function_values_on_validation_grid",
        method="function_sample_target",
        lambda_if_any=None,
        chebyshev_coefficients=None,
        monomial_coefficients=None,
        full_domain_grid=full_grid,
        full_domain_target=target_full,
        full_domain_polynomial=None,
        actual_singular_values=actual,
        actual_singular_targets=actual_target,
        actual_singular_polynomial=None,
        domain_min=float(context.domain_min),
        domain_max=float(context.domain_max),
        bounded_scaling_C=scale,
        native_max_error_full_domain=0.0,
        native_max_error_actual_singular_values=0.0,
        native_max_abs_value=float(np.max(np.abs(target_full))),
        bounded_in_native_basis=True,
        parity_error=0.0,
        monomial_dynamic_range=np.nan,
        monomial_bounded_after_conversion=False,
        supported_input_bases=("function_values",),
    )


def _fit_conditioned_chebyshev(
    *,
    context: ApproximationContext,
    alpha: float,
    degree: int,
    grid_size: int,
    penalty: float,
) -> np.ndarray:
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(context.domain_min, context.domain_max, max(grid_size, degree + 2)),
                context.normalized_singular_values,
            ]
        )
    )
    target, _ = _bounded_target(grid, context=context, alpha=alpha)
    odds = [index for index in range(degree + 1) if index % 2 == 1]
    basis = chebvander(grid, degree)[:, odds]
    lhs = basis.T @ basis + penalty * np.eye(len(odds), dtype=np.float64)
    rhs = basis.T @ target
    coefficients = np.linalg.solve(lhs, rhs)
    cheb = np.zeros(degree + 1, dtype=np.float64)
    cheb[odds] = coefficients
    return cheb


def _signed_domain_grid(context: ApproximationContext, grid_size: int) -> np.ndarray:
    positive = np.linspace(context.domain_min, context.domain_max, max(grid_size // 2, 2))
    return np.unique(np.concatenate([-positive[::-1], positive]))


def _bounded_target(
    points: np.ndarray,
    *,
    context: ApproximationContext,
    alpha: float,
    scale: float | None = None,
) -> tuple[np.ndarray, float]:
    values = regularized_filter_on_normalized_domain(
        np.asarray(points, dtype=np.float64),
        alpha=alpha,
        block_encoding_normalization=context.beta,
    )
    if scale is None:
        positive = np.linspace(context.domain_min, context.domain_max, 4097)
        positive_values = regularized_filter_on_normalized_domain(
            positive,
            alpha=alpha,
            block_encoding_normalization=context.beta,
        )
        scale = max(1.0, float(np.max(np.abs(positive_values))))
    return values / scale, float(scale)


def _coefficient_diagnostics(coefficients: np.ndarray) -> dict[str, float]:
    values = np.asarray(coefficients, dtype=np.float64)
    abs_values = np.abs(values)
    nonzero = abs_values[abs_values > 0.0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else np.nan
    max_abs = float(np.max(abs_values)) if abs_values.size else np.nan
    dynamic = float(max_abs / min_nonzero) if np.isfinite(min_nonzero) and min_nonzero else np.nan
    return {"coefficient_dynamic_range": dynamic}


def _candidate_name(method: str, degree: int, lambda_if_any: float | None) -> str:
    if lambda_if_any is None:
        return f"{method}_degree_{degree}"
    return f"{method}_degree_{degree}_lambda_{lambda_if_any:.0e}"


def _odd_padded(values: np.ndarray, degree: int) -> np.ndarray:
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    coefficients[: min(source.size, coefficients.size)] = source[: coefficients.size]
    coefficients[::2] = 0.0
    return coefficients


def _max_even_coefficient_abs(coefficients: np.ndarray) -> float:
    values = np.asarray(coefficients, dtype=np.float64)
    return float(np.max(np.abs(values[::2]))) if values.size else 0.0


def _as_odd_degree(degree: int) -> int:
    value = int(degree)
    return value if value % 2 == 1 else value + 1


def _context_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "matrix_source": config["matrix_source"],
        "case_name": config["case_name"],
        "case_source": config["case_source"],
        "seed": config["seed"],
        "fallback_to_synthetic": config["fallback_to_synthetic"],
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "degrees": [35, 51, 71, 101, 151, 201],
        "lambdas": [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4],
        "function_degree": 101,
        "fit_grid_size": 513,
        "validation_grid_size": 1201,
        "boundedness_grid_size": 2049,
        "bound_tolerance": 1.0e-5,
        "coefficient_dynamic_range_limit": 1.0e12,
        "conversion_error_tolerance": 1.0e-5,
    }
    if config:
        resolved.update(config)
    return resolved
