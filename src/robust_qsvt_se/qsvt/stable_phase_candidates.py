from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain
from robust_qsvt_se.qsvt.polynomial_approximation import (
    ApproximationContext,
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

TARGET_TOLERANCE = 1.0e-3
CANDIDATE_CAVEAT = (
    "Stable polynomial candidate diagnostic only. A safe candidate is a necessary "
    "input to phase synthesis, not proof of hardware execution, quantum speedup, "
    "quantum advantage, or QSVT superiority over Ridge/Tikhonov."
)

SUMMARY_COLUMNS = [
    "candidate_name",
    "alpha",
    "degree",
    "native_basis",
    "method",
    "lambda_if_any",
    "native_max_error",
    "native_mean_error",
    "native_rms_error",
    "native_max_abs_value",
    "bounded_in_native_basis",
    "parity_error",
    "conversion_method",
    "conversion_precision",
    "conversion_max_error",
    "max_abs_coefficient",
    "min_abs_nonzero_coefficient",
    "coefficient_dynamic_range",
    "bounded_after_conversion",
    "post_conversion_max_abs_value",
    "safe_for_phase_synthesis",
    "failure_reason_if_any",
    "recommended_next_step",
]


@dataclass(frozen=True, slots=True)
class StableCandidateConfig:
    output_dir: str = "outputs/qsvt_stable_phase_candidates"
    matrix_source: str = "ieee14_ac_weighted_jacobian"
    case_name: str = "ieee14"
    case_source: str = "pypower"
    seed: int = 123
    fallback_to_synthetic: bool = True
    alpha: float = 1.0e-2
    degrees: tuple[int, ...] = (35, 51, 71, 101, 151, 201)
    lambdas: tuple[float, ...] = (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4)
    approximation_grid_size: int = 513
    boundedness_grid_size: int = 2049
    conversion_grid_size: int = 2049
    bound_tolerance: float = 1.0e-5
    parity_tolerance: float = 1.0e-10
    conversion_error_tolerance: float = 1.0e-5
    coefficient_dynamic_range_limit: float = 1.0e12
    include_decimal_conversion: bool = True
    decimal_precision: int = 80

    def context_config(self) -> dict[str, Any]:
        return {
            "matrix_source": self.matrix_source,
            "case_name": self.case_name,
            "case_source": self.case_source,
            "seed": self.seed,
            "fallback_to_synthetic": self.fallback_to_synthetic,
        }


@dataclass(frozen=True, slots=True)
class CandidatePolynomial:
    candidate_base_name: str
    alpha: float
    degree: int
    method: str
    lambda_if_any: float | None
    chebyshev_coefficients: np.ndarray


def build_stable_phase_candidates(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    candidate_config = _config_from_dict(resolved)
    output_dir = ensure_directory(Path(candidate_config.output_dir))
    context = build_approximation_context(candidate_config.context_config())
    candidates = _candidate_polynomials(context, candidate_config)

    summary_rows: list[dict[str, Any]] = []
    cheb_rows: list[dict[str, Any]] = []
    monomial_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    boundedness_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        cheb_rows.extend(_chebyshev_coefficient_rows(candidate))
        for conversion_method in _conversion_methods(candidate_config):
            evaluation = _evaluate_candidate(
                candidate=candidate,
                context=context,
                config=candidate_config,
                conversion_method=conversion_method,
            )
            summary_rows.append(evaluation["summary"])
            monomial_rows.extend(evaluation["monomial_rows"])
            error_rows.extend(evaluation["error_rows"])
            boundedness_rows.extend(evaluation["boundedness_rows"])

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    cheb = pd.DataFrame(cheb_rows)
    monomial = pd.DataFrame(monomial_rows)
    errors = pd.DataFrame(error_rows)
    boundedness = pd.DataFrame(boundedness_rows)

    summary_csv = output_dir / "stable_phase_candidate_summary.csv"
    summary_json = output_dir / "stable_phase_candidate_summary.json"
    cheb_csv = output_dir / "candidate_coefficients_chebyshev.csv"
    monomial_csv = output_dir / "candidate_coefficients_monomial.csv"
    error_grid_csv = output_dir / "candidate_error_grid.csv"
    boundedness_grid_csv = output_dir / "candidate_boundedness_grid.csv"
    report_md = output_dir / "candidate_report.md"

    summary.to_csv(summary_csv, index=False)
    cheb.to_csv(cheb_csv, index=False)
    monomial.to_csv(monomial_csv, index=False)
    errors.to_csv(error_grid_csv, index=False)
    boundedness.to_csv(boundedness_grid_csv, index=False)
    write_json(summary_json, {"rows": summary_rows, "caveat": CANDIDATE_CAVEAT})
    report_md.write_text(_candidate_report(summary, candidate_config), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "stable_phase_candidate_summary_csv": str(summary_csv),
            "stable_phase_candidate_summary_json": str(summary_json),
            "candidate_coefficients_chebyshev_csv": str(cheb_csv),
            "candidate_coefficients_monomial_csv": str(monomial_csv),
            "candidate_error_grid_csv": str(error_grid_csv),
            "candidate_boundedness_grid_csv": str(boundedness_grid_csv),
            "candidate_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "stable_phase_candidate_summary_csv": summary_csv,
            "stable_phase_candidate_summary_json": summary_json,
            "candidate_coefficients_chebyshev_csv": cheb_csv,
            "candidate_coefficients_monomial_csv": monomial_csv,
            "candidate_error_grid_csv": error_grid_csv,
            "candidate_boundedness_grid_csv": boundedness_grid_csv,
            "candidate_report_md": report_md,
            "manifest": manifest,
        },
    }


def _candidate_polynomials(
    context: ApproximationContext,
    config: StableCandidateConfig,
) -> list[CandidatePolynomial]:
    candidates: list[CandidatePolynomial] = []
    for degree in config.degrees:
        odd_degree = _as_odd_degree(degree)
        for method in ["odd_chebyshev_minimax_lp", "odd_chebyshev_ls"]:
            try:
                result = evaluate_polynomial_approximation(
                    context=context,
                    alpha=config.alpha,
                    degree=odd_degree,
                    method=method,
                    grid_size=config.approximation_grid_size,
                )
            except Exception as exc:
                candidates.append(
                    CandidatePolynomial(
                        candidate_base_name=f"{method}_degree_{odd_degree}_failed_{_safe_slug(exc)}",
                        alpha=config.alpha,
                        degree=odd_degree,
                        method=f"{method}_construction_failed",
                        lambda_if_any=None,
                        chebyshev_coefficients=np.zeros(odd_degree + 1, dtype=np.float64),
                    )
                )
                continue
            candidates.append(
                CandidatePolynomial(
                    candidate_base_name=f"{method}_degree_{result.degree}",
                    alpha=config.alpha,
                    degree=int(result.degree),
                    method=method,
                    lambda_if_any=None,
                    chebyshev_coefficients=np.asarray(
                        result.chebyshev_coefficients,
                        dtype=np.float64,
                    ),
                )
            )
        for penalty in config.lambdas:
            cheb = _fit_conditioned_chebyshev(
                context=context,
                alpha=config.alpha,
                degree=odd_degree,
                grid_size=config.approximation_grid_size,
                penalty=float(penalty),
            )
            candidates.append(
                CandidatePolynomial(
                    candidate_base_name=(
                        f"conditioned_chebyshev_ls_degree_{odd_degree}_lambda_{penalty:.0e}"
                    ),
                    alpha=config.alpha,
                    degree=odd_degree,
                    method="conditioned_chebyshev_ls",
                    lambda_if_any=float(penalty),
                    chebyshev_coefficients=cheb,
                )
            )
    return candidates


def _fit_conditioned_chebyshev(
    *,
    context: ApproximationContext,
    alpha: float,
    degree: int,
    grid_size: int,
    penalty: float,
) -> np.ndarray:
    grid = _fit_grid(context, degree, grid_size)
    target = _bounded_target(grid, context=context, alpha=alpha)
    odds = _odd_indices(degree)
    basis = chebvander(grid, degree)[:, odds]
    lhs = basis.T @ basis + float(penalty) * np.eye(len(odds), dtype=np.float64)
    rhs = basis.T @ target
    coefficients = np.linalg.solve(lhs, rhs)
    cheb = np.zeros(degree + 1, dtype=np.float64)
    cheb[odds] = coefficients
    return cheb


def _evaluate_candidate(
    *,
    candidate: CandidatePolynomial,
    context: ApproximationContext,
    config: StableCandidateConfig,
    conversion_method: str,
) -> dict[str, Any]:
    cheb_coeffs = _odd_padded(candidate.chebyshev_coefficients, candidate.degree)
    conversion = _convert_chebyshev_to_monomial(cheb_coeffs, conversion_method, config)
    candidate_name = f"{candidate.candidate_base_name}_{conversion_method}"

    positive_grid = np.linspace(
        context.domain_min,
        context.domain_max,
        config.approximation_grid_size,
        dtype=np.float64,
    )
    unit_grid = np.linspace(-1.0, 1.0, config.boundedness_grid_size, dtype=np.float64)
    conversion_grid = np.linspace(-1.0, 1.0, config.conversion_grid_size, dtype=np.float64)

    native_poly = Chebyshev(cheb_coeffs, domain=[-1.0, 1.0])
    native_positive = native_poly(positive_grid)
    native_unit = native_poly(unit_grid)
    native_conversion = native_poly(conversion_grid)
    target_positive = _bounded_target(positive_grid, context=context, alpha=candidate.alpha)
    native_errors = np.abs(native_positive - target_positive)

    coefficients = conversion["coefficients"]
    converted_poly = Polynomial(coefficients)
    converted_positive = converted_poly(positive_grid)
    converted_unit = converted_poly(unit_grid)
    converted_conversion = converted_poly(conversion_grid)
    conversion_errors = np.abs(converted_conversion - native_conversion)

    coeff_diag = _coefficient_diagnostics(coefficients)
    native_max_error = float(np.max(native_errors))
    native_mean_error = float(np.mean(native_errors))
    native_rms_error = float(np.sqrt(np.mean(native_errors**2)))
    native_max_abs = float(np.max(np.abs(native_unit)))
    post_conversion_max_abs = float(np.max(np.abs(converted_unit)))
    parity_error = float(
        max(
            _max_even_coefficient_abs(cheb_coeffs),
            _max_even_coefficient_abs(coefficients),
        )
    )
    bounded_native = bool(native_max_abs <= 1.0 + config.bound_tolerance)
    bounded_after_conversion = bool(post_conversion_max_abs <= 1.0 + config.bound_tolerance)
    conversion_max_error = float(np.max(conversion_errors)) if conversion_errors.size else np.nan

    safe, reason = _candidate_gate(
        native_max_error=native_max_error,
        bounded_native=bounded_native,
        parity_error=parity_error,
        conversion_max_error=conversion_max_error,
        dynamic_range=coeff_diag["coefficient_dynamic_range"],
        bounded_after_conversion=bounded_after_conversion,
        post_conversion_max_abs=post_conversion_max_abs,
        config=config,
    )
    summary = {
        "candidate_name": candidate_name,
        "alpha": float(candidate.alpha),
        "degree": int(candidate.degree),
        "native_basis": "chebyshev_T_low_to_high_on_unit_interval",
        "method": candidate.method,
        "lambda_if_any": candidate.lambda_if_any,
        "native_max_error": native_max_error,
        "native_mean_error": native_mean_error,
        "native_rms_error": native_rms_error,
        "native_max_abs_value": native_max_abs,
        "bounded_in_native_basis": bounded_native,
        "parity_error": parity_error,
        "conversion_method": conversion_method,
        "conversion_precision": conversion["precision"],
        "conversion_max_error": conversion_max_error,
        "max_abs_coefficient": coeff_diag["max_abs_coefficient"],
        "min_abs_nonzero_coefficient": coeff_diag["min_abs_nonzero_coefficient"],
        "coefficient_dynamic_range": coeff_diag["coefficient_dynamic_range"],
        "bounded_after_conversion": bounded_after_conversion,
        "post_conversion_max_abs_value": post_conversion_max_abs,
        "safe_for_phase_synthesis": safe,
        "failure_reason_if_any": reason,
        "recommended_next_step": _recommended_next_step(safe, reason),
    }
    return {
        "summary": summary,
        "monomial_rows": _monomial_coefficient_rows(
            candidate_name,
            conversion_method,
            coefficients,
        ),
        "error_rows": _error_grid_rows(
            candidate_name=candidate_name,
            positive_grid=positive_grid,
            target=target_positive,
            native_values=native_positive,
            converted_values=converted_positive,
        ),
        "boundedness_rows": _boundedness_grid_rows(
            candidate_name=candidate_name,
            unit_grid=unit_grid,
            native_values=native_unit,
            converted_values=converted_unit,
        ),
    }


def _candidate_gate(
    *,
    native_max_error: float,
    bounded_native: bool,
    parity_error: float,
    conversion_max_error: float,
    dynamic_range: float,
    bounded_after_conversion: bool,
    post_conversion_max_abs: float,
    config: StableCandidateConfig,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if not np.isfinite(native_max_error) or native_max_error > TARGET_TOLERANCE:
        reasons.append("native approximation error exceeds 1e-3")
    if not bounded_native:
        reasons.append("native Chebyshev polynomial is not bounded by 1 on [-1, 1]")
    if not np.isfinite(parity_error) or parity_error > config.parity_tolerance:
        reasons.append("odd-parity error exceeds tolerance")
    if (
        not np.isfinite(conversion_max_error)
        or conversion_max_error > config.conversion_error_tolerance
    ):
        reasons.append("Chebyshev-to-monomial conversion error exceeds safety tolerance")
    if not np.isfinite(dynamic_range) or dynamic_range > config.coefficient_dynamic_range_limit:
        reasons.append("monomial coefficient dynamic range exceeds safety threshold")
    if not bounded_after_conversion:
        reasons.append("converted monomial polynomial is not bounded by 1 on [-1, 1]")
    if post_conversion_max_abs > 1.0 + config.bound_tolerance:
        reasons.append("converted polynomial exceeds boundedness tolerance")
    return len(reasons) == 0, "; ".join(reasons)


def _convert_chebyshev_to_monomial(
    cheb_coefficients: np.ndarray,
    method: str,
    config: StableCandidateConfig,
) -> dict[str, Any]:
    if method == "float64":
        coefficients = (
            Chebyshev(cheb_coefficients, domain=[-1.0, 1.0]).convert(kind=Polynomial).coef
        )
        return {"coefficients": np.asarray(coefficients, dtype=np.float64), "precision": "float64"}
    if method == "longdouble_recurrence":
        return {
            "coefficients": _chebyshev_to_power_longdouble(cheb_coefficients),
            "precision": "numpy.longdouble",
        }
    if method == "decimal_recurrence":
        return {
            "coefficients": _chebyshev_to_power_decimal(
                cheb_coefficients,
                precision=config.decimal_precision,
            ),
            "precision": f"decimal_{config.decimal_precision}",
        }
    raise ValueError(f"unknown conversion method: {method}")


def _chebyshev_to_power_longdouble(cheb_coefficients: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(cheb_coefficients, dtype=np.longdouble)
    if coeffs.size == 0:
        return np.asarray([], dtype=np.float64)
    polys: list[np.ndarray] = [np.asarray([1.0], dtype=np.longdouble)]
    if coeffs.size > 1:
        polys.append(np.asarray([0.0, 1.0], dtype=np.longdouble))
    for _degree in range(2, coeffs.size):
        prev = polys[-1]
        prev2 = polys[-2]
        shifted = np.concatenate([np.asarray([0.0], dtype=np.longdouble), 2.0 * prev])
        padded = np.pad(prev2, (0, shifted.size - prev2.size))
        polys.append(shifted - padded)
    power = np.zeros(coeffs.size, dtype=np.longdouble)
    for index, coefficient in enumerate(coeffs):
        if coefficient == 0:
            continue
        poly = polys[index]
        power[: poly.size] += coefficient * poly
    return np.asarray(power, dtype=np.float64)


def _chebyshev_to_power_decimal(
    cheb_coefficients: np.ndarray,
    *,
    precision: int,
) -> np.ndarray:
    previous_precision = getcontext().prec
    getcontext().prec = int(precision)
    try:
        coeffs = [Decimal(str(float(value))) for value in cheb_coefficients]
        if not coeffs:
            return np.asarray([], dtype=np.float64)
        polys: list[list[Decimal]] = [[Decimal(1)]]
        if len(coeffs) > 1:
            polys.append([Decimal(0), Decimal(1)])
        for _degree in range(2, len(coeffs)):
            prev = [Decimal(0), *[Decimal(2) * value for value in polys[-1]]]
            prev2 = [*polys[-2], *([Decimal(0)] * (len(prev) - len(polys[-2])))]
            polys.append([a - b for a, b in zip(prev, prev2, strict=True)])
        power = [Decimal(0) for _ in coeffs]
        for index, coefficient in enumerate(coeffs):
            if coefficient == 0:
                continue
            for power_index, value in enumerate(polys[index]):
                power[power_index] += coefficient * value
        return np.asarray([float(value) for value in power], dtype=np.float64)
    finally:
        getcontext().prec = previous_precision


def _fit_grid(context: ApproximationContext, degree: int, grid_size: int) -> np.ndarray:
    positive = np.linspace(
        context.domain_min,
        context.domain_max,
        max(int(grid_size), int(degree) + 2),
        dtype=np.float64,
    )
    return np.unique(np.concatenate([positive, context.normalized_singular_values]))


def _bounded_target(
    points: np.ndarray,
    *,
    context: ApproximationContext,
    alpha: float,
) -> np.ndarray:
    values = regularized_filter_on_normalized_domain(
        np.asarray(points, dtype=np.float64),
        alpha=float(alpha),
        block_encoding_normalization=context.beta,
    )
    scale = max(1.0, float(np.max(np.abs(values))))
    return values / scale


def _coefficient_diagnostics(coefficients: np.ndarray) -> dict[str, float]:
    values = np.asarray(coefficients, dtype=np.float64)
    abs_values = np.abs(values)
    nonzero = abs_values[abs_values > 0.0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else np.nan
    max_abs = float(np.max(abs_values)) if abs_values.size else np.nan
    dynamic = float(max_abs / min_nonzero) if min_nonzero and np.isfinite(min_nonzero) else np.nan
    return {
        "max_abs_coefficient": max_abs,
        "min_abs_nonzero_coefficient": min_nonzero,
        "coefficient_dynamic_range": dynamic,
    }


def _chebyshev_coefficient_rows(candidate: CandidatePolynomial) -> list[dict[str, Any]]:
    return [
        {
            "candidate_base_name": candidate.candidate_base_name,
            "alpha": float(candidate.alpha),
            "degree": int(candidate.degree),
            "method": candidate.method,
            "lambda_if_any": candidate.lambda_if_any,
            "coefficient_index": int(index),
            "chebyshev_coefficient": float(value),
        }
        for index, value in enumerate(candidate.chebyshev_coefficients)
    ]


def _monomial_coefficient_rows(
    candidate_name: str,
    conversion_method: str,
    coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_name": candidate_name,
            "conversion_method": conversion_method,
            "coefficient_index": int(index),
            "monomial_coefficient": float(value),
        }
        for index, value in enumerate(coefficients)
    ]


def _error_grid_rows(
    *,
    candidate_name: str,
    positive_grid: np.ndarray,
    target: np.ndarray,
    native_values: np.ndarray,
    converted_values: np.ndarray,
) -> list[dict[str, Any]]:
    native_error = np.abs(native_values - target)
    converted_error = np.abs(converted_values - target)
    return [
        {
            "candidate_name": candidate_name,
            "sigma_normalized": float(sigma),
            "target_value": float(target_value),
            "native_value": float(native_value),
            "converted_value": float(converted_value),
            "native_abs_error": float(n_error),
            "converted_abs_error": float(c_error),
        }
        for sigma, target_value, native_value, converted_value, n_error, c_error in zip(
            positive_grid,
            target,
            native_values,
            converted_values,
            native_error,
            converted_error,
            strict=True,
        )
    ]


def _boundedness_grid_rows(
    *,
    candidate_name: str,
    unit_grid: np.ndarray,
    native_values: np.ndarray,
    converted_values: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_name": candidate_name,
            "x": float(x),
            "native_value": float(native_value),
            "converted_value": float(converted_value),
            "native_abs_value": float(abs(native_value)),
            "converted_abs_value": float(abs(converted_value)),
        }
        for x, native_value, converted_value in zip(
            unit_grid,
            native_values,
            converted_values,
            strict=True,
        )
    ]


def _odd_indices(degree: int) -> list[int]:
    return [index for index in range(int(degree) + 1) if index % 2 == 1]


def _as_odd_degree(degree: int) -> int:
    value = int(degree)
    return value if value % 2 == 1 else value + 1


def _odd_padded(values: np.ndarray, degree: int) -> np.ndarray:
    coefficients = np.zeros(int(degree) + 1, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    coefficients[: min(coefficients.size, source.size)] = source[: coefficients.size]
    coefficients[::2] = 0.0
    return coefficients


def _max_even_coefficient_abs(coefficients: np.ndarray) -> float:
    values = np.asarray(coefficients, dtype=np.float64)
    return float(np.max(np.abs(values[::2]))) if values.size else 0.0


def _conversion_methods(config: StableCandidateConfig) -> list[str]:
    methods = ["float64", "longdouble_recurrence"]
    if config.include_decimal_conversion:
        methods.append("decimal_recurrence")
    return methods


def _recommended_next_step(safe: bool, reason: str) -> str:
    if safe:
        return "Attempt phase synthesis with the audited monomial phase backend."
    if "native approximation error" in reason:
        return "Do not synthesize phases; improve the native bounded approximant first."
    if "dynamic range" in reason or "conversion" in reason:
        return "Do not synthesize phases; use a Chebyshev-basis backend or better conditioning."
    if "bounded" in reason:
        return "Do not synthesize phases; enforce boundedness on [-1, 1]."
    return "Do not synthesize phases until all safety gates pass."


def _candidate_report(summary: pd.DataFrame, config: StableCandidateConfig) -> str:
    safe_count = int(summary["safe_for_phase_synthesis"].sum()) if not summary.empty else 0
    best = (
        summary.sort_values("native_max_error", na_position="last").head(8)
        if not summary.empty
        else pd.DataFrame()
    )
    lines = [
        "# Stable Polynomial Candidate Diagnostics",
        "",
        "## Verdict",
        "",
        (
            f"Safe candidates for phase synthesis: `{safe_count}`. A row is safe only "
            "if every configured approximation, boundedness, parity, conversion, and "
            "coefficient gate passes."
        ),
        "",
        "## Safety Thresholds",
        "",
        f"- native target tolerance: `{TARGET_TOLERANCE}`",
        f"- boundedness tolerance: `{config.bound_tolerance}`",
        f"- parity tolerance: `{config.parity_tolerance}`",
        f"- conversion error tolerance: `{config.conversion_error_tolerance}`",
        f"- coefficient dynamic range limit: `{config.coefficient_dynamic_range_limit}`",
        "",
        "## Best Native Approximation Rows",
        "",
        "| candidate | degree | method | native error | dynamic range | safe | reason |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            "| "
            f"{row.candidate_name} | {row.degree} | {row.method} | "
            f"{row.native_max_error:.6g} | {row.coefficient_dynamic_range:.6g} | "
            f"{row.safe_for_phase_synthesis} | {row.failure_reason_if_any} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Rows with accurate native Chebyshev approximation but unstable "
                "monomial conversion are intentionally not passed to phase synthesis."
            ),
            "",
            "## Claim Boundary",
            "",
            CANDIDATE_CAVEAT,
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_stable_phase_candidates",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "degrees": [35, 51, 71, 101, 151, 201],
        "lambdas": [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4],
        "approximation_grid_size": 513,
        "boundedness_grid_size": 2049,
        "conversion_grid_size": 2049,
        "bound_tolerance": 1.0e-5,
        "parity_tolerance": 1.0e-10,
        "conversion_error_tolerance": 1.0e-5,
        "coefficient_dynamic_range_limit": 1.0e12,
        "include_decimal_conversion": True,
        "decimal_precision": 80,
    }
    if config:
        resolved.update(config)
    return resolved


def _config_from_dict(config: dict[str, Any]) -> StableCandidateConfig:
    return StableCandidateConfig(
        output_dir=str(config["output_dir"]),
        matrix_source=str(config["matrix_source"]),
        case_name=str(config["case_name"]),
        case_source=str(config["case_source"]),
        seed=int(config["seed"]),
        fallback_to_synthetic=bool(config["fallback_to_synthetic"]),
        alpha=float(config["alpha"]),
        degrees=tuple(int(value) for value in config["degrees"]),
        lambdas=tuple(float(value) for value in config["lambdas"]),
        approximation_grid_size=int(config["approximation_grid_size"]),
        boundedness_grid_size=int(config["boundedness_grid_size"]),
        conversion_grid_size=int(config["conversion_grid_size"]),
        bound_tolerance=float(config["bound_tolerance"]),
        parity_tolerance=float(config["parity_tolerance"]),
        conversion_error_tolerance=float(config["conversion_error_tolerance"]),
        coefficient_dynamic_range_limit=float(config["coefficient_dynamic_range_limit"]),
        include_decimal_conversion=bool(config["include_decimal_conversion"]),
        decimal_precision=int(config["decimal_precision"]),
    )


def _safe_slug(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))[:40]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build stable QSVT phase candidates")
    parser.parse_args(argv)
    run = build_stable_phase_candidates()
    print(f"QSVT stable phase candidates complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
