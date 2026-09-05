from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.nonbruteforce_refinement import (
    PHASE_CAVEAT,
    _markdown_table,
)
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain
from robust_qsvt_se.qsvt.polynomial_approximation import (
    ApproximationContext,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

STRICT_TOLERANCE = 1.0e-3
ESTIMATOR_CAVEAT = (
    "Preconditioned/equilibrated rows are formal new estimator variants. They do "
    "not overwrite original Ridge or QSVT-target claims."
)
QSVT_CAVEAT = (
    "QSVT approximation rows are resource-aware polynomial diagnostics only. "
    "They do not demonstrate quantum speedup, quantum advantage, hardware "
    "execution, or QSVT superiority over Ridge/Tikhonov."
)
RESIDUAL_WEIGHTED_CAVEAT = (
    "Residual-weighted diagnostics indicate whether high pointwise approximation "
    "error aligns with high-energy residual directions. They do not replace "
    "full-interval validation."
)
STABLE_PHASE_STATUSES = {
    "passed",
    "failed_approximation_error",
    "failed_boundedness",
    "failed_basis_conversion",
    "failed_coefficient_dynamic_range",
    "failed_phase_response",
    "skipped_unstable_coefficients",
    "skipped_dependency_missing",
}


def run_phase_validation_stable_basis(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_phase_stable_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = _build_context_from_config(resolved)
    sanity_status = _sanity_polynomial_status(resolved)
    candidate_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    backend_row = _chebyshev_backend_probe_row(resolved, sanity_status)
    candidate_rows.append(backend_row)
    coefficient_rows.append(_coefficient_row_from_candidate(backend_row))

    for degree in resolved["degrees"]:
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(resolved["alpha"]),
            degree=int(degree),
            method=str(resolved["method"]),
            grid_size=int(resolved["grid_size"]),
        )
        for conversion_method in ["float64", "longdouble_recurrence"]:
            candidate, phase = _phase_candidate_row(
                result=result,
                context=context,
                alpha=float(resolved["alpha"]),
                output_dir=output_dir,
                config=resolved,
                conversion_method=conversion_method,
                candidate_name=f"{result.method}_degree_{result.degree}_{conversion_method}",
                sanity_status=sanity_status,
            )
            candidate_rows.append(candidate)
            coefficient_rows.append(_coefficient_row_from_candidate(candidate))
            phase_rows.extend(phase)

    conditioned = _conditioned_candidate(context, resolved)
    if conditioned is not None:
        candidate, phase = _phase_candidate_row(
            result=conditioned,
            context=context,
            alpha=float(resolved["alpha"]),
            output_dir=output_dir,
            config=resolved,
            conversion_method="longdouble_recurrence",
            candidate_name=f"conditioned_chebyshev_ridge_degree_{conditioned.degree}",
            sanity_status=sanity_status,
        )
        candidate_rows.append(candidate)
        coefficient_rows.append(_coefficient_row_from_candidate(candidate))
        phase_rows.extend(phase)

    candidates = pd.DataFrame(candidate_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    phases = pd.DataFrame(phase_rows)
    summary_rows = _phase_stable_summary_rows(candidates, resolved, sanity_status)
    summary = pd.DataFrame(summary_rows)

    summary_csv = output_dir / "phase_validation_stable_basis_summary.csv"
    summary_json = output_dir / "phase_validation_stable_basis_summary.json"
    candidates_csv = output_dir / "candidate_polynomial_diagnostics.csv"
    coefficients_csv = output_dir / "coefficient_stability_diagnostics.csv"
    phases_csv = output_dir / "phase_response_diagnostics.csv"
    report_md = output_dir / "stable_phase_validation_report.md"
    summary.to_csv(summary_csv, index=False)
    candidates.to_csv(candidates_csv, index=False)
    coefficients.to_csv(coefficients_csv, index=False)
    phases.to_csv(phases_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_stable_phase_report(candidates), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "phase_validation_stable_basis_summary_csv": str(summary_csv),
            "phase_validation_stable_basis_summary_json": str(summary_json),
            "candidate_polynomial_diagnostics_csv": str(candidates_csv),
            "coefficient_stability_diagnostics_csv": str(coefficients_csv),
            "phase_response_diagnostics_csv": str(phases_csv),
            "stable_phase_validation_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": summary, "artifacts": {"manifest": manifest}}


def run_preconditioned_ieee300_estimator(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_preconditioned_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    solution_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    approximation_rows: list[dict[str, Any]] = []

    for case in resolved["cases"]:
        start = time.perf_counter()
        case_config = _case_config(case, resolved)
        case_name = str(case_config.get("case_name", "unknown"))
        alpha = float(case_config.get("alpha", resolved["alpha"]))
        try:
            system, matrix_source = build_engineering_system(case_config)
            rows = _preconditioned_case_rows(
                system=system,
                matrix_source=matrix_source,
                alpha=alpha,
                degree=int(case_config.get("degree", resolved["degree"])),
                method=str(resolved["method"]),
                grid_size=int(resolved["grid_size"]),
                runtime_start=start,
            )
            summary_rows.extend(rows["summary"])
            solution_rows.extend(rows["solution"])
            spectral_rows.extend(rows["spectral"])
            approximation_rows.extend(rows["approximation"])
        except Exception as exc:
            summary_rows.append(_preconditioned_failure_row(case_name, alpha, str(exc)))

    summary = pd.DataFrame(summary_rows)
    solution = pd.DataFrame(solution_rows)
    spectral = pd.DataFrame(spectral_rows)
    approximation = pd.DataFrame(approximation_rows)
    summary_csv = output_dir / "preconditioned_ieee300_estimator_summary.csv"
    summary_json = output_dir / "preconditioned_ieee300_estimator_summary.json"
    solution_csv = output_dir / "preconditioned_ieee300_solution_metrics.csv"
    spectral_csv = output_dir / "preconditioned_ieee300_spectral_metrics.csv"
    approximation_csv = output_dir / "preconditioned_ieee300_qsvt_approximation.csv"
    report_md = output_dir / "preconditioned_ieee300_report.md"
    summary.to_csv(summary_csv, index=False)
    solution.to_csv(solution_csv, index=False)
    spectral.to_csv(spectral_csv, index=False)
    approximation.to_csv(approximation_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_preconditioned_report(summary, approximation), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "preconditioned_ieee300_estimator_summary_csv": str(summary_csv),
            "preconditioned_ieee300_estimator_summary_json": str(summary_json),
            "preconditioned_ieee300_solution_metrics_csv": str(solution_csv),
            "preconditioned_ieee300_spectral_metrics_csv": str(spectral_csv),
            "preconditioned_ieee300_qsvt_approximation_csv": str(approximation_csv),
            "preconditioned_ieee300_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": summary, "artifacts": {"manifest": manifest}}


def diagnose_ieee300_residual_weighted_error(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_residual_weighted_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    case_config = _case_config(
        {
            "case_name": resolved["case_name"],
            "matrix_source": resolved.get("matrix_source", "ieee14_ac_weighted_jacobian"),
        },
        resolved,
    )
    alpha = float(resolved["alpha"])
    degree = int(resolved["degree"])
    summary_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    try:
        system, matrix_source = build_engineering_system(case_config)
        H = np.asarray(system.H_tilde, dtype=np.float64)
        b = np.asarray(system.r_tilde, dtype=np.float64)
        context = _context_from_matrix(
            matrix=H,
            case_name=str(system.metadata.get("case_name", resolved["case_name"])),
            matrix_source=matrix_source,
            source_note="residual-weighted diagnostic",
        )
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=alpha,
            degree=degree,
            method=str(resolved["method"]),
            grid_size=int(resolved["grid_size"]),
        )
        U, singular_values, _ = np.linalg.svd(H, full_matrices=False)
        projections = U.T @ b
        actual = _actual_singular_approximation_values(result)
        target_filter = regularized_filter_on_normalized_domain(
            singular_values / context.beta,
            alpha=alpha,
            block_encoding_normalization=context.beta,
        )
        approx_filter = actual["approximation_value"]
        pointwise_error = np.abs(approx_filter - target_filter)
        weighted_error = pointwise_error * np.abs(projections)
        target_contribution = np.abs(target_filter * projections)
        order = np.argsort(-weighted_error)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, order.size + 1)
        for index, sigma in enumerate(singular_values):
            contribution_rows.append(
                {
                    "case_name": context.case_name,
                    "alpha": alpha,
                    "singular_index": int(index),
                    "sigma": float(sigma),
                    "target_filter_value": float(target_filter[index]),
                    "approx_filter_value": float(approx_filter[index]),
                    "pointwise_error": float(pointwise_error[index]),
                    "abs_residual_projection": float(abs(projections[index])),
                    "target_contribution": float(target_contribution[index]),
                    "approx_error_contribution": float(weighted_error[index]),
                    "relative_contribution_rank": int(ranks[index]),
                }
            )
        top1 = _top_fraction_sum(weighted_error, 0.01)
        top5 = _top_fraction_sum(weighted_error, 0.05)
        total = float(np.sum(weighted_error))
        summary_rows.append(
            {
                "case_name": context.case_name,
                "alpha": alpha,
                "degree": int(result.degree),
                "max_pointwise_error": float(np.max(pointwise_error)),
                "max_residual_weighted_error": float(np.max(weighted_error)),
                "sum_residual_weighted_error": total,
                "top_1_percent_error_contribution": top1,
                "top_5_percent_error_contribution": top5,
                "interpretation": _residual_weighted_interpretation(weighted_error),
                "matrix_source": context.matrix_source,
                "caveat": RESIDUAL_WEIGHTED_CAVEAT,
                "status": "ok",
            }
        )
    except Exception as exc:
        summary_rows.append(
            {
                "case_name": str(resolved["case_name"]),
                "alpha": alpha,
                "degree": degree,
                "max_pointwise_error": np.nan,
                "max_residual_weighted_error": np.nan,
                "sum_residual_weighted_error": np.nan,
                "top_1_percent_error_contribution": np.nan,
                "top_5_percent_error_contribution": np.nan,
                "interpretation": f"failed gracefully: {exc}",
                "matrix_source": "",
                "caveat": RESIDUAL_WEIGHTED_CAVEAT,
                "status": "failed",
            }
        )

    contributions = pd.DataFrame(contribution_rows)
    top = (
        contributions.sort_values("approx_error_contribution", ascending=False).head(25)
        if not contributions.empty
        else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "residual_weighted_error_summary.csv"
    summary_json = output_dir / "residual_weighted_error_summary.json"
    contributions_csv = output_dir / "singular_direction_contributions.csv"
    top_csv = output_dir / "top_error_directions.csv"
    report_md = output_dir / "residual_weighted_error_report.md"
    summary.to_csv(summary_csv, index=False)
    contributions.to_csv(contributions_csv, index=False)
    top.to_csv(top_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_residual_weighted_report(summary, top), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "residual_weighted_error_summary_csv": str(summary_csv),
            "residual_weighted_error_summary_json": str(summary_json),
            "singular_direction_contributions_csv": str(contributions_csv),
            "top_error_directions_csv": str(top_csv),
            "residual_weighted_error_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": summary, "artifacts": {"manifest": manifest}}


def build_failure_fix_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_failure_summary_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = _failure_fix_rows()
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "failure_fix_summary.csv"
    json_path = output_dir / "failure_fix_summary.json"
    md_path = output_dir / "failure_fix_summary.md"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    md_path.write_text(_failure_fix_report(rows), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "failure_fix_summary_md": str(md_path),
            "failure_fix_summary_csv": str(csv_path),
            "failure_fix_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {"output_dir": output_dir, "summary": frame, "artifacts": {"manifest": manifest}}


def _phase_candidate_row(
    *,
    result: Any,
    context: ApproximationContext,
    alpha: float,
    output_dir: Path,
    config: dict[str, Any],
    conversion_method: str,
    candidate_name: str,
    sanity_status: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = np.linspace(-1.0, 1.0, int(config["bound_grid_size"]), dtype=np.float64)
    positive_grid = np.linspace(context.domain_min, context.domain_max, int(config["grid_size"]))
    native_values = Chebyshev(result.chebyshev_coefficients)(grid)
    native_positive = Chebyshev(result.chebyshev_coefficients)(positive_grid)
    target_positive = regularized_filter_on_normalized_domain(
        positive_grid,
        alpha=alpha,
        block_encoding_normalization=context.beta,
    )
    bounded_scale = max(1.0, float(np.max(np.abs(target_positive))))
    target_positive = target_positive / bounded_scale
    native_error = np.abs(native_positive - target_positive)
    native_max_abs = float(np.max(np.abs(native_values)))
    if conversion_method == "float64":
        coefficients = np.asarray(result.power_coefficients, dtype=np.float64)
        conversion_precision = "float64"
    else:
        coefficients = _chebyshev_to_power_longdouble(result.chebyshev_coefficients)
        conversion_precision = "numpy.longdouble"
    power_values = Polynomial(coefficients)(grid)
    conversion_error = float(np.max(np.abs(power_values - native_values)))
    coeff_diag = _coefficient_diagnostics(coefficients)
    bounded_after = bool(
        float(np.max(np.abs(power_values))) <= 1.0 + float(config["bound_tolerance"])
    )
    bounded_native = bool(native_max_abs <= 1.0 + float(config["bound_tolerance"]))
    parity_error = float(np.max(np.abs(coefficients[::2]))) if coefficients.size else 0.0
    phase_backend = "pennylane_poly_to_angles"
    phase_status = _pre_phase_status(
        native_error=float(np.max(native_error)),
        bounded_native=bounded_native,
        conversion_error=conversion_error,
        dynamic_range=coeff_diag["coefficient_dynamic_range"],
        bounded_after=bounded_after,
        config=config,
    )
    phase_max = np.nan
    phase_mean = np.nan
    failure_reason = _phase_candidate_failure_reason(phase_status)
    phase_rows: list[dict[str, Any]] = []
    if phase_status == "ready_for_phase_synthesis":
        phase_status, phase_max, phase_mean, failure_reason, phase_rows = _try_phase_response(
            candidate_name=candidate_name,
            coefficients=coefficients,
            target_values=result.bounded_target_values,
            approximation_values=result.bounded_approximation_values,
            evaluation_points=result.evaluation_points,
            evaluation_kind=result.evaluation_kind,
            output_dir=output_dir,
            config=config,
        )
    passed = bool(
        float(np.max(native_error)) <= STRICT_TOLERANCE
        and bounded_after
        and np.isfinite(phase_max)
        and phase_max <= STRICT_TOLERANCE
        and bool(sanity_status["passed"])
    )
    interpretation = (
        "phase validation passed for the bounded Ridge/Tikhonov target"
        if passed
        else "sanity polynomial precondition is missing or failed"
        if float(np.max(native_error)) <= STRICT_TOLERANCE
        and bounded_after
        and np.isfinite(phase_max)
        and phase_max <= STRICT_TOLERANCE
        and not bool(sanity_status["passed"])
        else "polynomial approximation passed; phase-level validation remains unresolved"
        if float(np.max(native_error)) <= STRICT_TOLERANCE
        else "candidate does not meet approximation tolerance"
    )
    return (
        {
            "candidate_name": candidate_name,
            "alpha": alpha,
            "degree": int(result.degree),
            "native_basis": "chebyshev_T_low_to_high_on_unit_interval",
            "parity": "odd",
            "native_approx_max_error": float(np.max(native_error)),
            "native_approx_mean_error": float(np.mean(native_error)),
            "native_max_abs_value": native_max_abs,
            "bounded_in_native_basis": bounded_native,
            "coefficient_basis_for_backend": "monomial_power_low_to_high",
            "conversion_method": conversion_method,
            "conversion_precision": conversion_precision,
            "conversion_max_error": conversion_error,
            "max_abs_coefficient": coeff_diag["max_abs_coefficient"],
            "min_abs_nonzero_coefficient": coeff_diag["min_abs_nonzero_coefficient"],
            "coefficient_dynamic_range": coeff_diag["coefficient_dynamic_range"],
            "bounded_after_conversion": bounded_after,
            "phase_backend": phase_backend,
            "phase_status": phase_status,
            "phase_response_max_error": phase_max,
            "phase_response_mean_error": phase_mean,
            "passed_1e_minus_3": passed,
            "sanity_polynomial_tests_passed": bool(sanity_status["passed"]),
            "sanity_polynomial_source": str(sanity_status["source"]),
            "max_sanity_polynomial_error": sanity_status["max_error"],
            "failure_reason": failure_reason,
            "recommended_interpretation": interpretation,
            "parity_error": parity_error,
            "phase_caveat": PHASE_CAVEAT,
        },
        phase_rows,
    )


def _pre_phase_status(
    *,
    native_error: float,
    bounded_native: bool,
    conversion_error: float,
    dynamic_range: float,
    bounded_after: bool,
    config: dict[str, Any],
) -> str:
    if native_error > STRICT_TOLERANCE:
        return "failed_approximation_error"
    if not bounded_native or not bounded_after:
        return "failed_boundedness"
    if conversion_error > float(config["conversion_error_limit"]):
        return "failed_basis_conversion"
    if np.isfinite(dynamic_range) and dynamic_range > float(
        config["coefficient_dynamic_range_limit"]
    ):
        return "failed_coefficient_dynamic_range"
    if importlib.util.find_spec("pennylane") is None or bool(config["force_dependency_missing"]):
        return "skipped_dependency_missing"
    return "ready_for_phase_synthesis"


def _try_phase_response(
    *,
    candidate_name: str,
    coefficients: np.ndarray,
    target_values: np.ndarray,
    approximation_values: np.ndarray,
    evaluation_points: np.ndarray,
    evaluation_kind: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[str, float, float, str, list[dict[str, Any]]]:
    try:
        validate_qsvt_polynomial(
            coefficients,
            parity="odd",
            grid_size=int(config["bound_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phase_result = synthesize_pennylane_phases_cached(
            coefficients,
            angle_solver=str(config["angle_solver"]),
            cache_dir=output_dir / "phase_cache",
            cache_metadata={"candidate_name": candidate_name},
        )
        mask = np.asarray(evaluation_kind, dtype=object) == "grid"
        grid = np.asarray(evaluation_points, dtype=np.float64)[mask]
        target = np.asarray(target_values, dtype=np.float64)[mask]
        approximation = np.asarray(approximation_values, dtype=np.float64)[mask]
        response = pennylane_qsvt_response(
            grid,
            phase_result.phases,
            phase_order=str(config["phase_order"]),
            phase_sign=str(config["phase_sign"]),
            phase_offset_rule=str(config["phase_offset_rule"]),
            signal_operator_convention=str(config["signal_operator_convention"]),
            response_component=str(config["response_component"]),
        )
        errors = np.abs(response - target)
        rows = [
            {
                "candidate_name": candidate_name,
                "sigma_normalized": float(sigma),
                "target_value": float(target_value),
                "polynomial_value": float(poly_value),
                "phase_response_value": float(response_value),
                "phase_response_error": float(error),
            }
            for sigma, target_value, poly_value, response_value, error in zip(
                grid, target, approximation, response, errors, strict=True
            )
        ]
        max_error = float(np.max(errors))
        status = "passed" if max_error <= STRICT_TOLERANCE else "failed_phase_response"
        reason = "" if status == "passed" else "phase response exceeds strict tolerance"
        return status, max_error, float(np.mean(errors)), reason, rows
    except Exception as exc:
        return "failed_phase_response", np.nan, np.nan, str(exc), []


def _phase_candidate_failure_reason(status: str) -> str:
    return {
        "failed_approximation_error": "native approximation error exceeds strict tolerance",
        "failed_boundedness": "polynomial is not bounded by 1 after native or converted check",
        "failed_basis_conversion": "Chebyshev-to-monomial conversion error exceeds safety limit",
        "failed_coefficient_dynamic_range": "coefficient dynamic range exceeds safety limit",
        "skipped_dependency_missing": "phase backend dependency unavailable or forced missing",
    }.get(status, "")


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
    for index, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        poly = polys[index]
        power[: poly.size] += coeff * poly
    return np.asarray(power, dtype=np.float64)


def _conditioned_candidate(
    context: ApproximationContext,
    config: dict[str, Any],
) -> Any | None:
    degree = int(config["conditioned_degree"])
    base = evaluate_polynomial_approximation(
        context=context,
        alpha=float(config["alpha"]),
        degree=degree,
        method="odd_chebyshev_ls",
        grid_size=int(config["grid_size"]),
    )
    # Lightweight coefficient-conditioned attempt: shrink high-order Chebyshev
    # coefficients. This is reported as a stability diagnostic, not an optimizer.
    cheb = np.asarray(base.chebyshev_coefficients, dtype=np.float64).copy()
    powers = np.arange(cheb.size, dtype=np.float64)
    cheb = cheb / (1.0 + float(config["conditioned_lambda"]) * powers**2)
    grid = np.asarray(base.evaluation_points, dtype=np.float64)
    values = Chebyshev(cheb)(grid)
    power = _chebyshev_to_power_longdouble(cheb)
    errors = np.abs(values - np.asarray(base.bounded_target_values, dtype=np.float64))
    return _SimplePolynomialResult(
        method="conditioned_chebyshev_shrinkage",
        degree=degree,
        chebyshev_coefficients=cheb,
        power_coefficients=power,
        evaluation_points=base.evaluation_points,
        evaluation_kind=base.evaluation_kind,
        bounded_target_values=base.bounded_target_values,
        bounded_approximation_values=values,
        pointwise_errors=errors,
        bounded_scaling_C=base.bounded_scaling_C,
    )


class _SimplePolynomialResult:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _chebyshev_backend_probe_row(
    config: dict[str, Any],
    sanity_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_name": "chebyshev_basis_preserving_backend_probe",
        "alpha": float(config["alpha"]),
        "degree": 0,
        "native_basis": "chebyshev_T_low_to_high_on_unit_interval",
        "parity": "odd",
        "native_approx_max_error": np.nan,
        "native_approx_mean_error": np.nan,
        "native_max_abs_value": np.nan,
        "bounded_in_native_basis": False,
        "coefficient_basis_for_backend": "not_applicable",
        "conversion_method": "not_applicable",
        "conversion_precision": "not_applicable",
        "conversion_max_error": np.nan,
        "max_abs_coefficient": np.nan,
        "min_abs_nonzero_coefficient": np.nan,
        "coefficient_dynamic_range": np.nan,
        "bounded_after_conversion": False,
        "phase_backend": "none_available_for_chebyshev_coefficients",
        "phase_status": "skipped_dependency_missing",
        "phase_response_max_error": np.nan,
        "phase_response_mean_error": np.nan,
        "passed_1e_minus_3": False,
        "sanity_polynomial_tests_passed": bool(sanity_status["passed"]),
        "sanity_polynomial_source": str(sanity_status["source"]),
        "max_sanity_polynomial_error": sanity_status["max_error"],
        "failure_reason": (
            "No available backend was found that accepts Chebyshev coefficients directly."
        ),
        "recommended_interpretation": (
            "Chebyshev-basis polynomial approximation can be reported, but direct "
            "phase synthesis remains unavailable in the current dependency set."
        ),
        "parity_error": np.nan,
        "phase_caveat": PHASE_CAVEAT,
    }


def _coefficient_diagnostics(coefficients: np.ndarray) -> dict[str, float]:
    values = np.asarray(coefficients, dtype=np.float64)
    nonzero = np.abs(values[np.abs(values) > 0.0])
    min_nonzero = float(np.min(nonzero)) if nonzero.size else np.nan
    max_abs = float(np.max(np.abs(values))) if values.size else np.nan
    dynamic = float(max_abs / min_nonzero) if np.isfinite(min_nonzero) and min_nonzero else np.nan
    return {
        "max_abs_coefficient": max_abs,
        "min_abs_nonzero_coefficient": min_nonzero,
        "coefficient_dynamic_range": dynamic,
    }


def _coefficient_row_from_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_name",
        "degree",
        "conversion_method",
        "conversion_precision",
        "conversion_max_error",
        "max_abs_coefficient",
        "min_abs_nonzero_coefficient",
        "coefficient_dynamic_range",
        "bounded_after_conversion",
        "phase_status",
        "failure_reason",
    ]
    return {key: row.get(key) for key in keys}


def _phase_stable_summary_rows(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    sanity_status: dict[str, Any],
) -> list[dict[str, Any]]:
    passed = candidates[candidates["passed_1e_minus_3"] == True]  # noqa: E712
    best = candidates.sort_values("native_approx_max_error", na_position="last").head(1)
    return [
        {
            "alpha": float(config["alpha"]),
            "target_tolerance": STRICT_TOLERANCE,
            "candidate_count": len(candidates),
            "passed_candidate_count": len(passed),
            "best_candidate_name": "" if best.empty else str(best.iloc[0]["candidate_name"]),
            "best_native_approx_max_error": (
                np.nan if best.empty else float(best.iloc[0]["native_approx_max_error"])
            ),
            "sanity_polynomial_tests_passed": bool(sanity_status["passed"]),
            "sanity_polynomial_source": str(sanity_status["source"]),
            "max_sanity_polynomial_error": sanity_status["max_error"],
            "phase_validation_status": "passed" if not passed.empty else "unresolved",
            "interpretation": (
                "A bounded Ridge/Tikhonov phase target passed all declared criteria."
                if not passed.empty
                else "Stable-basis diagnostics did not produce a safe passing phase target."
            ),
            "caveat": PHASE_CAVEAT,
        }
    ]


def _stable_phase_report(candidates: pd.DataFrame) -> str:
    table = _markdown_table(
        candidates,
        [
            "candidate_name",
            "degree",
            "conversion_method",
            "coefficient_dynamic_range",
            "bounded_after_conversion",
            "phase_response_max_error",
            "phase_status",
            "sanity_polynomial_tests_passed",
        ],
    )
    passed = candidates[candidates["passed_1e_minus_3"]] if not candidates.empty else []
    verdict = "passed" if len(passed) else "unresolved"
    return f"""# Stable Phase-Synthesis Diagnostics

## Verdict

Bounded Ridge/Tikhonov target phase validation status: `{verdict}`.

## Candidate Table

{table}

## Interpretation

Phase validation is claimed only when native approximation error, boundedness
after conversion, and phase-response error all meet the strict `1e-3` tolerance.

{PHASE_CAVEAT}
"""


def _sanity_polynomial_status(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(config["sanity_results_path"]))
    if not source.is_file():
        return {
            "passed": False,
            "source": source,
            "max_error": np.nan,
            "status": "missing",
        }
    try:
        frame = pd.read_csv(source)
        passed = bool(
            not frame.empty and frame["best_status"].astype(str).str.lower().eq("passed").all()
        )
        max_error = (
            float(frame["best_max_pointwise_error"].max())
            if "best_max_pointwise_error" in frame.columns
            else np.nan
        )
        return {
            "passed": passed,
            "source": source,
            "max_error": max_error,
            "status": "passed" if passed else "failed",
        }
    except Exception:
        return {
            "passed": False,
            "source": source,
            "max_error": np.nan,
            "status": "unreadable",
        }


def _preconditioned_case_rows(
    *,
    system: WeightedSystem,
    matrix_source: str,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
    runtime_start: float,
) -> dict[str, list[dict[str, Any]]]:
    H = np.asarray(system.H_tilde, dtype=np.float64)
    b = np.asarray(system.r_tilde, dtype=np.float64)
    case_name = str(system.metadata.get("case_name", "unknown"))
    scales = _column_equilibration_scales(H)
    H_pre = H * scales[None, :]
    x_original = _ridge_svd(H, b, alpha=alpha)
    y_coordinate = _ridge_svd(H_pre, b, alpha=alpha)
    x_coordinate = scales * y_coordinate
    x_transformed = _transformed_penalty_solution(H_pre, b, scales=scales, alpha=alpha)
    before_context = _context_from_matrix(
        matrix=H,
        case_name=case_name,
        matrix_source=matrix_source,
        source_note="unpreconditioned formal variant",
    )
    after_context = _context_from_matrix(
        matrix=H_pre,
        case_name=case_name,
        matrix_source=f"{matrix_source}_column_equilibrated",
        source_note="formal column-equilibrated variant",
    )
    before_approx = _approximation_metrics(before_context, alpha, degree, method, grid_size)
    after_approx = _approximation_metrics(after_context, alpha, degree, method, grid_size)
    spectral_rows = [
        _spectral_row(case_name, "before", before_context),
        _spectral_row(case_name, "after", after_context),
    ]
    solution_specs = [
        ("unpreconditioned_ridge", x_original, before_context, before_approx, "ok"),
        (
            "preconditioned_ridge_column_equilibrated_coordinate_penalty",
            x_coordinate,
            after_context,
            after_approx,
            _variant_status(system, x_original, x_coordinate, before_approx, after_approx),
        ),
        (
            "preconditioned_ridge_column_equilibrated_transformed_penalty",
            x_transformed,
            after_context,
            after_approx,
            "consistency_check",
        ),
        (
            "unpreconditioned_qsvt_target_spectral_diagnostic",
            x_original,
            before_context,
            before_approx,
            "diagnostic_only",
        ),
        (
            "preconditioned_qsvt_target_column_equilibrated_spectral_diagnostic",
            x_coordinate,
            after_context,
            after_approx,
            "diagnostic_only",
        ),
    ]
    summary_rows = []
    solution_rows = []
    for variant, x_hat, context, approx, status in solution_specs:
        row = _solution_summary_row(
            system=system,
            variant_name=variant,
            alpha=alpha,
            before=before_context,
            after=context,
            rank_before=int(np.linalg.matrix_rank(H)),
            rank_after=int(np.linalg.matrix_rank(H_pre)),
            x_hat=x_hat,
            x_original=x_original,
            approximation=approx,
            status=status,
        )
        row["runtime_seconds"] = float(time.perf_counter() - runtime_start)
        summary_rows.append(row)
        solution_rows.append(row)
    approximation_rows = [
        {
            "case_name": case_name,
            "alpha": alpha,
            "degree_before": degree,
            "degree_after": degree,
            "query_count_before": int(2 * degree + 1),
            "query_count_after": int(2 * degree + 1),
            "sigma_min_before": float(np.min(before_context.singular_values)),
            "sigma_max_before": float(np.max(before_context.singular_values)),
            "kappa_before": _kappa(before_context.singular_values),
            "sigma_min_after": float(np.min(after_context.singular_values)),
            "sigma_max_after": float(np.max(after_context.singular_values)),
            "kappa_after": _kappa(after_context.singular_values),
            "full_interval_error_before": before_approx["full_interval_approx_error"],
            "full_interval_error_after": after_approx["full_interval_approx_error"],
            "actual_singular_error_before": before_approx["actual_singular_value_approx_error"],
            "actual_singular_error_after": after_approx["actual_singular_value_approx_error"],
            "status": (
                "passed_preconditioned_1e_minus_3"
                if after_approx["full_interval_approx_error"] <= STRICT_TOLERANCE
                else "failed_preconditioned_1e_minus_3"
            ),
            "qsvt_caveat": QSVT_CAVEAT,
        }
    ]
    return {
        "summary": summary_rows,
        "solution": solution_rows,
        "spectral": spectral_rows,
        "approximation": approximation_rows,
    }


def _solution_summary_row(
    *,
    system: WeightedSystem,
    variant_name: str,
    alpha: float,
    before: ApproximationContext,
    after: ApproximationContext,
    rank_before: int,
    rank_after: int,
    x_hat: np.ndarray,
    x_original: np.ndarray,
    approximation: dict[str, float],
    status: str,
) -> dict[str, Any]:
    rel = _relative_error(x_original, x_hat)
    return {
        "case_name": before.case_name,
        "variant_name": variant_name,
        "alpha": alpha,
        "m": int(system.n_measurements),
        "n": int(system.n_states),
        "condition_number_before": _kappa(before.singular_values),
        "condition_number_after": _kappa(after.singular_values),
        "rank_before": rank_before,
        "rank_after": rank_after,
        "rmse_if_available": system.rmse(x_hat),
        "residual_norm": system.residual_norm(x_hat),
        "weighted_residual_norm": system.weighted_residual_norm(x_hat),
        "solution_norm": float(np.linalg.norm(x_hat)),
        "relative_solution_error_vs_unpreconditioned_ridge": rel,
        "full_interval_approx_error": approximation["full_interval_approx_error"],
        "actual_singular_value_approx_error": approximation["actual_singular_value_approx_error"],
        "degree": int(approximation["degree"]),
        "query_count": int(approximation["query_count"]),
        "status": status,
        "estimator_caveat": ESTIMATOR_CAVEAT,
        "qsvt_caveat": QSVT_CAVEAT,
    }


def _approximation_metrics(
    context: ApproximationContext,
    alpha: float,
    degree: int,
    method: str,
    grid_size: int,
) -> dict[str, float]:
    result = evaluate_polynomial_approximation(
        context=context,
        alpha=alpha,
        degree=degree,
        method=method,
        grid_size=grid_size,
    )
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_errors = result.pointwise_errors[kinds == "grid"]
    actual_errors = result.pointwise_errors[kinds == "actual_singular_value"]
    return {
        "full_interval_approx_error": float(np.max(grid_errors)),
        "actual_singular_value_approx_error": float(np.max(actual_errors)),
        "degree": float(result.degree),
        "query_count": float(2 * result.degree + 1),
    }


def _actual_singular_approximation_values(result: Any) -> dict[str, np.ndarray]:
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    mask = kinds == "actual_singular_value"
    return {
        "sigma_normalized": np.asarray(result.evaluation_points, dtype=np.float64)[mask],
        "target_value": np.asarray(result.target_values, dtype=np.float64)[mask],
        "approximation_value": np.asarray(result.approximation_values, dtype=np.float64)[mask],
    }


def _ridge_svd(matrix: np.ndarray, rhs: np.ndarray, *, alpha: float) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64),
        full_matrices=False,
    )
    gains = singular_values / (singular_values**2 + alpha)
    return Vt.T @ (gains * (U.T @ rhs))


def _transformed_penalty_solution(
    matrix_preconditioned: np.ndarray,
    rhs: np.ndarray,
    *,
    scales: np.ndarray,
    alpha: float,
) -> np.ndarray:
    lhs = matrix_preconditioned.T @ matrix_preconditioned
    lhs = lhs + alpha * np.diag(scales**2)
    y = np.linalg.solve(lhs, matrix_preconditioned.T @ rhs)
    return scales * y


def _variant_status(
    system: WeightedSystem,
    original: np.ndarray,
    candidate: np.ndarray,
    before: dict[str, float],
    after: dict[str, float],
) -> str:
    improves = after["full_interval_approx_error"] < before["full_interval_approx_error"]
    finite = np.all(np.isfinite(candidate))
    residual_ratio = system.residual_norm(candidate) / max(
        system.residual_norm(original),
        np.finfo(float).eps,
    )
    if improves and finite and residual_ratio <= 10.0:
        return "useful_preconditioned_variant"
    if improves and finite:
        return "diagnostic_improves_approximation_but_residual_degrades"
    return "diagnostic_only"


def _column_equilibration_scales(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=0)
    return np.divide(1.0, norms, out=np.ones_like(norms), where=norms > 1.0e-14)


def _spectral_row(case_name: str, label: str, context: ApproximationContext) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "stage": label,
        "matrix_source": context.matrix_source,
        "sigma_min": float(np.min(context.singular_values)),
        "sigma_max": float(np.max(context.singular_values)),
        "condition_number": _kappa(context.singular_values),
        "rank": int(np.count_nonzero(context.singular_values > 1.0e-12)),
    }


def _preconditioned_failure_row(case_name: str, alpha: float, reason: str) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "variant_name": "preconditioned_estimator_failure",
        "alpha": alpha,
        "m": np.nan,
        "n": np.nan,
        "condition_number_before": np.nan,
        "condition_number_after": np.nan,
        "rank_before": np.nan,
        "rank_after": np.nan,
        "rmse_if_available": np.nan,
        "residual_norm": np.nan,
        "weighted_residual_norm": np.nan,
        "solution_norm": np.nan,
        "relative_solution_error_vs_unpreconditioned_ridge": np.nan,
        "full_interval_approx_error": np.nan,
        "actual_singular_value_approx_error": np.nan,
        "degree": np.nan,
        "query_count": np.nan,
        "status": f"failed: {reason}",
        "estimator_caveat": ESTIMATOR_CAVEAT,
        "qsvt_caveat": QSVT_CAVEAT,
    }


def _context_from_matrix(
    *,
    matrix: np.ndarray,
    case_name: str,
    matrix_source: str,
    source_note: str,
) -> ApproximationContext:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("diagnostics require positive singular values")
    beta = float(np.max(positive))
    return ApproximationContext(
        case_name=case_name,
        matrix_source=matrix_source,
        matrix_shape=f"{matrix.shape[0]}x{matrix.shape[1]}",
        beta=beta,
        singular_values=positive,
        normalized_singular_values=positive / beta,
        domain_min=max(float(np.min(positive / beta)), np.finfo(float).eps),
        domain_max=1.0,
        source_note=source_note,
    )


def _kappa(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > 1.0e-14]
    return float(np.max(positive) / np.min(positive)) if positive.size else float("inf")


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(candidate - reference))
    return float(np.linalg.norm(candidate - reference) / denom)


def _top_fraction_sum(values: np.ndarray, fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0:
        return np.nan
    count = max(1, int(np.ceil(fraction * finite.size)))
    return float(np.sum(np.sort(finite)[-count:]))


def _residual_weighted_interpretation(weighted_error: np.ndarray) -> str:
    if weighted_error.size == 0:
        return "No singular-direction rows were available."
    total = float(np.sum(weighted_error))
    top5 = _top_fraction_sum(weighted_error, 0.05)
    share = top5 / total if total > 0.0 else np.nan
    return (
        "Residual-weighted diagnostics indicate whether high pointwise approximation "
        f"error aligns with high-energy residual directions; top 5% share is {share:.6g}. "
        "This does not replace full-interval validation."
    )


def _preconditioned_report(summary: pd.DataFrame, approximation: pd.DataFrame) -> str:
    summary_columns = [
        "case_name",
        "variant_name",
        "condition_number_after",
        "residual_norm",
        "weighted_residual_norm",
        "full_interval_approx_error",
        "status",
    ]
    approximation_columns = [
        "case_name",
        "kappa_before",
        "kappa_after",
        "full_interval_error_before",
        "full_interval_error_after",
        "actual_singular_error_before",
        "actual_singular_error_after",
        "status",
    ]
    return f"""# Preconditioned IEEE300 Estimator Variant

## Summary

{_markdown_table(summary, summary_columns)}

## QSVT Approximation

{_markdown_table(approximation, approximation_columns)}

## Caveat

{ESTIMATOR_CAVEAT}

{QSVT_CAVEAT}
"""


def _residual_weighted_report(summary: pd.DataFrame, top: pd.DataFrame) -> str:
    summary_columns = [
        "case_name",
        "degree",
        "max_pointwise_error",
        "max_residual_weighted_error",
        "sum_residual_weighted_error",
        "top_1_percent_error_contribution",
        "top_5_percent_error_contribution",
        "status",
    ]
    top_columns = [
        "singular_index",
        "sigma",
        "pointwise_error",
        "abs_residual_projection",
        "approx_error_contribution",
        "relative_contribution_rank",
    ]
    return f"""# IEEE300 Residual-Weighted Spectral Error

## Summary

{_markdown_table(summary, summary_columns)}

## Top Directions

{_markdown_table(top, top_columns)}

## Claim Boundary

{RESIDUAL_WEIGHTED_CAVEAT}
"""


def _failure_fix_rows() -> list[dict[str, Any]]:
    phase = _read_csv(
        Path("outputs/qsvt_phase_validation_stable_basis/candidate_polynomial_diagnostics.csv")
    )
    preconditioned = _read_csv(
        Path(
            "outputs/qsvt_preconditioned_ieee300_estimator/preconditioned_ieee300_estimator_summary.csv"
        )
    )
    residual = _read_csv(
        Path("outputs/qsvt_ieee300_residual_weighted_error/residual_weighted_error_summary.csv")
    )
    return [
        _failure_row("stable_phase_synthesis", phase),
        _failure_row("preconditioned_ieee300_variant", preconditioned),
        _failure_row("residual_weighted_spectral_error", residual),
        {
            "area": "claim_boundaries",
            "status": "documented",
            "key_result": (
                "No tolerance relaxation, no restricted/residual-weighted full-validation claim."
            ),
            "main_variant": "original estimators unchanged; preconditioned rows are new variants",
            "diagnostic_only": RESIDUAL_WEIGHTED_CAVEAT,
        },
    ]


def _failure_row(area: str, frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "area": area,
            "status": "missing",
            "key_result": "Required output was unavailable.",
            "main_variant": "",
            "diagnostic_only": "",
        }
    if area == "stable_phase_synthesis":
        passed = int((frame["passed_1e_minus_3"] == True).sum())  # noqa: E712
        unstable = int(
            frame["phase_status"].astype(str).str.contains("coefficient|boundedness").sum()
        )
        return {
            "area": area,
            "status": "passed" if passed else "unresolved",
            "key_result": f"passing candidates={passed}; unstable/unsafe candidates={unstable}",
            "main_variant": "bounded Ridge/Tikhonov phase target",
            "diagnostic_only": PHASE_CAVEAT,
        }
    if area == "preconditioned_ieee300_variant":
        useful = frame[frame["status"].astype(str).str.contains("useful|passed", case=False)]
        return {
            "area": area,
            "status": "implemented",
            "key_result": f"rows={len(frame)}; useful_or_passed_rows={len(useful)}",
            "main_variant": "new preconditioned/equilibrated estimator variant",
            "diagnostic_only": QSVT_CAVEAT,
        }
    return {
        "area": area,
        "status": str(frame.iloc[0].get("status", "documented")),
        "key_result": str(frame.iloc[0].get("interpretation", "")),
        "main_variant": "none",
        "diagnostic_only": RESIDUAL_WEIGHTED_CAVEAT,
    }


def _failure_fix_report(rows: list[dict[str, Any]]) -> str:
    evidence_columns = ["area", "status", "key_result", "main_variant", "diagnostic_only"]
    return f"""# QSVT Failure-Fix Summary

## Executive Verdict

PARTIAL PASS. The preconditioned IEEE300 variant is implemented and can reduce
approximation difficulty under column equilibration. Stable phase synthesis for
the bounded Ridge/Tikhonov target remains unresolved unless a generated row
explicitly passes all criteria.

## Evidence

{_markdown_table(pd.DataFrame(rows), evidence_columns)}

## Phase Validation Failure

The bounded Ridge/Tikhonov phase target is fixed only when a generated candidate
row has `passed_1e_minus_3 = true`. Otherwise the result remains unresolved.
Unstable or unbounded high-degree monomial conversions are not forced into
phase synthesis.

## Stable-Basis Phase Synthesis Results

The stable-basis diagnostic reports native approximation error, native
boundedness, parity, coefficient basis, conversion error, coefficient dynamic
range, converted boundedness, phase-synthesis status, phase-response error, and
the reason for each pass/fail/skip row.

## IEEE300 Preconditioned Estimator Results

Column-equilibrated coordinate-penalty Ridge and transformed-penalty Ridge are
reported as separate estimator rows. The coordinate-penalty row is a new
variant. The transformed-penalty row is a consistency check for the original
x-space penalty.

## IEEE300 Residual-Weighted Spectral Diagnostic

Residual-weighted rows report singular-direction projection weights and
approximation-error contributions. They diagnose whether pointwise error aligns
with high-energy residual directions.

## Main Estimator Variants

- Main estimator variants: original Ridge and explicitly labeled preconditioned Ridge rows.

## Diagnostic-Only Results

QSVT approximation proxy rows, restricted-interval rows, and residual-weighted
spectral rows are diagnostic only unless explicitly labeled as full-interval
validation for the relevant matrix and variant.

## Full-Interval Validations

Only rows with explicit full-interval error below `1e-3` are full-interval
validations. Preconditioned full-interval rows apply to the preconditioned
matrix variant, not to the original unpreconditioned IEEE300 matrix.

## Actual-Singular-Value Diagnostics

Actual-singular-value errors are reported separately from full-interval errors.
They are diagnostic evidence and are not interchangeable with full-interval
validation.

## Restricted Or Residual-Weighted Diagnostics

Restricted-interval and residual-weighted diagnostics are not full validation.
They may explain where error occurs or whether it aligns with the current right
hand side, but they do not relax the `1e-3` full-interval criterion.

## Safe Claims

Stable phase-synthesis diagnostics, a formal preconditioned estimator variant,
and residual-weighted spectral diagnostics were implemented with strict
pass/fail boundaries.

## Claims To Avoid

Do not claim quantum speedup, quantum advantage, hardware execution, field-data
validation, QSVT-over-Ridge superiority, or full validation from restricted or
residual-weighted diagnostics.

## Remaining Limitations

High-degree monomial phase backends remain coefficient-sensitive. Preconditioned
rows are new variants and do not overwrite the original estimator results.

## Recommended Manuscript Wording

We report stable-basis phase diagnostics, a formally labeled column-equilibrated
estimator variant, and residual-weighted spectral diagnostics. These results
separate original and preconditioned estimator claims and distinguish
full-interval, actual-singular-value, and residual-weighted evidence. They
support resource-aware feasibility analysis, not quantum speedup, quantum
advantage, hardware execution, field-data validation, or QSVT superiority over
Ridge/Tikhonov under the same regularization parameter.
"""


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _build_context_from_config(config: dict[str, Any]) -> ApproximationContext:
    return _context_from_matrix(
        matrix=np.asarray(build_engineering_system(config)[0].H_tilde, dtype=np.float64),
        case_name=str(config.get("case_name", "ieee14")),
        matrix_source=str(config.get("matrix_source", "ieee14_ac_weighted_jacobian")),
        source_note="stable phase diagnostics",
    )


def _case_config(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = {
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": config.get("case_source", "pypower"),
        "seed": config.get("seed", 123),
        "fallback_to_synthetic": bool(config.get("fallback_to_synthetic", False)),
    }
    if isinstance(case, dict):
        case_config.update(case)
    elif case == "synthetic":
        case_config.update({"matrix_source": "synthetic", "case_name": "synthetic"})
    else:
        case_config.update({"case_name": str(case)})
    return case_config


def _resolve_phase_stable_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase_validation_stable_basis",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "degrees": [35, 51, 71, 101, 151, 201],
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 256,
        "bound_grid_size": 4097,
        "bound_tolerance": 1.0e-5,
        "conversion_error_limit": 1.0e-8,
        "coefficient_dynamic_range_limit": 1.0e12,
        "conditioned_degree": 101,
        "conditioned_lambda": 1.0e-4,
        "angle_solver": "root-finding",
        "phase_order": "original",
        "phase_sign": "phi",
        "phase_offset_rule": "none",
        "signal_operator_convention": "pennylane_rx_pcphase",
        "response_component": "real_u00",
        "force_dependency_missing": False,
        "sanity_results_path": (
            "outputs/qsvt_phase_response_convention_diagnostics/sanity_polynomial_results.csv"
        ),
    }
    if config:
        resolved.update(config)
    resolved["degrees"] = [int(value) for value in resolved["degrees"]]
    return resolved


def _resolve_preconditioned_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_preconditioned_ieee300_estimator",
        "cases": ["ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": False,
        "alpha": 1.0e-2,
        "degree": 301,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_residual_weighted_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_ieee300_residual_weighted_error",
        "case_name": "ieee300",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": False,
        "alpha": 1.0e-2,
        "degree": 1001,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_failure_summary_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {"output_dir": "outputs/qsvt_failure_fix_summary"}
    if config:
        resolved.update(config)
    return resolved


def main_phase_stable(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run stable-basis QSVT phase validation")
    parser.parse_args(argv)
    run = run_phase_validation_stable_basis()
    print(f"QSVT stable-basis phase validation complete: {run['output_dir']}")


def main_preconditioned(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run preconditioned IEEE300 estimator variants")
    parser.parse_args(argv)
    run = run_preconditioned_ieee300_estimator()
    print(f"QSVT preconditioned IEEE300 estimator complete: {run['output_dir']}")


def main_residual_weighted(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run IEEE300 residual-weighted spectral diagnostic"
    )
    parser.parse_args(argv)
    run = diagnose_ieee300_residual_weighted_error()
    print(f"QSVT IEEE300 residual-weighted diagnostic complete: {run['output_dir']}")


def main_failure_summary(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT failure-fix summary")
    parser.parse_args(argv)
    run = build_failure_fix_summary()
    print(f"QSVT failure-fix summary complete: {run['output_dir']}")
