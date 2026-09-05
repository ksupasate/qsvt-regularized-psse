from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev, Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import RESOURCE_CAVEAT, build_engineering_system
from robust_qsvt_se.qsvt.phase_response_conventions import pennylane_qsvt_response
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.polynomial_approximation import (
    ApproximationContext,
    PolynomialApproximationResult,
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

PHASE_UNRESOLVED_MESSAGE = (
    "Scalar phase-response convention is validated on sanity polynomials, but the "
    "bounded Ridge/Tikhonov target remains unresolved."
)
NO_BRUTE_FORCE_CAVEAT = (
    "No brute-force degree escalation was used, and the strict 1e-3 tolerance was "
    "not relaxed to create a passing result."
)
RESTRICTED_INTERVAL_CAVEAT = (
    "Restricted-interval diagnostics are diagnostic only and are not full-interval QSVT validation."
)
PRECONDITIONING_CAVEAT = (
    "Preconditioning and spectrum-aware diagnostics quantify whether approximation "
    "difficulty is driven by spectral spread or low-density interval regions. They "
    "do not prove quantum speedup or change the main estimator claims."
)
PHASE_CAVEAT = (
    "Phase-response diagnostics are scalar polynomial checks. They are not hardware "
    "execution, quantum speedup, quantum advantage, or evidence that QSVT "
    "outperforms Ridge/Tikhonov under the same alpha."
)
PHASE_TARGET_SUMMARY_PATH = Path(
    "outputs/qsvt_phase_target_failure_diagnostics/phase_target_failure_summary.csv"
)
STABLE_PHASE_SUMMARY_PATH = Path(
    "outputs/qsvt_stable_phase_validation_attempt/stable_phase_validation_summary.csv"
)
SPECTRAL_DIFFICULTY_SUMMARY_PATH = Path(
    "outputs/qsvt_ieee300_spectral_difficulty/spectral_difficulty_summary.csv"
)
SPECTRUM_AWARE_SUMMARY_PATH = Path(
    "outputs/qsvt_spectrum_aware_diagnostics/spectrum_aware_summary.csv"
)
IEEE118_REFINEMENT_SUMMARY_PATH = Path(
    "outputs/qsvt_ieee118_targeted_refinement/ieee118_refinement_summary.csv"
)


def diagnose_phase_target_failure(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_phase_target_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []

    for degree in resolved["degrees"]:
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(resolved["alpha"]),
            degree=int(degree),
            method=str(resolved["method"]),
            grid_size=int(resolved["grid_size"]),
        )
        coefficient = _coefficient_diagnostics(
            alpha=float(resolved["alpha"]),
            result=result,
            original_basis="chebyshev_T_low_to_high_on_unit_interval",
            passed_basis="monomial_power_low_to_high",
        )
        basis = _basis_conversion_diagnostics(result)
        phase = _phase_response_diagnostics(
            result=result,
            target_values=result.bounded_target_values,
            evaluation_points=result.evaluation_points,
            evaluation_kind=result.evaluation_kind,
            output_dir=output_dir,
            config=resolved,
            cache_metadata={
                "script": "diagnose_qsvt_phase_target_failure",
                "alpha": float(resolved["alpha"]),
                "degree": int(result.degree),
                "method": result.method,
            },
        )
        failure_class = _classify_phase_failure(
            coefficient=coefficient,
            basis=basis,
            phase=phase,
            polynomial_error=float(np.max(result.pointwise_errors)),
            tolerance=float(resolved["target_tolerance"]),
            coefficient_dynamic_range_limit=float(resolved["coefficient_dynamic_range_limit"]),
            basis_conversion_error_limit=float(resolved["basis_conversion_error_limit"]),
        )
        status = (
            "passed"
            if phase["status"] == "passed"
            else "diagnosed_failure"
            if phase["status"] != "skipped_dependency_missing"
            else "skipped_dependency_missing"
        )
        failure_reason = _phase_failure_reason(
            failure_class=failure_class,
            phase=phase,
            polynomial_error=float(np.max(result.pointwise_errors)),
            tolerance=float(resolved["target_tolerance"]),
        )
        recommended_fix = _phase_recommended_fix(failure_class)
        row = {
            "alpha": float(resolved["alpha"]),
            "degree": int(result.degree),
            "approximation_method": result.method,
            "polynomial_basis_original": "chebyshev_T_low_to_high_on_unit_interval",
            "polynomial_basis_passed_to_phase_synthesis": "monomial_power_low_to_high",
            "coefficient_order": "low_to_high",
            "max_abs_coefficient": coefficient["max_abs_coefficient"],
            "min_abs_nonzero_coefficient": coefficient["min_abs_nonzero_coefficient"],
            "coefficient_dynamic_range": coefficient["coefficient_dynamic_range"],
            "bounded_target_max_abs": coefficient["bounded_target_max_abs"],
            "polynomial_approx_max_abs": coefficient["polynomial_approx_max_abs"],
            "polynomial_approx_max_error": float(np.max(result.pointwise_errors)),
            "phase_response_max_error": phase["phase_response_max_error"],
            "phase_response_minus_polynomial_error": phase["phase_response_minus_polynomial_error"],
            "parity_error": coefficient["parity_error"],
            "boundedness_violation": coefficient["boundedness_violation"],
            "basis_conversion_error": basis["basis_conversion_error"],
            "status": status,
            "failure_reason": failure_reason,
            "recommended_fix": recommended_fix,
            "failure_class": failure_class,
            "phase_response_vs_polynomial_max_error": phase[
                "phase_response_vs_polynomial_max_error"
            ],
            "phase_status": phase["status"],
            "dependency_available": phase["dependency_available"],
            "phase_count": phase["phase_count"],
            "caveat": PHASE_CAVEAT,
        }
        summary_rows.append(row)
        coefficient_rows.append({**coefficient, "alpha": float(resolved["alpha"])})
        basis_rows.append({**basis, "alpha": float(resolved["alpha"])})
        breakdown_rows.extend(phase["breakdown_rows"])

    summary = pd.DataFrame(summary_rows)
    coefficient_frame = pd.DataFrame(coefficient_rows)
    basis_frame = pd.DataFrame(basis_rows)
    breakdown = pd.DataFrame(breakdown_rows)

    summary_csv = output_dir / "phase_target_failure_summary.csv"
    summary_json = output_dir / "phase_target_failure_summary.json"
    coefficient_csv = output_dir / "coefficient_diagnostics.csv"
    basis_csv = output_dir / "basis_conversion_diagnostics.csv"
    breakdown_csv = output_dir / "phase_response_error_breakdown.csv"
    report_md = output_dir / "phase_target_failure_report.md"

    summary.to_csv(summary_csv, index=False)
    coefficient_frame.to_csv(coefficient_csv, index=False)
    basis_frame.to_csv(basis_csv, index=False)
    breakdown.to_csv(breakdown_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_phase_target_failure_report(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "phase_target_failure_summary_csv": str(summary_csv),
            "phase_target_failure_summary_json": str(summary_json),
            "coefficient_diagnostics_csv": str(coefficient_csv),
            "basis_conversion_diagnostics_csv": str(basis_csv),
            "phase_response_error_breakdown_csv": str(breakdown_csv),
            "phase_target_failure_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "phase_target_failure_summary_csv": summary_csv,
            "phase_target_failure_summary_json": summary_json,
            "coefficient_diagnostics_csv": coefficient_csv,
            "basis_conversion_diagnostics_csv": basis_csv,
            "phase_response_error_breakdown_csv": breakdown_csv,
            "phase_target_failure_report_md": report_md,
            "manifest": manifest,
        },
    }


def run_stable_phase_validation_attempt(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_stable_phase_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for spec in _stable_phase_targets(context, resolved):
        row, points = _run_stable_phase_target(spec, output_dir, resolved)
        rows.append(row)
        point_rows.extend(points)

    summary = pd.DataFrame(rows)
    pointwise = pd.DataFrame(point_rows)
    summary_csv = output_dir / "stable_phase_validation_summary.csv"
    summary_json = output_dir / "stable_phase_validation_summary.json"
    pointwise_csv = output_dir / "stable_phase_pointwise_errors.csv"
    report_md = output_dir / "stable_phase_report.md"
    summary.to_csv(summary_csv, index=False)
    pointwise.to_csv(pointwise_csv, index=False)
    write_json(summary_json, {"rows": rows})
    report_md.write_text(_stable_phase_report(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "stable_phase_validation_summary_csv": str(summary_csv),
            "stable_phase_validation_summary_json": str(summary_json),
            "stable_phase_pointwise_errors_csv": str(pointwise_csv),
            "stable_phase_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "stable_phase_validation_summary_csv": summary_csv,
            "stable_phase_validation_summary_json": summary_json,
            "stable_phase_pointwise_errors_csv": pointwise_csv,
            "stable_phase_report_md": report_md,
            "manifest": manifest,
        },
    }


def diagnose_ieee300_spectral_difficulty(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_spectral_difficulty_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for case in resolved["cases"]:
        case_config = _case_config(case, resolved)
        case_name = str(case_config.get("case_name", "unknown"))
        alpha = float(case_config.get("alpha", resolved["alpha"]))
        degree = int(
            case_config.get(
                "degree",
                resolved["degree_by_case"].get(case_name, resolved["degree"]),
            )
        )
        method = str(case_config.get("method", resolved["method"]))
        try:
            context = build_approximation_context(case_config)
            result = evaluate_polynomial_approximation(
                context=context,
                alpha=alpha,
                degree=degree,
                method=method,
                grid_size=int(resolved["grid_size"]),
            )
            diagnostics = _spectral_error_diagnostics(context, result, alpha=alpha)
            m, n = (int(value) for value in context.matrix_shape.split("x"))
            q = _singular_quantiles(context.singular_values)
            row = {
                "case_name": context.case_name,
                "m": m,
                "n": n,
                "alpha": alpha,
                "degree": int(result.degree),
                "sigma_min": float(np.min(context.singular_values)),
                "sigma_max": float(np.max(context.singular_values)),
                "kappa": float(np.max(context.singular_values) / np.min(context.singular_values)),
                "singular_value_q001": q[0.001],
                "singular_value_q01": q[0.01],
                "singular_value_q05": q[0.05],
                "singular_value_q50": q[0.50],
                "singular_value_q95": q[0.95],
                "singular_value_q99": q[0.99],
                "singular_value_q999": q[0.999],
                "full_interval_max_error": diagnostics["full_interval_max_error"],
                "actual_singular_values_max_error": diagnostics["actual_singular_values_max_error"],
                "actual_singular_values_mean_error": diagnostics[
                    "actual_singular_values_mean_error"
                ],
                "central_99_interval_max_error": diagnostics["central_99_interval_max_error"],
                "central_95_interval_max_error": diagnostics["central_95_interval_max_error"],
                "error_peak_sigma": diagnostics["error_peak_sigma"],
                "nearest_actual_singular_value": diagnostics["nearest_actual_singular_value"],
                "distance_to_nearest_singular_value": diagnostics[
                    "distance_to_nearest_singular_value"
                ],
                "error_peak_region": diagnostics["error_peak_region"],
                "diagnostic_interpretation": diagnostics["diagnostic_interpretation"],
                "matrix_source": context.matrix_source,
                "status": "ok",
                "failure_reason_if_any": "",
                "interval_caveat": RESTRICTED_INTERVAL_CAVEAT,
            }
            summary_rows.append(row)
            quantile_rows.extend(
                _quantile_rows(context, alpha=alpha, degree=int(result.degree), q=q)
            )
            histogram_rows.extend(
                _histogram_rows(
                    context,
                    alpha=alpha,
                    degree=int(result.degree),
                    bins=int(resolved["histogram_bins"]),
                )
            )
            error_rows.append(
                {
                    "case_name": context.case_name,
                    "alpha": alpha,
                    "degree": int(result.degree),
                    **{
                        key: diagnostics[key]
                        for key in [
                            "full_interval_max_error",
                            "actual_singular_values_max_error",
                            "actual_singular_values_mean_error",
                            "error_peak_sigma",
                            "nearest_actual_singular_value",
                            "distance_to_nearest_singular_value",
                            "error_peak_region",
                        ]
                    },
                }
            )
            interval_rows.extend(_interval_rows(context, result, alpha=alpha))
        except Exception as exc:
            summary_rows.append(
                _spectral_failure_row(case_name, case_config, alpha, degree, str(exc))
            )

    summary = pd.DataFrame(summary_rows)
    quantiles = pd.DataFrame(quantile_rows)
    histograms = pd.DataFrame(histogram_rows)
    errors = pd.DataFrame(error_rows)
    intervals = pd.DataFrame(interval_rows)

    summary_csv = output_dir / "spectral_difficulty_summary.csv"
    summary_json = output_dir / "spectral_difficulty_summary.json"
    quantiles_csv = output_dir / "singular_value_quantiles.csv"
    histograms_csv = output_dir / "singular_value_histograms.csv"
    errors_csv = output_dir / "error_location_diagnostics.csv"
    intervals_csv = output_dir / "interval_restriction_diagnostics.csv"
    report_md = output_dir / "ieee300_spectral_difficulty_report.md"

    summary.to_csv(summary_csv, index=False)
    quantiles.to_csv(quantiles_csv, index=False)
    histograms.to_csv(histograms_csv, index=False)
    errors.to_csv(errors_csv, index=False)
    intervals.to_csv(intervals_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    report_md.write_text(_spectral_difficulty_report(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "spectral_difficulty_summary_csv": str(summary_csv),
            "spectral_difficulty_summary_json": str(summary_json),
            "singular_value_quantiles_csv": str(quantiles_csv),
            "singular_value_histograms_csv": str(histograms_csv),
            "error_location_diagnostics_csv": str(errors_csv),
            "interval_restriction_diagnostics_csv": str(intervals_csv),
            "ieee300_spectral_difficulty_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "spectral_difficulty_summary_csv": summary_csv,
            "spectral_difficulty_summary_json": summary_json,
            "singular_value_quantiles_csv": quantiles_csv,
            "singular_value_histograms_csv": histograms_csv,
            "error_location_diagnostics_csv": errors_csv,
            "interval_restriction_diagnostics_csv": intervals_csv,
            "ieee300_spectral_difficulty_report_md": report_md,
            "manifest": manifest,
        },
    }


def run_spectrum_aware_diagnostics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_spectrum_aware_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for case in resolved["cases"]:
        case_config = _case_config(case, resolved)
        case_name = str(case_config.get("case_name", "unknown"))
        alpha = float(case_config.get("alpha", resolved["alpha"]))
        degree = int(
            case_config.get(
                "degree",
                resolved["degree_by_case"].get(case_name, resolved["degree"]),
            )
        )
        try:
            baseline_context = build_approximation_context(case_config)
            baseline_result = evaluate_polynomial_approximation(
                context=baseline_context,
                alpha=alpha,
                degree=degree,
                method=str(resolved["method"]),
                grid_size=int(resolved["grid_size"]),
            )
            baseline_diag = _spectral_error_diagnostics(
                baseline_context,
                baseline_result,
                alpha=alpha,
            )
            rows.append(
                _column_equilibration_row(
                    baseline_context=baseline_context,
                    baseline_diag=baseline_diag,
                    case_config=case_config,
                    alpha=alpha,
                    degree=int(baseline_result.degree),
                    config=resolved,
                )
            )
            for low_q, high_q, label in [
                (0.001, 0.999, "central_99"),
                (0.05, 0.95, "central_95"),
            ]:
                interval_row = _restriction_diagnostic_row(
                    context=baseline_context,
                    result=baseline_result,
                    baseline_diag=baseline_diag,
                    alpha=alpha,
                    degree=int(baseline_result.degree),
                    low_q=low_q,
                    high_q=high_q,
                    label=label,
                )
                rows.append(interval_row)
                interval_rows.append(interval_row)
        except Exception as exc:
            rows.append(
                _spectrum_aware_failure_row(case_name, case_config, alpha, degree, str(exc))
            )

    summary = pd.DataFrame(rows)
    interval_frame = pd.DataFrame(interval_rows)
    summary_csv = output_dir / "spectrum_aware_summary.csv"
    summary_json = output_dir / "spectrum_aware_summary.json"
    interval_csv = output_dir / "preconditioning_interval_diagnostics.csv"
    report_md = output_dir / "spectrum_aware_report.md"
    summary.to_csv(summary_csv, index=False)
    interval_frame.to_csv(interval_csv, index=False)
    write_json(summary_json, {"rows": rows})
    report_md.write_text(_spectrum_aware_report(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "spectrum_aware_summary_csv": str(summary_csv),
            "spectrum_aware_summary_json": str(summary_json),
            "preconditioning_interval_diagnostics_csv": str(interval_csv),
            "spectrum_aware_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "spectrum_aware_summary_csv": summary_csv,
            "spectrum_aware_summary_json": summary_json,
            "preconditioning_interval_diagnostics_csv": interval_csv,
            "spectrum_aware_report_md": report_md,
            "manifest": manifest,
        },
    }


def run_ieee118_targeted_refinement(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_ieee118_refinement_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    case_config = _case_config(
        {
            "case_name": resolved["case_name"],
            "matrix_source": resolved.get("matrix_source", "ieee14_ac_weighted_jacobian"),
        },
        resolved,
    )
    try:
        context = build_approximation_context(case_config)
    except Exception as exc:
        rows.append(
            _ieee118_failure_row(
                case_name=str(resolved["case_name"]),
                alpha=float(resolved["alpha"]),
                degree=0,
                reason=str(exc),
                status="failed_matrix_construction",
            )
        )
    else:
        for degree in resolved["degrees"]:
            start = time.perf_counter()
            try:
                result = evaluate_polynomial_approximation(
                    context=context,
                    alpha=float(resolved["alpha"]),
                    degree=int(degree),
                    method=str(resolved["method"]),
                    grid_size=int(resolved["grid_size"]),
                )
                max_error = float(np.max(result.pointwise_errors))
                passed = bool(max_error <= float(resolved["target_tolerance"]))
                stability_status = str(result.numerical_stability_status)
                if passed and stability_status != "ok":
                    passed = False
                    status = "failed_numerical_instability"
                    failure_reason = (
                        "polynomial met the target tolerance, but the diagnostic monomial "
                        f"basis conversion reported {stability_status}"
                    )
                else:
                    status = "passed" if passed else "failed_target_tolerance"
                    failure_reason = ""
                row = {
                    "case_name": context.case_name,
                    "alpha": float(resolved["alpha"]),
                    "degree": int(result.degree),
                    "query_count": int(2 * result.degree + 1),
                    "max_pointwise_error": max_error,
                    "mean_pointwise_error": float(np.mean(result.pointwise_errors)),
                    "rms_pointwise_error": float(np.sqrt(np.mean(result.pointwise_errors**2))),
                    "passed_1e_minus_3": passed,
                    "runtime_seconds": float(time.perf_counter() - start),
                    "status": status,
                    "diagnostic_status": (
                        "diagnostic_only" if stability_status != "ok" else "not_applicable"
                    ),
                    "numerical_stability_status": stability_status,
                    "resource_caveat": RESOURCE_CAVEAT,
                    "failure_reason_if_any": failure_reason,
                    "degree_budget_caveat": (
                        "Only the approved targeted degree list was used; no arbitrary "
                        "higher-degree search was performed."
                    ),
                }
            except Exception as exc:
                row = _ieee118_failure_row(
                    case_name=context.case_name,
                    alpha=float(resolved["alpha"]),
                    degree=int(degree),
                    reason=str(exc),
                    status="failed_numerical_instability",
                    runtime_seconds=float(time.perf_counter() - start),
                )
            rows.append(row)
            if bool(row.get("passed_1e_minus_3", False)):
                break

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "ieee118_refinement_summary.csv"
    summary_json = output_dir / "ieee118_refinement_summary.json"
    trace_csv = output_dir / "ieee118_refinement_trace.csv"
    report_md = output_dir / "ieee118_refinement_report.md"
    summary.to_csv(summary_csv, index=False)
    summary.to_csv(trace_csv, index=False)
    write_json(summary_json, {"rows": rows})
    report_md.write_text(_ieee118_refinement_report(summary, resolved), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "ieee118_refinement_summary_csv": str(summary_csv),
            "ieee118_refinement_summary_json": str(summary_json),
            "ieee118_refinement_trace_csv": str(trace_csv),
            "ieee118_refinement_report_md": str(report_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "ieee118_refinement_summary_csv": summary_csv,
            "ieee118_refinement_summary_json": summary_json,
            "ieee118_refinement_trace_csv": trace_csv,
            "ieee118_refinement_report_md": report_md,
            "manifest": manifest,
        },
    }


def build_nonbruteforce_refinement_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_nonbruteforce_summary_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows = _consolidated_rows(resolved)
    summary = pd.DataFrame(rows)
    csv_path = output_dir / "nonbruteforce_refinement_summary.csv"
    json_path = output_dir / "nonbruteforce_refinement_summary.json"
    md_path = output_dir / "nonbruteforce_refinement_summary.md"
    summary.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    md_path.write_text(_consolidated_report(rows, resolved), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "nonbruteforce_refinement_summary_md": str(md_path),
            "nonbruteforce_refinement_summary_csv": str(csv_path),
            "nonbruteforce_refinement_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "nonbruteforce_refinement_summary_md": md_path,
            "nonbruteforce_refinement_summary_csv": csv_path,
            "nonbruteforce_refinement_summary_json": json_path,
            "manifest": manifest,
        },
    }


def _coefficient_diagnostics(
    *,
    alpha: float,
    result: PolynomialApproximationResult,
    original_basis: str,
    passed_basis: str,
) -> dict[str, Any]:
    coefficients = np.asarray(result.power_coefficients, dtype=np.float64)
    nonzero = np.abs(coefficients[np.abs(coefficients) > 0.0])
    min_nonzero = float(np.min(nonzero)) if nonzero.size else np.nan
    max_abs = float(np.max(np.abs(coefficients))) if coefficients.size else np.nan
    dynamic_range = (
        float(max_abs / min_nonzero) if min_nonzero and np.isfinite(min_nonzero) else np.nan
    )
    unit_grid = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)
    polynomial_values = Polynomial(coefficients)(unit_grid)
    max_abs_poly = float(np.max(np.abs(polynomial_values)))
    even_coefficients = coefficients[::2]
    parity_error = float(np.max(np.abs(even_coefficients))) if even_coefficients.size else 0.0
    return {
        "degree": int(result.degree),
        "approximation_method": result.method,
        "polynomial_basis_original": original_basis,
        "polynomial_basis_passed_to_phase_synthesis": passed_basis,
        "coefficient_order": "low_to_high",
        "coefficient_count": int(coefficients.size),
        "finite_coefficients": bool(np.all(np.isfinite(coefficients))),
        "max_abs_coefficient": max_abs,
        "min_abs_nonzero_coefficient": min_nonzero,
        "coefficient_dynamic_range": dynamic_range,
        "bounded_target_max_abs": float(np.max(np.abs(result.bounded_target_values))),
        "polynomial_approx_max_abs": max_abs_poly,
        "polynomial_approx_max_error": float(np.max(result.pointwise_errors)),
        "parity_error": parity_error,
        "boundedness_violation": max(0.0, max_abs_poly - 1.0),
        "alpha": alpha,
    }


def _basis_conversion_diagnostics(result: PolynomialApproximationResult) -> dict[str, Any]:
    grid = np.linspace(-1.0, 1.0, 4097, dtype=np.float64)
    cheb_values = Chebyshev(
        np.asarray(result.chebyshev_coefficients, dtype=np.float64),
        domain=[-1.0, 1.0],
    )(grid)
    power_values = Polynomial(np.asarray(result.power_coefficients, dtype=np.float64))(grid)
    difference = np.abs(cheb_values - power_values)
    return {
        "degree": int(result.degree),
        "approximation_method": result.method,
        "polynomial_basis_original": "chebyshev_T_low_to_high_on_unit_interval",
        "polynomial_basis_converted": "monomial_power_low_to_high",
        "basis_conversion_error": float(np.max(difference)),
        "basis_conversion_mean_error": float(np.mean(difference)),
        "basis_conversion_rms_error": float(np.sqrt(np.mean(difference**2))),
        "conversion_status": "ok" if float(np.max(difference)) <= 1.0e-8 else "lossy",
    }


def _phase_response_diagnostics(
    *,
    result: PolynomialApproximationResult,
    target_values: np.ndarray,
    evaluation_points: np.ndarray,
    evaluation_kind: list[str],
    output_dir: Path,
    config: dict[str, Any],
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    dependency_available = importlib.util.find_spec("pennylane") is not None
    if bool(config.get("force_dependency_missing", False)) or not dependency_available:
        return {
            "status": "skipped_dependency_missing",
            "dependency_available": bool(dependency_available),
            "phase_count": 0,
            "phase_response_max_error": np.nan,
            "phase_response_minus_polynomial_error": np.nan,
            "phase_response_vs_polynomial_max_error": np.nan,
            "failure_reason": "PennyLane unavailable or forced missing",
            "breakdown_rows": [],
        }
    try:
        validation = validate_qsvt_polynomial(
            result.power_coefficients,
            parity="odd",
            grid_size=int(config["bound_validation_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phase_result = synthesize_pennylane_phases_cached(
            result.power_coefficients,
            angle_solver=str(config["angle_solver"]),
            cache_dir=output_dir / "phase_cache",
            cache_metadata=cache_metadata,
        )
        mask = np.asarray(evaluation_kind, dtype=object) == "grid"
        grid = np.asarray(evaluation_points, dtype=np.float64)[mask]
        target = np.asarray(target_values, dtype=np.float64)[mask]
        polynomial = np.asarray(result.bounded_approximation_values, dtype=np.float64)[mask]
        response = pennylane_qsvt_response(
            grid,
            phase_result.phases,
            phase_order=str(config["phase_order"]),
            phase_sign=str(config["phase_sign"]),
            phase_offset_rule=str(config["phase_offset_rule"]),
            signal_operator_convention=str(config["signal_operator_convention"]),
            response_component=str(config["response_component"]),
        )
        phase_errors = np.abs(response - target)
        polynomial_errors = np.abs(polynomial - target)
        phase_vs_polynomial = np.abs(response - polynomial)
        max_phase = float(np.max(phase_errors))
        max_polynomial = float(np.max(polynomial_errors))
        status = (
            "passed" if max_phase <= float(config["target_tolerance"]) else "failed_phase_response"
        )
        rows = [
            {
                "degree": int(result.degree),
                "evaluation_index": int(index),
                "sigma_normalized": float(sigma),
                "bounded_target_value": float(target_value),
                "polynomial_value": float(polynomial_value),
                "phase_response_value": float(response_value),
                "polynomial_pointwise_error": float(poly_error),
                "phase_response_pointwise_error": float(phase_error),
                "phase_vs_polynomial_pointwise_error": float(poly_phase_error),
            }
            for index, (
                sigma,
                target_value,
                polynomial_value,
                response_value,
                poly_error,
                phase_error,
                poly_phase_error,
            ) in enumerate(
                zip(
                    grid,
                    target,
                    polynomial,
                    response,
                    polynomial_errors,
                    phase_errors,
                    phase_vs_polynomial,
                    strict=True,
                )
            )
        ]
        return {
            "status": status,
            "dependency_available": True,
            "phase_count": int(phase_result.phases.size),
            "phase_response_max_error": max_phase,
            "phase_response_minus_polynomial_error": float(max_phase - max_polynomial),
            "phase_response_vs_polynomial_max_error": float(np.max(phase_vs_polynomial)),
            "failure_reason": "" if status == "passed" else "phase response exceeded tolerance",
            "polynomial_validation_max_abs": validation["max_abs_on_unit_interval"],
            "breakdown_rows": rows,
        }
    except Exception as exc:
        return {
            "status": "failed_phase_response",
            "dependency_available": bool(dependency_available),
            "phase_count": 0,
            "phase_response_max_error": np.nan,
            "phase_response_minus_polynomial_error": np.nan,
            "phase_response_vs_polynomial_max_error": np.nan,
            "failure_reason": str(exc),
            "breakdown_rows": [],
        }


def _classify_phase_failure(
    *,
    coefficient: dict[str, Any],
    basis: dict[str, Any],
    phase: dict[str, Any],
    polynomial_error: float,
    tolerance: float,
    coefficient_dynamic_range_limit: float,
    basis_conversion_error_limit: float,
) -> str:
    classes: list[str] = []
    if float(coefficient["boundedness_violation"]) > 0.0:
        classes.append("boundedness_issue")
    if float(coefficient["parity_error"]) > 1.0e-8:
        classes.append("parity_issue")
    dynamic_range = float(coefficient["coefficient_dynamic_range"])
    if np.isfinite(dynamic_range) and dynamic_range > coefficient_dynamic_range_limit:
        classes.append("coefficient_dynamic_range_issue")
    if float(basis["basis_conversion_error"]) > basis_conversion_error_limit:
        classes.append("basis_conversion_instability")
    phase_extra = phase.get("phase_response_minus_polynomial_error", np.nan)
    phase_poly = phase.get("phase_response_vs_polynomial_max_error", np.nan)
    if phase["status"] == "skipped_dependency_missing":
        classes.append("unknown")
    elif (
        polynomial_error > tolerance
        and np.isfinite(phase_extra)
        and abs(float(phase_extra)) <= 5.0e-4
    ):
        classes.append("degree_too_low")
    elif np.isfinite(phase_poly) and float(phase_poly) > tolerance:
        classes.append("phase_backend_convention_unresolved")
    if not classes:
        classes.append("unknown")
    return ";".join(dict.fromkeys(classes))


def _phase_failure_reason(
    *,
    failure_class: str,
    phase: dict[str, Any],
    polynomial_error: float,
    tolerance: float,
) -> str:
    if "degree_too_low" in failure_class:
        return (
            "The phase response tracks the synthesized polynomial, but the polynomial "
            f"approximation error {polynomial_error:.6g} exceeds tolerance {tolerance:.6g}."
        )
    if phase["status"] == "skipped_dependency_missing":
        return str(phase["failure_reason"])
    if phase.get("failure_reason"):
        return str(phase["failure_reason"])
    return "No single dominant failure mode was isolated."


def _phase_recommended_fix(failure_class: str) -> str:
    if "degree_too_low" in failure_class:
        return (
            "Use a polynomial that already passes the bounded approximation diagnostic "
            "before making a phase-validation claim."
        )
    if "basis_conversion_instability" in failure_class:
        return (
            "Avoid unstable monomial conversion or use a phase backend that accepts a stable basis."
        )
    if "coefficient_dynamic_range_issue" in failure_class:
        return "Use lower-risk targets or numerical stabilization before phase synthesis."
    if "phase_backend_convention_unresolved" in failure_class:
        return (
            "Revisit phase-response extraction against sanity polynomial and "
            "polynomial-response checks."
        )
    if "boundedness_issue" in failure_class:
        return "Rescale or refit the bounded polynomial so it is bounded by 1 on [-1, 1]."
    if "parity_issue" in failure_class:
        return "Refit using an explicitly odd polynomial basis."
    return "Document unresolved status and avoid claiming phase validation."


def _stable_phase_targets(
    context: ApproximationContext,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grid = np.linspace(-0.95, 0.95, int(config["sanity_grid_size"]), dtype=np.float64)
    targets = [
        {
            "target": "x",
            "target_type": "sanity_polynomial",
            "degree": 1,
            "coefficients": np.asarray([0.0, 1.0], dtype=np.float64),
            "grid": grid,
            "bounded_target_values": grid,
            "polynomial_values": grid,
            "polynomial_approx_max_error": 0.0,
        },
        {
            "target": "0.5x",
            "target_type": "sanity_polynomial",
            "degree": 1,
            "coefficients": np.asarray([0.0, 0.5], dtype=np.float64),
            "grid": grid,
            "bounded_target_values": 0.5 * grid,
            "polynomial_values": 0.5 * grid,
            "polynomial_approx_max_error": 0.0,
        },
    ]
    for degree, label in [
        (int(config["low_degree"]), "low_degree_bounded_ridge_tikhonov"),
        (35, "degree_35_bounded_ridge_tikhonov"),
        (101, "degree_101_bounded_ridge_tikhonov"),
    ]:
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(config["alpha"]),
            degree=degree,
            method=str(config["method"]),
            grid_size=int(config["grid_size"]),
        )
        mask = np.asarray(result.evaluation_kind, dtype=object) == "grid"
        coefficient = _coefficient_diagnostics(
            alpha=float(config["alpha"]),
            result=result,
            original_basis="chebyshev_T_low_to_high_on_unit_interval",
            passed_basis="monomial_power_low_to_high",
        )
        targets.append(
            {
                "target": label,
                "target_type": "ridge_tikhonov_bounded_target",
                "degree": int(result.degree),
                "coefficients": result.power_coefficients,
                "grid": result.evaluation_points[mask],
                "bounded_target_values": result.bounded_target_values[mask],
                "polynomial_values": result.bounded_approximation_values[mask],
                "polynomial_approx_max_error": float(np.max(result.pointwise_errors)),
                "coefficient_dynamic_range": coefficient["coefficient_dynamic_range"],
                "boundedness_violation": coefficient["boundedness_violation"],
                "parity_error": coefficient["parity_error"],
            }
        )
    return targets


def _run_stable_phase_target(
    spec: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dependency_available = importlib.util.find_spec("pennylane") is not None
    dynamic_range = float(spec.get("coefficient_dynamic_range", 1.0))
    if bool(config["force_dependency_missing"]) or not dependency_available:
        return (
            _stable_phase_skip_row(
                spec,
                dependency_available=dependency_available,
                status="skipped_dependency_missing",
                reason="PennyLane unavailable or forced missing",
                config=config,
            ),
            [],
        )
    if (
        spec["target"] == "degree_101_bounded_ridge_tikhonov"
        and np.isfinite(dynamic_range)
        and dynamic_range > float(config["coefficient_dynamic_range_limit"])
    ):
        return (
            _stable_phase_skip_row(
                spec,
                dependency_available=dependency_available,
                status="skipped_coefficients_unstable",
                reason="coefficient dynamic range exceeded configured safety limit",
                config=config,
            ),
            [],
        )
    try:
        validate_qsvt_polynomial(
            np.asarray(spec["coefficients"], dtype=np.float64),
            parity="odd",
            grid_size=int(config["bound_validation_grid_size"]),
            bound_tolerance=float(config["bound_tolerance"]),
        )
        phase_result = synthesize_pennylane_phases_cached(
            np.asarray(spec["coefficients"], dtype=np.float64),
            angle_solver=str(config["angle_solver"]),
            cache_dir=output_dir / "phase_cache",
            cache_metadata={
                "script": "run_qsvt_stable_phase_validation_attempt",
                "target": str(spec["target"]),
                "degree": int(spec["degree"]),
            },
        )
        response = pennylane_qsvt_response(
            np.asarray(spec["grid"], dtype=np.float64),
            phase_result.phases,
            phase_order=str(config["phase_order"]),
            phase_sign=str(config["phase_sign"]),
            phase_offset_rule=str(config["phase_offset_rule"]),
            signal_operator_convention=str(config["signal_operator_convention"]),
            response_component=str(config["response_component"]),
        )
        target = np.asarray(spec["bounded_target_values"], dtype=np.float64)
        polynomial = np.asarray(spec["polynomial_values"], dtype=np.float64)
        phase_errors = np.abs(response - target)
        polynomial_errors = np.abs(polynomial - target)
        phase_vs_poly = np.abs(response - polynomial)
        max_error = float(np.max(phase_errors))
        polynomial_error = float(np.max(polynomial_errors))
        if polynomial_error > float(config["target_tolerance"]):
            status = "failed_polynomial_approximation"
            reason = "polynomial approximation exceeds target tolerance before phase synthesis"
        elif max_error <= float(config["target_tolerance"]):
            status = "passed"
            reason = ""
        else:
            status = "failed_phase_response"
            reason = "phase response exceeds tolerance despite polynomial approximation passing"
        row = {
            "target": str(spec["target"]),
            "target_type": str(spec["target_type"]),
            "alpha": (
                float(config["alpha"]) if spec["target_type"] != "sanity_polynomial" else np.nan
            ),
            "degree": int(spec["degree"]),
            "phase_count": int(phase_result.phases.size),
            "dependency_available": True,
            "coefficient_dynamic_range": dynamic_range,
            "polynomial_approx_max_error": polynomial_error,
            "max_error": max_error,
            "mean_error": float(np.mean(phase_errors)),
            "rms_error": float(np.sqrt(np.mean(phase_errors**2))),
            "phase_vs_polynomial_max_error": float(np.max(phase_vs_poly)),
            "status": status,
            "reason": reason,
            "caveat": PHASE_CAVEAT,
        }
        point_rows = [
            {
                "target": str(spec["target"]),
                "degree": int(spec["degree"]),
                "sigma_normalized": float(sigma),
                "target_value": float(target_value),
                "polynomial_value": float(poly_value),
                "phase_response_value": float(response_value),
                "pointwise_error": float(error),
                "phase_vs_polynomial_error": float(poly_error),
            }
            for sigma, target_value, poly_value, response_value, error, poly_error in zip(
                spec["grid"],
                target,
                polynomial,
                response,
                phase_errors,
                phase_vs_poly,
                strict=True,
            )
        ]
        return row, point_rows
    except Exception as exc:
        return (
            _stable_phase_skip_row(
                spec,
                dependency_available=dependency_available,
                status="failed_phase_response",
                reason=str(exc),
                config=config,
            ),
            [],
        )


def _stable_phase_skip_row(
    spec: dict[str, Any],
    *,
    dependency_available: bool,
    status: str,
    reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target": str(spec["target"]),
        "target_type": str(spec["target_type"]),
        "alpha": float(config["alpha"]) if spec["target_type"] != "sanity_polynomial" else np.nan,
        "degree": int(spec["degree"]),
        "phase_count": 0,
        "dependency_available": bool(dependency_available),
        "coefficient_dynamic_range": float(spec.get("coefficient_dynamic_range", np.nan)),
        "polynomial_approx_max_error": float(spec.get("polynomial_approx_max_error", np.nan)),
        "max_error": np.nan,
        "mean_error": np.nan,
        "rms_error": np.nan,
        "phase_vs_polynomial_max_error": np.nan,
        "status": status,
        "reason": reason,
        "caveat": PHASE_CAVEAT,
    }


def _spectral_error_diagnostics(
    context: ApproximationContext,
    result: PolynomialApproximationResult,
    *,
    alpha: float,
) -> dict[str, Any]:
    del alpha
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_mask = kinds == "grid"
    actual_mask = kinds == "actual_singular_value"
    grid_x = np.asarray(result.evaluation_points, dtype=np.float64)[grid_mask]
    grid_errors = np.asarray(result.pointwise_errors, dtype=np.float64)[grid_mask]
    actual_x = np.asarray(result.evaluation_points, dtype=np.float64)[actual_mask]
    actual_errors = np.asarray(result.pointwise_errors, dtype=np.float64)[actual_mask]
    peak_index = int(np.argmax(grid_errors))
    peak_x = float(grid_x[peak_index])
    peak_sigma = peak_x * context.beta
    actual_sigma = actual_x * context.beta
    nearest_index = int(np.argmin(np.abs(actual_sigma - peak_sigma)))
    nearest = float(actual_sigma[nearest_index])
    distance = float(abs(nearest - peak_sigma))
    q001, q999 = np.quantile(context.normalized_singular_values, [0.001, 0.999])
    q05, q95 = np.quantile(context.normalized_singular_values, [0.05, 0.95])
    central_99 = _max_error_in_interval(grid_x, grid_errors, float(q001), float(q999))
    central_95 = _max_error_in_interval(grid_x, grid_errors, float(q05), float(q95))
    region = _error_peak_region(
        peak_x=peak_x,
        domain_min=context.domain_min,
        nearest_distance_normalized=distance / context.beta,
    )
    full_error = float(np.max(grid_errors))
    actual_max = float(np.max(actual_errors))
    interpretation = _spectral_interpretation(
        full_error=full_error,
        actual_error=actual_max,
        central_95=central_95,
        region=region,
    )
    return {
        "full_interval_max_error": full_error,
        "actual_singular_values_max_error": actual_max,
        "actual_singular_values_mean_error": float(np.mean(actual_errors)),
        "central_99_interval_max_error": central_99,
        "central_95_interval_max_error": central_95,
        "error_peak_sigma": float(peak_sigma),
        "nearest_actual_singular_value": nearest,
        "distance_to_nearest_singular_value": distance,
        "error_peak_region": region,
        "diagnostic_interpretation": interpretation,
    }


def _max_error_in_interval(
    points: np.ndarray,
    errors: np.ndarray,
    low: float,
    high: float,
) -> float:
    mask = (points >= low) & (points <= high)
    if not np.any(mask):
        index = int(np.argmin(np.abs(points - (low + high) / 2.0)))
        return float(errors[index])
    return float(np.max(errors[mask]))


def _error_peak_region(
    *,
    peak_x: float,
    domain_min: float,
    nearest_distance_normalized: float,
) -> str:
    span = max(1.0 - domain_min, np.finfo(float).eps)
    if abs(peak_x - domain_min) <= 0.02 * span:
        return "near_sigma_min"
    if abs(1.0 - peak_x) <= 0.02 * span:
        return "near_sigma_max"
    if nearest_distance_normalized >= 0.02 * span:
        return "low_density_interior_interval"
    return "interior_with_spectral_mass"


def _spectral_interpretation(
    *,
    full_error: float,
    actual_error: float,
    central_95: float,
    region: str,
) -> str:
    if actual_error <= 1.0e-3 and full_error > 1.0e-3:
        return (
            "Full-interval error exceeds tolerance, while actual-singular-value error "
            "is small. This is diagnostic only and not full validation."
        )
    if central_95 < 0.5 * full_error and region == "low_density_interior_interval":
        return (
            "Restricted-interval diagnostics suggest that much of the full-interval "
            "error is concentrated in a low-density spectral region."
        )
    if region == "near_sigma_min":
        return "The largest error occurs near the smallest normalized singular values."
    if region == "near_sigma_max":
        return "The largest error occurs near the largest normalized singular values."
    return "Full-interval and actual-spectrum errors should be reported separately."


def _singular_quantiles(singular_values: np.ndarray) -> dict[float, float]:
    qs = [0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
    values = np.quantile(np.asarray(singular_values, dtype=np.float64), qs)
    return {q: float(value) for q, value in zip(qs, values, strict=True)}


def _quantile_rows(
    context: ApproximationContext,
    *,
    alpha: float,
    degree: int,
    q: dict[float, float],
) -> list[dict[str, Any]]:
    return [
        {
            "case_name": context.case_name,
            "alpha": alpha,
            "degree": degree,
            "quantile": quantile,
            "singular_value": value,
            "normalized_singular_value": value / context.beta,
        }
        for quantile, value in sorted(q.items())
    ]


def _histogram_rows(
    context: ApproximationContext,
    *,
    alpha: float,
    degree: int,
    bins: int,
) -> list[dict[str, Any]]:
    counts, edges = np.histogram(context.singular_values, bins=max(1, int(bins)))
    return [
        {
            "case_name": context.case_name,
            "alpha": alpha,
            "degree": degree,
            "bin_index": int(index),
            "bin_left": float(edges[index]),
            "bin_right": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _interval_rows(
    context: ApproximationContext,
    result: PolynomialApproximationResult,
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_mask = kinds == "grid"
    grid_x = np.asarray(result.evaluation_points, dtype=np.float64)[grid_mask]
    grid_errors = np.asarray(result.pointwise_errors, dtype=np.float64)[grid_mask]
    rows = []
    for low_q, high_q, label in [(0.001, 0.999, "central_99"), (0.05, 0.95, "central_95")]:
        low, high = np.quantile(context.normalized_singular_values, [low_q, high_q])
        rows.append(
            {
                "case_name": context.case_name,
                "alpha": alpha,
                "degree": int(result.degree),
                "interval_label": label,
                "low_quantile": low_q,
                "high_quantile": high_q,
                "sigma_low": float(low * context.beta),
                "sigma_high": float(high * context.beta),
                "normalized_low": float(low),
                "normalized_high": float(high),
                "restricted_interval_max_error": _max_error_in_interval(
                    grid_x,
                    grid_errors,
                    float(low),
                    float(high),
                ),
                "full_interval_max_error": float(np.max(grid_errors)),
                "caveat": RESTRICTED_INTERVAL_CAVEAT,
            }
        )
    return rows


def _column_equilibration_row(
    *,
    baseline_context: ApproximationContext,
    baseline_diag: dict[str, Any],
    case_config: dict[str, Any],
    alpha: float,
    degree: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    system, matrix_source = build_engineering_system(case_config)
    H = np.asarray(system.H_tilde, dtype=np.float64)
    column_norms = np.linalg.norm(H, axis=0)
    scales = np.divide(
        1.0,
        column_norms,
        out=np.ones_like(column_norms),
        where=column_norms > 1.0e-14,
    )
    scaled = H * scales[None, :]
    after_context = _context_from_matrix(
        matrix=scaled,
        case_name=baseline_context.case_name,
        matrix_source=f"{matrix_source}_column_equilibrated",
        source_note="diagnostic column equilibration only",
    )
    after_result = evaluate_polynomial_approximation(
        context=after_context,
        alpha=alpha,
        degree=degree,
        method=str(config["method"]),
        grid_size=int(config["grid_size"]),
    )
    after_diag = _spectral_error_diagnostics(after_context, after_result, alpha=alpha)
    return {
        "case_name": baseline_context.case_name,
        "diagnostic_type": "column_equilibration",
        "alpha": alpha,
        "degree": degree,
        "sigma_min_before": float(np.min(baseline_context.singular_values)),
        "sigma_max_before": float(np.max(baseline_context.singular_values)),
        "kappa_before": float(
            np.max(baseline_context.singular_values) / np.min(baseline_context.singular_values)
        ),
        "sigma_min_after": float(np.min(after_context.singular_values)),
        "sigma_max_after": float(np.max(after_context.singular_values)),
        "kappa_after": float(
            np.max(after_context.singular_values) / np.min(after_context.singular_values)
        ),
        "full_interval_error_before": baseline_diag["full_interval_max_error"],
        "full_interval_error_after": after_diag["full_interval_max_error"],
        "actual_singular_error_before": baseline_diag["actual_singular_values_max_error"],
        "actual_singular_error_after": after_diag["actual_singular_values_max_error"],
        "selected_interval": "full normalized interval after column scaling",
        "interval_caveat": "Preconditioning diagnostic only; main estimator results are unchanged.",
        "resource_caveat": RESOURCE_CAVEAT,
        "status": "diagnostic_only",
    }


def _restriction_diagnostic_row(
    *,
    context: ApproximationContext,
    result: PolynomialApproximationResult,
    baseline_diag: dict[str, Any],
    alpha: float,
    degree: int,
    low_q: float,
    high_q: float,
    label: str,
) -> dict[str, Any]:
    low, high = np.quantile(context.normalized_singular_values, [low_q, high_q])
    kinds = np.asarray(result.evaluation_kind, dtype=object)
    grid_mask = kinds == "grid"
    grid_x = np.asarray(result.evaluation_points, dtype=np.float64)[grid_mask]
    grid_errors = np.asarray(result.pointwise_errors, dtype=np.float64)[grid_mask]
    restricted = _max_error_in_interval(grid_x, grid_errors, float(low), float(high))
    sigma_low = float(low * context.beta)
    sigma_high = float(high * context.beta)
    return {
        "case_name": context.case_name,
        "diagnostic_type": f"{label}_interval_restriction",
        "alpha": alpha,
        "degree": degree,
        "sigma_min_before": float(np.min(context.singular_values)),
        "sigma_max_before": float(np.max(context.singular_values)),
        "kappa_before": float(np.max(context.singular_values) / np.min(context.singular_values)),
        "sigma_min_after": sigma_low,
        "sigma_max_after": sigma_high,
        "kappa_after": float(sigma_high / sigma_low) if sigma_low > 0.0 else np.inf,
        "full_interval_error_before": baseline_diag["full_interval_max_error"],
        "full_interval_error_after": restricted,
        "actual_singular_error_before": baseline_diag["actual_singular_values_max_error"],
        "actual_singular_error_after": baseline_diag["actual_singular_values_max_error"],
        "selected_interval": f"{low_q:g}-{high_q:g} singular-value quantile interval",
        "interval_caveat": RESTRICTED_INTERVAL_CAVEAT,
        "resource_caveat": RESOURCE_CAVEAT,
        "status": "diagnostic_only",
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
    normalized = positive / beta
    return ApproximationContext(
        case_name=case_name,
        matrix_source=matrix_source,
        matrix_shape=f"{matrix.shape[0]}x{matrix.shape[1]}",
        beta=beta,
        singular_values=positive,
        normalized_singular_values=normalized,
        domain_min=max(float(np.min(normalized)), np.finfo(float).eps),
        domain_max=1.0,
        source_note=source_note,
    )


def _spectral_failure_row(
    case_name: str,
    case_config: dict[str, Any],
    alpha: float,
    degree: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "m": np.nan,
        "n": np.nan,
        "alpha": alpha,
        "degree": degree,
        "sigma_min": np.nan,
        "sigma_max": np.nan,
        "kappa": np.nan,
        "singular_value_q001": np.nan,
        "singular_value_q01": np.nan,
        "singular_value_q05": np.nan,
        "singular_value_q50": np.nan,
        "singular_value_q95": np.nan,
        "singular_value_q99": np.nan,
        "singular_value_q999": np.nan,
        "full_interval_max_error": np.nan,
        "actual_singular_values_max_error": np.nan,
        "actual_singular_values_mean_error": np.nan,
        "central_99_interval_max_error": np.nan,
        "central_95_interval_max_error": np.nan,
        "error_peak_sigma": np.nan,
        "nearest_actual_singular_value": np.nan,
        "distance_to_nearest_singular_value": np.nan,
        "error_peak_region": "unavailable",
        "diagnostic_interpretation": "Matrix construction or approximation failed gracefully.",
        "matrix_source": str(case_config.get("matrix_source", "pypower_ac_weighted_jacobian")),
        "status": "failed",
        "failure_reason_if_any": reason,
        "interval_caveat": RESTRICTED_INTERVAL_CAVEAT,
    }


def _spectrum_aware_failure_row(
    case_name: str,
    case_config: dict[str, Any],
    alpha: float,
    degree: int,
    reason: str,
) -> dict[str, Any]:
    del case_config
    return {
        "case_name": case_name,
        "diagnostic_type": "unavailable",
        "alpha": alpha,
        "degree": degree,
        "sigma_min_before": np.nan,
        "sigma_max_before": np.nan,
        "kappa_before": np.nan,
        "sigma_min_after": np.nan,
        "sigma_max_after": np.nan,
        "kappa_after": np.nan,
        "full_interval_error_before": np.nan,
        "full_interval_error_after": np.nan,
        "actual_singular_error_before": np.nan,
        "actual_singular_error_after": np.nan,
        "selected_interval": "",
        "interval_caveat": reason,
        "resource_caveat": RESOURCE_CAVEAT,
        "status": "failed",
    }


def _ieee118_failure_row(
    *,
    case_name: str,
    alpha: float,
    degree: int,
    reason: str,
    status: str,
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "alpha": alpha,
        "degree": degree,
        "query_count": int(2 * degree + 1) if degree > 0 else 0,
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "passed_1e_minus_3": False,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "diagnostic_status": "diagnostic_only",
        "numerical_stability_status": "not_computed",
        "resource_caveat": RESOURCE_CAVEAT,
        "failure_reason_if_any": reason,
        "degree_budget_caveat": "Only the approved targeted degree list was used.",
    }


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


def _phase_target_failure_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Phase-Target Failure Diagnosis\n\nNo diagnostic rows were generated.\n"
    degree_35 = summary[summary["degree"] == 35]
    selected = degree_35.iloc[0] if not degree_35.empty else summary.iloc[-1]
    return f"""# Phase-Target Failure Diagnosis

## Executive Verdict

Failure class: `{selected["failure_class"]}`.

Polynomial approximation error: `{float(selected["polynomial_approx_max_error"]):.6g}`.
Phase-response error: `{float(selected["phase_response_max_error"]):.6g}`.
Phase response minus polynomial error:
`{float(selected["phase_response_minus_polynomial_error"]):.6g}`.

## Diagnosis

The coefficient basis passed to phase synthesis is `monomial_power_low_to_high`.
The original fitted basis is tracked separately as
`chebyshev_T_low_to_high_on_unit_interval`.

The main failure is not reported as a convention fix unless the phase response
itself passes the declared tolerance. When phase and polynomial errors are
nearly the same, the failure is attributed to the bounded polynomial target not
meeting the strict tolerance at that degree.

## Recommended Fix

{selected["recommended_fix"]}

## Claim Boundary

{PHASE_CAVEAT}
{NO_BRUTE_FORCE_CAVEAT}
"""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_No requested columns._"
    rows = frame[available].fillna("").astype(str)
    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    body = [
        "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |"
        for row in rows.to_numpy()
    ]
    return "\n".join([header, separator, *body])


def _escape_markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _stable_phase_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Stable Phase Validation Attempt\n\nNo rows were generated.\n"
    passed_ridge = summary[
        (summary["target_type"] == "ridge_tikhonov_bounded_target")
        & (summary["status"] == "passed")
    ]
    verdict = (
        "At least one bounded Ridge/Tikhonov target phase validation passed."
        if not passed_ridge.empty
        else PHASE_UNRESOLVED_MESSAGE
    )
    table = _markdown_table(summary, ["target", "degree", "status", "max_error", "reason"])
    return f"""# Stable Phase Validation Attempt

## Verdict

{verdict}

## Results

{table}

## Claim Boundary

Only rows with `status == passed` for the bounded Ridge/Tikhonov target support
the phrase "phase validation passed" for that target. Sanity-polynomial passes
alone do not validate the bounded Ridge/Tikhonov target.

{PHASE_CAVEAT}
{NO_BRUTE_FORCE_CAVEAT}
"""


def _spectral_difficulty_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# IEEE300 Spectral Difficulty Diagnostic\n\nNo rows were generated.\n"
    ieee300 = summary[summary["case_name"] == "ieee300"]
    selected = ieee300.iloc[0] if not ieee300.empty else summary.iloc[-1]
    full_error = float(selected.get("full_interval_max_error", np.nan))
    actual_error = float(selected.get("actual_singular_values_max_error", np.nan))
    passes = bool(np.isfinite(full_error) and full_error <= 1.0e-3)
    rows = _markdown_table(
        summary,
        [
            "case_name",
            "degree",
            "kappa",
            "full_interval_max_error",
            "actual_singular_values_max_error",
            "central_95_interval_max_error",
            "error_peak_region",
        ],
    )
    return f"""# IEEE300 Spectral Difficulty Diagnostic

## Executive Verdict

IEEE300 full-interval 1e-3 status: `{"passed" if passes else "failed"}`.

Full-interval error and actual-singular-value error are reported separately.
For IEEE300, full-interval max error is `{full_error:.6g}` and actual-singular
value max error is `{actual_error:.6g}`.

## Case Comparison

{rows}

## Interpretation

Restricted-interval diagnostics suggest where the full-interval error is
concentrated, but they are not full QSVT validation.

{RESTRICTED_INTERVAL_CAVEAT}
{NO_BRUTE_FORCE_CAVEAT}
"""


def _spectrum_aware_report(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Spectrum-Aware Diagnostics\n\nNo rows were generated.\n"
    table = _markdown_table(
        summary,
        [
            "case_name",
            "diagnostic_type",
            "kappa_before",
            "kappa_after",
            "full_interval_error_before",
            "full_interval_error_after",
            "status",
        ],
    )
    return f"""# Spectrum-Aware and Preconditioning Diagnostics

## Results

{table}

## Interpretation

{PRECONDITIONING_CAVEAT}

{RESTRICTED_INTERVAL_CAVEAT}
"""


def _ieee118_refinement_report(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    if summary.empty:
        return "# IEEE118 Targeted Refinement\n\nNo rows were generated.\n"
    passed = summary[summary["passed_1e_minus_3"] == True]  # noqa: E712
    verdict = (
        "IEEE118 passed strict 1e-3 within the approved targeted degree budget."
        if not passed.empty
        else "IEEE118 did not pass strict 1e-3 within the approved targeted degree budget."
    )
    table = _markdown_table(
        summary,
        ["degree", "query_count", "max_pointwise_error", "passed_1e_minus_3", "status"],
    )
    return f"""# IEEE118 Targeted Refinement

## Verdict

{verdict}

Approved degrees: `{list(config["degrees"])}`.

## Trace

{table}

## Claim Boundary

This is a targeted follow-up because IEEE118 narrowly missed at degree 1001. No
arbitrary higher-degree search was performed.
"""


def _consolidated_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    del config
    rows: list[dict[str, Any]] = []
    phase = _read_csv(PHASE_TARGET_SUMMARY_PATH)
    stable = _read_csv(STABLE_PHASE_SUMMARY_PATH)
    spectral = _read_csv(SPECTRAL_DIFFICULTY_SUMMARY_PATH)
    spectrum = _read_csv(SPECTRUM_AWARE_SUMMARY_PATH)
    ieee118 = _read_csv(IEEE118_REFINEMENT_SUMMARY_PATH)
    rows.append(_summary_row_from_frame("phase_target_failure_diagnosis", phase))
    rows.append(_summary_row_from_frame("stable_phase_validation_attempt", stable))
    rows.append(_summary_row_from_frame("ieee300_spectral_difficulty", spectral))
    rows.append(_summary_row_from_frame("spectrum_aware_diagnostics", spectrum))
    rows.append(_summary_row_from_frame("ieee118_targeted_refinement", ieee118))
    rows.append(
        {
            "area": "claim_boundary",
            "status": "documented",
            "key_result": NO_BRUTE_FORCE_CAVEAT,
            "full_interval_validation": "Only explicitly passing full-interval rows count.",
            "diagnostic_only": RESTRICTED_INTERVAL_CAVEAT,
            "supporting_output": "outputs/qsvt_nonbruteforce_refinement_summary",
        }
    )
    return rows


def _summary_row_from_frame(area: str, frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "area": area,
            "status": "missing",
            "key_result": "Required output was not available when summary was built.",
            "full_interval_validation": "not available",
            "diagnostic_only": "not available",
            "supporting_output": "",
        }
    if area == "phase_target_failure_diagnosis":
        row = (
            frame[frame["degree"] == 35].iloc[0]
            if (frame["degree"] == 35).any()
            else frame.iloc[-1]
        )
        return {
            "area": area,
            "status": str(row["status"]),
            "key_result": (
                f"failure_class={row['failure_class']}; "
                f"phase_error={float(row['phase_response_max_error']):.6g}"
            ),
            "full_interval_validation": "phase target passes only if status is passed",
            "diagnostic_only": "coefficient and basis diagnostics",
            "supporting_output": "outputs/qsvt_phase_target_failure_diagnostics",
        }
    if area == "stable_phase_validation_attempt":
        ridge = frame[frame["target_type"] == "ridge_tikhonov_bounded_target"]
        passed = int((ridge["status"] == "passed").sum()) if not ridge.empty else 0
        return {
            "area": area,
            "status": "passed_ridge_target" if passed else "ridge_target_unresolved",
            "key_result": f"bounded Ridge/Tikhonov passing rows={passed}",
            "full_interval_validation": "only passed Ridge target rows count",
            "diagnostic_only": "sanity polynomials do not validate Ridge target",
            "supporting_output": "outputs/qsvt_stable_phase_validation_attempt",
        }
    if area == "ieee300_spectral_difficulty":
        ieee300 = frame[frame["case_name"] == "ieee300"]
        row = ieee300.iloc[0] if not ieee300.empty else frame.iloc[-1]
        full_error = float(row["full_interval_max_error"])
        return {
            "area": area,
            "status": "passed_full_interval" if full_error <= 1.0e-3 else "failed_full_interval",
            "key_result": (
                f"IEEE300 full_interval_error={full_error:.6g}; "
                "actual_singular_error="
                f"{float(row['actual_singular_values_max_error']):.6g}"
            ),
            "full_interval_validation": "reported separately from actual singular values",
            "diagnostic_only": "central interval rows are diagnostic only",
            "supporting_output": "outputs/qsvt_ieee300_spectral_difficulty",
        }
    if area == "ieee118_targeted_refinement":
        passed = int((frame["passed_1e_minus_3"] == True).sum())  # noqa: E712
        degrees = ",".join(str(int(value)) for value in frame["degree"].dropna())
        return {
            "area": area,
            "status": "passed" if passed else "not_passed",
            "key_result": f"tested_degrees={degrees}; passing_rows={passed}",
            "full_interval_validation": "targeted full-interval approximation diagnostic",
            "diagnostic_only": "degree budget limited to approved list",
            "supporting_output": "outputs/qsvt_ieee118_targeted_refinement",
        }
    return {
        "area": area,
        "status": "documented",
        "key_result": f"{len(frame)} diagnostic rows generated",
        "full_interval_validation": "not a main full-interval validation claim",
        "diagnostic_only": PRECONDITIONING_CAVEAT,
        "supporting_output": f"outputs/qsvt_{area}",
    }


def _consolidated_report(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    del config
    row_frame = pd.DataFrame(rows)
    verdict = "PARTIAL PASS"
    phase = _read_csv(PHASE_TARGET_SUMMARY_PATH)
    stable = _read_csv(STABLE_PHASE_SUMMARY_PATH)
    spectral = _read_csv(SPECTRAL_DIFFICULTY_SUMMARY_PATH)
    ieee118 = _read_csv(IEEE118_REFINEMENT_SUMMARY_PATH)
    spectrum = _read_csv(SPECTRUM_AWARE_SUMMARY_PATH)
    phase_text = _phase_summary_text(phase)
    stable_text = _stable_summary_text(stable)
    spectral_text = _spectral_summary_text(spectral)
    ieee118_text = _ieee118_summary_text(ieee118)
    spectrum_text = _spectrum_summary_text(spectrum)
    table = _markdown_table(row_frame, list(row_frame.columns))
    return f"""# QSVT Non-Brute-Force Refinement Summary

## 1. Executive Verdict

{verdict}. The requested diagnostics were added and generated. Some outputs may
show unresolved or failed rows; those failures are reported rather than hidden.

## 2. Phase-Target Failure Diagnosis

{phase_text}

## 3. Phase Validation Status

{stable_text}

## 4. IEEE300 Spectral Difficulty Explanation

{spectral_text}

## 5. IEEE118 Targeted Refinement Result

{ieee118_text}

## 6. Spectrum-Aware Diagnostics

{spectrum_text}

## 7. Full-Interval Validations

Only rows explicitly labeled as full-interval and passing `1e-3` are full-interval
validations. Actual-singular-value and restricted-interval diagnostics are
reported separately.

## 8. Restricted/Diagnostic-Only Results

{RESTRICTED_INTERVAL_CAVEAT}

## 9. Resource Implications

Degree and query-count proxies remain resource diagnostics only. They exclude
oracle synthesis, state preparation, fault-tolerant compilation, and readout
costs.

## 10. Safe Claims

- QSVT-compatible approximation diagnostics were extended without brute-force escalation.
- Phase-response diagnostics separate sanity checks from bounded Ridge/Tikhonov target status.
- IEEE300 full-interval and actual-singular-value errors are reported separately.
- Spectrum-aware/preconditioning rows are diagnostic only.

## 11. Claims to Avoid

- Do not claim quantum speedup, quantum advantage, hardware execution, or field-data validation.
- Do not claim IEEE300 passed full-interval 1e-3 unless the full-interval row passes.
- Do not claim restricted-interval diagnostics are full QSVT validation.
- Do not claim QSVT outperforms Ridge/Tikhonov under the same alpha and filter.

## 12. Remaining Limitations

- Phase synthesis depends on optional PennyLane availability and monomial coefficient conditioning.
- Polynomial approximation evidence is not a scalable block-encoding or hardware implementation.
- IEEE300 remains a difficult full-interval approximation case when its full-interval row fails.

## 13. Recommended Manuscript Wording

We report QSVT-compatible polynomial and scalar phase-response diagnostics for
controlled IEEE/PYPOWER weighted Jacobian matrices. The diagnostics separate
full-interval approximation error, actual-singular-value error, and restricted
spectrum-aware checks. Sanity-polynomial phase responses validate the scalar
response convention, while bounded Ridge/Tikhonov target phase validation is
reported only when the target response itself meets the declared tolerance.
These results support resource-aware feasibility analysis, not quantum speedup,
quantum advantage, hardware execution, field-data validation, or superiority over
Ridge/Tikhonov under the same regularization parameter.

## 14. Evidence Table

{table}

{NO_BRUTE_FORCE_CAVEAT}
"""


def _phase_summary_text(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "Phase-target failure diagnostics were not available."
    row = frame[frame["degree"] == 35].iloc[0] if (frame["degree"] == 35).any() else frame.iloc[-1]
    return (
        f"Failure class `{row['failure_class']}` with polynomial error "
        f"`{float(row['polynomial_approx_max_error']):.6g}` and phase-response "
        f"error `{float(row['phase_response_max_error']):.6g}`. Recommended fix: "
        f"{row['recommended_fix']}"
    )


def _stable_summary_text(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "Stable phase validation output was not available."
    ridge = frame[frame["target_type"] == "ridge_tikhonov_bounded_target"]
    passed = ridge[ridge["status"] == "passed"] if not ridge.empty else pd.DataFrame()
    if passed.empty:
        return PHASE_UNRESOLVED_MESSAGE
    degrees = ", ".join(str(int(value)) for value in passed["degree"])
    return f"Bounded Ridge/Tikhonov phase validation passed for degree(s): {degrees}."


def _spectral_summary_text(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "IEEE300 spectral difficulty output was not available."
    ieee300 = frame[frame["case_name"] == "ieee300"]
    row = ieee300.iloc[0] if not ieee300.empty else frame.iloc[-1]
    return (
        f"IEEE300 full-interval max error `{float(row['full_interval_max_error']):.6g}`, "
        f"actual-singular-value max error "
        f"`{float(row['actual_singular_values_max_error']):.6g}`, peak region "
        f"`{row['error_peak_region']}`. {row['diagnostic_interpretation']}"
    )


def _ieee118_summary_text(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "IEEE118 targeted refinement output was not available."
    passed = frame[frame["passed_1e_minus_3"] == True]  # noqa: E712
    if passed.empty:
        return "IEEE118 did not pass within the approved targeted degree list."
    row = passed.iloc[0]
    return (
        f"IEEE118 passed at degree `{int(row['degree'])}` with max error "
        f"`{float(row['max_pointwise_error']):.6g}`."
    )


def _spectrum_summary_text(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "Spectrum-aware output was not available."
    return f"{len(frame)} spectrum-aware diagnostic rows were generated. {PRECONDITIONING_CAVEAT}"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _resolve_phase_target_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase_target_failure_diagnostics",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "degrees": [5, 15, 25, 35, 101],
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 256,
        "target_tolerance": 1.0e-3,
        "angle_solver": "root-finding",
        "bound_validation_grid_size": 1001,
        "bound_tolerance": 1.0e-5,
        "phase_order": "original",
        "phase_sign": "phi",
        "phase_offset_rule": "none",
        "signal_operator_convention": "pennylane_rx_pcphase",
        "response_component": "real_u00",
        "coefficient_dynamic_range_limit": 1.0e12,
        "basis_conversion_error_limit": 1.0e-8,
        "force_dependency_missing": False,
    }
    if config:
        resolved.update(config)
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    return resolved


def _resolve_stable_phase_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_stable_phase_validation_attempt",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": 1.0e-2,
        "low_degree": 5,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 256,
        "sanity_grid_size": 101,
        "target_tolerance": 1.0e-3,
        "angle_solver": "root-finding",
        "bound_validation_grid_size": 1001,
        "bound_tolerance": 1.0e-5,
        "phase_order": "original",
        "phase_sign": "phi",
        "phase_offset_rule": "none",
        "signal_operator_convention": "pennylane_rx_pcphase",
        "response_component": "real_u00",
        "coefficient_dynamic_range_limit": 1.0e12,
        "force_dependency_missing": False,
    }
    if config:
        resolved.update(config)
    return resolved


def _resolve_spectral_difficulty_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_ieee300_spectral_difficulty",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": False,
        "alpha": 1.0e-2,
        "degree": 1001,
        "degree_by_case": {},
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
        "histogram_bins": 25,
    }
    if config:
        resolved.update(config)
    resolved["degree_by_case"] = {
        str(key): int(value) for key, value in dict(resolved["degree_by_case"]).items()
    }
    return resolved


def _resolve_spectrum_aware_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_spectrum_aware_diagnostics",
        "cases": ["ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": False,
        "alpha": 1.0e-2,
        "degree": 301,
        "degree_by_case": {},
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
    }
    if config:
        resolved.update(config)
    resolved["degree_by_case"] = {
        str(key): int(value) for key, value in dict(resolved["degree_by_case"]).items()
    }
    return resolved


def _resolve_ieee118_refinement_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_ieee118_targeted_refinement",
        "case_name": "ieee118",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": False,
        "alpha": 1.0e-2,
        "degrees": [1201, 1501],
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
        "target_tolerance": 1.0e-3,
    }
    if config:
        resolved.update(config)
    approved = [1201, 1501, 2001]
    degrees = [int(degree) for degree in resolved["degrees"]]
    invalid = [degree for degree in degrees if degree not in approved]
    if invalid:
        raise ValueError(f"unapproved IEEE118 targeted degrees: {invalid}")
    resolved["degrees"] = degrees
    return resolved


def _resolve_nonbruteforce_summary_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {"output_dir": "outputs/qsvt_nonbruteforce_refinement_summary"}
    if config:
        resolved.update(config)
    return resolved


def main_phase_target_failure(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose bounded Ridge QSVT phase-target failure")
    parser.parse_args(argv)
    run = diagnose_phase_target_failure()
    print(f"QSVT phase-target failure diagnostics complete: {run['output_dir']}")


def main_stable_phase_validation(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run stable QSVT phase-validation attempts")
    parser.parse_args(argv)
    run = run_stable_phase_validation_attempt()
    print(f"QSVT stable phase validation attempt complete: {run['output_dir']}")


def main_ieee300_spectral_difficulty(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose IEEE300 QSVT spectral difficulty")
    parser.parse_args(argv)
    run = diagnose_ieee300_spectral_difficulty()
    print(f"QSVT IEEE300 spectral difficulty diagnostics complete: {run['output_dir']}")


def main_spectrum_aware(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run QSVT spectrum-aware diagnostics")
    parser.parse_args(argv)
    run = run_spectrum_aware_diagnostics()
    print(f"QSVT spectrum-aware diagnostics complete: {run['output_dir']}")


def main_ieee118_refinement(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run IEEE118 targeted QSVT refinement")
    parser.parse_args(argv)
    run = run_ieee118_targeted_refinement()
    print(f"QSVT IEEE118 targeted refinement complete: {run['output_dir']}")


def main_nonbruteforce_summary(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build non-brute-force QSVT refinement summary")
    parser.parse_args(argv)
    run = build_nonbruteforce_refinement_summary()
    print(f"QSVT non-brute-force refinement summary complete: {run['output_dir']}")
