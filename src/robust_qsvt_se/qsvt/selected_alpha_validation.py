from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, required_case_name
from robust_qsvt_se.qsvt.polynomial import (
    fit_odd_regularized_polynomial,
    regularized_filter_on_normalized_domain,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_ALPHA_GRID = [1.0e-4, 1.0e-2, 1.0]
DEFAULT_DEGREE_GRID = [5, 11, 21, 35, 51, 71, 101, 151, 201]
PHASE_VALIDATION_CAVEAT = (
    "Polynomial approximation fallback, not full phase synthesis. This validates a "
    "bounded target approximation path only; it is not hardware execution and does "
    "not demonstrate quantum speedup, quantum advantage, or QSVT superiority over "
    "Ridge/Tikhonov under the same alpha."
)


def run_selected_alpha_phase_validation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source, source_note = _build_system_with_fallback(resolved)
    H_tilde = np.asarray(system.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H_tilde, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected-alpha validation requires at least one positive singular value")

    beta = float(np.max(positive))
    normalized_singular_values = positive / beta
    domain_min = max(float(np.min(normalized_singular_values)), np.finfo(float).eps)
    domain_max = 1.0

    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for alpha in resolved["alpha"]:
        summary, values = _validate_alpha(
            alpha=float(alpha),
            beta=beta,
            normalized_singular_values=normalized_singular_values,
            domain_min=domain_min,
            domain_max=domain_max,
            candidate_degrees=list(resolved["degrees"]),
            grid_size=int(resolved["grid_size"]),
            tolerance=float(resolved["tolerance"]),
            case_name=required_case_name(system),
            matrix_source=matrix_source,
            matrix_shape=f"{H_tilde.shape[0]}x{H_tilde.shape[1]}",
            source_note=source_note,
        )
        summary_rows.append(summary)
        value_rows.extend(values)
        error_rows.extend(
            {
                "case_name": row["case_name"],
                "matrix_source": row["matrix_source"],
                "alpha": row["alpha"],
                "polynomial_degree": row["polynomial_degree"],
                "evaluation_kind": row["evaluation_kind"],
                "sigma_normalized": row["sigma_normalized"],
                "pointwise_error": row["pointwise_error"],
            }
            for row in values
        )

    summary_frame = pd.DataFrame(summary_rows)
    value_frame = pd.DataFrame(value_rows)
    error_frame = pd.DataFrame(error_rows)
    summary_csv = output_dir / "phase_validation_summary.csv"
    summary_json = output_dir / "phase_validation_summary.json"
    errors_csv = output_dir / "pointwise_errors.csv"
    values_csv = output_dir / "target_and_approx_values.csv"
    summary_frame.to_csv(summary_csv, index=False)
    value_frame.to_csv(values_csv, index=False)
    error_frame.to_csv(errors_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "phase_validation_summary_csv": str(summary_csv),
            "phase_validation_summary_json": str(summary_json),
            "pointwise_errors_csv": str(errors_csv),
            "target_and_approx_values_csv": str(values_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "phase_validation_summary_csv": summary_csv,
            "phase_validation_summary_json": summary_json,
            "pointwise_errors_csv": errors_csv,
            "target_and_approx_values_csv": values_csv,
            "manifest": manifest_path,
        },
    }


def _validate_alpha(
    *,
    alpha: float,
    beta: float,
    normalized_singular_values: np.ndarray,
    domain_min: float,
    domain_max: float,
    candidate_degrees: list[int],
    grid_size: int,
    tolerance: float,
    case_name: str,
    matrix_source: str,
    matrix_shape: str,
    source_note: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_summary: dict[str, Any] | None = None
    best_values: list[dict[str, Any]] = []
    for degree in candidate_degrees:
        summary, values = _evaluate_degree(
            alpha=alpha,
            beta=beta,
            normalized_singular_values=normalized_singular_values,
            domain_min=domain_min,
            domain_max=domain_max,
            degree=int(degree),
            grid_size=grid_size,
            tolerance=tolerance,
            case_name=case_name,
            matrix_source=matrix_source,
            matrix_shape=matrix_shape,
            source_note=source_note,
        )
        if best_summary is None or (
            summary["max_pointwise_target_error"] < best_summary["max_pointwise_target_error"]
        ):
            best_summary = summary
            best_values = values
        if bool(summary["passed"]):
            return summary, values
    if best_summary is None:
        raise ValueError("candidate degree grid must be non-empty")
    return best_summary, best_values


def _evaluate_degree(
    *,
    alpha: float,
    beta: float,
    normalized_singular_values: np.ndarray,
    domain_min: float,
    domain_max: float,
    degree: int,
    grid_size: int,
    tolerance: float,
    case_name: str,
    matrix_source: str,
    matrix_shape: str,
    source_note: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha,
        block_encoding_normalization=beta,
        degree=_as_odd_degree(degree),
        domain_min=domain_min,
        domain_max=domain_max,
        grid_size=max(grid_size, degree + 2),
    )
    polynomial = Polynomial(np.asarray(approximation.power_coefficients, dtype=np.float64))
    grid = np.linspace(domain_min, domain_max, max(grid_size, degree + 2), dtype=np.float64)
    evaluation_points = np.concatenate([grid, normalized_singular_values])
    evaluation_kind = ["grid"] * grid.size + [
        "actual_singular_value"
    ] * normalized_singular_values.size
    target = regularized_filter_on_normalized_domain(
        evaluation_points,
        alpha=alpha,
        block_encoding_normalization=beta,
    )
    observed = polynomial(evaluation_points)
    bounded_scaling = max(1.0, float(np.max(np.abs(target))))
    target_bounded = target / bounded_scaling
    observed_bounded = observed / bounded_scaling
    errors = np.abs(observed_bounded - target_bounded)
    grid_errors = errors[: grid.size]
    values = [
        {
            "case_name": case_name,
            "matrix_source": matrix_source,
            "alpha": float(alpha),
            "polynomial_degree": int(approximation.degree),
            "evaluation_kind": kind,
            "sigma_normalized": float(sigma),
            "target_value": float(raw_target),
            "approximation_value": float(raw_observed),
            "target_bounded_value": float(bounded_target),
            "approximation_bounded_value": float(bounded_observed),
            "pointwise_error": float(error),
        }
        for kind, sigma, raw_target, raw_observed, bounded_target, bounded_observed, error in zip(
            evaluation_kind,
            evaluation_points,
            target,
            observed,
            target_bounded,
            observed_bounded,
            errors,
            strict=True,
        )
    ]
    max_error = float(np.max(errors))
    summary = {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "matrix_shape": matrix_shape,
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma_min": float(np.min(normalized_singular_values) * beta),
        "sigma_max": float(np.max(normalized_singular_values) * beta),
        "normalized_sigma_min": float(np.min(normalized_singular_values)),
        "normalized_sigma_max": float(np.max(normalized_singular_values)),
        "bounded_scaling_C": float(bounded_scaling),
        "max_unbounded_filter_value": float(np.max(np.abs(target))),
        "max_bounded_filter_value": float(np.max(np.abs(target_bounded))),
        "max_bounded_approx_value": float(np.max(np.abs(observed_bounded))),
        "approximation_method": "bounded odd-polynomial approximation fallback",
        "polynomial_degree": int(approximation.degree),
        "phase_count_if_available": None,
        "query_count": int(2 * approximation.degree + 1),
        "query_count_estimate": int(2 * approximation.degree + 1),
        "max_pointwise_target_error": max_error,
        "mean_pointwise_target_error": float(np.mean(errors)),
        "rms_pointwise_target_error": float(np.sqrt(np.mean(errors**2))),
        "max_grid_pointwise_target_error": float(np.max(grid_errors)),
        "grid_size": int(grid.size),
        "actual_singular_value_count": int(normalized_singular_values.size),
        "tolerance": float(tolerance),
        "passed": bool(max_error <= tolerance),
        "source_note": source_note,
        "caveat": PHASE_VALIDATION_CAVEAT,
    }
    return summary, values


def _build_system_with_fallback(config: dict[str, Any]) -> tuple[Any, str, str]:
    try:
        system, matrix_source = build_engineering_system(config)
    except Exception as exc:
        if not bool(config["fallback_to_synthetic"]):
            raise
        fallback = dict(config)
        fallback["matrix_source"] = "synthetic"
        system, matrix_source = build_engineering_system(fallback)
        return system, f"{matrix_source}_fallback", f"synthetic fallback after: {exc}"
    return system, matrix_source, "primary matrix source"


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_selected_alpha_phase_validation",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": DEFAULT_ALPHA_GRID,
        "degrees": DEFAULT_DEGREE_GRID,
        "grid_size": 2048,
        "tolerance": 1.0e-3,
        "fallback_to_synthetic": True,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    if any(alpha <= 0.0 for alpha in resolved["alpha"]):
        raise ValueError("alpha values must be positive")
    resolved["degrees"] = sorted({_as_odd_degree(int(degree)) for degree in resolved["degrees"]})
    if not resolved["degrees"]:
        raise ValueError("degrees must be non-empty")
    if int(resolved["grid_size"]) < 2048:
        raise ValueError("grid_size must be at least 2048")
    if int(resolved["grid_size"]) <= max(resolved["degrees"]) + 1:
        raise ValueError("grid_size must exceed max degree + 1")
    if float(resolved["tolerance"]) <= 0.0:
        raise ValueError("tolerance must be positive")
    return resolved


def _as_odd_degree(degree: int) -> int:
    if degree < 1:
        return 1
    return degree if degree % 2 == 1 else degree + 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run selected-alpha QSVT polynomial/phase validation"
    )
    parser.parse_args(argv)
    run = run_selected_alpha_phase_validation()
    print(f"QSVT selected-alpha validation complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
