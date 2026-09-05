from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_DEGREES,
    DEFAULT_EPSILON,
    RESOURCE_CAVEAT,
    bounded_scaling_C,
    build_engineering_system,
    direction_metrics,
    estimate_degree_and_queries,
    required_case_name,
    ridge_svd_solution,
    singular_summary,
)
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.polynomial import (
    fit_odd_regularized_polynomial,
    regularized_filter_on_normalized_domain,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_SELECTED_ALPHA_GRID = [1.0e-4, 1.0e-2, 1.0]


def run_alpha_resource_sensitivity(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    singular_values = system.singular_values()
    summary = singular_summary(system.H_tilde)
    rows = [
        _alpha_row(
            alpha=float(alpha),
            system=system,
            singular_values=singular_values,
            condition_number=float(summary["condition_number"]),
            case_name=required_case_name(system),
            matrix_source=matrix_source,
            epsilon=float(resolved["epsilon"]),
            degrees=list(resolved["degrees"]),
        )
        for alpha in resolved["alpha_grid"]
    ]
    frame = pd.DataFrame(rows)
    polynomial_rows = selected_alpha_polynomial_validation_rows(
        singular_values,
        case_name=required_case_name(system),
        matrix_source=matrix_source,
        alpha_grid=list(resolved["selected_alpha_grid"]),
        epsilon=float(resolved["polynomial_error_tolerance"]),
        degrees=list(resolved["degrees"]),
        grid_size=int(resolved["polynomial_grid_size"]),
    )
    polynomial_frame = pd.DataFrame(polynomial_rows)
    csv_path = output_dir / "alpha_resource_sensitivity.csv"
    json_path = output_dir / "alpha_resource_sensitivity.json"
    polynomial_csv_path = output_dir / "selected_alpha_polynomial_validation.csv"
    polynomial_json_path = output_dir / "selected_alpha_polynomial_validation.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, {"rows": rows})
    polynomial_frame.to_csv(polynomial_csv_path, index=False)
    write_json(polynomial_json_path, {"rows": polynomial_rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "alpha_resource_sensitivity_csv": str(csv_path),
            "alpha_resource_sensitivity_json": str(json_path),
            "selected_alpha_polynomial_validation_csv": str(polynomial_csv_path),
            "selected_alpha_polynomial_validation_json": str(polynomial_json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "alpha_resource_sensitivity_csv": csv_path,
            "alpha_resource_sensitivity_json": json_path,
            "selected_alpha_polynomial_validation_csv": polynomial_csv_path,
            "selected_alpha_polynomial_validation_json": polynomial_json_path,
            "manifest": manifest_path,
        },
    }


def selected_alpha_polynomial_validation_rows(
    singular_values: np.ndarray,
    *,
    case_name: str,
    matrix_source: str,
    alpha_grid: list[float],
    epsilon: float,
    degrees: list[int],
    grid_size: int,
) -> list[dict[str, Any]]:
    values = np.asarray(singular_values, dtype=np.float64)
    positive = values[values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("singular_values must contain at least one positive value")
    beta = float(np.max(positive))
    domain_min = max(float(np.min(positive) / beta), np.finfo(float).eps)
    domain_max = 1.0
    candidate_degrees = sorted({_as_odd_degree(int(degree)) for degree in degrees})
    rows = []
    for alpha in alpha_grid:
        row = _selected_alpha_polynomial_row(
            alpha=float(alpha),
            case_name=case_name,
            matrix_source=matrix_source,
            beta=beta,
            domain_min=domain_min,
            domain_max=domain_max,
            epsilon=float(epsilon),
            candidate_degrees=candidate_degrees,
            grid_size=grid_size,
        )
        rows.append(row)
    return rows


def _selected_alpha_polynomial_row(
    *,
    alpha: float,
    case_name: str,
    matrix_source: str,
    beta: float,
    domain_min: float,
    domain_max: float,
    epsilon: float,
    candidate_degrees: list[int],
    grid_size: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for polynomial_degree in candidate_degrees:
        row = _polynomial_validation_row(
            alpha=alpha,
            case_name=case_name,
            matrix_source=matrix_source,
            beta=beta,
            domain_min=domain_min,
            domain_max=domain_max,
            epsilon=epsilon,
            polynomial_degree=polynomial_degree,
            candidate_degrees=candidate_degrees,
            grid_size=grid_size,
        )
        if best is None or (
            row["max_bounded_pointwise_target_error"] < best["max_bounded_pointwise_target_error"]
        ):
            best = row
        if row["passed"]:
            return row
    if best is None:
        raise ValueError("candidate_degrees must be non-empty")
    return best


def _polynomial_validation_row(
    *,
    alpha: float,
    case_name: str,
    matrix_source: str,
    beta: float,
    domain_min: float,
    domain_max: float,
    epsilon: float,
    polynomial_degree: int,
    candidate_degrees: list[int],
    grid_size: int,
) -> dict[str, Any]:
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha,
        block_encoding_normalization=beta,
        degree=polynomial_degree,
        domain_min=domain_min,
        domain_max=domain_max,
        grid_size=max(grid_size, polynomial_degree + 2),
    )
    polynomial = Polynomial(np.asarray(approximation.power_coefficients, dtype=np.float64))
    grid = np.linspace(domain_min, domain_max, max(grid_size, polynomial_degree + 2))
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=beta,
    )
    observed = polynomial(grid)
    C = float(approximation.scale_factor)
    max_error = float(np.max(np.abs(observed - target)))
    max_bounded_error = float(np.max(np.abs(observed / C - target / C)))
    return {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "alpha": alpha,
        "bounded_scaling_C": C,
        "polynomial_degree": polynomial_degree,
        "max_pointwise_target_error": max_error,
        "max_bounded_pointwise_target_error": max_bounded_error,
        "query_count": int(2 * polynomial_degree + 1),
        "passed": bool(max_bounded_error <= epsilon),
        "polynomial_error_tolerance": epsilon,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "candidate_degrees": ",".join(str(degree) for degree in candidate_degrees),
        "validation_type": "bounded_odd_polynomial_fit",
        "phase_validation_backend": "not_run_backend_free_polynomial_validation",
        "caveat": (
            "Polynomial validation only; no QSVT phase synthesis or hardware "
            "execution is claimed for this selected-alpha report."
        ),
    }


def _alpha_row(
    *,
    alpha: float,
    system: Any,
    singular_values: np.ndarray,
    condition_number: float,
    case_name: str,
    matrix_source: str,
    epsilon: float,
    degrees: list[int],
) -> dict[str, Any]:
    gains = ridge_filter(singular_values, alpha=alpha)
    C = bounded_scaling_C(singular_values, alpha=alpha)
    degree = estimate_degree_and_queries(
        singular_values,
        alpha=alpha,
        epsilon=epsilon,
        degrees=degrees,
    )
    ridge_solution = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=alpha)
    qsvt_target_solution = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=alpha)
    equivalence = direction_metrics(ridge_solution, qsvt_target_solution)
    return {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "alpha": alpha,
        "condition_number": condition_number,
        "max_filter_gain": float(np.max(np.abs(gains))),
        "bounded_scaling_C": C,
        "max_bounded_filter_value": float(np.max(np.abs(gains / C))),
        "estimated_qsvt_degree": int(degree["qsvt_degree_estimate"]),
        "estimated_query_count": int(degree["query_count_estimate"]),
        "ridge_rmse_if_available": system.rmse(ridge_solution),
        "ridge_residual_if_available": system.residual_norm(ridge_solution),
        "qsvt_target_relative_error_vs_ridge_if_available": equivalence["relative_error"],
        "resource_caveat": RESOURCE_CAVEAT,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_alpha_resource_sensitivity",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha_grid": DEFAULT_ALPHA_GRID,
        "selected_alpha_grid": DEFAULT_SELECTED_ALPHA_GRID,
        "epsilon": DEFAULT_EPSILON,
        "polynomial_error_tolerance": DEFAULT_EPSILON,
        "polynomial_grid_size": 2048,
        "degrees": DEFAULT_DEGREES,
    }
    if config:
        resolved.update(config)
    resolved["alpha_grid"] = [float(alpha) for alpha in resolved["alpha_grid"]]
    if any(alpha <= 0.0 for alpha in resolved["alpha_grid"]):
        raise ValueError("alpha_grid values must be positive")
    resolved["selected_alpha_grid"] = [float(alpha) for alpha in resolved["selected_alpha_grid"]]
    if any(alpha <= 0.0 for alpha in resolved["selected_alpha_grid"]):
        raise ValueError("selected_alpha_grid values must be positive")
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    if float(resolved["polynomial_error_tolerance"]) <= 0.0:
        raise ValueError("polynomial_error_tolerance must be positive")
    if int(resolved["polynomial_grid_size"]) <= max(resolved["degrees"]) + 1:
        raise ValueError("polynomial_grid_size must exceed max degree + 1")
    return resolved


def _as_odd_degree(degree: int) -> int:
    if degree < 1:
        return 1
    return degree if degree % 2 == 1 else degree + 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT alpha/resource sensitivity report")
    parser.parse_args(argv)
    run = run_alpha_resource_sensitivity()
    print(f"QSVT alpha/resource sensitivity complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
