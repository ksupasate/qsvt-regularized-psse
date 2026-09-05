from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.subproblem_sweep import select_subproblem
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.utils.io import write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_GRID = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
EPSILON_TARGETS = [1.0e-2, 1.0e-3, 1.0e-4]
DEGREE_GRID = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 201]
DEFAULT_SUBPROBLEMS = [
    {"case_name": "ieee14", "subproblem_size": 4, "selection_mode": "high_leverage"},
    {"case_name": "ieee14", "subproblem_size": 8, "selection_mode": "high_leverage"},
    {"case_name": "ieee30", "subproblem_size": 8, "selection_mode": "high_leverage"},
    {"case_name": "ieee57", "subproblem_size": 8, "selection_mode": "high_leverage"},
    {"case_name": "ieee57", "subproblem_size": 16, "selection_mode": "high_leverage"},
    {"case_name": "ieee118", "subproblem_size": 16, "selection_mode": "high_leverage"},
]

DEGREE_RESULTS_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "matrix_shape",
    "weighted_status",
    "alpha",
    "epsilon_target",
    "requested_degree",
    "degree",
    "degree_grid_index",
    "degree_adjustment_reason",
    "polynomial_parity",
    "C_alpha",
    "C_alpha_formula",
    "beta",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "max_approximation_error_on_dense_grid",
    "max_approximation_error_on_actual_singular_values",
    "mean_approximation_error_on_actual_singular_values",
    "max_polynomial_abs_on_unit_domain",
    "meets_epsilon_on_actual_singular_values",
    "phase_synthesis_status",
    "phase_count",
    "phase_synthesis_backend",
    "phase_synthesis_failure_reason",
    "nominal_phase_parameter_count",
    "gate_simulation_status",
    "gate_versus_polynomial_error",
    "gate_versus_ridge_target_error",
    "output_update_relative_error_against_matched_ridge",
    "residual_norm_ratio_against_no_update_baseline",
    "ridge_residual_norm_ratio_against_no_update_baseline",
    "success_probability",
    "success_probability_type",
    "run_status",
    "failure_mode",
    "failure_reason",
]

SUMMARY_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "matrix_shape",
    "alpha",
    "epsilon_target",
    "required_degree",
    "best_available_degree",
    "best_error_on_actual_singular_values",
    "best_dense_grid_error",
    "degree_grid",
    "degree_selection_status",
]


@dataclass(frozen=True, slots=True)
class SweepSubproblem:
    H_tilde: np.ndarray
    r_tilde: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolynomialDiagnostics:
    degree: int
    parity: str
    C_alpha: float
    chebyshev_coefficients: np.ndarray
    dense_error: float
    actual_error_max: float
    actual_error_mean: float
    max_polynomial_abs: float
    output_relative_error: float
    residual_ratio: float
    ridge_residual_ratio: float
    success_probability: float


def run_degree_alpha_precision_sweep(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    status = "completed"
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = paths["degree"]
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    rows: list[dict[str, Any]] = []
    coefficient_cache: dict[tuple[str, int, str, float, int], np.ndarray] = {}
    for spec in resolved["subproblems"]:
        try:
            subproblem = load_sweep_subproblem(spec, seed=int(resolved["seed"]))
        except Exception as exc:
            rows.extend(_subproblem_failure_rows(spec, resolved, exc))
            continue
        for alpha in resolved["alpha_grid"]:
            for degree_index, requested_degree in enumerate(resolved["degree_grid"]):
                degree_rows, coefficients = evaluate_degree_candidate(
                    subproblem=subproblem,
                    alpha=float(alpha),
                    epsilon_targets=resolved["epsilon_targets"],
                    requested_degree=int(requested_degree),
                    degree_grid_index=degree_index,
                    dense_grid_size=int(resolved["dense_grid_size"]),
                )
                rows.extend(degree_rows)
                if coefficients is not None:
                    key = _phase_key_from_row(degree_rows[0])
                    coefficient_cache[key] = coefficients

    summary_frame = select_required_degrees(pd.DataFrame(rows, columns=DEGREE_RESULTS_COLUMNS))
    _apply_phase_synthesis_budget(
        rows=rows,
        summary_frame=summary_frame,
        coefficient_cache=coefficient_cache,
        max_attempts=int(resolved["phase_synthesis_max_attempts"]),
        bound_tolerance=float(resolved["phase_synthesis_bound_tolerance"]),
    )

    results_frame = pd.DataFrame(rows, columns=DEGREE_RESULTS_COLUMNS)
    summary_frame = select_required_degrees(results_frame)

    results_csv = output_dir / "degree_alpha_precision_sweep_results.csv"
    metadata_json = output_dir / "degree_alpha_precision_sweep_metadata.json"
    summary_csv = tables_dir / "table_degree_alpha_precision_summary.csv"
    figure_path = figures_dir / "figure_degree_alpha_precision_tradeoff.png"
    report_path = reports_dir / "degree_alpha_precision_sweep_report.md"

    results_frame.to_csv(results_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    _plot_degree_tradeoff(summary_frame, figure_path)
    report_path.write_text(
        _degree_report_markdown(resolved, results_frame, summary_frame, results_csv, summary_csv),
        encoding="utf-8",
    )
    ended_at = utc_timestamp()
    artifacts = {
        "results_csv": str(results_csv),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "figure": str(figure_path),
        "report": str(report_path),
    }
    write_json(
        metadata_json,
        reproducibility_metadata(
            config=resolved,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            command=current_command(),
            artifacts=artifacts,
        ),
    )
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results_frame,
        "summary": summary_frame,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def bounded_ridge_normalization_C(alpha: float, beta: float) -> float:
    """Analytic bound for beta*x/(beta^2*x^2 + alpha) on x in [0, 1]."""

    alpha_value = float(alpha)
    beta_value = float(beta)
    if alpha_value <= 0.0:
        raise ValueError("alpha must be positive")
    if beta_value <= 0.0 or not np.isfinite(beta_value):
        raise ValueError("beta must be positive and finite")
    stationary_point = np.sqrt(alpha_value) / beta_value
    if stationary_point <= 1.0:
        peak = 1.0 / (2.0 * np.sqrt(alpha_value))
    else:
        peak = beta_value / (beta_value**2 + alpha_value)
    return float(max(1.0, peak))


def bounded_ridge_target(
    normalized_sigma: np.ndarray | list[float],
    *,
    alpha: float,
    beta: float,
    C_alpha: float | None = None,
) -> np.ndarray:
    values = np.asarray(normalized_sigma, dtype=np.float64)
    C_value = bounded_ridge_normalization_C(alpha, beta) if C_alpha is None else float(C_alpha)
    physical_sigma = float(beta) * values
    return physical_sigma / (physical_sigma**2 + float(alpha)) / C_value


def qsvt_odd_degree(requested_degree: int) -> tuple[int, str]:
    value = int(requested_degree)
    if value < 1:
        raise ValueError("degree must be positive")
    if value % 2 == 1:
        return value, ""
    return value + 1, "even requested degree advanced to next odd degree for odd QSVT parity"


def fit_bounded_ridge_polynomial(
    *,
    alpha: float,
    beta: float,
    degree: int,
) -> tuple[Chebyshev, np.ndarray, float]:
    used_degree, _ = qsvt_odd_degree(degree)
    C_alpha = bounded_ridge_normalization_C(alpha, beta)

    def target(points: np.ndarray) -> np.ndarray:
        return bounded_ridge_target(points, alpha=alpha, beta=beta, C_alpha=C_alpha)

    polynomial = Chebyshev.interpolate(target, deg=used_degree, domain=[-1.0, 1.0])
    coefficients = np.asarray(polynomial.coef, dtype=np.float64)
    coefficients[0::2] = 0.0
    return Chebyshev(coefficients, domain=[-1.0, 1.0]), coefficients, C_alpha


def evaluate_degree_candidate(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    epsilon_targets: list[float],
    requested_degree: int,
    degree_grid_index: int,
    dense_grid_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    used_degree, adjustment = qsvt_odd_degree(requested_degree)
    base = _base_row(
        subproblem=subproblem,
        alpha=alpha,
        requested_degree=requested_degree,
        used_degree=used_degree,
        degree_grid_index=degree_grid_index,
        degree_adjustment_reason=adjustment,
    )
    try:
        diagnostics = _evaluate_polynomial_diagnostics(
            subproblem=subproblem,
            alpha=alpha,
            degree=used_degree,
            dense_grid_size=dense_grid_size,
        )
    except Exception as exc:
        return [
            {
                **base,
                "epsilon_target": float(epsilon),
                "run_status": "failed",
                "failure_mode": "polynomial_approximation_failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "meets_epsilon_on_actual_singular_values": False,
            }
            for epsilon in epsilon_targets
        ], None

    rows = []
    for epsilon in epsilon_targets:
        rows.append(
            {
                **base,
                "epsilon_target": float(epsilon),
                "polynomial_parity": diagnostics.parity,
                "C_alpha": diagnostics.C_alpha,
                "C_alpha_formula": "max(1, max_{x in [0,1]} beta*x/(beta^2*x^2+alpha))",
                "max_approximation_error_on_dense_grid": diagnostics.dense_error,
                "max_approximation_error_on_actual_singular_values": diagnostics.actual_error_max,
                "mean_approximation_error_on_actual_singular_values": diagnostics.actual_error_mean,
                "max_polynomial_abs_on_unit_domain": diagnostics.max_polynomial_abs,
                "meets_epsilon_on_actual_singular_values": bool(
                    diagnostics.actual_error_max <= float(epsilon)
                ),
                "phase_synthesis_status": "skipped_not_attempted",
                "phase_count": 0,
                "phase_synthesis_backend": "",
                "phase_synthesis_failure_reason": "",
                "nominal_phase_parameter_count": used_degree + 1,
                "gate_simulation_status": "skipped_not_attempted",
                "gate_versus_polynomial_error": np.nan,
                "gate_versus_ridge_target_error": np.nan,
                "output_update_relative_error_against_matched_ridge": (
                    diagnostics.output_relative_error
                ),
                "residual_norm_ratio_against_no_update_baseline": diagnostics.residual_ratio,
                "ridge_residual_norm_ratio_against_no_update_baseline": (
                    diagnostics.ridge_residual_ratio
                ),
                "success_probability": diagnostics.success_probability,
                "success_probability_type": "bounded_polynomial_postselection_proxy",
                "run_status": "completed",
                "failure_mode": "",
                "failure_reason": "",
            }
        )
    return rows, diagnostics.chebyshev_coefficients


def select_required_degrees(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    group_columns = [
        "case_name",
        "subproblem_size",
        "selection_criterion",
        "matrix_shape",
        "alpha",
        "epsilon_target",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(group_columns, dropna=False):
        ordered = group.sort_values(["degree_grid_index", "degree"])
        completed = ordered[
            (ordered["run_status"] == "completed")
            & np.isfinite(ordered["max_approximation_error_on_actual_singular_values"])
        ]
        required_degree: int | None = None
        status = "no_degree_met_target"
        if not completed.empty:
            passed = completed[
                completed["max_approximation_error_on_actual_singular_values"]
                <= completed["epsilon_target"]
            ]
            if not passed.empty:
                first = passed.iloc[0]
                required_degree = int(first["degree"])
                status = "met_target"
            best_index = completed["max_approximation_error_on_actual_singular_values"].idxmin()
            best = completed.loc[best_index]
            best_degree = int(best["degree"])
            best_actual = float(best["max_approximation_error_on_actual_singular_values"])
            best_dense = float(best["max_approximation_error_on_dense_grid"])
        else:
            best_degree = None
            best_actual = np.nan
            best_dense = np.nan
            status = "all_rows_failed"
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "required_degree": required_degree,
                "best_available_degree": best_degree,
                "best_error_on_actual_singular_values": best_actual,
                "best_dense_grid_error": best_dense,
                "degree_grid": " ".join(str(int(value)) for value in ordered["degree"].unique()),
                "degree_selection_status": status,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def load_sweep_subproblem(spec: dict[str, Any], *, seed: int) -> SweepSubproblem:
    if "matrix" in spec:
        matrix = np.asarray(spec["matrix"], dtype=np.float64)
        r_tilde = np.asarray(spec.get("r_tilde", np.ones(matrix.shape[0])), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("synthetic matrix must be nonempty and two-dimensional")
        if r_tilde.shape != (matrix.shape[0],):
            raise ValueError("synthetic r_tilde length must match matrix rows")
        size = int(spec.get("subproblem_size", min(matrix.shape)))
        return SweepSubproblem(
            H_tilde=matrix,
            r_tilde=r_tilde,
            metadata={
                "case_name": str(spec.get("case_name", "synthetic")),
                "subproblem_size": size,
                "selection_mode": str(spec.get("selection_mode", "synthetic_fixed")),
                "matrix_source": "synthetic_config",
                "selected_measurement_indices": list(range(matrix.shape[0])),
                "selected_state_indices": list(range(matrix.shape[1])),
            },
        )

    case_name = str(spec["case_name"])
    selection_mode = str(spec.get("selection_mode", "high_leverage"))
    size = int(spec["subproblem_size"])
    case_source = str(spec.get("case_source", "pypower"))
    system, matrix_source = build_engineering_system(
        {
            "case_name": case_name,
            "case_source": case_source,
            "matrix_source": f"{case_name}_ac_weighted_jacobian",
            "seed": int(spec.get("seed", seed)),
            "measurement": dict(spec.get("measurement", {})),
            "linearization": dict(spec.get("linearization", {})),
        }
    )
    selected = select_subproblem(
        system=system,
        matrix_source=matrix_source,
        case=case_name,
        model="ac_linearized",
        submatrix_size=size,
        selection_mode=selection_mode,
        seed=int(spec.get("seed", seed)),
    )
    metadata = dict(selected.metadata)
    metadata["case_name"] = case_name
    metadata["subproblem_size"] = size
    metadata["selection_mode"] = selection_mode
    return SweepSubproblem(
        H_tilde=np.asarray(selected.H_tilde, dtype=np.float64),
        r_tilde=np.asarray(selected.r_tilde, dtype=np.float64),
        metadata=metadata,
    )


def _evaluate_polynomial_diagnostics(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    degree: int,
    dense_grid_size: int,
) -> PolynomialDiagnostics:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(H, full_matrices=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("subproblem must have at least one positive singular value")
    beta = float(np.max(positive))
    polynomial, coefficients, C_alpha = fit_bounded_ridge_polynomial(
        alpha=alpha,
        beta=beta,
        degree=degree,
    )
    dense_grid = np.linspace(0.0, 1.0, max(int(dense_grid_size), degree + 2), dtype=np.float64)
    dense_target = bounded_ridge_target(
        dense_grid,
        alpha=alpha,
        beta=beta,
        C_alpha=C_alpha,
    )
    dense_poly = polynomial(dense_grid)
    dense_error = float(np.max(np.abs(dense_poly - dense_target)))

    actual_normalized = singular_values / beta
    actual_target = bounded_ridge_target(
        actual_normalized,
        alpha=alpha,
        beta=beta,
        C_alpha=C_alpha,
    )
    actual_poly = polynomial(actual_normalized)
    actual_errors = np.abs(actual_poly - actual_target)

    ridge_filter = singular_values / (singular_values**2 + float(alpha))
    ridge_update = Vt.T @ (ridge_filter * (U.T @ r))
    polynomial_update = Vt.T @ ((C_alpha * actual_poly) * (U.T @ r))
    ridge_norm = float(np.linalg.norm(ridge_update))
    update_relative_error = (
        float(np.linalg.norm(polynomial_update - ridge_update))
        if ridge_norm == 0.0
        else float(np.linalg.norm(polynomial_update - ridge_update) / ridge_norm)
    )
    no_update_residual = float(np.linalg.norm(r))
    if no_update_residual == 0.0:
        residual_ratio = 0.0
        ridge_residual_ratio = 0.0
    else:
        residual_ratio = float(np.linalg.norm(H @ polynomial_update - r) / no_update_residual)
        ridge_residual_ratio = float(np.linalg.norm(H @ ridge_update - r) / no_update_residual)
    bounded_vector = Vt.T @ (actual_poly * (U.T @ r))
    success_probability = (
        0.0
        if no_update_residual == 0.0
        else float(np.linalg.norm(bounded_vector) ** 2 / no_update_residual**2)
    )
    unit_grid = np.linspace(-1.0, 1.0, max(4097, degree * 16 + 1), dtype=np.float64)
    max_abs = float(np.max(np.abs(polynomial(unit_grid))))
    return PolynomialDiagnostics(
        degree=int(degree),
        parity="odd",
        C_alpha=C_alpha,
        chebyshev_coefficients=coefficients,
        dense_error=dense_error,
        actual_error_max=float(np.max(actual_errors)),
        actual_error_mean=float(np.mean(actual_errors)),
        max_polynomial_abs=max_abs,
        output_relative_error=update_relative_error,
        residual_ratio=residual_ratio,
        ridge_residual_ratio=ridge_residual_ratio,
        success_probability=success_probability,
    )


def _base_row(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    requested_degree: int,
    used_degree: int,
    degree_grid_index: int,
    degree_adjustment_reason: str,
) -> dict[str, Any]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    beta = float(np.max(positive)) if positive.size else np.nan
    condition = float(np.max(positive) / np.min(positive)) if positive.size else np.inf
    metadata = subproblem.metadata
    return {
        "case_name": str(metadata.get("case_name", "unknown")),
        "subproblem_size": int(metadata.get("subproblem_size", min(H.shape))),
        "selection_criterion": str(metadata.get("selection_mode", "unknown")),
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "weighted_status": "weighted_jacobian_R_minus_half_H",
        "alpha": float(alpha),
        "epsilon_target": np.nan,
        "requested_degree": int(requested_degree),
        "degree": int(used_degree),
        "degree_grid_index": int(degree_grid_index),
        "degree_adjustment_reason": degree_adjustment_reason,
        "polynomial_parity": "",
        "C_alpha": np.nan,
        "C_alpha_formula": "",
        "beta": beta,
        "sigma_min": float(np.min(positive)) if positive.size else np.nan,
        "sigma_max": float(np.max(positive)) if positive.size else np.nan,
        "condition_number": condition,
        "max_approximation_error_on_dense_grid": np.nan,
        "max_approximation_error_on_actual_singular_values": np.nan,
        "mean_approximation_error_on_actual_singular_values": np.nan,
        "max_polynomial_abs_on_unit_domain": np.nan,
        "meets_epsilon_on_actual_singular_values": False,
        "phase_synthesis_status": "skipped_not_attempted",
        "phase_count": 0,
        "phase_synthesis_backend": "",
        "phase_synthesis_failure_reason": "",
        "nominal_phase_parameter_count": int(used_degree) + 1,
        "gate_simulation_status": "skipped_not_attempted",
        "gate_versus_polynomial_error": np.nan,
        "gate_versus_ridge_target_error": np.nan,
        "output_update_relative_error_against_matched_ridge": np.nan,
        "residual_norm_ratio_against_no_update_baseline": np.nan,
        "ridge_residual_norm_ratio_against_no_update_baseline": np.nan,
        "success_probability": np.nan,
        "success_probability_type": "",
        "run_status": "not_started",
        "failure_mode": "",
        "failure_reason": "",
    }


def _subproblem_failure_rows(
    spec: dict[str, Any],
    resolved: dict[str, Any],
    exc: Exception,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_name = str(spec.get("case_name", "unknown"))
    size = int(spec.get("subproblem_size", 0))
    selection = str(spec.get("selection_mode", "unknown"))
    for alpha in resolved["alpha_grid"]:
        for degree_index, requested_degree in enumerate(resolved["degree_grid"]):
            used_degree, adjustment = qsvt_odd_degree(int(requested_degree))
            for epsilon in resolved["epsilon_targets"]:
                row = {key: np.nan for key in DEGREE_RESULTS_COLUMNS}
                row.update(
                    {
                        "case_name": case_name,
                        "subproblem_size": size,
                        "selection_criterion": selection,
                        "matrix_shape": "",
                        "weighted_status": "weighted_jacobian_R_minus_half_H",
                        "alpha": float(alpha),
                        "epsilon_target": float(epsilon),
                        "requested_degree": int(requested_degree),
                        "degree": used_degree,
                        "degree_grid_index": degree_index,
                        "degree_adjustment_reason": adjustment,
                        "phase_synthesis_status": "skipped_subproblem_failed",
                        "phase_count": 0,
                        "gate_simulation_status": "skipped_subproblem_failed",
                        "run_status": "failed",
                        "failure_mode": "subproblem_extraction_failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                rows.append(row)
    return rows


def _apply_phase_synthesis_budget(
    *,
    rows: list[dict[str, Any]],
    summary_frame: pd.DataFrame,
    coefficient_cache: dict[tuple[str, int, str, float, int], np.ndarray],
    max_attempts: int,
    bound_tolerance: float,
) -> None:
    completed_indices = [
        index for index, row in enumerate(rows) if row["run_status"] == "completed"
    ]
    for index in completed_indices:
        rows[index]["phase_synthesis_status"] = (
            "skipped_phase_attempt_budget" if max_attempts > 0 else "skipped_not_attempted"
        )
    if max_attempts <= 0 or summary_frame.empty:
        return
    required = summary_frame[summary_frame["required_degree"].notna()].copy()
    required = required.sort_values(["case_name", "subproblem_size", "alpha", "epsilon_target"])
    attempts = 0
    attempted_keys: set[tuple[str, int, str, float, int]] = set()
    for _, summary in required.iterrows():
        key = (
            str(summary["case_name"]),
            int(summary["subproblem_size"]),
            str(summary["selection_criterion"]),
            float(summary["alpha"]),
            int(summary["required_degree"]),
        )
        if key in attempted_keys:
            continue
        attempted_keys.add(key)
        if attempts >= max_attempts:
            break
        coefficients = coefficient_cache.get(key)
        if coefficients is None:
            result = {
                "phase_synthesis_status": "failed_missing_coefficients",
                "phase_count": 0,
                "phase_synthesis_backend": "pyqsp_sym_qsp",
                "phase_synthesis_failure_reason": "required-degree coefficients unavailable",
            }
        else:
            max_abs = _max_polynomial_abs_from_coefficients(coefficients)
            if max_abs > 1.0 + bound_tolerance:
                result = {
                    "phase_synthesis_status": "skipped_unbounded_polynomial",
                    "phase_count": 0,
                    "phase_synthesis_backend": "pyqsp_sym_qsp",
                    "phase_synthesis_failure_reason": (
                        f"polynomial max abs {max_abs:.6g} exceeds 1 + tolerance"
                    ),
                }
            else:
                result = _try_pyqsp_phase_synthesis(coefficients)
                attempts += 1
        for row in rows:
            if row["run_status"] != "completed":
                continue
            if _phase_key_from_row(row) == key:
                row.update(result)


def _try_pyqsp_phase_synthesis(coefficients: np.ndarray) -> dict[str, Any]:
    if importlib.util.find_spec("pyqsp") is None:
        return {
            "phase_synthesis_status": "skipped_backend_unavailable",
            "phase_count": 0,
            "phase_synthesis_backend": "pyqsp_sym_qsp",
            "phase_synthesis_failure_reason": "pyqsp is unavailable",
        }
    try:
        angle_sequence = importlib.import_module("pyqsp.angle_sequence")
        phases_function = angle_sequence.QuantumSignalProcessingPhases

        buffer = StringIO()
        with redirect_stdout(buffer):
            result = phases_function(
                np.asarray(coefficients, dtype=np.float64),
                method="sym_qsp",
                chebyshev_basis=True,
            )
        phases = np.asarray(result[0], dtype=np.float64)
        return {
            "phase_synthesis_status": "passed_synthesis",
            "phase_count": int(phases.size),
            "phase_synthesis_backend": "pyqsp_sym_qsp",
            "phase_synthesis_failure_reason": "",
        }
    except Exception as exc:  # pragma: no cover - backend/version dependent
        return {
            "phase_synthesis_status": "failed_synthesis",
            "phase_count": 0,
            "phase_synthesis_backend": "pyqsp_sym_qsp",
            "phase_synthesis_failure_reason": f"{type(exc).__name__}: {exc}",
        }


def _phase_key_from_row(row: dict[str, Any]) -> tuple[str, int, str, float, int]:
    return (
        str(row["case_name"]),
        int(row["subproblem_size"]),
        str(row["selection_criterion"]),
        float(row["alpha"]),
        int(row["degree"]),
    )


def _max_polynomial_abs_from_coefficients(coefficients: np.ndarray) -> float:
    polynomial = Chebyshev(np.asarray(coefficients, dtype=np.float64), domain=[-1.0, 1.0])
    grid = np.linspace(-1.0, 1.0, max(4097, coefficients.size * 16), dtype=np.float64)
    return float(np.max(np.abs(polynomial(grid))))


def _plot_degree_tradeoff(summary_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if summary_frame.empty:
        ax.text(0.5, 0.5, "No degree sweep rows", ha="center", va="center")
    else:
        plot_frame = summary_frame.copy()
        plot_frame["required_degree_numeric"] = pd.to_numeric(
            plot_frame["required_degree"],
            errors="coerce",
        )
        grouped = (
            plot_frame.dropna(subset=["required_degree_numeric"])
            .groupby(["alpha", "epsilon_target"])["required_degree_numeric"]
            .median()
            .reset_index()
        )
        if grouped.empty:
            ax.text(0.5, 0.5, "No degree met target precision", ha="center", va="center")
        else:
            for epsilon, group in grouped.groupby("epsilon_target"):
                ordered = group.sort_values("alpha")
                ax.plot(
                    ordered["alpha"],
                    ordered["required_degree_numeric"],
                    marker="o",
                    linewidth=1.8,
                    label=f"epsilon={epsilon:g}",
                )
            ax.legend(frameon=False)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("alpha")
        ax.set_ylabel("required polynomial degree")
        ax.set_title("Bounded Ridge/Tikhonov QSVT Target Degree Cost")
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _degree_report_markdown(
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    results_csv: Path,
    summary_csv: Path,
) -> str:
    status_counts = results["run_status"].value_counts().to_dict() if not results.empty else {}
    phase_counts = (
        results["phase_synthesis_status"].value_counts().to_dict() if not results.empty else {}
    )
    met_count = (
        int((summary["degree_selection_status"] == "met_target").sum()) if not summary.empty else 0
    )
    no_pass_count = int(len(summary) - met_count)
    key_findings = _key_findings(summary)
    subproblem_lines = [
        "- "
        f"{spec.get('case_name')}, size={spec.get('subproblem_size')}, "
        f"selection={spec.get('selection_mode', 'high_leverage')}"
        for spec in config["subproblems"]
    ]
    return "\n".join(
        [
            "# Degree-Alpha-Precision Sweep Report",
            "",
            "## Command Used",
            "",
            f"`{current_command()}`",
            "",
            "## Environment Info",
            "",
            "- See `degree_alpha_precision_sweep_metadata.json` for Python, "
            "platform, package versions, seeds, and git commit if available.",
            "",
            "## Grids",
            "",
            f"- Alpha grid: {config['alpha_grid']}",
            f"- Epsilon grid: {config['epsilon_targets']}",
            f"- Requested degree grid: {config['degree_grid']}",
            "- Even requested degrees are advanced to the next odd degree to enforce "
            "odd QSVT parity.",
            "",
            "## Subproblem Selection",
            "",
            *subproblem_lines,
            "",
            "Subproblems use existing deterministic weighted-Jacobian selection "
            "utilities. The default criterion is high leverage: columns by largest "
            "weighted-Jacobian column norms and rows by largest restricted row norms.",
            "",
            "## Success and Failure Counts",
            "",
            f"- Degree-row status counts: {status_counts}",
            f"- Phase-synthesis status counts: {phase_counts}",
            f"- Summary rows meeting target precision: {met_count}",
            f"- Summary rows with no tested degree meeting target precision: {no_pass_count}",
            "",
            "## Key Numerical Findings",
            "",
            *key_findings,
            "",
            "## Claim-Safe Interpretation",
            "",
            "These results quantify the degree cost of implementing the bounded "
            "Ridge/Tikhonov spectral filter as a QSVT-compatible target. They do "
            "not imply QSVT numerically outperforms Ridge/Tikhonov under the "
            "matched spectral map.",
            "",
            "## Limitations",
            "",
            "- Polynomial approximation diagnostics are not hardware execution.",
            "- Gate-level simulation is skipped unless explicitly added in a separate "
            "validation run.",
            "- Dense-grid errors are approximation diagnostics over the normalized "
            "singular-value domain and do not prove scalable QSVT implementation.",
            "- Full-vector readout and scalable state preparation are outside this experiment.",
            "",
            "## Artifacts",
            "",
            f"- Results CSV: `{results_csv}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _key_findings(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- No summary rows were generated."]
    lines: list[str] = []
    passed = summary[summary["degree_selection_status"] == "met_target"]
    if not passed.empty:
        lines.append(
            "- Successful required degrees range from "
            f"{int(passed['required_degree'].min())} to {int(passed['required_degree'].max())}."
        )
    failed = summary[summary["degree_selection_status"] != "met_target"]
    lines.append(f"- Rows without a satisfying tested degree: {len(failed)}.")
    if "best_error_on_actual_singular_values" in summary:
        lines.append(
            "- Best actual-singular-value error range: "
            f"{summary['best_error_on_actual_singular_values'].min():.3e} to "
            f"{summary['best_error_on_actual_singular_values'].max():.3e}."
        )
    return lines


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_root": str(OUTPUT_ROOT),
        "seed": 123,
        "alpha_grid": ALPHA_GRID,
        "epsilon_targets": EPSILON_TARGETS,
        "degree_grid": DEGREE_GRID,
        "subproblems": DEFAULT_SUBPROBLEMS,
        "dense_grid_size": 4097,
        "phase_synthesis_max_attempts": 4,
        "phase_synthesis_bound_tolerance": 1.0e-6,
    }
    if config:
        resolved.update(config)
    resolved["alpha_grid"] = [float(value) for value in resolved["alpha_grid"]]
    resolved["epsilon_targets"] = [float(value) for value in resolved["epsilon_targets"]]
    resolved["degree_grid"] = [int(value) for value in resolved["degree_grid"]]
    resolved["subproblems"] = [dict(value) for value in resolved["subproblems"]]
    if any(value <= 0.0 for value in resolved["alpha_grid"]):
        raise ValueError("alpha_grid values must be positive")
    if any(value <= 0.0 for value in resolved["epsilon_targets"]):
        raise ValueError("epsilon_targets values must be positive")
    if any(value <= 0 for value in resolved["degree_grid"]):
        raise ValueError("degree_grid values must be positive")
    if int(resolved["dense_grid_size"]) < 129:
        raise ValueError("dense_grid_size must be at least 129")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run TQE QSVT degree-alpha-precision sweep",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = run_degree_alpha_precision_sweep({"output_root": args.output_root})
    print(f"TQE degree-alpha-precision sweep complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
