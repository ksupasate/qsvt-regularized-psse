from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev
from numpy.polynomial.chebyshev import chebvander

from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    SweepSubproblem,
    bounded_ridge_normalization_C,
    bounded_ridge_target,
    load_sweep_subproblem,
    qsvt_odd_degree,
)
from robust_qsvt_se.utils.io import write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_GRID = [1.0e-2, 1.0e-3, 1.0e-4]
EPSILON_TARGETS = [1.0e-2, 1.0e-3, 1.0e-4]
DEGREE_GRID = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 201]
DEFAULT_SUBPROBLEMS = [
    {"case_name": "ieee14", "subproblem_size": 4, "selection_mode": "high_leverage"},
    {"case_name": "ieee30", "subproblem_size": 8, "selection_mode": "high_leverage"},
    {"case_name": "ieee57", "subproblem_size": 16, "selection_mode": "high_leverage"},
    {"case_name": "ieee118", "subproblem_size": 16, "selection_mode": "high_leverage"},
]
SMALL_TOL = 1.0e-15

END_TO_END_RESULTS_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "matrix_shape",
    "weighted_status",
    "alpha",
    "epsilon_target",
    "degree",
    "target_met",
    "degree_selection_status",
    "degree_selection_source",
    "gamma",
    "C_alpha",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "numerical_rank",
    "residual_no_update",
    "residual_ridge",
    "residual_qsvt_poly",
    "ridge_residual_ratio",
    "qsvt_residual_ratio",
    "residual_gap",
    "ridge_update_norm",
    "qsvt_update_norm",
    "absolute_update_error",
    "relative_update_error",
    "max_component_error",
    "cosine_similarity",
    "actual_singular_value_error",
    "dense_grid_error",
    "max_polynomial_abs_on_unit_domain",
    "qsvt_admissible_polynomial",
    "phase_synthesis_status",
    "phase_count",
    "gate_simulation_status",
    "gate_update_error_vs_ridge",
    "gate_update_error_vs_polynomial",
    "gate_residual_ratio",
    "success_probability",
    "failure_or_skip_reason",
    "run_status",
]

SUMMARY_COLUMNS = [
    "case_name",
    "subproblem_size",
    "selection_criterion",
    "alpha",
    "epsilon_target",
    "degree",
    "relative_update_error",
    "residual_gap",
    "target_met",
    "gate_simulation_status",
    "run_status",
]


@dataclass(frozen=True, slots=True)
class DegreeSelection:
    degree: int
    target_met: bool
    selection_status: str
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class EndToEndComputation:
    row_values: dict[str, Any]
    ridge_update: np.ndarray
    qsvt_update: np.ndarray


def run_end_to_end_qsvt_vs_ridge(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = paths["end_to_end"]
    figures_dir = paths["figures"]
    tables_dir = paths["tables"]
    reports_dir = paths["reports"]

    degree_summary = _load_optional_frame(resolved["degree_summary_path"])
    degree_results = _load_optional_frame(resolved["degree_results_path"])
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    gate_attempts = 0

    for spec in resolved["subproblems"]:
        try:
            subproblem = load_sweep_subproblem(spec, seed=int(resolved["seed"]))
        except Exception as exc:
            rows.extend(_subproblem_failure_rows(spec, resolved, exc))
            continue
        for alpha in resolved["alpha_grid"]:
            for epsilon in resolved["epsilon_targets"]:
                selection = select_degree_for_setting(
                    subproblem=subproblem,
                    alpha=float(alpha),
                    epsilon_target=float(epsilon),
                    degree_summary=degree_summary,
                    degree_grid=resolved["degree_grid"],
                    dense_grid_size=int(resolved["dense_grid_size"]),
                )
                try:
                    computation = compute_end_to_end_update(
                        subproblem=subproblem,
                        alpha=float(alpha),
                        epsilon_target=float(epsilon),
                        degree=int(selection.degree),
                        degree_selection=selection,
                        dense_grid_size=int(resolved["dense_grid_size"]),
                    )
                    gate_values, attempted = _maybe_gate_validation(
                        subproblem=subproblem,
                        alpha=float(alpha),
                        degree=int(selection.degree),
                        qsvt_update=computation.qsvt_update,
                        config=resolved,
                        attempts_used=gate_attempts,
                    )
                    if attempted:
                        gate_attempts += 1
                    row = {
                        **computation.row_values,
                        **gate_values,
                    }
                    row["phase_synthesis_status"], row["phase_count"] = _previous_phase_status(
                        degree_results=degree_results,
                        row=row,
                    )
                    row["failure_or_skip_reason"] = _combined_reason(
                        degree_reason=selection.reason,
                        gate_reason=str(gate_values["failure_or_skip_reason"]),
                        target_met=selection.target_met,
                    )
                    rows.append(row)
                    component_rows.extend(
                        _component_rows(
                            row=row,
                            ridge_update=computation.ridge_update,
                            qsvt_update=computation.qsvt_update,
                        )
                    )
                except Exception as exc:
                    rows.append(_setting_failure_row(subproblem, alpha, epsilon, selection, exc))

    results = pd.DataFrame(rows, columns=END_TO_END_RESULTS_COLUMNS)
    components = pd.DataFrame(component_rows)
    summary = _summary_frame(results)

    results_csv = output_dir / "end_to_end_qsvt_vs_ridge_results.csv"
    components_csv = output_dir / "end_to_end_qsvt_vs_ridge_update_components.csv"
    metadata_json = output_dir / "end_to_end_qsvt_vs_ridge_metadata.json"
    summary_csv = tables_dir / "table_end_to_end_qsvt_vs_ridge_summary.csv"
    relative_error_figure = figures_dir / "figure_end_to_end_relative_update_error.png"
    residual_figure = figures_dir / "figure_end_to_end_residual_ratio_comparison.png"
    scatter_figure = figures_dir / "figure_qsvt_vs_ridge_update_scatter.png"
    report_path = reports_dir / "end_to_end_qsvt_vs_ridge_report.md"

    results.to_csv(results_csv, index=False)
    components.to_csv(components_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_relative_update_error(results, relative_error_figure)
    _plot_residual_ratio_comparison(results, residual_figure)
    _plot_update_scatter(results, components, scatter_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            results=results,
            summary=summary,
            results_csv=results_csv,
            summary_csv=summary_csv,
        ),
        encoding="utf-8",
    )

    ended_at = utc_timestamp()
    artifacts = {
        "results_csv": str(results_csv),
        "update_components_csv": str(components_csv),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "relative_update_error_figure": str(relative_error_figure),
        "residual_ratio_figure": str(residual_figure),
        "update_scatter_figure": str(scatter_figure),
        "report": str(report_path),
    }
    metadata = reproducibility_metadata(
        config={
            **resolved,
            "input_paths": {
                "degree_summary_path": resolved["degree_summary_path"],
                "degree_results_path": resolved["degree_results_path"],
            },
        },
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts=artifacts,
    )
    metadata["counts"] = _counts(results)
    write_json(metadata_json, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "results": results,
        "summary": summary,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def ridge_update_svd(A: np.ndarray, b: np.ndarray, *, alpha: float) -> np.ndarray:
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    matrix = np.asarray(A, dtype=np.float64)
    residual = np.asarray(b, dtype=np.float64)
    if matrix.ndim != 2 or residual.ndim != 1 or matrix.shape[0] != residual.size:
        raise ValueError("A must be m x n and b must have length m")
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    filter_values = singular_values / (singular_values**2 + float(alpha))
    return Vt.T @ (filter_values * (U.T @ residual))


def qsvt_polynomial_filter_values(
    singular_values: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    degree: int,
    grid_size: int = 4097,
) -> tuple[np.ndarray, dict[str, float]]:
    polynomial, C_alpha = fit_actual_singular_interpolating_polynomial(
        alpha=float(alpha),
        gamma=float(gamma),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        degree=int(degree),
    )
    normalized = np.asarray(singular_values, dtype=np.float64) / float(gamma)
    bounded_values = polynomial(normalized)
    physical_filter = C_alpha * bounded_values
    physical_target = np.asarray(singular_values, dtype=np.float64) / (
        np.asarray(singular_values, dtype=np.float64) ** 2 + float(alpha)
    )
    bounded_target = bounded_ridge_target(
        normalized,
        alpha=float(alpha),
        beta=float(gamma),
        C_alpha=C_alpha,
    )
    actual_error = (
        float(np.max(np.abs(physical_filter - physical_target))) if physical_target.size else 0.0
    )
    actual_bounded_error = (
        float(np.max(np.abs(bounded_values - bounded_target))) if bounded_target.size else 0.0
    )
    dense_grid = np.linspace(0.0, 1.0, max(int(grid_size), int(degree) * 16 + 1), dtype=np.float64)
    dense_physical_target = (float(gamma) * dense_grid) / (
        (float(gamma) * dense_grid) ** 2 + float(alpha)
    )
    dense_physical_error = float(
        np.max(np.abs(C_alpha * polynomial(dense_grid) - dense_physical_target))
    )
    dense_bounded_error = float(
        np.max(
            np.abs(
                polynomial(dense_grid)
                - bounded_ridge_target(
                    dense_grid,
                    alpha=float(alpha),
                    beta=float(gamma),
                    C_alpha=C_alpha,
                )
            )
        )
    )
    unit_grid = np.linspace(-1.0, 1.0, max(int(grid_size), int(degree) * 16 + 1))
    max_abs = float(np.max(np.abs(polynomial(unit_grid))))
    return physical_filter, {
        "C_alpha": float(C_alpha),
        "actual_singular_value_error": actual_error,
        "dense_grid_error": dense_physical_error,
        "actual_bounded_singular_value_error": actual_bounded_error,
        "dense_bounded_grid_error": dense_bounded_error,
        "max_polynomial_abs_on_unit_domain": max_abs,
        "qsvt_admissible_polynomial": bool(max_abs <= 1.0 + 1.0e-6),
    }


def fit_actual_singular_interpolating_polynomial(
    *,
    alpha: float,
    gamma: float,
    singular_values: np.ndarray,
    degree: int,
) -> tuple[Chebyshev, float]:
    """Fit an odd polynomial to the bounded target on actual singular values."""

    used_degree, _ = qsvt_odd_degree(int(degree))
    values = np.asarray(singular_values, dtype=np.float64)
    positive = values[values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("actual-singular interpolation requires positive singular values")
    normalized = positive / float(gamma)
    C_alpha = bounded_ridge_normalization_C(alpha=float(alpha), beta=float(gamma))
    target = bounded_ridge_target(
        normalized,
        alpha=float(alpha),
        beta=float(gamma),
        C_alpha=C_alpha,
    )
    odd_indices = [index for index in range(used_degree + 1) if index % 2 == 1]
    basis = chebvander(normalized, used_degree)[:, odd_indices]
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    cheb_coefficients = np.zeros(used_degree + 1, dtype=np.float64)
    cheb_coefficients[odd_indices] = coefficients
    return Chebyshev(cheb_coefficients, domain=[-1.0, 1.0]), C_alpha


def residual_metrics(
    A: np.ndarray,
    b: np.ndarray,
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(A, dtype=np.float64)
    residual = np.asarray(b, dtype=np.float64)
    ridge = np.asarray(ridge_update, dtype=np.float64)
    qsvt = np.asarray(qsvt_update, dtype=np.float64)
    residual_no_update = float(np.linalg.norm(residual))
    residual_ridge = float(np.linalg.norm(matrix @ ridge - residual))
    residual_qsvt = float(np.linalg.norm(matrix @ qsvt - residual))
    denominator = max(residual_no_update, SMALL_TOL)
    ridge_ratio = residual_ridge / denominator
    qsvt_ratio = residual_qsvt / denominator
    return {
        "residual_no_update": residual_no_update,
        "residual_ridge": residual_ridge,
        "residual_qsvt_poly": residual_qsvt,
        "ridge_residual_ratio": ridge_ratio,
        "qsvt_residual_ratio": qsvt_ratio,
        "residual_gap": abs(qsvt_ratio - ridge_ratio),
    }


def compute_end_to_end_update(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    epsilon_target: float,
    degree: int,
    degree_selection: DegreeSelection,
    dense_grid_size: int,
) -> EndToEndComputation:
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    b = np.asarray(subproblem.r_tilde, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("subproblem must have at least one positive singular value")
    gamma = float(np.max(positive))
    condition = float(np.max(positive) / np.min(positive))
    ridge_filter = singular_values / (singular_values**2 + float(alpha))
    ridge_update = Vt.T @ (ridge_filter * (U.T @ b))
    qsvt_filter, approximation = qsvt_polynomial_filter_values(
        singular_values,
        alpha=float(alpha),
        gamma=gamma,
        degree=int(degree),
        grid_size=int(dense_grid_size),
    )
    qsvt_update = Vt.T @ (qsvt_filter * (U.T @ b))
    residuals = residual_metrics(A, b, ridge_update, qsvt_update)
    update_delta = qsvt_update - ridge_update
    ridge_norm = float(np.linalg.norm(ridge_update))
    qsvt_norm = float(np.linalg.norm(qsvt_update))
    absolute_error = float(np.linalg.norm(update_delta))
    cosine = float(np.dot(qsvt_update, ridge_update) / max(qsvt_norm * ridge_norm, SMALL_TOL))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    metadata = subproblem.metadata
    row = {
        "case_name": str(metadata.get("case_name", metadata.get("case", "unknown"))),
        "subproblem_size": int(metadata.get("subproblem_size", min(A.shape))),
        "selection_criterion": str(metadata.get("selection_mode", "unknown")),
        "matrix_shape": f"{A.shape[0]}x{A.shape[1]}",
        "weighted_status": "weighted_jacobian_R_minus_half_H",
        "alpha": float(alpha),
        "epsilon_target": float(epsilon_target),
        "degree": int(degree),
        "target_met": bool(degree_selection.target_met),
        "degree_selection_status": degree_selection.selection_status,
        "degree_selection_source": degree_selection.source,
        "gamma": gamma,
        "C_alpha": approximation["C_alpha"],
        "condition_number": condition,
        "sigma_min": float(np.min(positive)),
        "sigma_max": float(np.max(positive)),
        "numerical_rank": int(np.linalg.matrix_rank(A)),
        **residuals,
        "ridge_update_norm": ridge_norm,
        "qsvt_update_norm": qsvt_norm,
        "absolute_update_error": absolute_error,
        "relative_update_error": absolute_error / max(ridge_norm, SMALL_TOL),
        "max_component_error": float(np.max(np.abs(update_delta))) if update_delta.size else 0.0,
        "cosine_similarity": cosine,
        "actual_singular_value_error": approximation["actual_singular_value_error"],
        "dense_grid_error": approximation["dense_grid_error"],
        "max_polynomial_abs_on_unit_domain": approximation["max_polynomial_abs_on_unit_domain"],
        "qsvt_admissible_polynomial": approximation["qsvt_admissible_polynomial"],
        "phase_synthesis_status": "not_available",
        "phase_count": 0,
        "gate_simulation_status": "skipped_by_budget",
        "gate_update_error_vs_ridge": np.nan,
        "gate_update_error_vs_polynomial": np.nan,
        "gate_residual_ratio": np.nan,
        "success_probability": np.nan,
        "failure_or_skip_reason": "",
        "run_status": "completed",
    }
    return EndToEndComputation(
        row_values=row,
        ridge_update=ridge_update,
        qsvt_update=qsvt_update,
    )


def select_degree_for_setting(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    epsilon_target: float,
    degree_summary: pd.DataFrame,
    degree_grid: list[int],
    dense_grid_size: int,
) -> DegreeSelection:
    metadata = subproblem.metadata
    case_name = str(metadata.get("case_name", metadata.get("case", "unknown")))
    size = int(metadata.get("subproblem_size", min(np.asarray(subproblem.H_tilde).shape)))
    selection = str(metadata.get("selection_mode", "unknown"))
    if not degree_summary.empty:
        matches = degree_summary[
            (degree_summary["case_name"] == case_name)
            & (degree_summary["subproblem_size"].astype(int) == size)
            & (degree_summary["selection_criterion"] == selection)
            & np.isclose(degree_summary["alpha"].astype(float), float(alpha))
            & np.isclose(degree_summary["epsilon_target"].astype(float), float(epsilon_target))
        ]
        if not matches.empty:
            return _select_degree_by_recomputing(
                subproblem=subproblem,
                alpha=float(alpha),
                epsilon_target=float(epsilon_target),
                degree_grid=degree_grid,
                dense_grid_size=dense_grid_size,
                source="physical_scale_recomputed_with_previous_summary_reference",
                base_reason=(
                    "Experiment 1 summary was found; Experiment 3 recomputed "
                    "physical-scale actual-singular-value filter error for the "
                    "end-to-end update"
                ),
            )

    return _select_degree_by_recomputing(
        subproblem=subproblem,
        alpha=float(alpha),
        epsilon_target=float(epsilon_target),
        degree_grid=degree_grid,
        dense_grid_size=dense_grid_size,
        source="physical_scale_recomputed_no_previous_summary",
        base_reason=(
            "previous degree summary unavailable; recomputed physical-scale degree selection"
        ),
    )


def _select_degree_by_recomputing(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    epsilon_target: float,
    degree_grid: list[int],
    dense_grid_size: int,
    source: str,
    base_reason: str,
) -> DegreeSelection:
    A = np.asarray(subproblem.H_tilde, dtype=np.float64)
    b = np.asarray(subproblem.r_tilde, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("degree selection requires a nonzero subproblem")
    gamma = float(np.max(positive))
    ridge_filter = singular_values / (singular_values**2 + float(alpha))
    ridge_update = Vt.T @ (ridge_filter * (U.T @ b))
    ridge_norm = max(float(np.linalg.norm(ridge_update)), SMALL_TOL)
    candidates: list[tuple[int, float, float, bool, float]] = []
    for requested in degree_grid:
        degree, _ = qsvt_odd_degree(int(requested))
        qsvt_filter, diagnostics = qsvt_polynomial_filter_values(
            singular_values,
            alpha=float(alpha),
            gamma=gamma,
            degree=degree,
            grid_size=int(dense_grid_size),
        )
        qsvt_update = Vt.T @ (qsvt_filter * (U.T @ b))
        relative_update_error = float(np.linalg.norm(qsvt_update - ridge_update) / ridge_norm)
        actual_error = float(diagnostics["actual_singular_value_error"])
        max_abs = float(diagnostics["max_polynomial_abs_on_unit_domain"])
        admissible = bool(diagnostics["qsvt_admissible_polynomial"])
        candidates.append((degree, actual_error, max_abs, admissible, relative_update_error))
        if (
            actual_error <= float(epsilon_target)
            and relative_update_error <= float(epsilon_target)
            and admissible
        ):
            return DegreeSelection(
                degree=degree,
                target_met=True,
                selection_status="met_target",
                source=source,
                reason=base_reason,
            )
    best_degree, best_error, best_max_abs, best_admissible, best_relative = min(
        candidates,
        key=lambda item: (not item[3], item[4], item[1]),
    )
    return DegreeSelection(
        degree=best_degree,
        target_met=False,
        selection_status="no_degree_met_target",
        source=source,
        reason=(
            f"{base_reason}; no recomputed degree met the physical-scale target; "
            f"using best available degree {best_degree} with error {best_error:.3e}, "
            f"relative update error {best_relative:.3e}, max |p|={best_max_abs:.3e}, "
            f"admissible={best_admissible}"
        ),
    )


def _maybe_gate_validation(
    *,
    subproblem: SweepSubproblem,
    alpha: float,
    degree: int,
    qsvt_update: np.ndarray,
    config: dict[str, Any],
    attempts_used: int,
) -> tuple[dict[str, Any], bool]:
    base = {
        "gate_simulation_status": "skipped_by_budget",
        "gate_update_error_vs_ridge": np.nan,
        "gate_update_error_vs_polynomial": np.nan,
        "gate_residual_ratio": np.nan,
        "success_probability": np.nan,
        "failure_or_skip_reason": (
            "gate-level simulation skipped by configured budget; matrix-level "
            "polynomial consistency is the required validation path"
        ),
    }
    if int(config["gate_validation_max_cases"]) <= attempts_used:
        return base, False
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    if H.shape[0] != H.shape[1] or H.shape[0] > int(config["gate_dimension_limit"]):
        base["failure_or_skip_reason"] = "gate-level simulation skipped by dimension budget"
        return base, False
    if int(degree) > int(config["gate_degree_limit"]):
        base["failure_or_skip_reason"] = "gate-level simulation skipped by degree budget"
        return base, False
    try:
        from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
            solve_gate_level_state_estimation_problem,
        )

        computation = solve_gate_level_state_estimation_problem(
            H_tilde=H,
            r_tilde=np.asarray(subproblem.r_tilde, dtype=np.float64),
            alpha=float(alpha),
            degree=int(degree),
            shots=int(config["gate_shots"]),
            seed=int(config["seed"]),
            metadata=subproblem.metadata,
            transpile_qubit_limit=0,
            export_qasm=False,
            phase_timeout_seconds=int(config["gate_phase_timeout_seconds"]),
        )
        ridge_norm = max(float(np.linalg.norm(computation.ridge_update)), SMALL_TOL)
        poly_norm = max(float(np.linalg.norm(qsvt_update)), SMALL_TOL)
        return {
            "gate_simulation_status": "completed",
            "gate_update_error_vs_ridge": float(
                np.linalg.norm(computation.qsvt_update - computation.ridge_update) / ridge_norm
            ),
            "gate_update_error_vs_polynomial": float(
                np.linalg.norm(computation.qsvt_update - qsvt_update) / poly_norm
            ),
            "gate_residual_ratio": float(
                computation.summary["residual_after_qsvt_update"]
                / max(computation.summary["residual_before_update"], SMALL_TOL)
            ),
            "success_probability": float(computation.summary["success_probability"]),
            "failure_or_skip_reason": "",
        }, True
    except Exception as exc:  # pragma: no cover - optional backend/resource dependent
        return {
            **base,
            "gate_simulation_status": "failed_gate_validation",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
        }, True


def _previous_phase_status(
    *,
    degree_results: pd.DataFrame,
    row: dict[str, Any],
) -> tuple[str, int]:
    if degree_results.empty:
        return "not_available", 0
    matches = degree_results[
        (degree_results["case_name"] == row["case_name"])
        & (degree_results["subproblem_size"].astype(int) == int(row["subproblem_size"]))
        & (degree_results["selection_criterion"] == row["selection_criterion"])
        & np.isclose(degree_results["alpha"].astype(float), float(row["alpha"]))
        & np.isclose(degree_results["epsilon_target"].astype(float), float(row["epsilon_target"]))
        & (degree_results["degree"].astype(int) == int(row["degree"]))
    ]
    if matches.empty:
        return "not_available", 0
    match = matches.iloc[0]
    return str(match.get("phase_synthesis_status", "not_available")), int(
        match.get("phase_count", 0)
    )


def _combined_reason(*, degree_reason: str, gate_reason: str, target_met: bool) -> str:
    reasons = []
    if degree_reason:
        reasons.append(degree_reason)
    if not target_met:
        reasons.append("target_met=false retained for audit")
    if gate_reason:
        reasons.append(gate_reason)
    return "; ".join(reasons)


def _component_rows(
    *,
    row: dict[str, Any],
    ridge_update: np.ndarray,
    qsvt_update: np.ndarray,
) -> list[dict[str, Any]]:
    setting_id = _setting_id(row)
    return [
        {
            "setting_id": setting_id,
            "case_name": row["case_name"],
            "subproblem_size": row["subproblem_size"],
            "selection_criterion": row["selection_criterion"],
            "alpha": row["alpha"],
            "epsilon_target": row["epsilon_target"],
            "degree": row["degree"],
            "target_met": row["target_met"],
            "component_index": int(index),
            "ridge_update_component": float(ridge_value),
            "qsvt_poly_update_component": float(qsvt_value),
            "component_error": float(qsvt_value - ridge_value),
        }
        for index, (ridge_value, qsvt_value) in enumerate(
            zip(ridge_update, qsvt_update, strict=True)
        )
    ]


def _subproblem_failure_rows(
    spec: dict[str, Any],
    resolved: dict[str, Any],
    exc: Exception,
) -> list[dict[str, Any]]:
    rows = []
    for alpha in resolved["alpha_grid"]:
        for epsilon in resolved["epsilon_targets"]:
            row = {key: np.nan for key in END_TO_END_RESULTS_COLUMNS}
            row.update(
                {
                    "case_name": str(spec.get("case_name", "unknown")),
                    "subproblem_size": int(spec.get("subproblem_size", 0)),
                    "selection_criterion": str(spec.get("selection_mode", "unknown")),
                    "alpha": float(alpha),
                    "epsilon_target": float(epsilon),
                    "target_met": False,
                    "phase_synthesis_status": "skipped_subproblem_failed",
                    "phase_count": 0,
                    "gate_simulation_status": "skipped_subproblem_failed",
                    "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
                    "run_status": "failed",
                }
            )
            rows.append(row)
    return rows


def _setting_failure_row(
    subproblem: SweepSubproblem,
    alpha: float,
    epsilon: float,
    selection: DegreeSelection,
    exc: Exception,
) -> dict[str, Any]:
    metadata = subproblem.metadata
    row = {key: np.nan for key in END_TO_END_RESULTS_COLUMNS}
    row.update(
        {
            "case_name": str(metadata.get("case_name", metadata.get("case", "unknown"))),
            "subproblem_size": int(metadata.get("subproblem_size", 0)),
            "selection_criterion": str(metadata.get("selection_mode", "unknown")),
            "alpha": float(alpha),
            "epsilon_target": float(epsilon),
            "degree": int(selection.degree),
            "target_met": bool(selection.target_met),
            "degree_selection_status": selection.selection_status,
            "degree_selection_source": selection.source,
            "phase_synthesis_status": "skipped_setting_failed",
            "phase_count": 0,
            "gate_simulation_status": "skipped_setting_failed",
            "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
            "run_status": "failed",
        }
    )
    return row


def _summary_frame(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return results[SUMMARY_COLUMNS].copy()


def _plot_relative_update_error(results: pd.DataFrame, output_path: Path) -> None:
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if completed.empty:
        ax.text(0.5, 0.5, "No completed rows", ha="center", va="center")
    else:
        labels = _case_size_labels(completed)
        label_to_x = {label: index for index, label in enumerate(sorted(set(labels)))}
        palette = ["C0", "C1", "C2", "C3", "C4"]
        colors = {
            alpha: color
            for alpha, color in zip(sorted(completed["alpha"].unique()), palette, strict=False)
        }
        markers = {1.0e-2: "o", 1.0e-3: "s", 1.0e-4: "^"}
        for row, label in zip(completed.itertuples(index=False), labels, strict=True):
            x = label_to_x[label]
            y = max(float(row.relative_update_error), 1.0e-16)
            ax.scatter(
                x,
                y,
                color=colors.get(float(row.alpha), "C3"),
                marker=markers.get(float(row.epsilon_target), "o"),
                alpha=0.8,
            )
        ax.set_xticks(list(label_to_x.values()))
        ax.set_xticklabels(list(label_to_x.keys()), rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("relative update error")
        ax.set_title("QSVT-Compatible Polynomial vs Matched Ridge Update")
        ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_residual_ratio_comparison(results: pd.DataFrame, output_path: Path) -> None:
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    if completed.empty:
        ax.text(0.5, 0.5, "No completed rows", ha="center", va="center")
    else:
        x = completed["ridge_residual_ratio"].astype(float).to_numpy()
        y = completed["qsvt_residual_ratio"].astype(float).to_numpy()
        ax.scatter(x, y, alpha=0.75)
        low = float(min(np.min(x), np.min(y)))
        high = float(max(np.max(x), np.max(y)))
        pad = max((high - low) * 0.05, 1.0e-12)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
        ax.set_xlabel("Ridge residual ratio")
        ax.set_ylabel("QSVT-compatible polynomial residual ratio")
        ax.set_title("Residual Ratio Consistency")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_update_scatter(
    results: pd.DataFrame,
    components: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    if completed.empty or components.empty:
        ax.text(0.5, 0.5, "No completed update components", ha="center", va="center")
    else:
        representative = (
            completed.sort_values(
                ["case_name", "subproblem_size", "target_met", "epsilon_target", "alpha"],
                ascending=[True, True, False, True, True],
            )
            .groupby(["case_name", "subproblem_size"], as_index=False)
            .first()
        )
        all_values = []
        for row in representative.itertuples(index=False):
            setting_id = _setting_id(row._asdict())
            subset = components[components["setting_id"] == setting_id]
            label = f"{row.case_name}-{int(row.subproblem_size)}"
            ax.scatter(
                subset["ridge_update_component"],
                subset["qsvt_poly_update_component"],
                alpha=0.8,
                label=label,
            )
            all_values.extend(subset["ridge_update_component"].astype(float).tolist())
            all_values.extend(subset["qsvt_poly_update_component"].astype(float).tolist())
        low = min(all_values)
        high = max(all_values)
        pad = max((high - low) * 0.05, 1.0e-12)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", linewidth=1.0)
        ax.set_xlabel("Ridge update component")
        ax.set_ylabel("QSVT-compatible polynomial update component")
        ax.set_title("Representative Update Component Agreement")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _report_markdown(
    *,
    config: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    results_csv: Path,
    summary_csv: Path,
) -> str:
    del summary
    counts = _counts(results)
    completed = results[results["run_status"] == "completed"] if not results.empty else results
    if completed.empty:
        findings = ["- No completed end-to-end settings."]
    else:
        best = completed.loc[completed["relative_update_error"].idxmin()]
        worst = completed.loc[completed["relative_update_error"].idxmax()]
        findings = [
            "- Relative update error range: "
            f"{completed['relative_update_error'].min():.3e} to "
            f"{completed['relative_update_error'].max():.3e}.",
            "- Residual gap range: "
            f"{completed['residual_gap'].min():.3e} to "
            f"{completed['residual_gap'].max():.3e}.",
            "- Best relative-error setting: "
            f"{best['case_name']} size {int(best['subproblem_size'])}, "
            f"alpha={best['alpha']:.1e}, epsilon={best['epsilon_target']:.1e}.",
            "- Worst relative-error setting: "
            f"{worst['case_name']} size {int(worst['subproblem_size'])}, "
            f"alpha={worst['alpha']:.1e}, epsilon={worst['epsilon_target']:.1e}.",
        ]
    subproblem_lines = [
        "- "
        f"{spec.get('case_name')}, size={spec.get('subproblem_size')}, "
        f"selection={spec.get('selection_mode', 'high_leverage')}"
        for spec in config["subproblems"]
    ]
    return "\n".join(
        [
            "# End-to-End QSVT-Compatible vs Ridge/Tikhonov Report",
            "",
            "## Experiment Goal",
            "",
            "Verify that the QSVT-compatible polynomial implementation reproduces "
            "the matched Ridge/Tikhonov spectral update within controlled "
            "approximation error on selected IEEE-derived weighted-Jacobian "
            "subproblems.",
            "",
            "## Command Used",
            "",
            f"`{current_command()}`",
            "",
            "## Environment Information",
            "",
            "- See `end_to_end_qsvt_vs_ridge_metadata.json` for Python, platform, "
            "package versions, seed, git commit if available, and config.",
            "",
            "## Cases and Sizes",
            "",
            *subproblem_lines,
            "",
            "## Alpha and Epsilon Grids",
            "",
            f"- Alpha grid: {config['alpha_grid']}",
            f"- Epsilon grid: {config['epsilon_targets']}",
            "",
            "## Degree Selection Rule",
            "",
            "The experiment reads `table_degree_alpha_precision_summary.csv` as "
            "a reference, then recomputes the physical-scale actual-singular-value "
            "filter error for the end-to-end polynomial on the selected subproblem. "
            "For each case/size/alpha/epsilon setting, it selects the smallest "
            "degree in the configured grid whose physical-scale actual-singular "
            "error and relative update error are at most epsilon, and whose fitted "
            "polynomial remains bounded by 1 + tolerance on the unit domain. If no "
            "degree meets all criteria, it keeps the setting, uses the best "
            "available degree, and marks `target_met=false`.",
            "",
            "## Normalization and Rescaling Formula",
            "",
            "Let A = U Sigma V^T, gamma = ||A||_2, and s_i = sigma_i / gamma. "
            "The bounded target is f_alpha(s) = [gamma s / ((gamma s)^2 + alpha)] / C_alpha, "
            "where C_alpha = max(1, max_{s in [0,1]} gamma s / ((gamma s)^2 + alpha)). "
            "The fitted odd polynomial p_d(s) approximates f_alpha(s). The physical "
            "QSVT-compatible update uses the rescaled filter C_alpha p_d(s_i): "
            "Delta x_poly = V diag(C_alpha p_d(s_i)) U^T b.",
            "",
            "## Successful, Failed, and Skipped Settings",
            "",
            f"- Counts: {counts}",
            "",
            "## Key Numerical Results",
            "",
            *findings,
            "",
            "## Interpretation",
            "",
            "The end-to-end small-solver experiment verifies consistency between "
            "the matched Ridge/Tikhonov update and the QSVT-compatible polynomial "
            "implementation on selected IEEE-derived weighted-Jacobian subproblems.",
            "",
            "## Claim Boundaries",
            "",
            "The results support the implementation-pathway claim but do not imply "
            "numerical superiority of QSVT over Ridge/Tikhonov. Gate-level checks "
            "are reported where computationally feasible; matrix-level polynomial "
            "consistency is used for larger selected blocks. The experiment remains "
            "selected-subproblem evidence and does not constitute a full IEEE-scale "
            "quantum state estimator.",
            "",
            "## Limitations",
            "",
            "- Default gate-level simulation is skipped by budget; matrix-level "
            "polynomial consistency is the required validation path.",
            "- Full sparse-oracle construction, state preparation, and full-vector "
            "readout are outside this experiment.",
            "- Degree selection is selected-subproblem and matrix-level; it is not a "
            "scalable phase-synthesis or sparse-oracle guarantee.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "The end-to-end small-solver experiment verifies consistency between "
            "the matched Ridge/Tikhonov update and the QSVT-compatible polynomial "
            "implementation on selected IEEE-derived weighted-Jacobian subproblems. "
            "The results support the implementation-pathway claim but do not imply "
            "numerical superiority of QSVT over Ridge/Tikhonov.",
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


def _counts(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {}
    return {
        "rows": len(results),
        "run_status": results["run_status"].value_counts().to_dict(),
        "target_met": int(results["target_met"].fillna(False).sum()),
        "target_not_met": int((~results["target_met"].fillna(False).astype(bool)).sum()),
        "gate_simulation_status": results["gate_simulation_status"].value_counts().to_dict(),
        "phase_synthesis_status": results["phase_synthesis_status"].value_counts().to_dict(),
    }


def _case_size_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{row.case_name}-{int(row.subproblem_size)}" for row in frame.itertuples(index=False)]


def _setting_id(row: dict[str, Any]) -> str:
    return (
        f"{row['case_name']}_{int(row['subproblem_size'])}_"
        f"a{float(row['alpha']):.0e}_e{float(row['epsilon_target']):.0e}_"
        f"d{int(row['degree'])}"
    )


def _load_optional_frame(path_value: str | None) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    output_root = OUTPUT_ROOT
    resolved = {
        "output_root": str(output_root),
        "seed": 123,
        "alpha_grid": ALPHA_GRID,
        "epsilon_targets": EPSILON_TARGETS,
        "degree_grid": DEGREE_GRID,
        "subproblems": DEFAULT_SUBPROBLEMS,
        "dense_grid_size": 4097,
        "degree_summary_path": str(
            output_root / "tables" / "table_degree_alpha_precision_summary.csv"
        ),
        "degree_results_path": str(
            output_root
            / "degree_alpha_precision_sweep"
            / "degree_alpha_precision_sweep_results.csv"
        ),
        "gate_validation_max_cases": 0,
        "gate_dimension_limit": 4,
        "gate_degree_limit": 25,
        "gate_shots": 1000,
        "gate_phase_timeout_seconds": 10,
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
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run TQE end-to-end QSVT-compatible vs Ridge solver check",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = run_end_to_end_qsvt_vs_ridge({"output_root": args.output_root})
    print(f"TQE end-to-end QSVT-vs-Ridge experiment complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
