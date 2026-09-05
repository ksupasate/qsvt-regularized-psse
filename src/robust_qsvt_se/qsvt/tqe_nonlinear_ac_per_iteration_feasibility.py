from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import (
    ACNonlinearProblem,
    _build_update_estimator,
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.measurement.ac_linear import ac_measurements_and_jacobian
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    NONLINEAR_FEASIBILITY_DIR,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    bounded_ridge_target,
    fit_bounded_ridge_polynomial,
    qsvt_odd_degree,
)
from robust_qsvt_se.qsvt.tqe_sparse_oracle_block_encoding_model import (
    SparseJacobianOracle,
    ceil_log2,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("tqe_nonlinear_ac_per_iteration_feasibility")
LOGGER.addHandler(logging.NullHandler())

DEFAULT_CASES = ["ieee14", "ieee57"]
DEFAULT_STRESS_SETTINGS = ["clean_noise", "bad_data_10_percent", "missing_measurements"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_ALPHA_GRID = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
DEFAULT_EPSILON_TARGETS = [1.0e-2, 1.0e-3, 1.0e-4]
DEFAULT_DEGREE_GRID = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 201]
DEFAULT_NONZERO_TOL = 1.0e-12
DEFAULT_DENSE_GRID_SIZE = 1025
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_NONLINEAR_RIDGE_ALPHA = 1.0e-4
DEFAULT_HUBER_DELTA = 1.5
SMALL_TOL = 1.0e-15

ITERATION_COLUMNS = [
    "case_name",
    "stress_setting",
    "seed",
    "estimator",
    "iteration",
    "converged",
    "stopping_reason",
    "m",
    "n",
    "nnz",
    "density",
    "numerical_rank",
    "sigma_min_nonzero",
    "sigma_max",
    "condition_number",
    "norm_2",
    "norm_fro",
    "norm_max_abs",
    "residual_norm_unweighted",
    "residual_norm_weighted",
    "update_norm",
    "rmse",
    "ridge_alpha",
    "ridge_update_norm",
    "ridge_residual_ratio",
    "qsvt_epsilon_target",
    "required_degree",
    "target_met",
    "best_degree",
    "best_actual_singular_error",
    "dense_grid_error",
    "C_alpha",
    "gamma",
    "phase_synthesis_status",
    "sparse_max_row_nnz",
    "sparse_alpha_max",
    "sparse_normalization_overhead",
    "row_qubits",
    "col_qubits",
    "nonzero_index_qubits",
    "simulation_status",
    "failure_or_skip_reason",
]

SUMMARY_COLUMNS = [
    "case_name",
    "stress_setting",
    "estimator",
    "ridge_alpha",
    "qsvt_epsilon_target",
    "median_condition_number",
    "max_condition_number",
    "median_required_degree",
    "max_required_degree",
    "percentage_target_met",
    "median_residual_ratio",
    "convergence_rate",
    "row_count",
]

RUN_SUMMARY_COLUMNS = [
    "case_name",
    "stress_setting",
    "seed",
    "estimator",
    "final_rmse",
    "final_residual_norm_weighted",
    "iterations",
    "converged",
    "stopping_reason",
    "median_condition_number",
    "max_condition_number",
    "median_required_degree",
    "max_required_degree",
    "simulation_status",
    "failure_or_skip_reason",
]


@dataclass(frozen=True, slots=True)
class SpectralDiagnostics:
    U: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    numerical_rank: int
    sigma_min_nonzero: float
    sigma_max: float
    condition_number: float
    norm_2: float
    norm_fro: float
    norm_max_abs: float


@dataclass(frozen=True, slots=True)
class IterationSnapshot:
    case_name: str
    stress_setting: str
    seed: int
    estimator: str
    iteration: int
    converged: bool
    stopping_reason: str
    system: WeightedSystem
    residual_norm_unweighted: float
    residual_norm_weighted: float
    update_norm: float
    rmse: float
    simulation_status: str
    failure_or_skip_reason: str


def run_nonlinear_ac_per_iteration_feasibility(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / NONLINEAR_FEASIBILITY_DIR)
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    if resolved["mock_iterations"]:
        iteration_rows = _run_mock_iterations(resolved)
    else:
        iteration_rows = _run_configured_nonlinear_cases(resolved)

    diagnostics = pd.DataFrame(iteration_rows, columns=ITERATION_COLUMNS)
    summary = summarize_iteration_diagnostics(diagnostics)
    run_summary = summarize_nonlinear_runs(diagnostics)

    diagnostics_csv = output_dir / "nonlinear_iteration_diagnostics.csv"
    summary_csv = tables_dir / "table_nonlinear_ac_qsvt_feasibility_summary.csv"
    run_summary_csv = output_dir / "nonlinear_run_summary.csv"
    metadata_json = output_dir / "nonlinear_ac_per_iteration_feasibility_metadata.json"
    condition_figure = figures_dir / "figure_nonlinear_condition_number_by_iteration.png"
    degree_figure = figures_dir / "figure_nonlinear_required_degree_by_iteration.png"
    residual_figure = figures_dir / "figure_nonlinear_rmse_residual_by_iteration.png"
    sparse_figure = figures_dir / "figure_nonlinear_sparse_oracle_overhead_by_iteration.png"
    report_path = reports_dir / "nonlinear_ac_per_iteration_feasibility_report.md"

    diagnostics.to_csv(diagnostics_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    run_summary.to_csv(run_summary_csv, index=False)
    _plot_condition_by_iteration(diagnostics, condition_figure)
    _plot_degree_by_iteration(diagnostics, degree_figure)
    _plot_residual_by_iteration(diagnostics, residual_figure)
    _plot_sparse_overhead_by_iteration(diagnostics, sparse_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            diagnostics=diagnostics,
            summary=summary,
            run_summary=run_summary,
            diagnostics_csv=diagnostics_csv,
            summary_csv=summary_csv,
            run_summary_csv=run_summary_csv,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "diagnostics_csv": str(diagnostics_csv),
        "summary_table_csv": str(summary_csv),
        "run_summary_csv": str(run_summary_csv),
        "metadata_json": str(metadata_json),
        "condition_number_figure": str(condition_figure),
        "required_degree_figure": str(degree_figure),
        "rmse_residual_figure": str(residual_figure),
        "sparse_overhead_figure": str(sparse_figure),
        "report": str(report_path),
    }
    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts=artifacts,
    )
    metadata.update(
        {
            "benchmark_cases": resolved["cases"],
            "stress_settings": resolved["stress_settings"],
            "nonlinear_solver_settings": {
                "max_iterations": resolved["max_iterations"],
                "update_tolerance": resolved["update_tolerance"],
                "residual_tolerance": resolved["residual_tolerance"],
                "damping": resolved["damping"],
                "max_update_norm": resolved["max_update_norm"],
                "residual_growth_limit": resolved["residual_growth_limit"],
            },
            "alpha_grid": resolved["alpha_grid"],
            "epsilon_targets": resolved["epsilon_targets"],
            "degree_grid": resolved["degree_grid"],
            "nonzero_threshold": resolved["nonzero_tol"],
            "status_counts": _status_counts(diagnostics),
            "phase_synthesis_policy": "skipped_not_required for per-iteration feasibility sweep",
        }
    )
    write_json(metadata_json, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "diagnostics": diagnostics,
        "summary": summary,
        "run_summary": run_summary,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def matrix_spectral_diagnostics(
    matrix: np.ndarray,
    *,
    nonzero_tol: float = DEFAULT_NONZERO_TOL,
) -> SpectralDiagnostics:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    U, singular_values, Vt = np.linalg.svd(values, full_matrices=False)
    positive = singular_values[singular_values > float(nonzero_tol)]
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    sigma_min = float(np.min(positive)) if positive.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf
    return SpectralDiagnostics(
        U=U,
        singular_values=singular_values,
        Vt=Vt,
        numerical_rank=int(positive.size),
        sigma_min_nonzero=sigma_min,
        sigma_max=sigma_max,
        condition_number=condition,
        norm_2=sigma_max,
        norm_fro=float(np.linalg.norm(values, ord="fro")),
        norm_max_abs=float(np.max(np.abs(values))) if values.size else 0.0,
    )


def select_required_degree_from_errors(
    errors_by_degree: dict[int, float],
    epsilon_target: float,
) -> tuple[int | None, int | None, float, bool]:
    finite = {
        int(degree): float(error)
        for degree, error in errors_by_degree.items()
        if np.isfinite(float(error))
    }
    if not finite:
        return None, None, np.nan, False
    ordered = sorted(finite)
    passed = [degree for degree in ordered if finite[degree] <= float(epsilon_target)]
    best_degree = min(ordered, key=lambda degree: finite[degree])
    if passed:
        return passed[0], best_degree, finite[best_degree], True
    return None, best_degree, finite[best_degree], False


def ridge_update_and_residual_ratio(
    matrix: np.ndarray,
    residual: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, float]:
    H = np.asarray(matrix, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if H.ndim != 2 or r.shape != (H.shape[0],):
        raise ValueError("matrix and residual dimensions are inconsistent")
    U, singular_values, Vt = np.linalg.svd(H, full_matrices=False)
    filter_values = singular_values / (singular_values**2 + float(alpha))
    update = Vt.T @ (filter_values * (U.T @ r))
    no_update = float(np.linalg.norm(r))
    residual_after = float(np.linalg.norm(H @ update - r))
    ratio = 0.0 if no_update <= SMALL_TOL else residual_after / no_update
    return update, residual_after, float(ratio)


def sparse_iteration_diagnostics(
    matrix: np.ndarray,
    *,
    nonzero_tol: float = DEFAULT_NONZERO_TOL,
) -> dict[str, Any]:
    H = np.asarray(matrix, dtype=np.float64)
    oracle = SparseJacobianOracle.from_matrix(H, nonzero_tol=float(nonzero_tol))
    m, n = H.shape
    nnz = int(sum(oracle.row_nnz(i) for i in range(m)))
    max_row = int(oracle.max_row_nnz())
    norm_2 = float(np.linalg.norm(H, ord=2))
    norm_max = float(np.max(np.abs(H))) if H.size else 0.0
    alpha_sparse = float(max_row * norm_max)
    return {
        "nnz": nnz,
        "density": float(nnz / max(m * n, 1)),
        "sparse_max_row_nnz": max_row,
        "sparse_alpha_max": alpha_sparse,
        "sparse_normalization_overhead": (
            float(alpha_sparse / norm_2) if norm_2 > SMALL_TOL else np.inf
        ),
        "row_qubits": ceil_log2(m),
        "col_qubits": ceil_log2(n),
        "nonzero_index_qubits": ceil_log2(max(max_row, 1)),
    }


def degree_feasibility_by_epsilon(
    singular_values: np.ndarray,
    *,
    alpha: float,
    epsilon_targets: list[float],
    degree_grid: list[int],
    dense_grid_size: int = DEFAULT_DENSE_GRID_SIZE,
) -> dict[float, dict[str, Any]]:
    values = np.asarray(singular_values, dtype=np.float64)
    positive = values[values > DEFAULT_NONZERO_TOL]
    if positive.size == 0:
        return {
            float(epsilon): {
                "required_degree": np.nan,
                "target_met": False,
                "best_degree": np.nan,
                "best_actual_singular_error": np.nan,
                "dense_grid_error": np.nan,
                "C_alpha": np.nan,
                "gamma": np.nan,
            }
            for epsilon in epsilon_targets
        }
    gamma = float(np.max(positive))
    actual_normalized = values / gamma
    actual_errors: dict[int, float] = {}
    dense_errors: dict[int, float] = {}
    c_alpha = np.nan
    for requested_degree in degree_grid:
        degree, _ = qsvt_odd_degree(int(requested_degree))
        polynomial, _, c_alpha = fit_bounded_ridge_polynomial(
            alpha=float(alpha),
            beta=gamma,
            degree=degree,
        )
        actual_target = bounded_ridge_target(
            actual_normalized,
            alpha=float(alpha),
            beta=gamma,
            C_alpha=float(c_alpha),
        )
        actual_errors[int(degree)] = float(
            np.max(np.abs(polynomial(actual_normalized) - actual_target))
        )
        dense_grid = np.linspace(
            0.0,
            1.0,
            max(int(dense_grid_size), int(degree) + 2),
            dtype=np.float64,
        )
        dense_target = bounded_ridge_target(
            dense_grid,
            alpha=float(alpha),
            beta=gamma,
            C_alpha=float(c_alpha),
        )
        dense_errors[int(degree)] = float(np.max(np.abs(polynomial(dense_grid) - dense_target)))

    results: dict[float, dict[str, Any]] = {}
    for epsilon in epsilon_targets:
        required, best, best_error, target_met = select_required_degree_from_errors(
            actual_errors,
            float(epsilon),
        )
        selected = required if required is not None else best
        results[float(epsilon)] = {
            "required_degree": np.nan if required is None else int(required),
            "target_met": bool(target_met),
            "best_degree": np.nan if best is None else int(best),
            "best_actual_singular_error": float(best_error),
            "dense_grid_error": (
                np.nan if selected is None else float(dense_errors.get(int(selected), np.nan))
            ),
            "C_alpha": float(c_alpha),
            "gamma": gamma,
        }
    return results


def rows_for_iteration_snapshot(
    snapshot: IterationSnapshot,
    *,
    alpha_grid: list[float],
    epsilon_targets: list[float],
    degree_grid: list[int],
    dense_grid_size: int,
    nonzero_tol: float,
) -> list[dict[str, Any]]:
    H = np.asarray(snapshot.system.H_tilde, dtype=np.float64)
    r = np.asarray(snapshot.system.r_tilde, dtype=np.float64)
    spectral = matrix_spectral_diagnostics(H, nonzero_tol=nonzero_tol)
    sparse = sparse_iteration_diagnostics(H, nonzero_tol=nonzero_tol)
    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        try:
            ridge_update, _, ridge_ratio = ridge_update_and_residual_ratio(H, r, float(alpha))
            ridge_update_norm = float(np.linalg.norm(ridge_update))
            degree_rows = degree_feasibility_by_epsilon(
                spectral.singular_values,
                alpha=float(alpha),
                epsilon_targets=epsilon_targets,
                degree_grid=degree_grid,
                dense_grid_size=dense_grid_size,
            )
            phase_status = "skipped_not_required"
            row_status = snapshot.simulation_status
            row_reason = snapshot.failure_or_skip_reason
        except Exception as exc:
            ridge_update_norm = np.nan
            ridge_ratio = np.nan
            degree_rows = _failed_degree_rows(epsilon_targets)
            phase_status = "skipped_polynomial_diagnostic_failed"
            row_status = "failed_diagnostic"
            row_reason = f"{type(exc).__name__}: {exc}"

        for epsilon in epsilon_targets:
            degree = degree_rows[float(epsilon)]
            rows.append(
                {
                    "case_name": snapshot.case_name,
                    "stress_setting": snapshot.stress_setting,
                    "seed": int(snapshot.seed),
                    "estimator": snapshot.estimator,
                    "iteration": int(snapshot.iteration),
                    "converged": bool(snapshot.converged),
                    "stopping_reason": snapshot.stopping_reason,
                    "m": int(H.shape[0]),
                    "n": int(H.shape[1]),
                    "nnz": sparse["nnz"],
                    "density": sparse["density"],
                    "numerical_rank": spectral.numerical_rank,
                    "sigma_min_nonzero": spectral.sigma_min_nonzero,
                    "sigma_max": spectral.sigma_max,
                    "condition_number": spectral.condition_number,
                    "norm_2": spectral.norm_2,
                    "norm_fro": spectral.norm_fro,
                    "norm_max_abs": spectral.norm_max_abs,
                    "residual_norm_unweighted": snapshot.residual_norm_unweighted,
                    "residual_norm_weighted": snapshot.residual_norm_weighted,
                    "update_norm": snapshot.update_norm,
                    "rmse": snapshot.rmse,
                    "ridge_alpha": float(alpha),
                    "ridge_update_norm": ridge_update_norm,
                    "ridge_residual_ratio": ridge_ratio,
                    "qsvt_epsilon_target": float(epsilon),
                    "required_degree": degree["required_degree"],
                    "target_met": bool(degree["target_met"]),
                    "best_degree": degree["best_degree"],
                    "best_actual_singular_error": degree["best_actual_singular_error"],
                    "dense_grid_error": degree["dense_grid_error"],
                    "C_alpha": degree["C_alpha"],
                    "gamma": degree["gamma"],
                    "phase_synthesis_status": phase_status,
                    "sparse_max_row_nnz": sparse["sparse_max_row_nnz"],
                    "sparse_alpha_max": sparse["sparse_alpha_max"],
                    "sparse_normalization_overhead": sparse["sparse_normalization_overhead"],
                    "row_qubits": sparse["row_qubits"],
                    "col_qubits": sparse["col_qubits"],
                    "nonzero_index_qubits": sparse["nonzero_index_qubits"],
                    "simulation_status": row_status,
                    "failure_or_skip_reason": row_reason,
                }
            )
    return rows


def summarize_iteration_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "case_name",
        "stress_setting",
        "estimator",
        "ridge_alpha",
        "qsvt_epsilon_target",
    ]
    for keys, group in diagnostics.groupby(group_cols, dropna=False):
        run_flags = group.groupby(["case_name", "stress_setting", "seed", "estimator"])[
            "converged"
        ].max()
        required = pd.to_numeric(group["required_degree"], errors="coerce").dropna()
        condition = pd.to_numeric(group["condition_number"], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "median_condition_number": float(condition.median()),
                "max_condition_number": float(condition.max()),
                "median_required_degree": (
                    float(required.median()) if not required.empty else np.nan
                ),
                "max_required_degree": float(required.max()) if not required.empty else np.nan,
                "percentage_target_met": float(group["target_met"].astype(bool).mean() * 100.0),
                "median_residual_ratio": float(
                    pd.to_numeric(group["ridge_residual_ratio"], errors="coerce").median()
                ),
                "convergence_rate": (
                    float(run_flags.astype(bool).mean()) if not run_flags.empty else 0.0
                ),
                "row_count": len(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize_nonlinear_runs(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)
    unique_iterations = diagnostics.drop_duplicates(
        ["case_name", "stress_setting", "seed", "estimator", "iteration"]
    )
    rows: list[dict[str, Any]] = []
    for keys, group in unique_iterations.groupby(
        ["case_name", "stress_setting", "seed", "estimator"],
        dropna=False,
    ):
        ordered = group.sort_values("iteration")
        final = ordered.iloc[-1]
        all_rows = diagnostics[
            (diagnostics["case_name"].astype(str) == str(keys[0]))
            & (diagnostics["stress_setting"].astype(str) == str(keys[1]))
            & (diagnostics["seed"].astype(int) == int(keys[2]))
            & (diagnostics["estimator"].astype(str) == str(keys[3]))
        ]
        required = pd.to_numeric(all_rows["required_degree"], errors="coerce").dropna()
        condition = pd.to_numeric(ordered["condition_number"], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rows.append(
            {
                "case_name": keys[0],
                "stress_setting": keys[1],
                "seed": int(keys[2]),
                "estimator": keys[3],
                "final_rmse": final["rmse"],
                "final_residual_norm_weighted": final["residual_norm_weighted"],
                "iterations": int(ordered["iteration"].max()) + 1,
                "converged": bool(ordered["converged"].max()),
                "stopping_reason": str(final["stopping_reason"]),
                "median_condition_number": float(condition.median()),
                "max_condition_number": float(condition.max()),
                "median_required_degree": (
                    float(required.median()) if not required.empty else np.nan
                ),
                "max_required_degree": float(required.max()) if not required.empty else np.nan,
                "simulation_status": str(final["simulation_status"]),
                "failure_or_skip_reason": str(final["failure_or_skip_reason"]),
            }
        )
    return pd.DataFrame(rows, columns=RUN_SUMMARY_COLUMNS)


def _run_configured_nonlinear_cases(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for case_spec in resolved["cases"]:
        case_name = _case_name_from_spec(case_spec)
        for stress in resolved["stress_settings"]:
            for seed in resolved["seeds"]:
                if perf_counter() - started > float(resolved["max_total_seconds"]):
                    rows.extend(
                        _failure_rows(
                            case_name=case_name,
                            stress_setting=str(stress),
                            seed=int(seed),
                            estimator="all_requested_estimators",
                            reason="wall-clock budget exhausted",
                            resolved=resolved,
                            status="skipped_by_budget",
                        )
                    )
                    continue
                estimators = _estimators_for_stress(str(stress), resolved)
                for estimator_name in estimators:
                    try:
                        rows.extend(
                            _run_one_nonlinear_estimator(
                                case_spec=case_spec,
                                stress=str(stress),
                                seed=int(seed),
                                estimator_name=str(estimator_name),
                                resolved=resolved,
                            )
                        )
                    except Exception as exc:
                        rows.extend(
                            _failure_rows(
                                case_name=case_name,
                                stress_setting=str(stress),
                                seed=int(seed),
                                estimator=str(estimator_name),
                                reason=f"{type(exc).__name__}: {exc}",
                                resolved=resolved,
                                status="failed_or_skipped",
                            )
                        )
    return rows


def _run_one_nonlinear_estimator(
    *,
    case_spec: str | dict[str, Any],
    stress: str,
    seed: int,
    estimator_name: str,
    resolved: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(case_spec, dict) and case_spec.get("force_missing", False):
        raise ValueError("forced missing nonlinear case for failure-recording test")
    cfg = _nonlinear_config(case_spec, stress, seed, estimator_name, resolved)
    problem = build_ac_nonlinear_problem(cfg)
    estimator = _build_update_estimator(cfg["estimators"][0])
    iteration = cfg["system"]["iteration"]
    max_iterations = int(iteration["max_iterations"])
    update_tolerance = float(iteration["update_tolerance"])
    residual_tolerance = float(iteration["residual_tolerance"])
    damping = float(iteration["damping"])
    max_update_norm = float(iteration.get("max_update_norm", 1.0e6))
    residual_growth_limit = float(iteration.get("residual_growth_limit", 1.0e6))
    min_voltage = float(cfg["system"]["linearization"].get("min_voltage_magnitude", 0.5))
    angle_count = len(problem.case.angle_state_buses)
    x = problem.initial_state.copy()
    batches: list[list[dict[str, Any]]] = []
    final_reason = "max_iterations_reached"
    final_converged = False

    for index in range(max_iterations):
        system, residual_unweighted, residual_weighted = _linearized_snapshot(problem, x)
        update_result = estimator.solve(system)
        if update_result.failed or not np.all(np.isfinite(update_result.x_hat)):
            snapshot = IterationSnapshot(
                case_name=_case_name_from_spec(case_spec),
                stress_setting=stress,
                seed=seed,
                estimator=estimator.name,
                iteration=index,
                converged=False,
                stopping_reason="update_solver_failed",
                system=system,
                residual_norm_unweighted=residual_unweighted,
                residual_norm_weighted=residual_weighted,
                update_norm=np.nan,
                rmse=_state_rmse(x, problem.true_state),
                simulation_status="failed_update_solver",
                failure_or_skip_reason=update_result.failure_reason or "update solver failed",
            )
            batches.append(_rows_for_snapshot(snapshot, resolved))
            final_reason = "update_solver_failed"
            break

        update = np.asarray(update_result.x_hat, dtype=np.float64)
        update_norm = float(np.linalg.norm(update))
        if update_norm > max_update_norm:
            snapshot = IterationSnapshot(
                case_name=_case_name_from_spec(case_spec),
                stress_setting=stress,
                seed=seed,
                estimator=estimator.name,
                iteration=index,
                converged=False,
                stopping_reason="update_norm_limit_exceeded",
                system=system,
                residual_norm_unweighted=residual_unweighted,
                residual_norm_weighted=residual_weighted,
                update_norm=update_norm,
                rmse=_state_rmse(x, problem.true_state),
                simulation_status="failed_update_norm_limit",
                failure_or_skip_reason=f"update norm exceeded max_update_norm={max_update_norm}",
            )
            batches.append(_rows_for_snapshot(snapshot, resolved))
            final_reason = "update_norm_limit_exceeded"
            break

        x_next = x + damping * update
        x_next[angle_count:] = np.maximum(x_next[angle_count:], min_voltage)
        residual_after = _weighted_residual_norm(problem, x_next)
        if not np.isfinite(residual_after):
            status = "failed_nonfinite_residual"
            reason = "weighted residual became nonfinite"
            stop = "nonfinite_residual"
            converged = False
        elif residual_weighted > 0.0 and residual_after / residual_weighted > residual_growth_limit:
            status = "failed_residual_growth"
            reason = (
                f"weighted residual growth exceeded residual_growth_limit={residual_growth_limit}"
            )
            stop = "residual_growth_limit_exceeded"
            converged = False
        else:
            status = "completed_iteration"
            reason = ""
            converged = update_norm <= update_tolerance or residual_after <= residual_tolerance
            stop = "converged_by_tolerance" if converged else "iteration_continues"

        snapshot = IterationSnapshot(
            case_name=_case_name_from_spec(case_spec),
            stress_setting=stress,
            seed=seed,
            estimator=estimator.name,
            iteration=index,
            converged=converged,
            stopping_reason=stop,
            system=system,
            residual_norm_unweighted=residual_unweighted,
            residual_norm_weighted=residual_weighted,
            update_norm=update_norm,
            rmse=_state_rmse(x_next, problem.true_state),
            simulation_status=status,
            failure_or_skip_reason=reason,
        )
        batches.append(_rows_for_snapshot(snapshot, resolved))
        x = x_next
        if converged or status.startswith("failed"):
            final_reason = stop
            final_converged = converged
            break
    else:
        final_reason = "max_iterations_reached"
        final_converged = False

    if batches and final_reason == "max_iterations_reached":
        for row in batches[-1]:
            row["stopping_reason"] = final_reason
            row["converged"] = final_converged
    return [row for batch in batches for row in batch]


def _linearized_snapshot(
    problem: ACNonlinearProblem,
    state: np.ndarray,
) -> tuple[WeightedSystem, float, float]:
    values_full, jacobian_full, rows = ac_measurements_and_jacobian(
        problem.case,
        state,
        problem.measurement_config,
    )
    values = values_full[problem.kept_row_indices]
    jacobian = jacobian_full[problem.kept_row_indices, :]
    kept_labels = [rows[index].label for index in problem.kept_row_indices]
    if kept_labels != problem.measurement_labels:
        raise ValueError("nonlinear AC measurement layout changed during solve")
    residual = problem.z - values
    H_tilde = jacobian / problem.measurement_stds[:, None]
    r_tilde = residual / problem.measurement_stds
    system = WeightedSystem(
        H_tilde=H_tilde,
        r_tilde=r_tilde,
        x_true=problem.true_state - state,
        metadata={
            **problem.config_metadata,
            "measurement_count": len(problem.z),
            "kept_row_indices": problem.kept_row_indices.astype(int).tolist(),
        },
    )
    return system, float(np.linalg.norm(residual)), float(np.linalg.norm(r_tilde))


def _rows_for_snapshot(
    snapshot: IterationSnapshot,
    resolved: dict[str, Any],
) -> list[dict[str, Any]]:
    return rows_for_iteration_snapshot(
        snapshot,
        alpha_grid=[float(value) for value in resolved["alpha_grid"]],
        epsilon_targets=[float(value) for value in resolved["epsilon_targets"]],
        degree_grid=[int(value) for value in resolved["degree_grid"]],
        dense_grid_size=int(resolved["dense_grid_size"]),
        nonzero_tol=float(resolved["nonzero_tol"]),
    )


def _run_mock_iterations(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in resolved["mock_iterations"]:
        if item.get("force_failure", False):
            rows.extend(
                _failure_rows(
                    case_name=str(item.get("case_name", "mock")),
                    stress_setting=str(item.get("stress_setting", "mock_stress")),
                    seed=int(item.get("seed", 0)),
                    estimator=str(item.get("estimator", "ridge")),
                    reason="forced mock iteration failure",
                    resolved=resolved,
                    status="skipped_input_unavailable",
                )
            )
            continue
        H = np.asarray(item["H_tilde"], dtype=np.float64)
        r = np.asarray(item["r_tilde"], dtype=np.float64)
        system = WeightedSystem(H_tilde=H, r_tilde=r)
        snapshot = IterationSnapshot(
            case_name=str(item.get("case_name", "mock")),
            stress_setting=str(item.get("stress_setting", "mock_stress")),
            seed=int(item.get("seed", 0)),
            estimator=str(item.get("estimator", "ridge")),
            iteration=int(item.get("iteration", 0)),
            converged=bool(item.get("converged", False)),
            stopping_reason=str(item.get("stopping_reason", "mock_iteration")),
            system=system,
            residual_norm_unweighted=float(item.get("residual_norm_unweighted", np.linalg.norm(r))),
            residual_norm_weighted=float(item.get("residual_norm_weighted", np.linalg.norm(r))),
            update_norm=float(item.get("update_norm", 0.0)),
            rmse=float(item.get("rmse", np.nan)),
            simulation_status=str(item.get("simulation_status", "completed_mock")),
            failure_or_skip_reason=str(item.get("failure_or_skip_reason", "")),
        )
        rows.extend(_rows_for_snapshot(snapshot, resolved))
    return rows


def _failure_rows(
    *,
    case_name: str,
    stress_setting: str,
    seed: int,
    estimator: str,
    reason: str,
    resolved: dict[str, Any],
    status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in resolved["alpha_grid"]:
        for epsilon in resolved["epsilon_targets"]:
            row = {key: np.nan for key in ITERATION_COLUMNS}
            row.update(
                {
                    "case_name": case_name,
                    "stress_setting": stress_setting,
                    "seed": int(seed),
                    "estimator": estimator,
                    "iteration": 0,
                    "converged": False,
                    "stopping_reason": "input_unavailable",
                    "ridge_alpha": float(alpha),
                    "qsvt_epsilon_target": float(epsilon),
                    "target_met": False,
                    "phase_synthesis_status": "skipped_input_unavailable",
                    "simulation_status": status,
                    "failure_or_skip_reason": reason,
                }
            )
            rows.append(row)
    return rows


def _failed_degree_rows(epsilon_targets: list[float]) -> dict[float, dict[str, Any]]:
    return {
        float(epsilon): {
            "required_degree": np.nan,
            "target_met": False,
            "best_degree": np.nan,
            "best_actual_singular_error": np.nan,
            "dense_grid_error": np.nan,
            "C_alpha": np.nan,
            "gamma": np.nan,
        }
        for epsilon in epsilon_targets
    }


def _nonlinear_config(
    case_spec: str | dict[str, Any],
    stress: str,
    seed: int,
    estimator_name: str,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    case_name = _case_name_from_spec(case_spec)
    case_source = (
        str(case_spec.get("case_source", resolved["case_source"]))
        if isinstance(case_spec, dict)
        else str(resolved["case_source"])
    )
    measurement = dict(resolved["measurement"])
    if isinstance(case_spec, dict):
        measurement.update(dict(case_spec.get("measurement", {})))
    linearization = dict(resolved["linearization"])
    if isinstance(case_spec, dict):
        linearization.update(dict(case_spec.get("linearization", {})))
    return {
        "run_name": f"tqe_nonlinear_feasibility_{case_name}_{stress}_seed{seed}_{estimator_name}",
        "seed": int(seed),
        "system": {
            "case_name": case_name,
            "case_source": case_source,
            "mode": "nonlinear_ac_state_estimation",
            "measurement": measurement,
            "linearization": linearization,
            "iteration": {
                "max_iterations": int(resolved["max_iterations"]),
                "update_tolerance": float(resolved["update_tolerance"]),
                "residual_tolerance": float(resolved["residual_tolerance"]),
                "damping": float(resolved["damping"]),
                "max_update_norm": float(resolved["max_update_norm"]),
                "residual_growth_limit": float(resolved["residual_growth_limit"]),
                "log_iteration_metrics": True,
                "log_condition_number": True,
            },
        },
        "scenario": _stress_scenario(stress),
        "estimators": [_estimator_config(estimator_name, resolved)],
        "output": {
            "root": "outputs",
            "run_id": f"tqe_nonlinear_feasibility_{case_name}_{stress}_seed{seed}_{estimator_name}",
        },
    }


def _stress_scenario(stress: str) -> dict[str, Any]:
    profiles = {
        "clean_noise": {"noise_std": 0.002, "missing_ratio": 0.0, "bad_ratio": 0.0},
        "bad_data_10_percent": {"noise_std": 0.002, "missing_ratio": 0.0, "bad_ratio": 0.10},
        "missing_measurements": {"noise_std": 0.002, "missing_ratio": 0.20, "bad_ratio": 0.0},
    }
    profile = profiles.get(stress)
    if profile is None:
        raise ValueError(f"unsupported nonlinear stress setting: {stress}")
    return {
        "name": stress,
        "noise_std": profile["noise_std"],
        "missing_ratio": profile["missing_ratio"],
        "bad_data": {
            "enabled": profile["bad_ratio"] > 0.0,
            "ratio": profile["bad_ratio"],
            "magnitude": 10.0,
            "target": "random",
        },
    }


def _estimator_config(name: str, resolved: dict[str, Any]) -> dict[str, Any]:
    if name == "ridge":
        return {"name": "ridge", "alpha": float(resolved["nonlinear_update_alpha"])}
    if name == "pseudoinverse":
        return {"name": "pseudoinverse", "rcond": float(resolved["pseudoinverse_rcond"])}
    if name == "huber_irls":
        return {
            "name": "huber_irls",
            "delta": float(resolved["huber_delta"]),
            "max_iterations": int(resolved["huber_max_iterations"]),
            "tolerance": float(resolved["huber_tolerance"]),
        }
    raise ValueError(f"unsupported Experiment 6 estimator: {name}")


def _estimators_for_stress(stress: str, resolved: dict[str, Any]) -> list[str]:
    if resolved.get("estimators_by_stress"):
        mapping = dict(resolved["estimators_by_stress"])
        if stress in mapping:
            return [str(value) for value in mapping[stress]]
    if stress == "bad_data_10_percent":
        return ["ridge", "pseudoinverse", "huber_irls"]
    return ["ridge", "pseudoinverse"]


def _case_name_from_spec(spec: str | dict[str, Any]) -> str:
    return str(spec.get("case_name", "unknown")) if isinstance(spec, dict) else str(spec)


def _state_rmse(state: np.ndarray, true_state: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(state) - np.asarray(true_state)) ** 2)))


def _plot_condition_by_iteration(diagnostics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    completed = _plot_source_rows(diagnostics)
    if completed.empty:
        ax.text(0.5, 0.5, "No completed nonlinear iteration diagnostics", ha="center", va="center")
    else:
        grouped = (
            completed.groupby(["case_name", "stress_setting", "iteration"], dropna=False)[
                "condition_number"
            ]
            .median()
            .reset_index()
        )
        for (case, stress), group in grouped.groupby(["case_name", "stress_setting"]):
            ordered = group.sort_values("iteration")
            ax.plot(
                ordered["iteration"],
                ordered["condition_number"],
                marker="o",
                label=f"{case}/{stress}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("nonlinear iteration")
        ax.set_ylabel(r"$\kappa(\tilde H_k)$")
        ax.set_title("Nonlinear AC Weighted-Jacobian Conditioning")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_degree_by_iteration(diagnostics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    completed = _plot_source_rows(diagnostics)
    if completed.empty:
        ax.text(0.5, 0.5, "No completed degree diagnostics", ha="center", va="center")
    else:
        alpha = 1.0e-3 if np.isclose(completed["ridge_alpha"].astype(float), 1.0e-3).any() else None
        epsilon = (
            1.0e-3
            if np.isclose(completed["qsvt_epsilon_target"].astype(float), 1.0e-3).any()
            else None
        )
        subset = completed
        if alpha is not None:
            subset = subset[np.isclose(subset["ridge_alpha"].astype(float), alpha)]
        if epsilon is not None:
            subset = subset[np.isclose(subset["qsvt_epsilon_target"].astype(float), epsilon)]
        grouped = (
            subset.groupby(["case_name", "stress_setting", "iteration"], dropna=False)[
                "required_degree"
            ]
            .median()
            .reset_index()
        )
        for (case, stress), group in grouped.groupby(["case_name", "stress_setting"]):
            ordered = group.sort_values("iteration")
            ax.plot(
                ordered["iteration"],
                ordered["required_degree"],
                marker="o",
                label=f"{case}/{stress}",
            )
        ax.set_xlabel("nonlinear iteration")
        ax.set_ylabel("required degree")
        ax.set_title("Per-Iteration QSVT-Compatible Degree Requirement")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_residual_by_iteration(diagnostics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    completed = _plot_source_rows(diagnostics)
    if completed.empty:
        ax.text(0.5, 0.5, "No completed residual diagnostics", ha="center", va="center")
    else:
        unique = completed.drop_duplicates(
            ["case_name", "stress_setting", "seed", "estimator", "iteration"]
        )
        grouped = (
            unique.groupby(["case_name", "stress_setting", "estimator", "iteration"], dropna=False)[
                "residual_norm_weighted"
            ]
            .median()
            .reset_index()
        )
        for (case, stress, estimator), group in grouped.groupby(
            ["case_name", "stress_setting", "estimator"]
        ):
            ordered = group.sort_values("iteration")
            ax.plot(
                ordered["iteration"],
                ordered["residual_norm_weighted"],
                marker="o",
                label=f"{case}/{stress}/{estimator}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("nonlinear iteration")
        ax.set_ylabel("weighted residual norm")
        ax.set_title("Nonlinear AC Residual Trajectories")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_sparse_overhead_by_iteration(diagnostics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    completed = _plot_source_rows(diagnostics)
    if completed.empty or "sparse_normalization_overhead" not in completed:
        ax.text(0.5, 0.5, "No sparse-oracle overhead diagnostics", ha="center", va="center")
    else:
        grouped = (
            completed.groupby(["case_name", "stress_setting", "iteration"], dropna=False)[
                "sparse_normalization_overhead"
            ]
            .median()
            .reset_index()
        )
        for (case, stress), group in grouped.groupby(["case_name", "stress_setting"]):
            ordered = group.sort_values("iteration")
            ax.plot(
                ordered["iteration"],
                ordered["sparse_normalization_overhead"],
                marker="o",
                label=f"{case}/{stress}",
            )
        ax.set_xlabel("nonlinear iteration")
        ax.set_ylabel(r"$s||\tilde H_k||_{\max}/||\tilde H_k||_2$")
        ax.set_title("Sparse-Oracle Normalization Overhead by Iteration")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_source_rows(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    completed = diagnostics[
        diagnostics["simulation_status"]
        .astype(str)
        .str.contains(
            "completed|mock",
            case=False,
            na=False,
        )
    ].copy()
    if completed.empty:
        return completed
    return completed.drop_duplicates(
        [
            "case_name",
            "stress_setting",
            "seed",
            "estimator",
            "iteration",
            "ridge_alpha",
            "qsvt_epsilon_target",
        ]
    )


def _report_markdown(
    *,
    config: dict[str, Any],
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    diagnostics_csv: Path,
    summary_csv: Path,
    run_summary_csv: Path,
) -> str:
    status_counts = _status_counts(diagnostics)
    completed = diagnostics[
        diagnostics["simulation_status"].astype(str).str.contains("completed|mock", na=False)
    ]
    target_rate = (
        float(completed["target_met"].astype(bool).mean() * 100.0) if not completed.empty else 0.0
    )
    lines = _key_result_lines(completed)
    convergence = (
        float(run_summary["converged"].astype(bool).mean() * 100.0)
        if not run_summary.empty
        else 0.0
    )
    return "\n".join(
        [
            "# Nonlinear AC Per-Iteration QSVT Feasibility Diagnostic Report",
            "",
            "## Goal",
            "",
            "This experiment analyzes the QSVT-compatible spectral target requirements "
            "at each iteration of a classical nonlinear AC state-estimation workflow.",
            "",
            "The nonlinear loop is classical; QSVT is not executed inside the loop.",
            "",
            "## Cases and Stress Settings",
            "",
            f"- Cases requested: {config['cases']}",
            f"- Stress settings: {config['stress_settings']}",
            f"- Seeds: {config['seeds']}",
            f"- Seed-grid note: {config['seed_grid_note']}",
            "- Measurement rows: voltage magnitudes, active/reactive injections, "
            "and active/reactive branch-flow rows.",
            "- Weighting convention: H_tilde_k = R^{-1/2} H_k and r_tilde_k = R^{-1/2} r_k.",
            "",
            "## Degree Selection Rule",
            "",
            f"- Alpha grid: {config['alpha_grid']}",
            f"- Epsilon targets: {config['epsilon_targets']}",
            f"- Degree grid: {config['degree_grid']}",
            "- For each iteration and alpha, the smallest odd-compatible degree "
            "whose actual-singular-value error is <= epsilon is reported. If no "
            "degree meets the target, the best available degree and error are retained.",
            "- Phase synthesis is marked `skipped_not_required`; this is a "
            "degree-feasibility sweep, not per-iteration pyqsp synthesis.",
            "",
            "## Status",
            "",
            f"- Iteration diagnostic rows: {len(diagnostics)}.",
            f"- Run summary rows: {len(run_summary)}.",
            f"- Status counts: {status_counts}.",
            f"- Target-met percentage across completed rows: {target_rate:.2f}%.",
            f"- Nonlinear convergence rate across run summaries: {convergence:.2f}%.",
            "",
            "## Key Results",
            "",
            *lines,
            "",
            "## Sparse-Oracle Diagnostics",
            "",
            "The per-iteration sparse diagnostics reuse the sparse access model from "
            "Experiment 5: nnz, density, max row sparsity s, register-size estimates, "
            "and alpha_sparse_max = s * ||H_tilde_k||_max.",
            "",
            "## Claim-Safe Interpretation",
            "",
            "Although this work does not implement a nonlinear QSVT-in-the-loop "
            "estimator, the per-iteration diagnostics quantify the conditioning and "
            "polynomial-degree requirements that a QSVT-compatible regularized "
            "update would face in a nonlinear AC workflow.",
            "",
            "The results quantify feasibility requirements only. They do not "
            "demonstrate a nonlinear quantum state estimator, quantum speedup, or "
            "numerical superiority of QSVT over Ridge/Tikhonov.",
            "",
            "## Limitations",
            "",
            "- The nonlinear update loop is classical; QSVT is not executed inside it.",
            "- Per-iteration phase synthesis and gate simulation are intentionally "
            "skipped by budget and scope.",
            "- The default run uses three deterministic seeds to control runtime; "
            "larger seed grids remain straightforward but more expensive.",
            "- Full-vector readout, state preparation, and hardware execution remain "
            "outside this experiment.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "Per-iteration diagnostics from a classical nonlinear AC state-estimation "
            "workflow show the condition-number, sparse-access normalization, and "
            "bounded-polynomial degree requirements that a future QSVT-compatible "
            "regularized update would face. These diagnostics do not constitute a "
            "nonlinear QSVT-in-the-loop solver; they provide feasibility evidence "
            "and resource-boundary information for future work.",
            "",
            "## Artifacts",
            "",
            f"- Per-iteration CSV: `{diagnostics_csv}`",
            f"- Summary table: `{summary_csv}`",
            f"- Per-run summary: `{run_summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _key_result_lines(completed: pd.DataFrame) -> list[str]:
    if completed.empty:
        return ["- No completed per-iteration rows were generated."]
    required = pd.to_numeric(completed["required_degree"], errors="coerce").dropna()
    condition = (
        pd.to_numeric(completed["condition_number"], errors="coerce")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )
    overhead = (
        pd.to_numeric(
            completed["sparse_normalization_overhead"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    residual = pd.to_numeric(completed["residual_norm_weighted"], errors="coerce").dropna()
    rmse = pd.to_numeric(completed["rmse"], errors="coerce").dropna()
    lines = []
    if not condition.empty:
        lines.append(f"- Condition-number range: {condition.min():.3e} to {condition.max():.3e}.")
    if not required.empty:
        lines.append(
            f"- Required-degree range for target-met rows: {int(required.min())} to "
            f"{int(required.max())}."
        )
    else:
        lines.append("- No rows met the requested epsilon targets within the degree grid.")
    if not residual.empty:
        lines.append(
            f"- Weighted residual norm range: {residual.min():.3e} to {residual.max():.3e}."
        )
    if not rmse.empty:
        lines.append(f"- State RMSE range: {rmse.min():.3e} to {rmse.max():.3e}.")
    if not overhead.empty:
        lines.append(
            f"- Sparse normalization overhead range: {overhead.min():.3e} to {overhead.max():.3e}."
        )
    return lines


def _status_counts(diagnostics: pd.DataFrame) -> dict[str, int]:
    if diagnostics.empty or "simulation_status" not in diagnostics:
        return {}
    return {
        str(key): int(value)
        for key, value in diagnostics["simulation_status"].value_counts(dropna=False).items()
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(config or {})
    resolved: dict[str, Any] = {
        "output_root": str(OUTPUT_ROOT),
        "cases": DEFAULT_CASES,
        "case_source": "pypower",
        "stress_settings": DEFAULT_STRESS_SETTINGS,
        "seeds": DEFAULT_SEEDS,
        "seed_grid_note": "default limited to seeds [0, 1, 2] for nonlinear runtime budget",
        "alpha_grid": DEFAULT_ALPHA_GRID,
        "epsilon_targets": DEFAULT_EPSILON_TARGETS,
        "degree_grid": DEFAULT_DEGREE_GRID,
        "dense_grid_size": DEFAULT_DENSE_GRID_SIZE,
        "nonzero_tol": DEFAULT_NONZERO_TOL,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "update_tolerance": 1.0e-7,
        "residual_tolerance": 1.0e-7,
        "damping": 1.0,
        "max_update_norm": 1000.0,
        "residual_growth_limit": 10000.0,
        "max_total_seconds": 900.0,
        "nonlinear_update_alpha": DEFAULT_NONLINEAR_RIDGE_ALPHA,
        "pseudoinverse_rcond": 1.0e-10,
        "huber_delta": DEFAULT_HUBER_DELTA,
        "huber_max_iterations": 10,
        "huber_tolerance": 1.0e-7,
        "measurement": {
            "include_voltage_magnitudes": True,
            "include_p_injections": True,
            "include_q_injections": True,
            "include_p_branch_flows": True,
            "include_q_branch_flows": True,
            "voltage_std": 0.01,
            "injection_p_std": 0.03,
            "injection_q_std": 0.03,
            "flow_p_std": 0.02,
            "flow_q_std": 0.02,
        },
        "linearization": {
            "angle_perturbation_std": 0.005,
            "voltage_perturbation_std": 0.005,
            "min_voltage_magnitude": 0.5,
        },
        "estimators_by_stress": None,
        "mock_iterations": [],
    }
    resolved.update(supplied)
    resolved["cases"] = list(resolved["cases"])
    resolved["stress_settings"] = [str(value) for value in resolved["stress_settings"]]
    resolved["seeds"] = [int(value) for value in resolved["seeds"]]
    resolved["alpha_grid"] = [float(value) for value in resolved["alpha_grid"]]
    resolved["epsilon_targets"] = [float(value) for value in resolved["epsilon_targets"]]
    resolved["degree_grid"] = [int(value) for value in resolved["degree_grid"]]
    resolved["mock_iterations"] = list(resolved.get("mock_iterations", []))
    if float(resolved["nonzero_tol"]) < 0.0:
        raise ValueError("nonzero_tol must be nonnegative")
    if int(resolved["max_iterations"]) <= 0:
        raise ValueError("max_iterations must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run TQE Experiment 6 nonlinear AC per-iteration QSVT feasibility diagnostic."
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--stress-settings", nargs="+", default=DEFAULT_STRESS_SETTINGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--case-source", default="pypower")
    args = parser.parse_args(argv)
    run = run_nonlinear_ac_per_iteration_feasibility(
        {
            "output_root": args.output_root,
            "cases": args.cases,
            "stress_settings": args.stress_settings,
            "seeds": args.seeds,
            "max_iterations": args.max_iterations,
            "case_source": args.case_source,
        }
    )
    print(f"Wrote nonlinear AC per-iteration feasibility outputs to {run['output_dir']}")


if __name__ == "__main__":
    main()
