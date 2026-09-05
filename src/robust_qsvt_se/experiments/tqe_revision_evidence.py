from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "robust_qsvt_se_matplotlib"))
import numpy as np
import pandas as pd

from robust_qsvt_se import __version__
from robust_qsvt_se.data.cases import load_ac_case
from robust_qsvt_se.estimators.base import Estimator
from robust_qsvt_se.estimators.pseudoinverse import PseudoinverseEstimator
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.experiments.iterative_ac import (
    ACNonlinearProblem,
    _linearized_update_system,
    _weighted_residual_norm,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.measurement.ac_linear import (
    ac_measurements_and_jacobian,
    default_ac_state_vector,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.filters import inverse_filter, ridge_filter
from robust_qsvt_se.qsvt.polynomial_approximation import (
    ApproximationContext,
    as_odd_degree,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.qsvt.research_matrix import (
    DEFAULT_LINEARIZATION_CONFIG,
    DEFAULT_MEASUREMENT_CONFIG,
)
from robust_qsvt_se.utils.config import DEFAULT_CONFIG
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.seed import make_rng

CLAIM_BOUNDARY = (
    "Small circuits validate instantiation. Larger matrix-level experiments validate the "
    "spectral-filter translation beyond toy examples. Full sparse-access diagnostics "
    "characterize scaling assumptions. None of these are full IEEE-scale quantum hardware "
    "executions, quantum-speedup claims, QSVT-over-Ridge claims, or real PMU/SCADA validation."
)
SPARSE_ACCESS_NOTE = (
    "Dense block encoding is used only for selected small-subproblem validation. Since the "
    "PSSE Jacobian is structurally sparse and diagonal weighting preserves sparsity, "
    "scalability is discussed through sparse-access modeling and resource diagnostics."
)

NONLINEAR_ITERATION_COLUMNS = [
    "case_name",
    "seed",
    "perturbation_name",
    "solver_name",
    "alpha",
    "iteration",
    "weighted_residual_norm",
    "update_norm",
    "state_rmse",
    "weighted_jacobian_cond",
    "sigma_max",
    "sigma_min",
    "jacobian_nnz",
    "jacobian_density",
    "converged",
    "stop_reason",
    "runtime_seconds",
]
NONLINEAR_SUMMARY_COLUMNS = [
    "case_name",
    "seed",
    "perturbation_name",
    "solver_name",
    "alpha",
    "converged",
    "stop_reason",
    "num_iterations",
    "final_weighted_residual_norm",
    "final_update_norm",
    "final_state_rmse",
    "min_state_rmse",
    "max_update_norm",
    "median_weighted_jacobian_cond",
    "max_weighted_jacobian_cond",
    "diverged_or_oscillated",
    "runtime_seconds",
]
QSVT_MATRIX_COLUMNS = [
    "case_name",
    "block_id",
    "block_shape",
    "selection_policy",
    "alpha",
    "target_epsilon",
    "degree",
    "phase_synthesis_status",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "target_scale_C",
    "grid_max_error",
    "actual_singular_value_max_error",
    "relative_update_error_vs_ridge",
    "residual_gap_vs_ridge",
    "runtime_seconds",
    "failure_reason",
]
SPARSE_ACCESS_COLUMNS = [
    "case_name",
    "num_rows",
    "num_cols",
    "num_nonzeros_H",
    "num_nonzeros_weighted_H",
    "density_H",
    "density_weighted_H",
    "max_row_nnz",
    "mean_row_nnz",
    "median_row_nnz",
    "max_col_nnz",
    "mean_col_nnz",
    "median_col_nnz",
    "sparsity_pattern_preserved",
    "weighting_is_diagonal",
    "sparse_access_notes",
    "runtime_seconds",
]


def run_tqe_revision_evidence(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the advisor-requested revision-evidence package.

    The package is deliberately claim-bounded: nonlinear AC comparison changes
    only the linear update solver, matrix-level QSVT validation uses polynomial
    spectral filtering rather than large circuit execution, and sparse-access
    diagnostics separate scalability assumptions from dense selected blocks.
    """

    resolved = _resolve_evidence_config(config)
    output_dir = _fresh_output_dir(Path(resolved["output_dir"]))
    ensure_directory(output_dir)
    artifacts: dict[str, Path] = {}
    skipped: dict[str, str] = {}

    if "nonlinear" in resolved["tasks"]:
        run = run_nonlinear_convergence_comparison(
            {
                **resolved["nonlinear"],
                "cases": resolved["cases"],
                "case_source": resolved["case_source"],
                "seeds": resolved["seeds"],
                "output_dir": str(output_dir),
            }
        )
        artifacts.update(run["artifacts"])
    else:
        skipped["nonlinear"] = "task not selected"

    if "qsvt-matrix" in resolved["tasks"]:
        run = run_larger_qsvt_matrix_validation(
            {
                **resolved["qsvt_matrix"],
                "cases": resolved["cases"],
                "case_source": resolved["case_source"],
                "seed": resolved["seeds"][0],
                "output_dir": str(output_dir),
            }
        )
        artifacts.update(run["artifacts"])
    else:
        skipped["qsvt-matrix"] = "task not selected"

    if "sparse" in resolved["tasks"]:
        run = run_sparse_access_diagnostics(
            {
                **resolved["sparse"],
                "cases": resolved["cases"],
                "case_source": resolved["case_source"],
                "seed": resolved["seeds"][0],
                "output_dir": str(output_dir),
            }
        )
        artifacts.update(run["artifacts"])
    else:
        skipped["sparse"] = "task not selected"

    readme = write_evidence_readme(output_dir=output_dir, artifacts=artifacts, skipped=skipped)
    artifacts["readme"] = readme
    manifest = write_evidence_manifest(
        output_dir=output_dir,
        config=resolved,
        artifacts=artifacts,
        skipped=skipped,
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "skipped": skipped,
        "config": resolved,
    }


def run_nonlinear_convergence_comparison(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_nonlinear_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    iteration_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case_name in resolved["cases"]:
        for seed in resolved["seeds"]:
            problem_config = _nonlinear_problem_config(resolved, case_name=case_name, seed=seed)
            problem = build_ac_nonlinear_problem(problem_config)
            solvers: list[tuple[str, float, Estimator]] = [
                ("pinv", np.nan, PseudoinverseEstimator(rcond=float(resolved["rcond"]))),
                ("ridge", float(resolved["alpha"]), RidgeEstimator(alpha=float(resolved["alpha"]))),
            ]
            for solver_name, alpha, estimator in solvers:
                rows, summary = _run_single_nonlinear_solver(
                    config=problem_config,
                    problem=problem,
                    estimator=estimator,
                    solver_name=solver_name,
                    alpha=alpha,
                    divergence_residual_growth_factor=float(
                        resolved["divergence_residual_growth_factor"]
                    ),
                )
                iteration_rows.extend(rows)
                summary_rows.append(summary)

    iteration_frame = _ordered_frame(iteration_rows, NONLINEAR_ITERATION_COLUMNS)
    summary_frame = _ordered_frame(summary_rows, NONLINEAR_SUMMARY_COLUMNS)
    iteration_csv = output_dir / "nonlinear_convergence_iterations.csv"
    summary_csv = output_dir / "nonlinear_convergence_summary.csv"
    iteration_frame.to_csv(iteration_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    artifacts = {
        "nonlinear_convergence_iterations": iteration_csv,
        "nonlinear_convergence_summary": summary_csv,
    }
    if bool(resolved["save_plots"]):
        artifacts.update(_write_nonlinear_plots(output_dir, iteration_frame))
    return {
        "output_dir": output_dir,
        "iteration_metrics": iteration_frame,
        "summary_metrics": summary_frame,
        "artifacts": artifacts,
    }


def run_larger_qsvt_matrix_validation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_qsvt_matrix_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case_name in resolved["cases"]:
        system, _matrix_source = build_engineering_system(
            {
                "case_name": case_name,
                "case_source": resolved["case_source"],
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["seed"]),
                "measurement": resolved["measurement"],
                "linearization": resolved["linearization"],
            }
        )
        for block_size in resolved["block_sizes"]:
            block_id = f"{case_name}_{block_size}x{block_size}_{resolved['selection_policy']}"
            start = time.perf_counter()
            try:
                H_block, r_block, selected_rows, selected_cols = select_deterministic_block(
                    system.H_tilde,
                    system.r_tilde,
                    row_count=int(block_size),
                    col_count=int(block_size),
                    policy=str(resolved["selection_policy"]),
                )
                row = _qsvt_matrix_validation_row(
                    case_name=case_name,
                    block_id=block_id,
                    H_block=H_block,
                    r_block=r_block,
                    selected_rows=selected_rows,
                    selected_cols=selected_cols,
                    alpha=float(resolved["alpha"]),
                    target_epsilon=float(resolved["target_epsilon"]),
                    degrees=[int(value) for value in resolved["degrees"]],
                    method=str(resolved["polynomial_method"]),
                    grid_size=int(resolved["grid_size"]),
                    selection_policy=str(resolved["selection_policy"]),
                    runtime_start=start,
                )
            except Exception as exc:
                row = _qsvt_failure_row(
                    case_name=case_name,
                    block_id=block_id,
                    block_shape=f"{block_size}x{block_size}",
                    selection_policy=str(resolved["selection_policy"]),
                    alpha=float(resolved["alpha"]),
                    target_epsilon=float(resolved["target_epsilon"]),
                    runtime_seconds=time.perf_counter() - start,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
    frame = _ordered_frame(rows, QSVT_MATRIX_COLUMNS)
    csv_path = output_dir / "larger_qsvt_matrix_validation.csv"
    table_path = output_dir / "larger_qsvt_matrix_validation_summary.csv"
    frame.to_csv(csv_path, index=False)
    _qsvt_summary_table(frame).to_csv(table_path, index=False)
    artifacts = {
        "larger_qsvt_matrix_validation": csv_path,
        "larger_qsvt_matrix_validation_summary": table_path,
    }
    if bool(resolved["save_plots"]):
        artifacts.update(_write_qsvt_matrix_plot(output_dir, frame))
    return {"output_dir": output_dir, "validation": frame, "artifacts": artifacts}


def run_sparse_access_diagnostics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_sparse_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case_name in resolved["cases"]:
        start = time.perf_counter()
        H, weighted_H = build_unweighted_and_weighted_ac_jacobian(
            case_name=case_name,
            case_source=str(resolved["case_source"]),
            seed=int(resolved["seed"]),
            measurement_config=dict(resolved["measurement"]),
            linearization_config=dict(resolved["linearization"]),
        )
        row = sparse_access_diagnostics_row(
            case_name=case_name,
            H=H,
            weighted_H=weighted_H,
            tolerance=float(resolved["tolerance"]),
            runtime_seconds=time.perf_counter() - start,
        )
        rows.append(row)
    frame = _ordered_frame(rows, SPARSE_ACCESS_COLUMNS)
    csv_path = output_dir / "sparse_access_diagnostics.csv"
    table_path = output_dir / "sparse_access_density_row_sparsity_table.csv"
    frame.to_csv(csv_path, index=False)
    frame[
        [
            "case_name",
            "num_rows",
            "num_cols",
            "density_weighted_H",
            "max_row_nnz",
            "mean_row_nnz",
            "max_col_nnz",
        ]
    ].to_csv(table_path, index=False)
    artifacts = {
        "sparse_access_diagnostics": csv_path,
        "sparse_access_density_row_sparsity_table": table_path,
    }
    if bool(resolved["save_plots"]):
        artifacts.update(_write_sparse_plot(output_dir, frame))
    return {"output_dir": output_dir, "diagnostics": frame, "artifacts": artifacts}


def ridge_svd_update(H_tilde: np.ndarray, r_tilde: np.ndarray, *, alpha: float) -> np.ndarray:
    matrix = np.asarray(H_tilde, dtype=np.float64)
    residual = np.asarray(r_tilde, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    return Vt.T @ (ridge_filter(singular_values, alpha=float(alpha)) * (U.T @ residual))


def pseudoinverse_svd_update(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    matrix = np.asarray(H_tilde, dtype=np.float64)
    residual = np.asarray(r_tilde, dtype=np.float64)
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    cutoff = float(rcond) * float(np.max(singular_values)) if singular_values.size else 0.0
    return Vt.T @ (inverse_filter(singular_values, eps=cutoff) * (U.T @ residual))


def select_deterministic_block(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    row_count: int,
    col_count: int,
    policy: str = "largest_row_col_norms",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(H_tilde, dtype=np.float64)
    residual = np.asarray(r_tilde, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("H_tilde must be two-dimensional")
    if residual.shape != (matrix.shape[0],):
        raise ValueError(f"r_tilde must have shape {(matrix.shape[0],)}")
    if row_count <= 0 or col_count <= 0:
        raise ValueError("row_count and col_count must be positive")
    if row_count > matrix.shape[0] or col_count > matrix.shape[1]:
        raise ValueError(
            f"requested block {row_count}x{col_count} exceeds matrix shape {matrix.shape}"
        )
    if policy == "top_left":
        rows = np.arange(row_count, dtype=np.int64)
        cols = np.arange(col_count, dtype=np.int64)
    elif policy == "largest_row_col_norms":
        cols = _top_indices(np.linalg.norm(matrix, axis=0), col_count)
        row_scores = np.linalg.norm(matrix[:, cols], axis=1)
        rows = _top_indices(row_scores, row_count)
    else:
        raise ValueError("selection policy must be 'top_left' or 'largest_row_col_norms'")
    return matrix[np.ix_(rows, cols)], residual[rows], rows, cols


def build_unweighted_and_weighted_ac_jacobian(
    *,
    case_name: str,
    case_source: str,
    seed: int,
    measurement_config: dict[str, Any] | None = None,
    linearization_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    measurement = {**DEFAULT_MEASUREMENT_CONFIG, **dict(measurement_config or {})}
    linearization = {**DEFAULT_LINEARIZATION_CONFIG, **dict(linearization_config or {})}
    rng = make_rng(int(seed))
    case = load_ac_case(case_name, case_source=case_source)
    true_state = default_ac_state_vector(case)
    angle_count = len(case.angle_state_buses)
    perturbation = np.concatenate(
        [
            rng.normal(
                0.0,
                float(linearization["angle_perturbation_std"]),
                size=angle_count,
            ),
            rng.normal(
                0.0,
                float(linearization["voltage_perturbation_std"]),
                size=len(case.voltage_state_buses),
            ),
        ]
    )
    state = true_state + perturbation
    state[angle_count:] = np.maximum(
        state[angle_count:],
        float(linearization["min_voltage_magnitude"]),
    )
    _, H, rows = ac_measurements_and_jacobian(case, state, measurement)
    stds = np.asarray([row.std for row in rows], dtype=np.float64)
    weighted_H = H / stds[:, None]
    return np.asarray(H, dtype=np.float64), np.asarray(weighted_H, dtype=np.float64)


def sparsity_pattern_preserved_by_diagonal_weighting(
    H: np.ndarray,
    weighted_H: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    raw_pattern = np.abs(np.asarray(H, dtype=np.float64)) > float(tolerance)
    weighted_pattern = np.abs(np.asarray(weighted_H, dtype=np.float64)) > float(tolerance)
    return bool(np.array_equal(raw_pattern, weighted_pattern))


def sparse_access_diagnostics_row(
    *,
    case_name: str,
    H: np.ndarray,
    weighted_H: np.ndarray,
    tolerance: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    raw = np.asarray(H, dtype=np.float64)
    weighted = np.asarray(weighted_H, dtype=np.float64)
    if raw.shape != weighted.shape:
        raise ValueError("H and weighted_H must have the same shape")
    raw_mask = np.abs(raw) > float(tolerance)
    weighted_mask = np.abs(weighted) > float(tolerance)
    row_counts = weighted_mask.sum(axis=1)
    col_counts = weighted_mask.sum(axis=0)
    rows, cols = weighted.shape
    extra = {
        "spectral_norm_estimate": float(np.linalg.svd(weighted, compute_uv=False)[0])
        if weighted.size
        else 0.0,
        "frobenius_norm": float(np.linalg.norm(weighted, ord="fro")),
    }
    row = {
        "case_name": case_name,
        "num_rows": int(rows),
        "num_cols": int(cols),
        "num_nonzeros_H": int(raw_mask.sum()),
        "num_nonzeros_weighted_H": int(weighted_mask.sum()),
        "density_H": float(raw_mask.sum() / raw.size),
        "density_weighted_H": float(weighted_mask.sum() / weighted.size),
        "max_row_nnz": int(row_counts.max()) if row_counts.size else 0,
        "mean_row_nnz": float(np.mean(row_counts)) if row_counts.size else 0.0,
        "median_row_nnz": float(np.median(row_counts)) if row_counts.size else 0.0,
        "max_col_nnz": int(col_counts.max()) if col_counts.size else 0,
        "mean_col_nnz": float(np.mean(col_counts)) if col_counts.size else 0.0,
        "median_col_nnz": float(np.median(col_counts)) if col_counts.size else 0.0,
        "row_sparsity_estimate": int(row_counts.max()) if row_counts.size else 0,
        "column_sparsity_estimate": int(col_counts.max()) if col_counts.size else 0,
        "sparsity_pattern_preserved": bool(np.array_equal(raw_mask, weighted_mask)),
        "weighting_is_diagonal": True,
        "sparse_access_notes": SPARSE_ACCESS_NOTE,
        "runtime_seconds": float(runtime_seconds),
    }
    row.update(extra)
    return row


def write_evidence_manifest(
    *,
    output_dir: Path,
    config: dict[str, Any],
    artifacts: dict[str, Path],
    skipped: dict[str, str],
) -> Path:
    path = output_dir / "manifest.json"
    output_files = {name: str(path) for name, path in sorted(artifacts.items())}
    output_files["manifest"] = str(path)
    manifest = {
        "git_commit_hash": _git_commit_hash(),
        "command_used": " ".join(sys.argv),
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "package_version": __version__,
        "key_package_versions": _package_versions(
            ["numpy", "pandas", "scipy", "matplotlib", "pypower"]
        ),
        "selected_cases": config["cases"],
        "selected_seeds": config["seeds"],
        "alpha_values": {
            "nonlinear_ridge_alpha": config["nonlinear"]["alpha"],
            "qsvt_matrix_alpha": config["qsvt_matrix"]["alpha"],
        },
        "tolerance_values": {
            "nonlinear_update_tolerance": config["nonlinear"]["update_tolerance"],
            "nonlinear_residual_tolerance": config["nonlinear"]["residual_tolerance"],
            "qsvt_target_epsilon": config["qsvt_matrix"]["target_epsilon"],
            "sparse_pattern_tolerance": config["sparse"]["tolerance"],
        },
        "experiment_configs": config,
        "output_files_generated": output_files,
        "skipped_tasks_and_reasons": skipped,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(path, manifest)
    return path


def write_evidence_readme(
    *,
    output_dir: Path,
    artifacts: dict[str, Path],
    skipped: dict[str, str],
) -> Path:
    lines = [
        "# TQE Revision Evidence Package",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Artifacts",
    ]
    for name, path in sorted(artifacts.items()):
        lines.append(f"- `{name}`: `{path.name}`")
    if skipped:
        lines.extend(["", "## Skipped Tasks"])
        for task, reason in sorted(skipped.items()):
            lines.append(f"- `{task}`: {reason}")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Ridge/Tikhonov damps weak singular-value directions; this is a "
            "stability-accuracy-speed tradeoff, not a guaranteed convergence-speed result.",
            "- Matrix-level polynomial validation checks the QSVT-compatible spectral-filter "
            "translation beyond 4x4 and 8x8 circuit examples.",
            "- Dense block encoding remains selected-subproblem validation only.",
            "- Sparse access is treated as the scalability pathway.",
        ]
    )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_single_nonlinear_solver(
    *,
    config: dict[str, Any],
    problem: ACNonlinearProblem,
    estimator: Estimator,
    solver_name: str,
    alpha: float,
    divergence_residual_growth_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iteration_config = config["system"]["iteration"]
    max_iterations = int(iteration_config["max_iterations"])
    update_tolerance = float(iteration_config["update_tolerance"])
    residual_tolerance = float(iteration_config["residual_tolerance"])
    damping = float(iteration_config["damping"])
    max_update_norm = float(iteration_config.get("max_update_norm", 1.0e6))
    residual_growth_limit = float(iteration_config.get("residual_growth_limit", 1.0e6))
    min_voltage = float(config["system"]["linearization"].get("min_voltage_magnitude", 0.5))
    x = problem.initial_state.copy()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    converged = False
    stop_reason = "max_iterations reached"
    initial_residual = _weighted_residual_norm(problem, x)

    for iteration in range(max_iterations):
        system, residual_before = _linearized_update_system(problem, x)
        singular_values = system.singular_values()
        update_result = estimator.solve(system)
        update = np.asarray(update_result.x_hat, dtype=np.float64)
        update_norm = float(np.linalg.norm(update)) if np.all(np.isfinite(update)) else np.nan
        failed = bool(update_result.failed or not np.all(np.isfinite(update)))
        if failed:
            stop_reason = update_result.failure_reason or "update solver failed"
            residual_after = np.nan
            state_rmse = np.nan
        elif update_norm > max_update_norm:
            failed = True
            stop_reason = f"update norm exceeded max_update_norm={max_update_norm}"
            residual_after = np.nan
            state_rmse = np.nan
        else:
            x_next = x + damping * update
            x_next[len(problem.case.angle_state_buses) :] = np.maximum(
                x_next[len(problem.case.angle_state_buses) :],
                min_voltage,
            )
            residual_after = _weighted_residual_norm(problem, x_next)
            state_rmse = _state_rmse(problem, x_next)
            if not np.isfinite(residual_after):
                failed = True
                stop_reason = "weighted residual became nonfinite"
            elif residual_before > 0.0 and residual_after / residual_before > residual_growth_limit:
                failed = True
                stop_reason = (
                    f"weighted residual growth exceeded residual_growth_limit="
                    f"{residual_growth_limit}"
                )
            else:
                converged = update_norm <= update_tolerance or residual_after <= residual_tolerance
                stop_reason = "converged" if converged else "running"
                x = x_next

        row_residual = residual_after if np.isfinite(residual_after) else residual_before
        row = _nonlinear_iteration_row(
            problem=problem,
            solver_name=solver_name,
            alpha=alpha,
            iteration=iteration,
            weighted_residual_norm=row_residual,
            update_norm=update_norm,
            state_rmse=state_rmse,
            singular_values=singular_values,
            H_tilde=system.H_tilde,
            converged=converged,
            stop_reason=stop_reason,
            runtime_seconds=time.perf_counter() - start,
        )
        rows.append(row)
        if failed or converged:
            break

    if rows and rows[-1]["stop_reason"] == "running":
        rows[-1]["stop_reason"] = "max_iterations reached"
    summary = _nonlinear_summary_row(
        rows=rows,
        solver_name=solver_name,
        alpha=alpha,
        problem=problem,
        runtime_seconds=time.perf_counter() - start,
        initial_residual=initial_residual,
        divergence_residual_growth_factor=divergence_residual_growth_factor,
    )
    return rows, summary


def _nonlinear_iteration_row(
    *,
    problem: ACNonlinearProblem,
    solver_name: str,
    alpha: float,
    iteration: int,
    weighted_residual_norm: float,
    update_norm: float,
    state_rmse: float,
    singular_values: np.ndarray,
    H_tilde: np.ndarray,
    converged: bool,
    stop_reason: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    mask = np.abs(H_tilde) > 1.0e-12
    sigma_min = float(np.min(singular_values)) if singular_values.size else np.nan
    sigma_max = float(np.max(singular_values)) if singular_values.size else np.nan
    condition = np.inf if sigma_min <= 1.0e-15 else sigma_max / sigma_min
    return {
        "case_name": problem.case.name,
        "seed": int(problem.config_metadata["seed"]),
        "perturbation_name": str(problem.config_metadata.get("scenario_name", "")),
        "solver_name": solver_name,
        "alpha": float(alpha),
        "iteration": int(iteration),
        "weighted_residual_norm": float(weighted_residual_norm),
        "update_norm": float(update_norm),
        "state_rmse": float(state_rmse),
        "weighted_jacobian_cond": float(condition),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "jacobian_nnz": int(mask.sum()),
        "jacobian_density": float(mask.sum() / mask.size),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "runtime_seconds": float(runtime_seconds),
    }


def _nonlinear_summary_row(
    *,
    rows: list[dict[str, Any]],
    solver_name: str,
    alpha: float,
    problem: ACNonlinearProblem,
    runtime_seconds: float,
    initial_residual: float,
    divergence_residual_growth_factor: float,
) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    final = frame.iloc[-1] if not frame.empty else pd.Series(dtype=object)
    residuals = pd.to_numeric(frame.get("weighted_residual_norm", pd.Series(dtype=float)))
    updates = pd.to_numeric(frame.get("update_norm", pd.Series(dtype=float)))
    rmses = pd.to_numeric(frame.get("state_rmse", pd.Series(dtype=float)))
    conds = pd.to_numeric(frame.get("weighted_jacobian_cond", pd.Series(dtype=float)))
    nonfinite = (
        bool(~np.isfinite(updates.dropna()).all())
        or bool(~np.isfinite(rmses.dropna()).all())
        or bool(~np.isfinite(residuals.dropna()).all())
    )
    residual_growth = (
        bool(residuals.max() > divergence_residual_growth_factor * initial_residual)
        if not residuals.dropna().empty and initial_residual > 0.0
        else False
    )
    converged = bool(final.get("converged", False))
    stop_reason = str(final.get("stop_reason", "not_started"))
    return {
        "case_name": problem.case.name,
        "seed": int(problem.config_metadata["seed"]),
        "perturbation_name": str(problem.config_metadata.get("scenario_name", "")),
        "solver_name": solver_name,
        "alpha": float(alpha),
        "converged": converged,
        "stop_reason": stop_reason,
        "num_iterations": len(rows),
        "final_weighted_residual_norm": _last_numeric(residuals),
        "final_update_norm": _last_numeric(updates),
        "final_state_rmse": _last_numeric(rmses),
        "min_state_rmse": float(rmses.min()) if not rmses.dropna().empty else np.nan,
        "max_update_norm": float(updates.max()) if not updates.dropna().empty else np.nan,
        "median_weighted_jacobian_cond": (
            float(conds.median()) if not conds.dropna().empty else np.nan
        ),
        "max_weighted_jacobian_cond": float(conds.max()) if not conds.dropna().empty else np.nan,
        "diverged_or_oscillated": bool(nonfinite or residual_growth or not converged),
        "runtime_seconds": float(runtime_seconds),
    }


def _qsvt_matrix_validation_row(
    *,
    case_name: str,
    block_id: str,
    H_block: np.ndarray,
    r_block: np.ndarray,
    selected_rows: np.ndarray,
    selected_cols: np.ndarray,
    alpha: float,
    target_epsilon: float,
    degrees: list[int],
    method: str,
    grid_size: int,
    selection_policy: str,
    runtime_start: float,
) -> dict[str, Any]:
    del selected_rows, selected_cols
    U, singular_values, Vt = np.linalg.svd(H_block, full_matrices=False)
    positive = singular_values[singular_values > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected block has no positive singular values")
    context = ApproximationContext(
        case_name=case_name,
        matrix_source="selected_weighted_jacobian_block",
        matrix_shape=f"{H_block.shape[0]}x{H_block.shape[1]}",
        beta=float(np.max(positive)),
        singular_values=positive,
        normalized_singular_values=positive / float(np.max(positive)),
        domain_min=max(float(np.min(positive) / np.max(positive)), np.finfo(float).eps),
        domain_max=1.0,
        source_note=f"deterministic {selection_policy} block",
    )
    best_result = None
    best_degree = None
    for degree in sorted({as_odd_degree(value) for value in degrees}):
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=float(alpha),
            degree=int(degree),
            method=method,
            grid_size=max(int(grid_size), int(degree) + 2),
        )
        grid_errors = result.pointwise_errors[
            np.asarray(result.evaluation_kind, dtype=object) == "grid"
        ]
        best_result = result
        best_degree = int(result.degree)
        if float(np.max(grid_errors)) <= float(target_epsilon):
            break
    if best_result is None or best_degree is None:
        raise RuntimeError("no polynomial approximation was evaluated")
    bounded_filter_values = _bounded_values_at_actual_singular_values(
        result=best_result,
        singular_values=singular_values,
    )
    qsvt_bounded_update = Vt.T @ (bounded_filter_values * (U.T @ r_block))
    qsvt_scaled_update = best_result.bounded_scaling_C * qsvt_bounded_update
    ridge_update = Vt.T @ (ridge_filter(singular_values, alpha=float(alpha)) * (U.T @ r_block))
    ridge_residual = float(np.linalg.norm(H_block @ ridge_update - r_block))
    qsvt_residual = float(np.linalg.norm(H_block @ qsvt_scaled_update - r_block))
    relative_update_error = _relative_error(qsvt_scaled_update, ridge_update)
    residual_gap = abs(qsvt_residual - ridge_residual) / max(ridge_residual, 1.0e-15)
    grid_errors = best_result.pointwise_errors[
        np.asarray(best_result.evaluation_kind, dtype=object) == "grid"
    ]
    singular_errors = best_result.pointwise_errors[
        np.asarray(best_result.evaluation_kind, dtype=object) == "actual_singular_value"
    ]
    grid_max_error = float(np.max(grid_errors))
    status = (
        "polynomial_tolerance_met_no_phase_synthesis"
        if grid_max_error <= float(target_epsilon)
        else "polynomial_tolerance_not_met_no_phase_synthesis"
    )
    return {
        "case_name": case_name,
        "block_id": block_id,
        "block_shape": f"{H_block.shape[0]}x{H_block.shape[1]}",
        "selection_policy": selection_policy,
        "alpha": float(alpha),
        "target_epsilon": float(target_epsilon),
        "degree": int(best_degree),
        "phase_synthesis_status": status,
        "sigma_min": float(np.min(positive)),
        "sigma_max": float(np.max(positive)),
        "condition_number": float(np.max(positive) / np.min(positive)),
        "target_scale_C": float(best_result.bounded_scaling_C),
        "grid_max_error": grid_max_error,
        "actual_singular_value_max_error": float(np.max(singular_errors)),
        "relative_update_error_vs_ridge": relative_update_error,
        "residual_gap_vs_ridge": float(residual_gap),
        "runtime_seconds": float(time.perf_counter() - runtime_start),
        "failure_reason": "",
    }


def _qsvt_failure_row(
    *,
    case_name: str,
    block_id: str,
    block_shape: str,
    selection_policy: str,
    alpha: float,
    target_epsilon: float,
    runtime_seconds: float,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "block_id": block_id,
        "block_shape": block_shape,
        "selection_policy": selection_policy,
        "alpha": float(alpha),
        "target_epsilon": float(target_epsilon),
        "degree": np.nan,
        "phase_synthesis_status": "failed_before_matrix_level_polynomial_validation",
        "sigma_min": np.nan,
        "sigma_max": np.nan,
        "condition_number": np.nan,
        "target_scale_C": np.nan,
        "grid_max_error": np.nan,
        "actual_singular_value_max_error": np.nan,
        "relative_update_error_vs_ridge": np.nan,
        "residual_gap_vs_ridge": np.nan,
        "runtime_seconds": float(runtime_seconds),
        "failure_reason": failure_reason,
    }


def _bounded_values_at_actual_singular_values(
    *,
    result: Any,
    singular_values: np.ndarray,
) -> np.ndarray:
    actual_mask = np.asarray(result.evaluation_kind, dtype=object) == "actual_singular_value"
    actual_values = np.asarray(result.bounded_approximation_values, dtype=np.float64)[actual_mask]
    positive_mask = np.asarray(singular_values, dtype=np.float64) > 1.0e-14
    if int(np.count_nonzero(positive_mask)) != actual_values.size:
        raise ValueError("polynomial evaluation points do not match positive singular values")
    bounded_filter_values = np.zeros_like(singular_values, dtype=np.float64)
    bounded_filter_values[positive_mask] = actual_values
    return bounded_filter_values


def _nonlinear_problem_config(
    resolved: dict[str, Any],
    *,
    case_name: str,
    seed: int,
) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update({"run_name": "tqe_revision_nonlinear", "seed": int(seed)})
    config["system"]["case_name"] = case_name
    config["system"]["case_source"] = resolved["case_source"]
    config["system"]["mode"] = "nonlinear_ac_state_estimation"
    config["system"]["measurement"] = deepcopy(resolved["measurement"])
    config["system"]["linearization"] = deepcopy(resolved["linearization"])
    config["system"]["iteration"] = {
        "max_iterations": int(resolved["max_iterations"]),
        "update_tolerance": float(resolved["update_tolerance"]),
        "residual_tolerance": float(resolved["residual_tolerance"]),
        "damping": float(resolved["damping"]),
        "max_update_norm": float(resolved["max_update_norm"]),
        "residual_growth_limit": float(resolved["residual_growth_limit"]),
    }
    config["scenario"] = {
        "name": str(resolved["perturbation_name"]),
        "noise_std": float(resolved["noise_std"]),
        "missing_ratio": float(resolved["missing_ratio"]),
        "bad_data": deepcopy(resolved["bad_data"]),
    }
    config["estimators"] = [
        {"name": "pseudoinverse", "rcond": float(resolved["rcond"])},
        {"name": "ridge", "alpha": float(resolved["alpha"])},
    ]
    config["output"] = {
        "root": str(resolved["output_dir"]),
        "run_id": "unused_internal",
        "save_plots": False,
        "overwrite": False,
    }
    return config


def _resolve_evidence_config(config: dict[str, Any] | None) -> dict[str, Any]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved = {
        "tasks": ["nonlinear", "qsvt-matrix", "sparse"],
        "cases": ["ieee14", "ieee30", "ieee57"],
        "case_source": "pypower",
        "seeds": [0, 1, 2],
        "output_dir": f"results/tqe_revision_evidence/{timestamp}",
        "nonlinear": {},
        "qsvt_matrix": {},
        "sparse": {},
    }
    if config:
        resolved.update({key: value for key, value in config.items() if key not in resolved})
        for key in ("tasks", "cases", "case_source", "seeds", "output_dir"):
            if key in config:
                resolved[key] = config[key]
        for key in ("nonlinear", "qsvt_matrix", "sparse"):
            resolved[key] = {**resolved[key], **dict(config.get(key, {}))}
    tasks = [str(task) for task in resolved["tasks"]]
    if "all" in tasks:
        tasks = ["nonlinear", "qsvt-matrix", "sparse"]
    resolved["tasks"] = tasks
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["seeds"] = [int(seed) for seed in resolved["seeds"]]
    if not resolved["seeds"]:
        raise ValueError("at least one seed is required")
    resolved["nonlinear"] = _resolve_nonlinear_config(resolved["nonlinear"])
    resolved["qsvt_matrix"] = _resolve_qsvt_matrix_config(resolved["qsvt_matrix"])
    resolved["sparse"] = _resolve_sparse_config(resolved["sparse"])
    return resolved


def _resolve_nonlinear_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "cases": ["ieee14", "ieee30", "ieee57"],
        "case_source": "pypower",
        "seeds": [0, 1, 2],
        "output_dir": "results/tqe_revision_evidence",
        "alpha": 1.0e-4,
        "rcond": 1.0e-12,
        "max_iterations": 8,
        "update_tolerance": 1.0e-7,
        "residual_tolerance": 1.0e-7,
        "damping": 1.0,
        "max_update_norm": 1000.0,
        "residual_growth_limit": 10000.0,
        "divergence_residual_growth_factor": 10.0,
        "noise_std": 0.002,
        "missing_ratio": 0.1,
        "perturbation_name": "controlled_noise_missing",
        "measurement": DEFAULT_MEASUREMENT_CONFIG,
        "linearization": DEFAULT_LINEARIZATION_CONFIG,
        "bad_data": {"enabled": False, "ratio": 0.0, "magnitude": 10.0, "target": "random"},
        "save_plots": True,
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["seeds"] = [int(seed) for seed in resolved["seeds"]]
    resolved["measurement"] = {**DEFAULT_MEASUREMENT_CONFIG, **dict(resolved["measurement"])}
    resolved["linearization"] = {**DEFAULT_LINEARIZATION_CONFIG, **dict(resolved["linearization"])}
    resolved["alpha"] = float(resolved["alpha"])
    if resolved["alpha"] <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def _resolve_qsvt_matrix_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "cases": ["ieee14", "ieee30", "ieee57"],
        "case_source": "pypower",
        "seed": 0,
        "output_dir": "results/tqe_revision_evidence",
        "block_sizes": [16, 32, 64],
        "selection_policy": "largest_row_col_norms",
        "alpha": 1.0e-4,
        "target_epsilon": 1.0e-3,
        "degrees": [15, 35, 51, 71],
        "polynomial_method": "odd_chebyshev_ls",
        "grid_size": 1024,
        "measurement": DEFAULT_MEASUREMENT_CONFIG,
        "linearization": DEFAULT_LINEARIZATION_CONFIG,
        "save_plots": True,
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["block_sizes"] = [int(size) for size in resolved["block_sizes"]]
    resolved["degrees"] = [as_odd_degree(int(degree)) for degree in resolved["degrees"]]
    resolved["measurement"] = {**DEFAULT_MEASUREMENT_CONFIG, **dict(resolved["measurement"])}
    resolved["linearization"] = {**DEFAULT_LINEARIZATION_CONFIG, **dict(resolved["linearization"])}
    return resolved


def _resolve_sparse_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "cases": ["ieee14", "ieee30", "ieee57"],
        "case_source": "pypower",
        "seed": 0,
        "output_dir": "results/tqe_revision_evidence",
        "measurement": DEFAULT_MEASUREMENT_CONFIG,
        "linearization": DEFAULT_LINEARIZATION_CONFIG,
        "tolerance": 1.0e-12,
        "save_plots": True,
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    resolved["measurement"] = {**DEFAULT_MEASUREMENT_CONFIG, **dict(resolved["measurement"])}
    resolved["linearization"] = {**DEFAULT_LINEARIZATION_CONFIG, **dict(resolved["linearization"])}
    return resolved


def _write_nonlinear_plots(output_dir: Path, frame: pd.DataFrame) -> dict[str, Path]:
    plt = _pyplot()

    metric_specs = {
        "weighted_residual_norm": "nonlinear_convergence_residual_vs_iter",
        "state_rmse": "nonlinear_convergence_rmse_vs_iter",
        "update_norm": "nonlinear_convergence_update_norm_vs_iter",
        "weighted_jacobian_cond": "nonlinear_convergence_condition_vs_iter",
    }
    artifacts: dict[str, Path] = {}
    for metric, stem in metric_specs.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for (case_name, solver_name), group in frame.groupby(["case_name", "solver_name"]):
            agg = (
                group.groupby("iteration", as_index=False)[metric]
                .mean(numeric_only=True)
                .sort_values("iteration")
            )
            ax.plot(agg["iteration"], agg[metric], marker="o", label=f"{case_name} {solver_name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.replace("_", " "))
        if metric in {"weighted_residual_norm", "state_rmse", "update_norm"}:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        artifacts.update(_save_figure(fig, output_dir, stem))
        plt.close(fig)
    return artifacts


def _write_qsvt_matrix_plot(output_dir: Path, frame: pd.DataFrame) -> dict[str, Path]:
    plt = _pyplot()

    success = frame[frame["failure_reason"].fillna("") == ""].copy()
    if success.empty:
        return {}
    success["block_size"] = success["block_shape"].str.split("x").str[0].astype(int)
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    for case_name, group in success.groupby("case_name"):
        group = group.sort_values("block_size")
        ax1.plot(
            group["block_size"],
            group["grid_max_error"],
            marker="o",
            label=f"{case_name} grid error",
        )
    ax1.set_xlabel("Block size")
    ax1.set_ylabel("Polynomial approximation error")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    for case_name, group in success.groupby("case_name"):
        group = group.sort_values("block_size")
        ax2.plot(
            group["block_size"],
            group["relative_update_error_vs_ridge"],
            marker="s",
            linestyle="--",
            label=f"{case_name} update error",
        )
    ax2.set_ylabel("Relative update error vs Ridge")
    ax2.set_yscale("log")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    artifacts = _save_figure(fig, output_dir, "larger_qsvt_matrix_validation_summary")
    plt.close(fig)
    return artifacts


def _write_sparse_plot(output_dir: Path, frame: pd.DataFrame) -> dict[str, Path]:
    plt = _pyplot()

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ordered = frame.sort_values("num_cols")
    ax1.bar(ordered["case_name"], ordered["density_weighted_H"], alpha=0.7, label="density")
    ax1.set_ylabel("Weighted Jacobian density")
    ax1.set_xlabel("Case")
    ax2 = ax1.twinx()
    ax2.plot(
        ordered["case_name"],
        ordered["max_row_nnz"],
        color="black",
        marker="o",
        label="max row nnz",
    )
    ax2.set_ylabel("Max row nonzeros")
    ax1.grid(True, axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    artifacts = _save_figure(fig, output_dir, "sparse_access_density_row_sparsity")
    plt.close(fig)
    return artifacts


def _save_figure(fig: Any, output_dir: Path, stem: str) -> dict[str, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    return {f"{stem}_png": png, f"{stem}_pdf": pdf}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _qsvt_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_name",
        "block_shape",
        "degree",
        "grid_max_error",
        "relative_update_error_vs_ridge",
        "phase_synthesis_status",
        "failure_reason",
    ]
    return frame.loc[:, columns].copy()


def _ordered_frame(rows: list[dict[str, Any]], required_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    extra = [column for column in frame.columns if column not in required_columns]
    return frame[required_columns + extra]


def _fresh_output_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"could not find a fresh output directory based on {path}")


def _state_rmse(problem: ACNonlinearProblem, state: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(state) - problem.true_state) ** 2)))


def _last_numeric(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-15)
    return float(np.linalg.norm(np.asarray(candidate) - np.asarray(reference)) / denominator)


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[:count]).astype(np.int64)


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TQE revision evidence package")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "nonlinear", "qsvt-matrix", "sparse"],
        help="Evidence tasks to run.",
    )
    parser.add_argument("--cases", nargs="+", default=["ieee14", "ieee30", "ieee57"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--polynomial-method", default="odd_chebyshev_ls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or f"results/tqe_revision_evidence/{timestamp}"
    config = {
        "tasks": args.tasks,
        "cases": args.cases,
        "case_source": args.case_source,
        "seeds": args.seeds,
        "output_dir": output_dir,
        "nonlinear": {"alpha": args.alpha, "save_plots": not args.no_plots},
        "qsvt_matrix": {
            "alpha": args.alpha,
            "block_sizes": args.block_sizes,
            "polynomial_method": args.polynomial_method,
            "save_plots": not args.no_plots,
        },
        "sparse": {"save_plots": not args.no_plots},
    }
    resolved = _resolve_evidence_config(config)
    if args.dry_run:
        write_json(Path("/tmp/tqe_revision_evidence_dry_run.json"), resolved)
        print(resolved)
        return
    run = run_tqe_revision_evidence(resolved)
    print(f"TQE revision evidence complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
