from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_end_to_end_qsvt_vs_ridge import (
    END_TO_END_RESULTS_COLUMNS,
    qsvt_polynomial_filter_values,
    residual_metrics,
    ridge_update_svd,
    run_end_to_end_qsvt_vs_ridge,
)


def test_ridge_update_svd_matches_direct_solve() -> None:
    A = np.array(
        [
            [1.0, 0.4],
            [0.2, 2.0],
            [0.3, -0.1],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -0.2, 0.5], dtype=np.float64)
    alpha = 1.0e-2

    svd_update = ridge_update_svd(A, b, alpha=alpha)
    direct_update = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)

    np.testing.assert_allclose(svd_update, direct_update, atol=1.0e-12, rtol=1.0e-12)


def test_normalized_polynomial_rescaling_reproduces_ridge_filter() -> None:
    singular_values = np.array([2.0, 0.5], dtype=np.float64)
    alpha = 2.0e-1
    gamma = float(np.max(singular_values))

    polynomial_filter, diagnostics = qsvt_polynomial_filter_values(
        singular_values,
        alpha=alpha,
        gamma=gamma,
        degree=101,
    )
    ridge_filter = singular_values / (singular_values**2 + alpha)

    np.testing.assert_allclose(polynomial_filter, ridge_filter, atol=1.0e-7, rtol=1.0e-7)
    assert diagnostics["actual_singular_value_error"] < 1.0e-7


def test_residual_metrics_are_finite_and_correct() -> None:
    A = np.eye(2)
    b = np.array([3.0, 4.0])
    ridge_update = np.array([2.0, 4.0])
    qsvt_update = np.array([2.5, 4.0])

    metrics = residual_metrics(A, b, ridge_update, qsvt_update)

    assert metrics["residual_no_update"] == 5.0
    assert metrics["residual_ridge"] == 1.0
    assert metrics["residual_qsvt_poly"] == 0.5
    assert metrics["ridge_residual_ratio"] == 0.2
    assert metrics["qsvt_residual_ratio"] == 0.1
    assert np.isfinite(list(metrics.values())).all()


def test_end_to_end_output_csv_contains_required_columns(tmp_path: Path) -> None:
    run = run_end_to_end_qsvt_vs_ridge(
        {
            "output_root": str(tmp_path),
            "subproblems": [
                {
                    "case_name": "synthetic",
                    "subproblem_size": 2,
                    "selection_mode": "synthetic_fixed",
                    "matrix": [[2.0, 0.0], [0.0, 0.5]],
                    "r_tilde": [1.0, -0.25],
                }
            ],
            "alpha_grid": [1.0e-2],
            "epsilon_targets": [1.0e-2],
            "degree_grid": [5],
            "degree_summary_path": None,
            "degree_results_path": None,
            "gate_validation_max_cases": 0,
        }
    )
    frame = pd.read_csv(run["artifacts"]["results_csv"])

    assert set(END_TO_END_RESULTS_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1
    assert frame.loc[0, "run_status"] == "completed"
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["relative_update_error_figure"].is_file()
    assert run["artifacts"]["residual_ratio_figure"].is_file()
    assert run["artifacts"]["update_scatter_figure"].is_file()


def test_end_to_end_records_subproblem_failures(tmp_path: Path) -> None:
    run = run_end_to_end_qsvt_vs_ridge(
        {
            "output_root": str(tmp_path),
            "subproblems": [
                {
                    "case_name": "bad_synthetic",
                    "subproblem_size": 2,
                    "selection_mode": "synthetic_bad_residual",
                    "matrix": [[1.0, 0.0], [0.0, 1.0]],
                    "r_tilde": [1.0],
                }
            ],
            "alpha_grid": [1.0e-2],
            "epsilon_targets": [1.0e-2],
            "degree_summary_path": None,
            "degree_results_path": None,
        }
    )
    frame = pd.read_csv(run["artifacts"]["results_csv"])

    assert len(frame) == 1
    assert frame.loc[0, "run_status"] == "failed"
    assert frame.loc[0, "gate_simulation_status"] == "skipped_subproblem_failed"
    assert "r_tilde length" in frame.loc[0, "failure_or_skip_reason"]
