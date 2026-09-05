from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_nonlinear_ac_per_iteration_feasibility import (
    ITERATION_COLUMNS,
    matrix_spectral_diagnostics,
    ridge_update_and_residual_ratio,
    run_nonlinear_ac_per_iteration_feasibility,
    select_required_degree_from_errors,
    sparse_iteration_diagnostics,
)


def test_iteration_diagnostic_schema_from_mock_run(tmp_path: Path) -> None:
    run = _run_tiny_mock(tmp_path)
    frame = pd.read_csv(run["artifacts"]["diagnostics_csv"])

    assert set(ITERATION_COLUMNS).issubset(frame.columns)
    assert len(frame) == 2
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["run_summary_csv"].is_file()
    assert run["artifacts"]["report"].is_file()


def test_condition_number_diagnostics_for_known_singular_values() -> None:
    diagnostics = matrix_spectral_diagnostics(np.diag([3.0, 2.0, 0.0]), nonzero_tol=1.0e-12)

    assert diagnostics.sigma_max == 3.0
    assert diagnostics.sigma_min_nonzero == 2.0
    assert diagnostics.condition_number == 1.5
    assert diagnostics.numerical_rank == 2


def test_required_degree_selection_uses_smallest_passing_degree() -> None:
    required, best, best_error, target_met = select_required_degree_from_errors(
        {5: 2.0e-2, 11: 8.0e-3, 21: 1.0e-3},
        1.0e-2,
    )

    assert required == 11
    assert best == 21
    assert best_error == 1.0e-3
    assert target_met is True


def test_required_degree_selection_records_no_passing_degree() -> None:
    required, best, best_error, target_met = select_required_degree_from_errors(
        {5: 2.0e-2, 11: 8.0e-3},
        1.0e-4,
    )

    assert required is None
    assert best == 11
    assert best_error == 8.0e-3
    assert target_met is False


def test_ridge_residual_ratio_matches_direct_tikhonov_solve() -> None:
    A = np.array([[2.0, 0.0], [0.0, 0.5], [1.0, 1.0]], dtype=np.float64)
    b = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    alpha = 1.0e-2

    update, _, ratio = ridge_update_and_residual_ratio(A, b, alpha)
    direct = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)

    np.testing.assert_allclose(update, direct, atol=1.0e-12)
    assert np.isfinite(ratio)
    assert ratio == np.linalg.norm(A @ update - b) / np.linalg.norm(b)


def test_sparse_iteration_diagnostics_for_known_matrix() -> None:
    matrix = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float64)
    diagnostics = sparse_iteration_diagnostics(matrix)

    assert diagnostics["nnz"] == 3
    assert diagnostics["density"] == 0.5
    assert diagnostics["sparse_max_row_nnz"] == 2
    assert diagnostics["sparse_alpha_max"] == 8.0
    assert diagnostics["row_qubits"] == 1
    assert diagnostics["col_qubits"] == 2
    assert diagnostics["nonzero_index_qubits"] == 1


def test_failure_or_skipped_iteration_is_recorded(tmp_path: Path) -> None:
    run = run_nonlinear_ac_per_iteration_feasibility(
        {
            "output_root": str(tmp_path),
            "alpha_grid": [1.0e-2],
            "epsilon_targets": [1.0e-2],
            "degree_grid": [5],
            "mock_iterations": [
                {
                    "case_name": "mock_missing",
                    "stress_setting": "clean_noise",
                    "seed": 0,
                    "estimator": "ridge",
                    "force_failure": True,
                }
            ],
        }
    )
    frame = pd.read_csv(run["artifacts"]["diagnostics_csv"])

    assert len(frame) == 1
    assert frame.iloc[0]["simulation_status"] == "skipped_input_unavailable"
    assert "forced mock iteration failure" in frame.iloc[0]["failure_or_skip_reason"]


def test_output_smoke_generates_figures_and_manifest(tmp_path: Path) -> None:
    run = _run_tiny_mock(tmp_path)

    assert run["artifacts"]["condition_number_figure"].is_file()
    assert run["artifacts"]["required_degree_figure"].is_file()
    assert run["artifacts"]["rmse_residual_figure"].is_file()
    assert run["artifacts"]["sparse_overhead_figure"].is_file()
    assert run["artifacts"]["manifest"].is_file()
    assert run["artifacts"]["final_report"].is_file()


def _run_tiny_mock(tmp_path: Path) -> dict[str, object]:
    return run_nonlinear_ac_per_iteration_feasibility(
        {
            "output_root": str(tmp_path),
            "alpha_grid": [1.0e-2],
            "epsilon_targets": [1.0e-1, 1.0e-2],
            "degree_grid": [5, 10],
            "dense_grid_size": 64,
            "mock_iterations": [
                {
                    "case_name": "mock_ieee",
                    "stress_setting": "clean_noise",
                    "seed": 0,
                    "estimator": "ridge",
                    "iteration": 0,
                    "H_tilde": [[2.0, 0.0], [0.0, 0.5], [1.0, 1.0]],
                    "r_tilde": [1.0, -0.5, 0.25],
                    "update_norm": 0.1,
                    "rmse": 0.01,
                    "converged": True,
                    "stopping_reason": "mock_converged",
                }
            ],
        }
    )
