from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt import tqe_degree_alpha_precision_sweep as degree_sweep
from robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep import (
    DEGREE_RESULTS_COLUMNS,
    SweepSubproblem,
    bounded_ridge_normalization_C,
    bounded_ridge_target,
    evaluate_degree_candidate,
    run_degree_alpha_precision_sweep,
    select_required_degrees,
)
from robust_qsvt_se.qsvt.tqe_explicit_block_encoding_demo import (
    BLOCK_RESULTS_COLUMNS,
    construct_padded_block_encoding,
    next_power_of_two,
    pad_to_square_power_of_two,
    run_explicit_block_encoding_demo,
    verify_padded_block_encoding,
)


def test_bounded_ridge_target_is_bounded_on_normalized_domain() -> None:
    alpha = 1.0e-6
    beta = 4.0
    C_alpha = bounded_ridge_normalization_C(alpha, beta)
    grid = np.linspace(-1.0, 1.0, 5001)
    values = bounded_ridge_target(grid, alpha=alpha, beta=beta, C_alpha=C_alpha)

    assert C_alpha >= 1.0
    assert np.max(np.abs(values)) <= 1.0 + 1.0e-12


def test_required_degree_selection_returns_smallest_passing_degree() -> None:
    rows = []
    for degree, error in [(5, 3.0e-2), (11, 8.0e-3), (15, 2.0e-3)]:
        row = {key: np.nan for key in DEGREE_RESULTS_COLUMNS}
        row.update(
            {
                "case_name": "toy",
                "subproblem_size": 2,
                "selection_criterion": "unit",
                "matrix_shape": "2x2",
                "alpha": 1.0e-3,
                "epsilon_target": 1.0e-2,
                "degree": degree,
                "requested_degree": degree,
                "degree_grid_index": len(rows),
                "max_approximation_error_on_actual_singular_values": error,
                "max_approximation_error_on_dense_grid": error,
                "run_status": "completed",
            }
        )
        rows.append(row)

    summary = select_required_degrees(pd.DataFrame(rows))

    assert int(summary.loc[0, "required_degree"]) == 11
    assert int(summary.loc[0, "best_available_degree"]) == 15
    assert summary.loc[0, "degree_selection_status"] == "met_target"


def test_degree_sweep_records_polynomial_failures(monkeypatch) -> None:
    def fail_fit(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(degree_sweep, "fit_bounded_ridge_polynomial", fail_fit)
    subproblem = SweepSubproblem(
        H_tilde=np.eye(2),
        r_tilde=np.array([1.0, 0.0]),
        metadata={
            "case_name": "toy",
            "subproblem_size": 2,
            "selection_mode": "unit_test",
        },
    )

    rows, coefficients = evaluate_degree_candidate(
        subproblem=subproblem,
        alpha=1.0e-3,
        epsilon_targets=[1.0e-2, 1.0e-3],
        requested_degree=5,
        degree_grid_index=0,
        dense_grid_size=257,
    )

    assert coefficients is None
    assert {row["run_status"] for row in rows} == {"failed"}
    assert all(row["failure_mode"] == "polynomial_approximation_failed" for row in rows)
    assert all("synthetic fit failure" in row["failure_reason"] for row in rows)


def test_degree_sweep_output_csv_contains_required_columns(tmp_path: Path) -> None:
    run = run_degree_alpha_precision_sweep(
        {
            "output_root": str(tmp_path),
            "subproblems": [
                {
                    "case_name": "synthetic",
                    "subproblem_size": 2,
                    "selection_mode": "synthetic_fixed",
                    "matrix": [[2.0, 0.0], [0.0, 0.2]],
                    "r_tilde": [1.0, 0.5],
                }
            ],
            "alpha_grid": [1.0e-2],
            "epsilon_targets": [1.0e-2],
            "degree_grid": [5],
            "dense_grid_size": 257,
            "phase_synthesis_max_attempts": 0,
        }
    )
    csv_path = run["artifacts"]["results_csv"]
    frame = pd.read_csv(csv_path)

    assert set(DEGREE_RESULTS_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["figure"].is_file()


def test_constructed_padded_block_encoding_is_unitary_and_matches_block() -> None:
    matrix = np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.1]], dtype=np.float64)
    encoding = construct_padded_block_encoding(matrix)
    verification = verify_padded_block_encoding(encoding)
    top_left_original = encoding.A_bar_padded[: matrix.shape[0], : matrix.shape[1]]

    assert encoding.gamma >= np.linalg.svd(matrix, compute_uv=False)[0] * (1.0 - 1.0e-12)
    assert encoding.padded_dimension == 4
    assert verification["block_error_frobenius"] <= 1.0e-9
    assert verification["unitarity_error_frobenius"] <= 1.0e-8
    assert verification["spectral_norm_A_bar"] <= 1.0 + 1.0e-12
    np.testing.assert_allclose(top_left_original, matrix / encoding.gamma)


def test_padding_preserves_original_matrix_block() -> None:
    matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    padded = pad_to_square_power_of_two(matrix)

    assert padded.shape == (4, 4)
    assert next_power_of_two(3) == 4
    np.testing.assert_allclose(padded[:1, :3], matrix)
    np.testing.assert_allclose(padded[1:, :], 0.0)


def test_block_encoding_output_csv_contains_required_columns(tmp_path: Path) -> None:
    run = run_explicit_block_encoding_demo(
        {
            "output_root": str(tmp_path),
            "subproblems": [
                {
                    "case_name": "synthetic",
                    "subproblem_size": 2,
                    "selection_mode": "synthetic_fixed",
                    "matrix": [[1.0, 0.2], [0.1, 0.7]],
                    "r_tilde": [1.0, 0.0],
                }
            ],
            "save_unitary_dimension_limit": 16,
        }
    )
    csv_path = run["artifacts"]["results_csv"]
    frame = pd.read_csv(csv_path)

    assert set(BLOCK_RESULTS_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "run_status"] == "completed"
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["figure"].is_file()
