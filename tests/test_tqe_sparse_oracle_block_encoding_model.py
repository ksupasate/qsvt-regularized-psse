from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_sparse_oracle_block_encoding_model import (
    ORACLE_COLUMNS,
    RESOURCE_COLUMNS,
    SPARSITY_COLUMNS,
    SparseJacobianOracle,
    ceil_log2,
    resource_estimate_rows,
    run_sparse_oracle_block_encoding_model,
    sparsity_audit_row,
)


def _known_sparse_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 2.0, 0.0, -1.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_sparse_oracle_reconstructs_known_matrix() -> None:
    matrix = _known_sparse_matrix()
    oracle = SparseJacobianOracle.from_matrix(matrix)

    np.testing.assert_allclose(oracle.reconstruct_dense(), matrix)
    np.testing.assert_allclose(oracle.reconstruct_sparse().toarray(), matrix)
    diagnostics = oracle.validate_against_matrix(matrix)
    assert diagnostics["reconstruction_fro_error"] == 0.0
    assert diagnostics["reconstruction_max_abs_error"] == 0.0


def test_index_oracle_returns_sorted_nonzero_columns() -> None:
    oracle = SparseJacobianOracle.from_matrix(_known_sparse_matrix())

    assert [oracle.index_oracle(0, ell) for ell in range(oracle.row_nnz(0))] == [1, 3]
    assert [oracle.index_oracle(2, ell) for ell in range(oracle.row_nnz(2))] == [1, 2]


def test_padding_convention_returns_sentinel_and_zero_value() -> None:
    oracle = SparseJacobianOracle.from_matrix(_known_sparse_matrix())

    assert oracle.index_oracle(1, 1) == -1
    assert oracle.value_oracle_by_position(1, 1) == 0.0


def test_value_oracle_returns_nonzeros_and_zero_for_missing_entries() -> None:
    oracle = SparseJacobianOracle.from_matrix(_known_sparse_matrix())

    assert oracle.value_oracle(0, 1) == 2.0
    assert oracle.value_oracle(0, 3) == -1.0
    assert oracle.value_oracle(0, 0) == 0.0
    assert oracle.value_oracle(0, -1) == 0.0


def test_sparse_normalization_bound_and_resource_qubits() -> None:
    matrix = _known_sparse_matrix()
    oracle = SparseJacobianOracle.from_matrix(matrix)
    case = _synthetic_case(matrix)
    audit = sparsity_audit_row(case=case, oracle=oracle, nonzero_tol=1.0e-12)
    row = resource_estimate_rows(case=case, oracle=oracle, audit=audit, value_bits_grid=[16])[0]

    assert row["alpha_sparse_max"] == oracle.max_row_nnz() * np.max(np.abs(matrix))
    assert row["alpha_sparse_max"] >= np.linalg.norm(matrix, ord=2) - 1.0e-12
    assert row["row_qubits"] == ceil_log2(matrix.shape[0])
    assert row["col_qubits"] == ceil_log2(matrix.shape[1])
    assert row["nonzero_index_qubits"] == ceil_log2(oracle.max_row_nnz())
    assert ceil_log2(1) == 0
    assert ceil_log2(5) == 3


def test_sparse_oracle_output_schema_contains_required_columns(tmp_path: Path) -> None:
    run = run_sparse_oracle_block_encoding_model(
        {
            "output_root": str(tmp_path),
            "cases": [
                {
                    "case_name": "synthetic_sparse",
                    "measurement_setting": "unit_test_fixture",
                    "matrix": _known_sparse_matrix().tolist(),
                }
            ],
            "value_bits_grid": [16],
        }
    )
    sparsity = pd.read_csv(run["artifacts"]["sparsity_audit_csv"])
    verification = pd.read_csv(run["artifacts"]["oracle_verification_csv"])
    resources = pd.read_csv(run["artifacts"]["resource_estimates_csv"])

    assert set(SPARSITY_COLUMNS).issubset(sparsity.columns)
    assert set(ORACLE_COLUMNS).issubset(verification.columns)
    assert set(RESOURCE_COLUMNS).issubset(resources.columns)
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["density_figure"].is_file()
    assert run["artifacts"]["row_nnz_figure"].is_file()
    assert run["artifacts"]["normalization_overhead_figure"].is_file()
    assert run["artifacts"]["qubit_estimates_figure"].is_file()


def test_missing_case_is_recorded_without_dropping_rows(tmp_path: Path) -> None:
    run = run_sparse_oracle_block_encoding_model(
        {
            "output_root": str(tmp_path),
            "cases": [{"case_name": "missing_fixture", "force_missing": True}],
            "value_bits_grid": [16, 24],
        }
    )

    assert len(run["sparsity"]) == 1
    assert len(run["verification"]) == 1
    assert len(run["resources"]) == 2
    assert run["sparsity"].iloc[0]["run_status"] == "failed_or_skipped"
    assert run["verification"].iloc[0]["oracle_status"] == "skipped_input_unavailable"
    assert "forced missing" in run["sparsity"].iloc[0]["failure_or_skip_reason"]


def _synthetic_case(matrix: np.ndarray):
    from robust_qsvt_se.qsvt.tqe_sparse_oracle_block_encoding_model import WeightedJacobianCase

    return WeightedJacobianCase(
        case_name="synthetic_sparse",
        measurement_setting="unit_test_fixture",
        matrix=np.asarray(matrix, dtype=np.float64),
        matrix_source="synthetic_test_fixture",
        metadata={},
    )
