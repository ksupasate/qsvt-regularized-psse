from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.sparse_access_oracle import build_sparse_access_oracle


def test_sparse_oracle_row_access_matches_dense_entries() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, -3.0, 0.0]])
    oracle = build_sparse_access_oracle(matrix)

    assert oracle.get_row_nonzero_col(0, 0) == 0
    assert oracle.get_row_nonzero_col(0, 1) == 2
    assert oracle.get_value(1, 1) == -3.0
    assert oracle.get_value(1, 2) == 0.0


def test_sparse_oracle_matvec_and_rmatvec_match_dense() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, -3.0, 0.5]])
    oracle = build_sparse_access_oracle(matrix)
    x = np.array([0.2, -0.1, 0.4])
    y = np.array([1.5, -2.0])

    np.testing.assert_allclose(oracle.matvec(x), matrix @ x)
    np.testing.assert_allclose(oracle.rmatvec(y), matrix.T @ y)


def test_sparse_oracle_sparsity_statistics_are_correct() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, -3.0, 0.5]])
    oracle = build_sparse_access_oracle(matrix)

    assert oracle.nnz == 4
    assert oracle.max_row_sparsity == 2
    assert oracle.max_col_sparsity == 2


def test_sparse_oracle_does_not_materialize_large_dense_matrix() -> None:
    matrix = np.eye(80)
    oracle = build_sparse_access_oracle(matrix)

    assert oracle.to_dense_if_small(max_dimension=16) is None
