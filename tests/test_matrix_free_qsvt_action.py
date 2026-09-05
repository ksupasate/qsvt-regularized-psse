from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.matrix_free_qsvt_action import (
    run_exact_svd_filter_action,
    run_matrix_free_polynomial_filter_action,
)
from robust_qsvt_se.qsvt.sparse_access_oracle import build_sparse_access_oracle


def test_matrix_free_action_approximates_exact_svd_on_tiny_matrix() -> None:
    matrix = np.array([[1.0, 0.0], [0.2, 0.5], [0.0, -0.3]])
    residual = np.array([1.0, -0.2, 0.4])
    exact = run_exact_svd_filter_action(matrix, residual, alpha=0.2)
    oracle = build_sparse_access_oracle(matrix)

    result = run_matrix_free_polynomial_filter_action(
        oracle,
        residual,
        alpha=0.2,
        degree=25,
    )

    relative_error = np.linalg.norm(result.update_vector - exact.update_vector) / np.linalg.norm(
        exact.update_vector
    )
    assert relative_error < 0.05


def test_matrix_free_error_does_not_catastrophically_increase_with_degree() -> None:
    matrix = np.array([[0.8, 0.0], [0.0, 0.5], [0.1, -0.2]])
    residual = np.array([0.4, -0.7, 0.3])
    oracle = build_sparse_access_oracle(matrix)

    low = run_matrix_free_polynomial_filter_action(oracle, residual, alpha=0.3, degree=5)
    high = run_matrix_free_polynomial_filter_action(oracle, residual, alpha=0.3, degree=25)

    assert high.relative_error_vs_exact_svd is not None
    assert low.relative_error_vs_exact_svd is not None
    assert high.relative_error_vs_exact_svd <= 2.0 * low.relative_error_vs_exact_svd


def test_matrix_free_large_oracle_does_not_build_dense_reference() -> None:
    matrix = np.eye(520)
    residual = np.ones(520)
    oracle = build_sparse_access_oracle(matrix)

    result = run_matrix_free_polynomial_filter_action(
        oracle,
        residual,
        alpha=0.5,
        degree=3,
    )

    assert result.error_vs_exact_svd is None
    assert result.matvec_calls > 0
    assert result.rmatvec_calls > 0


def test_matrix_free_counts_sparse_calls() -> None:
    matrix = np.eye(3)
    residual = np.ones(3)
    oracle = build_sparse_access_oracle(matrix)

    result = run_matrix_free_polynomial_filter_action(
        oracle,
        residual,
        alpha=0.5,
        degree=7,
    )

    assert result.matvec_calls == 7
    assert result.rmatvec_calls == 8
