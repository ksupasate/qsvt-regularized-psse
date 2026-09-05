from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.scalable_block_encoding import (
    BlockEncodingModel,
    estimate_block_encoding_resources,
)


def test_explicit_dense_resource_estimator_reports_qubits_and_error() -> None:
    matrix = np.array([[0.2, 0.1], [0.0, 0.3]])

    estimate = estimate_block_encoding_resources(
        matrix,
        BlockEncodingModel.EXPLICIT_DENSE,
        explicit_dimension_limit=2,
    )

    assert estimate.model == "explicit_dense"
    assert estimate.row_qubits == 1
    assert estimate.col_qubits == 1
    assert estimate.block_encoding_error is not None
    assert estimate.block_encoding_error < 1.0e-12


def test_sparse_oracle_resource_model_does_not_build_dense_unitary() -> None:
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    estimate = estimate_block_encoding_resources(matrix, "sparse_access_oracle")

    assert estimate.model == "sparse_access_oracle"
    assert estimate.block_encoding_error is None
    assert estimate.query_cost_per_block_encoding == 3
    assert "oracle construction is not implemented" in estimate.limitations
