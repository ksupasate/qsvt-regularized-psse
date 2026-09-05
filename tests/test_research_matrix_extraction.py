from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.research_matrix import (
    extract_qsvt_submatrix,
    extract_research_matrix,
    extract_weighted_jacobian_matrix,
)


def test_research_matrix_extraction_is_deterministic_and_normalized() -> None:
    config = {
        "matrix": {
            "case_name": "ieee14",
            "case_source": "pypower",
            "submatrix_size": 2,
            "seed": 123,
        }
    }

    first = extract_research_matrix(config)
    second = extract_research_matrix(config)

    assert np.allclose(first.normalized_matrix, second.normalized_matrix)
    assert first.normalized_matrix.shape == (2, 2)
    assert np.linalg.svd(first.normalized_matrix, compute_uv=False)[0] <= 1.0 + 1.0e-12
    assert first.metadata["source_case_name"] == "ieee14"
    assert len(first.metadata["selected_rows"]) == 2
    assert len(first.metadata["selected_columns"]) == 2
    assert first.metadata["normalization_factor"] > 0.0


def test_full_weighted_jacobian_and_submatrix_api_are_deterministic() -> None:
    full = extract_weighted_jacobian_matrix(
        case_name="ieee14",
        mode="ac_weighted_jacobian",
        case_source="pypower",
        seed=123,
    )
    submatrix = extract_qsvt_submatrix(
        full,
        target_shape=(4, 4),
        strategy="high_leverage",
        seed=123,
    )
    repeated = extract_qsvt_submatrix(
        full,
        target_shape=(4, 4),
        strategy="high_leverage",
        seed=123,
    )

    assert full.normalized_matrix.shape == (82, 27)
    assert full.metadata["is_full_matrix"] is True
    assert full.metadata["full_shape"] == [82, 27]
    assert np.linalg.svd(full.normalized_matrix, compute_uv=False)[0] <= 1.0 + 1.0e-12
    assert submatrix.normalized_matrix.shape == (4, 4)
    assert submatrix.metadata["is_full_matrix"] is False
    assert submatrix.metadata["used_shape"] == [4, 4]
    assert np.allclose(submatrix.normalized_matrix, repeated.normalized_matrix)
