from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.research_matrix import extract_research_matrix


def test_canonical_block_encoding_matches_research_matrix_top_left() -> None:
    research_matrix = extract_research_matrix(
        {
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "matrix_scope": "submatrix",
                "submatrix_size": 2,
                "selection_strategy": "high_leverage",
                "seed": 123,
            }
        }
    )
    block_encoding = canonical_square_block_encoding(research_matrix.normalized_matrix)
    validation = validate_block_encoding(block_encoding)

    singular_values = np.linalg.svd(block_encoding.source_matrix, compute_uv=False)
    assert float(singular_values[0]) <= 1.0 + 1.0e-12
    assert validation["top_left_block_valid"] is True
    assert validation["unitarity_valid"] is True
    assert block_encoding.summary["uses_dense_block_encoding_gate"] is True
    np.testing.assert_allclose(
        block_encoding.unitary[:2, :2],
        research_matrix.normalized_matrix,
        atol=1.0e-10,
    )
