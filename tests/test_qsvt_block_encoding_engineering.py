from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.block_encoding import (
    build_dense_block_encoding,
    extract_encoded_block,
    normalize_for_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.block_encoding_demo import run_block_encoding_demo


def test_rectangular_dense_block_encoding_validates() -> None:
    matrix = np.array(
        [
            [2.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.1],
        ],
        dtype=np.float64,
    )

    normalized, beta = normalize_for_block_encoding(matrix)
    assert np.linalg.svd(normalized, compute_uv=False)[0] <= 1.0 + 1.0e-12

    unitary = build_dense_block_encoding(normalized)
    validation = validate_block_encoding(normalized, unitary, beta=beta)

    assert validation["passed"] is True
    assert validation["encoded_block_error"] <= 1.0e-8
    assert validation["unitarity_error"] <= 1.0e-8
    np.testing.assert_allclose(
        extract_encoded_block(unitary, normalized.shape),
        normalized,
        atol=1.0e-10,
    )


def test_block_encoding_demo_writes_manifest_and_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_block_encoding_demo(
        {
            "output_dir": str(tmp_path / "block"),
            "matrix_source": "synthetic",
            "rectangular_shape": [3, 2],
        }
    )
    frame = run["summary"]
    output_dir = run["output_dir"]

    assert len(frame) == 2
    assert frame["passed"].all()
    assert (output_dir / "block_encoding_summary.csv").is_file()
    assert (output_dir / "block_encoding_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
