from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import (
    normalized_bounded_ridge_target,
    rescale_bounded_target_to_original,
    run_full_matrix_qsvt_demo,
)


def test_normalized_target_scales_back_to_original_ridge_filter() -> None:
    beta = 7.5
    alpha = 1.0e-3
    sigma = np.array([0.2, 1.0, 3.0], dtype=np.float64)
    s = sigma / beta
    alpha_norm = alpha / beta**2
    C = 11.0

    bounded = normalized_bounded_ridge_target(s, alpha_norm=alpha_norm, C=C)
    scaled_back = rescale_bounded_target_to_original(bounded, beta=beta, C=C)

    np.testing.assert_allclose(scaled_back, ridge_filter(sigma, alpha=alpha), rtol=1.0e-12)


def test_full_matrix_qsvt_demo_writes_outputs_and_validates_matrix_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_full_matrix_qsvt_demo(
        {
            "output_dir": str(tmp_path / "full_qsvt"),
            "case": "ieee14",
            "case_name": "ieee14",
            "matrix_source": "weighted_jacobian",
            "submatrix_size": 2,
            "alpha": 1.0e-4,
            "degree": 5,
            "max_synthesis_degree": 5,
            "grid_size": 512,
            "phase_timeout_seconds": 0,
        }
    )
    output_dir = run["output_dir"]
    expected = {
        "block_encoding_report.json",
        "matrix_metadata.json",
        "singular_values.csv",
        "qsvt_matrix_level_comparison.csv",
        "qsvt_state_solution_comparison.csv",
        "summary.md",
        "full_qsvt_small_matrix_demo.png",
    }

    assert expected.issubset({path.name for path in output_dir.iterdir()})

    block_report = json.loads((output_dir / "block_encoding_report.json").read_text())
    metadata = json.loads((output_dir / "matrix_metadata.json").read_text())
    comparison = pd.read_csv(output_dir / "qsvt_matrix_level_comparison.csv")
    state_comparison = pd.read_csv(output_dir / "qsvt_state_solution_comparison.csv")
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert metadata["matrix_source"] == "ieee14_pypower_ac_weighted_jacobian"
    assert metadata["matrix_orientation"] == "B = H_tilde.T"
    assert block_report["unitarity_error"] <= 1.0e-7
    assert block_report["top_left_block_error"] <= 1.0e-10
    assert comparison["abs_error_vs_polynomial_svd"].max() <= 1.0e-5
    assert state_comparison["abs_error_vs_ridge"].max() <= 1.0e-5
    assert "does not demonstrate full IEEE-scale quantum execution or quantum speedup" in summary_md
