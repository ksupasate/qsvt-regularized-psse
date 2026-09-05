from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.alpha_resource_sensitivity import run_alpha_resource_sensitivity


def test_alpha_sensitivity_has_one_row_per_alpha(tmp_path) -> None:  # type: ignore[no-untyped-def]
    alpha_grid = [1.0e-3, 1.0e-2, 1.0e-1]
    run = run_alpha_resource_sensitivity(
        {
            "output_dir": str(tmp_path / "alpha"),
            "matrix_source": "synthetic",
            "alpha_grid": alpha_grid,
            "degrees": [2, 4],
            "epsilon": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "alpha_resource_sensitivity.csv")
    polynomial = pd.read_csv(output_dir / "selected_alpha_polynomial_validation.csv")

    assert len(frame) == len(alpha_grid)
    assert set(frame["alpha"]) == set(alpha_grid)
    assert {
        "max_filter_gain",
        "bounded_scaling_C",
        "max_bounded_filter_value",
        "estimated_qsvt_degree",
        "estimated_query_count",
    }.issubset(frame.columns)
    assert (frame["estimated_query_count"] >= 1).all()
    assert (frame["qsvt_target_relative_error_vs_ridge_if_available"] <= 1.0e-12).all()
    assert set(polynomial["alpha"]) == {1.0e-4, 1.0e-2, 1.0}
    assert {
        "bounded_scaling_C",
        "polynomial_degree",
        "max_pointwise_target_error",
        "query_count",
        "passed",
    }.issubset(polynomial.columns)
    assert (polynomial["query_count"] >= 1).all()
    assert (output_dir / "alpha_resource_sensitivity.json").is_file()
    assert (output_dir / "selected_alpha_polynomial_validation.json").is_file()
    assert (output_dir / "manifest.json").is_file()
