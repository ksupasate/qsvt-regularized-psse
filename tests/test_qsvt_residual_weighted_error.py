from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.failure_fix import diagnose_ieee300_residual_weighted_error


def test_residual_weighted_diagnostics_do_not_claim_full_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = diagnose_ieee300_residual_weighted_error(
        {
            "output_dir": str(tmp_path / "residual_weighted"),
            "case_name": "synthetic",
            "matrix_source": "synthetic",
            "degree": 15,
            "grid_size": 80,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "residual_weighted_error_summary.csv")
    contributions = pd.read_csv(output_dir / "singular_direction_contributions.csv")
    report = (output_dir / "residual_weighted_error_report.md").read_text()

    assert {
        "max_pointwise_error",
        "max_residual_weighted_error",
        "sum_residual_weighted_error",
        "top_1_percent_error_contribution",
        "top_5_percent_error_contribution",
        "interpretation",
    }.issubset(summary.columns)
    assert {
        "case_name",
        "alpha",
        "singular_index",
        "sigma",
        "target_filter_value",
        "approx_filter_value",
        "pointwise_error",
        "abs_residual_projection",
        "target_contribution",
        "approx_error_contribution",
        "relative_contribution_rank",
    }.issubset(contributions.columns)
    assert "do not replace full-interval validation" in report
    assert (output_dir / "top_error_directions.csv").is_file()
