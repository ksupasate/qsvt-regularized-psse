from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.failure_fix import run_preconditioned_ieee300_estimator


def test_preconditioned_estimator_variants_are_labeled_separately(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_preconditioned_ieee300_estimator(
        {
            "output_dir": str(tmp_path / "preconditioned"),
            "cases": ["synthetic"],
            "degree": 15,
            "grid_size": 80,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "preconditioned_ieee300_estimator_summary.csv")
    approximation = pd.read_csv(output_dir / "preconditioned_ieee300_qsvt_approximation.csv")

    assert {
        "unpreconditioned_ridge",
        "preconditioned_ridge_column_equilibrated_coordinate_penalty",
        "preconditioned_ridge_column_equilibrated_transformed_penalty",
    }.issubset(set(summary["variant_name"]))
    assert not summary["variant_name"].str.contains("coordinate.*transformed").any()
    assert np.isfinite(summary["residual_norm"]).all()
    assert {"full_interval_error_before", "full_interval_error_after"}.issubset(
        approximation.columns
    )
    assert np.isfinite(approximation["full_interval_error_after"]).all()
    assert (output_dir / "preconditioned_ieee300_solution_metrics.csv").is_file()
    assert (output_dir / "preconditioned_ieee300_spectral_metrics.csv").is_file()
    assert (output_dir / "preconditioned_ieee300_report.md").is_file()
