from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.nonbruteforce_refinement import run_spectrum_aware_diagnostics


def test_spectrum_aware_diagnostics_include_caveats(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_spectrum_aware_diagnostics(
        {
            "output_dir": str(tmp_path / "spectrum_aware"),
            "cases": ["synthetic"],
            "degree": 15,
            "grid_size": 80,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "spectrum_aware_summary.csv")
    intervals = pd.read_csv(output_dir / "preconditioning_interval_diagnostics.csv")

    assert {
        "case_name",
        "diagnostic_type",
        "alpha",
        "degree",
        "sigma_min_before",
        "sigma_max_before",
        "kappa_before",
        "sigma_min_after",
        "sigma_max_after",
        "kappa_after",
        "full_interval_error_before",
        "full_interval_error_after",
        "actual_singular_error_before",
        "actual_singular_error_after",
        "selected_interval",
        "interval_caveat",
        "resource_caveat",
        "status",
    }.issubset(summary.columns)
    assert "diagnostic only" in " ".join(summary["interval_caveat"].astype(str)).lower()
    assert "diagnostic only" in intervals["interval_caveat"].iloc[0].lower()
    assert (output_dir / "spectrum_aware_report.md").is_file()
