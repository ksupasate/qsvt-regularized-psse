from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.nonbruteforce_refinement import diagnose_ieee300_spectral_difficulty


def test_spectral_difficulty_separates_error_types(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = diagnose_ieee300_spectral_difficulty(
        {
            "output_dir": str(tmp_path / "spectral"),
            "cases": ["synthetic", {"case_name": "not_a_case", "case_source": "pypower"}],
            "degree": 15,
            "grid_size": 80,
            "histogram_bins": 5,
            "fallback_to_synthetic": False,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "spectral_difficulty_summary.csv")
    intervals = pd.read_csv(output_dir / "interval_restriction_diagnostics.csv")
    report = (output_dir / "ieee300_spectral_difficulty_report.md").read_text()

    assert {
        "full_interval_max_error",
        "actual_singular_values_max_error",
        "central_99_interval_max_error",
        "central_95_interval_max_error",
        "error_peak_sigma",
        "nearest_actual_singular_value",
        "distance_to_nearest_singular_value",
        "diagnostic_interpretation",
    }.issubset(summary.columns)
    assert {"ok", "failed"}.issubset(set(summary["status"]))
    assert "Full-interval error and actual-singular-value error are reported separately" in report
    assert "diagnostic only" in intervals["caveat"].iloc[0]
    assert (output_dir / "singular_value_quantiles.csv").is_file()
    assert (output_dir / "singular_value_histograms.csv").is_file()
    assert (output_dir / "error_location_diagnostics.csv").is_file()
