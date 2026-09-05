from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.approximation_tradeoff_report import (
    build_approximation_tradeoff_report,
)


def test_approximation_tradeoff_report_is_generated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_approximation_tradeoff_report(
        {
            "output_dir": str(tmp_path / "tradeoff"),
            "matrix_source": "synthetic",
            "alpha": [1.0e-2],
            "degree": [15, 35],
            "grid_size": 500,
            "tolerances": [1.0e-2, 1.0e-3],
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "tradeoff_summary.csv")
    report = (output_dir / "tradeoff_report.md").read_text(encoding="utf-8")

    assert set(frame["target_tolerance"]) == {1.0e-2, 1.0e-3}
    assert "degree-error-query trade-off" in report
    assert (output_dir / "error_vs_degree.csv").is_file()
    assert (output_dir / "query_vs_degree.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
