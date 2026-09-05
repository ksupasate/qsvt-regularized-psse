from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.multicase_approximation_diagnostics import (
    build_multicase_approximation_diagnostics,
)


def test_multicase_approximation_diagnostics_logs_failures(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_multicase_approximation_diagnostics(
        {
            "output_dir": str(tmp_path / "multi"),
            "cases": ["synthetic", {"case_name": "not_a_case", "case_source": "pypower"}],
            "degree": 15,
            "grid_size": 500,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "multicase_approximation_summary.csv")
    failures = pd.read_csv(output_dir / "failure_log.csv")

    assert {"ok", "failed"}.issubset(set(frame["status"]))
    assert len(failures) == 1
    ok_row = frame[frame["status"] == "ok"].iloc[0]
    assert int(ok_row["query_count_estimate"]) == 31
    assert (output_dir / "multicase_approximation_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
