from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.multicase_resource_diagnostics import (
    build_multicase_resource_diagnostics,
)


def test_multicase_resource_diagnostics_logs_failures_without_crashing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_multicase_resource_diagnostics(
        {
            "output_dir": str(tmp_path / "multicase"),
            "cases": [
                "synthetic",
                {"case_name": "not_a_case", "case_source": "pypower"},
            ],
            "degrees": [5, 11],
            "epsilon": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "multicase_resource_summary.csv")
    failures = pd.read_csv(output_dir / "failure_log.csv")

    assert {"ok", "failed"}.issubset(set(frame["status"]))
    assert len(failures) == 1
    ok_row = frame[frame["status"] == "ok"].iloc[0]
    assert int(ok_row["qsvt_degree_estimate"]) >= 0
    assert int(ok_row["query_count_estimate"]) >= 1
    assert "Full-vector readout" in ok_row["readout_caveat"]
    assert (output_dir / "multicase_resource_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
