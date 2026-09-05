from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.adaptive_multicase_degree_search import (
    run_adaptive_multicase_degree_search,
)


def test_adaptive_multicase_degree_search_writes_trace_and_failures(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_adaptive_multicase_degree_search(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "cases": ["synthetic", {"case_name": "not_a_case", "case_source": "pypower"}],
            "alpha": [1.0e-2],
            "degree_schedule": [5, 11],
            "max_degree": 11,
            "grid_size": 40,
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "adaptive_multicase_summary.csv")
    trace = pd.read_csv(output_dir / "adaptive_multicase_search_trace.csv")
    failures = pd.read_csv(output_dir / "adaptive_multicase_failure_log.csv")

    assert {"synthetic", "not_a_case"}.issubset(set(summary["case_name"]))
    assert "failed_matrix_construction" in set(summary["status"])
    assert {"case_name", "degree", "query_count", "max_pointwise_error", "status"}.issubset(
        trace.columns
    )
    assert len(failures) >= 1
    assert (output_dir / "adaptive_multicase_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
