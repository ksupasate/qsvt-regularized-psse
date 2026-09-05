from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.adaptive_degree_selection import run_adaptive_degree_selection


def test_adaptive_degree_selection_passes_or_fails_without_crashing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_adaptive_degree_selection(
        {
            "output_dir": str(tmp_path / "adaptive"),
            "matrix_source": "synthetic",
            "alpha": [1.0e-2],
            "target_tolerance": [1.0e-2, 1.0e-6],
            "search_degrees": [15, 35],
            "max_degree": 35,
            "grid_size": 500,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "adaptive_degree_summary.csv")
    trace = pd.read_csv(output_dir / "adaptive_search_trace.csv")

    assert set(frame["status"]).issubset(
        {"passed", "failed_max_degree", "failed_numerical_instability"}
    )
    assert len(frame) == 2
    assert len(trace) >= 2
    assert (frame["selected_query_count"] >= 1).all()
    assert (output_dir / "adaptive_degree_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
