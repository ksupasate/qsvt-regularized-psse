"""Public checks for the selected-observable classical timing baseline."""

from __future__ import annotations

import pandas as pd


def test_classical_baseline_has_repeated_ieee300_timings() -> None:
    frame = pd.read_csv("outputs/classical_selected_observable_baseline/baseline_summary.csv")
    assert {
        "runtime_q1_seconds",
        "runtime_q3_seconds",
        "preprocessing_median_seconds",
        "query_median_seconds",
        "timing_repeats",
        "solver_type",
    }.issubset(frame.columns)
    ieee300 = frame[frame["case"] == "ieee300"]
    assert {"sparse_factorized", "adjoint_functional"} <= set(ieee300["method"])
    assert (ieee300["timing_repeats"] == 30).all()
