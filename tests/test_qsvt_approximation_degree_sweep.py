from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.approximation_degree_sweep import (
    run_approximation_degree_sweep,
)


def test_degree_sweep_rows_errors_and_query_monotonicity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_approximation_degree_sweep(
        {
            "output_dir": str(tmp_path / "degree"),
            "matrix_source": "synthetic",
            "alpha": [1.0e-4, 1.0e-2, 1.0],
            "degree": [15, 35],
            "grid_size": 500,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "degree_sweep_summary.csv")

    assert set(frame["alpha"]) == {1.0e-4, 1.0e-2, 1.0}
    assert set(frame["degree"]) == {15, 35}
    assert (frame["max_bounded_filter_value"] <= 1.0 + 1.0e-12).all()
    assert np.isfinite(frame["max_pointwise_error"]).all()
    for _, group in frame.groupby("alpha"):
        ordered = group.sort_values("degree")
        assert ordered["query_count_estimate"].is_monotonic_increasing
    assert (output_dir / "degree_sweep_pointwise_errors.csv").is_file()
    assert (output_dir / "degree_sweep_target_and_approx_values.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
