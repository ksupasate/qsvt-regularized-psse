from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.selected_alpha_validation import (
    run_selected_alpha_phase_validation,
)


def test_selected_alpha_phase_validation_outputs_required_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    alpha_grid = [1.0e-4, 1.0e-2, 1.0]
    run = run_selected_alpha_phase_validation(
        {
            "output_dir": str(tmp_path / "phase"),
            "matrix_source": "synthetic",
            "alpha": alpha_grid,
            "degrees": [5, 11],
            "grid_size": 2048,
            "tolerance": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "phase_validation_summary.csv")
    values = pd.read_csv(output_dir / "target_and_approx_values.csv")

    assert set(frame["alpha"]) == set(alpha_grid)
    assert len(frame) == len(alpha_grid)
    assert (frame["max_bounded_filter_value"] <= 1.0 + 1.0e-12).all()
    assert np.isfinite(frame["max_pointwise_target_error"]).all()
    assert {"bounded_scaling_C", "polynomial_degree", "query_count", "passed"}.issubset(
        frame.columns
    )
    assert {"grid", "actual_singular_value"}.issubset(set(values["evaluation_kind"]))
    assert (output_dir / "pointwise_errors.csv").is_file()
    assert (output_dir / "phase_validation_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
