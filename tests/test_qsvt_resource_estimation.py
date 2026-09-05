from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.matrix_resource_estimation import run_resource_estimation


def test_matrix_resource_estimates_cover_requested_cases(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_resource_estimation(
        {
            "resource": {
                "output_dir": str(tmp_path / "resource"),
                "cases": ["ieee14", "ieee30"],
                "degrees": [5, 11],
                "grid_size": 256,
                "target_error": 0.1,
            }
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "resource_estimates.csv")

    assert set(frame["case_name"]) == {"ieee14", "ieee30"}
    assert (output_dir / "resource_estimates_summary.json").is_file()
    assert (frame["estimated_qsvt_query_count"] > 0).all()
    assert {"matrix_shape", "condition_number", "full_statevector_simulation_feasible"}.issubset(
        frame.columns
    )
