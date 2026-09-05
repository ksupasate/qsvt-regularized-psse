from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    run_dense_explicit_solver_limit_study,
)


def test_dense_limit_study_handles_size_failure_gracefully(tmp_path: Path) -> None:
    run = run_dense_explicit_solver_limit_study(
        {
            "output_dir": str(tmp_path),
            "submatrix_sizes": [3],
            "degree": 5,
            "shots": 100,
            "seed": 123,
        }
    )
    summary = pd.read_csv(run["artifacts"]["dense_limit_summary"])

    assert run["artifacts"]["dense_limit_resource_summary"].is_file()
    assert run["artifacts"]["dense_limit_failures"].is_file()
    assert run["artifacts"]["dense_limit_interpretation"].is_file()
    assert summary.loc[0, "run_status"] == "failed"
    assert "failure_reason_if_any" in summary.columns
