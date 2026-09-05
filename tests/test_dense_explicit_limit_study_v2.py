from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.dense_explicit_limit_study_v2 import (
    run_dense_explicit_limit_study_v2,
)


def test_dense_limit_study_v2_separates_raw_and_transpiled_resources(tmp_path: Path) -> None:
    run = run_dense_explicit_limit_study_v2(
        {
            "output_dir": str(tmp_path),
            "submatrix_sizes": [4],
            "degree": 9,
            "shots": 100,
            "seed": 123,
            "transpile_qubit_limit": 3,
        }
    )
    construction = pd.read_csv(run["artifacts"]["construction_resource_results"])

    assert run["artifacts"]["executed_solver_results"].is_file()
    assert "raw_depth" in construction.columns
    assert "solver_validated" in construction.columns
    assert run["artifacts"]["transpilation_feasibility"].is_file()
