from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.preconditioning_diagnostics import (
    CLAIM_STRENGTH,
    run_preconditioning_diagnostics,
)


def test_preconditioning_diagnostics_are_caveated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_preconditioning_diagnostics(
        {
            "output_dir": str(tmp_path / "preconditioning"),
            "matrix_source": "synthetic",
            "alpha": 1.0e-2,
            "degrees": [2, 4],
            "epsilon": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "preconditioning_summary.csv")
    row = frame.loc[0]

    assert row["preconditioner_type"] == "column_equilibration"
    assert row["claim_strength"] == CLAIM_STRENGTH
    assert "speedup" not in row["claim_strength"]
    assert int(row["query_count_before"]) >= 1
    assert int(row["query_count_after"]) >= 1
    assert float(row["relative_solution_error_vs_unpreconditioned_ridge"]) >= 0.0
    assert (output_dir / "preconditioning_summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()
