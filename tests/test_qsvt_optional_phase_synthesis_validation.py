from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.optional_phase_synthesis_validation import (
    run_optional_phase_synthesis_validation,
)


def test_optional_phase_synthesis_skips_when_forced_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_optional_phase_synthesis_validation(
        {
            "output_dir": str(tmp_path / "phase"),
            "matrix_source": "synthetic",
            "alpha": [1.0e-2],
            "degree": 5,
            "grid_size": 64,
            "force_dependency_missing": True,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "phase_synthesis_summary.csv")
    phases = pd.read_csv(output_dir / "phase_angles.csv")

    assert frame.loc[0, "status"] == "skipped_dependency_missing"
    assert int(frame.loc[0, "phase_count"]) == 0
    assert phases.empty
    assert (output_dir / "phase_pointwise_errors.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
