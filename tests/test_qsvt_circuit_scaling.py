from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.circuit_scaling import run_circuit_scaling


def test_circuit_scaling_records_completed_and_infeasible_rows(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")

    output_dir = tmp_path / "scaling"
    run_circuit_scaling(
        {
            "scaling": {
                "run_id": "test_qsvt_circuit_scaling",
                "output_dir": str(output_dir),
                "cases": ["ieee14"],
                "sizes": [2, 4],
                "max_simulated_size": 2,
                "alpha": 0.05,
                "polynomial_degree": 5,
                "grid_size": 128,
                "phase_cache_dir": str(tmp_path / "phase_cache"),
            }
        }
    )

    for filename in [
        "config_resolved.yaml",
        "circuit_scaling_results.csv",
        "circuit_scaling_summary.json",
        "circuit_scaling_plot_depth.png",
        "circuit_scaling_plot_cx.png",
        "circuit_scaling_plot_error.png",
        "run.log",
    ]:
        assert (output_dir / filename).is_file()

    results = pd.read_csv(output_dir / "circuit_scaling_results.csv")
    assert set(results["status"]) == {"completed", "infeasible"}
    completed = results.loc[results["status"] == "completed"].iloc[0]
    assert completed["depth_after_transpile"] > 0
    assert completed["cx_count"] >= 0
    infeasible = results.loc[results["status"] == "infeasible"].iloc[0]
    assert "max_simulated_size" in str(infeasible["failure_reason"])
