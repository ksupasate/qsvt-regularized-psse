from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import robust_qsvt_se.qsvt.phase2_preconditioned_alpha as phase2
from robust_qsvt_se.measurement.linear_system import WeightedSystem


def test_phase2_output_includes_ieee118_ieee300_and_separate_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(phase2, "build_engineering_system", _fake_engineering_system)
    run = phase2.run_phase2_preconditioned_alpha_sweeps(
        {
            "output_dir": str(tmp_path / "phase2"),
            "cases": ["ieee118", "ieee300"],
            "alphas": [1.0e-2],
            "noise_stds": [0.0],
            "missing_ratios": [0.0],
            "bad_data_ratios": [0.0],
            "seeds": [10],
            "qsvt_degree": 5,
            "grid_size": 32,
            "fallback_to_synthetic": True,
        }
    )
    results = pd.read_csv(run["output_dir"] / "phase2_sweep_results.csv")

    assert {"ieee118", "ieee300"}.issubset(set(results["case_name"]))
    assert set(phase2.PHASE2_VARIANTS).issubset(set(results["variant_name"]))
    assert "coordinate_preconditioned_ridge" in set(results["variant_name"])
    assert "transformed_penalty_preconditioned_ridge" in set(results["variant_name"])
    assert not results["variant_name"].str.contains("coordinate.*transformed").any()
    assert (run["output_dir"] / "phase2_sweep_summary.csv").is_file()
    assert (run["output_dir"] / "phase2_failure_log.csv").is_file()


def _fake_engineering_system(config: dict[str, Any]) -> tuple[WeightedSystem, str]:
    case_name = str(config["case_name"])
    scale = 1.0 if case_name == "ieee118" else 1.5
    matrix = np.asarray(
        [
            [3.0 * scale, 0.1, 0.0],
            [0.0, 1.0 * scale, 0.2],
            [0.0, 0.0, 0.04],
            [1.0, -0.1, 0.0],
            [0.0, 0.3, 0.01],
        ],
        dtype=float,
    )
    x_true = np.asarray([0.2, -0.1, 0.05], dtype=float)
    return (
        WeightedSystem(
            matrix,
            matrix @ x_true,
            x_true,
            {"case_name": case_name},
        ),
        f"{case_name}_fake_weighted_jacobian",
    )
