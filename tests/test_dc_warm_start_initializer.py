"""Phase 3 hardening: clean DC warm-start initializer for nonlinear AC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.initializer_ablation import (
    _dc_warm_start_state,
    _safe_problem,
    build_initializer_ablation,
)


def test_dc_warm_start_does_not_use_true_state() -> None:
    problem = _safe_problem("ieee14", "pypower", 0)
    assert problem is not None
    x0 = _dc_warm_start_state(problem, "pypower")
    true = np.asarray(problem.true_state, dtype=np.float64)
    # Warm start is derived from measurements, not the true state.
    assert not np.allclose(x0, true)


def test_dc_warm_start_state_shape_and_flat_voltages() -> None:
    problem = _safe_problem("ieee14", "pypower", 0)
    assert problem is not None
    x0 = _dc_warm_start_state(problem, "pypower")
    angle_count = len(problem.case.angle_state_buses)
    voltage_count = len(problem.case.voltage_state_buses)
    assert x0.shape == (angle_count + voltage_count,)
    # Voltage magnitudes start flat at 1.0 p.u.
    assert np.allclose(x0[angle_count:], 1.0)


def test_dc_warm_start_beats_flat_start_initial_rmse() -> None:
    problem = _safe_problem("ieee14", "pypower", 0)
    assert problem is not None
    true = np.asarray(problem.true_state, dtype=np.float64)
    angle_count = len(problem.case.angle_state_buses)
    flat = np.zeros_like(true)
    flat[angle_count:] = 1.0
    dc = _dc_warm_start_state(problem, "pypower")

    def rmse(vector: np.ndarray) -> float:
        return float(np.sqrt(np.mean((vector - true) ** 2)))

    # A real DC warm start is a meaningfully better angle initialization than flat start.
    assert rmse(dc) < rmse(flat)


def test_dc_warm_start_computed_and_converges(tmp_path: Path) -> None:
    run = build_initializer_ablation(
        {
            "cases": ["ieee14"],
            "initializers": ["dc_warm_start"],
            "estimators": ["ridge_tikhonov", "qsvt_target_classical"],
            "alpha": 1e-4,
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "init_ablation"),
        }
    )
    results = pd.read_csv(run["artifacts"]["initializer_ablation_results"])
    dc = results[results["initializer"] == "dc_warm_start"]
    assert not dc.empty
    assert (dc["result_status"] == "computed").all()
    assert dc["converged"].all()
    # QSVT-target remains Ridge-equivalent under the DC warm start.
    assert run["qsvt_ridge_equivalent"] is True


def test_dc_warm_start_run_records_convergence(tmp_path: Path) -> None:
    run = build_initializer_ablation(
        {
            "cases": ["ieee14"],
            "initializers": ["dc_warm_start"],
            "estimators": ["ridge_tikhonov"],
            "alpha": 1e-4,
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "init_ablation"),
        }
    )
    results = pd.read_csv(run["artifacts"]["initializer_ablation_results"])
    dc = results[results["initializer"] == "dc_warm_start"].iloc[0]
    # Convergence / failure status is explicitly recorded.
    assert str(dc["result_status"]) in {"computed", "failed_with_error"}
    assert "converged" in results.columns
