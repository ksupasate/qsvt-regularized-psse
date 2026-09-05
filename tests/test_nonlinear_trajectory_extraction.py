from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.nonlinear_trajectory_extraction import (
    build_nonlinear_trajectory_extraction,
)


def _make_trace(input_root: Path) -> None:
    case_dir = input_root / "nonlinear_ac_ieee14_seed10"
    case_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "estimator": ["ridge", "ridge", "qsvt_regularized", "qsvt_regularized"],
            "iteration": [0, 1, 0, 1],
            "weighted_residual_before": [22.0, 5.0, 22.0, 5.0],
            "weighted_residual_after": [5.0, 4.9, 5.0, 4.9],
            "update_norm": [0.02, 0.001, 0.02, 0.001],
            "failed": [False, False, False, False],
            "converged": [False, True, False, True],
            "trial_id": ["noise_0", "noise_0", "noise_0", "noise_0"],
            "sweep_name": ["nonlinear_noise_sweep"] * 4,
            "sweep_parameter": ["scenario.noise_std"] * 4,
            "sweep_value": [0.0, 0.0, 0.0, 0.0],
            "seed": [101, 101, 101, 101],
        }
    ).to_csv(case_dir / "iteration_trace.csv", index=False)


def test_residual_and_update_extracted_as_real(tmp_path: Path) -> None:
    _make_trace(tmp_path)
    run = build_nonlinear_trajectory_extraction(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "traj")}
    )
    residual = pd.read_csv(run["artifacts"]["figure_data_nonlinear_residual_vs_iteration"])
    assert not residual.empty
    assert (residual["result_status"] == "extracted").all()
    # Estimator names mapped to paper-facing names.
    assert "qsvt_target_classical" in set(residual["estimator"])
    update = pd.read_csv(run["artifacts"]["figure_data_nonlinear_update_norm_vs_iteration"])
    assert not update.empty


def test_unavailable_logs_recorded_not_fabricated(tmp_path: Path) -> None:
    # No trace files at all -> every case is unavailable_logs.
    run = build_nonlinear_trajectory_extraction(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "traj")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_nonlinear_trajectory_logs"])
    assert (missing["result_status"] == "unavailable_logs").any()
    residual = pd.read_csv(run["artifacts"]["figure_data_nonlinear_residual_vs_iteration"])
    assert residual.empty


def test_rmse_and_kappa_not_recorded(tmp_path: Path) -> None:
    _make_trace(tmp_path)
    run = build_nonlinear_trajectory_extraction(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "traj")}
    )
    # RMSE / kappa per-iteration are header-only (no fabricated trajectories).
    rmse = pd.read_csv(run["artifacts"]["figure_data_nonlinear_rmse_vs_iteration"])
    assert rmse.empty
    missing = pd.read_csv(run["artifacts"]["missing_nonlinear_trajectory_logs"])
    assert (missing["result_status"] == "not_recorded_by_current_solver").any()
