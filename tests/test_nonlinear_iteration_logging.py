"""Phase 2 hardening: nonlinear AC per-iteration RMSE / condition-number logging."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.iterative_ac import (
    build_ac_nonlinear_problem,
    run_iterative_estimators,
)
from robust_qsvt_se.paper.nonlinear_logging_refresh import build_nonlinear_logging_refresh
from robust_qsvt_se.paper.nonlinear_trajectory_extraction import (
    build_nonlinear_trajectory_extraction,
)

_LOGGER = logging.getLogger("test_nonlinear_iteration_logging")


def _config(*, log: bool) -> dict:
    iteration = {
        "max_iterations": 5,
        "update_tolerance": 1.0e-7,
        "residual_tolerance": 1.0e-7,
        "damping": 1.0,
        "max_update_norm": 1000.0,
        "residual_growth_limit": 10000.0,
    }
    if log:
        iteration["log_iteration_metrics"] = True
        iteration["log_condition_number"] = True
    return {
        "run_name": "logging_test",
        "seed": 10,
        "system": {
            "case_name": "ieee14",
            "case_source": "pypower",
            "mode": "nonlinear_ac_state_estimation",
            "measurement": {
                "include_voltage_magnitudes": True,
                "include_p_injections": True,
                "include_q_injections": True,
                "include_p_branch_flows": True,
                "include_q_branch_flows": True,
                "voltage_std": 0.01,
                "injection_p_std": 0.03,
                "injection_q_std": 0.03,
                "flow_p_std": 0.02,
                "flow_q_std": 0.02,
            },
            "linearization": {
                "angle_perturbation_std": 0.005,
                "voltage_perturbation_std": 0.005,
                "min_voltage_magnitude": 0.5,
            },
            "iteration": iteration,
        },
        "scenario": {
            "name": "logging_test",
            "noise_std": 0.01,
            "missing_ratio": 0.0,
            "bad_data": {"enabled": False, "ratio": 0.0, "magnitude": 10.0, "target": "random"},
        },
        "estimators": [{"name": "ridge", "alpha": 1.0e-4}],
        "output": {"root": "outputs", "run_id": "logging_test"},
    }


def test_logging_off_preserves_legacy_trace_schema() -> None:
    config = _config(log=False)
    problem = build_ac_nonlinear_problem(config)
    _, trace = run_iterative_estimators(config=config, problem=problem, logger=_LOGGER)
    assert "state_rmse" not in trace.columns
    assert "condition_number" not in trace.columns


def test_logging_on_records_rmse_when_true_state_exists() -> None:
    config = _config(log=True)
    problem = build_ac_nonlinear_problem(config)
    _, trace = run_iterative_estimators(config=config, problem=problem, logger=_LOGGER)
    assert "state_rmse" in trace.columns
    assert trace["state_rmse"].notna().any()
    assert (pd.to_numeric(trace["state_rmse"], errors="coerce").dropna() >= 0).all()


def test_logging_on_records_positive_condition_number() -> None:
    config = _config(log=True)
    problem = build_ac_nonlinear_problem(config)
    _, trace = run_iterative_estimators(config=config, problem=problem, logger=_LOGGER)
    kappa = pd.to_numeric(trace["condition_number"], errors="coerce").dropna()
    assert not kappa.empty
    assert (kappa > 0).all()
    rank = pd.to_numeric(trace["numerical_rank"], errors="coerce").dropna()
    assert (rank > 0).all()


def test_refresh_logs_are_extractable(tmp_path: Path) -> None:
    refresh_dir = tmp_path / "nonlinear_logging_refresh"
    build_nonlinear_logging_refresh(
        {"cases": ["ieee14"], "seeds": [10], "output_dir": str(refresh_dir)}
    )
    run = build_nonlinear_trajectory_extraction(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "traj")}
    )
    rmse = pd.read_csv(run["artifacts"]["figure_data_nonlinear_rmse_vs_iteration"])
    kappa = pd.read_csv(run["artifacts"]["figure_data_nonlinear_kappa_vs_iteration"])
    assert not rmse.empty
    assert not kappa.empty
    assert (rmse["result_status"] == "extracted").all()
    assert np.isfinite(pd.to_numeric(kappa["metric_value"], errors="coerce")).all()


def test_old_logs_not_backfilled(tmp_path: Path) -> None:
    # A legacy-schema log (no state_rmse / condition_number) plus a refresh: the legacy
    # case must remain not_recorded; only the refresh populates rmse/kappa.
    legacy = tmp_path / "nonlinear_ac_ieee30_seed10"
    legacy.mkdir(parents=True)
    pd.DataFrame(
        {
            "estimator": ["ridge", "ridge"],
            "iteration": [0, 1],
            "weighted_residual_before": [10.0, 4.0],
            "weighted_residual_after": [4.0, 3.9],
            "update_norm": [0.05, 0.001],
            "failed": [False, False],
            "converged": [False, True],
            "trial_id": ["t", "t"],
            "sweep_name": ["s", "s"],
            "sweep_parameter": ["p", "p"],
            "sweep_value": [0.0, 0.0],
            "seed": [10, 10],
        }
    ).to_csv(legacy / "iteration_trace.csv", index=False)
    build_nonlinear_logging_refresh(
        {
            "cases": ["ieee14"],
            "seeds": [10],
            "output_dir": str(tmp_path / "nonlinear_logging_refresh"),
        }
    )
    run = build_nonlinear_trajectory_extraction(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "traj")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_nonlinear_trajectory_logs"])
    not_recorded = missing[missing["result_status"] == "not_recorded_by_current_solver"]
    assert "ieee30" in set(not_recorded["case"])
    rmse = pd.read_csv(run["artifacts"]["figure_data_nonlinear_rmse_vs_iteration"])
    # The legacy ieee30 log contributes no fabricated RMSE rows.
    assert "ieee30" not in set(rmse["case"].astype(str))
