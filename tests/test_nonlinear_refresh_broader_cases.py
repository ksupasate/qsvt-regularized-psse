"""Gap-resolution: broadened nonlinear per-iteration logging refresh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.nonlinear_logging_refresh import build_nonlinear_logging_refresh

_ROOT = Path(__file__).resolve().parents[1]


def test_ieee30_refresh_computes_rmse_and_positive_kappa(tmp_path: Path) -> None:
    run = build_nonlinear_logging_refresh(
        {
            "cases": ["ieee30"],
            "estimators": ["ridge_tikhonov", "huber_irls", "qsvt_target_classical"],
            "stress_types": ["clean_or_noise"],
            "seeds": [0],
            "output_dir": str(tmp_path / "refresh"),
        }
    )
    assert "ieee30" in run["computed_cases"]
    trace = run["trace"]
    assert "stress_type" in trace.columns
    rmse = pd.to_numeric(trace["state_rmse"], errors="coerce").dropna()
    kappa = pd.to_numeric(trace["condition_number"], errors="coerce").dropna()
    assert len(rmse) > 0 and np.isfinite(rmse).all()
    assert len(kappa) > 0 and bool((kappa > 0).all())
    # The robust Huber estimator is supported by the refresh.
    assert "huber_irls" in set(trace["estimator"].astype(str))


def test_runtime_budget_records_runtime_limited(tmp_path: Path) -> None:
    # An exhausted (negative) budget forces every requested combination into the
    # runtime-limited record rather than skipping it silently.
    run = build_nonlinear_logging_refresh(
        {
            "cases": ["ieee30"],
            "estimators": ["ridge_tikhonov"],
            "stress_types": ["clean_or_noise"],
            "seeds": [0],
            "max_total_seconds": -1.0,
            "output_dir": str(tmp_path / "refresh"),
        }
    )
    assert run["computed_cases"] == []
    assert len(run["runtime_limited"]) >= 1
    limited = pd.read_csv(run["artifacts"]["runtime_limited_cases"])
    assert (limited["result_status"] == "runtime_limited").all()
    assert "ieee30" in set(limited["case"].astype(str))


def test_qsvt_target_equals_ridge_per_iteration(tmp_path: Path) -> None:
    run = build_nonlinear_logging_refresh(
        {
            "cases": ["ieee14"],
            "estimators": ["ridge_tikhonov", "qsvt_target_classical"],
            "stress_types": ["clean_or_noise"],
            "seeds": [0],
            "output_dir": str(tmp_path / "refresh"),
        }
    )
    trace = run["trace"]
    keys = ["case", "stress_type", "seed", "iteration"]
    ridge = trace[trace["estimator"] == "ridge"].set_index(keys)
    qsvt = trace[trace["estimator"] == "qsvt_regularized"].set_index(keys)
    common = ridge.index.intersection(qsvt.index)
    assert len(common) > 0
    for column in ("state_rmse", "condition_number"):
        assert np.allclose(
            pd.to_numeric(ridge.loc[common, column]),
            pd.to_numeric(qsvt.loc[common, column]),
            rtol=1e-9,
            atol=1e-12,
        )


def test_stress_types_are_distinct(tmp_path: Path) -> None:
    run = build_nonlinear_logging_refresh(
        {
            "cases": ["ieee14"],
            "estimators": ["ridge_tikhonov"],
            "stress_types": ["clean_or_noise", "missing_20_percent"],
            "seeds": [0],
            "output_dir": str(tmp_path / "refresh"),
        }
    )
    stresses = set(run["trace"]["stress_type"].astype(str))
    assert {"clean_or_noise", "missing_20_percent"}.issubset(stresses)


def test_package_trajectories_cover_ieee30_and_ieee118_or_runtime_limited() -> None:
    traj = _ROOT / "outputs" / "final_manuscript_package" / "nonlinear_trajectories"
    rmse_path = traj / "figure_data_nonlinear_rmse_vs_iteration.csv"
    if not rmse_path.is_file():
        return
    rmse = pd.read_csv(rmse_path)
    if rmse.empty:
        return
    cases = set(rmse["case"].astype(str))
    refresh_limited = _ROOT / "outputs" / "nonlinear_logging_refresh" / "runtime_limited_cases.csv"
    limited_cases: set[str] = set()
    if refresh_limited.is_file():
        limited = pd.read_csv(refresh_limited)
        if not limited.empty:
            limited_cases = set(limited["case"].astype(str))
    for case in ("ieee30", "ieee118"):
        assert case in cases or case in limited_cases


def test_legacy_logs_not_backfilled() -> None:
    missing = (
        _ROOT
        / "outputs"
        / "final_manuscript_package"
        / "nonlinear_trajectories"
        / "missing_nonlinear_trajectory_logs.csv"
    )
    if not missing.is_file():
        return
    frame = pd.read_csv(missing)
    if frame.empty:
        return
    # Legacy logs without per-iteration RMSE/kappa stay marked, never fabricated.
    assert "not_recorded_by_current_solver" in set(frame["result_status"].astype(str))
