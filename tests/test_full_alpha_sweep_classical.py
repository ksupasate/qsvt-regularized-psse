from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.full_alpha_sweep_classical import (
    ALL_COLUMNS,
    build_full_alpha_sweep_classical,
)

_ALPHAS = [1.0e-4, 1.0e-2, 1.0]


def _run(tmp_path: Path) -> dict:
    return build_full_alpha_sweep_classical(
        {
            "cases": ["ieee14"],
            "stress_types": ["clean_or_noise", "high_condition"],
            "estimators": ["pseudoinverse", "ridge_tikhonov", "qsvt_target_classical"],
            "alphas": _ALPHAS,
            "seeds": [0],
            "fixed_alpha": 1.0e-4,
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "alpha_sweep"),
        }
    )


def test_only_traceable_alpha_values(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["alpha_sweep_all_results"])
    assert list(frame.columns) == ALL_COLUMNS
    alpha_rows = frame[frame["estimator"].isin(["ridge_tikhonov", "qsvt_target_classical"])]
    alphas = set(np.round(pd.to_numeric(alpha_rows["alpha"]), 12))
    # Every alpha is from the provided grid; none invented or blank.
    assert alphas <= set(np.round(_ALPHAS, 12))
    assert pd.to_numeric(alpha_rows["alpha"]).notna().all()
    # Non-alpha estimators are explicitly marked not_applicable.
    pinv = frame[frame["estimator"] == "pseudoinverse"]
    assert (pinv["alpha_role"] == "not_applicable").all()


def test_best_alpha_is_diagnostic_only(tmp_path: Path) -> None:
    run = _run(tmp_path)
    best = pd.read_csv(run["artifacts"]["alpha_sweep_best_alpha_diagnostic"])
    assert not best.empty
    assert (best["alpha_role"] == "best_alpha_diagnostic_only").all()
    # The reported operating point is flagged as fixed in the main results table.
    frame = pd.read_csv(run["artifacts"]["alpha_sweep_all_results"])
    fixed = frame[frame["alpha_role"] == "fixed_reported_alpha"]
    assert not fixed.empty
    assert np.allclose(pd.to_numeric(fixed["alpha"]), 1.0e-4)


def test_qsvt_equals_ridge_for_matched_alpha(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run["qsvt_ridge_equivalent"] is True
    frame = pd.read_csv(run["artifacts"]["alpha_sweep_all_results"])
    keys = ["case", "stress_type", "alpha", "seed"]
    ridge = frame[frame["estimator"] == "ridge_tikhonov"].set_index(keys)["rmse"]
    qsvt = frame[frame["estimator"] == "qsvt_target_classical"].set_index(keys)["rmse"]
    common = ridge.index.intersection(qsvt.index)
    assert len(common) > 0
    assert np.allclose(
        pd.to_numeric(ridge.loc[common]),
        pd.to_numeric(qsvt.loc[common]),
        rtol=1e-9,
        atol=1e-12,
    )
