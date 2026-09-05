from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.nonlinear_ac_alpha_stress import (
    ALL_COLUMNS,
    build_nonlinear_ac_alpha_stress,
)


def _run(tmp_path: Path) -> dict:
    return build_nonlinear_ac_alpha_stress(
        {
            "cases": ["ieee14"],
            "stress_types": ["clean_or_noise", "bad_data_5_percent", "weak_area_stress"],
            "estimators": ["ridge_tikhonov", "huber_irls", "qsvt_target_classical"],
            "alphas": [1.0e-4, 1.0e-2],
            "seeds": [0],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "nonlinear"),
        }
    )


def test_raw_perturbation_and_iteration_fields(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["nonlinear_alpha_sweep"])
    assert list(frame.columns) == ALL_COLUMNS
    # Nonlinear raw perturbation z = h(x_true) + e + b is recorded in the provenance.
    assert frame["source_artifact"].astype(str).str.contains(r"raw_z=h\(x\)\+e\+b").all()
    # Iteration / convergence fields are present and sane.
    assert frame["iteration_count"].notna().all()
    assert (frame["iteration_count"] >= 0).all()
    assert (frame["iteration_count"] <= frame["max_iterations"]).all()
    assert frame["notes"].astype(str).str.contains("Jacobian rebuilt").all()


def test_weak_area_stress_recorded(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["nonlinear_alpha_sweep"])
    weak = frame[frame["stress_type"] == "weak_area_stress"]
    assert not weak.empty
    assert (weak["weak_area_multiplier"] > 0).all()
    weak_file = pd.read_csv(run["artifacts"]["nonlinear_weak_area_stress"])
    assert not weak_file.empty
    assert (weak_file["stress_type"] == "weak_area_stress").all()


def test_qsvt_equals_ridge_nonlinear(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run["qsvt_ridge_equivalent"] is True
    frame = pd.read_csv(run["artifacts"]["nonlinear_alpha_sweep"])
    keys = ["stress_type", "alpha", "seed"]
    ridge = frame[frame["estimator"] == "ridge_tikhonov"].set_index(keys)["final_rmse"]
    qsvt = frame[frame["estimator"] == "qsvt_target_classical"].set_index(keys)["final_rmse"]
    common = ridge.index.intersection(qsvt.index)
    assert len(common) > 0
    assert np.allclose(
        pd.to_numeric(ridge.loc[common]),
        pd.to_numeric(qsvt.loc[common]),
        rtol=1e-8,
        atol=1e-12,
    )
