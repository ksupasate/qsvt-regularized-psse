from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.measurement_type_ablation import (
    ALL_COLUMNS,
    build_measurement_type_ablation,
)


def _run(tmp_path: Path) -> dict:
    return build_measurement_type_ablation(
        {
            "cases": ["ieee14"],
            "measurement_subsets": [
                "full_ac_measurement_set",
                "voltage_only",
                "drop_branch_flow_rows",
            ],
            "estimators": ["pseudoinverse", "ridge_tikhonov", "qsvt_target_classical"],
            "alpha": 1.0e-4,
            "seeds": [0],
            "input_root": str(tmp_path / "outputs"),
            "output_dir": str(tmp_path / "measurement_type_ablation"),
        }
    )


def test_subsets_have_different_row_counts(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["measurement_type_ablation_all"])
    assert list(frame.columns) == ALL_COLUMNS
    rows_by_subset = frame.groupby("measurement_subset")["n_rows"].first()
    # Dropping measurement types must change the row count (real ablation, not a relabel).
    assert rows_by_subset["full_ac_measurement_set"] > rows_by_subset["voltage_only"]
    assert rows_by_subset["full_ac_measurement_set"] > rows_by_subset["drop_branch_flow_rows"]
    assert rows_by_subset["voltage_only"] != rows_by_subset["drop_branch_flow_rows"]
    assert frame["source_artifact"].astype(str).str.startswith("computed:").all()


def test_rank_deficient_subset_recorded_explicitly(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["measurement_type_ablation_all"])
    voltage_only = frame[frame["measurement_subset"] == "voltage_only"]
    # voltage_only is underdetermined on ieee14 (14 rows < 27 states): never marked computed.
    assert (voltage_only["result_status"] == "rank_deficient").all()
    assert (voltage_only["n_rows"] < voltage_only["state_dimension"]).all()
    assert run["rank_deficient_rows"] > 0


def test_qsvt_equals_ridge_and_alpha_traceable(tmp_path: Path) -> None:
    run = _run(tmp_path)
    frame = pd.read_csv(run["artifacts"]["measurement_type_ablation_all"])
    full = frame[
        (frame["measurement_subset"] == "full_ac_measurement_set")
        & (frame["result_status"] == "computed")
    ]
    ridge = full[full["estimator"] == "ridge_tikhonov"]["rmse"].to_numpy()
    qsvt = full[full["estimator"] == "qsvt_target_classical"]["rmse"].to_numpy()
    assert ridge.size and qsvt.size
    assert abs(float(ridge[0]) - float(qsvt[0])) < 1e-12
    # Alpha is traceable only for the alpha-parametrized estimators.
    alpha_rows = frame[frame["estimator"].isin(["ridge_tikhonov", "qsvt_target_classical"])]
    assert (pd.to_numeric(alpha_rows["alpha"]) == 1.0e-4).all()
    pinv_alpha = frame[frame["estimator"] == "pseudoinverse"]["alpha"]
    assert pinv_alpha.isna().all() or (pinv_alpha.astype(str).str.strip() == "").all()
