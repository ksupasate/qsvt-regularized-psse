from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper.final_statistical_aggregation import (
    MANIFEST_COLUMNS,
    OUTPUT_FILES,
    STAT_COLUMNS,
    build_final_statistical_aggregation,
)


def _write_fixture(input_root: Path, package_root: Path) -> None:
    ablation_dir = input_root / "measurement_type_ablation"
    ablation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case": "ieee14",
                "workflow": "ac_linearized",
                "measurement_subset": "full_ac_measurement_set",
                "estimator": "ridge_tikhonov",
                "alpha": 1e-4,
                "rmse": 1.0,
                "weighted_residual_norm": 10.0,
                "condition_number": 100.0,
                "seed": 0,
                "result_status": "computed",
            },
            {
                "case": "ieee14",
                "workflow": "ac_linearized",
                "measurement_subset": "full_ac_measurement_set",
                "estimator": "ridge_tikhonov",
                "alpha": 1e-4,
                "rmse": 3.0,
                "weighted_residual_norm": 14.0,
                "condition_number": 100.0,
                "seed": 1,
                "result_status": "computed",
            },
            {
                "case": "ieee14",
                "workflow": "ac_linearized",
                "measurement_subset": "voltage_only",
                "estimator": "pseudoinverse",
                "alpha": np.nan,
                "rmse": 9.0,
                "weighted_residual_norm": 30.0,
                "condition_number": 1.0,
                "seed": 0,
                "result_status": "runtime_limited",
            },
            {
                "case": "ieee14",
                "workflow": "ac_linearized",
                "measurement_subset": "full_ac_measurement_set",
                "estimator": "qsvt_target_classical",
                "alpha": 1e-4,
                "rmse": 1.0,
                "weighted_residual_norm": 10.0,
                "condition_number": 100.0,
                "seed": 0,
                "result_status": "computed",
            },
        ]
    ).to_csv(ablation_dir / "measurement_type_ablation_all.csv", index=False)

    nonlinear_dir = package_root / "phase4_nonlinear_ac"
    nonlinear_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case": "ieee14",
                "estimator": "ridge_tikhonov",
                "alpha": 1e-4,
                "noise_level": 0.0,
                "missing_ratio": 0.0,
                "bad_data_ratio": 0.0,
                "seed": 0,
                "converged": "yes",
                "iteration_count": 3,
                "final_rmse": 0.1,
                "final_weighted_residual_norm": 1.0,
                "result_status": "completed",
            },
            {
                "case": "ieee14",
                "estimator": "ridge_tikhonov",
                "alpha": 1e-4,
                "noise_level": 0.0,
                "missing_ratio": 0.0,
                "bad_data_ratio": 0.0,
                "seed": 1,
                "converged": "partial",
                "iteration_count": 8,
                "final_rmse": 0.3,
                "final_weighted_residual_norm": 3.0,
                "result_status": "completed",
            },
        ]
    ).to_csv(nonlinear_dir / "paper_table_nonlinear_ac_convergence.csv", index=False)


def _run(tmp_path: Path) -> dict:
    input_root = tmp_path / "outputs"
    package_root = tmp_path / "package"
    _write_fixture(input_root, package_root)
    return build_final_statistical_aggregation(
        {
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(package_root / "statistical_summary"),
        }
    )


def test_aggregation_statistics_are_correct_on_synthetic_data(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["estimator_seed_variability"])
    ridge = summary[
        (summary["estimator"] == "ridge_tikhonov")
        & (summary["measurement_subset"] == "full_ac_measurement_set")
        & (summary["metric"] == "rmse")
    ].iloc[0]
    assert ridge["n_rows"] == 2
    assert ridge["n_seeds"] == 2
    assert ridge["mean"] == 2.0
    assert ridge["median"] == 2.0
    assert np.isclose(ridge["std"], np.sqrt(2.0))


def test_missing_source_artifacts_are_recorded_in_manifest(tmp_path: Path) -> None:
    run = _run(tmp_path)
    manifest = pd.read_csv(run["artifacts"]["statistical_aggregation_manifest"])
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert "missing_source_artifact" in set(manifest["status"])


def test_single_seed_standard_deviation_is_handled_honestly(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["estimator_seed_variability"])
    qsvt = summary[
        (summary["estimator"] == "qsvt_target_classical") & (summary["metric"] == "rmse")
    ].iloc[0]
    assert qsvt["n_seeds"] == 1
    assert pd.isna(qsvt["std"])
    assert "single seed" in str(qsvt["notes"])


def test_runtime_limited_rows_are_counted(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["measurement_ablation"])
    row = summary[
        (summary["measurement_subset"] == "voltage_only")
        & (summary["estimator"] == "pseudoinverse")
        & (summary["metric"] == "rmse")
    ].iloc[0]
    assert row["runtime_limited_count"] == 1


def test_qsvt_target_is_not_reported_as_outperforming_ridge(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = pd.read_csv(run["artifacts"]["estimator_seed_variability"])
    qsvt = summary[summary["estimator"] == "qsvt_target_classical"]
    assert not qsvt.empty
    assert (qsvt["qsvt_outperforms_ridge"] == False).all()  # noqa: E712
    assert qsvt["notes"].astype(str).str.contains("no superiority claim").all()


def test_summary_files_are_generated_with_required_columns(tmp_path: Path) -> None:
    run = _run(tmp_path)
    for key in OUTPUT_FILES:
        path = Path(run["artifacts"][key])
        assert path.is_file()
        assert list(pd.read_csv(path).columns) == STAT_COLUMNS
    summary = Path(run["artifacts"]["statistical_aggregation_summary"])
    assert summary.is_file()
