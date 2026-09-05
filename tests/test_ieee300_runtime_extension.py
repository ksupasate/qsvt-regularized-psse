from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.ieee300_runtime_extension import build_ieee300_runtime_extension


def test_runtime_limited_recorded_when_over_budget(tmp_path: Path) -> None:
    # Force the budget to zero so the (small) build is recorded as runtime_limited
    # rather than producing alpha-sweep rows. Uses ieee14 to keep the build fast.
    run = build_ieee300_runtime_extension(
        {
            "case": "ieee14",
            "alphas": [1e-4],
            "measurement_subsets": ["full_ac_measurement_set"],
            "estimators": ["ridge_tikhonov"],
            "seeds": [0],
            "max_build_seconds": 0.0,
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "ieee300"),
        }
    )
    limited = pd.read_csv(run["artifacts"]["ieee300_runtime_limited_rows"])
    assert not limited.empty
    assert (limited["result_status"] == "runtime_limited").all()
    sweep = pd.read_csv(run["artifacts"]["ieee300_alpha_sweep_reduced"])
    assert sweep.empty


def test_unknown_subset_recorded_not_applicable(tmp_path: Path) -> None:
    run = build_ieee300_runtime_extension(
        {
            "case": "ieee14",
            "measurement_subsets": ["not_a_real_subset"],
            "estimators": ["ridge_tikhonov"],
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "ieee300"),
        }
    )
    limited = pd.read_csv(run["artifacts"]["ieee300_runtime_limited_rows"])
    assert (limited["result_status"] == "not_applicable").any()


def test_small_case_runs_and_qsvt_equals_ridge(tmp_path: Path) -> None:
    # A real (generous-budget) reduced run on a small case verifies the compute path
    # and the QSVT-target == Ridge equivalence.
    run = build_ieee300_runtime_extension(
        {
            "case": "ieee14",
            "alphas": [1e-4, 1e-3],
            "measurement_subsets": ["full_ac_measurement_set", "drop_branch_flow_rows"],
            "estimators": ["ridge_tikhonov", "qsvt_target_classical"],
            "seeds": [0],
            "max_build_seconds": 600.0,
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "ieee300"),
        }
    )
    assert run["qsvt_ridge_equivalent"] is True
    ablation = pd.read_csv(run["artifacts"]["ieee300_measurement_ablation_reduced"])
    assert not ablation.empty
