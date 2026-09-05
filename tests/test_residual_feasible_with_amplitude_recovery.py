from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.residual_feasible_with_amplitude_recovery import (
    RESULT_COLUMNS,
    evaluate_amplitude_recovery_search,
    write_amplitude_recovery_outputs,
)


def _subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64),
        r_tilde=np.array([0.4, -0.2], dtype=np.float64),
        metadata={"case": "toy", "candidate_id": "toy_2x2"},
    )


def _rows() -> list[dict]:
    return evaluate_amplitude_recovery_search(
        subproblem=_subproblem(),
        alphas=[1.0e-3],
        degrees=[5],
        target_designs=["current_global", "support_scaled"],
        methods=[
            "exact_success_amplitude_diagnostic",
            "bernoulli_success_amplitude",
            "iterative_qae_if_available",
            "hybrid_known_C_amplitude",
            "best_scalar_upper_bound",
        ],
        shot_levels=[512],
        grover_powers=(0, 1, 2),
        seed=7,
    )


def test_diagnostic_methods_never_marked_deployable_feasible() -> None:
    rows = _rows()
    for row in rows:
        if row["norm_recovery_method"] in {
            "exact_success_amplitude_diagnostic",
            "best_scalar_upper_bound",
        }:
            assert row["residual_feasible"] is False


def test_feasible_requires_deployable_method_and_thresholds() -> None:
    rows = _rows()
    for row in rows:
        if row["residual_feasible"]:
            assert row["norm_recovery_method"] in {
                "bernoulli_success_amplitude",
                "iterative_qae",
                "hybrid_known_C_amplitude",
            }
            assert float(row["residual_ratio_vs_no_update"]) <= 0.1 + 1.0e-9
            assert float(row["direction_error_vs_ridge"]) <= 0.1 + 1.0e-9


def test_alias_normalized_and_columns_written(tmp_path: Path) -> None:
    rows = _rows()
    methods = {row["norm_recovery_method"] for row in rows}
    assert "iterative_qae" in methods
    assert "iterative_qae_if_available" not in methods

    artifacts = write_amplitude_recovery_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)
    for key in [
        "all_amplitude_recovery_configs",
        "residual_feasible_configs",
        "diagnostic_feasible_configs",
        "rejected_configs",
    ]:
        assert artifacts[key].is_file()
    frame = pd.read_csv(artifacts["all_amplitude_recovery_configs"])
    for column in RESULT_COLUMNS:
        assert column in frame.columns


def test_diagnostic_feasible_column_separates_diagnostics() -> None:
    rows = _rows()
    # diagnostic_feasible can only be True for diagnostic methods.
    for row in rows:
        if row["diagnostic_feasible"]:
            assert row["norm_recovery_method"] in {
                "exact_success_amplitude_diagnostic",
                "best_scalar_upper_bound",
                "observable_calibrated",
            }
            assert row["residual_feasible"] is False
