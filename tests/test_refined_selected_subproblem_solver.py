from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.refined_selected_subproblem_solver import (
    run_refined_selected_subproblem_solver,
)


def test_refined_solver_reports_best_configuration_per_subproblem(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selection = pd.DataFrame(
        [
            {
                "candidate_id": "toy_0",
                "case": "toy",
                "model": "ac_linearized",
                "case_source": "synthetic",
                "selection_source": "unit_test",
                "row_indices": "0 1",
                "col_indices": "0 1",
                "selected": True,
            }
        ]
    )
    selection_path = tmp_path / "selected.csv"
    selection.to_csv(selection_path, index=False)

    system = WeightedSystem(
        H_tilde=np.eye(2),
        r_tilde=np.array([1.0, 0.0]),
        metadata={"measurement_labels": ["m0", "m1"], "state_labels": ["x0", "x1"]},
    )
    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.refined_selected_subproblem_solver._build_system",
        lambda **kwargs: (system, "synthetic"),
    )

    def fake_grid(**kwargs):
        return [
            {
                "case": "toy",
                "model": "linear",
                "subproblem_id": "toy_0",
                "selection_mode": "unit_test",
                "matrix_shape": "2x2",
                "condition_number": 1.0,
                "alpha": 1.0e-4,
                "requested_degree": 3,
                "synthesized_degree": 3,
                "phase_count": 4,
                "query_count": 3,
                "qsvt_query_count": 3,
                "polynomial_target_error_if_available": 0.1,
                "state_error_vs_ridge": 0.1,
                "phase_or_sign_aligned_state_error": 0.1,
                "relative_update_error_raw": 0.1,
                "relative_update_error_best_scalar": 0.1,
                "residual_no_update": 1.0,
                "residual_qsvt_raw": 0.3,
                "residual_qsvt_best_scalar": 0.05,
                "residual_ridge": 1.0e-9,
                "residual_ratio_raw_vs_no_update": 0.3,
                "residual_ratio_best_scalar_vs_no_update": 0.05,
                "success_probability": 0.5,
                "postselection_cost_proxy": 2.0,
                "amplitude_amplification_cost_proxy": 2.0**0.5,
                "run_status": "completed",
                "failure_reason_if_any": "",
            }
        ]

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.refined_selected_subproblem_solver.evaluate_alpha_degree_grid",
        fake_grid,
    )
    run = run_refined_selected_subproblem_solver(
        {"selection_file": str(selection_path), "output_dir": str(tmp_path / "out")}
    )

    best = pd.read_csv(run["artifacts"]["best_configuration_per_subproblem"])
    assert len(best) == 1
    assert best.loc[0, "subproblem_classification"] == "strong_residual_reducing"
