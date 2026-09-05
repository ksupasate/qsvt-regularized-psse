from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.alpha_degree_refinement import (
    completed_refinement_row,
    evaluate_single_alpha_degree,
    write_refinement_outputs,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem


def test_alpha_degree_refinement_handles_failed_high_degree_rows(monkeypatch) -> None:
    def fail_solver(**kwargs):
        if int(kwargs["degree"]) > 3:
            raise RuntimeError("synthetic high-degree failure")
        return SimpleNamespace()

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.alpha_degree_refinement.solve_gate_level_state_estimation_problem",
        fail_solver,
    )
    subproblem = SelectedSubproblem(
        H_tilde=np.eye(2),
        r_tilde=np.array([1.0, 0.0]),
        metadata={"case": "toy"},
    )

    row = evaluate_single_alpha_degree(
        subproblem=subproblem,
        alpha=1.0e-4,
        degree=5,
        shots=10,
        seed=1,
        case="toy",
        model="linear",
        subproblem_id="toy_0",
        selection_mode="unit_test",
    )

    assert row["run_status"] == "failed"
    assert "synthetic high-degree failure" in row["failure_reason_if_any"]


def test_alpha_degree_refinement_writes_required_outputs(tmp_path: Path) -> None:
    computation = SimpleNamespace(
        H_tilde=np.eye(2),
        r_tilde=np.array([1.0, 0.0]),
        qsvt_update=np.array([0.9, 0.0]),
        ridge_update=np.array([1.0, 0.0]),
        summary={
            "success_probability": 0.25,
            "qsvt_query_count": 5,
            "synthesized_degree": 3,
            "phase_count": 4,
            "state_error_vs_ridge": 0.0,
            "phase_or_sign_aligned_state_error": 0.0,
        },
    )
    row = completed_refinement_row(
        computation=computation,
        alpha=1.0e-4,
        degree=3,
        case="toy",
        model="linear",
        subproblem_id="toy_0",
        selection_mode="unit_test",
    )

    artifacts = write_refinement_outputs(tmp_path, {"output_dir": str(tmp_path)}, [row])
    summary = pd.read_csv(artifacts["alpha_degree_refinement_summary"])

    assert summary.loc[0, "run_status"] == "completed"
    assert artifacts["best_residual_by_alpha"].is_file()
    assert artifacts["refinement_failures"].is_file()
