from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.best_config_gate_validation import (
    select_gate_validation_configs,
    write_gate_validation_outputs,
)


def test_best_config_gate_validation_selects_from_polynomial_results() -> None:
    frame = pd.DataFrame(
        [
            {
                "case": "toy",
                "model": "linear",
                "subproblem_id": "slow",
                "alpha": 1.0e-2,
                "requested_degree": 101,
                "effective_qsvt_degree": 101,
                "condition_number": 3.0,
                "polynomial_action_residual": 0.3,
                "best_scalar_polynomial_residual": 0.3,
                "success_probability_proxy": 0.1,
                "run_status": "completed",
            },
            {
                "case": "toy",
                "model": "linear",
                "subproblem_id": "best",
                "alpha": 1.0e-3,
                "requested_degree": 35,
                "effective_qsvt_degree": 35,
                "condition_number": 2.0,
                "polynomial_action_residual": 0.1,
                "best_scalar_polynomial_residual": 0.05,
                "success_probability_proxy": 0.2,
                "run_status": "completed",
            },
            {
                "case": "toy",
                "model": "linear",
                "subproblem_id": "failed",
                "alpha": 1.0e-4,
                "requested_degree": 51,
                "effective_qsvt_degree": np.nan,
                "condition_number": 2.0,
                "polynomial_action_residual": 0.01,
                "best_scalar_polynomial_residual": 0.01,
                "success_probability_proxy": 0.2,
                "run_status": "failed",
            },
        ]
    )

    selected = select_gate_validation_configs(frame, max_configs=1)

    assert len(selected) == 1
    assert selected.loc[0, "subproblem_id"] == "best"


def test_best_config_gate_validation_writes_required_outputs(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        [
            {
                "case": "toy",
                "model": "linear",
                "subproblem_id": "toy_0",
                "alpha": 1.0e-2,
                "requested_degree": 5,
                "best_scalar_polynomial_residual": 0.1,
                "run_status": "completed",
            }
        ]
    )
    rows = [
        {
            "subproblem_id": "toy_0",
            "alpha": 1.0e-2,
            "degree": 5,
            "synthesized_phase_degree": 5,
            "phase_count": 6,
            "query_count": 5,
            "polynomial_action_residual": 0.1,
            "gate_level_residual": 0.12,
            "best_scalar_gate_residual": 0.11,
            "ridge_residual": 0.1,
            "state_error_gate_vs_poly": 0.02,
            "state_error_gate_vs_ridge": 0.03,
            "success_probability": 0.25,
            "circuit_depth": 10,
            "two_qubit_gates": 2,
            "run_status": "completed",
            "failure_reason_if_any": "",
            "classification": "poly_matching_but_ridge_gap",
        }
    ]

    artifacts = write_gate_validation_outputs(
        tmp_path,
        {"output_dir": str(tmp_path)},
        selected,
        rows,
    )
    result = pd.read_csv(artifacts["gate_validation_results"])

    assert result.loc[0, "degree"] == 5
    assert artifacts["selected_gate_validation_configs"].is_file()
    assert artifacts["gate_vs_poly_vs_ridge_summary"].is_file()
