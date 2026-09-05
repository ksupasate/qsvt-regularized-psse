from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.polynomial_action_decomposition import (
    evaluate_single_polynomial_action,
    write_polynomial_action_outputs,
)


def _toy_subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64),
        r_tilde=np.array([1.0, 0.25], dtype=np.float64),
        metadata={"case": "toy"},
    )


def test_polynomial_action_decomposition_separates_error_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_synthesis(coefficients, **kwargs):
        return {
            "phases": np.zeros(6),
            "phase_count": 6,
            "cache_key": "poly-cache",
            "cache_hit": False,
            "metadata": {},
        }

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.polynomial_action_decomposition.synthesize_phases_with_cache",
        fake_synthesis,
    )

    row = evaluate_single_polynomial_action(
        subproblem=_toy_subproblem(),
        alpha=1.0e-2,
        requested_degree=5,
        cache_dir=tmp_path,
    )

    assert row["run_status"] == "completed"
    assert row["constructed_polynomial_degree"] == 5
    assert row["synthesized_phase_degree"] == 5
    assert row["ridge_residual"] >= 0.0
    assert row["exact_target_residual"] >= 0.0
    assert row["bounded_target_residual"] >= 0.0
    assert row["polynomial_action_residual"] >= 0.0
    assert "dominant_error_source" in row


def test_polynomial_action_records_failed_phase_synthesis(monkeypatch, tmp_path: Path) -> None:
    def fail_synthesis(coefficients, **kwargs):
        raise RuntimeError("phase backend unavailable")

    monkeypatch.setattr(
        "robust_qsvt_se.qsvt.polynomial_action_decomposition.synthesize_phases_with_cache",
        fail_synthesis,
    )

    row = evaluate_single_polynomial_action(
        subproblem=_toy_subproblem(),
        alpha=1.0e-2,
        requested_degree=5,
        cache_dir=tmp_path,
    )

    assert row["run_status"] == "phase_synthesis_failed"
    assert "phase backend unavailable" in row["failure_reason_if_any"]
    assert np.isfinite(row["polynomial_action_residual"])


def test_polynomial_action_outputs_required_files(tmp_path: Path) -> None:
    row = {
        "case": "toy",
        "model": "linear",
        "subproblem_id": "toy_0",
        "selection_mode": "unit_test",
        "alpha": 1.0e-2,
        "requested_degree": 5,
        "constructed_polynomial_degree": 5,
        "synthesized_phase_degree": 5,
        "effective_qsvt_degree": 5,
        "condition_number": 2.0,
        "sigma_min": 0.5,
        "sigma_max": 1.0,
        "pointwise_max_error_on_grid": 0.01,
        "pointwise_max_error_on_singular_values": 0.001,
        "ridge_residual": 0.1,
        "exact_target_residual": 0.1,
        "bounded_target_residual": 0.5,
        "polynomial_action_residual": 0.11,
        "best_scalar_polynomial_residual": 0.1,
        "gate_level_residual_if_available": np.nan,
        "state_error_target_vs_ridge": 0.0,
        "state_error_poly_vs_target": 0.01,
        "state_error_gate_vs_poly_if_available": np.nan,
        "dominant_error_source": "polynomial_action_matches_target",
        "run_status": "completed",
        "failure_reason_if_any": "",
    }

    artifacts = write_polynomial_action_outputs(tmp_path, {"output_dir": str(tmp_path)}, [row])
    frame = pd.read_csv(artifacts["polynomial_action_residuals"])

    assert frame.loc[0, "ridge_residual"] == 0.1
    assert artifacts["polynomial_pointwise_error"].is_file()
    assert artifacts["gate_vs_polynomial_error"].is_file()
