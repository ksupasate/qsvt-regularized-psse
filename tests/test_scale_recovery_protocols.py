from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.scale_recovery_protocols import (
    DEPLOYABILITY_LABELS,
    compute_bounded_qsvt_action,
    evaluate_scale_recovery_point,
    write_scale_recovery_outputs,
)


def _toy_subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64),
        r_tilde=np.array([1.0, 0.25], dtype=np.float64),
        metadata={"case": "toy"},
    )


def _rows_by_protocol(rows: list[dict]) -> dict[str, dict]:
    return {row["scale_protocol"]: row for row in rows}


def test_known_c_rescaling_matches_analytic_constant() -> None:
    subproblem = _toy_subproblem()
    H = subproblem.H_tilde
    r = subproblem.r_tilde
    action = compute_bounded_qsvt_action(H, r, alpha=1.0e-2, degree=5)
    expected = rescale_bounded_target_to_original(
        action.direction, beta=action.beta, C=action.scale_factor_C
    )

    rows = evaluate_scale_recovery_point(subproblem=subproblem, alpha=1.0e-2, degree=5)
    by_protocol = _rows_by_protocol(rows)
    known_c = by_protocol["known_C"]

    recovered_residual = float(np.linalg.norm(H @ expected - r))
    assert np.isclose(known_c["residual"], recovered_residual, rtol=0.0, atol=1.0e-12)
    assert np.isclose(known_c["scale_factor"], action.scale_factor_C / action.beta)
    assert known_c["requires_ridge_reference"] is False
    assert known_c["requires_simulator_metadata"] is False
    assert known_c["requires_amplitude_estimation"] is False
    assert known_c["deployability_class"] == "potentially_deployable_under_oracle_assumptions"
    assert np.isnan(known_c["query_cost_proxy"])


def test_best_scalar_row_is_classical_diagnostic_only() -> None:
    rows = evaluate_scale_recovery_point(subproblem=_toy_subproblem(), alpha=1.0e-2, degree=5)
    by_protocol = _rows_by_protocol(rows)
    best_scalar = by_protocol["best_scalar_diagnostic"]
    ridge_norm = by_protocol["ridge_norm"]

    assert best_scalar["deployability_class"] == "classical_diagnostic_only"
    assert best_scalar["requires_ridge_reference"] is True
    assert ridge_norm["deployability_class"] == "classical_diagnostic_only"
    assert ridge_norm["requires_ridge_reference"] is True


def test_proxy_rows_carry_proxy_flags() -> None:
    rows = evaluate_scale_recovery_point(subproblem=_toy_subproblem(), alpha=1.0e-2, degree=5)
    by_protocol = _rows_by_protocol(rows)

    success = by_protocol["success_amplitude_proxy"]
    assert success["deployability_class"] == "quantum_protocol_proxy"
    assert success["requires_simulator_metadata"] is True
    assert success["requires_amplitude_estimation"] is False

    amplitude = by_protocol["amplitude_estimation_proxy"]
    assert amplitude["deployability_class"] == "quantum_protocol_proxy"
    assert amplitude["requires_amplitude_estimation"] is True
    assert amplitude["requires_simulator_metadata"] is True
    assert np.isfinite(amplitude["query_cost_proxy"])
    assert amplitude["query_cost_proxy"] >= 1.0

    observable = by_protocol["observable_calibrated"]
    assert observable["deployability_class"] == "simulator_assisted_proxy"
    assert observable["requires_simulator_metadata"] is True


def test_all_rows_use_valid_labels_and_claims() -> None:
    rows = evaluate_scale_recovery_point(subproblem=_toy_subproblem(), alpha=1.0e-3, degree=7)
    assert {row["scale_protocol"] for row in rows} == {
        "known_C",
        "ridge_norm",
        "best_scalar_diagnostic",
        "success_amplitude_proxy",
        "amplitude_estimation_proxy",
        "observable_calibrated",
    }
    for row in rows:
        assert row["deployability_class"] in DEPLOYABILITY_LABELS
        assert row["claim_allowed"] == "scale-recovery protocol / norm-recovery proxy"
        assert row["claim_disallowed"] == "QSVT beats Ridge; quantum speedup"
        assert row["dominant_limitation"] in {
            "direction_limited",
            "magnitude_limited",
            "polynomial_or_conditioning",
        }


def test_outputs_write_required_files(tmp_path: Path) -> None:
    rows = evaluate_scale_recovery_point(subproblem=_toy_subproblem(), alpha=1.0e-2, degree=5)
    artifacts = write_scale_recovery_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)

    assert artifacts["scale_recovery_summary"].is_file()
    assert artifacts["residual_by_scale_protocol"].is_file()
    assert artifacts["norm_recovery_cost_proxy"].is_file()
    assert artifacts["scale_recovery_interpretation"].is_file()
    assert artifacts["manifest"].is_file()

    cost_frame = pd.read_csv(artifacts["norm_recovery_cost_proxy"])
    assert set(cost_frame["scale_protocol"]) == {
        "success_amplitude_proxy",
        "amplitude_estimation_proxy",
    }

    summary = pd.read_csv(artifacts["scale_recovery_summary"])
    assert "direction_error_vs_ridge" in summary.columns
    assert "query_cost_proxy" in summary.columns
