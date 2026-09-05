from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.gate_level_qsvt_convention import best_sign_l2_error
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    build_tiny_weighted_problem,
    extract_state_estimation_subproblem,
    run_ieee_gate_level_state_estimation_solver,
    run_tiny_gate_level_state_estimation_solver,
    solve_gate_level_state_estimation_problem,
)


def test_tiny_gate_level_solver_reduces_weighted_residual() -> None:
    problem = build_tiny_weighted_problem()
    computation = solve_gate_level_state_estimation_problem(
        H_tilde=problem.H_tilde,
        r_tilde=problem.r_tilde,
        alpha=1.0e-2,
        degree=5,
        shots=200,
        seed=123,
        metadata=problem.metadata,
        export_qasm=False,
    )
    complex_prefix = computation.statevector[: problem.H_tilde.shape[0]]
    complex_prefix = complex_prefix / np.linalg.norm(complex_prefix)
    uncorrected_error = best_sign_l2_error(computation.ridge_state, complex_prefix)

    assert (
        computation.summary["residual_after_qsvt_update"]
        < computation.summary["residual_before_update"]
    )
    assert computation.summary["phase_or_sign_aligned_state_error"] < uncorrected_error
    assert (
        computation.summary["correct_state_extraction_rule"]
        == "real_prefix_signal_quadrature_state"
    )


def test_tiny_solver_writes_required_outputs(tmp_path: Path) -> None:
    run = run_tiny_gate_level_state_estimation_solver(
        {
            "output_dir": str(tmp_path),
            "alpha": 1.0e-2,
            "degree": 5,
            "shots": 200,
            "seed": 123,
        }
    )

    for name in [
        "manifest",
        "solver_diagnostics",
        "state_error_summary",
        "residual_reduction_summary",
        "circuit_resource_summary",
        "solver_summary",
    ]:
        assert run["artifacts"][name].is_file()
    assert run["summary"]["residual_after_qsvt_update"] < run["summary"]["residual_before_update"]


def test_ieee14_subproblem_extraction_is_deterministic() -> None:
    first = extract_state_estimation_subproblem(
        case="ieee14",
        model="ac_linearized",
        submatrix_size=4,
        seed=123,
    )
    second = extract_state_estimation_subproblem(
        case="ieee14",
        model="ac_linearized",
        submatrix_size=4,
        seed=123,
    )

    np.testing.assert_allclose(first.H_tilde, second.H_tilde)
    np.testing.assert_allclose(first.r_tilde, second.r_tilde)
    assert first.metadata["selected_state_indices"] == second.metadata["selected_state_indices"]
    assert (
        first.metadata["selected_measurement_indices"]
        == second.metadata["selected_measurement_indices"]
    )


def test_ieee14_solver_writes_qasm_and_required_fields(tmp_path: Path) -> None:
    run = run_ieee_gate_level_state_estimation_solver(
        {
            "output_dir": str(tmp_path),
            "case": "ieee14",
            "model": "ac_linearized",
            "submatrix_size": 4,
            "alpha": 1.0e-4,
            "degree": 9,
            "shots": 100,
            "seed": 123,
        }
    )
    summary_path = run["artifacts"]["qsvt_solver_circuit_summary"]
    update_path = run["artifacts"]["qsvt_vs_ridge_update_summary"]

    assert run["artifacts"]["qsvt_solver_circuit_qasm"].is_file()
    assert run["summary"]["qasm_export_status"] in {"succeeded", "failed_with_text_fallback"}
    assert run["summary"]["residual_after_qsvt_update"] < run["summary"]["residual_before_update"]
    assert "state_error_vs_ridge" in summary_path.read_text(encoding="utf-8")
    assert not pd.read_csv(update_path).empty
