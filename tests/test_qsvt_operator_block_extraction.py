from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.gate_level_qsvt_convention import (
    CORRECT_OPERATOR_EXTRACTION_RULE,
    CORRECT_STATE_EXTRACTION_RULE,
    best_operator_rule,
    best_state_rule,
    build_convention_probe,
    operator_block_error_rows,
    qiskit_pennylane_comparison,
    state_extraction_error_rows,
)


def test_operator_block_extraction_finds_real_top_left_rule() -> None:
    probe = build_convention_probe(np.diag([0.25, 0.7]))
    rows = operator_block_error_rows(
        probe.qiskit_unitary,
        probe.target_block,
        encoded_dimension=2,
    )
    correct = next(
        row for row in rows if row["extraction_rule"] == CORRECT_OPERATOR_EXTRACTION_RULE
    )
    best = best_operator_rule(rows)

    assert correct["frobenius_error"] < 1.0e-10
    assert best["frobenius_error"] < 1.0e-10


def test_qiskit_and_pennylane_operators_match() -> None:
    probe = build_convention_probe(np.diag([0.2, 0.6]))
    comparison = qiskit_pennylane_comparison(probe)

    assert comparison["same_phase_sequence"] is True
    assert comparison["same_qubit_ordering"] is True
    assert comparison["frobenius_error"] < 1.0e-10


def test_state_extraction_prefers_real_signal_quadrature() -> None:
    probe = build_convention_probe(np.diag([0.25, 0.7]))
    residual = np.array([0.6, -0.8], dtype=np.complex128)
    full_input = np.zeros(probe.qiskit_unitary.shape[0], dtype=np.complex128)
    full_input[: residual.size] = residual
    state = probe.qiskit_unitary @ full_input
    target = probe.target_block @ residual
    rows = state_extraction_error_rows(state, target, encoded_dimension=2)
    correct = next(row for row in rows if row["extraction_rule"] == CORRECT_STATE_EXTRACTION_RULE)
    complex_prefix = next(
        row for row in rows if row["extraction_rule"] == "complex_prefix_postselected_state"
    )
    best = best_state_rule(rows)

    assert correct["best_sign_l2_error"] < 1.0e-10
    assert best["best_sign_l2_error"] < 1.0e-10
    assert correct["best_sign_l2_error"] < complex_prefix["best_sign_l2_error"]
