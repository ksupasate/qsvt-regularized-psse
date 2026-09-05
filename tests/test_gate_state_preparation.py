from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.gate_state_preparation import (
    build_initialize_circuit,
    normalize_and_pad_for_gate_preparation,
    validate_initialize_circuit,
)


def test_normalize_and_pad_for_gate_preparation() -> None:
    result = normalize_and_pad_for_gate_preparation(
        np.array([3.0, 4.0]),
        target_dimension=4,
    )

    assert result.original_norm == pytest.approx(5.0)
    assert result.n_qubits == 2
    np.testing.assert_allclose(result.padded_state, np.array([0.6, 0.8, 0.0, 0.0]))


def test_initialize_circuit_prepares_target_state() -> None:
    result = normalize_and_pad_for_gate_preparation(np.array([1.0, -1.0]))
    circuit = build_initialize_circuit(result.padded_state)
    validation = validate_initialize_circuit(circuit, result.padded_state)

    assert validation["state_preparation_l2_error"] < 1.0e-12
    assert validation["state_preparation_fidelity"] == pytest.approx(1.0)
