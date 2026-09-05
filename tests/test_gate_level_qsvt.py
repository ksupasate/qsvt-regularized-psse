from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    build_structured_qsvt_operator_circuit,
    projector_controlled_phase_matrix,
    qsvt_sequence_operation_counts,
    run_gate_level_qsvt_demo,
)


def test_projector_controlled_phase_matches_pennylane_pcphase_convention() -> None:
    matrix = projector_controlled_phase_matrix(0.25, encoded_dimension=2, total_dimension=4)

    np.testing.assert_allclose(np.diag(matrix)[:2], np.exp(0.25j))
    np.testing.assert_allclose(np.diag(matrix)[2:], np.exp(-0.25j))


def test_structured_qsvt_circuit_matches_pennylane_qsvt_matrix() -> None:
    import pennylane as qml
    from qiskit.quantum_info import Operator

    A = np.array([[0.2, 0.1], [0.0, 0.3]])
    block = canonical_square_block_encoding(A)
    phases = np.array([0.1, 0.2, -0.3, 0.4])
    bundle = build_structured_qsvt_operator_circuit(
        block.unitary,
        phases,
        encoded_dimension=2,
    )
    qiskit_matrix = np.asarray(Operator(bundle.qsvt_operator_circuit).data)
    wires = [0, 1]
    qml_operator = qml.QSVT(
        qml.QubitUnitary(block.unitary, wires=wires),
        [qml.PCPhase(float(phase), dim=2, wires=wires) for phase in phases],
    )
    pennylane_matrix = np.asarray(qml.matrix(qml_operator, wire_order=wires))

    np.testing.assert_allclose(qiskit_matrix, pennylane_matrix, atol=1.0e-12)
    assert bundle.block_encoding_gate_count == 3
    assert bundle.phase_gate_count == 4


def test_qsvt_operation_counts_separate_signal_calls_and_phases() -> None:
    counts = qsvt_sequence_operation_counts(32)
    assert counts == {
        "signal_unitary_calls": 31,
        "projector_phase_operations": 32,
        "alternating_sequence_length": 63,
    }


@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_gate_level_qsvt_demo_writes_required_outputs(tmp_path: Path) -> None:
    run = run_gate_level_qsvt_demo(
        {
            "output_dir": str(tmp_path),
            "case": "ieee14",
            "case_name": "ieee14",
            "submatrix_size": 4,
            "alpha": 1.0e-4,
            "degree": 5,
            "max_synthesis_degree": 5,
            "shots": 100,
            "seed": 123,
        }
    )

    for name in [
        "manifest",
        "circuit_summary",
        "transpiled_resource_summary",
        "statevector_comparison",
        "observable_shot_summary",
        "qsvt_circuit_qasm",
    ]:
        assert run["artifacts"][name].is_file()
    assert run["summary"]["uses_dense_block_encoding_gate"] is True
    assert run["statevector_comparison"]["success_probability"] > 0.0
