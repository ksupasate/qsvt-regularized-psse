from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.paper.circuit_signed_readout import (
    READOUT_CLASS_CIRCUIT,
    circuit_signed_readout_rows,
    estimate_overlap,
    hadamard_overlap_circuit,
)

pytest.importorskip("qiskit")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_overlap_exact_matches_inner_product(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=4)
    b = rng.normal(size=4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    estimate = estimate_overlap(a, b, shots=1000, seed=seed, prefer_aer=False)
    # The Hadamard-test ancilla probability encodes the exact signed overlap.
    assert estimate.overlap_exact == pytest.approx(float(a @ b), abs=1e-9)


def test_signed_readout_recovers_sign_on_known_toy_case():
    # Anti-aligned and aligned states must yield opposite-sign overlaps.
    aligned_a = np.array([1.0, 0.0, 0.0, 0.0])
    aligned_b = np.array([0.8, 0.6, 0.0, 0.0])
    anti_b = np.array([-0.8, 0.6, 0.0, 0.0])
    pos = estimate_overlap(aligned_a, aligned_b, shots=200_000, seed=3)
    neg = estimate_overlap(aligned_a, anti_b, shots=200_000, seed=3)
    assert pos.overlap_estimate > 0.0
    assert neg.overlap_estimate < 0.0
    assert pos.overlap_estimate == pytest.approx(0.8, abs=0.02)
    assert neg.overlap_estimate == pytest.approx(-0.8, abs=0.02)


def test_hadamard_overlap_circuit_structure():
    circuit = hadamard_overlap_circuit(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    # 1 system qubit + 1 ancilla, 1 classical bit measuring the ancilla.
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 1


def test_circuit_signed_readout_rows_recover_physical_value():
    # A real unit output state and a unit observable; physical value = scale * overlap.
    output_state = np.array([0.6, -0.8, 0.0, 0.0])
    observables = [
        {
            "observable_id": "e0",
            "observable_type": "single",
            "vector": np.array([1.0, 0.0, 0.0, 0.0]),
        }
    ]
    scale = 0.05
    ridge_reference = {"e0": scale * 0.6}
    rows = circuit_signed_readout_rows(
        observables=observables,
        output_state=output_state,
        physical_recovery_scale=scale,
        ridge_reference=ridge_reference,
        shots_grid=(200_000,),
        seed=11,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["readout_class"] == READOUT_CLASS_CIRCUIT
    assert row["full_vector_recovery"] is False
    # Exact (noiseless) physical value matches the reference; shot estimate is close.
    assert row["physical_signed_value_exact"] == pytest.approx(scale * 0.6, abs=1e-9)
    assert row["physical_signed_value_estimate"] == pytest.approx(scale * 0.6, abs=5e-3)


def test_complex_output_state_real_overlap_is_extracted():
    # The Hadamard test must return the real part of the overlap for a complex state.
    output_state = np.array([0.6 + 0.0j, 0.0 + 0.8j, 0.0, 0.0])
    observable = np.array([1.0, 0.0, 0.0, 0.0])
    estimate = estimate_overlap(observable, output_state, shots=1000, seed=1, prefer_aer=False)
    assert estimate.overlap_exact == pytest.approx(0.6, abs=1e-9)
