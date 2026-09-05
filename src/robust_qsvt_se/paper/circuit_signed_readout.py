"""Isolated signed-overlap shot-noise experiment with assumed state preparation.

This is a *concrete circuit-level* signed-readout primitive, upgrading the
previously synthetic (binomial-from-known-mean) sign-aware diagnostic. Given
state-preparation unitaries ``V_l`` and ``V_psi`` for the normalized observable
``|l_hat>`` and the postselected QSVT output ``|psi_out>`` (both real,
power-of-two dimension), the signed overlap

    mu = <l_hat | psi_out> = Re <0| V_l^dag V_psi |0>

is estimated by an explicit ancilla-controlled Hadamard-test circuit: an ancilla
in ``|+>``, a controlled ``U = V_l^dag V_psi``, a closing Hadamard, and an
ancilla measurement give ``P(anc=0) = (1 + mu)/2``; the estimator
``mu_hat = 2 P_hat(0) - 1`` is unbiased and recovers the sign. The physical
signed functional is ``y_l = (C/beta) ||r|| sqrt(p_success) ||l|| mu``.

Scope / claim boundary: the overlap circuit (ancilla, controlled unitary,
Hadamard, measurement, shot sampling) is synthesized and simulated only after
the postselected QSVT output vector has been computed classically and loaded by
``StatePreparation(output_state)``.  It therefore characterizes isolated
signed-overlap shot noise under assumed output-state preparation.  It is not an
integrated residual-loading--QSVT--postselection--readout execution, not a
scalable preparation proof, not full-vector recovery, and not a hardware run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CIRCUIT_READOUT_MODEL = "isolated_hadamard_overlap_assumed_output_state_preparation"
SIGN_CONVENTION = "P(anc=0) = (1 + <l_hat|psi_out>)/2; mu_hat = 2 P_hat(0) - 1"

# Readout-class labels used throughout the package to keep the boundary explicit.
READOUT_CLASS_CIRCUIT = "isolated_overlap_circuit_assumed_output_state_preparation"
READOUT_CLASS_SYNTHETIC = "synthetic_shot_model"
READOUT_CLASS_BASIS_SAMPLING = "basis_sampling_energy"
READOUT_CLASS_EXCLUDED_FULL_VECTOR = "excluded_full_vector_recovery"


@dataclass(frozen=True, slots=True)
class OverlapEstimate:
    overlap_exact: float
    overlap_estimate: float
    success_probability_ancilla: float
    shots: int
    standard_error: float
    backend: str


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _pad_unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.complex128).ravel()
    dimension = _next_power_of_two(values.size)
    padded = np.zeros(dimension, dtype=np.complex128)
    padded[: values.size] = values
    norm = float(np.linalg.norm(padded))
    if norm <= 1.0e-15:
        raise ValueError("cannot prepare a zero state for overlap readout")
    return padded / norm


def hadamard_overlap_circuit(observable: np.ndarray, output_state: np.ndarray) -> Any:
    """Return an isolated Hadamard-test circuit estimating ``<l_hat|psi_out>``.

    Both inputs are normalized and zero-padded to a common power-of-two dimension.
    In particular, ``output_state`` is loaded directly with ``StatePreparation``;
    the function does not compose the residual loader or the QSVT circuit.
    The circuit has one ancilla (control) plus ``log2(dim)`` system qubits and a
    single classical bit measuring the ancilla.
    """

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.library import StatePreparation  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for circuit-level signed readout") from exc

    a_state = _pad_unit(observable)
    b_state = _pad_unit(output_state)
    if a_state.size != b_state.size:
        raise ValueError("observable and output_state must share a padded dimension")
    n_system = int(np.log2(a_state.size))

    prep_a = StatePreparation(a_state)
    prep_b = StatePreparation(b_state)
    overlap_unitary = QuantumCircuit(n_system, name="Vl_dag_Vpsi")
    overlap_unitary.append(prep_b, range(n_system))
    overlap_unitary.append(prep_a.inverse(), range(n_system))
    controlled = overlap_unitary.to_gate().control(1)

    circuit = QuantumCircuit(n_system + 1, 1, name="hadamard_test_overlap")
    ancilla = n_system
    circuit.h(ancilla)
    circuit.append(controlled, [ancilla, *range(n_system)])
    circuit.h(ancilla)
    circuit.measure(ancilla, 0)
    return circuit


def _exact_ancilla_zero_probability(circuit: Any) -> float:
    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    without_measure = circuit.remove_final_measurements(inplace=False)
    state = Statevector(without_measure)
    ancilla_index = circuit.num_qubits - 1
    probabilities = np.asarray(state.probabilities([ancilla_index]), dtype=np.float64)
    return float(probabilities[0])


def estimate_overlap(
    observable: np.ndarray,
    output_state: np.ndarray,
    *,
    shots: int,
    seed: int,
    prefer_aer: bool = True,
) -> OverlapEstimate:
    """Estimate the signed overlap by sampling the synthesized Hadamard-test circuit.

    When ``qiskit-aer`` is available the circuit is sampled on the Aer backend
    (genuine shot sampling through the compiled circuit); otherwise the ancilla
    outcomes are drawn from the exact ``P(anc=0)`` (still a circuit-derived
    probability, not a precomputed observable value).
    """

    if int(shots) <= 0:
        raise ValueError("shots must be positive")
    circuit = hadamard_overlap_circuit(observable, output_state)
    p_zero_exact = _exact_ancilla_zero_probability(circuit)
    overlap_exact = 2.0 * p_zero_exact - 1.0

    backend = "exact_p0_binomial"
    p_zero_hat = p_zero_exact
    if prefer_aer:
        try:
            from qiskit import transpile  # type: ignore[import-not-found]
            from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

            simulator = AerSimulator()
            compiled = transpile(circuit, simulator)
            result = simulator.run(compiled, shots=int(shots), seed_simulator=int(seed)).result()
            counts = result.get_counts()
            zeros = int(counts.get("0", 0))
            p_zero_hat = zeros / float(shots)
            backend = f"aer:{AerSimulator().__class__.__name__}"
        except Exception:
            backend = "exact_p0_binomial"

    if backend == "exact_p0_binomial":
        rng = np.random.default_rng(int(seed))
        zeros = int(rng.binomial(int(shots), p_zero_exact))
        p_zero_hat = zeros / float(shots)

    overlap_estimate = 2.0 * p_zero_hat - 1.0
    variance = max(p_zero_exact * (1.0 - p_zero_exact), 0.0)
    standard_error = 2.0 * float(np.sqrt(variance / float(shots)))
    return OverlapEstimate(
        overlap_exact=float(overlap_exact),
        overlap_estimate=float(overlap_estimate),
        success_probability_ancilla=float(p_zero_exact),
        shots=int(shots),
        standard_error=standard_error,
        backend=backend,
    )


def circuit_signed_readout_rows(
    *,
    observables: list[dict[str, Any]],
    output_state: np.ndarray,
    physical_recovery_scale: float,
    ridge_reference: dict[str, float],
    shots_grid: tuple[int, ...] = (1_000, 10_000, 100_000),
    seed: int = 20260629,
) -> list[dict[str, Any]]:
    """Build circuit-level signed-readout rows for each signed observable.

    ``physical_recovery_scale`` is ``(C/beta) ||r|| sqrt(p_success)`` (the factor
    that maps the postselected unit output to the physical update). For an
    observable functional ``l`` the physical signed value is
    ``y_l = physical_recovery_scale * ||l|| * mu`` with ``mu = <l_hat|psi_out>``.
    """

    # Keep the postselected output state complex: the Hadamard test extracts the real
    # overlap Re<l_hat|psi_out>, which is the physical signed functional (dx is real).
    psi = np.asarray(output_state, dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    for observable in observables:
        ell = np.asarray(observable["vector"], dtype=np.float64)
        ell_norm = float(np.linalg.norm(ell))
        if ell_norm <= 0.0:
            continue
        scale = float(physical_recovery_scale) * ell_norm
        ridge_value = float(ridge_reference[observable["observable_id"]])
        for shots in shots_grid:
            estimate = estimate_overlap(ell, psi, shots=int(shots), seed=int(seed))
            physical_estimate = scale * estimate.overlap_estimate
            physical_exact = scale * estimate.overlap_exact
            abs_error = abs(physical_estimate - ridge_value)
            rel_error = abs_error / abs(ridge_value) if abs(ridge_value) > 1.0e-15 else float("nan")
            rows.append(
                {
                    "observable_id": observable["observable_id"],
                    "observable_type": observable["observable_type"],
                    "readout_class": READOUT_CLASS_CIRCUIT,
                    "readout_model": CIRCUIT_READOUT_MODEL,
                    "observable_state_preparation": "dense_amplitude_loading_state_preparation",
                    "qsvt_output_state_preparation": (
                        "direct_StatePreparation_of_classically_computed_postselected_output"
                    ),
                    "integrated_qsvt_readout": False,
                    "observable_normalized": True,
                    "sign_convention": SIGN_CONVENTION,
                    "shots": int(shots),
                    "backend": estimate.backend,
                    "overlap_exact": estimate.overlap_exact,
                    "overlap_estimate": estimate.overlap_estimate,
                    "ancilla_success_probability": estimate.success_probability_ancilla,
                    "standard_error_overlap": estimate.standard_error,
                    "physical_recovery_scale": scale,
                    "physical_signed_value_exact": physical_exact,
                    "physical_signed_value_estimate": physical_estimate,
                    "ridge_reference_value": ridge_value,
                    "absolute_error_vs_ridge": abs_error,
                    "relative_error_vs_ridge": rel_error,
                    "full_vector_recovery": False,
                }
            )
    return rows
