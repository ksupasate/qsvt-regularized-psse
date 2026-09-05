from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.block_encoding import (
    BlockEncodingResult,
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.phase_synthesis import synthesize_pennylane_phases_cached
from robust_qsvt_se.qsvt.polynomial import (
    OddPolynomialApproximation,
    fit_odd_regularized_polynomial,
)


@dataclass(frozen=True, slots=True)
class HardwareQSVTResult:
    block_encoding: BlockEncodingResult
    coefficients: np.ndarray
    approximation: OddPolynomialApproximation
    phases: np.ndarray
    circuit: Any
    transpiled_circuit: Any | None
    transformed_block: np.ndarray
    classical_transform: np.ndarray
    comparison: pd.DataFrame
    simulation: pd.DataFrame
    summary: dict[str, Any]
    block_encoding_summary: dict[str, Any]
    gate_counts: dict[str, int]
    transpiled_gate_counts: dict[str, int]


def run_explicit_hardware_qsvt(
    matrix: np.ndarray,
    *,
    alpha: float,
    polynomial_degree: int,
    grid_size: int,
    angle_solver: str,
    phase_cache_dir: str | Path,
    basis_gates: list[str],
    domain_min: float | None = None,
    transpile_optimization_level: int = 1,
) -> HardwareQSVTResult:
    """Build and validate a structured small-matrix block-encoding QSVT circuit.

    The block-encoding primitive is a dense canonical unitary for a small
    research-derived matrix. The full circuit is not a single dense QSVT
    unitary: it is a sequence of block-encoding calls, adjoint calls, and
    explicit projector phase rotations.
    """
    try:
        from qiskit import QuantumCircuit, transpile  # type: ignore[import-not-found]
        from qiskit.circuit.library import UnitaryGate  # type: ignore[import-not-found]
        from qiskit.quantum_info import Operator, Statevector  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for the hardware QSVT prototype") from exc

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("hardware QSVT requires a square matrix")
    singular_values = np.linalg.svd(values, compute_uv=False)
    positive_singular_values = singular_values[singular_values > 1.0e-14]
    if positive_singular_values.size == 0:
        raise ValueError("hardware QSVT matrix must have at least one positive singular value")
    fit_domain_min = (
        float(domain_min)
        if domain_min is not None
        else max(float(np.min(positive_singular_values)), 0.05)
    )
    fit_domain_min = min(fit_domain_min, 0.95)
    approximation = fit_odd_regularized_polynomial(
        alpha=float(alpha),
        block_encoding_normalization=1.0,
        degree=int(polynomial_degree),
        domain_min=fit_domain_min,
        domain_max=1.0,
        grid_size=int(grid_size),
    )
    coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
    scaled_coefficients = coefficients / approximation.scale_factor
    phase_result = synthesize_pennylane_phases_cached(
        scaled_coefficients,
        angle_solver=str(angle_solver),
        cache_dir=phase_cache_dir,
        cache_metadata={
            "alpha": float(alpha),
            "degree": int(polynomial_degree),
            "domain_min": fit_domain_min,
            "domain_max": 1.0,
            "scale_factor": float(approximation.scale_factor),
            "parity": "odd",
            "implementation": "explicit_hardware_qsvt",
        },
    )
    phases = phase_result.phases
    block_encoding = canonical_square_block_encoding(values)
    block_validation = validate_block_encoding(block_encoding)

    block_gate = UnitaryGate(block_encoding.unitary, label="U_A")
    n_qubits = int(np.log2(block_encoding.unitary.shape[0]))
    circuit = QuantumCircuit(n_qubits, name="explicit_research_matrix_qsvt")
    _append_qsvt_sequence(circuit, block_gate, phases)

    operator_matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
    dimension = values.shape[0]
    transformed_block = np.real(operator_matrix[:dimension, :dimension])
    classical = classical_spectral_transform(values, scaled_coefficients)
    comparison = comparison_frame(classical, transformed_block)

    state_start = time.perf_counter()
    state = Statevector.from_instruction(circuit)
    simulation_seconds = time.perf_counter() - state_start
    simulation = pd.DataFrame(
        {
            "state_index": np.arange(len(state.data)),
            "amplitude_real": np.real(state.data),
            "amplitude_imag": np.imag(state.data),
        }
    )

    gate_counts = dict(circuit.count_ops())
    transpiled_circuit = None
    transpiled_gate_counts: dict[str, int] = {}
    transpile_success = False
    transpile_message = ""
    transpile_seconds = 0.0
    try:
        start = time.perf_counter()
        transpiled_circuit = transpile(
            circuit,
            basis_gates=list(basis_gates),
            optimization_level=int(transpile_optimization_level),
        )
        transpile_seconds = time.perf_counter() - start
        transpiled_gate_counts = dict(transpiled_circuit.count_ops())
        transpile_success = True
        transpile_message = "transpile completed"
    except Exception as exc:  # pragma: no cover - backend-version dependent
        transpile_seconds = time.perf_counter() - start if "start" in locals() else 0.0
        transpile_message = str(exc)

    max_error = float(comparison["abs_error_to_classical_filter"].max())
    mean_error = float(comparison["abs_error_to_classical_filter"].mean())
    cx_count = int(transpiled_gate_counts.get("cx", 0))
    summary = {
        "block_encoding_type": block_encoding.summary["block_encoding_type"],
        "qsvt_construction_type": "explicit_block_encoding_qsvt_sequence",
        "qsvt_method": "qiskit_structured_block_encoding_qsvt",
        "matrix_shape": [int(values.shape[0]), int(values.shape[1])],
        "block_unitary_dimension": int(block_encoding.unitary.shape[0]),
        "qsvt_phase_count": int(phases.size),
        "n_phase_angles": int(phases.size),
        "qsvt_polynomial_degree": int(polynomial_degree),
        "polynomial_degree": int(polynomial_degree),
        "phase_synthesis_method": "pennylane_poly_to_angles",
        "phase_synthesis_angle_solver": str(angle_solver),
        "phase_cache_hit": bool(phase_result.cache_hit),
        "qubits_before_transpile": int(circuit.num_qubits),
        "qubits": int(circuit.num_qubits),
        "depth_before_transpile": int(circuit.depth()),
        "circuit_depth": int(circuit.depth()),
        "gate_counts_before_transpile": gate_counts,
        "gate_count_total": int(sum(gate_counts.values())),
        "basis_gates": list(basis_gates),
        "depth_after_transpile": (
            int(transpiled_circuit.depth()) if transpiled_circuit is not None else None
        ),
        "gate_counts_after_transpile": transpiled_gate_counts,
        "gate_count_total_after_transpile": int(sum(transpiled_gate_counts.values())),
        "cx_count_after_transpile": cx_count,
        "transpile_success": transpile_success,
        "transpile_message": transpile_message,
        "transpile_seconds": float(transpile_seconds),
        "simulation_backend": "qiskit.quantum_info.Statevector",
        "simulation_success": True,
        "simulation_seconds": float(simulation_seconds),
        "max_error_vs_classical": max_error,
        "mean_error_vs_classical": mean_error,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "is_dense_unitary_only": False,
        "uses_dense_block_encoding_gate": True,
        "implementation_scope": "explicit small-matrix block-encoding QSVT prototype",
        "scope_note": (
            "The block encoding is a dense canonical unitary for a small "
            "research-derived matrix. The QSVT circuit itself is a structured "
            "sequence of block-encoding calls and explicit projector phase rotations."
        ),
        **block_validation,
    }
    block_summary = {
        **block_encoding.summary,
        **block_validation,
        "normalization_required": "input matrix is expected to be pre-normalized",
    }
    return HardwareQSVTResult(
        block_encoding=block_encoding,
        coefficients=scaled_coefficients,
        approximation=approximation,
        phases=phases,
        circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        transformed_block=transformed_block,
        classical_transform=classical,
        comparison=comparison,
        simulation=simulation,
        summary=summary,
        block_encoding_summary=block_summary,
        gate_counts=gate_counts,
        transpiled_gate_counts=transpiled_gate_counts,
    )


def classical_spectral_transform(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64), full_matrices=False
    )
    values = Polynomial(np.asarray(coefficients, dtype=np.float64))(singular_values)
    return u @ np.diag(values) @ vh


def comparison_frame(classical: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    rows = []
    for row in range(classical.shape[0]):
        for column in range(classical.shape[1]):
            error = abs(float(observed[row, column] - classical[row, column]))
            rows.append(
                {
                    "row": row,
                    "column": column,
                    "classical_spectral_filter": float(classical[row, column]),
                    "hardware_qsvt_block_value": float(observed[row, column]),
                    "abs_error_to_classical_filter": error,
                    "max_error_vs_classical": error,
                }
            )
    return pd.DataFrame(rows)


def approximation_error_frame(approximation: OddPolynomialApproximation) -> pd.DataFrame:
    grid = np.linspace(approximation.domain_min, approximation.domain_max, 512, dtype=np.float64)
    target = grid / (grid**2 + approximation.alpha)
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients, dtype=np.float64) / approximation.scale_factor
    )
    scaled_target = target / approximation.scale_factor
    scaled_polynomial = polynomial(grid)
    return pd.DataFrame(
        {
            "normalized_singular_value": grid,
            "scaled_target": scaled_target,
            "scaled_polynomial": scaled_polynomial,
            "abs_error": np.abs(scaled_polynomial - scaled_target),
        }
    )


def _append_qsvt_sequence(circuit: Any, block_gate: Any, phases: np.ndarray) -> None:
    phase_qubit = circuit.num_qubits - 1
    qubits = list(range(circuit.num_qubits))
    phase_values = np.asarray(phases, dtype=np.float64)
    circuit.rz(-2.0 * float(phase_values[0]), phase_qubit)
    inverse_gate = block_gate.inverse()
    for index in range(1, phase_values.size - 1, 2):
        circuit.append(block_gate, qubits)
        circuit.rz(-2.0 * float(phase_values[index]), phase_qubit)
        circuit.append(inverse_gate, qubits)
        circuit.rz(-2.0 * float(phase_values[index + 1]), phase_qubit)
    if phase_values.size % 2 == 0:
        circuit.append(block_gate, qubits)
        circuit.rz(-2.0 * float(phase_values[-1]), phase_qubit)
