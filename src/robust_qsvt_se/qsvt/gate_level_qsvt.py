from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import (
    _build_phase_sequence,
    _build_source_matrix,
    rescale_bounded_target_to_original,
)
from robust_qsvt_se.qsvt.gate_state_preparation import (
    build_initialize_circuit,
    normalize_and_pad_for_gate_preparation,
    validate_initialize_circuit,
)
from robust_qsvt_se.qsvt.observable_measurement_circuits import (
    add_computational_basis_measurements,
    default_update_observables,
    exact_postselected_observable_values,
    observable_values_from_state,
    postselected_encoded_state,
    sample_statevector_counts,
    shot_observable_summary,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

GATE_LEVEL_QSVT_CLAIM = (
    "This adds small gate-level QSVT evidence and a toy sparse-oracle gate "
    "demonstration. It does not demonstrate full IEEE-scale hardware execution, "
    "quantum speedup, QSVT numerical superiority over Ridge/Tikhonov, or "
    "field-data validation."
)


@dataclass(frozen=True, slots=True)
class GateLevelQSVTCircuitBundle:
    qsvt_operator_circuit: Any
    full_state_circuit: Any
    measured_circuit: Any
    block_encoding_gate_count: int
    phase_gate_count: int
    n_qubits: int


def qsvt_sequence_operation_counts(phase_count: int) -> dict[str, int]:
    """Return the operation counts used by this repository's QSVT sequence.

    ``build_structured_qsvt_operator_circuit`` starts with one projector-phase
    operation and then alternates one signal-unitary call with one projector
    phase.  Consequently a degree-``d`` polynomial represented by ``d + 1``
    phases uses ``d`` calls to ``U_A`` or ``U_A^dagger`` and ``2d + 1`` total
    alternating operations.  The latter is a sequence length, not a
    block-encoding query count.
    """

    count = int(phase_count)
    if count < 1:
        raise ValueError("phase_count must be positive")
    signal_calls = 0
    for _index in range(1, count - 1, 2):
        signal_calls += 2
    if count % 2 == 0:
        signal_calls += 1
    return {
        "signal_unitary_calls": signal_calls,
        "projector_phase_operations": count,
        "alternating_sequence_length": signal_calls + count,
    }


def projector_controlled_phase_matrix(
    phase: float,
    *,
    encoded_dimension: int,
    total_dimension: int,
) -> np.ndarray:
    if encoded_dimension <= 0 or encoded_dimension > total_dimension:
        raise ValueError("encoded_dimension must be in (0, total_dimension]")
    diagonal = np.empty(int(total_dimension), dtype=np.complex128)
    diagonal[: int(encoded_dimension)] = np.exp(1j * float(phase))
    diagonal[int(encoded_dimension) :] = np.exp(-1j * float(phase))
    return np.diag(diagonal)


def build_structured_qsvt_operator_circuit(
    block_unitary: np.ndarray,
    phases: np.ndarray,
    *,
    encoded_dimension: int,
) -> GateLevelQSVTCircuitBundle:
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.library import UnitaryGate  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for gate-level QSVT circuits") from exc

    unitary = np.asarray(block_unitary, dtype=np.complex128)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("block_unitary must be square")
    total_dimension = int(unitary.shape[0])
    n_qubits = _qubits(total_dimension)
    if 2**n_qubits != total_dimension:
        raise ValueError("block_unitary dimension must be a power of two")
    phase_values = np.asarray(phases, dtype=np.float64)
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a nonempty vector")

    circuit = QuantumCircuit(n_qubits, name="structured_gate_level_qsvt")
    block_gate = UnitaryGate(unitary, label="U_A")
    inverse_gate = block_gate.inverse()

    def append_pcphase(phi: float) -> None:
        matrix = projector_controlled_phase_matrix(
            phi,
            encoded_dimension=encoded_dimension,
            total_dimension=total_dimension,
        )
        circuit.unitary(matrix, list(range(n_qubits)), label="PCPhase")

    append_pcphase(float(phase_values[0]))
    block_count = 0
    for index in range(1, phase_values.size - 1, 2):
        circuit.append(block_gate, list(range(n_qubits)))
        block_count += 1
        append_pcphase(float(phase_values[index]))
        circuit.append(inverse_gate, list(range(n_qubits)))
        block_count += 1
        append_pcphase(float(phase_values[index + 1]))
    if phase_values.size % 2 == 0:
        circuit.append(block_gate, list(range(n_qubits)))
        block_count += 1
        append_pcphase(float(phase_values[-1]))

    expected = qsvt_sequence_operation_counts(int(phase_values.size))
    if block_count != expected["signal_unitary_calls"]:
        raise RuntimeError("QSVT signal-unitary call accounting does not match the circuit")
    return GateLevelQSVTCircuitBundle(
        qsvt_operator_circuit=circuit,
        full_state_circuit=None,
        measured_circuit=None,
        block_encoding_gate_count=block_count,
        phase_gate_count=int(phase_values.size),
        n_qubits=n_qubits,
    )


def build_gate_level_qsvt_state_circuit(
    *,
    block_unitary: np.ndarray,
    phases: np.ndarray,
    residual_state: np.ndarray,
    encoded_dimension: int,
) -> GateLevelQSVTCircuitBundle:
    operator = build_structured_qsvt_operator_circuit(
        block_unitary,
        phases,
        encoded_dimension=encoded_dimension,
    )
    prep = build_initialize_circuit(residual_state, name="dense_residual_state_prep")
    full = prep.compose(operator.qsvt_operator_circuit)
    measured = add_computational_basis_measurements(full)
    return GateLevelQSVTCircuitBundle(
        qsvt_operator_circuit=operator.qsvt_operator_circuit,
        full_state_circuit=full,
        measured_circuit=measured,
        block_encoding_gate_count=operator.block_encoding_gate_count,
        phase_gate_count=operator.phase_gate_count,
        n_qubits=operator.n_qubits,
    )


def run_gate_level_qsvt_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_gate_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rng = np.random.default_rng(int(resolved["seed"]))

    source = _build_source_matrix(resolved)
    singular_values_B = np.linalg.svd(source.B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = source.B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    alpha_norm = float(resolved["alpha"]) / beta**2
    block_encoding = canonical_square_block_encoding(A, tolerance=float(resolved["tolerance"]))
    block_report = validate_block_encoding(
        block_encoding,
        beta=beta,
        tolerance=float(resolved["validation_tolerance"]),
    )
    phase_sequence = _build_phase_sequence(
        singular_values_A=singular_values_A,
        requested_degree=int(resolved["degree"]),
        alpha_norm=alpha_norm,
        grid_size=int(resolved["grid_size"]),
        max_synthesis_degree=int(resolved["max_synthesis_degree"]),
        angle_solver=str(resolved["angle_solver"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
    )
    state_prep = normalize_and_pad_for_gate_preparation(
        source.residual,
        target_dimension=block_encoding.unitary.shape[0],
    )
    circuit_bundle = build_gate_level_qsvt_state_circuit(
        block_unitary=block_encoding.unitary,
        phases=phase_sequence.phases,
        residual_state=state_prep.padded_state,
        encoded_dimension=A.shape[0],
    )
    state_prep_validation = validate_initialize_circuit(
        build_initialize_circuit(state_prep.padded_state),
        state_prep.padded_state,
    )
    statevector, simulation_seconds = _simulate_statevector(circuit_bundle.full_state_circuit)
    post_state, success_probability = postselected_encoded_state(
        statevector,
        encoded_dimension=A.shape[0],
    )
    signal_vector = np.real(statevector[: A.shape[0]])
    signal_norm = float(np.linalg.norm(signal_vector))
    if signal_norm <= 1.0e-15:
        raise ValueError("real signal quadrature is too small for update comparison")
    signal_state = signal_vector / signal_norm
    qsvt_update = rescale_bounded_target_to_original(
        signal_vector,
        beta=beta,
        C=phase_sequence.approximation.scale_factor,
    )
    ridge_update = _ridge_update(source.B, source.residual, float(resolved["alpha"]))
    ridge_state = ridge_update / max(float(np.linalg.norm(ridge_update)), 1.0e-15)
    statevector_comparison = _statevector_comparison(
        qsvt_update=qsvt_update,
        qsvt_postselected_state=post_state,
        qsvt_signal_state=signal_state,
        ridge_update=ridge_update,
        ridge_state=ridge_state,
        success_probability=success_probability,
        signal_quadrature_norm=signal_norm,
        beta=beta,
        C=phase_sequence.approximation.scale_factor,
    )
    observables = default_update_observables(A.shape[0])
    exact_qsvt_observables = exact_postselected_observable_values(
        statevector,
        encoded_dimension=A.shape[0],
        observables=observables,
    )
    ridge_observables = observable_values_from_state(ridge_state, observables)
    counts = sample_statevector_counts(
        statevector,
        shots=int(resolved["shots"]),
        rng=rng,
    )
    shot_rows = shot_observable_summary(
        counts,
        encoded_dimension=A.shape[0],
        observables=observables,
        exact_qsvt_values=exact_qsvt_observables,
        ridge_values=ridge_observables,
        shots=int(resolved["shots"]),
        seed=int(resolved["seed"]),
    )

    qsvt_operator_unitary_error = _operator_unitarity_error(circuit_bundle.qsvt_operator_circuit)
    transpile_summary = _transpile_and_summarize(
        circuit_bundle.full_state_circuit,
        basis_gates=list(resolved["basis_gates"]),
        optimization_level=int(resolved["transpile_optimization_level"]),
        transpile_qubit_limit=int(resolved["transpile_qubit_limit"]),
    )
    qasm_text = _export_qasm(
        transpile_summary["transpiled_circuit"] or circuit_bundle.full_state_circuit
    )
    circuit_summary = {
        "claim_boundary": GATE_LEVEL_QSVT_CLAIM,
        "case": resolved["case_name"],
        "matrix_source": source.metadata["matrix_source"],
        "matrix_size": [int(A.shape[0]), int(A.shape[1])],
        "submatrix_size": int(resolved["submatrix_size"]),
        "beta": beta,
        "alpha": float(resolved["alpha"]),
        "alpha_norm": alpha_norm,
        "requested_degree": int(resolved["degree"]),
        "constructed_polynomial_degree": int(phase_sequence.coefficients.size - 1),
        "synthesized_phase_degree": int(phase_sequence.actual_degree),
        "effective_qsvt_degree": int(phase_sequence.actual_degree),
        "synthesized_degree": int(phase_sequence.actual_degree),
        "phase_count": int(phase_sequence.phases.size),
        "qsvt_query_count": int(circuit_bundle.block_encoding_gate_count),
        "signal_unitary_calls": int(circuit_bundle.block_encoding_gate_count),
        "alternating_sequence_length": int(
            circuit_bundle.block_encoding_gate_count + circuit_bundle.phase_gate_count
        ),
        "block_encoding_gate_applications": int(circuit_bundle.block_encoding_gate_count),
        "projector_phase_gate_applications": int(circuit_bundle.phase_gate_count),
        "n_qubits": int(circuit_bundle.n_qubits),
        "ancilla_qubits": 1,
        "block_encoding_unitarity_error": block_report["unitarity_error"],
        "top_left_block_error": block_report["top_left_block_error"],
        "qsvt_operator_unitarity_error": qsvt_operator_unitary_error,
        "uses_dense_block_encoding_gate": True,
        "state_preparation": state_prep.to_dict(),
        "state_preparation_validation": state_prep_validation,
        "simulation_backend": "qiskit.quantum_info.Statevector",
        "simulation_seconds": simulation_seconds,
        "shot_sampling_method": "statevector_probabilities_multinomial",
        "shots": int(resolved["shots"]),
        "selected_state_labels": source.row_labels,
        "selected_measurement_labels": source.column_labels,
        "limitation": (
            "Gate-level dense QSVT is executed only for a small IEEE-derived "
            "submatrix. This is separate from matrix-free proxies and IEEE-scale "
            "resource estimates."
        ),
    }
    resources = {
        **transpile_summary["resource_row"],
        "case": resolved["case_name"],
        "submatrix_size": int(resolved["submatrix_size"]),
        "qsvt_query_count": circuit_summary["qsvt_query_count"],
        "phase_count": circuit_summary["phase_count"],
        "ancilla_qubits": circuit_summary["ancilla_qubits"],
        "two_qubit_gate_count_after_transpile": transpile_summary["two_qubit_gate_count"],
    }
    artifacts = _write_gate_level_outputs(
        output_dir=output_dir,
        resolved=resolved,
        circuit_summary=circuit_summary,
        resource_row=resources,
        statevector_comparison=statevector_comparison,
        shot_rows=shot_rows,
        qasm_text=qasm_text,
    )
    manifest = write_manifest(
        output_dir,
        artifacts={key: str(value) for key, value in artifacts.items()},
        input_config=resolved,
        claim_boundary=GATE_LEVEL_QSVT_CLAIM,
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "summary": circuit_summary,
        "statevector_comparison": statevector_comparison,
        "observable_shots": pd.DataFrame(shot_rows),
        "artifacts": artifacts,
    }


def _resolve_gate_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "case": "ieee14",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "max_synthesis_degree": 201,
        "grid_size": 2048,
        "angle_solver": "iterative",
        "phase_timeout_seconds": 25,
        "seed": 123,
        "shots": 1000,
        "output_dir": "outputs/gate_level_qsvt_demo",
        "tolerance": 1.0e-8,
        "validation_tolerance": 1.0e-7,
        "basis_gates": ["rz", "sx", "x", "cx"],
        "transpile_optimization_level": 1,
        "transpile_qubit_limit": 4,
    }
    if config:
        resolved.update(config)
    if "case" in resolved and "case_name" not in (config or {}):
        resolved["case_name"] = resolved["case"]
    size = int(resolved["submatrix_size"])
    if size <= 0 or not _is_power_of_two(size):
        raise ValueError("submatrix_size must be a positive power of two")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if int(resolved["degree"]) <= 0 or int(resolved["degree"]) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    max_degree = int(resolved["max_synthesis_degree"])
    if max_degree % 2 == 0:
        max_degree -= 1
    resolved["max_synthesis_degree"] = max(1, max_degree)
    resolved["submatrix_size"] = size
    resolved["grid_size"] = max(int(resolved["grid_size"]), int(resolved["degree"]) + 2)
    return resolved


def _ridge_update(B: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    H_tilde = np.asarray(B, dtype=np.float64).T
    U, singular_values, Vt = np.linalg.svd(H_tilde, full_matrices=False)
    return Vt.T @ (ridge_filter(singular_values, alpha=alpha) * (U.T @ residual))


def _statevector_comparison(
    *,
    qsvt_update: np.ndarray,
    qsvt_postselected_state: np.ndarray,
    qsvt_signal_state: np.ndarray,
    ridge_update: np.ndarray,
    ridge_state: np.ndarray,
    success_probability: float,
    signal_quadrature_norm: float,
    beta: float,
    C: float,
) -> dict[str, Any]:
    qsvt_update_real = np.real(qsvt_update)
    ridge_real = np.real(ridge_update)
    qsvt_update_norm = float(np.linalg.norm(qsvt_update_real))
    ridge_norm = float(np.linalg.norm(ridge_real))
    signal_error = float(np.linalg.norm(qsvt_signal_state - ridge_state))
    complex_state_error = float(np.linalg.norm(qsvt_postselected_state - ridge_state))
    sign_error = float(
        min(
            np.linalg.norm(qsvt_signal_state - ridge_state),
            np.linalg.norm(qsvt_signal_state + ridge_state),
        )
    )
    vector_error = float(np.linalg.norm(qsvt_update_real - ridge_real))
    relative_vector_error = vector_error / max(ridge_norm, 1.0e-15)
    overlap = complex(np.vdot(ridge_state, qsvt_signal_state))
    return {
        "success_probability": float(success_probability),
        "signal_quadrature_norm": float(signal_quadrature_norm),
        "beta": float(beta),
        "bounded_scaling_C": float(C),
        "ridge_update_norm": ridge_norm,
        "qsvt_rescaled_update_norm": qsvt_update_norm,
        "normalized_state_l2_error_vs_ridge": signal_error,
        "real_signal_quadrature_state_l2_error_vs_ridge": signal_error,
        "complex_postselected_state_l2_error_vs_ridge": complex_state_error,
        "best_sign_state_l2_error_vs_ridge": sign_error,
        "normalized_state_overlap_abs": float(abs(overlap)),
        "full_vector_l2_error_vs_ridge": vector_error,
        "relative_full_vector_error_vs_ridge": relative_vector_error,
        "qsvt_update": qsvt_update_real.tolist(),
        "ridge_update": ridge_real.tolist(),
        "qsvt_signal_quadrature_state": np.real(qsvt_signal_state).tolist(),
        "qsvt_postselected_state_real": np.real(qsvt_postselected_state).tolist(),
        "qsvt_postselected_state_imag": np.imag(qsvt_postselected_state).tolist(),
        "ridge_normalized_state": np.real(ridge_state).tolist(),
        "limitation": (
            "The phase convention encodes the polynomial target in the real signal "
            "quadrature of the encoded block for B=H_tilde.T. Full-vector comparison "
            "uses statevector simulator diagnostics and is not a claim that full "
            "update tomography or quadrature readout is efficient."
        ),
    }


def _simulate_statevector(circuit: Any) -> tuple[np.ndarray, float]:
    try:
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for statevector simulation") from exc
    start = time.perf_counter()
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    return state, float(time.perf_counter() - start)


def _operator_unitarity_error(circuit: Any) -> float:
    try:
        from qiskit.quantum_info import Operator  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for operator validation") from exc
    matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
    return float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(matrix.shape[0]))))


def _transpile_and_summarize(
    circuit: Any,
    *,
    basis_gates: list[str],
    optimization_level: int,
    transpile_qubit_limit: int,
) -> dict[str, Any]:
    try:
        from qiskit import transpile  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for transpilation") from exc
    before_counts = {str(key): int(value) for key, value in circuit.count_ops().items()}
    if circuit.num_qubits > int(transpile_qubit_limit):
        row = {
            "transpile_success": False,
            "transpile_message": (
                f"skipped: {circuit.num_qubits} qubits exceeds limit {transpile_qubit_limit}"
            ),
            "n_qubits": int(circuit.num_qubits),
            "depth_before_transpile": int(circuit.depth()),
            "depth_after_transpile": None,
            "operation_counts_before_transpile": before_counts,
            "operation_counts_after_transpile": {},
            "gate_count_total_before_transpile": int(sum(before_counts.values())),
            "gate_count_total_after_transpile": 0,
        }
        return {"transpiled_circuit": None, "resource_row": row, "two_qubit_gate_count": 0}
    start = time.perf_counter()
    try:
        transpiled = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=int(optimization_level),
        )
        seconds = time.perf_counter() - start
        after_counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        two_qubit = _two_qubit_count(transpiled)
        row = {
            "transpile_success": True,
            "transpile_message": "transpile completed",
            "transpile_seconds": seconds,
            "n_qubits": int(circuit.num_qubits),
            "depth_before_transpile": int(circuit.depth()),
            "depth_after_transpile": int(transpiled.depth()),
            "operation_counts_before_transpile": before_counts,
            "operation_counts_after_transpile": after_counts,
            "gate_count_total_before_transpile": int(sum(before_counts.values())),
            "gate_count_total_after_transpile": int(sum(after_counts.values())),
        }
        return {
            "transpiled_circuit": transpiled,
            "resource_row": row,
            "two_qubit_gate_count": two_qubit,
        }
    except Exception as exc:  # pragma: no cover - backend-version dependent
        seconds = time.perf_counter() - start
        row = {
            "transpile_success": False,
            "transpile_message": str(exc),
            "transpile_seconds": seconds,
            "n_qubits": int(circuit.num_qubits),
            "depth_before_transpile": int(circuit.depth()),
            "depth_after_transpile": None,
            "operation_counts_before_transpile": before_counts,
            "operation_counts_after_transpile": {},
            "gate_count_total_before_transpile": int(sum(before_counts.values())),
            "gate_count_total_after_transpile": 0,
        }
        return {"transpiled_circuit": None, "resource_row": row, "two_qubit_gate_count": 0}


def _two_qubit_count(circuit: Any) -> int:
    total = 0
    for instruction in circuit.data:
        if len(instruction.qubits) == 2:
            total += 1
    return int(total)


def _export_qasm(circuit: Any) -> str:
    try:
        from qiskit import qasm2  # type: ignore[import-not-found]

        return str(qasm2.dumps(circuit))
    except Exception as exc:
        return "\n".join(
            [
                "// QASM export failed for this Qiskit version/circuit.",
                f"// Reason: {type(exc).__name__}: {exc}",
                str(circuit.draw(output="text")),
            ]
        )


def _write_gate_level_outputs(
    *,
    output_dir: Path,
    resolved: dict[str, Any],
    circuit_summary: dict[str, Any],
    resource_row: dict[str, Any],
    statevector_comparison: dict[str, Any],
    shot_rows: list[dict[str, Any]],
    qasm_text: str,
) -> dict[str, Path]:
    circuit_summary_path = output_dir / "circuit_summary.json"
    resource_summary_path = output_dir / "transpiled_resource_summary.csv"
    statevector_path = output_dir / "statevector_comparison.json"
    shot_path = output_dir / "observable_shot_summary.csv"
    qasm_path = output_dir / "qsvt_circuit.qasm"
    config_path = output_dir / "resolved_config.json"
    write_json(circuit_summary_path, circuit_summary)
    pd.DataFrame([resource_row]).to_csv(resource_summary_path, index=False)
    write_json(statevector_path, statevector_comparison)
    pd.DataFrame(shot_rows).to_csv(shot_path, index=False)
    qasm_path.write_text(qasm_text, encoding="utf-8")
    write_json(config_path, resolved)
    return {
        "circuit_summary": circuit_summary_path,
        "transpiled_resource_summary": resource_summary_path,
        "statevector_comparison": statevector_path,
        "observable_shot_summary": shot_path,
        "qsvt_circuit_qasm": qasm_path,
        "resolved_config": config_path,
    }


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
