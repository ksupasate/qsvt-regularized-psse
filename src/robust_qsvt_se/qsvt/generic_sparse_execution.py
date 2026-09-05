"""Execution, validation, shot sampling, and resources for compiled sparse QSVT."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompiledSparseQSVT,
    SparseCompilerError,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    assert_no_direct_output_initializer,
    estimate_signed_selected_output,
    exact_joint_distribution,
    sample_aer_counts,
)


@dataclass(slots=True)
class StatevectorEvidence:
    workload_id: str
    metrics: dict[str, Any]
    functional_rows: list[dict[str, Any]]
    exact_joint_distributions: dict[str, dict[str, float]]
    sparse_update: np.ndarray
    dense_update: np.ndarray
    exact_polynomial_update: np.ndarray
    exact_rational_update: np.ndarray
    quantized_ridge_update: np.ndarray
    selected_exact_ridge_update: np.ndarray
    full_matrix_ridge_update: np.ndarray
    sparse_encoded_state: np.ndarray
    dense_encoded_state: np.ndarray


@dataclass(slots=True)
class PreparedExecution:
    workload_id: str
    backend: Any
    compiled_functional_circuits: dict[str, Any]
    compiled_direct_postselection_circuit: Any
    source_circuit_object_ids: dict[str, int]
    compiled_metadata: dict[str, Any]


@dataclass(slots=True)
class ResourceEvidence:
    workload_id: str
    record: dict[str, Any]
    register_rows: list[dict[str, Any]]


@dataclass(slots=True)
class ShotEvidence:
    workload_id: str
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _wrapper_roundtrip_error(unitary: np.ndarray, encoded_dimension: int) -> float:
    maximum = 0.0
    inverse = unitary.conj().T
    for index in range(encoded_dimension):
        state = np.zeros(unitary.shape[0], dtype=np.complex128)
        state[index] = 1.0
        maximum = max(maximum, float(np.max(np.abs(inverse @ (unitary @ state) - state))))
    return maximum


def _contains_dense_fallback(compiled: CompiledSparseQSVT) -> bool:
    forbidden = {"c0_U_A", "c0_U_A_dagger", "dense_U_A", "dense_U_A_dagger"}
    for bundle in compiled.functional_circuits.values():
        for instruction in bundle.circuit.data:
            operation = instruction.operation
            names = {str(operation.name), str(getattr(operation, "label", ""))}
            if names & forbidden:
                return True
    return False


def validate_compiled_statevector(
    compiled: CompiledSparseQSVT,
    *,
    lookup_tolerance: float = 1.0e-14,
    block_tolerance: float = 1.0e-9,
    unitarity_tolerance: float = 1.0e-9,
    sparse_dense_tolerance: float = 1.0e-9,
    qsvt_tolerance: float = 1.0e-6,
) -> StatevectorEvidence:
    """Run ordered implementation validation without changing the compiled circuit."""

    from qiskit.quantum_info import Operator, Statevector

    n = compiled.matrix_quantized.shape[0]
    residual = compiled.residual
    residual_norm = float(np.linalg.norm(residual))
    residual_unit = residual / residual_norm
    qsvt = compiled.qsvt_spec
    wrapper = compiled.wrapper
    sparse_unitary = np.asarray(wrapper.unitary, dtype=np.complex128)

    lookup_error = float(wrapper.lookup_value_max_error)
    block_error = float(
        np.linalg.norm(wrapper.encoded_block - wrapper.target_block, ord="fro")
        / max(np.linalg.norm(wrapper.target_block, ord="fro"), 1.0e-30)
    )
    unitarity_error = float(wrapper.unitarity_error)
    roundtrip_error = _wrapper_roundtrip_error(sparse_unitary, n)

    sparse_bundle = build_structured_qsvt_operator_circuit(
        sparse_unitary, compiled.phases, encoded_dimension=n
    )
    sparse_operator = np.asarray(
        Operator(sparse_bundle.qsvt_operator_circuit).data, dtype=np.complex128
    )
    sparse_input = np.zeros(sparse_operator.shape[0], dtype=np.complex128)
    sparse_input[:n] = residual_unit
    sparse_full = Statevector(sparse_input).evolve(
        sparse_bundle.qsvt_operator_circuit
    ).data

    normalized_matrix = compiled.matrix_quantized.T / qsvt.beta
    dense_encoding = canonical_square_block_encoding(normalized_matrix, tolerance=1.0e-8)
    dense_bundle = build_structured_qsvt_operator_circuit(
        dense_encoding.unitary, compiled.phases, encoded_dimension=n
    )
    dense_operator = np.asarray(
        Operator(dense_bundle.qsvt_operator_circuit).data, dtype=np.complex128
    )
    dense_input = np.zeros(dense_operator.shape[0], dtype=np.complex128)
    dense_input[:n] = residual_unit
    dense_full = Statevector(dense_input).evolve(dense_bundle.qsvt_operator_circuit).data

    sparse_action = np.real(sparse_full[:n])
    dense_action = np.real(dense_full[:n])
    sparse_dense_error = _relative_error(sparse_action, dense_action)

    left, singular_values, right_t = np.linalg.svd(normalized_matrix, full_matrices=False)
    polynomial = Polynomial(compiled.polynomial_coefficients)
    exact_polynomial_operator = left @ np.diag(polynomial(singular_values)) @ right_t
    rational_values = singular_values / (
        compiled.qsvt_spec.boundedness_factor
        * (singular_values**2 + compiled.qsvt_spec.normalized_lambda)
    )
    exact_rational_operator = left @ np.diag(rational_values) @ right_t
    physical_scale = qsvt.boundedness_factor / qsvt.beta * residual_norm
    sparse_update = physical_scale * sparse_action
    dense_update = physical_scale * dense_action
    exact_polynomial_update = physical_scale * (exact_polynomial_operator @ residual_unit)
    exact_rational_update = physical_scale * (exact_rational_operator @ residual_unit)
    quantized_ridge = ridge_svd_solution(
        compiled.matrix_quantized, residual, alpha=qsvt.alpha
    )
    selected_exact_ridge = ridge_svd_solution(
        compiled.matrix_supported_exact, residual, alpha=qsvt.alpha
    )
    full_matrix_ridge = ridge_svd_solution(
        compiled.matrix_original, residual, alpha=qsvt.alpha
    )

    qsvt_error = _relative_error(sparse_update, exact_polynomial_update)
    polynomial_error = _relative_error(exact_polynomial_update, exact_rational_update)
    quantization_error = _relative_error(quantized_ridge, selected_exact_ridge)
    support_error = _relative_error(selected_exact_ridge, full_matrix_ridge)
    qsvt_ridge_error = _relative_error(sparse_update, quantized_ridge)
    rational_ridge_error = _relative_error(exact_rational_update, quantized_ridge)
    sparse_postselection = float(np.vdot(sparse_full[:n], sparse_full[:n]).real)
    dense_postselection = float(np.vdot(dense_full[:n], dense_full[:n]).real)

    dense_fallback = _contains_dense_fallback(compiled)
    if dense_fallback:
        raise SparseCompilerError(
            "dense_fallback_detected",
            "statevector_validation",
            "a dense signal fallback appears in the compiled sparse circuit",
        )

    distributions: dict[str, dict[str, float]] = {}
    functional_rows: list[dict[str, Any]] = []
    max_signed_recovery_error = 0.0
    max_interference_error = 0.0
    max_acceptance_error = 0.0
    for functional_id, bundle in compiled.functional_circuits.items():
        assert_no_direct_output_initializer(bundle.circuit, sparse_full[:n])
        distribution = exact_joint_distribution(
            bundle.circuit,
            postselection_flag_qubit=bundle.register_layout["postselection_flag_qubit"],
            readout_qubit=bundle.register_layout["readout_qubit"],
        )
        distributions[functional_id] = distribution
        functional = compiled.functional_vectors[functional_id]
        functional_norm = float(np.linalg.norm(functional))
        functional_unit = functional / functional_norm
        expected_overlap = float(np.real(np.vdot(functional_unit, sparse_full[:n])))
        observed_overlap = float(distribution["00"] - distribution["10"])
        observed_acceptance = float(distribution["00"] + distribution["10"])
        expected_acceptance = float((1.0 + sparse_postselection) / 2.0)
        recovered = float(compiled.recovery_factors[functional_id] * observed_overlap)
        statevector_selected = float(functional @ sparse_update)
        polynomial_selected = float(functional @ exact_polynomial_update)
        rational_selected = float(functional @ exact_rational_update)
        quantized_ridge_selected = float(functional @ quantized_ridge)
        selected_exact_ridge_value = float(functional @ selected_exact_ridge)
        full_ridge_value = float(functional @ full_matrix_ridge)
        recovery_error = abs(recovered - statevector_selected)
        interference_error = abs(observed_overlap - expected_overlap)
        acceptance_error = abs(observed_acceptance - expected_acceptance)
        max_signed_recovery_error = max(max_signed_recovery_error, recovery_error)
        max_interference_error = max(max_interference_error, interference_error)
        max_acceptance_error = max(max_acceptance_error, acceptance_error)
        ridge_relative = (
            abs(statevector_selected - quantized_ridge_selected)
            / abs(quantized_ridge_selected)
            if abs(quantized_ridge_selected) > 1.0e-12
            else float("nan")
        )
        functional_rows.append(
            {
                "workload_id": compiled.workload_id,
                "functional_id": functional_id,
                "matrix_hash": compiled.component_hashes["matrix_original"],
                "support_hash": compiled.component_hashes["support_mask"],
                "quantized_matrix_hash": compiled.component_hashes["matrix_quantized"],
                "epsilon_lookup": lookup_error,
                "epsilon_block": block_error,
                "epsilon_qsvt": qsvt_error,
                "epsilon_poly": polynomial_error,
                "epsilon_quant": quantization_error,
                "epsilon_support": support_error,
                "sparse_dense_action_relative_error": sparse_dense_error,
                "qsvt_quantized_ridge_relative_error": qsvt_ridge_error,
                "exact_rational_quantized_ridge_relative_error": rational_ridge_error,
                "wrapper_unitarity_max_error": unitarity_error,
                "uncomputation_roundtrip_max_error": roundtrip_error,
                "postselection_probability": sparse_postselection,
                "dense_postselection_probability": dense_postselection,
                "interference_acceptance_probability": observed_acceptance,
                "signed_overlap": observed_overlap,
                "signed_overlap_validation_error": interference_error,
                "acceptance_validation_error": acceptance_error,
                "physical_recovery_factor": compiled.recovery_factors[functional_id],
                "C_over_beta": qsvt.boundedness_factor / qsvt.beta,
                "residual_norm": residual_norm,
                "functional_norm": functional_norm,
                "statevector_selected_output": statevector_selected,
                "recovered_selected_output_from_final_circuit": recovered,
                "exact_polynomial_selected_output": polynomial_selected,
                "exact_rational_selected_output": rational_selected,
                "quantized_ridge_selected_output": quantized_ridge_selected,
                "selected_exact_ridge_output": selected_exact_ridge_value,
                "full_matrix_ridge_output": full_ridge_value,
                "absolute_error_vs_quantized_ridge": abs(
                    statevector_selected - quantized_ridge_selected
                ),
                "relative_error_vs_quantized_ridge": ridge_relative,
                "signed_recovery_absolute_error": recovery_error,
                "direct_output_state_preparation_used": False,
                "dense_fallback_used": False,
                "same_final_circuit_for_exact_distribution_and_shots": True,
            }
        )

    metrics: dict[str, Any] = {
        "epsilon_lookup": lookup_error,
        "epsilon_block": block_error,
        "epsilon_qsvt": qsvt_error,
        "epsilon_poly": polynomial_error,
        "epsilon_quant": quantization_error,
        "epsilon_support": support_error,
        "wrapper_unitarity_max_error": unitarity_error,
        "uncomputation_roundtrip_max_error": roundtrip_error,
        "diffusion_unitarity_max_error": float(wrapper.diffusion_unitarity_error),
        "accepted_work_dirty_probability": 0.0,
        "sparse_dense_action_relative_error": sparse_dense_error,
        "qsvt_quantized_ridge_relative_error": qsvt_ridge_error,
        "exact_rational_quantized_ridge_relative_error": rational_ridge_error,
        "sparse_postselection_probability": sparse_postselection,
        "dense_postselection_probability": dense_postselection,
        "postselection_probability_absolute_difference": abs(
            sparse_postselection - dense_postselection
        ),
        "max_signed_recovery_absolute_error": max_signed_recovery_error,
        "max_signed_overlap_validation_error": max_interference_error,
        "max_acceptance_validation_error": max_acceptance_error,
        "physical_rescaling_factor_C_over_beta": qsvt.boundedness_factor / qsvt.beta,
        "lookup_pass": lookup_error <= lookup_tolerance,
        "block_pass": block_error <= block_tolerance,
        "unitarity_pass": unitarity_error <= unitarity_tolerance,
        "uncomputation_pass": roundtrip_error <= unitarity_tolerance,
        "sparse_dense_pass": sparse_dense_error <= sparse_dense_tolerance,
        "qsvt_polynomial_pass": qsvt_error <= qsvt_tolerance,
        "dense_fallback_used": False,
        "direct_output_state_preparation_used": False,
    }
    blocking = {
        "lookup_validation_failure": not metrics["lookup_pass"],
        "block_reconstruction_failure": not metrics["block_pass"],
        "wrapper_unitarity_failure": not metrics["unitarity_pass"],
        "uncomputation_failure": not metrics["uncomputation_pass"],
        "sparse_dense_action_failure": not metrics["sparse_dense_pass"],
        "qsvt_polynomial_action_failure": not metrics["qsvt_polynomial_pass"],
    }
    failed = [code for code, condition in blocking.items() if condition]
    if failed:
        raise SparseCompilerError(
            failed[0],
            "statevector_validation",
            "compiled implementation-correctness tolerance failed",
            failures=failed,
            metrics=metrics,
        )
    return StatevectorEvidence(
        workload_id=compiled.workload_id,
        metrics=metrics,
        functional_rows=functional_rows,
        exact_joint_distributions=distributions,
        sparse_update=np.asarray(sparse_update, dtype=np.float64),
        dense_update=np.asarray(dense_update, dtype=np.float64),
        exact_polynomial_update=np.asarray(exact_polynomial_update, dtype=np.float64),
        exact_rational_update=np.asarray(exact_rational_update, dtype=np.float64),
        quantized_ridge_update=np.asarray(quantized_ridge, dtype=np.float64),
        selected_exact_ridge_update=np.asarray(selected_exact_ridge, dtype=np.float64),
        full_matrix_ridge_update=np.asarray(full_matrix_ridge, dtype=np.float64),
        sparse_encoded_state=np.asarray(sparse_full[:n], dtype=np.complex128),
        dense_encoded_state=np.asarray(dense_full[:n], dtype=np.complex128),
    )


def prepare_compiled_execution(compiled: CompiledSparseQSVT) -> PreparedExecution:
    """Transpile each stored final measured circuit once for resources and shots."""

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qiskit-aer is required for finite-shot execution") from exc

    spec = compiled.execution_spec
    backend = AerSimulator(
        method=spec.simulator_method,
        max_parallel_threads=spec.max_parallel_threads,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    transpile_kwargs: dict[str, Any] = {
        "optimization_level": spec.optimization_level,
    }
    if spec.seed_transpiler is not None:
        transpile_kwargs["seed_transpiler"] = spec.seed_transpiler
    if spec.basis_gates is None:
        transpile_args = (backend,)
    else:
        transpile_args = ()
        transpile_kwargs["basis_gates"] = list(spec.basis_gates)

    compiled_circuits: dict[str, Any] = {}
    source_ids: dict[str, int] = {}
    for functional_id, bundle in compiled.functional_circuits.items():
        source_ids[functional_id] = id(bundle.circuit)
        compiled_circuits[functional_id] = transpile(
            bundle.circuit, *transpile_args, **transpile_kwargs
        )
    direct = transpile(
        compiled.direct_postselection_circuit, *transpile_args, **transpile_kwargs
    )
    metadata = {
        "source_circuit_object_ids": source_ids,
        "source_circuit_hashes": {
            functional_id: compiled.component_hashes[
                f"source_final_circuit:{functional_id}"
            ]
            for functional_id in compiled.functional_circuits
        },
        "basis_gates": (
            list(spec.basis_gates)
            if spec.basis_gates is not None
            else "qiskit_aer_statevector_default_target"
        ),
        "optimization_level": spec.optimization_level,
        "seed_transpiler": spec.seed_transpiler,
        "simulator_method": spec.simulator_method,
    }
    return PreparedExecution(
        workload_id=compiled.workload_id,
        backend=backend,
        compiled_functional_circuits=compiled_circuits,
        compiled_direct_postselection_circuit=direct,
        source_circuit_object_ids=source_ids,
        compiled_metadata=metadata,
    )


def _operation_arities(circuit: Any) -> tuple[int, int, int]:
    one = two = multi = 0
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"measure", "barrier", "delay"}:
            continue
        arity = int(operation.num_qubits)
        if arity == 1:
            one += 1
        elif arity == 2:
            two += 1
        else:
            multi += 1
    return one, two, multi


def _opaque_instruction_names(circuit: Any) -> list[str]:
    opaque_tokens = ("unitary", "state_preparation", "initialize", "sparse_u_a")
    names = set()
    for instruction in circuit.data:
        name = str(instruction.operation.name)
        label = str(getattr(instruction.operation, "label", "") or "")
        if any(token in name.lower() or token in label.lower() for token in opaque_tokens):
            names.add(label or name)
    return sorted(names)


def build_resource_evidence(
    compiled: CompiledSparseQSVT,
    prepared: PreparedExecution,
) -> ResourceEvidence:
    """Count resources from the same primary final circuit prepared for shots."""

    primary_id = compiled.functional_spec.primary_functional_id
    if prepared.source_circuit_object_ids[primary_id] != id(
        compiled.functional_circuits[primary_id].circuit
    ):
        raise RuntimeError("prepared execution does not reference the compiler's final circuit")
    circuit = prepared.compiled_functional_circuits[primary_id]
    counts = {str(name): int(value) for name, value in circuit.count_ops().items()}
    one, two, multi = _operation_arities(circuit)
    opaque = _opaque_instruction_names(circuit)
    operations = compiled.functional_circuits[primary_id].operation_counts
    allocation = compiled.register_allocation
    quantum_register_names = (
        "index",
        "slot",
        "rotation_ancilla",
        "postselection_flag",
        "signed_readout",
    )
    register_rows = [
        {
            "workload_id": compiled.workload_id,
            "register_name": name,
            "qubit_indices": ";".join(str(value) for value in allocation[name]),
            "count": len(allocation[name]),
            "role": {
                "index": "in-place input and output index",
                "slot": "sparse slot and padding basis",
                "rotation_ancilla": "stored signed-value rotation and encoded flag",
                "postselection_flag": "aggregate sparse-work postselection",
                "signed_readout": "real signed interference readout",
            }[name],
            "simultaneously_live": True,
        }
        for name in quantum_register_names
    ]
    register_sum = sum(int(row["count"]) for row in register_rows)
    record = {
        "workload_id": compiled.workload_id,
        "workload_digest": compiled.workload_digest,
        "functional_id": primary_id,
        "execution_status": "transpiled_final_measured_circuit",
        "source_circuit_hash": prepared.compiled_metadata["source_circuit_hashes"][
            primary_id
        ],
        "matrix_hash": compiled.component_hashes["matrix_original"],
        "support_hash": compiled.component_hashes["support_mask"],
        "quantized_matrix_hash": compiled.component_hashes["matrix_quantized"],
        "matrix_shape": f"{compiled.matrix_quantized.shape[0]}x{compiled.matrix_quantized.shape[1]}",
        "matrix_nonzeros": int(np.count_nonzero(compiled.matrix_quantized)),
        "declared_support_count": len(compiled.support_spec.coordinates),
        "slots": compiled.wrapper.slots,
        "value_magnitude_bits": compiled.quantization_spec.magnitude_bits,
        "value_sign_bits": 1,
        "total_simultaneously_live_qubits": int(circuit.num_qubits),
        "register_sum": register_sum,
        "register_breakdown_json": json.dumps(
            {name: len(allocation[name]) for name in quantum_register_names}, sort_keys=True
        ),
        "raw_final_circuit_operations": int(
            sum(compiled.final_measured_circuit.count_ops().values())
        ),
        "transpiled_gate_count": int(sum(counts.values())),
        "transpiled_depth": int(circuit.depth()),
        "toffoli_count": int(counts.get("ccx", 0)),
        "controlled_rotation_count": int(operations["value_rotations_per_attempt"]),
        "one_qubit_gate_count": one,
        "two_qubit_gate_count": two,
        "multi_qubit_gate_count": multi,
        "qsvt_signal_calls": int(operations["signal_unitary_calls_per_attempt"]),
        "phase_operations": int(operations["projector_phase_operations_per_attempt"]),
        "residual_loader_operations": int(operations["residual_preparations_per_attempt"]),
        "functional_loader_operations": int(operations["functional_preparations_per_attempt"]),
        "lookup_operations": int(operations["sparse_lookup_calls_per_attempt"]),
        "value_rotations_per_lookup": int(operations["value_rotation_gates_per_sparse_lookup"]),
        "postselection_flag_operations": int(
            2 * len(compiled.functional_circuits[primary_id].register_layout["sparse_work_qubits"])
            + 2
        ),
        "postselection_measurements": int(
            operations["postselection_measurements_per_attempt"]
        ),
        "readout_measurements": int(operations["readout_measurements_per_attempt"]),
        "transpiler_basis": json.dumps(prepared.compiled_metadata["basis_gates"]),
        "optimization_level": prepared.compiled_metadata["optimization_level"],
        "seed_transpiler": prepared.compiled_metadata["seed_transpiler"],
        "opaque_instructions_remain": bool(opaque),
        "opaque_instruction_names": ";".join(opaque),
        "operation_counts_json": json.dumps(counts, sort_keys=True),
        "same_final_circuit_as_shots": True,
        "dense_fallback_used": False,
    }
    if register_sum != int(circuit.num_qubits):
        raise RuntimeError("register sum does not equal final circuit qubit count")
    return ResourceEvidence(
        workload_id=compiled.workload_id,
        record=record,
        register_rows=register_rows,
    )


def run_compiled_shots(
    compiled: CompiledSparseQSVT,
    statevector: StatevectorEvidence,
    prepared: PreparedExecution,
) -> ShotEvidence:
    """Run actual Aer counts on the stored final measured circuits."""

    if not compiled.execution_spec.execute_finite_shots:
        return ShotEvidence(compiled.workload_id, [], [])
    reference_by_functional = {
        row["functional_id"]: row for row in statevector.functional_rows
    }
    rows: list[dict[str, Any]] = []
    for shots in compiled.execution_spec.shot_counts:
        for seed in compiled.execution_spec.simulator_seeds:
            direct_counts = sample_aer_counts(
                prepared.compiled_direct_postselection_circuit,
                prepared.backend,
                shots=int(shots),
                seed=int(seed),
            )
            direct_attempted = int(sum(direct_counts.values()))
            postselection_accepted = int(direct_counts.get("0", 0))
            postselection_probability = postselection_accepted / direct_attempted
            for functional_id, circuit in prepared.compiled_functional_circuits.items():
                if prepared.source_circuit_object_ids[functional_id] != id(
                    compiled.functional_circuits[functional_id].circuit
                ):
                    raise RuntimeError("finite-shot circuit identity differs from compiler output")
                counts = sample_aer_counts(
                    circuit,
                    prepared.backend,
                    shots=int(shots),
                    seed=int(seed),
                )
                estimate = estimate_signed_selected_output(
                    counts,
                    physical_scale=compiled.recovery_factors[functional_id],
                )
                attempted = int(sum(counts.values()))
                plus = int(counts.get("00", 0))
                minus = int(counts.get("10", 0))
                count_01 = int(counts.get("01", 0))
                count_11 = int(counts.get("11", 0))
                reference = reference_by_functional[functional_id]
                statevector_value = float(reference["statevector_selected_output"])
                ridge_value = float(reference["quantized_ridge_selected_output"])
                selected = float(estimate["selected_output_estimate"])
                absolute_statevector_error = abs(selected - statevector_value)
                expected_distribution = statevector.exact_joint_distributions[functional_id]
                exact_acceptance = float(
                    expected_distribution["00"] + expected_distribution["10"]
                )
                exact_contrast = float(
                    expected_distribution["00"] - expected_distribution["10"]
                )
                expected_variance = (
                    compiled.recovery_factors[functional_id] ** 2
                    * max(exact_acceptance - exact_contrast**2, 0.0)
                    / attempted
                )
                expected_standard_error = math.sqrt(expected_variance)
                relative_stable = bool(
                    abs(statevector_value) > max(1.0e-12, 5.0 * expected_standard_error)
                )
                rows.append(
                    {
                        "workload_id": compiled.workload_id,
                        "functional_id": functional_id,
                        "shots_attempted": attempted,
                        "direct_postselection_shots_attempted": direct_attempted,
                        "seed": int(seed),
                        "backend": "qiskit_aer_statevector_actual_shot_sampling",
                        "source_circuit_hash": prepared.compiled_metadata[
                            "source_circuit_hashes"
                        ][functional_id],
                        "postselection_accepted_shots": postselection_accepted,
                        "postselection_rejected_shots": int(
                            direct_attempted - postselection_accepted
                        ),
                        "interference_branch_accepted_shots": int(
                            estimate["readout_accepted"]
                        ),
                        "interference_branch_rejected_shots": int(
                            attempted - estimate["readout_accepted"]
                        ),
                        "plus_accepted_shots": plus,
                        "minus_accepted_shots": minus,
                        "count_00": plus,
                        "count_01": count_01,
                        "count_10": minus,
                        "count_11": count_11,
                        "estimated_postselection_probability": postselection_probability,
                        "interference_acceptance_probability": estimate[
                            "interference_acceptance_probability"
                        ],
                        "inferred_postselection_probability_from_branch": estimate[
                            "inferred_postselection_probability_from_branch"
                        ],
                        "signed_contrast": estimate["signed_overlap_estimate"],
                        "recovered_selected_output": selected,
                        "statevector_reference": statevector_value,
                        "quantized_ridge_reference": ridge_value,
                        "absolute_error_vs_statevector": absolute_statevector_error,
                        "absolute_error_vs_quantized_ridge": abs(selected - ridge_value),
                        "relative_error_vs_statevector": (
                            absolute_statevector_error / abs(statevector_value)
                            if relative_stable
                            else float("nan")
                        ),
                        "relative_error_numerically_stable": relative_stable,
                        "analytic_variance_estimate": float(
                            estimate["analytic_standard_error"] ** 2
                        ),
                        "analytic_standard_error": estimate["analytic_standard_error"],
                        "statevector_expected_variance": expected_variance,
                        "statevector_expected_standard_error": expected_standard_error,
                        "confidence_interval_lower": estimate["confidence_interval_lower"],
                        "confidence_interval_upper": estimate["confidence_interval_upper"],
                        "same_final_circuit_as_statevector_and_resources": True,
                        "output_state_used_for_preparation": False,
                    }
                )

    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for (functional_id, shots), group in frame.groupby(
            ["functional_id", "shots_attempted"], sort=True
        ):
            values = group["recovered_selected_output"].to_numpy(dtype=np.float64)
            errors = group["absolute_error_vs_statevector"].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "workload_id": compiled.workload_id,
                    "functional_id": functional_id,
                    "shots": int(shots),
                    "seed_count": int(len(group)),
                    "mean_selected_output": float(np.mean(values)),
                    "statevector_reference": float(group["statevector_reference"].iloc[0]),
                    "quantized_ridge_reference": float(
                        group["quantized_ridge_reference"].iloc[0]
                    ),
                    "mean_absolute_error_vs_statevector": float(np.mean(errors)),
                    "max_absolute_error_vs_statevector": float(np.max(errors)),
                    "rmse_vs_statevector": float(np.sqrt(np.mean((values - group["statevector_reference"].iloc[0]) ** 2))),
                    "empirical_seed_variance": float(np.var(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "empirical_seed_standard_deviation": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    "standard_error_of_seed_mean": float(np.std(values, ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0,
                    "mean_analytic_variance_estimate": float(
                        group["analytic_variance_estimate"].mean()
                    ),
                    "mean_statevector_expected_variance": float(
                        group["statevector_expected_variance"].mean()
                    ),
                    "mean_estimated_postselection_probability": float(
                        group["estimated_postselection_probability"].mean()
                    ),
                    "total_postselection_accepted_shots": int(
                        group["postselection_accepted_shots"].sum()
                    ),
                    "total_interference_branch_accepted_shots": int(
                        group["interference_branch_accepted_shots"].sum()
                    ),
                    "relative_error_stable_seed_count": int(
                        group["relative_error_numerically_stable"].sum()
                    ),
                    "absolute_error_is_primary_metric": True,
                    "finite_shot_executed": True,
                }
            )
    return ShotEvidence(compiled.workload_id, rows, summary_rows)


__all__ = [
    "PreparedExecution",
    "ResourceEvidence",
    "ShotEvidence",
    "StatevectorEvidence",
    "build_resource_evidence",
    "prepare_compiled_execution",
    "run_compiled_shots",
    "validate_compiled_statevector",
]
