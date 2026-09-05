"""Phase 10 WP A: complete 8x8 sparse block-encoding wrapper (blocker fixed).

Phase 9 validated the 8x8 sparse-access *lookup* oracle but reported the
complete block-encoding *wrapper* as blocked: the reused Konig edge-coloring
augmenting path did not terminate on the 8x8 sparsified nonzero pattern.  This
module resolves that blocker with the deterministic slot-assignment algorithm
in :mod:`robust_qsvt_se.qsvt.bipartite_slot_assignment` and compiles the
complete wrapper at 8x8.

The blocker had two layers, both documented in ``edge_coloring_validation.json``:

1. *Feasibility*: the Phase 9 pattern (row degrees all 2 after row
   thresholding) has maximum **column** degree 3, so the requested 2-slot
   coloring cannot exist; Konig's theorem guarantees exactly ``max_degree = 3``
   slots.  The old routine looped instead of detecting this.
2. *Implementation*: the old alternating-path recoloring could cycle forever
   because its color bookkeeping desynchronizes.  The replacement peels perfect
   matchings from a slots-regular bipartite multigraph under an explicit visit
   budget, so termination is structural.

The compiled wrapper generalizes the validated 4x4 demo to ``s`` slots:

    uniform slot diffusion V_s -> per-slot multiplexed value rotations (O_val
    role, zero-valued padding pairs rotate to the orthogonal ancilla state) ->
    slot-controlled in-place column permutations (O_col role) -> V_s^dagger,

so ``<0|_a <0|_k <i| U_A |0>_a |0>_k |j> = A_q^T[i, j] / (s mu)`` for the
sparsified, sign-magnitude-quantized selected block ``A_q`` (encoded transpose,
matching the ``A = H^T / beta`` QSVT orientation).  Per-slot value keying makes
double counting impossible even when padding pairs sit on real-entry positions.

A QSVT integration check applies matched-Ridge bounded-target phase sequences
(degree 31 first) directly to the wrapper unitary and compares the transformed
block and postselected update against the dense-dilation QSVT action, the exact
singular-value transform, and Ridge at the same alpha on the same quantized
block.  This is a tiny-instance completeness demonstration on a simulator; it
is not a scalable sparse-oracle compilation, not an IEEE-scale block encoding,
and not a hardware run.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.phase10_common import (
    assert_safe,
    json_ready,
    write_phase10_manifest,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    SlotAssignment,
    assign_slot_permutations,
    minimum_slot_count,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.sparse_block_encoding_wrapper import quantize_sign_magnitude
from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import sparsify_block
from robust_qsvt_se.utils.io import ensure_directory, write_json

OUTPUT_DIR = Path("outputs/phase10_sparse_wrapper_8x8_complete")
BLOCK_SIZE = 8
KEEP_PER_ROW = 2  # Phase 9 row-thresholding convention
VALUE_PRECISION_BITS = (4, 6, 8)
PRIMARY_PRECISION_BITS = 6  # 4x4 wrapper convention (6 magnitude + 1 sign bit)
VALIDATION_TOLERANCE = 1.0e-9
QSVT_DEGREES = (31, 39, 45)
QSVT_PASS_RELATIVE_TOLERANCE = 0.05

CLAIM = (
    "Complete 8x8 sparse block-encoding wrapper compiled and statevector-validated on the "
    "Phase 9 sparsified, quantized IEEE-14-derived selected block, with the Phase 9 "
    "edge-coloring blocker fixed by a deterministic terminating slot-assignment algorithm "
    "(the pattern needs 3 slots, not 2). QSVT phase-sequence integration is compared against "
    "dense-dilation QSVT, the exact singular-value transform, and matched Ridge at the same "
    "alpha on the same quantized block. Tiny-instance completeness evidence on a classical "
    "simulator; NOT an IEEE-scale sparse block encoding, NOT a scalable oracle synthesis, "
    "and NOT a hardware run."
)


@dataclass(frozen=True, slots=True)
class QuantizedSparseBlock:
    original: np.ndarray
    sparsified: np.ndarray
    quantized: np.ndarray
    mu: float
    magnitude_bits: int
    nnz: int
    quantization_step: float
    max_quantization_error: float
    sparsification_fro_error: float


@dataclass(slots=True)
class CompleteWrapperResult:
    circuit: Any
    unitary: np.ndarray
    encoded_block: np.ndarray
    target_block: np.ndarray
    assignment: SlotAssignment
    slots: int
    normalization_factor: float
    top_left_reconstruction_error: float
    unitarity_error: float
    statevector_max_error: float
    diffusion_unitarity_error: float
    lookup_value_max_error: float
    qubits: int
    gate_count: int
    depth: int
    transpiled_gate_count: int | None
    transpiled_depth: int | None
    transpiled_cx_count: int | None
    transpile_failure: str | None


def build_quantized_sparse_block(
    matrix: np.ndarray, *, magnitude_bits: int
) -> QuantizedSparseBlock:
    """Phase 9 convention: row-threshold to KEEP_PER_ROW, then quantize."""

    values = np.asarray(matrix, dtype=np.float64)
    sparsified = sparsify_block(values, keep_per_row=KEEP_PER_ROW)
    quantized, mu = quantize_sign_magnitude(sparsified, magnitude_bits=magnitude_bits)
    step = mu / ((1 << magnitude_bits) - 1)
    return QuantizedSparseBlock(
        original=values,
        sparsified=sparsified,
        quantized=quantized,
        mu=mu,
        magnitude_bits=int(magnitude_bits),
        nnz=int(np.count_nonzero(quantized)),
        quantization_step=step,
        max_quantization_error=float(np.max(np.abs(quantized - sparsified))),
        sparsification_fro_error=float(
            np.linalg.norm(sparsified - values) / max(np.linalg.norm(values), 1e-30)
        ),
    )


def _uniform_slot_diffusion(slots: int, slot_dimension: int) -> np.ndarray:
    """Deterministic real orthogonal V with V|0> uniform over the first ``slots`` states."""

    target = np.zeros(slot_dimension, dtype=np.float64)
    target[:slots] = 1.0 / math.sqrt(slots)
    w = target - np.eye(slot_dimension)[:, 0]
    norm2 = float(w @ w)
    if norm2 <= 1.0e-30:
        return np.eye(slot_dimension)
    return np.eye(slot_dimension) - 2.0 * np.outer(w, w) / norm2


def slot_values_from_assignment(
    matrix: np.ndarray, mu: float, assignment: SlotAssignment
) -> np.ndarray:
    """Per-slot normalized values v[k, j]; padding pairs are exactly zero."""

    n = matrix.shape[0]
    values = np.zeros((assignment.slots, n), dtype=np.float64)
    for k, (pi, mask) in enumerate(
        zip(assignment.permutations, assignment.real_edge_mask, strict=True)
    ):
        for j in range(n):
            if mask[j]:
                values[k, j] = float(matrix[pi[j], j] / mu)
    return values


def build_complete_wrapper_circuit(
    matrix: np.ndarray, mu: float, assignment: SlotAssignment
) -> tuple[Any, np.ndarray]:
    """Compile V_s -> per-slot value rotations -> slot-controlled perms -> V_s^dagger."""

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, UnitaryGate

    n = matrix.shape[0]
    index_qubits = int(math.log2(n))
    slots = assignment.slots
    slot_qubits = max(1, math.ceil(math.log2(slots)))
    slot_dimension = 1 << slot_qubits
    diffusion = _uniform_slot_diffusion(slots, slot_dimension)
    slot_values = slot_values_from_assignment(matrix, mu, assignment)

    total_qubits = index_qubits + slot_qubits + 1
    circuit = QuantumCircuit(total_qubits, name="complete_sparse_BE_wrapper_8x8")
    index_regs = list(range(index_qubits))
    slot_regs = list(range(index_qubits, index_qubits + slot_qubits))
    ancilla = index_qubits + slot_qubits

    circuit.append(UnitaryGate(diffusion, label="V_slot"), slot_regs)

    # Multiplexed value-rotation oracle keyed on (slot, column):
    # <0|Ry(2 arccos v)|0> = v; padding pairs use v = 0 exactly.
    for k in range(slots):
        for j in range(n):
            v = float(np.clip(slot_values[k, j], -1.0, 1.0))
            theta = 2.0 * math.acos(v)
            controls = index_regs + slot_regs
            ctrl_state = j + (k << index_qubits)
            gate = RYGate(theta).control(len(controls), ctrl_state=ctrl_state)
            circuit.append(gate, [*controls, ancilla])

    # Slot-controlled in-place column permutations (compiled O_col wrapper role).
    for k, pi in enumerate(assignment.permutations):
        perm_matrix = np.zeros((n, n))
        perm_matrix[list(pi), np.arange(n)] = 1.0
        gate = UnitaryGate(perm_matrix, label=f"P_slot{k}").control(slot_qubits, ctrl_state=k)
        circuit.append(gate, slot_regs + index_regs)

    circuit.append(UnitaryGate(diffusion.T, label="V_slot_dag"), slot_regs)
    return circuit, diffusion


def validate_complete_wrapper(
    block: QuantizedSparseBlock,
    *,
    encode_transpose: bool = True,
    transpile_circuit: bool = True,
) -> CompleteWrapperResult:
    from qiskit import transpile
    from qiskit.quantum_info import Operator, Statevector

    matrix = block.quantized.T if encode_transpose else block.quantized
    n = matrix.shape[0]
    pattern = np.abs(matrix) > 0.0
    assignment = assign_slot_permutations(pattern)
    slots = assignment.slots
    normalization = slots * block.mu
    target = matrix / normalization

    circuit, diffusion = build_complete_wrapper_circuit(matrix, block.mu, assignment)
    unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    dim = unitary.shape[0]
    unitarity_error = float(np.max(np.abs(unitary.conj().T @ unitary - np.eye(dim))))
    encoded = np.real(unitary[:n, :n])
    reconstruction_error = float(np.max(np.abs(encoded - target)))
    diffusion_unitarity_error = float(
        np.max(np.abs(diffusion.T @ diffusion - np.eye(diffusion.shape[0])))
    )

    slot_values = slot_values_from_assignment(matrix, block.mu, assignment)
    lookup_value_max_error = 0.0
    for k, (pi, mask) in enumerate(
        zip(assignment.permutations, assignment.real_edge_mask, strict=True)
    ):
        for j in range(n):
            expected = matrix[pi[j], j] / block.mu if mask[j] else 0.0
            lookup_value_max_error = max(
                lookup_value_max_error, abs(float(slot_values[k, j]) - float(expected))
            )

    statevector_max_error = 0.0
    for j in range(n):
        state = np.zeros(dim, dtype=np.complex128)
        state[j] = 1.0
        evolved = Statevector(state).evolve(circuit).data
        column_error = float(np.max(np.abs(np.real(evolved[:n]) - target[:, j])))
        statevector_max_error = max(statevector_max_error, column_error)

    transpiled_gate_count = transpiled_depth = transpiled_cx = None
    transpile_failure: str | None = None
    if transpile_circuit:
        try:
            transpiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
            counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
            transpiled_gate_count = int(sum(counts.values()))
            transpiled_depth = int(transpiled.depth())
            transpiled_cx = int(counts.get("cx", 0))
        except Exception as exc:  # honest failure recording, never silent
            transpile_failure = f"{type(exc).__name__}: {exc}"

    return CompleteWrapperResult(
        circuit=circuit,
        unitary=unitary,
        encoded_block=encoded,
        target_block=target,
        assignment=assignment,
        slots=slots,
        normalization_factor=normalization,
        top_left_reconstruction_error=reconstruction_error,
        unitarity_error=unitarity_error,
        statevector_max_error=statevector_max_error,
        diffusion_unitarity_error=diffusion_unitarity_error,
        lookup_value_max_error=lookup_value_max_error,
        qubits=int(circuit.num_qubits),
        gate_count=int(sum(circuit.count_ops().values())),
        depth=int(circuit.depth()),
        transpiled_gate_count=transpiled_gate_count,
        transpiled_depth=transpiled_depth,
        transpiled_cx_count=transpiled_cx,
        transpile_failure=transpile_failure,
    )


def qsvt_integration_comparison(
    block: QuantizedSparseBlock,
    wrapper: CompleteWrapperResult,
    r_block: np.ndarray,
    *,
    phase_cache_dir: Path,
    degrees: tuple[int, ...] = QSVT_DEGREES,
) -> list[dict[str, Any]]:
    """Apply matched-Ridge phases to the sparse wrapper and the dense dilation.

    Both encode ``A = H_q^T / (s mu)`` with the same normalization, so the same
    phase sequence must produce the same transformed block.  Ridge at the same
    alpha on the same quantized block is the classical reference.
    """

    from qiskit.quantum_info import Operator, Statevector

    H_q = block.quantized
    n = H_q.shape[0]
    r = np.asarray(r_block, dtype=np.float64)
    beta_be = wrapper.normalization_factor
    sigma = np.linalg.svd(H_q, compute_uv=False)
    sigma_pos = sigma[sigma > 1.0e-10]
    alpha = 4.0 * float(sigma_pos.min()) ** 2
    ridge_update = ridge_svd_solution(H_q, r, alpha=alpha)

    A = H_q.T / beta_be
    s_values = np.linalg.svd(A, compute_uv=False)
    s_pos = s_values[s_values > 1.0e-12]
    domain_min = float(np.clip(0.9 * s_pos.min(), 1.0e-4, 0.999))
    dense = canonical_square_block_encoding(A, tolerance=1.0e-8)

    base: dict[str, Any] = {
        "alpha": alpha,
        "beta_effective": beta_be,
        "lambda_alpha_over_beta2": alpha / beta_be**2,
        "sigma_min_positive_quantized": float(sigma_pos.min()),
        "sigma_max_quantized": float(sigma.max()),
        "rank_quantized": int(sigma_pos.size),
        "domain_min": domain_min,
    }
    rows: list[dict[str, Any]] = []
    for degree in degrees:
        record = dict(base)
        record["degree"] = int(degree)
        target = fit_codesigned_bounded_polynomial(
            beta=beta_be,
            alpha=alpha,
            domain_min=domain_min,
            domain_max=1.0,
            degree=degree,
            margin=1.05,
        )
        record["target_fit_error"] = target.fit_max_abs_error
        record["bound_C"] = target.bound_C
        record["physical_recovery_factor_C_over_beta"] = target.physical_recovery_factor
        try:
            validate_qsvt_polynomial(
                np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
            )
        except Exception as exc:
            record.update({"status": "bounded_polynomial_invalid", "failure_reason": str(exc)})
            rows.append(record)
            continue
        try:
            cached = synthesize_pennylane_phases_cached(
                np.asarray(target.coefficients),
                angle_solver="iterative",
                cache_dir=phase_cache_dir,
                cache_metadata={"wrapper": "phase10_sparse_8x8", "degree": degree, "alpha": alpha},
            )
            phases = np.asarray(cached.phases, dtype=np.float64)
        except Exception as exc:
            record.update({"status": "phase_synthesis_failed", "failure_reason": str(exc)})
            rows.append(record)
            continue
        record["phase_count"] = int(phases.size)

        sparse_bundle = build_structured_qsvt_operator_circuit(
            wrapper.unitary, phases, encoded_dimension=n
        )
        sparse_operator = np.asarray(
            Operator(sparse_bundle.qsvt_operator_circuit).data, dtype=np.complex128
        )
        dense_bundle = build_structured_qsvt_operator_circuit(
            dense.unitary, phases, encoded_dimension=n
        )
        dense_operator = np.asarray(
            Operator(dense_bundle.qsvt_operator_circuit).data, dtype=np.complex128
        )
        sparse_block_action = np.real(sparse_operator[:n, :n])
        dense_block_action = np.real(dense_operator[:n, :n])

        U_A, S_A, Vt_A = np.linalg.svd(A)
        exact = U_A @ np.diag(target.polynomial(S_A)) @ Vt_A
        record["sparse_vs_exact_svt_error"] = float(np.max(np.abs(sparse_block_action - exact)))
        record["dense_vs_exact_svt_error"] = float(np.max(np.abs(dense_block_action - exact)))
        record["sparse_vs_dense_action_error"] = float(
            np.max(np.abs(sparse_block_action - dense_block_action))
        )

        def _postselected_update(
            operator: np.ndarray, recovery_factor: float = target.physical_recovery_factor
        ) -> tuple[np.ndarray, float]:
            padded = np.zeros(operator.shape[0], dtype=np.complex128)
            padded[:n] = r / np.linalg.norm(r)
            evolved = operator @ padded
            encoded = evolved[:n]
            p_succ = float(np.vdot(encoded, encoded).real)
            psi = encoded / math.sqrt(p_succ) if p_succ > 1e-15 else encoded
            recovery = recovery_factor * float(np.linalg.norm(r)) * math.sqrt(p_succ)
            return recovery * np.real(psi), p_succ

        # Statevector.evolve on the compiled circuit cross-checks the Operator path once.
        padded_state = np.zeros(sparse_operator.shape[0], dtype=np.complex128)
        padded_state[:n] = r / np.linalg.norm(r)
        circuit_evolved = Statevector(padded_state).evolve(sparse_bundle.qsvt_operator_circuit).data
        record["sparse_operator_vs_statevector_error"] = float(
            np.max(np.abs(circuit_evolved - sparse_operator @ padded_state))
        )

        sparse_update, sparse_p = _postselected_update(sparse_operator)
        dense_update, dense_p = _postselected_update(dense_operator)
        rel_sparse = float(
            np.linalg.norm(sparse_update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
        )
        rel_dense = float(
            np.linalg.norm(dense_update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
        )
        record.update(
            {
                "sparse_postselection_probability": sparse_p,
                "dense_postselection_probability": dense_p,
                "sparse_update_relative_error_vs_ridge": rel_sparse,
                "dense_update_relative_error_vs_ridge": rel_dense,
                "sparse_vs_dense_update_error": float(np.linalg.norm(sparse_update - dense_update)),
                "selected_output_e1_sparse_qsvt": float(sparse_update[0]),
                "selected_output_e1_ridge": float(ridge_update[0]),
                "status": (
                    "statevector_validated"
                    if rel_sparse <= QSVT_PASS_RELATIVE_TOLERANCE
                    else "degree_limited"
                ),
            }
        )
        rows.append(record)
        if record["status"] == "statevector_validated":
            break
    return rows


def _build_block(seed: int) -> dict[str, Any]:
    system, matrix_source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(seed),
        }
    )
    H_block, r_block, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=BLOCK_SIZE,
        col_count=BLOCK_SIZE,
        policy="largest_row_col_norms",
    )
    return {
        "matrix_source": matrix_source,
        "H_block": H_block,
        "r_block": r_block,
        "selected_rows": rows,
        "selected_cols": cols,
    }


def _edge_coloring_report(block: QuantizedSparseBlock, wrapper: CompleteWrapperResult) -> dict:
    pattern = np.abs(block.quantized.T) > 0.0
    required = minimum_slot_count(pattern)
    infeasible_reason = None
    try:
        assign_slot_permutations(pattern, slots=2)
        two_slot_status = "feasible"
    except ValueError as exc:
        two_slot_status = "infeasible"
        infeasible_reason = str(exc)
    validation = validate_slot_assignment(pattern, wrapper.assignment)
    return {
        "pattern": {
            "encoded_orientation": "transpose (A = H_q^T / (s mu))",
            "row_degrees": [int(v) for v in pattern.sum(axis=1)],
            "col_degrees": [int(v) for v in pattern.sum(axis=0)],
            "nnz": int(pattern.sum()),
            "konig_minimum_slots": required,
        },
        "phase9_blocker_resolution": {
            "phase9_request": "2-slot Konig coloring via the reused augmenting-path routine",
            "layer_1_feasibility": {
                "two_slot_coloring_status": two_slot_status,
                "diagnosis": infeasible_reason,
                "note": (
                    "the Phase 9 pattern has maximum column degree 3 after row-only "
                    "thresholding, so the requested 2-slot coloring cannot exist; the old "
                    "routine looped instead of rejecting the request"
                ),
            },
            "layer_2_implementation": {
                "old_routine": "alternating-path recoloring with symmetric-difference "
                "bookkeeping (can cycle forever; recorded non-terminating in Phase 9)",
                "replacement": "regularize to a slots-regular bipartite multigraph and peel "
                "perfect matchings with Kuhn augmenting paths under an explicit visit budget",
                "termination": "structural (visit budget; regular multigraphs always admit "
                "perfect matchings)",
            },
        },
        "assignment": wrapper.assignment.to_metadata(),
        "validation": validation,
    }


def run_phase10_sparse_wrapper_complete(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "seed": 123,
        "value_precision_bits": list(VALUE_PRECISION_BITS),
        "primary_precision_bits": PRIMARY_PRECISION_BITS,
        "qsvt_degrees": list(QSVT_DEGREES),
        "transpile": True,
        "run_qsvt_integration": True,
        "command": "scripts/run_phase10_sparse_wrapper_8x8_complete.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")
    precisions = tuple(int(b) for b in resolved["value_precision_bits"])
    primary_bits = int(resolved["primary_precision_bits"])
    if primary_bits not in precisions:
        precisions = tuple(sorted({*precisions, primary_bits}))

    source = _build_block(int(resolved["seed"]))
    validation_rows: list[dict[str, Any]] = []
    wrappers: dict[int, CompleteWrapperResult] = {}
    blocks: dict[int, QuantizedSparseBlock] = {}
    for bits in precisions:
        block = build_quantized_sparse_block(source["H_block"], magnitude_bits=bits)
        started = time.perf_counter()
        wrapper = validate_complete_wrapper(
            block, encode_transpose=True, transpile_circuit=bool(resolved["transpile"])
        )
        elapsed = time.perf_counter() - started
        blocks[bits] = block
        wrappers[bits] = wrapper
        passed = (
            wrapper.unitarity_error <= VALIDATION_TOLERANCE
            and wrapper.top_left_reconstruction_error <= VALIDATION_TOLERANCE
            and wrapper.statevector_max_error <= VALIDATION_TOLERANCE
        )
        validation_rows.append(
            {
                "value_precision_bits": bits,
                "is_primary": bits == primary_bits,
                "block_shape": f"{BLOCK_SIZE}x{BLOCK_SIZE}",
                "nnz": block.nnz,
                "slots": wrapper.slots,
                "normalization_factor_s_mu": wrapper.normalization_factor,
                "mu_full_scale": block.mu,
                "quantization_step": block.quantization_step,
                "max_quantization_error": block.max_quantization_error,
                "sparsification_relative_fro_error": block.sparsification_fro_error,
                "top_left_reconstruction_error": wrapper.top_left_reconstruction_error,
                "unitarity_error": wrapper.unitarity_error,
                "statevector_max_error": wrapper.statevector_max_error,
                "diffusion_unitarity_error": wrapper.diffusion_unitarity_error,
                "lookup_value_max_error": wrapper.lookup_value_max_error,
                "qubits": wrapper.qubits,
                "raw_gate_count": wrapper.gate_count,
                "raw_depth": wrapper.depth,
                "transpiled_gate_count": wrapper.transpiled_gate_count,
                "transpiled_depth": wrapper.transpiled_depth,
                "transpiled_cx_count": wrapper.transpiled_cx_count,
                "transpile_failure": wrapper.transpile_failure,
                "compile_validate_seconds": elapsed,
                "status": "statevector_validated" if passed else "compiled_above_tolerance",
            }
        )

    primary_block = blocks[primary_bits]
    primary_wrapper = wrappers[primary_bits]
    qsvt_rows: list[dict[str, Any]] = []
    if bool(resolved["run_qsvt_integration"]):
        qsvt_rows = qsvt_integration_comparison(
            primary_block,
            primary_wrapper,
            source["r_block"],
            phase_cache_dir=phase_cache_dir,
            degrees=tuple(int(d) for d in resolved["qsvt_degrees"]),
        )

    all_validated = all(row["status"] == "statevector_validated" for row in validation_rows)
    qsvt_validated = any(row.get("status") == "statevector_validated" for row in qsvt_rows)
    overall_status = (
        "wrapper_and_qsvt_statevector_validated"
        if (all_validated and qsvt_validated)
        else (
            "wrapper_statevector_validated_qsvt_"
            + (qsvt_rows[-1].get("status", "not_attempted") if qsvt_rows else "not_attempted")
            if all_validated
            else "wrapper_validation_failed"
        )
    )

    block_source = (
        f"{source['matrix_source']} {BLOCK_SIZE}x{BLOCK_SIZE} selected block (rows "
        f"{' '.join(str(int(v)) for v in source['selected_rows'])}; cols "
        f"{' '.join(str(int(v)) for v in source['selected_cols'])}), row-thresholded to "
        f"<= {KEEP_PER_ROW} nonzeros per row (Phase 9 convention), sign-magnitude quantized"
    )

    validation_csv = output_dir / "sparse_wrapper_8x8_validation.csv"
    qsvt_csv = output_dir / "sparse_wrapper_8x8_qsvt_validation.csv"
    circuit_json = output_dir / "sparse_wrapper_8x8_circuit_metadata.json"
    reconstruction_json = output_dir / "sparse_wrapper_8x8_block_reconstruction.json"
    coloring_json = output_dir / "edge_coloring_validation.json"
    readme_md = output_dir / "README.md"

    pd.DataFrame(validation_rows).to_csv(validation_csv, index=False)
    pd.DataFrame(qsvt_rows).to_csv(qsvt_csv, index=False)
    write_json(
        circuit_json,
        json_ready(
            {
                "overall_status": overall_status,
                "block_source": block_source,
                "primary_precision_bits": primary_bits,
                "register_layout": {
                    "index_qubits": 3,
                    "slot_qubits": 2,
                    "rotation_ancilla_qubits": 1,
                    "total_qubits": primary_wrapper.qubits,
                    "encoded_subspace": "ancilla=0, slot=0, index=j (first 8 basis states)",
                },
                "slots": primary_wrapper.slots,
                "slot_diffusion": (
                    "deterministic Householder unitary V_s with V_s|0> uniform over the "
                    f"{primary_wrapper.slots} used slots; V_s^dagger uncomputes"
                ),
                "value_oracle": (
                    "multiplexed Ry(2 arccos v) keyed on (slot, column); padding pairs "
                    "rotate with v=0 exactly, so no entry is double counted"
                ),
                "column_oracle": "slot-controlled in-place column permutations",
                "normalization_factor_s_mu": primary_wrapper.normalization_factor,
                "raw_gate_count": primary_wrapper.gate_count,
                "raw_depth": primary_wrapper.depth,
                "transpiled_gate_count": primary_wrapper.transpiled_gate_count,
                "transpiled_depth": primary_wrapper.transpiled_depth,
                "transpiled_cx_count": primary_wrapper.transpiled_cx_count,
                "transpile_failure": primary_wrapper.transpile_failure,
                "claim_boundary": CLAIM,
            }
        ),
    )
    write_json(
        reconstruction_json,
        json_ready(
            {
                "precision_bits": primary_bits,
                "normalization_factor_s_mu": primary_wrapper.normalization_factor,
                "target_block_A_qT_over_s_mu": primary_wrapper.target_block,
                "encoded_top_left_block": primary_wrapper.encoded_block,
                "top_left_reconstruction_error": primary_wrapper.top_left_reconstruction_error,
                "unitarity_error": primary_wrapper.unitarity_error,
                "statevector_max_error": primary_wrapper.statevector_max_error,
                "quantized_block_H_q": primary_block.quantized,
                "sparsified_block": primary_block.sparsified,
                "original_block": primary_block.original,
            }
        ),
    )
    write_json(coloring_json, json_ready(_edge_coloring_report(primary_block, primary_wrapper)))
    readme_md.write_text(
        _readme(validation_rows, qsvt_rows, block_source, primary_wrapper, overall_status),
        encoding="utf-8",
    )

    artifacts = {
        "sparse_wrapper_8x8_validation_csv": validation_csv,
        "sparse_wrapper_8x8_qsvt_validation_csv": qsvt_csv,
        "sparse_wrapper_8x8_circuit_metadata_json": circuit_json,
        "sparse_wrapper_8x8_block_reconstruction_json": reconstruction_json,
        "edge_coloring_validation_json": coloring_json,
        "readme_md": readme_md,
    }
    manifest = write_phase10_manifest(
        output_dir=output_dir,
        experiment_id="phase10_sparse_wrapper_8x8_complete",
        script_name="scripts/run_phase10_sparse_wrapper_8x8_complete.py",
        command=str(resolved["command"]),
        description=CLAIM,
        artifacts=artifacts,
        seeds={"system_seed": int(resolved["seed"])},
        extra={
            "overall_status": overall_status,
            "block_source": block_source,
            "value_precision_bits": list(precisions),
            "primary_precision_bits": primary_bits,
            "qsvt_rows": qsvt_rows,
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "overall_status": overall_status,
        "validation_rows": validation_rows,
        "qsvt_rows": qsvt_rows,
        "wrapper": primary_wrapper,
        "block": primary_block,
        "artifacts": artifacts,
    }


def _readme(
    validation_rows: list[dict[str, Any]],
    qsvt_rows: list[dict[str, Any]],
    block_source: str,
    wrapper: CompleteWrapperResult,
    overall_status: str,
) -> str:
    lines = [
        "# Phase 10 WP A: Complete 8x8 Sparse Block-Encoding Wrapper",
        "",
        CLAIM,
        "",
        f"- Overall status: **{overall_status}**",
        f"- Block source: {block_source}",
        f"- Slots: {wrapper.slots} (Konig minimum for this pattern; the Phase 9 request of 2 "
        "slots was mathematically infeasible because the pattern has maximum column degree 3)",
        f"- Normalization factor s*mu = {wrapper.normalization_factor:.4f}",
        "",
        "## Phase 9 blocker resolution",
        "",
        "The Phase 9 blocker had two layers (details in `edge_coloring_validation.json`):",
        "",
        "1. the requested 2-slot coloring does not exist for this pattern (max column degree "
        "3; Konig's theorem needs 3 slots), and the old routine looped instead of rejecting;",
        "2. the old alternating-path recoloring could cycle forever due to desynchronized "
        "color bookkeeping.",
        "",
        "The replacement (`robust_qsvt_se/qsvt/bipartite_slot_assignment.py`) regularizes the "
        "pattern to a slots-regular bipartite multigraph and peels perfect matchings under an "
        "explicit visit budget, so termination is structural, the result is validated, and "
        "infeasible requests are rejected immediately with a diagnosis.",
        "",
        "## Wrapper validation (per value precision)",
        "",
        "| bits | slots | nnz | recon err | unitarity err | statevec err | qubits | raw gates "
        "| depth | transpiled (ops/depth/cx) | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in validation_rows:
        transpiled = (
            f"{row['transpiled_gate_count']}/{row['transpiled_depth']}/{row['transpiled_cx_count']}"
            if row["transpiled_gate_count"] is not None
            else f"omitted ({row['transpile_failure']})"
        )
        lines.append(
            f"| {row['value_precision_bits']} | {row['slots']} | {row['nnz']} | "
            f"{row['top_left_reconstruction_error']:.2e} | {row['unitarity_error']:.2e} | "
            f"{row['statevector_max_error']:.2e} | {row['qubits']} | {row['raw_gate_count']} | "
            f"{row['raw_depth']} | {transpiled} | {row['status']} |"
        )
    lines += [
        "",
        "The wrapper encodes the transpose orientation `A = H_q^T/(s mu)` used by the QSVT "
        "update pathway. Per-slot value keying assigns each real quantized entry to exactly "
        "one slot and gives padding pairs the exact value zero, so parallel padding cannot "
        "double count an entry (`lookup_value_max_error` is exactly checkable and reported).",
        "",
        "## QSVT integration on the sparse wrapper (primary precision)",
        "",
        "Matched-Ridge bounded-target phase sequences applied directly to the wrapper "
        "unitary, compared against the dense-dilation QSVT action, the exact singular-value "
        "transform, and Ridge at the same alpha on the same quantized block:",
        "",
        "| degree | status | sparse vs exact SVT | sparse vs dense | p_succ | update rel err "
        "vs Ridge |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in qsvt_rows:
        lines.append(
            f"| {row.get('degree')} | {row.get('status')} | "
            f"{_fmt(row.get('sparse_vs_exact_svt_error'))} | "
            f"{_fmt(row.get('sparse_vs_dense_action_error'))} | "
            f"{_fmt(row.get('sparse_postselection_probability'), '{:.4f}')} | "
            f"{_fmt(row.get('sparse_update_relative_error_vs_ridge'))} |"
        )
    lines += [
        "",
        "Physical recovery uses the single factor `C/beta` with `beta = s*mu` (Option B "
        "normalization convention). The Ridge reference uses the same quantized block and the "
        "same alpha, so this compares implementations of the same filter, not estimators.",
        "",
        "## Scope",
        "",
        "This is a tiny-instance (8x8) completeness demonstration of the sparse "
        "block-encoding wrapper interface on a classical simulator. It is not an IEEE-scale "
        "sparse block encoding, not a scalable oracle synthesis, and not a hardware run. "
        "The sparsification and quantization are recorded representation changes: the wrapper "
        "encodes the derived quantized sparse block exactly, and every comparison uses that "
        "same derived block.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _fmt(value: Any, spec: str = "{:.2e}") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10 WP A: complete 8x8 sparse block-encoding wrapper"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--value-precision-bits", nargs="+", type=int, default=list(VALUE_PRECISION_BITS)
    )
    parser.add_argument("--skip-transpile", action="store_true")
    parser.add_argument("--skip-qsvt", action="store_true")
    args = parser.parse_args(argv)
    run = run_phase10_sparse_wrapper_complete(
        {
            "output_dir": args.output_dir,
            "seed": args.seed,
            "value_precision_bits": args.value_precision_bits,
            "transpile": not args.skip_transpile,
            "run_qsvt_integration": not args.skip_qsvt,
            "command": "scripts/run_phase10_sparse_wrapper_8x8_complete.py " + " ".join(argv or []),
        }
    )
    print(f"Overall status: {run['overall_status']}")
    print(pd.DataFrame(run["validation_rows"]).to_string(index=False))
    if run["qsvt_rows"]:
        print(pd.DataFrame(run["qsvt_rows"]).to_string(index=False))
    print(f"Outputs: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
