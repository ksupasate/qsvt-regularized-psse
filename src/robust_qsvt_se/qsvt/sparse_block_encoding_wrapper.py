"""Goal B: a toy *complete* sparse block-encoding wrapper on a quantized IEEE block.

The previously compiled lookup circuits (``O_col``/``O_val``) validate reversible
index and value access only. This module compiles the remaining wrapper around the
same lookup data for one tiny instance: slot diffusion, a multiplexed value-rotation
oracle, in-place slot-controlled column permutations, and diffusion uncomputation,
so the resulting unitary is a complete block encoding

    <0|_a <0|_k <i| U_A |0>_a |0>_k |j> = A_q[i, j] / (s * mu),

where ``A_q`` is the sparsified, fixed-point-quantized IEEE-derived block, ``s`` is
the row/column sparsity bound, and ``mu`` is the quantization full-scale value.

Construction notes (honest scope):

* The nonzero pattern is edge-colored (Konig) into ``s`` partial permutations and
  padded to full permutations, so the slot-controlled column map is in-place
  reversible. General sparse patterns whose slot maps are not permutations would
  need out-of-place index registers plus uncomputation, which remain modeled.
* The value oracle is a multiplexed rotation enumerated over the stored lookup
  addresses (exact at toy scale). A scalable QROM-fed fixed-point-to-rotation
  arithmetic pipeline remains modeled.
* Feasible only because the instance is tiny; this is a completeness demonstration
  for the wrapper interface, not a scalable compilation and not a hardware run.

An optional integration check applies a synthesized QSVT phase sequence for the
matched Ridge/Tikhonov bounded target directly to this wrapper unitary and
compares the postselected selected outputs against Ridge at the same alpha on the
same quantized matrix.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    assert_safe,
    fit_codesigned_bounded_polynomial,
    write_demo_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_DIR = Path("outputs/sparse_block_encoding_wrapper_demo")
MAGNITUDE_BITS = 6  # matches the compiled O_val convention (6 magnitude + 1 sign bit)
ROW_COL_DEGREE_BOUND = 2
VALIDATION_TOLERANCE = 1.0e-9
QSVT_DEGREES = (31, 39, 45)
QSVT_PASS_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class SparsifiedQuantizedBlock:
    """IEEE-derived block after row/column-degree sparsification and quantization."""

    original: np.ndarray
    sparsified: np.ndarray
    quantized: np.ndarray
    mu: float
    sparsity_bound: int
    nnz: int
    quantization_step: float
    max_quantization_error: float
    sparsification_fro_error: float


def sparsify_row_col_degree(matrix: np.ndarray, *, bound: int) -> np.ndarray:
    """Keep at most ``bound`` largest-magnitude entries per row *and* per column.

    Rows are thresholded first; columns above the bound then drop their smallest
    surviving magnitudes. The result admits a ``bound``-color Konig edge coloring.
    """

    values = np.asarray(matrix, dtype=np.float64)
    out = np.zeros_like(values)
    for row in range(values.shape[0]):
        magnitudes = np.abs(values[row])
        keep = min(bound, int(np.count_nonzero(magnitudes)))
        if keep:
            columns = np.argsort(magnitudes)[::-1][:keep]
            out[row, columns] = values[row, columns]
    for col in range(out.shape[1]):
        rows = np.flatnonzero(np.abs(out[:, col]) > 0.0)
        if rows.size > bound:
            order = np.argsort(np.abs(out[rows, col]))
            for row in rows[order[: rows.size - bound]]:
                out[row, col] = 0.0
    return out


def quantize_sign_magnitude(matrix: np.ndarray, *, magnitude_bits: int) -> tuple[np.ndarray, float]:
    """Sign-magnitude fixed-point quantization on a uniform grid (O_val convention)."""

    values = np.asarray(matrix, dtype=np.float64)
    mu = float(np.max(np.abs(values)))
    if mu <= 0.0:
        raise ValueError("matrix must have at least one nonzero entry")
    levels = (1 << magnitude_bits) - 1
    codes = np.round(np.abs(values) / mu * levels)
    quantized = np.sign(values) * codes / levels * mu
    return quantized, mu


def build_sparsified_quantized_block(matrix: np.ndarray) -> SparsifiedQuantizedBlock:
    sparsified = sparsify_row_col_degree(matrix, bound=ROW_COL_DEGREE_BOUND)
    quantized, mu = quantize_sign_magnitude(sparsified, magnitude_bits=MAGNITUDE_BITS)
    step = mu / ((1 << MAGNITUDE_BITS) - 1)
    return SparsifiedQuantizedBlock(
        original=np.asarray(matrix, dtype=np.float64),
        sparsified=sparsified,
        quantized=quantized,
        mu=mu,
        sparsity_bound=ROW_COL_DEGREE_BOUND,
        nnz=int(np.count_nonzero(quantized)),
        quantization_step=step,
        max_quantization_error=float(np.max(np.abs(quantized - sparsified))),
        sparsification_fro_error=float(
            np.linalg.norm(sparsified - matrix) / max(np.linalg.norm(matrix), 1e-30)
        ),
    )


def konig_slot_permutations(pattern: np.ndarray, *, slots: int) -> list[np.ndarray]:
    """Edge-color the bipartite nonzero pattern into ``slots`` full permutations.

    Bipartite graphs with maximum degree ``slots`` are ``slots``-edge-colorable
    (Konig). Each color class is a partial matching (at most one entry per row and
    per column) padded greedily to a full permutation; padded pairs carry value 0.
    Returns ``pi_k`` arrays with ``pi_k[j] = i`` meaning slot ``k`` maps column
    ``j`` to row ``i``.
    """

    n = pattern.shape[0]
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(pattern), strict=True)]
    color_of: dict[tuple[int, int], int] = {}
    row_used: list[set[int]] = [set() for _ in range(n)]
    col_used: list[set[int]] = [set() for _ in range(n)]

    for i, j in edges:
        free = next(
            (c for c in range(slots) if c not in row_used[i] and c not in col_used[j]), None
        )
        if free is None:
            # Konig augmenting path: colors a free on the row side, b free on the col side.
            a = next(c for c in range(slots) if c not in row_used[i])
            b = next(c for c in range(slots) if c not in col_used[j])
            # Flip the alternating a/b path starting from column j.
            side_row, node, current = False, j, a
            while True:
                match = next(
                    (
                        edge
                        for edge, color in color_of.items()
                        if color == current
                        and ((side_row and edge[0] == node) or (not side_row and edge[1] == node))
                    ),
                    None,
                )
                if match is None:
                    break
                color_of[match] = b if current == a else a
                row_used[match[0]].symmetric_difference_update({a, b})
                col_used[match[1]].symmetric_difference_update({a, b})
                node = match[1] if side_row else match[0]
                side_row = not side_row
                current = b if current == a else a
            free = a
        color_of[(i, j)] = free
        row_used[i].add(free)
        col_used[j].add(free)

    permutations: list[np.ndarray] = []
    for color in range(slots):
        pi = np.full(n, -1, dtype=np.int64)
        used_rows: set[int] = set()
        for (i, j), c in color_of.items():
            if c == color:
                pi[j] = i
                used_rows.add(i)
        free_rows = [i for i in range(n) if i not in used_rows]
        for j in range(n):
            if pi[j] < 0:
                pi[j] = free_rows.pop()
        permutations.append(pi)

    for color, pi in enumerate(permutations):
        if sorted(pi.tolist()) != list(range(n)):
            raise RuntimeError(f"slot {color} map is not a permutation: {pi}")
    return permutations


@dataclass(slots=True)
class WrapperCircuitResult:
    circuit: Any
    unitary: np.ndarray
    encoded_block: np.ndarray
    target_block: np.ndarray
    normalization_factor: float
    top_left_reconstruction_error: float
    unitarity_error: float
    statevector_max_error: float
    qubits: int
    gate_count: int
    depth: int
    transpiled_gate_count: int
    transpiled_depth: int
    transpiled_cx_count: int


def build_wrapper_circuit(block: SparsifiedQuantizedBlock, *, encode_transpose: bool) -> Any:
    """Compile the complete wrapper: H(slot) -> O_val rotations -> O_col perms -> H(slot)."""

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, UnitaryGate

    matrix = block.quantized.T if encode_transpose else block.quantized
    n = matrix.shape[0]
    index_qubits = int(math.log2(n))
    slots = block.sparsity_bound
    slot_qubits = math.ceil(math.log2(slots))
    permutations = konig_slot_permutations(np.abs(matrix) > 0.0, slots=slots)

    total_qubits = index_qubits + slot_qubits + 1
    circuit = QuantumCircuit(total_qubits, name="sparse_BE_wrapper")
    index_regs = list(range(index_qubits))
    slot_regs = list(range(index_qubits, index_qubits + slot_qubits))
    ancilla = index_qubits + slot_qubits

    for q in slot_regs:
        circuit.h(q)

    # Multiplexed value-rotation oracle: <0|Ry(2 arccos(v))|0> = v for v in [-1, 1].
    for k, pi in enumerate(permutations):
        for j in range(n):
            v = float(matrix[pi[j], j] / block.mu)
            theta = 2.0 * math.acos(np.clip(v, -1.0, 1.0))
            controls = index_regs + slot_regs
            ctrl_state = j + (k << index_qubits)
            gate = RYGate(theta).control(len(controls), ctrl_state=ctrl_state)
            circuit.append(gate, [*controls, ancilla])

    # Slot-controlled in-place column permutations (compiled O_col wrapper role).
    for k, pi in enumerate(permutations):
        perm_matrix = np.zeros((n, n))
        perm_matrix[pi, np.arange(n)] = 1.0
        gate = UnitaryGate(perm_matrix, label=f"P_slot{k}").control(slot_qubits, ctrl_state=k)
        circuit.append(gate, slot_regs + index_regs)

    for q in slot_regs:
        circuit.h(q)
    return circuit


def validate_wrapper(
    block: SparsifiedQuantizedBlock, *, encode_transpose: bool
) -> WrapperCircuitResult:
    from qiskit import transpile
    from qiskit.quantum_info import Operator, Statevector

    matrix = block.quantized.T if encode_transpose else block.quantized
    n = matrix.shape[0]
    normalization = block.sparsity_bound * block.mu
    target = matrix / normalization

    circuit = build_wrapper_circuit(block, encode_transpose=encode_transpose)
    unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    dim = unitary.shape[0]
    unitarity_error = float(np.max(np.abs(unitary.conj().T @ unitary - np.eye(dim))))
    encoded = np.real(unitary[:n, :n])
    reconstruction_error = float(np.max(np.abs(encoded - target)))

    statevector_max_error = 0.0
    for j in range(n):
        state = np.zeros(dim, dtype=np.complex128)
        state[j] = 1.0
        evolved = Statevector(state).evolve(circuit).data
        column_error = float(np.max(np.abs(np.real(evolved[:n]) - target[:, j])))
        statevector_max_error = max(statevector_max_error, column_error)

    transpiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
    transpiled_counts = {str(k): int(v) for k, v in transpiled.count_ops().items()}
    return WrapperCircuitResult(
        circuit=circuit,
        unitary=unitary,
        encoded_block=encoded,
        target_block=target,
        normalization_factor=normalization,
        top_left_reconstruction_error=reconstruction_error,
        unitarity_error=unitarity_error,
        statevector_max_error=statevector_max_error,
        qubits=int(circuit.num_qubits),
        gate_count=int(sum(circuit.count_ops().values())),
        depth=int(circuit.depth()),
        transpiled_gate_count=int(sum(transpiled_counts.values())),
        transpiled_depth=int(transpiled.depth()),
        transpiled_cx_count=int(transpiled_counts.get("cx", 0)),
    )


def qsvt_integration_check(
    block: SparsifiedQuantizedBlock,
    wrapper: WrapperCircuitResult,
    r_block: np.ndarray,
    *,
    phase_cache_dir: Path,
) -> dict[str, Any]:
    """Apply a matched-Ridge QSVT phase sequence to the wrapper unitary.

    The wrapper encodes ``A = H_q^T / (s mu)`` so ``beta_be = s mu`` plays the
    normalization role. Ridge at the same alpha on the same quantized matrix is
    the reference. Failure modes are reported, not hidden.
    """

    from qiskit.quantum_info import Operator, Statevector

    H_q = block.quantized
    n = H_q.shape[0]
    r = np.asarray(r_block, dtype=np.float64)
    sigma = np.linalg.svd(H_q, compute_uv=False)
    sigma_min, sigma_max = float(sigma.min()), float(sigma.max())
    alpha = 4.0 * sigma_min**2
    beta_be = wrapper.normalization_factor
    ridge_update = ridge_svd_solution(H_q, r, alpha=alpha)

    A = H_q.T / beta_be
    s_values = np.linalg.svd(A, compute_uv=False)
    domain_min = float(np.clip(0.9 * s_values.min(), 1.0e-4, 0.999))

    record: dict[str, Any] = {
        "alpha": alpha,
        "beta_effective": beta_be,
        "lambda_alpha_over_beta2": alpha / beta_be**2,
        "sigma_min_quantized": sigma_min,
        "sigma_max_quantized": sigma_max,
        "kappa_quantized": sigma_max / sigma_min,
        "status": "not_attempted",
    }

    for degree in QSVT_DEGREES:
        # Fit on [domain_min, 1] like the anchor: sigma_max <= s*mu guarantees the
        # normalized spectrum lies inside, and a truncated domain lets the odd
        # Chebyshev fit blow past the QSVT bound outside it.
        target = fit_codesigned_bounded_polynomial(
            beta=beta_be,
            alpha=alpha,
            domain_min=domain_min,
            domain_max=1.0,
            degree=degree,
            margin=1.05,
        )
        try:
            validate_qsvt_polynomial(
                np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
            )
        except Exception as exc:
            record.update(
                {
                    "degree": degree,
                    "status": "tolerance_missing",
                    "failure_reason": f"boundedness: {exc}",
                }
            )
            continue
        try:
            cached = synthesize_pennylane_phases_cached(
                np.asarray(target.coefficients),
                angle_solver="iterative",
                cache_dir=phase_cache_dir,
                cache_metadata={"wrapper": "sparse_BE", "degree": degree, "alpha": alpha},
            )
            phases = np.asarray(cached.phases, dtype=np.float64)
        except Exception as exc:
            record.update({"degree": degree, "status": "phase_failed", "failure_reason": str(exc)})
            continue

        bundle = build_structured_qsvt_operator_circuit(
            wrapper.unitary, phases, encoded_dimension=n
        )
        operator = np.asarray(Operator(bundle.qsvt_operator_circuit).data, dtype=np.complex128)
        transformed = np.real(operator[:n, :n])
        U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=True)
        exact = U_A @ np.diag(target.polynomial(S_A)) @ Vt_A
        svt_error = float(np.max(np.abs(transformed - exact)))

        padded = np.zeros(operator.shape[0], dtype=np.complex128)
        padded[:n] = r / np.linalg.norm(r)
        evolved = Statevector(padded).evolve(bundle.qsvt_operator_circuit).data
        encoded = evolved[:n]
        p_succ = float(np.vdot(encoded, encoded).real)
        psi_out = encoded / math.sqrt(p_succ) if p_succ > 1e-15 else encoded
        recovery = target.physical_recovery_factor * float(np.linalg.norm(r)) * math.sqrt(p_succ)
        qsvt_update = recovery * np.real(psi_out)
        rel_error = float(
            np.linalg.norm(qsvt_update - ridge_update) / max(np.linalg.norm(ridge_update), 1e-30)
        )
        selected_output_qsvt = float(qsvt_update[0])
        selected_output_ridge = float(ridge_update[0])
        record.update(
            {
                "degree": degree,
                "phase_count": int(phases.size),
                "target_fit_error": target.fit_max_abs_error,
                "bound_C": target.bound_C,
                "circuit_vs_exact_svt_error": svt_error,
                "postselection_probability": p_succ,
                "update_relative_error_vs_ridge": rel_error,
                "selected_output_e1_qsvt": selected_output_qsvt,
                "selected_output_e1_ridge": selected_output_ridge,
                "status": (
                    "statevector_validated"
                    if rel_error <= QSVT_PASS_RELATIVE_TOLERANCE
                    else "degree_limited"
                ),
            }
        )
        if record["status"] == "statevector_validated":
            break
    return record


def run_wrapper_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": str(OUTPUT_DIR),
        "case": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "size": 4,
        "run_qsvt_integration": True,
        "command": "scripts/run_sparse_block_encoding_wrapper_demo.py",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    phase_cache_dir = ensure_directory(output_dir / "phase_cache")

    system, matrix_source = build_engineering_system(
        {
            "case_name": resolved["case"],
            "case_source": resolved["case_source"],
            "matrix_source": "weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
    )
    size = int(resolved["size"])
    H_block, r_block, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=size,
        col_count=size,
        policy="largest_row_col_norms",
    )
    block = build_sparsified_quantized_block(H_block)

    status = "failed"
    failure_reason = ""
    wrapper: WrapperCircuitResult | None = None
    try:
        wrapper = validate_wrapper(block, encode_transpose=True)
        checks_ok = (
            wrapper.unitarity_error <= VALIDATION_TOLERANCE
            and wrapper.top_left_reconstruction_error <= VALIDATION_TOLERANCE
            and wrapper.statevector_max_error <= VALIDATION_TOLERANCE
        )
        status = "statevector_validated" if checks_ok else "compiled"
        if not checks_ok:
            failure_reason = (
                f"compiled but reconstruction/unitarity above {VALIDATION_TOLERANCE:g} tolerance"
            )
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    qsvt_record: dict[str, Any] = {"status": "not_attempted"}
    if wrapper is not None and resolved["run_qsvt_integration"]:
        try:
            qsvt_record = qsvt_integration_check(
                block, wrapper, r_block, phase_cache_dir=phase_cache_dir
            )
        except Exception as exc:
            qsvt_record = {"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}

    row = {
        "matrix_source": (
            f"{matrix_source} {size}x{size} block (rows {' '.join(map(str, rows))}; "
            f"cols {' '.join(map(str, cols))}), sparsified to row/col degree "
            f"<= {ROW_COL_DEGREE_BOUND}, {MAGNITUDE_BITS}-bit sign-magnitude quantized"
        ),
        "matrix_shape": f"{size}x{size}",
        "sparsity": f"{block.nnz} nnz (row/col degree <= {block.sparsity_bound})",
        "quantization_bits": f"{MAGNITUDE_BITS} magnitude + 1 sign",
        "normalization_factor": wrapper.normalization_factor if wrapper else math.nan,
        "lookup_oracles_used": (
            "slot-controlled column permutations (O_col role, Konig edge coloring) + "
            "multiplexed value rotations from stored fixed-point words (O_val role)"
        ),
        "wrapper_type": (
            "complete sparse query-model block encoding: slot diffusion + value-rotation "
            "oracle + in-place column permutation + diffusion uncomputation"
        ),
        "top_left_reconstruction_error": (
            wrapper.top_left_reconstruction_error if wrapper else math.nan
        ),
        "unitarity_error": wrapper.unitarity_error if wrapper else math.nan,
        "statevector_max_error": wrapper.statevector_max_error if wrapper else math.nan,
        "qubits": wrapper.qubits if wrapper else math.nan,
        "gate_count": wrapper.gate_count if wrapper else math.nan,
        "depth": wrapper.depth if wrapper else math.nan,
        "transpiled_gate_count": wrapper.transpiled_gate_count if wrapper else math.nan,
        "transpiled_depth": wrapper.transpiled_depth if wrapper else math.nan,
        "transpiled_cx_count": wrapper.transpiled_cx_count if wrapper else math.nan,
        "sparsification_relative_fro_error": block.sparsification_fro_error,
        "max_quantization_error": block.max_quantization_error,
        "status": status,
        "failure_reason": failure_reason,
        "qsvt_integration_status": qsvt_record.get("status", "not_attempted"),
        "qsvt_integration_degree": qsvt_record.get("degree", math.nan),
        "qsvt_integration_update_rel_error": qsvt_record.get(
            "update_relative_error_vs_ridge", math.nan
        ),
        "qsvt_integration_postselection": qsvt_record.get("postselection_probability", math.nan),
    }

    frame = pd.DataFrame([row])
    results_csv = output_dir / "wrapper_demo_results.csv"
    results_json = output_dir / "wrapper_demo_results.json"
    frame.to_csv(results_csv, index=False)
    frame.to_json(results_json, orient="records", indent=2)

    readme = output_dir / "README.md"
    readme.write_text(_readme_markdown(row, block, qsvt_record), encoding="utf-8")

    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="sparse_block_encoding_wrapper_demo",
        description=(
            "Toy complete sparse block-encoding wrapper for a sparsified, six-bit-quantized "
            "IEEE-14-derived 4x4 weighted-Jacobian block: slot diffusion, multiplexed value "
            "rotations, in-place Konig column permutations, and uncomputation compiled and "
            "statevector-validated, with an optional QSVT phase-sequence integration check "
            "against matched Ridge. Tiny-instance completeness evidence; not a scalable "
            "compilation and not a hardware run."
        ),
        command=str(resolved["command"]),
        artifacts={
            "wrapper_demo_results_csv": results_csv,
            "wrapper_demo_results_json": results_json,
            "readme_md": readme,
        },
        input_files=[f"build_engineering_system:{resolved['case']}:weighted_jacobian:seed123"],
        extra={
            "seed_provenance": {
                "status": "recorded",
                "seeds": {"system_seed": int(resolved["seed"])},
                "source": "--seed argument (default 123)",
                "applies_to": "generated IEEE weighted-Jacobian system",
            },
            "qsvt_integration_record": {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in qsvt_record.items()
            },
            "validation_tolerance": VALIDATION_TOLERANCE,
        },
    )
    return {
        "output_dir": output_dir,
        "row": row,
        "wrapper": wrapper,
        "qsvt_record": qsvt_record,
        "manifest": manifest,
    }


def _readme_markdown(
    row: dict[str, Any], block: SparsifiedQuantizedBlock, qsvt_record: dict[str, Any]
) -> str:
    lines = [
        "# Toy Complete Sparse Block-Encoding Wrapper Demo (Goal B)",
        "",
        "Compiles the *wrapper* around validated lookup data for one tiny quantized",
        "IEEE-derived sparse block, so that a complete block encoding",
        "`<0|<0|<i| U_A |0>|0|j> = A_q[i,j] / (s mu)` exists as an executable circuit:",
        "",
        "```text",
        "CSR data -> Konig edge coloring -> slot diffusion (H)",
        "        -> multiplexed value rotations (O_val role)",
        "        -> slot-controlled in-place column permutations (O_col role)",
        "        -> diffusion uncomputation (H) -> U_A",
        "```",
        "",
        f"- Status: **{row['status']}**"
        + (f" ({row['failure_reason']})" if row["failure_reason"] else ""),
        f"- Matrix: {row['matrix_source']}",
        f"- Sparsity: {row['sparsity']}; quantization: {row['quantization_bits']} bits",
        f"- Normalization factor s*mu = {row['normalization_factor']:.4f}",
        "- Top-left reconstruction error vs A_q/(s mu): "
        f"{row['top_left_reconstruction_error']:.3e}",
        f"- Unitarity error: {row['unitarity_error']:.3e}; statevector max error: "
        f"{row['statevector_max_error']:.3e}",
        f"- Circuit: {row['qubits']} qubits, {row['gate_count']} gates (depth {row['depth']}); "
        f"transpiled to u3/cx: {row['transpiled_gate_count']} ops, depth "
        f"{row['transpiled_depth']}, {row['transpiled_cx_count']} cx",
        "",
        "## QSVT phase-sequence integration check",
        "",
    ]
    if qsvt_record.get("degree") is not None and "update_relative_error_vs_ridge" in qsvt_record:
        lines += [
            f"- Status: **{qsvt_record['status']}** at degree {qsvt_record['degree']}",
            f"- alpha = {qsvt_record['alpha']:.6g} (matched Ridge on the same quantized block), "
            f"lambda = alpha/(s mu)^2 = {qsvt_record['lambda_alpha_over_beta2']:.4g}",
            f"- circuit vs exact SVT error: {qsvt_record['circuit_vs_exact_svt_error']:.3e}",
            f"- postselection probability: {qsvt_record['postselection_probability']:.4f}",
            f"- postselected update vs matched Ridge (same alpha, quantized block): "
            f"{qsvt_record['update_relative_error_vs_ridge']:.3e} relative",
        ]
    else:
        reason = qsvt_record.get("failure_reason")
        suffix = f" ({reason})" if reason else ""
        lines.append(f"- Status: **{qsvt_record.get('status', 'not_attempted')}**{suffix}")
    lines += [
        "",
        "## What is compiled here vs what remains modeled",
        "",
        "| Wrapper piece | This demo | At scale |",
        "|---|---|---|",
        "| Slot diffusion / uncomputation | compiled (H gates) | compiled trivially |",
        "| Column access (O_col role) | compiled in-place permutations via Konig coloring | "
        "modeled: general patterns need out-of-place index registers + uncomputation |",
        "| Value access (O_val role) | compiled multiplexed rotations from stored fixed-point "
        "words | modeled: QROM-fed fixed-point-to-rotation arithmetic pipeline |",
        "| Normalization s*mu | recorded classical factor | recorded classical factor |",
        "| Controlled access for QSVT | compiled (projector-controlled phases applied to U_A) | "
        "modeled |",
        "",
        "The sparsification (row/col degree <= "
        f"{block.sparsity_bound}, relative Frobenius change "
        f"{block.sparsification_fro_error:.3f}) and {MAGNITUDE_BITS}-bit quantization "
        f"(max entry error {block.max_quantization_error:.3f}) are recorded representation",
        "changes: the wrapper encodes the *derived* quantized sparse block exactly, and the",
        "matched Ridge reference in the integration check uses that same derived block and",
        "the same alpha. This is tiny-instance completeness evidence for the wrapper",
        "interface; it is not a scalable sparse-oracle compilation, not an IEEE-scale block",
        "encoding, and not a quantum-hardware run.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Goal B sparse wrapper demo")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--skip-qsvt", action="store_true")
    args = parser.parse_args(argv)
    run = run_wrapper_demo(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "seed": args.seed,
            "size": args.size,
            "run_qsvt_integration": not args.skip_qsvt,
            "command": "scripts/run_sparse_block_encoding_wrapper_demo.py " + " ".join(argv or []),
        }
    )
    print(f"Wrapper status: {run['row']['status']}")
    print(f"QSVT integration: {run['qsvt_record'].get('status')}")
    print(f"Results: {run['output_dir']}/wrapper_demo_results.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
