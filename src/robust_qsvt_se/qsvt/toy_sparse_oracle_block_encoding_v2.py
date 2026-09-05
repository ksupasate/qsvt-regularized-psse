from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.hardware_aware_oracle_cost_model import matrix_sparse_stats
from robust_qsvt_se.qsvt.sparse_oracle_gates import (
    _transpile_and_export_qasm,
    _two_qubit_count,
    _with_control_pattern,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

CLAIM = (
    "Strengthened toy sparse-oracle prototype with a fixed-point value register, "
    "multiplexed value rotations, IEEE-derived sparse-block lookup validation, a "
    "value-loading precision sweep, transpiled controlled-rotation/two-qubit "
    "resource counts, and a sparse-vs-dense block-encoding resource comparison. "
    "This is a prototype plus cost model on tiny (4x4/8x8) blocks. It is NOT "
    "hardware execution, NOT a full hardware-native sparse-oracle synthesis, and "
    "NOT an IEEE-scale gate-level solver."
)

# Pre-transpile gate names that represent (multi-)controlled value rotations.
_CONTROLLED_ROTATION_NAMES = ("cry", "mcry", "cry_o0", "mcry_o0", "cu3", "crz", "cu")

VALIDATION_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class SparseOracleV2Entry:
    """One reversible lookup slot ``|row, local_index> -> |row, column, value>``."""

    row: int
    local_index: int
    column: int
    value: float
    normalized_value: float
    quantized_value: float
    quantization_abs_error: float
    fixed_point_bits: tuple[int, ...]
    rotation_angle: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "local_index": self.local_index,
            "column": self.column,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "quantized_value": self.quantized_value,
            "quantization_abs_error": self.quantization_abs_error,
            "fixed_point_bits": list(self.fixed_point_bits),
            "rotation_angle": self.rotation_angle,
        }


def sparsify_block(matrix: np.ndarray, *, keep_per_row: int = 2) -> np.ndarray:
    """Threshold each row to its ``keep_per_row`` largest-magnitude entries.

    IEEE-derived weighted-Jacobian subblocks are only moderately sparse; keeping
    a small fixed number of nonzeros per row yields a genuinely sparse block that
    matches the two-slot reversible-lookup structure of the toy oracle.
    """

    values = np.asarray(matrix, dtype=np.float64)
    out = np.zeros_like(values)
    for row in range(values.shape[0]):
        magnitudes = np.abs(values[row])
        if not np.any(magnitudes > 0.0):
            continue
        keep = min(int(keep_per_row), int(np.count_nonzero(magnitudes)))
        columns = np.argsort(magnitudes)[::-1][:keep]
        out[row, columns] = values[row, columns]
    return out


def fixed_point_encode(normalized_value: float, *, bits: int) -> tuple[float, tuple[int, ...]]:
    """Quantize a value in ``[-1, 1]`` to a signed ``bits``-bit fixed-point grid.

    The most significant bit carries the sign; the remaining ``bits - 1`` bits
    encode the magnitude on a uniform grid with step ``2^-(bits-1)``. Returns the
    reconstructed value and its bit pattern (MSB-first, sign bit last position).
    """

    if int(bits) < 2:
        raise ValueError("fixed-point value register requires at least 2 bits")
    value = float(np.clip(normalized_value, -1.0, 1.0))
    magnitude_bits = int(bits) - 1
    levels = (1 << magnitude_bits) - 1
    sign = 0 if value >= 0.0 else 1
    code = round(abs(value) * levels)
    code = min(code, levels)
    reconstructed = (-1.0 if sign else 1.0) * (code / levels if levels else 0.0)
    magnitude_pattern = tuple((code >> position) & 1 for position in range(magnitude_bits))
    return float(reconstructed), (*magnitude_pattern, sign)


def sparse_oracle_v2_entries(
    matrix: np.ndarray,
    *,
    value_precision_bits: int,
    slots_per_row: int = 2,
) -> tuple[list[SparseOracleV2Entry], float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("sparse oracle v2 requires a square matrix")
    beta = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    entries: list[SparseOracleV2Entry] = []
    for row in range(values.shape[0]):
        columns = np.flatnonzero(np.abs(values[row]) > 1.0e-12)
        for local_index in range(int(slots_per_row)):
            if local_index < columns.size:
                column = int(columns[local_index])
                value = float(values[row, column])
            else:
                column = 0
                value = 0.0
            normalized = float(np.clip(value / beta, -1.0, 1.0))
            quantized, bits = fixed_point_encode(normalized, bits=int(value_precision_bits))
            entries.append(
                SparseOracleV2Entry(
                    row=row,
                    local_index=local_index,
                    column=column,
                    value=value,
                    normalized_value=normalized,
                    quantized_value=quantized,
                    quantization_abs_error=abs(quantized - normalized),
                    fixed_point_bits=bits,
                    rotation_angle=float(2.0 * np.arcsin(np.clip(quantized, -1.0, 1.0))),
                )
            )
    return entries, beta


@dataclass(frozen=True, slots=True)
class SparseOracleV2Circuit:
    circuit: Any
    entries: list[SparseOracleV2Entry]
    row_qubits: int
    local_index_qubits: int
    column_qubits: int
    value_register_qubits: int
    value_rotation_qubit: int
    beta: float


def build_sparse_oracle_v2_circuit(
    matrix: np.ndarray,
    *,
    value_precision_bits: int,
    slots_per_row: int = 2,
) -> SparseOracleV2Circuit:
    """Build a reversible sparse-access oracle with a fixed-point value register.

    Layout: row register, local-index register, reversible column-output register,
    an ``n``-bit fixed-point value register loaded by multiplexed (pattern-
    controlled) X gates, and one value-rotation qubit driven by a multiplexed CRY
    whose angle is the quantized fixed-point value. Loading is conditioned on the
    (row, local-index) address, so the whole oracle is reversible.
    """

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for sparse-oracle v2 circuits") from exc

    values = np.asarray(matrix, dtype=np.float64)
    dimension = int(values.shape[0])
    entries, beta = sparse_oracle_v2_entries(
        values,
        value_precision_bits=int(value_precision_bits),
        slots_per_row=int(slots_per_row),
    )
    row_qubits = max(1, int(np.ceil(np.log2(dimension))))
    local_qubits = max(1, int(np.ceil(np.log2(max(int(slots_per_row), 2)))))
    column_qubits = row_qubits
    value_register_qubits = int(value_precision_bits)

    row_start = 0
    local_start = row_start + row_qubits
    column_start = local_start + local_qubits
    value_register_start = column_start + column_qubits
    value_rotation_qubit = value_register_start + value_register_qubits
    total_qubits = value_rotation_qubit + 1

    circuit = QuantumCircuit(total_qubits, name="sparse_access_oracle_v2")
    controls = list(range(row_start, local_start)) + list(range(local_start, column_start))

    for entry in entries:
        pattern = _address_pattern(entry, row_qubits, local_qubits)
        for bit_position in range(column_qubits):
            if (entry.column >> bit_position) & 1:
                target = column_start + bit_position
                _with_control_pattern(
                    circuit,
                    controls,
                    pattern,
                    lambda target=target: circuit.mcx(controls, target),
                )
        for bit_position, bit in enumerate(entry.fixed_point_bits):
            if bit:
                target = value_register_start + bit_position
                _with_control_pattern(
                    circuit,
                    controls,
                    pattern,
                    lambda target=target: circuit.mcx(controls, target),
                )
        if abs(entry.rotation_angle) > 1.0e-15:
            _with_control_pattern(
                circuit,
                controls,
                pattern,
                lambda angle=entry.rotation_angle: circuit.mcry(
                    angle,
                    controls,
                    value_rotation_qubit,
                    None,
                ),
            )
    return SparseOracleV2Circuit(
        circuit=circuit,
        entries=entries,
        row_qubits=row_qubits,
        local_index_qubits=local_qubits,
        column_qubits=column_qubits,
        value_register_qubits=value_register_qubits,
        value_rotation_qubit=value_rotation_qubit,
        beta=beta,
    )


def validate_sparse_oracle_v2_circuit(prototype: SparseOracleV2Circuit) -> dict[str, Any]:
    """Statevector check: every address loads the correct column, value bits, angle."""

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for sparse-oracle v2 validation") from exc

    circuit = prototype.circuit
    row_qubits = prototype.row_qubits
    local_qubits = prototype.local_index_qubits
    column_qubits = prototype.column_qubits
    column_start = row_qubits + local_qubits
    value_register_start = column_start + column_qubits
    value_rotation_qubit = prototype.value_rotation_qubit

    entry_rows: list[dict[str, Any]] = []
    max_column_error = 0
    max_value_register_error = 0
    max_value_probability_error = 0.0
    for entry in prototype.entries:
        init = QuantumCircuit(circuit.num_qubits)
        for bit in range(row_qubits):
            if (entry.row >> bit) & 1:
                init.x(bit)
        if entry.local_index & 1 and local_qubits >= 1:
            init.x(row_qubits)
        state = Statevector.from_instruction(init.compose(circuit))
        probabilities = np.asarray(state.probabilities(), dtype=np.float64)

        column_probs = np.zeros(1 << column_qubits, dtype=np.float64)
        register_probs = np.zeros(prototype.value_register_qubits, dtype=np.float64)
        value_one_probability = 0.0
        for basis_index, probability in enumerate(probabilities):
            if probability <= 0.0:
                continue
            column = sum(
                ((basis_index >> (column_start + b)) & 1) << b for b in range(column_qubits)
            )
            column_probs[column] += probability
            for b in range(prototype.value_register_qubits):
                if (basis_index >> (value_register_start + b)) & 1:
                    register_probs[b] += probability
            if (basis_index >> value_rotation_qubit) & 1:
                value_one_probability += probability

        observed_column = int(np.argmax(column_probs))
        observed_register = tuple(round(p) for p in register_probs)
        target_value_probability = float(np.sin(entry.rotation_angle / 2.0) ** 2)
        column_error = int(observed_column != entry.column)
        register_error = int(observed_register != entry.fixed_point_bits)
        value_error = abs(value_one_probability - target_value_probability)
        max_column_error = max(max_column_error, column_error)
        max_value_register_error = max(max_value_register_error, register_error)
        max_value_probability_error = max(max_value_probability_error, value_error)
        entry_rows.append(
            {
                **entry.to_dict(),
                "observed_column": observed_column,
                "column_success_probability": float(column_probs[entry.column]),
                "observed_value_register": list(observed_register),
                "value_register_correct": bool(register_error == 0),
                "value_one_probability": float(value_one_probability),
                "target_value_one_probability": target_value_probability,
                "value_probability_abs_error": float(value_error),
            }
        )
    passed = bool(
        max_column_error == 0
        and max_value_register_error == 0
        and max_value_probability_error < VALIDATION_TOLERANCE
    )
    return {
        "entries": entry_rows,
        "max_column_lookup_error": int(max_column_error),
        "max_value_register_error": int(max_value_register_error),
        "max_value_probability_error": float(max_value_probability_error),
        "validation_passed": passed,
        "limitation": CLAIM,
    }


def count_controlled_rotations(circuit: Any) -> int:
    return int(
        sum(
            1
            for instruction in circuit.data
            if instruction.operation.name.lower() in _CONTROLLED_ROTATION_NAMES
        )
    )


def sparse_oracle_resource_summary(
    prototype: SparseOracleV2Circuit,
    *,
    basis_gates: list[str],
) -> dict[str, Any]:
    circuit = prototype.circuit
    transpiled, qasm_text, transpile_summary = _transpile_and_export_qasm(
        circuit, basis_gates=basis_gates
    )
    return {
        "n_qubits": int(circuit.num_qubits),
        "row_qubits": int(prototype.row_qubits),
        "local_index_qubits": int(prototype.local_index_qubits),
        "column_qubits": int(prototype.column_qubits),
        "value_register_qubits": int(prototype.value_register_qubits),
        "circuit_depth_before_transpile": int(circuit.depth()),
        "circuit_depth_after_transpile": (None if transpiled is None else int(transpiled.depth())),
        "controlled_rotation_count": count_controlled_rotations(circuit),
        "two_qubit_gate_count_before_transpile": _two_qubit_count(circuit),
        "two_qubit_gate_count_after_transpile": int(
            transpile_summary["two_qubit_gate_count_after_transpile"]
        ),
        "operation_counts_before_transpile": {
            str(key): int(value) for key, value in circuit.count_ops().items()
        },
        "operation_counts_after_transpile": transpile_summary["operation_counts_after_transpile"],
        "transpile_succeeded": bool(transpiled is not None),
        "qasm_text": qasm_text,
    }


def dense_block_encoding_resource_estimate(
    matrix: np.ndarray,
    *,
    value_precision_bits: int,
) -> dict[str, Any]:
    """Resource estimate for a dense ``(2N)x(2N)`` canonical block encoding.

    A dense block encoding of an ``N x N`` matrix needs an arbitrary
    ``2N x 2N`` unitary. The standard Clifford+T arbitrary-unitary cost scales as
    ``O(M^2)`` two-qubit gates for an ``M``-dimensional unitary (Shende-Bullock-
    Markov), giving an explicit, audited comparison point against the sparse
    oracle's address-multiplexed cost.
    """

    values = np.asarray(matrix, dtype=np.float64)
    dimension = int(values.shape[0])
    encoded_dimension = 2 * dimension
    qubits = int(np.ceil(np.log2(max(encoded_dimension, 2))))
    # Quantum Shannon decomposition: ~ (M^2 - M) CNOTs for an M-dim unitary (M=2^qubits).
    matrix_dimension = 1 << qubits
    two_qubit_gates = int(matrix_dimension * matrix_dimension - matrix_dimension)
    controlled_rotations = int(0.5 * matrix_dimension * matrix_dimension)
    return {
        "encoding_type": "dense_canonical_block_encoding",
        "matrix_dimension": dimension,
        "encoded_unitary_dimension": encoded_dimension,
        "n_qubits": qubits,
        "two_qubit_gate_count": two_qubit_gates,
        "controlled_rotation_count": controlled_rotations,
        "value_precision_bits": int(value_precision_bits),
        "note": (
            "Dense block encoding loads the full unitary; cost scales with the "
            "square of the encoded dimension and ignores sparsity."
        ),
    }


def precision_sweep_rows(
    matrix: np.ndarray,
    *,
    value_precision_bits: list[int],
    slots_per_row: int = 2,
) -> list[dict[str, Any]]:
    """Quantify value-loading (quantization) error as a function of precision."""

    rows: list[dict[str, Any]] = []
    for bits in sorted({int(b) for b in value_precision_bits}):
        entries, beta = sparse_oracle_v2_entries(
            matrix, value_precision_bits=bits, slots_per_row=int(slots_per_row)
        )
        active = [e for e in entries if abs(e.value) > 1.0e-12]
        normalized_errors = np.asarray([e.quantization_abs_error for e in active], dtype=np.float64)
        physical_errors = normalized_errors * beta
        rows.append(
            {
                "value_precision_bits": bits,
                "active_entries": len(active),
                "fixed_point_step": float(2.0 ** -(bits - 1)),
                "max_normalized_value_loading_error": float(
                    np.max(normalized_errors) if normalized_errors.size else 0.0
                ),
                "mean_normalized_value_loading_error": float(
                    np.mean(normalized_errors) if normalized_errors.size else 0.0
                ),
                "max_physical_value_loading_error": float(
                    np.max(physical_errors) if physical_errors.size else 0.0
                ),
                "mean_physical_value_loading_error": float(
                    np.mean(physical_errors) if physical_errors.size else 0.0
                ),
                "normalization_beta": float(beta),
            }
        )
    return rows


def run_sparse_oracle_prototype_strengthening(config: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": "outputs/qsvt_sparse_oracle_prototype_strengthening",
        "case": "ieee14",
        "model": "dc_linearized",
        "case_source": "pypower",
        "matrix_size": 4,
        "slots_per_row": 2,
        "value_precision_bits": [4, 8, 12],
        "basis_gates": ["u3", "cx"],
        "seed": 123,
    }
    resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    matrix_size = int(resolved["matrix_size"])
    slots = int(resolved["slots_per_row"])
    precisions = [int(b) for b in resolved["value_precision_bits"]]
    basis_gates = list(resolved["basis_gates"])

    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=matrix_size,
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    dense_matrix = np.asarray(subproblem.H_tilde, dtype=np.float64)
    sparse_matrix = sparsify_block(dense_matrix, keep_per_row=slots)
    sparse_stats = matrix_sparse_stats(sparse_matrix)

    sweep_rows = precision_sweep_rows(
        sparse_matrix, value_precision_bits=precisions, slots_per_row=slots
    )

    resource_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {
        "case": str(resolved["case"]),
        "model": str(resolved["model"]),
        "matrix_size": matrix_size,
        "sparse_density": float(sparse_stats["density"]),
        "sparse_nnz": int(sparse_stats["nnz"]),
        "per_precision": {},
        "limitation": CLAIM,
    }
    qasm_text = ""
    overall_validation_passed = True
    for bits in precisions:
        prototype = build_sparse_oracle_v2_circuit(
            sparse_matrix, value_precision_bits=bits, slots_per_row=slots
        )
        bit_validation = validate_sparse_oracle_v2_circuit(prototype)
        overall_validation_passed = (
            overall_validation_passed and bit_validation["validation_passed"]
        )
        resource = sparse_oracle_resource_summary(prototype, basis_gates=basis_gates)
        if bits == precisions[-1]:
            qasm_text = resource.pop("qasm_text")
        else:
            resource.pop("qasm_text", None)
        sweep_for_bits = next(r for r in sweep_rows if r["value_precision_bits"] == bits)
        resource_rows.append(
            {
                "case": str(resolved["case"]),
                "matrix_shape": f"{matrix_size}x{matrix_size}",
                "value_precision_bits": bits,
                "n_qubits": resource["n_qubits"],
                "row_qubits": resource["row_qubits"],
                "column_qubits": resource["column_qubits"],
                "value_register_qubits": resource["value_register_qubits"],
                "controlled_rotation_count": resource["controlled_rotation_count"],
                "two_qubit_gate_count_before_transpile": resource[
                    "two_qubit_gate_count_before_transpile"
                ],
                "two_qubit_gate_count_after_transpile": resource[
                    "two_qubit_gate_count_after_transpile"
                ],
                "circuit_depth_before_transpile": resource["circuit_depth_before_transpile"],
                "circuit_depth_after_transpile": resource["circuit_depth_after_transpile"],
                "max_value_loading_error": sweep_for_bits["max_normalized_value_loading_error"],
                "lookup_validation_passed": bit_validation["validation_passed"],
                "transpile_succeeded": resource["transpile_succeeded"],
                "resource_status": "toy_prototype_gate_level",
            }
        )

        dense = dense_block_encoding_resource_estimate(sparse_matrix, value_precision_bits=bits)
        comparison_rows.append(
            {
                "case": str(resolved["case"]),
                "matrix_shape": f"{matrix_size}x{matrix_size}",
                "value_precision_bits": bits,
                "sparse_n_qubits": resource["n_qubits"],
                "dense_n_qubits": dense["n_qubits"],
                "sparse_controlled_rotation_count": resource["controlled_rotation_count"],
                "dense_controlled_rotation_count": dense["controlled_rotation_count"],
                "sparse_two_qubit_gate_count_after_transpile": resource[
                    "two_qubit_gate_count_after_transpile"
                ],
                "dense_two_qubit_gate_count": dense["two_qubit_gate_count"],
                "two_qubit_gate_ratio_sparse_over_dense": float(
                    resource["two_qubit_gate_count_after_transpile"]
                    / max(dense["two_qubit_gate_count"], 1)
                ),
                "sparse_density": float(sparse_stats["density"]),
                "encoding_comparison": "sparse_access_oracle_vs_dense_block_encoding",
            }
        )
        validation["per_precision"][str(bits)] = {
            "validation_passed": bit_validation["validation_passed"],
            "max_column_lookup_error": bit_validation["max_column_lookup_error"],
            "max_value_register_error": bit_validation["max_value_register_error"],
            "max_value_probability_error": bit_validation["max_value_probability_error"],
            "max_value_loading_error": sweep_for_bits["max_normalized_value_loading_error"],
        }
    validation["overall_validation_passed"] = bool(overall_validation_passed)

    artifacts = write_sparse_oracle_strengthening_outputs(
        output_dir,
        resolved,
        resource_rows=resource_rows,
        sweep_rows=sweep_rows,
        comparison_rows=comparison_rows,
        validation=validation,
        qasm_text=qasm_text,
        sparse_stats=sparse_stats,
    )
    return {
        "output_dir": output_dir,
        "resource_rows": resource_rows,
        "sweep_rows": sweep_rows,
        "comparison_rows": comparison_rows,
        "validation": validation,
        "artifacts": artifacts,
    }


def write_sparse_oracle_strengthening_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    resource_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    qasm_text: str,
    sparse_stats: dict[str, Any],
) -> dict[str, Path]:
    resource_path = output_dir / "sparse_oracle_v2_resource_summary.csv"
    sweep_path = output_dir / "sparse_oracle_precision_sweep.csv"
    comparison_path = output_dir / "sparse_vs_dense_resource_comparison.csv"
    validation_path = output_dir / "sparse_oracle_validation.json"
    qasm_path = output_dir / "sparse_oracle_v2.qasm"
    interpretation_path = output_dir / "sparse_oracle_strengthening_interpretation.md"

    pd.DataFrame(resource_rows).to_csv(resource_path, index=False)
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    write_json(validation_path, validation)
    qasm_path.write_text(qasm_text, encoding="utf-8")
    interpretation_path.write_text(
        _interpretation_markdown(
            resolved,
            resource_rows=resource_rows,
            sweep_rows=sweep_rows,
            comparison_rows=comparison_rows,
            validation=validation,
            sparse_stats=sparse_stats,
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(
        output_dir,
        artifacts={
            "sparse_oracle_v2_resource_summary": str(resource_path),
            "sparse_oracle_precision_sweep": str(sweep_path),
            "sparse_vs_dense_resource_comparison": str(comparison_path),
            "sparse_oracle_validation": str(validation_path),
            "sparse_oracle_v2_qasm": str(qasm_path),
            "sparse_oracle_strengthening_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CLAIM,
    )
    return {
        "manifest": manifest,
        "sparse_oracle_v2_resource_summary": resource_path,
        "sparse_oracle_precision_sweep": sweep_path,
        "sparse_vs_dense_resource_comparison": comparison_path,
        "sparse_oracle_validation": validation_path,
        "sparse_oracle_v2_qasm": qasm_path,
        "sparse_oracle_strengthening_interpretation": interpretation_path,
    }


def _address_pattern(
    entry: SparseOracleV2Entry, row_qubits: int, local_qubits: int
) -> tuple[int, ...]:
    row_bits = tuple((entry.row >> bit) & 1 for bit in range(row_qubits))
    local_bits = tuple((entry.local_index >> bit) & 1 for bit in range(local_qubits))
    return (*row_bits, *local_bits)


def _interpretation_markdown(
    resolved: dict[str, Any],
    *,
    resource_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    sparse_stats: dict[str, Any],
) -> str:
    ordered_sweep = sorted(sweep_rows, key=lambda r: r["value_precision_bits"])
    finest = ordered_sweep[-1]
    coarsest = ordered_sweep[0]
    finest_resource = max(resource_rows, key=lambda r: r["value_precision_bits"])
    finest_comparison = max(comparison_rows, key=lambda r: r["value_precision_bits"])
    sweep_lines = [
        f"- {row['value_precision_bits']} bits: "
        f"max normalized value-loading error {row['max_normalized_value_loading_error']:.3e}, "
        f"mean {row['mean_normalized_value_loading_error']:.3e}"
        for row in ordered_sweep
    ]
    return "\n".join(
        [
            "# Strengthened Sparse-Oracle Prototype and Cost Tightening",
            "",
            CLAIM,
            "",
            "## Evidence Categories (must be distinguished)",
            "1. Toy sparse-oracle prototype (implemented here): a reversible "
            f"{resolved['matrix_size']}x{resolved['matrix_size']} sparse-access "
            "oracle on an IEEE-derived, row-thresholded block, with a fixed-point "
            "value register, multiplexed value rotations, statevector lookup "
            "validation, and transpiled gate counts.",
            "2. Software sparse-access abstraction (existing): classical CSR/CSC "
            "row/column/value access and matrix-free matvec for resource modelling.",
            "3. Full IEEE-scale oracle-model estimate (existing cost model): "
            "asymptotic sparse-access, value-loading, amplification, and readout "
            "cost rows under stated assumptions; not synthesized.",
            "4. MISSING: full hardware-native sparse-oracle synthesis. No "
            "IEEE-scale sparse oracle has been compiled into a calibrated, "
            "hardware-native circuit, and no hardware execution is claimed.",
            "",
            "## Implemented Strengthening Improvements",
            "- (a) Fixed-point value register with multiplexed value rotations.",
            "- (b) Lookup validation on an IEEE-derived sparsified block.",
            "- (c) Precision sweep over the requested value-precision bits.",
            "- (d) Controlled-rotation and two-qubit gate resource counts via transpile.",
            "- (e) QASM export of the finest-precision oracle circuit.",
            "- (f) Sparse-vs-dense block-encoding resource comparison.",
            "",
            "## IEEE-Derived Sparse Block",
            f"- Case/model: {resolved['case']} / {resolved['model']}",
            f"- Sparse block shape: {sparse_stats['matrix_shape'][0]}x"
            f"{sparse_stats['matrix_shape'][1]}",
            f"- Nonzeros after thresholding: {sparse_stats['nnz']} "
            f"(density {sparse_stats['density']:.3f})",
            f"- Max nonzeros per row: {sparse_stats['max_row_sparsity']}",
            "",
            "## Value-Loading Precision Sweep",
            *sweep_lines,
            f"- Error is non-increasing from {coarsest['value_precision_bits']} to "
            f"{finest['value_precision_bits']} bits "
            f"({coarsest['max_normalized_value_loading_error']:.3e} -> "
            f"{finest['max_normalized_value_loading_error']:.3e}); finer fixed-point "
            "grids reduce value-loading error.",
            "",
            "## Resource Counts (finest precision)",
            f"- Value precision: {finest_resource['value_precision_bits']} bits",
            f"- Qubits: {finest_resource['n_qubits']}",
            f"- Controlled rotations (pre-transpile): "
            f"{finest_resource['controlled_rotation_count']}",
            f"- Two-qubit gates after transpile: "
            f"{finest_resource['two_qubit_gate_count_after_transpile']}",
            f"- Lookup validation passed: {finest_resource['lookup_validation_passed']}",
            "",
            "## Sparse-vs-Dense Resource Comparison (finest precision)",
            f"- Sparse oracle qubits {finest_comparison['sparse_n_qubits']} vs dense "
            f"block-encoding qubits {finest_comparison['dense_n_qubits']}.",
            f"- Sparse two-qubit gates (transpiled) "
            f"{finest_comparison['sparse_two_qubit_gate_count_after_transpile']} vs dense "
            f"{finest_comparison['dense_two_qubit_gate_count']} "
            f"(ratio {finest_comparison['two_qubit_gate_ratio_sparse_over_dense']:.3e}).",
            "- The dense block-encoding cost grows with the square of the encoded "
            "dimension and ignores sparsity; the sparse-access oracle multiplexes "
            "only over nonzero entries. This is a prototype-scale comparison, NOT a "
            "speedup or advantage claim.",
            "",
            f"## Validation Status: "
            f"overall_validation_passed={validation['overall_validation_passed']}",
            "",
        ]
    )
