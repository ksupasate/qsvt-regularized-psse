from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json

TOY_SPARSE_GATE_CLAIM = (
    "The toy sparse-oracle gate demo implements reversible lookup and value-rotation "
    "structure on a 4x4 matrix only. It is not an IEEE-scale sparse oracle circuit."
)


@dataclass(frozen=True, slots=True)
class SparseOracleGateEntry:
    row: int
    local_index: int
    column: int
    value: float
    normalized_value: float
    rotation_angle: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "local_index": self.local_index,
            "column": self.column,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "rotation_angle": self.rotation_angle,
        }


def toy_sparse_matrix(matrix_size: int = 4) -> np.ndarray:
    if int(matrix_size) != 4:
        raise ValueError("only matrix_size=4 is currently implemented")
    return np.asarray(
        [
            [0.5, 0.0, 0.0, 0.1],
            [0.0, -0.4, 0.2, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2, 0.4],
        ],
        dtype=np.float64,
    )


def sparse_oracle_entries(matrix: np.ndarray) -> list[SparseOracleGateEntry]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4):
        raise ValueError("toy sparse-oracle gates require a 4x4 matrix")
    beta = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    entries: list[SparseOracleGateEntry] = []
    for row in range(values.shape[0]):
        columns = np.flatnonzero(np.abs(values[row]) > 1.0e-12)
        for local_index in range(2):
            if local_index < columns.size:
                column = int(columns[local_index])
                value = float(values[row, column])
            else:
                column = 0
                value = 0.0
            normalized = float(np.clip(value / beta, -1.0, 1.0))
            entries.append(
                SparseOracleGateEntry(
                    row=row,
                    local_index=local_index,
                    column=column,
                    value=value,
                    normalized_value=normalized,
                    rotation_angle=float(2.0 * np.arcsin(normalized)),
                )
            )
    return entries


def build_toy_sparse_oracle_gate_circuit(
    matrix_size: int = 4,
) -> tuple[Any, list[SparseOracleGateEntry]]:
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for sparse-oracle gate demos") from exc

    matrix = toy_sparse_matrix(matrix_size)
    entries = sparse_oracle_entries(matrix)
    # q0-q1: row, q2: local nonzero index, q3-q4: output column, q5: value rotation.
    circuit = QuantumCircuit(6, name="toy_sparse_access_oracle")
    controls = [0, 1, 2]
    column_targets = [3, 4]
    value_target = 5
    for entry in entries:
        pattern = (entry.row & 1, (entry.row >> 1) & 1, entry.local_index)
        column_bits = (entry.column & 1, (entry.column >> 1) & 1)
        for bit, target in zip(column_bits, column_targets, strict=True):
            if bit:
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
                    value_target,
                    None,
                ),
            )
    return circuit, entries


def validate_toy_sparse_oracle_gate_circuit(
    circuit: Any,
    entries: list[SparseOracleGateEntry],
) -> dict[str, Any]:
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("Qiskit is required for sparse-oracle validation") from exc

    entry_rows: list[dict[str, Any]] = []
    max_column_error = 0
    max_value_probability_error = 0.0
    for entry in entries:
        init = QuantumCircuit(circuit.num_qubits)
        if entry.row & 1:
            init.x(0)
        if (entry.row >> 1) & 1:
            init.x(1)
        if entry.local_index:
            init.x(2)
        state = Statevector.from_instruction(init.compose(circuit))
        probabilities = np.asarray(state.probabilities(), dtype=np.float64)
        column_probs = np.zeros(4, dtype=np.float64)
        value_one_probability = 0.0
        for basis_index, probability in enumerate(probabilities):
            column = ((basis_index >> 3) & 1) + 2 * ((basis_index >> 4) & 1)
            column_probs[column] += probability
            if (basis_index >> 5) & 1:
                value_one_probability += probability
        observed_column = int(np.argmax(column_probs))
        target_value_probability = float(np.sin(entry.rotation_angle / 2.0) ** 2)
        column_error = int(observed_column != entry.column)
        value_error = abs(value_one_probability - target_value_probability)
        max_column_error = max(max_column_error, column_error)
        max_value_probability_error = max(max_value_probability_error, value_error)
        entry_rows.append(
            {
                **entry.to_dict(),
                "observed_column": observed_column,
                "column_success_probability": float(column_probs[entry.column]),
                "value_one_probability": float(value_one_probability),
                "target_value_one_probability": target_value_probability,
                "value_probability_abs_error": float(value_error),
            }
        )
    return {
        "entries": entry_rows,
        "max_column_lookup_error": int(max_column_error),
        "max_value_probability_error": float(max_value_probability_error),
        "validation_passed": bool(max_column_error == 0 and max_value_probability_error < 1.0e-12),
        "limitation": TOY_SPARSE_GATE_CLAIM,
    }


def run_toy_sparse_oracle_gate_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/toy_sparse_oracle_gate_demo",
        "matrix_size": 4,
        "seed": 123,
        "basis_gates": ["u3", "cx"],
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    circuit, entries = build_toy_sparse_oracle_gate_circuit(int(resolved["matrix_size"]))
    validation = validate_toy_sparse_oracle_gate_circuit(circuit, entries)
    transpiled, qasm_text, resource_summary = _transpile_and_export_qasm(
        circuit,
        basis_gates=list(resolved["basis_gates"]),
    )
    summary = {
        "claim_boundary": TOY_SPARSE_GATE_CLAIM,
        "matrix_size": int(resolved["matrix_size"]),
        "n_qubits": int(circuit.num_qubits),
        "circuit_depth_before_transpile": int(circuit.depth()),
        "circuit_depth_after_transpile": None if transpiled is None else int(transpiled.depth()),
        "operation_counts_before_transpile": {
            str(key): int(value) for key, value in circuit.count_ops().items()
        },
        "operation_counts_after_transpile": resource_summary["operation_counts_after_transpile"],
        "two_qubit_gate_count_after_transpile": resource_summary[
            "two_qubit_gate_count_after_transpile"
        ],
        "oracle_structure": (
            "row register q0-q1, local-index q2, reversible column output q3-q4, "
            "value-rotation target q5"
        ),
        "entries": [entry.to_dict() for entry in entries],
        "limitation": TOY_SPARSE_GATE_CLAIM,
    }
    summary_path = output_dir / "oracle_circuit_summary.json"
    validation_path = output_dir / "oracle_validation.json"
    qasm_path = output_dir / "toy_sparse_oracle.qasm"
    write_json(summary_path, summary)
    write_json(validation_path, validation)
    qasm_path.write_text(qasm_text, encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "oracle_circuit_summary": str(summary_path),
            "oracle_validation": str(validation_path),
            "toy_sparse_oracle_qasm": str(qasm_path),
        },
        input_config=resolved,
        claim_boundary=TOY_SPARSE_GATE_CLAIM,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "validation": validation,
        "artifacts": {
            "manifest": manifest,
            "oracle_circuit_summary": summary_path,
            "oracle_validation": validation_path,
            "toy_sparse_oracle_qasm": qasm_path,
        },
    }


def _with_control_pattern(
    circuit: Any,
    controls: list[int],
    pattern: tuple[int, ...],
    operation: Any,
) -> None:
    zero_controls = [control for control, bit in zip(controls, pattern, strict=True) if bit == 0]
    for control in zero_controls:
        circuit.x(control)
    operation()
    for control in reversed(zero_controls):
        circuit.x(control)


def _transpile_and_export_qasm(
    circuit: Any,
    *,
    basis_gates: list[str],
) -> tuple[Any | None, str, dict[str, Any]]:
    try:
        from qiskit import qasm2, transpile  # type: ignore[import-not-found]

        transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
        qasm_text = str(qasm2.dumps(transpiled))
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        return (
            transpiled,
            qasm_text,
            {
                "operation_counts_after_transpile": counts,
                "two_qubit_gate_count_after_transpile": _two_qubit_count(transpiled),
            },
        )
    except Exception as exc:
        return (
            None,
            "\n".join(
                [
                    "// QASM export failed for this Qiskit version/circuit.",
                    f"// Reason: {type(exc).__name__}: {exc}",
                    str(circuit.draw(output="text")),
                ]
            ),
            {
                "operation_counts_after_transpile": {},
                "two_qubit_gate_count_after_transpile": 0,
            },
        )


def _two_qubit_count(circuit: Any) -> int:
    return int(sum(1 for instruction in circuit.data if len(instruction.qubits) == 2))
