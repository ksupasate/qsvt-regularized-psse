"""Task 3: reversible sparse-access oracle prototype + formal gate-cost model.

Two reversible lookup oracles for a small weighted-Jacobian block are *synthesized
and statevector-validated*:

* column oracle ``O_col: |i, k, z> -> |i, k, z xor c(i,k)>`` writing the column
  index of the ``k``-th stored nonzero of row ``i`` (plus an ``invalid`` flag for
  padding slots),
* value oracle  ``O_val: |i, j, z> -> |i, j, z xor fix(H_ij)>`` writing a
  sign + ``p``-bit fixed-point magnitude of the matrix entry.

Both are permutations on the computational basis (multi-controlled-X writes), so a
basis-address input maps to a single basis output that is decoded and checked
against the matrix exactly (up to the documented fixed-point quantization).

For sizes that are not synthesized, a *formal resource model* gives explicit
qubit-register widths and Toffoli/T-count formulas instantiated for the selected
block and for IEEE 14/30/57/118/300. The synthesized circuits are small-scale
reversible-arithmetic evidence; they are not an IEEE-scale oracle compilation and
not a quantum-hardware run.
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
    ORACLE_DIR,
    assert_safe,
    write_demo_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_VALUE_BITS = 6
DEFAULT_IEEE_CASES = ("ieee14", "ieee30", "ieee57", "ieee118", "ieee300")
TOFFOLI_T_COST = 7  # T gates per Toffoli in a Clifford+T decomposition.

ORACLE_RESOURCE_COLUMNS = [
    "case",
    "status",
    "matrix_shape",
    "nnz",
    "max_nonzeros_per_row",
    "row_index_qubits",
    "slot_index_qubits",
    "column_register_qubits",
    "value_magnitude_qubits",
    "value_sign_qubits",
    "value_precision_bits",
    "sign_representation",
    "work_qubits",
    "uncomputation_required",
    "col_oracle_toffoli_qrom",
    "col_oracle_toffoli_naive",
    "val_oracle_toffoli_qrom",
    "val_oracle_toffoli_naive",
    "total_t_count_qrom",
    "col_oracle_synth_gate_count",
    "col_oracle_synth_depth",
    "val_oracle_synth_gate_count",
    "val_oracle_synth_depth",
    "invalid_index_behavior",
]


def _bits_for(count: int) -> int:
    return max(1, math.ceil(math.log2(max(int(count), 2))))


@dataclass(frozen=True, slots=True)
class RegisterLayout:
    row_qubits: int
    slot_qubits: int
    column_qubits: int
    value_magnitude_qubits: int
    value_sign_qubits: int
    invalid_qubits: int


def _row_nonzeros(matrix: np.ndarray, tolerance: float = 1.0e-12) -> list[np.ndarray]:
    return [np.flatnonzero(np.abs(matrix[i]) > tolerance) for i in range(matrix.shape[0])]


def column_map(matrix: np.ndarray, max_nonzeros: int) -> dict[tuple[int, int], tuple[int, bool]]:
    """Map ``(row, slot) -> (column, valid)`` for the stored nonzeros of each row."""

    nonzeros = _row_nonzeros(matrix)
    mapping: dict[tuple[int, int], tuple[int, bool]] = {}
    for row, columns in enumerate(nonzeros):
        for slot in range(max_nonzeros):
            if slot < columns.size:
                mapping[(row, slot)] = (int(columns[slot]), True)
            else:
                mapping[(row, slot)] = (0, False)  # padding slot -> invalid flag set
    return mapping


def quantize_value(value: float, scale: float, magnitude_bits: int) -> tuple[int, int]:
    """Return ``(sign_bit, magnitude_int)`` for the sign-magnitude fixed-point encoding."""

    levels = (1 << magnitude_bits) - 1
    sign_bit = 1 if value < 0.0 else 0
    magnitude = round(abs(value) / scale * levels) if scale > 0 else 0
    magnitude = int(np.clip(magnitude, 0, levels))
    return sign_bit, magnitude


def decode_value(sign_bit: int, magnitude_int: int, scale: float, magnitude_bits: int) -> float:
    levels = (1 << magnitude_bits) - 1
    sign = -1.0 if sign_bit else 1.0
    return sign * (magnitude_int / levels) * scale if levels > 0 else 0.0


def _apply_controlled_writes(
    circuit: Any, controls: list[int], pattern: list[int], targets: list[int]
) -> None:
    """Write ``X`` on each target, controlled on ``controls`` matching ``pattern``."""

    zero_controls = [c for c, bit in zip(controls, pattern, strict=True) if bit == 0]
    for control in zero_controls:
        circuit.x(control)
    for target in targets:
        circuit.mcx(controls, target)
    for control in reversed(zero_controls):
        circuit.x(control)


def build_column_oracle(matrix: np.ndarray, max_nonzeros: int) -> dict[str, Any]:
    """Synthesize the reversible column oracle ``O_col`` for a small matrix."""

    from qiskit import QuantumCircuit, QuantumRegister  # type: ignore[import-not-found]

    rows, cols = matrix.shape
    row_q = _bits_for(rows)
    slot_q = _bits_for(max_nonzeros)
    col_q = _bits_for(cols)
    qr_row = QuantumRegister(row_q, "row")
    qr_slot = QuantumRegister(slot_q, "slot")
    qr_col = QuantumRegister(col_q, "col")
    qr_invalid = QuantumRegister(1, "invalid")
    circuit = QuantumCircuit(qr_row, qr_slot, qr_col, qr_invalid, name="O_col")
    controls = list(range(row_q + slot_q))
    col_targets = list(range(row_q + slot_q, row_q + slot_q + col_q))
    invalid_target = row_q + slot_q + col_q

    mapping = column_map(matrix, max_nonzeros)
    for (row, slot), (column, valid) in mapping.items():
        pattern = [(row >> b) & 1 for b in range(row_q)] + [(slot >> b) & 1 for b in range(slot_q)]
        if valid:
            bits = [col_targets[b] for b in range(col_q) if (column >> b) & 1]
            if bits:
                _apply_controlled_writes(circuit, controls, pattern, bits)
        else:
            _apply_controlled_writes(circuit, controls, pattern, [invalid_target])

    offsets = {"row": 0, "slot": row_q, "col": row_q + slot_q, "invalid": row_q + slot_q + col_q}
    return {
        "circuit": circuit,
        "mapping": mapping,
        "offsets": offsets,
        "widths": {"row": row_q, "slot": slot_q, "col": col_q, "invalid": 1},
        "max_nonzeros": int(max_nonzeros),
    }


def build_value_oracle(matrix: np.ndarray, magnitude_bits: int) -> dict[str, Any]:
    """Synthesize the reversible value oracle ``O_val`` for a small matrix."""

    from qiskit import QuantumCircuit, QuantumRegister  # type: ignore[import-not-found]

    rows, cols = matrix.shape
    row_q = _bits_for(rows)
    col_q = _bits_for(cols)
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).eps)
    qr_row = QuantumRegister(row_q, "row")
    qr_col = QuantumRegister(col_q, "col")
    qr_mag = QuantumRegister(magnitude_bits, "mag")
    qr_sign = QuantumRegister(1, "sign")
    circuit = QuantumCircuit(qr_row, qr_col, qr_mag, qr_sign, name="O_val")
    controls = list(range(row_q + col_q))
    mag_targets = list(range(row_q + col_q, row_q + col_q + magnitude_bits))
    sign_target = row_q + col_q + magnitude_bits

    encoding: dict[tuple[int, int], tuple[int, int]] = {}
    for row in range(rows):
        for col in range(cols):
            value = float(matrix[row, col])
            sign_bit, magnitude = quantize_value(value, scale, magnitude_bits)
            encoding[(row, col)] = (sign_bit, magnitude)
            if magnitude == 0 and sign_bit == 0:
                continue
            pattern = [(row >> b) & 1 for b in range(row_q)]
            pattern += [(col >> b) & 1 for b in range(col_q)]
            targets = [mag_targets[b] for b in range(magnitude_bits) if (magnitude >> b) & 1]
            if sign_bit:
                targets = [*targets, sign_target]
            if targets:
                _apply_controlled_writes(circuit, controls, pattern, targets)

    offsets = {
        "row": 0,
        "col": row_q,
        "mag": row_q + col_q,
        "sign": row_q + col_q + magnitude_bits,
    }
    return {
        "circuit": circuit,
        "encoding": encoding,
        "scale": scale,
        "magnitude_bits": int(magnitude_bits),
        "offsets": offsets,
        "widths": {"row": row_q, "col": col_q, "mag": magnitude_bits, "sign": 1},
    }


def _decode_basis_index(index: int, offset: int, width: int) -> int:
    return (int(index) >> int(offset)) & ((1 << int(width)) - 1)


def _oracle_output_index(init_bits: dict[int, int], circuit: Any) -> int:
    from qiskit import QuantumCircuit  # type: ignore[import-not-found]
    from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]

    init = QuantumCircuit(circuit.num_qubits)
    for qubit, bit in init_bits.items():
        if bit:
            init.x(qubit)
    state = Statevector.from_instruction(init.compose(circuit))
    probabilities = np.asarray(state.probabilities(), dtype=np.float64)
    return int(np.argmax(probabilities))


def validate_column_oracle(built: dict[str, Any]) -> dict[str, Any]:
    circuit = built["circuit"]
    offsets = built["offsets"]
    widths = built["widths"]
    rows: list[dict[str, Any]] = []
    max_error = 0
    for (row, slot), (column, valid) in built["mapping"].items():
        init_bits: dict[int, int] = {}
        for b in range(widths["row"]):
            if (row >> b) & 1:
                init_bits[offsets["row"] + b] = 1
        for b in range(widths["slot"]):
            if (slot >> b) & 1:
                init_bits[offsets["slot"] + b] = 1
        out_index = _oracle_output_index(init_bits, circuit)
        observed_col = _decode_basis_index(out_index, offsets["col"], widths["col"])
        observed_invalid = _decode_basis_index(out_index, offsets["invalid"], 1)
        expected_col = column if valid else 0
        expected_invalid = 0 if valid else 1
        error = int(observed_col != expected_col or observed_invalid != expected_invalid)
        max_error = max(max_error, error)
        rows.append(
            {
                "row": row,
                "slot": slot,
                "expected_column": expected_col,
                "observed_column": observed_col,
                "valid": valid,
                "observed_invalid_flag": observed_invalid,
                "match": error == 0,
            }
        )
    return {"entries": rows, "max_lookup_error": int(max_error), "passed": bool(max_error == 0)}


def validate_value_oracle(built: dict[str, Any]) -> dict[str, Any]:
    circuit = built["circuit"]
    offsets = built["offsets"]
    widths = built["widths"]
    scale = built["scale"]
    magnitude_bits = built["magnitude_bits"]
    rows: list[dict[str, Any]] = []
    max_decode_error = 0.0
    quantization = scale / ((1 << magnitude_bits) - 1)
    for (row, col), (sign_bit, magnitude) in built["encoding"].items():
        init_bits: dict[int, int] = {}
        for b in range(widths["row"]):
            if (row >> b) & 1:
                init_bits[offsets["row"] + b] = 1
        for b in range(widths["col"]):
            if (col >> b) & 1:
                init_bits[offsets["col"] + b] = 1
        out_index = _oracle_output_index(init_bits, circuit)
        observed_mag = _decode_basis_index(out_index, offsets["mag"], widths["mag"])
        observed_sign = _decode_basis_index(out_index, offsets["sign"], 1)
        decoded = decode_value(observed_sign, observed_mag, scale, magnitude_bits)
        true_value = decode_value(sign_bit, magnitude, scale, magnitude_bits)
        decode_error = abs(decoded - true_value)
        max_decode_error = max(max_decode_error, decode_error)
        rows.append(
            {
                "row": row,
                "col": col,
                "expected_sign": sign_bit,
                "observed_sign": observed_sign,
                "expected_magnitude": magnitude,
                "observed_magnitude": observed_mag,
                "decoded_value": decoded,
                "match": bool(observed_sign == sign_bit and observed_mag == magnitude),
            }
        )
    bit_exact = all(entry["match"] for entry in rows)
    return {
        "entries": rows,
        "max_decode_error": float(max_decode_error),
        "quantization_step": float(quantization),
        "bit_exact": bool(bit_exact),
        "passed": bool(bit_exact),
    }


def _circuit_resource(circuit: Any) -> dict[str, int]:
    return {
        "n_qubits": int(circuit.num_qubits),
        "gate_count": int(sum(circuit.count_ops().values())),
        "depth": int(circuit.depth()),
    }


def formal_cost_model(
    *,
    case: str,
    rows: int,
    cols: int,
    nnz: int,
    max_nonzeros_per_row: int,
    value_bits: int,
) -> dict[str, Any]:
    """Explicit register widths and Toffoli/T-count formulas for the oracle pair."""

    row_q = _bits_for(rows)
    slot_q = _bits_for(max_nonzeros_per_row)
    col_q = _bits_for(cols)

    # QROM (unary-iteration) bound: one Toffoli per stored address.
    col_addresses = rows * max_nonzeros_per_row
    val_addresses = nnz  # sparse access reads only stored nonzeros
    col_toffoli_qrom = col_addresses
    val_toffoli_qrom = val_addresses

    # Naive multiplexed bound: each written output bit is a multi-controlled-X.
    col_address_qubits = row_q + slot_q
    val_address_qubits = row_q + col_q
    col_toffoli_naive = col_addresses * col_q * max(1, 2 * (col_address_qubits - 1))
    val_toffoli_naive = val_addresses * (value_bits + 1) * max(1, 2 * (val_address_qubits - 1))

    total_t_qrom = TOFFOLI_T_COST * (col_toffoli_qrom + val_toffoli_qrom)
    return {
        "row_index_qubits": row_q,
        "slot_index_qubits": slot_q,
        "column_register_qubits": col_q,
        "value_magnitude_qubits": value_bits,
        "value_sign_qubits": 1,
        "value_precision_bits": value_bits,
        "sign_representation": "sign_magnitude_one_sign_bit",
        "work_qubits": "unary-iteration ancilla register (modeled, O(address_qubits))",
        "uncomputation_required": True,
        "col_oracle_toffoli_qrom": int(col_toffoli_qrom),
        "col_oracle_toffoli_naive": int(col_toffoli_naive),
        "val_oracle_toffoli_qrom": int(val_toffoli_qrom),
        "val_oracle_toffoli_naive": int(val_toffoli_naive),
        "total_t_count_qrom": int(total_t_qrom),
        "invalid_index_behavior": (
            "padding slots (slot >= nnz(row)) set a dedicated invalid flag; querying a "
            "structural zero of O_val returns the zero value"
        ),
    }


def _case_sparsity(case: str, seed: int) -> dict[str, Any] | None:
    try:
        system, _ = build_engineering_system(
            {
                "case_name": case,
                "case_source": "pypower",
                "matrix_source": "weighted_jacobian",
                "seed": int(seed),
            }
        )
    except Exception:
        return None
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    nonzero_mask = np.abs(matrix) > 1.0e-12
    per_row = nonzero_mask.sum(axis=1)
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "nnz": int(nonzero_mask.sum()),
        "max_nonzeros_per_row": int(per_row.max()) if per_row.size else 0,
    }


def run_reversible_sparse_oracle(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    value_bits = int(resolved["value_bits"])

    block = _load_or_build_block(resolved)
    matrix = block["matrix"]
    rows, cols = matrix.shape
    nonzero_mask = np.abs(matrix) > 1.0e-12
    max_nnz = int(nonzero_mask.sum(axis=1).max())
    nnz = int(nonzero_mask.sum())

    col_built = build_column_oracle(matrix, max_nnz)
    val_built = build_value_oracle(matrix, value_bits)
    col_validation = validate_column_oracle(col_built)
    val_validation = validate_value_oracle(val_built)
    col_resource = _circuit_resource(col_built["circuit"])
    val_resource = _circuit_resource(val_built["circuit"])

    resource_rows: list[dict[str, Any]] = []
    synth_model = formal_cost_model(
        case=f"{block['label']}_synthesized",
        rows=rows,
        cols=cols,
        nnz=nnz,
        max_nonzeros_per_row=max_nnz,
        value_bits=value_bits,
    )
    resource_rows.append(
        _resource_row(
            case=f"{block['label']}",
            status="synthesized_small_scale",
            rows=rows,
            cols=cols,
            nnz=nnz,
            max_nnz=max_nnz,
            model=synth_model,
            col_resource=col_resource,
            val_resource=val_resource,
        )
    )

    for case in resolved["ieee_cases"]:
        sparsity = _case_sparsity(case, int(resolved["seed"]))
        if sparsity is None:
            resource_rows.append(_unavailable_row(case))
            continue
        model = formal_cost_model(
            case=case,
            rows=sparsity["rows"],
            cols=sparsity["cols"],
            nnz=sparsity["nnz"],
            max_nonzeros_per_row=sparsity["max_nonzeros_per_row"],
            value_bits=value_bits,
        )
        resource_rows.append(
            _resource_row(
                case=case,
                status="modeled",
                rows=sparsity["rows"],
                cols=sparsity["cols"],
                nnz=sparsity["nnz"],
                max_nnz=sparsity["max_nonzeros_per_row"],
                model=model,
                col_resource=None,
                val_resource=None,
            )
        )

    artifacts = _write_oracle_outputs(
        output_dir=output_dir,
        resource_rows=resource_rows,
        col_validation=col_validation,
        val_validation=val_validation,
        block=block,
        col_resource=col_resource,
        val_resource=val_resource,
        value_bits=value_bits,
        val_scale=val_built["scale"],
    )
    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="reversible_sparse_oracle",
        description=(
            "Synthesized, statevector-validated reversible column/value oracles for a small "
            "weighted-Jacobian block, plus a formal gate-cost model for IEEE 14/30/57/118/300. "
            "Small-scale reversible-arithmetic evidence; not an IEEE-scale oracle compilation "
            "and not a quantum-hardware run."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[block["source"]],
        extra={
            "column_oracle_passed": col_validation["passed"],
            "value_oracle_bit_exact": val_validation["passed"],
            "value_precision_bits": value_bits,
            "synthesized_block": block["label"],
        },
        manifest_name="oracle_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "resource_summary": pd.DataFrame(resource_rows, columns=ORACLE_RESOURCE_COLUMNS),
        "column_validation": col_validation,
        "value_validation": val_validation,
        "artifacts": artifacts,
    }


def _resource_row(
    *,
    case: str,
    status: str,
    rows: int,
    cols: int,
    nnz: int,
    max_nnz: int,
    model: dict[str, Any],
    col_resource: dict[str, int] | None,
    val_resource: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "case": case,
        "status": status,
        "matrix_shape": f"{rows}x{cols}",
        "nnz": int(nnz),
        "max_nonzeros_per_row": int(max_nnz),
        "row_index_qubits": model["row_index_qubits"],
        "slot_index_qubits": model["slot_index_qubits"],
        "column_register_qubits": model["column_register_qubits"],
        "value_magnitude_qubits": model["value_magnitude_qubits"],
        "value_sign_qubits": model["value_sign_qubits"],
        "value_precision_bits": model["value_precision_bits"],
        "sign_representation": model["sign_representation"],
        "work_qubits": model["work_qubits"],
        "uncomputation_required": model["uncomputation_required"],
        "col_oracle_toffoli_qrom": model["col_oracle_toffoli_qrom"],
        "col_oracle_toffoli_naive": model["col_oracle_toffoli_naive"],
        "val_oracle_toffoli_qrom": model["val_oracle_toffoli_qrom"],
        "val_oracle_toffoli_naive": model["val_oracle_toffoli_naive"],
        "total_t_count_qrom": model["total_t_count_qrom"],
        "col_oracle_synth_gate_count": col_resource["gate_count"] if col_resource else np.nan,
        "col_oracle_synth_depth": col_resource["depth"] if col_resource else np.nan,
        "val_oracle_synth_gate_count": val_resource["gate_count"] if val_resource else np.nan,
        "val_oracle_synth_depth": val_resource["depth"] if val_resource else np.nan,
        "invalid_index_behavior": model["invalid_index_behavior"],
    }


def _unavailable_row(case: str) -> dict[str, Any]:
    row = {column: "" for column in ORACLE_RESOURCE_COLUMNS}
    row["case"] = case
    row["status"] = "unavailable"
    row["invalid_index_behavior"] = "case could not be built"
    return row


def _load_or_build_block(resolved: dict[str, Any]) -> dict[str, Any]:
    block_path = Path(resolved["block_npy"])
    if block_path.is_file():
        matrix = np.load(block_path)
        return {
            "matrix": np.asarray(matrix, dtype=np.float64),
            "label": f"selected_block_{matrix.shape[0]}x{matrix.shape[1]}",
            "source": str(block_path),
        }
    system, _ = build_engineering_system(
        {
            "case_name": resolved["case"],
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
    )
    size = int(resolved["block_size"])
    matrix, _r, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=size,
        col_count=size,
        policy="largest_row_col_norms",
    )
    return {
        "matrix": np.asarray(matrix, dtype=np.float64),
        "label": f"selected_block_{size}x{size}",
        "source": f"build_engineering_system:{resolved['case']}:weighted_jacobian",
    }


def _write_oracle_outputs(
    *,
    output_dir: Path,
    resource_rows: list[dict[str, Any]],
    col_validation: dict[str, Any],
    val_validation: dict[str, Any],
    block: dict[str, Any],
    col_resource: dict[str, int],
    val_resource: dict[str, int],
    value_bits: int,
    val_scale: float,
) -> dict[str, Path]:
    summary_csv = output_dir / "oracle_resource_summary.csv"
    pd.DataFrame(resource_rows, columns=ORACLE_RESOURCE_COLUMNS).to_csv(summary_csv, index=False)

    metadata_json = output_dir / "oracle_metadata.json"
    write_json(
        metadata_json,
        {
            "synthesized_block": block["label"],
            "block_source": block["source"],
            "value_precision_bits": value_bits,
            "value_scale": float(val_scale),
            "sign_representation": "sign_magnitude_one_sign_bit",
            "column_oracle": {
                "definition": "O_col: |i,k,z> -> |i,k,z xor c(i,k)> with invalid flag",
                "resource": col_resource,
                "validation": {
                    "passed": col_validation["passed"],
                    "max_lookup_error": col_validation["max_lookup_error"],
                    "n_entries": len(col_validation["entries"]),
                },
            },
            "value_oracle": {
                "definition": "O_val: |i,j,z> -> |i,j,z xor fix(H_ij)> sign-magnitude fixed point",
                "resource": val_resource,
                "validation": {
                    "bit_exact": val_validation["passed"],
                    "max_decode_error": val_validation["max_decode_error"],
                    "quantization_step": val_validation["quantization_step"],
                    "n_entries": len(val_validation["entries"]),
                },
            },
            "status_label": "synthesized small-scale + modeled (IEEE 14/30/57/118/300)",
            "claim_boundary": (
                "Small synthesized reversible lookup oracles validated by statevector; "
                "larger cases are a formal cost model only. Not an IEEE-scale oracle "
                "compilation and not a quantum-hardware run."
            ),
        },
    )

    validation_json = output_dir / "oracle_validation.json"
    write_json(
        validation_json,
        {"column_oracle": col_validation, "value_oracle": val_validation},
    )

    readme = output_dir / "README.md"
    readme.write_text(
        _oracle_readme(resource_rows, col_validation, val_validation, block, value_bits),
        encoding="utf-8",
    )

    return {
        "oracle_resource_summary_csv": summary_csv,
        "oracle_metadata_json": metadata_json,
        "oracle_validation_json": validation_json,
        "readme_md": readme,
    }


def _oracle_readme(
    resource_rows: list[dict[str, Any]],
    col_validation: dict[str, Any],
    val_validation: dict[str, Any],
    block: dict[str, Any],
    value_bits: int,
) -> str:
    lines = [
        "# Reversible Sparse-Access Oracle",
        "",
        "Synthesized, statevector-validated reversible **column** and **value** oracles for "
        f"the `{block['label']}` weighted-Jacobian block, plus a formal gate-cost model for "
        "IEEE 14/30/57/118/300. Small-scale reversible-arithmetic evidence; **not** an "
        "IEEE-scale oracle compilation and **not** a quantum-hardware run.",
        "",
        "## Oracle definitions",
        "",
        "- `O_col: |i, k, z> -> |i, k, z xor c(i,k)>` writes the column index of the k-th "
        "stored nonzero of row i, with a dedicated `invalid` flag for padding slots.",
        "- `O_val: |i, j, z> -> |i, j, z xor fix(H_ij)>` writes a sign bit + "
        f"{value_bits}-bit fixed-point magnitude of the entry.",
        "",
        "## Synthesized validation",
        "",
        f"- column oracle: {len(col_validation['entries'])} addresses, "
        f"max lookup error = {col_validation['max_lookup_error']}, passed = "
        f"{col_validation['passed']}",
        f"- value oracle: {len(val_validation['entries'])} entries, bit-exact = "
        f"{val_validation['passed']}, quantization step = "
        f"{val_validation['quantization_step']:.4e}",
        "",
        "## Resource summary (synthesized small-scale + modeled)",
        "",
        "| Case | Status | Shape | nnz | row q | slot q | col q | val q | "
        "O_col Toffoli (QROM) | O_val Toffoli (QROM) | total T (QROM) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in resource_rows:
        val_q = (
            ""
            if row["value_magnitude_qubits"] == ""
            else f"{row['value_magnitude_qubits']}+{row['value_sign_qubits']}"
        )
        lines.append(
            f"| {row['case']} | {row['status']} | {row['matrix_shape']} | {row['nnz']} | "
            f"{row['row_index_qubits']} | {row['slot_index_qubits']} | "
            f"{row['column_register_qubits']} | {val_q} | "
            f"{row['col_oracle_toffoli_qrom']} | {row['val_oracle_toffoli_qrom']} | "
            f"{row['total_t_count_qrom']} |"
        )
    lines += [
        "",
        "Toffoli/T counts are a **modeled** unary-iteration (QROM) bound (one Toffoli per "
        "stored address; T-count = 7 x Toffoli) plus a naive multiplexed bound in the CSV. "
        "The synthesized small block reports actual qiskit gate counts and depth.",
        "",
        "Invalid-index behavior: padding slots set the `invalid` flag; querying a structural "
        "zero of `O_val` returns the zero value. Uncomputation of address/work registers is "
        "required for use inside a block-encoding and is accounted as modeled.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(ORACLE_DIR),
        "block_npy": "outputs/selected_observable_qsvt_demo/selected_block.npy",
        "case": "ieee14",
        "seed": 123,
        "block_size": 4,
        "value_bits": DEFAULT_VALUE_BITS,
        "ieee_cases": list(DEFAULT_IEEE_CASES),
        "command": "run_reversible_sparse_oracle",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthesize + model the reversible sparse oracle")
    parser.add_argument("--output-dir", default=str(ORACLE_DIR))
    parser.add_argument(
        "--block-npy", default="outputs/selected_observable_qsvt_demo/selected_block.npy"
    )
    parser.add_argument("--case", default="ieee14")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=DEFAULT_VALUE_BITS)
    parser.add_argument("--ieee-cases", nargs="+", default=list(DEFAULT_IEEE_CASES))
    args = parser.parse_args(argv)
    run = run_reversible_sparse_oracle(
        {
            "output_dir": args.output_dir,
            "block_npy": args.block_npy,
            "case": args.case,
            "seed": args.seed,
            "block_size": args.block_size,
            "value_bits": args.value_bits,
            "ieee_cases": args.ieee_cases,
            "command": "scripts/run_reversible_sparse_oracle_resource.py " + " ".join(argv or []),
        }
    )
    print(f"Reversible sparse oracle complete: {run['artifacts']['oracle_resource_summary_csv']}")
    print(
        f"column oracle passed: {run['column_validation']['passed']}, "
        f"value oracle bit-exact: {run['value_validation']['passed']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
