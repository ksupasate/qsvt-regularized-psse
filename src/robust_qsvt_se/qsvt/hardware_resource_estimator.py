from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.scalable_block_encoding import (
    BlockEncodingModel,
    estimate_block_encoding_resources,
)
from robust_qsvt_se.qsvt.sparse_access_oracle import build_sparse_access_oracle
from robust_qsvt_se.qsvt.state_preparation_model import (
    StatePreparationModel,
    estimate_state_preparation,
)
from robust_qsvt_se.utils.io import ensure_directory

HARDWARE_RESOURCE_CLAIM = (
    "Hardware-resource rows are proxy diagnostics for a QSVT-compatible "
    "implementation pathway. They do not demonstrate quantum speedup, calibrated "
    "hardware execution, or full IEEE-scale QSVT execution."
)


@dataclass(frozen=True, slots=True)
class HardwareResourceEstimate:
    matrix_shape: tuple[int, int]
    padded_dimension: int
    logical_index_qubits: int
    ancilla_qubits: int
    total_logical_qubits: int
    qsvt_degree: int
    phase_count: int
    query_count: int
    controlled_block_encoding_count: int
    circuit_depth_proxy: int
    two_qubit_gate_proxy: int
    readout_shots: int
    state_preparation_assumption: str
    block_encoding_assumption: str

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "matrix_shape": f"{self.matrix_shape[0]}x{self.matrix_shape[1]}",
            "padded_dimension": self.padded_dimension,
            "logical_index_qubits": self.logical_index_qubits,
            "ancilla_qubits": self.ancilla_qubits,
            "total_logical_qubits": self.total_logical_qubits,
            "qsvt_degree": self.qsvt_degree,
            "phase_count": self.phase_count,
            "query_count": self.query_count,
            "controlled_block_encoding_count": self.controlled_block_encoding_count,
            "circuit_depth_proxy": self.circuit_depth_proxy,
            "two_qubit_gate_proxy": self.two_qubit_gate_proxy,
            "readout_shots": self.readout_shots,
            "state_preparation_assumption": self.state_preparation_assumption,
            "block_encoding_assumption": self.block_encoding_assumption,
            "claim_boundary": HARDWARE_RESOURCE_CLAIM,
        }
        if extra:
            row.update(extra)
        return row


def estimate_hardware_resources(
    matrix: np.ndarray,
    *,
    qsvt_degree: int,
    phase_count: int | None = None,
    readout_shots: int = 10000,
    block_encoding_model: BlockEncodingModel | str = BlockEncodingModel.SPARSE_ACCESS_ORACLE,
    state_preparation_assumption: str = "amplitude state preparation oracle assumed",
) -> HardwareResourceEstimate:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    degree = int(qsvt_degree)
    if degree <= 0:
        raise ValueError("qsvt_degree must be positive")
    shots = int(readout_shots)
    if shots < 0:
        raise ValueError("readout_shots must be nonnegative")
    block = estimate_block_encoding_resources(values, block_encoding_model)
    padded_dimension = max(block.padded_shape)
    logical_index_qubits = max(block.row_qubits, block.col_qubits)
    phases = int(phase_count if phase_count is not None else degree + 1)
    query_count = 2 * degree + 1
    block_query_cost = block.query_cost_per_block_encoding or 1
    gate_proxy = block.gate_cost_proxy or max(1, values.size)
    depth_proxy = int(query_count * block_query_cost * max(1, logical_index_qubits))
    two_qubit_proxy = int(query_count * max(1, gate_proxy))
    return HardwareResourceEstimate(
        matrix_shape=(int(values.shape[0]), int(values.shape[1])),
        padded_dimension=int(padded_dimension),
        logical_index_qubits=int(logical_index_qubits),
        ancilla_qubits=int(block.ancilla_qubits),
        total_logical_qubits=int(logical_index_qubits + block.ancilla_qubits),
        qsvt_degree=degree,
        phase_count=phases,
        query_count=query_count,
        controlled_block_encoding_count=query_count,
        circuit_depth_proxy=depth_proxy,
        two_qubit_gate_proxy=two_qubit_proxy,
        readout_shots=shots,
        state_preparation_assumption=state_preparation_assumption,
        block_encoding_assumption="; ".join(block.assumptions),
    )


def build_hardware_resource_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_hardware_resources",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "seed": 123,
        "qsvt_degree": 51,
        "phase_count": 52,
        "readout_shots": 10000,
        "block_encoding_model": BlockEncodingModel.SPARSE_ACCESS_ORACLE.value,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    estimate = estimate_hardware_resources(
        system.H_tilde,
        qsvt_degree=int(resolved["qsvt_degree"]),
        phase_count=int(resolved["phase_count"]),
        readout_shots=int(resolved["readout_shots"]),
        block_encoding_model=str(resolved["block_encoding_model"]),
    )
    row = estimate.to_row(
        {
            "case_name": resolved["case_name"],
            "matrix_source": matrix_source,
            "block_encoding_model": str(resolved["block_encoding_model"]),
        }
    )
    frame = pd.DataFrame([row])
    summary_csv = output_dir / "hardware_resource_summary.csv"
    assumptions_md = output_dir / "hardware_resource_assumptions.md"
    query_csv = output_dir / "qsvt_query_count_summary.csv"
    frame.to_csv(summary_csv, index=False)
    pd.DataFrame(
        [
            {
                "case_name": resolved["case_name"],
                "qsvt_degree": estimate.qsvt_degree,
                "phase_count": estimate.phase_count,
                "query_count": estimate.query_count,
                "controlled_block_encoding_count": estimate.controlled_block_encoding_count,
            }
        ]
    ).to_csv(query_csv, index=False)
    assumptions_md.write_text(_assumptions_markdown(row), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "hardware_resource_summary": str(summary_csv),
            "hardware_resource_assumptions": str(assumptions_md),
            "qsvt_query_count_summary": str(query_csv),
        },
        input_config=resolved,
        claim_boundary=HARDWARE_RESOURCE_CLAIM,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "hardware_resource_summary": summary_csv,
            "hardware_resource_assumptions": assumptions_md,
            "qsvt_query_count_summary": query_csv,
            "manifest": manifest_path,
        },
    }


def qubit_convention_counts(
    matrix_shape: tuple[int, int],
    *,
    block_encoding_ancilla_qubits: int = 3,
    qsvt_signal_ancilla_qubits: int = 1,
    state_preparation_ancilla_qubits: int = 2,
) -> dict[str, int]:
    rows, cols = int(matrix_shape[0]), int(matrix_shape[1])
    row_qubits = _qubits(rows)
    col_qubits = _qubits(cols)
    padded_qubits = _qubits(max(rows, cols))
    total_row_col = (
        row_qubits
        + col_qubits
        + int(block_encoding_ancilla_qubits)
        + int(qsvt_signal_ancilla_qubits)
        + int(state_preparation_ancilla_qubits)
    )
    total_padded = (
        padded_qubits
        + int(block_encoding_ancilla_qubits)
        + int(qsvt_signal_ancilla_qubits)
        + int(state_preparation_ancilla_qubits)
    )
    return {
        "row_qubits": int(row_qubits),
        "col_qubits": int(col_qubits),
        "padded_dimension_qubits": int(padded_qubits),
        "block_encoding_ancilla_qubits": int(block_encoding_ancilla_qubits),
        "qsvt_signal_ancilla_qubits": int(qsvt_signal_ancilla_qubits),
        "state_preparation_ancilla_qubits": int(state_preparation_ancilla_qubits),
        "total_logical_qubits_row_col_convention": int(total_row_col),
        "total_logical_qubits_padded_convention": int(total_padded),
    }


def build_qubit_convention_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_scalable_qubit_convention_audit",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "seed": 123,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case in list(resolved["cases"]):
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": resolved["matrix_source"],
                "seed": int(resolved["seed"]),
            }
        )
        counts = qubit_convention_counts(system.H_tilde.shape)
        rows.append(
            {
                "case": case,
                "matrix_source": matrix_source,
                "matrix_rows": int(system.H_tilde.shape[0]),
                "matrix_cols": int(system.H_tilde.shape[1]),
                **counts,
                "convention_note": (
                    "row+column convention counts separate rectangular registers; "
                    "padded convention counts one square-dimension index register."
                ),
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "qubit_convention_summary.csv"
    md_path = output_dir / "qubit_convention_audit.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_qubit_audit_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "qubit_convention_summary": str(csv_path),
            "qubit_convention_audit": str(md_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "manifest": manifest,
            "qubit_convention_summary": csv_path,
            "qubit_convention_audit": md_path,
        },
    }


def build_oracle_model_resource_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_oracle_model_resources",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "seed": 123,
        "alpha": 1.0e-4,
        "degree": 51,
        "phase_count": 52,
        "readout_shots": 10000,
        "block_encoding_model": BlockEncodingModel.SPARSE_ACCESS_ORACLE.value,
        "state_preparation_model": StatePreparationModel.QRAM_AMPLITUDE_ORACLE.value,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    for case in list(resolved["cases"]):
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": resolved["matrix_source"],
                "seed": int(resolved["seed"]),
            }
        )
        oracle = build_sparse_access_oracle(system.H_tilde)
        prep = estimate_state_preparation(
            system.r_tilde,
            str(resolved["state_preparation_model"]),
        )
        counts = qubit_convention_counts(
            oracle.shape,
            block_encoding_ancilla_qubits=3,
            qsvt_signal_ancilla_qubits=1,
            state_preparation_ancilla_qubits=prep.ancilla_qubits,
        )
        degree = int(resolved["degree"])
        query_count = 2 * degree + 1
        block_query = max(1, oracle.max_row_sparsity + oracle.max_col_sparsity)
        rows.append(
            {
                "case": case,
                "matrix_source": matrix_source,
                "matrix_rows": oracle.shape[0],
                "matrix_cols": oracle.shape[1],
                "nnz": oracle.nnz,
                "density": oracle.nnz / float(oracle.shape[0] * oracle.shape[1]),
                "max_row_sparsity": oracle.max_row_sparsity,
                "max_col_sparsity": oracle.max_col_sparsity,
                **counts,
                "alpha": float(resolved["alpha"]),
                "degree": degree,
                "phase_count": int(resolved["phase_count"]),
                "qsvt_query_count": query_count,
                "block_encoding_query_cost": block_query,
                "state_preparation_query_cost": prep.estimated_query_cost,
                "readout_shots": int(resolved["readout_shots"]),
                "success_probability_proxy": None,
                "norm_recovery_model": "future_amplitude_estimation_or_simulator_metadata",
                "block_encoding_model": str(resolved["block_encoding_model"]),
                "state_preparation_model": prep.preparation_model,
                "implemented_or_estimated": "oracle_model_resource_estimate",
                "limitations": (
                    "sparse-access oracle, state preparation, and amplitude "
                    "estimation are resource models; no full IEEE-scale hardware "
                    "execution is implemented"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summary_csv = output_dir / "oracle_model_resource_summary.csv"
    assumptions_md = output_dir / "oracle_model_resource_assumptions.md"
    limitations_md = output_dir / "oracle_model_resource_limitations.md"
    frame.to_csv(summary_csv, index=False)
    assumptions_md.write_text(_oracle_resource_assumptions(), encoding="utf-8")
    limitations_md.write_text(_oracle_resource_limitations(), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "oracle_model_resource_summary": str(summary_csv),
            "oracle_model_resource_assumptions": str(assumptions_md),
            "oracle_model_resource_limitations": str(limitations_md),
        },
        input_config=resolved,
        claim_boundary=HARDWARE_RESOURCE_CLAIM,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "manifest": manifest,
            "oracle_model_resource_summary": summary_csv,
            "oracle_model_resource_assumptions": assumptions_md,
            "oracle_model_resource_limitations": limitations_md,
        },
    }


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _qubit_audit_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Qubit Convention Audit",
        "",
        "For a rectangular weighted Jacobian H in R^{m x n}, this audit reports:",
        "",
        "- row register qubits: ceil(log2(m))",
        "- column register qubits: ceil(log2(n))",
        "- padded square-dimension qubits: ceil(log2(max(m,n)))",
        "- block-encoding, QSVT signal, and state-preparation ancillas separately",
        "",
        "The older single total logical qubit value is not sufficient for rectangular "
        "resource accounting.",
        "",
    ]
    for row in frame.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['case']}",
                f"- Matrix: {row['matrix_rows']} x {row['matrix_cols']}",
                f"- Row qubits: {row['row_qubits']}",
                f"- Column qubits: {row['col_qubits']}",
                f"- Padded-dimension qubits: {row['padded_dimension_qubits']}",
                "- Ancillas: "
                f"block={row['block_encoding_ancilla_qubits']}, "
                f"signal={row['qsvt_signal_ancilla_qubits']}, "
                f"state-prep={row['state_preparation_ancilla_qubits']}",
                "- Totals: "
                f"row+column={row['total_logical_qubits_row_col_convention']}, "
                f"padded={row['total_logical_qubits_padded_convention']}",
                "",
            ]
        )
    return "\n".join(lines)


def _oracle_resource_assumptions() -> str:
    return "\n".join(
        [
            "# Oracle-Model Resource Assumptions",
            "",
            "- Matrix access is represented by sparse row-position and value oracles.",
            "- Row and column registers are counted separately for rectangular matrices.",
            "- QSVT query count is reported as 2 * degree + 1.",
            "- State preparation uses the configured access model and is not synthesized.",
            "",
        ]
    )


def _oracle_resource_limitations() -> str:
    return "\n".join(
        [
            "# Oracle-Model Resource Limitations",
            "",
            HARDWARE_RESOURCE_CLAIM,
            "",
            "- No sparse oracle circuit is synthesized for IEEE-scale matrices.",
            "- No fault-tolerant compilation, routing, or physical error model is included.",
            "- Success probability and norm recovery remain proxy or future-work items.",
            "",
        ]
    )


def _assumptions_markdown(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Hardware Resource Assumptions",
            "",
            HARDWARE_RESOURCE_CLAIM,
            "",
            f"- Case: {row['case_name']}",
            f"- Matrix shape: {row['matrix_shape']}",
            f"- Logical index qubits: {row['logical_index_qubits']}",
            f"- Ancilla qubits: {row['ancilla_qubits']}",
            f"- QSVT degree: {row['qsvt_degree']}",
            f"- Query count: {row['query_count']}",
            f"- Readout shots: {row['readout_shots']}",
            f"- State preparation assumption: {row['state_preparation_assumption']}",
            f"- Block-encoding assumption: {row['block_encoding_assumption']}",
            "",
            "No calibrated noise model, fault-tolerant compilation, qRAM construction, "
            "or full-vector readout is included.",
            "",
        ]
    )
