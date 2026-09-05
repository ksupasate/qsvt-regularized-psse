from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    spectral_norm_bound,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.utils.io import ensure_directory


class BlockEncodingModel(Enum):
    EXPLICIT_DENSE = "explicit_dense"
    SPARSE_ACCESS_ORACLE = "sparse_access_oracle"
    QRAM_ROW_STATE = "qram_row_state"
    LCU_RESOURCE_MODEL = "lcu_resource_model"


@dataclass(frozen=True, slots=True)
class BlockEncodingResourceEstimate:
    model: str
    matrix_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_qubits: int
    col_qubits: int
    ancilla_qubits: int
    normalization_beta: float
    sparsity_max_row: int
    sparsity_max_col: int
    frobenius_norm: float
    spectral_norm_estimate: float
    block_encoding_error: float | None
    query_cost_per_block_encoding: int | None
    gate_cost_proxy: int | None
    assumptions: list[str]
    limitations: list[str]

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "model": self.model,
            "matrix_shape": f"{self.matrix_shape[0]}x{self.matrix_shape[1]}",
            "padded_shape": f"{self.padded_shape[0]}x{self.padded_shape[1]}",
            "row_qubits": self.row_qubits,
            "col_qubits": self.col_qubits,
            "ancilla_qubits": self.ancilla_qubits,
            "normalization_beta": self.normalization_beta,
            "sparsity_max_row": self.sparsity_max_row,
            "sparsity_max_col": self.sparsity_max_col,
            "frobenius_norm": self.frobenius_norm,
            "spectral_norm_estimate": self.spectral_norm_estimate,
            "block_encoding_error": self.block_encoding_error,
            "query_cost_per_block_encoding": self.query_cost_per_block_encoding,
            "gate_cost_proxy": self.gate_cost_proxy,
            "assumptions": "; ".join(self.assumptions),
            "limitations": "; ".join(self.limitations),
        }
        if extra:
            row.update(extra)
        return row


def estimate_block_encoding_resources(
    matrix: np.ndarray,
    model: BlockEncodingModel | str = BlockEncodingModel.SPARSE_ACCESS_ORACLE,
    *,
    explicit_dimension_limit: int = 16,
    tolerance: float = 1.0e-8,
) -> BlockEncodingResourceEstimate:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    model_value = _model(model)
    rows, cols = values.shape
    padded_rows = _next_power_of_two(rows)
    padded_cols = _next_power_of_two(cols)
    row_qubits = _qubits(padded_rows)
    col_qubits = _qubits(padded_cols)
    nonzero_mask = np.abs(values) > float(tolerance)
    sparsity_max_row = int(nonzero_mask.sum(axis=1).max()) if rows else 0
    sparsity_max_col = int(nonzero_mask.sum(axis=0).max()) if cols else 0
    beta = spectral_norm_bound(values)
    frobenius = float(np.linalg.norm(values, ord="fro"))
    block_error: float | None = None

    assumptions: list[str]
    limitations: list[str]
    if model_value is BlockEncodingModel.EXPLICIT_DENSE:
        assumptions = [
            "dense Julia/canonical block encoding is built only for small contractions",
            "normalization beta is the spectral norm",
        ]
        limitations = [
            "not a scalable sparse-access or qRAM oracle",
            "not used for full IEEE-scale hardware execution",
        ]
        if rows == cols and rows <= explicit_dimension_limit and _is_power_of_two(rows):
            A = values / beta
            block = canonical_square_block_encoding(A, tolerance=tolerance)
            report = validate_block_encoding(block, beta=beta, tolerance=10.0 * tolerance)
            block_error = float(report["top_left_block_error"])
            query_cost = 1
            gate_proxy = int((2 * rows) ** 2)
        else:
            limitations.append(
                "dense unitary was not constructed because shape/dimension exceeded guard"
            )
            query_cost = None
            gate_proxy = None
    elif model_value is BlockEncodingModel.SPARSE_ACCESS_ORACLE:
        assumptions = [
            "row/column sparse-access oracles expose nonzero positions and values",
            "normalization beta is estimated by the spectral norm",
        ]
        limitations = [
            "oracle construction is not implemented",
            "gate constants, data loading, and fault tolerance are not modeled",
        ]
        query_cost = max(1, sparsity_max_row + sparsity_max_col)
        gate_proxy = int(query_cost * (row_qubits + col_qubits + 1))
    elif model_value is BlockEncodingModel.QRAM_ROW_STATE:
        assumptions = [
            "qRAM can prepare normalized row states and row norms",
            "normalization and addressing overhead are summarized as proxy counts",
        ]
        limitations = [
            "qRAM hardware and loading are assumptions, not implemented components",
            "norm and state-preparation errors are not calibrated",
        ]
        query_cost = 2
        gate_proxy = int((row_qubits + col_qubits + 2) * max(1, sparsity_max_row))
    else:
        assumptions = [
            "matrix is decomposed into weighted simple terms for LCU accounting",
            "nonzero count is used as a conservative proxy for term count",
        ]
        limitations = [
            "LCU decomposition is not explicitly synthesized",
            "oblivious amplitude-amplification costs are proxy-only",
        ]
        nonzeros = int(nonzero_mask.sum())
        query_cost = max(1, nonzeros)
        gate_proxy = int(query_cost * (row_qubits + col_qubits + 2))

    return BlockEncodingResourceEstimate(
        model=model_value.value,
        matrix_shape=(int(rows), int(cols)),
        padded_shape=(int(padded_rows), int(padded_cols)),
        row_qubits=int(row_qubits),
        col_qubits=int(col_qubits),
        ancilla_qubits=_ancilla_qubits(model_value),
        normalization_beta=float(beta),
        sparsity_max_row=sparsity_max_row,
        sparsity_max_col=sparsity_max_col,
        frobenius_norm=frobenius,
        spectral_norm_estimate=float(beta),
        block_encoding_error=block_error,
        query_cost_per_block_encoding=query_cost,
        gate_cost_proxy=gate_proxy,
        assumptions=assumptions,
        limitations=limitations,
    )


def build_block_encoding_resource_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/full_qsvt_ieee_block_encoding",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "seed": 123,
        "models": [
            BlockEncodingModel.EXPLICIT_DENSE.value,
            BlockEncodingModel.SPARSE_ACCESS_ORACLE.value,
            BlockEncodingModel.QRAM_ROW_STATE.value,
            BlockEncodingModel.LCU_RESOURCE_MODEL.value,
        ],
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    rows = []
    for model in resolved["models"]:
        estimate = estimate_block_encoding_resources(system.H_tilde, model)
        rows.append(
            estimate.to_row(
                {
                    "case_name": resolved["case_name"],
                    "matrix_source": matrix_source,
                }
            )
        )
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "block_encoding_resource_summary.csv"
    assumptions_path = output_dir / "block_encoding_assumptions.md"
    frame.to_csv(csv_path, index=False)
    assumptions_path.write_text(_assumptions_markdown(rows), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "block_encoding_resource_summary": str(csv_path),
            "block_encoding_assumptions": str(assumptions_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "block_encoding_resource_summary": csv_path,
            "block_encoding_assumptions": assumptions_path,
            "manifest": manifest_path,
        },
    }


def _model(value: BlockEncodingModel | str) -> BlockEncodingModel:
    if isinstance(value, BlockEncodingModel):
        return value
    return BlockEncodingModel(str(value))


def _next_power_of_two(value: int) -> int:
    return 1 << (max(int(value), 1) - 1).bit_length()


def _qubits(dimension: int) -> int:
    return int(np.ceil(np.log2(max(int(dimension), 2))))


def _ancilla_qubits(model: BlockEncodingModel) -> int:
    if model is BlockEncodingModel.EXPLICIT_DENSE:
        return 1
    if model is BlockEncodingModel.SPARSE_ACCESS_ORACLE:
        return 3
    if model is BlockEncodingModel.QRAM_ROW_STATE:
        return 4
    return 4


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _assumptions_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Block-Encoding Resource Assumptions",
        "",
        "Dense explicit block encoding is separated from scalable resource models.",
        "Sparse, qRAM, and LCU rows are resource models only; they do not construct a "
        "dense unitary or execute full IEEE-scale QSVT.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['model']}",
                f"- Matrix shape: {row['matrix_shape']}",
                f"- Padded shape: {row['padded_shape']}",
                f"- Qubits: row={row['row_qubits']}, col={row['col_qubits']}, "
                f"ancilla={row['ancilla_qubits']}",
                f"- Assumptions: {row['assumptions']}",
                f"- Limitations: {row['limitations']}",
                "",
            ]
        )
    return "\n".join(lines)
