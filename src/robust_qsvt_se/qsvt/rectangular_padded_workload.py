"""Rectangular 8x4 block zero-padded to 8x8 through the existing sparse-QSVT compiler.

The compiler's in-place wrapper supports square power-of-two matrices only.  A tall 8x4
weighted-Jacobian block is therefore embedded by zero-padding four state columns; the padded
Ridge action is exactly the rectangular Ridge action on the real columns (the padded
coordinates carry zero weight and zero output), which this module verifies both classically
and against the compiled statevector.  Everything reuses the frozen canonical degree-31
polynomial/phase pair and the existing compile path unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompiledSparseQSVT,
    compile_from_bundle,
)
from robust_qsvt_se.qsvt.generic_sparse_scaling import (
    _balanced_magnitude_support,
    _bundle_for_matrix,
)

WORKLOAD_ID = "ieee14_sparse_quantized_8x4pad8x8_d31_rectangular_v1"
MATRIX_SEED = 123
RECT_ROWS = 8
RECT_COLS = 4
PADDED_DIM = 8
SUPPORT_BUDGET = 12
SLOTS = 3
EQUIVALENCE_ATOL = 1.0e-12


def build_rectangular_block() -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Deterministic 8x4 IEEE-14 block + residual via the frozen outcome-independent extractor."""

    from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
    from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system

    system, _source = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": MATRIX_SEED,
        }
    )
    block, residual, rows, cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=RECT_ROWS,
        col_count=RECT_COLS,
    )
    return (
        np.asarray(block, dtype=np.float64),
        np.asarray(residual, dtype=np.float64),
        tuple(int(v) for v in rows),
        tuple(int(v) for v in cols),
    )


def zero_pad(block: np.ndarray) -> np.ndarray:
    padded = np.zeros((PADDED_DIM, PADDED_DIM), dtype=np.float64)
    padded[:, : block.shape[1]] = block
    return padded


def classical_padding_equivalence(
    rectangular: np.ndarray, padded: np.ndarray, residual: np.ndarray, alpha: float
) -> dict[str, float]:
    """Exact check: padded-square Ridge == rectangular Ridge on real columns, zero elsewhere."""

    rect_update = ridge_svd_solution(rectangular, residual, alpha=float(alpha))
    padded_update = ridge_svd_solution(padded, residual, alpha=float(alpha))
    real_gap = float(np.max(np.abs(padded_update[: rectangular.shape[1]] - rect_update)))
    pad_leak = float(np.max(np.abs(padded_update[rectangular.shape[1]:])))
    return {
        "max_abs_real_column_gap": real_gap,
        "max_abs_padded_coordinate": pad_leak,
        "equivalent": bool(real_gap <= EQUIVALENCE_ATOL and pad_leak <= EQUIVALENCE_ATOL),
    }


def build_padded_workload(canonical: CompiledSparseQSVT) -> tuple[CompiledSparseQSVT, dict]:
    """Compile the padded workload through the unchanged existing compiler path."""

    rectangular, residual, rows, cols = build_rectangular_block()
    padded = zero_pad(rectangular)
    support = _balanced_magnitude_support(padded, budget=SUPPORT_BUDGET, slots=SLOTS)
    if any(j >= RECT_COLS for _i, j in support):
        raise RuntimeError("support selection leaked into zero-padded columns")
    bundle = _bundle_for_matrix(
        canonical,
        matrix=padded,
        residual=residual,
        coordinates=support,
        matrix_id="ieee14_rectangular_8x4_zero_padded",
        workload_id=WORKLOAD_ID,
        source="ieee14_pypower_ac_weighted_jacobian seed 123 (8x4 extractor block, zero-padded)",
    )
    compiled = compile_from_bundle(bundle)
    metadata: dict[str, Any] = {
        "workload_id": WORKLOAD_ID,
        "rectangular_shape": [RECT_ROWS, RECT_COLS],
        "padded_shape": [PADDED_DIM, PADDED_DIM],
        "selected_rows": list(rows),
        "selected_columns": list(cols),
        "support_coordinates": [list(pair) for pair in support],
        "support_budget": SUPPORT_BUDGET,
        "slots": SLOTS,
        "rectangular_rank": int(np.linalg.matrix_rank(rectangular)),
        "padded_nonzeros": int(np.count_nonzero(padded)),
        "alpha": float(compiled.qsvt_spec.alpha),
        "beta": float(compiled.qsvt_spec.beta),
        "normalized_lambda": float(compiled.qsvt_spec.normalized_lambda),
        "degree": int(compiled.qsvt_spec.degree),
        "workload_digest": compiled.workload_digest,
    }
    return compiled, metadata


def statevector_rectangular_consistency(
    compiled: CompiledSparseQSVT, statevector_metrics_row: dict[str, Any]
) -> dict[str, float]:
    """Compare the compiled selected output against the exact RECTANGULAR quantized reference.

    The compiled chain acts on the padded quantized matrix; the reference here strips the
    padding and applies the identical Ridge filter to the 8x4 quantized block, so agreement
    demonstrates the rectangular semantics end to end.
    """

    quantized_rect = np.asarray(compiled.matrix_quantized, dtype=np.float64)[:, :RECT_COLS]
    update = ridge_svd_solution(
        quantized_rect, compiled.residual, alpha=float(compiled.qsvt_spec.alpha)
    )
    y_rect = float(update[0])
    y_padded_ridge = float(statevector_metrics_row["quantized_ridge_selected_output"])
    y_statevector = float(statevector_metrics_row["statevector_selected_output"])
    return {
        "rectangular_quantized_ridge_selected_output": y_rect,
        "padded_quantized_ridge_selected_output": y_padded_ridge,
        "statevector_selected_output": y_statevector,
        "abs_gap_rect_vs_padded_ridge": abs(y_rect - y_padded_ridge),
        "abs_gap_statevector_vs_rect_ridge": abs(y_statevector - y_rect),
    }


__all__ = [
    "EQUIVALENCE_ATOL",
    "MATRIX_SEED",
    "PADDED_DIM",
    "RECT_COLS",
    "RECT_ROWS",
    "SLOTS",
    "SUPPORT_BUDGET",
    "WORKLOAD_ID",
    "build_padded_workload",
    "build_rectangular_block",
    "classical_padding_equivalence",
    "statevector_rectangular_consistency",
    "zero_pad",
]
