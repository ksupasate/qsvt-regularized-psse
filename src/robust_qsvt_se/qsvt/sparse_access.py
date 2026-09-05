"""Concrete sparse-access oracle/emulator model for weighted PSSE Jacobians.

This module exposes the standard sparse-access oracle *interfaces* used by QSVT
block-encoding constructions for a row-sparse matrix ``H~``:

    O_col : |i, k>      -> |i, c(i, k)>          (k-th nonzero column of row i)
    O_val : |i, j, 0>   -> |i, j, H~_{ij}>       (weighted Jacobian value)

It is a **classical sparse-access emulator**: the index and value lookups are
exact CSR table lookups, validated against the source matrix. It is **not** a
synthesized reversible quantum oracle circuit and **not** a quantum-hardware
run. The model adds the resource bookkeeping (index-register qubits, value
precision, query assumptions) needed to make the QSVT pathway concrete while
staying inside the feasibility/boundary scope.

The heavy lifting (CSR construction, index/value lookup, spectral-norm
normalization) is delegated to the existing
:class:`robust_qsvt_se.qsvt.sparse_access_oracle.SparseAccessOracle`; this layer
adds the task-specified API surface and the qubit/precision metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.sparse_access_oracle import (
    SparseAccessOracle,
    build_sparse_access_oracle,
)

# Conservative, reused in summaries and manifests.
SPARSE_ACCESS_STATUS = "validated_exact_lookup"
SPARSE_ACCESS_LIMITATION = (
    "Classical sparse-access emulator with exact CSR index/value lookup. This is a "
    "modeled sparse-access pathway, not a reversible quantum oracle circuit and not a "
    "quantum-hardware run."
)

DEFAULT_VALUE_PRECISION_BITS = 8


def _index_qubits(dimension: int) -> int:
    """Qubits to address ``dimension`` basis states (at least one)."""

    return int(max(1, ceil(log2(max(int(dimension), 1)))))


@dataclass(slots=True)
class SparseAccessModel:
    """Sparse-access model for one weighted Jacobian with resource bookkeeping.

    Wraps a validated :class:`SparseAccessOracle` and records the oracle-interface
    metadata (qubit counts, value precision, query assumptions). The lookups are
    exact; ``reversible_oracle_synthesized`` is always ``False`` because no
    reversible circuit is compiled here.
    """

    oracle: SparseAccessOracle
    matrix_source: str
    value_precision_bits: int
    row_nonzero_counts: np.ndarray = field(repr=False)
    case: str | None = None

    # --- shape / sparsity -------------------------------------------------
    @property
    def num_rows(self) -> int:
        return int(self.oracle.shape[0])

    @property
    def num_cols(self) -> int:
        return int(self.oracle.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.oracle.nnz)

    @property
    def density(self) -> float:
        return float(self.nnz / (self.num_rows * self.num_cols))

    @property
    def max_row_nnz(self) -> int:
        return int(self.row_nonzero_counts.max()) if self.row_nonzero_counts.size else 0

    @property
    def mean_row_nnz(self) -> float:
        return float(self.row_nonzero_counts.mean()) if self.row_nonzero_counts.size else 0.0

    # --- resource bookkeeping --------------------------------------------
    @property
    def row_index_qubits(self) -> int:
        return _index_qubits(self.num_rows)

    @property
    def col_index_qubits(self) -> int:
        return _index_qubits(self.num_cols)

    @property
    def local_index_qubits(self) -> int:
        """Qubits for the local nonzero index ``k`` in ``O_col`` (sparsity register)."""

        return _index_qubits(self.max_row_nnz)

    @property
    def index_qubits(self) -> int:
        """Row + column index-register qubits for the ``|i, j>`` value-oracle register."""

        return self.row_index_qubits + self.col_index_qubits

    @property
    def value_register_qubits(self) -> int:
        """Fixed-point value-register width (sign folded into two's-complement encoding)."""

        return int(self.value_precision_bits)

    @property
    def normalization_beta(self) -> float:
        return float(self.oracle.normalization_beta)

    def query_assumptions(self) -> dict[str, Any]:
        """Modeled per-application oracle query counts for one block-encoding call."""

        return {
            "o_col_queries_per_block_encoding": 1,
            "o_val_queries_per_block_encoding": 1,
            "diffusion_queries_per_block_encoding": 1,
            "reversible_oracle_synthesized": False,
            "query_model": "modeled_sparse_access",
        }

    # --- oracle interfaces (exact emulation) ------------------------------
    def get_col(self, row_index: int, local_nonzero_index: int) -> int:
        """``O_col``: column of the ``local_nonzero_index``-th nonzero in ``row_index``."""

        return self.oracle.get_row_nonzero_col(row_index, local_nonzero_index)

    def get_val(self, row_index: int, col_index: int) -> float:
        """``O_val``: the weighted Jacobian value ``H~_{ij}`` (0.0 if structurally zero)."""

        return self.oracle.get_value(row_index, col_index)

    def get_row_nonzeros(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(columns, values)`` of the nonzeros in ``row_index``."""

        return self.oracle.get_row_entries(row_index)

    def validate_against_dense_or_csr(self, reference: np.ndarray | None = None) -> dict[str, Any]:
        """Exhaustively verify the emulated access matches the source matrix.

        Checks, over *every* structural nonzero: (1) ``O_val`` returns the stored
        value, (2) ``O_col`` enumerates the same columns as the CSR row, and
        (3) row nonzero counts and ``max_row_nnz`` agree. Returns a record with the
        number of entries checked, the maximum value-lookup error, and a status.
        """

        dense = self.oracle.to_dense_if_small(max_dimension=max(self.oracle.shape) + 1)
        if reference is not None:
            dense = np.asarray(reference, dtype=np.float64)
        entries_checked = 0
        max_value_error = 0.0
        col_mismatches = 0
        for row in range(self.num_rows):
            columns, values = self.get_row_nonzeros(row)
            if int(columns.size) != int(self.row_nonzero_counts[row]):
                col_mismatches += 1
            for local_index in range(columns.size):
                accessed_col = self.get_col(row, local_index)
                if int(accessed_col) != int(columns[local_index]):
                    col_mismatches += 1
                accessed_value = self.get_val(row, int(accessed_col))
                stored = float(values[local_index])
                max_value_error = max(max_value_error, abs(accessed_value - stored))
                if dense is not None:
                    max_value_error = max(
                        max_value_error, abs(accessed_value - float(dense[row, int(accessed_col)]))
                    )
                entries_checked += 1
        max_row_nnz_correct = self.max_row_nnz == (
            int(np.max(self.row_nonzero_counts)) if self.row_nonzero_counts.size else 0
        )
        invalid_index_raises = self._invalid_index_raises()
        validated = (
            max_value_error <= 1.0e-12
            and col_mismatches == 0
            and max_row_nnz_correct
            and invalid_index_raises
        )
        return {
            "case": self.case,
            "matrix_source": self.matrix_source,
            "entries_checked": int(entries_checked),
            "value_max_abs_error": float(max_value_error),
            "col_index_mismatches": int(col_mismatches),
            "max_row_nnz_correct": bool(max_row_nnz_correct),
            "invalid_index_raises": bool(invalid_index_raises),
            "access_status": SPARSE_ACCESS_STATUS if validated else "validation_failed",
            "reversible_oracle_synthesized": False,
            "notes": SPARSE_ACCESS_LIMITATION,
        }

    def _invalid_index_raises(self) -> bool:
        """Confirm out-of-range access raises (defensive lookup, not silent zero)."""

        ok = True
        try:
            self.get_col(self.num_rows, 0)
            ok = False
        except IndexError:
            pass
        try:
            self.get_val(self.num_rows, 0)
            ok = False
        except IndexError:
            pass
        try:
            # local index past the row's nonzeros must raise
            self.get_col(0, int(self.row_nonzero_counts[0]) + 5)
            ok = False
        except IndexError:
            pass
        return ok

    def summary_row(self) -> dict[str, Any]:
        """Row matching the required sparse-access summary columns."""

        return {
            "case": self.case,
            "matrix_source": self.matrix_source,
            "shape_rows": self.num_rows,
            "shape_cols": self.num_cols,
            "nnz": self.nnz,
            "density": self.density,
            "max_row_nnz": self.max_row_nnz,
            "mean_row_nnz": self.mean_row_nnz,
            "index_qubits": self.index_qubits,
            "value_precision_bits": int(self.value_precision_bits),
            "value_register_qubits": self.value_register_qubits,
            "access_status": SPARSE_ACCESS_STATUS,
            "reversible_oracle_synthesized": False,
            "notes": SPARSE_ACCESS_LIMITATION,
        }


def build_sparse_access_model(
    matrix: np.ndarray,
    *,
    case: str | None = None,
    matrix_source: str = "weighted_jacobian",
    value_precision_bits: int = DEFAULT_VALUE_PRECISION_BITS,
    normalization_beta: float | None = None,
    tolerance: float = 1.0e-12,
) -> SparseAccessModel:
    """Build a :class:`SparseAccessModel` from a (dense) weighted Jacobian."""

    if int(value_precision_bits) <= 0:
        raise ValueError("value_precision_bits must be positive")
    oracle = build_sparse_access_oracle(
        matrix,
        normalization_beta=normalization_beta,
        tolerance=tolerance,
    )
    row_counts = np.diff(oracle.row_ptr).astype(np.int64)
    return SparseAccessModel(
        oracle=oracle,
        matrix_source=str(matrix_source),
        value_precision_bits=int(value_precision_bits),
        row_nonzero_counts=row_counts,
        case=case,
    )
