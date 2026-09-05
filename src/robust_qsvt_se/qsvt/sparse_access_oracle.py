from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, matrix_density
from robust_qsvt_se.utils.io import ensure_directory, write_json

SPARSE_ORACLE_LIMITATION = (
    "This is a classical sparse-access oracle interface for resource and "
    "matrix-free simulation. It is not a synthesized quantum oracle circuit."
)


@dataclass(slots=True)
class SparseAccessOracle:
    """Classical row-sparse access model for a weighted Jacobian matrix.

    The row oracle corresponds to ``O_F: |i,l> -> |i,f(i,l)>`` where
    ``f(i,l)`` is represented by CSR ``row_ptr`` and ``col_ind`` arrays.
    Value access is represented by sparse lookup in the same CSR structure.
    """

    shape: tuple[int, int]
    nnz: int
    max_row_sparsity: int
    max_col_sparsity: int
    row_ptr: np.ndarray
    col_ind: np.ndarray
    data: np.ndarray
    normalization_beta: float
    dtype: str
    _csr: sparse.csr_matrix = field(repr=False)
    _csc: sparse.csc_matrix = field(repr=False)

    def get_row_nonzero_col(self, row: int, local_index: int) -> int:
        row_index = _validate_index(row, self.shape[0], "row")
        start = int(self.row_ptr[row_index])
        stop = int(self.row_ptr[row_index + 1])
        local = _validate_index(local_index, stop - start, "local_index")
        return int(self.col_ind[start + local])

    def get_value(self, row: int, col: int) -> float:
        row_index = _validate_index(row, self.shape[0], "row")
        col_index = _validate_index(col, self.shape[1], "col")
        start = int(self.row_ptr[row_index])
        stop = int(self.row_ptr[row_index + 1])
        columns = self.col_ind[start:stop]
        location = np.searchsorted(columns, col_index)
        if location < columns.size and int(columns[location]) == col_index:
            return float(self.data[start + int(location)])
        return 0.0

    def get_row_entries(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        row_index = _validate_index(row, self.shape[0], "row")
        start = int(self.row_ptr[row_index])
        stop = int(self.row_ptr[row_index + 1])
        return self.col_ind[start:stop].copy(), self.data[start:stop].copy()

    def get_column_entries(self, col: int) -> tuple[np.ndarray, np.ndarray]:
        col_index = _validate_index(col, self.shape[1], "col")
        start = int(self._csc.indptr[col_index])
        stop = int(self._csc.indptr[col_index + 1])
        return self._csc.indices[start:stop].copy(), self._csc.data[start:stop].copy()

    def matvec(self, x: np.ndarray) -> np.ndarray:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"x must have shape {(self.shape[1],)}, got {vector.shape}")
        return np.asarray(self._csr @ vector, dtype=np.float64)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        vector = np.asarray(y, dtype=np.float64)
        if vector.shape != (self.shape[0],):
            raise ValueError(f"y must have shape {(self.shape[0],)}, got {vector.shape}")
        return np.asarray(self._csr.T @ vector, dtype=np.float64)

    def to_dense_if_small(self, max_dimension: int = 64) -> np.ndarray | None:
        if max(self.shape) > int(max_dimension):
            return None
        return np.asarray(self._csr.toarray(), dtype=np.float64)

    def to_summary_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        rows, cols = self.shape
        row = {
            "matrix_rows": rows,
            "matrix_cols": cols,
            "nnz": self.nnz,
            "density": self.nnz / float(rows * cols),
            "max_row_sparsity": self.max_row_sparsity,
            "max_col_sparsity": self.max_col_sparsity,
            "normalization_beta": self.normalization_beta,
            "dtype": self.dtype,
            "limitation": SPARSE_ORACLE_LIMITATION,
        }
        if extra:
            row.update(extra)
        return row


def build_sparse_access_oracle(
    matrix: np.ndarray,
    *,
    normalization_beta: float | None = None,
    tolerance: float = 1.0e-12,
) -> SparseAccessOracle:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    cleaned = np.where(np.abs(values) > float(tolerance), values, 0.0)
    csr = sparse.csr_matrix(cleaned)
    csr.eliminate_zeros()
    csr.sort_indices()
    csc = csr.tocsc()
    row_counts = np.diff(csr.indptr)
    col_counts = np.diff(csc.indptr)
    beta = (
        float(normalization_beta)
        if normalization_beta is not None
        else _spectral_norm_estimate(csr)
    )
    if beta <= 0.0:
        beta = 1.0
    return SparseAccessOracle(
        shape=(int(values.shape[0]), int(values.shape[1])),
        nnz=int(csr.nnz),
        max_row_sparsity=int(row_counts.max()) if row_counts.size else 0,
        max_col_sparsity=int(col_counts.max()) if col_counts.size else 0,
        row_ptr=np.asarray(csr.indptr, dtype=np.int64),
        col_ind=np.asarray(csr.indices, dtype=np.int64),
        data=np.asarray(csr.data, dtype=np.float64),
        normalization_beta=float(beta),
        dtype=str(csr.data.dtype),
        _csr=csr,
        _csc=csc,
    )


def build_sparse_oracle_outputs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_sparse_access_oracle",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "seed": 123,
        "validation_vector_seed": 321,
    }
    if config:
        resolved.update(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rng = np.random.default_rng(int(resolved["validation_vector_seed"]))
    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
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
        x = rng.normal(size=oracle.shape[1])
        y = rng.normal(size=oracle.shape[0])
        matvec_error = float(np.linalg.norm(oracle.matvec(x) - system.H_tilde @ x))
        rmatvec_error = float(np.linalg.norm(oracle.rmatvec(y) - system.H_tilde.T @ y))
        rows.append(
            oracle.to_summary_row(
                {
                    "case": case,
                    "matrix_source": matrix_source,
                    "dense_matrix_density": matrix_density(system.H_tilde),
                    "dense_unitary_constructed": False,
                }
            )
        )
        validation[case] = {
            "matvec_l2_error": matvec_error,
            "rmatvec_l2_error": rmatvec_error,
            "row_access_checked": _row_access_validation(system.H_tilde, oracle),
            "dense_unitary_constructed": False,
        }
    summary_csv = output_dir / "sparse_oracle_summary.csv"
    validation_json = output_dir / "sparse_oracle_validation.json"
    assumptions_md = output_dir / "sparse_oracle_assumptions.md"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    write_json(validation_json, validation)
    assumptions_md.write_text(_sparse_oracle_assumptions(), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "sparse_oracle_summary": str(summary_csv),
            "sparse_oracle_validation": str(validation_json),
            "sparse_oracle_assumptions": str(assumptions_md),
        },
        input_config=resolved,
        claim_boundary=SPARSE_ORACLE_LIMITATION,
    )
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame(rows),
        "validation": validation,
        "artifacts": {
            "manifest": manifest,
            "sparse_oracle_summary": summary_csv,
            "sparse_oracle_validation": validation_json,
            "sparse_oracle_assumptions": assumptions_md,
        },
    }


def _spectral_norm_estimate(matrix: sparse.csr_matrix) -> float:
    if max(matrix.shape) <= 256:
        singular_values = np.linalg.svd(matrix.toarray(), compute_uv=False)
        return float(singular_values[0]) if singular_values.size else 0.0
    rng = np.random.default_rng(123)
    x = rng.normal(size=matrix.shape[1])
    norm = np.linalg.norm(x)
    x = x / norm if norm else np.ones(matrix.shape[1]) / np.sqrt(matrix.shape[1])
    for _ in range(30):
        y = matrix @ x
        z = matrix.T @ y
        z_norm = np.linalg.norm(z)
        if z_norm == 0.0:
            return 0.0
        x = z / z_norm
    return float(np.linalg.norm(matrix @ x))


def _validate_index(index: int, limit: int, name: str) -> int:
    value = int(index)
    if value < 0 or value >= int(limit):
        raise IndexError(f"{name} index {value} out of range for limit {limit}")
    return value


def _row_access_validation(matrix: np.ndarray, oracle: SparseAccessOracle) -> dict[str, Any]:
    checked = 0
    max_error = 0.0
    for row in range(min(5, oracle.shape[0])):
        columns, _values = oracle.get_row_entries(row)
        for local_index, col in enumerate(columns[: min(5, columns.size)]):
            accessed_col = oracle.get_row_nonzero_col(row, local_index)
            accessed_value = oracle.get_value(row, accessed_col)
            max_error = max(max_error, abs(accessed_value - float(matrix[row, col])))
            checked += 1
    return {"entries_checked": checked, "max_abs_error": float(max_error)}


def _sparse_oracle_assumptions() -> str:
    return "\n".join(
        [
            "# Sparse-Access Oracle Assumptions",
            "",
            SPARSE_ORACLE_LIMITATION,
            "",
            "- Row access uses CSR arrays for nonzero positions.",
            "- Value access returns classical matrix entries from the sparse table.",
            "- Matvec and rmatvec use sparse matrix multiplication.",
            "- No dense block-encoding unitary is constructed for IEEE-scale cases.",
            "",
        ]
    )
