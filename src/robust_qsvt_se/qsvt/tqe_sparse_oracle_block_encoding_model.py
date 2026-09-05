from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.tqe_additional_common import (
    CLAIM_BOUNDARY,
    OUTPUT_ROOT,
    current_command,
    ensure_tqe_output_tree,
    reproducibility_metadata,
    utc_timestamp,
    write_top_level_manifest_and_report,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robust_qsvt_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPARSE_ORACLE_DIR = "sparse_oracle_block_encoding_model"
DEFAULT_CASES = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]
DEFAULT_VALUE_BITS = [16, 24, 32]
DEFAULT_NONZERO_TOL = 1.0e-12
PADDING_COLUMN_SENTINEL = -1

SPARSITY_COLUMNS = [
    "case_name",
    "measurement_setting",
    "matrix_shape_m_by_n",
    "m",
    "n",
    "nnz",
    "density",
    "max_row_nnz",
    "mean_row_nnz",
    "median_row_nnz",
    "min_nonzero_row_nnz",
    "max_col_nnz",
    "mean_col_nnz",
    "median_col_nnz",
    "zero_row_count",
    "zero_col_count",
    "numerical_rank",
    "sigma_min",
    "sigma_max",
    "condition_number",
    "norm_2",
    "norm_fro",
    "norm_max_abs",
    "row_norm_min",
    "row_norm_max",
    "row_norm_mean",
    "weighted_status",
    "row_weighting_convention",
    "matrix_source",
    "run_status",
    "failure_or_skip_reason",
]

ORACLE_COLUMNS = [
    "case_name",
    "measurement_setting",
    "m",
    "n",
    "nnz",
    "max_row_nnz",
    "padding_column",
    "uses_row_padding",
    "reconstruction_fro_error",
    "reconstruction_spectral_error",
    "reconstruction_max_abs_error",
    "reconstruction_nnz_mismatch",
    "row_nnz_mismatch_count",
    "value_mismatch_count",
    "oracle_status",
    "failure_or_skip_reason",
]

RESOURCE_COLUMNS = [
    "case_name",
    "measurement_setting",
    "m",
    "n",
    "s",
    "nnz",
    "density",
    "row_qubits",
    "col_qubits",
    "nonzero_index_qubits",
    "value_bits",
    "sign_bit",
    "fixed_point_integer_bits",
    "fixed_point_fraction_bits",
    "alpha_sparse_max",
    "alpha_fro",
    "alpha_row_norm",
    "norm_2",
    "normalization_overhead_sparse_max",
    "normalization_overhead_fro",
    "normalization_overhead_row_norm",
    "sigma_max_over_alpha_sparse_max",
    "sigma_min_over_alpha_sparse_max",
    "effective_condition_number_if_rank_full",
    "padded_sparse_entries",
    "padding_overhead",
    "estimated_oracle_calls_per_block_encoding",
    "query_model",
    "block_encoding_normalization",
    "tiny_reversible_oracle_status",
    "notes",
    "resource_status",
    "failure_or_skip_reason",
]

SUMMARY_COLUMNS = [
    "case_name",
    "measurement_setting",
    "m",
    "n",
    "nnz",
    "density",
    "s",
    "max_col_nnz",
    "row_qubits",
    "col_qubits",
    "nonzero_index_qubits",
    "alpha_sparse_max",
    "norm_2",
    "normalization_overhead_sparse_max",
    "padding_overhead",
    "reconstruction_fro_error",
    "reconstruction_max_abs_error",
    "status",
]


@dataclass(frozen=True, slots=True)
class WeightedJacobianCase:
    case_name: str
    measurement_setting: str
    matrix: np.ndarray
    matrix_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SparseJacobianOracle:
    """Classical row-wise sparse index/value oracle emulator.

    Nonzero columns are stored in sorted order for each row. Padded row slots
    use `padding_column=-1`, and padded values are exactly zero.
    """

    shape: tuple[int, int]
    row_columns: tuple[np.ndarray, ...]
    row_values: tuple[np.ndarray, ...]
    nonzero_tol: float = DEFAULT_NONZERO_TOL
    padding_column: int = PADDING_COLUMN_SENTINEL

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        nonzero_tol: float = DEFAULT_NONZERO_TOL,
        padding_column: int = PADDING_COLUMN_SENTINEL,
    ) -> SparseJacobianOracle:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("matrix must be a nonempty two-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("matrix entries must be finite")
        row_columns: list[np.ndarray] = []
        row_values: list[np.ndarray] = []
        for row in values:
            columns = np.flatnonzero(np.abs(row) > float(nonzero_tol)).astype(np.int64)
            row_columns.append(columns)
            row_values.append(row[columns].astype(np.float64))
        return cls(
            shape=(int(values.shape[0]), int(values.shape[1])),
            row_columns=tuple(row_columns),
            row_values=tuple(row_values),
            nonzero_tol=float(nonzero_tol),
            padding_column=int(padding_column),
        )

    def row_nnz(self, i: int) -> int:
        self._check_row(i)
        return int(self.row_columns[int(i)].size)

    def max_row_nnz(self) -> int:
        return max((int(columns.size) for columns in self.row_columns), default=0)

    def index_oracle(self, i: int, ell: int) -> int:
        self._check_row(i)
        if int(ell) < 0:
            raise IndexError("ell must be nonnegative")
        columns = self.row_columns[int(i)]
        if int(ell) >= columns.size:
            return self.padding_column
        return int(columns[int(ell)])

    def value_oracle(self, i: int, j: int) -> float:
        self._check_row(i)
        column = int(j)
        if column < 0 or column >= self.shape[1]:
            return 0.0
        columns = self.row_columns[int(i)]
        position = np.searchsorted(columns, column)
        if position < columns.size and int(columns[position]) == column:
            return float(self.row_values[int(i)][position])
        return 0.0

    def value_oracle_by_position(self, i: int, ell: int) -> float:
        self._check_row(i)
        if int(ell) < 0:
            raise IndexError("ell must be nonnegative")
        values = self.row_values[int(i)]
        if int(ell) >= values.size:
            return 0.0
        return float(values[int(ell)])

    def reconstruct_dense(self) -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=np.float64)
        for i, (columns, values) in enumerate(zip(self.row_columns, self.row_values, strict=True)):
            matrix[i, columns] = values
        return matrix

    def reconstruct_sparse(self) -> sp.csr_matrix:
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for columns, values in zip(self.row_columns, self.row_values, strict=True):
            data.extend(float(value) for value in values)
            indices.extend(int(column) for column in columns)
            indptr.append(len(data))
        return sp.csr_matrix((data, indices, indptr), shape=self.shape)

    def validate_against_matrix(
        self,
        matrix: np.ndarray,
        *,
        spectral_norm_max_dim: int = 2500,
    ) -> dict[str, float | int]:
        original = np.asarray(matrix, dtype=np.float64)
        if original.shape != self.shape:
            raise ValueError("matrix shape does not match oracle shape")
        reconstructed = self.reconstruct_dense()
        delta = reconstructed - original
        original_nnz = np.count_nonzero(np.abs(original) > self.nonzero_tol, axis=1)
        oracle_nnz = np.array([self.row_nnz(i) for i in range(self.shape[0])], dtype=np.int64)
        nonzero_original = np.abs(original) > self.nonzero_tol
        value_mismatches = np.abs(delta[nonzero_original]) > self.nonzero_tol
        spectral_error = (
            float(np.linalg.norm(delta, ord=2))
            if max(self.shape) <= int(spectral_norm_max_dim)
            else np.nan
        )
        return {
            "reconstruction_fro_error": float(np.linalg.norm(delta, ord="fro")),
            "reconstruction_spectral_error": spectral_error,
            "reconstruction_max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "reconstruction_nnz_mismatch": int(
                np.count_nonzero(np.abs(reconstructed) > self.nonzero_tol)
                - np.count_nonzero(nonzero_original)
            ),
            "row_nnz_mismatch_count": int(np.count_nonzero(original_nnz != oracle_nnz)),
            "value_mismatch_count": int(np.count_nonzero(value_mismatches)),
        }

    def _check_row(self, i: int) -> None:
        row = int(i)
        if row < 0 or row >= self.shape[0]:
            raise IndexError("row index out of range")


def run_sparse_oracle_block_encoding_model(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = utc_timestamp()
    resolved = _resolve_config(config)
    paths = ensure_tqe_output_tree(resolved["output_root"])
    output_dir = ensure_directory(paths["root"] / SPARSE_ORACLE_DIR)
    tables_dir = paths["tables"]
    figures_dir = paths["figures"]
    reports_dir = paths["reports"]

    sparsity_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    sample_payload: dict[str, Any] = {
        "padding_convention": "index_oracle returns -1 for padded slots; values are 0",
        "samples": [],
    }
    row_nnz_payload: dict[str, list[int]] = {}

    for spec in resolved["cases"]:
        try:
            case = load_weighted_jacobian_case(
                spec,
                seed=int(resolved["seed"]),
                case_source=str(resolved["case_source"]),
            )
            matrix = np.asarray(case.matrix, dtype=np.float64)
            oracle = SparseJacobianOracle.from_matrix(
                matrix,
                nonzero_tol=float(resolved["nonzero_tol"]),
            )
            audit = sparsity_audit_row(
                case=case,
                oracle=oracle,
                nonzero_tol=float(resolved["nonzero_tol"]),
            )
            verification = oracle_verification_row(
                case=case,
                oracle=oracle,
                spectral_norm_max_dim=int(resolved["spectral_norm_max_dim"]),
            )
            resources = resource_estimate_rows(
                case=case,
                oracle=oracle,
                audit=audit,
                value_bits_grid=[int(value) for value in resolved["value_bits_grid"]],
            )
            samples = sample_oracle_queries(
                case=case,
                oracle=oracle,
                sample_rows=int(resolved["oracle_sample_rows"]),
            )
            row_nnz_payload[case.case_name] = [
                int(oracle.row_nnz(i)) for i in range(oracle.shape[0])
            ]
            sample_payload["samples"].extend(samples)
        except Exception as exc:
            name = _case_name_from_spec(spec)
            measurement = _measurement_setting_from_spec(spec)
            audit = _sparsity_failure_row(name, measurement, exc)
            verification = _verification_failure_row(name, measurement, exc)
            resources = [
                _resource_failure_row(name, measurement, int(value), exc)
                for value in resolved["value_bits_grid"]
            ]
        sparsity_rows.append(audit)
        verification_rows.append(verification)
        resource_rows.extend(resources)

    sparsity = pd.DataFrame(sparsity_rows, columns=SPARSITY_COLUMNS)
    verification = pd.DataFrame(verification_rows, columns=ORACLE_COLUMNS)
    resources = pd.DataFrame(resource_rows, columns=RESOURCE_COLUMNS)
    summary = summarize_sparse_oracle_results(sparsity, verification, resources)

    sparsity_csv = output_dir / "sparsity_audit_results.csv"
    verification_csv = output_dir / "oracle_reconstruction_verification.csv"
    resources_csv = output_dir / "sparse_oracle_resource_estimates.csv"
    samples_json = output_dir / "oracle_samples.json"
    metadata_json = output_dir / "sparse_oracle_block_encoding_model_metadata.json"
    summary_csv = tables_dir / "table_sparse_oracle_block_encoding_summary.csv"
    density_figure = figures_dir / "figure_sparse_jacobian_density.png"
    row_nnz_figure = figures_dir / "figure_sparse_row_nnz_distribution.png"
    normalization_figure = figures_dir / "figure_sparse_oracle_normalization_overhead.png"
    qubit_figure = figures_dir / "figure_sparse_oracle_qubit_estimates.png"
    report_path = reports_dir / "sparse_oracle_block_encoding_model_report.md"

    sparsity.to_csv(sparsity_csv, index=False)
    verification.to_csv(verification_csv, index=False)
    resources.to_csv(resources_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_json(samples_json, sample_payload)
    _plot_density(sparsity, density_figure)
    _plot_row_nnz_distribution(row_nnz_payload, row_nnz_figure)
    _plot_normalization_overhead(summary, normalization_figure)
    _plot_qubit_estimates(summary, qubit_figure)
    report_path.write_text(
        _report_markdown(
            config=resolved,
            sparsity=sparsity,
            verification=verification,
            resources=resources,
            summary=summary,
            sparsity_csv=sparsity_csv,
            verification_csv=verification_csv,
            resources_csv=resources_csv,
            summary_csv=summary_csv,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "sparsity_audit_csv": str(sparsity_csv),
        "oracle_verification_csv": str(verification_csv),
        "resource_estimates_csv": str(resources_csv),
        "oracle_samples_json": str(samples_json),
        "metadata_json": str(metadata_json),
        "summary_table_csv": str(summary_csv),
        "density_figure": str(density_figure),
        "row_nnz_figure": str(row_nnz_figure),
        "normalization_overhead_figure": str(normalization_figure),
        "qubit_estimates_figure": str(qubit_figure),
        "report": str(report_path),
    }
    ended_at = utc_timestamp()
    metadata = reproducibility_metadata(
        config=resolved,
        started_at=started_at,
        ended_at=ended_at,
        status="completed",
        command=current_command(),
        artifacts=artifacts,
    )
    metadata.update(
        {
            "input_benchmark_cases": [_case_name_from_spec(spec) for spec in resolved["cases"]],
            "measurement_settings": [
                _measurement_setting_from_spec(spec) for spec in resolved["cases"]
            ],
            "nonzero_threshold": float(resolved["nonzero_tol"]),
            "value_bits_grid": [int(value) for value in resolved["value_bits_grid"]],
            "status_counts": {
                "sparsity": sparsity["run_status"].value_counts(dropna=False).to_dict(),
                "oracle": verification["oracle_status"].value_counts(dropna=False).to_dict(),
                "resource": resources["resource_status"].value_counts(dropna=False).to_dict(),
            },
        }
    )
    write_json(metadata_json, metadata)
    top_level = write_top_level_manifest_and_report(paths["root"])
    artifacts.update({key: str(path) for key, path in top_level.items()})
    return {
        "output_root": paths["root"],
        "output_dir": output_dir,
        "sparsity": sparsity,
        "verification": verification,
        "resources": resources,
        "summary": summary,
        "artifacts": {key: Path(value) for key, value in artifacts.items()},
    }


def load_weighted_jacobian_case(
    spec: str | dict[str, Any],
    *,
    seed: int,
    case_source: str,
) -> WeightedJacobianCase:
    if isinstance(spec, dict) and spec.get("force_missing", False):
        raise ValueError("forced missing case for failure-recording test")
    if isinstance(spec, dict) and "matrix" in spec:
        matrix = np.asarray(spec["matrix"], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("synthetic fixture matrix must be two-dimensional")
        case_name = str(spec.get("case_name", "synthetic"))
        measurement = str(spec.get("measurement_setting", "synthetic_fixture"))
        return WeightedJacobianCase(
            case_name=case_name,
            measurement_setting=measurement,
            matrix=matrix,
            matrix_source="synthetic_test_fixture",
            metadata=dict(spec.get("metadata", {})),
        )

    case_name = _case_name_from_spec(spec)
    measurement = _measurement_setting_from_spec(spec)
    measurement_config = dict(spec.get("measurement", {})) if isinstance(spec, dict) else {}
    linearization_config = dict(spec.get("linearization", {})) if isinstance(spec, dict) else {}
    system, matrix_source = build_engineering_system(
        {
            "case_name": case_name,
            "case_source": str(spec.get("case_source", case_source))
            if isinstance(spec, dict)
            else case_source,
            "matrix_source": f"{case_name}_ac_weighted_jacobian",
            "seed": int(spec.get("seed", seed)) if isinstance(spec, dict) else int(seed),
            "measurement": measurement_config,
            "linearization": linearization_config,
        }
    )
    return WeightedJacobianCase(
        case_name=case_name,
        measurement_setting=measurement,
        matrix=np.asarray(system.H_tilde, dtype=np.float64),
        matrix_source=matrix_source,
        metadata=dict(system.metadata),
    )


def sparsity_audit_row(
    *,
    case: WeightedJacobianCase,
    oracle: SparseJacobianOracle,
    nonzero_tol: float,
) -> dict[str, Any]:
    matrix = np.asarray(case.matrix, dtype=np.float64)
    m, n = matrix.shape
    row_nnz = np.array([oracle.row_nnz(i) for i in range(m)], dtype=np.int64)
    col_nnz = np.count_nonzero(np.abs(matrix) > float(nonzero_tol), axis=0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > float(nonzero_tol)]
    row_norms = np.linalg.norm(matrix, axis=1)
    nnz = int(np.sum(row_nnz))
    sigma_min = float(np.min(positive)) if positive.size else 0.0
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf
    nonzero_rows = row_nnz[row_nnz > 0]
    return {
        "case_name": case.case_name,
        "measurement_setting": case.measurement_setting,
        "matrix_shape_m_by_n": f"{m}x{n}",
        "m": int(m),
        "n": int(n),
        "nnz": nnz,
        "density": float(nnz / max(m * n, 1)),
        "max_row_nnz": int(np.max(row_nnz)) if row_nnz.size else 0,
        "mean_row_nnz": float(np.mean(row_nnz)) if row_nnz.size else 0.0,
        "median_row_nnz": float(np.median(row_nnz)) if row_nnz.size else 0.0,
        "min_nonzero_row_nnz": int(np.min(nonzero_rows)) if nonzero_rows.size else 0,
        "max_col_nnz": int(np.max(col_nnz)) if col_nnz.size else 0,
        "mean_col_nnz": float(np.mean(col_nnz)) if col_nnz.size else 0.0,
        "median_col_nnz": float(np.median(col_nnz)) if col_nnz.size else 0.0,
        "zero_row_count": int(np.count_nonzero(row_nnz == 0)),
        "zero_col_count": int(np.count_nonzero(col_nnz == 0)),
        "numerical_rank": int(np.linalg.matrix_rank(matrix, tol=float(nonzero_tol))),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition,
        "norm_2": sigma_max,
        "norm_fro": float(np.linalg.norm(matrix, ord="fro")),
        "norm_max_abs": float(np.max(np.abs(matrix))) if matrix.size else 0.0,
        "row_norm_min": float(np.min(row_norms)) if row_norms.size else 0.0,
        "row_norm_max": float(np.max(row_norms)) if row_norms.size else 0.0,
        "row_norm_mean": float(np.mean(row_norms)) if row_norms.size else 0.0,
        "weighted_status": "weighted_jacobian_R_minus_half_H",
        "row_weighting_convention": "A_ij = H_ij / sigma_i from build_ac_weighted_system",
        "matrix_source": case.matrix_source,
        "run_status": "completed",
        "failure_or_skip_reason": "",
    }


def oracle_verification_row(
    *,
    case: WeightedJacobianCase,
    oracle: SparseJacobianOracle,
    spectral_norm_max_dim: int,
) -> dict[str, Any]:
    diagnostics = oracle.validate_against_matrix(
        case.matrix,
        spectral_norm_max_dim=spectral_norm_max_dim,
    )
    m, n = oracle.shape
    return {
        "case_name": case.case_name,
        "measurement_setting": case.measurement_setting,
        "m": int(m),
        "n": int(n),
        "nnz": int(sum(oracle.row_nnz(i) for i in range(m))),
        "max_row_nnz": int(oracle.max_row_nnz()),
        "padding_column": int(oracle.padding_column),
        "uses_row_padding": True,
        **diagnostics,
        "oracle_status": "completed",
        "failure_or_skip_reason": "",
    }


def resource_estimate_rows(
    *,
    case: WeightedJacobianCase,
    oracle: SparseJacobianOracle,
    audit: dict[str, Any],
    value_bits_grid: list[int],
) -> list[dict[str, Any]]:
    matrix = np.asarray(case.matrix, dtype=np.float64)
    m, n = oracle.shape
    s = int(oracle.max_row_nnz())
    nnz = int(audit["nnz"])
    norm_2 = float(audit["norm_2"])
    norm_fro = float(audit["norm_fro"])
    norm_max_abs = float(audit["norm_max_abs"])
    row_norm_max = float(audit["row_norm_max"])
    alpha_sparse_max = float(s * norm_max_abs)
    alpha_fro = norm_fro
    alpha_row_norm = float(row_norm_max * np.sqrt(m))
    sigma_min = float(audit["sigma_min"])
    sigma_max = float(audit["sigma_max"])
    row_qubits = ceil_log2(m)
    col_qubits = ceil_log2(n)
    index_qubits = ceil_log2(max(s, 1))
    padded_entries = int(m * s)
    padding_overhead = float((padded_entries - nnz) / max(nnz, 1))
    integer_bits = fixed_point_integer_bits(norm_max_abs)
    rows: list[dict[str, Any]] = []
    for value_bits in value_bits_grid:
        sign_bit = 1
        fraction_bits = max(int(value_bits) - sign_bit - integer_bits, 0)
        rows.append(
            {
                "case_name": case.case_name,
                "measurement_setting": case.measurement_setting,
                "m": int(m),
                "n": int(n),
                "s": s,
                "nnz": nnz,
                "density": float(audit["density"]),
                "row_qubits": row_qubits,
                "col_qubits": col_qubits,
                "nonzero_index_qubits": index_qubits,
                "value_bits": int(value_bits),
                "sign_bit": sign_bit,
                "fixed_point_integer_bits": integer_bits,
                "fixed_point_fraction_bits": fraction_bits,
                "alpha_sparse_max": alpha_sparse_max,
                "alpha_fro": alpha_fro,
                "alpha_row_norm": alpha_row_norm,
                "norm_2": norm_2,
                "normalization_overhead_sparse_max": _safe_ratio(alpha_sparse_max, norm_2),
                "normalization_overhead_fro": _safe_ratio(alpha_fro, norm_2),
                "normalization_overhead_row_norm": _safe_ratio(alpha_row_norm, norm_2),
                "sigma_max_over_alpha_sparse_max": _safe_ratio(sigma_max, alpha_sparse_max),
                "sigma_min_over_alpha_sparse_max": _safe_ratio(sigma_min, alpha_sparse_max),
                "effective_condition_number_if_rank_full": (
                    float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf
                ),
                "padded_sparse_entries": padded_entries,
                "padding_overhead": padding_overhead,
                "estimated_oracle_calls_per_block_encoding": 2,
                "query_model": "sparse_index_value_oracle",
                "block_encoding_normalization": "alpha_sparse_max = s * ||A||_max",
                "tiny_reversible_oracle_status": "skipped_not_required",
                "notes": (
                    "Oracle-level resource estimate only; no reversible sparse-oracle "
                    "gate synthesis or hardware execution is claimed."
                ),
                "resource_status": "completed",
                "failure_or_skip_reason": "",
            }
        )
    _ = matrix
    return rows


def summarize_sparse_oracle_results(
    sparsity: pd.DataFrame,
    verification: pd.DataFrame,
    resources: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, audit in sparsity.iterrows():
        case = str(audit["case_name"])
        measurement = str(audit["measurement_setting"])
        verify = _matching_row(verification, case, measurement)
        resource = _matching_resource_row(resources, case, measurement)
        rows.append(
            {
                "case_name": case,
                "measurement_setting": measurement,
                "m": audit["m"],
                "n": audit["n"],
                "nnz": audit["nnz"],
                "density": audit["density"],
                "s": resource.get("s", np.nan),
                "max_col_nnz": audit["max_col_nnz"],
                "row_qubits": resource.get("row_qubits", np.nan),
                "col_qubits": resource.get("col_qubits", np.nan),
                "nonzero_index_qubits": resource.get("nonzero_index_qubits", np.nan),
                "alpha_sparse_max": resource.get("alpha_sparse_max", np.nan),
                "norm_2": audit["norm_2"],
                "normalization_overhead_sparse_max": resource.get(
                    "normalization_overhead_sparse_max",
                    np.nan,
                ),
                "padding_overhead": resource.get("padding_overhead", np.nan),
                "reconstruction_fro_error": verify.get("reconstruction_fro_error", np.nan),
                "reconstruction_max_abs_error": verify.get(
                    "reconstruction_max_abs_error",
                    np.nan,
                ),
                "status": _combined_status(audit, verify, resource),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def sample_oracle_queries(
    *,
    case: WeightedJacobianCase,
    oracle: SparseJacobianOracle,
    sample_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_indices = _deterministic_sample_indices(oracle.shape[0], sample_rows)
    s = max(oracle.max_row_nnz(), 1)
    for i in row_indices:
        row_nnz = oracle.row_nnz(i)
        ell_values = sorted({0, max(row_nnz - 1, 0), row_nnz, s - 1})
        for ell in ell_values:
            column = oracle.index_oracle(i, ell)
            value = oracle.value_oracle_by_position(i, ell)
            rows.append(
                {
                    "case_name": case.case_name,
                    "measurement_setting": case.measurement_setting,
                    "i": int(i),
                    "ell": int(ell),
                    "f_i_ell": int(column),
                    "A_i_f_i_ell": float(value),
                    "row_nnz": int(row_nnz),
                    "padding_status": "padding" if column == oracle.padding_column else "nonzero",
                }
            )
    return rows


def ceil_log2(value: int) -> int:
    integer = int(value)
    if integer <= 1:
        return 0
    return (integer - 1).bit_length()


def fixed_point_integer_bits(max_abs_value: float) -> int:
    value = float(abs(max_abs_value))
    if value < 1.0:
        return 0
    return int(np.floor(np.log2(value))) + 1


def _plot_density(sparsity: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    completed = sparsity[sparsity["run_status"] == "completed"]
    if completed.empty:
        ax.text(0.5, 0.5, "No completed sparsity audits", ha="center", va="center")
    else:
        labels = completed["case_name"].astype(str).tolist()
        values = completed["density"].astype(float).to_numpy()
        bars = ax.bar(labels, values)
        ax.set_yscale("log")
        ax.set_ylabel("density")
        ax.set_title("Weighted Jacobian Density")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        for bar, nnz in zip(bars, completed["nnz"], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"nnz={int(nnz)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_row_nnz_distribution(row_nnz_payload: dict[str, list[int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if not row_nnz_payload:
        ax.text(0.5, 0.5, "No row sparsity data", ha="center", va="center")
    else:
        labels = list(row_nnz_payload)
        data = [row_nnz_payload[label] for label in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_ylabel("row nonzeros")
        ax.set_title("Row Sparsity Distribution")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_normalization_overhead(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    completed = summary[summary["status"] == "completed"]
    if completed.empty:
        ax.text(0.5, 0.5, "No normalization rows", ha="center", va="center")
    else:
        labels = completed["case_name"].astype(str).tolist()
        x = np.arange(len(completed))
        ax.bar(
            x,
            completed["normalization_overhead_sparse_max"].astype(float),
            label=r"$s||A||_{\max}/||A||_2$",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("normalization overhead")
        ax.set_title("Sparse-Oracle Normalization Overhead")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_qubit_estimates(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    completed = summary[summary["status"] == "completed"]
    if completed.empty:
        ax.text(0.5, 0.5, "No qubit estimate rows", ha="center", va="center")
    else:
        labels = completed["case_name"].astype(str).tolist()
        x = np.arange(len(completed))
        width = 0.25
        ax.bar(x - width, completed["row_qubits"].astype(float), width, label="row")
        ax.bar(x, completed["col_qubits"].astype(float), width, label="column")
        ax.bar(
            x + width,
            completed["nonzero_index_qubits"].astype(float),
            width,
            label="nonzero index",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("qubits")
        ax.set_title("Sparse Oracle Register Estimates")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _report_markdown(
    *,
    config: dict[str, Any],
    sparsity: pd.DataFrame,
    verification: pd.DataFrame,
    resources: pd.DataFrame,
    summary: pd.DataFrame,
    sparsity_csv: Path,
    verification_csv: Path,
    resources_csv: Path,
    summary_csv: Path,
) -> str:
    completed = summary[summary["status"] == "completed"]
    status_counts = summary["status"].value_counts(dropna=False).to_dict()
    lines = _report_metric_lines(completed)
    return "\n".join(
        [
            "# Sparse-Oracle Scalable Block-Encoding Model Report",
            "",
            "## Goal",
            "",
            "This experiment constructs and verifies a sparse index/value oracle "
            "emulator for generated weighted PSSE Jacobians.",
            "",
            "## Cases and Weighted-Jacobian Source",
            "",
            f"- Cases requested: {[_case_name_from_spec(spec) for spec in config['cases']]}",
            f"- Case source: {config['case_source']}",
            "- Matrix source: `build_engineering_system` using AC-linearized "
            "weighted Jacobians from `build_ac_weighted_system`.",
            f"- Nonzero threshold: {config['nonzero_tol']}",
            "",
            "## Sparse Oracle Definitions",
            "",
            "- Index oracle emulator: `index_oracle(i, ell)` returns the sorted column "
            "index of the ell-th nonzero in row i.",
            "- Value oracle emulator: `value_oracle(i, j)` returns A_ij and returns 0 "
            "for missing entries.",
            "- Padded convention: ell >= row_nnz(i) returns column -1 and value 0.",
            "- Internal indexing is zero-based.",
            "",
            "## Reconstruction Verification",
            "",
            f"- Status counts: {status_counts}.",
            *lines,
            "",
            "## Resource Estimate Convention",
            "",
            "- s is the maximum row sparsity.",
            "- alpha_sparse_max = s * ||A||_max.",
            "- Qubit counts are register-size estimates for row, column, and nonzero slot indices.",
            f"- Value-bit grid: {config['value_bits_grid']}.",
            "",
            "## Claim-Safe Interpretation",
            "",
            "The oracle emulator reconstructs the weighted Jacobian from row-wise "
            "nonzero indices and values, supporting a sparse-access block-encoding "
            "model. The reported qubit/register counts and normalization factors are "
            "oracle-level resource estimates, not compiled reversible oracle gate counts.",
            "",
            "The dense circuit experiments remain proof-of-concept implementations; "
            "the sparse-oracle model provides a route toward scalable access "
            "assumptions but does not constitute a full hardware-ready block encoding.",
            "",
            "## Limitations",
            "",
            "- No reversible sparse-oracle circuit is synthesized for the full cases.",
            "- No full IEEE-scale QSVT execution or hardware execution is claimed.",
            "- Normalization overhead can reduce effective singular-value scale and "
            "must be accounted for in any future QSVT cost model.",
            "",
            "## Recommended Manuscript Wording",
            "",
            "The weighted PSSE Jacobians generated from IEEE/PYPOWER cases admit a "
            "row-wise sparse index/value oracle representation in classical "
            "emulation. Reconstruction checks verify the oracle model against the "
            "generated weighted matrices, while sparsity, padding, register-size, "
            "and normalization diagnostics quantify the assumptions needed for "
            "future sparse-access QSVT-style block encodings.",
            "",
            "## Artifacts",
            "",
            f"- Sparsity audit CSV: `{sparsity_csv}`",
            f"- Oracle verification CSV: `{verification_csv}`",
            f"- Resource estimates CSV: `{resources_csv}`",
            f"- Summary table: `{summary_csv}`",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )


def _report_metric_lines(completed: pd.DataFrame) -> list[str]:
    if completed.empty:
        return ["- No completed sparse-oracle rows were generated."]
    return [
        f"- nnz range: {int(completed['nnz'].min())} to {int(completed['nnz'].max())}.",
        f"- density range: {completed['density'].min():.3e} to {completed['density'].max():.3e}.",
        f"- max row sparsity range: {int(completed['s'].min())} to {int(completed['s'].max())}.",
        "- reconstruction Frobenius error range: "
        f"{completed['reconstruction_fro_error'].min():.3e} to "
        f"{completed['reconstruction_fro_error'].max():.3e}.",
        "- alpha_sparse_max / ||A||_2 range: "
        f"{completed['normalization_overhead_sparse_max'].min():.3e} to "
        f"{completed['normalization_overhead_sparse_max'].max():.3e}.",
        "- padding overhead range: "
        f"{completed['padding_overhead'].min():.3e} to "
        f"{completed['padding_overhead'].max():.3e}.",
    ]


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(config or {})
    root = Path(supplied.get("output_root", OUTPUT_ROOT))
    resolved: dict[str, Any] = {
        "output_root": str(root),
        "cases": DEFAULT_CASES,
        "case_source": "pypower",
        "seed": 123,
        "nonzero_tol": DEFAULT_NONZERO_TOL,
        "value_bits_grid": DEFAULT_VALUE_BITS,
        "spectral_norm_max_dim": 2500,
        "oracle_sample_rows": 3,
    }
    resolved.update(supplied)
    resolved["cases"] = list(resolved["cases"])
    resolved["value_bits_grid"] = [int(value) for value in resolved["value_bits_grid"]]
    if float(resolved["nonzero_tol"]) < 0.0:
        raise ValueError("nonzero_tol must be nonnegative")
    return resolved


def _matching_row(frame: pd.DataFrame, case: str, measurement: str) -> dict[str, Any]:
    subset = frame[
        (frame["case_name"].astype(str) == case)
        & (frame["measurement_setting"].astype(str) == measurement)
    ]
    return {} if subset.empty else subset.iloc[0].to_dict()


def _matching_resource_row(frame: pd.DataFrame, case: str, measurement: str) -> dict[str, Any]:
    subset = frame[
        (frame["case_name"].astype(str) == case)
        & (frame["measurement_setting"].astype(str) == measurement)
        & (frame["value_bits"].astype(float) == 32.0)
    ]
    if subset.empty:
        subset = frame[
            (frame["case_name"].astype(str) == case)
            & (frame["measurement_setting"].astype(str) == measurement)
        ]
    return {} if subset.empty else subset.iloc[0].to_dict()


def _combined_status(
    audit: pd.Series | dict[str, Any],
    verify: dict[str, Any],
    resource: dict[str, Any],
) -> str:
    statuses = [
        str(audit.get("run_status", "")),
        str(verify.get("oracle_status", "")),
        str(resource.get("resource_status", "")),
    ]
    return "completed" if all(status == "completed" for status in statuses) else ";".join(statuses)


def _sparsity_failure_row(case_name: str, measurement: str, exc: Exception) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "measurement_setting": measurement,
        "matrix_shape_m_by_n": "",
        "m": np.nan,
        "n": np.nan,
        "nnz": np.nan,
        "density": np.nan,
        "max_row_nnz": np.nan,
        "mean_row_nnz": np.nan,
        "median_row_nnz": np.nan,
        "min_nonzero_row_nnz": np.nan,
        "max_col_nnz": np.nan,
        "mean_col_nnz": np.nan,
        "median_col_nnz": np.nan,
        "zero_row_count": np.nan,
        "zero_col_count": np.nan,
        "numerical_rank": np.nan,
        "sigma_min": np.nan,
        "sigma_max": np.nan,
        "condition_number": np.nan,
        "norm_2": np.nan,
        "norm_fro": np.nan,
        "norm_max_abs": np.nan,
        "row_norm_min": np.nan,
        "row_norm_max": np.nan,
        "row_norm_mean": np.nan,
        "weighted_status": "unavailable",
        "row_weighting_convention": "unavailable",
        "matrix_source": "unavailable",
        "run_status": "failed_or_skipped",
        "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
    }


def _verification_failure_row(case_name: str, measurement: str, exc: Exception) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "measurement_setting": measurement,
        "m": np.nan,
        "n": np.nan,
        "nnz": np.nan,
        "max_row_nnz": np.nan,
        "padding_column": PADDING_COLUMN_SENTINEL,
        "uses_row_padding": True,
        "reconstruction_fro_error": np.nan,
        "reconstruction_spectral_error": np.nan,
        "reconstruction_max_abs_error": np.nan,
        "reconstruction_nnz_mismatch": np.nan,
        "row_nnz_mismatch_count": np.nan,
        "value_mismatch_count": np.nan,
        "oracle_status": "skipped_input_unavailable",
        "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
    }


def _resource_failure_row(
    case_name: str,
    measurement: str,
    value_bits: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "measurement_setting": measurement,
        "m": np.nan,
        "n": np.nan,
        "s": np.nan,
        "nnz": np.nan,
        "density": np.nan,
        "row_qubits": np.nan,
        "col_qubits": np.nan,
        "nonzero_index_qubits": np.nan,
        "value_bits": int(value_bits),
        "sign_bit": 1,
        "fixed_point_integer_bits": np.nan,
        "fixed_point_fraction_bits": np.nan,
        "alpha_sparse_max": np.nan,
        "alpha_fro": np.nan,
        "alpha_row_norm": np.nan,
        "norm_2": np.nan,
        "normalization_overhead_sparse_max": np.nan,
        "normalization_overhead_fro": np.nan,
        "normalization_overhead_row_norm": np.nan,
        "sigma_max_over_alpha_sparse_max": np.nan,
        "sigma_min_over_alpha_sparse_max": np.nan,
        "effective_condition_number_if_rank_full": np.nan,
        "padded_sparse_entries": np.nan,
        "padding_overhead": np.nan,
        "estimated_oracle_calls_per_block_encoding": np.nan,
        "query_model": "sparse_index_value_oracle",
        "block_encoding_normalization": "alpha_sparse_max = s * ||A||_max",
        "tiny_reversible_oracle_status": "skipped_input_unavailable",
        "notes": "Input matrix unavailable; row retained for auditability.",
        "resource_status": "skipped_input_unavailable",
        "failure_or_skip_reason": f"{type(exc).__name__}: {exc}",
    }


def _case_name_from_spec(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, dict):
        return str(spec.get("case_name", "unknown"))
    return str(spec)


def _measurement_setting_from_spec(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, dict):
        return str(spec.get("measurement_setting", "full_ac_measurement_set"))
    return "full_ac_measurement_set"


def _deterministic_sample_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    raw = np.linspace(0, size - 1, num=min(size, count), dtype=int)
    return sorted(set(int(value) for value in raw))


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if denom == 0.0:
        return np.inf if float(numerator) != 0.0 else 0.0
    return float(numerator) / denom


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TQE sparse-oracle model experiment")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    args = parser.parse_args(argv)
    run = run_sparse_oracle_block_encoding_model(
        {
            "output_root": args.output_root,
            "cases": list(args.cases),
        }
    )
    print(f"TQE sparse-oracle block-encoding model complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
