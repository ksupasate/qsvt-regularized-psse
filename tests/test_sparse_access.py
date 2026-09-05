from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_common import forbidden_in
from robust_qsvt_se.paper.sparse_access_workload import (
    SUMMARY_COLUMNS,
    VALIDATION_COLUMNS,
    run_sparse_access_workload,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.sparse_access import build_sparse_access_model


def _ieee14_matrix() -> np.ndarray:
    system, _ = build_engineering_system(
        {"case_name": "ieee14", "case_source": "pypower", "matrix_source": "weighted_jacobian"}
    )
    return np.asarray(system.H_tilde, dtype=np.float64)


def test_lookup_matches_dense_matrix() -> None:
    matrix = _ieee14_matrix()
    model = build_sparse_access_model(matrix, case="ieee14")
    # Every structural entry recovered exactly through O_col / O_val.
    for row in range(model.num_rows):
        columns, values = model.get_row_nonzeros(row)
        for local_index, (col, value) in enumerate(zip(columns, values, strict=True)):
            assert model.get_col(row, local_index) == int(col)
            assert model.get_val(row, int(col)) == pytest.approx(float(value), abs=1e-12)
            assert model.get_val(row, int(col)) == pytest.approx(matrix[row, int(col)], abs=1e-12)


def test_row_counts_and_max_row_nnz() -> None:
    matrix = _ieee14_matrix()
    model = build_sparse_access_model(matrix, case="ieee14")
    expected_counts = np.count_nonzero(np.abs(matrix) > 1e-12, axis=1)
    assert np.array_equal(model.row_nonzero_counts, expected_counts)
    assert model.max_row_nnz == int(expected_counts.max())
    assert model.nnz == int(expected_counts.sum())
    # index_qubits = ceil(log2(rows)) + ceil(log2(cols)) for the |i,j> register.
    assert model.index_qubits == model.row_index_qubits + model.col_index_qubits


def test_validation_passes_on_exact_lookup() -> None:
    matrix = _ieee14_matrix()
    model = build_sparse_access_model(matrix, case="ieee14")
    record = model.validate_against_dense_or_csr(reference=matrix)
    assert record["access_status"] == "validated_exact_lookup"
    assert record["value_max_abs_error"] == 0.0
    assert record["col_index_mismatches"] == 0
    assert record["invalid_index_raises"] is True
    assert record["reversible_oracle_synthesized"] is False


def test_invalid_indices_raise() -> None:
    matrix = _ieee14_matrix()
    model = build_sparse_access_model(matrix, case="ieee14")
    with pytest.raises(IndexError):
        model.get_col(model.num_rows, 0)
    with pytest.raises(IndexError):
        model.get_val(model.num_rows, 0)
    with pytest.raises(IndexError):
        # local nonzero index past the row's nnz must raise
        model.get_col(0, int(model.row_nonzero_counts[0]) + 100)


def test_workload_outputs_have_required_columns(tmp_path: Path) -> None:
    run = run_sparse_access_workload(
        {"output_dir": str(tmp_path), "cases": ["ieee14", "ieee30"], "command": "test"}
    )
    summary = run["summary"]
    validation = run["validation"]
    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(VALIDATION_COLUMNS).issubset(validation.columns)
    assert bool((~summary["reversible_oracle_synthesized"]).all())
    assert bool((validation["value_max_abs_error"] == 0.0).all())

    for name in (
        "sparse_access_summary.csv",
        "sparse_access_summary.json",
        "sparse_access_validation.csv",
        "sparse_access_report.md",
        "sparse_access_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    report = (tmp_path / "sparse_access_report.md").read_text(encoding="utf-8")
    assert forbidden_in(report) == []
    payload = json.loads((tmp_path / "sparse_access_summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] and "reversible_oracle_synthesized" in payload["rows"][0]
