from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.paper.reversible_sparse_oracle import (
    build_column_oracle,
    build_value_oracle,
    decode_value,
    formal_cost_model,
    quantize_value,
    run_reversible_sparse_oracle,
    validate_column_oracle,
    validate_value_oracle,
)

pytest.importorskip("qiskit")


def _toy_sparse_matrix() -> np.ndarray:
    # Rows 0 and 3 have 1 nonzero; rows 1, 2 have 2 -> exercises padding slots.
    return np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.4, 0.2, 0.0],
            [0.3, 0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 0.4],
        ],
        dtype=np.float64,
    )


def test_column_oracle_returns_correct_columns():
    matrix = _toy_sparse_matrix()
    built = build_column_oracle(matrix, max_nonzeros=2)
    validation = validate_column_oracle(built)
    assert validation["passed"]
    assert validation["max_lookup_error"] == 0
    # First stored nonzero of row 1 is column 1; of row 2 is column 0.
    by_key = {(e["row"], e["slot"]): e for e in validation["entries"]}
    assert by_key[(1, 0)]["observed_column"] == 1
    assert by_key[(2, 0)]["observed_column"] == 0


def test_invalid_index_sets_invalid_flag():
    matrix = _toy_sparse_matrix()
    built = build_column_oracle(matrix, max_nonzeros=2)
    validation = validate_column_oracle(built)
    by_key = {(e["row"], e["slot"]): e for e in validation["entries"]}
    # Row 0 has a single nonzero; slot 1 is a padding (invalid) slot.
    assert by_key[(0, 1)]["valid"] is False
    assert by_key[(0, 1)]["observed_invalid_flag"] == 1
    # A valid slot keeps the invalid flag at 0.
    assert by_key[(0, 0)]["observed_invalid_flag"] == 0


def test_value_oracle_is_bit_exact():
    matrix = _toy_sparse_matrix()
    built = build_value_oracle(matrix, magnitude_bits=6)
    validation = validate_value_oracle(built)
    assert validation["bit_exact"]
    assert validation["max_decode_error"] == pytest.approx(0.0, abs=1e-12)


def test_value_quantization_roundtrip_within_step():
    scale = 0.5
    bits = 6
    for value in (-0.4, 0.2, -0.1, 0.5, 0.0):
        sign, magnitude = quantize_value(value, scale, bits)
        decoded = decode_value(sign, magnitude, scale, bits)
        assert abs(decoded - value) <= scale / ((1 << bits) - 1) + 1e-12


def test_formal_cost_model_register_widths():
    model = formal_cost_model(
        case="ieee300", rows=1722, cols=599, nnz=8054, max_nonzeros_per_row=24, value_bits=6
    )
    assert model["row_index_qubits"] == 11  # ceil(log2(1722))
    assert model["column_register_qubits"] == 10  # ceil(log2(599))
    assert model["value_precision_bits"] == 6
    assert model["uncomputation_required"] is True
    assert model["val_oracle_toffoli_qrom"] == 8054
    assert model["total_t_count_qrom"] > 0


def test_run_reversible_sparse_oracle_writes_outputs(tmp_path):
    run = run_reversible_sparse_oracle(
        {"output_dir": str(tmp_path), "ieee_cases": ["ieee14"], "block_npy": "does_not_exist.npy"}
    )
    assert run["column_validation"]["passed"]
    assert run["value_validation"]["passed"]
    summary = run["resource_summary"]
    assert "synthesized_small_scale" in set(summary["status"])
    assert "modeled" in set(summary["status"])
    assert (tmp_path / "oracle_resource_summary.csv").is_file()
    assert (tmp_path / "oracle_metadata.json").is_file()
