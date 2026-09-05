from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robust_qsvt_se.qsvt.toy_sparse_oracle_block_encoding_v2 import (
    build_sparse_oracle_v2_circuit,
    count_controlled_rotations,
    dense_block_encoding_resource_estimate,
    fixed_point_encode,
    precision_sweep_rows,
    run_sparse_oracle_prototype_strengthening,
    sparse_oracle_resource_summary,
    sparsify_block,
    validate_sparse_oracle_v2_circuit,
)

_TOY_4X4 = np.asarray(
    [
        [1000.0, 0.0, -120.0, 0.0],
        [0.0, -800.0, 60.0, 0.0],
        [300.0, 0.0, 0.0, -45.0],
        [0.0, 250.0, -200.0, 0.0],
    ],
    dtype=np.float64,
)


def test_fixed_point_encode_is_signed_and_bounded() -> None:
    value, bits = fixed_point_encode(0.5, bits=8)
    assert -1.0 <= value <= 1.0
    assert len(bits) == 8
    assert bits[-1] == 0  # positive sign bit
    neg_value, neg_bits = fixed_point_encode(-0.5, bits=8)
    assert neg_value < 0.0
    assert neg_bits[-1] == 1


def test_sparsify_block_keeps_two_per_row() -> None:
    sparse = sparsify_block(_TOY_4X4, keep_per_row=2)
    assert np.all(np.count_nonzero(sparse, axis=1) <= 2)
    # Largest-magnitude entry per row is retained.
    assert sparse[0, 0] == 1000.0


def test_oracle_validates_lookup_column_value_register_and_rotation() -> None:
    sparse = sparsify_block(_TOY_4X4, keep_per_row=2)
    prototype = build_sparse_oracle_v2_circuit(sparse, value_precision_bits=8, slots_per_row=2)
    validation = validate_sparse_oracle_v2_circuit(prototype)

    assert validation["validation_passed"] is True
    assert validation["max_column_lookup_error"] == 0
    assert validation["max_value_register_error"] == 0
    assert validation["max_value_probability_error"] < 1.0e-12
    # The rotation qubit reproduces the quantized value for every active entry.
    for entry in validation["entries"]:
        target = float(np.sin(entry["rotation_angle"] / 2.0) ** 2)
        assert abs(entry["value_one_probability"] - target) < 1.0e-12


def test_precision_sweep_error_is_non_increasing() -> None:
    sparse = sparsify_block(_TOY_4X4, keep_per_row=2)
    rows = precision_sweep_rows(sparse, value_precision_bits=[4, 8, 12], slots_per_row=2)
    errors = [row["max_normalized_value_loading_error"] for row in rows]
    assert [row["value_precision_bits"] for row in rows] == [4, 8, 12]
    assert errors[0] >= errors[1] >= errors[2]
    assert errors[2] < errors[0]  # strictly improves overall


def test_controlled_rotation_and_two_qubit_counts_positive() -> None:
    sparse = sparsify_block(_TOY_4X4, keep_per_row=2)
    prototype = build_sparse_oracle_v2_circuit(sparse, value_precision_bits=4, slots_per_row=2)
    resource = sparse_oracle_resource_summary(prototype, basis_gates=["u3", "cx"])
    assert resource["controlled_rotation_count"] >= 1
    assert resource["two_qubit_gate_count_after_transpile"] >= 1
    assert resource["transpile_succeeded"] is True
    assert count_controlled_rotations(prototype.circuit) == resource["controlled_rotation_count"]


def test_dense_estimate_and_sparse_comparison_are_well_formed() -> None:
    sparse = sparsify_block(_TOY_4X4, keep_per_row=2)
    prototype = build_sparse_oracle_v2_circuit(sparse, value_precision_bits=8, slots_per_row=2)
    resource = sparse_oracle_resource_summary(prototype, basis_gates=["u3", "cx"])
    dense = dense_block_encoding_resource_estimate(sparse, value_precision_bits=8)
    assert dense["two_qubit_gate_count"] > 0
    assert dense["encoded_unitary_dimension"] == 8
    # Dense block encoding loads the full 2N-dimensional unitary, so it needs more
    # qubits than the sparse-access oracle's address registers (encoded dim 8 -> 3 qubits).
    assert dense["n_qubits"] < resource["n_qubits"]
    assert resource["two_qubit_gate_count_after_transpile"] > 0


def test_run_writes_all_required_outputs(tmp_path: Path) -> None:
    run = run_sparse_oracle_prototype_strengthening(
        {
            "output_dir": str(tmp_path),
            "case": "ieee14",
            "model": "dc_linearized",
            "matrix_size": 4,
            "value_precision_bits": [4, 8, 12],
            "seed": 123,
        }
    )
    artifacts = run["artifacts"]
    for key in (
        "manifest",
        "sparse_oracle_v2_resource_summary",
        "sparse_oracle_precision_sweep",
        "sparse_vs_dense_resource_comparison",
        "sparse_oracle_validation",
        "sparse_oracle_v2_qasm",
        "sparse_oracle_strengthening_interpretation",
    ):
        assert artifacts[key].is_file(), key

    assert run["validation"]["overall_validation_passed"] is True

    comparison = run["comparison_rows"]
    assert len(comparison) == 3
    assert all("dense_two_qubit_gate_count" in row for row in comparison)

    validation = json.loads(artifacts["sparse_oracle_validation"].read_text(encoding="utf-8"))
    assert set(validation["per_precision"].keys()) == {"4", "8", "12"}

    interpretation = artifacts["sparse_oracle_strengthening_interpretation"].read_text(
        encoding="utf-8"
    )
    assert "MISSING" in interpretation
    assert "hardware-native sparse-oracle synthesis" in interpretation
