from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.sparse_oracle_gates import (
    build_toy_sparse_oracle_gate_circuit,
    run_toy_sparse_oracle_gate_demo,
    sparse_oracle_entries,
    toy_sparse_matrix,
    validate_toy_sparse_oracle_gate_circuit,
)


def test_sparse_oracle_entries_cover_two_slots_per_row() -> None:
    entries = sparse_oracle_entries(toy_sparse_matrix(4))

    assert len(entries) == 8
    assert {entry.row for entry in entries} == {0, 1, 2, 3}
    assert {entry.local_index for entry in entries} == {0, 1}


def test_toy_sparse_oracle_gate_circuit_validates_lookup_and_rotation() -> None:
    circuit, entries = build_toy_sparse_oracle_gate_circuit(4)
    validation = validate_toy_sparse_oracle_gate_circuit(circuit, entries)

    assert circuit.num_qubits == 6
    assert validation["validation_passed"] is True
    assert validation["max_column_lookup_error"] == 0
    assert validation["max_value_probability_error"] < 1.0e-12


def test_toy_sparse_oracle_gate_demo_writes_outputs(tmp_path: Path) -> None:
    run = run_toy_sparse_oracle_gate_demo(
        {
            "output_dir": str(tmp_path),
            "matrix_size": 4,
            "seed": 123,
        }
    )

    assert run["artifacts"]["manifest"].is_file()
    assert run["artifacts"]["oracle_circuit_summary"].is_file()
    assert run["artifacts"]["oracle_validation"].is_file()
    assert run["artifacts"]["toy_sparse_oracle_qasm"].is_file()
    assert run["validation"]["validation_passed"] is True
