from __future__ import annotations

import json

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.qiskit_matrix_qsvt import run_qiskit_matrix_qsvt


def test_qiskit_research_matrix_qsvt_builds_real_circuit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("qiskit")
    pytest.importorskip("pennylane")
    run = run_qiskit_matrix_qsvt(
        {
            "demo": {
                "output_dir": str(tmp_path / "qiskit_matrix"),
                "polynomial_degree": 5,
                "grid_size": 256,
                "angle_solver": "iterative",
            },
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "submatrix_size": 2,
                "seed": 123,
            },
        }
    )
    output_dir = run["output_dir"]
    comparison = pd.read_csv(output_dir / "comparison_to_classical.csv")
    with (output_dir / "circuit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert (output_dir / "circuit_draw.txt").is_file()
    assert summary["qiskit_available"] is True
    assert summary["is_full_matrix_qsvt"] is False
    assert summary["full_or_submatrix"] == "submatrix"
    assert summary["n_qubits"] >= 1
    assert summary["circuit_depth"] > 0
    assert summary["gate_count_total"] > 0
    assert "transpile_success" in summary
    assert (output_dir / "transpiled_circuit_summary.json").is_file()
    assert comparison["abs_error_to_classical_filter"].max() < 5.0e-2
