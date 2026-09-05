from __future__ import annotations

import json

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.qiskit_matrix_qsvt import run_qiskit_matrix_qsvt


def test_qiskit_research_matrix_4x4_records_transpilation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")
    run = run_qiskit_matrix_qsvt(
        {
            "demo": {
                "output_dir": str(tmp_path / "qk_4x4"),
                "polynomial_degree": 5,
                "grid_size": 256,
                "angle_solver": "iterative",
                "transpile_qubit_limit": 4,
            },
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "matrix_scope": "submatrix",
                "submatrix_size": 4,
                "selection_strategy": "high_leverage",
                "seed": 123,
            },
        }
    )
    output_dir = run["output_dir"]
    comparison = pd.read_csv(output_dir / "comparison_to_classical.csv")
    with (output_dir / "circuit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["full_or_submatrix"] == "submatrix"
    assert summary["simulation_success"] is True
    assert "transpile_seconds" in summary
    assert "basis_gates" in summary
    assert (output_dir / "transpiled_circuit_summary.json").is_file()
    assert comparison["abs_error_to_classical_filter"].max() < 5.0e-2
