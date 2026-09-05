from __future__ import annotations

import json

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.run_full_matrix_qsvt import run_full_matrix_qsvt


def test_full_matrix_qsvt_runner_writes_feasibility_and_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")
    run = run_full_matrix_qsvt(
        {
            "run": {
                "output_dir": str(tmp_path / "full_matrix"),
                "max_pennylane_full_qubits": 8,
                "max_qiskit_dense_full_qubits": 1,
                "submatrix_size": 2,
            },
            "demo": {
                "polynomial_degree": 5,
                "grid_size": 256,
                "angle_solver": "iterative",
                "transpile_qubit_limit": 4,
            },
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "mode": "ac_weighted_jacobian",
                "seed": 123,
            },
        }
    )
    output_dir = run["output_dir"]
    status = pd.read_csv(output_dir / "qsvt_full_matrix_status.csv")
    with (output_dir / "feasibility_decision.json").open("r", encoding="utf-8") as file:
        feasibility = json.load(file)

    assert feasibility["pennylane_full_feasible"] is True
    assert feasibility["qiskit_dense_full_feasible"] is False
    assert bool(status.loc[0, "resource_estimate"]) is True
    assert (output_dir / "research_matrix_metadata.json").is_file()
    assert (output_dir / "full_matrix_resource_estimate.json").is_file()
    assert (output_dir / "pennylane_summary.json").is_file()
    assert (output_dir / "qiskit_summary.json").is_file()
