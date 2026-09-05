from __future__ import annotations

import json

import pandas as pd
import pytest

from robust_qsvt_se.qsvt.pennylane_matrix_qsvt import run_pennylane_matrix_qsvt


def test_pennylane_research_matrix_qsvt_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_pennylane_matrix_qsvt(
        {
            "demo": {
                "output_dir": str(tmp_path / "pennylane_matrix"),
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

    assert (output_dir / "research_matrix_metadata.json").is_file()
    assert (output_dir / "circuit_draw.txt").is_file()
    assert summary["is_full_matrix_qsvt"] is False
    assert summary["full_or_submatrix"] == "submatrix"
    assert summary["source_case"] == "ieee14"
    assert comparison["abs_error_to_classical_filter"].max() < 5.0e-2


def test_pennylane_full_ieee14_matrix_qsvt_writes_scope_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_pennylane_matrix_qsvt(
        {
            "demo": {
                "output_dir": str(tmp_path / "pennylane_full_matrix"),
                "polynomial_degree": 5,
                "grid_size": 256,
                "angle_solver": "iterative",
            },
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "matrix_scope": "full_matrix",
                "use_full_matrix": True,
                "seed": 123,
            },
        }
    )
    summary = run["summary"]
    output_dir = run["output_dir"]
    comparison = pd.read_csv(output_dir / "comparison_to_classical.csv")

    assert summary["is_full_matrix_qsvt"] is True
    assert summary["full_or_submatrix"] == "full_matrix"
    assert summary["matrix_shape"] == [82, 27]
    assert (output_dir / "polynomial_coefficients.csv").is_file()
    assert (output_dir / "approximation_error.csv").is_file()
    assert comparison["abs_error_to_classical_filter"].max() < 5.0e-2
