from __future__ import annotations

from pathlib import Path

import pytest

from robust_qsvt_se.qsvt.run_hardware_qsvt_demo import run_hardware_qsvt_demo


def test_hardware_qsvt_demo_writes_structured_circuit_artifacts(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")

    output_dir = tmp_path / "hardware_qsvt"
    run = run_hardware_qsvt_demo(
        {
            "demo": {
                "run_id": "test_hardware_qsvt",
                "output_dir": str(output_dir),
                "alpha": 0.05,
                "polynomial_degree": 5,
                "grid_size": 128,
                "phase_cache_dir": str(tmp_path / "phase_cache"),
                "transpile_basis_gates": ["rz", "sx", "x", "cx"],
            },
            "matrix": {
                "case_name": "ieee14",
                "case_source": "pypower",
                "matrix_scope": "submatrix",
                "submatrix_size": 2,
                "selection_strategy": "high_leverage",
                "seed": 123,
            },
        }
    )

    required = [
        "config_resolved.yaml",
        "research_matrix_metadata.json",
        "singular_values.csv",
        "phase_angles.csv",
        "block_encoding_summary.json",
        "hardware_qsvt_circuit_summary.json",
        "hardware_qsvt_gate_counts.json",
        "hardware_qsvt_transpiled_summary.json",
        "hardware_qsvt_transpiled_gate_counts.json",
        "circuit_draw.txt",
        "transpiled_circuit_draw.txt",
        "simulation_results.csv",
        "comparison_to_classical.csv",
        "run.log",
    ]
    for filename in required:
        assert (output_dir / filename).is_file()

    summary = run["summary"]
    assert summary["is_dense_unitary_only"] is False
    assert summary["uses_dense_block_encoding_gate"] is True
    assert summary["transpile_success"] is True
    assert summary["max_error_vs_classical"] < 1.0e-4
    assert summary["n_phase_angles"] == 6
