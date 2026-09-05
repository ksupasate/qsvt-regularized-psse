from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.pennylane_demo import run_pennylane_demo
from robust_qsvt_se.qsvt.qiskit_demo import run_qiskit_demo


def test_pennylane_qsvt_demo_runs_when_available(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_pennylane_demo(
        {
            "demo": {
                "output_dir": str(tmp_path / "pennylane"),
                "matrix_diagonal": [0.2, 0.6],
                "alpha": 1.0,
            }
        }
    )
    output_dir = run["output_dir"]
    comparison = pd.read_csv(output_dir / "comparison_to_classical.csv")

    assert (output_dir / "circuit_draw.txt").is_file()
    assert (output_dir / "qsvt_pennylane_summary.json").is_file()
    assert np.all(np.isfinite(comparison["pennylane_qsvt_block_value"]))


def test_qiskit_demo_builds_real_circuit_when_available(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("qiskit")
    run = run_qiskit_demo(
        {
            "demo": {
                "output_dir": str(tmp_path / "qiskit"),
                "singular_values": [0.2, 0.6],
                "alpha": 1.0,
            }
        }
    )
    output_dir = run["output_dir"]
    with (output_dir / "circuit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert (output_dir / "circuit_draw.txt").is_file()
    assert summary["qiskit_available"] is True
    assert summary["n_qubits"] == 1
    assert summary["circuit_depth"] > 0
    assert summary["gate_count_total"] > 0
    assert summary["is_full_matrix_qsvt"] is False
