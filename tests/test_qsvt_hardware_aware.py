from __future__ import annotations

import json

import pandas as pd

from robust_qsvt_se.qsvt.hardware_aware_report import (
    HARDWARE_AWARE_CAVEAT,
    build_hardware_aware_report,
)


def test_hardware_aware_proxy_report_has_nonnegative_estimates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_hardware_aware_report(
        {
            "output_dir": str(tmp_path / "hardware"),
            "matrix_source": "synthetic",
            "degrees": [5, 11],
            "epsilon": 10.0,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "hardware_aware_summary.csv")
    payload = json.loads((output_dir / "hardware_aware_summary.json").read_text(encoding="utf-8"))

    row = frame.loc[0]
    assert int(row["logical_qubits_estimate"]) >= 1
    assert int(row["total_qubits_estimate"]) >= int(row["logical_qubits_estimate"])
    assert int(row["estimated_depth"]) >= 0
    assert int(row["estimated_two_qubit_gates"]) >= 0
    assert HARDWARE_AWARE_CAVEAT in row["hardware_caveat"]
    assert {item["dependency"] for item in payload["optional_dependencies"]} == {
        "qiskit",
        "qiskit_aer",
        "pennylane",
    }
    assert (output_dir / "hardware_assumptions.md").is_file()
    assert (output_dir / "manifest.json").is_file()
