from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.hardware_resource_estimator import (
    build_oracle_model_resource_report,
    qubit_convention_counts,
)


def test_qubit_convention_reports_row_and_column_registers() -> None:
    counts = qubit_convention_counts((82, 27))

    assert counts["row_qubits"] == 7
    assert counts["col_qubits"] == 5
    assert counts["padded_dimension_qubits"] == 7
    assert (
        counts["total_logical_qubits_row_col_convention"]
        > counts["total_logical_qubits_padded_convention"]
    )


def test_oracle_model_resource_report_has_required_columns(tmp_path: Path) -> None:
    run = build_oracle_model_resource_report(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "degree": 35,
            "phase_count": 36,
        }
    )
    frame = pd.read_csv(run["artifacts"]["oracle_model_resource_summary"])

    assert "row_qubits" in frame.columns
    assert "col_qubits" in frame.columns
    assert "total_logical_qubits_row_col_convention" in frame.columns
    assert frame.loc[0, "implemented_or_estimated"] == "oracle_model_resource_estimate"


def test_ieee300_resource_row_does_not_claim_dense_qsvt(tmp_path: Path) -> None:
    run = build_oracle_model_resource_report(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee300"],
            "degree": 35,
            "phase_count": 36,
        }
    )
    frame = pd.read_csv(run["artifacts"]["oracle_model_resource_summary"])

    assert frame.loc[0, "implemented_or_estimated"] == "oracle_model_resource_estimate"
    assert "hardware execution" in frame.loc[0, "limitations"]
