"""One-status guards for IEEE case evidence tiers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"


def test_ieee57_is_matrix_action_only() -> None:
    matrix = pd.read_csv(OUT / "ieee_case_evidence_matrix.csv", keep_default_na=False)
    row = matrix[matrix["case"] == "IEEE-57"].iloc[0]
    assert "qsvt_matrix_action" in row["status"]
    assert row["sampled_readout"] == "no"
    assert row["dense_statevector"].startswith("no")
    assert row["complete_transpilation"] == "no"


def test_case_rows_preserve_family_specific_tiers() -> None:
    matrix = pd.read_csv(OUT / "ieee_case_evidence_matrix.csv", keep_default_na=False)
    assert set(matrix["case"]) == {"IEEE-14", "IEEE-30", "IEEE-57"}
    row14 = matrix[matrix["case"] == "IEEE-14"].iloc[0]
    row30 = matrix[matrix["case"] == "IEEE-30"].iloc[0]
    assert "sampled_simulator" in row14["status"]
    assert "qsvt_matrix_action" in row30["status"]
    assert row30["sampled_readout"] == "no"

