"""Tests for structured PSSE access (WP-L).

Reads structured_psse_access_ieee14.csv. Integrity invariants:
  - the classical sparse reconstruction is EXECUTED with reconstruction error 0;
  - the quantum oracle is MODELED, NOT EXECUTED;
  - no dense preconstructed unitary is hidden in the structured-access claim
    (reconstruction must come from sparse rows, verified to match the dense H).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "structured_psse_access_ieee14.csv"


def test_structured_access_recorded():
    df = pd.read_csv(CSV)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["classical_sparse_access_status"] == "EXECUTED"
    assert r["quantum_oracle_status"] == "MODELED"
    assert r["overall_status"] == "STRUCTURED_ACCESS_MODELED_ONLY"


def test_quantum_oracle_not_labeled_executed():
    """Integrity invariant: modeled access must NOT be labeled executed."""

    df = pd.read_csv(CSV)
    r = df.iloc[0]
    assert r["quantum_oracle_status"] != "EXECUTED"


def test_classical_reconstruction_exact():
    df = pd.read_csv(CSV)
    r = df.iloc[0]
    assert r["reconstruction_error_executed"] == 0.0


def test_sparsity_structure_present():
    df = pd.read_csv(CSV)
    r = df.iloc[0]
    assert r["max_row_sparsity"] <= 16  # PSSE rows are sparse
    assert r["address_width_bits"] >= 1
    assert r["total_nnz"] > 0
