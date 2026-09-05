"""Tests for the generalized error budget (section 20).

Reads generalized_error_budget.csv. Integrity invariant: application and
implementation errors are NOT combined; the convention/rectangular/block-encoding
errors are at machine precision (~1e-14), separating them from the dominant
application/polynomial-approximation terms.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "generalized_error_budget.csv"


def test_budget_has_all_sources():
    df = pd.read_csv(CSV)
    assert len(df) == 12


def test_application_and_implementation_separated():
    df = pd.read_csv(CSV)
    tiers = set(df["tier"])
    assert {"application", "deterministic", "statistical", "not_applied"}.issubset(tiers)


def test_convention_errors_at_machine_precision():
    df = pd.read_csv(CSV).set_index("source")
    assert df.loc["4_convention_conversion", "value"] < 1e-10
    assert df.loc["6_rectangular_action", "value"] < 1e-10
    assert df.loc["7_block_encoding", "value"] < 1e-10


def test_dominant_error_is_approximation_not_convention():
    """The largest deterministic error must be polynomial approximation, NOT convention."""

    df = pd.read_csv(CSV)
    det = df[df["tier"] == "deterministic"].dropna(subset=["value"])
    conv_val = det[det["source"] == "4_convention_conversion"]["value"].iloc[0]
    poly_val = det[det["source"] == "2_polynomial_approximation"]["value"].iloc[0]
    assert poly_val > conv_val * 1e6  # poly approx dominates convention by many orders
