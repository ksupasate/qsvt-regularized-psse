"""Tests for IEEE-57 escalation (WP-I).

Reads ieee57_escalation_results.csv. Integrity invariant: IEEE-57 execution is
documented with the convention at machine precision; useful-overlap status
(pass or fail) is recorded honestly, not omitted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "ieee57_escalation_results.csv"


def test_escalation_recorded():
    df = pd.read_csv(CSV)
    assert len(df) >= 6
    headline = df[(df["lambda"] == 1e-3) & (df["degree"].isin([127, 255]))]
    assert set(headline["degree"].astype(int)) == {127, 255}


def test_convention_exact_at_ieee57():
    df = pd.read_csv(CSV)
    assert df["convention_block_error_vs_exact_svd"].max() < 1e-10


def test_status_honestly_recorded():
    df = pd.read_csv(CSV)
    allowed = {"STATEVECTOR_PASSED", "APPLICATION_FAILED", "RESOURCE_LIMIT"}
    assert set(df["status"]).issubset(allowed)


def test_headline_useful_overlap_rows_reproduced():
    df = pd.read_csv(CSV)
    headline = df[(df["lambda"] == 1e-3) & (df["degree"].isin([127, 255]))]
    assert len(headline) == 2
    assert (headline["status"] == "STATEVECTOR_PASSED").all()
    assert (headline["rmse_ratio"] <= 1.25).all()


def test_wall_time_recorded():
    """IEEE-57 execution feasibility is evidenced by a measured wall time."""

    df = pd.read_csv(CSV)
    assert (df["wall_time_s"] > 0).all()
