"""Tests for IEEE-30 useful-overlap search (WP-H).

Reads ieee30_useful_overlap_search.csv. Integrity invariants:
  - the search is staged (degrees 31..255) and every candidate is recorded;
  - unsuccessful candidates are NOT hidden (APPLICATION_FAILED rows present);
  - the convention block error is machine-precision on EVERY candidate
    (the convention generalizes to IEEE-30 even where the application fails).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "ieee30_useful_overlap_search.csv"


def test_search_recorded_all_candidates():
    df = pd.read_csv(CSV)
    assert len(df) >= 12
    assert {31, 63, 127, 255}.issubset(set(int(d) for d in df["degree"]))


def test_failed_candidates_not_hidden():
    df = pd.read_csv(CSV)
    assert (df["status"] == "APPLICATION_FAILED").sum() >= 1, (
        "failed candidates must remain visible"
    )


def test_convention_exact_on_all_candidates():
    df = pd.read_csv(CSV)
    assert df["convention_block_error_vs_exact_svd"].max() < 1e-10


def test_at_least_one_useful_overlap():
    df = pd.read_csv(CSV)
    passed = df[df["status"] == "STATEVECTOR_PASSED"]
    assert len(passed) >= 1, (
        "IEEE-30 must reach useful overlap in at least one staged configuration"
    )
    assert (passed["rmse_ratio"] <= 1.25).all()
