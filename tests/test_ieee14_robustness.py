"""Tests for IEEE-14 robustness (WP-G).

Reads ieee14_robustness_results.csv. Integrity invariants:
  - all perturbation families and >=10 seeds per family are present;
  - no failed IEEE case is omitted (every recorded case is visible);
  - the convention block error stays at machine precision under perturbation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "ieee14_robustness_results.csv"


def test_robustness_results_exist():
    df = pd.read_csv(CSV)
    assert len(df) >= 30


def test_all_families_present():
    df = pd.read_csv(CSV)
    assert {"gaussian", "bad_data", "missing"}.issubset(set(df["family"]))


def test_at_least_10_seeds_per_family():
    df = pd.read_csv(CSV)
    for fam in ["gaussian", "bad_data", "missing"]:
        assert len(df[df["family"] == fam]) >= 10


def test_no_hidden_failures():
    """Every recorded case is visible; none silently dropped."""

    df = pd.read_csv(CSV)
    statuses = set(df["status"])
    # STATEVECTOR_PASSED or a documented failure stage; no row may be 'hidden'
    allowed = {
        "STATEVECTOR_PASSED",
        "APPLICATION_FAILED",
        "POLYNOMIAL_FAILED",
        "PHASE_FAILED",
        "RECTANGULAR_ACTION_FAILED",
        "RESOURCE_LIMIT",
    }
    assert statuses.issubset(allowed), f"unexpected status: {statuses - allowed}"


def test_convention_exact_under_perturbation():
    df = pd.read_csv(CSV)
    assert df["convention_block_error_vs_exact_svd"].max() < 1e-10


def test_application_passes_threshold():
    df = pd.read_csv(CSV)
    # primary criterion: RMSE ratio <= 1.25 for the validated configurations
    assert df["rmse_ratio"].max() <= 1.25
