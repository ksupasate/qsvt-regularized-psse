"""Tests for postselection-mitigation prototype (WP-K).

Reads postselection_mitigation_executed_results.csv and cost_comparison.csv.
Integrity invariant (MUST FAIL if modeled mitigation is labeled executed):
  - the controlled case is labeled EXECUTED and runs on Aer;
  - the IEEE-14 cost rows are labeled MODELED, never EXECUTED.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
EXEC = OUT / "postselection_mitigation_executed_results.csv"
COST = OUT / "postselection_mitigation_cost_comparison.csv"


def test_executed_case_present():
    df = pd.read_csv(EXEC)
    assert len(df) >= 2
    assert (df["method"].isin(["MLAE", "direct_sampling"])).all()


def test_executed_case_runs_on_aer():
    df = pd.read_csv(EXEC)
    # the executed controlled case must be labeled as Aer-executed
    assert (df["setting"] == "controlled_aer_executed").all()


def test_modeled_ieee14_never_labeled_executed():
    """Integrity invariant: modeled mitigation must NOT be labeled executed."""

    df = pd.read_csv(COST)
    ieee = df[df["setting"].str.contains("ieee14", na=False)]
    assert (ieee["status"] == "MODELED").all(), (
        "IEEE-14 mitigation cost labeled executed (should be MODELED)"
    )


def test_executed_mlae_accurate():
    df = pd.read_csv(EXEC)
    mlae = df[df["method"] == "MLAE"]
    # MLAE estimates should be within ~0.05 of the true amplitude on the controlled case
    assert (mlae["abs_error"] < 0.05).all()
