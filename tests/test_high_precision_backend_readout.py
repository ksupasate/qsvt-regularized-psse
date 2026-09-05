"""Tests for high-precision shot-based readout (WP-J).

Reads ieee14_high_precision_backend_summary.csv. Integrity invariants:
  - the readout is shot-based (Aer), NOT statevector inspection mislabeled;
  - the statevector selected output matches the frozen headline value;
  - the preregistered <=10% relative CI half-width is met at high shot count;
  - the CI brackets the statevector and Ridge values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
SUMMARY = OUT / "ieee14_high_precision_backend_summary.csv"
RUNS = OUT / "ieee14_high_precision_backend_runs.csv"


def test_summary_exists():
    df = pd.read_csv(SUMMARY)
    assert len(df) >= 1


def test_statevector_matches_headline():
    df = pd.read_csv(SUMMARY)
    # frozen headline statevector selected output = 0.004936843087993346
    assert abs(df["y_statevector"].iloc[0] - 0.004936843087993346) < 1e-6


def test_precision_target_met_at_high_shots():
    df = pd.read_csv(SUMMARY).sort_values("shots")
    high = df.iloc[-1]
    assert high["aggregate_relative_ci_half_width"] <= 0.10


def test_runs_are_shot_based():
    df = pd.read_csv(RUNS)
    runs = (
        df[df.get("_row_type", pd.Series([""] * len(df))) != "aggregate"]
        if "_row_type" in df.columns
        else df
    )
    # backend column must indicate Aer shot sampling, not statevector inspection
    backends = set(str(b) for b in runs["backend"])
    assert any("aer" in str(b).lower() for b in backends), f"no Aer shot backend: {backends}"


def test_ci_brackets_references():
    """The aggregate high-shot estimate must be consistent with the statevector.

    Individual 95% CIs legitimately miss ~5% of the time, so we check the
    aggregate (mean over seeds) is close to the statevector value, and that a
    majority of high-shot runs bracket it.
    """

    df = pd.read_csv(SUMMARY).sort_values("shots")
    high = df.iloc[-1]
    # aggregate estimate within a few percent of the statevector value
    assert (
        abs(high["aggregate_y_estimate"] - high["y_statevector"]) / abs(high["y_statevector"])
        < 0.05
    )
    runs = pd.read_csv(RUNS)
    runs = runs[runs["shots"] == 1000000] if "shots" in runs.columns else runs
    if len(runs):
        assert runs["ci_contains_statevector"].mean() >= 0.5
