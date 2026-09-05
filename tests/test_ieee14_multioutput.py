"""Tests for IEEE-14 multi-output validation (WP-F).

Reads ieee14_multioutput_statevector.csv and asserts three PRE-SELECTED outputs
(theta_2, V_1, area-aggregate angle) all pass with convention error at machine
precision and QSVT matching Ridge. Integrity invariant: outputs must be the
pre-registered set, not selected post-hoc.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "generalized_rectangular_qsvt"
CSV = OUT / "ieee14_multioutput_statevector.csv"

EXPECTED_OUTPUT_KEYWORDS = ["theta_2", "V_1", "area_aggregate_angle"]


def test_multioutput_results_exist():
    df = pd.read_csv(CSV)
    assert len(df) == 3


def test_outputs_are_preselected():
    df = pd.read_csv(CSV)
    for kw in EXPECTED_OUTPUT_KEYWORDS:
        assert any(kw in str(o) for o in df["output"]), f"pre-selected output {kw} missing"


def test_all_outputs_convention_exact():
    df = pd.read_csv(CSV)
    assert (df["selected_rel_err_vs_exact"] < 1e-10).all()


def test_qsvt_matches_ridge():
    df = pd.read_csv(CSV)
    assert (df["selected_rel_err_vs_ridge"] < 0.01).all()


def test_headline_output_reproduced():
    df = pd.read_csv(CSV)
    theta2 = df[df["output"].str.contains("theta_2")].iloc[0]
    # headline y_QSVT = 0.004936843087993346
    assert abs(theta2["y_qsvt"] - 0.004936843087993346) < 1e-5
