from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import (
    classify_matched_errors,
    paired_instance_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_win_tie_loss_uses_declared_relative_tolerance() -> None:
    assert classify_matched_errors(0.5, 1.0, relative_tolerance=0.01, epsilon=1e-15) == "win"
    assert classify_matched_errors(1.0, 0.5, relative_tolerance=0.01, epsilon=1e-15) == "loss"
    assert classify_matched_errors(0.995, 1.0, relative_tolerance=0.01, epsilon=1e-15) == "tie"


def test_bootstrap_resamples_instances_and_is_deterministic() -> None:
    differences = np.array([-0.3, 0.1, -0.2, 0.4])
    first = paired_instance_bootstrap(differences, samples=200, seed=77)
    second = paired_instance_bootstrap(differences, samples=200, seed=77)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["instance_count"]) == {4}
    assert set(first["resampling_unit"]) == {"instance"}


def test_primary_comparison_and_case_counts_are_frozen_and_reconcile() -> None:
    result = json.loads((OUT / "generalization_primary_test.json").read_text())
    declaration = result["primary_comparison_declaration"]
    assert declaration["candidate_selector"] == "sensitivity_initial_mean"
    assert declaration["baseline_selector"] == "balanced_magnitude"
    assert (declaration["k_budget"], declaration["slot_budget"]) == (16, 3)
    overall = result["primary_overall_win_tie_loss"]
    case_total = {
        key: sum(values[key] for values in result["primary_case_win_tie_loss"].values())
        for key in ("win", "tie", "loss")
    }
    assert overall == case_total
    bootstrap = pd.read_csv(OUT / "generalization_bootstrap.csv")
    assert len(bootstrap) == 10_000
    assert set(bootstrap["resampling_unit"]) == {"instance"}

