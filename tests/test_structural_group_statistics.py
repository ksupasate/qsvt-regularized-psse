from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.output_aware_structural_generalization import (
    structural_group_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_group_bootstrap_is_deterministic_and_does_not_resample_realizations() -> None:
    pairs = pd.DataFrame(
        {
            "structural_group_id": ["a", "b", "c", "d"],
            "ieee_case": ["ieee14", "ieee14", "ieee30", "ieee30"],
            "paired_difference_candidate_minus_baseline": [-0.2, 0.1, -0.3, 0.2],
        }
    )
    first = structural_group_bootstrap(pairs, samples=100, seed=9, case_stratified=True)
    second = structural_group_bootstrap(pairs, samples=100, seed=9, case_stratified=True)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["resampling_unit"]) == {"structural_group"}
    assert set(first["structural_group_count"]) == {4}


def test_primary_counts_bootstraps_and_group_metric_reconcile() -> None:
    primary = json.loads((OUT / "structural_primary_test.json").read_text())
    pairs = pd.read_csv(OUT / "structural_primary_matched_pairs.csv")
    bootstrap = pd.read_csv(OUT / "structural_group_bootstrap.csv")
    stratified = pd.read_csv(OUT / "structural_case_stratified_bootstrap.csv")
    assert len(pairs) == primary["structural_groups"] == 12
    assert sum(primary["overall_win_tie_loss"].values()) == 12
    assert {
        outcome: sum(values[outcome] for values in primary["case_win_tie_loss"].values())
        for outcome in ("win", "tie", "loss")
    } == primary["overall_win_tie_loss"]
    assert len(bootstrap) == len(stratified) == 10_000
    assert set(bootstrap["resampling_unit"]) == {"structural_group"}
    assert set(stratified["bootstrap_mode"]) == {"case_stratified"}
    assert set(pairs["realization_combination"]) == {"mean_of_realization_medians"}


def test_functional_pairs_cover_every_group_and_functional() -> None:
    frame = pd.read_csv(OUT / "structural_primary_functional_pairs.csv")
    assert len(frame) == 12 * 3
    assert (frame.groupby(["structural_group_id", "functional_id"]).size() == 1).all()
    assert set(frame["functional_id"]) == {
        "coordinate_e0",
        "signed_difference_e0_minus_e1",
        "aggregate_e0_to_e3",
    }
