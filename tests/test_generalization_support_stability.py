from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import jaccard_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_jaccard_handles_identical_disjoint_and_empty_supports() -> None:
    first = np.array([[True, False], [False, True]])
    second = np.array([[False, True], [True, False]])
    assert jaccard_similarity(first, first) == 1.0
    assert jaccard_similarity(first, second) == 0.0
    assert jaccard_similarity(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0


def test_training_subset_schedule_is_predeclared_and_used() -> None:
    config = json.loads((ROOT / "configs/output_aware_generalization.json").read_text())
    expected = set(config["stability"]["training_subset_schedules"])
    frame = pd.read_csv(OUT / "support_stability.csv")
    assert set(frame["training_subset"]) == expected
    assert set(frame["selection_data_split"]) == {"training_only"}
    assert (~frame["held_out_used_for_support_construction"].astype(bool)).all()
    assert frame["jaccard_similarity"].between(0.0, 1.0).all()


def test_magnitude_stability_is_one_and_summary_handles_relations() -> None:
    frame = pd.read_csv(OUT / "support_stability.csv")
    magnitude = frame[frame["selector"] == "balanced_magnitude"]
    assert np.allclose(magnitude["jaccard_similarity"], 1.0)
    summary = pd.read_csv(OUT / "support_stability_summary.csv")
    assert set(summary["selector"]) == {
        "balanced_magnitude",
        "sensitivity_initial_mean",
        "sensitivity_refined_mean",
    }
    assert summary["median_jaccard"].between(0.0, 1.0).all()

