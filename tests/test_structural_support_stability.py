from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_stability_registry_covers_frozen_subsets_selectors_and_realizations() -> None:
    frame = pd.read_csv(OUT / "support_stability.csv")
    assert len(frame) == 24 * 3 * 4
    assert frame["instance_id"].nunique() == 24
    assert frame["structural_group_id"].nunique() == 12
    assert set(frame["training_subset"]) == {
        "first_half",
        "second_half",
        "even_positions",
        "odd_positions",
    }
    assert set(frame["selector"]) == {
        "balanced_magnitude",
        "sensitivity_initial_mean",
        "sensitivity_refined_mean",
    }
    assert frame["jaccard_similarity"].between(0.0, 1.0).all()
    assert (~frame["held_out_used_for_support_construction"].astype(bool)).all()


def test_stability_summary_reports_median_worst_and_heldout_association() -> None:
    summary = pd.read_csv(OUT / "support_stability_summary.csv")
    assert set(summary["selector"]) == {
        "balanced_magnitude",
        "sensitivity_initial_mean",
        "sensitivity_refined_mean",
    }
    assert summary["median_jaccard"].between(0.0, 1.0).all()
    assert summary["worst_jaccard"].between(0.0, 1.0).all()
    assert "spearman_instability_vs_heldout_error" in summary
    grouped = pd.read_csv(OUT / "support_stability_group_summary.csv")
    assert grouped["structural_group_id"].nunique() == 12
