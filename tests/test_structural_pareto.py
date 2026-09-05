from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def _assert_frontier_is_nondominated(
    candidates: pd.DataFrame, frontier: pd.DataFrame, cost: str
) -> None:
    assert not frontier.empty
    assert frontier["nondominated"].astype(bool).all()
    for row in frontier.itertuples(index=False):
        local = candidates[candidates["instance_id"] == row.instance_id]
        dominates = (
            (local["median_normalized_error"] <= row.median_normalized_error)
            & (local[cost] <= getattr(row, cost))
            & (
                (local["median_normalized_error"] < row.median_normalized_error)
                | (local[cost] < getattr(row, cost))
            )
        )
        assert not dominates.any()




def test_pareto_accuracy_objective_is_absolute_normalized_error() -> None:
    candidates = pd.read_csv(OUT / "pareto_candidates_error_nnz.csv")
    assert (candidates["median_normalized_error"] >= 0.0).all()
    assert set(candidates["accuracy_objective"]) == {
        "median_heldout_normalized_error_absolute_not_signed"
    }
