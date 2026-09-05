from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.output_aware_generalization import grouped_pareto_frontier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_equal_cost_lower_error_dominates_and_construction_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "instance_id": ["a", "a", "a", "b"],
            "support_id": ["high", "low", "tradeoff", "other"],
            "error": [0.2, 0.1, 0.05, 0.9],
            "cost": [10.0, 10.0, 20.0, 1.0],
        }
    )
    first_candidates, first_frontier = grouped_pareto_frontier(
        frame,
        group_columns=["instance_id"],
        error_column="error",
        cost_column="cost",
    )
    second_candidates, second_frontier = grouped_pareto_frontier(
        frame,
        group_columns=["instance_id"],
        error_column="error",
        cost_column="cost",
    )
    pd.testing.assert_frame_equal(first_candidates, second_candidates)
    pd.testing.assert_frame_equal(first_frontier, second_frontier)
    assert "high" not in set(first_frontier["support_id"])
    assert {"low", "tradeoff", "other"} == set(first_frontier["support_id"])


def test_campaign_pareto_uses_normalized_not_signed_error_and_retains_candidates() -> None:
    heldout = pd.read_csv(OUT / "heldout_instance_summary.csv")
    candidates = pd.read_csv(OUT / "pareto_candidates_error_nnz.csv")
    frontier = pd.read_csv(OUT / "pareto_frontier_error_nnz.csv")
    completed = heldout[heldout["status"] == "completed"]
    assert len(candidates) == len(completed)
    assert set(candidates["accuracy_objective"]) == {
        "median_heldout_normalized_error_absolute_not_signed"
    }
    assert "signed_error" not in candidates.columns
    assert set(frontier["support_id"]).issubset(set(candidates["support_id"]))
    assert frontier["nondominated"].astype(bool).all()


def test_missing_gate_costs_are_not_zero_and_complete_registry_is_retained() -> None:
    candidates = pd.read_csv(OUT / "pareto_candidates_error_gates.csv")
    missing = candidates["signal_unitary_gate_count"].isna()
    assert (candidates.loc[missing, "missing_resource_cost_is_zero"] == False).all()  # noqa: E712
    assert candidates.loc[missing, "nondominated"].eq(False).all()

