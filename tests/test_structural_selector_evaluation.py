from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_support_registry_has_frozen_selector_budget_multiplicity() -> None:
    registry = pd.read_csv(OUT / "support_registry.csv")
    assert len(registry) == 24 * 25 * 36
    deterministic = registry[registry["selector"] != "random_objective_feasible"]
    random = registry[registry["selector"] == "random_objective_feasible"]
    assert len(deterministic) == 24 * 25 * 6
    assert len(random) == 24 * 25 * 30
    assert (
        deterministic.groupby(["instance_id", "k_budget", "slot_budget", "selector"]).size() == 1
    ).all()
    assert (random.groupby(["instance_id", "k_budget", "slot_budget"]).size() == 30).all()




def test_heldout_summary_retains_two_realizations_per_group_selector() -> None:
    summary = pd.read_csv(OUT / "heldout_instance_summary.csv")
    selected = summary[
        (summary["k_budget"] == 16)
        & (summary["slot_budget"] == 3)
        & summary["selector"].isin(["balanced_magnitude", "sensitivity_initial_mean"])
        & (summary["status"] == "completed")
    ]
    assert (
        selected.groupby(["structural_group_id", "selector"])["instance_id"].nunique() == 2
    ).all()
