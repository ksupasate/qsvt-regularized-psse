from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_candidate_descriptor_and_group_selection_are_output_blind() -> None:
    candidates = pd.read_csv(OUT / "candidate_registry.csv")
    descriptors = pd.read_csv(OUT / "structural_descriptors.csv")
    groups = pd.read_csv(OUT / "structural_group_registry.csv")
    assert candidates["outcome_independent"].astype(bool).all()
    assert (~candidates["selector_outcomes_used_for_inclusion"].astype(bool)).all()
    assert (~descriptors["descriptor_uses_selector_or_output_results"].astype(bool)).all()
    assert (~groups["selector_outcomes_used_for_selection"].astype(bool)).all()


def test_training_and_heldout_residual_seeds_are_disjoint_per_realization() -> None:
    residuals = pd.read_csv(OUT / "residual_registry.csv")
    for _instance, local in residuals.groupby("instance_id"):
        training = set(local.loc[local["split"] == "training", "residual_seed"])
        heldout = set(local.loc[local["split"] == "held_out", "residual_seed"])
        assert training.isdisjoint(heldout)
        assert len(training) == len(heldout) == 20


