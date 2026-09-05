from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_training_and_heldout_residuals_are_disjoint_per_instance() -> None:
    residuals = pd.read_csv(OUT / "residual_registry.csv")
    for _instance_id, group in residuals.groupby("instance_id", sort=True):
        training = set(group[group["split"] == "training"]["residual_seed"])
        heldout = set(group[group["split"] == "held_out"]["residual_seed"])
        assert training.isdisjoint(heldout)
        assert len(training) == 20
        assert len(heldout) == 20


def test_heldout_never_enters_scoring_or_refinement() -> None:
    supports = pd.read_csv(OUT / "support_registry.csv")
    sensitive = supports[supports["selector"].str.startswith("sensitivity")]
    assert (sensitive["selection_data_split"] == "training_only").all()
    assert not sensitive["selection_data_split"].str.contains("held", case=False).any()
    scores = pd.read_csv(OUT / "entry_scores.csv")
    assert set(scores["selection_data_split"]) == {"training_only"}


def test_instance_inclusion_precedes_support_evaluation() -> None:
    checkpoint = json.loads((OUT / "checkpoint.json").read_text())
    instances_time = checkpoint["stages"]["instances"]["completed_at"]
    supports_time = checkpoint["stages"]["supports"]["completed_at"]
    assert instances_time <= supports_time


def test_qsvt_instance_subset_is_explicitly_predeclared() -> None:
    config = json.loads((ROOT / "configs/output_aware_generalization.json").read_text())
    identifiers = config["qsvt"]["predeclared_instance_ids"]
    assert len(identifiers) == 6
    assert sum(value.startswith("ieee14") for value in identifiers) == 2
    assert sum(value.startswith("ieee30") for value in identifiers) == 2
    assert sum(value.startswith("ieee57") for value in identifiers) == 2


def test_functionals_are_deterministic_and_metadata_grounded() -> None:
    registry = pd.read_csv(OUT / "functional_registry.csv")
    assert len(registry) == 45
    assert set(registry["functional_family"]) == {"coordinate", "difference", "aggregate"}
    assert registry["semantic_status"].str.startswith("metadata_grounded").all()
    assert (registry["selection_data_used"] == "state_metadata_only_no_output_accuracy").all()

