from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_candidate_registry_has_predeclared_case_policy_balance() -> None:
    config = json.loads((ROOT / "configs/output_aware_structural_generalization.json").read_text())
    registry = pd.read_csv(OUT / "candidate_registry.csv")
    assert len(registry) == 120
    assert set(registry["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert set(registry["policy"]) == set(config["candidate_pool"]["policies"])
    assert (registry.groupby(["ieee_case", "policy"]).size() == 4).all()
    assert registry["outcome_independent"].astype(bool).all()
    assert (~registry["selector_outcomes_used_for_inclusion"].astype(bool)).all()




def test_candidate_exclusions_are_retained_with_reasons() -> None:
    exclusions = pd.read_csv(OUT / "candidate_exclusion_registry.csv")
    if not exclusions.empty:
        assert exclusions["failure_reason"].fillna("").str.len().gt(0).all()
        assert (exclusions["status"] == "excluded").all()
