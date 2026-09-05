from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.output_aware_structural_generalization import (
    composite_structural_distance,
    select_structurally_diverse_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def _descriptor(identifier: str, offset: int) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "ieee_case": "ieee14",
        "row_set": [offset, offset + 1],
        "column_set": [offset, offset + 2],
        "support_cells": [[offset, offset], [offset + 1, offset + 2]],
        "measurement_type_proportions": {"p": 0.5, "q": 0.5},
        "state_type_proportions": {"angle": 0.5, "voltage": 0.5},
    }


def test_composite_distance_is_symmetric_bounded_and_zero_on_identity() -> None:
    weights = {"rows": 0.25, "columns": 0.25, "support": 0.2, "measurement": 0.15, "state": 0.15}
    first = _descriptor("a", 0)
    second = _descriptor("b", 3)
    same = composite_structural_distance(first, first, weights)
    forward = composite_structural_distance(first, second, weights)
    reverse = composite_structural_distance(second, first, weights)
    assert same["composite_distance"] == 0.0
    assert forward == reverse
    assert 0.0 <= forward["composite_distance"] <= 1.0


def test_farthest_point_selection_is_deterministic() -> None:
    weights = {"rows": 0.25, "columns": 0.25, "support": 0.2, "measurement": 0.15, "state": 0.15}
    values = [_descriptor(f"c{index}", 3 * index) for index in range(4)]
    first = select_structurally_diverse_candidates(
        values,
        previous_descriptor=_descriptor("previous", 30),
        weights=weights,
        preferred_count=3,
        minimum_distance=0.1,
    )
    second = select_structurally_diverse_candidates(
        values,
        previous_descriptor=_descriptor("previous", 30),
        weights=weights,
        preferred_count=3,
        minimum_distance=0.1,
    )
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]


def test_selected_groups_meet_diversity_contract() -> None:
    config = json.loads((ROOT / "configs/output_aware_structural_generalization.json").read_text())
    groups = pd.read_csv(OUT / "structural_group_registry.csv")
    assert len(groups) == 12
    assert (groups.groupby("ieee_case").size() == 4).all()
    later = groups[groups["selection_order"] > 1]
    assert (
        later["minimum_distance_to_earlier_selected"]
        >= config["structural_selection"]["minimum_pairwise_distance"]
    ).all()
    for _case, local in groups.groupby("ieee_case"):
        assert local["selected_rows"].nunique() == len(local)
        assert local["selected_columns"].nunique() == len(local)
        assert local["support_pattern_fingerprint"].nunique() == len(local)
    assert (~groups["selector_outcomes_used_for_selection"].astype(bool)).all()
