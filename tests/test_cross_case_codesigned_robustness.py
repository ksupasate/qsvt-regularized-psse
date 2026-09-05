from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_codesigned_robustness import (
    CROSS_CASE_CLASSES,
    DEFAULT_SELECTION_MODES,
    _mark_gate_validation_recommended,
    classify_case,
    classify_cross_case,
)


def _by_subproblem(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_classify_case_follows_documented_rules() -> None:
    none_feasible = _by_subproblem(
        [
            {
                "selection_mode": "high_leverage",
                "any_residual_feasible": False,
                "is_control": False,
            },
        ]
    )
    assert classify_case(none_feasible) == "no_feasible_selected_blocks"

    single = _by_subproblem(
        [
            {"selection_mode": "high_leverage", "any_residual_feasible": True, "is_control": False},
            {
                "selection_mode": "best_conditioned",
                "any_residual_feasible": False,
                "is_control": False,
            },
        ]
    )
    assert classify_case(single) == "single_selected_block"

    family = _by_subproblem(
        [
            {"selection_mode": "high_leverage", "any_residual_feasible": True, "is_control": False},
            {
                "selection_mode": "metadata_mapped",
                "any_residual_feasible": True,
                "is_control": False,
            },
        ]
    )
    assert classify_case(family) == "selected_subproblem_family"


def test_cross_case_classification_requires_transfer_and_uses_criteria_modes() -> None:
    by_case = pd.DataFrame(
        [
            {"case": "ieee14", "case_classification": "selected_subproblem_family"},
            {"case": "ieee30", "case_classification": "single_selected_block"},
            {"case": "ieee57", "case_classification": "no_feasible_selected_blocks"},
        ]
    )
    classification, per_case = classify_cross_case(by_case)
    assert classification == "cross_case_partial"
    assert classification in CROSS_CASE_CLASSES
    assert per_case["ieee57"] == "no_feasible_selected_blocks"

    both = pd.DataFrame(
        [
            {"case": "ieee30", "case_classification": "selected_subproblem_family"},
            {"case": "ieee57", "case_classification": "single_selected_block"},
        ]
    )
    assert classify_cross_case(both)[0] == "cross_case_selected_family"

    neither = pd.DataFrame(
        [
            {"case": "ieee30", "case_classification": "no_feasible_selected_blocks"},
            {"case": "ieee57", "case_classification": "no_feasible_selected_blocks"},
        ]
    )
    assert classify_cross_case(neither)[0] == "ieee14_only"

    # Criteria-based selection: the control mode is part of the default set but is a failure case.
    assert "worst_conditioned_control" in DEFAULT_SELECTION_MODES
    assert "high_leverage" in DEFAULT_SELECTION_MODES


def test_worst_conditioned_control_never_marked_for_gate_validation() -> None:
    rows = [
        {
            "case": "ieee30",
            "subproblem_id": "worst_conditioned_control_09",
            "selection_mode": "worst_conditioned_control",
            "residual_feasible": True,
            "residual_ratio_vs_no_update": 0.001,
            "gate_validation_recommended": False,
        },
        {
            "case": "ieee30",
            "subproblem_id": "high_leverage_00",
            "selection_mode": "high_leverage",
            "residual_feasible": True,
            "residual_ratio_vs_no_update": 0.02,
            "gate_validation_recommended": False,
        },
    ]
    marked = _mark_gate_validation_recommended(rows)
    control = [row for row in marked if row["selection_mode"] == "worst_conditioned_control"]
    high = [row for row in marked if row["selection_mode"] == "high_leverage"]
    assert control[0]["gate_validation_recommended"] is False
    assert high[0]["gate_validation_recommended"] is True
