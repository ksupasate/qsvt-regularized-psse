from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.ieee118_selected_robustness import (
    DEFAULT_SELECTION_MODES,
    classify_ieee118,
)


def _by_subproblem(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_selection_modes_are_criteria_based_only() -> None:
    # No mode selects subproblems by post hoc QSVT performance; all are numerical/metadata criteria.
    assert set(DEFAULT_SELECTION_MODES) == {
        "high_leverage",
        "metadata_mapped",
        "residual_supported",
        "best_conditioned",
        "random_seeded_pool",
        "worst_conditioned_control",
    }
    assert not any("qsvt" in mode for mode in DEFAULT_SELECTION_MODES)


def test_worst_conditioned_control_not_counted_as_positive_evidence() -> None:
    # Only the control is feasible -> still classified as no feasible selected blocks.
    frame = _by_subproblem(
        [
            {
                "selection_mode": "worst_conditioned_control",
                "any_residual_feasible": True,
                "is_control": True,
            },
            {
                "selection_mode": "high_leverage",
                "any_residual_feasible": False,
                "is_control": False,
            },
        ]
    )
    assert classify_ieee118(frame) == "no_feasible_selected_blocks"


def test_classify_ieee118_family_and_single() -> None:
    single = _by_subproblem(
        [
            {"selection_mode": "high_leverage", "any_residual_feasible": True, "is_control": False},
            {
                "selection_mode": "metadata_mapped",
                "any_residual_feasible": False,
                "is_control": False,
            },
        ]
    )
    assert classify_ieee118(single) == "single_selected_block"

    family = _by_subproblem(
        [
            {"selection_mode": "high_leverage", "any_residual_feasible": True, "is_control": False},
            {
                "selection_mode": "metadata_mapped",
                "any_residual_feasible": True,
                "is_control": False,
            },
            {
                "selection_mode": "residual_supported",
                "any_residual_feasible": True,
                "is_control": False,
            },
            {
                "selection_mode": "worst_conditioned_control",
                "any_residual_feasible": True,
                "is_control": True,
            },
        ]
    )
    # The control's feasibility does not inflate the count beyond the two non-control blocks.
    assert classify_ieee118(family) == "selected_subproblem_family"
