from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.codesigned_subproblem_robustness import (
    DEFAULT_SELECTION_MODES,
    ROBUSTNESS_CLASSES,
    classify_robustness,
)


def _by_subproblem(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_robustness_classification_follows_documented_rules() -> None:
    # Only the high-leverage block is feasible -> single_block_only.
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
    classification, modes = classify_robustness(single)
    assert classification == "single_block_only"
    assert modes == ["high_leverage"]

    # Three distinct non-control modes feasible -> moderate_selected_family.
    moderate = _by_subproblem(
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
        ]
    )
    classification, _ = classify_robustness(moderate)
    assert classification == "moderate_selected_family"
    assert classification in ROBUSTNESS_CLASSES


def test_robustness_excludes_control_and_uses_criteria_selection() -> None:
    # The worst-conditioned control is criteria-selected as a failure case and must never
    # count toward feasibility, even if it were marked feasible.
    frame = _by_subproblem(
        [
            {
                "selection_mode": "worst_conditioned_control",
                "any_residual_feasible": True,
                "is_control": True,
            },
        ]
    )
    classification, modes = classify_robustness(frame)
    assert classification == "inconclusive"
    assert modes == []

    # The default selection modes are numerical/metadata criteria, not QSVT-performance picks.
    assert "worst_conditioned_control" in DEFAULT_SELECTION_MODES
    assert "high_leverage" in DEFAULT_SELECTION_MODES
    assert "metadata_mapped" in DEFAULT_SELECTION_MODES
