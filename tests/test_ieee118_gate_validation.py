from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.ieee118_gate_validation import (
    GATE_COLUMNS,
    select_ieee118_gate_candidates,
    surviving_subproblems,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "ieee118",
                "subproblem_id": "high_leverage_00",
                "selection_mode": "high_leverage",
                "row_indices": "185 303 536 722",
                "col_indices": "67 114 184 232",
                "alpha": 1.0e-4,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 10.1,
                "residual_ratio_vs_no_update": 0.0002,
            },
            {
                "case": "ieee118",
                "subproblem_id": "metadata_mapped_01",
                "selection_mode": "metadata_mapped",
                "row_indices": "118 236 355 541",
                "col_indices": "0 1 117 118",
                "alpha": 1.0e-6,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 7.0,
                "residual_ratio_vs_no_update": 0.00006,
            },
            {
                "case": "ieee118",
                "subproblem_id": "worst_conditioned_control_09",
                "selection_mode": "worst_conditioned_control",
                "row_indices": "1 2 3 4",
                "col_indices": "1 2 3 4",
                "alpha": 1.0e-4,
                "degree": 25,
                "target_family": "residual_aware",
                "condition_number": 1.0e17,
                "residual_ratio_vs_no_update": 0.07,
            },
        ]
    )


def test_selection_reads_only_candidates_and_excludes_control() -> None:
    selected = select_ieee118_gate_candidates(_candidates(), max_configs=3)
    modes = set(selected["selection_mode"].astype(str))
    # The control is never selected for positive gate validation.
    assert "worst_conditioned_control" not in modes
    assert {"high_leverage", "metadata_mapped"}.issubset(modes)
    # Only candidate-file columns are carried through; indices survive for reconstruction.
    assert (selected["row_indices"].astype(str).str.len() > 0).all()
    assert (selected["col_indices"].astype(str).str.len() > 0).all()


def test_selection_respects_max_configs_and_empty_input() -> None:
    selected = select_ieee118_gate_candidates(_candidates(), max_configs=1)
    assert len(selected) == 1
    # The preferred high_leverage candidate fills the single slot.
    assert selected.iloc[0]["selection_mode"] == "high_leverage"

    empty = select_ieee118_gate_candidates(pd.DataFrame(), max_configs=3)
    assert empty.empty


def test_gate_columns_include_ridge_residual_and_indices() -> None:
    for column in [
        "ridge_residual",
        "state_error_gate_vs_polynomial",
        "row_indices",
        "col_indices",
    ]:
        assert column in GATE_COLUMNS


def test_surviving_subproblems_reads_feasibility_flag() -> None:
    results = pd.DataFrame(
        [
            {"subproblem_id": "high_leverage_00", "residual_feasible_after_gate": True},
            {"subproblem_id": "metadata_mapped_01", "residual_feasible_after_gate": False},
        ]
    )
    assert surviving_subproblems(results) == ["high_leverage_00"]
    assert surviving_subproblems(pd.DataFrame()) == []
