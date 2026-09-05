from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_gate_validation import (
    GATE_COLUMNS,
    select_cross_case_gate_candidates,
    surviving_cases,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "ieee14",
                "subproblem_id": "high_leverage_00",
                "selection_mode": "high_leverage",
                "row_indices": "0 1 2 3",
                "col_indices": "0 1 2 3",
                "alpha": 1.0e-3,
                "degree": 25,
                "target_family": "residual_aware",
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.0002,
            },
            {
                "case": "ieee30",
                "subproblem_id": "high_leverage_00",
                "selection_mode": "high_leverage",
                "row_indices": "1 4 7 9",
                "col_indices": "2 3 5 8",
                "alpha": 1.0e-3,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 1.3,
                "residual_ratio_vs_no_update": 2.6e-10,
            },
            {
                "case": "ieee57",
                "subproblem_id": "high_leverage_00",
                "selection_mode": "high_leverage",
                "row_indices": "3 6 8 11",
                "col_indices": "1 5 7 9",
                "alpha": 1.0e-5,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 8.5,
                "residual_ratio_vs_no_update": 0.00022,
            },
            {
                "case": "ieee30",
                "subproblem_id": "worst_conditioned_control_09",
                "selection_mode": "worst_conditioned_control",
                "row_indices": "0 2 4 6",
                "col_indices": "0 2 4 6",
                "alpha": 1.0e-4,
                "degree": 25,
                "target_family": "residual_aware",
                "condition_number": 1.0e17,
                "residual_ratio_vs_no_update": 0.05,
            },
        ]
    )


def test_selection_covers_cross_cases_and_excludes_control() -> None:
    selected = select_cross_case_gate_candidates(_candidates(), max_configs=6)
    cases = set(selected["case"].astype(str))
    modes = set(selected["selection_mode"].astype(str))

    assert "ieee30" in cases
    assert "ieee57" in cases
    # The worst-conditioned control is never selected for gate validation.
    assert "worst_conditioned_control" not in modes
    # Indices are carried through so the subproblem can be reconstructed downstream.
    assert (selected["row_indices"].astype(str).str.len() > 0).all()


def test_selection_respects_max_configs_and_reads_only_candidates() -> None:
    selected = select_cross_case_gate_candidates(_candidates(), max_configs=2)
    assert len(selected) == 2
    # With only two slots, both must be the guaranteed cross-case representatives.
    assert set(selected["case"].astype(str)) == {"ieee30", "ieee57"}

    # Empty input yields an empty, well-formed selection.
    empty = select_cross_case_gate_candidates(pd.DataFrame(), max_configs=6)
    assert empty.empty


def test_surviving_cases_reads_feasibility_flag() -> None:
    results = pd.DataFrame(
        [
            {"case": "ieee30", "residual_feasible_after_gate": True},
            {"case": "ieee57", "residual_feasible_after_gate": False},
            {"case": "ieee14", "residual_feasible_after_gate": True},
        ]
    )
    assert surviving_cases(results) == ["ieee14", "ieee30"]
    assert surviving_cases(pd.DataFrame()) == []


def test_gate_columns_include_required_and_indices() -> None:
    for column in [
        "case",
        "cross_case_validation_status",
        "residual_feasible_after_gate",
        "state_error_gate_vs_polynomial",
    ]:
        assert column in GATE_COLUMNS
    # Indices are appended so the observable-readout phase can reconstruct the subproblem.
    assert "row_indices" in GATE_COLUMNS
    assert "col_indices" in GATE_COLUMNS
