from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.ieee118_extension_audit import (
    REUSABLE_COMPONENT_COLUMNS,
    SELECTION_MODE_COLUMNS,
    ieee118_candidate_selection_modes,
    ieee118_load_status,
    reusable_cross_case_components,
)


def test_reusable_components_are_marked_cross_case() -> None:
    components = reusable_cross_case_components()
    assert list(components.columns) == REUSABLE_COMPONENT_COLUMNS
    # Every listed component must be reusable for IEEE118 (case-agnostic pipeline).
    assert components["supports_ieee118"].astype(bool).all()
    assert "_build_system" in set(components["component"].astype(str))


def test_ieee118_loads_and_supports_4x4_extraction() -> None:
    modes = ieee118_candidate_selection_modes(
        case="ieee118",
        model="ac_linearized",
        case_source="pypower",
        submatrix_size=4,
        selection_modes=[
            "high_leverage",
            "metadata_mapped",
            "residual_supported",
            "best_conditioned",
            "random_seeded_pool",
            "worst_conditioned_control",
        ],
        seed=123,
    )
    assert list(modes.columns) == SELECTION_MODE_COLUMNS
    status = ieee118_load_status(modes)
    # IEEE118 loads through the shared pypower pipeline and supports 4x4 extraction.
    assert status["loads"] is True
    assert status["supports_4x4_extraction"] is True
    # The worst-conditioned control never counts as positive evidence.
    assert "worst_conditioned_control" not in status["positive_evidence_modes"]
    assert "high_leverage" in status["positive_evidence_modes"]


def test_load_status_records_structured_failure_when_unavailable() -> None:
    # A structured failure (empty/unavailable selection modes) is reported, not an exception.
    failed = pd.DataFrame(
        [
            {
                "case": "ieee118",
                "selection_mode": "high_leverage",
                "criteria_based": True,
                "available": False,
                "n_candidates": 0,
                "counts_as_positive_evidence": True,
                "note": "load_failed:RuntimeError",
            }
        ],
        columns=SELECTION_MODE_COLUMNS,
    )
    status = ieee118_load_status(failed)
    assert status["loads"] is False
    assert status["supports_4x4_extraction"] is False
    assert status["positive_evidence_modes"] == []

    empty_status = ieee118_load_status(pd.DataFrame(columns=SELECTION_MODE_COLUMNS))
    assert empty_status["loads"] is False
