from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_gate_observable_readout import _successful_gate_rows
from robust_qsvt_se.qsvt.ieee118_gate_observable_readout import (
    DEFAULT_OBSERVABLES,
    SUMMARY_COLUMNS,
    evaluate_cross_case_observable_readout,
    ieee118_observable_interpretation,
)


def test_default_observables_exclude_full_vector_reconstruction() -> None:
    assert "full_state_vector_reconstruction" not in DEFAULT_OBSERVABLES
    # The summary still tracks whether each observable would require full-vector readout.
    assert "requires_full_vector_readout" in SUMMARY_COLUMNS


def test_full_vector_reconstruction_is_filtered_out() -> None:
    # Even if a caller requests full-vector reconstruction, the readout drops it (no rows, no
    # systems built because the successful frame is empty).
    rows = evaluate_cross_case_observable_readout(
        successful=pd.DataFrame(),
        model="ac_linearized",
        case_source="pypower",
        observables=["full_state_vector_reconstruction", *DEFAULT_OBSERVABLES],
        shots=[1000],
    )
    assert rows == []


def test_successful_gate_rows_filters_completed_only() -> None:
    gate = pd.DataFrame(
        [
            {"case": "ieee118", "gate_status": "completed"},
            {"case": "ieee118", "gate_status": "not_run"},
            {"case": "ieee118", "gate_status": "failed"},
        ]
    )
    successful = _successful_gate_rows(gate)
    assert len(successful) == 1
    assert successful.iloc[0]["gate_status"] == "completed"


def test_interpretation_handles_empty_frame() -> None:
    text = ieee118_observable_interpretation(pd.DataFrame(columns=SUMMARY_COLUMNS))
    assert "IEEE118 Gate-Level Observable Readout" in text
    assert "No IEEE118 gate-validated configuration" in text
