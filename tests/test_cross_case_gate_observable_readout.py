from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.cross_case_gate_observable_readout import (
    SUMMARY_COLUMNS,
    _successful_gate_rows,
    evaluate_cross_case_observable_readout,
    write_cross_case_observable_outputs,
)


def test_successful_gate_rows_keeps_only_completed() -> None:
    frame = pd.DataFrame(
        [
            {"case": "ieee30", "gate_status": "completed"},
            {"case": "ieee57", "gate_status": "not_run"},
            {"case": "ieee14", "gate_status": "failed"},
        ]
    )
    kept = _successful_gate_rows(frame)
    assert list(kept["case"]) == ["ieee30"]


def test_full_vector_reconstruction_is_excluded() -> None:
    # No gate rows -> no readout; but the requested-observable filter must drop full-vector.
    rows = evaluate_cross_case_observable_readout(
        successful=pd.DataFrame(),
        model="ac_linearized",
        case_source="pypower",
        observables=["top_k_update_identification", "full_state_vector_reconstruction"],
        shots=[1000],
    )
    assert rows == []


def test_outputs_written_and_columns_present(tmp_path) -> None:
    rows = [
        {
            "case": "ieee30",
            "observable_name": "top_k_update_identification",
            "physical_meaning": "top-k update support",
            "subproblem_id": "high_leverage_00",
            "selection_mode": "high_leverage",
            "alpha": 1.0e-3,
            "degree": 35,
            "target_family": "residual_aware",
            "ridge_value": 1.0,
            "polynomial_value": 1.0,
            "gate_value": 1.0,
            "shot_estimated_value_if_available": 1.0,
            "absolute_error_gate_vs_ridge": 0.0,
            "relative_error_gate_vs_ridge": 0.0,
            "top_k_match_if_applicable": 1.0,
            "requires_norm_recovery": False,
            "requires_signed_overlap": False,
            "requires_full_vector_readout": False,
            "readout_protocol": "topk",
            "readout_cost_proxy": 10000.0,
            "practical_status": "practical_without_norm_recovery",
            "claim_allowed": "observable-first QSVT readout",
            "claim_disallowed": "no quantum speedup",
        }
    ]
    artifacts = write_cross_case_observable_outputs(tmp_path, {"input": "x"}, rows)
    for name in [
        "manifest",
        "cross_case_gate_observable_values",
        "cross_case_gate_observable_accuracy_cost",
        "cross_case_gate_observable_interpretation",
    ]:
        assert artifacts[name].is_file()
    values = pd.read_csv(artifacts["cross_case_gate_observable_values"])
    assert set(SUMMARY_COLUMNS).issubset(values.columns)
    # The requires_full_vector_readout column exists so full-vector observables stay flagged.
    assert "requires_full_vector_readout" in values.columns
    assert not bool(np.any(values["requires_full_vector_readout"].to_numpy()))
