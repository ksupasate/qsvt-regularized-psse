from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.readout_limitation_formalization import (
    OBSERVABLE_COLUMNS,
    build_readout_limitation_formalization,
)


def _write_inputs(input_root: Path) -> None:
    readout = input_root / "qsvt_cross_case_gate_observable_readout"
    readout.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "case": ["ieee14", "ieee14", "ieee57", "ieee57"],
            "observable_name": [
                "bus_angle_update",
                "top_k_update_identification",
                "bus_angle_update",
                "top_k_update_identification",
            ],
            "alpha": [1e-4, 1e-4, 1e-4, 1e-4],
            "degree": [15, 15, 15, 15],
            "absolute_error_gate_vs_ridge": [1e-4, 0.0, 2e-4, 0.0],
            "relative_error_gate_vs_ridge": [1e-3, 0.0, 2e-3, 0.0],
            "top_k_match_if_applicable": [None, 1.0, None, 0.0],
            "requires_norm_recovery": [True, False, True, False],
            "requires_full_vector_readout": [False, False, False, False],
            "readout_cost_proxy": [10000.0, 10000.0, 10000.0, 10000.0],
            "practical_status": [
                "practical_with_norm_recovery",
                "practical_without_norm_recovery",
                "practical_with_norm_recovery",
                "practical_without_norm_recovery",
            ],
        }
    ).to_csv(readout / "cross_case_gate_observable_accuracy_cost.csv", index=False)
    pd.DataFrame({"observable_name": ["bus_angle_update", "top_k_update_identification"]}).to_csv(
        readout / "cross_case_gate_observable_values.csv", index=False
    )


def test_full_vector_readout_is_assumption_only(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    run = build_readout_limitation_formalization(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "readout")}
    )
    frame = pd.read_csv(run["artifacts"]["observable_vs_full_vector_readout"])
    assert list(frame.columns) == OBSERVABLE_COLUMNS
    full = frame[frame["observable_name"] == "full_update_vector"]
    assert not full.empty
    assert (full["requires_full_vector_readout"] == "yes").all()
    assert (full["supported_by_gate_values"] == "no").all()
    assert run["full_vector_assumption_recorded"] is True
    # The gate-computed observables avoid full-vector reconstruction.
    gate = frame[frame["observable_name"] == "bus_angle_update"]
    assert (gate["requires_full_vector_readout"] == "no").all()


def test_topk_exactness_conditioned_on_margin(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    run = build_readout_limitation_formalization(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "readout")}
    )
    frame = pd.read_csv(run["artifacts"]["observable_vs_full_vector_readout"])
    topk = frame[frame["observable_name"] == "top_k_update_identification"]
    assert (topk["requires_topk_margin"] == "yes").all()
    margin = pd.read_csv(run["artifacts"]["topk_margin_sensitivity"])
    assert not margin.empty
    assert margin["margin_condition"].astype(str).str.contains("separation").all()
    # The non-matching top-k case (match=0) is not reported as exact.
    assert set(margin["exact_when_well_separated"]) == {"yes", "no"}


def test_summary_keeps_full_vector_unsolved(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    run = build_readout_limitation_formalization(
        {"input_root": str(tmp_path), "output_dir": str(tmp_path / "readout")}
    )
    summary = Path(run["artifacts"]["readout_limitation_summary"]).read_text(encoding="utf-8")
    assert "Full-vector readout is **not** claimed solved" in summary
    assert "Top-k exactness is conditioned on margin" in summary
