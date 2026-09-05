"""Workstream 4 tests: cost-model arithmetic, evidence status, and Pareto logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.reviewer_blocking.resource_pareto import (
    FIXED_ERROR_COLUMNS,
    _best_class_at_threshold,
    _fixed_error_costs,
    _pareto_fronts,
    _resource_status_ledger,
    _write_summary,
    cost_model,
)


def _executed(signal_gates: int, p_post: float) -> dict:
    return {
        "signal_unitary_gate_count": signal_gates,
        "postselection_probability": p_post,
    }


def test_cost_model_arithmetic_matches_declared_formula():
    dimension, degree, shots = 8, 31, 100000
    executed = _executed(1000, 0.5)
    costs = cost_model(executed, dimension=dimension, degree=degree, shots=shots)
    c_load = 2 * dimension - 2
    c_readout = dimension
    total_per_attempt = c_load + degree * 1000 + c_readout
    assert costs["total_estimated_gates_per_attempt"] == total_per_attempt
    assert costs["c_total_gates"] == total_per_attempt / 0.5
    assert costs["expected_postselection_attempts"] == 1.0 / 0.5
    assert costs["modeled_total_shot_attempts"] == shots / 0.5


def test_cost_model_handles_zero_postselection_safely():
    costs = cost_model(_executed(500, 0.0), dimension=8, degree=31, shots=1000)
    assert np.isinf(costs["c_total_gates"])
    assert np.isinf(costs["expected_postselection_attempts"])
    assert np.isinf(costs["modeled_total_shot_attempts"])
    # Finite per-attempt gate count remains well defined.
    assert np.isfinite(costs["total_estimated_gates_per_attempt"])


def test_resource_status_ledger_separates_executed_and_modeled():
    ledger = _resource_status_ledger()
    executed = ledger[ledger["evidence_status"].str.startswith("executed")]["component"].tolist()
    modeled = ledger[ledger["evidence_status"].str.startswith("modeled")]["component"].tolist()
    assert "signal_unitary_gates" in executed
    assert "postselection_probability" in executed
    assert "residual_loader" in modeled
    assert "selected_output_readout" in modeled
    # C_total is explicitly flagged as mixed, never silently merged.
    ctotal = ledger[ledger["component"] == "c_total"]["evidence_status"].iloc[0]
    assert ctotal == "mixed_executed_and_modeled"


def _completed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"selector": "sensitivity_refined_mean", "selector_class": "output_aware",
             "k_budget": 12, "slot_budget": 3, "actual_nonzeros": 12, "slot_count": 3,
             "mean_heldout_normalized_error": 0.30, "executed_c_signal_gates": 3600,
             "total_estimated_gates_per_attempt": 111614, "c_total_gates": 160000.0,
             "expected_postselection_attempts": 1.4},
            {"selector": "sensitivity_refined_mean", "selector_class": "output_aware",
             "k_budget": 24, "slot_budget": 3, "actual_nonzeros": 24, "slot_count": 3,
             "mean_heldout_normalized_error": 0.27, "executed_c_signal_gates": 3900,
             "total_estimated_gates_per_attempt": 120000, "c_total_gates": 158000.0,
             "expected_postselection_attempts": 1.3},
            {"selector": "global_magnitude", "selector_class": "output_agnostic",
             "k_budget": 12, "slot_budget": 2, "actual_nonzeros": 12, "slot_count": 2,
             "mean_heldout_normalized_error": 0.79, "executed_c_signal_gates": 3300,
             "total_estimated_gates_per_attempt": 102000, "c_total_gates": 112000.0,
             "expected_postselection_attempts": 1.1},
            {"selector": "global_magnitude", "selector_class": "output_agnostic",
             "k_budget": 24, "slot_budget": 3, "actual_nonzeros": 24, "slot_count": 3,
             "mean_heldout_normalized_error": 0.28, "executed_c_signal_gates": 4400,
             "total_estimated_gates_per_attempt": 136000, "c_total_gates": 198000.0,
             "expected_postselection_attempts": 1.45},
        ]
    )


def test_fixed_error_costs_reports_min_and_reachability():
    completed = _completed_frame()
    fixed = _fixed_error_costs(completed, [0.5, 1.0])
    aware_05 = fixed[
        (fixed["error_threshold"] == 0.5) & (fixed["selector"] == "sensitivity_refined_mean")
    ].iloc[0]
    agnostic_05 = fixed[
        (fixed["error_threshold"] == 0.5) & (fixed["selector"] == "global_magnitude")
    ].iloc[0]
    assert aware_05["reached"]
    assert agnostic_05["reached"]
    # At the stringent threshold output-aware reaches it with a smaller support.
    assert aware_05["min_k"] < agnostic_05["min_k"]
    assert aware_05["min_c_total_gates"] < agnostic_05["min_c_total_gates"]


def test_fixed_error_costs_records_unreachable_threshold():
    completed = _completed_frame()
    fixed = _fixed_error_costs(completed, [0.1])
    assert (~fixed["reached"]).all()  # 0.1 is below every achieved error


def test_pareto_fronts_dominance_is_correct():
    completed = _completed_frame()
    fronts = _pareto_fronts(completed)
    axis = fronts[fronts["cost_axis"] == "c_total"].sort_values("cost")
    nd = axis[axis["nondominated"]]
    # A point is nondominated iff no other point has both <= error and <= cost.
    for _, row in axis.iterrows():
        dominated = (
            (axis["error"] <= row["error"])
            & (axis["cost"] <= row["cost"])
            & ((axis["error"] < row["error"]) | (axis["cost"] < row["cost"]))
        ).any()
        assert bool(row["nondominated"]) == (not dominated)
    assert len(nd) >= 1


def _write_reporting_summary(tmp_path, completed, thresholds):
    frame = completed.assign(status="completed") if not completed.empty else pd.DataFrame()
    fixed = _fixed_error_costs(completed, thresholds)
    _write_summary(
        tmp_path,
        {"degree": 31, "slot_budgets": [2, 3, 4]},
        frame,
        completed,
        fixed,
        thresholds,
    )
    return fixed, (tmp_path / "resource_pareto_summary.md").read_text()


def test_summary_all_selectors_infeasible_retains_nullable_threshold_row(tmp_path):
    completed = _completed_frame().assign(mean_heldout_normalized_error=2.0)
    fixed, report = _write_reporting_summary(tmp_path, completed, [0.5])
    assert list(fixed.columns) == list(FIXED_ERROR_COLUMNS)
    assert (fixed["reached"] == False).all()  # noqa: E712 - explicit schema assertion
    assert fixed["min_c_total_gates"].isna().all()
    assert "| 0.5 | false | unavailable |  | false | unavailable |  | both infeasible |" in report


def test_summary_one_selector_feasible_reports_only_that_selector(tmp_path):
    completed = _completed_frame().assign(mean_heldout_normalized_error=2.0)
    completed.loc[completed.index[0], "mean_heldout_normalized_error"] = 0.4
    fixed, report = _write_reporting_summary(tmp_path, completed, [0.5])
    aware = _best_class_at_threshold(fixed, 0.5, "output_aware")
    agnostic = _best_class_at_threshold(fixed, 0.5, "output_agnostic")
    assert aware["feasible"] and aware["selector"] == "sensitivity_refined_mean"
    assert not agnostic["feasible"] and pd.isna(agnostic["selector"])
    assert "aware only" in report


def test_summary_multiple_selectors_tied_are_reported_deterministically(tmp_path):
    completed = _completed_frame().iloc[[0, 2]].copy()
    completed["mean_heldout_normalized_error"] = 0.4
    completed["c_total_gates"] = 100000.0
    completed["selector_class"] = "output_aware"
    completed["selector"] = ["z_selector", "a_selector"]
    fixed, _ = _write_reporting_summary(tmp_path, completed, [0.5])
    best = _best_class_at_threshold(fixed, 0.5, "output_aware")
    assert best == {
        "feasible": True,
        "selector": "a_selector, z_selector",
        "min_c_total_gates": 100000.0,
    }


def test_summary_mixed_finite_and_unavailable_costs_uses_finite_only(tmp_path):
    completed = _completed_frame().iloc[[0, 1]].copy()
    completed["mean_heldout_normalized_error"] = 0.4
    completed.loc[completed.index[0], "c_total_gates"] = np.nan
    completed.loc[completed.index[1], "c_total_gates"] = 158000.0
    fixed, _ = _write_reporting_summary(tmp_path, completed, [0.5])
    best = _best_class_at_threshold(fixed, 0.5, "output_aware")
    assert best["feasible"]
    assert best["min_c_total_gates"] == 158000.0


def test_summary_all_thresholds_infeasible_preserves_every_threshold(tmp_path):
    completed = _completed_frame().assign(mean_heldout_normalized_error=2.0)
    thresholds = [0.5, 0.6, 0.75, 1.0, 1.5]
    fixed, report = _write_reporting_summary(tmp_path, completed, thresholds)
    assert set(fixed["error_threshold"]) == set(thresholds)
    assert report.count("both infeasible") == len(thresholds)
    assert all(f"| {threshold:g} |" in report for threshold in thresholds)


def test_empty_pareto_front_retains_thresholds_and_writer_does_not_crash(tmp_path):
    empty = pd.DataFrame()
    assert _pareto_fronts(empty).empty
    thresholds = [0.5, 1.0]
    fixed, report = _write_reporting_summary(tmp_path, empty, thresholds)
    assert len(fixed) == len(thresholds)
    assert fixed["selector"].isna().all()
    assert fixed["min_c_total_gates"].isna().all()
    assert report.count("both infeasible") == len(thresholds)
