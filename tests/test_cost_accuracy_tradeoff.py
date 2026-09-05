"""Phase 7: cost-accuracy and degree-cost trade-off tables."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def _three_view(case="ieee57", stype="high_leverage"):
    rows = []
    for view, degree in (
        ("matched_alpha_common_reference", 35),
        ("matched_alpha_refined_qsp", 41),
        ("per_subproblem_relaxed_alpha_qsvt_validated", 45),
    ):
        rows.append(
            {
                "case": case,
                "subproblem_id": "s",
                "subproblem_type": stype,
                "view": view,
                "alpha": 1e-2,
                "degree": degree,
                "phase_solver_variant": "default",
                "fit_mode": "global_uniform_fit",
                "weight_mode": "uniform_support_weight",
                "gate_error_vs_matching_ridge_alpha": 0.1,
                "ridge_alpha_shift_vs_common_alpha": 0.0,
                "exact_target_vs_polynomial_error": 0.1,
                "polynomial_vs_gate_error": 0.01,
                "shot_readout_error": float("nan"),
                "success_probability": 0.5,
                "postselection_overhead": 2.0,
                "estimated_total_readout_shots": 10000,
                "status": "pass",
                "interpretation": "x",
                "source_artifact": "y",
                "notes": "z",
            }
        )
    return rows


def test_cost_rows_use_real_gate_query_count(tiny_context, fake_realize, monkeypatch) -> None:
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    contexts = {("ieee57", "high_leverage"): tiny_context}
    cost, degree_cost = psr.build_cost_accuracy_rows(_three_view(), contexts=contexts, seed=1)
    assert len(cost) == 3
    assert all(set(psr.COST_ACCURACY_COLUMNS) == set(r) for r in cost)
    assert all(set(psr.DEGREE_COST_COLUMNS) == set(r) for r in degree_cost)
    # query count comes from the (synthetic) gate and increases with degree
    by_degree = {r["degree"]: r["query_count"] for r in cost}
    assert by_degree[45] > by_degree[35]


def test_postselection_overhead_included(tiny_context, fake_realize, monkeypatch) -> None:
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    contexts = {("ieee57", "high_leverage"): tiny_context}
    cost, _ = psr.build_cost_accuracy_rows(_three_view(), contexts=contexts, seed=1)
    assert all(
        np.isfinite(r["postselection_overhead"]) and r["postselection_overhead"] > 0 for r in cost
    )
    assert all(np.isfinite(r["accuracy_cost_score"]) for r in cost)


def test_no_speedup_claim_in_notes(tiny_context, fake_realize, monkeypatch) -> None:
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    contexts = {("ieee57", "high_leverage"): tiny_context}
    cost, degree_cost = psr.build_cost_accuracy_rows(_three_view(), contexts=contexts, seed=1)
    for row in cost + degree_cost:
        note = row["notes"].lower()
        # speedup may only appear when explicitly disclaimed ("no speedup is claimed")
        assert "speedup" not in note or "no speedup" in note


def test_cost_figure_rows_nonnegative(tiny_context, fake_realize, monkeypatch) -> None:
    monkeypatch.setattr(psr, "realize_config_gate", fake_realize())
    contexts = {("ieee57", "high_leverage"): tiny_context}
    cost, _ = psr.build_cost_accuracy_rows(_three_view(), contexts=contexts, seed=1)
    figs = psr.build_cost_figure_rows(cost)
    assert figs
    assert all(r["gate_error_vs_matching_ridge_alpha"] >= 0 for r in figs)
