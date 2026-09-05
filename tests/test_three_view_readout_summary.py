"""Phase 4: the three-view readout summary (common reference, refined QSP, relaxed alpha)."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def _fit_rows(baseline_rows, *, error: float = 0.03):
    rows = []
    for b in baseline_rows:
        for alpha in (b["current_best_alpha"], 1e-4):
            rows.append(
                {
                    "case": b["case"],
                    "subproblem_id": b["subproblem_id"],
                    "subproblem_type": b["subproblem_type"],
                    "alpha": float(alpha),
                    "degree": int(b["current_best_degree"]),
                    "fit_mode": "weighted_singular_support_fit",
                    "weight_mode": "residual_energy_weight",
                    "support_weighted_error": 0.02,
                    "gate_error_vs_polynomial": 0.01,
                    "gate_error_vs_ridge": error,
                    "boundedness_ok": True,
                    "eligible_for_best_config": True,
                }
            )
    return rows


def _contexts(baseline_rows):
    pairs = [(b["case"], b["subproblem_type"]) for b in baseline_rows]
    from pathlib import Path

    return psr._build_contexts(pairs, Path("outputs"), seed=123)






def test_refined_view_adopts_weighted_fit(baseline_rows) -> None:
    contexts = _contexts(baseline_rows)
    rows = psr.build_three_view_summary(
        baseline_rows=baseline_rows,
        matched_view_rows=[],
        validated_view_rows=[],
        fit_comparison_rows=_fit_rows(baseline_rows, error=0.03),
        contexts=contexts,
        target_error=1e-2,
    )
    refined = [r for r in rows if r["view"] == "matched_alpha_refined_qsp"]
    assert all(r["fit_mode"] == "weighted_singular_support_fit" for r in refined)
    assert all(r["status"] == "pass" for r in refined)


def test_relaxed_view_reports_ridge_alpha_shift(baseline_rows) -> None:
    contexts = _contexts(baseline_rows)
    rows = psr.build_three_view_summary(
        baseline_rows=baseline_rows,
        matched_view_rows=[],
        validated_view_rows=[],
        fit_comparison_rows=_fit_rows(baseline_rows),
        contexts=contexts,
        target_error=1e-2,
    )
    relaxed = [r for r in rows if r["view"] == "per_subproblem_relaxed_alpha_qsvt_validated"]
    # the ridge alpha-shift column is present and finite for every relaxed row
    assert all(np.isfinite(r["ridge_alpha_shift_vs_common_alpha"]) for r in relaxed)


def test_three_view_markdown_no_superiority_claim(baseline_rows) -> None:
    contexts = _contexts(baseline_rows)
    rows = psr.build_three_view_summary(
        baseline_rows=baseline_rows,
        matched_view_rows=[],
        validated_view_rows=[],
        fit_comparison_rows=_fit_rows(baseline_rows),
        contexts=contexts,
        target_error=1e-2,
    )
    md = psr.three_view_summary_markdown(rows)
    lowered = md.lower()
    assert "outperform" not in lowered or "does not" in lowered
    assert "quantum speedup" not in lowered or "no" in lowered
    assert "same alpha" in lowered
