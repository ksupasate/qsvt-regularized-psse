"""Phase 5: the selection-leakage audit confirms no true-state / oracle tuning."""

from __future__ import annotations

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def test_no_stage_uses_true_state(baseline_rows) -> None:
    rows = psr.build_selection_leakage_audit(baseline_rows)
    assert rows
    assert all(set(psr.SELECTION_LEAKAGE_COLUMNS) == set(r) for r in rows)
    assert all(r["uses_true_state"] is False for r in rows)
    assert all(r["uses_state_rmse"] is False for r in rows)


def test_every_allowed_stage_documented(baseline_rows) -> None:
    rows = psr.build_selection_leakage_audit(baseline_rows)
    # no row is simultaneously true-state-using and allowed for selection
    assert not any(
        (r["uses_true_state"] or r["uses_state_rmse"]) and r["allowed_for_selection"] for r in rows
    )
    stages = {r["selection_stage"] for r in rows}
    assert "weighted_singular_support_fit" in stages
    assert "three_view_best" in stages


def test_audit_notes_ridge_is_reference_not_superiority(baseline_rows) -> None:
    rows = psr.build_selection_leakage_audit(baseline_rows)
    assert all("classical reference" in r["notes"] for r in rows)


def test_leakage_markdown_reports_clean(baseline_rows) -> None:
    md = psr.selection_leakage_markdown(psr.build_selection_leakage_audit(baseline_rows))
    assert "Any stage uses true state: False" in md
    assert "Forbidden metrics" in md
