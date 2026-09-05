"""Phase 6: the phase-synthesis failure taxonomy."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def test_high_error_rows_get_a_failure_label(baseline_rows) -> None:
    rows = psr.build_failure_taxonomy(
        baseline_rows=baseline_rows, diagnostic_rows=[], comparison_rows=[]
    )
    assert rows
    assert all(set(psr.FAILURE_TAXONOMY_COLUMNS) == set(r) for r in rows)
    labels = {r["subproblem_type"]: r["failure_label"] for r in rows}
    # metadata_mapped has a degraded gate-vs-poly -> phase synthesis layer
    assert labels["metadata_mapped"] == "phase_synthesis_degraded"


def test_labels_correspond_to_metrics(baseline_rows) -> None:
    rows = psr.build_failure_taxonomy(
        baseline_rows=baseline_rows, diagnostic_rows=[], comparison_rows=[]
    )
    for row in rows:
        assert np.isfinite(row["evidence_value"])
        if row["failure_label"] == "phase_synthesis_degraded":
            assert row["failure_layer"] == "phase_synthesis"
            assert row["evidence_metric"] == "polynomial_vs_gate_error"


def test_no_readout_failure_label_without_shot_dominance(baseline_rows) -> None:
    rows = psr.build_failure_taxonomy(
        baseline_rows=baseline_rows, diagnostic_rows=[], comparison_rows=[]
    )
    assert all("readout" not in r["failure_label"] for r in rows)
    assert all("readout protocol is not the failure source" in r["notes"] for r in rows)


def test_overshoot_and_unbounded_fit_recorded(baseline_rows) -> None:
    diagnostic = [
        {
            "case": "ieee14",
            "subproblem_id": "x",
            "subproblem_type": "metadata_mapped",
            "alpha": 1e-5,
            "degree": 47,
            "overshoot_flag": True,
            "gate_run_status": "failed",
            "dense_grid_max_abs_value": 1.8,
        }
    ]
    comparison = [
        {
            "case": "ieee57",
            "subproblem_id": "y",
            "subproblem_type": "high_leverage",
            "alpha": 1e-2,
            "degree": 41,
            "boundedness_ok": False,
            "fit_mode": "residual_aware_fit",
            "dense_grid_max_error": 0.9,
        }
    ]
    rows = psr.build_failure_taxonomy(
        baseline_rows=baseline_rows, diagnostic_rows=diagnostic, comparison_rows=comparison
    )
    labels = {r["failure_label"] for r in rows}
    assert "overshoot_on_dense_grid" in labels
    assert "support_fit_good_dense_bad" in labels


def test_taxonomy_markdown_counts(baseline_rows) -> None:
    rows = psr.build_failure_taxonomy(
        baseline_rows=baseline_rows, diagnostic_rows=[], comparison_rows=[]
    )
    md = psr.failure_taxonomy_markdown(rows)
    assert "Failure label counts" in md
    assert "shot reconstruction dominates" in md
