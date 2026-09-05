"""Phase 1: degree 47/49 diagnostic-only sweep and the degree-ceiling interpretation."""

from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr


def test_diagnostic_cell_is_diagnostic_only(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(2.0))
    cell = psr.evaluate_degree_diagnostic_cell(
        tiny_context, alpha=1e-2, degree=47, seed=1, target_error=1e-2
    )
    assert cell["overshoot_diagnostic_only"] is True
    assert cell["eligible_for_best_config"] is False
    assert cell["degree"] == 47
    assert set(psr.DEGREE_DIAGNOSTIC_COLUMNS) == set(cell)


def test_diagnostic_cell_records_phase_and_gate(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(2.0))
    cell = psr.evaluate_degree_diagnostic_cell(
        tiny_context, alpha=1e-2, degree=47, seed=1, target_error=1e-2
    )
    assert cell["gate_run_status"] == "computed"
    assert np.isfinite(cell["gate_error_vs_ridge"])
    assert np.isfinite(cell["phase_objective"])
    assert np.isfinite(cell["dense_grid_max_abs_value"])
    assert cell["success_probability"] > 0


def test_diagnostic_cell_marks_missing_gate(tiny_context, fake_gate, monkeypatch) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(2.0, available=False))
    cell = psr.evaluate_degree_diagnostic_cell(
        tiny_context, alpha=1e-2, degree=47, seed=1, target_error=1e-2
    )
    assert cell["gate_run_status"] == "failed"
    assert cell["status"] == "gate_unavailable"


def test_degree_ceiling_label_phase_synthesis(tiny_context) -> None:
    # polynomial improves but gate worsens / synthesis degrades -> phase_synthesis_ceiling
    cell = {
        "boundedness_ok": True,
        "overshoot_flag": False,
        "polynomial_error_vs_exact_target": 0.10,
        "gate_error_vs_ridge": 1.30,
        "gate_error_vs_polynomial": 0.50,
    }
    label = psr._degree_ceiling_label(cell, baseline_best_error=0.17, baseline_poly_error=0.20)
    assert label == "phase_synthesis_ceiling"


def test_degree_ceiling_label_overshoot() -> None:
    cell = {
        "boundedness_ok": False,
        "overshoot_flag": True,
        "polynomial_error_vs_exact_target": 0.05,
        "gate_error_vs_ridge": 0.05,
        "gate_error_vs_polynomial": 0.01,
    }
    label = psr._degree_ceiling_label(cell, baseline_best_error=0.17, baseline_poly_error=0.20)
    assert label == "boundedness_or_overshoot_ceiling"


def test_degree_ceiling_label_helpful() -> None:
    cell = {
        "boundedness_ok": True,
        "overshoot_flag": False,
        "polynomial_error_vs_exact_target": 0.05,
        "gate_error_vs_ridge": 0.05,
        "gate_error_vs_polynomial": 0.001,
    }
    label = psr._degree_ceiling_label(cell, baseline_best_error=0.17, baseline_poly_error=0.20)
    assert label == "higher_degree_potentially_useful"


def test_degree_ceiling_summary(tiny_context, fake_gate, monkeypatch, baseline_rows) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(40.0))
    cells = [
        psr.evaluate_degree_diagnostic_cell(
            tiny_context, alpha=1e-2, degree=degree, seed=1, target_error=1e-2
        )
        for degree in (47, 49)
    ]
    summary = psr.build_degree_ceiling_summary(cells, baseline_rows)
    assert len(summary) == 1
    row = summary[0]
    assert set(psr.DEGREE_CEILING_COLUMNS) == set(row)
    assert row["ceiling_label"] in {
        "phase_synthesis_ceiling",
        "boundedness_or_overshoot_ceiling",
        "higher_degree_potentially_useful",
        "higher_degree_not_helpful",
    }
