"""Phase 0 and shared primitives: baseline freeze, the QSVT phase response, the custom-
coefficient gate runner, and the unified config-gate routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pennylane as qml
from numpy.polynomial import Polynomial

from robust_qsvt_se.paper import phase_synthesis_refinement as psr
from robust_qsvt_se.paper.full_vector_readout import _bounded_scale_and_polynomial

_TWO_VIEW = (
    "case,subproblem_id,subproblem_type,matched_alpha,matched_degree,matched_alpha_gate_error,"
    "validated_alpha,validated_degree,validated_gate_error,improvement_factor,"
    "matched_dominant_error_source,validated_dominant_error_source,final_status,"
    "manuscript_interpretation,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,3.0189,1e-05,45,0.4718,6.40,"
    "exact_target_vs_polynomial,exact_target_vs_polynomial,polynomial_limited_after_sweep,x,y\n"
    "ieee57,high_leverage_00,high_leverage,0.0001,35,0.323,0.01,41,0.1717,1.88,"
    "exact_target_vs_polynomial,shot_reconstruction_vs_ridge,polynomial_limited_after_sweep,x,y\n"
)

_BEST = (
    "case,subproblem_id,subproblem_type,matched_alpha,matched_degree,matched_alpha_error,"
    "best_alpha,best_degree,best_gate_error_vs_ridge,best_polynomial_error_vs_exact_target,"
    "best_gate_vs_polynomial_error,best_shot_reconstruction_error_vs_ridge,best_residual_ratio,"
    "best_sign_reliable_fraction,best_top_k_match,improvement_factor,selection_rule,"
    "selected_for_main_paper,status,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,3.0189,1e-05,45,0.4718,2.56,0.705,0.47,"
    "0.53,1.0,1.0,6.40,rule,reported_as_limitation,qsvt_polynomial_limited,n\n"
    "ieee57,high_leverage_00,high_leverage,0.0001,35,0.323,0.01,41,0.1717,0.171,0.0013,0.17,0.22,"
    "1.0,0.5,1.88,rule,reported_as_limitation,qsvt_polynomial_limited,n\n"
)


def _seed_readout(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "readout_two_view_summary.csv").write_text(_TWO_VIEW, encoding="utf-8")
    (out / "readout_selected_best_configs.csv").write_text(_BEST, encoding="utf-8")
    (out / "readout_config_diagnostic_classification.csv").write_text(
        "case,subproblem_type,dominant_error_source\n"
        "ieee14,metadata_mapped,exact_target_vs_polynomial\n",
        encoding="utf-8",
    )


def test_freeze_baseline_derives_from_existing(tmp_path: Path) -> None:
    _seed_readout(tmp_path)
    rows = {(r["case"], r["subproblem_type"]): r for r in psr.freeze_baseline(tmp_path)}
    assert set(rows) == {("ieee14", "metadata_mapped"), ("ieee57", "high_leverage")}
    meta = rows[("ieee14", "metadata_mapped")]
    assert meta["matched_alpha_error"] == 3.0189
    assert meta["current_best_error"] == 0.4718
    assert meta["phase_synthesis_status"] == "phase_synthesis_degraded_at_best_degree"
    # ieee57 has a clean gate-vs-poly at its best degree
    assert rows[("ieee57", "high_leverage")]["phase_synthesis_status"] == (
        "phase_synthesis_clean_at_best_degree"
    )


def test_freeze_baseline_does_not_write(tmp_path: Path) -> None:
    _seed_readout(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    psr.freeze_baseline(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_baseline_summary_markdown_lists_high_error(tmp_path: Path) -> None:
    _seed_readout(tmp_path)
    md = psr.baseline_summary_markdown(psr.freeze_baseline(tmp_path))
    assert "Frozen co-design baseline" in md
    assert "ieee14/metadata_mapped" in md


def test_metadata_mapped_limitation_note_preserves_polynomial_limited_status(
    tmp_path: Path,
) -> None:
    _seed_readout(tmp_path)
    rows = [
        {
            "case": "ieee14",
            "subproblem_type": "metadata_mapped",
            "view": "matched_alpha_refined_qsp",
            "gate_error_vs_matching_ridge_alpha": 0.17149,
            "status": "qsvt_polynomial_limited",
        }
    ]
    md = psr.metadata_mapped_limitation_note_markdown(rows)
    assert "remains polynomial-limited" in md
    assert "finite-degree QSP limitation" in md
    assert "not a readout failure and not a full QSVT failure" in md
    assert "qsvt_polynomial_limited" in md


def test_qsvt_phase_response_reproduces_polynomial() -> None:
    coeffs = np.zeros(8)
    coeffs[1], coeffs[3], coeffs[5] = 0.6, -0.2, 0.05
    phases = np.asarray(qml.poly_to_angles(coeffs, "QSVT", angle_solver="iterative"), dtype=float)
    grid = np.array([-0.8, -0.3, 0.1, 0.45, 0.77])
    response = psr.qsvt_phase_response(grid, phases)
    assert np.allclose(response, Polynomial(coeffs)(grid), atol=1e-6)


def test_gate_update_from_coefficients_runs_real_circuit(tiny_context) -> None:
    state = psr.qsvt_target_readout(tiny_context.H, tiny_context.r, alpha=0.1, degree=5)
    sv_A = np.asarray(state.singular_values) / max(float(state.singular_values[0]), 1e-300)
    domain_min = psr._domain_min(sv_A)
    scale_C, coeffs, ok = _bounded_scale_and_polynomial(
        sv_A, alpha_norm=float(state.alpha_norm), degree=5
    )
    assert ok
    _ = domain_min
    gate = psr.gate_update_from_coefficients(
        tiny_context.H, tiny_context.r, coeffs, scale_C, seed=1
    )
    assert gate["available"]
    assert np.all(np.isfinite(gate["gate_update"]))
    assert gate["gate_update"].shape == tiny_context.r.shape


def test_gate_update_from_coefficients_reports_failure() -> None:
    H = np.eye(4)
    r = np.ones(4)
    gate = psr.gate_update_from_coefficients(H, r, np.array([np.nan, np.nan]), 1.0, seed=1)
    assert gate["available"] is False
    assert "reason" in gate


def test_realize_config_gate_uniform_uses_standard_solver(
    tiny_context, fake_gate, monkeypatch
) -> None:
    monkeypatch.setattr(psr, "gate_level_subproblem_readout", fake_gate(1.0))
    cache: dict = {}
    result = psr.realize_config_gate(tiny_context, alpha=0.1, degree=5, seed=1, cache=cache)
    assert result["available"]
    assert np.isfinite(result["gate_error_vs_ridge"])
    # cache hit returns the same object
    assert psr.realize_config_gate(tiny_context, alpha=0.1, degree=5, seed=1, cache=cache) is result
