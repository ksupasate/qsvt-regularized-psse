"""Phases 1-5: diagnostic classification, matched-alpha view, targeted alpha/degree sweep,
best-config selection, and the per-subproblem validated view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.paper import readout_alpha_degree_codesign as codesign
from robust_qsvt_se.paper.full_vector_readout import ridge_update
from robust_qsvt_se.paper.readout_alpha_degree_codesign import (
    BEST_CONFIG_COLUMNS,
    CLASSIFICATION_COLUMNS,
    MATCHED_ALPHA_VIEW_COLUMNS,
    SWEEP_COLUMNS,
    VALIDATED_VIEW_COLUMNS,
    build_matched_alpha_view,
    build_validated_view,
    classify_readout_configs,
    evaluate_sweep_cell,
    run_alpha_degree_sweep,
    select_best_configs,
    select_targeted_configs,
)

_GATE_SUMMARY = (
    "case,subproblem_id,subproblem_type,alpha,degree,state_dimension,"
    "vector_relative_l2_error_vs_ridge,vector_relative_l2_error_vs_exact_target,"
    "max_coordinate_abs_error_vs_ridge,sign_accuracy,residual_ratio_after_reconstruction,"
    "gate_available,status,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,4,3.0189,3.0189,0.022,0.5,3.87,True,gr,n\n"
    "ieee57,high_leverage_00,high_leverage,0.0001,35,4,0.323,0.323,0.0035,0.75,0.41,True,gr,n\n"
    "ieee30,high_leverage_00,high_leverage,0.0001,35,4,1e-9,1e-9,8e-12,1.0,5e-10,True,gr,n\n"
)

_DECOMPOSITION = (
    "case,subproblem_id,subproblem_type,alpha,degree,ridge_vs_exact_target_error,"
    "exact_target_vs_polynomial_error,polynomial_vs_gate_error,gate_vs_statevector_readout_error,"
    "statevector_vs_shot_reconstruction_error,shot_reconstruction_vs_ridge_error,"
    "dominant_error_source,status,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,0.0,3.0178,4.3e-4,0.0,2.3e-3,2.3e-3,"
    "exact_target_vs_polynomial,decomposed,n\n"
    "ieee57,high_leverage_00,high_leverage,0.0001,35,0.0,0.3230,3.5e-5,0.0,2.2e-3,2.2e-3,"
    "exact_target_vs_polynomial,decomposed,n\n"
    "ieee30,high_leverage_00,high_leverage,0.0001,35,3e-17,3e-11,6e-10,0.0,2.4e-3,2.4e-3,"
    "statevector_vs_shot_reconstruction,decomposed,n\n"
)


def _seed_matched(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_level_full_vector_summary.csv").write_text(_GATE_SUMMARY, encoding="utf-8")
    (out / "readout_error_decomposition.csv").write_text(_DECOMPOSITION, encoding="utf-8")


def _fake_gate(error_scale: float, *, available: bool = True):
    """Deterministic gate whose vector error vs Ridge equals error_scale/degree."""

    def gate(H, r, *, alpha, degree, seed, shots=1000, case="custom"):
        if not available:
            return {"available": False, "reason": "synthetic synthesis failure"}
        ridge = ridge_update(H, r, alpha=float(alpha))
        err = float(error_scale) / float(degree)
        pert = np.arange(1, ridge.size + 1, dtype=np.float64)
        pert[0] *= -1.0
        pert = pert / np.linalg.norm(pert)
        gate_update = ridge + err * float(np.linalg.norm(ridge)) * pert
        gate_state = gate_update / float(np.linalg.norm(gate_update))
        return {
            "available": True,
            "gate_update": gate_update,
            "gate_state": gate_state,
            "statevector": np.concatenate([gate_state, np.zeros_like(gate_state)]),
            "ridge_update": ridge,
            "success_probability": 0.5,
            "synthesized_degree": int(degree),
            "state_error_vs_ridge": err,
        }

    return gate


# --------------------------------------------------------------------------- Phase 1
def test_classification_assigns_pass_warning_fail(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = {(r["case"], r["subproblem_type"]): r for r in classify_readout_configs(tmp_path)}
    assert rows[("ieee14", "metadata_mapped")]["diagnostic_status"] == "fail_diagnostic"
    assert rows[("ieee57", "high_leverage")]["diagnostic_status"] == "fail_diagnostic"
    assert rows[("ieee30", "high_leverage")]["diagnostic_status"] == "pass"


def test_classification_includes_mandatory_metadata_mapped(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = classify_readout_configs(tmp_path)
    assert any(r["case"] == "ieee14" and r["subproblem_type"] == "metadata_mapped" for r in rows)


def test_classification_preserves_dominant_error_source(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = {(r["case"], r["subproblem_type"]): r for r in classify_readout_configs(tmp_path)}
    assert (
        rows[("ieee14", "metadata_mapped")]["dominant_error_source"] == "exact_target_vs_polynomial"
    )
    assert rows[("ieee30", "high_leverage")]["dominant_error_source"] == (
        "statevector_vs_shot_reconstruction"
    )
    assert (
        rows[("ieee14", "metadata_mapped")]["recommended_action"] == "alpha_degree_codesign_sweep"
    )


def test_classification_columns_complete(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = classify_readout_configs(tmp_path)
    assert set(rows[0].keys()) == set(CLASSIFICATION_COLUMNS)


# --------------------------------------------------------------------------- Phase 2
def test_matched_view_keeps_all_rows_and_alpha(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = build_matched_alpha_view(tmp_path)
    assert len(rows) == 3
    assert all(r["view"] == codesign.MATCHED_VIEW for r in rows)
    assert all(abs(r["alpha"] - codesign.MATCHED_ALPHA) < 1e-15 for r in rows)
    assert set(rows[0].keys()) == set(MATCHED_ALPHA_VIEW_COLUMNS)


def test_matched_view_labels_high_error_as_diagnostic(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    rows = {(r["case"], r["subproblem_type"]): r for r in build_matched_alpha_view(tmp_path)}
    high = rows[("ieee14", "metadata_mapped")]
    assert high["interpretation"] == "finite_degree_limited_common_alpha"
    assert "not readout protocol failure" in high["notes"]
    assert rows[("ieee30", "high_leverage")]["interpretation"] == "matched_alpha_pass"


def test_matched_view_does_not_overwrite_source(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    before = (tmp_path / "gate_level_full_vector_summary.csv").read_text(encoding="utf-8")
    build_matched_alpha_view(tmp_path)
    classify_readout_configs(tmp_path)
    assert (tmp_path / "gate_level_full_vector_summary.csv").read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- Phase 3
def _entry() -> dict:
    return {
        "case": "ieee14",
        "subproblem_id": "metadata_mapped_02",
        "subproblem_type": "metadata_mapped",
    }


def _tiny_problem() -> tuple[np.ndarray, np.ndarray]:
    H = np.array(
        [
            [1.2, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.3, 1.1, 0.2],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    r = np.array([0.5, -0.3, 0.4, -0.2], dtype=np.float64)
    return H, r


def test_sweep_cell_writes_all_columns_and_tracks_polynomial(monkeypatch) -> None:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate(2.0))
    H, r = _tiny_problem()
    row = evaluate_sweep_cell(
        _entry(), H, r, alpha=1e-4, degree=35, seed=7, shots=5000, trials=2, target_error=1e-2
    )
    assert set(row.keys()) == set(SWEEP_COLUMNS)
    assert row["eligible_for_best_config"] == "yes"
    assert row["gate_run_status"] == "computed"
    assert abs(row["gate_error_vs_ridge"] - 2.0 / 35.0) < 1e-6
    assert row["estimated_total_shots"] > 0


def test_sweep_cell_missing_gate_is_marked_not_inferred(monkeypatch) -> None:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate(1.0, available=False))
    H, r = _tiny_problem()
    row = evaluate_sweep_cell(
        _entry(), H, r, alpha=1e-4, degree=35, seed=7, shots=5000, trials=2, target_error=1e-2
    )
    assert row["status"] == "missing_gate_config"
    assert row["gate_run_status"] == "failed"
    assert row["eligible_for_best_config"] == "no"
    assert not np.isfinite(row["gate_error_vs_ridge"])
    # exact-target / polynomial diagnostics are still recorded (gate-independent).
    assert np.isfinite(row["exact_target_error_vs_ridge"])


def test_sweep_cell_overshoot_is_flagged_and_ineligible(monkeypatch) -> None:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate(1.0))
    H, r = _tiny_problem()
    row = evaluate_sweep_cell(
        _entry(),
        H,
        r,
        alpha=1e-4,
        degree=47,
        seed=7,
        shots=5000,
        trials=2,
        target_error=1e-2,
        overshoot_diagnostic_only=True,
    )
    assert row["overshoot_diagnostic_only"] is True
    assert row["overshoot_flag"] is True
    assert row["eligible_for_best_config"] == "no"
    assert row["selection_score"] >= 10.0  # overshoot penalty applied


def test_sweep_cell_selection_score_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate(2.0))
    H, r = _tiny_problem()
    a = evaluate_sweep_cell(
        _entry(), H, r, alpha=1e-4, degree=35, seed=7, shots=5000, trials=3, target_error=1e-2
    )
    b = evaluate_sweep_cell(
        _entry(), H, r, alpha=1e-4, degree=35, seed=7, shots=5000, trials=3, target_error=1e-2
    )
    assert a["selection_score"] == b["selection_score"]
    assert a["shot_reconstruction_error_vs_ridge"] == b["shot_reconstruction_error_vs_ridge"]


def test_run_sweep_includes_mandatory_excludes_fabricated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate(2.0))
    out = tmp_path / "full_vector_readout"
    _seed_matched(out)
    run = run_alpha_degree_sweep(
        {
            "input_root": "outputs",
            "output_dir": str(out),
            "alpha_grid": [1e-4, 1e-2],
            "degree_grid": [15, 45],
            "shots": 5000,
            "trials": 2,
            "mandatory": [("ieee14", "metadata_mapped"), ("ieee999", "fabricated")],
            "top_k_high_error": 0,
        }
    )
    assert ("ieee14", "metadata_mapped") in run["targeted"]
    cases = {r["case"] for r in run["sweep_rows"]}
    assert "ieee999" not in cases  # fabricated mandatory never produces sweep rows
    sweep = pd.read_csv(out / "readout_alpha_degree_sweep.csv")
    assert list(sweep.columns) == SWEEP_COLUMNS


# --------------------------------------------------------------------------- Phase 4
def _sweep_row(
    case, stype, alpha, degree, gate_err, *, eligible="yes", overshoot=False, bounded=True
) -> dict:
    return {
        "case": case,
        "subproblem_id": f"{stype}_00",
        "subproblem_type": stype,
        "alpha": alpha,
        "degree": degree,
        "target_scale_C": 1.0,
        "boundedness_ok": bounded,
        "overshoot_flag": overshoot,
        "overshoot_diagnostic_only": overshoot,
        "phase_synthesis_status": "synthesized",
        "gate_run_status": "computed",
        "exact_target_error_vs_ridge": 1e-16,
        "polynomial_error_vs_exact_target": gate_err,
        "gate_error_vs_polynomial": 1e-4,
        "gate_error_vs_ridge": gate_err,
        "shot_reconstruction_error_vs_gate": 2e-3,
        "shot_reconstruction_error_vs_ridge": gate_err + 2e-3,
        "residual_ratio_after_gate": gate_err,
        "residual_ratio_after_shot_reconstruction": gate_err,
        "sign_reliable_fraction": 1.0,
        "top_k_match": 1.0,
        "estimated_total_shots": 100000,
        "selection_score": gate_err
        + 0.1 * degree / 45.0
        + (10.0 if overshoot else 0.0)
        + (0.0 if bounded else 10.0),
        "eligible_for_best_config": eligible,
        "status": "computed",
        "notes": "n",
    }


def _matched_row(case, stype, gate_err, degree=35, dominant="exact_target_vs_polynomial") -> dict:
    return {
        "case": case,
        "subproblem_id": f"{stype}_00",
        "subproblem_type": stype,
        "view": codesign.MATCHED_VIEW,
        "alpha": codesign.MATCHED_ALPHA,
        "degree": degree,
        "gate_error_vs_ridge": gate_err,
        "exact_target_error_vs_ridge": 1e-16,
        "finite_polynomial_error": gate_err,
        "gate_vs_polynomial_error": 1e-4,
        "shot_readout_error": 2e-3,
        "residual_ratio_after_gate": gate_err,
        "residual_ratio_after_shot_reconstruction": gate_err,
        "dominant_error_source": dominant,
        "diagnostic_status": "fail_diagnostic",
        "interpretation": "finite_degree_limited_common_alpha",
        "source_artifact": "x",
        "notes": "n",
    }


def test_best_config_selects_from_eligible_only() -> None:
    sweep = [
        _sweep_row("ieee57", "high_leverage", 1e-2, 49, 0.01, eligible="no", overshoot=True),
        _sweep_row("ieee57", "high_leverage", 1e-6, 15, 0.005, eligible="no", bounded=False),
        _sweep_row("ieee57", "high_leverage", 1e-2, 45, 0.04),
        _sweep_row("ieee57", "high_leverage", 1e-4, 35, 0.32),
    ]
    matched = [_matched_row("ieee57", "high_leverage", 0.32)]
    best = select_best_configs(
        sweep, matched, [("ieee57", "high_leverage")], available_keys={("ieee57", "high_leverage")}
    )[0]
    assert best["best_degree"] == 45  # the eligible, low-error config
    assert best["best_alpha"] == 1e-2
    assert set(best.keys()) == set(BEST_CONFIG_COLUMNS)


def test_best_config_improvement_factor_and_status() -> None:
    sweep = [_sweep_row("ieee57", "high_leverage", 1e-2, 45, 0.04)]
    matched = [_matched_row("ieee57", "high_leverage", 0.32)]
    best = select_best_configs(
        sweep, matched, [("ieee57", "high_leverage")], available_keys={("ieee57", "high_leverage")}
    )[0]
    assert abs(best["improvement_factor"] - 0.32 / 0.04) < 1e-6
    assert best["status"] == "improved_by_codesign"
    assert best["selected_for_main_paper"] == "yes"


def test_best_config_limitation_aware_when_no_improvement() -> None:
    sweep = [_sweep_row("ieee14", "metadata_mapped", 1e-2, 45, 1.19)]
    matched = [_matched_row("ieee14", "metadata_mapped", 3.02)]
    best = select_best_configs(
        sweep,
        matched,
        [("ieee14", "metadata_mapped")],
        available_keys={("ieee14", "metadata_mapped")},
    )[0]
    # 1.19 is still > 0.10 and 3.02/1.19 ~= 2.5x < 10x, so it must be reported as a limitation.
    assert best["status"] == "qsvt_polynomial_limited"
    assert best["selected_for_main_paper"] == "reported_as_limitation"
    assert "finite-degree polynomial" in best["notes"]


def test_best_config_missing_when_no_eligible() -> None:
    sweep = [_sweep_row("ieee57", "high_leverage", 1e-2, 49, 0.01, eligible="no", overshoot=True)]
    matched = [_matched_row("ieee57", "high_leverage", 0.32)]
    best = select_best_configs(
        sweep, matched, [("ieee57", "high_leverage")], available_keys={("ieee57", "high_leverage")}
    )[0]
    assert best["status"] == "missing_valid_gate_config"
    assert best["best_degree"] == -1


def test_metadata_mapped_has_status_either_improved_or_limited() -> None:
    for best_err, expected in [(0.05, "improved_by_codesign"), (1.19, "qsvt_polynomial_limited")]:
        sweep = [_sweep_row("ieee14", "metadata_mapped", 1e-2, 45, best_err)]
        matched = [_matched_row("ieee14", "metadata_mapped", 3.02)]
        best = select_best_configs(
            sweep,
            matched,
            [("ieee14", "metadata_mapped")],
            available_keys={("ieee14", "metadata_mapped")},
        )[0]
        assert best["status"] == expected


# --------------------------------------------------------------------------- Phase 5
def test_validated_view_distinct_and_labeled() -> None:
    sweep = [_sweep_row("ieee57", "high_leverage", 1e-2, 45, 0.04)]
    matched = [_matched_row("ieee57", "high_leverage", 0.32)]
    best = select_best_configs(
        sweep, matched, [("ieee57", "high_leverage")], available_keys={("ieee57", "high_leverage")}
    )
    view = build_validated_view(best, sweep)[0]
    assert view["view"] == codesign.VALIDATED_VIEW
    assert view["interpretation"] == "validated_readout_pass"
    assert set(view.keys()) == set(VALIDATED_VIEW_COLUMNS)
    assert "readout_alpha_degree_sweep.csv" in view["source_sweep_artifact"]


def test_validated_view_marks_polynomial_limited() -> None:
    sweep = [_sweep_row("ieee14", "metadata_mapped", 1e-2, 45, 1.19)]
    matched = [_matched_row("ieee14", "metadata_mapped", 3.02)]
    best = select_best_configs(
        sweep,
        matched,
        [("ieee14", "metadata_mapped")],
        available_keys={("ieee14", "metadata_mapped")},
    )
    view = build_validated_view(best, sweep)[0]
    assert view["interpretation"] == "still_polynomial_limited"


def test_select_targeted_top_k_plus_mandatory(tmp_path: Path) -> None:
    _seed_matched(tmp_path)
    classification = classify_readout_configs(tmp_path)
    targeted = select_targeted_configs(
        classification, mandatory=(("ieee14", "metadata_mapped"),), top_k_high_error=2
    )
    assert ("ieee14", "metadata_mapped") in targeted
    assert ("ieee57", "high_leverage") in targeted
    assert ("ieee30", "high_leverage") not in targeted  # it passed, not high-error


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
