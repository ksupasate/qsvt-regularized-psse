"""Phases 6-8: two-view summary, figure data, and the co-design scope note."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.paper import readout_alpha_degree_codesign as codesign
from robust_qsvt_se.paper.claim_lint import scan_file
from robust_qsvt_se.paper.final_artifact_validator import build_final_artifact_validator
from robust_qsvt_se.paper.full_vector_readout import build_readout_claim_family_rows, ridge_update
from robust_qsvt_se.paper.readout_alpha_degree_codesign import (
    FIGURE_DECOMP_COLUMNS,
    FIGURE_TWO_VIEW_COLUMNS,
    TWO_VIEW_SUMMARY_COLUMNS,
    build_error_decomposition_figure_rows,
    build_two_view_figure_rows,
    build_two_view_summary,
    readout_codesign_scope_note_markdown,
    run_alpha_degree_sweep,
    run_readout_two_view_summary,
    two_view_summary_markdown,
)


def _matched(case, stype, err, degree=35, dominant="exact_target_vs_polynomial") -> dict:
    return {
        "case": case,
        "subproblem_id": f"{stype}_00",
        "subproblem_type": stype,
        "view": codesign.MATCHED_VIEW,
        "alpha": codesign.MATCHED_ALPHA,
        "degree": degree,
        "gate_error_vs_ridge": err,
        "exact_target_error_vs_ridge": 1e-16,
        "finite_polynomial_error": err,
        "gate_vs_polynomial_error": 4e-4,
        "shot_readout_error": 2e-3,
        "residual_ratio_after_gate": err,
        "residual_ratio_after_shot_reconstruction": err,
        "dominant_error_source": dominant,
        "diagnostic_status": "fail_diagnostic",
        "interpretation": "finite_degree_limited_common_alpha",
        "source_artifact": "x",
        "notes": "n",
    }


def _validated(case, stype, err, degree, status, interp) -> dict:
    return {
        "case": case,
        "subproblem_id": f"{stype}_00",
        "subproblem_type": stype,
        "view": codesign.VALIDATED_VIEW,
        "alpha": 1e-2,
        "degree": degree,
        "selection_rule": "min",
        "gate_error_vs_ridge": err,
        "exact_target_error_vs_ridge": 1e-16,
        "finite_polynomial_error": err,
        "gate_vs_polynomial_error": 3e-4,
        "shot_readout_error": err + 2e-3,
        "residual_ratio_after_gate": err,
        "residual_ratio_after_shot_reconstruction": err,
        "sign_reliable_fraction": 1.0,
        "top_k_match": 1.0,
        "status": status,
        "interpretation": interp,
        "source_sweep_artifact": "readout_alpha_degree_sweep.csv",
        "notes": "n",
    }


def _best(case, stype, matched_err, best_err, degree, status, selected) -> dict:
    impr = matched_err / best_err if best_err > 0 else float("nan")
    return {
        "case": case,
        "subproblem_id": f"{stype}_00",
        "subproblem_type": stype,
        "matched_alpha": codesign.MATCHED_ALPHA,
        "matched_degree": 35,
        "matched_alpha_error": matched_err,
        "best_alpha": 1e-2,
        "best_degree": degree,
        "best_gate_error_vs_ridge": best_err,
        "best_polynomial_error_vs_exact_target": best_err,
        "best_gate_vs_polynomial_error": 3e-4,
        "best_shot_reconstruction_error_vs_ridge": best_err + 2e-3,
        "best_residual_ratio": best_err,
        "best_sign_reliable_fraction": 1.0,
        "best_top_k_match": 1.0,
        "improvement_factor": impr,
        "selection_rule": "min",
        "selected_for_main_paper": selected,
        "status": status,
        "notes": "finite-degree polynomial note" if status == "qsvt_polynomial_limited" else "n",
    }


def _scenario() -> tuple[list, list, list, list]:
    matched = [
        _matched("ieee57", "high_leverage", 0.32),
        _matched("ieee14", "metadata_mapped", 3.02),
    ]
    validated = [
        _validated(
            "ieee57", "high_leverage", 0.04, 45, "improved_by_codesign", "validated_readout_pass"
        ),
        _validated(
            "ieee14",
            "metadata_mapped",
            1.19,
            45,
            "qsvt_polynomial_limited",
            "still_polynomial_limited",
        ),
    ]
    best = [
        _best("ieee57", "high_leverage", 0.32, 0.04, 45, "improved_by_codesign", "yes"),
        _best(
            "ieee14",
            "metadata_mapped",
            3.02,
            1.19,
            45,
            "qsvt_polynomial_limited",
            "reported_as_limitation",
        ),
    ]
    targeted = [("ieee57", "high_leverage"), ("ieee14", "metadata_mapped")]
    return matched, validated, best, targeted


# --------------------------------------------------------------------------- Phase 6
def test_summary_includes_both_views_and_all_targets() -> None:
    matched, validated, best, _ = _scenario()
    rows = build_two_view_summary(matched, validated, best)
    assert {(r["case"], r["subproblem_type"]) for r in rows} == {
        ("ieee57", "high_leverage"),
        ("ieee14", "metadata_mapped"),
    }
    assert set(rows[0].keys()) == set(TWO_VIEW_SUMMARY_COLUMNS)
    by_pair = {(r["case"], r["subproblem_type"]): r for r in rows}
    assert by_pair[("ieee57", "high_leverage")]["final_status"] == "improved_by_codesign"
    assert (
        by_pair[("ieee14", "metadata_mapped")]["final_status"] == "polynomial_limited_after_sweep"
    )
    # both matched and validated errors present
    assert by_pair[("ieee57", "high_leverage")]["matched_alpha_gate_error"] == 0.32
    assert by_pair[("ieee57", "high_leverage")]["validated_gate_error"] == 0.04


def test_summary_does_not_overclaim_full_reconstruction() -> None:
    matched, validated, best, _ = _scenario()
    rows = build_two_view_summary(matched, validated, best)
    md = two_view_summary_markdown(rows)
    assert "accurately reconstructs Ridge for all" not in md
    assert "not all gate-level configs accurately reconstruct ridge" in md.lower()


def test_summary_states_scope_boundary() -> None:
    matched, validated, best, _ = _scenario()
    md = two_view_summary_markdown(build_two_view_summary(matched, validated, best))
    assert "efficient full ieee-scale full-vector readout remains outside" in md.lower()
    assert "matched-alpha view fixes alpha=1e-4" in md.lower()


# --------------------------------------------------------------------------- Phase 7
def test_figure_two_view_has_both_views() -> None:
    matched, validated, _, targeted = _scenario()
    rows = build_two_view_figure_rows(matched, validated, targeted)
    views = {r["view"] for r in rows}
    assert views == {codesign.MATCHED_VIEW, codesign.VALIDATED_VIEW}
    assert set(rows[0].keys()) == set(FIGURE_TWO_VIEW_COLUMNS)


def test_decomposition_figure_includes_high_error_and_is_nonnegative() -> None:
    matched, validated, _, targeted = _scenario()
    rows = build_error_decomposition_figure_rows(matched, validated, targeted)
    assert any(r["case"] == "ieee14" and r["subproblem_type"] == "metadata_mapped" for r in rows)
    assert set(rows[0].keys()) == set(FIGURE_DECOMP_COLUMNS)
    finite = [r["error_value"] for r in rows if np.isfinite(r["error_value"])]
    assert finite and all(value >= 0.0 for value in finite)


def test_decomposition_figure_marks_missing_components() -> None:
    matched, validated, _, targeted = _scenario()
    rows = build_error_decomposition_figure_rows(matched, validated, targeted)
    # gate_vs_shot_reconstruction is not separately recorded in the views -> labelled, not faked.
    gate_vs_shot = [r for r in rows if r["error_component"] == "gate_vs_shot_reconstruction"]
    assert gate_vs_shot and all(not np.isfinite(r["error_value"]) for r in gate_vs_shot)
    assert all("not separately recorded" in r["notes"] for r in gate_vs_shot)


# --------------------------------------------------------------------------- Phase 8
def test_scope_note_has_required_statements_and_subclaim() -> None:
    matched, validated, best, _ = _scenario()
    note = readout_codesign_scope_note_markdown(build_two_view_summary(matched, validated, best))
    lowered = note.lower()
    assert "separates common-alpha diagnostic behavior" in lowered
    assert "are not hidden" in lowered
    assert "reported as" in lowered and "polynomial-limited" in lowered
    assert "does not establish efficient full ieee-scale full-vector readout" in lowered
    assert "c10a.1" in lowered  # subclaim documented
    assert "not an independent readout claim" in lowered


def test_scope_note_is_claim_lint_clean(tmp_path: Path) -> None:
    matched, validated, best, _ = _scenario()
    note_path = tmp_path / "readout_codesign_scope_note.md"
    note_path.write_text(
        readout_codesign_scope_note_markdown(build_two_view_summary(matched, validated, best)),
        encoding="utf-8",
    )
    summary_path = tmp_path / "readout_two_view_summary.md"
    summary_path.write_text(
        two_view_summary_markdown(build_two_view_summary(matched, validated, best)),
        encoding="utf-8",
    )
    assert not [r for r in scan_file(note_path) if r["risk_level"] == "high"]
    assert not [r for r in scan_file(summary_path) if r["risk_level"] == "high"]


def test_lint_flags_full_scale_overclaim(tmp_path: Path) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(
        "With two-view co-design, full-vector readout solved at IEEE scale.", encoding="utf-8"
    )
    high = [r for r in scan_file(draft) if r["risk_level"] == "high"]
    assert any(r["matched_phrase"] == "full-vector readout solved" for r in high)


def test_two_view_does_not_promote_full_scale_claims() -> None:
    # C10a stays selected-subproblem; C10b stays future-work; co-design changes neither.
    rows = {r["claim_id"]: r for r in build_readout_claim_family_rows(full_vector_supported=True)}
    assert rows["C10a"]["claim_role"] == "selected_subproblem_readout_claim"
    assert rows["C10a"]["status"] == "supported_with_limitations"
    assert rows["C10b"]["status"] == "assumption_only"
    assert rows["C10"]["counts_as_independent_claim"] == "no"


# ----------------------------------------------------------- end-to-end (mocked gate)
def _fake_gate(H, r, *, alpha, degree, seed, shots=1000, case="custom"):
    ridge = ridge_update(H, r, alpha=float(alpha))
    err = 2.0 / float(degree)
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


_GATE_SUMMARY = (
    "case,subproblem_id,subproblem_type,alpha,degree,state_dimension,"
    "vector_relative_l2_error_vs_ridge,vector_relative_l2_error_vs_exact_target,"
    "max_coordinate_abs_error_vs_ridge,sign_accuracy,residual_ratio_after_reconstruction,"
    "gate_available,status,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,4,3.0189,3.0189,0.022,0.5,3.87,True,gr,n\n"
)
_DECOMP = (
    "case,subproblem_id,subproblem_type,alpha,degree,ridge_vs_exact_target_error,"
    "exact_target_vs_polynomial_error,polynomial_vs_gate_error,gate_vs_statevector_readout_error,"
    "statevector_vs_shot_reconstruction_error,shot_reconstruction_vs_ridge_error,"
    "dominant_error_source,status,notes\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,0.0,3.0178,4.3e-4,0.0,2.3e-3,2.3e-3,"
    "exact_target_vs_polynomial,decomposed,n\n"
)




# --------------------------------------------------------------------------- Phase 9
def _seed_codesign_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(codesign, "gate_level_subproblem_readout", _fake_gate)
    root = tmp_path / "pkg"
    out = root / "full_vector_readout"
    out.mkdir(parents=True)
    (out / "gate_level_full_vector_summary.csv").write_text(_GATE_SUMMARY, encoding="utf-8")
    (out / "readout_error_decomposition.csv").write_text(_DECOMP, encoding="utf-8")
    run_alpha_degree_sweep(
        {
            "input_root": "outputs",
            "output_dir": str(out),
            "alpha_grid": [1e-4, 1e-2],
            "degree_grid": [15, 45],
            "shots": 5000,
            "trials": 2,
            "mandatory": [("ieee14", "metadata_mapped")],
            "top_k_high_error": 0,
        }
    )
    run_readout_two_view_summary({"input_dir": str(out), "output_dir": str(out)})
    return root




def test_validator_v20_passes_and_adds_no_readout_failures(tmp_path: Path, monkeypatch) -> None:
    root = _seed_codesign_root(tmp_path, monkeypatch)
    run = build_final_artifact_validator({"input_root": str(root)})
    rows = pd.read_csv(run["artifacts"]["artifact_validation_report"])
    v20 = rows[rows["check_id"] == "V20"].iloc[0]
    assert v20["status"] == "passed"
    failed = rows[rows["status"] == "failed"]
    assert not failed["file_path"].astype(str).str.contains("full_vector_readout").any()


def test_validator_v20_not_applicable_without_codesign(tmp_path: Path) -> None:
    root = tmp_path / "empty_pkg"
    root.mkdir()
    run = build_final_artifact_validator({"input_root": str(root)})
    rows = pd.read_csv(run["artifacts"]["artifact_validation_report"])
    v20 = rows[rows["check_id"] == "V20"].iloc[0]
    assert v20["status"] == "not_applicable"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
