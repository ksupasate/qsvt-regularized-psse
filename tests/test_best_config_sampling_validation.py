"""Phase 8: best-config sampling validation across shot levels and trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.paper import phase_synthesis_refinement as psr

_THREE_VIEW = (
    "case,subproblem_id,subproblem_type,view,alpha,degree,phase_solver_variant,fit_mode,"
    "weight_mode,gate_error_vs_matching_ridge_alpha,ridge_alpha_shift_vs_common_alpha,"
    "exact_target_vs_polynomial_error,polynomial_vs_gate_error,shot_readout_error,"
    "success_probability,postselection_overhead,estimated_total_readout_shots,status,"
    "interpretation,source_artifact,notes\n"
    "ieee57,high_leverage_00,high_leverage,matched_alpha_common_reference,0.0001,35,default,"
    "global_uniform_fit,uniform_support_weight,0.32,0.0,0.32,0.01,,0.5,2.0,10000,"
    "qsvt_polynomial_limited,x,y,z\n"
    "ieee57,high_leverage_00,high_leverage,per_subproblem_relaxed_alpha_qsvt_validated,0.01,41,"
    "default,weighted_singular_support_fit,residual_energy_weight,0.03,0.0,0.02,0.01,,0.5,2.0,"
    "10000,pass,x,y,z\n"
)


def _patch(monkeypatch):
    rng = np.random.default_rng(1)
    H = np.diag([3.0, 2.0, 1.4, 0.7])
    r = rng.standard_normal(4)
    ctx = psr._RefinementContext(
        {"case": "ieee57", "subproblem_id": "high_leverage_00", "subproblem_type": "high_leverage"},
        H,
        r,
    )
    monkeypatch.setattr(
        psr,
        "_build_contexts",
        lambda targeted, input_root, seed: {("ieee57", "high_leverage"): ctx},
    )

    def realize(
        context, *, alpha, degree, fit_mode, weight_mode, seed, cache=None, angle_solver="iterative"
    ):
        from robust_qsvt_se.paper.full_vector_readout import ridge_update

        ridge = ridge_update(context.H, context.r, alpha=float(alpha))
        gate_state = ridge / float(np.linalg.norm(ridge))
        return {
            "available": True,
            "gate_update": ridge,
            "gate_state": gate_state,
            "ridge_update": ridge,
            "success_probability": 0.5,
        }

    monkeypatch.setattr(psr, "realize_config_gate", realize)


def test_sampling_runs_only_selected_configs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "three_view_readout_summary.csv").write_text(_THREE_VIEW, encoding="utf-8")
    _patch(monkeypatch)
    out = run = psr.run_best_config_sampling_validation(
        {
            "input_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "input_root": "outputs",
            "shots": [1000, 10000],
            "trials": 5,
        }
    )
    # only the relaxed (selected) view is validated, not the common reference
    assert {r["view"] for r in run["trial_rows"]} == {"per_subproblem_relaxed_alpha_qsvt_validated"}
    assert {r["shots"] for r in run["trial_rows"]} == {1000, 10000}
    assert len(run["summary_rows"]) == 2  # one per shot level
    _ = out


def test_seeds_and_trials_recorded(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "three_view_readout_summary.csv").write_text(_THREE_VIEW, encoding="utf-8")
    _patch(monkeypatch)
    run = psr.run_best_config_sampling_validation(
        {"input_dir": str(tmp_path), "output_dir": str(tmp_path), "shots": [1000], "trials": 5}
    )
    seeds = [r["rng_seed"] for r in run["trial_rows"]]
    assert len(seeds) == 5
    assert len(set(seeds)) == 5  # distinct per trial
    assert all(set(psr.SAMPLING_VALIDATION_COLUMNS) == set(r) for r in run["trial_rows"])


def test_summary_from_multiple_trials(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "three_view_readout_summary.csv").write_text(_THREE_VIEW, encoding="utf-8")
    _patch(monkeypatch)
    run = psr.run_best_config_sampling_validation(
        {"input_dir": str(tmp_path), "output_dir": str(tmp_path), "shots": [10000], "trials": 8}
    )
    summary = run["summary_rows"][0]
    assert set(psr.SAMPLING_VALIDATION_SUMMARY_COLUMNS) == set(summary)
    assert summary["trials"] == 8
    assert summary["p95_vector_relative_l2_error"] >= summary["p50_vector_relative_l2_error"]
    # higher shots -> not statevector-only; error is finite and positive
    assert summary["mean_vector_relative_l2_error"] > 0


def test_empty_when_no_three_view(tmp_path: Path, monkeypatch) -> None:
    _patch(monkeypatch)
    run = psr.run_best_config_sampling_validation(
        {"input_dir": str(tmp_path), "output_dir": str(tmp_path)}
    )
    assert run["trial_rows"] == []
    assert (tmp_path / "best_config_sampling_validation.csv").is_file()
