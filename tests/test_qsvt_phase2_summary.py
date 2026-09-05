from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.phase2_preconditioned_alpha import build_phase2_summary


def test_phase2_summary_builds_from_sweep_and_alpha_outputs(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    alpha_dir = tmp_path / "alpha"
    sweep_dir.mkdir()
    alpha_dir.mkdir()
    pd.DataFrame(
        [
            {
                "case_name": "ieee118",
                "variant_name": "original_ridge",
                "alpha": 1.0e-2,
                "scenario_count": 1,
                "mean_residual_norm": 1.0,
                "mean_weighted_residual_norm": 1.0,
                "mean_rmse_if_available": 0.1,
                "mean_qsvt_full_interval_approx_error": 1.0e-4,
                "mean_qsvt_actual_singular_value_error": 1.0e-4,
                "mean_qsvt_degree": 201,
                "mean_qsvt_query_count": 403,
                "median_condition_number_original": 10.0,
                "median_condition_number_preconditioned_if_applicable": 5.0,
                "failure_count": 0,
                "status": "ok",
                "interpretation": "Original Ridge/Tikhonov reference.",
            }
        ]
    ).to_csv(sweep_dir / "phase2_sweep_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "case_name": "ieee118",
                "variant_name": "original_ridge",
                "selected_alpha": 1.0e-2,
                "selection_criterion": "joint_score_alpha",
                "score": 0.0,
                "metric_used": "diagnostic joint score",
                "caveat": "Diagnostic alpha-selection rule only",
            }
        ]
    ).to_csv(alpha_dir / "alpha_selection_summary.csv", index=False)

    run = build_phase2_summary(
        {
            "output_dir": str(tmp_path / "summary"),
            "sweep_summary_csv": str(sweep_dir / "phase2_sweep_summary.csv"),
            "alpha_selection_summary_csv": str(alpha_dir / "alpha_selection_summary.csv"),
        }
    )
    summary = pd.read_csv(run["output_dir"] / "phase2_summary.csv")
    text = (run["output_dir"] / "phase2_summary.md").read_text(encoding="utf-8")

    assert "ieee118" in set(summary["case_name"])
    assert "original_ridge" in set(summary["variant_name"])
    assert "Coordinate-preconditioned Ridge is a separate estimator" in text
    assert (run["output_dir"] / "manifest.json").is_file()
