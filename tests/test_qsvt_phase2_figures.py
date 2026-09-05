from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.phase2_completion import build_phase2_figures


def test_phase2_figures_and_captions_exist(tmp_path: Path) -> None:
    sweep_path = tmp_path / "phase2_sweep_summary.csv"
    trace_path = tmp_path / "alpha_selection_trace.csv"
    _sweep_summary_frame().to_csv(sweep_path, index=False)
    _alpha_trace_frame().to_csv(trace_path, index=False)

    output_dir = tmp_path / "figures"
    build_phase2_figures(
        {
            "output_dir": str(output_dir),
            "sweep_summary_csv": str(sweep_path),
            "alpha_selection_trace_csv": str(trace_path),
        }
    )

    required = [
        "fig_phase2_ieee300_residual_vs_alpha.png",
        "fig_phase2_ieee300_rmse_vs_alpha.png",
        "fig_phase2_ieee300_qsvt_error_vs_alpha.png",
        "fig_phase2_ieee300_residual_rmse_qsvt_tradeoff.png",
        "fig_phase2_ieee118_qsvt_error_vs_alpha.png",
        "fig_phase2_ieee118_residual_vs_alpha.png",
        "fig_phase2_variant_comparison_ieee300.png",
        "fig_phase2_original_vs_preconditioned_kappa.png",
        "fig_phase2_alpha_selection_score.png",
        "phase2_figure_captions.md",
    ]
    for filename in required:
        assert (output_dir / filename).is_file()
    assert (output_dir / "fig_phase2_ieee300_residual_rmse_qsvt_tradeoff.pdf").is_file()
    assert "claim-safe interpretation" in (output_dir / "phase2_figure_captions.md").read_text(
        encoding="utf-8"
    )


def _sweep_summary_frame() -> pd.DataFrame:
    rows = []
    variants = [
        "original_ridge",
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
        "original_qsvt_diagnostic",
        "preconditioned_qsvt_diagnostic",
    ]
    for case_name in ["ieee118", "ieee300"]:
        for variant in variants:
            for alpha in [0.001, 0.01, 0.1]:
                preconditioned = "preconditioned" in variant
                rows.append(
                    {
                        "case_name": case_name,
                        "variant_name": variant,
                        "alpha": alpha,
                        "mean_residual_norm": 2.0 if variant.startswith("coordinate") else 1.0,
                        "mean_rmse_if_available": (
                            0.2 if variant.startswith("coordinate") else 0.1
                        ),
                        "mean_qsvt_full_interval_approx_error": (
                            1.0e-5 if preconditioned else 1.0e-1
                        )
                        / alpha,
                        "median_condition_number_original": 100.0,
                        "median_condition_number_preconditioned_if_applicable": (
                            10.0 if preconditioned else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _alpha_trace_frame() -> pd.DataFrame:
    rows = []
    for case_name in ["ieee118", "ieee300"]:
        for variant in [
            "original_ridge",
            "coordinate_preconditioned_ridge",
            "transformed_penalty_preconditioned_ridge",
            "original_qsvt_diagnostic",
            "preconditioned_qsvt_diagnostic",
        ]:
            for alpha, score in [(0.001, 0.5), (0.01, 0.1), (0.1, 0.3)]:
                rows.append(
                    {
                        "case_name": case_name,
                        "variant_name": variant,
                        "alpha": alpha,
                        "joint_score": score,
                    }
                )
    return pd.DataFrame(rows)
