from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.phase2_completion import build_phase2_complete_summary


def test_phase2_complete_summary_exists_and_separates_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    sweep_path, alpha_path, trace_path = _write_phase2_inputs(tmp_path)

    run = build_phase2_complete_summary(
        {
            "output_dir": str(tmp_path / "complete"),
            "sweep_results_csv": str(sweep_path),
            "alpha_selection_summary_csv": str(alpha_path),
            "alpha_selection_trace_csv": str(trace_path),
        }
    )
    output_dir = run["output_dir"]
    summary = pd.read_csv(output_dir / "phase2_complete_summary.csv")

    assert (output_dir / "phase2_complete_summary.json").is_file()
    assert (output_dir / "phase2_best_alpha_by_metric.csv").is_file()
    assert {"ieee118", "ieee300"}.issubset(set(summary["case_name"]))
    assert {
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
    }.issubset(set(summary["variant_name"]))
    assert (output_dir / "phase2_manifest.json").is_file()


def _write_phase2_inputs(root: Path) -> tuple[Path, Path, Path]:
    variants = [
        "original_ridge",
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
        "original_qsvt_diagnostic",
        "preconditioned_qsvt_diagnostic",
    ]
    rows = []
    alpha_rows = []
    trace_rows = []
    for case_name in ["ieee118", "ieee300"]:
        for variant in variants:
            rows.append(_result_row(case_name, variant))
            alpha_rows.append(
                {
                    "case_name": case_name,
                    "variant_name": variant,
                    "selected_alpha": 0.01,
                    "selection_criterion": "joint_score_alpha",
                    "score": 0.0,
                    "metric_used": "diagnostic joint score",
                    "caveat": "Diagnostic alpha-selection rule only",
                }
            )
            trace_rows.append(
                {
                    "case_name": case_name,
                    "variant_name": variant,
                    "alpha": 0.01,
                    "joint_score": 0.0,
                }
            )
    sweep_path = root / "phase2_sweep_results.csv"
    alpha_path = root / "alpha_selection_summary.csv"
    trace_path = root / "alpha_selection_trace.csv"
    pd.DataFrame(rows).to_csv(sweep_path, index=False)
    pd.DataFrame(alpha_rows).to_csv(alpha_path, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    return sweep_path, alpha_path, trace_path


def _result_row(case_name: str, variant: str) -> dict:
    preconditioned = "preconditioned" in variant
    qsvt_error = 1.0e-5 if preconditioned else 1.0e-1
    residual = 2.0 if variant == "coordinate_preconditioned_ridge" else 1.0
    return {
        "case_name": case_name,
        "variant_name": variant,
        "alpha": 0.01,
        "m": 10,
        "n": 4,
        "rank": 4,
        "condition_number_original": 100.0,
        "condition_number_preconditioned_if_applicable": 10.0 if preconditioned else "",
        "rmse_if_available": 0.2 if variant == "coordinate_preconditioned_ridge" else 0.1,
        "residual_norm": residual,
        "weighted_residual_norm": residual,
        "solution_norm": 1.0,
        "relative_solution_error_vs_original_ridge": 0.0,
        "relative_solution_error_vs_transformed_penalty": 0.0,
        "qsvt_full_interval_approx_error": qsvt_error,
        "qsvt_actual_singular_value_error": qsvt_error,
        "qsvt_degree": 201,
        "qsvt_query_count": 403,
        "phase_validation_status": "passed_scalar_full_domain",
        "status": "ok",
    }
