from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.residual_feasible_codesigned_search import (
    RESULT_COLUMNS,
    evaluate_codesigned_feasibility,
    write_codesigned_search_outputs,
)


def _summary_frame() -> pd.DataFrame:
    rows = [
        # Deployable, QSVT-safe, feasible.
        {
            "case": "ieee14",
            "model": "ac_linearized",
            "subproblem_id": "toy",
            "alpha": 1.0e-4,
            "degree": 35,
            "target_family": "weighted_support_ls",
            "weighting_scheme": "combined_singular_support",
            "qsvt_safe": True,
            "deployability_class": "instance_aware_qsvt_safe",
            "success_probability_proxy": 0.02,
            "residual_ratio_vs_no_update": 0.0003,
            "direction_error_vs_ridge": 0.0005,
            "cos_angle_vs_ridge": 0.9999,
            "residual_amplitude_recovered_proxy": 0.006,
            "residual_best_scalar_diagnostic": 0.005,
        },
        # Diagnostic-only: a non-deployable family whose best-scalar lower bound is feasible.
        {
            "case": "ieee14",
            "model": "ac_linearized",
            "subproblem_id": "toy",
            "alpha": 1.0e-4,
            "degree": 35,
            "target_family": "direction_preserving",
            "weighting_scheme": "ridge_direction",
            "qsvt_safe": True,
            "deployability_class": "diagnostic_only",
            "success_probability_proxy": 0.5,
            "residual_ratio_vs_no_update": 0.8,
            "direction_error_vs_ridge": 0.02,
            "residual_amplitude_recovered_proxy": 16.0,
            "residual_best_scalar_diagnostic": 0.4,  # ratio 0.02 vs no_update 20 -> feasible
        },
        # Rejected: direction error too large.
        {
            "case": "ieee14",
            "model": "ac_linearized",
            "subproblem_id": "toy",
            "alpha": 1.0e-2,
            "degree": 51,
            "target_family": "weighted_support_ls",
            "weighting_scheme": "combined_singular_support",
            "qsvt_safe": True,
            "deployability_class": "instance_aware_qsvt_safe",
            "success_probability_proxy": 0.1,
            "residual_ratio_vs_no_update": 5.0,
            "direction_error_vs_ridge": 1.8,
            "residual_amplitude_recovered_proxy": 100.0,
            "residual_best_scalar_diagnostic": 90.0,
        },
    ]
    return pd.DataFrame(rows)


def test_diagnostic_only_configs_are_not_counted_as_deployable() -> None:
    rows = evaluate_codesigned_feasibility(_summary_frame())
    by_family = {(row["target_family"], row["degree"]): row for row in rows}

    deployable = by_family[("weighted_support_ls", 35)]
    assert deployable["residual_feasible"] is True
    assert deployable["diagnostic_feasible"] is False

    diagnostic = by_family[("direction_preserving", 35)]
    assert diagnostic["residual_feasible"] is False
    assert diagnostic["diagnostic_feasible"] is True

    rejected = by_family[("weighted_support_ls", 51)]
    assert rejected["residual_feasible"] is False
    assert rejected["diagnostic_feasible"] is False


def test_outputs_separate_feasible_and_diagnostic(tmp_path) -> None:
    rows = evaluate_codesigned_feasibility(_summary_frame())
    artifacts = write_codesigned_search_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)

    feasible = pd.read_csv(artifacts["deployable_residual_feasible_configs"])
    diagnostic = pd.read_csv(artifacts["diagnostic_feasible_configs"])
    assert len(feasible) == 1
    assert len(diagnostic) == 1
    # Deployable feasible rows are never diagnostic-only families.
    assert set(feasible["deployability_class"]).issubset(
        {"general_qsvt_safe", "instance_aware_qsvt_safe"}
    )
    for column in RESULT_COLUMNS:
        assert column in feasible.columns
