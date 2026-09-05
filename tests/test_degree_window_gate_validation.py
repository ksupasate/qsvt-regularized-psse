from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.degree_window_gate_validation import (
    GATE_COLUMNS,
    select_degree_window_gate_configs,
    write_degree_window_gate_outputs,
)


def _feasible_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "toy",
                "alpha": 1.0e-2,
                "degree": 37,
                "target_family": "weighted_support_ls",
                "weighting_scheme": "combined_singular_support",
                "degree_window_class": "residual_feasible",
                "residual_ratio_vs_no_update": 0.00006,
                "direction_error_vs_ridge": 0.0003,
                "success_probability_proxy": 0.15,
            },
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "toy",
                "alpha": 1.0e-6,
                "degree": 15,
                "target_family": "weighted_support_ls",
                "weighting_scheme": "combined_singular_support",
                "degree_window_class": "residual_feasible",
                "residual_ratio_vs_no_update": 0.017,
                "direction_error_vs_ridge": 0.02,
                "success_probability_proxy": 0.98,
            },
            {
                "case": "ieee14",
                "model": "ac_linearized",
                "subproblem_id": "toy",
                "alpha": 1.0e-6,
                "degree": 45,
                "target_family": "residual_aware",
                "weighting_scheme": "residual_least_squares",
                "degree_window_class": "residual_feasible",
                "residual_ratio_vs_no_update": 0.08,
                "direction_error_vs_ridge": 0.03,
                "success_probability_proxy": 0.8,
            },
        ]
    )


def test_select_reads_configs_from_feasible_degree_window() -> None:
    selected = select_degree_window_gate_configs(_feasible_frame(), max_configs=5)
    assert not selected.empty
    assert "selection_reason" in selected.columns
    degrees = set(int(value) for value in selected["degree"])
    # The lowest and highest feasible degrees must both be represented among the picks.
    assert 15 in degrees
    assert 45 in degrees
    # Best-residual intent selects the lowest residual ratio (degree 37).
    assert 37 in degrees


def test_select_returns_empty_for_empty_input() -> None:
    selected = select_degree_window_gate_configs(pd.DataFrame(), max_configs=5)
    assert selected.empty


def test_write_outputs_when_no_results(tmp_path) -> None:
    selected = select_degree_window_gate_configs(pd.DataFrame(), max_configs=5)
    artifacts = write_degree_window_gate_outputs(
        tmp_path, {"output_dir": str(tmp_path)}, selected, []
    )
    assert artifacts["degree_window_gate_results"].is_file()
    results = pd.read_csv(artifacts["degree_window_gate_results"])
    assert results.empty or set(GATE_COLUMNS).issubset(set(results.columns))
