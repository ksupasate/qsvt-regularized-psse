from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.codesigned_gate_validation import (
    GATE_COLUMNS,
    select_codesigned_gate_configs,
    write_codesigned_gate_outputs,
)


def _feasible_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subproblem_id": "toy",
                "alpha": 1.0e-4,
                "degree": 25,
                "target_family": "residual_aware",
                "deployability_class": "general_qsvt_safe",
                "residual_ratio_vs_no_update": 0.0002,
                "direction_error_vs_ridge": 0.0009,
            },
            {
                "subproblem_id": "toy",
                "alpha": 1.0e-2,
                "degree": 35,
                "target_family": "weighted_support_ls",
                "deployability_class": "instance_aware_qsvt_safe",
                "residual_ratio_vs_no_update": 0.0003,
                "direction_error_vs_ridge": 0.0005,
            },
        ]
    )


def _diagnostic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subproblem_id": "toy",
                "alpha": 1.0e-3,
                "degree": 35,
                "target_family": "direction_preserving",
                "deployability_class": "diagnostic_only",
                "residual_ratio_vs_no_update": 0.9,
                "direction_error_vs_ridge": 0.02,
            }
        ]
    )


def test_select_reads_deployable_feasible_configs_first() -> None:
    selected, source = select_codesigned_gate_configs(
        _feasible_frame(), _diagnostic_frame(), max_configs=3
    )
    assert source == "deployable_residual_feasible"
    assert len(selected) == 2
    # Ranked by residual_ratio_vs_no_update ascending.
    assert float(selected.iloc[0]["residual_ratio_vs_no_update"]) <= float(
        selected.iloc[1]["residual_ratio_vs_no_update"]
    )


def test_select_falls_back_to_diagnostic_when_feasible_empty() -> None:
    empty = pd.DataFrame(columns=_feasible_frame().columns)
    selected, source = select_codesigned_gate_configs(empty, _diagnostic_frame(), max_configs=3)
    assert source == "diagnostic_feasible"
    assert len(selected) == 1


def test_select_returns_none_when_both_empty() -> None:
    empty = pd.DataFrame(columns=_feasible_frame().columns)
    selected, source = select_codesigned_gate_configs(empty, empty, max_configs=3)
    assert source == "none"
    assert selected.empty


def test_write_outputs_when_no_configs(tmp_path) -> None:
    empty = pd.DataFrame(columns=_feasible_frame().columns)
    selected, source = select_codesigned_gate_configs(empty, empty, max_configs=3)
    artifacts = write_codesigned_gate_outputs(
        tmp_path, {"output_dir": str(tmp_path)}, selected, [], source=source
    )
    assert artifacts["codesigned_gate_validation_results"].is_file()
    results = pd.read_csv(artifacts["codesigned_gate_validation_results"])
    assert results.empty or set(GATE_COLUMNS).issubset(set(results.columns))
