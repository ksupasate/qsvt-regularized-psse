from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_full_gate_level_qsvt_coverage import (
    COVERAGE_COLUMNS,
    DegreeSelection,
    empty_coverage_row,
    evaluate_tiny_mock_gate_case,
    phase_failure_row,
    ridge_update_comparison_metrics,
    run_full_gate_level_qsvt_coverage,
    select_degree_from_previous_sweep,
    skipped_by_budget_row,
)
from robust_qsvt_se.qsvt.tqe_integrated_small_qsvt_circuit import PhaseSynthesisResult


def test_output_schema_smoke_with_budget_skip(tmp_path: Path) -> None:
    run = run_full_gate_level_qsvt_coverage(
        {
            "output_root": str(tmp_path),
            "case_specs": [
                {
                    "tier": "unit_test",
                    "case_name": "ieee14",
                    "subproblem_size": 4,
                    "skip_by_budget": True,
                    "skip_reason": "unit-test budget skip",
                }
            ],
            "degree_summary_path": str(tmp_path / "missing_summary.csv"),
            "degree_results_path": str(tmp_path / "missing_results.csv"),
        }
    )
    frame = pd.read_csv(run["artifacts"]["results_csv"])

    assert set(COVERAGE_COLUMNS).issubset(frame.columns)
    assert len(frame) == 1
    assert run["artifacts"]["metadata_json"].is_file()


def test_selected_degree_lookup_prefers_required_degree(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "case_name": "ieee14",
                "subproblem_size": 4,
                "selection_criterion": "high_leverage",
                "alpha": 1.0e-2,
                "epsilon_target": 1.0e-2,
                "required_degree": 5,
                "best_available_degree": 11,
            }
        ]
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    selection = select_degree_from_previous_sweep(
        summary_path=summary_path,
        results_path=tmp_path / "missing.csv",
        case_name="ieee14",
        subproblem_size=4,
        selection_criterion="high_leverage",
        alpha=1.0e-2,
        epsilon_target=1.0e-2,
        fallback_degree=11,
    )

    assert selection.degree == 5
    assert selection.target_met
    assert selection.source == "degree_alpha_precision_summary_required_degree"


def test_phase_synthesis_failure_is_recorded() -> None:
    row = phase_failure_row(
        case_spec={
            "tier": "unit",
            "case_name": "ieee14",
            "subproblem_size": 4,
            "selection_mode": "high_leverage",
        },
        config={"alpha": 1.0e-2, "epsilon_target": 1.0e-2},
        degree_selection=DegreeSelection(5, "unit", False, ""),
        phase_result=PhaseSynthesisResult(
            phases=np.array([], dtype=np.float64),
            status="failed",
            failure_reason="forced failure",
            convention="unit-test",
        ),
        reason="forced failure",
    )

    assert row["phase_synthesis_status"] == "failed"
    assert row["simulation_status"] == "skipped_phase_synthesis_failed"
    assert "forced failure" in row["failure_or_skip_reason"]


def test_qsvt_circuit_metrics_are_finite_on_tiny_mock_case() -> None:
    row = evaluate_tiny_mock_gate_case()

    assert row["phase_synthesis_status"] == "completed"
    assert row["qsvt_circuit_status"] == "completed"
    assert row["simulation_status"] == "completed"
    assert np.isfinite(row["transform_block_fro_error"])
    assert np.isfinite(row["circuit_vs_polynomial_fro_error"])
    assert np.isfinite(row["success_probability"])


def test_skipped_by_budget_recording() -> None:
    row = skipped_by_budget_row(
        case_spec={
            "tier": "tier3",
            "case_name": "ieee57",
            "subproblem_size": 16,
            "selection_mode": "high_leverage",
        },
        config={"alpha": 1.0e-2, "epsilon_target": 1.0e-2},
        degree_selection=DegreeSelection(5, "unit", False, "fallback"),
        runtime_seconds=0.01,
        reason="too large for unit-test budget",
    )

    assert row["qsvt_circuit_status"] == "skipped_by_budget"
    assert row["simulation_status"] == "skipped_by_budget"
    assert row["transpilation_status"] == "skipped_by_budget"
    assert "too large" in row["failure_or_skip_reason"]


def test_ridge_comparison_metric() -> None:
    metrics = ridge_update_comparison_metrics(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1.0, 2.1], dtype=np.float64),
    )

    assert np.isclose(metrics["absolute_update_error"], 0.1)
    assert np.isclose(metrics["relative_update_error"], 0.1 / np.sqrt(5.0))
    assert np.isclose(metrics["max_component_error"], 0.1)


def test_empty_coverage_row_has_required_columns() -> None:
    row = empty_coverage_row(
        tier="unit",
        case_name="case",
        subproblem_size=4,
        selection_criterion="mock",
    )

    assert set(COVERAGE_COLUMNS).issubset(row.keys())
