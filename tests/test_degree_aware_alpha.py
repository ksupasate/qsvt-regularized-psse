from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from robust_qsvt_se.paper.degree_aware_alpha import (
    GRID_COLUMNS,
    SUMMARY_COLUMNS,
    run_degree_aware_alpha_selection,
)
from robust_qsvt_se.paper.selected_observable_common import forbidden_in


@pytest.fixture(scope="module")
def ieee14_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("degree_alpha")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_degree_aware_alpha_selection(
            {"output_dir": str(out), "cases": ["ieee14", "ieee57"], "command": "test"}
        )


def test_grid_and_summary_columns(ieee14_run: dict) -> None:
    assert set(GRID_COLUMNS).issubset(ieee14_run["grid"].columns)
    assert set(SUMMARY_COLUMNS).issubset(ieee14_run["summary"].columns)


def test_selected_rows_satisfy_degree_budget(ieee14_run: dict) -> None:
    grid = ieee14_run["grid"]
    selected = grid[grid["selected"]]
    assert not selected.empty
    # Every degree-aware-selected row must meet its degree budget and the target.
    assert bool(selected["degree_budget_met"].all())
    assert bool(selected["target_met"].all())
    for _, row in selected.iterrows():
        assert row["degree_required"] != ""
        assert int(row["degree_required"]) <= int(row["degree_budget"])


def test_degree_aware_rule_only_picks_budget_feasible_alpha(ieee14_run: dict) -> None:
    summary = ieee14_run["summary"]
    degree_aware = summary[summary["selection_rule"] == "degree_aware_under_dmax"]
    assert not degree_aware.empty
    for _, row in degree_aware.iterrows():
        if row["selected_alpha"] != "":
            assert row["degree_budget_met"] is True or row["degree_budget_met"] == True  # noqa: E712
            assert int(row["degree_required_at_selected"]) <= int(row["degree_budget"])


def test_best_classical_can_exceed_budget_while_degree_aware_feasible(ieee14_run: dict) -> None:
    """The manuscript point: best-classical alpha need not be QSVT-implementable."""

    summary = ieee14_run["summary"]
    # ieee57: best classical alpha is degree-infeasible; degree-aware remains feasible.
    ieee57 = summary[summary["case"] == "ieee57"]
    best = ieee57[ieee57["selection_rule"] == "best_classical_rmse"].iloc[0]
    degree_aware = ieee57[ieee57["selection_rule"] == "degree_aware_under_dmax"].iloc[0]
    assert bool(best["degree_budget_met"]) is False
    assert bool(degree_aware["degree_budget_met"]) is True


def test_smaller_alpha_is_degree_cheaper_under_bounded_convention(ieee14_run: dict) -> None:
    grid = ieee14_run["grid"]
    slice_df = (
        grid[
            (grid["case"] == "ieee57") & (grid["tolerance"] == 1e-3) & (grid["degree_budget"] == 51)
        ]
        .drop_duplicates(subset=["alpha"])
        .sort_values("alpha")
    )
    # Smallest alpha is degree-feasible; the largest alpha is not (bounded convention).
    feasible = slice_df[slice_df["degree_required"] != ""]
    assert not feasible.empty
    assert float(feasible.iloc[0]["alpha"]) <= 1e-4


def test_report_is_claim_safe(ieee14_run: dict) -> None:
    output_dir = Path(ieee14_run["output_dir"])
    report = (output_dir / "degree_aware_alpha_report.md").read_text(encoding="utf-8")
    assert forbidden_in(report) == []
    assert "not automatically the best QSVT-implementable" in report
