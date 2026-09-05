"""Tests for Experiment B: conditioning / alpha / degree / phase boundary."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.tqe_revision_conditioning_boundary import (
    GRID_COLUMNS,
    run_conditioning_boundary,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in

pytest.importorskip("pennylane")


@pytest.fixture(scope="module")
def boundary_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("conditioning_boundary")
    return run_conditioning_boundary(
        {
            "output_dir": str(output_dir),
            "sizes": [4],
            "kappa_grid": [100.0, 1_000_000.0],
            "matrix_seeds": [0],
            "lambda_grid": [1e-4, 6.9e-2],
            "degree_grid": [15, 31, 63],
            "ieee_cases": ["ieee14"],
            "ieee_sizes": [4],
            "include_stressed": False,
            "max_synthesis_degree": 45,
        }
    )


def test_required_output_files_created(boundary_run):
    output_dir = boundary_run["output_dir"]
    for name in [
        "boundary_grid_results.csv",
        "boundary_summary_by_kappa_alpha.csv",
        "phase_synthesis_failures.csv",
        "degree_vs_kappa_alpha.pdf",
        "degree_vs_kappa_alpha.png",
        "psucc_vs_kappa_alpha.pdf",
        "psucc_vs_kappa_alpha.png",
        "boundary_heatmap.tex",
        "boundary_summary_table.tex",
        "manifest.json",
        "README.md",
    ]:
        assert (output_dir / name).is_file(), name


def test_grid_has_required_columns(boundary_run):
    grid = boundary_run["grid"]
    for column in GRID_COLUMNS:
        assert column in grid.columns, column


def test_records_both_successes_and_failures(boundary_run):
    statuses = set(boundary_run["grid"]["pipeline_status"].unique())
    assert "success" in statuses
    # Degree 63 is above the ceiling; some non-success status must be present and kept.
    assert statuses & {
        "degree_above_supported_ceiling",
        "target_fit_failed",
        "phase_synthesis_failed",
    }


def test_no_rows_dropped(boundary_run):
    grid = boundary_run["grid"]
    # controlled: 1 size * 2 kappa * 1 seed * 2 lambda * 3 degree = 12; ieee14 4x4: 2*3 = 6.
    assert len(grid) == 12 + 6


def test_failures_file_keeps_failed_configs(boundary_run):
    failures = boundary_run["failures"]
    assert not failures.empty
    assert (failures["pipeline_status"] != "success").all()


def test_benign_regime_feasible_light_regime_not(boundary_run):
    summary = boundary_run["summary"]
    benign = summary[summary["lambda_alpha_over_beta2"] == 6.9e-2]
    light = summary[summary["lambda_alpha_over_beta2"] == 1e-4]
    assert benign["feasible_at_tolerance"].any()
    # Light regularization is outside the feasible region at these degrees.
    assert not light["feasible_at_tolerance"].any()


def test_postselection_present_and_finite(boundary_run):
    grid = boundary_run["grid"]
    success = grid[grid["pipeline_status"] == "success"]
    assert (success["postselection_probability"] > 0).all()
    assert (success["postselection_probability"] <= 1.0 + 1e-9).all()


def test_manifest_is_safe(boundary_run):
    manifest = json.loads((boundary_run["output_dir"] / "manifest.json").read_text())
    assert manifest["fabricates_results"] is False
    assert not forbidden_in(manifest["claim_boundary"])
    assert not forbidden_in((boundary_run["output_dir"] / "README.md").read_text())
