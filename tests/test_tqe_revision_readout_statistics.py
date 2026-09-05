"""Tests for Experiment A: seed-resolved readout statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in
from robust_qsvt_se.paper.tqe_revision_readout_statistics import (
    SEED_RESULT_COLUMNS,
    run_readout_statistics,
)

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

_NUM_SEEDS = 3
_SHOTS = [500, 5000]


@pytest.fixture(scope="module")
def readout_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("readout_statistics")
    return run_readout_statistics(
        {
            "output_dir": str(output_dir),
            "num_seeds": _NUM_SEEDS,
            "shots_grid": _SHOTS,
        }
    )


def test_required_output_files_created(readout_run):
    output_dir = readout_run["output_dir"]
    for name in [
        "readout_seed_results.csv",
        "readout_shot_scaling_summary.csv",
        "readout_error_vs_shots.pdf",
        "readout_error_vs_shots.png",
        "readout_error_vs_shots.tex",
        "readout_statistics_table.tex",
        "manifest.json",
        "README.md",
    ]:
        assert (output_dir / name).is_file(), name


def test_seed_results_have_required_columns(readout_run):
    frame = readout_run["seed_results"]
    for column in SEED_RESULT_COLUMNS:
        assert column in frame.columns, column


def test_no_configurations_dropped(readout_run):
    frame = readout_run["seed_results"]
    n_signed = frame["observable_label"].nunique()
    assert len(frame) == n_signed * len(_SHOTS) * _NUM_SEEDS
    # No silently dropped finite-shot estimates.
    assert frame["finite_shot_estimate"].notna().all()


def test_summary_aggregates_seeds(readout_run):
    summary = readout_run["summary"]
    pooled = summary[summary["observable_label"] == "__all_signed_pooled__"]
    assert not pooled.empty
    assert (pooled["num_seeds"] == _NUM_SEEDS).all()
    assert (pooled["num_failures"] == 0).all()
    # Relative error should decrease with more shots (shot-noise averaging).
    ordered = pooled.sort_values("shots")
    errors = ordered["mean_relative_error_vs_ridge"].to_numpy()
    assert errors[-1] <= errors[0]


def test_exact_svt_matches_statevector_functional(readout_run):
    frame = readout_run["seed_results"]
    diff = np.abs(frame["exact_svt_functional"] - frame["exact_qsvt_statevector_functional"])
    assert float(diff.max()) < 1.0e-6


def test_sampling_mode_labelled(readout_run):
    modes = set(readout_run["seed_results"]["sampling_mode"].unique())
    allowed = {"aer_circuit_shot_sampling", "finite_shot_sampling_proxy_from_exact_overlap"}
    assert modes <= allowed and modes


def test_readout_is_explicitly_isolated_from_qsvt_execution(readout_run):
    frame = readout_run["seed_results"]
    assert not frame["integrated_qsvt_readout"].any()
    assert set(frame["output_state_access_model"]) == {
        "direct_StatePreparation_of_classically_computed_postselected_output"
    }


def test_manifest_is_safe_and_complete(readout_run):
    manifest = json.loads((readout_run["output_dir"] / "manifest.json").read_text())
    for key in ["claim_boundary", "random_seeds", "interpretation_boundary", "outputs_generated"]:
        assert key in manifest
    assert manifest["fabricates_results"] is False
    assert not forbidden_in(manifest["claim_boundary"])
    assert not forbidden_in((readout_run["output_dir"] / "README.md").read_text())
