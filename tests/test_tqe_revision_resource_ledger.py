"""Tests for Experiment C: fixed-case end-to-end resource ledger."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in
from robust_qsvt_se.paper.tqe_revision_resource_ledger import run_resource_ledger

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")


@pytest.fixture(scope="module")
def ledger_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("resource_ledger")
    return run_resource_ledger({"output_dir": str(output_dir), "timing_repeats": 5})


def test_required_output_files_created(ledger_run):
    output_dir = ledger_run["output_dir"]
    for name in [
        "fixed_case_resource_ledger.csv",
        "classical_adjoint_baseline.csv",
        "quantum_vs_classical_boundary.csv",
        "resource_waterfall.pdf",
        "resource_waterfall.png",
        "resource_table.tex",
        "classical_baseline_table.tex",
        "assumptions.md",
        "manifest.json",
        "README.md",
    ]:
        assert (output_dir / name).is_file(), name


def test_ledger_separates_tiers(ledger_run):
    ledger = ledger_run["ledger"]
    tiers = set(ledger["tier"].unique())
    # Implemented, proxy, and modeled tiers must all be represented and distinguished.
    assert {"implemented", "proxy", "modeled", "finite_shot", "statevector"} <= tiers
    fields = set(ledger["field"])
    assert {
        "shots_for_target_error",
        "total_signal_unitary_calls_without_AA",
        "postselection_probability",
        "state_preparation_status",
    } <= fields


def test_classical_baseline_matches_ridge(ledger_run):
    classical = ledger_run["classical"]
    assert len(classical) == 3
    # The adjoint/direct/sparse selected-observable values equal the Ridge reference.
    assert (classical["abs_difference_from_ridge_reference"] < 1.0e-9).all()
    assert (classical["median_runtime_seconds"] > 0).all()
    assert (classical["num_timing_repeats"] == 5).all()


def test_boundary_has_both_sides(ledger_run):
    boundary = ledger_run["boundary"]
    sides = set(boundary["side"])
    assert {"classical_adjoint", "quantum_qsvt", "conclusion"} <= sides


def test_resource_conclusion_is_not_a_speed_claim(ledger_run):
    manifest = json.loads((ledger_run["output_dir"] / "manifest.json").read_text())
    assert manifest["resource_conclusion_type"] == 1
    assert not forbidden_in((ledger_run["output_dir"] / "assumptions.md").read_text())
    assert not forbidden_in((ledger_run["output_dir"] / "README.md").read_text())


def test_shots_and_queries_are_positive(ledger_run):
    manifest = json.loads((ledger_run["output_dir"] / "manifest.json").read_text())
    assert manifest["shots_for_target_error"] > 0
    assert manifest["total_signal_unitary_calls_without_AA"] > 0


def test_ledger_separates_signal_calls_from_sequence_length(ledger_run):
    ledger = ledger_run["ledger"].set_index("field")
    assert int(ledger.loc["signal_unitary_calls_per_attempt", "value"]) == 31
    assert int(ledger.loc["alternating_sequence_length_per_attempt", "value"]) == 63
