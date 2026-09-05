"""Tests for Phase 10 WP E end-to-end resource and classical comparator ledger."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.paper.phase10_resource_ledger import (
    run_phase10_resource_ledger,
)


@pytest.fixture(scope="module")
def ledger_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase10_resource_ledger")
    return run_phase10_resource_ledger({"output_dir": str(output_dir)})


def test_all_workloads_present_with_tiers(ledger_run):
    master = {row["workload"]: row for row in ledger_run["rows"]["master"]}
    assert "selected_4x4_integrated_chain" in master
    assert "selected_8x8_integrated_chain" in master
    assert "sparse_wrapper_8x8" in master
    assert "full_rectangular_ieee14" in master
    assert "full_rectangular_ieee30" in master
    assert "full_rectangular_ieee57" in master
    assert "full_rectangular_ieee300" in master
    # Execution tiers are explicit.
    assert master["full_rectangular_ieee14"]["execution_tier"] == "executed_statevector"
    assert master["full_rectangular_ieee300"]["execution_tier"] == "modeled"


def test_p_succ_matches_executed_packages(ledger_run):
    master = {row["workload"]: row for row in ledger_run["rows"]["master"]}
    # Sparse wrapper p_succ matches the WP A executed value it cites.
    assert master["sparse_wrapper_8x8"]["postselection_probability"] == pytest.approx(
        0.609042, abs=2e-3
    )
    # Full rectangular matches WP B executed values.
    assert master["full_rectangular_ieee14"]["postselection_probability"] == pytest.approx(
        0.848362, abs=2e-3
    )
    # 4x4 chain matches phase8 measured value.
    assert master["selected_4x4_integrated_chain"]["postselection_probability"] == pytest.approx(
        0.9904, abs=3e-3
    )
    assert master["selected_4x4_integrated_chain"]["qubits"] == 4
    # 8x8 chain is loaded from the executed Phase 9 integrated-readout package.
    assert master["selected_8x8_integrated_chain"]["postselection_probability"] == pytest.approx(
        0.9519140790609424, abs=1e-12
    )
    assert master["selected_8x8_integrated_chain"]["qubits"] == 5


def test_modeled_rows_have_no_postselection(ledger_run):
    master = {row["workload"]: row for row in ledger_run["rows"]["master"]}
    for case in ("ieee57", "ieee118", "ieee300"):
        row = master[f"full_rectangular_{case}"]
        assert row["execution_tier"] == "modeled"
        assert row["postselection_probability"] is None
        # Classical comparators are still measured for modeled rows.
        assert row["classical_full_ridge_seconds"] > 0


def test_classical_and_quantum_units_not_merged(ledger_run):
    classical = ledger_run["rows"]["classical"]
    for row in classical:
        assert "seconds" in row["unit"]
        assert row["full_ridge_seconds_median_best"] > 0
    # Quantum ledger reports counts, not seconds.
    quantum = ledger_run["rows"]["quantum"]
    for row in quantum:
        assert "seconds" not in str(row.get("block_encoding", ""))


def test_readout_full_vector_is_n_times_selected(ledger_run):
    readout = ledger_run["rows"]["readout"]
    assert readout
    for row in readout:
        if row["attempts_per_functional_no_AA"] and row["full_vector_recovery_attempts"]:
            # Full-vector recovery ~ n * selected.
            assert row["full_vector_recovery_attempts"] >= row["attempts_per_functional_no_AA"]


def test_nonlinear_repetition_present(ledger_run):
    nonlinear = ledger_run["rows"]["nonlinear"]
    assert nonlinear
    for row in nonlinear:
        assert row["residual_and_jacobian_rebuilt_per_iteration"] is True
        assert row["iterations"] == 8


def test_outputs_and_claim_safe(ledger_run):
    output_dir = ledger_run["output_dir"]
    for name in (
        "end_to_end_resource_ledger.csv",
        "classical_comparator_ledger.csv",
        "quantum_component_ledger.csv",
        "readout_cost_ledger.csv",
        "nonlinear_repetition_ledger.csv",
        "README.md",
        "manifest.json",
        "checksums.sha256",
        "command_log.txt",
    ):
        assert (output_dir / name).is_file(), name
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "phase10_end_to_end_resource_ledger"
    assert manifest["tier_counts"]["executed_statevector"] == 3
