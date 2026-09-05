"""Tests for Phase 10 WP C residual loading and repeat-cost accounting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.qsvt.binary_tree_state_loader import (
    build_binary_tree_circuit,
    compute_binary_tree_plan,
    validate_binary_tree_circuit,
)


def test_dense_loader_fidelity_small():
    from robust_qsvt_se.qsvt.gate_state_preparation import (
        build_initialize_circuit,
        normalize_and_pad_for_gate_preparation,
        validate_initialize_circuit,
    )

    rng = np.random.default_rng(1)
    vector = rng.standard_normal(6)  # non-power-of-two -> padded to 8
    prep = normalize_and_pad_for_gate_preparation(vector)
    circuit = build_initialize_circuit(prep.padded_state)
    validation = validate_initialize_circuit(circuit, prep.padded_state)
    assert validation["state_preparation_l2_error"] <= 1e-10
    assert validation["state_preparation_fidelity"] == pytest.approx(1.0, abs=1e-10)


def test_binary_tree_loader_fidelity_small_signed_vectors():
    rng = np.random.default_rng(7)
    for size in (2, 4, 5, 8, 16):
        vector = rng.standard_normal(size)  # signed, arbitrary
        plan = compute_binary_tree_plan(vector)
        # N - 1 rotations for a dimension-N (padded) loader.
        assert plan.rotation_count == plan.dimension - 1
        assert plan.reconstruction_error <= 1e-12
        circuit = build_binary_tree_circuit(plan)
        validation = validate_binary_tree_circuit(circuit, plan)
        assert validation["state_preparation_l2_error"] <= 1e-10
        assert validation["state_preparation_fidelity"] == pytest.approx(1.0, abs=1e-10)


def test_binary_tree_angle_plan_exact_for_full_residual():
    # The angle plan must be exact even where the compiled circuit is skipped.
    rng = np.random.default_rng(3)
    vector = rng.standard_normal(82)
    plan = compute_binary_tree_plan(vector)
    assert plan.dimension == 128
    assert plan.rotation_count == 127
    assert plan.reconstruction_error <= 1e-12


def test_qrom_cost_scales_with_vector_length():
    from robust_qsvt_se.paper.phase10_residual_loading import qrom_loader_cost

    small = qrom_loader_cost(np.ones(8), precision_bits=8)
    large = qrom_loader_cost(np.ones(64), precision_bits=8)
    assert large["stored_values"] > small["stored_values"]
    assert large["toffoli_estimate"] > small["toffoli_estimate"]
    assert large["t_count_estimate"] > small["t_count_estimate"]
    # T-count model: 4 T per Toffoli over SELECT + value-load + uncompute.
    assert small["t_count_estimate"] == 4 * (
        small["toffoli_estimate"] + small["uncompute_toffoli_estimate"]
    )


@pytest.fixture(scope="module")
def loading_run(tmp_path_factory):
    from robust_qsvt_se.paper.phase10_residual_loading import run_phase10_residual_loading

    output_dir = tmp_path_factory.mktemp("phase10_residual_loading")
    return run_phase10_residual_loading({"output_dir": str(output_dir)})


def test_p_succ_matches_executed_convention(loading_run):
    # p_succ must match the executed WP B / phase8 values (full complex amplitude).
    summary = loading_run["summary"]
    assert summary["full_rectangular_ieee14"]["p_succ"] == pytest.approx(0.8484, abs=2e-3)
    assert summary["full_rectangular_ieee30"]["p_succ"] == pytest.approx(0.8887, abs=2e-3)
    assert summary["selected_4x4_integrated_chain"]["p_succ"] == pytest.approx(0.9904, abs=3e-3)


def test_t_prep_counted_per_attempt(loading_run):
    for row in loading_run["selected_rows"] + loading_run["full_rows"]:
        # State preparation is invoked once per postselection attempt.
        assert row["T_prep_counted_per_attempt"] is True
        assert row["state_prep_invocations"] == row["postselection_attempts_no_AA"]
        # Attempts scale as q * N_shots / p_succ.
        expected = np.ceil(
            row["q_functionals"] * row["readout_shots_per_functional"] / row["p_succ"]
        )
        assert row["postselection_attempts_no_AA"] == expected
    # Full-vector recovery costs n times the single-functional readout.
    single = next(
        r
        for r in loading_run["full_rows"]
        if r["workload"] == "full_rectangular_ieee14"
        and r["q_label"] == "single_functional"
        and r["readout_epsilon"] == 1e-2
    )
    full = next(
        r
        for r in loading_run["full_rows"]
        if r["workload"] == "full_rectangular_ieee14"
        and r["q_label"] == "full_vector"
        and r["readout_epsilon"] == 1e-2
    )
    assert full["q_functionals"] == single["n_states"]


def test_nonlinear_loop_counts_reload_per_iteration(loading_run):
    rows = loading_run["nonlinear_rows"]
    assert rows
    for row in rows:
        assert row["residual_reload_per_iteration"] is True
        assert row["jacobian_rebuild_per_iteration"] is True
        assert (
            row["state_prep_invocations_total_loop"]
            == row["state_prep_invocations_per_iteration"] * row["iterations"]
        )


def test_outputs_and_claim_safe(loading_run):
    output_dir = loading_run["output_dir"]
    for name in (
        "residual_loading_modes.csv",
        "residual_loading_selected_workloads.csv",
        "residual_loading_full_rectangular.csv",
        "residual_loading_nonlinear_loop.csv",
        "residual_loading_resource_summary.json",
        "README.md",
        "manifest.json",
        "checksums.sha256",
        "command_log.txt",
    ):
        assert (output_dir / name).is_file(), name
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "phase10_residual_loading_accounting"
