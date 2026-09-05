"""Finite-shot signed-readout and uncertainty tests for the sparse integrated chain."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    build_default_sparse_integrated_inputs,
    build_integrated_sparse_selected_output_circuit,
    compile_for_aer,
    estimate_signed_selected_output,
    exact_joint_distribution,
    sample_aer_counts,
    statevector_validate_integrated_chain,
)


@pytest.fixture(scope="module")
def readout_context(tmp_path_factory):
    inputs = build_default_sparse_integrated_inputs(
        tmp_path_factory.mktemp("sparse_integrated_readout"),
        shot_counts=(20_000,),
        seeds=(0, 1, 2),
    )
    functional = inputs.selected_functionals["coordinate_e0"]
    bundle = build_integrated_sparse_selected_output_circuit(
        inputs.config,
        matrix=inputs.matrix_quantized,
        residual=inputs.residual,
        selected_functional=functional,
        phases=inputs.phases,
    )
    validation = statevector_validate_integrated_chain(inputs, bundle)
    distribution = exact_joint_distribution(
        bundle.circuit,
        postselection_flag_qubit=bundle.register_layout["postselection_flag_qubit"],
        readout_qubit=bundle.register_layout["readout_qubit"],
    )
    compiled, simulator = compile_for_aer(bundle.circuit)
    return inputs, bundle, validation, distribution, compiled, simulator


def test_signed_estimator_handles_positive_negative_and_near_zero_counts():
    positive = estimate_signed_selected_output(
        {"00": 600, "10": 200, "01": 100, "11": 100}, physical_scale=2.0
    )
    negative = estimate_signed_selected_output(
        {"00": 200, "10": 600, "01": 100, "11": 100}, physical_scale=2.0
    )
    near_zero = estimate_signed_selected_output(
        {"00": 401, "10": 399, "01": 100, "11": 100}, physical_scale=2.0
    )
    assert positive["selected_output_estimate"] == pytest.approx(0.8)
    assert negative["selected_output_estimate"] == pytest.approx(-0.8)
    assert abs(near_zero["selected_output_estimate"]) < 0.01
    assert positive["readout_accepted"] == 800
    assert positive["inferred_postselection_probability_from_branch"] == pytest.approx(0.6)


def test_exact_integrated_distribution_matches_statevector_action(readout_context):
    inputs, _bundle, validation, distribution, _compiled, _simulator = readout_context
    functional = inputs.selected_functionals["coordinate_e0"]
    expected_z = float(
        np.real(np.vdot(functional / np.linalg.norm(functional), validation.sparse_encoded_state))
    )
    expected_acceptance = (
        1.0 + validation.metrics["sparse_postselection_probability"]
    ) / 2.0
    assert distribution["00"] + distribution["10"] == pytest.approx(
        expected_acceptance, abs=1e-9
    )
    assert distribution["00"] - distribution["10"] == pytest.approx(expected_z, abs=1e-9)


def test_actual_aer_sampling_is_unbiased_within_declared_uncertainty(readout_context):
    inputs, _bundle, validation, _distribution, compiled, simulator = readout_context
    functional = inputs.selected_functionals["coordinate_e0"]
    scale = (
        inputs.config.contraction_c
        / inputs.config.beta
        * np.linalg.norm(inputs.residual)
        * np.linalg.norm(functional)
    )
    reference = float(functional @ validation.sparse_update)
    estimates = []
    standard_errors = []
    for seed in inputs.config.seeds:
        counts = sample_aer_counts(compiled, simulator, shots=20_000, seed=seed)
        assert sum(counts.values()) == 20_000
        estimate = estimate_signed_selected_output(counts, physical_scale=scale)
        estimates.append(estimate["selected_output_estimate"])
        standard_errors.append(estimate["analytic_standard_error"])
        assert estimate["readout_accepted"] <= 20_000
    mean = float(np.mean(estimates))
    mean_se = float(np.mean(standard_errors)) / math.sqrt(len(estimates))
    assert abs(mean - reference) <= 5.0 * mean_se


def test_analytic_variance_matches_monte_carlo_and_ci_coverage(readout_context):
    inputs, _bundle, validation, distribution, _compiled, _simulator = readout_context
    functional = inputs.selected_functionals["coordinate_e0"]
    scale = (
        inputs.config.contraction_c
        / inputs.config.beta
        * np.linalg.norm(inputs.residual)
        * np.linalg.norm(functional)
    )
    keys = ["00", "01", "10", "11"]
    probabilities = np.asarray([distribution[key] for key in keys], dtype=np.float64)
    shots = 2_000
    rng = np.random.default_rng(20260710)
    estimates = []
    covered = []
    reference = float(functional @ validation.sparse_update)
    for draw in rng.multinomial(shots, probabilities, size=2_000):
        counts = {key: int(value) for key, value in zip(keys, draw, strict=True)}
        estimate = estimate_signed_selected_output(counts, physical_scale=scale)
        estimates.append(estimate["selected_output_estimate"])
        covered.append(
            estimate["confidence_interval_lower"]
            <= reference
            <= estimate["confidence_interval_upper"]
        )
    exact_f = distribution["00"] + distribution["10"]
    exact_z = distribution["00"] - distribution["10"]
    analytic_variance = scale**2 * (exact_f - exact_z**2) / shots
    empirical_variance = float(np.var(estimates, ddof=1))
    assert empirical_variance / analytic_variance == pytest.approx(1.0, rel=0.12)
    assert 0.92 <= float(np.mean(covered)) <= 0.98


def test_direct_and_interference_acceptance_counts_are_not_conflated(readout_context):
    _inputs, bundle, _validation, _distribution, _compiled, _simulator = readout_context
    compiled, simulator = compile_for_aer(bundle.direct_postselection_circuit)
    direct_counts = sample_aer_counts(compiled, simulator, shots=5_000, seed=9)
    postselection_accepted = direct_counts.get("0", 0)
    assert sum(direct_counts.values()) == 5_000
    assert 0 < postselection_accepted < 5_000
    # The direct event estimates p_post; the interference event instead estimates (1+p)/2.
    assert postselection_accepted / 5_000 < 0.75
