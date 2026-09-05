"""Configuration, sparse-block, circuit-composition, and numerical integration tests."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from qiskit.quantum_info import Statevector

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    slot_values_from_assignment,
)
from robust_qsvt_se.qsvt.bipartite_slot_assignment import validate_slot_assignment
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    PHASE_CONVENTION,
    assert_no_direct_output_initializer,
    build_default_sparse_integrated_inputs,
    build_integrated_sparse_selected_output_circuit,
    load_sparse_integrated_config,
    stable_array_fingerprint,
    statevector_validate_integrated_chain,
    validate_integrated_inputs,
)


@pytest.fixture(scope="module")
def integrated_context(tmp_path_factory):
    inputs = build_default_sparse_integrated_inputs(
        tmp_path_factory.mktemp("sparse_integrated_config"),
        shot_counts=(2_000,),
        seeds=(0,),
    )
    bundle = build_integrated_sparse_selected_output_circuit(
        inputs.config,
        matrix=inputs.matrix_quantized,
        residual=inputs.residual,
        selected_functional=inputs.selected_functionals[inputs.config.selected_output_name],
        phases=inputs.phases,
    )
    validation = statevector_validate_integrated_chain(inputs, bundle)
    return inputs, bundle, validation


def test_valid_frozen_configuration_and_formula(integrated_context):
    inputs, _bundle, _validation = integrated_context
    config = inputs.config
    assert config.configuration_id == "ieee14_sparse_quantized_8x8_d31_selected_v1"
    assert config.phase_convention == PHASE_CONVENTION
    assert config.matrix_shape == (8, 8)
    assert config.polynomial_degree == 31
    assert inputs.phases.size == config.polynomial_degree + 1
    assert config.normalized_lambda == pytest.approx(config.alpha / config.beta**2, rel=1e-13)


def test_valid_frozen_configuration_loads(integrated_context, tmp_path):
    inputs, _bundle, _validation = integrated_context
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(inputs.config.to_json_dict()), encoding="utf-8")
    loaded = load_sparse_integrated_config(path)
    assert loaded == inputs.config


def test_configuration_rejects_even_degree_and_inconsistent_lambda(integrated_context):
    inputs, _bundle, _validation = integrated_context
    with pytest.raises(ValueError, match="odd degree"):
        replace(inputs.config, polynomial_degree=30)
    with pytest.raises(ValueError, match="normalized_lambda"):
        replace(inputs.config, normalized_lambda=inputs.config.normalized_lambda * 1.01)


def test_dimensions_phase_count_complex_mode_and_fingerprints_are_guarded(integrated_context):
    inputs, _bundle, _validation = integrated_context
    kwargs = {
        "matrix": inputs.matrix_quantized,
        "residual": inputs.residual,
        "selected_functional": inputs.selected_functionals["coordinate_e0"],
        "phases": inputs.phases,
    }
    validate_integrated_inputs(inputs.config, **kwargs)
    assert stable_array_fingerprint(inputs.matrix_quantized) == inputs.config.matrix_fingerprint
    with pytest.raises(ValueError, match="residual length"):
        validate_integrated_inputs(inputs.config, **{**kwargs, "residual": inputs.residual[:-1]})
    with pytest.raises(ValueError, match="functional length"):
        validate_integrated_inputs(
            inputs.config,
            **{**kwargs, "selected_functional": np.ones(7)},
        )
    with pytest.raises(ValueError, match="phase sequence length"):
        validate_integrated_inputs(inputs.config, **{**kwargs, "phases": inputs.phases[:-1]})
    complex_matrix = inputs.matrix_quantized.astype(np.complex128)
    complex_matrix[0, 0] += 1.0e-3j
    with pytest.raises(ValueError, match="unsupported complex matrix"):
        validate_integrated_inputs(inputs.config, **{**kwargs, "matrix": complex_matrix})


def test_dense_and_sparse_reference_matrix_must_be_identical(integrated_context):
    inputs, _bundle, _validation = integrated_context
    dense = inputs.matrix_quantized.copy()
    dense[0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="identical quantized matrix"):
        validate_integrated_inputs(
            inputs.config,
            matrix=inputs.matrix_quantized,
            residual=inputs.residual,
            selected_functional=inputs.selected_functionals["coordinate_e0"],
            phases=inputs.phases,
            dense_reference_matrix=dense,
        )


def test_sparse_lookup_assignment_padding_and_reversibility(integrated_context):
    inputs, bundle, _validation = integrated_context
    pattern = np.abs(inputs.matrix_quantized.T) > 0.0
    report = validate_slot_assignment(pattern, bundle.wrapper.assignment)
    assert report["real_edges_covered_exactly_once"] is True
    values = slot_values_from_assignment(
        inputs.matrix_quantized.T,
        float(np.max(np.abs(inputs.matrix_quantized))),
        bundle.wrapper.assignment,
    )
    for slot, mask in enumerate(bundle.wrapper.assignment.real_edge_mask):
        for column, is_real in enumerate(mask):
            if not is_real:
                assert values[slot, column] == 0.0
    unitary = np.asarray(bundle.wrapper.unitary)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-10)
    basis = np.zeros(unitary.shape[0], dtype=np.complex128)
    basis[0] = 1.0
    round_trip = Statevector(basis).evolve(bundle.wrapper.circuit).evolve(
        bundle.wrapper.circuit.inverse()
    )
    np.testing.assert_allclose(round_trip.data, basis, atol=1e-10)


def test_sparse_block_reconstructs_same_dense_target(integrated_context):
    inputs, bundle, validation = integrated_context
    target = inputs.matrix_quantized.T / inputs.config.beta
    dense = canonical_square_block_encoding(target)
    np.testing.assert_allclose(bundle.wrapper.encoded_block, target, atol=1e-9)
    np.testing.assert_allclose(dense.unitary[:8, :8], target, atol=1e-12)
    assert validation.metrics["block_reconstruction_relative_fro_error"] < 1e-9


def test_integrated_circuit_contains_every_required_stage(integrated_context):
    _inputs, bundle, _validation = integrated_context
    labels = [str(item.operation.label or item.operation.name) for item in bundle.circuit.data]
    assert labels.count("c0_residual_prep") == 1
    assert labels.count("c0_sparse_U_A") == 16
    assert labels.count("c0_sparse_U_A_dagger") == 15
    sparse_operations = [
        item.operation
        for item in bundle.circuit.data
        if str(item.operation.label or "").startswith("c0_sparse_U_A")
    ]
    assert all(
        "complete_sparse_BE_wrapper_8x8" in operation.name
        for operation in sparse_operations
    )
    assert labels.count("c0_PCPhase") == 32
    assert labels.count("c1_functional_prep") == 1
    assert labels.count("measure") == 2
    assert labels.count("mcx") == 1
    assert bundle.operation_counts["signal_unitary_calls_per_attempt"] == 31
    assert bundle.operation_counts["projector_phase_operations_per_attempt"] == 32
    assert bundle.output_state_used_for_preparation is False


def test_classically_computed_output_is_not_initialized(integrated_context):
    _inputs, bundle, validation = integrated_context
    assert_no_direct_output_initializer(bundle.circuit, validation.sparse_encoded_state)


def test_sparse_dense_exact_svt_and_quantized_ridge_agree(integrated_context):
    _inputs, _bundle, validation = integrated_context
    metrics = validation.metrics
    assert metrics["sparse_dense_action_relative_l2_error"] < 1e-8
    assert metrics["qsvt_exact_polynomial_svt_relative_l2_error"] < 1e-6
    assert metrics["qsvt_quantized_ridge_relative_l2_error"] < 0.05
    assert metrics["exact_rational_svt_quantized_ridge_relative_l2_error"] < 1e-12
    assert metrics["quantized_ridge_original_ridge_relative_l2_difference"] > 0.0


def test_guard_would_reject_output_state_initializer(integrated_context):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    _inputs, _bundle, validation = integrated_context
    output = validation.sparse_encoded_state
    normalized = output / np.linalg.norm(output)
    bad = QuantumCircuit(3)
    bad.append(StatePreparation(normalized), range(3))
    with pytest.raises(RuntimeError, match="directly initializes"):
        assert_no_direct_output_initializer(bad, output)
