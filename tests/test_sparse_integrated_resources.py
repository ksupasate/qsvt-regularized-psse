"""Executed resource-accounting guards for the sparse integrated circuit."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    _build_dense_circuits,
    build_default_sparse_integrated_inputs,
    build_executed_resource_records,
    build_integrated_sparse_selected_output_circuit,
    compile_for_aer,
)


@pytest.fixture(scope="module")
def resource_context(tmp_path_factory):
    inputs = build_default_sparse_integrated_inputs(
        tmp_path_factory.mktemp("sparse_integrated_resources"),
        shot_counts=(1_000,),
        seeds=(0,),
    )
    functional = inputs.selected_functionals[inputs.config.selected_output_name]
    sparse = build_integrated_sparse_selected_output_circuit(
        inputs.config,
        matrix=inputs.matrix_quantized,
        residual=inputs.residual,
        selected_functional=functional,
        phases=inputs.phases,
    )
    dense, _direct = _build_dense_circuits(inputs, functional)
    compiled_sparse, _sparse_simulator = compile_for_aer(sparse.circuit)
    compiled_dense, _dense_simulator = compile_for_aer(dense)
    records, metadata = build_executed_resource_records(
        inputs,
        sparse,
        compiled_sparse=compiled_sparse,
        compiled_dense=compiled_dense,
    )
    return inputs, sparse, records, metadata


def _record(records, category):
    return next(row for row in records if row["resource_category"] == category)


def test_degree_dependent_signal_and_phase_counts(resource_context):
    inputs, sparse, records, _metadata = resource_context
    expected = qsvt_sequence_operation_counts(inputs.config.polynomial_degree + 1)
    assert expected["signal_unitary_calls"] == inputs.config.polynomial_degree
    assert expected["projector_phase_operations"] == inputs.config.polynomial_degree + 1
    assert sparse.operation_counts["signal_unitary_calls_per_attempt"] == 31
    assert sparse.operation_counts["forward_sparse_lookup_calls_per_attempt"] == 16
    assert sparse.operation_counts["inverse_lookup_calls_per_attempt"] == 15
    executed = _record(records, "executed_small_scale_sparse_integrated")
    assert executed["signal_unitary_calls_per_attempt"] == 31
    assert executed["projector_phase_operations_per_attempt"] == 32


def test_actual_sparse_circuit_resource_counts_are_populated(resource_context):
    _inputs, _sparse, records, metadata = resource_context
    executed = _record(records, "executed_small_scale_sparse_integrated")
    assert executed["total_logical_qubits"] == 8
    assert executed["ancilla_and_work_qubits"] == 5
    assert executed["transpiled_gate_count"] > 0
    assert executed["transpiled_depth"] > 0
    assert executed["toffoli_count"] > 0
    assert executed["controlled_rotation_count"] == 31 * 3 * 8
    assert executed["residual_preparations_per_attempt"] == 1
    assert metadata["sparse"]["gate_count"] == executed["transpiled_gate_count"]


def test_unknown_resource_fields_are_not_encoded_as_zero(resource_context):
    _inputs, _sparse, records, _metadata = resource_context
    modeled = _record(records, "modeled_ieee_scale_sparse_access")
    for key in (
        "transpiled_gate_count",
        "transpiled_depth",
        "toffoli_count",
        "controlled_rotation_count",
        "value_register_qubits",
    ):
        assert modeled[key] != 0
        assert "not_" in str(modeled[key])
    dense = _record(records, "executed_small_scale_dense_integrated")
    assert dense["controlled_rotation_count"] == "not_estimated"


def test_sparse_and_dense_records_share_configuration_fingerprint(resource_context):
    inputs, _sparse, records, _metadata = resource_context
    sparse_record = _record(records, "executed_small_scale_sparse_integrated")
    dense_record = _record(records, "executed_small_scale_dense_integrated")
    assert sparse_record["configuration_id"] == dense_record["configuration_id"]
    assert sparse_record["matrix_fingerprint"] == dense_record["matrix_fingerprint"]
    assert sparse_record["matrix_fingerprint"] == inputs.config.matrix_fingerprint


def test_executed_and_modeled_resource_categories_are_distinct(resource_context):
    _inputs, _sparse, records, _metadata = resource_context
    categories = {row["resource_category"] for row in records}
    assert categories == {
        "executed_small_scale_sparse_integrated",
        "executed_small_scale_dense_integrated",
        "modeled_ieee_scale_sparse_access",
    }
    modeled = _record(records, "modeled_ieee_scale_sparse_access")
    assert modeled["execution_status"] == "not_executed_in_this_experiment"
