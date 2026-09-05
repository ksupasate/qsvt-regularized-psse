from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

import robust_qsvt_se.qsvt.generic_sparse_compiler as compiler_module
from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompilerInputBundle,
    ExecutionSpec,
    FunctionalDefinition,
    FunctionalSpec,
    MatrixSpec,
    QSVTSpec,
    QuantizationSpec,
    ResidualSpec,
    SparseCompilerError,
    SupportSpec,
    compile_from_bundle,
)
from robust_qsvt_se.qsvt.generic_sparse_execution import (
    build_resource_evidence,
    prepare_compiled_execution,
    validate_compiled_statevector,
)
from robust_qsvt_se.qsvt.generic_sparse_workloads import (
    build_canonical_compiler_inputs,
    build_second_ieee30_compiler_inputs,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    PHASE_CONVENTION,
    estimate_signed_selected_output,
)


def _synthetic_bundle(
    matrix: np.ndarray | None = None,
    *,
    slots: int = 3,
    bits: int = 6,
    coordinates: tuple[tuple[int, int], ...] | None = None,
) -> CompilerInputBundle:
    values = np.asarray(
        matrix
        if matrix is not None
        else [
            [2.0, -1.0, 0.0, 0.5],
            [0.0, 1.5, -0.75, 0.0],
            [1.0, 0.0, 0.5, -0.25],
            [0.0, -0.5, 1.0, 1.25],
        ]
    )
    if coordinates is None:
        coordinates = tuple(
            (int(i), int(j)) for i, j in zip(*np.nonzero(values), strict=True)
        )
    selected = np.zeros(values.shape, dtype=np.complex128 if np.iscomplexobj(values) else float)
    for coordinate in coordinates:
        selected[coordinate] = values[coordinate]
    mu = float(np.max(np.abs(selected), initial=1.0))
    beta = float(slots * mu)
    normalized_lambda = 0.1
    dimension = int(values.shape[0])
    coordinate = np.zeros(dimension)
    coordinate[0] = 1.0
    aggregate = np.ones(dimension) / math.sqrt(dimension)
    return CompilerInputBundle(
        matrix_spec=MatrixSpec(
            values=values,
            shape=values.shape,
            matrix_id="synthetic_in_memory",
            workload_id=None,
        ),
        support_spec=SupportSpec(
            coordinates=coordinates,
            support_id="synthetic_support",
            slots=slots,
            allow_zero_source_values=False,
        ),
        quantization_spec=QuantizationSpec(
            magnitude_bits=bits,
            scale=mu,
        ),
        qsvt_spec=QSVTSpec(
            alpha=normalized_lambda * beta**2,
            beta=beta,
            normalized_lambda=normalized_lambda,
            boundedness_factor=1.0,
            polynomial_coefficients=np.array([0.0, 0.5]),
            polynomial_id="synthetic_half_x",
            phases=np.array([-4.97418837, -0.26179939]),
            degree=1,
            parity="odd",
            phase_convention=PHASE_CONVENTION,
        ),
        residual_spec=ResidualSpec(
            vector=np.arange(1, dimension + 1, dtype=float),
            residual_id="synthetic_residual",
        ),
        functional_spec=FunctionalSpec(
            functionals=(
                FunctionalDefinition("coordinate", coordinate, "coordinate"),
                FunctionalDefinition("aggregate", aggregate, "aggregate"),
            ),
            primary_functional_id="coordinate",
        ),
        execution_spec=ExecutionSpec(
            shot_counts=(100,),
            simulator_seeds=(0,),
            execute_finite_shots=False,
        ),
    )


def _assert_code(bundle: CompilerInputBundle, code: str) -> None:
    with pytest.raises(SparseCompilerError) as error:
        compile_from_bundle(bundle)
    assert error.value.code == code
    assert error.value.to_record()["stage"]


@pytest.fixture(scope="module")
def canonical_compiled():
    return compile_from_bundle(build_canonical_compiler_inputs())


@pytest.fixture(scope="module")
def second_compiled():
    return compile_from_bundle(build_second_ieee30_compiler_inputs())


@pytest.fixture(scope="module")
def canonical_statevector(canonical_compiled):
    return validate_compiled_statevector(canonical_compiled)


@pytest.fixture(scope="module")
def second_statevector(second_compiled):
    return validate_compiled_statevector(second_compiled)


def test_valid_in_memory_configuration_compiles_without_repository_state(tmp_path, monkeypatch):
    bundle = _synthetic_bundle()
    monkeypatch.chdir(tmp_path)
    compiled = compile_from_bundle(bundle)
    assert compiled.matrix_spec.source == "in_memory"
    assert compiled.workload_id.startswith("sparse_qsvt_")
    assert compiled.final_measured_circuit is not None


def test_workload_ids_and_hashes_are_deterministic():
    first = compile_from_bundle(_synthetic_bundle())
    second = compile_from_bundle(_synthetic_bundle())
    assert first.workload_id == second.workload_id
    assert first.workload_digest == second.workload_digest
    assert first.component_hashes == second.component_hashes


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_multiple_value_precisions_compile(bits):
    compiled = compile_from_bundle(_synthetic_bundle(bits=bits))
    assert compiled.quantization_spec.magnitude_bits == bits
    assert np.all(np.isfinite(compiled.matrix_quantized))


@pytest.mark.parametrize(
    "matrix,slots",
    [
        (np.diag([1.0, 1.0, -0.5, 0.0]), 1),
        (
            np.array(
                [
                        [1.0, -0.25, 0.0, 0.0],
                        [0.0, 1.0, 0.5, 0.0],
                        [-0.5, 0.0, 0.0, 0.25],
                        [0.0, 0.0, -0.5, 1.0],
                ]
            ),
            2,
        ),
        (np.diag(np.linspace(0.25, 2.0, 8)), 1),
    ],
)
def test_generic_square_matrices_cover_repeated_zero_negative_and_unequal_degree(matrix, slots):
    bundle = _synthetic_bundle(matrix, slots=slots)
    compiled = compile_from_bundle(bundle)
    assert compiled.matrix_original.shape == matrix.shape
    assert len(compiled.functional_vectors) == 2
    assert compiled.wrapper.slots == slots


@pytest.mark.parametrize("shape", [(3, 4), (4, 3)])
def test_rectangular_inputs_are_explicitly_rejected(shape):
    matrix = np.ones(shape)
    bundle = _synthetic_bundle(np.eye(4))
    bundle = replace(bundle, matrix_spec=replace(bundle.matrix_spec, values=matrix, shape=shape))
    _assert_code(bundle, "unsupported_rectangular_orientation")


def test_duplicate_support_is_structured_failure():
    bundle = _synthetic_bundle()
    coordinates = bundle.support_spec.coordinates
    _assert_code(
        replace(bundle, support_spec=replace(bundle.support_spec, coordinates=coordinates + (coordinates[0],))),
        "duplicate_support_coordinates",
    )


def test_out_of_bounds_support_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(
        replace(bundle, support_spec=replace(bundle.support_spec, coordinates=((0, 0), (4, 0)))),
        "support_out_of_bounds",
    )


def test_slot_overflow_is_structured_failure():
    bundle = _synthetic_bundle(slots=3)
    beta = bundle.qsvt_spec.beta / 3.0
    altered = replace(
        bundle,
        support_spec=replace(bundle.support_spec, slots=1),
        qsvt_spec=replace(
            bundle.qsvt_spec,
            beta=beta,
            alpha=bundle.qsvt_spec.normalized_lambda * beta**2,
        ),
    )
    _assert_code(altered, "slot_overflow")


def test_complex_matrix_is_structured_failure():
    bundle = _synthetic_bundle()
    values = bundle.matrix_spec.values.astype(complex)
    values[0, 0] += 0.1j
    _assert_code(replace(bundle, matrix_spec=replace(bundle.matrix_spec, values=values)), "unsupported_complex_matrix")


def test_nonfinite_matrix_is_structured_failure():
    bundle = _synthetic_bundle()
    values = bundle.matrix_spec.values.copy()
    values[0, 0] = np.nan
    _assert_code(replace(bundle, matrix_spec=replace(bundle.matrix_spec, values=values)), "non_finite_values")


def test_unsupported_quantization_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(
        replace(bundle, quantization_spec=replace(bundle.quantization_spec, sign_representation="twos_complement")),
        "unsupported_quantization",
    )


def test_invalid_phase_count_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(replace(bundle, qsvt_spec=replace(bundle.qsvt_spec, phases=np.array([0.0]))), "invalid_phase_count")


def test_degree_parity_mismatch_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(replace(bundle, qsvt_spec=replace(bundle.qsvt_spec, degree=2, parity="even")), "degree_parity_mismatch")


def test_missing_uncomputation_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(replace(bundle, qsvt_spec=replace(bundle.qsvt_spec, require_uncomputation=False)), "missing_uncomputation")


def test_zero_residual_is_structured_failure():
    bundle = _synthetic_bundle()
    _assert_code(replace(bundle, residual_spec=replace(bundle.residual_spec, vector=np.zeros(4))), "zero_residual_norm")


def test_incompatible_functional_dimension_is_structured_failure():
    bundle = _synthetic_bundle()
    invalid = FunctionalDefinition("bad", np.ones(3), "coordinate")
    _assert_code(
        replace(bundle, functional_spec=FunctionalSpec((invalid,), "bad")),
        "incompatible_functional_dimension",
    )


def test_register_collision_is_structured_failure():
    bundle = _synthetic_bundle()
    override = (
        ("index", (0, 1)),
        ("slot", (1, 2)),
        ("rotation_ancilla", (3,)),
        ("postselection_flag", (4,)),
        ("signed_readout", (5,)),
    )
    _assert_code(
        replace(bundle, execution_spec=replace(bundle.execution_spec, register_override=override)),
        "register_collision",
    )


def test_compiler_source_has_no_workload_registry_or_frozen_paths():
    source = inspect.getsource(compiler_module)
    forbidden = (
        "generic_sparse_workloads",
        "outputs/",
        "_build_block",
        "selected_rows =",
        "selected_columns =",
        "phase_angles.csv",
        "seed123",
    )
    assert not any(token in source for token in forbidden)


def test_sparse_lookup_sign_quantization_and_slot_permutations(canonical_compiled):
    compiled = canonical_compiled
    assert compiled.sparse_value_sign_lookup["lookup_value_max_error"] == 0.0
    assert any(row["sign"] < 0 for row in compiled.value_controlled_rotations)
    assert any(row["sign"] > 0 for row in compiled.value_controlled_rotations)
    assert len(compiled.slot_controlled_permutations) == 3
    assert compiled.sparse_index_lookup["exact_support_coverage"] is True


def test_wrapper_inverse_uncomputes_every_basis_state(canonical_compiled):
    unitary = canonical_compiled.wrapper.unitary
    identity = unitary.conj().T @ unitary
    assert np.max(np.abs(identity - np.eye(identity.shape[0]))) <= 1.0e-9
    assert canonical_compiled.inverse_uncomputation_path["wrapper_inverse_is_complete"] is True
    assert canonical_compiled.inverse_uncomputation_path["discarded_work_registers"] is False


def test_canonical_hashes_and_integer_resources_reproduce(canonical_compiled):
    assert canonical_compiled.workload_id == "ieee14_sparse_quantized_8x8_d31_selected_v1"
    assert canonical_compiled.component_hashes["matrix_original"] == "b158d34b86b778f0c290519ca98985345107012e225798a4cfc7fbf9178df7f9"
    assert canonical_compiled.component_hashes["matrix_quantized"] == "26159050694e76abc32692332daba94e9cd5e22d958a242236b4d57509aeab21"
    prepared = prepare_compiled_execution(canonical_compiled)
    resources = build_resource_evidence(canonical_compiled, prepared).record
    assert resources["total_simultaneously_live_qubits"] == 8
    assert resources["transpiled_gate_count"] == 186191
    assert resources["transpiled_depth"] == 180380
    assert resources["toffoli_count"] == 51898
    assert resources["controlled_rotation_count"] == 744
    assert resources["register_sum"] == resources["total_simultaneously_live_qubits"]


def test_canonical_statevector_matches_historical_metrics(canonical_statevector):
    metrics = canonical_statevector.metrics
    assert metrics["epsilon_block"] == pytest.approx(1.0096734076438567e-12, abs=5e-15)
    assert metrics["sparse_dense_action_relative_error"] == pytest.approx(1.5402592989062325e-12, abs=5e-15)
    assert metrics["epsilon_qsvt"] == pytest.approx(1.1157854519147143e-8, abs=5e-15)
    assert metrics["qsvt_quantized_ridge_relative_error"] == pytest.approx(1.6278664099110887e-4, abs=5e-15)
    assert metrics["sparse_postselection_probability"] == pytest.approx(0.6090421558900074, abs=5e-15)


def test_second_workload_is_frozen_without_leakage(second_compiled):
    compiled = second_compiled
    assert compiled.component_hashes["matrix_original"] == "634f0fd7e657faf78b834dbe34f1196af4b2836c3935c38fd774ad57dec1c963"
    assert compiled.component_hashes["support_mask"] == "b24bc7e54c4067ba4d49179190fb209c022bd7e3d875551edf44cbece5994adc"
    assert compiled.residual_spec.provenance["seed"] == 2000
    assert 2000 not in compiled.support_spec.provenance["training_seed_ids"]
    assert compiled.support_spec.provenance["held_out_or_truth_used"] is False
    assert compiled.qsvt_spec.provenance["output_metrics_used_for_selection"] is False
    assert compiled.qsvt_spec.degree == 31


def test_second_functionals_are_metadata_defined_and_unavailable_request_retained(second_compiled):
    assert set(second_compiled.functional_vectors) == {
        "coordinate_angle_bus4",
        "branch_angle_diff_4_6",
    }
    assert any(
        item["requested_functional_id"] == "connected_block_angle_area_aggregate"
        for item in second_compiled.functional_spec.unavailable_requests
    )
    assert all(definition.metadata for definition in second_compiled.functional_spec.functionals)


def test_second_end_to_end_action_and_recovery_pass(second_statevector):
    metrics = second_statevector.metrics
    assert metrics["epsilon_lookup"] <= 1e-14
    assert metrics["epsilon_block"] <= 1e-9
    assert metrics["sparse_dense_action_relative_error"] <= 1e-9
    assert metrics["epsilon_qsvt"] <= 1e-6
    assert metrics["max_signed_recovery_absolute_error"] <= 1e-12
    assert metrics["dense_fallback_used"] is False
    assert metrics["direct_output_state_preparation_used"] is False


@pytest.mark.parametrize("fixture_name", ["canonical_compiled", "second_compiled"])
def test_phase_convention_applied_once_and_final_circuit_is_shared(request, fixture_name):
    compiled = request.getfixturevalue(fixture_name)
    assert len(compiled.phases) == compiled.qsvt_spec.degree + 1
    assert compiled.qsvt_spec.phase_convention == PHASE_CONVENTION
    primary = compiled.functional_spec.primary_functional_id
    assert compiled.final_measured_circuit is compiled.functional_circuits[primary].circuit
    operation_names = {
        instruction.operation.name for instruction in compiled.final_measured_circuit.data
    }
    assert "initialize" not in operation_names
    assert "dense_U_A" not in operation_names


def test_physical_recovery_factor_is_exact(second_compiled):
    for functional_id, vector in second_compiled.functional_vectors.items():
        expected = (
            second_compiled.qsvt_spec.boundedness_factor
            / second_compiled.qsvt_spec.beta
            * np.linalg.norm(second_compiled.residual)
            * np.linalg.norm(vector)
        )
        assert second_compiled.recovery_factors[functional_id] == pytest.approx(expected, abs=1e-15)


def test_signed_readout_distinguishes_postselection_and_interference_events():
    counts = {"00": 40, "10": 20, "01": 25, "11": 15}
    estimate = estimate_signed_selected_output(counts, physical_scale=2.0)
    assert estimate["readout_accepted"] == 60
    assert estimate["interference_acceptance_probability"] == pytest.approx(0.6)
    assert estimate["signed_overlap_estimate"] == pytest.approx(0.2)
    assert estimate["selected_output_estimate"] == pytest.approx(0.4)


def test_near_zero_signed_readout_is_retained():
    estimate = estimate_signed_selected_output(
        {"00": 50, "10": 50, "01": 0, "11": 0}, physical_scale=3.0
    )
    assert estimate["selected_output_estimate"] == 0.0
    assert math.isfinite(estimate["analytic_standard_error"])
