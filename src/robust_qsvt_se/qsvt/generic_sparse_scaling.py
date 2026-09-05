"""Controlled compiled scaling experiments for the generic sparse-QSVT compiler."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.generic_sparse_compiler import (
    CompiledSparseQSVT,
    CompilerInputBundle,
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
    ResourceEvidence,
    StatevectorEvidence,
    build_resource_evidence,
    prepare_compiled_execution,
    validate_compiled_statevector,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import stable_array_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class ScalingStudyResult:
    dimension_rows: list[dict[str, Any]]
    slot_rows: list[dict[str, Any]]
    precision_rows: list[dict[str, Any]]
    degree_rows: list[dict[str, Any]]
    failure_rows: list[dict[str, Any]]


def _balanced_magnitude_support(
    matrix: np.ndarray, *, budget: int, slots: int
) -> tuple[tuple[int, int], ...]:
    """Choose support by matrix magnitudes alone with deterministic row/column caps."""

    candidates = sorted(
        (
            (-abs(float(matrix[i, j])), int(i), int(j))
            for i, j in zip(*np.nonzero(matrix), strict=True)
        ),
        key=lambda value: (value[0], value[1], value[2]),
    )
    row_degree = np.zeros(matrix.shape[0], dtype=int)
    column_degree = np.zeros(matrix.shape[1], dtype=int)
    selected: list[tuple[int, int]] = []
    for _, row, column in candidates:
        if row_degree[row] >= slots or column_degree[column] >= slots:
            continue
        selected.append((row, column))
        row_degree[row] += 1
        column_degree[column] += 1
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise RuntimeError(
            f"matrix-only balanced support found {len(selected)} of {budget} requested edges"
        )
    return tuple(sorted(selected))


def _scaling_execution(canonical: CompiledSparseQSVT):
    return replace(
        canonical.execution_spec,
        shot_counts=(10_000,),
        simulator_seeds=(0,),
        execute_statevector=False,
        execute_finite_shots=False,
    )


def _bundle_for_matrix(
    canonical: CompiledSparseQSVT,
    *,
    matrix: np.ndarray,
    residual: np.ndarray,
    coordinates: tuple[tuple[int, int], ...],
    matrix_id: str,
    workload_id: str,
    source: str,
    slots: int = 3,
    value_bits: int = 6,
    degree: int | None = None,
    coefficients: np.ndarray | None = None,
    phases: np.ndarray | None = None,
    polynomial_id: str | None = None,
) -> CompilerInputBundle:
    values = np.asarray(matrix, dtype=np.float64)
    selected = np.zeros_like(values)
    for coordinate in coordinates:
        selected[coordinate] = values[coordinate]
    mu = float(np.max(np.abs(selected)))
    beta = float(slots * mu)
    normalized_lambda = float(canonical.qsvt_spec.normalized_lambda)
    alpha = float(normalized_lambda * beta**2)
    resolved_degree = canonical.qsvt_spec.degree if degree is None else int(degree)
    resolved_coefficients = (
        canonical.polynomial_coefficients
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64)
    )
    resolved_phases = canonical.phases if phases is None else np.asarray(phases, dtype=np.float64)
    functional = np.zeros(values.shape[1], dtype=np.float64)
    functional[0] = 1.0
    return CompilerInputBundle(
        matrix_spec=MatrixSpec(
            values=values,
            shape=values.shape,
            matrix_id=matrix_id,
            source=source,
            workload_id=workload_id,
            expected_hash=stable_array_fingerprint(values),
            metadata={"scaling_study": True},
        ),
        support_spec=SupportSpec(
            coordinates=coordinates,
            support_id=f"{matrix_id}_support_s{slots}",
            slots=slots,
            provenance={
                "selection_inputs": "matrix values and frozen support registry only",
                "output_metrics_used": False,
            },
        ),
        quantization_spec=QuantizationSpec(
            magnitude_bits=value_bits,
            sign_representation="sign_magnitude",
            rule="sign_magnitude_round_to_nearest",
            scale=mu,
        ),
        qsvt_spec=QSVTSpec(
            alpha=alpha,
            beta=beta,
            normalized_lambda=normalized_lambda,
            boundedness_factor=canonical.qsvt_spec.boundedness_factor,
            polynomial_coefficients=resolved_coefficients,
            polynomial_id=polynomial_id or canonical.qsvt_spec.polynomial_id,
            phases=resolved_phases,
            degree=resolved_degree,
            parity="odd",
            phase_convention=canonical.qsvt_spec.phase_convention,
            require_uncomputation=True,
            bound_tolerance=2.0e-3,
            provenance={
                "scaling_rule": "fixed normalized target unless degree is varied",
                "reference_workload": canonical.workload_id,
            },
        ),
        residual_spec=ResidualSpec(
            vector=np.asarray(residual, dtype=np.float64),
            residual_id=f"{matrix_id}_frozen_residual",
            data_split="frozen_scaling_anchor",
            expected_hash=stable_array_fingerprint(np.asarray(residual, dtype=np.float64)),
        ),
        functional_spec=FunctionalSpec(
            functionals=(
                FunctionalDefinition(
                    functional_id="coordinate_e0",
                    vector=functional,
                    kind="coordinate",
                    metadata={"selection_rule": "first local coordinate"},
                ),
            ),
            primary_functional_id="coordinate_e0",
        ),
        execution_spec=_scaling_execution(canonical),
        provenance={"study": "generic_sparse_qsvt_compiler_scaling_v1"},
    )


def _resource_columns(resource: ResourceEvidence) -> dict[str, Any]:
    record = resource.record
    keys = (
        "workload_digest",
        "total_simultaneously_live_qubits",
        "register_breakdown_json",
        "transpiled_gate_count",
        "transpiled_depth",
        "toffoli_count",
        "controlled_rotation_count",
        "one_qubit_gate_count",
        "two_qubit_gate_count",
        "multi_qubit_gate_count",
        "qsvt_signal_calls",
        "phase_operations",
        "residual_loader_operations",
        "functional_loader_operations",
        "lookup_operations",
        "postselection_flag_operations",
        "postselection_measurements",
        "readout_measurements",
        "transpiler_basis",
        "optimization_level",
        "seed_transpiler",
        "opaque_instructions_remain",
    )
    return {key: record[key] for key in keys}


def _compiled_row(
    compiled: CompiledSparseQSVT,
    resource: ResourceEvidence,
    *,
    factor: str,
    level: int,
    status: str,
    statevector_executed: bool,
    finite_shot_executed: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "scaling_factor": factor,
        "level": level,
        "workload_id": compiled.workload_id,
        "matrix_shape": f"{compiled.matrix_original.shape[0]}x{compiled.matrix_original.shape[1]}",
        "matrix_hash": compiled.component_hashes["matrix_original"],
        "support_hash": compiled.component_hashes["support_mask"],
        "support_count": len(compiled.support_spec.coordinates),
        "slot_count": compiled.wrapper.slots,
        "value_magnitude_bits": compiled.quantization_spec.magnitude_bits,
        "qsvt_degree": compiled.qsvt_spec.degree,
        "evidence_status": status,
        "statevector_executed": statevector_executed,
        "finite_shot_executed": finite_shot_executed,
        "transpiled": True,
        "analytically_modeled": False,
        "source": source,
        "failure_code": "",
        "failure_reason": "",
        **_resource_columns(resource),
    }


def _failed_row(
    *,
    factor: str,
    level: int,
    slots: int,
    bits: int,
    degree: int,
    code: str,
    reason: str,
    source: str,
) -> dict[str, Any]:
    return {
        "scaling_factor": factor,
        "level": level,
        "workload_id": "",
        "matrix_shape": "8x8" if factor != "dimension" else f"{level}x{level}",
        "matrix_hash": "",
        "support_hash": "",
        "support_count": "",
        "slot_count": slots,
        "value_magnitude_bits": bits,
        "qsvt_degree": degree,
        "evidence_status": "failed",
        "statevector_executed": False,
        "finite_shot_executed": False,
        "transpiled": False,
        "analytically_modeled": False,
        "source": source,
        "failure_code": code,
        "failure_reason": reason,
    }


def _compile_and_resource(bundle: CompilerInputBundle) -> tuple[CompiledSparseQSVT, ResourceEvidence]:
    compiled = compile_from_bundle(bundle)
    prepared = prepare_compiled_execution(compiled)
    return compiled, build_resource_evidence(compiled, prepared)


def _record_failure(
    failures: list[dict[str, Any]], *, factor: str, level: int, exc: Exception
) -> tuple[str, str]:
    if isinstance(exc, SparseCompilerError):
        record = exc.to_record()
        code = str(record["code"])
        stage = str(record["stage"])
        details = json.dumps(record["details"], sort_keys=True)
    else:
        code = f"{type(exc).__name__}"
        stage = "scaling_execution"
        details = "{}"
    reason = str(exc)
    failures.append(
        {
            "workstream": "compiled_resource_scaling",
            "scaling_factor": factor,
            "level": level,
            "failure_code": code,
            "stage": stage,
            "reason": reason,
            "details_json": details,
            "retained": True,
        }
    )
    return code, reason


def run_compiled_scaling_study(
    canonical: CompiledSparseQSVT,
    canonical_resource: ResourceEvidence,
    canonical_statevector: StatevectorEvidence,
    *,
    output_root: str | Path,
) -> ScalingStudyResult:
    """Compile the four one-factor-at-a-time grids and retain failures."""

    output_path = Path(output_root)
    failures: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    precisions: list[dict[str, Any]] = []
    degrees: list[dict[str, Any]] = []

    # Dimension: frozen IEEE-14 anchors. Only the 4x4 support is newly declared,
    # by a matrix-only balanced-magnitude rule, before any QSVT output evaluation.
    matrix4 = np.load(REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_4x4.npy")
    residual4 = np.load(REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_residual_ieee14_4x4.npy")
    support4 = _balanced_magnitude_support(matrix4, budget=12, slots=3)
    bundle4 = _bundle_for_matrix(
        canonical,
        matrix=matrix4,
        residual=residual4,
        coordinates=support4,
        matrix_id="ieee14_primary_4x4_anchor",
        workload_id="ieee14_sparse_quantized_4x4_d31_scaling_anchor_v1",
        source="outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_4x4.npy",
    )
    try:
        compiled4, resource4 = _compile_and_resource(bundle4)
        validate_compiled_statevector(compiled4)
        dimensions.append(
            _compiled_row(
                compiled4,
                resource4,
                factor="dimension",
                level=4,
                status="statevector executed",
                statevector_executed=True,
                finite_shot_executed=False,
                source=bundle4.matrix_spec.source,
            )
        )
    except Exception as exc:
        code, reason = _record_failure(failures, factor="dimension", level=4, exc=exc)
        dimensions.append(
            _failed_row(
                factor="dimension", level=4, slots=3, bits=6, degree=31,
                code=code, reason=reason, source=bundle4.matrix_spec.source,
            )
        )
    dimensions.append(
        _compiled_row(
            canonical,
            canonical_resource,
            factor="dimension",
            level=8,
            status="statevector and finite-shot executed",
            statevector_executed=True,
            finite_shot_executed=True,
            source=canonical.matrix_spec.source,
        )
    )
    matrix16 = np.load(REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_block_ieee14_16x16.npy")
    residual16 = np.load(REPO_ROOT / "outputs/ieee_qsvt_pipeline_boundary/selected_residual_ieee14_16x16.npy")
    support_payload = json.loads(
        (REPO_ROOT / "outputs/cross_case_larger_block_validation/larger_block_16x16/support_paths.json").read_text(
            encoding="utf-8"
        )
    )
    support16_id = "ieee14_block_16x16_seed123_sensitivity_refined_mean_k16_s3"
    support16 = tuple(tuple(int(value) for value in pair) for pair in support_payload[support16_id])
    bundle16 = _bundle_for_matrix(
        canonical,
        matrix=matrix16,
        residual=residual16,
        coordinates=support16,
        matrix_id="ieee14_frozen_16x16_block",
        workload_id="ieee14_sparse_quantized_16x16_d31_scaling_v1",
        source="outputs/cross_case_larger_block_validation/larger_block_16x16/block_inventory.json",
    )
    try:
        compiled16, resource16 = _compile_and_resource(bundle16)
        dimensions.append(
            _compiled_row(
                compiled16,
                resource16,
                factor="dimension",
                level=16,
                status="transpiled only",
                statevector_executed=False,
                finite_shot_executed=False,
                source=bundle16.matrix_spec.source,
            )
        )
    except Exception as exc:
        code, reason = _record_failure(failures, factor="dimension", level=16, exc=exc)
        dimensions.append(
            _failed_row(
                factor="dimension", level=16, slots=3, bits=6, degree=31,
                code=code, reason=reason, source=bundle16.matrix_spec.source,
            )
        )

    # Slots: the frozen canonical support is infeasible at s=2; preserve that row.
    for slot_count in (2, 3, 4):
        if slot_count == 3:
            slots.append(
                _compiled_row(
                    canonical,
                    canonical_resource,
                    factor="slots",
                    level=slot_count,
                    status="statevector and finite-shot executed",
                    statevector_executed=True,
                    finite_shot_executed=True,
                    source="frozen canonical support",
                )
            )
            continue
        bundle = _bundle_for_matrix(
            canonical,
            matrix=canonical.matrix_original,
            residual=canonical.residual,
            coordinates=canonical.support_spec.coordinates,
            matrix_id="canonical_ieee14_slot_scaling",
            workload_id=f"ieee14_sparse_quantized_8x8_s{slot_count}_d31_scaling_v1",
            source="frozen canonical matrix and support",
            slots=slot_count,
        )
        try:
            compiled, resource = _compile_and_resource(bundle)
            slots.append(
                _compiled_row(
                    compiled,
                    resource,
                    factor="slots",
                    level=slot_count,
                    status="transpiled only",
                    statevector_executed=False,
                    finite_shot_executed=False,
                    source=bundle.matrix_spec.source,
                )
            )
        except Exception as exc:
            code, reason = _record_failure(
                failures, factor="slots", level=slot_count, exc=exc
            )
            slots.append(
                _failed_row(
                    factor="slots", level=slot_count, slots=slot_count, bits=6,
                    degree=31, code=code, reason=reason, source=bundle.matrix_spec.source,
                )
            )

    # Precision: keep support, slots, degree, phase target, scale, and alpha fixed.
    primary_functional = canonical.functional_vectors[
        canonical.functional_spec.primary_functional_id
    ]
    for bits in (4, 6, 8):
        if bits == 6:
            compiled, resource = canonical, canonical_resource
            status = "statevector and finite-shot executed"
            statevector_executed = True
            finite_shot_executed = True
        else:
            bundle = _bundle_for_matrix(
                canonical,
                matrix=canonical.matrix_original,
                residual=canonical.residual,
                coordinates=canonical.support_spec.coordinates,
                matrix_id="canonical_ieee14_precision_scaling",
                workload_id=f"ieee14_sparse_quantized_8x8_b{bits}_d31_scaling_v1",
                source="frozen canonical matrix and support",
                value_bits=bits,
            )
            try:
                compiled, resource = _compile_and_resource(bundle)
            except Exception as exc:
                code, reason = _record_failure(
                    failures, factor="value_precision", level=bits, exc=exc
                )
                precisions.append(
                    _failed_row(
                        factor="value_precision", level=bits, slots=3, bits=bits,
                        degree=31, code=code, reason=reason, source=bundle.matrix_spec.source,
                    )
                )
                continue
            status = "transpiled only"
            statevector_executed = False
            finite_shot_executed = False
        row = _compiled_row(
            compiled,
            resource,
            factor="value_precision",
            level=bits,
            status=status,
            statevector_executed=statevector_executed,
            finite_shot_executed=finite_shot_executed,
            source="frozen canonical matrix and support",
        )
        row["matrix_quantization_relative_error"] = float(
            np.linalg.norm(compiled.matrix_quantized - compiled.matrix_supported_exact)
            / np.linalg.norm(compiled.matrix_supported_exact)
        )
        quantized_update = ridge_svd_solution(
            compiled.matrix_quantized, compiled.residual, alpha=compiled.qsvt_spec.alpha
        )
        exact_update = ridge_svd_solution(
            compiled.matrix_supported_exact, compiled.residual, alpha=compiled.qsvt_spec.alpha
        )
        row["selected_output_quantization_absolute_error"] = abs(
            float(primary_functional @ quantized_update)
            - float(primary_functional @ exact_update)
        )
        row["selected_output_quantized_ridge"] = float(primary_functional @ quantized_update)
        row["selected_output_exact_supported_ridge"] = float(primary_functional @ exact_update)
        precisions.append(row)

    # Degree: synthesize only through the existing fit and PennyLane protocol.
    degree_specs: dict[int, tuple[np.ndarray, np.ndarray, float]] = {
        31: (
            canonical.polynomial_coefficients,
            canonical.phases,
            0.0006270745208463713,
        )
    }
    for degree in (15, 63):
        try:
            target = fit_codesigned_bounded_polynomial(
                beta=canonical.qsvt_spec.beta,
                alpha=canonical.qsvt_spec.alpha,
                domain_min=0.11860862497508919,
                domain_max=1.0,
                degree=degree,
                margin=1.05,
            )
            coefficients = np.asarray(target.coefficients, dtype=np.float64)
            validate_qsvt_polynomial(
                coefficients, parity="odd", bound_tolerance=2.0e-3
            )
            cached = synthesize_pennylane_phases_cached(
                coefficients,
                angle_solver="iterative",
                cache_dir=output_path / "scaling_phase_cache",
                cache_metadata={
                    "study_id": "generic_sparse_qsvt_compiler_scaling_v1",
                    "degree": degree,
                    "alpha": canonical.qsvt_spec.alpha,
                    "beta": canonical.qsvt_spec.beta,
                    "domain_min": 0.11860862497508919,
                    "domain_max": 1.0,
                    "margin": 1.05,
                },
            )
            degree_specs[degree] = (
                coefficients,
                np.asarray(cached.phases, dtype=np.float64),
                float(target.fit_max_abs_error),
            )
        except Exception as exc:
            code, reason = _record_failure(
                failures, factor="degree", level=degree, exc=exc
            )
            degrees.append(
                _failed_row(
                    factor="degree", level=degree, slots=3, bits=6, degree=degree,
                    code=code, reason=reason,
                    source="existing phase-synthesis protocol",
                )
            )
    for degree in sorted(degree_specs):
        coefficients, phases, fit_error = degree_specs[degree]
        if degree == 31:
            compiled, resource = canonical, canonical_resource
            status = "statevector and finite-shot executed"
            statevector_executed = True
            finite_shot_executed = True
        else:
            bundle = _bundle_for_matrix(
                canonical,
                matrix=canonical.matrix_original,
                residual=canonical.residual,
                coordinates=canonical.support_spec.coordinates,
                matrix_id="canonical_ieee14_degree_scaling",
                workload_id=f"ieee14_sparse_quantized_8x8_d{degree}_scaling_v1",
                source="frozen canonical matrix/support and existing phase-synthesis protocol",
                degree=degree,
                coefficients=coefficients,
                phases=phases,
                polynomial_id=f"codesigned_bounded_ridge_d{degree}",
            )
            try:
                compiled, resource = _compile_and_resource(bundle)
                validate_compiled_statevector(compiled)
            except Exception as exc:
                code, reason = _record_failure(
                    failures, factor="degree", level=degree, exc=exc
                )
                degrees.append(
                    _failed_row(
                        factor="degree", level=degree, slots=3, bits=6, degree=degree,
                        code=code, reason=reason, source=bundle.matrix_spec.source,
                    )
                )
                continue
            status = "statevector executed"
            statevector_executed = True
            finite_shot_executed = False
        row = _compiled_row(
            compiled,
            resource,
            factor="degree",
            level=degree,
            status=status,
            statevector_executed=statevector_executed,
            finite_shot_executed=finite_shot_executed,
            source="existing phase-synthesis protocol",
        )
        row["polynomial_uniform_fit_error"] = fit_error
        row["phase_count"] = len(phases)
        row["polynomial_hash"] = stable_array_fingerprint(coefficients)
        row["phase_hash"] = stable_array_fingerprint(phases)
        degrees.append(row)
    degrees.sort(key=lambda row: int(row["level"]))

    return ScalingStudyResult(dimensions, slots, precisions, degrees, failures)


__all__ = ["ScalingStudyResult", "run_compiled_scaling_study"]
