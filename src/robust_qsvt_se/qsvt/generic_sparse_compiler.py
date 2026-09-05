"""Parameter-driven compiler for small-scale sparse selected-output QSVT circuits.

The compiler contains no workload registry, IEEE case identifier, fixed support,
residual, functional, phase path, degree, slot count, value precision, or matrix
dimension. Workload adapters live in a separate module. The current reversible
architecture supports real square power-of-two matrices and rejects rectangular
inputs with a structured API-boundary error.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    CompleteWrapperResult,
    build_complete_wrapper_circuit,
    slot_values_from_assignment,
)
from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    SlotAssignment,
    assign_slot_permutations,
    minimum_slot_count,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    IntegratedSparseQSVTCircuit,
    PHASE_CONVENTION,
    _append_direct_sparse_qsvt,
    _append_postselection_flag,
    _append_sparse_qsvt_branch,
    stable_array_fingerprint,
)


@dataclass(frozen=True, slots=True)
class SparseCompilerFailure:
    """Stable machine-readable compiler failure record."""

    code: str
    stage: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "details": _json_ready(dict(self.details)),
        }


class SparseCompilerError(ValueError):
    """Exception carrying a stable failure code and structured context."""

    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.failure = SparseCompilerFailure(code, stage, message, details)

    @property
    def code(self) -> str:
        return self.failure.code

    @property
    def stage(self) -> str:
        return self.failure.stage

    def to_record(self) -> dict[str, Any]:
        return self.failure.to_record()


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    values: np.ndarray
    shape: tuple[int, int]
    matrix_id: str
    source: str = "in_memory"
    workload_id: str | None = None
    normalization_orientation: str = "transpose"
    expected_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SupportSpec:
    coordinates: tuple[tuple[int, int], ...]
    support_id: str = "in_memory_support"
    slots: int | None = None
    slot_assignment: SlotAssignment | None = None
    assignment_rule: str = "deterministic_bipartite_permutations"
    encode_transpose: bool = True
    allow_zero_source_values: bool = False
    expected_support_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    magnitude_bits: int
    sign_representation: str = "sign_magnitude"
    rule: str = "sign_magnitude_round_to_nearest"
    scale: float | None = None
    expected_quantized_hash: str | None = None


@dataclass(frozen=True, slots=True)
class QSVTSpec:
    alpha: float
    beta: float
    normalized_lambda: float
    boundedness_factor: float
    polynomial_coefficients: np.ndarray
    polynomial_id: str
    phases: np.ndarray
    degree: int
    parity: str = "odd"
    phase_convention: str = PHASE_CONVENTION
    require_uncomputation: bool = True
    bound_tolerance: float = 2.0e-3
    expected_polynomial_hash: str | None = None
    expected_phase_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResidualSpec:
    vector: np.ndarray
    residual_id: str
    preparation_convention: str = "controlled_state_preparation_normalized_input"
    data_split: str = "unspecified"
    expected_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionalDefinition:
    functional_id: str
    vector: np.ndarray
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionalSpec:
    functionals: tuple[FunctionalDefinition, ...]
    primary_functional_id: str
    postselection_convention: str = "flag_zero_iff_sparse_work_zero"
    readout_convention: str = "real_branch_hadamard_c1c0"
    unavailable_requests: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    shot_counts: tuple[int, ...] = (10_000, 100_000, 1_000_000)
    simulator_seeds: tuple[int, ...] = tuple(range(10))
    simulator_method: str = "statevector"
    basis_gates: tuple[str, ...] | None = None
    optimization_level: int = 0
    seed_transpiler: int | None = None
    execute_statevector: bool = True
    execute_finite_shots: bool = True
    max_parallel_threads: int = 1
    register_override: tuple[tuple[str, tuple[int, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class CompilerInputBundle:
    matrix_spec: MatrixSpec
    support_spec: SupportSpec
    quantization_spec: QuantizationSpec
    qsvt_spec: QSVTSpec
    residual_spec: ResidualSpec
    functional_spec: FunctionalSpec
    execution_spec: ExecutionSpec
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompiledSparseQSVT:
    workload_id: str
    workload_digest: str
    validated_input_metadata: dict[str, Any]
    padded_dimensions: dict[str, int]
    register_allocation: dict[str, Any]
    sparse_index_lookup: dict[str, Any]
    sparse_value_sign_lookup: dict[str, Any]
    value_controlled_rotations: tuple[dict[str, Any], ...]
    slot_controlled_permutations: tuple[dict[str, Any], ...]
    inverse_uncomputation_path: dict[str, Any]
    wrapper: CompleteWrapperResult
    sparse_block_encoding_wrapper: Any
    qsvt_sequence: dict[str, Any]
    postselection_logic: dict[str, Any]
    signed_readout_logic: dict[str, Any]
    functional_circuits: dict[str, IntegratedSparseQSVTCircuit]
    final_measured_circuit: Any
    direct_postselection_circuit: Any
    recovery_factors: dict[str, float]
    component_hashes: dict[str, str]
    failure_records: tuple[SparseCompilerFailure, ...]
    matrix_original: np.ndarray
    matrix_supported_exact: np.ndarray
    matrix_quantized: np.ndarray
    residual: np.ndarray
    functional_vectors: dict[str, np.ndarray]
    polynomial_coefficients: np.ndarray
    phases: np.ndarray
    matrix_spec: MatrixSpec
    support_spec: SupportSpec
    quantization_spec: QuantizationSpec
    qsvt_spec: QSVTSpec
    residual_spec: ResidualSpec
    functional_spec: FunctionalSpec
    execution_spec: ExecutionSpec


def _raise(code: str, stage: str, message: str, **details: Any) -> None:
    raise SparseCompilerError(code, stage, message, **details)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_real_array(name: str, values: np.ndarray, *, stage: str) -> np.ndarray:
    source = np.asarray(values)
    if np.iscomplexobj(source):
        imaginary = float(np.max(np.abs(np.imag(source)), initial=0.0))
        if imaginary > 1.0e-14:
            code = "unsupported_complex_matrix" if name == "matrix" else "non_finite_values"
            _raise(code, stage, f"{name} must be real", max_imaginary=imaginary)
    result = np.asarray(np.real(source), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        _raise("non_finite_values", stage, f"{name} contains non-finite values")
    return result


def _validate_matrix(spec: MatrixSpec) -> np.ndarray:
    matrix = _as_real_array("matrix", spec.values, stage="matrix_validation")
    if matrix.ndim != 2 or matrix.size == 0:
        _raise("invalid_shape", "matrix_validation", "matrix must be a nonempty 2D array")
    if tuple(int(value) for value in matrix.shape) != tuple(spec.shape):
        _raise(
            "invalid_shape",
            "matrix_validation",
            "matrix shape disagrees with the declared shape",
            actual=matrix.shape,
            declared=spec.shape,
        )
    rows, columns = matrix.shape
    if rows != columns:
        _raise(
            "unsupported_rectangular_orientation",
            "matrix_validation",
            "the in-place sparse wrapper supports square matrices only",
            shape=matrix.shape,
        )
    if rows & (rows - 1):
        _raise(
            "unsupported_dimension",
            "matrix_validation",
            "matrix dimension must be a power of two",
            dimension=rows,
        )
    if spec.normalization_orientation != "transpose":
        _raise(
            "unsupported_rectangular_orientation",
            "matrix_validation",
            "only the H-transpose sparse encoding orientation is supported",
            orientation=spec.normalization_orientation,
        )
    if not spec.matrix_id.strip():
        _raise("invalid_shape", "matrix_validation", "matrix_id must be nonempty")
    fingerprint = stable_array_fingerprint(matrix)
    if spec.expected_hash is not None and fingerprint != spec.expected_hash:
        _raise(
            "matrix_hash_mismatch",
            "matrix_validation",
            "matrix hash does not match the frozen specification",
            expected=spec.expected_hash,
            actual=fingerprint,
        )
    return matrix


def _validate_support(matrix: np.ndarray, spec: SupportSpec) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows, columns = matrix.shape
    coordinates = [(int(i), int(j)) for i, j in spec.coordinates]
    if len(set(coordinates)) != len(coordinates):
        duplicates = sorted({coord for coord in coordinates if coordinates.count(coord) > 1})
        _raise(
            "duplicate_support_coordinates",
            "support_validation",
            "support contains duplicate coordinates",
            duplicates=duplicates,
        )
    if not coordinates:
        _raise("invalid_shape", "support_validation", "support cannot be empty")
    outside = [coord for coord in coordinates if not (0 <= coord[0] < rows and 0 <= coord[1] < columns)]
    if outside:
        _raise(
            "support_out_of_bounds",
            "support_validation",
            "support contains coordinates outside the matrix",
            coordinates=outside,
            shape=matrix.shape,
        )
    zero_source = [coord for coord in coordinates if matrix[coord] == 0.0]
    if zero_source and not spec.allow_zero_source_values:
        _raise(
            "support_value_missing",
            "support_validation",
            "support contains zero-valued source coordinates",
            coordinates=zero_source,
        )
    mask = np.zeros(matrix.shape, dtype=bool)
    for coordinate in coordinates:
        mask[coordinate] = True
    support_hash = stable_array_fingerprint(mask.astype(np.float64))
    if spec.expected_support_hash is not None and support_hash != spec.expected_support_hash:
        _raise(
            "support_hash_mismatch",
            "support_validation",
            "support hash does not match the frozen specification",
            expected=spec.expected_support_hash,
            actual=support_hash,
        )
    return mask, sorted(coordinates)


def _quantize(
    matrix: np.ndarray,
    support_mask: np.ndarray,
    spec: QuantizationSpec,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if spec.sign_representation != "sign_magnitude":
        _raise(
            "unsupported_quantization",
            "quantization",
            "only sign-magnitude representation is supported",
            sign_representation=spec.sign_representation,
        )
    if not isinstance(spec.magnitude_bits, int) or not 1 <= spec.magnitude_bits <= 16:
        _raise(
            "unsupported_quantization",
            "quantization",
            "magnitude_bits must be an integer from 1 through 16",
            magnitude_bits=spec.magnitude_bits,
        )
    supported = np.where(support_mask, matrix, 0.0)
    maximum = float(np.max(np.abs(supported)))
    if maximum <= 0.0:
        _raise("unsupported_quantization", "quantization", "supported matrix is identically zero")
    scale = maximum if spec.scale is None else float(spec.scale)
    if not math.isfinite(scale) or scale <= 0.0 or scale + 1.0e-12 < maximum:
        _raise(
            "unsupported_quantization",
            "quantization",
            "quantization scale must be finite, positive, and cover all supported values",
            scale=scale,
            maximum=maximum,
        )
    if spec.rule == "prequantized":
        quantized = supported.copy()
    elif spec.rule == "sign_magnitude_round_to_nearest":
        levels = (1 << spec.magnitude_bits) - 1
        codes = np.round(np.abs(supported) / scale * levels)
        quantized = np.sign(supported) * codes / levels * scale
    else:
        _raise(
            "unsupported_quantization",
            "quantization",
            "unsupported quantization rule",
            rule=spec.rule,
        )
    fingerprint = stable_array_fingerprint(quantized)
    if spec.expected_quantized_hash is not None and fingerprint != spec.expected_quantized_hash:
        _raise(
            "quantized_matrix_hash_mismatch",
            "quantization",
            "quantized matrix hash does not match the frozen specification",
            expected=spec.expected_quantized_hash,
            actual=fingerprint,
        )
    max_error = float(np.max(np.abs(quantized - supported)))
    return supported, quantized, scale, max_error


def _validate_qsvt(spec: QSVTSpec, *, slots: int, mu: float) -> tuple[np.ndarray, np.ndarray]:
    scalars = {
        "alpha": spec.alpha,
        "beta": spec.beta,
        "normalized_lambda": spec.normalized_lambda,
        "boundedness_factor": spec.boundedness_factor,
        "bound_tolerance": spec.bound_tolerance,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        _raise("non_finite_values", "qsvt_validation", "QSVT scalars must be finite", **scalars)
    if spec.alpha <= 0.0 or spec.beta <= 0.0 or spec.boundedness_factor <= 0.0:
        _raise("invalid_polynomial", "qsvt_validation", "alpha, beta, and C must be positive")
    expected_beta = float(slots * mu)
    if not math.isclose(spec.beta, expected_beta, rel_tol=1.0e-12, abs_tol=1.0e-12):
        _raise(
            "normalization_mismatch",
            "qsvt_validation",
            "beta must equal slots times the value scale",
            expected=expected_beta,
            actual=spec.beta,
        )
    expected_lambda = spec.alpha / spec.beta**2
    if not math.isclose(spec.normalized_lambda, expected_lambda, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _raise(
            "normalization_mismatch",
            "qsvt_validation",
            "normalized lambda must equal alpha divided by beta squared",
            expected=expected_lambda,
            actual=spec.normalized_lambda,
        )
    if not spec.require_uncomputation:
        _raise(
            "missing_uncomputation",
            "qsvt_validation",
            "the sparse wrapper inverse and slot-diffusion inverse are mandatory",
        )
    if spec.phase_convention != PHASE_CONVENTION:
        _raise(
            "unsupported_phase_convention",
            "qsvt_validation",
            "unsupported QSVT phase convention",
            convention=spec.phase_convention,
        )
    if spec.parity != "odd" or spec.degree <= 0 or spec.degree % 2 == 0:
        _raise(
            "degree_parity_mismatch",
            "qsvt_validation",
            "the verified convention requires positive odd degree and odd parity",
            degree=spec.degree,
            parity=spec.parity,
        )
    coefficients = _as_real_array(
        "polynomial coefficients", spec.polynomial_coefficients, stage="qsvt_validation"
    )
    phases = _as_real_array("phase sequence", spec.phases, stage="qsvt_validation")
    if coefficients.ndim != 1 or coefficients.size != spec.degree + 1:
        _raise(
            "degree_parity_mismatch",
            "qsvt_validation",
            "coefficient count must equal degree plus one",
            coefficient_count=coefficients.size,
            degree=spec.degree,
        )
    if phases.ndim != 1 or phases.size != spec.degree + 1:
        _raise(
            "invalid_phase_count",
            "qsvt_validation",
            "phase count must equal degree plus one",
            phase_count=phases.size,
            degree=spec.degree,
        )
    try:
        validate_qsvt_polynomial(
            coefficients, parity="odd", bound_tolerance=float(spec.bound_tolerance)
        )
    except Exception as exc:
        _raise("invalid_polynomial", "qsvt_validation", str(exc))
    coefficient_hash = stable_array_fingerprint(coefficients)
    phase_hash = stable_array_fingerprint(phases)
    if spec.expected_polynomial_hash is not None and coefficient_hash != spec.expected_polynomial_hash:
        _raise(
            "polynomial_hash_mismatch",
            "qsvt_validation",
            "polynomial hash does not match the frozen specification",
            expected=spec.expected_polynomial_hash,
            actual=coefficient_hash,
        )
    if spec.expected_phase_hash is not None and phase_hash != spec.expected_phase_hash:
        _raise(
            "phase_hash_mismatch",
            "qsvt_validation",
            "phase hash does not match the frozen specification",
            expected=spec.expected_phase_hash,
            actual=phase_hash,
        )
    return coefficients, phases


def _validate_residual(spec: ResidualSpec, rows: int) -> np.ndarray:
    residual = _as_real_array("residual", spec.vector, stage="residual_validation")
    if residual.ndim != 1 or residual.size != rows:
        _raise(
            "invalid_shape",
            "residual_validation",
            "residual length must match matrix rows",
            residual_shape=residual.shape,
            rows=rows,
        )
    norm = float(np.linalg.norm(residual))
    if norm <= 1.0e-15:
        _raise("zero_residual_norm", "residual_validation", "residual norm is zero")
    fingerprint = stable_array_fingerprint(residual)
    if spec.expected_hash is not None and fingerprint != spec.expected_hash:
        _raise(
            "residual_hash_mismatch",
            "residual_validation",
            "residual hash does not match the frozen specification",
            expected=spec.expected_hash,
            actual=fingerprint,
        )
    return residual


def _validate_functionals(spec: FunctionalSpec, columns: int) -> dict[str, np.ndarray]:
    if not spec.functionals:
        _raise("incompatible_functional_dimension", "functional_validation", "at least one functional is required")
    result: dict[str, np.ndarray] = {}
    for definition in spec.functionals:
        if not definition.functional_id.strip() or definition.functional_id in result:
            _raise(
                "incompatible_functional_dimension",
                "functional_validation",
                "functional identifiers must be nonempty and unique",
                functional_id=definition.functional_id,
            )
        vector = _as_real_array(
            f"functional {definition.functional_id}",
            definition.vector,
            stage="functional_validation",
        )
        if vector.ndim != 1 or vector.size != columns:
            _raise(
                "incompatible_functional_dimension",
                "functional_validation",
                "functional length must match matrix columns",
                functional_id=definition.functional_id,
                functional_shape=vector.shape,
                columns=columns,
            )
        if float(np.linalg.norm(vector)) <= 1.0e-15:
            _raise(
                "zero_functional_norm",
                "functional_validation",
                "functional norm is zero",
                functional_id=definition.functional_id,
            )
        result[definition.functional_id] = vector
    if spec.primary_functional_id not in result:
        _raise(
            "incompatible_functional_dimension",
            "functional_validation",
            "primary functional is not present in the functional registry",
            primary_functional_id=spec.primary_functional_id,
        )
    if spec.postselection_convention != "flag_zero_iff_sparse_work_zero":
        _raise(
            "incompatible_functional_dimension",
            "functional_validation",
            "unsupported postselection convention",
            convention=spec.postselection_convention,
        )
    if spec.readout_convention != "real_branch_hadamard_c1c0":
        _raise(
            "incompatible_functional_dimension",
            "functional_validation",
            "unsupported readout convention",
            convention=spec.readout_convention,
        )
    return result


def _validate_execution(spec: ExecutionSpec) -> None:
    if not spec.shot_counts or any(int(value) <= 0 for value in spec.shot_counts):
        _raise("invalid_execution", "execution_validation", "shot counts must be positive")
    if not spec.simulator_seeds or len(set(spec.simulator_seeds)) != len(spec.simulator_seeds):
        _raise("invalid_execution", "execution_validation", "simulator seeds must be nonempty and unique")
    if spec.optimization_level not in {0, 1, 2, 3}:
        _raise("invalid_execution", "execution_validation", "optimization level must be 0, 1, 2, or 3")
    if spec.simulator_method != "statevector":
        _raise("invalid_execution", "execution_validation", "only Aer statevector method is supported")
    if spec.max_parallel_threads <= 0:
        _raise("invalid_execution", "execution_validation", "max_parallel_threads must be positive")


def _assignment_for_support(
    encoded_support: np.ndarray,
    spec: SupportSpec,
) -> SlotAssignment:
    required = minimum_slot_count(encoded_support)
    declared = required if spec.slots is None else int(spec.slots)
    if declared < required:
        _raise(
            "slot_overflow",
            "slot_assignment",
            "declared slot count is below the support maximum degree",
            declared_slots=declared,
            required_slots=required,
        )
    try:
        if spec.slot_assignment is None:
            assignment = assign_slot_permutations(encoded_support, slots=declared)
        else:
            assignment = spec.slot_assignment
            if assignment.slots != declared:
                _raise(
                    "inconsistent_slot_assignment",
                    "slot_assignment",
                    "explicit assignment slot count disagrees with SupportSpec",
                    assignment_slots=assignment.slots,
                    declared_slots=declared,
                )
        validate_slot_assignment(encoded_support, assignment)
    except SparseCompilerError:
        raise
    except ValueError as exc:
        _raise("inconsistent_slot_assignment", "slot_assignment", str(exc))
    except RuntimeError as exc:
        _raise("inconsistent_slot_assignment", "slot_assignment", str(exc))
    return assignment


def _build_validated_wrapper(
    matrix_quantized: np.ndarray,
    support_mask: np.ndarray,
    mu: float,
    beta: float,
    assignment: SlotAssignment,
    *,
    block_tolerance: float = 1.0e-9,
    unitarity_tolerance: float = 1.0e-9,
) -> tuple[CompleteWrapperResult, np.ndarray, tuple[dict[str, Any], ...]]:
    from qiskit.quantum_info import Operator, Statevector

    encoded_matrix = matrix_quantized.T
    encoded_support = support_mask.T
    try:
        validate_slot_assignment(encoded_support, assignment)
        circuit, diffusion = build_complete_wrapper_circuit(encoded_matrix, mu, assignment)
        unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    except Exception as exc:
        _raise(
            "construction_failure",
            "wrapper_construction",
            f"failed to construct the sparse wrapper: {type(exc).__name__}: {exc}",
        )
    n = encoded_matrix.shape[0]
    target = encoded_matrix / beta
    # The verified signal convention is the real top-left block.  Preserve the
    # historical wrapper metric by discarding numerical imaginary roundoff here;
    # full-unitary validation below still uses the unmodified complex operator.
    encoded = np.asarray(np.real(unitary[:n, :n]), dtype=np.float64)
    reconstruction_max = float(np.max(np.abs(encoded - target)))
    reconstruction_relative = float(
        np.linalg.norm(encoded - target, ord="fro")
        / max(np.linalg.norm(target, ord="fro"), 1.0e-30)
    )
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    unitarity_error = float(np.max(np.abs(unitary.conj().T @ unitary - identity)))
    diffusion_error = float(
        np.max(np.abs(diffusion.T @ diffusion - np.eye(diffusion.shape[0])))
    )
    slot_values = slot_values_from_assignment(encoded_matrix, mu, assignment)
    lookup_error = 0.0
    rotations: list[dict[str, Any]] = []
    for slot, (permutation, real_mask) in enumerate(
        zip(assignment.permutations, assignment.real_edge_mask, strict=True)
    ):
        for encoded_column in range(n):
            encoded_row = int(permutation[encoded_column])
            expected = (
                float(encoded_matrix[encoded_row, encoded_column] / mu)
                if real_mask[encoded_column]
                else 0.0
            )
            actual = float(slot_values[slot, encoded_column])
            lookup_error = max(lookup_error, abs(actual - expected))
            rotations.append(
                {
                    "slot": slot,
                    "encoded_column": encoded_column,
                    "encoded_row": encoded_row,
                    "original_coordinate": [encoded_column, encoded_row],
                    "is_declared_support_edge": bool(real_mask[encoded_column]),
                    "normalized_signed_value": actual,
                    "sign": int(np.sign(actual)),
                    "rotation_angle": 2.0 * math.acos(float(np.clip(actual, -1.0, 1.0))),
                }
            )
    if lookup_error > 1.0e-14:
        _raise(
            "lookup_validation_failure",
            "wrapper_validation",
            "slot value lookup disagrees with the quantized matrix",
            error=lookup_error,
        )
    if reconstruction_relative > block_tolerance:
        _raise(
            "block_reconstruction_failure",
            "wrapper_validation",
            "sparse wrapper top block exceeds tolerance",
            error=reconstruction_relative,
            tolerance=block_tolerance,
        )
    if unitarity_error > unitarity_tolerance:
        _raise(
            "wrapper_unitarity_failure",
            "wrapper_validation",
            "sparse wrapper is not unitary within tolerance",
            error=unitarity_error,
            tolerance=unitarity_tolerance,
        )
    statevector_max_error = 0.0
    for column in range(n):
        initial = np.zeros(unitary.shape[0], dtype=np.complex128)
        initial[column] = 1.0
        evolved = Statevector(initial).evolve(circuit).data
        statevector_max_error = max(
            statevector_max_error,
            float(np.max(np.abs(evolved[:n] - target[:, column]))),
        )
    wrapper = CompleteWrapperResult(
        circuit=circuit,
        unitary=unitary,
        encoded_block=encoded,
        target_block=target,
        assignment=assignment,
        slots=assignment.slots,
        normalization_factor=beta,
        top_left_reconstruction_error=reconstruction_max,
        unitarity_error=unitarity_error,
        statevector_max_error=statevector_max_error,
        diffusion_unitarity_error=diffusion_error,
        lookup_value_max_error=lookup_error,
        qubits=int(circuit.num_qubits),
        gate_count=int(sum(circuit.count_ops().values())),
        depth=int(circuit.depth()),
        transpiled_gate_count=None,
        transpiled_depth=None,
        transpiled_cx_count=None,
        transpile_failure=None,
    )
    return wrapper, slot_values, tuple(rotations)


def _standard_register_allocation(n: int, wrapper_qubits: int) -> dict[str, Any]:
    index_count = int(math.log2(n))
    slot_count = wrapper_qubits - index_count - 1
    index = list(range(index_count))
    slot = list(range(index_count, index_count + slot_count))
    rotation = index_count + slot_count
    postselection = wrapper_qubits
    readout = wrapper_qubits + 1
    return {
        "index": index,
        "slot": slot,
        "rotation_ancilla": [rotation],
        "postselection_flag": [postselection],
        "signed_readout": [readout],
        "classical_postselection_bit": 0,
        "classical_readout_bit": 1,
        "total_logical_qubits": wrapper_qubits + 2,
    }


def _validate_registers(allocation: dict[str, Any], spec: ExecutionSpec) -> None:
    quantum_names = (
        "index",
        "slot",
        "rotation_ancilla",
        "postselection_flag",
        "signed_readout",
    )
    selected = {name: tuple(int(q) for q in allocation[name]) for name in quantum_names}
    if spec.register_override:
        override = {name: tuple(int(q) for q in qubits) for name, qubits in spec.register_override}
        flattened_override = [q for qubits in override.values() for q in qubits]
        if len(flattened_override) != len(set(flattened_override)):
            _raise(
                "register_collision",
                "register_allocation",
                "register override assigns a qubit to more than one live register",
                override=override,
            )
        if override != selected:
            _raise(
                "register_layout_mismatch",
                "register_allocation",
                "the current circuit builder accepts only its deterministic contiguous allocation",
                requested=override,
                required=selected,
            )
    flattened = [q for name in quantum_names for q in selected[name]]
    if len(flattened) != len(set(flattened)):
        _raise(
            "register_collision",
            "register_allocation",
            "compiler-generated live registers collide",
            allocation=selected,
        )


def _build_functional_circuit(
    *,
    wrapper: CompleteWrapperResult,
    residual: np.ndarray,
    functional: np.ndarray,
    phases: np.ndarray,
) -> IntegratedSparseQSVTCircuit:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    n = residual.size
    index_count = int(math.log2(n))
    chain_count = int(wrapper.circuit.num_qubits)
    slot_count = chain_count - index_count - 1
    index = list(range(index_count))
    slot = list(range(index_count, index_count + slot_count))
    rotation_ancilla = index_count + slot_count
    work = [*slot, rotation_ancilla]
    chain = list(range(chain_count))
    postselection_flag = chain_count
    readout = chain_count + 1

    circuit = QuantumCircuit(chain_count + 2, 2, name="sparse_integrated_qsvt_readout")
    circuit.h(readout)
    residual_unit = residual / np.linalg.norm(residual)
    residual_prep = StatePreparation(residual_unit).control(
        1, ctrl_state=0, annotated=False
    )
    residual_prep.label = "c0_residual_prep"
    circuit.append(residual_prep, [readout, *index])

    wrapper_gate = wrapper.circuit.to_gate(label="sparse_U_A")
    accounting = _append_sparse_qsvt_branch(
        circuit,
        wrapper_gate=wrapper_gate,
        phases=phases,
        chain_qubits=chain,
        work_qubits=work,
        readout_qubit=readout,
    )
    functional_unit = functional / np.linalg.norm(functional)
    functional_prep = StatePreparation(functional_unit).control(
        1, ctrl_state=1, annotated=False
    )
    functional_prep.label = "c1_functional_prep"
    circuit.append(functional_prep, [readout, *index])
    _append_postselection_flag(circuit, work, postselection_flag)
    circuit.h(readout)
    circuit.measure(postselection_flag, 0)
    circuit.measure(readout, 1)

    direct = QuantumCircuit(chain_count + 1, 1, name="sparse_direct_postselection")
    direct.append(StatePreparation(residual_unit), index)
    direct_wrapper_gate = wrapper.circuit.to_gate(label="sparse_U_A")
    _append_direct_sparse_qsvt(
        direct,
        wrapper_gate=direct_wrapper_gate,
        phases=phases,
        chain_qubits=chain,
        work_qubits=work,
    )
    _append_postselection_flag(direct, work, postselection_flag)
    direct.measure(postselection_flag, 0)

    operation_counts = {
        **accounting,
        "sparse_lookup_calls_per_attempt": accounting["signal_unitary_calls_per_attempt"],
        "value_rotation_gates_per_sparse_lookup": int(n * wrapper.slots),
        "value_rotations_per_attempt": int(
            n * wrapper.slots * accounting["signal_unitary_calls_per_attempt"]
        ),
        "residual_preparations_per_attempt": 1,
        "functional_preparations_per_attempt": 1,
        "postselection_measurements_per_attempt": 1,
        "readout_measurements_per_attempt": 1,
    }
    return IntegratedSparseQSVTCircuit(
        circuit=circuit,
        direct_postselection_circuit=direct,
        wrapper=wrapper,
        register_layout={
            "index_qubits": index,
            "slot_qubits": slot,
            "rotation_ancilla_qubit": rotation_ancilla,
            "sparse_work_qubits": work,
            "postselection_flag_qubit": postselection_flag,
            "readout_qubit": readout,
            "classical_postselection_bit": 0,
            "classical_readout_bit": 1,
            "encoded_subspace": "all slot and rotation-work qubits are zero",
        },
        operation_counts=operation_counts,
        preparation_records=(
            {
                "role": "residual_input",
                "fingerprint": stable_array_fingerprint(residual_unit),
            },
            {
                "role": "selected_functional_reference",
                "fingerprint": stable_array_fingerprint(functional_unit),
            },
        ),
        output_state_used_for_preparation=False,
    )


def compile_sparse_selected_output_qsvt(
    matrix_spec: MatrixSpec,
    support_spec: SupportSpec,
    quantization_spec: QuantizationSpec,
    qsvt_spec: QSVTSpec,
    residual_spec: ResidualSpec,
    functional_spec: FunctionalSpec,
    execution_spec: ExecutionSpec,
) -> CompiledSparseQSVT:
    """Validate and construct a complete sparse selected-output QSVT circuit.

    This function performs no statevector, transpilation, or shot execution.
    """

    matrix = _validate_matrix(matrix_spec)
    support_mask, coordinates = _validate_support(matrix, support_spec)
    supported, quantized, mu, quantization_error = _quantize(
        matrix, support_mask, quantization_spec
    )
    encoded_support = support_mask.T if support_spec.encode_transpose else support_mask
    if not support_spec.encode_transpose:
        _raise(
            "unsupported_rectangular_orientation",
            "support_validation",
            "the verified QSVT path requires transpose encoding",
        )
    assignment = _assignment_for_support(encoded_support, support_spec)
    coefficients, phases = _validate_qsvt(qsvt_spec, slots=assignment.slots, mu=mu)
    residual = _validate_residual(residual_spec, matrix.shape[0])
    functionals = _validate_functionals(functional_spec, matrix.shape[1])
    _validate_execution(execution_spec)

    wrapper, slot_values, rotations = _build_validated_wrapper(
        quantized,
        support_mask,
        mu,
        qsvt_spec.beta,
        assignment,
    )
    allocation = _standard_register_allocation(matrix.shape[0], wrapper.qubits)
    _validate_registers(allocation, execution_spec)

    circuits: dict[str, IntegratedSparseQSVTCircuit] = {}
    try:
        for functional_id, vector in functionals.items():
            circuits[functional_id] = _build_functional_circuit(
                wrapper=wrapper,
                residual=residual,
                functional=vector,
                phases=phases,
            )
    except SparseCompilerError:
        raise
    except Exception as exc:
        _raise(
            "construction_failure",
            "integrated_circuit_construction",
            f"failed to construct the integrated circuit: {type(exc).__name__}: {exc}",
        )

    primary = circuits[functional_spec.primary_functional_id]
    sequence_counts = qsvt_sequence_operation_counts(phases.size)
    component_hashes = {
        "matrix_original": stable_array_fingerprint(matrix),
        "support_mask": stable_array_fingerprint(support_mask.astype(np.float64)),
        "matrix_supported_exact": stable_array_fingerprint(supported),
        "matrix_quantized": stable_array_fingerprint(quantized),
        "slot_assignment": _canonical_json_hash(assignment.to_metadata()),
        "slot_values": stable_array_fingerprint(slot_values),
        "residual": stable_array_fingerprint(residual),
        "polynomial_coefficients": stable_array_fingerprint(coefficients),
        "phases": stable_array_fingerprint(phases),
        "execution_settings": _canonical_json_hash(
            {
                "shot_counts": execution_spec.shot_counts,
                "simulator_seeds": execution_spec.simulator_seeds,
                "simulator_method": execution_spec.simulator_method,
                "basis_gates": execution_spec.basis_gates,
                "optimization_level": execution_spec.optimization_level,
                "seed_transpiler": execution_spec.seed_transpiler,
            }
        ),
    }
    component_hashes["sparse_wrapper_circuit"] = _canonical_json_hash(
        {
            "matrix_quantized": component_hashes["matrix_quantized"],
            "slot_assignment": component_hashes["slot_assignment"],
            "slot_values": component_hashes["slot_values"],
            "normalization": qsvt_spec.beta,
            "qubits": wrapper.qubits,
            "gate_count": wrapper.gate_count,
            "depth": wrapper.depth,
        }
    )
    for functional_id, vector in functionals.items():
        component_hashes[f"functional:{functional_id}"] = stable_array_fingerprint(vector)
        component_hashes[f"source_final_circuit:{functional_id}"] = _canonical_json_hash(
            {
                "architecture": "sparse_integrated_qsvt_readout",
                "wrapper": component_hashes["sparse_wrapper_circuit"],
                "residual": component_hashes["residual"],
                "functional": component_hashes[f"functional:{functional_id}"],
                "phases": component_hashes["phases"],
                "register_allocation": allocation,
                "postselection": functional_spec.postselection_convention,
                "readout": functional_spec.readout_convention,
                "operation_counts": circuits[functional_id].operation_counts,
            }
        )
    component_hashes["direct_postselection_circuit"] = _canonical_json_hash(
        {
            "architecture": "sparse_direct_postselection",
            "wrapper": component_hashes["sparse_wrapper_circuit"],
            "residual": component_hashes["residual"],
            "phases": component_hashes["phases"],
            "register_allocation": allocation,
            "postselection": functional_spec.postselection_convention,
        }
    )
    workload_payload = {
        "matrix_id": matrix_spec.matrix_id,
        "matrix_hash": component_hashes["matrix_original"],
        "support_id": support_spec.support_id,
        "support_hash": component_hashes["support_mask"],
        "quantized_hash": component_hashes["matrix_quantized"],
        "assignment_hash": component_hashes["slot_assignment"],
        "residual_id": residual_spec.residual_id,
        "residual_hash": component_hashes["residual"],
        "functional_hashes": {
            key: value for key, value in component_hashes.items() if key.startswith("functional:")
        },
        "alpha": qsvt_spec.alpha,
        "beta": qsvt_spec.beta,
        "lambda": qsvt_spec.normalized_lambda,
        "C": qsvt_spec.boundedness_factor,
        "degree": qsvt_spec.degree,
        "polynomial_hash": component_hashes["polynomial_coefficients"],
        "phase_hash": component_hashes["phases"],
        "phase_convention": qsvt_spec.phase_convention,
        "postselection": functional_spec.postselection_convention,
        "readout": functional_spec.readout_convention,
        "execution_hash": component_hashes["execution_settings"],
    }
    workload_digest = _canonical_json_hash(workload_payload)
    workload_id = (
        matrix_spec.workload_id
        if matrix_spec.workload_id is not None
        else f"sparse_qsvt_{workload_digest[:20]}"
    )
    recovery = {
        functional_id: float(
            qsvt_spec.boundedness_factor
            / qsvt_spec.beta
            * np.linalg.norm(residual)
            * np.linalg.norm(vector)
        )
        for functional_id, vector in functionals.items()
    }
    permutations = tuple(
        {
            "slot": slot,
            "permutation": list(permutation),
            "real_edge_mask": list(real_mask),
        }
        for slot, (permutation, real_mask) in enumerate(
            zip(assignment.permutations, assignment.real_edge_mask, strict=True)
        )
    )
    metadata = {
        "workload_id": workload_id,
        "workload_digest": workload_digest,
        "matrix_id": matrix_spec.matrix_id,
        "matrix_source": matrix_spec.source,
        "matrix_shape": list(matrix.shape),
        "support_id": support_spec.support_id,
        "support_coordinates": [list(coord) for coord in coordinates],
        "support_nonzeros": len(coordinates),
        "quantized_nonzeros": int(np.count_nonzero(quantized)),
        "slots": assignment.slots,
        "minimum_slots": assignment.max_degree,
        "mu": mu,
        "magnitude_bits": quantization_spec.magnitude_bits,
        "sign_representation": quantization_spec.sign_representation,
        "quantization_rule": quantization_spec.rule,
        "max_quantization_error": quantization_error,
        "alpha": qsvt_spec.alpha,
        "beta": qsvt_spec.beta,
        "normalized_lambda": qsvt_spec.normalized_lambda,
        "C": qsvt_spec.boundedness_factor,
        "degree": qsvt_spec.degree,
        "phase_count": int(phases.size),
        "polynomial_id": qsvt_spec.polynomial_id,
        "phase_convention": qsvt_spec.phase_convention,
        "residual_id": residual_spec.residual_id,
        "residual_data_split": residual_spec.data_split,
        "primary_functional_id": functional_spec.primary_functional_id,
        "available_functional_ids": list(functionals),
        "unavailable_functional_requests": _json_ready(functional_spec.unavailable_requests),
        "postselection_convention": functional_spec.postselection_convention,
        "readout_convention": functional_spec.readout_convention,
        "dense_fallback_used": False,
        "direct_output_state_preparation_used": False,
        "construction_only": True,
    }
    slot_qubits = len(allocation["slot"])
    padded = {
        "matrix_rows": matrix.shape[0],
        "matrix_columns": matrix.shape[1],
        "index_dimension": 1 << len(allocation["index"]),
        "slot_dimension": 1 << slot_qubits,
        "declared_slots": assignment.slots,
        "wrapper_hilbert_dimension": 1 << wrapper.qubits,
    }
    qsvt_sequence = {
        "degree": qsvt_spec.degree,
        "phase_count": int(phases.size),
        "phase_convention": qsvt_spec.phase_convention,
        "signal_unitary_calls": int(sequence_counts["signal_unitary_calls"]),
        "forward_wrapper_calls": primary.operation_counts[
            "forward_sparse_lookup_calls_per_attempt"
        ],
        "inverse_wrapper_calls": primary.operation_counts["inverse_lookup_calls_per_attempt"],
        "projector_phase_operations": primary.operation_counts[
            "projector_phase_operations_per_attempt"
        ],
        "phases_hash": component_hashes["phases"],
        "polynomial_hash": component_hashes["polynomial_coefficients"],
        "uncomputation_required": True,
    }
    return CompiledSparseQSVT(
        workload_id=workload_id,
        workload_digest=workload_digest,
        validated_input_metadata=metadata,
        padded_dimensions=padded,
        register_allocation=allocation,
        sparse_index_lookup={
            "assignment_rule": support_spec.assignment_rule,
            "assignment": assignment.to_metadata(),
            "support_hash": component_hashes["support_mask"],
            "exact_support_coverage": True,
        },
        sparse_value_sign_lookup={
            "slot_values": slot_values.tolist(),
            "value_scale_mu": mu,
            "sign_representation": quantization_spec.sign_representation,
            "magnitude_bits": quantization_spec.magnitude_bits,
            "lookup_value_max_error": wrapper.lookup_value_max_error,
            "separate_value_register": False,
        },
        value_controlled_rotations=rotations,
        slot_controlled_permutations=permutations,
        inverse_uncomputation_path={
            "slot_diffusion_inverse": "V_slot_dag",
            "inverse_wrapper_calls": primary.operation_counts["inverse_lookup_calls_per_attempt"],
            "wrapper_inverse_is_complete": True,
            "discarded_work_registers": False,
        },
        wrapper=wrapper,
        sparse_block_encoding_wrapper=wrapper.circuit,
        qsvt_sequence=qsvt_sequence,
        postselection_logic={
            "event": "postselection flag is zero iff every slot and rotation-work qubit is zero",
            "classical_bit": 0,
            "distinct_from_interference_acceptance": True,
        },
        signed_readout_logic={
            "quadrature": "real",
            "bit_order": "c1c0",
            "readout_bit": 1,
            "recovery_formula": "(C/beta)||r||||ell||(N_00-N_10)/shots",
            "branch_probability": 0.5,
        },
        functional_circuits=circuits,
        final_measured_circuit=primary.circuit,
        direct_postselection_circuit=primary.direct_postselection_circuit,
        recovery_factors=recovery,
        component_hashes=component_hashes,
        failure_records=(),
        matrix_original=matrix,
        matrix_supported_exact=supported,
        matrix_quantized=quantized,
        residual=residual,
        functional_vectors=functionals,
        polynomial_coefficients=coefficients,
        phases=phases,
        matrix_spec=matrix_spec,
        support_spec=support_spec,
        quantization_spec=quantization_spec,
        qsvt_spec=qsvt_spec,
        residual_spec=residual_spec,
        functional_spec=functional_spec,
        execution_spec=execution_spec,
    )


def compile_from_bundle(bundle: CompilerInputBundle) -> CompiledSparseQSVT:
    """Convenience wrapper preserving the seven-record public contract."""

    return compile_sparse_selected_output_qsvt(
        bundle.matrix_spec,
        bundle.support_spec,
        bundle.quantization_spec,
        bundle.qsvt_spec,
        bundle.residual_spec,
        bundle.functional_spec,
        bundle.execution_spec,
    )


__all__ = [
    "CompiledSparseQSVT",
    "CompilerInputBundle",
    "ExecutionSpec",
    "FunctionalDefinition",
    "FunctionalSpec",
    "MatrixSpec",
    "QSVTSpec",
    "QuantizationSpec",
    "ResidualSpec",
    "SparseCompilerError",
    "SparseCompilerFailure",
    "SupportSpec",
    "compile_from_bundle",
    "compile_sparse_selected_output_qsvt",
]
