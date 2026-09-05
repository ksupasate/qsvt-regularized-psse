"""End-to-end sparse-access selected-output QSVT chain on the verified 8x8 workload.

The implementation composes the repository's complete Phase 10 sparse wrapper with
input residual preparation, the verified PCPhase/U/U-dagger QSVT sequence, coherent
encoded-subspace flagging, and a branch-Hadamard signed selected-output readout.  It is
an executable small-instance simulator experiment.  The stored slot/value realization
is intentionally not represented as a scalable QROM or IEEE-scale oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.paper.phase8_integrated_readout import (
    build_direct_chain_circuit as build_dense_direct_chain_circuit,
)
from robust_qsvt_se.paper.phase8_integrated_readout import (
    build_integrated_readout_circuit as build_dense_integrated_readout_circuit,
)
from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    CompleteWrapperResult,
    QuantizedSparseBlock,
    _build_block,
    build_quantized_sparse_block,
    validate_complete_wrapper,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    CodesignedBoundedTarget,
    fit_codesigned_bounded_polynomial,
)
from robust_qsvt_se.paper.tqe_revision_support_common import (
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.qsvt.block_encoding import canonical_square_block_encoding
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    build_structured_qsvt_operator_circuit,
    qsvt_sequence_operation_counts,
)
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_OUTPUT_DIR = Path("outputs/sparse_integrated_chain")
FROZEN_PHASE_PATH = Path(
    "outputs/phase10_sparse_wrapper_8x8_complete/phase_cache/"
    "38b8a10ad6ec88f509459bf3_phase_angles.csv"
)
CONFIGURATION_ID = "ieee14_sparse_quantized_8x8_d31_selected_v1"
PHASE_CONVENTION = "pennylane_qsvt_pcphase_u_udagger_real_top_left"
SUPPORTED_PHASE_CONVENTIONS = frozenset({PHASE_CONVENTION})
DEFAULT_SHOT_COUNTS = (10_000, 100_000, 1_000_000)
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_DEGREE = 31
DEFAULT_VALUE_BITS = 6
TARGET_MARGIN = 1.05
RELATIVE_BLOCK_TOLERANCE = 1.0e-9
SPARSE_DENSE_TOLERANCE = 1.0e-8
QSVT_EXACT_SVT_TOLERANCE = 1.0e-6
QSVT_RIDGE_TOLERANCE = 5.0e-2

CLAIM_BOUNDARY = (
    "Executed 8x8 sparse-access selected-output chain on a classical simulator using the "
    "repository's enumerated slot/value wrapper. This is not an IEEE-scale sparse oracle, "
    "not a scalable state loader, not a hardware run, and not a speedup or practical-"
    "competitiveness claim."
)


def stable_array_fingerprint(values: np.ndarray) -> str:
    """Return a cross-process SHA-256 fingerprint of contiguous float64 array bytes."""

    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class SparseIntegratedQSVTConfig:
    """Frozen configuration tying all components of the integrated experiment together."""

    configuration_id: str
    case_name: str
    case_source: str
    matrix_source: str
    matrix_path: Path
    residual_path: Path
    phase_path: Path
    matrix_fingerprint: str
    residual_fingerprint: str
    matrix_shape: tuple[int, int]
    matrix_value_bits: int
    alpha: float
    beta: float
    normalized_lambda: float
    contraction_c: float
    polynomial_degree: int
    phase_convention: str
    selected_output_name: str
    selected_output_vector: tuple[float, ...]
    shot_counts: tuple[int, ...]
    seeds: tuple[int, ...]
    selected_rows: tuple[int, ...]
    selected_columns: tuple[int, ...]
    normalization: str = "A = H_q^T / beta; beta = slots * mu"

    def __post_init__(self) -> None:
        if not self.configuration_id.strip():
            raise ValueError("configuration_id must be nonempty")
        if not self.matrix_fingerprint.strip() or not self.residual_fingerprint.strip():
            raise ValueError("matrix and residual fingerprints must be nonempty")
        rows, columns = self.matrix_shape
        if rows <= 0 or columns <= 0 or rows != columns:
            raise ValueError("the current sparse wrapper requires a nonempty square matrix")
        if rows & (rows - 1):
            raise ValueError("the current sparse wrapper requires power-of-two dimensions")
        if self.matrix_value_bits <= 0:
            raise ValueError("matrix_value_bits must be positive")
        if self.alpha <= 0.0 or self.beta <= 0.0 or self.contraction_c <= 0.0:
            raise ValueError("alpha, beta, and C must be positive")
        expected_lambda = self.alpha / self.beta**2
        if not math.isclose(self.normalized_lambda, expected_lambda, rel_tol=1.0e-12):
            raise ValueError("normalized_lambda must equal alpha / beta**2")
        if self.polynomial_degree <= 0 or self.polynomial_degree % 2 == 0:
            raise ValueError("the verified convention supports positive odd degree only")
        if self.phase_convention not in SUPPORTED_PHASE_CONVENTIONS:
            raise ValueError(f"unsupported phase convention: {self.phase_convention}")
        if len(self.selected_output_vector) != columns:
            raise ValueError("selected-output functional length does not match matrix columns")
        if not any(abs(value) > 0.0 for value in self.selected_output_vector):
            raise ValueError("selected-output functional cannot be zero")
        if not self.shot_counts or any(int(shots) <= 0 for shots in self.shot_counts):
            raise ValueError("shot counts must be positive")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("seeds must be nonempty and unique")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("matrix_path", "residual_path", "phase_path"):
            payload[key] = str(payload[key])
        for key in (
            "matrix_shape",
            "selected_output_vector",
            "shot_counts",
            "seeds",
            "selected_rows",
            "selected_columns",
        ):
            payload[key] = list(payload[key])
        payload["C"] = payload["contraction_c"]
        payload["lambda"] = payload["normalized_lambda"]
        payload["quantization"] = f"{self.matrix_value_bits} magnitude bits + 1 sign bit"
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> SparseIntegratedQSVTConfig:
        """Load the typed record from the generated configuration JSON schema."""

        return cls(
            configuration_id=str(payload["configuration_id"]),
            case_name=str(payload["case_name"]),
            case_source=str(payload["case_source"]),
            matrix_source=str(payload["matrix_source"]),
            matrix_path=Path(payload["matrix_path"]),
            residual_path=Path(payload["residual_path"]),
            phase_path=Path(payload["phase_path"]),
            matrix_fingerprint=str(payload["matrix_fingerprint"]),
            residual_fingerprint=str(payload["residual_fingerprint"]),
            matrix_shape=tuple(int(value) for value in payload["matrix_shape"]),
            matrix_value_bits=int(payload["matrix_value_bits"]),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            normalized_lambda=float(payload["normalized_lambda"]),
            contraction_c=float(payload["contraction_c"]),
            polynomial_degree=int(payload["polynomial_degree"]),
            phase_convention=str(payload["phase_convention"]),
            selected_output_name=str(payload["selected_output_name"]),
            selected_output_vector=tuple(
                float(value) for value in payload["selected_output_vector"]
            ),
            shot_counts=tuple(int(value) for value in payload["shot_counts"]),
            seeds=tuple(int(value) for value in payload["seeds"]),
            selected_rows=tuple(int(value) for value in payload["selected_rows"]),
            selected_columns=tuple(int(value) for value in payload["selected_columns"]),
            normalization=str(payload["normalization"]),
        )


def load_sparse_integrated_config(path: str | Path) -> SparseIntegratedQSVTConfig:
    """Read and validate a generated frozen configuration record."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration JSON must contain an object")
    return SparseIntegratedQSVTConfig.from_json_dict(payload)


@dataclass(slots=True)
class IntegratedSparseQSVTCircuit:
    """Composed sampled circuit and its direct-postselection companion."""

    circuit: Any
    direct_postselection_circuit: Any
    wrapper: CompleteWrapperResult
    register_layout: dict[str, Any]
    operation_counts: dict[str, int]
    preparation_records: tuple[dict[str, str], ...]
    output_state_used_for_preparation: bool = False


@dataclass(slots=True)
class DefaultSparseIntegratedInputs:
    config: SparseIntegratedQSVTConfig
    matrix_quantized: np.ndarray
    matrix_sparsified: np.ndarray
    matrix_original: np.ndarray
    residual: np.ndarray
    phases: np.ndarray
    target: CodesignedBoundedTarget
    selected_functionals: dict[str, np.ndarray]
    source_metadata: dict[str, Any]


@dataclass(slots=True)
class StatevectorValidation:
    metrics: dict[str, float]
    sparse_update: np.ndarray
    dense_update: np.ndarray
    exact_polynomial_svt_update: np.ndarray
    exact_rational_svt_update: np.ndarray
    quantized_ridge_update: np.ndarray
    original_ridge_update: np.ndarray
    sparse_encoded_state: np.ndarray
    dense_encoded_state: np.ndarray
    target_block: np.ndarray
    sparse_block: np.ndarray


def predetermined_selected_functionals(dimension: int) -> dict[str, np.ndarray]:
    """Return deterministic functionals fixed independently of all solved outputs."""

    if dimension < 4:
        raise ValueError("the predetermined extension requires dimension at least four")
    coordinate = np.zeros(dimension, dtype=np.float64)
    coordinate[0] = 1.0
    difference = np.zeros(dimension, dtype=np.float64)
    difference[0], difference[1] = 1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)
    aggregate = np.zeros(dimension, dtype=np.float64)
    aggregate[:4] = 0.5
    return {
        "coordinate_e0": coordinate,
        "signed_difference_e0_minus_e1": difference,
        "aggregate_e0_to_e3": aggregate,
    }


def load_frozen_phases(path: str | Path = FROZEN_PHASE_PATH) -> np.ndarray:
    phase_path = Path(path)
    if not phase_path.is_file():
        raise FileNotFoundError(f"frozen Phase 10 phase sequence is missing: {phase_path}")
    frame = pd.read_csv(phase_path)
    if "phase_angle" not in frame:
        raise ValueError("phase CSV must contain phase_angle")
    phases = frame["phase_angle"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(phases)):
        raise ValueError("phase sequence contains non-finite entries")
    return phases


def build_default_sparse_integrated_inputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    shot_counts: tuple[int, ...] = DEFAULT_SHOT_COUNTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> DefaultSparseIntegratedInputs:
    """Rebuild the frozen Phase 10 matrix and bind it to the integrated configuration."""

    destination = Path(output_dir)
    source = _build_block(123)
    quantized = build_quantized_sparse_block(source["H_block"], magnitude_bits=DEFAULT_VALUE_BITS)
    matrix = np.asarray(quantized.quantized, dtype=np.float64)
    residual = np.asarray(source["r_block"], dtype=np.float64)
    encoded_pattern = np.abs(matrix.T) > 0.0
    slots = int(max(encoded_pattern.sum(axis=0).max(), encoded_pattern.sum(axis=1).max()))
    beta = float(slots * quantized.mu)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-10]
    if positive.size == 0:
        raise ValueError(
            "the verified sparse workload unexpectedly has no positive singular values"
        )
    alpha = 4.0 * float(positive.min()) ** 2
    domain_min = float(np.clip(0.9 * positive.min() / beta, 1.0e-4, 0.999))
    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=alpha,
        domain_min=domain_min,
        domain_max=1.0,
        degree=DEFAULT_DEGREE,
        margin=TARGET_MARGIN,
    )
    validate_qsvt_polynomial(
        np.asarray(target.coefficients), parity="odd", bound_tolerance=2.0e-3
    )
    phases = load_frozen_phases()
    functionals = predetermined_selected_functionals(matrix.shape[1])
    config = SparseIntegratedQSVTConfig(
        configuration_id=CONFIGURATION_ID,
        case_name="ieee14",
        case_source="pypower",
        matrix_source=str(source["matrix_source"]),
        matrix_path=destination / "matrix_quantized.npy",
        residual_path=destination / "residual.npy",
        phase_path=FROZEN_PHASE_PATH,
        matrix_fingerprint=stable_array_fingerprint(matrix),
        residual_fingerprint=stable_array_fingerprint(residual),
        matrix_shape=tuple(int(value) for value in matrix.shape),
        matrix_value_bits=DEFAULT_VALUE_BITS,
        alpha=alpha,
        beta=beta,
        normalized_lambda=alpha / beta**2,
        contraction_c=target.bound_C,
        polynomial_degree=DEFAULT_DEGREE,
        phase_convention=PHASE_CONVENTION,
        selected_output_name="coordinate_e0",
        selected_output_vector=tuple(float(value) for value in functionals["coordinate_e0"]),
        shot_counts=tuple(int(value) for value in shot_counts),
        seeds=tuple(int(value) for value in seeds),
        selected_rows=tuple(int(value) for value in source["selected_rows"]),
        selected_columns=tuple(int(value) for value in source["selected_cols"]),
    )
    validate_integrated_inputs(
        config,
        matrix=matrix,
        residual=residual,
        selected_functional=functionals[config.selected_output_name],
        phases=phases,
        dense_reference_matrix=matrix.copy(),
    )
    return DefaultSparseIntegratedInputs(
        config=config,
        matrix_quantized=matrix,
        matrix_sparsified=np.asarray(quantized.sparsified, dtype=np.float64),
        matrix_original=np.asarray(quantized.original, dtype=np.float64),
        residual=residual,
        phases=phases,
        target=target,
        selected_functionals=functionals,
        source_metadata={
            "system_seed": 123,
            "selected_rows": config.selected_rows,
            "selected_columns": config.selected_columns,
            "slots": slots,
            "mu": quantized.mu,
            "nnz": quantized.nnz,
            "quantization_step": quantized.quantization_step,
            "max_quantization_error": quantized.max_quantization_error,
            "sparsification_relative_fro_error": quantized.sparsification_fro_error,
            "domain_min": domain_min,
            "singular_values_quantized": singular_values,
        },
    )


def _real_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) and np.max(np.abs(np.imag(array)), initial=0.0) > 1.0e-14:
        raise ValueError(f"unsupported complex {name} mode")
    result = np.asarray(np.real(array), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite entries")
    return result


def validate_integrated_inputs(
    config: SparseIntegratedQSVTConfig,
    *,
    matrix: np.ndarray,
    residual: np.ndarray,
    selected_functional: np.ndarray,
    phases: np.ndarray,
    dense_reference_matrix: np.ndarray | None = None,
) -> None:
    """Fail fast on every cross-component incompatibility in the integrated chain."""

    matrix_values = _real_array("matrix", matrix)
    residual_values = _real_array("residual", residual)
    functional_values = _real_array("selected functional", selected_functional)
    phase_values = _real_array("phase sequence", phases)
    if matrix_values.shape != config.matrix_shape:
        raise ValueError("matrix dimensions do not match the frozen configuration")
    if residual_values.ndim != 1 or residual_values.size != matrix_values.shape[0]:
        raise ValueError("residual length does not match matrix rows")
    if functional_values.ndim != 1 or functional_values.size != matrix_values.shape[1]:
        raise ValueError("selected-functional length does not match matrix columns")
    if np.linalg.norm(residual_values) <= 1.0e-15:
        raise ValueError("residual cannot be zero")
    if np.linalg.norm(functional_values) <= 1.0e-15:
        raise ValueError("selected functional cannot be zero")
    if phase_values.ndim != 1 or phase_values.size != config.polynomial_degree + 1:
        raise ValueError("phase sequence length must equal polynomial degree + 1")
    if stable_array_fingerprint(matrix_values) != config.matrix_fingerprint:
        raise ValueError("matrix fingerprint does not match the frozen configuration")
    if stable_array_fingerprint(residual_values) != config.residual_fingerprint:
        raise ValueError("residual fingerprint does not match the frozen configuration")
    if not math.isclose(
        config.normalized_lambda, config.alpha / config.beta**2, rel_tol=1.0e-12
    ):
        raise ValueError("lambda is inconsistent with alpha and beta")
    spectral_norm = float(np.linalg.svd(matrix_values, compute_uv=False).max())
    if config.beta + 1.0e-12 < spectral_norm:
        raise ValueError("beta must upper-bound the matrix spectral norm")
    if dense_reference_matrix is not None:
        dense = _real_array("dense reference matrix", dense_reference_matrix)
        if dense.shape != matrix_values.shape or not np.array_equal(dense, matrix_values):
            raise ValueError("dense and sparse references must use the identical quantized matrix")


def _as_quantized_block(matrix: np.ndarray, magnitude_bits: int) -> QuantizedSparseBlock:
    values = np.asarray(matrix, dtype=np.float64)
    mu = float(np.max(np.abs(values)))
    levels = (1 << int(magnitude_bits)) - 1
    return QuantizedSparseBlock(
        original=values.copy(),
        sparsified=values.copy(),
        quantized=values.copy(),
        mu=mu,
        magnitude_bits=int(magnitude_bits),
        nnz=int(np.count_nonzero(values)),
        quantization_step=mu / levels,
        max_quantization_error=0.0,
        sparsification_fro_error=0.0,
    )


def _branch_pcphase_gate(phase: float, work_qubits: int) -> Any:
    """PCPhase on work=0, controlled by the readout branch being zero."""

    from qiskit import QuantumCircuit

    if work_qubits < 1:
        raise ValueError("PCPhase requires at least one sparse work qubit")
    gate_circuit = QuantumCircuit(work_qubits + 1, name="c0_PCPhase")
    all_qubits = list(range(work_qubits + 1))
    gate_circuit.x(all_qubits)
    gate_circuit.p(-float(phase), 0)
    gate_circuit.mcp(
        2.0 * float(phase),
        list(range(work_qubits)),
        work_qubits,
    )
    gate_circuit.x(all_qubits)
    return gate_circuit.to_gate(label="c0_PCPhase")


def _direct_pcphase_gate(phase: float, work_qubits: int) -> Any:
    """PCPhase on the sparse encoded subspace (all work qubits zero)."""

    from qiskit import QuantumCircuit

    gate_circuit = QuantumCircuit(work_qubits, name="PCPhase")
    gate_circuit.global_phase = -float(phase)
    gate_circuit.x(range(work_qubits))
    if work_qubits == 1:
        gate_circuit.p(2.0 * float(phase), 0)
    else:
        gate_circuit.mcp(
            2.0 * float(phase),
            list(range(work_qubits - 1)),
            work_qubits - 1,
        )
    gate_circuit.x(range(work_qubits))
    return gate_circuit.to_gate(label="PCPhase")


def _append_sparse_qsvt_branch(
    circuit: Any,
    *,
    wrapper_gate: Any,
    phases: np.ndarray,
    chain_qubits: list[int],
    work_qubits: list[int],
    readout_qubit: int,
) -> dict[str, int]:
    controlled_forward = wrapper_gate.control(1, ctrl_state=0, annotated=False)
    controlled_forward.label = "c0_sparse_U_A"
    controlled_inverse = wrapper_gate.inverse().control(1, ctrl_state=0, annotated=False)
    controlled_inverse.label = "c0_sparse_U_A_dagger"
    phase_values = np.asarray(phases, dtype=np.float64)
    circuit.append(
        _branch_pcphase_gate(float(phase_values[0]), len(work_qubits)),
        [readout_qubit, *work_qubits],
    )
    forward_calls = 0
    inverse_calls = 0
    for index in range(1, phase_values.size - 1, 2):
        circuit.append(controlled_forward, [readout_qubit, *chain_qubits])
        forward_calls += 1
        circuit.append(
            _branch_pcphase_gate(float(phase_values[index]), len(work_qubits)),
            [readout_qubit, *work_qubits],
        )
        circuit.append(controlled_inverse, [readout_qubit, *chain_qubits])
        inverse_calls += 1
        circuit.append(
            _branch_pcphase_gate(float(phase_values[index + 1]), len(work_qubits)),
            [readout_qubit, *work_qubits],
        )
    if phase_values.size % 2 == 0:
        circuit.append(controlled_forward, [readout_qubit, *chain_qubits])
        forward_calls += 1
        circuit.append(
            _branch_pcphase_gate(float(phase_values[-1]), len(work_qubits)),
            [readout_qubit, *work_qubits],
        )
    expected = qsvt_sequence_operation_counts(int(phase_values.size))
    if forward_calls + inverse_calls != expected["signal_unitary_calls"]:
        raise RuntimeError("sparse signal-call count disagrees with the QSVT convention")
    return {
        "signal_unitary_calls_per_attempt": forward_calls + inverse_calls,
        "forward_sparse_lookup_calls_per_attempt": forward_calls,
        "inverse_lookup_calls_per_attempt": inverse_calls,
        "projector_phase_operations_per_attempt": int(phase_values.size),
        "alternating_sequence_length": forward_calls + inverse_calls + int(phase_values.size),
    }


def _append_direct_sparse_qsvt(
    circuit: Any,
    *,
    wrapper_gate: Any,
    phases: np.ndarray,
    chain_qubits: list[int],
    work_qubits: list[int],
) -> None:
    inverse = wrapper_gate.inverse()
    inverse.label = "sparse_U_A_dagger"
    phase_values = np.asarray(phases, dtype=np.float64)
    circuit.append(
        _direct_pcphase_gate(float(phase_values[0]), len(work_qubits)), work_qubits
    )
    for index in range(1, phase_values.size - 1, 2):
        circuit.append(wrapper_gate, chain_qubits)
        circuit.append(
            _direct_pcphase_gate(float(phase_values[index]), len(work_qubits)), work_qubits
        )
        circuit.append(inverse, chain_qubits)
        circuit.append(
            _direct_pcphase_gate(float(phase_values[index + 1]), len(work_qubits)),
            work_qubits,
        )
    if phase_values.size % 2 == 0:
        circuit.append(wrapper_gate, chain_qubits)
        circuit.append(
            _direct_pcphase_gate(float(phase_values[-1]), len(work_qubits)), work_qubits
        )


def _append_postselection_flag(circuit: Any, work_qubits: list[int], flag_qubit: int) -> None:
    """Set flag=0 exactly on the all-zero sparse work subspace; flag=1 otherwise."""

    circuit.x(flag_qubit)
    circuit.x(work_qubits)
    circuit.mcx(work_qubits, flag_qubit)
    circuit.x(work_qubits)


def build_integrated_sparse_selected_output_circuit(
    config: SparseIntegratedQSVTConfig,
    *,
    matrix: np.ndarray,
    residual: np.ndarray,
    selected_functional: np.ndarray,
    phases: np.ndarray,
) -> IntegratedSparseQSVTCircuit:
    """Build the complete measured residual-to-signed-output sparse QSVT chain."""

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    validate_integrated_inputs(
        config,
        matrix=matrix,
        residual=residual,
        selected_functional=selected_functional,
        phases=phases,
    )
    matrix_values = np.asarray(matrix, dtype=np.float64)
    residual_values = np.asarray(residual, dtype=np.float64)
    functional_values = np.asarray(selected_functional, dtype=np.float64)
    block = _as_quantized_block(matrix_values, config.matrix_value_bits)
    wrapper = validate_complete_wrapper(
        block, encode_transpose=True, transpile_circuit=False
    )
    if not math.isclose(wrapper.normalization_factor, config.beta, rel_tol=1.0e-12):
        raise ValueError("sparse wrapper normalization does not match configured beta")
    if wrapper.top_left_reconstruction_error > RELATIVE_BLOCK_TOLERANCE:
        raise RuntimeError("sparse wrapper reconstruction exceeds the integration tolerance")

    index_qubits = round(math.log2(matrix_values.shape[1]))
    chain_qubit_count = int(wrapper.circuit.num_qubits)
    slot_qubits = chain_qubit_count - index_qubits - 1
    index = list(range(index_qubits))
    slot = list(range(index_qubits, index_qubits + slot_qubits))
    rotation_ancilla = index_qubits + slot_qubits
    work = [*slot, rotation_ancilla]
    chain = list(range(chain_qubit_count))
    postselection_flag = chain_qubit_count
    readout = chain_qubit_count + 1
    circuit = QuantumCircuit(chain_qubit_count + 2, 2, name="sparse_integrated_qsvt_readout")
    circuit.h(readout)

    residual_unit = residual_values / np.linalg.norm(residual_values)
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

    functional_unit = functional_values / np.linalg.norm(functional_values)
    functional_prep = StatePreparation(functional_unit).control(
        1, ctrl_state=1, annotated=False
    )
    functional_prep.label = "c1_functional_prep"
    circuit.append(functional_prep, [readout, *index])
    _append_postselection_flag(circuit, work, postselection_flag)
    circuit.h(readout)
    circuit.measure(postselection_flag, 0)
    circuit.measure(readout, 1)

    direct = QuantumCircuit(chain_qubit_count + 1, 1, name="sparse_direct_postselection")
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
        "value_rotation_gates_per_sparse_lookup": int(matrix_values.shape[1] * wrapper.slots),
        "value_rotations_per_attempt": int(
            matrix_values.shape[1]
            * wrapper.slots
            * accounting["signal_unitary_calls_per_attempt"]
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
            "encoded_subspace": "slot=00 and rotation_ancilla=0; index unrestricted",
        },
        operation_counts=operation_counts,
        preparation_records=(
            {"role": "residual_input", "fingerprint": stable_array_fingerprint(residual_unit)},
            {
                "role": "selected_functional_reference",
                "fingerprint": stable_array_fingerprint(functional_unit),
            },
        ),
    )


def assert_no_direct_output_initializer(circuit: Any, output_state: np.ndarray) -> None:
    """Reject a direct StatePreparation/Initialize of a computed postselected output."""

    target = np.asarray(output_state, dtype=np.complex128).ravel()
    norm = float(np.linalg.norm(target))
    if norm <= 1.0e-15:
        return
    target = target / norm
    for instruction in circuit.data:
        operation = instruction.operation
        candidates = [operation]
        base_gate = getattr(operation, "base_gate", None)
        if base_gate is not None:
            candidates.append(base_gate)
        for candidate in candidates:
            if candidate.name not in {"state_preparation", "initialize"}:
                continue
            try:
                prepared = np.asarray(candidate.params, dtype=np.complex128).ravel()
            except (TypeError, ValueError):
                continue
            if prepared.size != target.size:
                continue
            prepared_norm = float(np.linalg.norm(prepared))
            if prepared_norm <= 1.0e-15:
                continue
            fidelity = abs(np.vdot(target, prepared / prepared_norm)) ** 2
            if fidelity >= 1.0 - 1.0e-12:
                raise RuntimeError("integrated circuit directly initializes the computed output")


def statevector_validate_integrated_chain(
    inputs: DefaultSparseIntegratedInputs,
    bundle: IntegratedSparseQSVTCircuit,
) -> StatevectorValidation:
    """Validate sparse block/action, dense action, exact SVT, and matched Ridge."""

    from qiskit.quantum_info import Operator, Statevector

    config = inputs.config
    matrix = inputs.matrix_quantized
    residual = inputs.residual
    n = matrix.shape[1]
    residual_norm = float(np.linalg.norm(residual))
    residual_unit = residual / residual_norm
    sparse_unitary = np.asarray(bundle.wrapper.unitary, dtype=np.complex128)
    sparse_qsvt = build_structured_qsvt_operator_circuit(
        sparse_unitary, inputs.phases, encoded_dimension=n
    )
    sparse_operator = np.asarray(
        Operator(sparse_qsvt.qsvt_operator_circuit).data, dtype=np.complex128
    )
    sparse_input = np.zeros(sparse_operator.shape[0], dtype=np.complex128)
    sparse_input[:n] = residual_unit
    sparse_full = Statevector(sparse_input).evolve(sparse_qsvt.qsvt_operator_circuit).data

    normalized_matrix = matrix.T / config.beta
    dense_encoding = canonical_square_block_encoding(normalized_matrix, tolerance=1.0e-8)
    dense_qsvt = build_structured_qsvt_operator_circuit(
        dense_encoding.unitary, inputs.phases, encoded_dimension=n
    )
    dense_operator = np.asarray(
        Operator(dense_qsvt.qsvt_operator_circuit).data, dtype=np.complex128
    )
    dense_input = np.zeros(dense_operator.shape[0], dtype=np.complex128)
    dense_input[:n] = residual_unit
    dense_full = Statevector(dense_input).evolve(dense_qsvt.qsvt_operator_circuit).data

    sparse_block = np.asarray(bundle.wrapper.encoded_block, dtype=np.complex128)
    target_block = normalized_matrix
    block_error = float(
        np.linalg.norm(sparse_block - target_block, ord="fro")
        / max(np.linalg.norm(target_block, ord="fro"), 1.0e-30)
    )
    sparse_action = np.real(sparse_full[:n])
    dense_action = np.real(dense_full[:n])
    sparse_dense_error = float(
        np.linalg.norm(sparse_action - dense_action)
        / max(np.linalg.norm(dense_action), 1.0e-30)
    )

    u_matrix, singular_values, vh_matrix = np.linalg.svd(
        normalized_matrix, full_matrices=False
    )
    polynomial = Polynomial(np.asarray(inputs.target.coefficients, dtype=np.float64))
    exact_polynomial_operator = (
        u_matrix @ np.diag(polynomial(singular_values)) @ vh_matrix
    )
    rational_values = (
        singular_values
        / (singular_values**2 + config.normalized_lambda)
        / config.contraction_c
    )
    exact_rational_operator = u_matrix @ np.diag(rational_values) @ vh_matrix
    physical_scale = config.contraction_c / config.beta * residual_norm
    sparse_update = physical_scale * sparse_action
    dense_update = physical_scale * dense_action
    exact_polynomial_update = physical_scale * (exact_polynomial_operator @ residual_unit)
    exact_rational_update = physical_scale * (exact_rational_operator @ residual_unit)
    quantized_ridge = ridge_svd_solution(matrix, residual, alpha=config.alpha)
    original_ridge = ridge_svd_solution(inputs.matrix_original, residual, alpha=config.alpha)
    qsvt_exact_error = float(
        np.linalg.norm(sparse_update - exact_polynomial_update)
        / max(np.linalg.norm(exact_polynomial_update), 1.0e-30)
    )
    qsvt_ridge_error = float(
        np.linalg.norm(sparse_update - quantized_ridge)
        / max(np.linalg.norm(quantized_ridge), 1.0e-30)
    )
    exact_rational_ridge_error = float(
        np.linalg.norm(exact_rational_update - quantized_ridge)
        / max(np.linalg.norm(quantized_ridge), 1.0e-30)
    )
    quantization_ridge_error = float(
        np.linalg.norm(quantized_ridge - original_ridge)
        / max(np.linalg.norm(original_ridge), 1.0e-30)
    )
    sparse_postselection = float(np.vdot(sparse_full[:n], sparse_full[:n]).real)
    dense_postselection = float(np.vdot(dense_full[:n], dense_full[:n]).real)
    metrics = {
        "block_reconstruction_relative_fro_error": block_error,
        "sparse_dense_action_relative_l2_error": sparse_dense_error,
        "qsvt_exact_polynomial_svt_relative_l2_error": qsvt_exact_error,
        "qsvt_quantized_ridge_relative_l2_error": qsvt_ridge_error,
        "exact_rational_svt_quantized_ridge_relative_l2_error": exact_rational_ridge_error,
        "quantized_ridge_original_ridge_relative_l2_difference": quantization_ridge_error,
        "sparse_postselection_probability": sparse_postselection,
        "dense_postselection_probability": dense_postselection,
        "postselection_probability_absolute_difference": abs(
            sparse_postselection - dense_postselection
        ),
        "physical_rescaling_factor_C_over_beta": config.contraction_c / config.beta,
        "residual_norm": residual_norm,
    }
    if block_error > RELATIVE_BLOCK_TOLERANCE:
        raise RuntimeError("relative sparse block reconstruction validation failed")
    if sparse_dense_error > SPARSE_DENSE_TOLERANCE:
        raise RuntimeError("sparse and dense QSVT actions disagree")
    if qsvt_exact_error > QSVT_EXACT_SVT_TOLERANCE:
        raise RuntimeError("sparse QSVT action disagrees with exact polynomial SVT")
    if qsvt_ridge_error > QSVT_RIDGE_TOLERANCE:
        raise RuntimeError("sparse QSVT action disagrees with matched quantized Ridge")
    return StatevectorValidation(
        metrics=metrics,
        sparse_update=np.asarray(sparse_update, dtype=np.float64),
        dense_update=np.asarray(dense_update, dtype=np.float64),
        exact_polynomial_svt_update=np.asarray(exact_polynomial_update, dtype=np.float64),
        exact_rational_svt_update=np.asarray(exact_rational_update, dtype=np.float64),
        quantized_ridge_update=np.asarray(quantized_ridge, dtype=np.float64),
        original_ridge_update=np.asarray(original_ridge, dtype=np.float64),
        sparse_encoded_state=np.asarray(sparse_full[:n], dtype=np.complex128),
        dense_encoded_state=np.asarray(dense_full[:n], dtype=np.complex128),
        target_block=np.asarray(target_block, dtype=np.float64),
        sparse_block=np.asarray(sparse_block, dtype=np.complex128),
    )


def estimate_signed_selected_output(
    counts: dict[str, int],
    *,
    physical_scale: float,
) -> dict[str, float]:
    """Recover signed output and plug-in finite-shot uncertainty from joint counts."""

    attempted = int(sum(counts.values()))
    if attempted <= 0:
        raise ValueError("counts must contain at least one shot")
    plus = int(counts.get("00", 0))
    minus = int(counts.get("10", 0))
    accepted = plus + minus
    if accepted != sum(value for key, value in counts.items() if key[-1] == "0"):
        raise RuntimeError("joint readout count bookkeeping is inconsistent")
    acceptance = accepted / attempted
    signed_overlap = (plus - minus) / attempted
    sign_mean = (plus - minus) / accepted if accepted else float("nan")
    inferred_postselection = 2.0 * acceptance - 1.0
    overlap_variance = max(acceptance - signed_overlap**2, 0.0) / attempted
    standard_error = float(physical_scale * math.sqrt(overlap_variance))
    estimate = float(physical_scale * signed_overlap)
    return {
        "readout_accepted": float(accepted),
        "interference_acceptance_probability": acceptance,
        "readout_sign_mean_accepted": sign_mean,
        "signed_overlap_estimate": signed_overlap,
        "inferred_postselection_probability_from_branch": inferred_postselection,
        "selected_output_estimate": estimate,
        "analytic_standard_error": standard_error,
        "confidence_interval_lower": estimate - 1.959963984540054 * standard_error,
        "confidence_interval_upper": estimate + 1.959963984540054 * standard_error,
    }


def exact_joint_distribution(
    circuit: Any, *, postselection_flag_qubit: int, readout_qubit: int
) -> dict[str, float]:
    """Return exact c1c0 probabilities for validation only, never sampled fake counts."""

    from qiskit.quantum_info import Statevector

    measurement_free = circuit.remove_final_measurements(inplace=False)
    state = Statevector(measurement_free)
    probabilities = np.asarray(
        state.probabilities([postselection_flag_qubit, readout_qubit]), dtype=np.float64
    )
    return {
        "00": float(probabilities[0]),
        "01": float(probabilities[1]),
        "10": float(probabilities[2]),
        "11": float(probabilities[3]),
    }


def compile_for_aer(circuit: Any) -> tuple[Any, Any]:
    """Compile a measured circuit to the actual Aer simulator target."""

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("qiskit-aer is required for actual finite-shot execution") from exc
    simulator = AerSimulator(
        method="statevector",
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    compiled = transpile(circuit, simulator, optimization_level=0)
    return compiled, simulator


def sample_aer_counts(
    compiled_circuit: Any,
    simulator: Any,
    *,
    shots: int,
    seed: int,
) -> dict[str, int]:
    """Run genuine Aer shot sampling; no statevector-derived count fallback is allowed."""

    result = simulator.run(
        compiled_circuit, shots=int(shots), seed_simulator=int(seed)
    ).result()
    raw = result.get_counts()
    return {str(key).replace(" ", ""): int(value) for key, value in raw.items()}


def _direct_postselection_estimate(counts: dict[str, int]) -> tuple[int, float]:
    attempted = int(sum(counts.values()))
    if attempted <= 0:
        raise ValueError("direct postselection counts are empty")
    accepted = int(counts.get("0", 0))
    return accepted, accepted / attempted


def _resource_counts(compiled: Any) -> dict[str, Any]:
    counts = {str(key): int(value) for key, value in compiled.count_ops().items()}
    return {
        "operation_counts": counts,
        "gate_count": int(sum(counts.values())),
        "depth": int(compiled.depth()),
        "toffoli_count": int(counts.get("ccx", 0)),
    }


def build_executed_resource_records(
    inputs: DefaultSparseIntegratedInputs,
    sparse_bundle: IntegratedSparseQSVTCircuit,
    *,
    compiled_sparse: Any,
    compiled_dense: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build separate executed-small-scale and modeled-IEEE resource categories."""

    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation

    sparse_counts = _resource_counts(compiled_sparse)
    dense_counts = _resource_counts(compiled_dense)
    wrapper_transpiled = transpile(
        sparse_bundle.wrapper.circuit,
        basis_gates=["u3", "cx"],
        optimization_level=1,
    )
    wrapper_counts = {
        str(key): int(value) for key, value in wrapper_transpiled.count_ops().items()
    }
    residual_unit = inputs.residual / np.linalg.norm(inputs.residual)
    residual_probe = QuantumCircuit(4)
    residual_probe.append(
        StatePreparation(residual_unit).control(1, ctrl_state=0, annotated=False),
        range(4),
    )
    residual_compiled, _simulator = compile_for_aer(residual_probe.measure_all(inplace=False))
    residual_gate_count = int(
        sum(int(value) for value in residual_compiled.count_ops().values())
        - residual_compiled.count_ops().get("measure", 0)
    )
    config = inputs.config
    operations = sparse_bundle.operation_counts
    executed_sparse = {
        "resource_category": "executed_small_scale_sparse_integrated",
        "execution_status": "executed_sampled_counts_and_statevector",
        "configuration_id": config.configuration_id,
        "matrix_fingerprint": config.matrix_fingerprint,
        "matrix_shape": "8x8",
        "matrix_nonzeros": int(np.count_nonzero(inputs.matrix_quantized)),
        "maximum_row_sparsity": int(np.count_nonzero(inputs.matrix_quantized, axis=1).max()),
        "maximum_encoded_row_or_column_degree": sparse_bundle.wrapper.slots,
        "value_bits": config.matrix_value_bits,
        "value_sign_bits": 1,
        "value_total_bits": config.matrix_value_bits + 1,
        "row_register_qubits": 3,
        "slot_register_qubits": 2,
        "column_register_qubits": 3,
        "index_register_alias": "same in-place 3-qubit register is column input and row output",
        "value_register_qubits": "not_allocated_direct_multiplexed_rotation",
        "work_qubits": 3,
        "postselection_flag_qubits": 1,
        "signal_qubits": 3,
        "readout_qubits": 1,
        "total_logical_qubits": int(sparse_bundle.circuit.num_qubits),
        "ancilla_and_work_qubits": int(sparse_bundle.circuit.num_qubits - 3),
        "polynomial_degree": config.polynomial_degree,
        "signal_unitary_calls_per_attempt": operations["signal_unitary_calls_per_attempt"],
        "projector_phase_operations_per_attempt": operations[
            "projector_phase_operations_per_attempt"
        ],
        "sparse_lookup_calls_per_attempt": operations["sparse_lookup_calls_per_attempt"],
        "forward_sparse_lookup_calls_per_attempt": operations[
            "forward_sparse_lookup_calls_per_attempt"
        ],
        "inverse_lookup_calls_per_attempt": operations["inverse_lookup_calls_per_attempt"],
        "value_rotations_per_attempt": operations["value_rotations_per_attempt"],
        "residual_preparations_per_attempt": 1,
        "postselection_measurements_per_attempt": 1,
        "readout_measurements_per_attempt": 1,
        "transpiled_gate_count": sparse_counts["gate_count"],
        "transpiled_depth": sparse_counts["depth"],
        "toffoli_count": sparse_counts["toffoli_count"],
        "controlled_rotation_count": operations["value_rotations_per_attempt"],
        "one_signal_unitary_gate_count": int(sum(wrapper_counts.values())),
        "one_signal_unitary_depth": int(wrapper_transpiled.depth()),
        "one_signal_unitary_cx_count": int(wrapper_counts.get("cx", 0)),
        "residual_preparation_gate_count": residual_gate_count,
        "total_gate_count_per_attempt": sparse_counts["gate_count"],
        "total_depth_per_attempt": sparse_counts["depth"],
        "estimated_attempts_per_accepted_sample": 1.0
        / inputs_statevector_postselection(inputs, sparse_bundle),
        "selected_output_estimate_shots": max(config.shot_counts),
        "estimated_gates_per_selected_output_estimate": int(
            sparse_counts["gate_count"] * max(config.shot_counts)
        ),
        "transpilation_target": "qiskit_aer_statevector_default_target_optimization_level_0",
        "operation_counts_json": json.dumps(sparse_counts["operation_counts"], sort_keys=True),
    }
    executed_dense = {
        "resource_category": "executed_small_scale_dense_integrated",
        "execution_status": "executed_sampled_counts_and_statevector",
        "configuration_id": config.configuration_id,
        "matrix_fingerprint": config.matrix_fingerprint,
        "matrix_shape": "8x8",
        "matrix_nonzeros": int(np.count_nonzero(inputs.matrix_quantized)),
        "maximum_row_sparsity": "not_applicable_dense_dilation",
        "maximum_encoded_row_or_column_degree": "not_applicable_dense_dilation",
        "value_bits": config.matrix_value_bits,
        "value_sign_bits": 1,
        "value_total_bits": config.matrix_value_bits + 1,
        "row_register_qubits": "not_applicable_dense_dilation",
        "slot_register_qubits": "not_applicable_dense_dilation",
        "column_register_qubits": 3,
        "index_register_alias": "dense system register",
        "value_register_qubits": "not_applicable_dense_dilation",
        "work_qubits": 1,
        "postselection_flag_qubits": 1,
        "signal_qubits": 3,
        "readout_qubits": 1,
        "total_logical_qubits": int(compiled_dense.num_qubits),
        "ancilla_and_work_qubits": int(compiled_dense.num_qubits - 3),
        "polynomial_degree": config.polynomial_degree,
        "signal_unitary_calls_per_attempt": config.polynomial_degree,
        "projector_phase_operations_per_attempt": config.polynomial_degree + 1,
        "sparse_lookup_calls_per_attempt": "not_applicable_dense_dilation",
        "forward_sparse_lookup_calls_per_attempt": "not_applicable_dense_dilation",
        "inverse_lookup_calls_per_attempt": "not_applicable_dense_dilation",
        "value_rotations_per_attempt": "not_estimated",
        "residual_preparations_per_attempt": 1,
        "postselection_measurements_per_attempt": 1,
        "readout_measurements_per_attempt": 1,
        "transpiled_gate_count": "not_estimated_opaque_dense_unitaries",
        "transpiled_depth": "not_estimated_opaque_dense_unitaries",
        "toffoli_count": "not_estimated_opaque_dense_unitaries",
        "controlled_rotation_count": "not_estimated",
        "one_signal_unitary_gate_count": "not_estimated_opaque_dense_reference",
        "one_signal_unitary_depth": "not_estimated_opaque_dense_reference",
        "one_signal_unitary_cx_count": "not_estimated_opaque_dense_reference",
        "residual_preparation_gate_count": residual_gate_count,
        "total_gate_count_per_attempt": "not_estimated_opaque_dense_unitaries",
        "total_depth_per_attempt": "not_estimated_opaque_dense_unitaries",
        "estimated_attempts_per_accepted_sample": "reported_in_comparison",
        "selected_output_estimate_shots": max(config.shot_counts),
        "estimated_gates_per_selected_output_estimate": "not_estimated_opaque_dense_unitaries",
        "transpilation_target": "qiskit_aer_statevector_default_target_optimization_level_0",
        "operation_counts_json": json.dumps(dense_counts["operation_counts"], sort_keys=True),
    }
    modeled = {
        "resource_category": "modeled_ieee_scale_sparse_access",
        "execution_status": "not_executed_in_this_experiment",
        "configuration_id": config.configuration_id,
        "matrix_fingerprint": "not_applicable_full_ieee_matrix_differs",
        "matrix_shape": "not_executed",
        "matrix_nonzeros": "not_estimated_here",
        "maximum_row_sparsity": "not_estimated_here",
        "maximum_encoded_row_or_column_degree": "not_estimated_here",
        "value_bits": config.matrix_value_bits,
        "value_sign_bits": 1,
        "value_total_bits": config.matrix_value_bits + 1,
        "row_register_qubits": "not_estimated_here",
        "slot_register_qubits": "not_estimated_here",
        "column_register_qubits": "not_estimated_here",
        "index_register_alias": "modeled only; see existing hardware-aware oracle ledger",
        "value_register_qubits": "not_estimated_here",
        "work_qubits": "not_estimated_here",
        "postselection_flag_qubits": "not_estimated_here",
        "signal_qubits": "not_estimated_here",
        "readout_qubits": "not_estimated_here",
        "total_logical_qubits": "not_estimated_here",
        "ancilla_and_work_qubits": "not_estimated_here",
        "polynomial_degree": "not_executed",
        "signal_unitary_calls_per_attempt": "not_executed",
        "projector_phase_operations_per_attempt": "not_executed",
        "sparse_lookup_calls_per_attempt": "not_executed",
        "forward_sparse_lookup_calls_per_attempt": "not_executed",
        "inverse_lookup_calls_per_attempt": "not_executed",
        "value_rotations_per_attempt": "not_executed",
        "residual_preparations_per_attempt": "not_executed",
        "postselection_measurements_per_attempt": "not_executed",
        "readout_measurements_per_attempt": "not_executed",
        "transpiled_gate_count": "not_estimated",
        "transpiled_depth": "not_estimated",
        "toffoli_count": "not_estimated",
        "controlled_rotation_count": "not_estimated",
        "one_signal_unitary_gate_count": "not_estimated",
        "one_signal_unitary_depth": "not_estimated",
        "one_signal_unitary_cx_count": "not_estimated",
        "residual_preparation_gate_count": "not_estimated",
        "total_gate_count_per_attempt": "not_estimated",
        "total_depth_per_attempt": "not_estimated",
        "estimated_attempts_per_accepted_sample": "not_estimated",
        "selected_output_estimate_shots": "not_applicable",
        "estimated_gates_per_selected_output_estimate": "not_estimated",
        "transpilation_target": "not_applicable_modeled_category",
        "operation_counts_json": "not_estimated",
        "source": "existing outputs/hardware_aware_oracle_cost_model (not copied or extrapolated)",
    }
    records = [executed_sparse, executed_dense, modeled]
    return records, {
        "sparse": sparse_counts,
        "dense_opaque_instruction_diagnostic": {
            **dense_counts,
            "reporting_status": (
                "not a full gate-level resource count; dense UnitaryGate instructions remain"
            ),
        },
        "one_sparse_signal": {
            "operation_counts": wrapper_counts,
            "gate_count": int(sum(wrapper_counts.values())),
            "depth": int(wrapper_transpiled.depth()),
        },
    }


def inputs_statevector_postselection(
    inputs: DefaultSparseIntegratedInputs, bundle: IntegratedSparseQSVTCircuit
) -> float:
    """Compute the direct sparse encoded-prefix probability for resource normalization."""

    from qiskit.quantum_info import Statevector

    n = inputs.matrix_quantized.shape[1]
    operator = build_structured_qsvt_operator_circuit(
        bundle.wrapper.unitary, inputs.phases, encoded_dimension=n
    )
    initial = np.zeros(bundle.wrapper.unitary.shape[0], dtype=np.complex128)
    initial[:n] = inputs.residual / np.linalg.norm(inputs.residual)
    evolved = Statevector(initial).evolve(operator.qsvt_operator_circuit).data
    return float(np.vdot(evolved[:n], evolved[:n]).real)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _statevector_rows(
    inputs: DefaultSparseIntegratedInputs,
    validation: StatevectorValidation,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = validation.metrics
    for functional_id, functional in inputs.selected_functionals.items():
        ell = np.asarray(functional, dtype=np.float64)
        ridge_value = float(ell @ validation.quantized_ridge_update)
        sparse_value = float(ell @ validation.sparse_update)
        dense_value = float(ell @ validation.dense_update)
        exact_value = float(ell @ validation.exact_polynomial_svt_update)
        rational_value = float(ell @ validation.exact_rational_svt_update)
        original_value = float(ell @ validation.original_ridge_update)
        rows.append(
            {
                "configuration_id": inputs.config.configuration_id,
                "functional_id": functional_id,
                "matrix_fingerprint": inputs.config.matrix_fingerprint,
                "block_reconstruction_relative_fro_error": common[
                    "block_reconstruction_relative_fro_error"
                ],
                "sparse_dense_action_relative_l2_error": common[
                    "sparse_dense_action_relative_l2_error"
                ],
                "qsvt_exact_polynomial_svt_relative_l2_error": common[
                    "qsvt_exact_polynomial_svt_relative_l2_error"
                ],
                "qsvt_quantized_ridge_relative_l2_error": common[
                    "qsvt_quantized_ridge_relative_l2_error"
                ],
                "selected_output_absolute_error_vs_quantized_ridge": abs(
                    sparse_value - ridge_value
                ),
                "selected_output_relative_error_vs_quantized_ridge": abs(
                    sparse_value - ridge_value
                )
                / max(abs(ridge_value), 1.0e-30),
                "sparse_statevector_selected_output": sparse_value,
                "dense_statevector_selected_output": dense_value,
                "exact_polynomial_svt_selected_output": exact_value,
                "exact_rational_svt_selected_output": rational_value,
                "quantized_ridge_selected_output": ridge_value,
                "original_unquantized_ridge_selected_output": original_value,
                "quantization_and_sparsification_selected_output_difference": abs(
                    ridge_value - original_value
                ),
                "sparse_postselection_probability": common[
                    "sparse_postselection_probability"
                ],
                "dense_postselection_probability": common["dense_postselection_probability"],
                "physical_rescaling_factor_C_over_beta": common[
                    "physical_rescaling_factor_C_over_beta"
                ],
                "residual_norm": common["residual_norm"],
                "selected_functional_norm": float(np.linalg.norm(ell)),
                "selected_output_sign": int(np.sign(sparse_value)),
                "sparse_statevector_reference": json.dumps(
                    _json_ready(validation.sparse_encoded_state), sort_keys=True
                ),
                "dense_statevector_reference": json.dumps(
                    _json_ready(validation.dense_encoded_state), sort_keys=True
                ),
                "primary_reference": "Ridge on the identical quantized sparse matrix",
            }
        )
    return rows


def _summarize_finite_shots(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["chain_type", "functional_id", "shots_attempted"], sort=True)
    for (chain_type, functional_id, shots), group in grouped:
        estimates = group["selected_output_estimate"].to_numpy(dtype=np.float64)
        analytic = group["analytic_standard_error"].to_numpy(dtype=np.float64)
        expected = group["statevector_expected_standard_error"].to_numpy(dtype=np.float64)
        statevector = float(group["statevector_reference"].iloc[0])
        ridge = float(group["quantized_ridge_reference"].iloc[0])
        seed_std = float(np.std(estimates, ddof=1)) if estimates.size > 1 else float("nan")
        uncertainty_mean = seed_std / math.sqrt(estimates.size) if estimates.size > 1 else math.nan
        empirical_variance = float(np.var(estimates, ddof=1)) if estimates.size > 1 else math.nan
        expected_variance = float(np.mean(expected**2))
        variance_ratio = (
            empirical_variance / expected_variance if expected_variance > 0 else math.nan
        )
        mean_estimate = float(np.mean(estimates))
        coverage = float(
            np.mean(
                (group["confidence_interval_lower"].to_numpy(dtype=np.float64) <= statevector)
                & (
                    statevector
                    <= group["confidence_interval_upper"].to_numpy(dtype=np.float64)
                )
            )
        )
        rows.append(
            {
                "configuration_id": str(group["configuration_id"].iloc[0]),
                "chain_type": str(chain_type),
                "functional_id": str(functional_id),
                "shots": int(shots),
                "num_seeds": int(group["seed"].nunique()),
                "mean_measured_postselection_probability": float(
                    group["measured_postselection_probability"].mean()
                ),
                "std_measured_postselection_probability_across_seeds": float(
                    group["measured_postselection_probability"].std(ddof=1)
                ),
                "mean_interference_acceptance_probability": float(
                    group["interference_acceptance_probability"].mean()
                ),
                "mean_selected_output_estimate": mean_estimate,
                "selected_output_std_across_seeds": seed_std,
                "mean_analytic_standard_error_one_estimate": float(np.mean(analytic)),
                "uncertainty_of_mean_across_seeds": uncertainty_mean,
                "mean_confidence_interval_lower": mean_estimate
                - 1.959963984540054 * uncertainty_mean,
                "mean_confidence_interval_upper": mean_estimate
                + 1.959963984540054 * uncertainty_mean,
                "statevector_reference": statevector,
                "quantized_ridge_reference": ridge,
                "absolute_error_of_mean_vs_quantized_ridge": abs(mean_estimate - ridge),
                "relative_error_of_mean_vs_quantized_ridge": abs(mean_estimate - ridge)
                / max(abs(ridge), 1.0e-30),
                "absolute_error_of_mean_vs_statevector": abs(mean_estimate - statevector),
                "empirical_variance_across_seeds": empirical_variance,
                "analytic_variance_one_estimate": expected_variance,
                "empirical_to_analytic_variance_ratio": variance_ratio,
                "variance_consistency_status": (
                    "consistent_with_10_seed_monte_carlo"
                    if np.isfinite(variance_ratio) and 0.1 <= variance_ratio <= 4.0
                    else "outside_declared_10_seed_consistency_band"
                ),
                "statevector_95pct_ci_coverage_across_seeds": coverage,
                "mean_shots_attempted": float(group["shots_attempted"].mean()),
                "mean_postselection_accepted_direct_chain": float(
                    group["postselection_accepted"].mean()
                ),
                "mean_readout_accepted_interference_chain": float(
                    group["readout_accepted"].mean()
                ),
                "uncertainty_note": (
                    "analytic_standard_error is for one finite-shot estimate; seed std is "
                    "variation across seeds; uncertainty_of_mean divides seed std by sqrt(K)"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_dense_circuits(
    inputs: DefaultSparseIntegratedInputs,
    functional: np.ndarray,
) -> tuple[Any, Any]:
    normalized_matrix = inputs.matrix_quantized.T / inputs.config.beta
    encoding = canonical_square_block_encoding(normalized_matrix, tolerance=1.0e-8)
    residual_padded = np.zeros(encoding.unitary.shape[0], dtype=np.complex128)
    residual_padded[: inputs.residual.size] = inputs.residual / np.linalg.norm(inputs.residual)
    integrated, _accounting = build_dense_integrated_readout_circuit(
        block_unitary=encoding.unitary,
        phases=inputs.phases,
        padded_residual=residual_padded,
        functional_unit=functional / np.linalg.norm(functional),
    )
    direct = build_dense_direct_chain_circuit(
        block_unitary=encoding.unitary,
        phases=inputs.phases,
        padded_residual=residual_padded,
    )
    return integrated, direct


def _finite_shot_rows(
    inputs: DefaultSparseIntegratedInputs,
    validation: StatevectorValidation,
    sparse_bundles: dict[str, IntegratedSparseQSVTCircuit],
    dense_circuits: dict[str, Any],
    sparse_compiled: dict[str, tuple[Any, Any]],
    dense_compiled: dict[str, tuple[Any, Any]],
    sparse_direct_compiled: tuple[Any, Any],
    dense_direct_compiled: tuple[Any, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct_cache: dict[tuple[str, int, int], tuple[int, float]] = {}
    for chain_type, direct_pair in (
        ("sparse", sparse_direct_compiled),
        ("dense", dense_direct_compiled),
    ):
        direct_circuit, direct_simulator = direct_pair
        for shots in inputs.config.shot_counts:
            for seed in inputs.config.seeds:
                counts = sample_aer_counts(
                    direct_circuit, direct_simulator, shots=shots, seed=seed
                )
                direct_cache[(chain_type, shots, seed)] = _direct_postselection_estimate(counts)

    for chain_type, compiled_by_functional in (
        ("sparse", sparse_compiled),
        ("dense", dense_compiled),
    ):
        encoded_state = (
            validation.sparse_encoded_state
            if chain_type == "sparse"
            else validation.dense_encoded_state
        )
        update = validation.sparse_update if chain_type == "sparse" else validation.dense_update
        p_post = float(np.vdot(encoded_state, encoded_state).real)
        for functional_id, functional in inputs.selected_functionals.items():
            ell = np.asarray(functional, dtype=np.float64)
            ell_norm = float(np.linalg.norm(ell))
            ell_unit = ell / ell_norm
            physical_scale = (
                inputs.config.contraction_c
                / inputs.config.beta
                * float(np.linalg.norm(inputs.residual))
                * ell_norm
            )
            exact_z = float(np.real(np.vdot(ell_unit, encoded_state)))
            exact_f = (1.0 + p_post) / 2.0
            statevector_value = float(ell @ update)
            ridge_value = float(ell @ validation.quantized_ridge_update)
            compiled, simulator = compiled_by_functional[functional_id]
            for shots in inputs.config.shot_counts:
                expected_se = physical_scale * math.sqrt(
                    max(exact_f - exact_z**2, 0.0) / shots
                )
                for seed in inputs.config.seeds:
                    counts = sample_aer_counts(compiled, simulator, shots=shots, seed=seed)
                    estimate = estimate_signed_selected_output(
                        counts, physical_scale=physical_scale
                    )
                    post_accepted, measured_post = direct_cache[(chain_type, shots, seed)]
                    selected = estimate["selected_output_estimate"]
                    rows.append(
                        {
                            "configuration_id": inputs.config.configuration_id,
                            "chain_type": chain_type,
                            "functional_id": functional_id,
                            "shots_attempted": int(shots),
                            "direct_postselection_shots_attempted": int(shots),
                            "seed": int(seed),
                            "backend": "qiskit_aer_statevector_actual_shot_sampling",
                            "postselection_accepted": int(post_accepted),
                            "readout_accepted": int(estimate["readout_accepted"]),
                            "measured_postselection_probability": measured_post,
                            "interference_acceptance_probability": estimate[
                                "interference_acceptance_probability"
                            ],
                            "inferred_postselection_probability_from_branch": estimate[
                                "inferred_postselection_probability_from_branch"
                            ],
                            "branch_probability": 0.5,
                            "quadrature": "real",
                            "quadrature_probability": 1.0,
                            "readout_sign_mean_accepted": estimate[
                                "readout_sign_mean_accepted"
                            ],
                            "signed_overlap_estimate": estimate["signed_overlap_estimate"],
                            "selected_output_estimate": selected,
                            "absolute_error_vs_quantized_ridge": abs(selected - ridge_value),
                            "relative_error_vs_quantized_ridge": abs(selected - ridge_value)
                            / max(abs(ridge_value), 1.0e-30),
                            "absolute_error_vs_statevector": abs(
                                selected - statevector_value
                            ),
                            "analytic_standard_error": estimate["analytic_standard_error"],
                            "statevector_expected_standard_error": expected_se,
                            "confidence_interval_lower": estimate[
                                "confidence_interval_lower"
                            ],
                            "confidence_interval_upper": estimate[
                                "confidence_interval_upper"
                            ],
                            "statevector_reference": statevector_value,
                            "quantized_ridge_reference": ridge_value,
                            "statevector_postselection_probability": p_post,
                            "physical_recovery_scale": physical_scale,
                            "output_state_used_for_preparation": False,
                        }
                    )
    return rows


def _dense_sparse_comparison(
    inputs: DefaultSparseIntegratedInputs,
    validation: StatevectorValidation,
    finite_summary: pd.DataFrame,
    resources: list[dict[str, Any]],
) -> pd.DataFrame:
    resource_by_category = {row["resource_category"]: row for row in resources}
    sparse_resource = resource_by_category["executed_small_scale_sparse_integrated"]
    dense_resource = resource_by_category["executed_small_scale_dense_integrated"]
    rows: list[dict[str, Any]] = []
    max_shots = max(inputs.config.shot_counts)
    for functional_id, functional in inputs.selected_functionals.items():
        ell = np.asarray(functional, dtype=np.float64)
        sparse_shot = finite_summary[
            (finite_summary["chain_type"] == "sparse")
            & (finite_summary["functional_id"] == functional_id)
            & (finite_summary["shots"] == max_shots)
        ].iloc[0]
        dense_shot = finite_summary[
            (finite_summary["chain_type"] == "dense")
            & (finite_summary["functional_id"] == functional_id)
            & (finite_summary["shots"] == max_shots)
        ].iloc[0]
        ridge_value = float(ell @ validation.quantized_ridge_update)
        sparse_value = float(ell @ validation.sparse_update)
        dense_value = float(ell @ validation.dense_update)
        rows.append(
            {
                "configuration_id": inputs.config.configuration_id,
                "functional_id": functional_id,
                "matrix_fingerprint_sparse": inputs.config.matrix_fingerprint,
                "matrix_fingerprint_dense": inputs.config.matrix_fingerprint,
                "polynomial_degree_sparse": inputs.config.polynomial_degree,
                "polynomial_degree_dense": inputs.config.polynomial_degree,
                "postselection_probability_sparse": validation.metrics[
                    "sparse_postselection_probability"
                ],
                "postselection_probability_dense": validation.metrics[
                    "dense_postselection_probability"
                ],
                "statevector_selected_output_sparse": sparse_value,
                "statevector_selected_output_dense": dense_value,
                "finite_shot_budget": max_shots,
                "finite_shot_selected_output_sparse": float(
                    sparse_shot["mean_selected_output_estimate"]
                ),
                "finite_shot_selected_output_dense": float(
                    dense_shot["mean_selected_output_estimate"]
                ),
                "quantized_ridge_reference": ridge_value,
                "statevector_error_vs_quantized_ridge_sparse": abs(
                    sparse_value - ridge_value
                ),
                "statevector_error_vs_quantized_ridge_dense": abs(dense_value - ridge_value),
                "finite_shot_error_vs_quantized_ridge_sparse": abs(
                    float(sparse_shot["mean_selected_output_estimate"]) - ridge_value
                ),
                "finite_shot_error_vs_quantized_ridge_dense": abs(
                    float(dense_shot["mean_selected_output_estimate"]) - ridge_value
                ),
                "logical_qubits_sparse": sparse_resource["total_logical_qubits"],
                "logical_qubits_dense": dense_resource["total_logical_qubits"],
                "ancilla_work_qubits_sparse": sparse_resource["ancilla_and_work_qubits"],
                "ancilla_work_qubits_dense": dense_resource["ancilla_and_work_qubits"],
                "circuit_depth_sparse": sparse_resource["transpiled_depth"],
                "circuit_depth_dense": dense_resource["transpiled_depth"],
                "gate_count_sparse": sparse_resource["transpiled_gate_count"],
                "gate_count_dense": dense_resource["transpiled_gate_count"],
                "toffoli_count_sparse": sparse_resource["toffoli_count"],
                "toffoli_count_dense": dense_resource["toffoli_count"],
                "controlled_rotations_sparse": sparse_resource[
                    "controlled_rotation_count"
                ],
                "controlled_rotations_dense": dense_resource["controlled_rotation_count"],
                "residual_preparation_gates_sparse": sparse_resource[
                    "residual_preparation_gate_count"
                ],
                "residual_preparation_gates_dense": dense_resource[
                    "residual_preparation_gate_count"
                ],
            }
        )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_manifest_and_checksums(output_dir: str | Path) -> None:
    """Refresh the self-contained manifest and a shasum-compatible checksum ledger."""

    directory = Path(output_dir)
    manifest_path = directory / "manifest.json"
    checksum_path = directory / "checksums.sha256"
    artifact_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"}
    )
    manifest = {
        "experiment_id": "sparse_integrated_chain",
        "configuration_id": CONFIGURATION_ID,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_tier": "executed_statevector_and_sampled_counts_small_scale",
        "changes_estimator_behavior": False,
        "fabricates_results": False,
        "output_state_used_for_preparation": False,
        "artifacts": [str(path) for path in artifact_paths],
        "artifact_checksums": {path.name: _sha256_file(path) for path in artifact_paths},
        "key_package_versions": package_versions(
            ["numpy", "pandas", "scipy", "pennylane", "qiskit", "qiskit-aer", "pypower"]
        ),
    }
    write_json(manifest_path, manifest)
    checked_paths = [*artifact_paths, manifest_path]
    checksum_path.write_text(
        "".join(f"{_sha256_file(path)}  {path}\n" for path in sorted(checked_paths)),
        encoding="utf-8",
    )


def run_sparse_integrated_chain(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the complete sparse/dense statevector and finite-shot experiment."""

    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", DEFAULT_OUTPUT_DIR)))
    shots = tuple(int(value) for value in resolved.get("shot_counts", DEFAULT_SHOT_COUNTS))
    seeds = tuple(int(value) for value in resolved.get("seeds", DEFAULT_SEEDS))
    inputs = build_default_sparse_integrated_inputs(
        output_dir, shot_counts=shots, seeds=seeds
    )
    np.save(inputs.config.matrix_path, inputs.matrix_quantized)
    np.save(inputs.config.residual_path, inputs.residual)
    np.save(output_dir / "phases.npy", inputs.phases)
    np.save(
        output_dir / "polynomial_coefficients.npy",
        np.asarray(inputs.target.coefficients, dtype=np.float64),
    )
    write_json(output_dir / "configuration.json", _json_ready(inputs.config.to_json_dict()))

    matrix_metadata = {
        "configuration_id": inputs.config.configuration_id,
        "source": inputs.config.matrix_source,
        "case": inputs.config.case_name,
        "selected_rows": inputs.config.selected_rows,
        "selected_columns": inputs.config.selected_columns,
        "shape": inputs.config.matrix_shape,
        "quantization_magnitude_bits": inputs.config.matrix_value_bits,
        "quantization_sign_bits": 1,
        "matrix_fingerprint_quantized": inputs.config.matrix_fingerprint,
        "matrix_fingerprint_sparsified": stable_array_fingerprint(
            inputs.matrix_sparsified
        ),
        "matrix_fingerprint_original": stable_array_fingerprint(inputs.matrix_original),
        "residual_fingerprint": inputs.config.residual_fingerprint,
        "normalization": inputs.config.normalization,
        "alpha": inputs.config.alpha,
        "beta": inputs.config.beta,
        "lambda": inputs.config.normalized_lambda,
        "C": inputs.config.contraction_c,
        "polynomial_degree": inputs.config.polynomial_degree,
        "phase_count": int(inputs.phases.size),
        "phase_fingerprint": stable_array_fingerprint(inputs.phases),
        "row_nonzeros": np.count_nonzero(inputs.matrix_quantized, axis=1),
        "column_nonzeros": np.count_nonzero(inputs.matrix_quantized, axis=0),
        **inputs.source_metadata,
        "quantization_and_sparsification_are_reported_not_hidden": True,
    }
    write_json(output_dir / "matrix_metadata.json", _json_ready(matrix_metadata))
    functionals_metadata = {
        "selection_timing": "fixed in implementation_audit.md before solving",
        "physical_semantics": "deterministic linear functionals; no fabricated bus semantics",
        "functionals": [
            {
                "functional_id": name,
                "vector": vector.tolist(),
                "norm": float(np.linalg.norm(vector)),
                "kind": (
                    "coordinate"
                    if name == "coordinate_e0"
                    else "signed_difference"
                    if name.startswith("signed_difference")
                    else "normalized_aggregate"
                ),
            }
            for name, vector in inputs.selected_functionals.items()
        ],
    }
    write_json(output_dir / "selected_functionals.json", functionals_metadata)

    sparse_bundles = {
        name: build_integrated_sparse_selected_output_circuit(
            inputs.config,
            matrix=inputs.matrix_quantized,
            residual=inputs.residual,
            selected_functional=functional,
            phases=inputs.phases,
        )
        for name, functional in inputs.selected_functionals.items()
    }
    primary_bundle = sparse_bundles[inputs.config.selected_output_name]
    validation = statevector_validate_integrated_chain(inputs, primary_bundle)
    for bundle in sparse_bundles.values():
        assert_no_direct_output_initializer(bundle.circuit, validation.sparse_encoded_state)

    statevector_frame = pd.DataFrame(_statevector_rows(inputs, validation))
    statevector_frame.to_csv(output_dir / "statevector_validation.csv", index=False)

    dense_integrated: dict[str, Any] = {}
    dense_direct = None
    for name, functional in inputs.selected_functionals.items():
        integrated, direct = _build_dense_circuits(inputs, functional)
        dense_integrated[name] = integrated
        if dense_direct is None:
            dense_direct = direct
    if dense_direct is None:  # pragma: no cover - deterministic nonempty functionals
        raise RuntimeError("no dense direct circuit was built")

    exact_circuit_checks: dict[str, Any] = {}
    for name, bundle in sparse_bundles.items():
        distribution = exact_joint_distribution(
            bundle.circuit,
            postselection_flag_qubit=bundle.register_layout["postselection_flag_qubit"],
            readout_qubit=bundle.register_layout["readout_qubit"],
        )
        ell = inputs.selected_functionals[name]
        ell_unit = ell / np.linalg.norm(ell)
        exact_z = float(np.real(np.vdot(ell_unit, validation.sparse_encoded_state)))
        exact_acceptance = (1.0 + validation.metrics["sparse_postselection_probability"]) / 2
        observed_acceptance = distribution["00"] + distribution["10"]
        observed_z = distribution["00"] - distribution["10"]
        if abs(observed_acceptance - exact_acceptance) > 1.0e-9:
            raise RuntimeError("integrated sparse postselection branch validation failed")
        if abs(observed_z - exact_z) > 1.0e-9:
            raise RuntimeError("integrated sparse signed-overlap validation failed")
        exact_circuit_checks[name] = {
            "joint_probabilities_c1c0": distribution,
            "acceptance_validation_error": abs(observed_acceptance - exact_acceptance),
            "signed_overlap_validation_error": abs(observed_z - exact_z),
        }

    sparse_compiled: dict[str, tuple[Any, Any]] = {}
    dense_compiled: dict[str, tuple[Any, Any]] = {}
    for name in inputs.selected_functionals:
        sparse_compiled[name] = compile_for_aer(sparse_bundles[name].circuit)
        dense_compiled[name] = compile_for_aer(dense_integrated[name])
    sparse_direct_compiled = compile_for_aer(primary_bundle.direct_postselection_circuit)
    dense_direct_compiled = compile_for_aer(dense_direct)

    finite_rows = _finite_shot_rows(
        inputs,
        validation,
        sparse_bundles,
        dense_integrated,
        sparse_compiled,
        dense_compiled,
        sparse_direct_compiled,
        dense_direct_compiled,
    )
    finite_frame = pd.DataFrame(finite_rows)
    finite_summary = _summarize_finite_shots(finite_frame)
    finite_frame.to_csv(output_dir / "finite_shot_results.csv", index=False)
    finite_summary.to_csv(output_dir / "finite_shot_summary.csv", index=False)

    resource_records, transpile_metadata = build_executed_resource_records(
        inputs,
        primary_bundle,
        compiled_sparse=sparse_compiled[inputs.config.selected_output_name][0],
        compiled_dense=dense_compiled[inputs.config.selected_output_name][0],
    )
    resource_frame = pd.DataFrame(resource_records)
    resource_frame.to_csv(output_dir / "resource_ledger.csv", index=False)
    write_json(
        output_dir / "resource_ledger.json",
        _json_ready(
            {
                "configuration_id": inputs.config.configuration_id,
                "records": resource_records,
                "definitions": {
                    "executed": "counts from actual small-scale circuits compiled to Aer",
                    "modeled": "separate category; no executed 8x8 counts extrapolated",
                    "sparse_lookup_calls_per_attempt": (
                        "all U_A or U_A_dagger wrapper invocations; inverse calls are a subset"
                    ),
                    "controlled_rotation_count": (
                        "24 stored slot-column rotations per wrapper call times degree 31"
                    ),
                    "not_estimated": "unknown/unavailable values are never encoded as zero",
                },
            }
        ),
    )
    comparison = _dense_sparse_comparison(
        inputs, validation, finite_summary, resource_records
    )
    comparison.to_csv(output_dir / "dense_sparse_comparison.csv", index=False)

    circuit_metadata = {
        "configuration_id": inputs.config.configuration_id,
        "architecture": (
            "branch-Hadamard residual preparation -> controlled sparse wrapper QSVT -> "
            "aggregate sparse-work postselection flag -> signed selected-output measurement"
        ),
        "register_layout": primary_bundle.register_layout,
        "operation_counts": primary_bundle.operation_counts,
        "wrapper": {
            "slots": primary_bundle.wrapper.slots,
            "normalization_factor": primary_bundle.wrapper.normalization_factor,
            "raw_gate_count": primary_bundle.wrapper.gate_count,
            "raw_depth": primary_bundle.wrapper.depth,
            "top_left_reconstruction_error_max_abs": (
                primary_bundle.wrapper.top_left_reconstruction_error
            ),
            "slot_assignment": primary_bundle.wrapper.assignment.to_metadata(),
            "sparse_lookup_realization": (
                "multiplexed stored-value rotations plus slot-controlled in-place "
                "matching permutations and inverse slot diffusion"
            ),
            "separate_value_register": False,
        },
        "qsvt": {
            "degree": inputs.config.polynomial_degree,
            "phase_count": int(inputs.phases.size),
            "phase_convention": inputs.config.phase_convention,
            "signal_extraction": "real encoded top-left quadrature",
        },
        "postselection": {
            "event": "aggregate flag 0 iff slot=00 and rotation ancilla=0",
            "direct_p_post_is_separate_from_interference_acceptance": True,
        },
        "readout": {
            "event": "readout ancilla after closing Hadamard",
            "bit_order": "c1c0: left=readout sign, right=aggregate postselection flag",
            "recovery": "(C/beta)*||r||*||ell||*(N_00-N_10)/shots",
            "branch_probability": 0.5,
            "quadrature": "real",
        },
        "output_state_used_for_preparation": False,
        "preparation_records": primary_bundle.preparation_records,
        "guard_test": "assert_no_direct_output_initializer",
        "exact_integrated_circuit_checks": exact_circuit_checks,
        "transpilation": transpile_metadata,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(output_dir / "circuit_metadata.json", _json_ready(circuit_metadata))

    primary_state = statevector_frame[
        statevector_frame["functional_id"] == inputs.config.selected_output_name
    ].iloc[0]
    primary_shots = finite_summary[
        (finite_summary["chain_type"] == "sparse")
        & (finite_summary["functional_id"] == inputs.config.selected_output_name)
        & (finite_summary["shots"] == max(inputs.config.shot_counts))
    ].iloc[0]
    verification_report = "\n".join(
        [
            "# Sparse Integrated Chain Verification Report",
            "",
            f"- Configuration: `{inputs.config.configuration_id}`",
            f"- Execution: {CLAIM_BOUNDARY}",
            f"- Relative sparse block reconstruction error: "
            f"{validation.metrics['block_reconstruction_relative_fro_error']:.6e}",
            f"- Sparse-versus-dense action error: "
            f"{validation.metrics['sparse_dense_action_relative_l2_error']:.6e}",
            f"- QSVT-versus-exact polynomial SVT error: "
            f"{validation.metrics['qsvt_exact_polynomial_svt_relative_l2_error']:.6e}",
            f"- QSVT-versus-quantized-Ridge error: "
            f"{validation.metrics['qsvt_quantized_ridge_relative_l2_error']:.6e}",
            f"- Sparse postselection probability: "
            f"{validation.metrics['sparse_postselection_probability']:.10f}",
            f"- Primary statevector selected output: "
            f"{float(primary_state['sparse_statevector_selected_output']):.12e}",
            f"- Primary quantized Ridge output: "
            f"{float(primary_state['quantized_ridge_selected_output']):.12e}",
            f"- Primary {max(inputs.config.shot_counts)}-shot mean over "
            f"{len(inputs.config.seeds)} seeds: "
            f"{float(primary_shots['mean_selected_output_estimate']):.12e}",
            "",
            "## Statistical interpretation",
            "",
            "Each per-seed analytic standard error describes one finite-shot estimate. "
            "The across-seed standard deviation and the standard error of the seed mean "
            "are stored separately. The finite-shot counts are produced by actual Aer "
            "sampling of the measured circuits; exact statevector distributions are used "
            "only for validation and analytic variance references.",
            "",
            "## Verification command status",
            "",
            "Command results are appended after the repository verification commands run.",
            "",
        ]
    )
    (output_dir / "verification_report.md").write_text(
        verification_report, encoding="utf-8"
    )
    refresh_manifest_and_checksums(output_dir)
    return {
        "output_dir": output_dir,
        "inputs": inputs,
        "statevector": validation,
        "statevector_frame": statevector_frame,
        "finite_shot_results": finite_frame,
        "finite_shot_summary": finite_summary,
        "dense_sparse_comparison": comparison,
        "resource_records": resource_records,
        "circuit_metadata": circuit_metadata,
        "sparse_bundles": sparse_bundles,
    }
