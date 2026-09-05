"""Error-source ablation and precision-resource study for the integrated sparse QSVT chain.

The study quantifies, on the verified ``ieee14_sparse_quantized_8x8_d31_selected_v1``
workload, the incremental effect of sparsification, matrix-value quantization, QSVT
polynomial/phase implementation, and finite-shot readout along the declared path

    H_original -> H_sparse_exact -> H_sparse_quantized -> QSVT statevector -> finite shots,

together with the executed circuit resources of every sampled configuration.  One frozen
QSVT design (alpha, beta, lambda, C, degree-31 coefficients, full-precision phases) is
used for every primary sweep point; nothing is retuned per matrix.  All heavy reusable
components (quantizer, sparse wrapper, integrated chain, Aer sampling, Ridge reference)
come from the existing verified modules and are not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    CompleteWrapperResult,
    validate_complete_wrapper,
)
from robust_qsvt_se.paper.tqe_revision_support_common import (
    git_commit_hash,
    now_iso,
    package_versions,
)
from robust_qsvt_se.qsvt.bipartite_slot_assignment import (
    minimum_slot_count,
    validate_slot_assignment,
)
from robust_qsvt_se.qsvt.engineering_utils import ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    build_structured_qsvt_operator_circuit,
    qsvt_sequence_operation_counts,
)
from robust_qsvt_se.qsvt.sparse_block_encoding_wrapper import quantize_sign_magnitude
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    DefaultSparseIntegratedInputs,
    _as_quantized_block,
    _direct_postselection_estimate,
    assert_no_direct_output_initializer,
    build_default_sparse_integrated_inputs,
    build_integrated_sparse_selected_output_circuit,
    compile_for_aer,
    estimate_signed_selected_output,
    exact_joint_distribution,
    sample_aer_counts,
    stable_array_fingerprint,
    statevector_validate_integrated_chain,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

STUDY_ID = "sparse_error_precision_study_v1"
BASELINE_CONFIGURATION_ID = "ieee14_sparse_quantized_8x8_d31_selected_v1"
DEFAULT_STUDY_DIR = Path("outputs/sparse_error_precision_study")
DEFAULT_CONFIG_PATH = Path("configs/sparse_error_precision_study.json")

VALUE_BITS_SWEEP = (4, 6, 8, 10, 12)
EXACT_VALUE_KEY = "exact"
PHASE_BITS_SWEEP = (8, 10, 12, 16, 24)
FULL_PHASE_KEY = "full"
# Sentinel used only for numeric plotting/dominance ordering of the unquantized points
# (float64 mantissa width); it is documentation, not a synthesized register width.
FULL_PRECISION_NUMERIC_SENTINEL = 53.0

FUNCTIONAL_IDS = (
    "coordinate_e0",
    "signed_difference_e0_minus_e1",
    "aggregate_e0_to_e3",
)
PRIMARY_FUNCTIONAL_ID = "coordinate_e0"

# Predeclared finite-shot subset (never replaced after observing results).
FINITE_SHOT_CONFIGURATIONS = (
    {"label": "baseline", "value_bits": "6", "phase_bits": "full"},
    {"label": "low_precision", "value_bits": "4", "phase_bits": "8"},
    {"label": "medium_precision", "value_bits": "8", "phase_bits": "12"},
    {"label": "high_precision", "value_bits": "12", "phase_bits": "16"},
    {"label": "value_limited", "value_bits": "4", "phase_bits": "full"},
    {"label": "phase_limited", "value_bits": "12", "phase_bits": "8"},
)
SHOT_BUDGETS = (10_000, 100_000, 1_000_000)
SEED_COUNT = 10

STAGES = (
    "audit",
    "baseline",
    "matrices",
    "statevector",
    "finite-shot",
    "resources",
    "sparsity-extension",
    "pareto",
    "verify",
)

SPARSITY_BUDGETS = (8, 12, 16, 24, 32, 64)

FLOAT_REL_TOLERANCE = 1.0e-9
FINGERPRINT_TOLERANCE = "exact"
IDENTITY_ABS_TOLERANCE = 1.0e-12
TRIANGLE_ABS_TOLERANCE = 1.0e-12
BLOCK_RECONSTRUCTION_TOLERANCE = 1.0e-9
CIRCUIT_STATEVECTOR_CROSSCHECK_TOLERANCE = 1.0e-9
CI_Z = 1.959963984540054

FAILURE_CLASSES = (
    "normalization_violation",
    "phase_quantization_invalid",
    "block_reconstruction_failure",
    "slot_assignment_failure",
    "statevector_memory_limit",
    "transpilation_limit",
    "finite_shot_runtime_limit",
    "numerical_instability",
    "unsupported_precision",
    "other_verified_failure",
)

DIRECT_ROTATION_LIMITATION = "not_estimated_under_direct_rotation_architecture"

CLAIM_BOUNDARY = (
    "Executed 8x8 error-source ablation and precision-resource study on a classical "
    "simulator using the repository's verified sparse integrated chain. Value and phase "
    "bit widths change stored rotation parameters, not synthesized gate counts, under the "
    "direct multiplexed-rotation architecture. This is not an IEEE-scale result, not a "
    "hardware or fault-tolerant estimate, and not a speedup or practical-competitiveness "
    "claim."
)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON durably: temp file in the same directory, fsync, atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def precision_key_to_numeric(key: str) -> float:
    """Numeric precision for ordering; unquantized points map to the documented sentinel."""

    if key in (EXACT_VALUE_KEY, FULL_PHASE_KEY):
        return FULL_PRECISION_NUMERIC_SENTINEL
    return float(int(key))


def _close(a: float, b: float, rel: float = FLOAT_REL_TOLERANCE) -> bool:
    return bool(math.isclose(float(a), float(b), rel_tol=rel, abs_tol=rel))


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


class StudyCheckpoint:
    """Durable per-unit checkpoint store: one atomic JSON part per completed unit."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "checkpoint.json"
        self.parts_dir = self.output_dir / "checkpoint_parts"

    def _part_path(self, stage: str, key: str) -> Path:
        safe = key.replace("/", "_").replace(" ", "_")
        return self.parts_dir / stage / f"{safe}.json"

    def load_part(self, stage: str, key: str) -> dict[str, Any] | None:
        """Return the payload only for a verified completed part; else None (rerun)."""

        path = self._part_path(stage, key)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(record, dict) or record.get("status") != "completed":
            return None
        if record.get("key") != key:
            return None
        return record.get("payload")

    def write_part(self, stage: str, key: str, payload: dict[str, Any]) -> None:
        record = {
            "status": "completed",
            "stage": stage,
            "key": key,
            "timestamp": now_iso(),
            "payload": json_ready(payload),
        }
        atomic_write_json(self._part_path(stage, key), record)
        self._touch_state(stage, key)

    def completed_keys(self, stage: str) -> list[str]:
        directory = self.parts_dir / stage
        if not directory.is_dir():
            return []
        keys: list[str] = []
        for path in sorted(directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(record, dict) and record.get("status") == "completed":
                keys.append(str(record.get("key")))
        return keys

    def clear_stage(self, stage: str) -> None:
        directory = self.parts_dir / stage
        if directory.is_dir():
            for path in directory.glob("*.json"):
                path.unlink()
        state = self._read_state()
        state.get("stages", {}).pop(stage, None)
        atomic_write_json(self.state_path, state)

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"study_id": STUDY_ID, "stages": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"study_id": STUDY_ID, "stages": {}}
        if not isinstance(state, dict):
            return {"study_id": STUDY_ID, "stages": {}}
        state.setdefault("stages", {})
        return state

    def _touch_state(self, stage: str, key: str) -> None:
        state = self._read_state()
        entry = state["stages"].setdefault(stage, {"completed_keys": []})
        if key not in entry["completed_keys"]:
            entry["completed_keys"].append(key)
        entry["last_update"] = now_iso()
        atomic_write_json(self.state_path, state)

    def mark_stage_complete(self, stage: str, summary: dict[str, Any] | None = None) -> None:
        state = self._read_state()
        entry = state["stages"].setdefault(stage, {"completed_keys": []})
        entry["stage_status"] = "completed"
        entry["completed_at"] = now_iso()
        if summary:
            entry["summary"] = json_ready(summary)
        atomic_write_json(self.state_path, state)


# --------------------------------------------------------------------------------------
# Frozen design and sweep inputs
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FrozenStudyDesign:
    """One frozen QSVT design and every sweep input derived from the verified baseline."""

    inputs: DefaultSparseIntegratedInputs
    matrix_original: np.ndarray
    matrix_sparse_exact: np.ndarray
    residual: np.ndarray
    matrices_by_value_key: dict[str, np.ndarray]
    phases_by_phase_key: dict[str, np.ndarray]
    functionals: dict[str, np.ndarray]
    alpha: float
    beta: float
    normalized_lambda: float
    contraction_c: float
    degree: int
    coefficients: np.ndarray
    mu: float
    slots: int
    support: np.ndarray

    @property
    def ridge_updates(self) -> dict[str, np.ndarray]:
        """Ridge updates at the single frozen physical alpha for every matrix stage."""

        updates = {
            "original": ridge_svd_solution(self.matrix_original, self.residual, alpha=self.alpha),
            "sparse_exact": ridge_svd_solution(
                self.matrix_sparse_exact, self.residual, alpha=self.alpha
            ),
        }
        for key, matrix in self.matrices_by_value_key.items():
            updates[f"value_bits_{key}"] = ridge_svd_solution(
                matrix, self.residual, alpha=self.alpha
            )
        return updates


def quantize_phase_sequence(phases: np.ndarray, phase_key: str) -> np.ndarray:
    """Deterministic fixed-point phase rounding on a ``2*pi / 2**bits`` grid.

    The rule (declared in the Phase 0 audit before execution): each angle is rounded to
    the nearest multiple of ``2*pi / 2**bits`` with numpy round-half-even.  Ordering,
    length, parity, and projector convention are untouched, and the rounded sequence is
    never refitted.  ``full`` returns the frozen sequence unchanged.
    """

    values = np.asarray(phases, dtype=np.float64)
    if phase_key == FULL_PHASE_KEY:
        return values.copy()
    bits = int(phase_key)
    if bits < 1 or bits > 62:
        raise ValueError(f"unsupported_precision: phase bits {bits} outside [1, 62]")
    if np.max(np.abs(values)) >= math.pi:
        raise ValueError("phase_quantization_invalid: sequence leaves the [-pi, pi) range")
    step = 2.0 * math.pi / float(1 << bits)
    rounded = np.round(values / step) * step
    if not np.all(np.isfinite(rounded)):
        raise ValueError("phase_quantization_invalid: rounded sequence is non-finite")
    return rounded


def build_frozen_design(config_path_dir: str | Path = DEFAULT_STUDY_DIR) -> FrozenStudyDesign:
    """Rebuild the verified baseline inputs and derive every frozen sweep input."""

    inputs = build_default_sparse_integrated_inputs(Path(config_path_dir))
    matrix_original = np.asarray(inputs.matrix_original, dtype=np.float64)
    matrix_sparse_exact = np.asarray(inputs.matrix_sparsified, dtype=np.float64)
    support = matrix_sparse_exact != 0.0
    mu = float(np.max(np.abs(matrix_sparse_exact)))
    slots = minimum_slot_count(np.abs(matrix_sparse_exact.T) > 0.0)
    matrices: dict[str, np.ndarray] = {}
    for bits in VALUE_BITS_SWEEP:
        quantized, mu_bits = quantize_sign_magnitude(matrix_sparse_exact, magnitude_bits=bits)
        if not _close(mu_bits, mu, rel=1.0e-12):
            raise RuntimeError("quantizer full-scale drifted; frozen design assumption broken")
        matrices[str(bits)] = quantized
    matrices[EXACT_VALUE_KEY] = matrix_sparse_exact.copy()
    if not np.array_equal(matrices["6"], inputs.matrix_quantized):
        raise RuntimeError("six-bit sweep matrix does not equal the frozen baseline matrix")
    for key, matrix in matrices.items():
        if not np.array_equal(matrix != 0.0, support):
            raise RuntimeError(f"value precision {key} changed the frozen sparse support")
    phases_full = np.asarray(inputs.phases, dtype=np.float64)
    phase_map: dict[str, np.ndarray] = {
        str(bits): quantize_phase_sequence(phases_full, str(bits)) for bits in PHASE_BITS_SWEEP
    }
    phase_map[FULL_PHASE_KEY] = phases_full.copy()
    config = inputs.config
    beta = float(config.beta)
    for key, matrix in matrices.items():
        spectral = float(np.linalg.svd(matrix, compute_uv=False).max())
        if spectral > beta + 1.0e-9:
            raise RuntimeError(
                f"normalization_violation: ||H_(S,{key})||_2 = {spectral} exceeds frozen beta"
            )
    return FrozenStudyDesign(
        inputs=inputs,
        matrix_original=matrix_original,
        matrix_sparse_exact=matrix_sparse_exact,
        residual=np.asarray(inputs.residual, dtype=np.float64),
        matrices_by_value_key=matrices,
        phases_by_phase_key=phase_map,
        functionals={key: np.asarray(vec, dtype=np.float64) for key, vec in
                     inputs.selected_functionals.items()},
        alpha=float(config.alpha),
        beta=beta,
        normalized_lambda=float(config.normalized_lambda),
        contraction_c=float(config.contraction_c),
        degree=int(config.polynomial_degree),
        coefficients=np.asarray(inputs.target.coefficients, dtype=np.float64),
        mu=mu,
        slots=int(slots),
        support=support,
    )


def sweep_point_configuration_id(value_key: str, phase_key: str) -> str:
    return f"{BASELINE_CONFIGURATION_ID}__bv{value_key}_bp{phase_key}"


def build_point_config(
    design: FrozenStudyDesign, value_key: str, phase_key: str, output_dir: Path
) -> Any:
    """Frozen per-point configuration: only identity/fingerprint fields change."""

    if value_key == EXACT_VALUE_KEY:
        raise ValueError("measured-circuit configs are defined for quantized value points only")
    matrix = design.matrices_by_value_key[value_key]
    matrices_dir = Path(output_dir) / "matrices"
    return replace(
        design.inputs.config,
        configuration_id=sweep_point_configuration_id(value_key, phase_key),
        matrix_fingerprint=stable_array_fingerprint(matrix),
        matrix_value_bits=int(value_key),
        matrix_path=matrices_dir / f"matrix_value_bits_{value_key}.npy",
        residual_path=matrices_dir / "residual.npy",
    )


# --------------------------------------------------------------------------------------
# Statevector evaluation
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class StatevectorPointResult:
    value_key: str
    phase_key: str
    encoded_state: np.ndarray
    update: np.ndarray
    postselection_probability: float
    block_reconstruction_error: float
    qsvt_action_error: float
    wrapper_normalization: float
    status: str
    failure_reason: str


def build_sweep_wrapper(design: FrozenStudyDesign, value_key: str) -> CompleteWrapperResult:
    """Compile the existing sparse wrapper for one sweep matrix (no core changes)."""

    matrix = design.matrices_by_value_key[value_key]
    bits = 53 if value_key == EXACT_VALUE_KEY else int(value_key)
    block = _as_quantized_block(matrix, bits)
    wrapper = validate_complete_wrapper(block, encode_transpose=True, transpile_circuit=False)
    if not _close(wrapper.normalization_factor, design.beta, rel=1.0e-12):
        raise RuntimeError(
            "normalization_violation: wrapper normalization "
            f"{wrapper.normalization_factor} != frozen beta {design.beta}"
        )
    if wrapper.top_left_reconstruction_error > BLOCK_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            "block_reconstruction_failure: error "
            f"{wrapper.top_left_reconstruction_error} above tolerance"
        )
    return wrapper


def evaluate_statevector_point(
    design: FrozenStudyDesign,
    value_key: str,
    phase_key: str,
    wrapper: CompleteWrapperResult | None = None,
) -> StatevectorPointResult:
    """Evolve the residual through the QSVT sequence for one (value, phase) grid point."""

    from qiskit.quantum_info import Statevector

    matrix = design.matrices_by_value_key[value_key]
    n = matrix.shape[1]
    try:
        phases = design.phases_by_phase_key[phase_key]
        if wrapper is None:
            wrapper = build_sweep_wrapper(design, value_key)
        bundle = build_structured_qsvt_operator_circuit(
            wrapper.unitary, phases, encoded_dimension=n
        )
        residual_unit = design.residual / np.linalg.norm(design.residual)
        state = np.zeros(wrapper.unitary.shape[0], dtype=np.complex128)
        state[:n] = residual_unit
        evolved = Statevector(state).evolve(bundle.qsvt_operator_circuit).data
        encoded = np.asarray(evolved[:n], dtype=np.complex128)
        if not np.all(np.isfinite(encoded.view(np.float64))):
            raise FloatingPointError("numerical_instability: non-finite encoded state")
        postselection = float(np.vdot(encoded, encoded).real)
        physical_scale = design.contraction_c / design.beta * float(
            np.linalg.norm(design.residual)
        )
        update = physical_scale * np.real(encoded)
        normalized = matrix.T / design.beta
        u_mat, sigma, vh_mat = np.linalg.svd(normalized, full_matrices=False)
        polynomial = Polynomial(design.coefficients)
        exact_action = (u_mat @ np.diag(polynomial(sigma)) @ vh_mat) @ residual_unit
        action_error = float(
            np.linalg.norm(np.real(encoded) - exact_action)
            / max(np.linalg.norm(exact_action), 1.0e-30)
        )
        return StatevectorPointResult(
            value_key=value_key,
            phase_key=phase_key,
            encoded_state=encoded,
            update=update,
            postselection_probability=postselection,
            block_reconstruction_error=float(wrapper.top_left_reconstruction_error),
            qsvt_action_error=action_error,
            wrapper_normalization=float(wrapper.normalization_factor),
            status="completed",
            failure_reason="",
        )
    except Exception as exc:  # retained-failure policy: classify, never repair silently
        reason = str(exc)
        failure_class = "other_verified_failure"
        for candidate in FAILURE_CLASSES:
            if candidate in reason:
                failure_class = candidate
                break
        if isinstance(exc, MemoryError):
            failure_class = "statevector_memory_limit"
        return StatevectorPointResult(
            value_key=value_key,
            phase_key=phase_key,
            encoded_state=np.full(n, np.nan, dtype=np.complex128),
            update=np.full(n, np.nan),
            postselection_probability=float("nan"),
            block_reconstruction_error=float("nan"),
            qsvt_action_error=float("nan"),
            wrapper_normalization=float("nan"),
            status="failed",
            failure_reason=f"{failure_class}: {type(exc).__name__}: {reason}",
        )


# --------------------------------------------------------------------------------------
# Stage: audit
# --------------------------------------------------------------------------------------


def load_study_configuration(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study configuration must be a JSON object")
    return payload


def stage_audit(context: StudyContext) -> dict[str, Any]:
    """Record the declared study configuration; verify the Phase 0 audit exists."""

    audit_path = context.output_dir / "implementation_audit.md"
    if not audit_path.is_file():
        raise RuntimeError(
            "implementation_audit.md must be written before the sweep (Phase 0 contract)"
        )
    declared = load_study_configuration(context.config_path)
    expected_subset = [dict(item) for item in FINITE_SHOT_CONFIGURATIONS]
    if declared.get("finite_shot_configurations") != expected_subset:
        raise RuntimeError(
            "declared finite-shot subset in the config file disagrees with the module "
            "predeclaration; refusing to run with an undeclared sampling plan"
        )
    record = {
        "study_id": STUDY_ID,
        "baseline_configuration_id": BASELINE_CONFIGURATION_ID,
        "declared_configuration_file": str(context.config_path),
        "declared_configuration": declared,
        "value_bits_sweep": [*list(VALUE_BITS_SWEEP), EXACT_VALUE_KEY],
        "phase_bits_sweep": [*list(PHASE_BITS_SWEEP), FULL_PHASE_KEY],
        "functionals": list(FUNCTIONAL_IDS),
        "finite_shot_configurations": expected_subset,
        "shot_budgets": list(SHOT_BUDGETS),
        "seeds": list(context.seeds),
        "max_workers_requested": context.max_workers,
        "max_workers_effective_heavy_stages": 1,
        "tolerances": {
            "float_relative": FLOAT_REL_TOLERANCE,
            "fingerprints": FINGERPRINT_TOLERANCE,
            "signed_identity_absolute": IDENTITY_ABS_TOLERANCE,
            "triangle_bound_absolute": TRIANGLE_ABS_TOLERANCE,
            "block_reconstruction": BLOCK_RECONSTRUCTION_TOLERANCE,
            "circuit_statevector_crosscheck": CIRCUIT_STATEVECTOR_CROSSCHECK_TOLERANCE,
        },
        "environment": {
            "MPLBACKEND": os.environ.get("MPLBACKEND"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "aer_configuration": "statevector method, 1 thread, 1 experiment, 1 shot worker",
        },
        "git_commit_hash": git_commit_hash(),
        "timestamp": now_iso(),
        "key_package_versions": package_versions(
            ["numpy", "pandas", "scipy", "pennylane", "qiskit", "qiskit-aer", "pypower"]
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    atomic_write_json(context.output_dir / "study_configuration.json", json_ready(record))
    context.checkpoint.mark_stage_complete("audit", {"audit_file": str(audit_path)})
    return record


# --------------------------------------------------------------------------------------
# Stage: baseline reproduction
# --------------------------------------------------------------------------------------


def _compare(
    checks: list[dict[str, Any]], name: str, expected: Any, actual: Any, kind: str
) -> None:
    passed = expected == actual if kind == "exact" else _close(float(expected), float(actual))
    checks.append(
        {
            "check": name,
            "expected": json_ready(expected),
            "actual": json_ready(actual),
            "comparison": kind,
            "status": "pass" if passed else "fail",
        }
    )


def stage_baseline(context: StudyContext) -> dict[str, Any]:
    """Reproduce the frozen integrated baseline before any sweep work."""

    design = context.design
    inputs = design.inputs
    baseline_dir = Path("outputs/sparse_integrated_chain")
    stored_config = json.loads((baseline_dir / "configuration.json").read_text("utf-8"))
    stored_matrix_meta = json.loads((baseline_dir / "matrix_metadata.json").read_text("utf-8"))
    stored_statevector = pd.read_csv(baseline_dir / "statevector_validation.csv")
    stored_resources = pd.read_csv(baseline_dir / "resource_ledger.csv")
    stored_shots = pd.read_csv(baseline_dir / "finite_shot_summary.csv")

    checks: list[dict[str, Any]] = []
    config = inputs.config
    _compare(checks, "configuration_id", stored_config["configuration_id"],
             config.configuration_id, "exact")
    _compare(checks, "matrix_fingerprint", stored_config["matrix_fingerprint"],
             config.matrix_fingerprint, "exact")
    _compare(checks, "residual_fingerprint", stored_config["residual_fingerprint"],
             config.residual_fingerprint, "exact")
    _compare(checks, "matrix_fingerprint_original",
             stored_matrix_meta["matrix_fingerprint_original"],
             stable_array_fingerprint(design.matrix_original), "exact")
    _compare(checks, "matrix_fingerprint_sparsified",
             stored_matrix_meta["matrix_fingerprint_sparsified"],
             stable_array_fingerprint(design.matrix_sparse_exact), "exact")
    _compare(checks, "phase_fingerprint", stored_matrix_meta["phase_fingerprint"],
             stable_array_fingerprint(design.phases_by_phase_key[FULL_PHASE_KEY]), "exact")
    _compare(checks, "alpha", stored_config["alpha"], design.alpha, "float")
    _compare(checks, "beta", stored_config["beta"], design.beta, "float")
    _compare(checks, "lambda", stored_config["normalized_lambda"],
             design.normalized_lambda, "float")
    _compare(checks, "C", stored_config["contraction_c"], design.contraction_c, "float")
    _compare(checks, "polynomial_degree", int(stored_config["polynomial_degree"]),
             design.degree, "exact")
    _compare(checks, "phase_count", int(stored_matrix_meta["phase_count"]),
             int(design.phases_by_phase_key[FULL_PHASE_KEY].size), "exact")
    _compare(checks, "nnz", int(stored_matrix_meta["nnz"]),
             int(np.count_nonzero(design.matrices_by_value_key["6"])), "exact")
    _compare(checks, "slots", int(stored_matrix_meta["slots"]), design.slots, "exact")
    stored_coefficients = np.load(baseline_dir / "polynomial_coefficients.npy")
    _compare(checks, "polynomial_coefficients_bit_identical", True,
             bool(np.array_equal(stored_coefficients, design.coefficients)), "exact")
    stored_phases = np.load(baseline_dir / "phases.npy")
    _compare(checks, "phase_sequence_bit_identical", True,
             bool(np.array_equal(stored_phases, design.phases_by_phase_key[FULL_PHASE_KEY])),
             "exact")

    bundle = build_integrated_sparse_selected_output_circuit(
        config,
        matrix=inputs.matrix_quantized,
        residual=inputs.residual,
        selected_functional=inputs.selected_functionals[PRIMARY_FUNCTIONAL_ID],
        phases=inputs.phases,
    )
    validation = statevector_validate_integrated_chain(inputs, bundle)
    assert_no_direct_output_initializer(bundle.circuit, validation.sparse_encoded_state)
    _compare(checks, "block_reconstruction_relative_fro_error_below_tolerance", True,
             bool(validation.metrics["block_reconstruction_relative_fro_error"]
                  <= BLOCK_RECONSTRUCTION_TOLERANCE), "exact")
    for functional_id in FUNCTIONAL_IDS:
        stored_row = stored_statevector[
            stored_statevector["functional_id"] == functional_id
        ].iloc[0]
        ell = design.functionals[functional_id]
        _compare(checks, f"statevector_selected_output[{functional_id}]",
                 float(stored_row["sparse_statevector_selected_output"]),
                 float(ell @ validation.sparse_update), "float")
        _compare(checks, f"quantized_ridge_selected_output[{functional_id}]",
                 float(stored_row["quantized_ridge_selected_output"]),
                 float(ell @ validation.quantized_ridge_update), "float")
        _compare(checks, f"original_ridge_selected_output[{functional_id}]",
                 float(stored_row["original_unquantized_ridge_selected_output"]),
                 float(ell @ validation.original_ridge_update), "float")
    _compare(checks, "sparse_postselection_probability",
             float(stored_statevector.iloc[0]["sparse_postselection_probability"]),
             validation.metrics["sparse_postselection_probability"], "float")

    compiled, _simulator = compile_for_aer(bundle.circuit)
    ops = {str(key): int(value) for key, value in compiled.count_ops().items()}
    stored_sparse = stored_resources[
        stored_resources["resource_category"] == "executed_small_scale_sparse_integrated"
    ].iloc[0]
    _compare(checks, "transpiled_gate_count", int(stored_sparse["transpiled_gate_count"]),
             int(sum(ops.values())), "exact")
    _compare(checks, "transpiled_depth", int(stored_sparse["transpiled_depth"]),
             int(compiled.depth()), "exact")
    _compare(checks, "toffoli_count", int(stored_sparse["toffoli_count"]),
             int(ops.get("ccx", 0)), "exact")
    _compare(checks, "controlled_rotation_count",
             int(stored_sparse["controlled_rotation_count"]),
             int(bundle.operation_counts["value_rotations_per_attempt"]), "exact")
    _compare(checks, "signal_unitary_calls_per_attempt",
             int(stored_sparse["signal_unitary_calls_per_attempt"]),
             int(bundle.operation_counts["signal_unitary_calls_per_attempt"]), "exact")
    _compare(checks, "projector_phase_operations_per_attempt",
             int(stored_sparse["projector_phase_operations_per_attempt"]),
             int(bundle.operation_counts["projector_phase_operations_per_attempt"]), "exact")
    stored_primary_shots = stored_shots[
        (stored_shots["chain_type"] == "sparse")
        & (stored_shots["functional_id"] == PRIMARY_FUNCTIONAL_ID)
        & (stored_shots["shots"] == max(SHOT_BUDGETS))
    ].iloc[0]
    _compare(checks, "stored_finite_shot_statevector_reference",
             float(stored_primary_shots["statevector_reference"]),
             float(design.functionals[PRIMARY_FUNCTIONAL_ID] @ validation.sparse_update),
             "float")

    failures = [check for check in checks if check["status"] != "pass"]
    record = {
        "configuration_id": BASELINE_CONFIGURATION_ID,
        "status": "reproduced" if not failures else "reproduction_failed",
        "checks": checks,
        "failed_checks": failures,
        "declared_tolerances": {
            "floats_relative": FLOAT_REL_TOLERANCE,
            "fingerprints_and_counts": "exact",
        },
        "compiled_operation_counts": ops,
        "timestamp": now_iso(),
    }
    atomic_write_json(context.output_dir / "baseline_reproduction.json", json_ready(record))
    if failures:
        raise RuntimeError(
            f"baseline reproduction failed on {len(failures)} checks; sweep must not start"
        )
    context.checkpoint.mark_stage_complete("baseline", {"checks": len(checks)})
    return record


# --------------------------------------------------------------------------------------
# Stage: matrices (Phase 2 + Phase 3 + design mode)
# --------------------------------------------------------------------------------------


def stage_matrices(context: StudyContext) -> dict[str, Any]:
    design = context.design
    output_dir = context.output_dir
    matrices_dir = ensure_directory(output_dir / "matrices")
    np.save(output_dir / "matrix_original.npy", design.matrix_original)
    np.save(output_dir / "matrix_sparse_exact.npy", design.matrix_sparse_exact)
    np.save(matrices_dir / "residual.npy", design.residual)
    for key, matrix in design.matrices_by_value_key.items():
        np.save(matrices_dir / f"matrix_value_bits_{key}.npy", matrix)

    rows, cols = np.nonzero(design.support)
    support_payload = {
        "definition": "sparsify_block(H_original, keep_per_row=2): per row keep the two "
        "largest-|value| entries (frozen Phase 9/10 convention)",
        "shape": list(design.matrix_original.shape),
        "nnz": int(design.support.sum()),
        "entries": [[int(r), int(c)] for r, c in zip(rows, cols, strict=True)],
        "row_nonzeros": [int(v) for v in design.support.sum(axis=1)],
        "column_nonzeros": [int(v) for v in design.support.sum(axis=0)],
        "encoded_orientation_minimum_slots": design.slots,
        "support_equals_baseline_six_bit_support": bool(
            np.array_equal(design.support, design.matrices_by_value_key["6"] != 0.0)
        ),
    }
    atomic_write_json(output_dir / "sparse_support.json", support_payload)

    ridge = design.ridge_updates
    sigma_sparse = np.linalg.svd(design.matrix_sparse_exact, compute_uv=False)
    norm_h0_fro = float(np.linalg.norm(design.matrix_original, ord="fro"))
    norm_h0_2 = float(np.linalg.svd(design.matrix_original, compute_uv=False).max())
    registry_rows: list[dict[str, Any]] = []
    bound_table: dict[str, float] = {}
    for key in [*(str(b) for b in VALUE_BITS_SWEEP), EXACT_VALUE_KEY]:
        matrix = design.matrices_by_value_key[key]
        sigma = np.linalg.svd(matrix, compute_uv=False)
        positive = sigma[sigma > 1.0e-10]
        spectral = float(sigma.max())
        bound_table[key] = spectral
        diff_sparse = matrix - design.matrix_sparse_exact
        diff_original = matrix - design.matrix_original
        row: dict[str, Any] = {
            "value_bits": key,
            "value_bits_numeric": precision_key_to_numeric(key),
            "matrix_fingerprint": stable_array_fingerprint(matrix),
            "matrix_file": str(matrices_dir / f"matrix_value_bits_{key}.npy"),
            "quantizer": "quantize_sign_magnitude (sign + uniform magnitude grid)"
            if key != EXACT_VALUE_KEY
            else "none (exact sparse values)",
            "nnz": int(np.count_nonzero(matrix)),
            "unique_encoded_magnitudes": int(np.unique(np.abs(matrix[matrix != 0.0])).size),
            "mu_full_scale": design.mu,
            "quantization_step": design.mu / ((1 << int(key)) - 1)
            if key != EXACT_VALUE_KEY
            else 0.0,
            "relative_frobenius_error_vs_sparse_exact": float(
                np.linalg.norm(diff_sparse, ord="fro")
                / max(np.linalg.norm(design.matrix_sparse_exact, ord="fro"), 1.0e-30)
            ),
            "relative_spectral_error_vs_sparse_exact": float(
                np.linalg.svd(diff_sparse, compute_uv=False).max()
                / max(float(sigma_sparse.max()), 1.0e-30)
            ),
            "max_entrywise_error_vs_sparse_exact": float(np.max(np.abs(diff_sparse))),
            "relative_frobenius_error_vs_original": float(
                np.linalg.norm(diff_original, ord="fro") / max(norm_h0_fro, 1.0e-30)
            ),
            "relative_spectral_error_vs_original": float(
                np.linalg.svd(diff_original, compute_uv=False).max() / max(norm_h0_2, 1.0e-30)
            ),
            "max_singular_value_perturbation_vs_sparse_exact": float(
                np.max(np.abs(sigma - sigma_sparse))
            ),
            "spectral_norm": spectral,
            "sigma_min_positive": float(positive.min()) if positive.size else float("nan"),
            "effective_condition_number_sigma_gt_1e-10": float(spectral / positive.min())
            if positive.size
            else float("inf"),
            "rank_sigma_gt_1e-10": int(positive.size),
            "support_equals_frozen_support": bool(np.array_equal(matrix != 0.0,
                                                                 design.support)),
            "spectral_norm_below_frozen_beta": bool(spectral <= design.beta + 1.0e-9),
        }
        for functional_id in FUNCTIONAL_IDS:
            ell = design.functionals[functional_id]
            row[f"ridge_output_{functional_id}"] = float(ell @ ridge[f"value_bits_{key}"])
        registry_rows.append(row)
    for functional_id in FUNCTIONAL_IDS:
        ell = design.functionals[functional_id]
        for label, update in (("original", ridge["original"]),
                              ("sparse_exact", ridge["sparse_exact"])):
            registry_rows.append(
                {
                    "value_bits": label,
                    "value_bits_numeric": float("nan"),
                    "matrix_fingerprint": stable_array_fingerprint(
                        design.matrix_original
                        if label == "original"
                        else design.matrix_sparse_exact
                    ),
                    f"ridge_output_{functional_id}": float(ell @ update),
                }
            )
    registry = (
        pd.DataFrame(registry_rows)
        .groupby("value_bits", as_index=False, sort=False)
        .first()
    )
    registry.to_csv(output_dir / "matrix_precision_registry.csv", index=False)

    sparse_metrics = {
        "relative_frobenius_error_H0_vs_HS": float(
            np.linalg.norm(design.matrix_original - design.matrix_sparse_exact, ord="fro")
            / max(norm_h0_fro, 1.0e-30)
        ),
        "relative_spectral_error_H0_vs_HS": float(
            np.linalg.svd(
                design.matrix_original - design.matrix_sparse_exact, compute_uv=False
            ).max()
            / max(norm_h0_2, 1.0e-30)
        ),
        "singular_values_original": np.linalg.svd(
            design.matrix_original, compute_uv=False
        ).tolist(),
        "singular_values_sparse_exact": sigma_sparse.tolist(),
        "selected_output_changes": {
            functional_id: {
                "original": float(design.functionals[functional_id] @ ridge["original"]),
                "sparse_exact": float(
                    design.functionals[functional_id] @ ridge["sparse_exact"]
                ),
                "signed_delta": float(
                    design.functionals[functional_id]
                    @ (ridge["sparse_exact"] - ridge["original"])
                ),
            }
            for functional_id in FUNCTIONAL_IDS
        },
    }
    design_mode = {
        "mode": "frozen_design",
        "reason": "one frozen QSVT design (alpha, beta, lambda, C, degree-31 coefficients, "
        "full-precision phases) upper-bounds every sweep matrix, so no common redesign or "
        "per-matrix retuning is needed or performed",
        "alpha": design.alpha,
        "beta": design.beta,
        "lambda": design.normalized_lambda,
        "C": design.contraction_c,
        "degree": design.degree,
        "margin": 1.05,
        "coefficients_fingerprint": stable_array_fingerprint(design.coefficients),
        "full_phase_fingerprint": stable_array_fingerprint(
            design.phases_by_phase_key[FULL_PHASE_KEY]
        ),
        "spectral_norms_vs_beta": {
            key: {"spectral_norm": value, "frozen_beta": design.beta,
                  "bounded": bool(value <= design.beta + 1.0e-9)}
            for key, value in bound_table.items()
        },
        "per_matrix_phase_refitting": "forbidden_and_not_performed",
        "sparsification_stage_metrics": sparse_metrics,
        "timestamp": now_iso(),
    }
    atomic_write_json(output_dir / "design_mode.json", json_ready(design_mode))
    context.checkpoint.mark_stage_complete(
        "matrices", {"value_points": len(design.matrices_by_value_key)}
    )
    return {"registry_rows": len(registry_rows), "design_mode": "frozen_design"}


# --------------------------------------------------------------------------------------
# Stage: statevector grid (Phases 4-6) and phase registry (Phase 5)
# --------------------------------------------------------------------------------------


def _grid_keys() -> list[tuple[str, str]]:
    value_keys = [*(str(b) for b in VALUE_BITS_SWEEP), EXACT_VALUE_KEY]
    phase_keys = [FULL_PHASE_KEY, *(str(b) for b in PHASE_BITS_SWEEP)]
    return [(v, p) for v in value_keys for p in phase_keys]


def _grid_point_payload(
    design: FrozenStudyDesign,
    value_key: str,
    phase_key: str,
    wrapper: CompleteWrapperResult | None,
) -> dict[str, Any]:
    result = evaluate_statevector_point(design, value_key, phase_key, wrapper)
    phases = design.phases_by_phase_key[phase_key]
    full = design.phases_by_phase_key[FULL_PHASE_KEY]
    payload: dict[str, Any] = {
        "value_key": value_key,
        "phase_key": phase_key,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "postselection_probability": result.postselection_probability,
        "block_reconstruction_error": result.block_reconstruction_error,
        "qsvt_action_error": result.qsvt_action_error,
        "wrapper_normalization": result.wrapper_normalization,
        "matrix_fingerprint": stable_array_fingerprint(
            design.matrices_by_value_key[value_key]
        ),
        "phase_fingerprint": stable_array_fingerprint(phases),
        "max_phase_perturbation": float(np.max(np.abs(phases - full))),
        "rms_phase_perturbation": float(np.sqrt(np.mean((phases - full) ** 2))),
        "update": result.update.tolist(),
        "encoded_state_real": np.real(result.encoded_state).tolist(),
        "encoded_state_imag": np.imag(result.encoded_state).tolist(),
        "selected_outputs": {
            functional_id: float(design.functionals[functional_id] @ result.update)
            for functional_id in FUNCTIONAL_IDS
        },
    }
    return payload


def stage_statevector(context: StudyContext) -> dict[str, Any]:
    design = context.design
    checkpoint = context.checkpoint
    keys = _grid_keys()
    wrapper_cache: dict[str, CompleteWrapperResult | None] = {}
    computed = 0
    for value_key, phase_key in keys:
        part_key = f"bv{value_key}_bp{phase_key}"
        if context.resume and checkpoint.load_part("statevector", part_key) is not None:
            continue
        if value_key not in wrapper_cache:
            try:
                wrapper_cache[value_key] = build_sweep_wrapper(design, value_key)
            except Exception:
                wrapper_cache[value_key] = None
        payload = _grid_point_payload(design, value_key, phase_key, wrapper_cache[value_key])
        checkpoint.write_part("statevector", part_key, payload)
        computed += 1

    grid_rows, decomposition_rows = _assemble_grid_tables(context)
    pd.DataFrame(grid_rows).to_csv(
        context.output_dir / "statevector_precision_grid.csv", index=False
    )
    _write_phase_registry(context, grid_rows)
    pd.DataFrame(decomposition_rows).to_csv(
        context.output_dir / "error_decomposition_statevector.csv", index=False
    )
    checkpoint.mark_stage_complete(
        "statevector", {"grid_points": len(keys), "newly_computed": computed}
    )
    return {"grid_points": len(keys), "newly_computed": computed}


def _load_grid_payloads(context: StudyContext) -> dict[tuple[str, str], dict[str, Any]]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for value_key, phase_key in _grid_keys():
        part = context.checkpoint.load_part("statevector", f"bv{value_key}_bp{phase_key}")
        if part is not None:
            payloads[(value_key, phase_key)] = part
    return payloads


def _assemble_grid_tables(
    context: StudyContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    design = context.design
    ridge = design.ridge_updates
    payloads = _load_grid_payloads(context)
    grid_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for (value_key, phase_key), payload in sorted(payloads.items()):
        full_reference = payloads.get((value_key, FULL_PHASE_KEY))
        for functional_id in FUNCTIONAL_IDS:
            ell = design.functionals[functional_id]
            y_original = float(ell @ ridge["original"])
            y_sparse = float(ell @ ridge["sparse_exact"])
            y_quantized = float(ell @ ridge[f"value_bits_{value_key}"])
            failed = payload["status"] != "completed"
            y_qsvt = (
                float("nan") if failed else float(payload["selected_outputs"][functional_id])
            )
            delta_sparse = y_sparse - y_original
            delta_quant = y_quantized - y_sparse
            delta_qsvt = y_qsvt - y_quantized
            total_signed = y_qsvt - y_original
            y_qsvt_full = (
                float(full_reference["selected_outputs"][functional_id])
                if full_reference is not None and full_reference["status"] == "completed"
                else float("nan")
            )
            phase_delta = y_qsvt - y_qsvt_full
            polynomial_delta = y_qsvt_full - y_quantized
            grid_rows.append(
                {
                    "configuration_id": sweep_point_configuration_id(value_key, phase_key),
                    "value_bits": value_key,
                    "phase_bits": phase_key,
                    "value_bits_numeric": precision_key_to_numeric(value_key),
                    "phase_bits_numeric": precision_key_to_numeric(phase_key),
                    "functional_id": functional_id,
                    "matrix_fingerprint": payload["matrix_fingerprint"],
                    "phase_fingerprint": payload["phase_fingerprint"],
                    "alpha": design.alpha,
                    "beta": design.beta,
                    "lambda": design.normalized_lambda,
                    "C": design.contraction_c,
                    "degree": design.degree,
                    "ridge_original_output": y_original,
                    "ridge_sparse_exact_output": y_sparse,
                    "ridge_quantized_output": y_quantized,
                    "qsvt_statevector_output": y_qsvt,
                    "sparsification_signed_delta": delta_sparse,
                    "quantization_signed_delta": delta_quant,
                    "qsvt_signed_delta": delta_qsvt,
                    "phase_signed_delta": phase_delta,
                    "polynomial_signed_delta": polynomial_delta,
                    "sparsification_absolute_error": abs(delta_sparse),
                    "quantization_absolute_error": abs(delta_quant),
                    "qsvt_absolute_error": abs(delta_qsvt),
                    "phase_absolute_error": abs(phase_delta),
                    "total_statevector_signed_delta_vs_original": total_signed,
                    "total_statevector_error_vs_original": abs(total_signed),
                    "signed_identity_residual": total_signed
                    - (delta_sparse + delta_quant + delta_qsvt),
                    "postselection_probability": payload["postselection_probability"],
                    "block_reconstruction_error": payload["block_reconstruction_error"],
                    "qsvt_action_error": payload["qsvt_action_error"],
                    "max_phase_perturbation": payload["max_phase_perturbation"],
                    "rms_phase_perturbation": payload["rms_phase_perturbation"],
                    "status": payload["status"],
                    "failure_reason": payload["failure_reason"],
                }
            )
            triangle_sum = abs(delta_sparse) + abs(delta_quant) + abs(delta_qsvt)
            decomposition_rows.append(
                {
                    "value_bits": value_key,
                    "phase_bits": phase_key,
                    "functional_id": functional_id,
                    "estimate_kind": "statevector",
                    "shots_attempted": float("nan"),
                    "num_seeds": float("nan"),
                    "y_original_ridge": y_original,
                    "y_sparse_exact_ridge": y_sparse,
                    "y_quantized_ridge": y_quantized,
                    "y_qsvt_statevector": y_qsvt,
                    "y_estimate": y_qsvt,
                    "sparsification_signed_delta": delta_sparse,
                    "quantization_signed_delta": delta_quant,
                    "qsvt_signed_delta": delta_qsvt,
                    "sampling_signed_delta": float("nan"),
                    "total_signed_delta": total_signed,
                    "cumulative_identity_residual": total_signed
                    - (delta_sparse + delta_quant + delta_qsvt),
                    "sparsification_absolute_error": abs(delta_sparse),
                    "quantization_absolute_error": abs(delta_quant),
                    "qsvt_absolute_error": abs(delta_qsvt),
                    "sampling_absolute_error": float("nan"),
                    "total_absolute_error": abs(total_signed),
                    "triangle_bound_sum": triangle_sum,
                    "triangle_bound_satisfied": bool(
                        abs(total_signed) <= triangle_sum + TRIANGLE_ABS_TOLERANCE
                    )
                    if not failed
                    else False,
                    "status": payload["status"],
                }
            )
    return grid_rows, decomposition_rows


def _write_phase_registry(context: StudyContext, grid_rows: list[dict[str, Any]]) -> None:
    """Phase-precision registry measured on the primary six-bit matrix (Phase 5)."""

    frame = pd.DataFrame(grid_rows)
    subset = frame[
        (frame["value_bits"] == "6") & (frame["functional_id"] == PRIMARY_FUNCTIONAL_ID)
    ]
    full_rows = subset[subset["phase_bits"] == FULL_PHASE_KEY]
    if full_rows.empty:
        return
    full_row = full_rows.iloc[0]
    rows: list[dict[str, Any]] = []
    for phase_key in [*(str(b) for b in PHASE_BITS_SWEEP), FULL_PHASE_KEY]:
        matching = subset[subset["phase_bits"] == phase_key]
        if matching.empty:
            continue
        row = matching.iloc[0]
        rows.append(
            {
                "phase_bits": phase_key,
                "phase_bits_numeric": precision_key_to_numeric(phase_key),
                "phase_fingerprint": row["phase_fingerprint"],
                "phase_count": int(context.design.degree) + 1,
                "quantization_rule": "round-to-nearest multiple of 2*pi/2^bits "
                "(numpy round-half-even), range [-pi, pi), no refitting",
                "max_phase_perturbation": float(row["max_phase_perturbation"]),
                "rms_phase_perturbation": float(row["rms_phase_perturbation"]),
                "qsvt_action_error": float(row["qsvt_action_error"]),
                "postselection_probability": float(row["postselection_probability"]),
                "selected_output_error_vs_full_phase": abs(
                    float(row["qsvt_statevector_output"])
                    - float(full_row["qsvt_statevector_output"])
                ),
                "selected_output_error_vs_quantized_ridge": abs(
                    float(row["qsvt_signed_delta"])
                ),
                "status": row["status"],
                "failure_reason": row["failure_reason"],
            }
        )
    pd.DataFrame(rows).to_csv(
        context.output_dir / "phase_precision_registry.csv", index=False
    )


# --------------------------------------------------------------------------------------
# Stage: finite-shot campaign (Phase 7)
# --------------------------------------------------------------------------------------


def _finite_shot_config_key(config: dict[str, str]) -> str:
    return f"bv{config['value_bits']}_bp{config['phase_bits']}"


def run_finite_shot_configuration(
    context: StudyContext, sampled: dict[str, str]
) -> dict[str, Any]:
    """Build, validate, compile, and genuinely sample one predeclared configuration."""

    design = context.design
    value_key, phase_key = sampled["value_bits"], sampled["phase_bits"]
    matrix = design.matrices_by_value_key[value_key]
    phases = design.phases_by_phase_key[phase_key]
    point_config = build_point_config(design, value_key, phase_key, context.output_dir)
    grid = context.checkpoint.load_part("statevector", f"bv{value_key}_bp{phase_key}")
    if grid is None or grid["status"] != "completed":
        raise RuntimeError(
            "finite_shot_runtime_limit: statevector grid point missing or failed for "
            f"{point_config.configuration_id}; run the statevector stage first"
        )
    encoded_state = np.asarray(grid["encoded_state_real"], dtype=np.float64) + 1j * np.asarray(
        grid["encoded_state_imag"], dtype=np.float64
    )
    p_post = float(grid["postselection_probability"])
    ridge = design.ridge_updates
    residual_norm = float(np.linalg.norm(design.residual))

    bundles: dict[str, Any] = {}
    compiled: dict[str, tuple[Any, Any]] = {}
    resource_capture: dict[str, Any] = {}
    for functional_id in FUNCTIONAL_IDS:
        bundle = build_integrated_sparse_selected_output_circuit(
            point_config,
            matrix=matrix,
            residual=design.residual,
            selected_functional=design.functionals[functional_id],
            phases=phases,
        )
        assert_no_direct_output_initializer(bundle.circuit, encoded_state)
        distribution = exact_joint_distribution(
            bundle.circuit,
            postselection_flag_qubit=bundle.register_layout["postselection_flag_qubit"],
            readout_qubit=bundle.register_layout["readout_qubit"],
        )
        ell = design.functionals[functional_id]
        ell_unit = ell / np.linalg.norm(ell)
        exact_z = float(np.real(np.vdot(ell_unit, encoded_state)))
        observed_z = distribution["00"] - distribution["10"]
        observed_acceptance = distribution["00"] + distribution["10"]
        if abs(observed_acceptance - (1.0 + p_post) / 2.0) > (
            CIRCUIT_STATEVECTOR_CROSSCHECK_TOLERANCE
        ):
            raise RuntimeError(
                "numerical_instability: measured-circuit acceptance disagrees with the "
                "statevector grid reference"
            )
        if abs(observed_z - exact_z) > CIRCUIT_STATEVECTOR_CROSSCHECK_TOLERANCE:
            raise RuntimeError(
                "numerical_instability: measured-circuit signed overlap disagrees with "
                "the statevector grid reference"
            )
        bundles[functional_id] = bundle
        compiled[functional_id] = compile_for_aer(bundle.circuit)
        ops = {
            str(key): int(value)
            for key, value in compiled[functional_id][0].count_ops().items()
        }
        resource_capture[functional_id] = {
            "operation_counts": ops,
            "transpiled_gate_count": int(sum(ops.values())),
            "transpiled_depth": int(compiled[functional_id][0].depth()),
            "toffoli_count": int(ops.get("ccx", 0)),
            "cx_count": int(ops.get("cx", 0)),
            "total_logical_qubits": int(compiled[functional_id][0].num_qubits),
        }
    primary_bundle = bundles[PRIMARY_FUNCTIONAL_ID]
    direct_compiled, direct_simulator = compile_for_aer(
        primary_bundle.direct_postselection_circuit
    )
    direct_ops = {str(k): int(v) for k, v in direct_compiled.count_ops().items()}
    resource_capture["direct_postselection"] = {
        "operation_counts": direct_ops,
        "transpiled_gate_count": int(sum(direct_ops.values())),
        "transpiled_depth": int(direct_compiled.depth()),
        "toffoli_count": int(direct_ops.get("ccx", 0)),
        "cx_count": int(direct_ops.get("cx", 0)),
        "total_logical_qubits": int(direct_compiled.num_qubits),
    }

    rows: list[dict[str, Any]] = []
    direct_cache: dict[tuple[int, int], tuple[int, float]] = {}
    for shots in SHOT_BUDGETS:
        for seed in context.seeds:
            counts = sample_aer_counts(
                direct_compiled, direct_simulator, shots=shots, seed=seed
            )
            direct_cache[(shots, seed)] = _direct_postselection_estimate(counts)
    for functional_id in FUNCTIONAL_IDS:
        ell = design.functionals[functional_id]
        ell_norm = float(np.linalg.norm(ell))
        ell_unit = ell / ell_norm
        physical_scale = (
            design.contraction_c / design.beta * residual_norm * ell_norm
        )
        exact_z = float(np.real(np.vdot(ell_unit, encoded_state)))
        exact_acceptance = (1.0 + p_post) / 2.0
        statevector_value = float(grid["selected_outputs"][functional_id])
        y_quantized = float(ell @ ridge[f"value_bits_{value_key}"])
        y_original = float(ell @ ridge["original"])
        circuit, simulator = compiled[functional_id]
        for shots in SHOT_BUDGETS:
            expected_se = physical_scale * math.sqrt(
                max(exact_acceptance - exact_z**2, 0.0) / shots
            )
            for seed in context.seeds:
                counts = sample_aer_counts(circuit, simulator, shots=shots, seed=seed)
                estimate = estimate_signed_selected_output(
                    counts, physical_scale=physical_scale
                )
                accepted_direct, measured_post = direct_cache[(shots, seed)]
                selected = float(estimate["selected_output_estimate"])
                rows.append(
                    {
                        "configuration_id": point_config.configuration_id,
                        "configuration_label": sampled["label"],
                        "value_bits": value_key,
                        "phase_bits": phase_key,
                        "functional_id": functional_id,
                        "shots_attempted": int(shots),
                        "direct_postselection_shots_attempted": int(shots),
                        "seed": int(seed),
                        "backend": "qiskit_aer_statevector_actual_shot_sampling",
                        "postselection_accepted_direct": int(accepted_direct),
                        "readout_accepted_interference": int(estimate["readout_accepted"]),
                        "measured_postselection_probability": float(measured_post),
                        "interference_acceptance_probability": float(
                            estimate["interference_acceptance_probability"]
                        ),
                        "inferred_postselection_probability_from_branch": float(
                            estimate["inferred_postselection_probability_from_branch"]
                        ),
                        "readout_sign_mean_accepted": float(
                            estimate["readout_sign_mean_accepted"]
                        ),
                        "signed_overlap_estimate": float(
                            estimate["signed_overlap_estimate"]
                        ),
                        "selected_output_estimate": selected,
                        "statevector_reference": statevector_value,
                        "quantized_ridge_reference": y_quantized,
                        "original_ridge_reference": y_original,
                        "sampling_signed_delta": selected - statevector_value,
                        "sampling_absolute_error": abs(selected - statevector_value),
                        "absolute_error_vs_quantized_ridge": abs(selected - y_quantized),
                        "absolute_error_vs_original_ridge": abs(selected - y_original),
                        "analytic_standard_error": float(
                            estimate["analytic_standard_error"]
                        ),
                        "statevector_expected_standard_error": float(expected_se),
                        "confidence_interval_lower": float(
                            estimate["confidence_interval_lower"]
                        ),
                        "confidence_interval_upper": float(
                            estimate["confidence_interval_upper"]
                        ),
                        "statevector_postselection_probability": p_post,
                        "physical_recovery_scale": physical_scale,
                        "status": "completed",
                        "failure_stage": "",
                        "failure_reason": "",
                        "exception_type": "",
                    }
                )
    return {
        "configuration_id": point_config.configuration_id,
        "label": sampled["label"],
        "value_bits": value_key,
        "phase_bits": phase_key,
        "status": "completed",
        "rows": rows,
        "resource_capture": resource_capture,
        "operation_counts_convention": json_ready(primary_bundle.operation_counts),
        "postselection_probability_statevector": p_post,
    }


def stage_finite_shot(context: StudyContext) -> dict[str, Any]:
    checkpoint = context.checkpoint
    statuses: dict[str, str] = {}
    for sampled in FINITE_SHOT_CONFIGURATIONS:
        part_key = _finite_shot_config_key(sampled)
        if context.resume and checkpoint.load_part("finite-shot", part_key) is not None:
            statuses[part_key] = "already_completed"
            continue
        started = time.perf_counter()
        try:
            payload = run_finite_shot_configuration(context, dict(sampled))
        except Exception as exc:  # retain the failed configuration; never replace it
            reason = str(exc)
            failure_class = "other_verified_failure"
            for candidate in FAILURE_CLASSES:
                if candidate in reason:
                    failure_class = candidate
                    break
            payload = {
                "configuration_id": sweep_point_configuration_id(
                    sampled["value_bits"], sampled["phase_bits"]
                ),
                "label": sampled["label"],
                "value_bits": sampled["value_bits"],
                "phase_bits": sampled["phase_bits"],
                "status": "failed",
                "failure_stage": "finite-shot",
                "failure_reason": f"{failure_class}: {reason}",
                "exception_type": type(exc).__name__,
                "last_completed_checkpoint": "statevector",
                "rows": [],
                "resource_capture": {},
            }
        payload["elapsed_seconds"] = time.perf_counter() - started
        checkpoint.write_part("finite-shot", part_key, payload)
        statuses[part_key] = payload["status"]

    frame, summary = _assemble_finite_shot_tables(context)
    frame.to_csv(context.output_dir / "finite_shot_results.csv", index=False)
    summary.to_csv(context.output_dir / "finite_shot_summary.csv", index=False)
    _write_error_decomposition(context, summary)
    checkpoint.mark_stage_complete("finite-shot", {"configurations": statuses})
    return {"configurations": statuses}


def _assemble_finite_shot_tables(context: StudyContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sampled in FINITE_SHOT_CONFIGURATIONS:
        part = context.checkpoint.load_part("finite-shot", _finite_shot_config_key(sampled))
        if part is None:
            continue
        if part["status"] != "completed":
            failures.append(
                {
                    "configuration_id": part["configuration_id"],
                    "configuration_label": part["label"],
                    "value_bits": part["value_bits"],
                    "phase_bits": part["phase_bits"],
                    "functional_id": "all",
                    "status": "failed",
                    "failure_stage": part.get("failure_stage", "finite-shot"),
                    "failure_reason": part.get("failure_reason", ""),
                    "exception_type": part.get("exception_type", ""),
                }
            )
            continue
        rows.extend(part["rows"])
    frame = pd.DataFrame(rows)
    if failures:
        frame = pd.concat([frame, pd.DataFrame(failures)], ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    if not frame.empty and "seed" in frame:
        completed = frame[frame["status"] == "completed"]
        grouped = completed.groupby(
            ["configuration_id", "configuration_label", "value_bits", "phase_bits",
             "functional_id", "shots_attempted"],
            sort=True,
        )
        for (config_id, label, value_key, phase_key, functional_id, shots), group in grouped:
            estimates = group["selected_output_estimate"].to_numpy(dtype=np.float64)
            statevector_value = float(group["statevector_reference"].iloc[0])
            y_quantized = float(group["quantized_ridge_reference"].iloc[0])
            y_original = float(group["original_ridge_reference"].iloc[0])
            seed_count = int(group["seed"].nunique())
            mean_estimate = float(np.mean(estimates))
            seed_std = float(np.std(estimates, ddof=1)) if seed_count > 1 else float("nan")
            se_of_mean = (
                seed_std / math.sqrt(seed_count) if seed_count > 1 else float("nan")
            )
            analytic = group["analytic_standard_error"].to_numpy(dtype=np.float64)
            expected = group["statevector_expected_standard_error"].to_numpy(
                dtype=np.float64
            )
            empirical_variance = (
                float(np.var(estimates, ddof=1)) if seed_count > 1 else float("nan")
            )
            analytic_variance = float(np.mean(expected**2))
            coverage = float(
                np.mean(
                    (group["confidence_interval_lower"] <= statevector_value)
                    & (statevector_value <= group["confidence_interval_upper"])
                )
            )
            summary_rows.append(
                {
                    "configuration_id": config_id,
                    "configuration_label": label,
                    "value_bits": value_key,
                    "phase_bits": phase_key,
                    "functional_id": functional_id,
                    "shots_attempted": int(shots),
                    "num_seeds": seed_count,
                    "mean_shots_attempted": float(group["shots_attempted"].mean()),
                    "mean_postselection_accepted_direct": float(
                        group["postselection_accepted_direct"].mean()
                    ),
                    "mean_readout_accepted_interference": float(
                        group["readout_accepted_interference"].mean()
                    ),
                    "mean_measured_postselection_probability": float(
                        group["measured_postselection_probability"].mean()
                    ),
                    "mean_interference_acceptance_probability": float(
                        group["interference_acceptance_probability"].mean()
                    ),
                    "mean_selected_output_estimate": mean_estimate,
                    "selected_output_std_across_seeds": seed_std,
                    "uncertainty_of_seed_mean": se_of_mean,
                    "seed_mean_confidence_interval_lower": mean_estimate - CI_Z * se_of_mean,
                    "seed_mean_confidence_interval_upper": mean_estimate + CI_Z * se_of_mean,
                    "mean_analytic_standard_error_one_estimate": float(np.mean(analytic)),
                    "empirical_variance_across_seeds": empirical_variance,
                    "analytic_variance_one_estimate": analytic_variance,
                    "empirical_to_analytic_variance_ratio": (
                        empirical_variance / analytic_variance
                        if analytic_variance > 0
                        else float("nan")
                    ),
                    "statevector_95pct_ci_coverage_across_seeds": coverage,
                    "statevector_reference": statevector_value,
                    "quantized_ridge_reference": y_quantized,
                    "original_ridge_reference": y_original,
                    "sampling_signed_delta_of_mean": mean_estimate - statevector_value,
                    "sampling_absolute_error_of_mean": abs(mean_estimate - statevector_value),
                    "absolute_error_of_mean_vs_quantized_ridge": abs(
                        mean_estimate - y_quantized
                    ),
                    "absolute_error_of_mean_vs_original_ridge": abs(
                        mean_estimate - y_original
                    ),
                    "uncertainty_note": (
                        "analytic_standard_error describes one finite-shot estimate; the "
                        "seed std and the standard error of the seed mean are separate "
                        "across-seed quantities"
                    ),
                    "status": "completed",
                }
            )
    return frame, pd.DataFrame(summary_rows)


def _write_error_decomposition(context: StudyContext, summary: pd.DataFrame) -> None:
    """Full decomposition ledger: statevector rows plus sampled finite-shot rows."""

    statevector_path = context.output_dir / "error_decomposition_statevector.csv"
    _, statevector_rows = _assemble_grid_tables(context)
    pd.DataFrame(statevector_rows).to_csv(statevector_path, index=False)
    rows = list(statevector_rows)
    design = context.design
    ridge = design.ridge_updates
    if not summary.empty:
        for _, entry in summary.iterrows():
            functional_id = str(entry["functional_id"])
            value_key = str(entry["value_bits"])
            ell = design.functionals[functional_id]
            y_original = float(ell @ ridge["original"])
            y_sparse = float(ell @ ridge["sparse_exact"])
            y_quantized = float(ell @ ridge[f"value_bits_{value_key}"])
            y_qsvt = float(entry["statevector_reference"])
            y_estimate = float(entry["mean_selected_output_estimate"])
            delta_sparse = y_sparse - y_original
            delta_quant = y_quantized - y_sparse
            delta_qsvt = y_qsvt - y_quantized
            delta_sampling = y_estimate - y_qsvt
            total_signed = y_estimate - y_original
            triangle_sum = (
                abs(delta_sparse) + abs(delta_quant) + abs(delta_qsvt) + abs(delta_sampling)
            )
            rows.append(
                {
                    "value_bits": value_key,
                    "phase_bits": str(entry["phase_bits"]),
                    "functional_id": functional_id,
                    "estimate_kind": "finite_shot_seed_mean",
                    "shots_attempted": int(entry["shots_attempted"]),
                    "num_seeds": int(entry["num_seeds"]),
                    "y_original_ridge": y_original,
                    "y_sparse_exact_ridge": y_sparse,
                    "y_quantized_ridge": y_quantized,
                    "y_qsvt_statevector": y_qsvt,
                    "y_estimate": y_estimate,
                    "sparsification_signed_delta": delta_sparse,
                    "quantization_signed_delta": delta_quant,
                    "qsvt_signed_delta": delta_qsvt,
                    "sampling_signed_delta": delta_sampling,
                    "total_signed_delta": total_signed,
                    "cumulative_identity_residual": total_signed
                    - (delta_sparse + delta_quant + delta_qsvt + delta_sampling),
                    "sparsification_absolute_error": abs(delta_sparse),
                    "quantization_absolute_error": abs(delta_quant),
                    "qsvt_absolute_error": abs(delta_qsvt),
                    "sampling_absolute_error": abs(delta_sampling),
                    "total_absolute_error": abs(total_signed),
                    "triangle_bound_sum": triangle_sum,
                    "triangle_bound_satisfied": bool(
                        abs(total_signed) <= triangle_sum + TRIANGLE_ABS_TOLERANCE
                    ),
                    "status": "completed",
                }
            )
    pd.DataFrame(rows).to_csv(context.output_dir / "error_decomposition.csv", index=False)


# --------------------------------------------------------------------------------------
# Stage: resources (Phase 8)
# --------------------------------------------------------------------------------------


def _wrapper_resource_registry(context: StudyContext) -> pd.DataFrame:
    """Transpiled one-signal-call resources per value precision (u3/cx basis)."""

    from qiskit import transpile

    design = context.design
    rows: list[dict[str, Any]] = []
    for value_key in [*(str(b) for b in VALUE_BITS_SWEEP), EXACT_VALUE_KEY]:
        wrapper = build_sweep_wrapper(design, value_key)
        transpiled = transpile(wrapper.circuit, basis_gates=["u3", "cx"], optimization_level=1)
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        rows.append(
            {
                "value_bits": value_key,
                "value_bits_numeric": precision_key_to_numeric(value_key),
                "one_signal_unitary_gate_count": int(sum(counts.values())),
                "one_signal_unitary_depth": int(transpiled.depth()),
                "one_signal_unitary_cx_count": int(counts.get("cx", 0)),
                "value_rotation_gates_per_signal_call": int(
                    design.matrix_original.shape[1] * design.slots
                ),
                "precision_dependent_gate_count_scaling": DIRECT_ROTATION_LIMITATION,
            }
        )
    return pd.DataFrame(rows)


def stage_resources(context: StudyContext) -> dict[str, Any]:
    design = context.design
    convention = qsvt_sequence_operation_counts(design.degree + 1)
    residual_prep_gates = _residual_preparation_gate_count(design)
    rows: list[dict[str, Any]] = []
    for sampled in FINITE_SHOT_CONFIGURATIONS:
        part = context.checkpoint.load_part("finite-shot", _finite_shot_config_key(sampled))
        if part is None:
            raise RuntimeError(
                "resources stage requires completed finite-shot checkpoints; run "
                "--stage finite-shot first"
            )
        if part["status"] != "completed":
            rows.append(
                {
                    "configuration_id": part["configuration_id"],
                    "configuration_label": part["label"],
                    "value_bits": part["value_bits"],
                    "phase_bits": part["phase_bits"],
                    "functional_id": "all",
                    "status": "failed",
                    "failure_reason": part.get("failure_reason", ""),
                }
            )
            continue
        p_post = float(part["postselection_probability_statevector"])
        capture = part["resource_capture"]
        for functional_id in FUNCTIONAL_IDS:
            resource = capture[functional_id]
            gate_count = int(resource["transpiled_gate_count"])
            for shots in SHOT_BUDGETS:
                rows.append(
                    {
                        "configuration_id": part["configuration_id"],
                        "configuration_label": part["label"],
                        "value_bits": part["value_bits"],
                        "phase_bits": part["phase_bits"],
                        "functional_id": functional_id,
                        "logical_qubits": int(resource["total_logical_qubits"]),
                        "ancilla_work_qubits": int(resource["total_logical_qubits"]) - 3,
                        "polynomial_degree": design.degree,
                        "signal_unitary_calls": int(convention["signal_unitary_calls"]),
                        "phase_operations": int(convention["projector_phase_operations"]),
                        "sparse_lookup_calls": int(convention["signal_unitary_calls"]),
                        "inverse_lookup_calls": int(convention["signal_unitary_calls"] // 2),
                        "controlled_value_rotations": int(
                            design.matrix_original.shape[1]
                            * design.slots
                            * convention["signal_unitary_calls"]
                        ),
                        "residual_preparation_gates": residual_prep_gates,
                        "transpiled_gate_count_per_attempt": gate_count,
                        "transpiled_depth_per_attempt": int(resource["transpiled_depth"]),
                        "toffoli_count_per_attempt": int(resource["toffoli_count"]),
                        "cx_count_per_attempt": int(resource["cx_count"]),
                        "postselection_probability": p_post,
                        "attempts_per_accepted_direct_sample": 1.0 / p_post,
                        "gates_per_accepted_direct_sample": gate_count / p_post,
                        "shots_attempted": int(shots),
                        "estimated_total_gate_applications": int(shots) * gate_count,
                        "value_precision_gate_count_scaling": DIRECT_ROTATION_LIMITATION,
                        "phase_precision_gate_count_scaling": DIRECT_ROTATION_LIMITATION,
                        "resource_semantics": "executed_transpiled_counts_per_attempt; "
                        "totals are attempted-shot multiples, not hardware estimates",
                        "status": "completed",
                        "failure_reason": "",
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(context.output_dir / "resource_sweep.csv", index=False)
    wrapper_registry = _wrapper_resource_registry(context)
    wrapper_registry.to_csv(
        context.output_dir / "signal_unitary_resource_registry.csv", index=False
    )
    context.checkpoint.mark_stage_complete("resources", {"rows": len(rows)})
    return {"rows": len(rows)}


def _residual_preparation_gate_count(design: FrozenStudyDesign) -> int:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    residual_unit = design.residual / np.linalg.norm(design.residual)
    probe = QuantumCircuit(4)
    probe.append(
        StatePreparation(residual_unit).control(1, ctrl_state=0, annotated=False), range(4)
    )
    compiled, _simulator = compile_for_aer(probe.measure_all(inplace=False))
    counts = compiled.count_ops()
    return int(sum(int(v) for v in counts.values()) - int(counts.get("measure", 0)))


# --------------------------------------------------------------------------------------
# Stage: optional sparsity-budget extension (Phase 9)
# --------------------------------------------------------------------------------------


def deterministic_topk_support(matrix: np.ndarray, k: int) -> np.ndarray:
    """Global largest-|value| support with deterministic row-major tie-breaking.

    The rule is predeclared in the study configuration and never depends on any solved
    output.  Ties in magnitude are broken by flat row-major index (stable ordering).
    """

    values = np.asarray(matrix, dtype=np.float64)
    flat = np.abs(values).ravel()
    if not 1 <= int(k) <= flat.size:
        raise ValueError(f"unsupported sparsity budget {k}")
    order = np.lexsort((np.arange(flat.size), -flat))
    support = np.zeros(flat.size, dtype=bool)
    support[order[: int(k)]] = True
    return support.reshape(values.shape)


def stage_sparsity_extension(context: StudyContext) -> dict[str, Any]:
    """Classical + block-encoding evidence for deterministic sparsity budgets.

    Per predeclared budget: freeze the support before solving, keep exact values,
    validate reversible slot assignment and the compiled wrapper encoding, record
    executed wrapper resources, matrix errors, and Ridge selected outputs, then apply
    the same value-precision sweep classically.  QSVT execution on non-baseline
    supports is excluded: each support changes ``beta = slots * mu`` and would require
    per-support phase synthesis, which the primary-study fairness rules forbid.
    """

    from qiskit import transpile

    design = context.design
    rows: list[dict[str, Any]] = []
    for budget in SPARSITY_BUDGETS:
        part_key = f"k{budget}"
        if context.resume:
            cached = context.checkpoint.load_part("sparsity-extension", part_key)
            if cached is not None:
                rows.extend(cached["rows"])
                continue
        budget_rows: list[dict[str, Any]] = []
        try:
            support = deterministic_topk_support(design.matrix_original, budget)
            matrix = np.where(support, design.matrix_original, 0.0)
            pattern = np.abs(matrix.T) > 0.0
            slots = minimum_slot_count(pattern)
            mu = float(np.max(np.abs(matrix)))
            wrapper = validate_complete_wrapper(
                _as_quantized_block(matrix, 53), encode_transpose=True,
                transpile_circuit=False,
            )
            slot_validation = validate_slot_assignment(pattern, wrapper.assignment)
            transpiled = transpile(
                wrapper.circuit, basis_gates=["u3", "cx"], optimization_level=1
            )
            counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
            base_row = {
                "nonzero_budget": int(budget),
                "support_rule": "deterministic_global_largest_magnitude_row_major_ties",
                "support_fingerprint": stable_array_fingerprint(support.astype(np.float64)),
                "matrix_fingerprint": stable_array_fingerprint(matrix),
                "value_bits": EXACT_VALUE_KEY,
                "nnz": int(np.count_nonzero(matrix)),
                "slots": int(slots),
                "mu": mu,
                "beta_support": float(slots * mu),
                "slot_assignment_valid": bool(slot_validation.get("valid", True)),
                "block_reconstruction_error": float(
                    wrapper.top_left_reconstruction_error
                ),
                "wrapper_qubits": int(wrapper.qubits),
                "one_signal_unitary_gate_count": int(sum(counts.values())),
                "one_signal_unitary_depth": int(transpiled.depth()),
                "one_signal_unitary_cx_count": int(counts.get("cx", 0)),
                "relative_frobenius_error_vs_original": float(
                    np.linalg.norm(matrix - design.matrix_original, ord="fro")
                    / max(np.linalg.norm(design.matrix_original, ord="fro"), 1.0e-30)
                ),
                "relative_spectral_error_vs_original": float(
                    np.linalg.svd(matrix - design.matrix_original, compute_uv=False).max()
                    / max(
                        np.linalg.svd(design.matrix_original, compute_uv=False).max(),
                        1.0e-30,
                    )
                ),
                "qsvt_execution": "excluded_requires_per_support_phase_synthesis",
                "status": "completed",
                "failure_reason": "",
            }
            update = ridge_svd_solution(matrix, design.residual, alpha=design.alpha)
            for functional_id in FUNCTIONAL_IDS:
                ell = design.functionals[functional_id]
                base_row[f"ridge_output_{functional_id}"] = float(ell @ update)
                base_row[f"ridge_signed_delta_vs_original_{functional_id}"] = float(
                    ell @ update
                ) - float(ell @ design.ridge_updates["original"])
            budget_rows.append(base_row)
            for bits in VALUE_BITS_SWEEP:
                quantized, _mu = quantize_sign_magnitude(matrix, magnitude_bits=bits)
                q_update = ridge_svd_solution(quantized, design.residual, alpha=design.alpha)
                quant_row = {
                    "nonzero_budget": int(budget),
                    "support_rule": base_row["support_rule"],
                    "support_fingerprint": base_row["support_fingerprint"],
                    "matrix_fingerprint": stable_array_fingerprint(quantized),
                    "value_bits": str(bits),
                    "nnz": int(np.count_nonzero(quantized)),
                    "slots": int(slots),
                    "mu": mu,
                    "beta_support": float(slots * mu),
                    "relative_frobenius_error_vs_original": float(
                        np.linalg.norm(quantized - design.matrix_original, ord="fro")
                        / max(np.linalg.norm(design.matrix_original, ord="fro"), 1.0e-30)
                    ),
                    "qsvt_execution": "excluded_requires_per_support_phase_synthesis",
                    "status": "completed",
                    "failure_reason": "",
                }
                for functional_id in FUNCTIONAL_IDS:
                    ell = design.functionals[functional_id]
                    quant_row[f"ridge_output_{functional_id}"] = float(ell @ q_update)
                budget_rows.append(quant_row)
        except Exception as exc:  # retained-failure policy
            budget_rows.append(
                {
                    "nonzero_budget": int(budget),
                    "value_bits": EXACT_VALUE_KEY,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        context.checkpoint.write_part(
            "sparsity-extension", part_key, {"rows": budget_rows}
        )
        rows.extend(budget_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(context.output_dir / "sparsity_extension_registry.csv", index=False)
    executed = int((frame["status"] == "completed").sum())
    failed = int((frame["status"] != "completed").sum())
    status_lines = [
        "# Sparsity-Budget Extension Status",
        "",
        "Status: **partially executed by design** (classical and block-encoding tiers).",
        "",
        "Executed per predeclared budget k in {8, 12, 16, 24, 32, 64} with exact values:",
        "",
        "- deterministic global largest-|value| support (row-major tie-breaking), recorded",
        "  before any solve;",
        "- reversible slot-assignment validation on the encoded pattern;",
        "- compiled sparse block-encoding wrapper with exact reconstruction check;",
        "- executed one-signal-call wrapper resources (u3/cx transpilation);",
        "- matrix representation errors and Ridge selected outputs at the frozen physical",
        "  alpha, plus the same value-precision sweep applied classically per support.",
        "",
        "Not executed: QSVT circuits on non-baseline supports. Each support changes the",
        "slot count and hence `beta = slots * mu`, so a convention-correct QSVT run would",
        "require fitting a new bounded polynomial and synthesizing a new phase sequence",
        "per support. That is exactly the per-matrix retuning the primary study's",
        "fairness rules forbid, and phase synthesis is outside the frozen-design scope",
        "declared in `design_mode.json`. The minimum task is complete without this tier.",
        "",
        f"Rows: {executed} completed, {failed} retained failures",
        "(registry: `sparsity_extension_registry.csv`).",
        "",
    ]
    (context.output_dir / "sparsity_extension_status.md").write_text(
        "\n".join(status_lines), encoding="utf-8"
    )
    context.checkpoint.mark_stage_complete(
        "sparsity-extension", {"rows": len(rows), "failed": failed}
    )
    return {"rows": len(rows), "failed": failed}


# --------------------------------------------------------------------------------------
# Stage: pareto (Phase 10)
# --------------------------------------------------------------------------------------


def pareto_nondominated(
    frame: pd.DataFrame, *, error_column: str, cost_column: str
) -> pd.Series:
    """Deterministic dominance flags: True where no other row is <= on both axes and
    strictly < on at least one.  NaN rows are never nondominated."""

    errors = frame[error_column].to_numpy(dtype=np.float64)
    costs = frame[cost_column].to_numpy(dtype=np.float64)
    flags = np.zeros(len(frame), dtype=bool)
    for i in range(len(frame)):
        if not (np.isfinite(errors[i]) and np.isfinite(costs[i])):
            continue
        dominated = False
        for j in range(len(frame)):
            if i == j or not (np.isfinite(errors[j]) and np.isfinite(costs[j])):
                continue
            if (
                errors[j] <= errors[i]
                and costs[j] <= costs[i]
                and (errors[j] < errors[i] or costs[j] < costs[i])
            ):
                dominated = True
                break
        flags[i] = not dominated
    return pd.Series(flags, index=frame.index, name="nondominated")


def build_value_precision_pareto_tables(
    grid: pd.DataFrame,
    wrapper_resources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the executed value-precision frontier from the complete candidate set.

    Direct value rotations change parameters but not circuit topology.  Consequently the
    measured cost must come from the per-value signal-unitary resource registry, not from
    the finite-shot subset.  Matrix-representation error is the componentwise sum of the
    absolute sparsification and quantization increments; signed cancellation is never used
    as an accuracy objective.
    """

    required_grid = {
        "configuration_id",
        "functional_id",
        "value_bits",
        "phase_bits",
        "sparsification_absolute_error",
        "quantization_absolute_error",
        "status",
    }
    required_resources = {"value_bits", "one_signal_unitary_gate_count"}
    missing_grid = required_grid.difference(grid.columns)
    missing_resources = required_resources.difference(wrapper_resources.columns)
    if missing_grid:
        raise ValueError(f"precision grid missing columns: {sorted(missing_grid)}")
    if missing_resources:
        raise ValueError(
            f"wrapper resource registry missing columns: {sorted(missing_resources)}"
        )

    matrix_rows = grid[grid["phase_bits"].astype(str) == FULL_PHASE_KEY].copy()
    resources = wrapper_resources.copy()
    resources["value_bits"] = resources["value_bits"].astype(str)
    if resources["value_bits"].duplicated().any():
        raise ValueError("wrapper resource registry has duplicate value_bits entries")
    matrix_rows["value_bits"] = matrix_rows["value_bits"].astype(str)
    matrix_rows = matrix_rows.merge(
        resources[["value_bits", "one_signal_unitary_gate_count"]],
        on="value_bits",
        how="left",
        validate="many_to_one",
    )
    matrix_rows["frontier_kind"] = "matrix_representation_error_vs_signal_unitary_gates"
    matrix_rows["cost_axis"] = "one_signal_unitary_gate_count"
    matrix_rows["cost_value"] = pd.to_numeric(
        matrix_rows["one_signal_unitary_gate_count"], errors="coerce"
    )
    matrix_rows["error_value"] = (
        pd.to_numeric(matrix_rows["sparsification_absolute_error"], errors="coerce")
        + pd.to_numeric(matrix_rows["quantization_absolute_error"], errors="coerce")
    )
    matrix_rows["error_semantics"] = (
        "componentwise_absolute_sum_no_signed_cancellation"
    )

    groups: list[pd.DataFrame] = []
    for _functional_id, group in matrix_rows.groupby("functional_id", sort=True):
        candidate = group.copy()
        candidate["nondominated"] = pareto_nondominated(
            candidate, error_column="error_value", cost_column="cost_value"
        )
        groups.append(candidate)
    candidates = pd.concat(groups, ignore_index=True) if groups else matrix_rows.copy()
    candidates = candidates.sort_values(
        ["functional_id", "cost_value", "error_value", "value_bits", "configuration_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    frontier = candidates[candidates["nondominated"]].reset_index(drop=True)
    return candidates, frontier


def build_phase_rounding_sensitivity(grid: pd.DataFrame) -> pd.DataFrame:
    """Return phase-parameter sensitivity rows without inventing a resource frontier."""

    required = {
        "configuration_id",
        "functional_id",
        "value_bits",
        "phase_bits",
        "phase_bits_numeric",
        "qsvt_absolute_error",
        "status",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"precision grid missing columns: {sorted(missing)}")
    sensitivity = grid[grid["value_bits"].astype(str) == "6"].copy()
    sensitivity["curve_kind"] = "phase_rounding_sensitivity"
    sensitivity["x_axis"] = "phase_bits_parameter_precision"
    sensitivity["x_value"] = pd.to_numeric(
        sensitivity["phase_bits_numeric"], errors="coerce"
    )
    sensitivity["error_value"] = pd.to_numeric(
        sensitivity["qsvt_absolute_error"], errors="coerce"
    )
    sensitivity["executed_resource_frontier"] = False
    sensitivity["resource_semantics"] = (
        "rotation_parameters_only_no_discrete_gate_synthesis_cost"
    )
    return sensitivity.sort_values(
        ["functional_id", "x_value", "phase_bits", "configuration_id"], kind="stable"
    ).reset_index(drop=True)


def stage_pareto(context: StudyContext) -> dict[str, Any]:
    output_dir = context.output_dir
    summary = pd.read_csv(
        output_dir / "finite_shot_summary.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    resources = pd.read_csv(
        output_dir / "resource_sweep.csv", dtype={"value_bits": str, "phase_bits": str}
    )
    grid = pd.read_csv(
        output_dir / "statevector_precision_grid.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    wrapper_resources = pd.read_csv(
        output_dir / "signal_unitary_resource_registry.csv", dtype={"value_bits": str}
    )

    if summary.empty or resources.empty:
        raise RuntimeError(
            "pareto stage requires non-empty finite-shot and resource tables; if every "
            "sampled configuration failed, the retained failure registry is the result"
        )
    resource_ok = resources[resources["status"] == "completed"]
    merged = summary.merge(
        resource_ok[
            ["configuration_id", "functional_id", "shots_attempted",
             "transpiled_gate_count_per_attempt", "estimated_total_gate_applications"]
        ],
        on=["configuration_id", "functional_id", "shots_attempted"],
        how="left",
    )
    accuracy_rows: list[pd.DataFrame] = []
    for cost_axis, cost_column in (
        ("attempted_shots", "shots_attempted"),
        ("total_gate_applications", "estimated_total_gate_applications"),
    ):
        for _functional_id, group in merged.groupby("functional_id", sort=True):
            candidate = group.copy()
            candidate["cost_axis"] = cost_axis
            candidate["cost_value"] = candidate[cost_column].astype(float)
            candidate["error_value"] = candidate[
                "absolute_error_of_mean_vs_original_ridge"
            ].astype(float)
            candidate["nondominated"] = pareto_nondominated(
                candidate, error_column="error_value", cost_column="cost_value"
            )
            accuracy_rows.append(candidate)
    accuracy_full = pd.concat(accuracy_rows, ignore_index=True)
    accuracy_columns = [
        "cost_axis", "functional_id", "configuration_id", "configuration_label",
        "value_bits", "phase_bits", "shots_attempted",
        "transpiled_gate_count_per_attempt", "estimated_total_gate_applications",
        "cost_value", "error_value", "sampling_absolute_error_of_mean",
        "absolute_error_of_mean_vs_quantized_ridge", "nondominated",
    ]
    accuracy_full[accuracy_columns].to_csv(
        output_dir / "pareto_candidates_accuracy_cost.csv", index=False
    )
    accuracy_full[accuracy_full["nondominated"]][accuracy_columns].to_csv(
        output_dir / "pareto_frontier_accuracy_cost.csv", index=False
    )

    precision_full, precision_frontier = build_value_precision_pareto_tables(
        grid, wrapper_resources
    )
    precision_columns = [
        "frontier_kind", "cost_axis", "functional_id", "configuration_id", "value_bits",
        "phase_bits", "cost_value", "error_value", "error_semantics", "status",
        "nondominated",
    ]
    precision_full[precision_columns].to_csv(
        output_dir / "pareto_candidates_precision_cost.csv", index=False
    )
    precision_frontier[precision_columns].to_csv(
        output_dir / "pareto_frontier_precision_cost.csv", index=False
    )
    phase_sensitivity = build_phase_rounding_sensitivity(grid)
    phase_sensitivity_columns = [
        "curve_kind", "x_axis", "functional_id", "configuration_id", "value_bits",
        "phase_bits", "x_value", "error_value", "status",
        "executed_resource_frontier", "resource_semantics",
    ]
    phase_sensitivity[phase_sensitivity_columns].to_csv(
        output_dir / "phase_rounding_sensitivity.csv", index=False
    )
    context.checkpoint.mark_stage_complete(
        "pareto",
        {
            "accuracy_candidates": len(accuracy_full),
            "accuracy_frontier": int(accuracy_full["nondominated"].sum()),
            "precision_candidates": len(precision_full),
            "precision_frontier": len(precision_frontier),
            "phase_sensitivity_rows": len(phase_sensitivity),
        },
    )
    return {
        "accuracy_frontier": int(accuracy_full["nondominated"].sum()),
        "precision_frontier": len(precision_frontier),
        "phase_sensitivity_rows": len(phase_sensitivity),
    }


# --------------------------------------------------------------------------------------
# Stage: verify + manifest
# --------------------------------------------------------------------------------------


def _verify_decomposition(output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        output_dir / "error_decomposition.csv", dtype={"value_bits": str, "phase_bits": str}
    )
    completed = frame[frame["status"] == "completed"]
    identity = completed["cumulative_identity_residual"].abs()
    triangle = completed["triangle_bound_satisfied"]
    statevector_rows = completed[completed["estimate_kind"] == "statevector"]
    sampled_rows = completed[completed["estimate_kind"] == "finite_shot_seed_mean"]
    return {
        "rows_total": len(frame),
        "rows_completed": len(completed),
        "max_abs_cumulative_identity_residual": float(identity.max()),
        "identity_within_tolerance": bool((identity <= IDENTITY_ABS_TOLERANCE).all()),
        "triangle_bound_all_satisfied": bool(triangle.all()),
        "statevector_rows": len(statevector_rows),
        "finite_shot_rows": len(sampled_rows),
        "sparsification_and_quantization_reported_separately": bool(
            {"sparsification_signed_delta", "quantization_signed_delta"}.issubset(
                frame.columns
            )
        ),
    }


def _verify_finite_shot(output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        output_dir / "finite_shot_results.csv", dtype={"value_bits": str, "phase_bits": str}
    )
    completed = frame[frame["status"] == "completed"]
    consistent_counts = bool(
        (
            (completed["readout_accepted_interference"] <= completed["shots_attempted"])
            & (completed["postselection_accepted_direct"] <= completed["shots_attempted"])
        ).all()
    )
    return {
        "rows_total": len(frame),
        "rows_completed": len(completed),
        "attempted_accepted_consistent": consistent_counts,
        "failed_configurations_retained": int((frame["status"] != "completed").sum()),
        "configurations_sampled": sorted(completed["configuration_label"].unique().tolist()),
    }


def refresh_manifest_and_checksums(output_dir: Path) -> None:
    directory = Path(output_dir)
    skip = {"manifest.json", "checksums.sha256"}
    artifact_paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name not in skip and ".tmp." not in path.name
    )
    manifest = {
        "experiment_id": STUDY_ID,
        "baseline_configuration_id": BASELINE_CONFIGURATION_ID,
        "timestamp": now_iso(),
        "git_commit_hash": git_commit_hash(),
        "claim_boundary": CLAIM_BOUNDARY,
        "execution_tier": "executed_statevector_and_sampled_counts_small_scale",
        "changes_estimator_behavior": False,
        "fabricates_results": False,
        "artifacts": [str(path.relative_to(directory)) for path in artifact_paths],
        "artifact_checksums": {
            str(path.relative_to(directory)): _sha256_file(path) for path in artifact_paths
        },
        "key_package_versions": package_versions(
            ["numpy", "pandas", "scipy", "pennylane", "qiskit", "qiskit-aer", "pypower"]
        ),
    }
    write_json(directory / "manifest.json", manifest)
    checked = [*artifact_paths, directory / "manifest.json"]
    (directory / "checksums.sha256").write_text(
        "".join(f"{_sha256_file(path)}  {path}\n" for path in sorted(checked)),
        encoding="utf-8",
    )


def stage_verify(context: StudyContext) -> dict[str, Any]:
    output_dir = context.output_dir
    decomposition = _verify_decomposition(output_dir)
    finite = _verify_finite_shot(output_dir)
    grid = pd.read_csv(
        output_dir / "statevector_precision_grid.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    baseline = json.loads((output_dir / "baseline_reproduction.json").read_text("utf-8"))
    summary_frame = pd.read_csv(
        output_dir / "finite_shot_summary.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    baseline_cross = _crosscheck_against_stored_baseline(context)
    dominant = _dominant_error_sources(output_dir)
    report = {
        "study_id": STUDY_ID,
        "baseline_reproduction_status": baseline["status"],
        "grid_rows": len(grid),
        "grid_completed": int((grid["status"] == "completed").sum()),
        "grid_failed": int((grid["status"] != "completed").sum()),
        "decomposition": decomposition,
        "finite_shot": finite,
        "baseline_resample_crosscheck": baseline_cross,
        "dominant_error_sources": dominant,
        "timestamp": now_iso(),
    }
    atomic_write_json(output_dir / "verification_summary.json", json_ready(report))
    _write_verification_report(context, report, summary_frame)
    _write_summary_markdown(context, report)
    refresh_manifest_and_checksums(output_dir)
    return report


def _crosscheck_against_stored_baseline(context: StudyContext) -> dict[str, Any]:
    """Compare the resampled (6, full) configuration against the frozen chain artifact."""

    stored_path = Path("outputs/sparse_integrated_chain/finite_shot_results.csv")
    ours_path = context.output_dir / "finite_shot_results.csv"
    if not stored_path.is_file() or not ours_path.is_file():
        return {"status": "not_available"}
    stored = pd.read_csv(stored_path)
    ours = pd.read_csv(ours_path, dtype={"value_bits": str, "phase_bits": str})
    ours = ours[
        (ours["configuration_label"] == "baseline") & (ours["status"] == "completed")
    ]
    stored = stored[
        (stored["chain_type"] == "sparse")
        & (stored["functional_id"] == PRIMARY_FUNCTIONAL_ID)
    ]
    ours = ours[ours["functional_id"] == PRIMARY_FUNCTIONAL_ID]
    merged = stored.merge(
        ours,
        left_on=["shots_attempted", "seed"],
        right_on=["shots_attempted", "seed"],
        suffixes=("_stored", "_study"),
    )
    if merged.empty:
        return {"status": "not_available"}
    exact = bool(
        np.allclose(
            merged["selected_output_estimate_stored"],
            merged["selected_output_estimate_study"],
            rtol=0.0,
            atol=0.0,
        )
    )
    max_abs = float(
        np.max(
            np.abs(
                merged["selected_output_estimate_stored"]
                - merged["selected_output_estimate_study"]
            )
        )
    )
    return {
        "status": "exact_match" if exact else "difference_recorded",
        "compared_runs": len(merged),
        "max_absolute_difference": max_abs,
    }


def _dominant_error_sources(output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        output_dir / "error_decomposition.csv", dtype={"value_bits": str, "phase_bits": str}
    )
    baseline = frame[
        (frame["estimate_kind"] == "statevector")
        & (frame["value_bits"] == "6")
        & (frame["phase_bits"] == FULL_PHASE_KEY)
    ]
    result: dict[str, Any] = {}
    for _, row in baseline.iterrows():
        magnitudes = {
            "sparsification": float(row["sparsification_absolute_error"]),
            "quantization": float(row["quantization_absolute_error"]),
            "qsvt": float(row["qsvt_absolute_error"]),
        }
        result[str(row["functional_id"])] = {
            "magnitudes": magnitudes,
            "dominant": max(magnitudes, key=lambda key: magnitudes[key]),
        }
    return result


def _write_verification_report(
    context: StudyContext, report: dict[str, Any], summary: pd.DataFrame
) -> None:
    lines = [
        "# Sparse Error and Precision Study Verification Report",
        "",
        f"- Study: `{STUDY_ID}`",
        f"- Baseline reproduction: **{report['baseline_reproduction_status']}**",
        f"- Statevector grid rows: {report['grid_rows']} "
        f"(completed {report['grid_completed']}, failed {report['grid_failed']})",
        "- Cumulative signed decomposition: max |residual| = "
        f"{report['decomposition']['max_abs_cumulative_identity_residual']:.3e} "
        f"(within tolerance: {report['decomposition']['identity_within_tolerance']})",
        f"- Triangle bound satisfied on all completed rows: "
        f"{report['decomposition']['triangle_bound_all_satisfied']}",
        f"- Finite-shot rows: {report['finite_shot']['rows_total']} "
        f"(completed {report['finite_shot']['rows_completed']}, retained failures "
        f"{report['finite_shot']['failed_configurations_retained']})",
        f"- Attempted/accepted count consistency: "
        f"{report['finite_shot']['attempted_accepted_consistent']}",
        f"- Baseline resample cross-check vs frozen chain artifact: "
        f"{report['baseline_resample_crosscheck']}",
        "",
        "## Dominant error source at the baseline configuration (statevector tier)",
        "",
    ]
    for functional_id, entry in report["dominant_error_sources"].items():
        magnitudes = entry["magnitudes"]
        lines.append(
            f"- `{functional_id}`: dominant = **{entry['dominant']}** "
            f"(sparsification {magnitudes['sparsification']:.3e}, "
            f"quantization {magnitudes['quantization']:.3e}, "
            f"QSVT {magnitudes['qsvt']:.3e})"
        )
    lines += [
        "",
        "## Statistical interpretation",
        "",
        "Attempted shots, direct postselection acceptances, and interference/readout "
        "acceptances are stored separately. Analytic per-estimate standard errors and "
        "across-seed empirical variation are stored separately and never conflated.",
        "",
        "## Claim boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    (context.output_dir / "verification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_summary_markdown(context: StudyContext, report: dict[str, Any]) -> None:
    design = context.design
    grid = pd.read_csv(
        context.output_dir / "statevector_precision_grid.csv",
        dtype={"value_bits": str, "phase_bits": str},
    )
    primary = grid[grid["functional_id"] == PRIMARY_FUNCTIONAL_ID]
    lines = [
        "# Sparse QSVT Error-Source Ablation and Precision-Resource Study",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Frozen design: alpha={design.alpha}, beta={design.beta}, "
        f"lambda={design.normalized_lambda}, C={design.contraction_c}, "
        f"degree={design.degree} (one design for every primary sweep point)",
        f"- Grid: value bits {list(VALUE_BITS_SWEEP)} + exact; phase bits "
        f"{list(PHASE_BITS_SWEEP)} + full; functionals {list(FUNCTIONAL_IDS)}",
        "",
        "## Error decomposition at the baseline point (value=6, phase=full)",
        "",
        "| functional | sparsification | quantization | qsvt | dominant |",
        "| --- | --- | --- | --- | --- |",
    ]
    for functional_id, entry in report["dominant_error_sources"].items():
        magnitudes = entry["magnitudes"]
        lines.append(
            f"| {functional_id} | {magnitudes['sparsification']:.3e} | "
            f"{magnitudes['quantization']:.3e} | {magnitudes['qsvt']:.3e} | "
            f"{entry['dominant']} |"
        )
    lines += [
        "",
        "## Primary-functional statevector grid (selected columns)",
        "",
        "| value bits | phase bits | y_qsvt | quantization delta | qsvt delta | p_post |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in primary.sort_values(["value_bits_numeric", "phase_bits_numeric"]).iterrows():
        lines.append(
            f"| {row['value_bits']} | {row['phase_bits']} | "
            f"{row['qsvt_statevector_output']:.6e} | "
            f"{row['quantization_signed_delta']:+.3e} | {row['qsvt_signed_delta']:+.3e} | "
            f"{row['postselection_probability']:.4f} |"
        )
    lines += [
        "",
        "Signed increments follow the declared computational path "
        "H_original -> H_sparse_exact -> H_quantized -> QSVT -> finite shots and may "
        "partially cancel; absolute magnitudes are reported alongside and never merged.",
        "",
    ]
    (context.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class StudyContext:
    output_dir: Path
    config_path: Path
    checkpoint: StudyCheckpoint
    resume: bool
    force: bool
    max_workers: int
    base_seed: int
    _design: FrozenStudyDesign | None = None

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.base_seed, self.base_seed + SEED_COUNT))

    @property
    def design(self) -> FrozenStudyDesign:
        if self._design is None:
            self._design = build_frozen_design(self.output_dir)
        return self._design


def make_context(
    *,
    output_dir: str | Path = DEFAULT_STUDY_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    resume: bool = False,
    force: bool = False,
    max_workers: int = 1,
    seed: int = 0,
) -> StudyContext:
    directory = ensure_directory(output_dir)
    return StudyContext(
        output_dir=Path(directory),
        config_path=Path(config_path),
        checkpoint=StudyCheckpoint(Path(directory)),
        resume=bool(resume),
        force=bool(force),
        max_workers=max(1, int(max_workers)),
        base_seed=int(seed),
    )


STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "baseline": stage_baseline,
    "matrices": stage_matrices,
    "statevector": stage_statevector,
    "finite-shot": stage_finite_shot,
    "resources": stage_resources,
    "sparsity-extension": stage_sparsity_extension,
    "pareto": stage_pareto,
    "verify": stage_verify,
}


def run_study(context: StudyContext, stage: str = "all") -> dict[str, Any]:
    if stage != "all" and stage not in STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage}; expected one of {('all', *STAGES)}")
    selected = list(STAGES) if stage == "all" else [stage]
    if context.force:
        for name in selected:
            context.checkpoint.clear_stage(name)
    results: dict[str, Any] = {}
    for name in selected:
        started = time.perf_counter()
        results[name] = STAGE_FUNCTIONS[name](context)
        results[name] = {
            "elapsed_seconds": time.perf_counter() - started,
            **(results[name] if isinstance(results[name], dict) else {}),
        }
    return results
