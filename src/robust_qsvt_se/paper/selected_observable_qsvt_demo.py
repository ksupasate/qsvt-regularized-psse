"""Small selected-submatrix QSVT transform plus an isolated overlap-shot test.

End-to-end pathway on an IEEE-derived weighted-Jacobian block:

1. select a deterministic small block ``H_B`` and residual ``r_B``,
2. compute the matched Ridge/Tikhonov reference ``dx_alpha`` and the
   *predetermined* selected observables ``y_l = l^T dx_alpha``,
3. normalize ``A = H_B^T / beta`` (``beta = sigma_max``), dense block-encode it,
4. fit a bounded odd QSVT polynomial for the same regularized filter,
5. synthesize phases (PennyLane), build the gate-level QSVT circuit,
6. apply it to the prepared residual state, postselect the encoded branch,
7. rescale the statevector result to physical units with the factor ``C/beta``,
8. compare against the Ridge reference, and
9. separately characterize signed-overlap shot noise by directly preparing the
   classically computed postselected output state.

The 4x4 IEEE-14 block is a co-designed *pass*; the 8x8 block (high condition
number) is reported honestly as degree/synthesis-limited. Ridge is the matched
reference and QSVT implements the *same* filter at the *same* alpha; no speedup,
The shot experiment is not integrated residual-loading--QSVT--postselection--
readout execution. No QSVT-over-Ridge superiority, full-vector readout, or
quantum-hardware run is claimed.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.circuit_signed_readout import (
    READOUT_CLASS_BASIS_SAMPLING,
    READOUT_CLASS_EXCLUDED_FULL_VECTOR,
    circuit_signed_readout_rows,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    DEMO_CLAIM_BOUNDARY,
    DEMO_DIR,
    array_checksum,
    assert_safe,
    fit_codesigned_bounded_polynomial,
    write_demo_manifest,
)
from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit
from robust_qsvt_se.qsvt.gate_state_preparation import (
    build_initialize_circuit,
    normalize_and_pad_for_gate_preparation,
)
from robust_qsvt_se.qsvt.phase_synthesis import (
    synthesize_pennylane_phases_cached,
    validate_qsvt_polynomial,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_CASE = "ieee14"
DEFAULT_SEED = 123
PASS_RELATIVE_TOLERANCE = 0.05
PHASE_CACHE_DIR = DEMO_DIR / "phase_cache"

DEMO_SUMMARY_COLUMNS = [
    "case",
    "matrix_source",
    "block_shape",
    "block_checksum",
    "residual_checksum",
    "selected_rows",
    "selected_cols",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "alpha",
    "alpha_over_sigma_min_sq",
    "beta",
    "bound_C",
    "alpha_normalized",
    "physical_recovery_factor_C_over_beta",
    "residual_norm",
    "degree",
    "polynomial_fit_max_abs_error",
    "bounded_max_abs",
    "boundedness_ok",
    "phase_count",
    "phase_synthesis_status",
    "angle_solver",
    "num_qubits",
    "raw_gate_count",
    "raw_circuit_depth",
    "transpiled_depth",
    "transpiled_cx_count",
    "transpiled_total_ops",
    "block_encoding_unitarity_error",
    "block_encoding_top_left_error",
    "svt_circuit_vs_exact_max_error",
    "postselection_probability",
    "postselection_normalization",
    "output_state_norm",
    "update_relative_error_vs_ridge",
    "observable_id",
    "observable_type",
    "observable_physical_meaning",
    "observable_support",
    "observable_readout_class",
    "ridge_reference_value",
    "qsvt_statevector_value",
    "absolute_error",
    "relative_error",
    "status_label",
]


@dataclass(frozen=True, slots=True)
class BlockObservable:
    observable_id: str
    observable_type: str
    physical_meaning: str
    vector: np.ndarray = field(repr=False)
    readout_class: str
    sign_aware_required: bool

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.nonzero(self.vector)[0])

    def exact_value(self, update: np.ndarray) -> float:
        dx = np.asarray(update, dtype=np.float64)
        if self.readout_class == READOUT_CLASS_BASIS_SAMPLING:
            return float(np.sum(dx[list(self.support)] ** 2))
        return float(self.vector @ dx)


@dataclass(slots=True)
class BlockDemoResult:
    row_common: dict[str, Any]
    observable_rows: list[dict[str, Any]]
    pipeline_metadata: dict[str, Any]
    H_block: np.ndarray
    r_block: np.ndarray
    observables: list[BlockObservable]
    output_state: np.ndarray | None
    physical_recovery_scale: float
    ridge_update: np.ndarray
    status_label: str


def _state_labels_for_cols(
    system_metadata: dict[str, Any], cols: np.ndarray
) -> list[dict[str, Any]]:
    """Map selected block columns to physical state descriptors when available."""

    labels: list[dict[str, Any]] = []
    try:
        metadata = build_state_metadata_from_system_metadata(system_metadata)
    except Exception:
        metadata = None
    for position, full_index in enumerate(cols):
        descriptor = {
            "block_column": int(position),
            "full_state_index": int(full_index),
            "state_type": "unknown",
            "bus_id": None,
        }
        if metadata is not None and 0 <= int(full_index) < metadata.dimension:
            try:
                record = metadata.record_for_index(int(full_index))
                descriptor["state_type"] = str(record.state_type)
                descriptor["bus_id"] = None if record.bus_id is None else int(record.bus_id)
            except Exception:
                pass
        labels.append(descriptor)
    return labels


def build_block_observables(
    dimension: int, column_labels: list[dict[str, Any]]
) -> list[BlockObservable]:
    """Predetermined selected observables on the block column space (pre-solve)."""

    def label(position: int) -> str:
        descriptor = column_labels[position] if position < len(column_labels) else {}
        state_type = descriptor.get("state_type", "unknown")
        bus = descriptor.get("bus_id")
        suffix = f"{state_type}" + (f"_bus{bus}" if bus is not None else "")
        return suffix

    observables: list[BlockObservable] = []

    single = np.zeros(dimension, dtype=np.float64)
    single[0] = 1.0
    observables.append(
        BlockObservable(
            observable_id="state_correction_0",
            observable_type="single_state_correction",
            physical_meaning=f"signed correction of selected state 0 ({label(0)})",
            vector=single,
            readout_class="isolated_hadamard_overlap_assumed_output_state_preparation",
            sign_aware_required=True,
        )
    )

    if dimension >= 2:
        diff = np.zeros(dimension, dtype=np.float64)
        diff[0] = 1.0
        diff[1] = -1.0
        observables.append(
            BlockObservable(
                observable_id="state_difference_0_1",
                observable_type="branch_angle_difference_like",
                physical_meaning=(
                    f"signed difference of selected states 0 and 1 ({label(0)} - {label(1)})"
                ),
                vector=diff,
                readout_class="isolated_hadamard_overlap_assumed_output_state_preparation",
                sign_aware_required=True,
            )
        )

    half = max(1, (dimension + 1) // 2)
    area = np.zeros(dimension, dtype=np.float64)
    area[:half] = 1.0
    observables.append(
        BlockObservable(
            observable_id="area_aggregate_first_half",
            observable_type="area_aggregate_functional",
            physical_meaning=f"signed aggregate of the first {half} selected states",
            vector=area,
            readout_class="isolated_hadamard_overlap_assumed_output_state_preparation",
            sign_aware_required=True,
        )
    )

    energy = np.zeros(dimension, dtype=np.float64)
    energy[:half] = 1.0
    observables.append(
        BlockObservable(
            observable_id="energy_block_first_half",
            observable_type="energy_block",
            physical_meaning=f"energy-style ||Pi dx||^2 on the first {half} selected states",
            vector=energy,
            readout_class=READOUT_CLASS_BASIS_SAMPLING,
            sign_aware_required=False,
        )
    )
    return observables


def _transpile_stats(circuit: Any, *, qubit_limit: int = 6) -> dict[str, Any]:
    if int(circuit.num_qubits) > int(qubit_limit):
        return {
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
        }
    try:
        from qiskit import transpile  # type: ignore[import-not-found]

        transpiled = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=1)
        counts = {str(key): int(value) for key, value in transpiled.count_ops().items()}
        return {
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_cx_count": int(counts.get("cx", 0)),
            "transpiled_total_ops": int(sum(counts.values())),
        }
    except Exception:
        return {
            "transpiled_depth": np.nan,
            "transpiled_cx_count": np.nan,
            "transpiled_total_ops": np.nan,
        }


def run_demo_for_block(
    *,
    case: str,
    matrix_source: str,
    H_block: np.ndarray,
    r_block: np.ndarray,
    selected_rows: np.ndarray,
    selected_cols: np.ndarray,
    column_labels: list[dict[str, Any]],
    alpha: float,
    degree: int,
    angle_solver: str,
    margin: float,
    domain_low_factor: float,
    pass_relative_tolerance: float,
    bound_tolerance: float = 2.0e-3,
    phase_cache_dir: str | Path = PHASE_CACHE_DIR,
) -> BlockDemoResult:
    """Run the complete QSVT pathway for one square block and compare to Ridge."""

    H = np.asarray(H_block, dtype=np.float64)
    r = np.asarray(r_block, dtype=np.float64)
    n = int(H.shape[0])
    singular_values = np.linalg.svd(H, compute_uv=False)
    sigma_min = float(singular_values.min())
    sigma_max = float(singular_values.max())
    condition = float(sigma_max / sigma_min) if sigma_min > 0 else math.inf
    residual_norm = float(np.linalg.norm(r))

    ridge_update = ridge_svd_solution(H, r, alpha=float(alpha))
    observables = build_block_observables(n, column_labels)

    beta = sigma_max
    B = H.T
    A = B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    domain_min = float(min(singular_values_A) * domain_low_factor)
    domain_min = float(np.clip(domain_min, 1.0e-4, 0.999))

    target = fit_codesigned_bounded_polynomial(
        beta=beta,
        alpha=float(alpha),
        domain_min=domain_min,
        domain_max=1.0,
        degree=int(degree),
        margin=float(margin),
    )
    try:
        validation = validate_qsvt_polynomial(
            np.asarray(target.coefficients), parity="odd", bound_tolerance=bound_tolerance
        )
        bounded_ok = True
        bounded_max_abs = float(validation["max_abs_on_unit_interval"])
    except Exception:
        bounded_ok = False
        bounded_max_abs = float(target.bounded_max_abs)

    block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    block_report = validate_block_encoding(block_encoding, beta=beta, tolerance=1.0e-7)

    phase_status = "completed"
    phase_failure = ""
    phases = np.zeros(0, dtype=np.float64)
    if bounded_ok:
        try:
            cached = synthesize_pennylane_phases_cached(
                np.asarray(target.coefficients),
                angle_solver=str(angle_solver),
                cache_dir=phase_cache_dir,
                cache_metadata={
                    "case": case,
                    "block": f"{n}x{n}",
                    "alpha": float(alpha),
                    "degree": int(degree),
                },
            )
            phases = np.asarray(cached.phases, dtype=np.float64)
        except Exception as exc:  # phase synthesis ceiling is a real boundary
            phase_status = "phase_unavailable"
            phase_failure = f"{type(exc).__name__}: {exc}"
    else:
        phase_status = "failed_boundedness"

    circuit_stats = {
        "num_qubits": int(np.log2(block_encoding.unitary.shape[0])),
        "raw_gate_count": np.nan,
        "raw_circuit_depth": np.nan,
        "transpiled_depth": np.nan,
        "transpiled_cx_count": np.nan,
        "transpiled_total_ops": np.nan,
    }
    svt_error = np.nan
    success_probability = np.nan
    output_state: np.ndarray | None = None
    update_relative_error = np.nan
    qsvt_values: dict[str, float] = {}
    physical_recovery_scale = 0.0

    if phase_status == "completed" and phases.size > 0:
        from qiskit.quantum_info import Operator, Statevector  # type: ignore[import-not-found]

        bundle = build_structured_qsvt_operator_circuit(
            block_encoding.unitary, phases, encoded_dimension=n
        )
        operator_circuit = bundle.qsvt_operator_circuit
        operator_matrix = np.asarray(Operator(operator_circuit).data, dtype=np.complex128)
        transformed = np.real(operator_matrix[:n, :n])

        U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=True)
        exact_transform = U_A @ np.diag(target.polynomial(S_A)) @ Vt_A
        svt_error = float(np.max(np.abs(transformed - exact_transform)))

        preparation = normalize_and_pad_for_gate_preparation(
            r, target_dimension=block_encoding.unitary.shape[0]
        )
        full_circuit = build_initialize_circuit(preparation.padded_state).compose(operator_circuit)
        evolved = np.asarray(Statevector(full_circuit).data, dtype=np.complex128)
        encoded = evolved[:n]
        # Postselection success probability is the (complex) norm of the encoded branch.
        success_probability = float(np.vdot(encoded, encoded).real)
        if success_probability > 1.0e-15:
            output_state = encoded / math.sqrt(success_probability)  # complex unit |psi_out>
        else:
            output_state = encoded

        # dx_alpha ~ (C/beta) ||r|| real(encoded). The single physical recovery scale
        # for a unit |psi_out> is (C/beta) ||r|| sqrt(p_success); applied to the in-phase
        # (real) part this reproduces the matched Ridge update.
        physical_recovery_scale = (
            target.physical_recovery_factor
            * residual_norm
            * math.sqrt(max(success_probability, 0.0))
        )
        qsvt_update = physical_recovery_scale * np.real(output_state)
        update_relative_error = float(
            np.linalg.norm(qsvt_update - ridge_update) / max(np.linalg.norm(ridge_update), 1.0e-30)
        )
        for observable in observables:
            if observable.readout_class == READOUT_CLASS_BASIS_SAMPLING:
                qsvt_values[observable.observable_id] = float(
                    np.sum(qsvt_update[list(observable.support)] ** 2)
                )
            else:
                qsvt_values[observable.observable_id] = float(observable.vector @ qsvt_update)

        circuit_stats.update(
            {
                "raw_gate_count": int(sum(operator_circuit.count_ops().values())),
                "raw_circuit_depth": int(operator_circuit.depth()),
                **_transpile_stats(operator_circuit),
            }
        )

    if phase_status != "completed":
        status_label = phase_status
    elif not bounded_ok:
        status_label = "failed_boundedness"
    elif np.isfinite(update_relative_error) and update_relative_error <= pass_relative_tolerance:
        status_label = "pass"
    else:
        status_label = "degree_limited"

    row_common = {
        "case": case,
        "matrix_source": matrix_source,
        "block_shape": f"{n}x{n}",
        "block_checksum": array_checksum(H),
        "residual_checksum": array_checksum(r),
        "selected_rows": " ".join(str(int(i)) for i in selected_rows),
        "selected_cols": " ".join(str(int(i)) for i in selected_cols),
        "condition_number": condition,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "alpha": float(alpha),
        "alpha_over_sigma_min_sq": float(alpha) / sigma_min**2 if sigma_min > 0 else math.nan,
        "beta": beta,
        "bound_C": target.bound_C,
        "alpha_normalized": target.alpha_normalized,
        "physical_recovery_factor_C_over_beta": target.physical_recovery_factor,
        "residual_norm": residual_norm,
        "degree": int(degree),
        "polynomial_fit_max_abs_error": target.fit_max_abs_error,
        "bounded_max_abs": bounded_max_abs,
        "boundedness_ok": bool(bounded_ok),
        "phase_count": int(phases.size),
        "phase_synthesis_status": phase_status,
        "angle_solver": str(angle_solver),
        "num_qubits": circuit_stats["num_qubits"],
        "raw_gate_count": circuit_stats["raw_gate_count"],
        "raw_circuit_depth": circuit_stats["raw_circuit_depth"],
        "transpiled_depth": circuit_stats["transpiled_depth"],
        "transpiled_cx_count": circuit_stats["transpiled_cx_count"],
        "transpiled_total_ops": circuit_stats["transpiled_total_ops"],
        "block_encoding_unitarity_error": float(block_report["unitarity_error"]),
        "block_encoding_top_left_error": float(block_report["top_left_block_error"]),
        "svt_circuit_vs_exact_max_error": svt_error,
        "postselection_probability": success_probability,
        "postselection_normalization": math.sqrt(success_probability)
        if np.isfinite(success_probability)
        else math.nan,
        "output_state_norm": float(np.linalg.norm(np.real(output_state)))
        if output_state is not None
        else math.nan,
        "update_relative_error_vs_ridge": update_relative_error,
        "status_label": status_label,
    }

    observable_rows: list[dict[str, Any]] = []
    for observable in observables:
        ridge_value = observable.exact_value(ridge_update)
        qsvt_value = qsvt_values.get(observable.observable_id, math.nan)
        abs_error = abs(qsvt_value - ridge_value) if np.isfinite(qsvt_value) else math.nan
        rel_error = (
            abs_error / abs(ridge_value)
            if np.isfinite(abs_error) and abs(ridge_value) > 1.0e-15
            else math.nan
        )
        row = dict(row_common)
        row.update(
            {
                "observable_id": observable.observable_id,
                "observable_type": observable.observable_type,
                "observable_physical_meaning": observable.physical_meaning,
                "observable_support": " ".join(str(i) for i in observable.support),
                "observable_readout_class": observable.readout_class,
                "ridge_reference_value": ridge_value,
                "qsvt_statevector_value": qsvt_value,
                "absolute_error": abs_error,
                "relative_error": rel_error,
            }
        )
        observable_rows.append(row)

    pipeline_metadata = {
        "case": case,
        "matrix_source": matrix_source,
        "block_shape": f"{n}x{n}",
        "selected_rows": [int(i) for i in selected_rows],
        "selected_cols": [int(i) for i in selected_cols],
        "column_labels": column_labels,
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition,
        "alpha": float(alpha),
        "beta": beta,
        "bound_C": target.bound_C,
        "alpha_normalized": target.alpha_normalized,
        "physical_recovery_factor_C_over_beta": target.physical_recovery_factor,
        "degree": int(degree),
        "reduced_degree": (int(degree) - 1) // 2,
        "domain_min": target.domain_min,
        "domain_max": target.domain_max,
        "polynomial_coefficients_bounded": list(target.coefficients),
        "polynomial_fit_max_abs_error": target.fit_max_abs_error,
        "bounded_max_abs": bounded_max_abs,
        "boundedness_ok": bool(bounded_ok),
        "phase_synthesis_status": phase_status,
        "phase_synthesis_failure_reason": phase_failure,
        "phase_count": int(phases.size),
        "angle_solver": str(angle_solver),
        "block_encoding": {
            key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in block_report.items()
        },
        "svt_circuit_vs_exact_max_error": svt_error,
        "postselection_probability": success_probability,
        "update_relative_error_vs_ridge": update_relative_error,
        "ridge_update": [float(value) for value in ridge_update],
        "status_label": status_label,
        "normalization_convention": (
            "p_bounded(s) ~ f(s)/C, f(s)=s/(s^2+alpha/beta^2)=beta*P_alpha(beta s); "
            "physical recovery factor C/beta; dx = (C/beta) ||r|| sqrt(p_success) |psi_out>"
        ),
    }

    return BlockDemoResult(
        row_common=row_common,
        observable_rows=observable_rows,
        pipeline_metadata=pipeline_metadata,
        H_block=H,
        r_block=r,
        observables=observables,
        output_state=output_state,
        physical_recovery_scale=physical_recovery_scale,
        ridge_update=ridge_update,
        status_label=status_label,
    )


def run_selected_observable_qsvt_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    system, matrix_source = build_engineering_system(
        {
            "case_name": resolved["case"],
            "case_source": resolved["case_source"],
            "matrix_source": "weighted_jacobian",
            "seed": int(resolved["seed"]),
        }
    )
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)

    # Resolve any unset per-block alpha to alpha_mult * sigma_min^2 (co-design near
    # the smallest retained singular value); keep explicit alphas untouched.
    resolved["blocks"] = _materialize_block_alphas(resolved["blocks"], H_full, r_full)

    summary_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    headline: BlockDemoResult | None = None

    for spec in resolved["blocks"]:
        size = int(spec["size"])
        H_block, r_block, rows, cols = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        column_labels = _state_labels_for_cols(system.metadata, cols)
        result = run_demo_for_block(
            case=resolved["case"],
            matrix_source=matrix_source,
            H_block=H_block,
            r_block=r_block,
            selected_rows=rows,
            selected_cols=cols,
            column_labels=column_labels,
            alpha=float(spec["alpha"]),
            degree=int(spec["degree"]),
            angle_solver=resolved["angle_solver"],
            margin=resolved["margin"],
            domain_low_factor=resolved["domain_low_factor"],
            pass_relative_tolerance=resolved["pass_relative_tolerance"],
            phase_cache_dir=resolved["phase_cache_dir"],
        )
        summary_rows.extend(result.observable_rows)
        if spec.get("headline") and headline is None:
            headline = result

    if headline is None:
        headline = result

    # Isolated signed-overlap shot test on the headline block's computed output state.
    readout_metadata: dict[str, Any] = {
        "readout_circuit_model": "isolated_hadamard_overlap_with_direct_state_preparation",
        "output_state_access_model": (
            "direct_StatePreparation_of_classically_computed_postselected_output"
        ),
        "integrated_qsvt_readout": False,
        "headline_block": headline.row_common["block_shape"],
        "headline_status": headline.status_label,
        "shots_grid": list(resolved["readout_shots"]),
        "full_vector_recovery": False,
        "readout_class_excluded": READOUT_CLASS_EXCLUDED_FULL_VECTOR,
    }
    if headline.output_state is not None and headline.status_label == "pass":
        signed_observables = [
            {
                "observable_id": obs.observable_id,
                "observable_type": obs.observable_type,
                "vector": obs.vector,
            }
            for obs in headline.observables
            if obs.sign_aware_required
        ]
        ridge_reference = {
            obs.observable_id: obs.exact_value(headline.ridge_update)
            for obs in headline.observables
        }
        readout_rows = circuit_signed_readout_rows(
            observables=signed_observables,
            output_state=headline.output_state,
            physical_recovery_scale=headline.physical_recovery_scale,
            ridge_reference=ridge_reference,
            shots_grid=tuple(resolved["readout_shots"]),
            seed=int(resolved["readout_seed"]),
        )
        readout_metadata["status"] = "isolated_overlap_shot_experiment_completed"
    else:
        readout_metadata["status"] = "headline_block_not_pass_isolated_readout_skipped"

    artifacts = _write_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        headline=headline,
        readout_rows=readout_rows,
        readout_metadata=readout_metadata,
        resolved=resolved,
    )
    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="selected_observable_qsvt_demo",
        description=(
            "Small selected-submatrix QSVT transform on an IEEE-derived weighted-Jacobian "
            "block plus a separate signed-overlap shot-noise experiment under direct "
            "output-state preparation. Ridge/Tikhonov is "
            "the matched reference; QSVT implements the same filter at the same alpha. "
            "The readout is not integrated with residual loading, QSVT, and postselection; "
            "not full-vector readout and not a quantum-hardware run."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[f"build_engineering_system:{resolved['case']}:weighted_jacobian"],
        extra={
            "blocks": resolved["blocks"],
            "pass_relative_tolerance": resolved["pass_relative_tolerance"],
            "headline_status": headline.status_label,
            "selected_signed_observables_only": True,
            "random_seeds": {
                "system_seed": int(resolved["seed"]),
                "readout_shot_seed": int(resolved["readout_seed"]),
            },
            "seed_provenance": {
                "status": "recorded",
                "seeds": {
                    "system_seed": int(resolved["seed"]),
                    "readout_shot_seed": int(resolved["readout_seed"]),
                },
                "source": (
                    "scripts/run_selected_observable_qsvt_demo.py (--seed default 123; "
                    "readout seed fixed in the demo config)"
                ),
                "applies_to": (
                    "system_seed drives IEEE measurement-row generation in "
                    "build_engineering_system; readout_shot_seed drives isolated "
                    "Hadamard-overlap sampling under direct output-state preparation; "
                    "block selection, "
                    "polynomial fitting, and phase synthesis are deterministic"
                ),
            },
        },
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "summary": pd.DataFrame(summary_rows, columns=DEMO_SUMMARY_COLUMNS),
        "headline": headline,
        "readout_rows": readout_rows,
        "artifacts": artifacts,
    }


def _write_outputs(
    *,
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    headline: BlockDemoResult,
    readout_rows: list[dict[str, Any]],
    readout_metadata: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, Path]:
    summary_frame = pd.DataFrame(summary_rows, columns=DEMO_SUMMARY_COLUMNS)
    demo_summary_csv = output_dir / "demo_summary.csv"
    demo_summary_json = output_dir / "demo_summary.json"
    summary_frame.to_csv(demo_summary_csv, index=False)
    summary_frame.to_json(demo_summary_json, orient="records", indent=2)

    pipeline_json = output_dir / "qsvt_pipeline_metadata.json"
    write_json(pipeline_json, headline.pipeline_metadata)

    block_npy = output_dir / "selected_block.npy"
    residual_npy = output_dir / "selected_residual.npy"
    observable_npy = output_dir / "observable_vector.npy"
    np.save(block_npy, headline.H_block)
    np.save(residual_npy, headline.r_block)
    np.save(observable_npy, np.vstack([obs.vector for obs in headline.observables]))

    circuit_stats_json = output_dir / "circuit_stats.json"
    write_json(
        circuit_stats_json,
        {
            "headline_block": headline.row_common["block_shape"],
            "num_qubits": headline.row_common["num_qubits"],
            "raw_gate_count": headline.row_common["raw_gate_count"],
            "raw_circuit_depth": headline.row_common["raw_circuit_depth"],
            "transpiled_depth": headline.row_common["transpiled_depth"],
            "transpiled_cx_count": headline.row_common["transpiled_cx_count"],
            "transpiled_total_ops": headline.row_common["transpiled_total_ops"],
            "phase_count": headline.row_common["phase_count"],
            "degree": headline.row_common["degree"],
            "postselection_probability": headline.row_common["postselection_probability"],
            "basis_gates": ["u3", "cx"],
            "note": (
                "Gate-level QSVT operator circuit (dense block-encoding gate + "
                "projector-controlled phases); small simulator-scale evidence, not "
                "a scalable oracle compilation."
            ),
        },
    )

    normalization_json = output_dir / "normalization_record.json"
    write_json(
        normalization_json,
        {
            "convention": headline.pipeline_metadata["normalization_convention"],
            "beta": headline.row_common["beta"],
            "bound_C": headline.row_common["bound_C"],
            "alpha": headline.row_common["alpha"],
            "alpha_normalized": headline.row_common["alpha_normalized"],
            "physical_recovery_factor_C_over_beta": headline.row_common[
                "physical_recovery_factor_C_over_beta"
            ],
            "residual_norm": headline.row_common["residual_norm"],
            "postselection_probability": headline.row_common["postselection_probability"],
            "postselection_normalization": headline.row_common["postselection_normalization"],
            "physical_recovery_scale_for_unit_output": headline.physical_recovery_scale,
            "reconstruction": (
                "dx_alpha ~ (C/beta) * ||r|| * sqrt(p_success) * |psi_out>; "
                "y_l = (C/beta) ||r|| sqrt(p_success) ||l|| <l_hat|psi_out>"
            ),
        },
    )

    readout_csv = output_dir / "readout_diagnostics.csv"
    readout_meta_json = output_dir / "readout_metadata.json"
    if readout_rows:
        pd.DataFrame(readout_rows).to_csv(readout_csv, index=False)
    else:
        pd.DataFrame(columns=["observable_id", "shots", "physical_signed_value_estimate"]).to_csv(
            readout_csv, index=False
        )
    write_json(readout_meta_json, readout_metadata)

    readme = output_dir / "README.md"
    readme.write_text(_readme_markdown(summary_frame, headline, readout_rows), encoding="utf-8")

    return {
        "demo_summary_csv": demo_summary_csv,
        "demo_summary_json": demo_summary_json,
        "qsvt_pipeline_metadata_json": pipeline_json,
        "selected_block_npy": block_npy,
        "selected_residual_npy": residual_npy,
        "observable_vector_npy": observable_npy,
        "circuit_stats_json": circuit_stats_json,
        "normalization_record_json": normalization_json,
        "readout_diagnostics_csv": readout_csv,
        "readout_metadata_json": readout_meta_json,
        "readme_md": readme,
    }


def _readme_markdown(
    summary: pd.DataFrame, headline: BlockDemoResult, readout_rows: list[dict[str, Any]]
) -> str:
    head = headline.row_common
    lines = [
        "# Selected-Submatrix QSVT Transform and Isolated Overlap-Shot Test",
        "",
        DEMO_CLAIM_BOUNDARY,
        "",
        "Small selected-submatrix QSVT transform on an IEEE-derived weighted "
        "Jacobian block. The Ridge/Tikhonov update `dx_alpha = (H~^T H~ + alpha I)^{-1} "
        "H~^T r~` is the matched reference; the QSVT circuit implements the **same** "
        "regularized spectral filter at the **same** alpha. Selected linear functionals "
        "`y_l = l^T dx_alpha` only; this is **not** full-vector readout and **not** a "
        "quantum-hardware run.",
        "",
        f"## Headline block: {head['block_shape']} ({head['case']}), status = "
        f"`{head['status_label']}`",
        "",
        f"- condition number = {head['condition_number']:.3f}, "
        f"sigma in [{head['sigma_min']:.3g}, {head['sigma_max']:.3g}]",
        f"- alpha = {head['alpha']:.6g} (= {head['alpha_over_sigma_min_sq']:.3g} * sigma_min^2), "
        f"alpha/beta^2 = {head['alpha_normalized']:.4g}",
        f"- beta = {head['beta']:.6g}, bounded scale C = {head['bound_C']:.4f}, "
        f"physical recovery factor C/beta = {head['physical_recovery_factor_C_over_beta']:.6g}",
        f"- degree = {head['degree']}, phase count = {head['phase_count']} "
        f"({head['phase_synthesis_status']}), bounded max|p| = {head['bounded_max_abs']:.4f}",
        f"- polynomial fit max abs error = {head['polynomial_fit_max_abs_error']:.3e}",
        f"- circuit qubits = {head['num_qubits']}, raw depth = {head['raw_circuit_depth']}, "
        f"transpiled depth = {head['transpiled_depth']}, transpiled CX = "
        f"{head['transpiled_cx_count']}",
        f"- block-encoding unitarity error = {head['block_encoding_unitarity_error']:.2e}, "
        f"circuit-vs-exact SVT error = {head['svt_circuit_vs_exact_max_error']:.2e}",
        f"- postselection probability = {head['postselection_probability']:.4f}, "
        f"update relative error vs Ridge = {head['update_relative_error_vs_ridge']:.3e}",
        "",
        "## Selected observables (noiseless circuit reconstruction vs Ridge)",
        "",
        "| Observable | Type | Readout class | Ridge | QSVT | abs err | rel err |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary[summary["block_shape"] == head["block_shape"]].iterrows():
        lines.append(
            f"| `{row['observable_id']}` | {row['observable_type']} | "
            f"{row['observable_readout_class']} | {row['ridge_reference_value']:+.5e} | "
            f"{row['qsvt_statevector_value']:+.5e} | {row['absolute_error']:.2e} | "
            f"{row['relative_error']:.2e} |"
        )
    lines += [
        "",
        "## Isolated signed-overlap shot-noise experiment",
        "",
    ]
    if readout_rows:
        lines += [
            "| Observable | Shots | Backend | overlap est | signed value est | Ridge | abs err |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in readout_rows:
            lines.append(
                f"| `{row['observable_id']}` | {int(row['shots'])} | {row['backend']} | "
                f"{row['overlap_estimate']:+.4f} | "
                f"{row['physical_signed_value_estimate']:+.4e} | "
                f"{row['ridge_reference_value']:+.4e} | "
                f"{row['absolute_error_vs_ridge']:.2e} |"
            )
    else:
        lines.append("_Isolated overlap shots are reported only for a passing headline block._")
    lines += [
        "",
        "## Readout-class boundary",
        "",
        "- **Isolated overlap readout**: the signed functionals above directly load the "
        "classically computed postselected output state before Hadamard-overlap sampling.",
        "- **Integration status**: residual loading, QSVT, postselection, and shot readout "
        "are not composed into one executed circuit.",
        "- **Basis-sampling energy readout**: the energy-style observable is "
        "computational-basis accessible and does not recover the sign.",
        "- **Excluded full-vector recovery**: recovering every signed component would need "
        "one functional per coordinate and is out of scope.",
        "",
        "The 8x8 block (high condition number) is reported as degree/synthesis-limited "
        "rather than removed; the degree is the lever and the phase-synthesis ceiling is "
        "the binding constraint there.",
        "",
    ]
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(DEMO_DIR),
        "case": DEFAULT_CASE,
        "case_source": "pypower",
        "seed": DEFAULT_SEED,
        "angle_solver": "iterative",
        "margin": 1.05,
        "domain_low_factor": 0.9,
        "pass_relative_tolerance": PASS_RELATIVE_TOLERANCE,
        "readout_shots": [1_000, 10_000, 100_000],
        "readout_seed": 20260629,
        "blocks": None,
        "phase_cache_dir": str(PHASE_CACHE_DIR),
        "command": "run_selected_observable_qsvt_demo",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    if resolved["blocks"] is None:
        # 4x4 headline pass (alpha co-designed near sigma_min^2); 8x8 boundary row.
        resolved["blocks"] = [
            {"size": 4, "alpha": None, "degree": 31, "headline": True},
            {"size": 8, "alpha": None, "degree": 31, "headline": False},
        ]
    # Resolve per-block alpha defaults (alpha = alpha_mult * sigma_min^2) lazily here.
    resolved["readout_shots"] = [int(s) for s in resolved["readout_shots"]]
    return resolved


def _materialize_block_alphas(
    blocks: list[dict[str, Any]], H_full: np.ndarray, r_full: np.ndarray
) -> list[dict[str, Any]]:
    materialized = []
    for spec in blocks:
        size = int(spec["size"])
        H_block, _r, _rows, _cols = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy="largest_row_col_norms"
        )
        sigma_min = float(np.linalg.svd(H_block, compute_uv=False).min())
        new_spec = dict(spec)
        if new_spec.get("alpha") is None:
            multiplier = float(new_spec.get("alpha_mult", 4.0))
            new_spec["alpha"] = multiplier * sigma_min**2
        materialized.append(new_spec)
    return materialized


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the selected-observable QSVT demonstration")
    parser.add_argument("--output-dir", default=str(DEMO_DIR))
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--angle-solver", default="iterative")
    parser.add_argument("--degree-4x4", type=int, default=31)
    parser.add_argument("--degree-8x8", type=int, default=31)
    parser.add_argument("--alpha-mult", type=float, default=4.0)
    parser.add_argument("--readout-shots", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    args = parser.parse_args(argv)

    system, _ = build_engineering_system(
        {
            "case_name": args.case,
            "case_source": args.case_source,
            "matrix_source": "weighted_jacobian",
            "seed": int(args.seed),
        }
    )
    blocks = _materialize_block_alphas(
        [
            {
                "size": 4,
                "degree": int(args.degree_4x4),
                "alpha_mult": args.alpha_mult,
                "headline": True,
            },
            {
                "size": 8,
                "degree": int(args.degree_8x8),
                "alpha_mult": args.alpha_mult,
                "headline": False,
            },
        ],
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
    )
    run = run_selected_observable_qsvt_demo(
        {
            "output_dir": args.output_dir,
            "case": args.case,
            "case_source": args.case_source,
            "seed": args.seed,
            "angle_solver": args.angle_solver,
            "blocks": blocks,
            "readout_shots": args.readout_shots,
            "command": "scripts/run_selected_observable_qsvt_demo.py " + " ".join(argv or []),
        }
    )
    print(f"Selected-observable QSVT demo complete: {run['artifacts']['demo_summary_csv']}")
    print(f"Headline status: {run['headline'].status_label}")


if __name__ == "__main__":  # pragma: no cover
    main()
