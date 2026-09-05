from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.measurement.dc_linear import build_dc_weighted_system
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import (
    _build_phase_sequence,
    rescale_bounded_target_to_original,
)
from robust_qsvt_se.qsvt.gate_level_qsvt import (
    _export_qasm,
    _simulate_statevector,
    _transpile_and_summarize,
    build_gate_level_qsvt_state_circuit,
)
from robust_qsvt_se.qsvt.gate_level_qsvt_convention import (
    CORRECT_OPERATOR_EXTRACTION_RULE,
    CORRECT_STATE_EXTRACTION_RULE,
    IDENTIFIED_ERROR_SOURCE,
    global_phase_aligned,
    sign_aligned,
)
from robust_qsvt_se.qsvt.gate_state_preparation import (
    normalize_and_pad_for_gate_preparation,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.seed import make_rng

GATE_LEVEL_SOLVER_CLAIM = (
    "This is a small gate-level QSVT state-estimation solver for selected "
    "IEEE-derived weighted linearized subproblems. It does not demonstrate "
    "full IEEE-scale hardware execution, quantum speedup, QSVT superiority "
    "over Ridge/Tikhonov, field PMU/SCADA validation, or a full nonlinear "
    "AC QSVT solver."
)


@dataclass(frozen=True, slots=True)
class SelectedSubproblem:
    H_tilde: np.ndarray
    r_tilde: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GateLevelSolverComputation:
    H_tilde: np.ndarray
    r_tilde: np.ndarray
    qsvt_update: np.ndarray
    ridge_update: np.ndarray
    qsvt_state: np.ndarray
    ridge_state: np.ndarray
    statevector: np.ndarray
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    update_rows: list[dict[str, Any]]
    residual_rows: list[dict[str, Any]]
    observable_rows: list[dict[str, Any]]
    resource_row: dict[str, Any]
    qasm_text: str
    qasm_export_status: str
    report_markdown: str


def build_tiny_weighted_problem() -> SelectedSubproblem:
    """Return a deterministic 2x2 weighted state-estimation toy problem."""

    H_tilde = np.array(
        [
            [1.0, 0.24],
            [0.18, 0.82],
        ],
        dtype=np.float64,
    )
    r_tilde = np.array([0.37, -0.21], dtype=np.float64)
    metadata = {
        "case": "tiny_deterministic",
        "model": "weighted_linearized",
        "matrix_source": "deterministic_tiny_weighted_problem",
        "selected_measurement_indices": [0, 1],
        "selected_state_indices": [0, 1],
        "selected_measurement_labels": ["tiny_measurement_0", "tiny_measurement_1"],
        "selected_state_labels": ["tiny_state_0", "tiny_state_1"],
        "subproblem_selection_rule": "fixed deterministic 2x2 problem",
        "state_index_mapping": [
            {"local_index": 0, "source_index": 0, "label": "tiny_state_0", "state_type": "unknown"},
            {"local_index": 1, "source_index": 1, "label": "tiny_state_1", "state_type": "unknown"},
        ],
    }
    return SelectedSubproblem(H_tilde=H_tilde, r_tilde=r_tilde, metadata=metadata)


def extract_state_estimation_subproblem(
    *,
    case: str,
    model: str,
    submatrix_size: int,
    seed: int,
    case_source: str = "pypower",
) -> SelectedSubproblem:
    """Extract a deterministic square weighted state-estimation subproblem."""

    size = int(submatrix_size)
    if size <= 0 or not _is_power_of_two(size):
        raise ValueError("submatrix_size must be a positive power of two")
    model_name = str(model)
    if model_name == "ac_linearized":
        system, matrix_source = build_engineering_system(
            {
                "case_name": str(case),
                "case_source": str(case_source),
                "matrix_source": f"{case}_ac_weighted_jacobian",
                "seed": int(seed),
            }
        )
    elif model_name == "dc_linearized":
        system = build_dc_weighted_system(
            case_name=str(case),
            case_source=str(case_source),
            angle_scale=0.05,
            measurement_config={
                "include_branch_flows": True,
                "include_bus_injections": True,
                "angle_buses": (),
                "flow_std": 0.02,
                "injection_std": 0.03,
            },
            rng=make_rng(int(seed)),
            metadata={"qsvt_gate_level_solver_source": True},
        )
        matrix_source = f"{case}_{case_source}_dc_weighted_jacobian"
    else:
        raise ValueError("model must be ac_linearized or dc_linearized")

    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    if size > min(H_full.shape):
        raise ValueError(f"submatrix_size={size} exceeds weighted system shape {H_full.shape}")
    selected_columns = _top_indices(np.linalg.norm(H_full, axis=0), size)
    row_scores = np.linalg.norm(H_full[:, selected_columns], axis=1)
    selected_rows = _top_indices(row_scores, size)
    H_sub = H_full[np.ix_(selected_rows, selected_columns)]
    r_sub = np.asarray(system.r_tilde, dtype=np.float64)[selected_rows]
    metadata = _subproblem_metadata(
        system=system,
        matrix_source=matrix_source,
        case=case,
        model=model_name,
        selected_rows=selected_rows,
        selected_columns=selected_columns,
    )
    return SelectedSubproblem(H_tilde=H_sub, r_tilde=r_sub, metadata=metadata)


def ridge_tikhonov_update(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Compute ``(H.T H + alpha I)^-1 H.T r`` by SVD."""

    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    if H.ndim != 2 or r.ndim != 1 or H.shape[0] != r.size:
        raise ValueError("H_tilde must be m x n and r_tilde must have length m")
    U, singular_values, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt.T @ (ridge_filter(singular_values, alpha=float(alpha)) * (U.T @ r))


def solve_gate_level_state_estimation_problem(
    *,
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    alpha: float,
    degree: int,
    shots: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
    basis_gates: list[str] | None = None,
    transpile_optimization_level: int = 1,
    transpile_qubit_limit: int = 4,
    export_qasm: bool = True,
    angle_solver: str = "iterative",
    phase_grid_size: int | None = None,
    phase_timeout_seconds: int = 25,
) -> GateLevelSolverComputation:
    """Run the corrected gate-level QSVT update-state solver.

    ``angle_solver``, ``phase_grid_size`` and ``phase_timeout_seconds`` expose the
    already-configurable phase-synthesis settings of :func:`_build_phase_sequence`. The
    defaults reproduce the historical behaviour exactly (PennyLane ``iterative`` angle solver,
    grid ``max(2048, degree + 2)``, 25-second timeout); refinement passes vary them to probe the
    high-degree phase-synthesis ceiling without changing any other solver behaviour.
    """

    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("gate-level dense solver currently requires a square H_tilde")
    if not _is_power_of_two(H.shape[0]):
        raise ValueError("gate-level dense solver dimension must be a power of two")
    if r.ndim != 1 or r.size != H.shape[0]:
        raise ValueError("r_tilde length must match H_tilde rows")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if int(degree) <= 0 or int(degree) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if int(shots) <= 0:
        raise ValueError("shots must be positive")

    run_metadata = dict(metadata or {})
    dimension = int(H.shape[1])
    B = H.T
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = B / beta
    alpha_norm = float(alpha) / beta**2
    block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    block_report = validate_block_encoding(block_encoding, beta=beta, tolerance=1.0e-7)
    resolved_grid_size = int(phase_grid_size) if phase_grid_size else max(2048, int(degree) + 2)
    phase_sequence = _build_phase_sequence(
        singular_values_A=np.linalg.svd(A, compute_uv=False),
        requested_degree=int(degree),
        alpha_norm=alpha_norm,
        grid_size=max(resolved_grid_size, int(degree) + 2),
        max_synthesis_degree=int(degree),
        angle_solver=str(angle_solver),
        phase_timeout_seconds=int(phase_timeout_seconds),
    )
    state_preparation = normalize_and_pad_for_gate_preparation(
        r,
        target_dimension=block_encoding.unitary.shape[0],
    )
    circuit_bundle = build_gate_level_qsvt_state_circuit(
        block_unitary=block_encoding.unitary,
        phases=phase_sequence.phases,
        residual_state=state_preparation.padded_state,
        encoded_dimension=dimension,
    )
    statevector, simulation_seconds = _simulate_statevector(circuit_bundle.full_state_circuit)
    encoded_prefix = statevector[:dimension]
    bounded_update_on_normalized_residual = np.real(encoded_prefix)
    qsvt_update = state_preparation.original_norm * rescale_bounded_target_to_original(
        bounded_update_on_normalized_residual,
        beta=beta,
        C=phase_sequence.approximation.scale_factor,
    )
    ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
    qsvt_state = _normalize(qsvt_update)
    ridge_state = _normalize(ridge_update)
    phase_aligned_state, _ = global_phase_aligned(ridge_state, qsvt_state)
    sign_aligned_state, _ = sign_aligned(ridge_state, qsvt_state)

    residual_before = float(np.linalg.norm(r))
    residual_after_qsvt = float(np.linalg.norm(H @ qsvt_update - r))
    residual_after_ridge = float(np.linalg.norm(H @ ridge_update - r))
    full_update_error_abs = float(np.linalg.norm(qsvt_update - ridge_update))
    ridge_norm = float(np.linalg.norm(ridge_update))
    full_update_error_rel = full_update_error_abs / max(ridge_norm, 1.0e-15)
    state_error = float(np.linalg.norm(qsvt_state - ridge_state))
    phase_error = float(np.linalg.norm(phase_aligned_state - ridge_state))
    sign_error = float(np.linalg.norm(sign_aligned_state - ridge_state))
    success_probability = float(np.sum(np.abs(encoded_prefix) ** 2))

    transpile_summary = _transpile_and_summarize(
        circuit_bundle.full_state_circuit,
        basis_gates=basis_gates or ["rz", "sx", "x", "cx"],
        optimization_level=int(transpile_optimization_level),
        transpile_qubit_limit=int(transpile_qubit_limit),
    )
    qasm_text = (
        _export_qasm(transpile_summary["transpiled_circuit"] or circuit_bundle.full_state_circuit)
        if export_qasm
        else "// QASM export skipped for this run.\n"
    )
    qasm_status = "succeeded"
    if qasm_text.startswith("// QASM export failed"):
        qasm_status = "failed_with_text_fallback"
    elif qasm_text.startswith("// QASM export skipped"):
        qasm_status = "skipped"

    observables = _observable_rows(
        qsvt_state=qsvt_state,
        ridge_state=ridge_state,
        shots=int(shots),
        seed=int(seed),
        metadata=run_metadata,
    )
    max_exact_observable_error = _max_finite(row["absolute_error_vs_ridge"] for row in observables)
    max_shot_observable_error = _max_finite(row["absolute_sampling_error"] for row in observables)

    circuit_depth = (
        transpile_summary["resource_row"].get("depth_after_transpile")
        or transpile_summary["resource_row"]["depth_before_transpile"]
    )
    resource_row = {
        **transpile_summary["resource_row"],
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "requested_degree": int(degree),
        "constructed_polynomial_degree": int(phase_sequence.coefficients.size - 1),
        "synthesized_phase_degree": int(phase_sequence.actual_degree),
        "effective_qsvt_degree": int(phase_sequence.actual_degree),
        "synthesized_degree": int(phase_sequence.actual_degree),
        "phase_count": int(phase_sequence.phases.size),
        "qsvt_query_count": int(circuit_bundle.block_encoding_gate_count),
        "circuit_depth": int(circuit_depth),
        "two_qubit_gate_count": int(transpile_summary["two_qubit_gate_count"]),
    }
    summary = {
        "claim_boundary": GATE_LEVEL_SOLVER_CLAIM,
        "model": run_metadata.get("model", "weighted_linearized"),
        "case": run_metadata.get("case", "custom"),
        "submatrix_size": int(dimension),
        "matrix_shape": [int(H.shape[0]), int(H.shape[1])],
        "beta": beta,
        "alpha": float(alpha),
        "alpha_norm": alpha_norm,
        "requested_degree": int(degree),
        "constructed_polynomial_degree": int(phase_sequence.coefficients.size - 1),
        "synthesized_phase_degree": int(phase_sequence.actual_degree),
        "effective_qsvt_degree": int(phase_sequence.actual_degree),
        "synthesized_degree": int(phase_sequence.actual_degree),
        "phase_count": int(phase_sequence.phases.size),
        "qsvt_query_count": int(circuit_bundle.block_encoding_gate_count),
        "state_error_vs_ridge": state_error,
        "phase_or_sign_aligned_state_error": min(phase_error, sign_error),
        "phase_aligned_state_error": phase_error,
        "sign_aligned_state_error": sign_error,
        "full_update_error_if_available": full_update_error_rel,
        "full_update_l2_error": full_update_error_abs,
        "residual_before_update": residual_before,
        "residual_after_qsvt_update": residual_after_qsvt,
        "residual_after_ridge_update": residual_after_ridge,
        "residual_reduction_ratio_vs_no_update": (
            residual_after_qsvt / max(residual_before, 1.0e-15)
        ),
        "success_probability": success_probability,
        "postselection_probability": success_probability,
        "observables_evaluated": len(observables),
        "max_exact_observable_error": max_exact_observable_error,
        "max_shot_observable_error": max_shot_observable_error,
        "shot_based_observable_error": max_shot_observable_error,
        "circuit_depth": int(circuit_depth),
        "two_qubit_gate_count": int(transpile_summary["two_qubit_gate_count"]),
        "qasm_export_status": qasm_status,
        "correct_operator_extraction_rule": CORRECT_OPERATOR_EXTRACTION_RULE,
        "correct_state_extraction_rule": CORRECT_STATE_EXTRACTION_RULE,
        "identified_error_source": IDENTIFIED_ERROR_SOURCE,
        "simulation_seconds": simulation_seconds,
        "block_encoding_report": block_report,
        "selected_state_labels": run_metadata.get("selected_state_labels", []),
        "selected_measurement_labels": run_metadata.get("selected_measurement_labels", []),
        "selected_state_indices": run_metadata.get("selected_state_indices", []),
        "selected_measurement_indices": run_metadata.get("selected_measurement_indices", []),
        "limitations": [
            "Dense block encoding and Qiskit Initialize are small simulator primitives.",
            "The update vector is recovered from statevector/norm metadata.",
            "Real-quadrature extraction is validated by simulator matrix access.",
            "Ridge/Tikhonov is the reference target, not a method being outperformed.",
        ],
    }
    update_rows = _update_rows(qsvt_update, ridge_update, run_metadata)
    residual_rows = [
        {
            "residual_before_update": residual_before,
            "residual_after_qsvt_update": residual_after_qsvt,
            "residual_after_ridge_update": residual_after_ridge,
            "residual_reduction_ratio_vs_no_update": summary[
                "residual_reduction_ratio_vs_no_update"
            ],
        }
    ]
    diagnostics = {
        **summary,
        "state_preparation": state_preparation.to_dict(),
        "qsvt_update": qsvt_update.tolist(),
        "ridge_update": ridge_update.tolist(),
        "qsvt_normalized_state": np.real(qsvt_state).tolist(),
        "ridge_normalized_state": np.real(ridge_state).tolist(),
        "singular_values_B": singular_values_B.tolist(),
        "singular_values_H": np.linalg.svd(H, compute_uv=False).tolist(),
        "bounded_scaling_C": float(phase_sequence.approximation.scale_factor),
        "qsvt_polynomial_coefficients": phase_sequence.coefficients.tolist(),
        "phase_selection_reason": phase_sequence.selection_reason,
        "candidate_phase_errors": phase_sequence.candidate_errors,
        "metadata": run_metadata,
    }
    report = _solver_report_markdown(summary, update_rows, residual_rows, observables, run_metadata)
    return GateLevelSolverComputation(
        H_tilde=H,
        r_tilde=r,
        qsvt_update=qsvt_update,
        ridge_update=ridge_update,
        qsvt_state=qsvt_state,
        ridge_state=ridge_state,
        statevector=statevector,
        summary=summary,
        diagnostics=diagnostics,
        update_rows=update_rows,
        residual_rows=residual_rows,
        observable_rows=observables,
        resource_row=resource_row,
        qasm_text=qasm_text,
        qasm_export_status=qasm_status,
        report_markdown=report,
    )


def run_tiny_gate_level_state_estimation_solver(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = {
        "alpha": 1.0e-2,
        "degree": 9,
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/gate_level_qsvt_tiny_solver",
    }
    if config:
        resolved.update(config)
    problem = build_tiny_weighted_problem()
    computation = solve_gate_level_state_estimation_problem(
        H_tilde=problem.H_tilde,
        r_tilde=problem.r_tilde,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        shots=int(resolved["shots"]),
        seed=int(resolved["seed"]),
        metadata=problem.metadata,
        transpile_qubit_limit=3,
        export_qasm=False,
    )
    output_dir = ensure_directory(resolved["output_dir"])
    artifacts = _write_tiny_outputs(output_dir, resolved, computation)
    return {
        "output_dir": output_dir,
        "summary": computation.summary,
        "diagnostics": computation.diagnostics,
        "artifacts": artifacts,
    }


def run_ieee_gate_level_state_estimation_solver(
    config: dict[str, Any],
) -> dict[str, Any]:
    resolved = _resolve_ieee_solver_config(config)
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    computation = solve_gate_level_state_estimation_problem(
        H_tilde=subproblem.H_tilde,
        r_tilde=subproblem.r_tilde,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        shots=int(resolved["shots"]),
        seed=int(resolved["seed"]),
        metadata=subproblem.metadata,
        transpile_qubit_limit=int(resolved["transpile_qubit_limit"]),
        export_qasm=True,
    )
    output_dir = ensure_directory(resolved["output_dir"])
    artifacts = _write_ieee_outputs(output_dir, resolved, computation)
    return {
        "output_dir": output_dir,
        "summary": computation.summary,
        "diagnostics": computation.diagnostics,
        "artifacts": artifacts,
    }


def run_dense_explicit_solver_limit_study(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_sizes": [4, 8, 16],
        "alpha": 1.0e-4,
        "degree": 35,
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/dense_explicit_qsvt_solver_limit_study",
        "transpile_qubit_limit": 3,
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    summary_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for size in [int(value) for value in resolved["submatrix_sizes"]]:
        start = time.perf_counter()
        try:
            subproblem = extract_state_estimation_subproblem(
                case=str(resolved["case"]),
                model=str(resolved["model"]),
                submatrix_size=size,
                seed=int(resolved["seed"]),
                case_source=str(resolved["case_source"]),
            )
            computation = solve_gate_level_state_estimation_problem(
                H_tilde=subproblem.H_tilde,
                r_tilde=subproblem.r_tilde,
                alpha=float(resolved["alpha"]),
                degree=int(resolved["degree"]),
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                metadata=subproblem.metadata,
                transpile_qubit_limit=int(resolved["transpile_qubit_limit"]),
                export_qasm=False,
            )
            seconds = time.perf_counter() - start
            row = _dense_limit_row(size, computation, seconds, "completed", "")
            summary_rows.append(row)
            resource_rows.append({**row, **computation.resource_row})
        except Exception as exc:  # pragma: no cover - exercised by resource limits/version issues
            seconds = time.perf_counter() - start
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(f"- {size}x{size}: {reason}")
            row = _dense_failure_row(size, seconds, reason)
            summary_rows.append(row)
            resource_rows.append(row)

    summary_path = output_dir / "dense_limit_summary.csv"
    resource_path = output_dir / "dense_limit_resource_summary.csv"
    failure_path = output_dir / "dense_limit_failures.md"
    interpretation_path = output_dir / "dense_limit_interpretation.md"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(resource_rows).to_csv(resource_path, index=False)
    failure_path.write_text(_dense_failures_markdown(failures), encoding="utf-8")
    interpretation_path.write_text(
        _dense_interpretation_markdown(summary_rows),
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "resource_rows": resource_rows,
        "artifacts": {
            "dense_limit_summary": summary_path,
            "dense_limit_resource_summary": resource_path,
            "dense_limit_failures": failure_path,
            "dense_limit_interpretation": interpretation_path,
        },
    }


def _resolve_ieee_solver_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/gate_level_qsvt_ieee14_solver",
        "transpile_qubit_limit": 4,
        "observable_mode": "default",
        "use_corrected_extraction_rule": True,
        "attempt_rescaled_update": True,
    }
    resolved.update(config)
    if str(resolved["model"]) not in {"ac_linearized", "dc_linearized"}:
        raise ValueError("model must be ac_linearized or dc_linearized")
    size = int(resolved["submatrix_size"])
    if size <= 0 or not _is_power_of_two(size):
        raise ValueError("submatrix_size must be a positive power of two")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if int(resolved["degree"]) <= 0 or int(resolved["degree"]) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if int(resolved["shots"]) <= 0:
        raise ValueError("shots must be positive")
    return resolved


def _write_tiny_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    computation: GateLevelSolverComputation,
) -> dict[str, Path]:
    diagnostics_path = output_dir / "solver_diagnostics.json"
    state_path = output_dir / "state_error_summary.csv"
    residual_path = output_dir / "residual_reduction_summary.csv"
    resource_path = output_dir / "circuit_resource_summary.csv"
    summary_path = output_dir / "solver_summary.md"
    write_json(diagnostics_path, computation.diagnostics)
    pd.DataFrame([_state_error_row(computation.summary)]).to_csv(state_path, index=False)
    pd.DataFrame(computation.residual_rows).to_csv(residual_path, index=False)
    pd.DataFrame([computation.resource_row]).to_csv(resource_path, index=False)
    summary_path.write_text(computation.report_markdown, encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "solver_diagnostics": str(diagnostics_path),
            "state_error_summary": str(state_path),
            "residual_reduction_summary": str(residual_path),
            "circuit_resource_summary": str(resource_path),
            "solver_summary": str(summary_path),
        },
        input_config=resolved,
        claim_boundary=GATE_LEVEL_SOLVER_CLAIM,
    )
    return {
        "manifest": manifest,
        "solver_diagnostics": diagnostics_path,
        "state_error_summary": state_path,
        "residual_reduction_summary": residual_path,
        "circuit_resource_summary": resource_path,
        "solver_summary": summary_path,
    }


def _write_ieee_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    computation: GateLevelSolverComputation,
) -> dict[str, Path]:
    qasm_path = output_dir / "qsvt_solver_circuit.qasm"
    summary_path = output_dir / "qsvt_solver_circuit_summary.json"
    resource_path = output_dir / "qsvt_solver_transpiled_resources.csv"
    update_path = output_dir / "qsvt_vs_ridge_update_summary.csv"
    residual_path = output_dir / "residual_reduction_summary.csv"
    observable_path = output_dir / "observable_readout_summary.csv"
    report_path = output_dir / "state_estimation_solver_report.md"
    diagnostics_path = output_dir / "solver_diagnostics.json"
    qasm_path.write_text(computation.qasm_text, encoding="utf-8")
    write_json(summary_path, computation.summary)
    write_json(diagnostics_path, computation.diagnostics)
    pd.DataFrame([computation.resource_row]).to_csv(resource_path, index=False)
    pd.DataFrame(computation.update_rows).to_csv(update_path, index=False)
    pd.DataFrame(computation.residual_rows).to_csv(residual_path, index=False)
    pd.DataFrame(computation.observable_rows).to_csv(observable_path, index=False)
    report_path.write_text(computation.report_markdown, encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "qsvt_solver_circuit_qasm": str(qasm_path),
            "qsvt_solver_circuit_summary": str(summary_path),
            "qsvt_solver_transpiled_resources": str(resource_path),
            "qsvt_vs_ridge_update_summary": str(update_path),
            "residual_reduction_summary": str(residual_path),
            "observable_readout_summary": str(observable_path),
            "state_estimation_solver_report": str(report_path),
            "solver_diagnostics": str(diagnostics_path),
        },
        input_config=resolved,
        claim_boundary=GATE_LEVEL_SOLVER_CLAIM,
    )
    return {
        "manifest": manifest,
        "qsvt_solver_circuit_qasm": qasm_path,
        "qsvt_solver_circuit_summary": summary_path,
        "qsvt_solver_transpiled_resources": resource_path,
        "qsvt_vs_ridge_update_summary": update_path,
        "residual_reduction_summary": residual_path,
        "observable_readout_summary": observable_path,
        "state_estimation_solver_report": report_path,
        "solver_diagnostics": diagnostics_path,
    }


def _subproblem_metadata(
    *,
    system: WeightedSystem,
    matrix_source: str,
    case: str,
    model: str,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> dict[str, Any]:
    measurement_labels = list(system.metadata.get("measurement_labels", []))
    measurement_types = list(system.metadata.get("measurement_types", []))
    if len(measurement_labels) < system.n_measurements:
        measurement_labels = [f"measurement_{index}" for index in range(system.n_measurements)]
    state_labels = _state_labels(system.metadata, system.n_states)
    selected_state_labels = [state_labels[index] for index in selected_columns]
    selected_measurement_labels = [measurement_labels[index] for index in selected_rows]
    try:
        state_metadata = build_state_metadata_from_system_metadata(system.metadata)
        state_records = []
        for local_index, source_index in enumerate(selected_columns):
            record = state_metadata.record_for_index(int(source_index))
            state_records.append(
                {
                    "local_index": int(local_index),
                    "source_index": int(source_index),
                    "label": selected_state_labels[local_index],
                    "state_type": record.state_type,
                    "bus_id": record.bus_id,
                    "description": record.description,
                }
            )
    except Exception:
        state_records = [
            {
                "local_index": int(local_index),
                "source_index": int(source_index),
                "label": selected_state_labels[local_index],
                "state_type": _state_type_from_label(selected_state_labels[local_index]),
                "bus_id": None,
                "description": selected_state_labels[local_index],
            }
            for local_index, source_index in enumerate(selected_columns)
        ]
    return {
        "case": str(case),
        "model": str(model),
        "matrix_source": matrix_source,
        "full_weighted_jacobian_shape": [int(system.n_measurements), int(system.n_states)],
        "subproblem_selection_rule": (
            "state columns selected by largest weighted-Jacobian column norms; "
            "measurement rows selected by largest norms restricted to those columns"
        ),
        "selected_measurement_indices": selected_rows.astype(int).tolist(),
        "selected_state_indices": selected_columns.astype(int).tolist(),
        "selected_measurement_labels": selected_measurement_labels,
        "selected_measurement_types": [
            measurement_types[index] if index < len(measurement_types) else "unknown"
            for index in selected_rows
        ],
        "selected_state_labels": selected_state_labels,
        "state_index_mapping": state_records,
        "source_system_metadata": {
            "mode": system.metadata.get("mode"),
            "note": system.metadata.get("note"),
            "slack_bus": system.metadata.get("slack_bus"),
        },
    }


def _state_labels(metadata: dict[str, Any], count: int) -> list[str]:
    angle_buses = list(metadata.get("angle_state_buses", []))
    voltage_buses = list(metadata.get("voltage_state_buses", []))
    if angle_buses or voltage_buses:
        labels = [f"theta_{bus}" for bus in angle_buses] + [f"V_{bus}" for bus in voltage_buses]
    else:
        buses = list(metadata.get("state_buses", []))
        labels = [f"theta_{bus}" for bus in buses]
    if len(labels) < int(count):
        labels = [f"state_{index}" for index in range(int(count))]
    return labels


def _state_type_from_label(label: str) -> str:
    if label.startswith("theta_"):
        return "angle"
    if label.startswith("V_"):
        return "voltage"
    return "unknown"


def _observable_rows(
    *,
    qsvt_state: np.ndarray,
    ridge_state: np.ndarray,
    shots: int,
    seed: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    probabilities = np.abs(qsvt_state) ** 2
    counts = rng.multinomial(int(shots), probabilities / max(float(np.sum(probabilities)), 1.0e-15))
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, tuple[int, ...]]] = [
        ("component_0_probability", "basis_probability", (0,)),
        ("component_1_probability", "basis_probability", (min(1, qsvt_state.size - 1),)),
        ("first_two_state_energy", "subset_probability", tuple(range(min(2, qsvt_state.size)))),
    ]
    angle_indices = [
        int(record["local_index"])
        for record in metadata.get("state_index_mapping", [])
        if record.get("state_type") == "angle"
    ]
    if angle_indices:
        specs.append(("bus_angle_component_probability", "basis_probability", (angle_indices[0],)))
    if len(angle_indices) >= 2:
        exact_qsvt = float(abs(qsvt_state[angle_indices[0]] - qsvt_state[angle_indices[1]]) ** 2)
        exact_ridge = float(abs(ridge_state[angle_indices[0]] - ridge_state[angle_indices[1]]) ** 2)
        rows.append(
            {
                "observable_name": "branch_angle_difference_proxy",
                "observable_type": "difference_proxy",
                "indices": f"{angle_indices[0]} {angle_indices[1]}",
                "shots": int(shots),
                "seed": int(seed),
                "true_qsvt_value": exact_qsvt,
                "shot_estimate": np.nan,
                "standard_error": np.nan,
                "absolute_sampling_error": np.nan,
                "ridge_reference_value": exact_ridge,
                "absolute_error_vs_ridge": abs(exact_qsvt - exact_ridge),
                "notes": (
                    "Exact statevector proxy for angle-difference energy; no dedicated "
                    "Hadamard-test circuit is implemented here."
                ),
            }
        )
    for name, observable_type, indices in specs:
        exact_qsvt = _observable_value(qsvt_state, observable_type, indices)
        exact_ridge = _observable_value(ridge_state, observable_type, indices)
        selected_count = int(sum(counts[index] for index in indices))
        estimate = selected_count / float(shots)
        rows.append(
            {
                "observable_name": name,
                "observable_type": observable_type,
                "indices": " ".join(str(index) for index in indices),
                "shots": int(shots),
                "seed": int(seed),
                "true_qsvt_value": exact_qsvt,
                "shot_estimate": estimate,
                "standard_error": _bernoulli_se(exact_qsvt, int(shots)),
                "absolute_sampling_error": abs(estimate - exact_qsvt),
                "ridge_reference_value": exact_ridge,
                "absolute_error_vs_ridge": abs(exact_qsvt - exact_ridge),
                "notes": (
                    "Multinomial sampling proxy on the corrected extracted update state; "
                    "full quadrature readout circuitry is not implemented."
                ),
            }
        )
    return rows


def _observable_value(state: np.ndarray, observable_type: str, indices: tuple[int, ...]) -> float:
    if observable_type == "basis_probability":
        return float(np.abs(state[indices[0]]) ** 2)
    if observable_type == "subset_probability":
        return float(np.sum(np.abs(state[list(indices)]) ** 2))
    raise ValueError(f"unsupported observable_type: {observable_type}")


def _update_rows(
    qsvt_update: np.ndarray,
    ridge_update: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = list(metadata.get("selected_state_labels", []))
    indices = list(metadata.get("selected_state_indices", []))
    rows = []
    for local_index, (qsvt_value, ridge_value) in enumerate(
        zip(qsvt_update, ridge_update, strict=True)
    ):
        rows.append(
            {
                "local_state_index": int(local_index),
                "source_state_index": (
                    int(indices[local_index]) if local_index < len(indices) else int(local_index)
                ),
                "state_label": labels[local_index] if local_index < len(labels) else "",
                "qsvt_update": float(qsvt_value),
                "ridge_update": float(ridge_value),
                "absolute_error": abs(float(qsvt_value - ridge_value)),
            }
        )
    return rows


def _state_error_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_error_vs_ridge": summary["state_error_vs_ridge"],
        "phase_or_sign_aligned_state_error": summary["phase_or_sign_aligned_state_error"],
        "full_update_error_if_available": summary["full_update_error_if_available"],
        "success_probability": summary["success_probability"],
        "postselection_probability": summary["postselection_probability"],
        "correct_extraction_rule": summary["correct_state_extraction_rule"],
    }


def _solver_report_markdown(
    summary: dict[str, Any],
    update_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    residual = residual_rows[0]
    lines = [
        "# Gate-Level QSVT State-Estimation Solver",
        "",
        GATE_LEVEL_SOLVER_CLAIM,
        "",
        "## Solver Summary",
        f"- Model: {summary['model']}",
        f"- Matrix shape: {summary['matrix_shape'][0]} x {summary['matrix_shape'][1]}",
        f"- Alpha: {summary['alpha']:.17g}",
        f"- Beta: {summary['beta']:.17g}",
        f"- Degree: {summary['synthesized_degree']} (requested {summary['requested_degree']})",
        f"- Extraction rule: {summary['correct_state_extraction_rule']}",
        f"- State error vs Ridge: {summary['state_error_vs_ridge']:.17g}",
        f"- Phase/sign-aligned state error: {summary['phase_or_sign_aligned_state_error']:.17g}",
        f"- Success probability: {summary['success_probability']:.17g}",
        "",
        "## Residual Comparison",
        f"- Before update: {residual['residual_before_update']:.17g}",
        f"- After QSVT update: {residual['residual_after_qsvt_update']:.17g}",
        f"- After Ridge update: {residual['residual_after_ridge_update']:.17g}",
        "",
        "## Update-Vector Comparison",
        "| local index | source index | label | QSVT update | Ridge update | abs error |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in update_rows:
        lines.append(
            f"| {row['local_state_index']} | {row['source_state_index']} | "
            f"{row['state_label']} | {row['qsvt_update']:.8e} | "
            f"{row['ridge_update']:.8e} | {row['absolute_error']:.8e} |"
        )
    lines.extend(
        [
            "",
            "## Observable Readout",
            f"- Observables evaluated: {len(observable_rows)}",
            "- Shot estimates are multinomial proxies on the corrected extracted state; "
            "a dedicated quadrature-readout circuit is not implemented.",
            "",
            "## One-Step State-Estimation Interpretation",
            f"- Model type: {summary['model']}.",
            f"- Subproblem selection rule: {metadata.get('subproblem_selection_rule', 'unknown')}.",
            "- State-index mapping:",
        ]
    )
    for record in metadata.get("state_index_mapping", []):
        lines.append(
            f"  - local {record.get('local_index')} -> source {record.get('source_index')}: "
            f"{record.get('label')} ({record.get('state_type')})"
        )
    records = metadata.get("state_index_mapping", [])
    selected_types = sorted({str(record.get("state_type", "unknown")) for record in records})
    lines.extend(
        [
            "- Selected components correspond to: "
            f"{', '.join(selected_types) if selected_types else 'index-level update entries'}.",
            "- The circuit update state is compared component-wise against the "
            "Ridge/Tikhonov one-step update.",
            "- The residual comparison uses the selected weighted subproblem "
            "H_tilde_sub Delta x ~= r_tilde_sub.",
            "- Observable readout reports component probabilities, subset energy, "
            "and angle proxies when state metadata exposes angle entries.",
            "- Limitation: this is a selected subproblem and one linearized update, "
            "not a full nonlinear AC QSVT solve.",
            "",
            "## Limitations",
            "- Dense block encoding is an explicit small-matrix simulator primitive.",
            "- Norm recovery and real-quadrature extraction use simulator metadata.",
            "- The QASM artifact may contain dense unitary gates or a Qiskit text fallback.",
            "- Ridge/Tikhonov is the reference target; no superiority claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _dense_limit_row(
    size: int,
    computation: GateLevelSolverComputation,
    seconds: float,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "ancilla_qubits": 1,
        "depth": computation.summary["circuit_depth"],
        "two_qubit_gates": computation.summary["two_qubit_gate_count"],
        "qsvt_query_count": computation.summary["qsvt_query_count"],
        "transpile_time": computation.resource_row.get("transpile_seconds", np.nan),
        "state_error_vs_ridge": computation.summary["state_error_vs_ridge"],
        "residual_reduction_ratio": computation.summary["residual_reduction_ratio_vs_no_update"],
        "success_probability": computation.summary["success_probability"],
        "run_status": status,
        "failure_reason_if_any": failure_reason,
        "wall_seconds": float(seconds),
    }


def _dense_failure_row(size: int, seconds: float, reason: str) -> dict[str, Any]:
    return {
        "submatrix_size": int(size),
        "matrix_shape": f"{size}x{size}",
        "qubits": int(np.ceil(np.log2(2 * int(size)))),
        "ancilla_qubits": 1,
        "depth": np.nan,
        "two_qubit_gates": np.nan,
        "qsvt_query_count": np.nan,
        "transpile_time": np.nan,
        "state_error_vs_ridge": np.nan,
        "residual_reduction_ratio": np.nan,
        "success_probability": np.nan,
        "run_status": "failed",
        "failure_reason_if_any": reason,
        "wall_seconds": float(seconds),
    }


def _dense_failures_markdown(failures: list[str]) -> str:
    if not failures:
        return "# Dense Explicit QSVT Solver Limit Failures\n\nNo solver failures were raised.\n"
    return "# Dense Explicit QSVT Solver Limit Failures\n\n" + "\n".join(failures) + "\n"


def _dense_interpretation_markdown(rows: list[dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    largest = max((int(row["submatrix_size"]) for row in completed), default=None)
    return "\n".join(
        [
            "# Dense Explicit QSVT Solver Limit Interpretation",
            "",
            GATE_LEVEL_SOLVER_CLAIM,
            "",
            f"- Sizes attempted: {', '.join(str(row['submatrix_size']) for row in rows)}",
            f"- Largest completed statevector run: {largest}",
            "- Dense unitary depth and two-qubit counts are resource diagnostics for "
            "small explicit circuits, not scalability evidence.",
            "- Transpilation may be skipped by qubit limit to avoid turning dense "
            "unitaries into very large gate lists.",
            "",
        ]
    )


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    indices = np.arange(values.size)
    order = np.lexsort((indices, -np.asarray(values, dtype=np.float64)))
    return np.sort(order[: int(count)])


def _normalize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _bernoulli_se(probability: float, shots: int) -> float:
    p = float(np.clip(probability, 0.0, 1.0))
    return float(np.sqrt(p * (1.0 - p) / max(int(shots), 1)))


def _max_finite(values: Any) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def _is_power_of_two(value: int) -> bool:
    return int(value) > 0 and (int(value) & (int(value) - 1)) == 0
