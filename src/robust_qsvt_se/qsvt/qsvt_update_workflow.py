from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.block_encoding import (
    canonical_square_block_encoding,
    validate_block_encoding,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import (
    _apply_explicit_block_encoded_qsvt,
    _build_phase_sequence,
    _build_source_matrix,
    _is_power_of_two,
    rescale_bounded_target_to_original,
)
from robust_qsvt_se.qsvt.norm_success import compute_norm_success_diagnostics
from robust_qsvt_se.qsvt.partial_observable_readout import normalize_state
from robust_qsvt_se.utils.io import ensure_directory, write_json

UPDATE_WORKFLOW_CLAIM = (
    "This is a full QSVT implementation pathway with small explicit simulations "
    "and resource-aware diagnostics. It does not demonstrate quantum speedup, "
    "full IEEE-scale hardware execution, or QSVT numerical superiority over "
    "Ridge/Tikhonov."
)


@dataclass(frozen=True, slots=True)
class ResidualStatePreparationResult:
    normalized_residual_state: np.ndarray
    padded_residual_state: np.ndarray
    residual_norm: float
    dimension: int
    padded_dimension: int
    padding_width: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_residual_state": self.normalized_residual_state.tolist(),
            "padded_residual_state": self.padded_residual_state.tolist(),
            "residual_norm": float(self.residual_norm),
            "dimension": int(self.dimension),
            "padded_dimension": int(self.padded_dimension),
            "padding_width": int(self.padding_width),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class RidgeReferenceResult:
    update_vector: np.ndarray
    normalized_update_state: np.ndarray
    update_norm: float
    residual_norm: float
    filter_values: np.ndarray
    singular_values: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_vector": self.update_vector.tolist(),
            "normalized_update_state": self.normalized_update_state.tolist(),
            "update_norm": float(self.update_norm),
            "residual_norm": float(self.residual_norm),
            "filter_values": self.filter_values.tolist(),
            "singular_values": self.singular_values.tolist(),
        }


@dataclass(frozen=True, slots=True)
class QSVTUpdateResult:
    qsvt_normalized_update_state: np.ndarray
    qsvt_unnormalized_vector: np.ndarray
    observable_ready_state: np.ndarray
    ridge_reference: RidgeReferenceResult
    residual_state_preparation: ResidualStatePreparationResult
    success_probability_proxy: float
    bounded_target_scaling_C: float
    beta: float
    phase_count: int
    synthesized_degree: int
    requested_degree: int
    query_count_estimate: int
    state_l2_error_against_ridge: float
    phase_aligned_state_l2_error: float
    real_sign_aligned_state_l2_error: float
    normalized_state_overlap_abs: float
    block_encoding_report: dict[str, Any]
    simulation_mode: str
    phase_method: str
    limitations: tuple[str, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "ridge_update_norm": float(self.ridge_reference.update_norm),
            "qsvt_state_norm_before_normalization": float(
                np.linalg.norm(self.qsvt_unnormalized_vector)
            ),
            "qsvt_state_norm_after_normalization": float(
                np.linalg.norm(self.qsvt_normalized_update_state)
            ),
            "normalized_state_overlap_abs": float(self.normalized_state_overlap_abs),
            "state_l2_error_against_ridge": float(self.state_l2_error_against_ridge),
            "phase_aligned_state_l2_error": float(self.phase_aligned_state_l2_error),
            "real_sign_aligned_state_l2_error": float(self.real_sign_aligned_state_l2_error),
            "success_probability_proxy": float(self.success_probability_proxy),
            "bounded_target_scaling_C": float(self.bounded_target_scaling_C),
            "beta": float(self.beta),
            "requested_degree": int(self.requested_degree),
            "synthesized_degree": int(self.synthesized_degree),
            "phase_count": int(self.phase_count),
            "query_count_estimate": int(self.query_count_estimate),
            "simulation_mode": self.simulation_mode,
            "phase_method": self.phase_method,
            "block_encoding_report": self.block_encoding_report,
            "limitations": list(self.limitations),
        }


def prepare_weighted_residual_state(
    r_tilde: np.ndarray,
    eps: float = 1.0e-15,
) -> ResidualStatePreparationResult:
    """Prepare ``|r_tilde> = r_tilde / ||r_tilde||`` with padding metadata."""

    residual = np.asarray(r_tilde, dtype=np.complex128)
    if residual.ndim != 1:
        raise ValueError("r_tilde must be one-dimensional")
    if not np.all(np.isfinite(residual)):
        raise ValueError("r_tilde must contain finite values")
    norm = float(np.linalg.norm(residual))
    if norm <= float(eps):
        raise ValueError("weighted residual norm is too small for state preparation")
    normalized = residual / norm
    padded_dimension = _next_power_of_two(residual.size)
    padding_width = padded_dimension - residual.size
    padded = np.pad(normalized, (0, padding_width))
    warnings: list[str] = []
    if padding_width:
        warnings.append(
            f"residual state padded from dimension {residual.size} to {padded_dimension}"
        )
    if norm <= 1.0e-12:
        warnings.append("residual norm is very small; state preparation is ill-conditioned")
    return ResidualStatePreparationResult(
        normalized_residual_state=normalized,
        padded_residual_state=padded,
        residual_norm=norm,
        dimension=int(residual.size),
        padded_dimension=int(padded_dimension),
        padding_width=int(padding_width),
        warnings=tuple(warnings),
    )


def compute_ridge_update_reference(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    alpha: float,
) -> RidgeReferenceResult:
    """Compute the Ridge/Tikhonov reference update by SVD."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    matrix = np.asarray(H_tilde, dtype=np.float64)
    residual = np.asarray(r_tilde, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("H_tilde must be a matrix")
    if residual.ndim != 1 or residual.size != matrix.shape[0]:
        raise ValueError("r_tilde length must match H_tilde rows")
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    filter_values = ridge_filter(singular_values, alpha=float(alpha))
    update = Vt.T @ (filter_values * (U.T @ residual))
    update_state, update_norm = normalize_state(update)
    return RidgeReferenceResult(
        update_vector=np.real(update),
        normalized_update_state=update_state,
        update_norm=float(update_norm),
        residual_norm=float(np.linalg.norm(residual)),
        filter_values=filter_values,
        singular_values=singular_values,
    )


def run_qsvt_update_state_simulation(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    alpha: float,
    degree: int,
    block_encoding_mode: str = "explicit_dense",
    phase_method: str = "pennylane_poly_to_angles",
    seed: int = 123,
) -> QSVTUpdateResult:
    """Run a small QSVT-oriented update-state simulation.

    ``explicit_dense`` uses the repository's dense block encoding and PennyLane
    QSVT phase path. Other modes return a clearly labeled SVD target proxy for
    resource accounting only; they are not reported as implemented dense QSVT.
    """

    _ = int(seed)  # deterministic placeholder for future randomized state prep.
    residual_state = prepare_weighted_residual_state(r_tilde)
    ridge = compute_ridge_update_reference(H_tilde, r_tilde, alpha=float(alpha))
    B = np.asarray(H_tilde, dtype=np.float64).T
    mode = str(block_encoding_mode)
    if mode == "explicit_dense" and _can_run_explicit_dense(B):
        result = _run_explicit_dense_update(
            B=B,
            residual_preparation=residual_state,
            ridge=ridge,
            alpha=float(alpha),
            degree=int(degree),
            phase_method=str(phase_method),
        )
    else:
        result = _run_resource_proxy_update(
            B=B,
            residual_state=residual_state,
            ridge=ridge,
            alpha=float(alpha),
            degree=int(degree),
            mode=mode,
            phase_method=str(phase_method),
        )
    return result


def build_qsvt_update_workflow_artifacts(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_workflow_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    source = _build_source_matrix(
        {
            "case": resolved["case"],
            "case_name": resolved["case_name"],
            "case_source": resolved["case_source"],
            "matrix_source": resolved["matrix_source"],
            "submatrix_size": resolved["submatrix_size"],
            "alpha": resolved["alpha"],
            "degree": resolved["degree"],
            "seed": resolved["seed"],
            "tolerance": 1.0e-8,
            "validation_tolerance": 1.0e-7,
        }
    )
    H_sub = np.asarray(source.B, dtype=np.float64).T
    r_sub = np.asarray(source.residual, dtype=np.float64)
    result = run_qsvt_update_state_simulation(
        H_tilde=H_sub,
        r_tilde=r_sub,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        block_encoding_mode=str(resolved["block_encoding_mode"]),
        phase_method=str(resolved["phase_method"]),
        seed=int(resolved["seed"]),
    )
    diagnostics = result.diagnostics()
    diagnostics.update(
        {
            "case_name": resolved["case_name"],
            "matrix_source": source.metadata["matrix_source"],
            "selected_state_labels": source.row_labels,
            "selected_measurement_labels": source.column_labels,
            "selected_state_indices": source.selected_rows.astype(int).tolist(),
            "selected_measurement_indices": source.selected_columns.astype(int).tolist(),
            "claim_boundary": UPDATE_WORKFLOW_CLAIM,
        }
    )
    norm_success = compute_norm_success_diagnostics(
        ridge_update=result.ridge_reference.update_vector,
        qsvt_vector=result.qsvt_unnormalized_vector,
        bounded_scaling_C=result.bounded_target_scaling_C,
        beta=result.beta,
        residual_norm=result.residual_state_preparation.residual_norm,
        success_probability_proxy=result.success_probability_proxy,
        norm_recovery_method="classical_simulator_metadata",
    )
    diagnostics["norm_success_diagnostics"] = norm_success

    diagnostics_path = output_dir / "update_state_diagnostics.json"
    ridge_path = output_dir / "ridge_reference_update.npy"
    qsvt_path = output_dir / "qsvt_update_state.npy"
    summary_path = output_dir / "qsvt_update_summary.md"
    write_json(diagnostics_path, diagnostics)
    np.save(ridge_path, result.ridge_reference.update_vector)
    np.save(qsvt_path, result.qsvt_normalized_update_state)
    summary_path.write_text(_update_summary_markdown(diagnostics), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "update_state_diagnostics": str(diagnostics_path),
            "ridge_reference_update": str(ridge_path),
            "qsvt_update_state": str(qsvt_path),
            "qsvt_update_summary": str(summary_path),
        },
        input_config=resolved,
        claim_boundary=UPDATE_WORKFLOW_CLAIM,
    )
    return {
        "output_dir": output_dir,
        "result": result,
        "diagnostics": diagnostics,
        "artifacts": {
            "manifest": manifest_path,
            "update_state_diagnostics": diagnostics_path,
            "ridge_reference_update": ridge_path,
            "qsvt_update_state": qsvt_path,
            "qsvt_update_summary": summary_path,
        },
    }


def best_global_phase_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    overlap = np.vdot(ref, cand)
    if abs(overlap) <= 1.0e-15:
        return float(np.linalg.norm(cand - ref))
    aligned = cand * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(aligned - ref))


def best_real_sign_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128)
    cand = np.asarray(candidate, dtype=np.complex128)
    return float(min(np.linalg.norm(cand - ref), np.linalg.norm(cand + ref)))


def _run_explicit_dense_update(
    *,
    B: np.ndarray,
    residual_preparation: ResidualStatePreparationResult,
    ridge: RidgeReferenceResult,
    alpha: float,
    degree: int,
    phase_method: str,
) -> QSVTUpdateResult:
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = B / beta
    alpha_norm = alpha / beta**2
    block_encoding = canonical_square_block_encoding(A, tolerance=1.0e-8)
    block_report = validate_block_encoding(block_encoding, beta=beta, tolerance=1.0e-7)
    phase_sequence = _build_phase_sequence(
        singular_values_A=np.linalg.svd(A, compute_uv=False),
        requested_degree=int(degree),
        alpha_norm=float(alpha_norm),
        grid_size=2048,
        max_synthesis_degree=int(degree),
        angle_solver="iterative",
        phase_timeout_seconds=25,
    )
    qsvt = _apply_explicit_block_encoded_qsvt(
        A=A,
        block_unitary=block_encoding.unitary,
        phases=phase_sequence.phases,
    )
    bounded_block = np.asarray(qsvt["transformed_block"], dtype=np.float64)
    qsvt_operator = rescale_bounded_target_to_original(
        bounded_block,
        beta=beta,
        C=phase_sequence.approximation.scale_factor,
    )
    residual_state = np.real(residual_preparation.normalized_residual_state)
    qsvt_vector = residual_preparation.residual_norm * (qsvt_operator @ residual_state)
    bounded_state = bounded_block @ residual_state
    success_probability = float(min(1.0, np.linalg.norm(bounded_state) ** 2))
    return _make_update_result(
        qsvt_vector=qsvt_vector,
        ridge=ridge,
        residual_state=residual_preparation,
        success_probability=success_probability,
        C=float(phase_sequence.approximation.scale_factor),
        beta=beta,
        requested_degree=int(degree),
        synthesized_degree=int(phase_sequence.actual_degree),
        phase_count=int(phase_sequence.phases.size),
        phase_method=phase_method,
        simulation_mode="explicit_dense_block_encoded_qsvt",
        block_report=block_report,
        limitations=(
            "dense block encoding is a small correctness simulation only",
            "state-preparation and norm recovery use simulator metadata",
        ),
    )


def _run_resource_proxy_update(
    *,
    B: np.ndarray,
    residual_state: ResidualStatePreparationResult,
    ridge: RidgeReferenceResult,
    alpha: float,
    degree: int,
    mode: str,
    phase_method: str,
) -> QSVTUpdateResult:
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = B / beta
    alpha_norm = alpha / beta**2
    positive = np.linalg.svd(A, compute_uv=False)
    domain_min = max(1.0e-6, 0.9 * float(np.min(positive[positive > 1.0e-14])))
    from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial

    approximation = fit_odd_regularized_polynomial(
        alpha=alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=min(domain_min, 0.95),
        domain_max=1.0,
        grid_size=max(512, int(degree) + 2),
    )
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients) / approximation.scale_factor
    )
    polynomial_operator = _singular_value_transform(
        A, polynomial(np.linalg.svd(A, compute_uv=False))
    )
    qsvt_operator = rescale_bounded_target_to_original(
        polynomial_operator,
        beta=beta,
        C=approximation.scale_factor,
    )
    qsvt_vector = residual_state.residual_norm * (
        qsvt_operator @ np.real(residual_state.normalized_residual_state)
    )
    success_probability = float(
        min(
            1.0,
            np.linalg.norm(polynomial_operator @ np.real(residual_state.normalized_residual_state))
            ** 2,
        )
    )
    return _make_update_result(
        qsvt_vector=qsvt_vector,
        ridge=ridge,
        residual_state=residual_state,
        success_probability=success_probability,
        C=float(approximation.scale_factor),
        beta=beta,
        requested_degree=int(degree),
        synthesized_degree=int(degree),
        phase_count=int(degree) + 1,
        phase_method=phase_method,
        simulation_mode=f"{mode}_polynomial_svd_resource_proxy",
        block_report={
            "matrix_shape": [int(A.shape[0]), int(A.shape[1])],
            "beta": beta,
            "unitarity_error": None,
            "top_left_block_error": None,
            "passed": False,
            "resource_proxy_only": True,
        },
        limitations=(
            "this row uses a polynomial SVD target proxy, not an executed dense QSVT unitary",
            "use explicit_dense on a square power-of-two subproblem for implemented QSVT",
        ),
    )


def _make_update_result(
    *,
    qsvt_vector: np.ndarray,
    ridge: RidgeReferenceResult,
    residual_state: ResidualStatePreparationResult | None,
    success_probability: float,
    C: float,
    beta: float,
    requested_degree: int,
    synthesized_degree: int,
    phase_count: int,
    phase_method: str,
    simulation_mode: str,
    block_report: dict[str, Any],
    limitations: tuple[str, ...],
) -> QSVTUpdateResult:
    qsvt_state, _ = normalize_state(qsvt_vector)
    overlap_abs = float(abs(np.vdot(ridge.normalized_update_state, qsvt_state)))
    phase_error = best_global_phase_l2_error(ridge.normalized_update_state, qsvt_state)
    sign_error = best_real_sign_l2_error(ridge.normalized_update_state, qsvt_state)
    residual_preparation = residual_state or ResidualStatePreparationResult(
        normalized_residual_state=np.array([], dtype=np.complex128),
        padded_residual_state=np.array([], dtype=np.complex128),
        residual_norm=float(ridge.residual_norm),
        dimension=0,
        padded_dimension=0,
        padding_width=0,
        warnings=(),
    )
    return QSVTUpdateResult(
        qsvt_normalized_update_state=qsvt_state,
        qsvt_unnormalized_vector=np.real(qsvt_vector),
        observable_ready_state=qsvt_state,
        ridge_reference=ridge,
        residual_state_preparation=residual_preparation,
        success_probability_proxy=float(success_probability),
        bounded_target_scaling_C=float(C),
        beta=float(beta),
        phase_count=int(phase_count),
        synthesized_degree=int(synthesized_degree),
        requested_degree=int(requested_degree),
        query_count_estimate=int(2 * synthesized_degree + 1),
        state_l2_error_against_ridge=float(
            np.linalg.norm(qsvt_state - ridge.normalized_update_state)
        ),
        phase_aligned_state_l2_error=phase_error,
        real_sign_aligned_state_l2_error=sign_error,
        normalized_state_overlap_abs=overlap_abs,
        block_encoding_report=block_report,
        simulation_mode=simulation_mode,
        phase_method=phase_method,
        limitations=limitations,
    )


def _singular_value_transform(
    matrix: np.ndarray, transformed_singular_values: np.ndarray
) -> np.ndarray:
    U, _, Vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    return U @ (np.asarray(transformed_singular_values, dtype=np.float64)[:, None] * Vh)


def _can_run_explicit_dense(B: np.ndarray) -> bool:
    return bool(B.ndim == 2 and B.shape[0] == B.shape[1] and _is_power_of_two(B.shape[0]))


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    return 1 << (int(value) - 1).bit_length()


def _resolve_workflow_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "case": "ieee14",
        "case_name": "ieee14",
        "case_source": "pypower",
        "matrix_source": "weighted_jacobian",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "seed": 123,
        "block_encoding_mode": "explicit_dense",
        "phase_method": "pennylane_poly_to_angles",
        "output_dir": "outputs/full_qsvt_ieee_update_workflow",
    }
    if config:
        resolved.update(config)
    if "case" in resolved and "case_name" not in (config or {}):
        resolved["case_name"] = resolved["case"]
    if int(resolved["submatrix_size"]) <= 0:
        raise ValueError("submatrix_size must be positive")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if int(resolved["degree"]) <= 0 or int(resolved["degree"]) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    return resolved


def _update_summary_markdown(diagnostics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# QSVT Update-State Workflow",
            "",
            UPDATE_WORKFLOW_CLAIM,
            "",
            "## Setup",
            f"- Case: {diagnostics['case_name']}",
            f"- Matrix source: {diagnostics['matrix_source']}",
            f"- Selected states: {', '.join(diagnostics['selected_state_labels'])}",
            f"- Selected measurements: {', '.join(diagnostics['selected_measurement_labels'])}",
            f"- Beta: {diagnostics['beta']:.17g}",
            f"- Bounded scaling C: {diagnostics['bounded_target_scaling_C']:.17g}",
            f"- QSVT degree: {diagnostics['synthesized_degree']} "
            f"(requested {diagnostics['requested_degree']})",
            f"- Phases: {diagnostics['phase_count']}",
            "",
            "## State Diagnostics",
            f"- Ridge update norm: {diagnostics['ridge_update_norm']:.17g}",
            f"- QSVT state norm before normalization: "
            f"{diagnostics['qsvt_state_norm_before_normalization']:.17g}",
            f"- State overlap magnitude: {diagnostics['normalized_state_overlap_abs']:.17g}",
            f"- Best phase-aligned L2 error: {diagnostics['phase_aligned_state_l2_error']:.17g}",
            f"- Best real sign-aligned L2 error: "
            f"{diagnostics['real_sign_aligned_state_l2_error']:.17g}",
            f"- Success probability proxy: {diagnostics['success_probability_proxy']:.17g}",
            "",
            "## Limitations",
            "- Dense block encoding is used only for small selected subproblems.",
            "- Norm recovery is simulator metadata unless a separate amplitude-estimation "
            "routine is implemented.",
            "",
        ]
    )


def build_subproblem_from_engineering_system(
    *,
    case: str,
    matrix_source: str,
    submatrix_size: int,
    seed: int,
    case_source: str = "pypower",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = _build_source_matrix(
        {
            "case": case,
            "case_name": case,
            "case_source": case_source,
            "matrix_source": matrix_source,
            "submatrix_size": int(submatrix_size),
            "alpha": 1.0e-4,
            "degree": 51,
            "seed": int(seed),
            "tolerance": 1.0e-8,
            "validation_tolerance": 1.0e-7,
        }
    )
    return (
        np.asarray(source.B, dtype=np.float64).T,
        np.asarray(source.residual, dtype=np.float64),
        {
            "selected_state_indices": source.selected_rows.astype(int).tolist(),
            "selected_measurement_indices": source.selected_columns.astype(int).tolist(),
            "selected_state_labels": source.row_labels,
            "selected_measurement_labels": source.column_labels,
            "matrix_source": source.metadata["matrix_source"],
        },
    )


def full_ridge_reference_for_case(
    *,
    case: str,
    matrix_source: str,
    alpha: float,
    seed: int,
    case_source: str = "pypower",
) -> tuple[np.ndarray, dict[str, Any]]:
    system, source = build_engineering_system(
        {
            "case_name": case,
            "case_source": case_source,
            "matrix_source": matrix_source,
            "seed": int(seed),
        }
    )
    update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=float(alpha))
    return update, {"system": system, "matrix_source": source}
