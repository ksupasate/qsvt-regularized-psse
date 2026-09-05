from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.degree_control_audit import synthesize_phases_with_cache
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.phase_synthesis import validate_qsvt_polynomial
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory

POLYNOMIAL_ACTION_CLAIM = (
    "This decomposition separates Ridge/Tikhonov reference action, exact bounded "
    "QSVT target action, degree-controlled polynomial action, and any available "
    "gate-level extraction error. It is not a claim of QSVT superiority over Ridge."
)

DECOMPOSITION_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "alpha",
    "requested_degree",
    "constructed_polynomial_degree",
    "synthesized_phase_degree",
    "effective_qsvt_degree",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "pointwise_max_error_on_grid",
    "pointwise_max_error_on_singular_values",
    "ridge_residual",
    "exact_target_residual",
    "bounded_target_residual",
    "polynomial_action_residual",
    "best_scalar_polynomial_residual",
    "gate_level_residual_if_available",
    "state_error_target_vs_ridge",
    "state_error_poly_vs_target",
    "state_error_gate_vs_poly_if_available",
    "dominant_error_source",
    "run_status",
    "failure_reason_if_any",
]


def run_polynomial_action_decomposition(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75, 101, 151, 201],
        "seed": 123,
        "grid_size": 4096,
        "angle_solver": "iterative",
        "phase_timeout_seconds": 20,
        "phase_cache_dir": "outputs/qsvt_degree_control_audit/phase_cache",
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "selection_mode": "high_leverage",
        "output_dir": "outputs/qsvt_polynomial_action_decomposition",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    subproblem = extract_state_estimation_subproblem(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        seed=int(resolved["seed"]),
        case_source=str(resolved["case_source"]),
    )
    rows = evaluate_polynomial_action_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        selection_mode=str(resolved["selection_mode"]),
        grid_size=int(resolved["grid_size"]),
        angle_solver=str(resolved["angle_solver"]),
        phase_timeout_seconds=int(resolved["phase_timeout_seconds"]),
        cache_dir=resolved.get("phase_cache_dir") or (output_dir / "phase_cache"),
    )
    artifacts = write_polynomial_action_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_polynomial_action_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    grid_size: int = 4096,
    angle_solver: str = "iterative",
    phase_timeout_seconds: int = 20,
    cache_dir: str | Path = "outputs/qsvt_polynomial_action_decomposition/phase_cache",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for degree in degrees:
            rows.append(
                evaluate_single_polynomial_action(
                    subproblem=subproblem,
                    alpha=float(alpha),
                    requested_degree=int(degree),
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    selection_mode=selection_mode,
                    grid_size=int(grid_size),
                    angle_solver=str(angle_solver),
                    phase_timeout_seconds=int(phase_timeout_seconds),
                    cache_dir=cache_dir,
                )
            )
    return rows


def evaluate_single_polynomial_action(
    *,
    subproblem: SelectedSubproblem,
    alpha: float,
    requested_degree: int,
    case: str = "custom",
    model: str = "weighted_linearized",
    subproblem_id: str = "subproblem",
    selection_mode: str = "unit_test",
    grid_size: int = 4096,
    angle_solver: str = "iterative",
    phase_timeout_seconds: int = 20,
    cache_dir: str | Path = "outputs/qsvt_polynomial_action_decomposition/phase_cache",
) -> dict[str, Any]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    row = _base_decomposition_row(
        case=case,
        model=model,
        subproblem_id=subproblem_id,
        selection_mode=selection_mode,
        alpha=alpha,
        requested_degree=requested_degree,
        H=H,
    )
    try:
        if H.ndim != 2 or r.ndim != 1 or H.shape[0] != r.size:
            raise ValueError("H_tilde must be m x n and r_tilde must have length m")
        ridge = ridge_tikhonov_update(H, r, alpha=float(alpha))
        exact_target = _exact_spectral_update(H, r, alpha=float(alpha))
        B = H.T
        beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
        A = B / beta
        singular_values_A = np.linalg.svd(A, compute_uv=False)
        positive = singular_values_A[singular_values_A > 1.0e-14]
        if positive.size == 0:
            raise ValueError("selected subproblem has no positive singular values")
        domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
        alpha_norm = float(alpha) / beta**2
        approximation = fit_odd_regularized_polynomial(
            alpha=alpha_norm,
            block_encoding_normalization=1.0,
            degree=int(requested_degree),
            domain_min=domain_min,
            domain_max=1.0,
            grid_size=max(int(grid_size), int(requested_degree) + 2),
        )
        constructed_degree = int(approximation.degree)
        coefficients = np.asarray(approximation.power_coefficients, dtype=np.float64)
        scaled_coefficients = coefficients / float(approximation.scale_factor)
        validation = validate_qsvt_polynomial(
            scaled_coefficients,
            parity="odd",
            grid_size=max(8193, 128 * int(requested_degree) + 1),
            bound_tolerance=1.0e-5,
        )
        polynomial = Polynomial(scaled_coefficients)
        grid = np.linspace(domain_min, 1.0, max(int(grid_size), 512), dtype=np.float64)
        bounded_target_grid = grid / (float(approximation.scale_factor) * (grid**2 + alpha_norm))
        polynomial_grid = polynomial(grid)
        pointwise_grid_error = float(np.max(np.abs(polynomial_grid - bounded_target_grid)))
        bounded_target_singular = singular_values_A / (
            float(approximation.scale_factor) * (singular_values_A**2 + alpha_norm)
        )
        polynomial_singular = polynomial(singular_values_A)
        pointwise_singular_error = float(
            np.max(np.abs(polynomial_singular - bounded_target_singular))
        )
        U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
        polynomial_operator = U @ (polynomial(singular_values)[:, None] * Vh)
        polynomial_update = rescale_bounded_target_to_original(
            polynomial_operator @ r,
            beta=beta,
            C=float(approximation.scale_factor),
        )
        bounded_target_update = exact_target / float(approximation.scale_factor)
        best_scalar = best_scalar_for_residual(H, polynomial_update, r)
        phase_metadata = _try_phase_synthesis(
            scaled_coefficients=scaled_coefficients,
            alpha=alpha,
            alpha_norm=alpha_norm,
            requested_degree=requested_degree,
            constructed_degree=constructed_degree,
            domain_min=domain_min,
            scale_factor=float(approximation.scale_factor),
            angle_solver=angle_solver,
            phase_timeout_seconds=phase_timeout_seconds,
            cache_dir=cache_dir,
        )
        phase_failure = str(phase_metadata.get("failure_reason_if_any", ""))
        run_status = "completed" if not phase_failure else "phase_synthesis_failed"
        row.update(
            {
                "constructed_polynomial_degree": constructed_degree,
                "synthesized_phase_degree": phase_metadata.get("synthesized_phase_degree", np.nan),
                "effective_qsvt_degree": phase_metadata.get("effective_qsvt_degree", np.nan),
                "phase_count": phase_metadata.get("phase_count", np.nan),
                "cache_key": phase_metadata.get("cache_key", ""),
                "cache_hit": bool(phase_metadata.get("cache_hit", False)),
                "polynomial_scale_factor": float(approximation.scale_factor),
                "max_abs_on_unit_interval": float(validation["max_abs_on_unit_interval"]),
                "pointwise_max_error_on_grid": pointwise_grid_error,
                "pointwise_max_error_on_singular_values": pointwise_singular_error,
                "ridge_residual": _residual(H, ridge, r),
                "exact_target_residual": _residual(H, exact_target, r),
                "bounded_target_residual": _residual(H, bounded_target_update, r),
                "polynomial_action_residual": _residual(H, polynomial_update, r),
                "best_scalar_polynomial_residual": _residual(H, best_scalar * polynomial_update, r),
                "state_error_target_vs_ridge": _relative_error(exact_target, ridge),
                "state_error_poly_vs_target": _relative_error(polynomial_update, exact_target),
                "dominant_error_source": _dominant_error_source(
                    ridge_residual=_residual(H, ridge, r),
                    exact_target_residual=_residual(H, exact_target, r),
                    bounded_target_residual=_residual(H, bounded_target_update, r),
                    polynomial_residual=_residual(H, polynomial_update, r),
                    best_scalar_residual=_residual(H, best_scalar * polynomial_update, r),
                    pointwise_singular_error=pointwise_singular_error,
                    phase_failure=phase_failure,
                    condition_number=float(row["condition_number"]),
                ),
                "success_probability_proxy": float(
                    min(1.0, np.linalg.norm(polynomial_operator @ r) ** 2)
                ),
                "run_status": run_status,
                "failure_reason_if_any": phase_failure,
            }
        )
    except Exception as exc:
        row.update(
            {
                "run_status": "failed",
                "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def write_polynomial_action_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = pd.DataFrame(rows)
    for column in DECOMPOSITION_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    pointwise_path = output_dir / "polynomial_pointwise_error.csv"
    residual_path = output_dir / "polynomial_action_residuals.csv"
    comparison_path = output_dir / "ridge_target_poly_comparison.csv"
    gate_path = output_dir / "gate_vs_polynomial_error.csv"
    interpretation_path = output_dir / "polynomial_action_interpretation.md"
    frame[
        [
            "case",
            "model",
            "subproblem_id",
            "alpha",
            "requested_degree",
            "constructed_polynomial_degree",
            "synthesized_phase_degree",
            "pointwise_max_error_on_grid",
            "pointwise_max_error_on_singular_values",
            "run_status",
            "failure_reason_if_any",
        ]
    ].to_csv(pointwise_path, index=False)
    frame.to_csv(residual_path, index=False)
    frame[
        [
            "case",
            "model",
            "subproblem_id",
            "selection_mode",
            "alpha",
            "requested_degree",
            "ridge_residual",
            "exact_target_residual",
            "bounded_target_residual",
            "polynomial_action_residual",
            "best_scalar_polynomial_residual",
            "state_error_target_vs_ridge",
            "state_error_poly_vs_target",
            "dominant_error_source",
            "run_status",
        ]
    ].to_csv(comparison_path, index=False)
    frame[
        [
            "case",
            "model",
            "subproblem_id",
            "alpha",
            "requested_degree",
            "polynomial_action_residual",
            "gate_level_residual_if_available",
            "state_error_gate_vs_poly_if_available",
            "run_status",
            "failure_reason_if_any",
        ]
    ].to_csv(gate_path, index=False)
    interpretation_path.write_text(polynomial_action_interpretation(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "polynomial_pointwise_error": str(pointwise_path),
            "polynomial_action_residuals": str(residual_path),
            "ridge_target_poly_comparison": str(comparison_path),
            "gate_vs_polynomial_error": str(gate_path),
            "polynomial_action_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=POLYNOMIAL_ACTION_CLAIM,
    )
    return {
        "manifest": manifest,
        "polynomial_pointwise_error": pointwise_path,
        "polynomial_action_residuals": residual_path,
        "ridge_target_poly_comparison": comparison_path,
        "gate_vs_polynomial_error": gate_path,
        "polynomial_action_interpretation": interpretation_path,
    }


def polynomial_action_interpretation(frame: pd.DataFrame) -> str:
    completed = frame[frame["run_status"] == "completed"].copy()
    if completed.empty:
        best_lines = ["- No completed true-degree polynomial-action rows."]
        higher_improved = "inconclusive"
    else:
        best = completed.loc[completed["best_scalar_polynomial_residual"].astype(float).idxmin()]
        best_lines = [
            f"- Best alpha: {float(best['alpha']):.17g}",
            f"- Best requested degree: {int(best['requested_degree'])}",
            f"- Best polynomial-action residual: {float(best['polynomial_action_residual']):.17g}",
            "- Best scalar polynomial residual: "
            f"{float(best['best_scalar_polynomial_residual']):.17g}",
            f"- Ridge residual at best row: {float(best['ridge_residual']):.17g}",
            f"- Exact target residual at best row: {float(best['exact_target_residual']):.17g}",
            f"- Bounded target residual at best row: {float(best['bounded_target_residual']):.17g}",
            f"- Dominant error source: {best['dominant_error_source']}",
        ]
        higher_improved = _higher_degree_improved(completed)
    return "\n".join(
        [
            "# Polynomial-Action Decomposition",
            "",
            POLYNOMIAL_ACTION_CLAIM,
            "",
            "## Required Answers",
            "1. The exact unbounded spectral target matches Ridge up to numerical SVD error.",
            "2. Bounded scaling changes update magnitude before the known QSVT rescaling; "
            "after rescaling it should preserve the target action.",
            "3. Degree improvement is evaluated on actual singular values and the fitting grid.",
            "4. Polynomial action approaches the Ridge residual only when the degree-controlled "
            "polynomial and known scaling are accurate on the selected spectrum.",
            "5. If polynomial action is good but gate-level output is not, the bottleneck is "
            "gate-level extraction/state preparation/norm-sign recovery rather than target math.",
            "6. If polynomial action is poor, the remaining issue is target scaling, degree, "
            "or conditioning sensitivity before gate execution.",
            "",
            "## Best Row",
            *best_lines,
            f"- Higher degree improved residual: {higher_improved}",
            "",
        ]
    )


def _try_phase_synthesis(
    *,
    scaled_coefficients: np.ndarray,
    alpha: float,
    alpha_norm: float,
    requested_degree: int,
    constructed_degree: int,
    domain_min: float,
    scale_factor: float,
    angle_solver: str,
    phase_timeout_seconds: int,
    cache_dir: str | Path,
) -> dict[str, Any]:
    try:
        result = synthesize_phases_with_cache(
            scaled_coefficients,
            angle_solver=angle_solver,
            cache_dir=cache_dir,
            cache_metadata={
                "alpha": float(alpha),
                "alpha_norm": float(alpha_norm),
                "requested_degree": int(requested_degree),
                "constructed_polynomial_degree": int(constructed_degree),
                "domain_min": float(domain_min),
                "bound_tolerance": 1.0e-5,
                "scale_factor": float(scale_factor),
            },
            timeout_seconds=int(phase_timeout_seconds),
        )
        phase_count = int(result["phase_count"])
        degree = max(phase_count - 1, 0)
        return {
            "synthesized_phase_degree": degree,
            "effective_qsvt_degree": degree,
            "phase_count": phase_count,
            "cache_key": result["cache_key"],
            "cache_hit": bool(result["cache_hit"]),
            "failure_reason_if_any": "",
        }
    except Exception as exc:
        return {
            "synthesized_phase_degree": np.nan,
            "effective_qsvt_degree": np.nan,
            "phase_count": np.nan,
            "cache_key": "",
            "cache_hit": False,
            "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
        }


def _exact_spectral_update(H: np.ndarray, r: np.ndarray, *, alpha: float) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(np.asarray(H, dtype=np.float64), full_matrices=False)
    return Vt.T @ ((singular_values / (singular_values**2 + float(alpha))) * (U.T @ r))


def _base_decomposition_row(
    *,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    alpha: float,
    requested_degree: int,
    H: np.ndarray,
) -> dict[str, Any]:
    singular_values = np.linalg.svd(np.asarray(H, dtype=np.float64), compute_uv=False)
    sigma_max = float(np.max(singular_values)) if singular_values.size else np.nan
    sigma_min = float(np.min(singular_values)) if singular_values.size else np.nan
    condition = sigma_max / sigma_min if sigma_min and sigma_min > 1.0e-14 else float("inf")
    row = {column: np.nan for column in DECOMPOSITION_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "selection_mode": selection_mode,
            "alpha": float(alpha),
            "requested_degree": int(requested_degree),
            "condition_number": float(condition),
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "gate_level_residual_if_available": np.nan,
            "state_error_gate_vs_poly_if_available": np.nan,
            "run_status": "not_run",
            "failure_reason_if_any": "",
        }
    )
    return row


def _dominant_error_source(
    *,
    ridge_residual: float,
    exact_target_residual: float,
    bounded_target_residual: float,
    polynomial_residual: float,
    best_scalar_residual: float,
    pointwise_singular_error: float,
    phase_failure: str,
    condition_number: float,
) -> str:
    if phase_failure:
        return "phase_synthesis"
    if abs(exact_target_residual - ridge_residual) > max(1.0e-8, 10.0 * ridge_residual):
        return "target_mismatch"
    if bounded_target_residual > max(polynomial_residual, ridge_residual) * 2.0:
        return "bounded_scaling"
    if pointwise_singular_error > 1.0e-3 or polynomial_residual > best_scalar_residual * 2.0:
        return "polynomial_degree"
    if condition_number > 1.0e8:
        return "conditioning_sensitivity"
    return "polynomial_action_matches_target"


def _higher_degree_improved(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "inconclusive"
    by_degree = (
        frame.groupby("requested_degree", sort=True)["best_scalar_polynomial_residual"]
        .min()
        .reset_index()
    )
    if len(by_degree) < 2:
        return "inconclusive"
    first = float(by_degree.iloc[0]["best_scalar_polynomial_residual"])
    last = float(by_degree.iloc[-1]["best_scalar_polynomial_residual"])
    return "yes" if last < 0.8 * first else "no"


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_values - reference_values)
        / max(float(np.linalg.norm(reference_values)), 1.0e-15)
    )
