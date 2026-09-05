from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import (
    _build_phase_sequence,
    rescale_bounded_target_to_original,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    GATE_LEVEL_SOLVER_CLAIM,
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

NORM_GAP_CLAIM = (
    "This is a norm and residual-gap audit for a selected IEEE-derived "
    "gate-level QSVT update. It decomposes simulator evidence and does not "
    "claim QSVT numerical superiority over Ridge/Tikhonov."
)


def best_scalar_for_residual(
    H_tilde: np.ndarray,
    qsvt_update: np.ndarray,
    r_tilde: np.ndarray,
) -> float:
    """Return the scalar minimizing ``||H (s qsvt_update) - r||_2``."""

    H = np.asarray(H_tilde, dtype=np.float64)
    qsvt = np.asarray(qsvt_update, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    action = H @ qsvt
    denominator = float(np.dot(action, action))
    if denominator <= 1.0e-30:
        return 0.0
    return float(np.dot(action, r) / denominator)


def polynomial_action_reference(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    alpha: float,
    degree: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the SVD polynomial-action reference used by the gate-level solver."""

    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    B = H.T
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    beta = max(float(singular_values_B[0]), np.finfo(float).eps)
    A = B / beta
    alpha_norm = float(alpha) / beta**2
    phase_sequence = _build_phase_sequence(
        singular_values_A=np.linalg.svd(A, compute_uv=False),
        requested_degree=int(degree),
        alpha_norm=alpha_norm,
        grid_size=max(2048, int(degree) + 2),
        max_synthesis_degree=int(degree),
        angle_solver="iterative",
        phase_timeout_seconds=25,
    )
    polynomial = Polynomial(phase_sequence.coefficients)
    U, singular_values_A, Vh = np.linalg.svd(A, full_matrices=False)
    polynomial_operator = U @ (polynomial(singular_values_A)[:, None] * Vh)
    residual_norm = float(np.linalg.norm(r))
    residual_state = r / max(residual_norm, 1.0e-15)
    bounded_vector = polynomial_operator @ residual_state
    update = residual_norm * rescale_bounded_target_to_original(
        bounded_vector,
        beta=beta,
        C=phase_sequence.approximation.scale_factor,
    )
    return np.real(update), {
        "beta": beta,
        "alpha_norm": alpha_norm,
        "bounded_scaling_C": float(phase_sequence.approximation.scale_factor),
        "synthesized_degree": int(phase_sequence.actual_degree),
        "phase_count": int(phase_sequence.phases.size),
        "polynomial_coefficients": phase_sequence.coefficients.tolist(),
    }


def run_norm_residual_gap_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alpha": 1.0e-4,
        "degree": 51,
        "seed": 123,
        "shots": 1000,
        "output_dir": "outputs/qsvt_norm_residual_gap_audit",
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
    computation = solve_gate_level_state_estimation_problem(
        H_tilde=subproblem.H_tilde,
        r_tilde=subproblem.r_tilde,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
        shots=int(resolved["shots"]),
        seed=int(resolved["seed"]),
        metadata=subproblem.metadata,
        transpile_qubit_limit=0,
        export_qasm=False,
    )
    H = computation.H_tilde
    r = computation.r_tilde
    qsvt = computation.qsvt_update
    ridge = computation.ridge_update
    poly, poly_metadata = polynomial_action_reference(
        H,
        r,
        alpha=float(resolved["alpha"]),
        degree=int(resolved["degree"]),
    )
    qsvt_norm = float(np.linalg.norm(qsvt))
    ridge_norm = float(np.linalg.norm(ridge))
    qsvt_direction = qsvt / max(qsvt_norm, 1.0e-15)
    ridge_norm_rescaled = ridge_norm * qsvt_direction
    best_scalar = best_scalar_for_residual(H, qsvt, r)
    best_scalar_update = best_scalar * qsvt

    update_variants = {
        "no_update": np.zeros_like(ridge),
        "qsvt_raw": qsvt,
        "qsvt_direction_only_unit_norm": qsvt_direction,
        "qsvt_ridge_norm_rescaled": ridge_norm_rescaled,
        "qsvt_best_scalar_rescaled": best_scalar_update,
        "polynomial_reference": poly,
        "ridge_reference": ridge,
    }
    residual_rows = [
        _variant_row(name, update, H=H, r=r, ridge=ridge)
        for name, update in update_variants.items()
    ]
    summary = _summary_row(
        resolved=resolved,
        H=H,
        r=r,
        qsvt=qsvt,
        ridge=ridge,
        poly=poly,
        best_scalar=best_scalar,
        computation=computation,
        residual_rows=residual_rows,
    )
    sensitivity_rows = _conditioning_sensitivity_rows(H, r, qsvt, seed=int(resolved["seed"]))
    scale_rows = _scale_sweep_rows(H, r, qsvt, ridge_norm=ridge_norm, best_scalar=best_scalar)
    poly_rows = [
        {
            "metric": "polynomial_reference_relative_update_error",
            "value": _relative_error(poly, ridge),
        },
        {
            "metric": "polynomial_reference_residual",
            "value": _residual_norm(H, poly, r),
        },
        {
            "metric": "gate_qsvt_vs_polynomial_relative_update_error",
            "value": _relative_error(qsvt, poly),
        },
        {
            "metric": "gate_qsvt_vs_polynomial_residual_difference",
            "value": abs(_residual_norm(H, qsvt, r) - _residual_norm(H, poly, r)),
        },
    ]
    artifacts = _write_outputs(
        output_dir=output_dir,
        resolved=resolved,
        summary=summary,
        residual_rows=residual_rows,
        scale_rows=scale_rows,
        poly_rows=poly_rows,
        sensitivity_rows=sensitivity_rows,
        poly_metadata=poly_metadata,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "residual_rows": residual_rows,
        "scale_rows": scale_rows,
        "polynomial_rows": poly_rows,
        "sensitivity_rows": sensitivity_rows,
        "artifacts": artifacts,
    }


def _summary_row(
    *,
    resolved: dict[str, Any],
    H: np.ndarray,
    r: np.ndarray,
    qsvt: np.ndarray,
    ridge: np.ndarray,
    poly: np.ndarray,
    best_scalar: float,
    computation: Any,
    residual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    singular_values = np.linalg.svd(H, compute_uv=False)
    condition = _condition_number(singular_values)
    qsvt_norm = float(np.linalg.norm(qsvt))
    ridge_norm = float(np.linalg.norm(ridge))
    ridge_norm_rescaled = ridge_norm * qsvt / max(qsvt_norm, 1.0e-15)
    best_scalar_update = best_scalar * qsvt
    residual_by_name = {row["update_variant"]: row["residual_norm"] for row in residual_rows}
    sensitivity = _sensitivity_score(H, r, qsvt, seed=int(resolved["seed"]))
    dominant = _dominant_gap_source(
        residual_no_update=residual_by_name["no_update"],
        residual_raw=residual_by_name["qsvt_raw"],
        residual_best_scalar=residual_by_name["qsvt_best_scalar_rescaled"],
        residual_ridge=residual_by_name["ridge_reference"],
        residual_poly=residual_by_name["polynomial_reference"],
        sensitivity=sensitivity,
    )
    return {
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "alpha": float(resolved["alpha"]),
        "degree": int(resolved["degree"]),
        "synthesized_degree": int(computation.summary["synthesized_degree"]),
        "condition_number": condition,
        "sigma_max": float(np.max(singular_values)),
        "sigma_min": float(np.min(singular_values)),
        "ridge_update_norm": ridge_norm,
        "qsvt_update_norm": qsvt_norm,
        "norm_ratio_qsvt_to_ridge": qsvt_norm / max(ridge_norm, 1.0e-15),
        "state_direction_error": computation.summary["phase_or_sign_aligned_state_error"],
        "relative_update_error_raw": _relative_error(qsvt, ridge),
        "relative_update_error_ridge_norm": _relative_error(ridge_norm_rescaled, ridge),
        "relative_update_error_best_scalar": _relative_error(best_scalar_update, ridge),
        "relative_update_error_poly_reference": _relative_error(poly, ridge),
        "residual_no_update": residual_by_name["no_update"],
        "residual_qsvt_raw": residual_by_name["qsvt_raw"],
        "residual_qsvt_ridge_norm": residual_by_name["qsvt_ridge_norm_rescaled"],
        "residual_qsvt_best_scalar": residual_by_name["qsvt_best_scalar_rescaled"],
        "residual_ridge": residual_by_name["ridge_reference"],
        "residual_poly_reference_if_available": residual_by_name["polynomial_reference"],
        "best_scalar": float(best_scalar),
        "success_probability": computation.summary["success_probability"],
        "postselection_probability": computation.summary["postselection_probability"],
        "conditioning_sensitivity_score": sensitivity,
        "dominant_gap_source": dominant,
    }


def _dominant_gap_source(
    *,
    residual_no_update: float,
    residual_raw: float,
    residual_best_scalar: float,
    residual_ridge: float,
    residual_poly: float,
    sensitivity: float,
) -> str:
    close_to_ridge = max(10.0 * residual_ridge, 1.0e-6 * max(residual_no_update, 1.0))
    if residual_best_scalar <= close_to_ridge:
        return "norm_scaling_recovery"
    if sensitivity > 5.0:
        return "direction_accuracy_and_conditioning_sensitivity"
    if abs(residual_poly - residual_raw) <= 0.1 * max(residual_raw, 1.0e-15):
        return "polynomial_approximation_or_filter_degree"
    return "direction_polynomial_extraction_or_conditioning"


def _conditioning_sensitivity_rows(
    H: np.ndarray,
    r: np.ndarray,
    qsvt: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    qsvt_norm = float(np.linalg.norm(qsvt))
    direction = qsvt / max(qsvt_norm, 1.0e-15)
    rows: list[dict[str, Any]] = []
    for epsilon in [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1]:
        residuals = []
        for trial in range(5):
            noise = rng.normal(size=direction.size)
            noise -= direction * float(np.dot(direction, noise))
            noise_norm = float(np.linalg.norm(noise))
            if noise_norm > 0.0:
                noise /= noise_norm
            perturbed = direction + float(epsilon) * noise
            perturbed /= max(float(np.linalg.norm(perturbed)), 1.0e-15)
            update = qsvt_norm * perturbed
            residuals.append(_residual_norm(H, update, r))
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "trial": int(trial),
                    "residual_norm": residuals[-1],
                    "relative_to_unperturbed_qsvt": residuals[-1]
                    / max(_residual_norm(H, qsvt, r), 1.0e-15),
                }
            )
    return rows


def _sensitivity_score(H: np.ndarray, r: np.ndarray, qsvt: np.ndarray, *, seed: int) -> float:
    rows = _conditioning_sensitivity_rows(H, r, qsvt, seed=seed)
    base = np.mean([row["residual_norm"] for row in rows if row["epsilon"] == 0.0])
    perturbed = np.mean([row["residual_norm"] for row in rows if row["epsilon"] == 1.0e-2])
    return float(abs(perturbed - base) / max(1.0e-2 * base, 1.0e-15))


def _scale_sweep_rows(
    H: np.ndarray,
    r: np.ndarray,
    qsvt: np.ndarray,
    *,
    ridge_norm: float,
    best_scalar: float,
) -> list[dict[str, Any]]:
    qsvt_norm = float(np.linalg.norm(qsvt))
    norm_scalar = ridge_norm / max(qsvt_norm, 1.0e-15)
    scalars = sorted(
        {
            0.0,
            1.0,
            float(norm_scalar),
            float(best_scalar),
            0.5 * float(best_scalar),
            1.5 * float(best_scalar),
            2.0 * float(best_scalar),
        }
    )
    return [
        {
            "scale": scalar,
            "residual_norm": _residual_norm(H, scalar * qsvt, r),
            "is_best_scalar": bool(np.isclose(scalar, best_scalar)),
            "is_ridge_norm_scalar": bool(np.isclose(scalar, norm_scalar)),
        }
        for scalar in scalars
    ]


def _variant_row(
    name: str,
    update: np.ndarray,
    *,
    H: np.ndarray,
    r: np.ndarray,
    ridge: np.ndarray,
) -> dict[str, Any]:
    return {
        "update_variant": name,
        "update_norm": float(np.linalg.norm(update)),
        "residual_norm": _residual_norm(H, update, r),
        "relative_update_error_vs_ridge": _relative_error(update, ridge),
        "state_direction_error_vs_ridge": _direction_error(update, ridge),
    }


def _write_outputs(
    *,
    output_dir: Path,
    resolved: dict[str, Any],
    summary: dict[str, Any],
    residual_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    poly_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    poly_metadata: dict[str, Any],
) -> dict[str, Path]:
    summary_path = output_dir / "norm_gap_summary.csv"
    residual_path = output_dir / "residual_gap_decomposition.csv"
    scale_path = output_dir / "scale_optimized_residuals.csv"
    poly_path = output_dir / "polynomial_reference_comparison.csv"
    sensitivity_path = output_dir / "conditioning_sensitivity.csv"
    interpretation_path = output_dir / "norm_recovery_interpretation.md"
    metadata_path = output_dir / "polynomial_metadata.json"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)
    pd.DataFrame(scale_rows).to_csv(scale_path, index=False)
    pd.DataFrame(poly_rows).to_csv(poly_path, index=False)
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    write_json(metadata_path, poly_metadata)
    interpretation_path.write_text(_interpretation_markdown(summary), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "norm_gap_summary": str(summary_path),
            "residual_gap_decomposition": str(residual_path),
            "scale_optimized_residuals": str(scale_path),
            "polynomial_reference_comparison": str(poly_path),
            "conditioning_sensitivity": str(sensitivity_path),
            "norm_recovery_interpretation": str(interpretation_path),
            "polynomial_metadata": str(metadata_path),
        },
        input_config=resolved,
        claim_boundary=NORM_GAP_CLAIM,
    )
    return {
        "manifest": manifest,
        "norm_gap_summary": summary_path,
        "residual_gap_decomposition": residual_path,
        "scale_optimized_residuals": scale_path,
        "polynomial_reference_comparison": poly_path,
        "conditioning_sensitivity": sensitivity_path,
        "norm_recovery_interpretation": interpretation_path,
        "polynomial_metadata": metadata_path,
    }


def _interpretation_markdown(summary: dict[str, Any]) -> str:
    if summary["dominant_gap_source"] == "norm_scaling_recovery":
        conclusion = "The main residual gap is norm/scaling recovery."
    elif summary["dominant_gap_source"] == "direction_accuracy_and_conditioning_sensitivity":
        conclusion = (
            "The selected subproblem is numerically sensitive, and residual accuracy "
            "requires stronger direction accuracy or better conditioning."
        )
    else:
        conclusion = (
            "The main residual gap is direction, polynomial, extraction, or "
            "conditioning sensitivity."
        )
    return "\n".join(
        [
            "# QSVT Norm and Residual-Gap Audit",
            "",
            NORM_GAP_CLAIM,
            "",
            f"- Condition number: {summary['condition_number']:.17g}",
            f"- Raw QSVT residual: {summary['residual_qsvt_raw']:.17g}",
            f"- Ridge residual: {summary['residual_ridge']:.17g}",
            f"- Best scalar: {summary['best_scalar']:.17g}",
            f"- Best scalar-rescaled residual: {summary['residual_qsvt_best_scalar']:.17g}",
            f"- Dominant residual-gap source: {summary['dominant_gap_source']}",
            "",
            conclusion,
            "",
            GATE_LEVEL_SOLVER_CLAIM,
            "",
        ]
    )


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), 1.0e-15)
    )


def _direction_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    cand_norm = float(np.linalg.norm(cand))
    ref_norm = float(np.linalg.norm(ref))
    if cand_norm <= 1.0e-15 or ref_norm <= 1.0e-15:
        return float("nan")
    positive = np.linalg.norm(cand / cand_norm - ref / ref_norm)
    negative = np.linalg.norm(cand / cand_norm + ref / ref_norm)
    return float(min(positive, negative))


def _residual_norm(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H) @ np.asarray(update) - np.asarray(r)))


def _condition_number(singular_values: np.ndarray) -> float:
    positive = np.asarray(singular_values, dtype=np.float64)
    positive = positive[positive > 1.0e-14]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())
