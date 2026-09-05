from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
)
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory

WEIGHTED_SUPPORT_CLAIM = (
    "Weighted singular-support diagnostic for the QSVT-compatible regularized "
    "spectral filter on a selected IEEE14-derived subproblem. It identifies which "
    "singular directions drive the Ridge/Tikhonov update and the residual, so a "
    "co-designed bounded QSVT target can be fitted where it matters. Ridge/Tikhonov "
    "remains the reference filter; no QSVT superiority over Ridge/Tikhonov, quantum "
    "speedup, quantum advantage, or hardware execution is claimed."
)

# Default reference degree used to evaluate the current bounded QSVT polynomial error.
DEFAULT_CURRENT_DEGREE = 51
HIGH_WEIGHT_CUMULATIVE_FRACTION = 0.9

SUMMARY_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "singular_index",
    "sigma",
    "residual_projection",
    "ridge_filter_value",
    "ridge_update_component_norm",
    "residual_projection_weight",
    "update_magnitude_weight",
    "residual_sensitivity_weight",
    "combined_weight",
    "high_weight_direction",
    "current_polynomial_value_if_available",
    "current_weighted_filter_error_if_available",
]

HIGH_WEIGHT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "singular_index",
    "sigma",
    "combined_weight",
    "ridge_update_component_norm",
    "current_weighted_filter_error_if_available",
]

POLYNOMIAL_ERROR_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "current_degree",
    "singular_index",
    "sigma",
    "combined_weight",
    "high_weight_direction",
    "ridge_filter_value",
    "current_polynomial_value_if_available",
    "absolute_filter_error",
    "weighted_filter_error_contribution",
]


@dataclass(frozen=True, slots=True)
class SingularSupport:
    """Per-direction weights describing where the Ridge update and residual live."""

    sigma: np.ndarray
    sigma_normalized: np.ndarray
    beta: float
    alpha_norm: float
    domain_min: float
    U: np.ndarray
    Vh: np.ndarray
    residual_projection: np.ndarray
    ridge_filter_value: np.ndarray
    ridge_update_component_norm: np.ndarray
    residual_projection_weight: np.ndarray
    update_magnitude_weight: np.ndarray
    residual_sensitivity_weight: np.ndarray
    combined_weight: np.ndarray
    high_weight_direction: np.ndarray


def compute_singular_support(H: np.ndarray, r: np.ndarray, *, alpha: float) -> SingularSupport:
    """Compute singular-direction importance weights from the SVD of H.

    Ridge/Tikhonov acts on the SVD of H directly: with H = U S V^T and
    c_i = u_i^T r, the update is sum_i P_alpha(sigma_i) c_i v_i. Three weight
    families rank directions by residual projection |c_i|, update magnitude
    |P_alpha(sigma_i) c_i|, and residual sensitivity |P_alpha c_i| * sigma_i.
    """

    H = np.asarray(H, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if H.ndim != 2 or r.ndim != 1 or H.shape[0] != r.size:
        raise ValueError("H must be m x n and r must have length m")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")

    U, sigma, Vh = np.linalg.svd(H, full_matrices=False)
    positive = sigma[sigma > 1.0e-14]
    if positive.size == 0:
        raise ValueError("selected subproblem has no positive singular values")
    beta = max(float(sigma[0]), np.finfo(float).eps)
    sigma_normalized = sigma / beta
    alpha_norm = float(alpha) / beta**2
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive) / beta)), 0.95)

    residual_projection = U.T @ r
    ridge_filter_value = ridge_filter(sigma, alpha=float(alpha))
    ridge_component_coefficient = ridge_filter_value * residual_projection
    ridge_update_component_norm = np.abs(ridge_component_coefficient)

    residual_projection_weight = np.abs(residual_projection)
    update_magnitude_weight = np.abs(ridge_component_coefficient)
    residual_sensitivity_weight = update_magnitude_weight * sigma

    combined_weight = _combine_weights(
        residual_projection_weight, update_magnitude_weight, residual_sensitivity_weight
    )
    high_weight_direction = _high_weight_mask(combined_weight)

    return SingularSupport(
        sigma=sigma,
        sigma_normalized=sigma_normalized,
        beta=beta,
        alpha_norm=alpha_norm,
        domain_min=domain_min,
        U=U,
        Vh=Vh,
        residual_projection=residual_projection,
        ridge_filter_value=ridge_filter_value,
        ridge_update_component_norm=ridge_update_component_norm,
        residual_projection_weight=residual_projection_weight,
        update_magnitude_weight=update_magnitude_weight,
        residual_sensitivity_weight=residual_sensitivity_weight,
        combined_weight=combined_weight,
        high_weight_direction=high_weight_direction,
    )


def current_polynomial_filter(
    support: SingularSupport, *, degree: int, grid_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Physical filter value of the current bounded QSVT polynomial at each sigma.

    Returns (physical_filter_value, absolute_error_vs_ridge). The bounded QSVT
    polynomial approximates x/(x^2 + alpha_norm) on the normalized A-domain; its
    physical filter value is (1/beta) p_raw(sigma/beta), directly comparable to
    the Ridge filter value P_alpha(sigma).
    """

    if int(degree) < 1 or int(degree) % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    approximation = fit_odd_regularized_polynomial(
        alpha=support.alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=support.domain_min,
        domain_max=1.0,
        grid_size=max(int(grid_size), int(degree) + 2),
    )
    raw_polynomial = Polynomial(np.asarray(approximation.power_coefficients, dtype=np.float64))
    physical_value = raw_polynomial(support.sigma_normalized) / support.beta
    absolute_error = np.abs(physical_value - support.ridge_filter_value)
    return physical_value, absolute_error


def run_qsvt_weighted_singular_support(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "current_degree": DEFAULT_CURRENT_DEGREE,
        "grid_size": 4096,
        "seed": 123,
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "output_dir": "outputs/qsvt_weighted_singular_support",
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
    summary_rows, error_rows = evaluate_weighted_singular_support(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        current_degree=int(resolved["current_degree"]),
        grid_size=int(resolved["grid_size"]),
    )
    artifacts = write_weighted_singular_support_outputs(
        output_dir, resolved, summary_rows, error_rows
    )
    return {
        "output_dir": output_dir,
        "rows": summary_rows,
        "error_rows": error_rows,
        "artifacts": artifacts,
    }


def evaluate_weighted_singular_support(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    case: str,
    model: str,
    subproblem_id: str,
    current_degree: int = DEFAULT_CURRENT_DEGREE,
    grid_size: int = 4096,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for alpha in alphas:
        support = compute_singular_support(H, r, alpha=float(alpha))
        try:
            physical_value, absolute_error = current_polynomial_filter(
                support, degree=int(current_degree), grid_size=int(grid_size)
            )
            weight_sum = float(np.sum(support.combined_weight))
            weighted_contribution = (
                support.combined_weight * absolute_error / weight_sum
                if weight_sum > 0.0
                else np.full_like(absolute_error, np.nan)
            )
            polynomial_available = True
        except Exception:
            physical_value = np.full(support.sigma.size, np.nan)
            absolute_error = np.full(support.sigma.size, np.nan)
            weighted_contribution = np.full(support.sigma.size, np.nan)
            polynomial_available = False

        base = {"case": case, "model": model, "subproblem_id": subproblem_id, "alpha": float(alpha)}
        for index in range(support.sigma.size):
            summary_rows.append(
                {
                    **base,
                    "singular_index": int(index),
                    "sigma": float(support.sigma[index]),
                    "residual_projection": float(support.residual_projection[index]),
                    "ridge_filter_value": float(support.ridge_filter_value[index]),
                    "ridge_update_component_norm": float(
                        support.ridge_update_component_norm[index]
                    ),
                    "residual_projection_weight": float(support.residual_projection_weight[index]),
                    "update_magnitude_weight": float(support.update_magnitude_weight[index]),
                    "residual_sensitivity_weight": float(
                        support.residual_sensitivity_weight[index]
                    ),
                    "combined_weight": float(support.combined_weight[index]),
                    "high_weight_direction": bool(support.high_weight_direction[index]),
                    "current_polynomial_value_if_available": float(physical_value[index]),
                    "current_weighted_filter_error_if_available": (
                        float(weighted_contribution[index])
                        if polynomial_available
                        else float("nan")
                    ),
                }
            )
            error_rows.append(
                {
                    **base,
                    "current_degree": int(current_degree),
                    "singular_index": int(index),
                    "sigma": float(support.sigma[index]),
                    "combined_weight": float(support.combined_weight[index]),
                    "high_weight_direction": bool(support.high_weight_direction[index]),
                    "ridge_filter_value": float(support.ridge_filter_value[index]),
                    "current_polynomial_value_if_available": float(physical_value[index]),
                    "absolute_filter_error": float(absolute_error[index]),
                    "weighted_filter_error_contribution": float(weighted_contribution[index]),
                }
            )
    return summary_rows, error_rows


def write_weighted_singular_support_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_frame = _frame_with_columns(summary_rows, SUMMARY_COLUMNS)
    high_weight_frame = summary_frame[summary_frame["high_weight_direction"] == True][  # noqa: E712
        HIGH_WEIGHT_COLUMNS
    ].copy()
    error_frame = _frame_with_columns(error_rows, POLYNOMIAL_ERROR_COLUMNS)

    summary_path = output_dir / "singular_support_weights.csv"
    high_weight_path = output_dir / "high_weight_direction_summary.csv"
    error_path = output_dir / "current_polynomial_error_on_weighted_support.csv"
    interpretation_path = output_dir / "weighted_singular_support_interpretation.md"

    summary_frame.to_csv(summary_path, index=False)
    high_weight_frame.to_csv(high_weight_path, index=False)
    error_frame.to_csv(error_path, index=False)
    interpretation_path.write_text(
        weighted_singular_support_interpretation(summary_frame, error_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "singular_support_weights": str(summary_path),
            "high_weight_direction_summary": str(high_weight_path),
            "current_polynomial_error_on_weighted_support": str(error_path),
            "weighted_singular_support_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=WEIGHTED_SUPPORT_CLAIM,
    )
    return {
        "manifest": manifest,
        "singular_support_weights": summary_path,
        "high_weight_direction_summary": high_weight_path,
        "current_polynomial_error_on_weighted_support": error_path,
        "weighted_singular_support_interpretation": interpretation_path,
    }


def weighted_singular_support_interpretation(
    summary_frame: pd.DataFrame, error_frame: pd.DataFrame
) -> str:
    if summary_frame.empty:
        return "\n".join(["# Weighted Singular-Support Diagnostic", "", WEIGHTED_SUPPORT_CLAIM, ""])

    high = summary_frame[summary_frame["high_weight_direction"] == True]  # noqa: E712
    high_count = len(high)
    total = len(summary_frame)

    errors_completed = error_frame.dropna(subset=["weighted_filter_error_contribution"])
    concentration = _error_concentration(errors_completed)
    failure_mode = (
        "concentrated on high-weight directions"
        if concentration.startswith("concentrated")
        else "spread across the support"
    )

    dominant = (
        summary_frame.loc[summary_frame["combined_weight"].astype(float).idxmax()]
        if not summary_frame.empty
        else None
    )
    dominant_line = (
        f"singular_index={int(dominant['singular_index'])} at sigma="
        f"{float(dominant['sigma']):.6g} (alpha={float(dominant['alpha']):.3g}, "
        f"combined_weight={float(dominant['combined_weight']):.4g})"
        if dominant is not None
        else "n/a"
    )

    return "\n".join(
        [
            "# Weighted Singular-Support Diagnostic",
            "",
            WEIGHTED_SUPPORT_CLAIM,
            "",
            "## Counts",
            f"- Total (alpha, singular_index) rows: {total}",
            f"- High-weight directions (cumulative {HIGH_WEIGHT_CUMULATIVE_FRACTION:g} of "
            f"combined weight): {high_count}",
            "",
            "## Required Answers",
            f"1. Which singular directions matter most? Dominant direction: {dominant_line}.",
            f"2. Dominant weighted-error directions: {concentration}.",
            "3. Current polynomial failure mode: the bounded QSVT polynomial error, when "
            f"weighted by singular-direction importance, is {failure_mode}; "
            "this tells the co-designed target where to reduce error.",
            "",
            "## Interpretation",
            "- residual_projection_weight ranks directions by |u_i^T r| (where the residual "
            "energy lives).",
            "- update_magnitude_weight ranks by |P_alpha(sigma_i) (u_i^T r)| (where the Ridge "
            "update magnitude lives).",
            "- residual_sensitivity_weight ranks by |P_alpha(sigma_i)(u_i^T r)| * sigma_i (how "
            "much each direction moves the residual H Delta x).",
            "- combined_weight is the normalized average of the three; high_weight_direction "
            f"marks the smallest set covering {HIGH_WEIGHT_CUMULATIVE_FRACTION:g} of the combined "
            "weight.",
            "",
        ]
    )


def _combine_weights(*weights: np.ndarray) -> np.ndarray:
    normalized = []
    for weight in weights:
        total = float(np.sum(weight))
        normalized.append(weight / total if total > 0.0 else np.zeros_like(weight))
    combined = np.mean(normalized, axis=0)
    total = float(np.sum(combined))
    if total <= 0.0:
        size = combined.size
        return np.full(size, 1.0 / size) if size else combined
    return combined / total


def _high_weight_mask(combined_weight: np.ndarray) -> np.ndarray:
    mask = np.zeros(combined_weight.size, dtype=bool)
    if combined_weight.size == 0:
        return mask
    order = np.argsort(-combined_weight, kind="stable")
    cumulative = 0.0
    for position in order:
        mask[position] = True
        cumulative += float(combined_weight[position])
        if cumulative >= HIGH_WEIGHT_CUMULATIVE_FRACTION:
            break
    return mask


def _error_concentration(error_frame: pd.DataFrame) -> str:
    if error_frame.empty:
        return "unavailable (current polynomial not synthesized)"
    high = error_frame[error_frame["high_weight_direction"] == True]  # noqa: E712
    low = error_frame[error_frame["high_weight_direction"] != True]  # noqa: E712
    high_error = float(high["weighted_filter_error_contribution"].astype(float).sum())
    low_error = float(low["weighted_filter_error_contribution"].astype(float).sum())
    if high_error >= low_error:
        return (
            f"concentrated on high-weight directions (high-weight weighted error {high_error:.4g} "
            f">= remaining {low_error:.4g})"
        )
    return (
        f"spread across the support (high-weight weighted error {high_error:.4g} < remaining "
        f"{low_error:.4g})"
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
