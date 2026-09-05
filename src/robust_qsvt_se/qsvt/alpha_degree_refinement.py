from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.utils.io import ensure_directory

ALPHA_DEGREE_REFINEMENT_CLAIM = (
    "This alpha/degree refinement tests residual sensitivity for selected "
    "small IEEE-derived gate-level QSVT subproblems. Ridge/Tikhonov remains "
    "the reference target."
)

REFINEMENT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "matrix_shape",
    "condition_number",
    "alpha",
    "requested_degree",
    "synthesized_degree",
    "phase_count",
    "query_count",
    "qsvt_query_count",
    "polynomial_target_error_if_available",
    "state_error_vs_ridge",
    "phase_or_sign_aligned_state_error",
    "relative_update_error_raw",
    "relative_update_error_best_scalar",
    "residual_no_update",
    "residual_qsvt_raw",
    "residual_qsvt_best_scalar",
    "residual_ridge",
    "residual_ratio_raw_vs_no_update",
    "residual_ratio_best_scalar_vs_no_update",
    "success_probability",
    "postselection_cost_proxy",
    "amplitude_amplification_cost_proxy",
    "run_status",
    "failure_reason_if_any",
]


def run_alpha_degree_residual_refinement(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75, 101, 151, 201],
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/qsvt_alpha_degree_residual_refinement",
        "subproblem_id": "ieee14_ac_high_leverage_4x4",
        "selection_mode": "high_leverage",
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
    rows = evaluate_alpha_degree_grid(
        subproblem=subproblem,
        alphas=[float(value) for value in resolved["alphas"]],
        degrees=[int(value) for value in resolved["degrees"]],
        shots=int(resolved["shots"]),
        seed=int(resolved["seed"]),
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        subproblem_id=str(resolved["subproblem_id"]),
        selection_mode=str(resolved["selection_mode"]),
    )
    artifacts = write_refinement_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_alpha_degree_grid(
    *,
    subproblem: SelectedSubproblem,
    alphas: list[float],
    degrees: list[int],
    shots: int,
    seed: int,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for degree in degrees:
            rows.append(
                evaluate_single_alpha_degree(
                    subproblem=subproblem,
                    alpha=float(alpha),
                    degree=int(degree),
                    shots=int(shots),
                    seed=int(seed),
                    case=case,
                    model=model,
                    subproblem_id=subproblem_id,
                    selection_mode=selection_mode,
                )
            )
    return rows


def evaluate_single_alpha_degree(
    *,
    subproblem: SelectedSubproblem,
    alpha: float,
    degree: int,
    shots: int,
    seed: int,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
) -> dict[str, Any]:
    try:
        computation = solve_gate_level_state_estimation_problem(
            H_tilde=subproblem.H_tilde,
            r_tilde=subproblem.r_tilde,
            alpha=float(alpha),
            degree=int(degree),
            shots=int(shots),
            seed=int(seed),
            metadata=subproblem.metadata,
            transpile_qubit_limit=0,
            export_qasm=False,
        )
        return completed_refinement_row(
            computation=computation,
            alpha=float(alpha),
            degree=int(degree),
            case=case,
            model=model,
            subproblem_id=subproblem_id,
            selection_mode=selection_mode,
        )
    except Exception as exc:  # pragma: no cover - version/resource dependent
        return failure_refinement_row(
            H_tilde=subproblem.H_tilde,
            alpha=float(alpha),
            degree=int(degree),
            case=case,
            model=model,
            subproblem_id=subproblem_id,
            selection_mode=selection_mode,
            failure=f"{type(exc).__name__}: {exc}",
        )


def completed_refinement_row(
    *,
    computation: Any,
    alpha: float,
    degree: int,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
) -> dict[str, Any]:
    H = np.asarray(computation.H_tilde, dtype=np.float64)
    r = np.asarray(computation.r_tilde, dtype=np.float64)
    qsvt = np.asarray(computation.qsvt_update, dtype=np.float64)
    ridge = np.asarray(computation.ridge_update, dtype=np.float64)
    singular_values = np.linalg.svd(H, compute_uv=False)
    best_scalar = best_scalar_for_residual(H, qsvt, r)
    best_update = best_scalar * qsvt
    residual_no_update = float(np.linalg.norm(r))
    residual_raw = float(np.linalg.norm(H @ qsvt - r))
    residual_best = float(np.linalg.norm(H @ best_update - r))
    residual_ridge = float(np.linalg.norm(H @ ridge - r))
    relative_raw = _relative_error(qsvt, ridge)
    relative_best = _relative_error(best_update, ridge)
    polynomial_error = _polynomial_reference_error(H, r, ridge, computation=computation)
    p_success = _clip_success(computation.summary["success_probability"])
    query_count = int(computation.summary["qsvt_query_count"])
    return {
        "case": case,
        "model": model,
        "subproblem_id": subproblem_id,
        "selection_mode": selection_mode,
        "matrix_shape": f"{H.shape[0]}x{H.shape[1]}",
        "condition_number": _condition_number(singular_values),
        "alpha": float(alpha),
        "requested_degree": int(degree),
        "synthesized_degree": int(computation.summary["synthesized_degree"]),
        "phase_count": int(computation.summary["phase_count"]),
        "query_count": query_count,
        "qsvt_query_count": query_count,
        "polynomial_target_error_if_available": polynomial_error,
        "state_error_vs_ridge": float(computation.summary["state_error_vs_ridge"]),
        "phase_or_sign_aligned_state_error": float(
            computation.summary["phase_or_sign_aligned_state_error"]
        ),
        "relative_update_error_raw": relative_raw,
        "relative_update_error_best_scalar": relative_best,
        "residual_no_update": residual_no_update,
        "residual_qsvt_raw": residual_raw,
        "residual_qsvt_best_scalar": residual_best,
        "residual_ridge": residual_ridge,
        "residual_ratio_raw_vs_no_update": residual_raw / max(residual_no_update, 1.0e-15),
        "residual_ratio_best_scalar_vs_no_update": residual_best / max(residual_no_update, 1.0e-15),
        "success_probability": p_success,
        "postselection_cost_proxy": 1.0 / p_success,
        "amplitude_amplification_cost_proxy": 1.0 / np.sqrt(p_success),
        "run_status": "completed",
        "failure_reason_if_any": "",
    }


def failure_refinement_row(
    *,
    H_tilde: np.ndarray,
    alpha: float,
    degree: int,
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    failure: str,
) -> dict[str, Any]:
    H = np.asarray(H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H, compute_uv=False) if H.size else np.array([])
    row = {column: np.nan for column in REFINEMENT_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "selection_mode": selection_mode,
            "matrix_shape": f"{H.shape[0]}x{H.shape[1]}" if H.ndim == 2 else "",
            "condition_number": _condition_number(singular_values),
            "alpha": float(alpha),
            "requested_degree": int(degree),
            "run_status": "failed",
            "failure_reason_if_any": failure,
        }
    )
    return row


def write_refinement_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    summary_path = output_dir / "alpha_degree_refinement_summary.csv"
    by_alpha_path = output_dir / "best_residual_by_alpha.csv"
    by_degree_path = output_dir / "best_residual_by_degree.csv"
    tradeoff_path = output_dir / "accuracy_success_tradeoff.csv"
    failures_path = output_dir / "refinement_failures.csv"
    interpretation_path = output_dir / "alpha_degree_refinement_interpretation.md"
    frame = pd.DataFrame(rows, columns=REFINEMENT_COLUMNS)
    completed = frame[frame["run_status"] == "completed"].copy()
    frame.to_csv(summary_path, index=False)
    _best_by(completed, "alpha").to_csv(by_alpha_path, index=False)
    _best_by(completed, "requested_degree").to_csv(by_degree_path, index=False)
    _tradeoff_frame(completed).to_csv(tradeoff_path, index=False)
    frame[frame["run_status"] != "completed"].to_csv(failures_path, index=False)
    interpretation_path.write_text(_interpretation_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "alpha_degree_refinement_summary": str(summary_path),
            "best_residual_by_alpha": str(by_alpha_path),
            "best_residual_by_degree": str(by_degree_path),
            "accuracy_success_tradeoff": str(tradeoff_path),
            "refinement_failures": str(failures_path),
            "alpha_degree_refinement_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=ALPHA_DEGREE_REFINEMENT_CLAIM,
    )
    return {
        "manifest": manifest,
        "alpha_degree_refinement_summary": summary_path,
        "best_residual_by_alpha": by_alpha_path,
        "best_residual_by_degree": by_degree_path,
        "accuracy_success_tradeoff": tradeoff_path,
        "refinement_failures": failures_path,
        "alpha_degree_refinement_interpretation": interpretation_path,
    }


def classify_refinement_result(frame: pd.DataFrame) -> str:
    completed = frame[frame["run_status"] == "completed"].copy()
    if completed.empty:
        return "inconclusive_due_to_runtime_limits"
    best = completed.loc[completed["residual_qsvt_best_scalar"].astype(float).idxmin()]
    p_success = float(best["success_probability"])
    synthesized_unique = set(int(value) for value in completed["synthesized_degree"].dropna())
    by_degree = _best_by(completed, "requested_degree")
    degree_improves = _series_improves(
        by_degree["requested_degree"].to_numpy(dtype=float),
        by_degree["residual_qsvt_best_scalar"].to_numpy(dtype=float),
    )
    by_alpha = _best_by(completed, "alpha")
    alpha_span = float(by_alpha["residual_qsvt_best_scalar"].max()) / max(
        float(by_alpha["residual_qsvt_best_scalar"].min()),
        1.0e-15,
    )
    residual_ridge = float(best["residual_ridge"])
    residual_best = float(best["residual_qsvt_best_scalar"])
    ridge_floor = max(10.0 * residual_ridge, 1.0e-8)
    if p_success < 1.0e-3:
        return "success_probability_limited"
    if residual_best <= ridge_floor:
        if alpha_span > 2.0:
            return "alpha_sensitive_and_improvable"
        return "degree_limited_and_improvable"
    if len(synthesized_unique) <= 1 and len(set(completed["requested_degree"])) > 1:
        return "inconclusive_due_to_runtime_limits"
    if degree_improves:
        return "degree_limited_and_improvable"
    if alpha_span > 2.0:
        return "alpha_sensitive_and_improvable"
    return "not_fixed_by_degree_or_alpha"


def _polynomial_reference_error(
    H: np.ndarray,
    r: np.ndarray,
    ridge: np.ndarray,
    *,
    computation: Any,
) -> float:
    try:
        diagnostics = computation.diagnostics
        coefficients = np.asarray(diagnostics["qsvt_polynomial_coefficients"], dtype=np.float64)
        beta = float(computation.summary["beta"])
        scale_factor = float(diagnostics["bounded_scaling_C"])
        B = np.asarray(H, dtype=np.float64).T
        A = B / max(beta, np.finfo(float).eps)
        U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
        polynomial = Polynomial(coefficients)
        polynomial_operator = U @ (polynomial(singular_values)[:, None] * Vh)
        residual_norm = float(np.linalg.norm(r))
        residual_state = np.asarray(r, dtype=np.float64) / max(residual_norm, 1.0e-15)
        bounded_vector = polynomial_operator @ residual_state
        poly = residual_norm * rescale_bounded_target_to_original(
            bounded_vector,
            beta=beta,
            C=scale_factor,
        )
    except Exception:
        return float("nan")
    return _relative_error(poly, ridge)


def _best_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=REFINEMENT_COLUMNS)
    rows = []
    for _, group in frame.groupby(column, sort=True):
        idx = group["residual_qsvt_best_scalar"].astype(float).idxmin()
        rows.append(frame.loc[idx].to_dict())
    return pd.DataFrame(rows, columns=REFINEMENT_COLUMNS)


def _tradeoff_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "subproblem_id",
                "alpha",
                "requested_degree",
                "synthesized_degree",
                "residual_qsvt_best_scalar",
                "residual_ratio_best_scalar_vs_no_update",
                "success_probability",
                "postselection_cost_proxy",
                "amplitude_amplification_cost_proxy",
            ]
        )
    columns = [
        "subproblem_id",
        "alpha",
        "requested_degree",
        "synthesized_degree",
        "residual_qsvt_best_scalar",
        "residual_ratio_best_scalar_vs_no_update",
        "success_probability",
        "postselection_cost_proxy",
        "amplitude_amplification_cost_proxy",
    ]
    return frame[columns].sort_values(
        ["residual_qsvt_best_scalar", "amplitude_amplification_cost_proxy"],
        kind="mergesort",
    )


def _interpretation_markdown(frame: pd.DataFrame) -> str:
    classification = classify_refinement_result(frame)
    completed = frame[frame["run_status"] == "completed"].copy()
    failed = int((frame["run_status"] != "completed").sum())
    if completed.empty:
        best_lines = ["- No completed refinement rows."]
    else:
        best = completed.loc[completed["residual_qsvt_best_scalar"].astype(float).idxmin()]
        best_lines = [
            f"- Best alpha: {float(best['alpha']):.17g}",
            f"- Best requested degree: {int(best['requested_degree'])}",
            f"- Best synthesized degree: {int(best['synthesized_degree'])}",
            f"- Best raw QSVT residual: {float(best['residual_qsvt_raw']):.17g}",
            f"- Best scalar-rescaled residual: {float(best['residual_qsvt_best_scalar']):.17g}",
            f"- Ridge residual at best row: {float(best['residual_ridge']):.17g}",
            f"- Success probability at best row: {float(best['success_probability']):.17g}",
        ]
    return "\n".join(
        [
            "# Alpha/Degree Residual Refinement",
            "",
            ALPHA_DEGREE_REFINEMENT_CLAIM,
            "",
            *best_lines,
            f"- Failed rows: {failed}",
            f"- Residual-gap classification: {classification}",
            "",
            "Requested degree and synthesized degree are reported separately; "
            "the current dense phase-synthesis path may cap the actual degree.",
            "",
        ]
    )


def _series_improves(x_values: np.ndarray, y_values: np.ndarray) -> bool:
    if x_values.size < 2:
        return False
    order = np.argsort(x_values)
    sorted_y = y_values[order]
    return bool(sorted_y[-1] < 0.8 * sorted_y[0])


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(float(np.linalg.norm(reference)), 1.0e-15)
    )


def _condition_number(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.size == 0:
        return float("inf")
    sigma_max = float(np.max(values))
    sigma_min = float(np.min(values))
    if sigma_min <= 1.0e-14:
        return float("inf")
    return sigma_max / sigma_min


def _clip_success(value: float) -> float:
    return float(np.clip(float(value), 1.0e-300, 1.0))
