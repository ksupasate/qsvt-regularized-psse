from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import rescale_bounded_target_to_original
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    extract_state_estimation_subproblem,
    solve_gate_level_state_estimation_problem,
)
from robust_qsvt_se.qsvt.norm_residual_gap_audit import best_scalar_for_residual
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.utils.io import ensure_directory

BEST_CONFIG_GATE_CLAIM = (
    "Gate-level validation is run only for configurations selected from "
    "polynomial-action residual evidence. It remains selected-subproblem dense "
    "simulator evidence, not full IEEE-scale hardware execution."
)

GATE_VALIDATION_COLUMNS = [
    "subproblem_id",
    "alpha",
    "degree",
    "synthesized_phase_degree",
    "phase_count",
    "query_count",
    "polynomial_action_residual",
    "gate_level_residual",
    "best_scalar_gate_residual",
    "ridge_residual",
    "state_error_gate_vs_poly",
    "state_error_gate_vs_ridge",
    "success_probability",
    "circuit_depth",
    "two_qubit_gates",
    "run_status",
    "failure_reason_if_any",
    "classification",
]


def run_best_config_gate_validation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_polynomial_action_decomposition/polynomial_action_residuals.csv",
        "max_configs": 5,
        "shots": 1000,
        "seed": 123,
        "case_source": "pypower",
        "submatrix_size": 4,
        "max_manageable_degree": 101,
        "min_success_probability_proxy": 1.0e-12,
        "output_dir": "outputs/qsvt_best_config_gate_validation",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    input_frame = pd.read_csv(resolved["input"])
    selected = select_gate_validation_configs(
        input_frame,
        max_configs=int(resolved["max_configs"]),
        max_manageable_degree=int(resolved["max_manageable_degree"]),
        min_success_probability_proxy=float(resolved["min_success_probability_proxy"]),
    )
    rows = [
        validate_single_gate_config(
            row=row,
            shots=int(resolved["shots"]),
            seed=int(resolved["seed"]),
            case_source=str(resolved["case_source"]),
            default_submatrix_size=int(resolved["submatrix_size"]),
        )
        for row in selected.to_dict("records")
    ]
    artifacts = write_gate_validation_outputs(output_dir, resolved, selected, rows)
    return {
        "output_dir": output_dir,
        "selected_configs": selected,
        "rows": rows,
        "artifacts": artifacts,
    }


def select_gate_validation_configs(
    frame: pd.DataFrame,
    *,
    max_configs: int,
    max_manageable_degree: int = 101,
    min_success_probability_proxy: float = 1.0e-12,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(frame.columns))
    data = frame.copy()
    if "run_status" in data.columns:
        data = data[data["run_status"] == "completed"].copy()
    if data.empty:
        return data
    degree_column = (
        "effective_qsvt_degree" if "effective_qsvt_degree" in data.columns else "requested_degree"
    )
    data["degree_for_validation"] = data[degree_column].fillna(data["requested_degree"]).astype(int)
    data = data[data["degree_for_validation"] <= int(max_manageable_degree)].copy()
    if "success_probability_proxy" in data.columns:
        data = data[
            data["success_probability_proxy"].fillna(1.0) >= float(min_success_probability_proxy)
        ].copy()
    if data.empty:
        return data
    sort_columns = [
        "best_scalar_polynomial_residual",
        "polynomial_action_residual",
        "degree_for_validation",
        "condition_number",
    ]
    existing = [column for column in sort_columns if column in data.columns]
    data = data.sort_values(existing, kind="mergesort") if existing else data
    return data.head(int(max_configs)).reset_index(drop=True)


def validate_single_gate_config(
    *,
    row: dict[str, Any],
    shots: int,
    seed: int,
    case_source: str = "pypower",
    default_submatrix_size: int = 4,
) -> dict[str, Any]:
    alpha = float(row["alpha"])
    degree = int(row.get("degree_for_validation", row.get("requested_degree")))
    case = str(row.get("case", "ieee14"))
    model = str(row.get("model", "ac_linearized"))
    subproblem_id = str(row.get("subproblem_id", f"{case}_{model}_{default_submatrix_size}x"))
    submatrix_size = _submatrix_size_from_row(row, default_submatrix_size)
    result = _failure_row(subproblem_id=subproblem_id, alpha=alpha, degree=degree)
    try:
        subproblem = extract_state_estimation_subproblem(
            case=case,
            model=model,
            submatrix_size=submatrix_size,
            seed=int(seed),
            case_source=str(case_source),
        )
        polynomial_update = _polynomial_update(
            subproblem.H_tilde,
            subproblem.r_tilde,
            alpha=alpha,
            degree=degree,
        )
        computation = solve_gate_level_state_estimation_problem(
            H_tilde=subproblem.H_tilde,
            r_tilde=subproblem.r_tilde,
            alpha=alpha,
            degree=degree,
            shots=int(shots),
            seed=int(seed),
            metadata=subproblem.metadata,
            transpile_qubit_limit=0,
            export_qasm=False,
        )
        H = np.asarray(computation.H_tilde, dtype=np.float64)
        r = np.asarray(computation.r_tilde, dtype=np.float64)
        qsvt = np.asarray(computation.qsvt_update, dtype=np.float64)
        ridge = np.asarray(computation.ridge_update, dtype=np.float64)
        best_scalar = best_scalar_for_residual(H, qsvt, r)
        gate_residual = _residual(H, qsvt, r)
        best_gate_residual = _residual(H, best_scalar * qsvt, r)
        ridge_residual = _residual(H, ridge, r)
        resource = getattr(computation, "resource_row", {})
        result.update(
            {
                "synthesized_phase_degree": int(computation.summary["synthesized_degree"]),
                "phase_count": int(computation.summary["phase_count"]),
                "query_count": int(computation.summary["qsvt_query_count"]),
                "polynomial_action_residual": float(row.get("polynomial_action_residual", np.nan)),
                "gate_level_residual": gate_residual,
                "best_scalar_gate_residual": best_gate_residual,
                "ridge_residual": ridge_residual,
                "state_error_gate_vs_poly": _relative_error(qsvt, polynomial_update),
                "state_error_gate_vs_ridge": _relative_error(qsvt, ridge),
                "success_probability": float(computation.summary["success_probability"]),
                "circuit_depth": int(computation.summary["circuit_depth"]),
                "two_qubit_gates": int(
                    computation.summary.get(
                        "two_qubit_gate_count",
                        resource.get("two_qubit_gate_count", 0),
                    )
                ),
                "run_status": "completed",
                "failure_reason_if_any": "",
            }
        )
        result["classification"] = classify_gate_validation(result)
    except Exception as exc:
        result.update(
            {
                "run_status": "failed",
                "failure_reason_if_any": f"{type(exc).__name__}: {exc}",
                "classification": "not_successful",
            }
        )
    return result


def write_gate_validation_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    selected: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    selected_path = output_dir / "selected_gate_validation_configs.csv"
    results_path = output_dir / "gate_validation_results.csv"
    summary_path = output_dir / "gate_vs_poly_vs_ridge_summary.csv"
    interpretation_path = output_dir / "gate_validation_interpretation.md"
    selected.to_csv(selected_path, index=False)
    results = pd.DataFrame(rows)
    for column in GATE_VALIDATION_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan
    results.to_csv(results_path, index=False)
    results[
        [
            "subproblem_id",
            "alpha",
            "degree",
            "polynomial_action_residual",
            "gate_level_residual",
            "best_scalar_gate_residual",
            "ridge_residual",
            "state_error_gate_vs_poly",
            "state_error_gate_vs_ridge",
            "classification",
            "run_status",
        ]
    ].to_csv(summary_path, index=False)
    interpretation_path.write_text(gate_validation_interpretation(results), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "selected_gate_validation_configs": str(selected_path),
            "gate_validation_results": str(results_path),
            "gate_vs_poly_vs_ridge_summary": str(summary_path),
            "gate_validation_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=BEST_CONFIG_GATE_CLAIM,
    )
    return {
        "manifest": manifest,
        "selected_gate_validation_configs": selected_path,
        "gate_validation_results": results_path,
        "gate_vs_poly_vs_ridge_summary": summary_path,
        "gate_validation_interpretation": interpretation_path,
    }


def classify_gate_validation(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "not_successful"
    ridge = float(row.get("ridge_residual", np.nan))
    gate = float(row.get("best_scalar_gate_residual", row.get("gate_level_residual", np.nan)))
    poly = float(row.get("polynomial_action_residual", np.nan))
    gate_poly_error = float(row.get("state_error_gate_vs_poly", np.inf))
    gate_ridge_error = float(row.get("state_error_gate_vs_ridge", np.inf))
    if np.isfinite(gate) and np.isfinite(ridge) and gate <= max(10.0 * ridge, 1.0e-8):
        return "ridge_approximating"
    if np.isfinite(poly) and np.isfinite(gate) and gate_poly_error <= 0.1:
        return "poly_matching_but_ridge_gap"
    if gate_poly_error > 0.25:
        return "gate_extraction_limited"
    if gate_ridge_error > 0.25:
        return "degree_limited"
    return "not_successful"


def gate_validation_interpretation(frame: pd.DataFrame) -> str:
    completed = frame[frame["run_status"] == "completed"].copy() if not frame.empty else frame
    if completed.empty:
        best_lines = ["- No successful gate-level validation rows."]
    else:
        best = completed.loc[completed["best_scalar_gate_residual"].astype(float).idxmin()]
        best_lines = [
            f"- Best alpha: {float(best['alpha']):.17g}",
            f"- Best degree: {int(best['degree'])}",
            f"- Best gate-level residual: {float(best['gate_level_residual']):.17g}",
            f"- Best scalar gate residual: {float(best['best_scalar_gate_residual']):.17g}",
            f"- State error vs polynomial: {float(best['state_error_gate_vs_poly']):.17g}",
            f"- State error vs Ridge: {float(best['state_error_gate_vs_ridge']):.17g}",
            f"- Classification: {best['classification']}",
        ]
    return "\n".join(
        [
            "# Best-Configuration Gate Validation",
            "",
            BEST_CONFIG_GATE_CLAIM,
            "",
            *best_lines,
            "",
            "Configurations are selected from polynomial-action residual evidence, "
            "then validated with the dense selected-subproblem gate-level solver.",
            "",
        ]
    )


def _polynomial_update(
    H_tilde: np.ndarray,
    r_tilde: np.ndarray,
    *,
    alpha: float,
    degree: int,
) -> np.ndarray:
    H = np.asarray(H_tilde, dtype=np.float64)
    r = np.asarray(r_tilde, dtype=np.float64)
    B = H.T
    beta = max(float(np.linalg.svd(B, compute_uv=False)[0]), np.finfo(float).eps)
    A = B / beta
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    positive = singular_values_A[singular_values_A > 1.0e-14]
    domain_min = min(max(1.0e-6, 0.9 * float(np.min(positive))), 0.95)
    alpha_norm = float(alpha) / beta**2
    approximation = fit_odd_regularized_polynomial(
        alpha=alpha_norm,
        block_encoding_normalization=1.0,
        degree=int(degree),
        domain_min=domain_min,
        domain_max=1.0,
        grid_size=max(2048, int(degree) + 2),
    )
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients, dtype=np.float64)
        / float(approximation.scale_factor)
    )
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    polynomial_operator = U @ (polynomial(singular_values)[:, None] * Vh)
    return rescale_bounded_target_to_original(
        polynomial_operator @ r,
        beta=beta,
        C=float(approximation.scale_factor),
    )


def _failure_row(*, subproblem_id: str, alpha: float, degree: int) -> dict[str, Any]:
    row = {column: np.nan for column in GATE_VALIDATION_COLUMNS}
    row.update(
        {
            "subproblem_id": subproblem_id,
            "alpha": float(alpha),
            "degree": int(degree),
            "run_status": "not_run",
            "failure_reason_if_any": "",
            "classification": "not_successful",
        }
    )
    return row


def _submatrix_size_from_row(row: dict[str, Any], default: int) -> int:
    matrix_shape = str(row.get("matrix_shape", ""))
    if "x" in matrix_shape:
        try:
            return int(matrix_shape.split("x", maxsplit=1)[0])
        except ValueError:
            pass
    return int(default)


def _residual(H: np.ndarray, update: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=np.float64) @ update - r))


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_values - reference_values)
        / max(float(np.linalg.norm(reference_values)), 1.0e-15)
    )
