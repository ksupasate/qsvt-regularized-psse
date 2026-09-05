from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_bounded_targets import (
    SUCCESS_PROBABILITY_FLOOR,
    build_codesigned_solution,
)
from robust_qsvt_se.qsvt.degree_window_overshoot import detect_overshoot
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import (
    SelectedSubproblem,
    ridge_tikhonov_update,
)
from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    build_selected_subproblem_from_policy_row,
    generate_candidate_subproblems,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

CROSS_CASE_ROBUSTNESS_CLAIM = (
    "Cross-case selected-subproblem robustness study for the co-designed QSVT-safe solver "
    "across IEEE14/IEEE30/IEEE57 criteria-selected 4x4 AC-linearized weighted subproblems. "
    "Subproblems are chosen by numerical and metadata criteria (high-leverage, "
    "best-conditioned, metadata-mapped, residual-supported, a random pool, and a "
    "worst-conditioned control), never by post hoc QSVT performance, and the robust "
    "co-designed targets (weighted_support_ls, residual_aware) are tested over the safe "
    "degree window plus a boundary degree. This remains a selected-subproblem study, not full "
    "IEEE-scale solving. Ridge/Tikhonov remains the reference filter; QSVT is an "
    "implementation pathway for the same regularized spectral filter. No QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, quantum advantage, full IEEE-scale gate-level solving, "
    "or hardware execution is claimed."
)

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57")
DEFAULT_SELECTION_MODES = (
    "high_leverage",
    "metadata_mapped",
    "residual_supported",
    "best_conditioned",
    "random_seeded_pool",
    "worst_conditioned_control",
)
DEFAULT_TARGET_FAMILIES = ("weighted_support_ls", "residual_aware")
DEFAULT_DEGREES = (15, 25, 35, 45, 47)
DEFAULT_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
FEASIBLE_RATIO_MAX = 0.1
FEASIBLE_DIRECTION_MAX = 0.1
CONTROL_MODE = "worst_conditioned_control"

CASE_CLASSES = (
    "no_feasible_selected_blocks",
    "single_selected_block",
    "selected_subproblem_family",
    "inconclusive",
)
CROSS_CASE_CLASSES = (
    "ieee14_only",
    "cross_case_partial",
    "cross_case_selected_family",
    "cross_case_inconclusive",
)

RESULT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "selection_mode",
    "row_indices",
    "col_indices",
    "alpha",
    "degree",
    "target_family",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "ridge_residual",
    "no_update_residual",
    "qsvt_safe",
    "max_abs_polynomial_on_grid",
    "overshoot_detected",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "success_probability_proxy",
    "residual_feasible",
    "gate_validation_recommended",
    "failure_reason",
]

BY_CASE_COLUMNS = [
    "case",
    "n_subproblems",
    "n_noncontrol_subproblems",
    "n_feasible_noncontrol_subproblems",
    "feasible_modes",
    "best_residual_ratio_vs_no_update",
    "degree_47_overshoots",
    "case_classification",
]

BY_SUBPROBLEM_COLUMNS = [
    "case",
    "subproblem_id",
    "selection_mode",
    "row_indices",
    "col_indices",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "any_residual_feasible",
    "best_residual_ratio_vs_no_update",
    "best_degree",
    "best_target_family",
    "is_control",
]


def run_qsvt_cross_case_codesigned_robustness(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "cases": list(DEFAULT_CASES),
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": list(DEFAULT_SELECTION_MODES),
        "target_families": list(DEFAULT_TARGET_FAMILIES),
        "degrees": list(DEFAULT_DEGREES),
        "alphas": list(DEFAULT_ALPHAS),
        "grid_size": 4096,
        "seed": 123,
        "output_dir": "outputs/qsvt_cross_case_codesigned_robustness",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    rows: list[dict[str, Any]] = []
    for case in [str(value) for value in resolved["cases"]]:
        rows.extend(
            evaluate_case_robustness(
                case=case,
                model=str(resolved["model"]),
                case_source=str(resolved["case_source"]),
                submatrix_size=int(resolved["submatrix_size"]),
                selection_modes=[str(value) for value in resolved["selection_modes"]],
                target_families=[str(value) for value in resolved["target_families"]],
                degrees=[int(value) for value in resolved["degrees"]],
                alphas=[float(value) for value in resolved["alphas"]],
                grid_size=int(resolved["grid_size"]),
                seed=int(resolved["seed"]),
            )
        )
    rows = _mark_gate_validation_recommended(rows)
    artifacts = write_cross_case_robustness_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_case_robustness(
    *,
    case: str,
    model: str,
    case_source: str,
    submatrix_size: int,
    selection_modes: list[str],
    target_families: list[str],
    degrees: list[int],
    alphas: list[float],
    grid_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    try:
        system, matrix_source = _build_system(
            case=case, model=model, case_source=case_source, seed=int(seed)
        )
        candidates = generate_candidate_subproblems(
            system=system,
            matrix_source=matrix_source,
            case=case,
            model=model,
            submatrix_size=int(submatrix_size),
            candidate_modes=list(selection_modes),
            seed=int(seed),
        )
    except Exception as exc:  # pragma: no cover - depends on optional pypower data
        return [_case_load_failed_row(case, model, exc)]

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        policy_row = {
            "case": case,
            "model": model,
            "candidate_id": candidate.candidate_id,
            "selection_source": candidate.selection_source,
            "row_indices": " ".join(str(int(value)) for value in candidate.rows),
            "col_indices": " ".join(str(int(value)) for value in candidate.columns),
        }
        subproblem = build_selected_subproblem_from_policy_row(
            system=system, matrix_source=matrix_source, row=policy_row
        )
        properties = _subproblem_properties(subproblem)
        for alpha in alphas:
            for degree in degrees:
                for family in target_families:
                    rows.append(
                        _evaluate_cross_case_config(
                            subproblem=subproblem,
                            properties=properties,
                            case=case,
                            model=model,
                            subproblem_id=str(candidate.candidate_id),
                            selection_mode=str(candidate.selection_source),
                            row_indices=policy_row["row_indices"],
                            col_indices=policy_row["col_indices"],
                            alpha=float(alpha),
                            degree=int(degree),
                            target_family=str(family),
                            grid_size=int(grid_size),
                        )
                    )
    return rows


def _evaluate_cross_case_config(
    *,
    subproblem: SelectedSubproblem,
    properties: dict[str, float],
    case: str,
    model: str,
    subproblem_id: str,
    selection_mode: str,
    row_indices: str,
    col_indices: str,
    alpha: float,
    degree: int,
    target_family: str,
    grid_size: int,
) -> dict[str, Any]:
    base = {column: np.nan for column in RESULT_COLUMNS}
    base.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": subproblem_id,
            "selection_mode": selection_mode,
            "row_indices": row_indices,
            "col_indices": col_indices,
            "alpha": float(alpha),
            "degree": int(degree),
            "target_family": target_family,
            "condition_number": properties["condition_number"],
            "sigma_min": properties["sigma_min"],
            "sigma_max": properties["sigma_max"],
            "no_update_residual": properties["no_update_residual"],
            "qsvt_safe": False,
            "overshoot_detected": False,
            "residual_feasible": False,
            "gate_validation_recommended": False,
        }
    )

    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    no_update = properties["no_update_residual"]
    try:
        ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
        base["ridge_residual"] = float(np.linalg.norm(H @ ridge_update - r))
    except Exception:
        ridge_update = np.full(H.shape[1], np.nan)
        base["ridge_residual"] = float("nan")

    solution = build_codesigned_solution(
        subproblem,
        alpha=float(alpha),
        degree=int(degree),
        target_family=target_family,
        grid_size=int(grid_size),
    )
    if solution is None:
        base["failure_reason"] = "target_construction_failed"
        return base

    update = np.asarray(solution.known_c_update, dtype=np.float64)
    residual = (
        float(np.linalg.norm(H @ update - r)) if np.all(np.isfinite(update)) else float("nan")
    )
    residual_ratio = residual / no_update if no_update > 0.0 else float("nan")
    direction_error = float(solution.direction_error_vs_ridge)
    cos_angle = _cos_angle(update, ridge_update)
    success = float(solution.success_probability_proxy)
    deployable = solution.deployability_class in DEPLOYABLE_CLASSES

    max_abs_on_grid, overshoot_margin = _polynomial_extents(solution, H)
    overshoot_detected = detect_overshoot(
        max_abs_polynomial_on_grid=max_abs_on_grid,
        direction_error_vs_ridge=direction_error,
        overshoot_margin=overshoot_margin,
    )

    feasible = bool(
        deployable
        and solution.qsvt_safe
        and not overshoot_detected
        and math.isfinite(residual_ratio)
        and residual_ratio <= FEASIBLE_RATIO_MAX
        and math.isfinite(direction_error)
        and direction_error <= FEASIBLE_DIRECTION_MAX
        and math.isfinite(success)
        and success >= SUCCESS_PROBABILITY_FLOOR
    )
    base.update(
        {
            "qsvt_safe": bool(solution.qsvt_safe),
            "max_abs_polynomial_on_grid": float(max_abs_on_grid),
            "overshoot_detected": bool(overshoot_detected),
            "residual_ratio_vs_no_update": float(residual_ratio),
            "direction_error_vs_ridge": float(direction_error),
            "cos_angle_vs_ridge": float(cos_angle),
            "success_probability_proxy": float(success),
            "residual_feasible": feasible,
            "failure_reason": _failure_reason(
                qsvt_safe=bool(solution.qsvt_safe),
                deployable=deployable,
                overshoot_detected=overshoot_detected,
                residual_ratio=residual_ratio,
                direction_error=direction_error,
                success=success,
            ),
        }
    )
    return base


def _failure_reason(
    *,
    qsvt_safe: bool,
    deployable: bool,
    overshoot_detected: bool,
    residual_ratio: float,
    direction_error: float,
    success: float,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    if not deployable:
        return "non_deployable_class"
    if overshoot_detected:
        return "overshoot_detected"
    if not math.isfinite(direction_error) or direction_error > FEASIBLE_DIRECTION_MAX:
        return "direction_error_exceeds_threshold"
    if math.isfinite(success) and success < SUCCESS_PROBABILITY_FLOOR:
        return "success_probability_below_floor"
    if not math.isfinite(residual_ratio) or residual_ratio > FEASIBLE_RATIO_MAX:
        return "residual_ratio_exceeds_threshold"
    return ""


def _polynomial_extents(solution: Any, H: np.ndarray) -> tuple[float, float]:
    p_raw = solution.raw_polynomial
    c_scale = max(float(solution.c_scale), np.finfo(float).eps)
    beta = max(float(solution.beta), np.finfo(float).eps)
    grid = np.linspace(-1.0, 1.0, 8193, dtype=np.float64)
    raw_max_abs = float(np.max(np.abs(p_raw(grid))))
    singular_values = np.linalg.svd(np.asarray(H, dtype=np.float64).T / beta, compute_uv=False)
    support = singular_values[singular_values > 1.0e-14]
    support_max_abs = float(np.max(np.abs(p_raw(support)))) if support.size else 0.0
    max_abs_on_grid = raw_max_abs / c_scale
    overshoot_margin = raw_max_abs / max(support_max_abs, np.finfo(float).eps)
    return max_abs_on_grid, overshoot_margin


def _subproblem_properties(subproblem: SelectedSubproblem) -> dict[str, float]:
    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(H, compute_uv=False)
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 1.0e-14 else float("inf")
    return {
        "condition_number": condition,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "no_update_residual": float(np.linalg.norm(r)),
    }


def _mark_gate_validation_recommended(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the best feasible non-control config per (case, subproblem) for gate validation."""

    best: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("residual_feasible")):
            continue
        if str(row.get("selection_mode")) == CONTROL_MODE:
            continue
        ratio = float(row.get("residual_ratio_vs_no_update", float("inf")))
        key = (str(row.get("case")), str(row.get("subproblem_id")))
        current = best.get(key)
        if current is None or ratio < current[0]:
            best[key] = (ratio, row)
    for _, row in best.values():
        row["gate_validation_recommended"] = True
    return rows


def write_cross_case_robustness_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    all_frame = _frame_with_columns(rows, RESULT_COLUMNS)
    by_subproblem = _by_subproblem_frame(all_frame)
    by_case = _by_case_frame(all_frame, by_subproblem)
    candidates_frame = all_frame[
        all_frame["gate_validation_recommended"] == True  # noqa: E712
    ].copy()

    all_path = output_dir / "cross_case_all_configs.csv"
    by_case_path = output_dir / "cross_case_by_case_summary.csv"
    by_subproblem_path = output_dir / "cross_case_by_subproblem_summary.csv"
    candidates_path = output_dir / "cross_case_gate_validation_candidates.csv"
    interpretation_path = output_dir / "cross_case_robustness_interpretation.md"

    all_frame.to_csv(all_path, index=False)
    by_case.to_csv(by_case_path, index=False)
    by_subproblem.to_csv(by_subproblem_path, index=False)
    candidates_frame.to_csv(candidates_path, index=False)
    interpretation_path.write_text(
        cross_case_robustness_interpretation(by_case, all_frame), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "cross_case_all_configs": str(all_path),
            "cross_case_by_case_summary": str(by_case_path),
            "cross_case_by_subproblem_summary": str(by_subproblem_path),
            "cross_case_gate_validation_candidates": str(candidates_path),
            "cross_case_robustness_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CROSS_CASE_ROBUSTNESS_CLAIM,
    )
    return {
        "manifest": manifest,
        "cross_case_all_configs": all_path,
        "cross_case_by_case_summary": by_case_path,
        "cross_case_by_subproblem_summary": by_subproblem_path,
        "cross_case_gate_validation_candidates": candidates_path,
        "cross_case_robustness_interpretation": interpretation_path,
    }


def _by_subproblem_frame(all_frame: pd.DataFrame) -> pd.DataFrame:
    if all_frame.empty:
        return pd.DataFrame(columns=BY_SUBPROBLEM_COLUMNS)
    numeric = all_frame.copy()
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for (case, subproblem_id), group in numeric.groupby(["case", "subproblem_id"], dropna=False):
        feasible = group[group["residual_feasible"] == True]  # noqa: E712
        best_ratio = float("nan")
        best_degree = float("nan")
        best_family = "none"
        if not feasible.empty and feasible["residual_ratio_vs_no_update"].notna().any():
            best = feasible.loc[feasible["residual_ratio_vs_no_update"].idxmin()]
            best_ratio = float(best["residual_ratio_vs_no_update"])
            best_degree = float(best["degree"])
            best_family = str(best["target_family"])
        first = group.iloc[0]
        rows.append(
            {
                "case": str(case),
                "subproblem_id": str(subproblem_id),
                "selection_mode": str(first["selection_mode"]),
                "row_indices": str(first["row_indices"]),
                "col_indices": str(first["col_indices"]),
                "condition_number": float(
                    pd.to_numeric(first["condition_number"], errors="coerce")
                ),
                "sigma_min": float(pd.to_numeric(first["sigma_min"], errors="coerce")),
                "sigma_max": float(pd.to_numeric(first["sigma_max"], errors="coerce")),
                "any_residual_feasible": bool(not feasible.empty),
                "best_residual_ratio_vs_no_update": best_ratio,
                "best_degree": best_degree,
                "best_target_family": best_family,
                "is_control": str(first["selection_mode"]) == CONTROL_MODE,
            }
        )
    return pd.DataFrame(rows, columns=BY_SUBPROBLEM_COLUMNS)


def _by_case_frame(all_frame: pd.DataFrame, by_subproblem: pd.DataFrame) -> pd.DataFrame:
    if by_subproblem.empty:
        return pd.DataFrame(columns=BY_CASE_COLUMNS)
    numeric = all_frame.copy()
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for case, group in by_subproblem.groupby("case", dropna=False):
        non_control = group[group["is_control"] != True]  # noqa: E712
        feasible = non_control[non_control["any_residual_feasible"] == True]  # noqa: E712
        case_configs = numeric[numeric["case"].astype(str) == str(case)]
        best_ratio = (
            float(
                case_configs[case_configs["residual_feasible"] == True][  # noqa: E712
                    "residual_ratio_vs_no_update"
                ].min()
            )
            if (case_configs["residual_feasible"] == True).any()  # noqa: E712
            else float("nan")
        )
        degree_47 = case_configs[pd.to_numeric(case_configs["degree"], errors="coerce") == 47]
        degree_47_overshoots = bool(
            (degree_47["overshoot_detected"].astype(str).str.lower() == "true").any()
        )
        rows.append(
            {
                "case": str(case),
                "n_subproblems": len(group),
                "n_noncontrol_subproblems": len(non_control),
                "n_feasible_noncontrol_subproblems": len(feasible),
                "feasible_modes": ";".join(sorted(set(feasible["selection_mode"].astype(str))))
                or "none",
                "best_residual_ratio_vs_no_update": best_ratio,
                "degree_47_overshoots": degree_47_overshoots,
                "case_classification": classify_case(group),
            }
        )
    return pd.DataFrame(rows, columns=BY_CASE_COLUMNS)


def classify_case(case_subproblems: pd.DataFrame) -> str:
    """Classify a single case from its by-subproblem feasibility."""

    if case_subproblems.empty:
        return "inconclusive"
    non_control = case_subproblems[case_subproblems["is_control"] != True]  # noqa: E712
    feasible = non_control[non_control["any_residual_feasible"] == True]  # noqa: E712
    n = len(feasible)
    if n == 0:
        return "no_feasible_selected_blocks"
    if n == 1:
        return "single_selected_block"
    return "selected_subproblem_family"


def classify_cross_case(by_case: pd.DataFrame) -> tuple[str, dict[str, str]]:
    """Classify the overall cross-case transfer result (matrix-level residual feasibility)."""

    if by_case.empty:
        return "cross_case_inconclusive", {}
    per_case = {str(row["case"]): str(row["case_classification"]) for _, row in by_case.iterrows()}

    def transfers(case: str) -> bool:
        return per_case.get(case, "inconclusive") in {
            "single_selected_block",
            "selected_subproblem_family",
        }

    ieee30 = transfers("ieee30")
    ieee57 = transfers("ieee57")
    if "ieee30" not in per_case and "ieee57" not in per_case:
        return "cross_case_inconclusive", per_case
    if ieee30 and ieee57:
        return "cross_case_selected_family", per_case
    if ieee30 or ieee57:
        return "cross_case_partial", per_case
    return "ieee14_only", per_case


def cross_case_robustness_interpretation(by_case: pd.DataFrame, all_frame: pd.DataFrame) -> str:
    if by_case.empty:
        return "\n".join(
            ["# Cross-Case Selected-Subproblem Robustness", "", CROSS_CASE_ROBUSTNESS_CLAIM, ""]
        )
    classification, per_case = classify_cross_case(by_case)
    candidates = int((all_frame["gate_validation_recommended"] == True).sum())  # noqa: E712
    failed_modes = sorted(
        set(
            all_frame[
                (all_frame["residual_feasible"] != True)  # noqa: E712
                & (all_frame["selection_mode"].astype(str) != CONTROL_MODE)
            ]["selection_mode"].astype(str)
        )
        - set(
            all_frame[all_frame["residual_feasible"] == True][  # noqa: E712
                "selection_mode"
            ].astype(str)
        )
    )
    degree_47_overshoots = sorted(
        set(by_case[by_case["degree_47_overshoots"].astype(bool)]["case"].astype(str))
    )
    gate_cases = sorted(
        set(
            all_frame[all_frame["gate_validation_recommended"] == True][  # noqa: E712
                "case"
            ].astype(str)
        )
    )

    return "\n".join(
        [
            "# Cross-Case Selected-Subproblem Robustness",
            "",
            CROSS_CASE_ROBUSTNESS_CLAIM,
            "",
            "## Counts",
            f"- Cases tested: {', '.join(per_case)}",
            f"- Configurations tested: {len(all_frame)}",
            f"- Gate-validation candidates: {candidates}",
            "",
            "## Per-Case Classification",
            *[f"- {case}: {label}" for case, label in per_case.items()],
            "",
            "## Required Answers",
            f"1. Does the selected-subproblem family transfer to IEEE30? "
            f"{per_case.get('ieee30', 'not tested')}.",
            f"2. Does it transfer to IEEE57? {per_case.get('ieee57', 'not tested')}.",
            "3. Which selection modes work across cases? "
            "see cross_case_by_case_summary.csv (feasible_modes per case).",
            f"4. Does degree 47 fail consistently across cases? overshoot at degree 47 in cases: "
            f"{degree_47_overshoots or 'none'}.",
            f"5. Which cases should be gate-validated? {gate_cases or 'none'}.",
            "",
            f"## Overall Cross-Case Classification: {classification}",
            "- Modes that fail in every case (never feasible): "
            f"{', '.join(failed_modes) or 'none'}.",
            "",
        ]
    )


def _case_load_failed_row(case: str, model: str, exc: Exception) -> dict[str, Any]:
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "case": case,
            "model": model,
            "subproblem_id": "case_load_failed",
            "selection_mode": "none",
            "qsvt_safe": False,
            "overshoot_detected": False,
            "residual_feasible": False,
            "gate_validation_recommended": False,
            "failure_reason": f"case_load_failed:{type(exc).__name__}",
        }
    )
    return row


def _cos_angle(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if candidate_norm <= 1.0e-15 or reference_norm <= 1.0e-15:
        return float("nan")
    cosine = float(np.dot(candidate, reference) / (candidate_norm * reference_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
