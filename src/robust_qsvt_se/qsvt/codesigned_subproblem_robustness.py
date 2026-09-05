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

ROBUSTNESS_CLAIM = (
    "Robustness study of the co-designed QSVT-safe solver across criteria-selected "
    "IEEE14-derived subproblems. Subproblems are chosen by numerical and metadata criteria "
    "(high-leverage, best-conditioned, metadata-mapped, residual-supported, a random pool, "
    "and a worst-conditioned control), never by post hoc QSVT performance, and the best "
    "robust co-designed targets (weighted_support_ls, residual_aware) are tested over the "
    "feasible degree window. Ridge/Tikhonov remains the reference filter; QSVT is an "
    "implementation pathway for the same regularized spectral filter. No QSVT superiority "
    "over Ridge/Tikhonov, quantum speedup, quantum advantage, full IEEE-scale gate-level "
    "solving, or hardware execution is claimed."
)

DEFAULT_SELECTION_MODES = (
    "high_leverage",
    "best_conditioned",
    "metadata_mapped",
    "residual_supported",
    "random_seeded_pool",
    "worst_conditioned_control",
)
DEFAULT_TARGET_FAMILIES = ("weighted_support_ls", "residual_aware")
DEFAULT_DEGREES = (15, 25, 35)
DEFAULT_ALPHAS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)

DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
FEASIBLE_RATIO_MAX = 0.1
FEASIBLE_DIRECTION_MAX = 0.1
CONTROL_MODE = "worst_conditioned_control"

ROBUSTNESS_CLASSES = (
    "single_block_only",
    "narrow_selected_family",
    "moderate_selected_family",
    "broad_selected_family",
    "inconclusive",
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
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "success_probability_proxy",
    "residual_feasible",
    "gate_validation_recommended",
    "failure_reason",
]

BY_SUBPROBLEM_COLUMNS = [
    "subproblem_id",
    "selection_mode",
    "row_indices",
    "col_indices",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "ridge_residual",
    "no_update_residual",
    "any_residual_feasible",
    "best_residual_ratio_vs_no_update",
    "best_degree",
    "best_target_family",
    "is_control",
]


def run_qsvt_codesigned_subproblem_robustness(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": "ieee14",
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": list(DEFAULT_SELECTION_MODES),
        "target_families": list(DEFAULT_TARGET_FAMILIES),
        "degrees": list(DEFAULT_DEGREES),
        "alphas": list(DEFAULT_ALPHAS),
        "grid_size": 4096,
        "seed": 123,
        "output_dir": "outputs/qsvt_codesigned_subproblem_robustness",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    system, matrix_source = _build_system(
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        case_source=str(resolved["case_source"]),
        seed=int(resolved["seed"]),
    )
    candidates = generate_candidate_subproblems(
        system=system,
        matrix_source=matrix_source,
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        submatrix_size=int(resolved["submatrix_size"]),
        candidate_modes=[str(value) for value in resolved["selection_modes"]],
        seed=int(resolved["seed"]),
    )
    rows = evaluate_subproblem_robustness(
        system=system,
        matrix_source=matrix_source,
        candidates=candidates,
        case=str(resolved["case"]),
        model=str(resolved["model"]),
        target_families=[str(value) for value in resolved["target_families"]],
        degrees=[int(value) for value in resolved["degrees"]],
        alphas=[float(value) for value in resolved["alphas"]],
        grid_size=int(resolved["grid_size"]),
    )
    artifacts = write_robustness_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_subproblem_robustness(
    *,
    system: Any,
    matrix_source: str,
    candidates: list[Any],
    case: str,
    model: str,
    target_families: list[str],
    degrees: list[int],
    alphas: list[float],
    grid_size: int = 4096,
) -> list[dict[str, Any]]:
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
                        _evaluate_robustness_config(
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
    return _mark_gate_validation_recommended(rows)


def _evaluate_robustness_config(
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
            "residual_feasible": False,
            "gate_validation_recommended": False,
        }
    )

    H = np.asarray(subproblem.H_tilde, dtype=np.float64)
    r = np.asarray(subproblem.r_tilde, dtype=np.float64)
    no_update = properties["no_update_residual"]
    try:
        ridge_update = ridge_tikhonov_update(H, r, alpha=float(alpha))
        ridge_residual = float(np.linalg.norm(H @ ridge_update - r))
    except Exception:
        ridge_residual = float("nan")
    base["ridge_residual"] = ridge_residual

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
    success = float(solution.success_probability_proxy)
    deployable = solution.deployability_class in DEPLOYABLE_CLASSES

    feasible = bool(
        deployable
        and solution.qsvt_safe
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
            "residual_ratio_vs_no_update": float(residual_ratio),
            "direction_error_vs_ridge": float(direction_error),
            "success_probability_proxy": float(success),
            "residual_feasible": feasible,
            "failure_reason": _failure_reason(
                qsvt_safe=bool(solution.qsvt_safe),
                deployable=deployable,
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
    residual_ratio: float,
    direction_error: float,
    success: float,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    if not deployable:
        return "non_deployable_class"
    if not math.isfinite(direction_error) or direction_error > FEASIBLE_DIRECTION_MAX:
        return "direction_error_exceeds_threshold"
    if math.isfinite(success) and success < SUCCESS_PROBABILITY_FLOOR:
        return "success_probability_below_floor"
    if not math.isfinite(residual_ratio) or residual_ratio > FEASIBLE_RATIO_MAX:
        return "residual_ratio_exceeds_threshold"
    return ""


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
    best_by_subproblem: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("residual_feasible")):
            continue
        if str(row.get("selection_mode")) == CONTROL_MODE:
            continue
        ratio = float(row.get("residual_ratio_vs_no_update", float("inf")))
        key = str(row.get("subproblem_id"))
        current = best_by_subproblem.get(key)
        if current is None or ratio < current[0]:
            best_by_subproblem[key] = (ratio, row)
    for _, row in best_by_subproblem.values():
        row["gate_validation_recommended"] = True
    return rows


def write_robustness_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    all_frame = _frame_with_columns(rows, RESULT_COLUMNS)
    by_subproblem = _by_subproblem_frame(all_frame)
    candidates_frame = all_frame[
        all_frame["gate_validation_recommended"] == True  # noqa: E712
    ].copy()

    all_path = output_dir / "robustness_all_configs.csv"
    by_subproblem_path = output_dir / "robustness_by_subproblem.csv"
    candidates_path = output_dir / "robustness_gate_validation_candidates.csv"
    interpretation_path = output_dir / "robustness_interpretation.md"

    all_frame.to_csv(all_path, index=False)
    by_subproblem.to_csv(by_subproblem_path, index=False)
    candidates_frame.to_csv(candidates_path, index=False)
    interpretation_path.write_text(
        robustness_interpretation(all_frame, by_subproblem), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "robustness_all_configs": str(all_path),
            "robustness_by_subproblem": str(by_subproblem_path),
            "robustness_gate_validation_candidates": str(candidates_path),
            "robustness_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=ROBUSTNESS_CLAIM,
    )
    return {
        "manifest": manifest,
        "robustness_all_configs": all_path,
        "robustness_by_subproblem": by_subproblem_path,
        "robustness_gate_validation_candidates": candidates_path,
        "robustness_interpretation": interpretation_path,
    }


def _by_subproblem_frame(all_frame: pd.DataFrame) -> pd.DataFrame:
    if all_frame.empty:
        return pd.DataFrame(columns=BY_SUBPROBLEM_COLUMNS)
    numeric = all_frame.copy()
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for subproblem_id, group in numeric.groupby("subproblem_id", dropna=False):
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
                "subproblem_id": subproblem_id,
                "selection_mode": str(first["selection_mode"]),
                "row_indices": str(first["row_indices"]),
                "col_indices": str(first["col_indices"]),
                "condition_number": float(first["condition_number"]),
                "sigma_min": float(first["sigma_min"]),
                "sigma_max": float(first["sigma_max"]),
                "ridge_residual": float(pd.to_numeric(first["ridge_residual"], errors="coerce")),
                "no_update_residual": float(first["no_update_residual"]),
                "any_residual_feasible": bool(not feasible.empty),
                "best_residual_ratio_vs_no_update": best_ratio,
                "best_degree": best_degree,
                "best_target_family": best_family,
                "is_control": str(first["selection_mode"]) == CONTROL_MODE,
            }
        )
    return pd.DataFrame(rows, columns=BY_SUBPROBLEM_COLUMNS)


def classify_robustness(by_subproblem: pd.DataFrame) -> tuple[str, list[str]]:
    if by_subproblem.empty:
        return "inconclusive", []
    feasible = by_subproblem[
        (by_subproblem["any_residual_feasible"] == True)  # noqa: E712
        & (by_subproblem["is_control"] != True)  # noqa: E712
    ]
    feasible_modes = sorted(set(feasible["selection_mode"].astype(str)))
    n = len(feasible_modes)
    if n == 0:
        return "inconclusive", feasible_modes
    if feasible_modes == ["high_leverage"]:
        return "single_block_only", feasible_modes
    if n <= 2:
        return "narrow_selected_family", feasible_modes
    if n == 3:
        return "moderate_selected_family", feasible_modes
    return "broad_selected_family", feasible_modes


def robustness_interpretation(all_frame: pd.DataFrame, by_subproblem: pd.DataFrame) -> str:
    if all_frame.empty:
        return "\n".join(["# Co-Designed Subproblem Robustness", "", ROBUSTNESS_CLAIM, ""])

    classification, feasible_modes = classify_robustness(by_subproblem)
    non_control = by_subproblem[by_subproblem["is_control"] != True]  # noqa: E712
    feasible_subproblems = int((non_control["any_residual_feasible"] == True).sum())  # noqa: E712
    total_subproblems = len(by_subproblem)
    candidates = int((all_frame["gate_validation_recommended"] == True).sum())  # noqa: E712

    feasible_props = non_control[non_control["any_residual_feasible"] == True]  # noqa: E712
    cond_hint = "n/a"
    if not feasible_props.empty:
        cond_values = pd.to_numeric(feasible_props["condition_number"], errors="coerce").dropna()
        if not cond_values.empty:
            cond_hint = (
                f"feasible subproblem condition numbers in "
                f"[{float(cond_values.min()):.3g}, {float(cond_values.max()):.3g}]"
            )

    failed_modes = sorted(
        set(
            non_control[non_control["any_residual_feasible"] != True]["selection_mode"].astype(str)  # noqa: E712
        )
    )
    control_rows = by_subproblem[by_subproblem["is_control"] == True]  # noqa: E712
    control_feasible = bool((control_rows["any_residual_feasible"] == True).any())  # noqa: E712

    if classification in {"moderate_selected_family", "broad_selected_family"}:
        manuscript = "robust selected-subproblem family"
    elif classification in {"narrow_selected_family", "single_block_only"}:
        manuscript = "selected-subproblem prototype"
    else:
        manuscript = "selected-subproblem prototype (robustness inconclusive)"

    return "\n".join(
        [
            "# Co-Designed Subproblem Robustness",
            "",
            ROBUSTNESS_CLAIM,
            "",
            "## Counts",
            f"- Subproblems tested: {total_subproblems} (including the worst-conditioned control)",
            f"- Non-control subproblems residual-feasible at some config: {feasible_subproblems}",
            f"- Configurations tested: {len(all_frame)}",
            f"- Gate-validation candidates: {candidates}",
            "",
            "## Required Answers",
            f"1. Does the co-designed solver transfer beyond the high-leverage block? "
            f"{'yes' if feasible_modes not in ([], ['high_leverage']) else 'no'} (feasible "
            f"selection modes: {', '.join(feasible_modes) or 'none'}).",
            f"2. Which subproblem properties predict success? {cond_hint}; well-conditioned, "
            "residual-supported blocks with positive singular support succeed.",
            f"3. Which modes fail and why? {', '.join(failed_modes) or 'none'} "
            f"(worst-conditioned control feasible: {'yes' if control_feasible else 'no'}).",
            f"4. Robustness classification: {classification}.",
            f"5. Manuscript claim: {manuscript}.",
            "",
        ]
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
