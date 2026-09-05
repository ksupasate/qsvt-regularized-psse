from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.alpha_degree_refinement import (
    REFINEMENT_COLUMNS,
    evaluate_alpha_degree_grid,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    build_selected_subproblem_from_policy_row,
)
from robust_qsvt_se.qsvt.subproblem_sweep import _build_system
from robust_qsvt_se.utils.io import ensure_directory

REFINED_SELECTED_SOLVER_CLAIM = (
    "This refined solver evaluates gate-level QSVT only after a criteria-based "
    "subproblem selection policy. Ridge/Tikhonov remains the reference target."
)


def run_refined_selected_subproblem_solver(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "selection_file": "outputs/qsvt_subproblem_selection_policy/selected_subproblems.csv",
        "alphas": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        "degrees": [35, 51, 75, 101, 151, 201],
        "shots": 1000,
        "seed": 123,
        "output_dir": "outputs/qsvt_refined_selected_subproblem_solver",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    selected_frame = pd.read_csv(resolved["selection_file"])
    all_rows: list[dict[str, Any]] = []
    for _, selected_row in selected_frame.iterrows():
        policy_row = selected_row.to_dict()
        case = str(policy_row.get("case", "ieee14"))
        model = str(policy_row.get("model", "ac_linearized"))
        case_source = str(policy_row.get("case_source", "pypower"))
        system, matrix_source = _build_system(
            case=case,
            model=model,
            case_source=case_source,
            seed=int(resolved["seed"]),
        )
        subproblem = build_selected_subproblem_from_policy_row(
            system=system,
            matrix_source=matrix_source,
            row=policy_row,
        )
        all_rows.extend(
            evaluate_alpha_degree_grid(
                subproblem=subproblem,
                alphas=[float(value) for value in resolved["alphas"]],
                degrees=[int(value) for value in resolved["degrees"]],
                shots=int(resolved["shots"]),
                seed=int(resolved["seed"]),
                case=case,
                model=model,
                subproblem_id=str(policy_row["candidate_id"]),
                selection_mode=str(policy_row["selection_source"]),
            )
        )
    best_rows = best_configuration_rows(all_rows)
    artifacts = _write_outputs(output_dir, resolved, all_rows, best_rows)
    return {
        "output_dir": output_dir,
        "all_rows": all_rows,
        "best_rows": best_rows,
        "artifacts": artifacts,
    }


def best_configuration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=REFINEMENT_COLUMNS)
    completed = frame[frame["run_status"] == "completed"].copy()
    best_rows: list[dict[str, Any]] = []
    for subproblem_id, group in completed.groupby("subproblem_id", sort=True):
        idx = group["residual_qsvt_best_scalar"].astype(float).idxmin()
        row = frame.loc[idx].to_dict()
        row["subproblem_classification"] = classify_refined_subproblem(row)
        row["subproblem_id"] = subproblem_id
        best_rows.append(row)
    failed_ids = set(frame["subproblem_id"]) - {str(row["subproblem_id"]) for row in best_rows}
    for subproblem_id in sorted(str(value) for value in failed_ids):
        failure = frame[frame["subproblem_id"] == subproblem_id].iloc[0].to_dict()
        failure["subproblem_classification"] = "runtime_limited"
        best_rows.append(failure)
    return best_rows


def classify_refined_subproblem(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "runtime_limited"
    no_update = float(row["residual_no_update"])
    best = float(row["residual_qsvt_best_scalar"])
    ridge = float(row["residual_ridge"])
    if ridge > 1.0e-8 and best <= 10.0 * ridge:
        return "ridge_approximating"
    ratio = best / max(no_update, 1.0e-15)
    if ratio <= 0.1:
        return "strong_residual_reducing"
    if ratio <= 0.5:
        return "moderate_residual_reducing"
    return "not_successful"


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    all_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    all_path = output_dir / "refined_solver_all_runs.csv"
    best_path = output_dir / "best_configuration_per_subproblem.csv"
    summary_path = output_dir / "refined_solver_summary.md"
    failures_path = output_dir / "refined_solver_failures.csv"
    all_frame = pd.DataFrame(all_rows, columns=REFINEMENT_COLUMNS)
    best_frame = pd.DataFrame(best_rows)
    all_frame.to_csv(all_path, index=False)
    best_frame.to_csv(best_path, index=False)
    all_frame[all_frame["run_status"] != "completed"].to_csv(failures_path, index=False)
    summary_path.write_text(_summary_markdown(best_rows), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "refined_solver_all_runs": str(all_path),
            "best_configuration_per_subproblem": str(best_path),
            "refined_solver_summary": str(summary_path),
            "refined_solver_failures": str(failures_path),
        },
        input_config=resolved,
        claim_boundary=REFINED_SELECTED_SOLVER_CLAIM,
    )
    return {
        "manifest": manifest,
        "refined_solver_all_runs": all_path,
        "best_configuration_per_subproblem": best_path,
        "refined_solver_summary": summary_path,
        "refined_solver_failures": failures_path,
    }


def _summary_markdown(best_rows: list[dict[str, Any]]) -> str:
    classifications = pd.Series(
        [row.get("subproblem_classification", "unknown") for row in best_rows]
    ).value_counts()
    best = _best_overall(best_rows)
    best_lines = []
    if best:
        best_lines = [
            f"- Best subproblem: {best['subproblem_id']}",
            f"- Best alpha: {float(best['alpha']):.17g}",
            f"- Best requested degree: {int(best['requested_degree'])}",
            f"- Best scalar residual: {float(best['residual_qsvt_best_scalar']):.17g}",
            f"- Ridge residual: {float(best['residual_ridge']):.17g}",
            f"- Success probability: {float(best['success_probability']):.17g}",
        ]
    return "\n".join(
        [
            "# Refined Selected-Subproblem QSVT Solver",
            "",
            REFINED_SELECTED_SOLVER_CLAIM,
            "",
            *best_lines,
            f"- Classification counts: {classifications.to_dict()}",
            "",
        ]
    )


def _best_overall(best_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [
        row
        for row in best_rows
        if row.get("run_status") == "completed"
        and np.isfinite(float(row.get("residual_qsvt_best_scalar", np.nan)))
    ]
    if not completed:
        return None
    return min(completed, key=lambda row: float(row["residual_qsvt_best_scalar"]))
