from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.cross_case_codesigned_robustness import (
    BY_SUBPROBLEM_COLUMNS,
    CONTROL_MODE,
    RESULT_COLUMNS,
    _by_subproblem_frame,
    _mark_gate_validation_recommended,
    classify_case,
    evaluate_case_robustness,
)
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

IEEE118_ROBUSTNESS_CLAIM = (
    "IEEE118 selected-subproblem robustness study for the co-designed QSVT-safe solver on "
    "criteria-selected 4x4 AC-linearized weighted subproblems of the IEEE118 benchmark. "
    "Subproblems are chosen by numerical and metadata criteria (high-leverage, best-conditioned, "
    "metadata-mapped, residual-supported, a random pool, and a worst-conditioned control), never "
    "by post hoc QSVT performance, and the robust co-designed targets (weighted_support_ls, "
    "residual_aware) are tested over the safe degree window plus the boundary degree 47. This is "
    "a larger benchmark selected-subproblem test, not a full IEEE118-scale QSVT solver. "
    "Ridge/Tikhonov remains the reference filter; QSVT is an implementation pathway for the same "
    "regularized spectral filter. No full IEEE-scale QSVT solver, quantum speedup, quantum "
    "advantage, QSVT superiority over Ridge/Tikhonov, or hardware execution is claimed."
)

DEFAULT_CASE = "ieee118"
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

CASE_CLASSES = (
    "no_feasible_selected_blocks",
    "single_selected_block",
    "selected_subproblem_family",
    "inconclusive",
)


def run_qsvt_ieee118_selected_robustness(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "case": DEFAULT_CASE,
        "model": "ac_linearized",
        "case_source": "pypower",
        "submatrix_size": 4,
        "selection_modes": list(DEFAULT_SELECTION_MODES),
        "target_families": list(DEFAULT_TARGET_FAMILIES),
        "degrees": list(DEFAULT_DEGREES),
        "alphas": list(DEFAULT_ALPHAS),
        "grid_size": 4096,
        "seed": 123,
        "output_dir": "outputs/qsvt_ieee118_selected_robustness",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    rows = evaluate_case_robustness(
        case=str(resolved["case"]),
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
    rows = _mark_gate_validation_recommended(rows)
    artifacts = write_ieee118_robustness_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def classify_ieee118(by_subproblem: pd.DataFrame) -> str:
    """Classify the IEEE118 selected-subproblem result (matrix-level residual feasibility)."""

    return classify_case(by_subproblem)


def write_ieee118_robustness_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    all_frame = _frame_with_columns(rows, RESULT_COLUMNS)
    by_subproblem = _by_subproblem_frame(all_frame)
    candidates = all_frame[all_frame["gate_validation_recommended"] == True].copy()  # noqa: E712

    all_path = output_dir / "ieee118_all_configs.csv"
    by_subproblem_path = output_dir / "ieee118_by_subproblem_summary.csv"
    candidates_path = output_dir / "ieee118_gate_validation_candidates.csv"
    interpretation_path = output_dir / "ieee118_selected_robustness_interpretation.md"

    all_frame.to_csv(all_path, index=False)
    by_subproblem.to_csv(by_subproblem_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    interpretation_path.write_text(
        ieee118_robustness_interpretation(all_frame, by_subproblem), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "ieee118_all_configs": str(all_path),
            "ieee118_by_subproblem_summary": str(by_subproblem_path),
            "ieee118_gate_validation_candidates": str(candidates_path),
            "ieee118_selected_robustness_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=IEEE118_ROBUSTNESS_CLAIM,
    )
    return {
        "manifest": manifest,
        "ieee118_all_configs": all_path,
        "ieee118_by_subproblem_summary": by_subproblem_path,
        "ieee118_gate_validation_candidates": candidates_path,
        "ieee118_selected_robustness_interpretation": interpretation_path,
    }


def ieee118_robustness_interpretation(all_frame: pd.DataFrame, by_subproblem: pd.DataFrame) -> str:
    if all_frame.empty or by_subproblem.empty:
        return "\n".join(
            ["# IEEE118 Selected-Subproblem Robustness", "", IEEE118_ROBUSTNESS_CLAIM, ""]
        )
    classification = classify_ieee118(by_subproblem)
    numeric = all_frame.copy()
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    numeric["degree"] = pd.to_numeric(numeric["degree"], errors="coerce")

    non_control = by_subproblem[by_subproblem["is_control"] != True]  # noqa: E712
    feasible = non_control[non_control["any_residual_feasible"] == True]  # noqa: E712
    feasible_modes = sorted(set(feasible["selection_mode"].astype(str)))
    candidates = int((all_frame["gate_validation_recommended"] == True).sum())  # noqa: E712

    best_ratio = (
        float(numeric[numeric["residual_feasible"] == True]["residual_ratio_vs_no_update"].min())  # noqa: E712
        if (numeric["residual_feasible"] == True).any()  # noqa: E712
        else float("nan")
    )
    failed_modes = sorted(
        set(
            all_frame[
                (all_frame["residual_feasible"] != True)  # noqa: E712
                & (all_frame["selection_mode"].astype(str) != CONTROL_MODE)
            ]["selection_mode"].astype(str)
        )
        - set(feasible_modes)
    )
    degree_47 = numeric[numeric["degree"] == 47]
    degree_47_overshoots = bool(
        (degree_47["overshoot_detected"].astype(str).str.lower() == "true").any()
    )
    if classification in {"single_selected_block", "selected_subproblem_family"}:
        manuscript_role = "positive evidence (the selected-subproblem family extends to IEEE118)"
    elif classification == "no_feasible_selected_blocks":
        manuscript_role = (
            "boundary/negative evidence (no IEEE118 selected block is residual-feasible under the "
            "current criteria)"
        )
    else:
        manuscript_role = "inconclusive"

    return "\n".join(
        [
            "# IEEE118 Selected-Subproblem Robustness",
            "",
            IEEE118_ROBUSTNESS_CLAIM,
            "",
            "## Counts",
            f"- Configurations tested: {len(all_frame)}",
            f"- Residual-feasible configurations: {int((all_frame['residual_feasible'] == True).sum())}",  # noqa: E501, E712
            f"- Feasible non-control selected subproblems: {len(feasible)}",
            f"- Gate-validation candidates: {candidates}",
            "",
            f"## IEEE118 Robustness Classification: {classification}",
            "",
            "## Required Answers",
            f"1. Does the selected-subproblem solver transfer to IEEE118? "
            f"{'yes' if classification in {'single_selected_block', 'selected_subproblem_family'} else 'no'} "  # noqa: E501
            f"({classification}).",
            f"2. Which IEEE118 selection modes are residual-feasible? "
            f"{', '.join(feasible_modes) or 'none'} "
            "(criteria-based; the worst-conditioned control never counts as positive evidence).",
            f"3. Does degree 47 fail consistently? overshoot detected at degree 47: "
            f"{'yes' if degree_47_overshoots else 'no'}.",
            f"4. Which candidates should be gate-validated? {candidates} candidate(s) marked "
            "(best feasible non-control config per subproblem).",
            f"5. Manuscript role of IEEE118 evidence: {manuscript_role}.",
            "",
            f"- Best residual_ratio_vs_no_update: {best_ratio:.6g}.",
            "- Modes that fail (never residual-feasible, excluding control): "
            f"{', '.join(failed_modes) or 'none'}.",
            "",
        ]
    )


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


__all__ = [
    "BY_SUBPROBLEM_COLUMNS",
    "CASE_CLASSES",
    "RESULT_COLUMNS",
    "classify_ieee118",
    "ieee118_robustness_interpretation",
    "run_qsvt_ieee118_selected_robustness",
    "write_ieee118_robustness_outputs",
]
