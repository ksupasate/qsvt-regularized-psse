from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.codesigned_bounded_targets import run_qsvt_codesigned_bounded_target_study
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

CODESIGNED_SEARCH_CLAIM = (
    "Residual-feasible search over co-designed QSVT-safe bounded targets on a "
    "selected IEEE14-derived subproblem. Deployable feasibility uses only QSVT-safe, "
    "general or instance-aware targets whose amplitude/known-C recovered update meets "
    "the residual and Ridge-direction thresholds at a non-vanishing success amplitude. "
    "Diagnostic feasibility uses the non-deployable best-scalar lower bound. "
    "Ridge/Tikhonov remains the reference filter; no QSVT superiority over "
    "Ridge/Tikhonov, quantum speedup, quantum advantage, full IEEE-scale gate-level "
    "solving, or hardware execution is claimed."
)

RESIDUAL_RATIO_FEASIBLE_MAX = 0.1
DIRECTION_ERROR_FEASIBLE_MAX = 0.1
SUCCESS_PROBABILITY_MIN = 1.0e-3
DEPLOYABLE_CLASSES = ("general_qsvt_safe", "instance_aware_qsvt_safe")
GATE_VALIDATION_DEGREE_MAX = 101

RESULT_COLUMNS = [
    "case",
    "model",
    "subproblem_id",
    "alpha",
    "degree",
    "target_family",
    "weighting_scheme",
    "deployability_class",
    "qsvt_safe",
    "success_probability_proxy",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "cos_angle_vs_ridge",
    "residual_feasible",
    "diagnostic_feasible",
    "gate_validation_recommended",
    "rejection_reason",
]


def run_qsvt_residual_feasible_codesigned_search(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input": "outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv",
        "build_if_missing": True,
        "output_dir": "outputs/qsvt_residual_feasible_codesigned_search",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])

    frame = _load_codesigned_summary(
        Path(resolved["input"]), build_if_missing=bool(resolved["build_if_missing"])
    )
    rows = evaluate_codesigned_feasibility(frame)
    rows = _mark_gate_validation_recommended(rows)
    artifacts = write_codesigned_search_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def evaluate_codesigned_feasibility(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        rows.append(_classify_codesigned_row(record))
    return rows


def _classify_codesigned_row(record: dict[str, Any]) -> dict[str, Any]:
    qsvt_safe = _as_bool(record.get("qsvt_safe"))
    deployability_class = str(record.get("deployability_class", ""))
    success_probability = _as_float(record.get("success_probability_proxy"))
    residual_ratio = _as_float(record.get("residual_ratio_vs_no_update"))
    direction_error = _as_float(record.get("direction_error_vs_ridge"))
    best_scalar_ratio = _best_scalar_ratio(record)

    residual_feasible, diagnostic_feasible, rejection = _decide_codesigned_feasibility(
        qsvt_safe=qsvt_safe,
        deployability_class=deployability_class,
        success_probability=success_probability,
        residual_ratio=residual_ratio,
        direction_error=direction_error,
        best_scalar_ratio=best_scalar_ratio,
    )
    return {
        "case": record.get("case"),
        "model": record.get("model"),
        "subproblem_id": record.get("subproblem_id"),
        "alpha": _as_float(record.get("alpha")),
        "degree": _as_int(record.get("degree")),
        "target_family": record.get("target_family"),
        "weighting_scheme": record.get("weighting_scheme"),
        "deployability_class": deployability_class,
        "qsvt_safe": bool(qsvt_safe),
        "success_probability_proxy": float(success_probability),
        "residual_ratio_vs_no_update": float(residual_ratio),
        "direction_error_vs_ridge": float(direction_error),
        "cos_angle_vs_ridge": _as_float(record.get("cos_angle_vs_ridge")),
        "residual_feasible": bool(residual_feasible),
        "diagnostic_feasible": bool(diagnostic_feasible),
        "gate_validation_recommended": False,
        "rejection_reason": rejection,
    }


def _decide_codesigned_feasibility(
    *,
    qsvt_safe: bool,
    deployability_class: str,
    success_probability: float,
    residual_ratio: float,
    direction_error: float,
    best_scalar_ratio: float,
) -> tuple[bool, bool, str]:
    thresholds_met_direction = math.isfinite(direction_error) and (
        direction_error <= DIRECTION_ERROR_FEASIBLE_MAX
    )
    deployable_residual_ok = math.isfinite(residual_ratio) and (
        residual_ratio <= RESIDUAL_RATIO_FEASIBLE_MAX
    )

    if (
        qsvt_safe
        and deployability_class in DEPLOYABLE_CLASSES
        and deployable_residual_ok
        and thresholds_met_direction
        and math.isfinite(success_probability)
        and success_probability >= SUCCESS_PROBABILITY_MIN
    ):
        return True, False, ""

    diagnostic_residual_ok = math.isfinite(best_scalar_ratio) and (
        best_scalar_ratio <= RESIDUAL_RATIO_FEASIBLE_MAX
    )
    if diagnostic_residual_ok and thresholds_met_direction:
        return False, True, "diagnostic_only_best_scalar_not_deployable"

    return (
        False,
        False,
        _rejection_reason(
            qsvt_safe=qsvt_safe,
            deployability_class=deployability_class,
            success_probability=success_probability,
            residual_ratio=residual_ratio,
            direction_error=direction_error,
        ),
    )


def _rejection_reason(
    *,
    qsvt_safe: bool,
    deployability_class: str,
    success_probability: float,
    residual_ratio: float,
    direction_error: float,
) -> str:
    if not qsvt_safe:
        return "not_qsvt_safe"
    if deployability_class not in DEPLOYABLE_CLASSES:
        return f"non_deployable_class:{deployability_class or 'unknown'}"
    if not math.isfinite(direction_error) or direction_error > DIRECTION_ERROR_FEASIBLE_MAX:
        return (
            f"direction_error_vs_ridge_{direction_error:.3g}_"
            f"exceeds_{DIRECTION_ERROR_FEASIBLE_MAX:g}"
        )
    if math.isfinite(success_probability) and success_probability < SUCCESS_PROBABILITY_MIN:
        return f"success_probability_{success_probability:.3g}_below_{SUCCESS_PROBABILITY_MIN:g}"
    if not math.isfinite(residual_ratio) or residual_ratio > RESIDUAL_RATIO_FEASIBLE_MAX:
        return (
            f"residual_ratio_vs_no_update_{residual_ratio:.3g}_"
            f"exceeds_{RESIDUAL_RATIO_FEASIBLE_MAX:g}"
        )
    return "not_feasible"


def _best_scalar_ratio(record: dict[str, Any]) -> float:
    best_scalar = _as_float(record.get("residual_best_scalar_diagnostic"))
    amplitude = _as_float(record.get("residual_amplitude_recovered_proxy"))
    ratio = _as_float(record.get("residual_ratio_vs_no_update"))
    if not math.isfinite(best_scalar):
        return float("nan")
    no_update = (
        amplitude / ratio
        if (math.isfinite(amplitude) and math.isfinite(ratio) and ratio > 0.0)
        else float("nan")
    )
    if not math.isfinite(no_update) or no_update <= 0.0:
        return float("nan")
    return best_scalar / no_update


def _mark_gate_validation_recommended(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deployable = [
        row
        for row in rows
        if bool(row.get("residual_feasible"))
        and int(row.get("degree", 0)) <= GATE_VALIDATION_DEGREE_MAX
    ]
    pool = deployable
    if not pool:
        pool = [
            row
            for row in rows
            if bool(row.get("diagnostic_feasible"))
            and int(row.get("degree", 0)) <= GATE_VALIDATION_DEGREE_MAX
        ]
    pool.sort(
        key=lambda row: (
            float(row.get("residual_ratio_vs_no_update", float("inf"))),
            float(row.get("direction_error_vs_ridge", float("inf"))),
            int(row.get("degree", 0)),
            str(row.get("target_family", "")),
        )
    )
    for row in pool[:3]:
        row["gate_validation_recommended"] = True
    return rows


def write_codesigned_search_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, RESULT_COLUMNS)
    feasible_frame = frame[frame["residual_feasible"] == True].copy()  # noqa: E712
    diagnostic_frame = frame[frame["diagnostic_feasible"] == True].copy()  # noqa: E712
    rejected_frame = frame[
        (frame["residual_feasible"] != True) & (frame["diagnostic_feasible"] != True)  # noqa: E712
    ].copy()

    all_path = output_dir / "all_codesigned_configs.csv"
    feasible_path = output_dir / "deployable_residual_feasible_configs.csv"
    diagnostic_path = output_dir / "diagnostic_feasible_configs.csv"
    rejected_path = output_dir / "rejected_codesigned_configs.csv"
    interpretation_path = output_dir / "residual_feasible_codesigned_interpretation.md"

    frame.to_csv(all_path, index=False)
    feasible_frame.to_csv(feasible_path, index=False)
    diagnostic_frame.to_csv(diagnostic_path, index=False)
    rejected_frame.to_csv(rejected_path, index=False)
    interpretation_path.write_text(codesigned_search_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "all_codesigned_configs": str(all_path),
            "deployable_residual_feasible_configs": str(feasible_path),
            "diagnostic_feasible_configs": str(diagnostic_path),
            "rejected_codesigned_configs": str(rejected_path),
            "residual_feasible_codesigned_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=CODESIGNED_SEARCH_CLAIM,
    )
    return {
        "manifest": manifest,
        "all_codesigned_configs": all_path,
        "deployable_residual_feasible_configs": feasible_path,
        "diagnostic_feasible_configs": diagnostic_path,
        "rejected_codesigned_configs": rejected_path,
        "residual_feasible_codesigned_interpretation": interpretation_path,
    }


def codesigned_search_interpretation(frame: pd.DataFrame) -> str:
    total = len(frame)
    feasible_mask = frame["residual_feasible"] == True  # noqa: E712
    diagnostic_mask = frame["diagnostic_feasible"] == True  # noqa: E712
    feasible = int(feasible_mask.sum())
    diagnostic = int(diagnostic_mask.sum())
    rejected = int(total - feasible - diagnostic)

    rejection_counts = (
        frame[~feasible_mask & ~diagnostic_mask]["rejection_reason"]
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .to_dict()
    )
    top_reasons = sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))[:5]

    instance_required = "no"
    if feasible:
        feasible_rows = frame[feasible_mask]
        if (feasible_rows["deployability_class"] == "general_qsvt_safe").any():
            instance_required = "no (a general QSVT-safe target is feasible)"
        else:
            instance_required = "yes (only instance-aware targets are feasible)"
        best = feasible_rows.loc[
            pd.to_numeric(feasible_rows["residual_ratio_vs_no_update"], errors="coerce").idxmin()
        ]
        best_ratio_value = float(best["residual_ratio_vs_no_update"])
        best_line = (
            f"target_family={best['target_family']}, alpha={float(best['alpha']):.3g}, "
            f"degree={int(best['degree'])}, ratio={best_ratio_value:.6g}, "
            f"direction_error={float(best['direction_error_vs_ridge']):.6g}, "
            f"success_proxy={float(best['success_probability_proxy']):.4g}"
        )
    else:
        best_line = "none"

    success_blocked = _success_blocks_feasibility(frame)
    readiness = (
        "solver_prototype_ready_selected_subproblems (if gate validation confirms)"
        if feasible
        else (
            "engineering_framework_ready_with_negative_result"
            if not diagnostic
            else "draft_ready_with_major_limitations"
        )
    )

    return "\n".join(
        [
            "# Residual-Feasible Search With Co-Designed Targets",
            "",
            CODESIGNED_SEARCH_CLAIM,
            "",
            "## Counts",
            f"- Configurations tested: {total}",
            f"- Deployable residual-feasible: {feasible}",
            f"- Diagnostic-feasible (best-scalar lower bound): {diagnostic}",
            f"- Rejected: {rejected}",
            "",
            "## Best Deployable Feasible Configuration",
            f"- {best_line}",
            "",
            "## Main Rejection Reasons",
            *(
                [f"- {reason}: {count}" for reason, count in top_reasons]
                if top_reasons
                else ["- None recorded."]
            ),
            "",
            "## Required Answers",
            f"1. Any deployable residual-feasible configurations? {'yes' if feasible else 'no'} "
            f"({feasible} configs).",
            f"2. Any diagnostic-feasible configurations? {'yes' if diagnostic else 'no'} "
            f"({diagnostic} configs).",
            f"3. Does feasibility require instance-aware target design? {instance_required}.",
            f"4. Does success probability block otherwise-feasible configs? {success_blocked}.",
            f"5. Solver-readiness implication: {readiness}.",
            "",
        ]
    )


def _success_blocks_feasibility(frame: pd.DataFrame) -> str:
    numeric = frame.copy()
    numeric["success_probability_proxy"] = pd.to_numeric(
        numeric["success_probability_proxy"], errors="coerce"
    )
    numeric["residual_ratio_vs_no_update"] = pd.to_numeric(
        numeric["residual_ratio_vs_no_update"], errors="coerce"
    )
    numeric["direction_error_vs_ridge"] = pd.to_numeric(
        numeric["direction_error_vs_ridge"], errors="coerce"
    )
    blocked = numeric[
        (numeric["qsvt_safe"] == True)  # noqa: E712
        & numeric["deployability_class"].isin(DEPLOYABLE_CLASSES)
        & (numeric["residual_ratio_vs_no_update"] <= RESIDUAL_RATIO_FEASIBLE_MAX)
        & (numeric["direction_error_vs_ridge"] <= DIRECTION_ERROR_FEASIBLE_MAX)
        & (numeric["success_probability_proxy"] < SUCCESS_PROBABILITY_MIN)
    ]
    if blocked.empty:
        return (
            "no (no config with feasible residual/direction is blocked solely by "
            "success probability)"
        )
    return (
        f"yes ({len(blocked)} configs meet residual/direction but fail the "
        "success-probability floor)"
    )


def _load_codesigned_summary(input_path: Path, *, build_if_missing: bool) -> pd.DataFrame:
    if input_path.is_file():
        try:
            frame = pd.read_csv(input_path)
            if not frame.empty:
                return frame
        except Exception:
            pass
    if build_if_missing:
        output_dir = input_path.parent
        run = run_qsvt_codesigned_bounded_target_study({"output_dir": str(output_dir)})
        return pd.read_csv(run["artifacts"]["codesigned_target_summary"])
    return pd.DataFrame()


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
