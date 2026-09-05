from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.success_amplification_cost import bottleneck_severity, cost_row
from robust_qsvt_se.utils.io import ensure_directory

ACCURACY_SUCCESS_CLAIM = (
    "This study connects residual accuracy with postselection and "
    "amplitude-amplification cost proxies. It does not implement amplitude "
    "amplification as a hardware circuit."
)

TRADEOFF_COLUMNS = [
    "subproblem_id",
    "alpha",
    "degree",
    "best_scalar_residual",
    "residual_ratio_vs_no_update",
    "residual_ratio_vs_ridge_if_defined",
    "success_probability",
    "postselection_cost_proxy",
    "amplitude_amplification_cost_proxy",
    "qsvt_query_count",
    "amplified_query_cost_proxy",
    "pareto_optimal",
    "tradeoff_classification",
]


def run_accuracy_success_tradeoff(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_dirs": [
            "outputs/qsvt_alpha_degree_residual_refinement",
            "outputs/qsvt_refined_selected_subproblem_solver",
            "outputs/qsvt_success_amplification_cost",
        ],
        "output_dir": "outputs/qsvt_accuracy_success_tradeoff",
    }
    resolved.update(config)
    output_dir = ensure_directory(resolved["output_dir"])
    rows = collect_accuracy_success_rows([Path(value) for value in resolved["input_dirs"]])
    rows = mark_pareto_and_classify(rows)
    artifacts = _write_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def collect_accuracy_success_rows(input_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.rglob("*.csv")):
            rows.extend(_rows_from_csv(path))
    return rows


def mark_pareto_and_classify(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    pareto = pareto_frontier(rows)
    marked = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["pareto_optimal"] = index in pareto
        updated["tradeoff_classification"] = classify_tradeoff(updated)
        marked.append(updated)
    return marked


def pareto_frontier(rows: list[dict[str, Any]]) -> set[int]:
    frontier: set[int] = set()
    for i, candidate in enumerate(rows):
        candidate_residual = float(candidate["residual_ratio_vs_no_update"])
        candidate_success = float(candidate["success_probability"])
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            other_residual = float(other["residual_ratio_vs_no_update"])
            other_success = float(other["success_probability"])
            no_worse = other_residual <= candidate_residual and other_success >= candidate_success
            strictly_better = (
                other_residual < candidate_residual or other_success > candidate_success
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(i)
    return frontier


def classify_tradeoff(row: dict[str, Any]) -> str:
    if not bool(row.get("pareto_optimal", False)):
        return "dominated_configuration"
    accuracy_good = float(row["residual_ratio_vs_no_update"]) <= 0.1
    success_ok = bottleneck_severity(float(row["success_probability"])) == "low"
    if accuracy_good and success_ok:
        return "accuracy_improves_success_ok"
    if accuracy_good:
        return "accuracy_improves_success_costly"
    if success_ok:
        return "accuracy_poor_success_ok"
    return "accuracy_poor_success_costly"


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    required_any = {"residual_qsvt_best_scalar", "best_scalar_residual"}
    if not required_any.intersection(frame.columns):
        return []
    rows = []
    for index, row in frame.iterrows():
        converted = _convert_row(path, index, row)
        if converted is not None:
            rows.append(converted)
    return rows


def _convert_row(path: Path, index: int, row: pd.Series) -> dict[str, Any] | None:
    p_success = _column_float(row, ["success_probability"])
    residual = _column_float(row, ["residual_qsvt_best_scalar", "best_scalar_residual"])
    no_update_ratio = _column_float(
        row,
        ["residual_ratio_best_scalar_vs_no_update", "residual_ratio_vs_no_update"],
    )
    if not all(np.isfinite(value) for value in [p_success, residual, no_update_ratio]):
        return None
    ridge = _column_float(row, ["residual_ridge"])
    residual_ratio_vs_ridge = residual / ridge if np.isfinite(ridge) and ridge > 1.0e-12 else np.nan
    query_count = _column_float(row, ["qsvt_query_count", "query_count"])
    costs = cost_row(str(path), p_success, query_count if np.isfinite(query_count) else None)
    return {
        "subproblem_id": str(_column_value(row, ["subproblem_id"]) or f"{path.name}#{index}"),
        "alpha": _column_float(row, ["alpha"]),
        "degree": _column_float(row, ["requested_degree", "degree"]),
        "best_scalar_residual": residual,
        "residual_ratio_vs_no_update": no_update_ratio,
        "residual_ratio_vs_ridge_if_defined": residual_ratio_vs_ridge,
        "success_probability": costs["success_probability"],
        "postselection_cost_proxy": costs["postselection_cost"],
        "amplitude_amplification_cost_proxy": costs["amplitude_amplification_cost_proxy"],
        "qsvt_query_count": query_count,
        "amplified_query_cost_proxy": costs["amplified_qsvt_query_proxy"],
        "pareto_optimal": False,
        "tradeoff_classification": "unclassified",
    }


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    residual_success_path = output_dir / "residual_vs_success_probability.csv"
    residual_amp_path = output_dir / "residual_vs_amplification_cost.csv"
    pareto_path = output_dir / "pareto_frontier.csv"
    interpretation_path = output_dir / "accuracy_success_tradeoff_interpretation.md"
    frame = pd.DataFrame(rows, columns=TRADEOFF_COLUMNS)
    frame.to_csv(residual_success_path, index=False)
    frame[
        [
            "subproblem_id",
            "alpha",
            "degree",
            "best_scalar_residual",
            "residual_ratio_vs_no_update",
            "amplitude_amplification_cost_proxy",
            "amplified_query_cost_proxy",
            "tradeoff_classification",
        ]
    ].to_csv(residual_amp_path, index=False)
    frame[frame["pareto_optimal"] == True].to_csv(pareto_path, index=False)  # noqa: E712
    interpretation_path.write_text(_interpretation_markdown(frame), encoding="utf-8")
    manifest = write_manifest(
        output_dir,
        artifacts={
            "residual_vs_success_probability": str(residual_success_path),
            "residual_vs_amplification_cost": str(residual_amp_path),
            "pareto_frontier": str(pareto_path),
            "accuracy_success_tradeoff_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=ACCURACY_SUCCESS_CLAIM,
    )
    return {
        "manifest": manifest,
        "residual_vs_success_probability": residual_success_path,
        "residual_vs_amplification_cost": residual_amp_path,
        "pareto_frontier": pareto_path,
        "accuracy_success_tradeoff_interpretation": interpretation_path,
    }


def _interpretation_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        success_range = "nan to nan"
        worst_amp = float("nan")
        classifications: dict[str, int] = {}
        pareto_count = 0
    else:
        success_range = (
            f"{float(frame['success_probability'].min()):.17g} to "
            f"{float(frame['success_probability'].max()):.17g}"
        )
        worst_amp = float(frame["amplitude_amplification_cost_proxy"].max())
        classifications = frame["tradeoff_classification"].value_counts().to_dict()
        pareto_count = int((frame["pareto_optimal"] == True).sum())  # noqa: E712
    return "\n".join(
        [
            "# QSVT Accuracy-Success Probability Tradeoff",
            "",
            ACCURACY_SUCCESS_CLAIM,
            "",
            f"- Success probability range: {success_range}",
            f"- Worst amplitude-amplification cost proxy: {worst_amp:.17g}",
            f"- Pareto-optimal configurations: {pareto_count}",
            f"- Tradeoff classifications: {classifications}",
            "",
        ]
    )


def _column_value(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row.index:
            value = row[column]
            if pd.isna(value):
                return None
            return value
    return None


def _column_float(row: pd.Series, columns: list[str]) -> float:
    value = _column_value(row, columns)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
