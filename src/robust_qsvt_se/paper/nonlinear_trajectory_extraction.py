"""Phase 6 / hardening Phase 2: nonlinear AC trajectory extraction.

Extracts per-iteration trajectory figure data from the nonlinear-AC iteration logs
(``iteration_trace.csv``). The legacy ``*_seed10`` logs record the weighted residual and
update norm per iteration, so those trajectories are extracted as real values; they do
not record per-iteration RMSE or condition number, so for those logs RMSE/kappa are
reported as ``not_recorded_by_current_solver`` with no fabricated trajectory. The
logging-refresh log (``nonlinear_logging_refresh/iteration_trace.csv``), produced with
per-iteration logging enabled, does record state RMSE and the iteration-Jacobian
condition number, so the RMSE-vs-iteration and kappa-vs-iteration figure data are
populated from it. Cases with no log are recorded as ``unavailable_logs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, resolved_alpha_map, rows_to_table
from robust_qsvt_se.paper._estimation import ALPHA_ESTIMATORS, INTERNAL_TO_PAPER
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/extract_nonlinear_ac_trajectories.py"

TRAJECTORY_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "estimator",
    "alpha",
    "seed",
    "iteration",
    "metric_value",
    "source_artifact",
    "result_status",
    "notes",
]
MISSING_COLUMNS = ["case", "trajectory", "reason", "result_status"]

_CASE_DIRS = (
    "nonlinear_ac_ieee14_seed10",
    "nonlinear_ac_ieee30_seed10",
    "nonlinear_ac_ieee57_seed10",
    "nonlinear_ac_ieee118_seed10",
    "nonlinear_ac_ieee300_seed10",
)
_REFRESH_DIR = "nonlinear_logging_refresh"
# Per-iteration metric -> trace column. The legacy solver records residual + update_norm;
# state_rmse + condition_number are only present in logging-refresh traces.
_RECORDED = {
    "residual": "weighted_residual_after",
    "update_norm": "update_norm",
    "rmse": "state_rmse",
    "kappa": "condition_number",
}
_LOGGED_METRICS = ("rmse", "kappa")


def build_nonlinear_trajectory_extraction(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "nonlinear_trajectories"))
    ensure_directory(output_dir)

    rows_by_metric: dict[str, list[dict[str, Any]]] = {metric: [] for metric in _RECORDED}
    missing_rows: list[dict[str, Any]] = []
    logs_found: list[str] = []
    refresh_cases: list[str] = []

    for case_dir in _CASE_DIRS:
        case = _case_from_dir(case_dir)
        trace = read_csv(input_root / case_dir / "iteration_trace.csv")
        if trace.empty or "iteration" not in trace.columns:
            missing_rows.append(
                {
                    "case": case,
                    "trajectory": "all",
                    "reason": "iteration_trace.csv missing or empty",
                    "result_status": "unavailable_logs",
                }
            )
            continue
        logs_found.append(case)
        alpha_map = resolved_alpha_map(input_root / case_dir / "config_resolved.yaml")
        rel = f"{case_dir}/iteration_trace.csv"
        for metric in _RECORDED:
            extracted = _extract(trace, case, metric, alpha_map, rel)
            rows_by_metric[metric].extend(extracted)
            # The legacy solver does not record RMSE / condition number; mark, never fabricate.
            if metric in _LOGGED_METRICS and not extracted:
                missing_rows.append(
                    {
                        "case": case,
                        "trajectory": f"nonlinear_{metric}_vs_iteration",
                        "reason": "per-iteration metric not recorded by the legacy solver",
                        "result_status": "not_recorded_by_current_solver",
                    }
                )

    refresh = read_csv(input_root / _REFRESH_DIR / "iteration_trace.csv")
    if not refresh.empty and "iteration" in refresh.columns and "case" in refresh.columns:
        rel = f"{_REFRESH_DIR}/iteration_trace.csv"
        for case_value, group in refresh.groupby(refresh["case"].astype(str)):
            refresh_cases.append(str(case_value))
            alpha_map = _refresh_alpha_map(group)
            for metric in _RECORDED:
                rows_by_metric[metric].extend(
                    _extract(group, str(case_value), metric, alpha_map, rel)
                )
    else:
        missing_rows.append(
            {
                "case": "all",
                "trajectory": "nonlinear_rmse_and_kappa_vs_iteration",
                "reason": "no logging-refresh trace; run scripts/run_nonlinear_logging_refresh.py",
                "result_status": "refresh_not_run",
            }
        )

    return _write_outputs(
        output_dir=output_dir,
        rows_by_metric=rows_by_metric,
        missing_rows=missing_rows,
        logs_found=logs_found,
        refresh_cases=refresh_cases,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
        },
    )


def _case_from_dir(case_dir: str) -> str:
    for token in case_dir.split("_"):
        if token.startswith("ieee"):
            return token
    return case_dir


def _refresh_alpha_map(group: pd.DataFrame) -> dict[str, Any]:
    alpha_map: dict[str, Any] = {}
    if "alpha" not in group.columns:
        return alpha_map
    for _, record in group.iterrows():
        alpha = record.get("alpha")
        if pd.notna(alpha) and str(alpha) != "":
            alpha_map[str(record.get("estimator", ""))] = alpha
    return alpha_map


def _extract(
    trace: pd.DataFrame, case: str, metric: str, alpha_map: dict[str, Any], rel: str
) -> list[dict[str, Any]]:
    column = _RECORDED[metric]
    if column not in trace.columns:
        return []
    rows = []
    for _, record in trace.iterrows():
        internal = str(record.get("estimator", ""))
        paper = INTERNAL_TO_PAPER.get(internal, internal)
        value = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            continue
        alpha = alpha_map.get(internal, "") if paper in ALPHA_ESTIMATORS else ""
        stress_type = record.get("stress_type")
        if stress_type is None or str(stress_type) == "" or pd.isna(stress_type):
            stress_type = record.get("sweep_name", "")
        rows.append(
            {
                "case": case,
                "workflow": "nonlinear_ac_iterative",
                "stress_type": str(stress_type),
                "estimator": paper,
                "alpha": alpha,
                "seed": record.get("seed", ""),
                "iteration": int(record.get("iteration", 0)),
                "metric_value": float(value),
                "source_artifact": rel,
                "result_status": "extracted",
                "notes": (
                    f"{metric} per iteration; trial={record.get('trial_id', '')}; "
                    f"sweep_value={record.get('sweep_value', '')}"
                ),
            }
        )
    return rows


def _status_markdown(
    logs_found: list[str],
    refresh_cases: list[str],
    rows_by_metric: dict[str, list[dict[str, Any]]],
    missing_rows: list[dict[str, Any]],
) -> str:
    unavailable = [r["case"] for r in missing_rows if r["result_status"] == "unavailable_logs"]
    not_recorded = sorted(
        {r["case"] for r in missing_rows if r["result_status"] == "not_recorded_by_current_solver"}
    )
    lines = [
        "# Nonlinear AC Trajectory Status",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "## What was extracted",
        f"- Legacy iteration logs found for cases: {', '.join(logs_found) or 'none'}.",
        f"- Residual-vs-iteration rows extracted: {len(rows_by_metric['residual'])} (real "
        "`weighted_residual_after`).",
        f"- Update-norm-vs-iteration rows extracted: {len(rows_by_metric['update_norm'])} "
        "(real `update_norm`).",
        f"- RMSE-vs-iteration rows extracted: {len(rows_by_metric['rmse'])} (real `state_rmse` "
        "from the logging refresh).",
        f"- Condition-number-vs-iteration rows extracted: {len(rows_by_metric['kappa'])} (real "
        r"`condition_number` = \(\kappa(\tilde H_k)\) from the logging refresh).",
        f"- Logging-refresh cases: {', '.join(sorted(set(refresh_cases))) or 'none'}.",
        "",
        "## What is not available",
        "- The legacy `*_seed10` logs predate per-iteration logging and do NOT record RMSE or "
        "condition number; for those logs the metrics are reported as "
        "`not_recorded_by_current_solver` and are not backfilled.",
        f"- Legacy logs without per-iteration RMSE/kappa: {', '.join(not_recorded) or 'none'}.",
        f"- Cases with no iteration log: {', '.join(unavailable) or 'none'}.",
        "",
        "## Interpretation",
        "- The extracted residual / RMSE trajectories follow",
        "",
        r"\[",
        r"r_k = z - h(x_k),",
        r"\qquad",
        r"H_k = \left.\frac{\partial h}{\partial x}\right|_{x=x_k},",
        r"\]",
        "",
        "with the Jacobian rebuilt each iteration. These are classical nonlinear-AC consistency "
        "trajectories, not a nonlinear QSVT loop; QSVT-target classical equals Ridge for matched "
        "alpha at every iteration.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    rows_by_metric: dict[str, list[dict[str, Any]]],
    missing_rows: list[dict[str, Any]],
    logs_found: list[str],
    refresh_cases: list[str],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    residual_path = rows_to_table(
        rows_by_metric["residual"],
        output_dir / "figure_data_nonlinear_residual_vs_iteration.csv",
        TRAJECTORY_COLUMNS,
    )
    update_path = rows_to_table(
        rows_by_metric["update_norm"],
        output_dir / "figure_data_nonlinear_update_norm_vs_iteration.csv",
        TRAJECTORY_COLUMNS,
    )
    rmse_path = rows_to_table(
        rows_by_metric["rmse"],
        output_dir / "figure_data_nonlinear_rmse_vs_iteration.csv",
        TRAJECTORY_COLUMNS,
    )
    kappa_path = rows_to_table(
        rows_by_metric["kappa"],
        output_dir / "figure_data_nonlinear_kappa_vs_iteration.csv",
        TRAJECTORY_COLUMNS,
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_nonlinear_trajectory_logs.csv", MISSING_COLUMNS
    )
    status_path = output_dir / "nonlinear_trajectory_status.md"
    status_path.write_text(
        _status_markdown(logs_found, refresh_cases, rows_by_metric, missing_rows),
        encoding="utf-8",
    )

    artifacts = {
        "figure_data_nonlinear_residual_vs_iteration": str(residual_path),
        "figure_data_nonlinear_update_norm_vs_iteration": str(update_path),
        "figure_data_nonlinear_rmse_vs_iteration": str(rmse_path),
        "figure_data_nonlinear_kappa_vs_iteration": str(kappa_path),
        "missing_nonlinear_trajectory_logs": str(missing_path),
        "nonlinear_trajectory_status": str(status_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    n_extracted = sum(len(rows) for rows in rows_by_metric.values())
    return {
        "output_dir": output_dir,
        "rows_by_metric": rows_by_metric,
        "residual_rows": rows_by_metric["residual"],
        "update_rows": rows_by_metric["update_norm"],
        "rmse_rows": rows_by_metric["rmse"],
        "kappa_rows": rows_by_metric["kappa"],
        "missing_rows": missing_rows,
        "logs_found": logs_found,
        "refresh_cases": refresh_cases,
        "n_extracted": n_extracted,
        "artifacts": artifacts,
    }
