"""Phase B: paper-level alpha-sensitivity consolidation.

Consolidates alpha (Tikhonov) evidence from existing artifacts only. Per-row
alpha is recorded when traceable (resolved configs, alpha-swept QSVT artifacts);
otherwise the row is flagged as not alpha-resolved and listed as missing, never
fabricated. The QSVT-target filter stays Ridge-equivalent for matched alpha.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, resolved_alpha_map, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

ALPHA_SENSITIVITY_COLUMNS = [
    "case",
    "workflow",
    "experiment_group",
    "stress_type",
    "estimator",
    "alpha",
    "alpha_source",
    "rmse",
    "residual_norm",
    "weighted_residual_norm",
    "condition_number",
    "seed",
    "metric_source",
    "alpha_resolved",
    "result_status",
    "notes",
]

ALPHA_RULE_COLUMNS = [
    "rule_id",
    "rule_name",
    "uses_training_split",
    "uses_oracle_best_alpha",
    "uses_fixed_alpha",
    "uses_condition_number",
    "uses_residual",
    "uses_rmse",
    "allowed_for_main_claim",
    "claim_boundary",
    "notes",
]

QSVT_TRADEOFF_COLUMNS = [
    "case",
    "subproblem_id",
    "selection_mode",
    "target_family",
    "alpha",
    "degree",
    "qsvt_safe",
    "residual_ratio_vs_no_update",
    "direction_error_vs_ridge",
    "success_probability",
    "phase_count",
    "query_count",
    "source_artifact",
    "notes",
]

MISSING_COLUMNS = [
    "missing_output",
    "needed_for",
    "importance",
    "reason_missing",
    "recommended_action",
]

RMSE_FIG_COLUMNS = ["case", "alpha", "estimator", "rmse", "source_artifact"]
RESIDUAL_FIG_COLUMNS = ["case", "alpha", "residual_metric", "value", "source_artifact"]
DEGREE_FIG_COLUMNS = [
    "case",
    "alpha",
    "synthesized_degree",
    "phase_count",
    "query_count",
    "source_artifact",
]

# Result directories whose fixed regularization alpha is recoverable from the resolved config.
_FIXED_ALPHA_DIRS: tuple[tuple[str, str, str], ...] = (
    ("real_ieee14_seed10", "ieee14", "AC-linearized weighted update"),
    ("real_ieee30_seed10", "ieee30", "AC-linearized weighted update"),
    ("real_ieee57_seed10", "ieee57", "AC-linearized weighted update"),
    ("real_ieee118_seed10", "ieee118", "AC-linearized weighted update"),
    ("real_ieee300_seed10", "ieee300", "AC-linearized weighted update"),
    ("nonlinear_ac_ieee14_seed10", "ieee14", "Nonlinear AC iterative"),
    ("nonlinear_ac_ieee30_seed10", "ieee30", "Nonlinear AC iterative"),
    ("nonlinear_ac_ieee57_seed10", "ieee57", "Nonlinear AC iterative"),
    ("nonlinear_ac_ieee118_seed10", "ieee118", "Nonlinear AC iterative"),
    ("diagnostic_missing_baselines", "synthetic", "Synthetic conditioning diagnostic"),
)

# Canonical estimator name for the two alpha-parametrized estimators (run name -> paper name).
_ALPHA_ESTIMATORS = {"ridge": "ridge_tikhonov", "qsvt_regularized": "qsvt_target_classical"}

ALPHA_SELECTION_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "AR1",
        "rule_name": "fixed_alpha_grid_reported",
        "uses_training_split": "no",
        "uses_oracle_best_alpha": "no",
        "uses_fixed_alpha": "yes",
        "uses_condition_number": "no",
        "uses_residual": "no",
        "uses_rmse": "no",
        "allowed_for_main_claim": "yes",
        "claim_boundary": "fixed, pre-declared alpha; reported as-is, no test leakage",
        "notes": "Main benchmark runs fix alpha (1e-4 IEEE AC, 1e-3 synthetic diagnostic).",
    },
    {
        "rule_id": "AR2",
        "rule_name": "best_alpha_diagnostic_only",
        "uses_training_split": "no",
        "uses_oracle_best_alpha": "yes",
        "uses_fixed_alpha": "no",
        "uses_condition_number": "no",
        "uses_residual": "yes",
        "uses_rmse": "yes",
        "allowed_for_main_claim": "no",
        "claim_boundary": "oracle best-alpha selection uses the evaluation metric; DIAGNOSTIC ONLY",
        "notes": "Not allowed for deployable claims unless selected without test leakage.",
    },
    {
        "rule_id": "AR3",
        "rule_name": "condition_number_guided_alpha_candidate",
        "uses_training_split": "no",
        "uses_oracle_best_alpha": "no",
        "uses_fixed_alpha": "no",
        "uses_condition_number": "yes",
        "uses_residual": "no",
        "uses_rmse": "no",
        "allowed_for_main_claim": "candidate",
        "claim_boundary": "alpha tied to conditioning of the weighted Jacobian; data-independent",
        "notes": "Candidate rule; not yet validated as a deployable selection policy.",
    },
    {
        "rule_id": "AR4",
        "rule_name": "residual_curve_elbow_candidate",
        "uses_training_split": "no",
        "uses_oracle_best_alpha": "no",
        "uses_fixed_alpha": "no",
        "uses_condition_number": "no",
        "uses_residual": "yes",
        "uses_rmse": "no",
        "allowed_for_main_claim": "candidate",
        "claim_boundary": "L-curve / residual elbow heuristic; uses residual only, not truth",
        "notes": "Candidate rule for alpha without oracle RMSE access.",
    },
    {
        "rule_id": "AR5",
        "rule_name": "qsvt_degree_feasibility_candidate",
        "uses_training_split": "no",
        "uses_oracle_best_alpha": "no",
        "uses_fixed_alpha": "no",
        "uses_condition_number": "yes",
        "uses_residual": "no",
        "uses_rmse": "no",
        "allowed_for_main_claim": "candidate",
        "claim_boundary": "alpha chosen so the bounded QSVT target is degree-feasible",
        "notes": "Larger alpha lowers the QSVT polynomial degree; couples alpha to circuit cost.",
    },
)


def build_alpha_sensitivity_consolidation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "config_root": "configs",
        "output_dir": "outputs/final_manuscript_package/phase3_alpha_sensitivity",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    sensitivity_rows = _alpha_sensitivity_rows(input_root)
    rule_rows = [dict(r) for r in ALPHA_SELECTION_RULES]
    tradeoff_rows = _qsvt_degree_tradeoff_rows(input_root)
    rmse_fig, residual_fig, degree_fig = _figure_rows(input_root)
    missing_rows = _missing_rows(sensitivity_rows, tradeoff_rows, rmse_fig)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        sensitivity_rows=sensitivity_rows,
        rule_rows=rule_rows,
        tradeoff_rows=tradeoff_rows,
        rmse_fig=rmse_fig,
        residual_fig=residual_fig,
        degree_fig=degree_fig,
        missing_rows=missing_rows,
    )
    resolved_count = sum(1 for r in sensitivity_rows if r["alpha_resolved"] == "yes")
    return {
        "output_dir": output_dir,
        "sensitivity_rows": sensitivity_rows,
        "rule_rows": rule_rows,
        "tradeoff_rows": tradeoff_rows,
        "missing_rows": missing_rows,
        "rows_with_alpha": resolved_count,
        "rows_without_alpha": len(sensitivity_rows) - resolved_count,
        "alpha_grid": _alpha_grid(input_root),
        "artifacts": artifacts,
    }


def _alpha_sensitivity_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_fixed_alpha_rows(input_root))
    rows.extend(_swept_alpha_rows(input_root))
    return rows


def _fixed_alpha_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_dir, case, workflow in _FIXED_ALPHA_DIRS:
        summary = read_csv(input_root / rel_dir / "summary_metrics.csv")
        if summary.empty or "estimator" not in summary.columns:
            continue
        alpha_map = resolved_alpha_map(input_root / rel_dir / "config_resolved.yaml")
        for _, record in summary.iterrows():
            run_name = str(record.get("estimator", ""))
            if run_name not in _ALPHA_ESTIMATORS:
                continue
            alpha = alpha_map.get(run_name)
            resolved = alpha is not None
            parameter = str(record.get("sweep_parameter", ""))
            value = record.get("sweep_value", "")
            rows.append(
                {
                    "case": case,
                    "workflow": workflow,
                    "experiment_group": rel_dir,
                    "stress_type": f"{parameter}={value}",
                    "estimator": _ALPHA_ESTIMATORS[run_name],
                    "alpha": alpha if resolved else "",
                    "alpha_source": "resolved_config" if resolved else "not_traceable",
                    "rmse": _num(record, "rmse_median"),
                    "residual_norm": _num(record, "residual_norm_median"),
                    "weighted_residual_norm": _num(record, "weighted_residual_norm_median"),
                    "condition_number": _num(record, "condition_number_median"),
                    "seed": "aggregated",
                    "metric_source": "summary_metrics",
                    "alpha_resolved": "yes" if resolved else "no",
                    "result_status": "single_alpha_fixed",
                    "notes": f"run name {run_name}; fixed alpha (not swept)",
                }
            )
    return rows


def _swept_alpha_rows(input_root: Path) -> list[dict[str, Any]]:
    frame = read_csv(
        input_root / "qsvt_alpha_resource_sensitivity" / "alpha_resource_sensitivity.csv"
    )
    if frame.empty or "alpha" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        case = str(record.get("case_name", ""))
        alpha = record.get("alpha")
        rmse = _coerce(record.get("ridge_rmse_if_available"))
        residual = _coerce(record.get("ridge_residual_if_available"))
        condition = _coerce(record.get("condition_number"))
        for estimator in ("ridge_tikhonov", "qsvt_target_classical"):
            note = "ridge alpha sweep"
            if estimator == "qsvt_target_classical":
                note = "QSVT target == Ridge (relative error vs Ridge ~ 0)"
            rows.append(
                {
                    "case": case,
                    "workflow": "alpha resource sensitivity (QSVT target)",
                    "experiment_group": "qsvt_alpha_resource_sensitivity",
                    "stress_type": "baseline",
                    "estimator": estimator,
                    "alpha": alpha,
                    "alpha_source": "alpha_resource_sensitivity_artifact",
                    "rmse": rmse,
                    "residual_norm": residual,
                    "weighted_residual_norm": "",
                    "condition_number": condition,
                    "seed": "fixed",
                    "metric_source": "qsvt_alpha_resource_sensitivity",
                    "alpha_resolved": "yes",
                    "result_status": "alpha_swept",
                    "notes": note,
                }
            )
    return rows


def _qsvt_degree_tradeoff_rows(input_root: Path) -> list[dict[str, Any]]:
    codesigned = read_csv(
        input_root / "qsvt_codesigned_bounded_target_study" / "codesigned_target_summary.csv"
    )
    if codesigned.empty:
        return []
    phase_lookup = _phase_query_lookup(input_root)
    source = "outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv"
    rows: list[dict[str, Any]] = []
    for _, record in codesigned.iterrows():
        case = str(record.get("case", ""))
        alpha = record.get("alpha")
        degree = record.get("degree")
        subproblem = str(record.get("subproblem_id", ""))
        phase_count, query_count = phase_lookup.get((case, _key(alpha), _key(degree)), ("", ""))
        rows.append(
            {
                "case": case,
                "subproblem_id": subproblem,
                "selection_mode": "high_leverage" if "high_leverage" in subproblem else "selected",
                "target_family": str(record.get("target_family", "")),
                "alpha": alpha,
                "degree": degree,
                "qsvt_safe": record.get("qsvt_safe", ""),
                "residual_ratio_vs_no_update": _coerce(record.get("residual_ratio_vs_no_update")),
                "direction_error_vs_ridge": _coerce(record.get("direction_error_vs_ridge")),
                "success_probability": _coerce(record.get("success_probability_proxy")),
                "phase_count": phase_count,
                "query_count": query_count,
                "source_artifact": source,
                "notes": "phase/query joined from qsvt_alpha_degree_tradeoff where degree matches",
            }
        )
    return rows


def _phase_query_lookup(input_root: Path) -> dict[tuple[str, str, str], tuple[Any, Any]]:
    frame = read_csv(input_root / "qsvt_alpha_degree_tradeoff" / "alpha_degree_summary.csv")
    lookup: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    if frame.empty:
        return lookup
    for _, record in frame.iterrows():
        key = (
            str(record.get("case", "")),
            _key(record.get("alpha")),
            _key(record.get("synthesized_degree")),
        )
        lookup[key] = (
            _int(record.get("phase_count")),
            _int(record.get("query_count_estimate")),
        )
    return lookup


def _figure_rows(
    input_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rmse_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []

    resource = read_csv(
        input_root / "qsvt_alpha_resource_sensitivity" / "alpha_resource_sensitivity.csv"
    )
    if not resource.empty and "alpha" in resource.columns:
        source = "outputs/qsvt_alpha_resource_sensitivity/alpha_resource_sensitivity.csv"
        for _, record in resource.iterrows():
            case = str(record.get("case_name", ""))
            alpha = record.get("alpha")
            rmse = _coerce(record.get("ridge_rmse_if_available"))
            residual = _coerce(record.get("ridge_residual_if_available"))
            for estimator in ("ridge_tikhonov", "qsvt_target_classical"):
                rmse_rows.append(
                    {
                        "case": case,
                        "alpha": alpha,
                        "estimator": estimator,
                        "rmse": rmse,
                        "source_artifact": source,
                    }
                )
            residual_rows.append(
                {
                    "case": case,
                    "alpha": alpha,
                    "residual_metric": "ridge_residual_norm",
                    "value": residual,
                    "source_artifact": source,
                }
            )

    codesigned = read_csv(
        input_root / "qsvt_codesigned_bounded_target_study" / "codesigned_target_summary.csv"
    )
    if not codesigned.empty and "alpha" in codesigned.columns:
        source = "outputs/qsvt_codesigned_bounded_target_study/codesigned_target_summary.csv"
        ratio = pd.to_numeric(codesigned["residual_ratio_vs_no_update"], errors="coerce")
        best = codesigned.assign(_ratio=ratio).groupby(["case", "alpha"])["_ratio"].min()
        for (case, alpha), value in best.items():
            residual_rows.append(
                {
                    "case": str(case),
                    "alpha": alpha,
                    "residual_metric": "qsvt_min_residual_ratio_vs_no_update",
                    "value": _round(value),
                    "source_artifact": source,
                }
            )

    degree = read_csv(input_root / "qsvt_alpha_degree_tradeoff" / "alpha_degree_summary.csv")
    if not degree.empty and "alpha" in degree.columns:
        source = "outputs/qsvt_alpha_degree_tradeoff/alpha_degree_summary.csv"
        for _, record in degree.iterrows():
            degree_rows.append(
                {
                    "case": str(record.get("case", "")),
                    "alpha": record.get("alpha"),
                    "synthesized_degree": _int(record.get("synthesized_degree")),
                    "phase_count": _int(record.get("phase_count")),
                    "query_count": _int(record.get("query_count_estimate")),
                    "source_artifact": source,
                }
            )
    return rmse_rows, residual_rows, degree_rows


def _missing_rows(
    sensitivity_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
    rmse_fig: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    swept_cases = {r["case"] for r in rmse_fig}
    classical_cases = {"ieee14", "ieee30", "ieee57", "ieee118", "ieee300"}
    missing_swept = sorted(classical_cases - swept_cases)
    rows.append(
        {
            "missing_output": "classical RMSE/residual vs alpha sweep for all benchmark cases",
            "needed_for": "case-by-case alpha-sensitivity of the classical estimators",
            "importance": "medium",
            "reason_missing": (
                "alpha-swept classical metrics exist only for "
                f"{sorted(swept_cases) or 'no'} case(s); missing for {missing_swept or 'none'}"
            ),
            "recommended_action": "extend qsvt_alpha_resource_sensitivity / run an alpha grid",
        }
    )
    if not any(r["alpha_resolved"] == "no" for r in sensitivity_rows):
        unresolved_note = "all consolidated rows are alpha-resolved"
    else:
        unresolved_note = "some rows lack a traceable alpha and are flagged alpha_resolved=no"
    rows.append(
        {
            "missing_output": "populated alpha_sensitivity_summary / sensitivity_summary sweep",
            "needed_for": "dedicated alpha-sensitivity tables in the standard summaries",
            "importance": "low",
            "reason_missing": (
                "alpha_sensitivity_summary/* and sensitivity_summary/alpha_sensitivity.csv "
                f"are header-only (0 rows); {unresolved_note}"
            ),
            "recommended_action": "regenerate the alpha sweep or use the QSVT alpha-degree data",
        }
    )
    rows.append(
        {
            "missing_output": "nonlinear-AC alpha sweep (RMSE vs alpha under Gauss-Newton)",
            "needed_for": "alpha sensitivity of the iterative nonlinear workflow",
            "importance": "low",
            "reason_missing": "nonlinear AC runs fix alpha; no nonlinear alpha grid is recorded",
            "recommended_action": "out of scope; record as future work, do not fabricate",
        }
    )
    return rows


def _alpha_grid(input_root: Path) -> list[Any]:
    frame = read_csv(input_root / "qsvt_alpha_degree_tradeoff" / "alpha_degree_summary.csv")
    if frame.empty or "alpha" not in frame.columns:
        return []
    return sorted(pd.to_numeric(frame["alpha"], errors="coerce").dropna().unique().tolist())


def _num(record: pd.Series, column: str) -> Any:
    if column not in record:
        return ""
    value = pd.to_numeric(record[column], errors="coerce")
    return "" if pd.isna(value) else round(float(value), 8)


def _coerce(value: Any) -> Any:
    number = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(number) else round(float(number), 8)


def _round(value: Any) -> Any:
    return "" if pd.isna(value) else round(float(value), 8)


def _int(value: Any) -> Any:
    number = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(number) else int(number)


def _key(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(number) else format(float(number), ".12g")


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    sensitivity_rows: list[dict[str, Any]],
    rule_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
    rmse_fig: list[dict[str, Any]],
    residual_fig: list[dict[str, Any]],
    degree_fig: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    sensitivity_path = rows_to_table(
        sensitivity_rows,
        output_dir / "paper_table_alpha_sensitivity.csv",
        ALPHA_SENSITIVITY_COLUMNS,
    )
    rule_path = rows_to_table(
        rule_rows, output_dir / "paper_table_alpha_selection_rule.csv", ALPHA_RULE_COLUMNS
    )
    tradeoff_path = rows_to_table(
        tradeoff_rows,
        output_dir / "paper_table_alpha_qsvt_degree_tradeoff.csv",
        QSVT_TRADEOFF_COLUMNS,
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_alpha_sensitivity_outputs.csv", MISSING_COLUMNS
    )
    rmse_path = rows_to_table(
        rmse_fig, output_dir / "figure_data_rmse_vs_alpha.csv", RMSE_FIG_COLUMNS
    )
    residual_path = rows_to_table(
        residual_fig, output_dir / "figure_data_residual_vs_alpha.csv", RESIDUAL_FIG_COLUMNS
    )
    degree_path = rows_to_table(
        degree_fig, output_dir / "figure_data_qsvt_degree_vs_alpha.csv", DEGREE_FIG_COLUMNS
    )
    rule_md_path = output_dir / "alpha_selection_rule.md"
    rule_md_path.write_text(_alpha_selection_markdown(sensitivity_rows, tradeoff_rows), "utf-8")
    status_path = output_dir / "alpha_sensitivity_status.md"
    status_path.write_text(
        _status_markdown(sensitivity_rows, tradeoff_rows, missing_rows), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "paper_table_alpha_sensitivity": str(sensitivity_path),
            "paper_table_alpha_selection_rule": str(rule_path),
            "paper_table_alpha_qsvt_degree_tradeoff": str(tradeoff_path),
            "missing_alpha_sensitivity_outputs": str(missing_path),
            "figure_data_rmse_vs_alpha": str(rmse_path),
            "figure_data_residual_vs_alpha": str(residual_path),
            "figure_data_qsvt_degree_vs_alpha": str(degree_path),
            "alpha_selection_rule": str(rule_md_path),
            "alpha_sensitivity_status": str(status_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "paper_table_alpha_sensitivity": sensitivity_path,
        "paper_table_alpha_selection_rule": rule_path,
        "paper_table_alpha_qsvt_degree_tradeoff": tradeoff_path,
        "missing_alpha_sensitivity_outputs": missing_path,
        "figure_data_rmse_vs_alpha": rmse_path,
        "figure_data_residual_vs_alpha": residual_path,
        "figure_data_qsvt_degree_vs_alpha": degree_path,
        "alpha_selection_rule": rule_md_path,
        "alpha_sensitivity_status": status_path,
    }


def _alpha_selection_markdown(
    sensitivity_rows: list[dict[str, Any]], tradeoff_rows: list[dict[str, Any]]
) -> str:
    swept = sorted(
        {r["alpha"] for r in sensitivity_rows if r["result_status"] == "alpha_swept"},
        key=lambda v: float(v),
    )
    tradeoff_alphas = sorted({r["alpha"] for r in tradeoff_rows}, key=lambda v: float(v))
    return "\n".join(
        [
            "# Alpha Selection Rule (manuscript-safe)",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "The Ridge/Tikhonov reference filter is",
            "",
            "\\[",
            "P_{\\alpha}(\\sigma)",
            "=",
            "\\frac{\\sigma}{\\sigma^2+\\alpha}.",
            "\\]",
            "",
            "## 1. Which alpha values are actually tested?",
            "- Fixed main-run alpha: 1e-4 (IEEE AC-linearized and nonlinear AC), 1e-3 (synthetic "
            "diagnostic). Recovered per run from the resolved configs.",
            f"- Alpha sweep (QSVT target / Ridge): {swept or 'none'} on the IEEE14 leading block.",
            f"- QSVT alpha-degree grid: {tradeoff_alphas or 'none'}.",
            "",
            "## 2. Which workflows are alpha-resolved?",
            "- AC-linearized and nonlinear AC main runs: alpha-resolved at a single fixed value.",
            "- QSVT alpha-resource-sensitivity and codesigned bounded-target study: alpha-resolved "
            "across the grid above.",
            "",
            "## 3. Which workflows only have aggregate alpha information?",
            "- The standard alpha_sensitivity_summary / sensitivity_summary alpha tables are "
            "header-only (no rows); they contribute no per-row alpha and are listed as missing.",
            "",
            "## 4. Is alpha chosen by oracle best performance or by a rule?",
            "- Main results use a fixed, pre-declared alpha (rule AR1), not an oracle best-alpha. "
            "An oracle best-alpha (AR2) is treated as DIAGNOSTIC ONLY and is not allowed for "
            "deployable claims because it consumes the evaluation metric.",
            "",
            "## 5. Does the selected QSVT degree window depend on alpha?",
            "- Yes. Larger alpha yields a more bounded target and a lower feasible QSVT polynomial "
            "degree; see paper_table_alpha_qsvt_degree_tradeoff.csv and "
            "figure_data_qsvt_degree_vs_alpha.csv.",
            "",
            "## 6. How should the manuscript describe alpha selection conservatively?",
            "- State that a single fixed Tikhonov alpha is used and reported (not tuned on test "
            "metric); present the alpha sweep as sensitivity evidence; and note that alpha couples "
            "to QSVT degree feasibility. Do not claim an optimal or learned alpha and do not claim "
            "QSVT superiority over Ridge.",
            "",
        ]
    )


def _status_markdown(
    sensitivity_rows: list[dict[str, Any]],
    tradeoff_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    resolved = sum(1 for r in sensitivity_rows if r["alpha_resolved"] == "yes")
    swept = sum(1 for r in sensitivity_rows if r["result_status"] == "alpha_swept")
    cases = sorted({r["case"] for r in sensitivity_rows})
    return "\n".join(
        [
            "# Alpha-Sensitivity Consolidation Status",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Summary",
            f"- Alpha-sensitivity rows consolidated: {len(sensitivity_rows)} (cases {cases}).",
            f"- Rows with a traceable alpha: {resolved} / {len(sensitivity_rows)}.",
            f"- Alpha-swept rows: {swept}; remaining rows are at a single fixed alpha.",
            f"- QSVT alpha-degree tradeoff rows: {len(tradeoff_rows)}.",
            f"- Missing alpha outputs recorded: {len(missing_rows)}.",
            "",
            "## Interpretation",
            "- The classical RMSE/residual results are reported at a fixed, pre-declared Tikhonov "
            "alpha (no test-metric tuning).",
            "- The QSVT polynomial degree is alpha-sensitive: larger alpha lowers the feasible "
            "degree (existing-artifact consolidation; see the alpha-degree tradeoff table).",
            "- The QSVT-target filter equals Ridge in the classical simulator when alpha is the "
            "same; no QSVT-over-Ridge advantage is claimed.",
            "",
            "## Conclusion",
            "Alpha selection can be answered conservatively for reviewers from existing artifacts. "
            "Per-row alpha is present only when traceable; remaining alpha-sweep gaps are listed "
            "in missing_alpha_sensitivity_outputs.csv and are not fabricated.",
            "",
        ]
    )
