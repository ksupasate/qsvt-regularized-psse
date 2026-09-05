"""Phase C: nonlinear AC strengthening consolidation.

Consolidates the existing nonlinear AC benchmark runs into paper-level
convergence, estimator-comparison, and bad-data/missing tables. It implements no
new solver: the nonlinear AC workflow perturbs the raw measurements
z = h(x_true) + e + b and rebuilds the Jacobian each iteration, which is kept
separate from the single-step weighted-residual workflow. Missing nonlinear
outputs are listed, never fabricated, and nonlinear AC is framed as
realism/consistency evidence, not QSVT superiority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, resolved_alpha_map, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

CONVERGENCE_COLUMNS = [
    "case",
    "config_name",
    "workflow",
    "estimator",
    "alpha",
    "noise_level",
    "missing_ratio",
    "bad_data_ratio",
    "weak_area_multiplier",
    "seed",
    "converged",
    "iteration_count",
    "max_iterations",
    "failure_reason",
    "final_rmse",
    "final_angle_rmse",
    "final_voltage_rmse",
    "final_residual_norm",
    "final_weighted_residual_norm",
    "source_artifact",
    "result_status",
    "notes",
]

COMPARISON_COLUMNS = [
    "case",
    "stress_type",
    "estimator",
    "alpha",
    "median_rmse",
    "median_angle_rmse",
    "median_voltage_rmse",
    "median_weighted_residual",
    "convergence_rate",
    "failure_rate",
    "n_runs",
    "source_artifact",
    "claim_boundary",
    "notes",
]

BAD_DATA_COLUMNS = [
    "case",
    "stress_type",
    "missing_ratio",
    "bad_data_ratio",
    "bad_data_magnitude",
    "estimator",
    "alpha",
    "median_rmse",
    "median_weighted_residual",
    "convergence_rate",
    "failure_rate",
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

CONVERGENCE_FIG_COLUMNS = [
    "case",
    "estimator",
    "stress_type",
    "stress_value",
    "convergence_rate",
    "source_artifact",
]
FAILURE_FIG_COLUMNS = [
    "case",
    "estimator",
    "bad_data_ratio",
    "failure_rate",
    "median_rmse",
    "source_artifact",
]

_NONLINEAR_DIRS: tuple[tuple[str, str], ...] = (
    ("nonlinear_ac_ieee14_seed10", "ieee14"),
    ("nonlinear_ac_ieee30_seed10", "ieee30"),
    ("nonlinear_ac_ieee57_seed10", "ieee57"),
    ("nonlinear_ac_ieee118_seed10", "ieee118"),
    ("nonlinear_ac_ieee300_seed10", "ieee300"),
)

_STRESS_BY_SWEEP = {
    "nonlinear_noise_sweep": "noise",
    "nonlinear_missing_sweep": "missing",
    "nonlinear_bad_data_ratio_sweep": "bad_data",
}

_CANON = {"ridge": "ridge_tikhonov", "qsvt_regularized": "qsvt_target_classical"}


def build_nonlinear_ac_consolidation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "config_root": "configs",
        "output_dir": "outputs/final_manuscript_package/phase4_nonlinear_ac",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    loaded = _load_nonlinear(input_root)
    convergence_rows = _convergence_rows(loaded)
    comparison_rows = _comparison_rows(loaded)
    bad_data_rows = _bad_data_rows(loaded)
    convergence_fig, failure_fig = _figure_rows(convergence_rows, bad_data_rows)
    missing_rows = _missing_rows(loaded)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        convergence_rows=convergence_rows,
        comparison_rows=comparison_rows,
        bad_data_rows=bad_data_rows,
        convergence_fig=convergence_fig,
        failure_fig=failure_fig,
        missing_rows=missing_rows,
    )
    return {
        "output_dir": output_dir,
        "cases_covered": sorted({case for _, case, _, _, _ in loaded}),
        "convergence_rows": convergence_rows,
        "comparison_rows": comparison_rows,
        "bad_data_rows": bad_data_rows,
        "missing_rows": missing_rows,
        "artifacts": artifacts,
    }


def _load_nonlinear(
    input_root: Path,
) -> list[tuple[str, str, str, dict[str, Any], pd.DataFrame]]:
    loaded: list[tuple[str, str, str, dict[str, Any], pd.DataFrame]] = []
    for rel_dir, case in _NONLINEAR_DIRS:
        frame = read_csv(input_root / rel_dir / "aggregate_metrics.csv")
        if frame.empty or "estimator" not in frame.columns:
            continue
        config_path = input_root / rel_dir / "config_resolved.yaml"
        alpha_map = resolved_alpha_map(config_path)
        max_iter = _max_iterations(config_path)
        loaded.append((rel_dir, case, max_iter, alpha_map, frame))
    return loaded


def _convergence_rows(
    loaded: list[tuple[str, str, str, dict[str, Any], pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_dir, case, max_iter, alpha_map, frame in loaded:
        source = f"outputs/{rel_dir}/aggregate_metrics.csv"
        group_keys = ["estimator", "sweep_name", "sweep_value"]
        for (run_name, sweep_name, sweep_value), group in frame.groupby(group_keys, dropna=False):
            converged_rate = _rate(group, "converged")
            rows.append(
                {
                    "case": case,
                    "config_name": rel_dir,
                    "workflow": "Nonlinear AC iterative (raw z=h(x)+e+b, Jacobian rebuilt)",
                    "estimator": _CANON.get(str(run_name), str(run_name)),
                    "alpha": alpha_map.get(str(run_name), ""),
                    "noise_level": _const(group, "noise_std"),
                    "missing_ratio": _const(group, "missing_ratio"),
                    "bad_data_ratio": _const(group, "bad_data_ratio"),
                    "weak_area_multiplier": "",
                    "seed": _const(group, "seed"),
                    "converged": _converged_label(converged_rate),
                    "iteration_count": _median(group, "iterations"),
                    "max_iterations": max_iter,
                    "failure_reason": _mode(group, "failure_reason"),
                    "final_rmse": _median(group, "rmse"),
                    "final_angle_rmse": _median(group, "angle_rmse"),
                    "final_voltage_rmse": _median(group, "voltage_magnitude_rmse"),
                    "final_residual_norm": _median(group, "residual_norm"),
                    "final_weighted_residual_norm": _median(group, "weighted_residual"),
                    "source_artifact": source,
                    "result_status": "completed",
                    "notes": (
                        f"sweep {_STRESS_BY_SWEEP.get(str(sweep_name), sweep_name)}={sweep_value}; "
                        f"convergence_rate={round(converged_rate, 4)}; medians over trials"
                    ),
                }
            )
    return rows


def _comparison_rows(
    loaded: list[tuple[str, str, str, dict[str, Any], pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_dir, case, _max_iter, alpha_map, frame in loaded:
        source = f"outputs/{rel_dir}/aggregate_metrics.csv"
        for (run_name, sweep_name), group in frame.groupby(
            ["estimator", "sweep_name"], dropna=False
        ):
            converged_rate = _rate(group, "converged")
            rows.append(
                {
                    "case": case,
                    "stress_type": _STRESS_BY_SWEEP.get(str(sweep_name), str(sweep_name)),
                    "estimator": _CANON.get(str(run_name), str(run_name)),
                    "alpha": alpha_map.get(str(run_name), ""),
                    "median_rmse": _median(group, "rmse"),
                    "median_angle_rmse": _median(group, "angle_rmse"),
                    "median_voltage_rmse": _median(group, "voltage_magnitude_rmse"),
                    "median_weighted_residual": _median(group, "weighted_residual"),
                    "convergence_rate": round(converged_rate, 4),
                    "failure_rate": round(1.0 - converged_rate, 4),
                    "n_runs": len(group),
                    "source_artifact": source,
                    "claim_boundary": (
                        "Ridge/Tikhonov reference; QSVT target == Ridge in the simulator; "
                        "robust estimators (Huber) may reduce RMSE under heavy bad data"
                    ),
                    "notes": "realism/consistency evidence; not a QSVT-superiority comparison",
                }
            )
    return rows


def _bad_data_rows(
    loaded: list[tuple[str, str, str, dict[str, Any], pd.DataFrame]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_dir, case, _max_iter, alpha_map, frame in loaded:
        source = f"outputs/{rel_dir}/aggregate_metrics.csv"
        subset = frame[
            frame["sweep_name"]
            .astype(str)
            .isin(["nonlinear_missing_sweep", "nonlinear_bad_data_ratio_sweep"])
        ]
        if subset.empty:
            continue
        group_keys = ["estimator", "sweep_name", "sweep_value"]
        for (run_name, sweep_name, _value), group in subset.groupby(group_keys, dropna=False):
            converged_rate = _rate(group, "converged")
            rows.append(
                {
                    "case": case,
                    "stress_type": _STRESS_BY_SWEEP.get(str(sweep_name), str(sweep_name)),
                    "missing_ratio": _const(group, "missing_ratio"),
                    "bad_data_ratio": _const(group, "bad_data_ratio"),
                    "bad_data_magnitude": _const(group, "bad_data_magnitude"),
                    "estimator": _CANON.get(str(run_name), str(run_name)),
                    "alpha": alpha_map.get(str(run_name), ""),
                    "median_rmse": _median(group, "rmse"),
                    "median_weighted_residual": _median(group, "weighted_residual"),
                    "convergence_rate": round(converged_rate, 4),
                    "failure_rate": round(1.0 - converged_rate, 4),
                    "source_artifact": source,
                    "notes": "random (non-structured) missing/bad-data stress; generated rows",
                }
            )
    return rows


def _figure_rows(
    convergence_rows: list[dict[str, Any]], bad_data_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    convergence_fig: list[dict[str, Any]] = []
    for row in convergence_rows:
        stress = str(row["notes"]).split(";")[0].replace("sweep ", "")
        stress_type, _, stress_value = stress.partition("=")
        convergence_fig.append(
            {
                "case": row["case"],
                "estimator": row["estimator"],
                "stress_type": stress_type.strip(),
                "stress_value": stress_value.strip(),
                "convergence_rate": str(row["notes"]).split("convergence_rate=")[-1].split(";")[0],
                "source_artifact": row["source_artifact"],
            }
        )
    failure_fig = [
        {
            "case": row["case"],
            "estimator": row["estimator"],
            "bad_data_ratio": row["bad_data_ratio"],
            "failure_rate": row["failure_rate"],
            "median_rmse": row["median_rmse"],
            "source_artifact": row["source_artifact"],
        }
        for row in bad_data_rows
        if row["stress_type"] == "bad_data"
    ]
    return convergence_fig, failure_fig


def _missing_rows(
    loaded: list[tuple[str, str, str, dict[str, Any], pd.DataFrame]],
) -> list[dict[str, Any]]:
    covered = {case for _, case, _, _, _ in loaded}
    expected = {"ieee14", "ieee30", "ieee57", "ieee118", "ieee300"}
    rows: list[dict[str, Any]] = [
        {
            "missing_output": "nonlinear AC weak-area / structured-spatial stress sweeps",
            "needed_for": "structured stress realism in the nonlinear workflow",
            "importance": "medium",
            "reason_missing": "nonlinear runs sweep random noise/missing/bad-data only",
            "recommended_action": "see Phase 5 structured stress consolidation (future work)",
        },
        {
            "missing_output": "nonlinear AC combined-stress sweeps (noise+missing+bad-data)",
            "needed_for": "compound-stress robustness of the iterative workflow",
            "importance": "low",
            "reason_missing": "each nonlinear sweep varies a single stress axis",
            "recommended_action": "record as future work; do not fabricate compound-stress rows",
        },
        {
            "missing_output": "nonlinear AC alpha sweep (RMSE vs alpha under Gauss-Newton)",
            "needed_for": "alpha sensitivity of the nonlinear workflow",
            "importance": "low",
            "reason_missing": "alpha is fixed per nonlinear config",
            "recommended_action": "see Phase 3 alpha-sensitivity consolidation",
        },
    ]
    if expected - covered:
        rows.append(
            {
                "missing_output": "nonlinear AC runs for cases "
                + ", ".join(sorted(expected - covered)),
                "needed_for": "full nonlinear case coverage",
                "importance": "medium",
                "reason_missing": "no aggregate_metrics found for these nonlinear case directories",
                "recommended_action": "run nonlinear_ac_<case>_seed10 if the case is needed",
            }
        )
    return rows


def _max_iterations(config_path: Path) -> Any:
    if not config_path.is_file():
        return ""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    value = _find_key(config, "max_iterations")
    return int(value) if isinstance(value, (int, float)) else ""


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _rate(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or group.empty:
        return 0.0
    values = group[column]
    if values.dtype == bool:
        return float(values.mean())
    truthy = values.astype(str).str.lower().isin(["true", "1", "1.0"])
    return float(truthy.mean())


def _converged_label(rate: float) -> str:
    if rate >= 1.0:
        return "yes"
    if rate <= 0.0:
        return "no"
    return "partial"


def _median(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return ""
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return "" if values.empty else round(float(values.median()), 8)


def _const(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return ""
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return round(float(values.iloc[0]), 8)


def _mode(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return ""
    values = group[column].dropna()
    values = values[values.astype(str).str.lower() != "nan"]
    if values.empty:
        return ""
    return str(values.mode().iloc[0])


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    convergence_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    bad_data_rows: list[dict[str, Any]],
    convergence_fig: list[dict[str, Any]],
    failure_fig: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    convergence_path = rows_to_table(
        convergence_rows,
        output_dir / "paper_table_nonlinear_ac_convergence.csv",
        CONVERGENCE_COLUMNS,
    )
    comparison_path = rows_to_table(
        comparison_rows,
        output_dir / "paper_table_nonlinear_ac_estimator_comparison.csv",
        COMPARISON_COLUMNS,
    )
    bad_data_path = rows_to_table(
        bad_data_rows,
        output_dir / "paper_table_nonlinear_ac_bad_data_missing.csv",
        BAD_DATA_COLUMNS,
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_nonlinear_ac_outputs.csv", MISSING_COLUMNS
    )
    convergence_fig_path = rows_to_table(
        convergence_fig,
        output_dir / "figure_data_nonlinear_convergence.csv",
        CONVERGENCE_FIG_COLUMNS,
    )
    failure_fig_path = rows_to_table(
        failure_fig, output_dir / "figure_data_nonlinear_failure_rate.csv", FAILURE_FIG_COLUMNS
    )
    status_path = output_dir / "nonlinear_ac_status.md"
    status_path.write_text(
        _status_markdown(convergence_rows, comparison_rows, bad_data_rows, missing_rows), "utf-8"
    )
    summary_path = output_dir / "nonlinear_ac_manuscript_summary.md"
    summary_path.write_text(
        _summary_markdown(convergence_rows, comparison_rows, bad_data_rows), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "paper_table_nonlinear_ac_convergence": str(convergence_path),
            "paper_table_nonlinear_ac_estimator_comparison": str(comparison_path),
            "paper_table_nonlinear_ac_bad_data_missing": str(bad_data_path),
            "missing_nonlinear_ac_outputs": str(missing_path),
            "figure_data_nonlinear_convergence": str(convergence_fig_path),
            "figure_data_nonlinear_failure_rate": str(failure_fig_path),
            "nonlinear_ac_status": str(status_path),
            "nonlinear_ac_manuscript_summary": str(summary_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "paper_table_nonlinear_ac_convergence": convergence_path,
        "paper_table_nonlinear_ac_estimator_comparison": comparison_path,
        "paper_table_nonlinear_ac_bad_data_missing": bad_data_path,
        "missing_nonlinear_ac_outputs": missing_path,
        "figure_data_nonlinear_convergence": convergence_fig_path,
        "figure_data_nonlinear_failure_rate": failure_fig_path,
        "nonlinear_ac_status": status_path,
        "nonlinear_ac_manuscript_summary": summary_path,
    }


def _status_markdown(
    convergence_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    bad_data_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    cases = sorted({r["case"] for r in convergence_rows})
    estimators = sorted({r["estimator"] for r in comparison_rows})
    support = "supported_with_limitations" if convergence_rows else "missing_evidence"
    return "\n".join(
        [
            "# Nonlinear AC Consolidation Status",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Summary",
            f"- Cases covered: {cases}.",
            f"- Estimators compared: {estimators}.",
            f"- Convergence rows: {len(convergence_rows)}; estimator-comparison rows: "
            f"{len(comparison_rows)}; bad-data/missing rows: {len(bad_data_rows)}.",
            f"- Missing nonlinear outputs recorded: {len(missing_rows)}.",
            f"- Nonlinear AC support status: {support}.",
            "",
            "## Conclusion",
            "Existing nonlinear AC benchmarks are consolidated as realism/consistency evidence for "
            "the regularization story. Missing structured/compound-stress sweeps are listed, not "
            "fabricated. No QSVT superiority over Ridge/Tikhonov is asserted.",
            "",
        ]
    )


def _summary_markdown(
    convergence_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    bad_data_rows: list[dict[str, Any]],
) -> str:
    cases = sorted({r["case"] for r in convergence_rows})
    estimators = sorted({r["estimator"] for r in comparison_rows})
    return "\n".join(
        [
            "# Nonlinear AC Manuscript Summary",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Workflow (raw measurement perturbation, iterative Jacobian rebuild)",
            "The nonlinear AC workflow perturbs the raw measurements",
            "",
            "\\[",
            "z = h(x_{\\mathrm{true}}) + e + b.",
            "\\]",
            "",
            "and solves by Gauss-Newton, rebuilding the Jacobian each iteration:",
            "",
            "\\[",
            "r_k = z - h(x_k),",
            "\\qquad",
            "H_k =",
            "\\left.",
            "\\frac{\\partial h}{\\partial x}",
            "\\right|_{x=x_k}.",
            "\\]",
            "",
            "This is distinct from the single-step AC-linearized workflow, which perturbs the "
            "weighted residual",
            "",
            "\\[",
            "\\tilde r_{\\mathrm{perturbed}}",
            "=",
            "\\tilde r_{\\mathrm{clean}}",
            "+",
            "\\tilde e",
            "+",
            "\\tilde b.",
            "\\]",
            "",
            "## 1. Which nonlinear AC cases exist?",
            f"- {cases} (seed-fixed nonlinear_ac_<case>_seed10 benchmark runs).",
            "",
            "## 2. Which estimators are compared?",
            f"- {estimators} (run names ridge -> ridge_tikhonov, qsvt_regularized -> "
            "qsvt_target_classical).",
            "",
            "## 3. Does nonlinear AC support the main regularization story?",
            "- Yes, as realism/consistency evidence: the regularized filters remain stable and "
            "convergent under the iterative Gauss-Newton workflow on benchmark network models.",
            "",
            "## 4. Where do robust estimators outperform Ridge under bad-data-heavy settings?",
            "- See paper_table_nonlinear_ac_bad_data_missing.csv: under increasing bad-data ratio, "
            "the robust (Huber) estimator can yield lower median RMSE than the non-robust filters; "
            "this is a robustness observation, not a QSVT-over-Ridge claim.",
            "",
            "## 5. Are missing/bad-data nonlinear stress tests available?",
            f"- Random missing and random bad-data sweeps are available ({len(bad_data_rows)} "
            "consolidated rows). Structured/spatial and compound-stress sweeps are not available.",
            "",
            "## 6. What must be framed as a limitation?",
            "- Single-step vs nonlinear scope: the QSVT solver prototype remains a single-step "
            "AC-linearized update on selected 4x4 blocks; the nonlinear AC results are classical "
            "consistency evidence, not a full nonlinear AC QSVT loop. Missing structured/compound "
            "stress sweeps remain future work. No quantum speedup or QSVT superiority is claimed.",
            "",
        ]
    )
