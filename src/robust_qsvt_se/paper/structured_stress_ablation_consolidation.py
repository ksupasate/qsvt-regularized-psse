"""Phase D: structured stress and measurement ablation consolidation.

Consolidates the existing single-axis stress sweeps, conditioning diagnostics,
and measurement-redundancy data into manuscript-ready tables. Random missing
data is labelled as random missing (never as structured/spatial missing),
field-calibrated stress statistics are never claimed, and unavailable
measurement-type ablations and compound-stress sweeps are listed as missing,
not fabricated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

STRESS_COLUMNS = [
    "case",
    "workflow",
    "stress_type",
    "stress_subtype",
    "estimator",
    "alpha",
    "noise_level",
    "missing_ratio",
    "bad_data_ratio",
    "weak_area_multiplier",
    "rmse",
    "weighted_residual_norm",
    "condition_number",
    "seed",
    "source_artifact",
    "result_status",
    "notes",
]

MEASUREMENT_COLUMNS = [
    "case",
    "workflow",
    "measurement_subset",
    "measurement_types_included",
    "n_rows",
    "state_dimension",
    "redundancy_ratio",
    "condition_number",
    "sigma_min",
    "sigma_max",
    "estimator",
    "alpha",
    "rmse",
    "weighted_residual_norm",
    "source_artifact",
    "result_status",
    "notes",
]

DIAGNOSTIC_COLUMNS = [
    "case",
    "estimator",
    "alpha",
    "target_condition_number",
    "missing_ratio",
    "rmse",
    "weighted_residual_norm",
    "failure_rate",
    "unstable_ablation",
    "hhl_instability_flag",
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

STRESS_FIG_COLUMNS = [
    "case",
    "estimator",
    "stress_type",
    "stress_value",
    "rmse",
    "condition_number",
    "source_artifact",
]
CONDITIONING_FIG_COLUMNS = [
    "case",
    "measurement_subset",
    "n_rows",
    "state_dimension",
    "redundancy_ratio",
    "condition_number",
    "source_artifact",
]

_SENSITIVITY_FILES = {
    "noise_sensitivity.csv": ("noise_only", "noise_level"),
    "missing_sensitivity.csv": ("missing_only", "missing_ratio"),
    "bad_data_sensitivity.csv": ("bad_data_only", "bad_data_ratio"),
}

_FULL_AC_TYPES = "voltage_magnitude; p_injection; q_injection; p_branch_flow; q_branch_flow"
_AC_GROUPS = {
    "AC-linearized built-in IEEE14",
    "AC-linearized built-in IEEE14 / bad-data stress",
    "Nonlinear AC built-in IEEE14",
    "PYPOWER AC-linearized",
    "PYPOWER nonlinear AC",
}


def build_structured_stress_ablation_consolidation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "config_root": "configs",
        "output_dir": "outputs/final_manuscript_package/phase5_structured_stress_ablation",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    stress_rows = _stress_rows(input_root)
    measurement_rows = _measurement_rows(input_root)
    diagnostic_rows = _diagnostic_rows(input_root)
    stress_fig, conditioning_fig = _figure_rows(stress_rows, measurement_rows)
    missing_rows = _missing_rows(stress_rows, measurement_rows)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        stress_rows=stress_rows,
        measurement_rows=measurement_rows,
        diagnostic_rows=diagnostic_rows,
        stress_fig=stress_fig,
        conditioning_fig=conditioning_fig,
        missing_rows=missing_rows,
    )
    available_stress = sorted({r["stress_type"] for r in stress_rows})
    return {
        "output_dir": output_dir,
        "stress_rows": stress_rows,
        "measurement_rows": measurement_rows,
        "diagnostic_rows": diagnostic_rows,
        "missing_rows": missing_rows,
        "available_stress_types": available_stress,
        "structured_stress_available": bool(stress_rows),
        "artifacts": artifacts,
    }


def _stress_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, (stress_type, axis) in _SENSITIVITY_FILES.items():
        frame = read_csv(input_root / "sensitivity_summary" / file_name)
        if frame.empty:
            continue
        source = f"outputs/sensitivity_summary/{file_name}"
        subtype = "random" if stress_type == "missing_only" else "single_axis"
        for _, record in frame.iterrows():
            value = record.get("sweep_value", "")
            case = _case_from_source(str(record.get("source_output", "")))
            rows.append(
                {
                    "case": case,
                    "workflow": _workflow_from_source(str(record.get("source_output", ""))),
                    "stress_type": stress_type,
                    "stress_subtype": subtype,
                    "estimator": _canon(str(record.get("estimator", ""))),
                    "alpha": "",
                    "noise_level": value if axis == "noise_level" else "",
                    "missing_ratio": value if axis == "missing_ratio" else "",
                    "bad_data_ratio": value if axis == "bad_data_ratio" else "",
                    "weak_area_multiplier": "",
                    "rmse": _num(record, "rmse_median", "rmse_mean"),
                    "weighted_residual_norm": _num(
                        record, "weighted_residual_norm_mean", "weighted_residual_mean"
                    ),
                    "condition_number": _num(record, "condition_number_mean"),
                    "seed": "aggregated",
                    "source_artifact": source,
                    "result_status": "completed",
                    "notes": (
                        "random (non-structured) missing rows"
                        if stress_type == "missing_only"
                        else "single-axis controlled stress sweep; generated rows"
                    ),
                }
            )
    return rows


def _measurement_rows(input_root: Path) -> list[dict[str, Any]]:
    frame = read_csv(input_root / "measurement_redundancy" / "measurement_redundancy_by_case.csv")
    if frame.empty:
        return []
    source = "outputs/measurement_redundancy/measurement_redundancy_by_case.csv"
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, record in frame.iterrows():
        group = str(record.get("experiment_group", ""))
        if group not in _AC_GROUPS:
            continue
        case = str(record.get("case_name", "")).lower()
        key = (case, group)
        if key in seen:
            continue
        seen.add(key)
        sigma_min, sigma_max = _baseline_sigma(input_root, case)
        rows.append(
            {
                "case": case,
                "workflow": group,
                "measurement_subset": "full_ac_measurement_set",
                "measurement_types_included": _FULL_AC_TYPES,
                "n_rows": _int(record.get("measurement_rows_before_missing")),
                "state_dimension": _int(record.get("state_dimension")),
                "redundancy_ratio": _num(record, "redundancy_ratio"),
                "condition_number": _num(record, "condition_number_before_missing"),
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "estimator": "(conditioning only)",
                "alpha": "",
                "rmse": "",
                "weighted_residual_norm": "",
                "source_artifact": source,
                "result_status": "full_set_only",
                "notes": "full AC measurement set; per-type drop ablation not available",
            }
        )
    return rows


def _diagnostic_rows(input_root: Path) -> list[dict[str, Any]]:
    summary = read_csv(input_root / "diagnostic_missing_baselines" / "summary_metrics.csv")
    if summary.empty:
        return []
    aggregate = read_csv(input_root / "diagnostic_missing_baselines" / "aggregate_metrics.csv")
    unstable = _unstable_lookup(aggregate)
    source = "outputs/diagnostic_missing_baselines/summary_metrics.csv"
    rows: list[dict[str, Any]] = []
    for _, record in summary.iterrows():
        estimator = str(record.get("estimator", ""))
        flags = unstable.get(estimator, ("", ""))
        rows.append(
            {
                "case": "synthetic",
                "estimator": _canon(estimator),
                "alpha": "",
                "target_condition_number": _num(record, "sweep_value"),
                "missing_ratio": 0.0,
                "rmse": _num(record, "rmse_median", "rmse_mean"),
                "weighted_residual_norm": _num(
                    record, "weighted_residual_norm_median", "weighted_residual_median"
                ),
                "failure_rate": _num(record, "failure_rate"),
                "unstable_ablation": flags[0],
                "hhl_instability_flag": flags[1],
                "source_artifact": source,
                "notes": "synthetic high-condition diagnostic (conditioning sweep; missing=0)",
            }
        )
    return rows


def _figure_rows(
    stress_rows: list[dict[str, Any]], measurement_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stress_fig = [
        {
            "case": row["case"],
            "estimator": row["estimator"],
            "stress_type": row["stress_type"],
            "stress_value": row["noise_level"] or row["missing_ratio"] or row["bad_data_ratio"],
            "rmse": row["rmse"],
            "condition_number": row["condition_number"],
            "source_artifact": row["source_artifact"],
        }
        for row in stress_rows
    ]
    conditioning_fig = [
        {
            "case": row["case"],
            "measurement_subset": row["measurement_subset"],
            "n_rows": row["n_rows"],
            "state_dimension": row["state_dimension"],
            "redundancy_ratio": row["redundancy_ratio"],
            "condition_number": row["condition_number"],
            "source_artifact": row["source_artifact"],
        }
        for row in measurement_rows
    ]
    return stress_fig, conditioning_fig


def _missing_rows(
    stress_rows: list[dict[str, Any]], measurement_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    available = {r["stress_type"] for r in stress_rows}
    rows: list[dict[str, Any]] = []
    for stress_type in (
        "weak_area_only",
        "noise_plus_missing",
        "noise_plus_bad_data",
        "missing_plus_bad_data",
    ):
        if stress_type not in available:
            rows.append(
                {
                    "missing_output": f"{stress_type} structured stress sweep",
                    "needed_for": "compound / weak-area structured stress ablation",
                    "importance": "medium",
                    "reason_missing": "only single-axis random stress sweeps were run",
                    "recommended_action": "future work; do not fabricate compound-stress rows",
                }
            )
    rows.append(
        {
            "missing_output": "structured / contiguous-area (spatial) missing-data sweep",
            "needed_for": "structured (non-random) missing-data ablation",
            "importance": "medium",
            "reason_missing": "available missing-data sweeps drop rows at random, not by area",
            "recommended_action": "future work; random missing is not relabelled as structured",
        }
    )
    for subset in (
        "voltage_only",
        "voltage_plus_injection",
        "drop_voltage_rows",
        "drop_injection_rows",
        "drop_branch_flow_rows",
    ):
        rows.append(
            {
                "missing_output": f"measurement-type ablation: {subset}",
                "needed_for": "per-measurement-type conditioning ablation",
                "importance": "medium",
                "reason_missing": "only full-AC-set redundancy is recorded; no per-type subsets",
                "recommended_action": "future work; full set provided, type drops not fabricated",
            }
        )
    rows.append(
        {
            "missing_output": "field-calibrated stress / missing / bad-data distributions",
            "needed_for": "realistic stress statistics",
            "importance": "low",
            "reason_missing": "stress is synthetic/generated on benchmark network models",
            "recommended_action": "do not claim field-calibrated statistics (out of scope)",
        }
    )
    return rows


def _case_from_source(source: str) -> str:
    match = re.search(r"ieee(\d+)", source.lower())
    return f"ieee{match.group(1)}" if match else "unknown_or_aggregate"


def _workflow_from_source(source: str) -> str:
    lowered = source.lower()
    if "nonlinear" in lowered:
        return "Nonlinear AC iterative"
    if "report" in lowered:
        return "consolidated manuscript report"
    return "AC-linearized weighted update"


def _canon(name: str) -> str:
    return {"ridge": "ridge_tikhonov", "qsvt_regularized": "qsvt_target_classical"}.get(name, name)


def _baseline_sigma(input_root: Path, case: str) -> tuple[Any, Any]:
    frame = read_csv(input_root / f"real_{case}_seed10" / "singular_values.csv")
    if frame.empty or "singular_value" not in frame.columns:
        return "", ""
    if "estimator" in frame.columns and not frame.empty:
        frame = frame[frame["estimator"] == frame["estimator"].iloc[0]]
    if "trial_id" in frame.columns and not frame.empty:
        frame = frame[frame["trial_id"] == frame["trial_id"].iloc[0]]
    if "singular_index" in frame.columns:
        frame = frame.drop_duplicates(subset="singular_index", keep="first")
    sigma = pd.to_numeric(frame["singular_value"], errors="coerce").to_numpy()
    sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if sigma.size == 0:
        return "", ""
    return _sig(float(sigma.min())), _sig(float(sigma.max()))


def _unstable_lookup(aggregate: pd.DataFrame) -> dict[str, tuple[Any, Any]]:
    lookup: dict[str, tuple[Any, Any]] = {}
    if aggregate.empty or "estimator" not in aggregate.columns:
        return lookup
    for estimator, group in aggregate.groupby("estimator"):
        unstable = ""
        flag = ""
        if "unstable_ablation" in group.columns:
            unstable = (
                "yes"
                if bool(group["unstable_ablation"].astype(str).str.lower().eq("true").any())
                else "no"
            )
        if "hhl_instability_flag" in group.columns:
            flag = (
                "yes"
                if bool(group["hhl_instability_flag"].astype(str).str.lower().eq("true").any())
                else "no"
            )
        lookup[str(estimator)] = (unstable, flag)
    return lookup


def _num(record: pd.Series, *columns: str) -> Any:
    for column in columns:
        if column in record:
            value = pd.to_numeric(record[column], errors="coerce")
            if not pd.isna(value):
                return round(float(value), 8)
    return ""


def _int(value: Any) -> Any:
    number = pd.to_numeric(value, errors="coerce")
    return "" if pd.isna(number) else int(number)


def _sig(value: float, digits: int = 6) -> float:
    if not np.isfinite(value) or value == 0:
        return float(value)
    from math import floor, log10

    return round(value, -floor(log10(abs(value))) + (digits - 1))


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    stress_rows: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    stress_fig: list[dict[str, Any]],
    conditioning_fig: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    stress_path = rows_to_table(
        stress_rows, output_dir / "paper_table_structured_stress_ablation.csv", STRESS_COLUMNS
    )
    measurement_path = rows_to_table(
        measurement_rows,
        output_dir / "paper_table_measurement_type_ablation.csv",
        MEASUREMENT_COLUMNS,
    )
    diagnostic_path = rows_to_table(
        diagnostic_rows, output_dir / "paper_table_missing_data_diagnostics.csv", DIAGNOSTIC_COLUMNS
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_structured_stress_outputs.csv", MISSING_COLUMNS
    )
    stress_fig_path = rows_to_table(
        stress_fig, output_dir / "figure_data_stress_ablation_rmse.csv", STRESS_FIG_COLUMNS
    )
    conditioning_fig_path = rows_to_table(
        conditioning_fig,
        output_dir / "figure_data_measurement_type_conditioning.csv",
        CONDITIONING_FIG_COLUMNS,
    )
    status_path = output_dir / "structured_stress_status.md"
    status_path.write_text(
        _status_markdown(stress_rows, measurement_rows, diagnostic_rows, missing_rows), "utf-8"
    )
    summary_path = output_dir / "structured_stress_manuscript_summary.md"
    summary_path.write_text(
        _summary_markdown(stress_rows, measurement_rows, missing_rows), encoding="utf-8"
    )

    manifest = write_manifest(
        output_dir,
        artifacts={
            "paper_table_structured_stress_ablation": str(stress_path),
            "paper_table_measurement_type_ablation": str(measurement_path),
            "paper_table_missing_data_diagnostics": str(diagnostic_path),
            "missing_structured_stress_outputs": str(missing_path),
            "figure_data_stress_ablation_rmse": str(stress_fig_path),
            "figure_data_measurement_type_conditioning": str(conditioning_fig_path),
            "structured_stress_status": str(status_path),
            "structured_stress_manuscript_summary": str(summary_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "paper_table_structured_stress_ablation": stress_path,
        "paper_table_measurement_type_ablation": measurement_path,
        "paper_table_missing_data_diagnostics": diagnostic_path,
        "missing_structured_stress_outputs": missing_path,
        "figure_data_stress_ablation_rmse": stress_fig_path,
        "figure_data_measurement_type_conditioning": conditioning_fig_path,
        "structured_stress_status": status_path,
        "structured_stress_manuscript_summary": summary_path,
    }


def _status_markdown(
    stress_rows: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    available = sorted({r["stress_type"] for r in stress_rows})
    support = "supported_with_limitations" if stress_rows else "missing_evidence"
    return "\n".join(
        [
            "# Structured Stress / Measurement Ablation Status",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Summary",
            f"- Structured stress rows: {len(stress_rows)} (available stress types: {available}).",
            f"- Measurement-type (full-set) rows: {len(measurement_rows)}.",
            f"- Conditioning-diagnostic rows: {len(diagnostic_rows)}.",
            f"- Missing structured-stress/ablation outputs recorded: {len(missing_rows)}.",
            f"- C14 (structured stress / measurement ablation) status: {support}.",
            "",
            "## Conclusion",
            "Single-axis stress sweeps and conditioning diagnostics are consolidated. Random "
            "missing is labelled random (not structured). Per-type drops, compound stress, "
            "weak-area/spatial stress, and field-calibrated distributions remain future work and "
            "are not fabricated.",
            "",
        ]
    )


def _summary_markdown(
    stress_rows: list[dict[str, Any]],
    measurement_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> str:
    available = sorted({r["stress_type"] for r in stress_rows})
    cases = sorted({r["case"] for r in measurement_rows})
    return "\n".join(
        [
            "# Structured Stress / Measurement Ablation Summary",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "The weighted Jacobian condition number is",
            "",
            "\\[",
            "\\kappa(\\tilde H)",
            "=",
            "\\frac{\\sigma_{\\max}(\\tilde H)}",
            "{\\sigma_{\\min}(\\tilde H)}.",
            "\\]",
            "",
            "## 1. Which structured stress types are available?",
            f"- {available} (single-axis controlled stress sweeps) plus a synthetic conditioning "
            "diagnostic. Random missing data is labelled random, not structured.",
            "",
            "## 2. Which measurement ablations are available?",
            f"- Full AC measurement set conditioning per case ({cases}). Per-type drops "
            "are not available.",
            "",
            "## 3. Which stress/ablation outputs are missing?",
            f"- {len(missing_rows)} items, including compound stress, weak-area/spatial missing, "
            "per-type measurement drops, and field-calibrated distributions.",
            "",
            "## 4. Does regularization help because conditioning worsens?",
            "- The conditioning diagnostic shows that as the weighted Jacobian condition number "
            "grows, unregularized inverses (pseudoinverse, HHL-style proxy) degrade while the "
            "regularized filter (Ridge / QSVT-target) stays stable. This motivates regularization; "
            "it is not a QSVT-over-Ridge claim.",
            "",
            "## 5. Which measurement types appear most important for conditioning?",
            "- The full AC set (voltage magnitudes, P/Q injections, P/Q branch flows) is "
            "overdetermined; per-type importance requires the missing type-drop ablation and is "
            "left as future work rather than asserted.",
            "",
            "## 6. Which claims must remain future work?",
            "- Per-type ablation, compound/weak-area/spatial structured stress, and any "
            "field-calibrated stress statistics remain future work. The structured-stress claim "
            "(C14) is supported only with these documented limitations.",
            "",
        ]
    )
