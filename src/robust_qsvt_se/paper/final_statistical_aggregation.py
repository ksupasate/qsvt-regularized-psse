"""Final statistical aggregation from existing manuscript artifacts.

This generator does not run new experiments. It reads existing CSV artifacts, computes
descriptive statistics only from rows that are already present, records missing sources in
the manifest, and keeps the QSVT-target/Ridge equivalence boundary explicit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_final_statistical_aggregation.py"

STAT_COLUMNS = [
    "summary_name",
    "case",
    "workflow",
    "stress_type",
    "measurement_subset",
    "setup",
    "subproblem_type",
    "view",
    "estimator",
    "alpha",
    "degree",
    "shots",
    "metric",
    "n_rows",
    "n_cases",
    "n_seeds",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "failure_count",
    "failure_rate",
    "runtime_limited_count",
    "convergence_rate",
    "qsvt_outperforms_ridge",
    "source_artifact",
    "status",
    "notes",
]

MANIFEST_COLUMNS = [
    "output_table",
    "source_artifact",
    "status",
    "rows_read",
    "rows_written",
    "notes",
]

OUTPUT_FILES = {
    "estimator_seed_variability": "estimator_seed_variability_summary.csv",
    "nonlinear_convergence": "nonlinear_convergence_summary.csv",
    "measurement_ablation": "measurement_ablation_statistical_summary.csv",
    "reactive_conditioning": "reactive_conditioning_statistical_summary.csv",
    "readout_sampling": "readout_sampling_statistical_summary.csv",
    "phase_refinement": "phase_refinement_statistical_summary.csv",
}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    output_key: str
    source_options: tuple[str, ...]
    group_columns: tuple[str, ...]
    metric_columns: tuple[str, ...]
    seed_columns: tuple[str, ...] = ("seed", "rng_seed", "trial_id")
    status_columns: tuple[str, ...] = ("result_status", "status")
    convergence_column: str | None = None
    already_aggregated: bool = False


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        output_key="estimator_seed_variability",
        source_options=("measurement_type_ablation/measurement_type_ablation_all.csv",),
        group_columns=("case", "workflow", "measurement_subset", "estimator", "alpha"),
        metric_columns=("rmse", "weighted_residual_norm", "condition_number"),
    ),
    SourceSpec(
        output_key="nonlinear_convergence",
        source_options=(
            "final_manuscript_package/phase4_nonlinear_ac/paper_table_nonlinear_ac_convergence.csv",
            "phase4_nonlinear_ac/paper_table_nonlinear_ac_convergence.csv",
        ),
        group_columns=("case", "stress_type", "estimator", "alpha"),
        metric_columns=("final_rmse", "final_weighted_residual_norm", "iteration_count"),
        convergence_column="converged",
    ),
    SourceSpec(
        output_key="measurement_ablation",
        source_options=("measurement_type_ablation/measurement_type_ablation_all.csv",),
        group_columns=("case", "workflow", "measurement_subset", "estimator", "alpha"),
        metric_columns=("rmse", "weighted_residual_norm", "condition_number"),
    ),
    SourceSpec(
        output_key="reactive_conditioning",
        source_options=(
            "final_manuscript_package/reactive_power_conditioning/"
            "reactive_power_conditioning_table.csv",
            "reactive_power_conditioning/reactive_power_conditioning_table.csv",
        ),
        group_columns=("case", "setup_id"),
        metric_columns=(
            "condition_number",
            "angle_rmse",
            "voltage_rmse",
            "total_rmse",
            "weighted_residual_norm",
        ),
        already_aggregated=True,
    ),
    SourceSpec(
        output_key="readout_sampling",
        source_options=("full_vector_readout/readout_sampling_trials.csv",),
        group_columns=(
            "case",
            "subproblem_type",
            "method",
            "alpha",
            "degree",
            "shots",
        ),
        metric_columns=(
            "vector_relative_l2_error",
            "max_coordinate_abs_error",
            "sign_accuracy_reliable_coordinates",
            "fraction_reliable_coordinates",
            "top_k_match",
            "norm_relative_error",
        ),
    ),
    SourceSpec(
        output_key="phase_refinement",
        source_options=("phase_synthesis_refinement/cost_accuracy_tradeoff.csv",),
        group_columns=(
            "case",
            "subproblem_type",
            "view",
            "alpha",
            "degree",
            "fit_mode",
        ),
        metric_columns=(
            "gate_error_vs_matching_ridge_alpha",
            "success_probability",
            "postselection_overhead",
            "estimated_total_readout_shots",
            "query_count",
            "accuracy_cost_score",
        ),
        already_aggregated=True,
    ),
    SourceSpec(
        output_key="phase_refinement",
        source_options=("phase_synthesis_refinement/phase_solver_variant_comparison.csv",),
        group_columns=(
            "case",
            "subproblem_type",
            "phase_solver_variant",
            "alpha",
            "degree",
        ),
        metric_columns=(
            "phase_residual",
            "gate_error_vs_polynomial",
            "gate_error_vs_ridge",
            "runtime_seconds",
        ),
    ),
    SourceSpec(
        output_key="phase_refinement",
        source_options=("phase_synthesis_refinement/best_config_sampling_validation_summary.csv",),
        group_columns=("case", "subproblem_type", "view", "alpha", "degree", "shots"),
        metric_columns=(
            "mean_vector_relative_l2_error",
            "p95_vector_relative_l2_error",
            "mean_norm_relative_error",
            "mean_sign_accuracy_reliable",
            "mean_fraction_reliable_signs",
            "top_k_match_rate",
        ),
        already_aggregated=True,
    ),
)


def build_final_statistical_aggregation(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "statistical_summary"))
    ensure_directory(output_dir)

    output_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in OUTPUT_FILES}
    manifest_rows: list[dict[str, Any]] = []
    for spec in SOURCES:
        source_path = _resolve_source(input_root, package_root, spec.source_options)
        if source_path is None:
            manifest_rows.append(
                _manifest_row(
                    spec.output_key,
                    "|".join(spec.source_options),
                    "missing_source_artifact",
                    0,
                    0,
                    "source artifact missing; no rows fabricated",
                )
            )
            continue
        frame = read_csv(source_path)
        if frame.empty:
            manifest_rows.append(
                _manifest_row(
                    spec.output_key,
                    str(source_path),
                    "empty_or_unreadable_source",
                    0,
                    0,
                    "source artifact exists but has no readable rows",
                )
            )
            continue
        prepared = _prepare_frame(frame, spec)
        rows = aggregate_statistics(prepared, spec, str(source_path))
        output_rows[spec.output_key].extend(rows)
        manifest_rows.append(
            _manifest_row(
                spec.output_key,
                str(source_path),
                "read",
                len(frame),
                len(rows),
                "already aggregated source"
                if spec.already_aggregated
                else "computed from existing row-level artifact",
            )
        )

    artifacts = _write_outputs(output_dir, output_rows, manifest_rows, input_root, package_root)
    return {
        "output_dir": output_dir,
        "artifacts": artifacts,
        "rows": output_rows,
        "manifest_rows": manifest_rows,
    }


def aggregate_statistics(
    frame: pd.DataFrame, spec: SourceSpec, source_artifact: str
) -> list[dict[str, Any]]:
    """Aggregate configured metrics for a source frame."""

    if frame.empty:
        return []
    group_cols = [column for column in spec.group_columns if column in frame.columns]
    for column in group_cols:
        if column in {"alpha", "degree", "shots"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not group_cols:
        grouped: Iterable[tuple[Any, pd.DataFrame]] = [((), frame)]
    else:
        grouped = frame.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        key_values = _group_key_values(group_cols, keys)
        for metric in spec.metric_columns:
            if metric not in group.columns:
                continue
            values = _finite_numeric(group[metric])
            rows.append(
                _stat_row(
                    summary_name=spec.output_key,
                    key_values=key_values,
                    metric=metric,
                    values=values,
                    group=group,
                    spec=spec,
                    source_artifact=source_artifact,
                )
            )
    return rows


def _prepare_frame(frame: pd.DataFrame, spec: SourceSpec) -> pd.DataFrame:
    out = frame.copy()
    if spec.output_key == "nonlinear_convergence" and "stress_type" not in out.columns:
        out["stress_type"] = out.apply(_derive_nonlinear_stress_type, axis=1)
    if spec.output_key == "phase_refinement" and "view" not in out.columns:
        out["view"] = out.get("phase_solver_variant", "phase_solver_variant")
    return out


def _derive_nonlinear_stress_type(row: pd.Series) -> str:
    bad = _as_float(row.get("bad_data_ratio"))
    missing = _as_float(row.get("missing_ratio"))
    weak = _as_float(row.get("weak_area_multiplier"))
    noise = _as_float(row.get("noise_level"))
    if bad is not None and bad > 0:
        return "bad_data"
    if missing is not None and missing > 0:
        return "missing"
    if weak is not None and weak > 1:
        return "weak_area"
    if noise is not None and noise > 0:
        return "noise"
    return "clean_or_noise"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _group_key_values(group_cols: list[str], keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return dict(zip(group_cols, keys, strict=True))


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def _stat_row(
    *,
    summary_name: str,
    key_values: dict[str, Any],
    metric: str,
    values: pd.Series,
    group: pd.DataFrame,
    spec: SourceSpec,
    source_artifact: str,
) -> dict[str, Any]:
    seed_count = _seed_count(group, spec)
    failure_count = _failure_count(group, spec)
    runtime_limited_count = _runtime_limited_count(group, spec)
    convergence_rate = _convergence_rate(group, spec)
    note_parts = []
    if seed_count <= 1:
        note_parts.append("single seed/source row; std left null")
    if spec.already_aggregated:
        note_parts.append("source table is already aggregated; no seed variability invented")
    if key_values.get("estimator") == "qsvt_target_classical":
        note_parts.append(
            "QSVT target is Ridge/Tikhonov-equivalent for matched alpha; no superiority claim"
        )
    row = {column: "" for column in STAT_COLUMNS}
    row.update(
        {
            "summary_name": summary_name,
            "case": key_values.get("case", ""),
            "workflow": key_values.get("workflow", ""),
            "stress_type": key_values.get("stress_type", ""),
            "measurement_subset": key_values.get("measurement_subset", ""),
            "setup": key_values.get("setup_id", ""),
            "subproblem_type": key_values.get("subproblem_type", ""),
            "view": key_values.get("view", key_values.get("phase_solver_variant", "")),
            "estimator": key_values.get("estimator", key_values.get("method", "")),
            "alpha": key_values.get("alpha", ""),
            "degree": key_values.get("degree", ""),
            "shots": key_values.get("shots", ""),
            "metric": metric,
            "n_rows": len(group),
            "n_cases": int(group["case"].nunique()) if "case" in group.columns else "",
            "n_seeds": int(seed_count),
            "failure_count": int(failure_count),
            "failure_rate": failure_count / len(group) if len(group) else float("nan"),
            "runtime_limited_count": int(runtime_limited_count),
            "convergence_rate": convergence_rate,
            "qsvt_outperforms_ridge": False,
            "source_artifact": source_artifact,
            "status": "computed" if not values.empty else "no_numeric_values",
            "notes": "; ".join(note_parts),
        }
    )
    if not values.empty:
        row.update(_describe(values, seed_count))
    else:
        row.update({key: float("nan") for key in _STAT_VALUE_KEYS})
    return row


_STAT_VALUE_KEYS = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
)


def _describe(values: pd.Series, seed_count: int) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if seed_count > 1 and len(values) > 1 else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
    }


def _seed_count(group: pd.DataFrame, spec: SourceSpec) -> int:
    for column in spec.seed_columns:
        if column in group.columns:
            count = int(group[column].nunique(dropna=True))
            return max(count, 1)
    if "trials" in group.columns:
        trials = pd.to_numeric(group["trials"], errors="coerce").dropna()
        if not trials.empty:
            return max(int(trials.max()), 1)
    if "n_trials" in group.columns:
        trials = pd.to_numeric(group["n_trials"], errors="coerce").dropna()
        if not trials.empty:
            return max(int(trials.max()), 1)
    return 1


def _failure_count(group: pd.DataFrame, spec: SourceSpec) -> int:
    count = 0
    for column in spec.status_columns:
        if column in group.columns:
            status = group[column].astype(str).str.lower()
            count += int(status.str.contains("fail|error").sum())
    if "converged" in group.columns:
        converged = group["converged"].astype(str).str.lower()
        count += int((~converged.isin(["yes", "true", "1", "converged"])).sum())
    return count


def _runtime_limited_count(group: pd.DataFrame, spec: SourceSpec) -> int:
    count = 0
    for column in spec.status_columns:
        if column in group.columns:
            status = group[column].astype(str).str.lower()
            count += int(status.str.contains("runtime_limited|runtime limited|timeout").sum())
    if "notes" in group.columns:
        notes = group["notes"].astype(str).str.lower()
        count += int(notes.str.contains("runtime_limited|runtime limited|timeout").sum())
    return count


def _convergence_rate(group: pd.DataFrame, spec: SourceSpec) -> float:
    if spec.convergence_column and spec.convergence_column in group.columns:
        values = group[spec.convergence_column].astype(str).str.lower()
        converged = values.isin(["yes", "true", "1", "converged"])
        return float(converged.mean()) if len(converged) else float("nan")
    if "convergence_rate" in group.columns:
        values = _finite_numeric(group["convergence_rate"])
        return float(values.mean()) if not values.empty else float("nan")
    return float("nan")


def _resolve_source(input_root: Path, package_root: Path, options: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for option in options:
        rel = Path(option)
        candidates.extend([input_root / rel, package_root / rel])
        if option.startswith("final_manuscript_package/"):
            candidates.append(input_root / rel)
            candidates.append(package_root / Path(option).relative_to("final_manuscript_package"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _manifest_row(
    output_key: str,
    source_artifact: str,
    status: str,
    rows_read: int,
    rows_written: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "output_table": OUTPUT_FILES[output_key],
        "source_artifact": source_artifact,
        "status": status,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "notes": notes,
    }


def _write_outputs(
    output_dir: Path,
    output_rows: dict[str, list[dict[str, Any]]],
    manifest_rows: list[dict[str, Any]],
    input_root: Path,
    package_root: Path,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for key, filename in OUTPUT_FILES.items():
        path = rows_to_table(output_rows[key], output_dir / filename, STAT_COLUMNS)
        artifacts[key] = str(path)
    manifest_path = rows_to_table(
        manifest_rows,
        output_dir / "statistical_aggregation_manifest.csv",
        MANIFEST_COLUMNS,
    )
    summary_path = output_dir / "statistical_aggregation_summary.md"
    summary_path.write_text(_summary_markdown(output_rows, manifest_rows), encoding="utf-8")
    artifacts["statistical_aggregation_manifest"] = str(manifest_path)
    artifacts["statistical_aggregation_summary"] = str(summary_path)
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
        },
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return artifacts


def _summary_markdown(
    output_rows: dict[str, list[dict[str, Any]]], manifest_rows: list[dict[str, Any]]
) -> str:
    missing = [row for row in manifest_rows if str(row["status"]).startswith("missing")]
    single_seed = sum(
        1 for rows in output_rows.values() for row in rows if int(row.get("n_seeds") or 0) <= 1
    )
    return "\n".join(
        [
            "# Final Statistical Aggregation",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "These summaries are generated only from existing artifacts. Missing inputs are "
            "recorded in the manifest, and no seed variability is invented for already "
            "aggregated or single-seed sources.",
            "",
            "## Tables",
            *[
                f"- {OUTPUT_FILES[key]}: {len(rows)} summary rows"
                for key, rows in output_rows.items()
            ],
            "",
            "## Source Manifest",
            f"- Source records: {len(manifest_rows)}.",
            f"- Missing source records: {len(missing)}.",
            f"- Single-seed/already-aggregated rows with null std: {single_seed}.",
            "",
            "## Claim Boundary",
            "- QSVT-target rows remain matched-alpha Ridge/Tikhonov target rows and are not "
            "reported as outperforming Ridge.",
            "- Readout and phase summaries remain selected-subproblem evidence, not full "
            "IEEE-scale readout or speedup evidence.",
            "",
        ]
    )
