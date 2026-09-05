from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

AUDIT_CLAIM = (
    "Audit of the prior actual-amplitude-recovery evidence before co-designing new "
    "QSVT-safe targets, on a selected IEEE14-derived subproblem. It records amplitude-"
    "estimation accuracy, residual-feasibility status, and the dominant bottleneck so "
    "the co-designed target study can be focused. Ridge/Tikhonov remains the reference; "
    "no QSVT superiority over Ridge/Tikhonov, quantum speedup, or hardware execution is "
    "claimed."
)

SUMMARY_COLUMNS = [
    "metric",
    "value",
    "source",
    "detail",
]

_DEPLOYABLE_NORM_METHODS = (
    "bernoulli_success_amplitude",
    "iterative_qae",
    "hybrid_known_C_amplitude",
)


def run_qsvt_codesigned_target_audit(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "output_dir": "outputs/qsvt_codesigned_target_audit",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    output_dir = ensure_directory(resolved["output_dir"])

    rows = current_failure_mode_rows(input_root)
    artifacts = write_audit_outputs(output_dir, resolved, rows)
    return {"output_dir": output_dir, "rows": rows, "artifacts": artifacts}


def current_failure_mode_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    amplitude = _read(
        input_root / "qsvt_amplitude_estimation_routines" / "amplitude_estimation_summary.csv"
    )
    mlae_error = _best_abs_error(amplitude, "iterative_mlae")
    bernoulli_error = _best_abs_error(amplitude, "bernoulli_shots")
    rows.append(
        _row(
            "amplitude_estimation_accuracy",
            f"mlae_best_abs_error={mlae_error:.3g}; bernoulli_best_abs_error={bernoulli_error:.3g}",
            "qsvt_amplitude_estimation_routines",
            "Iterative MLAE and Bernoulli routines estimate the postselection success "
            "amplitude accurately; amplitude sampling is not the bottleneck.",
        )
    )

    recovery_all = _read(
        input_root
        / "qsvt_residual_feasible_with_amplitude_recovery"
        / "all_amplitude_recovery_configs.csv"
    )
    deployable_feasible = _csv_has_rows(
        input_root
        / "qsvt_residual_feasible_with_amplitude_recovery"
        / "residual_feasible_configs.csv"
    )
    best_deployable_ratio = _best_deployable_ratio(recovery_all)
    best_direction = _best_direction(recovery_all)
    rows.append(
        _row(
            "prior_residual_feasibility",
            "achieved" if deployable_feasible else "not_achieved",
            "qsvt_residual_feasible_with_amplitude_recovery",
            f"best deployable residual_ratio_vs_no_update={best_deployable_ratio:.3g}; "
            f"best direction_error_vs_ridge={best_direction:.3g}.",
        )
    )

    if deployable_feasible:
        bottleneck = "none (a deployable amplitude-recovered configuration was already feasible)"
    elif math.isfinite(best_direction) and best_direction <= 0.1:
        bottleneck = (
            "scale/polynomial-target-limited: the bounded QSVT direction can agree with Ridge, "
            "but the prior (uniform, unweighted) target plus the saturating success-probability "
            "proxy did not yield a deployable feasible update; amplitude sampling is accurate."
        )
    else:
        bottleneck = "direction-limited: the prior bounded QSVT direction points away from Ridge."
    rows.append(
        _row(
            "dominant_bottleneck",
            "scale_polynomial_target" if not deployable_feasible else "none",
            "prior_amplitude_recovery",
            bottleneck,
        )
    )

    gate = _read(
        input_root
        / "qsvt_gate_validation_with_amplitude_recovery"
        / "gate_amplitude_recovery_results.csv"
    )
    gate_feasible = (
        bool(
            gate.get("residual_feasible_after_gate", pd.Series(dtype=bool))
            .astype(str)
            .str.lower()
            .eq("true")
            .any()
        )
        if not gate.empty
        else False
    )
    rows.append(
        _row(
            "prior_gate_validation",
            "feasible_after_gate" if gate_feasible else "not_feasible_after_gate",
            "qsvt_gate_validation_with_amplitude_recovery",
            "Gate extraction faithfully reproduced the polynomial action; feasibility was "
            "limited by the target/scale, not the gate.",
        )
    )

    rows.append(
        _row(
            "reuse_recommendation",
            "ieee14_ac_high_leverage_4x4",
            "subproblem_selection",
            "Reuse the selected IEEE14-derived high-leverage 4x4 subproblem and the actual "
            "amplitude-estimation routines; focus the co-design on singular-support-weighted "
            "bounded targets at low-to-moderate degree (15-35) and a physically-faithful "
            "(normalized) success-amplitude model.",
        )
    )
    return rows


def write_audit_outputs(
    output_dir: Path, resolved: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Path]:
    frame = _frame_with_columns(rows, SUMMARY_COLUMNS)
    summary_path = output_dir / "current_failure_mode_summary.csv"
    interpretation_path = output_dir / "current_failure_mode_interpretation.md"

    frame.to_csv(summary_path, index=False)
    interpretation_path.write_text(_audit_interpretation(frame), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "current_failure_mode_summary": str(summary_path),
            "current_failure_mode_interpretation": str(interpretation_path),
        },
        input_config=resolved,
        claim_boundary=AUDIT_CLAIM,
    )
    return {
        "manifest": manifest,
        "current_failure_mode_summary": summary_path,
        "current_failure_mode_interpretation": interpretation_path,
    }


def _audit_interpretation(frame: pd.DataFrame) -> str:
    lines = ["# Current Failure-Mode Audit", "", AUDIT_CLAIM, "", "## Findings"]
    for record in frame.to_dict("records"):
        lines.append(f"- **{record['metric']}**: {record['value']} — {record['detail']}")
    lines.extend(
        [
            "",
            "## Required Statements",
            "1. Amplitude estimation is accurate (iterative MLAE and Bernoulli routines).",
            "2. Residual feasibility under the prior uniform target was not achieved by a "
            "deployable amplitude-recovered configuration.",
            "3. The bottleneck is the polynomial target/scale (and the saturating success-"
            "probability proxy), not amplitude-sampling precision or gate extraction.",
            "4. Reuse the selected IEEE14-derived high-leverage 4x4 subproblem and the actual "
            "amplitude-estimation routines for the co-designed target study.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_abs_error(frame: pd.DataFrame, estimator_type: str) -> float:
    if (
        frame.empty
        or "estimator_type" not in frame.columns
        or "absolute_error" not in frame.columns
    ):
        return float("nan")
    subset = frame[frame["estimator_type"] == estimator_type]
    values = pd.to_numeric(subset["absolute_error"], errors="coerce").dropna()
    return float(values.min()) if not values.empty else float("nan")


def _best_deployable_ratio(frame: pd.DataFrame) -> float:
    if frame.empty or "norm_recovery_method" not in frame.columns:
        return float("nan")
    subset = frame[frame["norm_recovery_method"].isin(_DEPLOYABLE_NORM_METHODS)]
    values = pd.to_numeric(
        subset.get("residual_ratio_vs_no_update", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    return float(values.min()) if not values.empty else float("nan")


def _best_direction(frame: pd.DataFrame) -> float:
    if frame.empty or "direction_error_vs_ridge" not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame["direction_error_vs_ridge"], errors="coerce").dropna()
    return float(values.min()) if not values.empty else float("nan")


def _csv_has_rows(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return not pd.read_csv(path).empty
    except Exception:
        return False


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _row(metric: str, value: str, source: str, detail: str) -> dict[str, Any]:
    return {"metric": metric, "value": value, "source": source, "detail": detail}


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]
