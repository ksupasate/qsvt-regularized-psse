from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_extension_report import build_engineering_extension_summary
from robust_qsvt_se.qsvt.engineering_io import (
    CLAIM_BOUNDARY,
    current_command,
    git_commit,
    utc_timestamp,
)
from robust_qsvt_se.qsvt.paper_finalization import build_paper_ready_qsvt_tables
from robust_qsvt_se.utils.io import ensure_directory, write_json

PYQSP_PHASE_TARGET = "bounded_ridge_tikhonov_pyqsp"
PYQSP_BACKEND = "pyqsp_sym_qsp"
PYQSP_CANDIDATE = "coefficient_conditioned_chebyshev_degree_201_lambda_1e-04"
PYQSP_TOLERANCE = 1.0e-3


def finalize_qsvt_phase1_artifacts(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    if bool(resolved["regenerate_upstream"]):
        build_engineering_extension_summary({"output_dir": "outputs/qsvt_engineering_extension"})
        build_paper_ready_qsvt_tables({"output_dir": "outputs/paper_ready_qsvt_tables"})

    output_dir = ensure_directory(Path(resolved["output_dir"]))
    pyqsp_row = _pyqsp_pass_row(Path(resolved["phase_validation_summary_csv"]))
    table_index = _table_index(Path(resolved["paper_ready_table_dir"]))
    claim_delta = _claim_delta(pyqsp_row)
    summary = _summary_frame(pyqsp_row)

    summary_csv = output_dir / "phase1_finalization_summary.csv"
    summary_json = output_dir / "phase1_finalization_summary.json"
    summary_md = output_dir / "phase1_finalization_summary.md"
    claim_delta_csv = output_dir / "phase1_claim_delta.csv"
    table_index_csv = output_dir / "phase1_updated_table_index.csv"
    manifest_path = output_dir / "manifest.json"

    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": summary.to_dict(orient="records")})
    summary_md.write_text(_summary_markdown(pyqsp_row, table_index), encoding="utf-8")
    claim_delta.to_csv(claim_delta_csv, index=False)
    table_index.to_csv(table_index_csv, index=False)
    write_json(
        manifest_path,
        {
            "generated_at": utc_timestamp(),
            "command": current_command(),
            "git_commit": git_commit(),
            "input_config": resolved,
            "artifacts": {
                "phase1_finalization_summary_md": str(summary_md),
                "phase1_finalization_summary_csv": str(summary_csv),
                "phase1_finalization_summary_json": str(summary_json),
                "phase1_claim_delta_csv": str(claim_delta_csv),
                "phase1_updated_table_index_csv": str(table_index_csv),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )
    return {
        "output_dir": output_dir,
        "pyqsp_row": pyqsp_row,
        "summary": summary,
        "claim_delta": claim_delta,
        "table_index": table_index,
        "artifacts": {"manifest": manifest_path},
    }


def _pyqsp_pass_row(summary_csv: Path) -> dict[str, Any]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"missing pyqsp phase validation summary: {summary_csv}")
    frame = pd.read_csv(summary_csv)
    candidates = frame[
        (frame["backend_name"].astype(str) == PYQSP_BACKEND)
        & (frame["candidate_name"].astype(str) == PYQSP_CANDIDATE)
    ].copy()
    if candidates.empty:
        raise ValueError("pyqsp bounded Ridge/Tikhonov pass row was not found")
    passed = candidates[
        (candidates["passed_1e_minus_3_full_domain"] == True)  # noqa: E712
        & (candidates["passed_1e_minus_3_actual_singular_values"] == True)  # noqa: E712
    ]
    row = passed.iloc[0] if not passed.empty else candidates.iloc[0]
    full_error = float(row["phase_response_max_error_full_domain"])
    actual_error = float(row["phase_response_max_error_actual_singular_values_if_available"])
    if not np.isfinite(full_error) or full_error > PYQSP_TOLERANCE:
        raise ValueError(
            "pyqsp bounded Ridge/Tikhonov row does not pass the declared full-domain tolerance"
        )
    return {
        "target": PYQSP_PHASE_TARGET,
        "backend": str(row["backend_name"]),
        "candidate": str(row["candidate_name"]),
        "degree": int(row["degree"]),
        "phase_count": int(row["phase_count"]),
        "input_basis": "Chebyshev",
        "full_domain_max_error": full_error,
        "actual_singular_value_max_error": actual_error,
        "tolerance": PYQSP_TOLERANCE,
        "status": "passed_scalar_full_domain",
        "claim_supported": "scalar full-domain phase-response validation",
        "limitation": "not hardware execution or block-encoded matrix execution",
    }


def _summary_frame(pyqsp_row: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "Phase 1",
                "verdict": "PASS",
                "statement": "Phase 1 PASS for scalar full-domain phase-response validation.",
                **pyqsp_row,
                "caveat": (
                    "This remains scalar phase-response validation only and does not "
                    "constitute hardware execution, full block-encoded matrix execution, "
                    "or quantum speedup evidence."
                ),
            }
        ]
    )


def _claim_delta(pyqsp_row: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "claim": (
                "The bounded Ridge/Tikhonov target passed scalar full-domain "
                "phase-response validation using pyqsp symmetric-QSP phases."
            ),
            "old_status": "unresolved_or_backend_specific_failure",
            "new_status": "supported",
            "supporting_outputs": (
                "outputs/qsvt_external_backend_phase_validation/"
                "external_backend_phase_validation_summary.csv"
            ),
            "recommended_wording": (
                "The bounded Ridge/Tikhonov target passed scalar full-domain "
                "phase-response validation with pyqsp symmetric-QSP phases."
            ),
            "avoid_wording": "The result demonstrates hardware execution or quantum speedup.",
        },
        {
            "claim": "Historical PennyLane and monomial-instability failures remain diagnostic.",
            "old_status": "latest_unresolved",
            "new_status": "historical_failure_superseded_by_pyqsp_phase_validation",
            "supporting_outputs": (
                "outputs/qsvt_phase_validation_stable_basis/; "
                "outputs/qsvt_stable_target_phase_validation/"
            ),
            "recommended_wording": (
                "PennyLane monomial-path failures are historical backend-specific "
                "diagnostics superseded by the pyqsp Chebyshev-basis pass."
            ),
            "avoid_wording": "Historical failures remain the latest final phase status.",
        },
    ]
    rows[0].update(
        {
            "backend": pyqsp_row["backend"],
            "degree": pyqsp_row["degree"],
            "phase_count": pyqsp_row["phase_count"],
            "full_domain_max_error": pyqsp_row["full_domain_max_error"],
            "actual_singular_value_max_error": pyqsp_row["actual_singular_value_max_error"],
        }
    )
    return pd.DataFrame(rows)


def _table_index(table_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(table_dir.glob("table_*.csv")):
        try:
            frame = pd.read_csv(path)
            status = "ok" if not frame.empty else "empty"
            rows.append(
                {
                    "table": path.stem,
                    "path": str(path),
                    "rows": len(frame),
                    "columns": "; ".join(frame.columns),
                    "status": status,
                }
            )
        except pd.errors.EmptyDataError:
            rows.append(
                {
                    "table": path.stem,
                    "path": str(path),
                    "rows": 0,
                    "columns": "",
                    "status": "empty",
                }
            )
    return pd.DataFrame(rows)


def _summary_markdown(pyqsp_row: dict[str, Any], table_index: pd.DataFrame) -> str:
    return f"""# QSVT Phase 1 Finalization Summary

Phase 1 PASS for scalar full-domain phase-response validation.

This remains scalar phase-response validation only and does not constitute
hardware execution, full block-encoded matrix execution, or quantum speedup
evidence.

## Latest Passing Row

- target: {pyqsp_row["target"]}
- backend: {pyqsp_row["backend"]}
- candidate: {pyqsp_row["candidate"]}
- input basis: {pyqsp_row["input_basis"]}
- degree: {pyqsp_row["degree"]}
- phase count: {pyqsp_row["phase_count"]}
- full-domain max error: {pyqsp_row["full_domain_max_error"]:.3e}
- actual singular value max error: {pyqsp_row["actual_singular_value_max_error"]:.3e}
- tolerance: {pyqsp_row["tolerance"]:.1e}
- status: {pyqsp_row["status"]}

Historical PennyLane and monomial-conversion failures remain preserved as
backend-specific diagnostics, but they are superseded as the latest final
paper-ready status by the pyqsp Chebyshev-basis full-domain pass.

## Updated Tables

{_markdown_table(table_index, ["table", "rows", "status"])}

## Claim Boundary

{CLAIM_BOUNDARY}
"""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(no rows)"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(subset.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    lines = [header, separator]
    for row in subset.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_phase1_finalization",
        "phase_validation_summary_csv": (
            "outputs/qsvt_external_backend_phase_validation/"
            "external_backend_phase_validation_summary.csv"
        ),
        "paper_ready_table_dir": "outputs/paper_ready_qsvt_tables",
        "regenerate_upstream": True,
    }
    if config:
        resolved.update(config)
    return resolved
