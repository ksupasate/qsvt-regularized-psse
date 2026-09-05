"""Task B workload: sparse-access summary for generated weighted PSSE Jacobians.

For each benchmark case this builds the weighted Jacobian via the existing
engineering-system path, wraps it in a validated
:class:`robust_qsvt_se.qsvt.sparse_access.SparseAccessModel`, runs an exhaustive
exact-lookup validation, and writes the paper-ready sparse-access tables. The
sparse access is a classical emulator with exact CSR lookups; it is not a
reversible quantum oracle circuit and not a quantum-hardware run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.paper.selected_observable_common import (
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
    write_workload_manifest,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system
from robust_qsvt_se.qsvt.sparse_access import (
    SPARSE_ACCESS_LIMITATION,
    build_sparse_access_model,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_CASES = ("ieee14", "ieee30", "ieee57", "ieee118")

SUMMARY_COLUMNS = [
    "case",
    "matrix_source",
    "shape_rows",
    "shape_cols",
    "nnz",
    "density",
    "max_row_nnz",
    "mean_row_nnz",
    "index_qubits",
    "value_precision_bits",
    "value_register_qubits",
    "access_status",
    "reversible_oracle_synthesized",
    "notes",
]

VALIDATION_COLUMNS = [
    "case",
    "matrix_source",
    "entries_checked",
    "value_max_abs_error",
    "col_index_mismatches",
    "max_row_nnz_correct",
    "invalid_index_raises",
    "access_status",
    "reversible_oracle_synthesized",
    "notes",
]


def run_sparse_access_workload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for case in resolved["cases"]:
        system, matrix_source = build_engineering_system(
            {
                "case_name": case,
                "case_source": resolved["case_source"],
                "matrix_source": "weighted_jacobian",
                "seed": int(resolved["seed"]),
            }
        )
        model = build_sparse_access_model(
            system.H_tilde,
            case=case,
            matrix_source=matrix_source,
            value_precision_bits=int(resolved["value_precision_bits"]),
        )
        summary_rows.append(model.summary_row())
        validation_rows.append(model.validate_against_dense_or_csr(reference=system.H_tilde))

    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    validation_frame = pd.DataFrame(validation_rows, columns=VALIDATION_COLUMNS)

    summary_csv = output_dir / "sparse_access_summary.csv"
    summary_json = output_dir / "sparse_access_summary.json"
    validation_csv = output_dir / "sparse_access_validation.csv"
    report_md = output_dir / "sparse_access_report.md"

    rows_to_table(summary_rows, summary_csv, SUMMARY_COLUMNS)
    rows_to_table(validation_rows, validation_csv, VALIDATION_COLUMNS)
    write_json(summary_json, {"claim_boundary": WORKLOAD_CLAIM_BOUNDARY, "rows": summary_rows})
    report_text = _report_markdown(summary_frame, validation_frame, resolved)
    assert_safe(report_text)
    report_md.write_text(report_text, encoding="utf-8")

    artifacts = {
        "sparse_access_summary_csv": summary_csv,
        "sparse_access_summary_json": summary_json,
        "sparse_access_validation_csv": validation_csv,
        "sparse_access_report_md": report_md,
    }
    manifest = write_workload_manifest(
        output_dir=output_dir,
        artifact_name="sparse_access_workload",
        description=(
            "Classical sparse-access emulator summary for generated weighted PSSE Jacobians, "
            "with exhaustive exact-lookup validation. Modeled sparse-access pathway; not a "
            "reversible quantum oracle circuit and not a quantum-hardware run."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[
            f"build_engineering_system:{case}:weighted_jacobian" for case in resolved["cases"]
        ],
        reran_long_experiments=False,
        aggregated_from_existing=False,
        extra={
            "cases": list(resolved["cases"]),
            "value_precision_bits": int(resolved["value_precision_bits"]),
            "access_status": "validated_exact_lookup",
            "reversible_oracle_synthesized": False,
            "manifest_name": "sparse_access_manifest.json",
        },
        manifest_name="sparse_access_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "validation": validation_frame,
        "artifacts": artifacts,
    }


def _report_markdown(
    summary: pd.DataFrame, validation: pd.DataFrame, resolved: dict[str, Any]
) -> str:
    lines = [
        "# Sparse-Access Model for Weighted PSSE Jacobians",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "## Scope",
        "",
        f"- {SPARSE_ACCESS_LIMITATION}",
        "- Oracle interfaces modeled: `O_col: |i,k> -> |i,c(i,k)>` and "
        "`O_val: |i,j,0> -> |i,j,H~_ij>`.",
        "- Index/value lookups are exact CSR table lookups, validated against the source "
        "weighted Jacobian (every structural nonzero checked).",
        "",
        "## Sparse-Access Summary",
        "",
        "| Case | Source | Shape | nnz | Density | Max row nnz | Mean row nnz | Index qubits | "
        "Value bits | Access status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['case']} | {row['matrix_source']} | "
            f"{int(row['shape_rows'])}x{int(row['shape_cols'])} | {int(row['nnz'])} | "
            f"{float(row['density']):.4f} | {int(row['max_row_nnz'])} | "
            f"{float(row['mean_row_nnz']):.2f} | {int(row['index_qubits'])} | "
            f"{int(row['value_precision_bits'])} | {row['access_status']} |"
        )
    lines += [
        "",
        "## Exact-Lookup Validation",
        "",
        "| Case | Entries checked | Max value error | Col mismatches | Max-row-nnz ok | "
        "Invalid index raises |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in validation.iterrows():
        lines.append(
            f"| {row['case']} | {int(row['entries_checked'])} | "
            f"{float(row['value_max_abs_error']):.2e} | {int(row['col_index_mismatches'])} | "
            f"{row['max_row_nnz_correct']} | {row['invalid_index_raises']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The weighted PSSE Jacobians are sparse (low mean row nnz), so a row-sparse access "
        "model is the natural block-encoding input.",
        "- The emulator reproduces every stored value exactly and rejects out-of-range "
        "indices, so the modeled access pathway is internally consistent.",
        "- `reversible_oracle_synthesized` is `False` for every case: this is a modeled "
        "sparse-access pathway, not a compiled reversible circuit and not a quantum-hardware "
        "run.",
        "",
    ]
    return "\n".join(lines)


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(WORKLOAD_DIR),
        "cases": list(DEFAULT_CASES),
        "case_source": "pypower",
        "seed": 123,
        "value_precision_bits": 8,
        "command": "run_sparse_access_workload",
    }
    if config:
        resolved.update(config)
    resolved["cases"] = [str(case) for case in resolved["cases"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the sparse-access workload tables")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--case-source", default="pypower")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--value-precision-bits", type=int, default=8)
    args = parser.parse_args(argv)
    run = run_sparse_access_workload(
        {
            "output_dir": args.output_dir,
            "cases": args.cases,
            "case_source": args.case_source,
            "seed": args.seed,
            "value_precision_bits": args.value_precision_bits,
            "command": "scripts/run_sparse_access_workload.py " + " ".join(argv or []),
        }
    )
    print(f"Sparse-access workload complete: {run['artifacts']['sparse_access_summary_csv']}")


if __name__ == "__main__":  # pragma: no cover
    main()
