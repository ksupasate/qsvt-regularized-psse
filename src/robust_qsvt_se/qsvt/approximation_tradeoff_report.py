from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.approximation_degree_sweep import run_approximation_degree_sweep
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory, write_json

TRADEOFF_WORDING = (
    "The degree-error-query trade-off indicates that tighter pointwise-error "
    "tolerance may require substantially higher polynomial degree and query count, "
    "especially for smaller alpha values or sharper spectral filters."
)
STRICT_FAIL_WORDING = (
    "The current fallback approximation did not meet the strict 1e-3 maximum "
    "pointwise-error tolerance over the tested grid. This motivates either "
    "higher-degree approximation, improved minimax/QSP synthesis, or reporting "
    "the result as diagnostic rather than validated high-accuracy phase synthesis."
)


def build_approximation_tradeoff_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    sweep_run = run_approximation_degree_sweep(
        {
            **resolved,
            "output_dir": str(output_dir / "_degree_sweep_source"),
            "degree": resolved["degree"],
            "alpha": resolved["alpha"],
            "method": resolved["method"],
            "grid_size": resolved["grid_size"],
        }
    )
    sweep = sweep_run["summary"].copy()
    tradeoff_rows = _tradeoff_rows(sweep, list(resolved["tolerances"]))
    tradeoff_frame = pd.DataFrame(tradeoff_rows)
    error_vs_degree = sweep[
        [
            "case_name",
            "matrix_source",
            "alpha",
            "degree",
            "approximation_method",
            "max_pointwise_error",
            "mean_pointwise_error",
            "rms_pointwise_error",
        ]
    ].copy()
    query_vs_degree = sweep[
        [
            "case_name",
            "matrix_source",
            "alpha",
            "degree",
            "query_count_estimate",
            "passed_tol_1e_minus_2",
            "passed_tol_5e_minus_3",
            "passed_tol_1e_minus_3",
            "passed_tol_1e_minus_4",
        ]
    ].copy()

    summary_csv = output_dir / "tradeoff_summary.csv"
    summary_json = output_dir / "tradeoff_summary.json"
    report_md = output_dir / "tradeoff_report.md"
    error_csv = output_dir / "error_vs_degree.csv"
    query_csv = output_dir / "query_vs_degree.csv"
    tradeoff_frame.to_csv(summary_csv, index=False)
    error_vs_degree.to_csv(error_csv, index=False)
    query_vs_degree.to_csv(query_csv, index=False)
    write_json(summary_json, {"rows": tradeoff_rows})
    report_md.write_text(
        _tradeoff_markdown(
            sweep=sweep,
            tradeoff=tradeoff_frame,
            methods=[str(resolved["method"])],
            tolerances=list(resolved["tolerances"]),
        ),
        encoding="utf-8",
    )
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "tradeoff_summary_csv": str(summary_csv),
            "tradeoff_summary_json": str(summary_json),
            "tradeoff_report_md": str(report_md),
            "error_vs_degree_csv": str(error_csv),
            "query_vs_degree_csv": str(query_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": tradeoff_frame,
        "artifacts": {
            "tradeoff_summary_csv": summary_csv,
            "tradeoff_summary_json": summary_json,
            "tradeoff_report_md": report_md,
            "error_vs_degree_csv": error_csv,
            "query_vs_degree_csv": query_csv,
            "manifest": manifest_path,
        },
    }


def _tradeoff_rows(sweep: pd.DataFrame, tolerances: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha, group in sweep.groupby("alpha"):
        ordered = group.sort_values("max_pointwise_error")
        best = ordered.iloc[0]
        for tolerance in tolerances:
            passing = group[group["max_pointwise_error"] <= float(tolerance)].sort_values("degree")
            if passing.empty:
                selected = best
                status = "no_tested_degree_passed"
            else:
                selected = passing.iloc[0]
                status = "passed"
            rows.append(
                {
                    "case_name": selected["case_name"],
                    "matrix_source": selected["matrix_source"],
                    "alpha": float(alpha),
                    "target_tolerance": float(tolerance),
                    "method": selected["approximation_method"],
                    "smallest_passing_degree": (
                        None if status != "passed" else int(selected["degree"])
                    ),
                    "best_degree": int(best["degree"]),
                    "selected_degree_for_row": int(selected["degree"]),
                    "achieved_max_error": float(selected["max_pointwise_error"]),
                    "best_max_error": float(best["max_pointwise_error"]),
                    "query_count_estimate": int(selected["query_count_estimate"]),
                    "status": status,
                    "claim_safe_interpretation": TRADEOFF_WORDING,
                }
            )
    return rows


def _tradeoff_markdown(
    *,
    sweep: pd.DataFrame,
    tradeoff: pd.DataFrame,
    methods: list[str],
    tolerances: list[float],
) -> str:
    alpha_values = ", ".join(f"{alpha:g}" for alpha in sorted(sweep["alpha"].unique()))
    degree_values = ", ".join(str(int(degree)) for degree in sorted(sweep["degree"].unique()))
    methods_text = ", ".join(methods)
    best_lines = []
    for alpha, group in sweep.groupby("alpha"):
        best = group.sort_values("max_pointwise_error").iloc[0]
        best_lines.append(
            f"- alpha `{alpha:g}`: degree `{int(best['degree'])}`, "
            f"query count `{int(best['query_count_estimate'])}`, "
            f"max error `{float(best['max_pointwise_error']):.6g}`"
        )
    tolerance_lines = []
    for _, row in tradeoff.iterrows():
        if row["status"] == "passed":
            tolerance_lines.append(
                f"- alpha `{row['alpha']:g}`, tolerance `{row['target_tolerance']:.1e}`: "
                f"degree `{int(row['smallest_passing_degree'])}`"
            )
        else:
            tolerance_lines.append(
                f"- alpha `{row['alpha']:g}`, tolerance `{row['target_tolerance']:.1e}`: "
                "no tested degree passed"
            )
    strict_pass = bool(sweep["passed_tol_1e_minus_3"].any())
    strict_text = (
        "At least one tested degree met the strict 1e-3 tolerance."
        if strict_pass
        else STRICT_FAIL_WORDING
    )
    return f"""# QSVT Approximation Trade-Off Report

## Methods Tested

{methods_text}

## Alpha Values Tested

{alpha_values}

## Degree Values Tested

{degree_values}

## Tolerances Tested

{", ".join(f"{tolerance:.1e}" for tolerance in tolerances)}

## Best Result Per Alpha

{chr(10).join(best_lines)}

## Smallest Degree Passing Each Tolerance

{chr(10).join(tolerance_lines)}

## Query-Count Implications

The query-count proxy is `2 * degree + 1`, so tighter tolerances directly raise
the QSVT query-count estimate in this diagnostic model.

{TRADEOFF_WORDING}

## Strict 1e-3 Status

{strict_text}

## Claim-Safe Interpretation

These approximation diagnostics support resource-aware feasibility discussion.
They are polynomial diagnostics unless an explicit phase-synthesis report says
otherwise. They do not demonstrate quantum speedup, quantum advantage, full
hardware execution, PMU/SCADA field-data validation, or QSVT superiority over
Ridge/Tikhonov under the same alpha.
"""


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_approximation_tradeoff",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": [1.0e-4, 1.0e-2, 1.0],
        "degree": [15, 25, 35, 51, 71, 101, 151, 201],
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 1000,
        "tolerances": [1.0e-2, 5.0e-3, 1.0e-3, 1.0e-4],
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    resolved["degree"] = [int(degree) for degree in resolved["degree"]]
    resolved["tolerances"] = [float(tolerance) for tolerance in resolved["tolerances"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT approximation trade-off report")
    parser.parse_args(argv)
    run = build_approximation_tradeoff_report()
    print(f"QSVT approximation trade-off report complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
