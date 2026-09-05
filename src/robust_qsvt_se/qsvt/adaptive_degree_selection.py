from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import RESOURCE_CAVEAT
from robust_qsvt_se.qsvt.polynomial_approximation import (
    DEFAULT_ADAPTIVE_DEGREES,
    DEFAULT_ALPHA_GRID,
    as_odd_degree,
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json


def run_adaptive_degree_selection(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for alpha in resolved["alpha"]:
        for tolerance in resolved["target_tolerance"]:
            summary, trace = _select_degree(
                context=context,
                alpha=float(alpha),
                target_tolerance=float(tolerance),
                degrees=list(resolved["search_degrees"]),
                method=str(resolved["method"]),
                grid_size=int(resolved["grid_size"]),
                max_degree=int(resolved["max_degree"]),
            )
            summary_rows.append(summary)
            trace_rows.extend(trace)
    summary_frame = pd.DataFrame(summary_rows)
    trace_frame = pd.DataFrame(trace_rows)
    summary_csv = output_dir / "adaptive_degree_summary.csv"
    summary_json = output_dir / "adaptive_degree_summary.json"
    trace_csv = output_dir / "adaptive_search_trace.csv"
    summary_frame.to_csv(summary_csv, index=False)
    trace_frame.to_csv(trace_csv, index=False)
    write_json(summary_json, {"rows": summary_rows})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "adaptive_degree_summary_csv": str(summary_csv),
            "adaptive_degree_summary_json": str(summary_json),
            "adaptive_search_trace_csv": str(trace_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "adaptive_degree_summary_csv": summary_csv,
            "adaptive_degree_summary_json": summary_json,
            "adaptive_search_trace_csv": trace_csv,
            "manifest": manifest_path,
        },
    }


def _select_degree(
    *,
    context: Any,
    alpha: float,
    target_tolerance: float,
    degrees: list[int],
    method: str,
    grid_size: int,
    max_degree: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for degree in degrees:
        if int(degree) > max_degree:
            continue
        try:
            result = evaluate_polynomial_approximation(
                context=context,
                alpha=alpha,
                degree=int(degree),
                method=method,
                grid_size=grid_size,
            )
            errors = result.pointwise_errors
            max_error = float(np.max(errors))
            row = {
                "case_name": context.case_name,
                "matrix_source": context.matrix_source,
                "alpha": alpha,
                "target_tolerance": target_tolerance,
                "degree": int(result.degree),
                "query_count_estimate": int(2 * result.degree + 1),
                "max_pointwise_error": max_error,
                "mean_pointwise_error": float(np.mean(errors)),
                "rms_pointwise_error": float(np.sqrt(np.mean(errors**2))),
                "status": "passed" if max_error <= target_tolerance else "candidate_failed",
                "failure_reason": "",
                "approximation_method": method,
                "bounded_scaling_C": result.bounded_scaling_C,
            }
        except Exception as exc:
            row = {
                "case_name": context.case_name,
                "matrix_source": context.matrix_source,
                "alpha": alpha,
                "target_tolerance": target_tolerance,
                "degree": int(degree),
                "query_count_estimate": int(2 * int(degree) + 1),
                "max_pointwise_error": np.nan,
                "mean_pointwise_error": np.nan,
                "rms_pointwise_error": np.nan,
                "status": "failed_numerical_instability",
                "failure_reason": str(exc),
                "approximation_method": method,
                "bounded_scaling_C": 1.0,
            }
        trace.append(row)
        if np.isfinite(row["max_pointwise_error"]) and (
            best is None or row["max_pointwise_error"] < best["max_pointwise_error"]
        ):
            best = row
        if row["status"] == "passed":
            return _summary_row(context, row, target_tolerance, max_degree, "passed", ""), trace
    if best is None:
        return (
            _summary_row(
                context,
                trace[-1],
                target_tolerance,
                max_degree,
                "failed_numerical_instability",
                "all candidate degrees failed numerically",
            ),
            trace,
        )
    return (
        _summary_row(
            context,
            best,
            target_tolerance,
            max_degree,
            "failed_max_degree",
            "target tolerance not met by configured odd degrees up to max_degree",
        ),
        trace,
    )


def _summary_row(
    context: Any,
    row: dict[str, Any],
    target_tolerance: float,
    max_degree: int,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "case_name": context.case_name,
        "matrix_source": context.matrix_source,
        "alpha": float(row["alpha"]),
        "target_tolerance": float(target_tolerance),
        "selected_degree": int(row["degree"]),
        "max_degree": int(max_degree),
        "selected_query_count": int(row["query_count_estimate"]),
        "achieved_max_error": float(row["max_pointwise_error"]),
        "achieved_mean_error": float(row["mean_pointwise_error"]),
        "achieved_rms_error": float(row["rms_pointwise_error"]),
        "status": status,
        "failure_reason": failure_reason,
        "approximation_method": row["approximation_method"],
        "bounded_scaling_C": float(row["bounded_scaling_C"]),
        "resource_caveat": RESOURCE_CAVEAT,
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_adaptive_degree_selection",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": DEFAULT_ALPHA_GRID,
        "target_tolerance": [1.0e-2, 5.0e-3, 1.0e-3, 1.0e-4],
        "max_degree": 301,
        "search_degrees": DEFAULT_ADAPTIVE_DEGREES,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 1000,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    resolved["target_tolerance"] = [float(tolerance) for tolerance in resolved["target_tolerance"]]
    resolved["search_degrees"] = [
        as_odd_degree(int(degree))
        for degree in resolved.get("search_degrees", DEFAULT_ADAPTIVE_DEGREES)
    ]
    resolved["max_degree"] = as_odd_degree(int(resolved["max_degree"]))
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run adaptive QSVT degree selection")
    parser.parse_args(argv)
    run = run_adaptive_degree_selection()
    print(f"QSVT adaptive degree selection complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
