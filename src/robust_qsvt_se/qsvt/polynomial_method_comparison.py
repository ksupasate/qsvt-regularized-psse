from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.polynomial_approximation import (
    DEFAULT_ALPHA_GRID,
    approximation_summary_row,
    as_odd_degree,
    build_approximation_context,
    evaluate_polynomial_approximation,
    pointwise_rows,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

DEFAULT_COMPARISON_METHODS = [
    "odd_chebyshev_reduced_y",
    "odd_chebyshev_ls",
    "odd_chebyshev_minimax_lp",
    "chebyshev_interpolation_positive",
]
DEFAULT_COMPARISON_DEGREES = [35, 71, 101]


def compare_polynomial_approximation_methods(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for alpha in resolved["alpha"]:
        for degree in resolved["degree"]:
            for method in resolved["methods"]:
                try:
                    result = evaluate_polynomial_approximation(
                        context=context,
                        alpha=float(alpha),
                        degree=int(degree),
                        method=str(method),
                        grid_size=int(resolved["grid_size"]),
                    )
                    row = approximation_summary_row(
                        context=context,
                        alpha=float(alpha),
                        result=result,
                        tolerance=1.0e-3,
                    )
                except Exception as exc:
                    row = _failed_row(context, float(alpha), int(degree), str(method), str(exc))
                    summary_rows.append(row)
                    continue
                summary_rows.append(row)
                error_rows.extend(
                    pointwise_rows(
                        context=context,
                        alpha=float(alpha),
                        result=result,
                        include_values=False,
                    )
                )
    summary_frame = pd.DataFrame(summary_rows)
    error_frame = pd.DataFrame(error_rows)
    summary_csv = output_dir / "method_comparison_summary.csv"
    summary_json = output_dir / "method_comparison_summary.json"
    error_csv = output_dir / "method_pointwise_errors.csv"
    summary_frame.to_csv(summary_csv, index=False)
    error_frame.to_csv(error_csv, index=False)
    write_json(
        summary_json,
        {
            "rows": summary_rows,
            "best_by_alpha": _best_by_alpha(summary_frame),
            "best_1e_minus_3": _best_passing(summary_frame),
        },
    )
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "method_comparison_summary_csv": str(summary_csv),
            "method_comparison_summary_json": str(summary_json),
            "method_pointwise_errors_csv": str(error_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "method_comparison_summary_csv": summary_csv,
            "method_comparison_summary_json": summary_json,
            "method_pointwise_errors_csv": error_csv,
            "manifest": manifest_path,
        },
    }


def _failed_row(
    context: Any,
    alpha: float,
    degree: int,
    method: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_name": context.case_name,
        "matrix_source": context.matrix_source,
        "matrix_shape": context.matrix_shape,
        "alpha": alpha,
        "degree": as_odd_degree(degree),
        "sigma_min": float(np.min(context.singular_values)),
        "sigma_max": float(np.max(context.singular_values)),
        "method": method,
        "approximation_method": method,
        "parity": "unknown",
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "max_error_on_singular_values": np.nan,
        "mean_error_on_singular_values": np.nan,
        "numerical_stability_status": "failed_numerical_instability",
        "requires_optional_dependency": False,
        "dependency_name_if_any": "",
        "query_count_estimate": int(2 * as_odd_degree(degree) + 1),
        "passed_1e_minus_3": False,
        "caveat": f"Method failed numerically: {reason}",
    }


def _best_by_alpha(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for alpha, group in frame.dropna(subset=["max_pointwise_error"]).groupby("alpha"):
        best = group.sort_values("max_pointwise_error").iloc[0]
        rows.append(
            {
                "alpha": float(alpha),
                "method": str(best["method"]),
                "degree": int(best["degree"]),
                "max_pointwise_error": float(best["max_pointwise_error"]),
                "passed_1e_minus_3": bool(best["passed_1e_minus_3"]),
            }
        )
    return rows


def _best_passing(frame: pd.DataFrame) -> list[dict[str, Any]]:
    passed = frame[frame.get("passed_1e_minus_3", False) == True]  # noqa: E712
    if passed.empty:
        return []
    rows = []
    for alpha, group in passed.groupby("alpha"):
        best = group.sort_values(["degree", "max_pointwise_error"]).iloc[0]
        rows.append(
            {
                "alpha": float(alpha),
                "method": str(best["method"]),
                "degree": int(best["degree"]),
                "max_pointwise_error": float(best["max_pointwise_error"]),
            }
        )
    return rows


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_polynomial_method_comparison",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": DEFAULT_ALPHA_GRID,
        "degree": DEFAULT_COMPARISON_DEGREES,
        "methods": DEFAULT_COMPARISON_METHODS,
        "grid_size": 1000,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    resolved["degree"] = [as_odd_degree(int(degree)) for degree in resolved["degree"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare QSVT polynomial approximation methods")
    parser.parse_args(argv)
    run = compare_polynomial_approximation_methods()
    print(f"QSVT polynomial method comparison complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
