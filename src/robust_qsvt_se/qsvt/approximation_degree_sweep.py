from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.polynomial_approximation import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_DEGREE_GRID,
    approximation_summary_row,
    as_odd_degree,
    build_approximation_context,
    evaluate_polynomial_approximation,
    pointwise_rows,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

NO_STRICT_PASS_MESSAGE = (
    "No tested degree met the strict 1e-3 maximum pointwise-error tolerance "
    "under the configured approximation method."
)


def run_approximation_degree_sweep(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    context = build_approximation_context(resolved)
    summary_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for alpha in resolved["alpha"]:
        for degree in resolved["degree"]:
            result = evaluate_polynomial_approximation(
                context=context,
                alpha=float(alpha),
                degree=int(degree),
                method=str(resolved["method"]),
                grid_size=int(resolved["grid_size"]),
            )
            summary_rows.append(
                approximation_summary_row(
                    context=context,
                    alpha=float(alpha),
                    result=result,
                    tolerance=float(resolved["strict_tolerance"]),
                )
            )
            rows = pointwise_rows(
                context=context,
                alpha=float(alpha),
                result=result,
                include_values=True,
            )
            value_rows.extend(rows)
            error_rows.extend(
                {
                    key: row[key]
                    for key in (
                        "case_name",
                        "matrix_source",
                        "alpha",
                        "degree",
                        "method",
                        "evaluation_index",
                        "evaluation_kind",
                        "sigma_normalized",
                        "pointwise_error",
                    )
                }
                for row in rows
            )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["interpretation"] = _interpretation(summary_frame)
    error_frame = pd.DataFrame(error_rows)
    value_frame = pd.DataFrame(value_rows)

    summary_csv = output_dir / "degree_sweep_summary.csv"
    summary_json = output_dir / "degree_sweep_summary.json"
    error_csv = output_dir / "degree_sweep_pointwise_errors.csv"
    value_csv = output_dir / "degree_sweep_target_and_approx_values.csv"
    summary_frame.to_csv(summary_csv, index=False)
    error_frame.to_csv(error_csv, index=False)
    value_frame.to_csv(value_csv, index=False)
    write_json(
        summary_json,
        {"rows": summary_rows, "interpretation": _interpretation(summary_frame)},
    )
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "degree_sweep_summary_csv": str(summary_csv),
            "degree_sweep_summary_json": str(summary_json),
            "degree_sweep_pointwise_errors_csv": str(error_csv),
            "degree_sweep_target_and_approx_values_csv": str(value_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "degree_sweep_summary_csv": summary_csv,
            "degree_sweep_summary_json": summary_json,
            "degree_sweep_pointwise_errors_csv": error_csv,
            "degree_sweep_target_and_approx_values_csv": value_csv,
            "manifest": manifest_path,
        },
    }


def _interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No degree-sweep rows were generated."
    strict_pass = bool(frame["passed_tol_1e_minus_3"].any())
    trend_parts = []
    for alpha, group in frame.groupby("alpha"):
        ordered = group.sort_values("degree")
        first_error = float(ordered.iloc[0]["max_pointwise_error"])
        last_error = float(ordered.iloc[-1]["max_pointwise_error"])
        trend = "decreased" if last_error <= first_error else "did_not_decrease"
        trend_parts.append(
            f"alpha={alpha:g}: error {trend} from {first_error:.6g} to {last_error:.6g}"
        )
    suffix = (
        "At least one tested degree met the strict 1e-3 maximum pointwise-error tolerance."
        if strict_pass
        else NO_STRICT_PASS_MESSAGE
    )
    return "; ".join(trend_parts) + ". " + suffix


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_approximation_degree_sweep",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "fallback_to_synthetic": True,
        "alpha": DEFAULT_ALPHA_GRID,
        "degree": DEFAULT_DEGREE_GRID,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 1000,
        "strict_tolerance": 1.0e-3,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    if any(alpha <= 0.0 for alpha in resolved["alpha"]):
        raise ValueError("alpha values must be positive")
    resolved["degree"] = [as_odd_degree(int(degree)) for degree in resolved["degree"]]
    if int(resolved["grid_size"]) < 500:
        raise ValueError("grid_size must be at least 500")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run QSVT approximation degree sweep")
    parser.parse_args(argv)
    run = run_approximation_degree_sweep()
    print(f"QSVT approximation degree sweep complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
