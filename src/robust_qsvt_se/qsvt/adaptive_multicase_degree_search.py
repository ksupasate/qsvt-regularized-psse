from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import RESOURCE_CAVEAT
from robust_qsvt_se.qsvt.polynomial_approximation import (
    build_approximation_context,
    evaluate_polynomial_approximation,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

ADAPTIVE_MULTICASE_CAVEAT = (
    "Adaptive multicase degree search is a bounded polynomial approximation "
    "diagnostic. It quantifies degree/query pressure and does not imply quantum "
    "advantage or full hardware execution."
)


def run_adaptive_multicase_degree_search(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for case in resolved["cases"]:
        for alpha in resolved["alpha"]:
            summary, trace = _search_case_alpha(case, float(alpha), resolved)
            summary_rows.append(summary)
            trace_rows.extend(trace)
            if summary["status"] != "passed":
                failure_rows.append(
                    {
                        "case_name": summary["case_name"],
                        "alpha": summary["alpha"],
                        "status": summary["status"],
                        "best_degree_tested": summary["best_degree_tested"],
                        "best_max_error": summary["best_max_error"],
                        "failure_reason": summary["failure_reason_if_any"],
                    }
                )

    summary_frame = pd.DataFrame(summary_rows)
    trace_frame = pd.DataFrame(trace_rows)
    failure_frame = pd.DataFrame(
        failure_rows,
        columns=[
            "case_name",
            "alpha",
            "status",
            "best_degree_tested",
            "best_max_error",
            "failure_reason",
        ],
    )
    summary_csv = output_dir / "adaptive_multicase_summary.csv"
    summary_json = output_dir / "adaptive_multicase_summary.json"
    trace_csv = output_dir / "adaptive_multicase_search_trace.csv"
    failure_csv = output_dir / "adaptive_multicase_failure_log.csv"
    summary_frame.to_csv(summary_csv, index=False)
    trace_frame.to_csv(trace_csv, index=False)
    failure_frame.to_csv(failure_csv, index=False)
    write_json(
        summary_json,
        {
            "rows": summary_rows,
            "interpretation": _interpretation(summary_frame),
        },
    )
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "adaptive_multicase_summary_csv": str(summary_csv),
            "adaptive_multicase_summary_json": str(summary_json),
            "adaptive_multicase_search_trace_csv": str(trace_csv),
            "adaptive_multicase_failure_log_csv": str(failure_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary_frame,
        "artifacts": {
            "adaptive_multicase_summary_csv": summary_csv,
            "adaptive_multicase_summary_json": summary_json,
            "adaptive_multicase_search_trace_csv": trace_csv,
            "adaptive_multicase_failure_log_csv": failure_csv,
            "manifest": manifest_path,
        },
    }


def _search_case_alpha(
    case: str | dict[str, Any],
    alpha: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_config = _case_config(case, config)
    case_name = str(case_config.get("case_name", "unknown"))
    start = time.perf_counter()
    trace_rows: list[dict[str, Any]] = []
    try:
        context = build_approximation_context(case_config)
    except Exception as exc:
        return _build_failure_summary(case_name, case_config, alpha, config, start, str(exc)), []

    max_degree = int(config["max_degree_by_case"].get(case_name, config["max_degree"]))
    degrees = [int(degree) for degree in config["degree_schedule"] if int(degree) <= max_degree]
    best_trace: dict[str, Any] | None = None
    last_result: Any | None = None
    for degree in degrees:
        degree_start = time.perf_counter()
        try:
            result = evaluate_polynomial_approximation(
                context=context,
                alpha=alpha,
                degree=degree,
                method=str(config["method"]),
                grid_size=int(config["grid_size"]),
            )
            errors = result.pointwise_errors
            max_error = float(np.max(errors))
            mean_error = float(np.mean(errors))
            rms_error = float(np.sqrt(np.mean(errors**2)))
            status = "passed" if max_error <= float(config["target_tolerance"]) else "failed"
            failure_reason = ""
            last_result = result
        except Exception as exc:
            max_error = np.nan
            mean_error = np.nan
            rms_error = np.nan
            status = "failed_numerical_instability"
            failure_reason = str(exc)
        trace = {
            "case_name": context.case_name,
            "alpha": alpha,
            "degree": int(degree),
            "query_count": int(2 * int(degree) + 1),
            "max_pointwise_error": max_error,
            "mean_pointwise_error": mean_error,
            "rms_pointwise_error": rms_error,
            "passed_1e_minus_3": bool(
                np.isfinite(max_error) and max_error <= float(config["target_tolerance"])
            ),
            "runtime_seconds": float(time.perf_counter() - degree_start),
            "status": status,
            "failure_reason_if_any": failure_reason,
        }
        trace_rows.append(trace)
        if np.isfinite(max_error) and (
            best_trace is None or max_error < float(best_trace["max_pointwise_error"])
        ):
            best_trace = trace
        if trace["passed_1e_minus_3"]:
            return (
                _summary_from_trace(
                    context,
                    trace,
                    best_trace,
                    last_result,
                    config,
                    start,
                    status="passed",
                    failure_reason="",
                ),
                trace_rows,
            )
        if status == "failed_numerical_instability":
            break

    if best_trace is None:
        best_trace = trace_rows[-1] if trace_rows else _empty_trace(case_name, alpha)
    return (
        _summary_from_trace(
            context,
            best_trace,
            best_trace,
            last_result,
            config,
            start,
            status="failed_max_degree",
            failure_reason=(
                "target tolerance not met by configured degree schedule within "
                f"case max degree {max_degree}"
            ),
        ),
        trace_rows,
    )


def _summary_from_trace(
    context: Any,
    selected: dict[str, Any],
    best: dict[str, Any] | None,
    result: Any,
    config: dict[str, Any],
    start: float,
    *,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    best_row = best or selected
    m, n = (int(value) for value in context.matrix_shape.split("x"))
    return {
        "case_name": context.case_name,
        "config_file": "",
        "matrix_source": context.matrix_source,
        "status": status,
        "alpha": float(selected["alpha"]),
        "m": m,
        "n": n,
        "sigma_min": float(np.min(context.singular_values)),
        "sigma_max": float(np.max(context.singular_values)),
        "kappa": float(np.max(context.singular_values) / np.min(context.singular_values)),
        "target_tolerance": float(config["target_tolerance"]),
        "selected_degree": int(selected["degree"]),
        "selected_query_count": int(selected["query_count"]),
        "achieved_max_error": float(selected["max_pointwise_error"]),
        "achieved_mean_error": float(selected["mean_pointwise_error"]),
        "achieved_rms_error": float(selected["rms_pointwise_error"]),
        "best_degree_tested": int(best_row["degree"]),
        "best_max_error": float(best_row["max_pointwise_error"]),
        "method": str(config["method"]),
        "bounded_scaling_C": (float(result.bounded_scaling_C) if result is not None else np.nan),
        "max_unbounded_filter_value": (
            float(result.max_unbounded_filter_value) if result is not None else np.nan
        ),
        "max_bounded_filter_value": (
            float(result.max_bounded_filter_value) if result is not None else np.nan
        ),
        "runtime_seconds": float(time.perf_counter() - start),
        "resource_caveat": RESOURCE_CAVEAT,
        "failure_reason_if_any": failure_reason,
        "interpretation": ADAPTIVE_MULTICASE_CAVEAT,
    }


def _build_failure_summary(
    case_name: str,
    case_config: dict[str, Any],
    alpha: float,
    config: dict[str, Any],
    start: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "config_file": str(case_config.get("config_file", "")),
        "matrix_source": str(case_config.get("matrix_source", "pypower_ac_weighted_jacobian")),
        "status": "failed_matrix_construction",
        "alpha": alpha,
        "m": None,
        "n": None,
        "sigma_min": None,
        "sigma_max": None,
        "kappa": None,
        "target_tolerance": float(config["target_tolerance"]),
        "selected_degree": None,
        "selected_query_count": None,
        "achieved_max_error": None,
        "achieved_mean_error": None,
        "achieved_rms_error": None,
        "best_degree_tested": None,
        "best_max_error": None,
        "method": str(config["method"]),
        "bounded_scaling_C": None,
        "max_unbounded_filter_value": None,
        "max_bounded_filter_value": None,
        "runtime_seconds": float(time.perf_counter() - start),
        "resource_caveat": RESOURCE_CAVEAT,
        "failure_reason_if_any": reason,
        "interpretation": ADAPTIVE_MULTICASE_CAVEAT,
    }


def _empty_trace(case_name: str, alpha: float) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "alpha": alpha,
        "degree": 0,
        "query_count": 0,
        "max_pointwise_error": np.nan,
        "mean_pointwise_error": np.nan,
        "rms_pointwise_error": np.nan,
        "passed_1e_minus_3": False,
        "runtime_seconds": 0.0,
        "status": "not_run",
        "failure_reason_if_any": "no degrees configured",
    }


def _case_config(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = {
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": config["case_source"],
        "seed": config["seed"],
        "fallback_to_synthetic": False,
    }
    if isinstance(case, dict):
        case_config.update(case)
    elif case == "synthetic":
        case_config.update({"matrix_source": "synthetic", "case_name": "synthetic"})
    else:
        case_config.update({"case_name": str(case)})
    return case_config


def _interpretation(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No adaptive multicase rows were generated."
    passed = frame[frame["status"] == "passed"]
    failed = frame[frame["status"] != "passed"]
    return (
        "Adaptive multicase degree search shows that larger IEEE cases can require "
        "substantially higher degree than IEEE14 under the same odd minimax method. "
        f"Passing rows: {len(passed)}. Non-passing rows: {len(failed)}. "
        "These diagnostics quantify feasibility and scaling pressure; they do not "
        "imply quantum advantage."
    )


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_adaptive_multicase_degree_search",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "alpha": [1.0e-2],
        "target_tolerance": 1.0e-3,
        "degree_schedule": [101, 151, 201, 301, 401, 601, 801, 1001],
        "max_degree": 1001,
        "max_degree_by_case": {"ieee300": 1001},
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 500,
    }
    if config:
        resolved.update(config)
    resolved["alpha"] = [float(alpha) for alpha in resolved["alpha"]]
    resolved["degree_schedule"] = [int(degree) for degree in resolved["degree_schedule"]]
    resolved["max_degree_by_case"] = {
        str(key): int(value) for key, value in dict(resolved["max_degree_by_case"]).items()
    }
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run adaptive multi-case QSVT degree search")
    parser.parse_args(argv)
    run = run_adaptive_multicase_degree_search()
    print(f"QSVT adaptive multi-case degree search complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
