from __future__ import annotations

import argparse
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


def build_multicase_approximation_diagnostics(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in resolved["cases"]:
        row = _case_row(case, resolved)
        rows.append(row)
        if row["status"] != "ok":
            failures.append(
                {
                    "case_name": row["case_name"],
                    "matrix_source": row["matrix_source"],
                    "failure_reason": row["failure_reason_if_any"],
                }
            )
    frame = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures, columns=["case_name", "matrix_source", "failure_reason"])
    summary_csv = output_dir / "multicase_approximation_summary.csv"
    summary_json = output_dir / "multicase_approximation_summary.json"
    failure_csv = output_dir / "failure_log.csv"
    frame.to_csv(summary_csv, index=False)
    failure_frame.to_csv(failure_csv, index=False)
    write_json(summary_json, {"rows": rows, "failures": failures})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "multicase_approximation_summary_csv": str(summary_csv),
            "multicase_approximation_summary_json": str(summary_json),
            "failure_log_csv": str(failure_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "multicase_approximation_summary_csv": summary_csv,
            "multicase_approximation_summary_json": summary_json,
            "failure_log_csv": failure_csv,
            "manifest": manifest_path,
        },
    }


def _case_row(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = _case_config(case, config)
    case_name = str(case_config.get("case_name", "unknown"))
    alpha = float(case_config.get("alpha", config["alpha"]))
    degree = int(case_config.get("degree", config["degree"]))
    method = str(case_config.get("method", config["method"]))
    try:
        context = build_approximation_context(case_config)
        result = evaluate_polynomial_approximation(
            context=context,
            alpha=alpha,
            degree=degree,
            method=method,
            grid_size=int(config["grid_size"]),
        )
        m, n = (int(value) for value in context.matrix_shape.split("x"))
        max_error = float(np.max(result.pointwise_errors))
        return {
            "case_name": case_name,
            "config_file": str(case_config.get("config_file", "")),
            "matrix_source": context.matrix_source,
            "status": "ok",
            "m": m,
            "n": n,
            "sigma_min": float(np.min(context.singular_values)),
            "sigma_max": float(np.max(context.singular_values)),
            "kappa": float(np.max(context.singular_values) / np.min(context.singular_values)),
            "alpha": alpha,
            "degree": int(result.degree),
            "method": method,
            "max_pointwise_error": max_error,
            "mean_pointwise_error": float(np.mean(result.pointwise_errors)),
            "query_count_estimate": int(2 * result.degree + 1),
            "resource_caveat": RESOURCE_CAVEAT,
            "failure_reason_if_any": "",
        }
    except Exception as exc:
        return {
            "case_name": case_name,
            "config_file": str(case_config.get("config_file", "")),
            "matrix_source": str(case_config.get("matrix_source", "pypower_ac_weighted_jacobian")),
            "status": "failed",
            "m": None,
            "n": None,
            "sigma_min": None,
            "sigma_max": None,
            "kappa": None,
            "alpha": alpha,
            "degree": degree,
            "method": method,
            "max_pointwise_error": None,
            "mean_pointwise_error": None,
            "query_count_estimate": int(2 * degree + 1),
            "resource_caveat": RESOURCE_CAVEAT,
            "failure_reason_if_any": str(exc),
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


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_multicase_approximation_diagnostics",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "degree": 101,
        "method": "odd_chebyshev_minimax_lp",
        "grid_size": 600,
    }
    if config:
        resolved.update(config)
    if not list(resolved["cases"]):
        raise ValueError("cases must be non-empty")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-case QSVT approximation diagnostics")
    parser.parse_args(argv)
    run = build_multicase_approximation_diagnostics()
    print(f"QSVT multi-case approximation diagnostics complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
