from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    DEFAULT_DEGREES,
    DEFAULT_EPSILON,
    build_engineering_system,
    estimate_degree_and_queries,
    matrix_density,
    singular_summary,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

READOUT_CAVEAT = (
    "Full-vector readout is not assumed; resource rows are proxy diagnostics for "
    "selected observables or components."
)


def build_multicase_resource_diagnostics(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in resolved["cases"]:
        row = _case_resource_row(case, resolved)
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
    csv_path = output_dir / "multicase_resource_summary.csv"
    json_path = output_dir / "multicase_resource_summary.json"
    failure_path = output_dir / "failure_log.csv"
    frame.to_csv(csv_path, index=False)
    failure_frame.to_csv(failure_path, index=False)
    write_json(json_path, {"rows": rows, "failures": failures})
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "multicase_resource_summary_csv": str(csv_path),
            "multicase_resource_summary_json": str(json_path),
            "failure_log_csv": str(failure_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "multicase_resource_summary_csv": csv_path,
            "multicase_resource_summary_json": json_path,
            "failure_log_csv": failure_path,
            "manifest": manifest_path,
        },
    }


def _case_resource_row(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = _case_config(case, config)
    case_name = str(case_config.get("case_name", "unknown"))
    alpha = float(case_config.get("alpha", config["alpha"]))
    epsilon = float(case_config.get("epsilon", config["epsilon"]))
    try:
        system, matrix_source = build_engineering_system(case_config)
        H = np.asarray(system.H_tilde, dtype=np.float64)
        summary = singular_summary(H)
        degree = estimate_degree_and_queries(
            summary["singular_values"],
            alpha=alpha,
            epsilon=epsilon,
            degrees=list(config["degrees"]),
        )
        nonzeros = int(np.count_nonzero(np.abs(H) > 1.0e-12))
        query_count = int(degree["query_count_estimate"])
        return {
            "case_name": case_name,
            "config_file": str(case_config.get("config_file", "")),
            "matrix_source": matrix_source,
            "status": "ok",
            "m": int(H.shape[0]),
            "n": int(H.shape[1]),
            "nonzeros": nonzeros,
            "density": matrix_density(H),
            "rank_if_available": summary["rank"],
            "kappa_if_available": summary["condition_number"],
            "beta_if_available": summary["spectral_norm"],
            "alpha": alpha,
            "epsilon": epsilon,
            "qsvt_degree_estimate": int(degree["qsvt_degree_estimate"]),
            "query_count_estimate": query_count,
            "logical_qubits_estimate": int(np.ceil(np.log2(max(H.shape[0] + H.shape[1], 2)))),
            "ancilla_qubits_estimate": 2,
            "depth_estimate": int(query_count * max(nonzeros, 1)),
            "readout_caveat": READOUT_CAVEAT,
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
            "nonzeros": None,
            "density": None,
            "rank_if_available": None,
            "kappa_if_available": None,
            "beta_if_available": None,
            "alpha": alpha,
            "epsilon": epsilon,
            "qsvt_degree_estimate": None,
            "query_count_estimate": None,
            "logical_qubits_estimate": None,
            "ancilla_qubits_estimate": None,
            "depth_estimate": None,
            "readout_caveat": READOUT_CAVEAT,
            "failure_reason_if_any": str(exc),
        }


def _case_config(case: str | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    case_config = {
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_source": config["case_source"],
        "seed": config["seed"],
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
        "output_dir": "outputs/qsvt_multicase_resource_diagnostics",
        "cases": ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"],
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "epsilon": DEFAULT_EPSILON,
        "degrees": DEFAULT_DEGREES,
    }
    if config:
        resolved.update(config)
    if not list(resolved["cases"]):
        raise ValueError("cases must be non-empty")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(resolved["epsilon"]) <= 0.0:
        raise ValueError("epsilon must be positive")
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build multi-case QSVT resource diagnostics")
    parser.parse_args(argv)
    run = build_multicase_resource_diagnostics()
    print(f"QSVT multi-case resource diagnostics complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
