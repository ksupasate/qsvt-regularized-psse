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
    direction_metrics,
    estimate_degree_and_queries,
    required_case_name,
    ridge_svd_solution,
    singular_summary,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

CLAIM_STRENGTH = "diagnostic evidence for possible resource reduction"


def run_preconditioning_diagnostics(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    alpha = float(resolved["alpha"])
    H = np.asarray(system.H_tilde, dtype=np.float64)
    column_norms = np.linalg.norm(H, axis=0)
    scales = np.divide(
        1.0,
        column_norms,
        out=np.ones_like(column_norms),
        where=column_norms > 1.0e-14,
    )
    H_preconditioned = H * scales[None, :]

    before = singular_summary(H)
    after = singular_summary(H_preconditioned)
    resource_before = estimate_degree_and_queries(
        before["singular_values"],
        alpha=alpha,
        epsilon=float(resolved["epsilon"]),
        degrees=list(resolved["degrees"]),
    )
    resource_after = estimate_degree_and_queries(
        after["singular_values"],
        alpha=alpha,
        epsilon=float(resolved["epsilon"]),
        degrees=list(resolved["degrees"]),
    )

    unpreconditioned = ridge_svd_solution(H, system.r_tilde, alpha=alpha)
    y_solution = ridge_svd_solution(H_preconditioned, system.r_tilde, alpha=alpha)
    preconditioned_solution = scales * y_solution
    comparison = direction_metrics(unpreconditioned, preconditioned_solution)
    row = {
        "case_name": required_case_name(system),
        "matrix_source": matrix_source,
        "preconditioner_type": "column_equilibration",
        "kappa_before": before["condition_number"],
        "kappa_after": after["condition_number"],
        "rank_before": before["rank"],
        "rank_after": after["rank"],
        "alpha": alpha,
        "estimated_qsvt_degree_before": int(resource_before["qsvt_degree_estimate"]),
        "estimated_qsvt_degree_after": int(resource_after["qsvt_degree_estimate"]),
        "query_count_before": int(resource_before["query_count_estimate"]),
        "query_count_after": int(resource_after["query_count_estimate"]),
        "relative_solution_error_vs_unpreconditioned_ridge": comparison["relative_error"],
        "residual_norm": system.residual_norm(preconditioned_solution),
        "claim_strength": CLAIM_STRENGTH,
    }
    frame = pd.DataFrame([row])
    csv_path = output_dir / "preconditioning_summary.csv"
    json_path = output_dir / "preconditioning_summary.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, row)
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "preconditioning_summary_csv": str(csv_path),
            "preconditioning_summary_json": str(json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "preconditioning_summary_csv": csv_path,
            "preconditioning_summary_json": json_path,
            "manifest": manifest_path,
        },
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_preconditioning_diagnostics",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "epsilon": DEFAULT_EPSILON,
        "degrees": DEFAULT_DEGREES,
    }
    if config:
        resolved.update(config)
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT preconditioning diagnostics")
    parser.parse_args(argv)
    run = run_preconditioning_diagnostics()
    print(f"QSVT preconditioning diagnostics complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
