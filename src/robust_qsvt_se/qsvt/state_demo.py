from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.estimators.qsvt_spectral import QSVTSpectralEstimator
from robust_qsvt_se.estimators.ridge import RidgeEstimator
from robust_qsvt_se.qsvt.block_encoding import normalize_for_block_encoding
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    bounded_scaling_C,
    build_engineering_system,
    direction_metrics,
    singular_summary,
)
from robust_qsvt_se.qsvt.filters import ridge_filter
from robust_qsvt_se.utils.io import ensure_directory, write_json


def run_state_demo(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    alpha = float(resolved["alpha"])
    system, matrix_source = build_engineering_system(resolved)

    normalized_matrix, beta = normalize_for_block_encoding(system.H_tilde)
    singular_values = system.singular_values()
    normalized_singular_values = np.linalg.svd(normalized_matrix, compute_uv=False)
    original_summary = singular_summary(system.H_tilde)
    normalized_summary = singular_summary(normalized_matrix)

    ridge = RidgeEstimator(alpha=alpha).solve(system)
    qsvt_target = QSVTSpectralEstimator(alpha=alpha).solve(system)
    metrics = direction_metrics(ridge.x_hat, qsvt_target.x_hat)
    C = bounded_scaling_C(singular_values, alpha=alpha)
    relative_error = metrics["relative_error"]
    cosine = metrics["cosine_similarity"]

    summary_row = {
        "case_name": str(system.metadata.get("case_name", "unknown")),
        "matrix_source": matrix_source,
        "matrix_shape": f"{system.n_measurements}x{system.n_states}",
        "state_dimension": system.n_states,
        "row_count": system.n_measurements,
        "alpha": alpha,
        "beta": beta,
        "condition_number_original": original_summary["condition_number"],
        "condition_number_normalized": normalized_summary["condition_number"],
        "ridge_solution_norm": float(np.linalg.norm(ridge.x_hat)),
        "qsvt_target_solution_norm": float(np.linalg.norm(qsvt_target.x_hat)),
        "relative_error_vs_ridge": relative_error,
        "cosine_similarity_vs_ridge": cosine,
        "state_fidelity_vs_ridge_direction": metrics["state_fidelity"],
        "residual_norm_ridge": ridge.residual_norm,
        "residual_norm_qsvt_target": qsvt_target.residual_norm,
        "weighted_residual_norm_ridge": ridge.weighted_residual_norm,
        "weighted_residual_norm_qsvt_target": qsvt_target.weighted_residual_norm,
        "phase_or_polynomial_error_if_available": None,
        "bounded_scaling_C": C,
        "max_bounded_filter_value": float(np.max(ridge_filter(singular_values, alpha=alpha) / C)),
        "passed_equivalence_check": bool(relative_error <= 1.0e-8 and cosine >= 1.0 - 1.0e-8),
        "interpretation": (
            "Exact QSVT-target spectral filtering uses the same regularized filter as "
            "Ridge/Tikhonov for the same alpha; equality is expected."
        ),
    }
    summary = pd.DataFrame([summary_row])
    singular_frame = pd.DataFrame(
        {
            "singular_index": np.arange(singular_values.size),
            "singular_value": singular_values,
            "normalized_singular_value": normalized_singular_values,
        }
    )
    filter_values = ridge_filter(singular_values, alpha=alpha)
    filter_frame = pd.DataFrame(
        {
            "singular_index": np.arange(singular_values.size),
            "singular_value": singular_values,
            "normalized_singular_value": normalized_singular_values,
            "ridge_filter_value": filter_values,
            "qsvt_target_filter_value": filter_values,
            "bounded_qsvt_filter_value": filter_values / C,
        }
    )

    summary_csv = output_dir / "state_demo_summary.csv"
    summary_json = output_dir / "state_demo_summary.json"
    singular_csv = output_dir / "singular_values.csv"
    filter_csv = output_dir / "filter_values.csv"
    summary.to_csv(summary_csv, index=False)
    write_json(summary_json, summary_row)
    singular_frame.to_csv(singular_csv, index=False)
    filter_frame.to_csv(filter_csv, index=False)
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "state_demo_summary_csv": str(summary_csv),
            "state_demo_summary_json": str(summary_json),
            "singular_values_csv": str(singular_csv),
            "filter_values_csv": str(filter_csv),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "artifacts": {
            "state_demo_summary_csv": summary_csv,
            "state_demo_summary_json": summary_json,
            "singular_values_csv": singular_csv,
            "filter_values_csv": filter_csv,
            "manifest": manifest_path,
        },
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_end_to_end_state_demo",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
    }
    if config:
        resolved.update(config)
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run exact QSVT-target/Ridge state demo")
    parser.parse_args(argv)
    run = run_state_demo()
    print(f"QSVT end-to-end state demo complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
