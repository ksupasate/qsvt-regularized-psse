from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.block_encoding import spectral_norm_bound
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.qsvt.engineering_utils import (
    DEFAULT_DEGREES,
    DEFAULT_EPSILON,
    RESOURCE_CAVEAT,
    build_engineering_system,
    estimate_degree_and_queries,
    matrix_density,
    required_case_name,
    ridge_svd_solution,
    singular_summary,
)
from robust_qsvt_se.qsvt.readout_analysis import (
    readout_summary_markdown,
    resource_assumptions_markdown,
    shot_level_readout_rows,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json


def estimate_qsvt_filter_resources(
    matrix: np.ndarray,
    *,
    alpha: float,
    epsilon: float = DEFAULT_EPSILON,
    degrees: list[int] | None = None,
    case_name: str = "unknown",
    matrix_source: str = "unknown",
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    summary = singular_summary(values)
    singular_values = summary["singular_values"]
    degree = estimate_degree_and_queries(
        singular_values,
        alpha=alpha,
        epsilon=epsilon,
        degrees=degrees or DEFAULT_DEGREES,
    )
    nonzero = int(np.count_nonzero(np.abs(values) > 1.0e-12))
    logical_qubits = int(np.ceil(np.log2(values.shape[0] + values.shape[1])))
    ancilla_qubits = 2
    query_count = int(degree["query_count_estimate"])
    return {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "m": int(values.shape[0]),
        "n": int(values.shape[1]),
        "sparsity": matrix_density(values),
        "rank": summary["rank"],
        "kappa": summary["condition_number"],
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "beta": spectral_norm_bound(values),
        "qsvt_degree_estimate": int(degree["qsvt_degree_estimate"]),
        "query_count_estimate": query_count,
        "logical_qubits_estimate": logical_qubits,
        "ancilla_qubits_estimate": ancilla_qubits,
        "depth_estimate": int(query_count * max(1, nonzero)),
        "estimated_controlled_block_encoding_calls": query_count,
        "state_preparation_model": "placeholder_amplitude_state_preparation_not_implemented",
        "state_preparation_cost_placeholder": "not modeled",
        "readout_model": "selected_component_or_observable_estimation",
        "readout_cost_placeholder": "depends on target observable and sampling tolerance",
        "full_vector_readout_required": False,
        "readout_caveat": (
            "Full state-vector reconstruction can require many measurements; "
            "selected components or observables are the intended readout model."
        ),
        "claim_strength": "feasibility discussion only",
        "resource_caveat": RESOURCE_CAVEAT,
    }


def run_resource_readout_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    row = estimate_qsvt_filter_resources(
        np.asarray(system.H_tilde, dtype=np.float64),
        alpha=float(resolved["alpha"]),
        epsilon=float(resolved["epsilon"]),
        degrees=list(resolved["degrees"]),
        case_name=required_case_name(system),
        matrix_source=matrix_source,
    )
    frame = pd.DataFrame([row])
    x_hat = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=float(resolved["alpha"]))
    shot_rows = shot_level_readout_rows(
        H_tilde=np.asarray(system.H_tilde, dtype=np.float64),
        r_tilde=np.asarray(system.r_tilde, dtype=np.float64),
        x_hat=x_hat,
        shot_count=int(resolved["shot_count"]),
    )
    shot_frame = pd.DataFrame(shot_rows)

    csv_path = output_dir / "resource_summary.csv"
    json_path = output_dir / "resource_summary.json"
    readout_path = output_dir / "readout_summary.md"
    assumptions_path = output_dir / "resource_assumptions.md"
    shot_csv_path = output_dir / "shot_readout_summary.csv"
    shot_json_path = output_dir / "shot_readout_summary.json"
    frame.to_csv(csv_path, index=False)
    write_json(json_path, row)
    shot_frame.to_csv(shot_csv_path, index=False)
    write_json(shot_json_path, {"rows": shot_rows})
    readout_path.write_text(readout_summary_markdown(row, shot_rows), encoding="utf-8")
    assumptions_path.write_text(resource_assumptions_markdown(row), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "resource_summary_csv": str(csv_path),
            "resource_summary_json": str(json_path),
            "readout_summary_md": str(readout_path),
            "resource_assumptions_md": str(assumptions_path),
            "shot_readout_summary_csv": str(shot_csv_path),
            "shot_readout_summary_json": str(shot_json_path),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "resource_summary_csv": csv_path,
            "resource_summary_json": json_path,
            "readout_summary_md": readout_path,
            "resource_assumptions_md": assumptions_path,
            "shot_readout_summary_csv": shot_csv_path,
            "shot_readout_summary_json": shot_json_path,
            "manifest": manifest_path,
        },
    }


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_resource_readout",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "epsilon": DEFAULT_EPSILON,
        "degrees": DEFAULT_DEGREES,
        "shot_count": 4096,
    }
    if config:
        resolved.update(config)
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(resolved["epsilon"]) <= 0.0:
        raise ValueError("epsilon must be positive")
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    if int(resolved["shot_count"]) <= 0:
        raise ValueError("shot_count must be positive")
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QSVT resource/readout report")
    parser.parse_args(argv)
    run = run_resource_readout_report()
    print(f"QSVT resource/readout report complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
