from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.measurement.ac_linear import build_ac_weighted_system
from robust_qsvt_se.qsvt.resources import estimate_qsvt_resources
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml
from robust_qsvt_se.utils.logging import configure_run_logger
from robust_qsvt_se.utils.seed import make_rng

DEFAULT_CASES = ["ieee14", "ieee30", "ieee57", "ieee118", "ieee300"]
DEFAULT_MEASUREMENT = {
    "include_voltage_magnitudes": True,
    "include_p_injections": True,
    "include_q_injections": True,
    "include_p_branch_flows": True,
    "include_q_branch_flows": True,
    "voltage_std": 0.01,
    "injection_p_std": 0.03,
    "injection_q_std": 0.03,
    "flow_p_std": 0.02,
    "flow_q_std": 0.02,
    "weak_area_buses": [],
    "weak_area_std_multiplier": 1.0,
}
DEFAULT_LINEARIZATION = {
    "angle_perturbation_std": 0.005,
    "voltage_perturbation_std": 0.005,
    "min_voltage_magnitude": 0.5,
}


def load_resource_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("resource estimation config must contain a mapping")
    return validate_resource_config(loaded)


def validate_resource_config(config: dict[str, Any]) -> dict[str, Any]:
    resource = dict(config.get("resource", config))
    defaults: dict[str, Any] = {
        "run_id": "qsvt_resource_full_ieee",
        "output_dir": "outputs/qsvt_resource_full_ieee",
        "case_source": "pypower",
        "cases": DEFAULT_CASES,
        "seed": 123,
        "alpha": 0.01,
        "degrees": [5, 11, 21, 35],
        "grid_size": 1024,
        "target_error": 1.0e-2,
        "simulation_qubit_limit": 12,
        "measurement": DEFAULT_MEASUREMENT,
        "linearization": DEFAULT_LINEARIZATION,
    }
    resolved = {**defaults, **resource}
    resolved["measurement"] = {**DEFAULT_MEASUREMENT, **dict(resolved["measurement"])}
    resolved["linearization"] = {**DEFAULT_LINEARIZATION, **dict(resolved["linearization"])}
    if not isinstance(resolved["cases"], list) or not resolved["cases"]:
        raise ValueError("resource.cases must be a non-empty list")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("resource.alpha must be positive")
    if not isinstance(resolved["degrees"], list) or not resolved["degrees"]:
        raise ValueError("resource.degrees must be a non-empty list")
    if int(resolved["grid_size"]) <= max(int(degree) for degree in resolved["degrees"]) + 1:
        raise ValueError("resource.grid_size must exceed max degree + 1")
    if float(resolved["target_error"]) <= 0.0:
        raise ValueError("resource.target_error must be positive")
    return {"resource": resolved}


def run_resource_estimation(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_resource_config(config)["resource"]
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting full IEEE matrix QSVT resource estimates")
    rows = [
        _estimate_case(case_name=str(case_name), resolved=resolved)
        for case_name in resolved["cases"]
    ]
    frame = pd.DataFrame(rows)
    summary = {
        "run_id": resolved["run_id"],
        "case_count": len(rows),
        "cases": list(resolved["cases"]),
        "notes": (
            "Resource estimates are proxy calculations for weighted Jacobian matrices; "
            "they are not hardware execution, oracle construction, or quantum speedup evidence."
        ),
    }
    write_yaml(output_dir / "config_resolved.yaml", {"resource": resolved})
    frame.to_csv(output_dir / "resource_estimates.csv", index=False)
    frame.to_csv(output_dir / "qsvt_resource_estimates.csv", index=False)
    write_json(output_dir / "resource_estimates_summary.json", summary)
    logger.info("Completed full IEEE matrix QSVT resource estimates")
    return {"output_dir": output_dir, "resource_estimates": frame, "summary": summary}


def _estimate_case(*, case_name: str, resolved: dict[str, Any]) -> dict[str, Any]:
    rng = make_rng(int(resolved["seed"]))
    system = build_ac_weighted_system(
        case_name=case_name,
        case_source=str(resolved["case_source"]),
        linearization_config=dict(resolved["linearization"]),
        measurement_config=dict(resolved["measurement"]),
        rng=rng,
    )
    matrix = np.asarray(system.H_tilde, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > 1.0e-14]
    condition = float(np.inf) if positive.size == 0 else float(positive.max() / positive.min())
    resource_rows = estimate_qsvt_resources(
        singular_values,
        alpha=float(resolved["alpha"]),
        degrees=[int(degree) for degree in resolved["degrees"]],
        grid_size=int(resolved["grid_size"]),
        target_error=float(resolved["target_error"]),
    )
    recommended = next(
        (row.recommended_degree for row in resource_rows if row.recommended_degree is not None),
        None,
    )
    polynomial_degree = int(recommended if recommended is not None else max(resolved["degrees"]))
    selected_resource = next(
        (row for row in resource_rows if row.degree == polynomial_degree),
        resource_rows[-1],
    )
    phase_count = polynomial_degree + 1
    qubit_estimate = int(np.ceil(np.log2(matrix.shape[0] + matrix.shape[1])))
    ancilla_qubits = 1
    query_count = 2 * polynomial_degree + 1
    total_qubits = qubit_estimate + ancilla_qubits
    statevector_dimension = int(2**total_qubits)
    estimated_statevector_memory_gb = float(statevector_dimension * 16 / 1.0e9)
    simulation_feasible = total_qubits <= int(resolved["simulation_qubit_limit"])
    feasible_reason = (
        "estimated total qubits within configured simulation limit"
        if simulation_feasible
        else (
            f"estimated total qubits {total_qubits} exceeds configured simulation limit "
            f"{resolved['simulation_qubit_limit']}"
        )
    )
    nonzero = int(np.count_nonzero(np.abs(matrix) > 1.0e-12))
    return {
        "case_name": case_name,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "nonzero_entries": nonzero,
        "sparsity_fraction": float(nonzero / matrix.size),
        "spectral_norm": float(singular_values[0]),
        "condition_number": condition,
        "degree": polynomial_degree,
        "max_error": selected_resource.max_error,
        "block_encoding_normalization": selected_resource.block_encoding_normalization,
        "effective_condition_number": selected_resource.effective_condition_number,
        "proxy_query_count": selected_resource.proxy_query_count,
        "polynomial_degree": polynomial_degree,
        "phase_count": phase_count,
        "estimated_block_encoding_qubits": qubit_estimate,
        "estimated_ancilla_qubits": ancilla_qubits,
        "estimated_total_qubits": total_qubits,
        "estimated_qsvt_query_count": query_count,
        "estimated_statevector_dimension": statevector_dimension,
        "estimated_statevector_memory_gb": estimated_statevector_memory_gb,
        "estimated_circuit_depth_proxy": int(query_count * max(1, nonzero)),
        "estimated_gate_count_proxy": int(query_count * max(1, nonzero) * 2),
        "full_statevector_simulation_feasible": bool(simulation_feasible),
        "full_simulation_feasible_reason": feasible_reason,
        "recommended_degree": recommended,
        "target_error": float(resolved["target_error"]),
        "alpha": float(resolved["alpha"]),
        "dataset_source": system.metadata.get("dataset_source"),
        "dataset_source_detail": system.metadata.get("dataset_source_detail"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate QSVT resources for full IEEE cases")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run = run_resource_estimation(load_resource_config(args.config))
    print(f"QSVT resource estimates complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
