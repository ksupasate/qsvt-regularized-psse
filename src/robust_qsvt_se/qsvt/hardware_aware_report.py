from __future__ import annotations

import argparse
import importlib.util
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
    required_case_name,
    singular_summary,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

HARDWARE_AWARE_CAVEAT = (
    "This is a hardware-aware simulation or proxy-cost report. It is not full "
    "IEEE-scale hardware execution and does not demonstrate quantum advantage."
)


def build_hardware_aware_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    system, matrix_source = build_engineering_system(resolved)
    row = hardware_proxy_row(
        matrix=np.asarray(system.H_tilde, dtype=np.float64),
        case_name=required_case_name(system),
        matrix_source=matrix_source,
        alpha=float(resolved["alpha"]),
        epsilon=float(resolved["epsilon"]),
        degrees=list(resolved["degrees"]),
        routing_overhead_factor=float(resolved["routing_overhead_factor"]),
        shot_budget=int(resolved["shot_budget"]),
        attempt_optional_quantum=bool(resolved["attempt_optional_quantum"]),
    )
    dependency_rows = [
        {"dependency": name, "available": available}
        for name, available in _optional_dependency_availability().items()
    ]
    frame = pd.DataFrame([row])
    summary_csv = output_dir / "hardware_aware_summary.csv"
    summary_json = output_dir / "hardware_aware_summary.json"
    assumptions_md = output_dir / "hardware_assumptions.md"
    frame.to_csv(summary_csv, index=False)
    write_json(summary_json, {"rows": [row], "optional_dependencies": dependency_rows})
    assumptions_md.write_text(_assumptions_markdown(row, dependency_rows), encoding="utf-8")
    manifest_path = write_manifest(
        output_dir,
        artifacts={
            "hardware_aware_summary_csv": str(summary_csv),
            "hardware_aware_summary_json": str(summary_json),
            "hardware_assumptions_md": str(assumptions_md),
        },
        input_config=resolved,
    )
    return {
        "output_dir": output_dir,
        "summary": frame,
        "artifacts": {
            "hardware_aware_summary_csv": summary_csv,
            "hardware_aware_summary_json": summary_json,
            "hardware_assumptions_md": assumptions_md,
            "manifest": manifest_path,
        },
    }


def hardware_proxy_row(
    *,
    matrix: np.ndarray,
    case_name: str,
    matrix_source: str,
    alpha: float,
    epsilon: float,
    degrees: list[int],
    routing_overhead_factor: float,
    shot_budget: int,
    attempt_optional_quantum: bool,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    summary = singular_summary(values)
    degree = estimate_degree_and_queries(
        summary["singular_values"],
        alpha=alpha,
        epsilon=epsilon,
        degrees=degrees,
    )
    nonzeros = int(np.count_nonzero(np.abs(values) > 1.0e-12))
    logical_qubits = int(np.ceil(np.log2(max(values.shape[0] + values.shape[1], 2))))
    ancilla_qubits = 2
    query_count = int(degree["query_count_estimate"])
    controlled_calls = query_count
    one_qubit_gates = int(routing_overhead_factor * query_count * max(logical_qubits, 1) * 4)
    two_qubit_gates = int(routing_overhead_factor * query_count * max(nonzeros, 1))
    depth = int(routing_overhead_factor * query_count * max(nonzeros, values.shape[1], 1))
    return {
        "case_name": case_name,
        "matrix_source": matrix_source,
        "m": int(values.shape[0]),
        "n": int(values.shape[1]),
        "nonzeros": nonzeros,
        "density": matrix_density(values),
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "simulation_mode": "dependency_free_proxy",
        "dependency_used": "none_proxy_only",
        "optional_quantum_attempted": bool(attempt_optional_quantum),
        "logical_qubits_estimate": logical_qubits,
        "ancilla_qubits_estimate": ancilla_qubits,
        "total_qubits_estimate": logical_qubits + ancilla_qubits,
        "qsvt_degree_estimate": int(degree["qsvt_degree_estimate"]),
        "query_count_estimate": query_count,
        "estimated_controlled_block_encoding_calls": controlled_calls,
        "estimated_one_qubit_gates": one_qubit_gates,
        "estimated_two_qubit_gates": two_qubit_gates,
        "estimated_depth": depth,
        "routing_overhead_factor": float(routing_overhead_factor),
        "noise_model_if_any": "none_proxy_only",
        "shots_if_any": int(shot_budget),
        "state_fidelity_if_available": None,
        "hardware_caveat": HARDWARE_AWARE_CAVEAT,
    }


def _optional_dependency_availability() -> dict[str, bool]:
    return {
        "qiskit": importlib.util.find_spec("qiskit") is not None,
        "qiskit_aer": importlib.util.find_spec("qiskit_aer") is not None,
        "pennylane": importlib.util.find_spec("pennylane") is not None,
    }


def _assumptions_markdown(row: dict[str, Any], dependencies: list[dict[str, Any]]) -> str:
    dependency_lines = "\n".join(
        f"- `{dependency['dependency']}` available: `{dependency['available']}`"
        for dependency in dependencies
    )
    return f"""# QSVT Hardware-Aware Proxy Assumptions

{HARDWARE_AWARE_CAVEAT}

The default mode is dependency-free and proxy-only. Optional quantum packages
are detected for transparency but are not required and are not used by this
report unless a future explicit configuration adds backend execution.

## Proxy Model

- Matrix shape: `{row["m"]}x{row["n"]}`
- QSVT degree estimate: `{row["qsvt_degree_estimate"]}`
- Query count estimate: `{row["query_count_estimate"]}`
- Controlled block-encoding call estimate: `{row["estimated_controlled_block_encoding_calls"]}`
- Routing overhead factor: `{row["routing_overhead_factor"]}`
- Shot budget placeholder: `{row["shots_if_any"]}`

## Optional Dependency Detection

{dependency_lines}

This report does not include hardware-native sparse oracle synthesis, calibrated
noise, error correction, state-preparation implementation, or full-vector
readout.
"""


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = {
        "output_dir": "outputs/qsvt_hardware_aware",
        "matrix_source": "ieee14_ac_weighted_jacobian",
        "case_name": "ieee14",
        "case_source": "pypower",
        "seed": 123,
        "alpha": 1.0e-2,
        "epsilon": DEFAULT_EPSILON,
        "degrees": DEFAULT_DEGREES,
        "routing_overhead_factor": 1.5,
        "shot_budget": 10000,
        "attempt_optional_quantum": False,
    }
    if config:
        resolved.update(config)
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(resolved["epsilon"]) <= 0.0:
        raise ValueError("epsilon must be positive")
    if float(resolved["routing_overhead_factor"]) < 1.0:
        raise ValueError("routing_overhead_factor must be at least 1")
    if int(resolved["shot_budget"]) < 0:
        raise ValueError("shot_budget must be nonnegative")
    resolved["degrees"] = [int(degree) for degree in resolved["degrees"]]
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hardware-aware QSVT proxy report")
    parser.parse_args(argv)
    run = build_hardware_aware_report()
    print(f"QSVT hardware-aware report complete: {run['output_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
