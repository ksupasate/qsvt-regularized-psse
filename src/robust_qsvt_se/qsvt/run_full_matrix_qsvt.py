from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.qsvt.pennylane_matrix_qsvt import run_pennylane_matrix_qsvt
from robust_qsvt_se.qsvt.qiskit_matrix_qsvt import run_qiskit_matrix_qsvt
from robust_qsvt_se.qsvt.research_matrix import (
    extract_qsvt_submatrix,
    extract_weighted_jacobian_matrix,
    singular_values_frame,
)
from robust_qsvt_se.qsvt.resources import estimate_qsvt_resources
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml
from robust_qsvt_se.utils.logging import configure_run_logger


def load_full_matrix_qsvt_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("full-matrix QSVT config must contain a mapping")
    return validate_full_matrix_qsvt_config(loaded)


def validate_full_matrix_qsvt_config(config: dict[str, Any]) -> dict[str, Any]:
    run = dict(config.get("run", {}))
    matrix = dict(config.get("matrix", {}))
    demo = dict(config.get("demo", {}))
    defaults_run = {
        "run_id": "qsvt_full_matrix_ieee14",
        "output_dir": "outputs/qsvt_full_matrix_ieee14",
        "run_pennylane": True,
        "run_qiskit": True,
        "max_pennylane_full_qubits": 8,
        "max_qiskit_dense_full_qubits": 8,
        "submatrix_size": 4,
    }
    defaults_matrix = {
        "case_name": "ieee14",
        "case_source": "pypower",
        "mode": "ac_weighted_jacobian",
        "seed": 123,
        "selection_strategy": "high_leverage",
    }
    defaults_demo = {
        "alpha": 0.05,
        "polynomial_degree": 5,
        "grid_size": 512,
        "angle_solver": "iterative",
        "block_encoding": "embedding",
        "transpile_qubit_limit": 4,
    }
    resolved_run = {**defaults_run, **run}
    resolved_matrix = {**defaults_matrix, **matrix}
    resolved_demo = {**defaults_demo, **demo}
    if int(resolved_run["submatrix_size"]) <= 0:
        raise ValueError("run.submatrix_size must be positive")
    if (
        int(resolved_demo["polynomial_degree"]) < 1
        or int(resolved_demo["polynomial_degree"]) % 2 == 0
    ):
        raise ValueError("demo.polynomial_degree must be a positive odd integer")
    return {"run": resolved_run, "matrix": resolved_matrix, "demo": resolved_demo}


def run_full_matrix_qsvt(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_full_matrix_qsvt_config(config)
    run = resolved["run"]
    matrix_config = resolved["matrix"]
    demo = resolved["demo"]
    output_dir = ensure_directory(Path(run["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting full-matrix QSVT feasibility run %s", run["run_id"])

    full_matrix = extract_weighted_jacobian_matrix(
        case_name=str(matrix_config["case_name"]),
        mode=str(matrix_config["mode"]),
        case_source=str(matrix_config["case_source"]),
        normalize=True,
        seed=int(matrix_config["seed"]),
        measurement_config=matrix_config.get("measurement"),
        linearization_config=matrix_config.get("linearization"),
    )
    full_shape = full_matrix.normalized_matrix.shape
    full_qubits = int(np.ceil(np.log2(full_shape[0] + full_shape[1])))
    feasibility = _feasibility_decision(
        case_name=str(matrix_config["case_name"]),
        full_shape=full_shape,
        full_qubits=full_qubits,
        polynomial_degree=int(demo["polynomial_degree"]),
        max_pennylane_full_qubits=int(run["max_pennylane_full_qubits"]),
        max_qiskit_dense_full_qubits=int(run["max_qiskit_dense_full_qubits"]),
    )
    resource_estimate = _resource_estimate(
        full_matrix,
        alpha=float(demo["alpha"]),
        degree=int(demo["polynomial_degree"]),
        grid_size=int(demo["grid_size"]),
    )
    submatrix = extract_qsvt_submatrix(
        full_matrix,
        target_shape=(int(run["submatrix_size"]), int(run["submatrix_size"])),
        strategy=str(matrix_config.get("selection_strategy", "high_leverage")),
        seed=int(matrix_config["seed"]),
    )

    pennylane_summary: dict[str, Any] = {
        "status": "not_run",
        "reason": "run_pennylane=false",
    }
    qiskit_summary: dict[str, Any] = {
        "status": "not_run",
        "reason": "run_qiskit=false",
    }
    comparison = pd.DataFrame()
    phase_angles = pd.DataFrame(columns=["phase_index", "phase_angle"])

    if bool(run["run_pennylane"]):
        pl_scope = "full_matrix" if feasibility["pennylane_full_feasible"] else "submatrix"
        pl_output = output_dir / (
            "pennylane_full" if pl_scope == "full_matrix" else "pennylane_submatrix"
        )
        pl_config = _backend_config(
            resolved,
            output_dir=pl_output,
            matrix_scope=pl_scope,
            submatrix_size=int(run["submatrix_size"]),
        )
        pl_run = run_pennylane_matrix_qsvt(pl_config)
        pennylane_summary = dict(pl_run["summary"])
        pennylane_summary["status"] = "completed"
        pennylane_summary["full_matrix_attempted"] = pl_scope == "full_matrix"
        pennylane_summary["fallback_used"] = pl_scope != "full_matrix"
        comparison = pd.read_csv(pl_run["output_dir"] / "comparison_to_classical.csv")
        phase_angles = pd.read_csv(pl_run["output_dir"] / "phase_angles.csv")

    if bool(run["run_qiskit"]):
        qk_scope = "full_matrix" if feasibility["qiskit_dense_full_feasible"] else "submatrix"
        qk_output = output_dir / (
            "qiskit_full" if qk_scope == "full_matrix" else "qiskit_submatrix"
        )
        qk_config = _backend_config(
            resolved,
            output_dir=qk_output,
            matrix_scope=qk_scope,
            submatrix_size=int(run["submatrix_size"]),
        )
        qk_run = run_qiskit_matrix_qsvt(qk_config)
        qiskit_summary = dict(qk_run["summary"])
        qiskit_summary["status"] = "completed"
        qiskit_summary["full_matrix_attempted"] = qk_scope == "full_matrix"
        qiskit_summary["fallback_used"] = qk_scope != "full_matrix"
        if comparison.empty:
            comparison = pd.read_csv(qk_run["output_dir"] / "comparison_to_classical.csv")
            phase_angles = pd.read_csv(qk_run["output_dir"] / "phase_angles.csv")

    full_matrix.metadata["full_matrix_qsvt_feasibility"] = feasibility
    full_matrix.metadata["submatrix_fallback_metadata"] = submatrix.metadata
    write_yaml(output_dir / "config_resolved.yaml", resolved)
    write_json(output_dir / "research_matrix_metadata.json", full_matrix.metadata)
    write_json(output_dir / "full_matrix_resource_estimate.json", resource_estimate)
    singular_values_frame(full_matrix).to_csv(output_dir / "singular_values.csv", index=False)
    phase_angles.to_csv(output_dir / "phase_angles.csv", index=False)
    write_json(output_dir / "pennylane_summary.json", pennylane_summary)
    write_json(output_dir / "qiskit_summary.json", qiskit_summary)
    write_json(
        output_dir / "circuit_summary.json",
        {
            "run_id": run["run_id"],
            "source_case": matrix_config["case_name"],
            "case_name": matrix_config["case_name"],
            "matrix_shape": list(full_shape),
            "matrix_scope": (
                "full_matrix"
                if feasibility["full_matrix_feasible"]
                else "resource_estimate_with_submatrix_fallback"
            ),
            "qsvt_method": "unified full-matrix feasibility runner",
            "qsvt_construction_type": "full_matrix_or_submatrix_by_feasibility",
            "is_full_matrix_qsvt": bool(feasibility["full_matrix_feasible"]),
            "pennylane_full_feasible": feasibility["pennylane_full_feasible"],
            "qiskit_dense_full_feasible": feasibility["qiskit_dense_full_feasible"],
            "pennylane_max_abs_error": pennylane_summary.get("max_abs_error"),
            "qiskit_max_abs_error": qiskit_summary.get("max_abs_error"),
            "max_abs_error": _first_float(
                pennylane_summary.get("max_abs_error"),
                qiskit_summary.get("max_abs_error"),
            ),
            "mean_abs_error": _first_float(
                pennylane_summary.get("mean_abs_error"),
                qiskit_summary.get("mean_abs_error"),
            ),
            "n_phase_angles": _first_float(
                pennylane_summary.get("n_phase_angles"),
                qiskit_summary.get("n_phase_angles"),
            ),
            "circuit_depth": qiskit_summary.get("circuit_depth"),
            "gate_count_total": qiskit_summary.get("gate_count_total"),
            "scope_note": feasibility["safe_claim"],
        },
    )
    comparison.to_csv(output_dir / "comparison_to_classical.csv", index=False)
    write_json(output_dir / "feasibility_decision.json", feasibility)
    status = pd.DataFrame(
        [
            {
                "case_name": matrix_config["case_name"],
                "full_matrix_shape": f"{full_shape[0]}x{full_shape[1]}",
                "full_qsvt_simulated": bool(
                    pennylane_summary.get("full_matrix_attempted")
                    or qiskit_summary.get("full_matrix_attempted")
                ),
                "decision": feasibility["decision"],
                "pennylane_status": pennylane_summary.get("status"),
                "qiskit_status": qiskit_summary.get("status"),
                "submatrix_run": bool(
                    pennylane_summary.get("fallback_used") or qiskit_summary.get("fallback_used")
                ),
                "resource_estimate": True,
                "safe_claim": feasibility["safe_claim"],
            }
        ]
    )
    status.to_csv(output_dir / "qsvt_full_matrix_status.csv", index=False)
    logger.info("Completed full-matrix QSVT feasibility run %s", run["run_id"])
    return {
        "output_dir": output_dir,
        "feasibility": feasibility,
        "pennylane_summary": pennylane_summary,
        "qiskit_summary": qiskit_summary,
    }


def _backend_config(
    resolved: dict[str, Any],
    *,
    output_dir: Path,
    matrix_scope: str,
    submatrix_size: int,
) -> dict[str, Any]:
    matrix = dict(resolved["matrix"])
    matrix["matrix_scope"] = matrix_scope
    matrix["use_full_matrix"] = matrix_scope == "full_matrix"
    matrix["submatrix_size"] = submatrix_size
    demo = {
        **dict(resolved["demo"]),
        "output_dir": str(output_dir),
        "run_id": output_dir.name,
    }
    return {"demo": demo, "matrix": matrix}


def _feasibility_decision(
    *,
    case_name: str,
    full_shape: tuple[int, int],
    full_qubits: int,
    polynomial_degree: int,
    max_pennylane_full_qubits: int,
    max_qiskit_dense_full_qubits: int,
) -> dict[str, Any]:
    pennylane_full_feasible = full_qubits <= max_pennylane_full_qubits
    qiskit_full_feasible = full_qubits <= max_qiskit_dense_full_qubits
    reason = []
    if not pennylane_full_feasible:
        reason.append(
            f"PennyLane full matrix skipped: {full_qubits} block-encoding qubits exceeds "
            f"limit {max_pennylane_full_qubits}."
        )
    if not qiskit_full_feasible:
        reason.append(
            f"Qiskit dense full matrix skipped: {full_qubits} block-encoding qubits exceeds "
            f"limit {max_qiskit_dense_full_qubits}."
        )
    statevector_dimension = int(2**full_qubits)
    estimated_memory_gb = float(statevector_dimension * 16 / 1.0e9)
    attempt_full_matrix = bool(pennylane_full_feasible or qiskit_full_feasible)
    decision = "full_matrix_completed" if attempt_full_matrix else "submatrix_completed"
    return {
        "case_name": case_name,
        "full_matrix_shape": [int(full_shape[0]), int(full_shape[1])],
        "estimated_qubits": int(full_qubits),
        "estimated_block_encoding_qubits": int(full_qubits),
        "estimated_statevector_dimension": statevector_dimension,
        "estimated_memory_gb": estimated_memory_gb,
        "estimated_qsvt_queries": int(2 * polynomial_degree + 1),
        "estimated_transpilation_cost": (
            "dense-unitary transpilation attempted only within configured qubit limit"
        ),
        "attempt_full_matrix": attempt_full_matrix,
        "decision": decision,
        "pennylane_full_feasible": bool(pennylane_full_feasible),
        "qiskit_dense_full_feasible": bool(qiskit_full_feasible),
        "full_matrix_feasible": bool(pennylane_full_feasible or qiskit_full_feasible),
        "fallback_required": not bool(pennylane_full_feasible and qiskit_full_feasible),
        "reason": " ".join(reason)
        if reason
        else "Full matrix simulation is within configured limits.",
        "safe_claim": (
            "Full weighted-Jacobian matrix QSVT simulated within configured limits."
            if pennylane_full_feasible or qiskit_full_feasible
            else (
                "Full matrix circuit simulation was not run; deterministic submatrix "
                "QSVT and resource estimates are reported."
            )
        ),
    }


def _resource_estimate(
    research_matrix: Any,
    *,
    alpha: float,
    degree: int,
    grid_size: int,
) -> dict[str, Any]:
    matrix = research_matrix.normalized_matrix
    singular_values = research_matrix.singular_values
    resource = estimate_qsvt_resources(
        singular_values,
        alpha=alpha,
        degrees=[degree],
        grid_size=max(grid_size, degree + 2),
        target_error=1.0e-2,
    )[0]
    nonzero = int(np.count_nonzero(np.abs(matrix) > 1.0e-12))
    qubits = int(np.ceil(np.log2(matrix.shape[0] + matrix.shape[1])))
    return {
        "case_name": research_matrix.metadata.get("source_case_name"),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nonzero_entries": nonzero,
        "sparsity_fraction": float(nonzero / matrix.size),
        "condition_number": research_matrix.metadata.get("condition_number"),
        "polynomial_degree": degree,
        "phase_count": degree + 1,
        "estimated_block_encoding_qubits": qubits,
        "estimated_ancilla_qubits": 1,
        "estimated_qsvt_query_count": 2 * degree + 1,
        "estimated_circuit_depth_proxy": int((2 * degree + 1) * max(1, nonzero)),
        "estimated_gate_count_proxy": int((2 * degree + 1) * max(1, nonzero) * 2),
        "max_error": resource.max_error,
        "notes": (
            "Proxy resource estimate only; no oracle construction, state preparation, "
            "readout, or quantum speedup is modeled."
        ),
    }


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run full/research-matrix QSVT feasibility")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run = run_full_matrix_qsvt(load_full_matrix_qsvt_config(args.config))
    print(f"Full-matrix QSVT run complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
