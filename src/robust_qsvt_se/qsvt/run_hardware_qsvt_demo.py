from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.qsvt.hardware_qsvt import (
    approximation_error_frame,
    run_explicit_hardware_qsvt,
)
from robust_qsvt_se.qsvt.research_matrix import (
    extract_research_matrix,
    singular_values_frame,
    validate_research_matrix_config,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml
from robust_qsvt_se.utils.logging import configure_run_logger


def load_hardware_qsvt_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("hardware QSVT config must contain a mapping")
    return validate_hardware_qsvt_config(loaded)


def validate_hardware_qsvt_config(config: dict[str, Any]) -> dict[str, Any]:
    demo = dict(config.get("demo", {}))
    matrix = validate_research_matrix_config({"matrix": config.get("matrix", {})})["matrix"]
    defaults: dict[str, Any] = {
        "run_id": "qsvt_hardware_ieee14_4x4",
        "output_dir": "outputs/qsvt_hardware_ieee14_4x4",
        "alpha": 0.05,
        "polynomial_degree": 5,
        "grid_size": 512,
        "angle_solver": "iterative",
        "phase_cache_dir": "outputs/qsvt_phase_cache",
        "transpile_basis_gates": ["rz", "sx", "x", "cx"],
        "transpile_optimization_level": 1,
        "domain_min": None,
    }
    resolved = {**defaults, **demo}
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("demo.alpha must be positive")
    degree = int(resolved["polynomial_degree"])
    if degree < 1 or degree % 2 == 0:
        raise ValueError("demo.polynomial_degree must be a positive odd integer")
    if int(resolved["grid_size"]) <= degree + 1:
        raise ValueError("demo.grid_size must be greater than polynomial_degree + 1")
    if str(resolved["angle_solver"]) not in {"root-finding", "iterative", "iterative-optax"}:
        raise ValueError("demo.angle_solver is invalid")
    if not isinstance(resolved["transpile_basis_gates"], list) or not all(
        isinstance(item, str) and item for item in resolved["transpile_basis_gates"]
    ):
        raise ValueError("demo.transpile_basis_gates must be a list of gate names")
    return {"demo": resolved, "matrix": matrix}


def run_hardware_qsvt_demo(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_hardware_qsvt_config(config)
    demo = resolved["demo"]
    output_dir = ensure_directory(Path(demo["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting explicit hardware QSVT prototype %s", demo["run_id"])

    research_matrix = extract_research_matrix({"matrix": resolved["matrix"]})
    matrix = research_matrix.normalized_matrix
    result = run_explicit_hardware_qsvt(
        matrix,
        alpha=float(demo["alpha"]),
        polynomial_degree=int(demo["polynomial_degree"]),
        grid_size=int(demo["grid_size"]),
        angle_solver=str(demo["angle_solver"]),
        phase_cache_dir=str(demo["phase_cache_dir"]),
        basis_gates=list(demo["transpile_basis_gates"]),
        domain_min=(None if demo.get("domain_min") is None else float(demo.get("domain_min"))),
        transpile_optimization_level=int(demo["transpile_optimization_level"]),
    )

    metadata = {
        **research_matrix.metadata,
        "qsvt_implementation_scope": "explicit small-matrix block-encoding QSVT prototype",
    }
    summary = {
        **result.summary,
        "run_id": demo["run_id"],
        "case_name": metadata.get("case_name") or metadata.get("source_case_name"),
        "source_case": metadata.get("source_case_name") or metadata.get("case_name"),
        "matrix_source": "weighted_jacobian",
        "matrix_shape": list(matrix.shape),
        "matrix_scope": metadata.get("matrix_scope"),
        "full_or_submatrix": "full_matrix" if metadata.get("is_full_matrix") else "submatrix",
        "normalization_factor": metadata.get("normalization_factor"),
        "condition_number": metadata.get("condition_number"),
        "selected_rows": metadata.get("selected_rows"),
        "selected_columns": metadata.get("selected_columns"),
    }
    block_summary = {
        **result.block_encoding_summary,
        "case_name": summary["case_name"],
        "matrix_source": "weighted_jacobian",
        "matrix_shape": list(matrix.shape),
        "normalization_factor": metadata.get("normalization_factor"),
    }
    artifacts = _write_artifacts(
        output_dir=output_dir,
        resolved_config=resolved,
        metadata=metadata,
        singular_values=singular_values_frame(research_matrix),
        phases=result.phases,
        coefficients=result.coefficients,
        approximation_error=approximation_error_frame(result.approximation),
        block_encoding_summary=block_summary,
        summary=summary,
        gate_counts=result.gate_counts,
        transpiled_gate_counts=result.transpiled_gate_counts,
        circuit=result.circuit,
        transpiled_circuit=result.transpiled_circuit,
        simulation=result.simulation,
        comparison=result.comparison,
    )
    logger.info(
        "Completed explicit hardware QSVT prototype %s with max error %.6g",
        demo["run_id"],
        summary["max_error_vs_classical"],
    )
    return {"output_dir": output_dir, "artifacts": artifacts, "summary": summary}


def _write_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    metadata: dict[str, Any],
    singular_values: pd.DataFrame,
    phases: np.ndarray,
    coefficients: np.ndarray,
    approximation_error: pd.DataFrame,
    block_encoding_summary: dict[str, Any],
    summary: dict[str, Any],
    gate_counts: dict[str, int],
    transpiled_gate_counts: dict[str, int],
    circuit: Any,
    transpiled_circuit: Any | None,
    simulation: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, str]:
    write_yaml(output_dir / "config_resolved.yaml", resolved_config)
    write_json(output_dir / "research_matrix_metadata.json", metadata)
    singular_values.to_csv(output_dir / "singular_values.csv", index=False)
    pd.DataFrame({"phase_index": np.arange(phases.size), "phase_angle": phases}).to_csv(
        output_dir / "phase_angles.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "coefficient_index": np.arange(coefficients.size),
            "scaled_power_coefficient": coefficients,
        }
    ).to_csv(output_dir / "polynomial_coefficients.csv", index=False)
    approximation_error.to_csv(output_dir / "approximation_error.csv", index=False)
    write_json(output_dir / "block_encoding_summary.json", block_encoding_summary)
    write_json(output_dir / "hardware_qsvt_circuit_summary.json", summary)
    write_json(output_dir / "circuit_summary.json", summary)
    write_json(output_dir / "hardware_qsvt_gate_counts.json", gate_counts)
    write_json(output_dir / "hardware_qsvt_transpiled_gate_counts.json", transpiled_gate_counts)
    write_json(
        output_dir / "hardware_qsvt_transpiled_summary.json",
        {
            "transpile_success": summary.get("transpile_success"),
            "transpile_message": summary.get("transpile_message"),
            "transpile_seconds": summary.get("transpile_seconds"),
            "basis_gates": summary.get("basis_gates"),
            "depth_after_transpile": summary.get("depth_after_transpile"),
            "gate_counts_after_transpile": transpiled_gate_counts,
        },
    )
    (output_dir / "circuit_draw.txt").write_text(str(circuit.draw(output="text")), encoding="utf-8")
    transpiled_text = (
        str(transpiled_circuit.draw(output="text"))
        if transpiled_circuit is not None
        else "Transpilation skipped or failed; see hardware_qsvt_transpiled_summary.json."
    )
    (output_dir / "transpiled_circuit_draw.txt").write_text(transpiled_text, encoding="utf-8")
    simulation.to_csv(output_dir / "simulation_results.csv", index=False)
    comparison.to_csv(output_dir / "comparison_to_classical.csv", index=False)
    return {
        "config_resolved": str(output_dir / "config_resolved.yaml"),
        "research_matrix_metadata": str(output_dir / "research_matrix_metadata.json"),
        "singular_values": str(output_dir / "singular_values.csv"),
        "phase_angles": str(output_dir / "phase_angles.csv"),
        "polynomial_coefficients": str(output_dir / "polynomial_coefficients.csv"),
        "approximation_error": str(output_dir / "approximation_error.csv"),
        "block_encoding_summary": str(output_dir / "block_encoding_summary.json"),
        "hardware_qsvt_circuit_summary": str(output_dir / "hardware_qsvt_circuit_summary.json"),
        "hardware_qsvt_gate_counts": str(output_dir / "hardware_qsvt_gate_counts.json"),
        "hardware_qsvt_transpiled_summary": str(
            output_dir / "hardware_qsvt_transpiled_summary.json"
        ),
        "hardware_qsvt_transpiled_gate_counts": str(
            output_dir / "hardware_qsvt_transpiled_gate_counts.json"
        ),
        "circuit_draw": str(output_dir / "circuit_draw.txt"),
        "transpiled_circuit_draw": str(output_dir / "transpiled_circuit_draw.txt"),
        "simulation_results": str(output_dir / "simulation_results.csv"),
        "comparison_to_classical": str(output_dir / "comparison_to_classical.csv"),
        "run_log": str(output_dir / "run.log"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run explicit research-matrix hardware QSVT demo")
    parser.add_argument("--config", required=True, help="Path to the hardware QSVT config")
    args = parser.parse_args(argv)
    run_hardware_qsvt_demo(load_hardware_qsvt_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
