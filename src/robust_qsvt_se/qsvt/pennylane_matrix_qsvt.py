from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.qsvt.research_matrix import (
    extract_research_matrix,
    singular_values_frame,
    validate_research_matrix_config,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json, write_yaml
from robust_qsvt_se.utils.logging import configure_run_logger


def load_pennylane_matrix_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("PennyLane matrix QSVT config must contain a mapping")
    return validate_pennylane_matrix_config(loaded)


def validate_pennylane_matrix_config(config: dict[str, Any]) -> dict[str, Any]:
    demo = dict(config.get("demo", {}))
    matrix = validate_research_matrix_config({"matrix": config.get("matrix", {})})["matrix"]
    defaults = {
        "run_id": "qsvt_pennylane_matrix_ieee14",
        "output_dir": "outputs/qsvt_pennylane_matrix_ieee14",
        "alpha": 0.05,
        "polynomial_degree": 5,
        "grid_size": 1024,
        "angle_solver": "iterative",
        "block_encoding": "embedding",
    }
    resolved = {**defaults, **demo}
    if int(resolved["polynomial_degree"]) < 1 or int(resolved["polynomial_degree"]) % 2 == 0:
        raise ValueError("demo.polynomial_degree must be a positive odd integer")
    if int(resolved["grid_size"]) <= int(resolved["polynomial_degree"]) + 1:
        raise ValueError("demo.grid_size must be greater than polynomial_degree + 1")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("demo.alpha must be positive")
    if str(resolved["angle_solver"]) not in {"root-finding", "iterative", "iterative-optax"}:
        raise ValueError("demo.angle_solver is invalid")
    return {"demo": resolved, "matrix": matrix}


def run_pennylane_matrix_qsvt(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_pennylane_matrix_config(config)
    demo = resolved["demo"]
    output_dir = ensure_directory(Path(demo["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting PennyLane research-matrix QSVT demo %s", demo["run_id"])
    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for this demo") from exc

    research_matrix = extract_research_matrix({"matrix": resolved["matrix"]})
    matrix = research_matrix.normalized_matrix
    rows, columns = matrix.shape
    wires = list(range(int(np.ceil(np.log2(rows + columns)))))
    block_encoding = qml.BlockEncode(matrix, wires=wires)
    block_encoding_normalization = float(block_encoding.hyperparameters["norm"])
    effective_matrix = matrix / block_encoding_normalization
    effective_singular_values = np.linalg.svd(effective_matrix, compute_uv=False)
    domain_min = max(
        float(np.min(effective_singular_values[effective_singular_values > 1.0e-14])), 0.05
    )
    approximation = fit_odd_regularized_polynomial(
        alpha=float(demo["alpha"]),
        block_encoding_normalization=1.0,
        degree=int(demo["polynomial_degree"]),
        domain_min=domain_min,
        domain_max=1.0,
        grid_size=int(demo["grid_size"]),
    )
    scaled_coefficients = np.asarray(approximation.power_coefficients) / approximation.scale_factor
    phases = np.asarray(
        qml.poly_to_angles(scaled_coefficients, "QSVT", angle_solver=str(demo["angle_solver"])),
        dtype=np.float64,
    )
    operator = qml.qsvt(
        matrix,
        scaled_coefficients,
        encoding_wires=wires,
        block_encoding=str(demo["block_encoding"]),
        angle_solver=str(demo["angle_solver"]),
    )
    qsvt_unitary = qml.matrix(operator, wire_order=wires)
    transformed_block = np.real(qsvt_unitary[:rows, :columns])
    classical = _classical_spectral_transform(effective_matrix, scaled_coefficients)
    comparison = _comparison_frame(classical, transformed_block)

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit() -> Any:
        qml.apply(operator)
        return qml.state()

    _ = circuit()
    circuit_draw = qml.draw(circuit)()
    summary = {
        "run_id": demo["run_id"],
        "pennylane_available": True,
        "pennylane_version": qml.__version__,
        "source_case": research_matrix.metadata["source_case_name"],
        "case_name": research_matrix.metadata["source_case_name"],
        "matrix_source": "weighted_jacobian",
        "matrix_shape": list(matrix.shape),
        "full_matrix_shape": research_matrix.metadata.get("full_shape"),
        "used_matrix_shape": list(matrix.shape),
        "matrix_scope": research_matrix.metadata.get("matrix_scope"),
        "full_or_submatrix": (
            "full_matrix" if research_matrix.metadata.get("is_full_matrix") else "submatrix"
        ),
        "submatrix_selection_strategy": research_matrix.metadata.get("row_selection_strategy"),
        "condition_number": research_matrix.metadata.get("condition_number"),
        "singular_values": research_matrix.metadata.get("singular_values"),
        "normalization_factor": research_matrix.metadata["normalization_factor"],
        "pennylane_block_encoding_norm": block_encoding_normalization,
        "effective_spectral_norm": float(effective_singular_values[0]),
        "block_encoding_method": f"qml.qsvt block_encoding={demo['block_encoding']}",
        "qsvt_method": "qml.qsvt research-matrix block encoding",
        "qsvt_construction_type": "pennylane_block_encoded_matrix_qsvt",
        "simulator_backend": "default.qubit",
        "polynomial_degree": int(demo["polynomial_degree"]),
        "n_phase_angles": int(phases.size),
        "phase_synthesis_method": "pennylane_poly_to_angles",
        "phase_synthesis_angle_solver": str(demo["angle_solver"]),
        "max_abs_error": float(comparison["abs_error_to_classical_filter"].max()),
        "mean_abs_error": float(comparison["abs_error_to_classical_filter"].mean()),
        "max_error_vs_classical": float(comparison["abs_error_to_classical_filter"].max()),
        "mean_error_vs_classical": float(comparison["abs_error_to_classical_filter"].mean()),
        "simulation_success": True,
        "is_full_matrix_qsvt": bool(research_matrix.metadata.get("is_full_matrix")),
        "is_hardware_native_decomposition": False,
        "scope_note": (
            "Research-derived weighted-Jacobian QSVT simulation using PennyLane's "
            "matrix block encoding. Full IEEE300 circuit simulation is not claimed."
        ),
    }
    artifacts = _write_artifacts(
        output_dir=output_dir,
        resolved_config=resolved,
        metadata=research_matrix.metadata,
        singular_values=singular_values_frame(research_matrix),
        phases=phases,
        coefficients=scaled_coefficients,
        approximation_error=_approximation_error_frame(approximation),
        circuit_draw=circuit_draw,
        transformed_block=transformed_block,
        comparison=comparison,
        summary=summary,
    )
    logger.info("Completed PennyLane research-matrix QSVT demo %s", demo["run_id"])
    return {"output_dir": output_dir, "artifacts": artifacts, "summary": summary}


def _classical_spectral_transform(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    values = Polynomial(coefficients)(singular_values)
    return u @ np.diag(values) @ vh


def _comparison_frame(classical: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    rows = []
    for row in range(classical.shape[0]):
        for column in range(classical.shape[1]):
            rows.append(
                {
                    "row": row,
                    "column": column,
                    "classical_spectral_filter": float(classical[row, column]),
                    "qsvt_block_value": float(observed[row, column]),
                    "abs_error_to_classical_filter": abs(
                        float(observed[row, column] - classical[row, column])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _approximation_error_frame(approximation: Any) -> pd.DataFrame:
    grid = np.linspace(approximation.domain_min, approximation.domain_max, 512, dtype=np.float64)
    target = grid / (grid**2 + approximation.alpha)
    polynomial = Polynomial(
        np.asarray(approximation.power_coefficients) / approximation.scale_factor
    )
    values = polynomial(grid)
    return pd.DataFrame(
        {
            "normalized_singular_value": grid,
            "scaled_target": target / approximation.scale_factor,
            "scaled_polynomial": values,
            "abs_error": np.abs(values - target / approximation.scale_factor),
        }
    )


def _write_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    metadata: dict[str, Any],
    singular_values: pd.DataFrame,
    phases: np.ndarray,
    coefficients: np.ndarray,
    approximation_error: pd.DataFrame,
    circuit_draw: str,
    transformed_block: np.ndarray,
    comparison: pd.DataFrame,
    summary: dict[str, Any],
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
    (output_dir / "circuit_draw.txt").write_text(circuit_draw, encoding="utf-8")
    pd.DataFrame(transformed_block).to_csv(output_dir / "qsvt_matrix_results.csv", index=False)
    comparison.to_csv(output_dir / "comparison_to_classical.csv", index=False)
    write_json(output_dir / "circuit_summary.json", summary)
    return {
        "config_resolved": str(output_dir / "config_resolved.yaml"),
        "research_matrix_metadata": str(output_dir / "research_matrix_metadata.json"),
        "singular_values": str(output_dir / "singular_values.csv"),
        "phase_angles": str(output_dir / "phase_angles.csv"),
        "polynomial_coefficients": str(output_dir / "polynomial_coefficients.csv"),
        "approximation_error": str(output_dir / "approximation_error.csv"),
        "circuit_draw": str(output_dir / "circuit_draw.txt"),
        "qsvt_matrix_results": str(output_dir / "qsvt_matrix_results.csv"),
        "comparison_to_classical": str(output_dir / "comparison_to_classical.csv"),
        "circuit_summary": str(output_dir / "circuit_summary.json"),
        "run_log": str(output_dir / "run.log"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PennyLane research-matrix QSVT demo")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run = run_pennylane_matrix_qsvt(load_pennylane_matrix_config(args.config))
    print(f"PennyLane matrix QSVT complete: {run['output_dir']}")


if __name__ == "__main__":
    main()
