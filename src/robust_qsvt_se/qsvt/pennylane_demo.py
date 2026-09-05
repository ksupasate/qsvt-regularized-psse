from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.logging import configure_run_logger


def load_pennylane_demo_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("PennyLane QSVT demo config must contain a mapping")
    return validate_pennylane_demo_config(loaded)


def validate_pennylane_demo_config(config: dict[str, Any]) -> dict[str, Any]:
    demo = dict(config.get("demo", config))
    defaults = {
        "run_id": "qsvt_pennylane_demo",
        "output_dir": "outputs/qsvt_pennylane_demo",
        "alpha": 1.0,
        "matrix_diagonal": [0.2, 0.6],
        "polynomial_degree": 1,
        "block_encoding": "embedding",
    }
    resolved = {**defaults, **demo}
    if not isinstance(resolved["run_id"], str) or not resolved["run_id"]:
        raise ValueError("demo.run_id must be a non-empty string")
    if not isinstance(resolved["output_dir"], str) or not resolved["output_dir"]:
        raise ValueError("demo.output_dir must be a non-empty string")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("demo.alpha must be positive")
    if int(resolved["polynomial_degree"]) != 1:
        raise ValueError("demo.polynomial_degree must be 1 for this small proof of concept")
    diagonal = np.asarray(resolved["matrix_diagonal"], dtype=np.float64)
    if diagonal.ndim != 1 or diagonal.size != 2:
        raise ValueError("demo.matrix_diagonal must contain exactly two values")
    if np.any(diagonal < 0.0) or np.any(diagonal > 1.0):
        raise ValueError("demo.matrix_diagonal values must lie in [0, 1]")
    resolved["matrix_diagonal"] = diagonal.tolist()
    return {"demo": resolved}


def run_pennylane_demo(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_pennylane_demo_config(config)["demo"]
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting PennyLane QSVT demo %s", resolved["run_id"])

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("PennyLane is required for this demo") from exc

    alpha = float(resolved["alpha"])
    singular_values = np.asarray(resolved["matrix_diagonal"], dtype=np.float64)
    matrix = np.diag(singular_values)
    exact_filter = singular_values / (singular_values**2 + alpha)
    coefficient = _best_linear_coefficient(singular_values, exact_filter)
    polynomial = np.array([0.0, coefficient], dtype=np.float64)

    op = qml.qsvt(matrix, polynomial, encoding_wires=[0, 1])
    qsvt_matrix = qml.matrix(op)
    transformed_block = np.real(qsvt_matrix[:2, :2]).astype(np.float64)
    transformed_diagonal = np.diag(transformed_block)
    phases = np.asarray(op.data[1:], dtype=np.float64)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit() -> Any:
        qml.apply(op)
        return qml.state()

    state = circuit()
    circuit_draw = qml.draw(circuit)()
    comparison = pd.DataFrame(
        {
            "singular_value": singular_values,
            "exact_filter": exact_filter,
            "linear_polynomial_filter": coefficient * singular_values,
            "pennylane_qsvt_block_value": transformed_diagonal,
            "abs_error_to_classical_filter": np.abs(transformed_diagonal - exact_filter),
        }
    )
    summary = {
        "run_id": resolved["run_id"],
        "pennylane_available": True,
        "pennylane_version": qml.__version__,
        "matrix_size": 2,
        "block_encoding_method": "qml.qsvt default BlockEncode",
        "qsvt_method": "qml.qsvt",
        "simulator_backend": "default.qubit",
        "polynomial_coefficients": polynomial.tolist(),
        "n_phase_angles": int(phases.size),
        "max_abs_error": float(comparison["abs_error_to_classical_filter"].max()),
        "mean_abs_error": float(comparison["abs_error_to_classical_filter"].mean()),
        "scope_note": "Tiny matrix proof of concept; not an IEEE case circuit.",
    }

    artifacts = _write_pennylane_artifacts(
        output_dir=output_dir,
        resolved_config={"demo": resolved},
        phases=phases,
        circuit_draw=circuit_draw,
        state=np.asarray(state),
        comparison=comparison,
        summary=summary,
    )
    logger.info("Completed PennyLane QSVT demo %s", resolved["run_id"])
    return {"output_dir": output_dir, "artifacts": artifacts, "summary": summary}


def _best_linear_coefficient(singular_values: np.ndarray, exact_filter: np.ndarray) -> float:
    denominator = float(np.dot(singular_values, singular_values))
    if denominator <= 0.0:
        raise ValueError("singular values must contain a positive value")
    return float(np.clip(np.dot(singular_values, exact_filter) / denominator, -1.0, 1.0))


def _write_pennylane_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    phases: np.ndarray,
    circuit_draw: str,
    state: np.ndarray,
    comparison: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    config_path = output_dir / "config_resolved.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=True)
    phase_path = output_dir / "phase_angles.csv"
    draw_path = output_dir / "circuit_draw.txt"
    results_path = output_dir / "qsvt_pennylane_results.csv"
    summary_path = output_dir / "qsvt_pennylane_summary.json"
    circuit_summary_path = output_dir / "circuit_summary.json"
    comparison_path = output_dir / "comparison_to_classical.csv"

    pd.DataFrame({"phase_index": np.arange(len(phases)), "phase_angle": phases}).to_csv(
        phase_path,
        index=False,
    )
    draw_path.write_text(circuit_draw, encoding="utf-8")
    pd.DataFrame(
        {
            "state_index": np.arange(state.size),
            "amplitude_real": np.real(state),
            "amplitude_imag": np.imag(state),
        }
    ).to_csv(results_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_json(summary_path, summary)
    write_json(circuit_summary_path, summary)
    return {
        "config_resolved": str(config_path),
        "phase_angles": str(phase_path),
        "circuit_draw": str(draw_path),
        "qsvt_pennylane_results": str(results_path),
        "qsvt_pennylane_summary": str(summary_path),
        "comparison_to_classical": str(comparison_path),
        "run_log": str(output_dir / "run.log"),
    }
