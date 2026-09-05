from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from robust_qsvt_se.utils.io import ensure_directory, write_json
from robust_qsvt_se.utils.logging import configure_run_logger


def load_qiskit_demo_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Qiskit QSVT demo config must contain a mapping")
    return validate_qiskit_demo_config(loaded)


def validate_qiskit_demo_config(config: dict[str, Any]) -> dict[str, Any]:
    demo = dict(config.get("demo", config))
    defaults = {
        "run_id": "qsvt_qiskit_demo",
        "output_dir": "outputs/qsvt_qiskit_demo",
        "alpha": 1.0,
        "singular_values": [0.2, 0.6],
        "polynomial_degree": 1,
    }
    resolved = {**defaults, **demo}
    if not isinstance(resolved["run_id"], str) or not resolved["run_id"]:
        raise ValueError("demo.run_id must be a non-empty string")
    if not isinstance(resolved["output_dir"], str) or not resolved["output_dir"]:
        raise ValueError("demo.output_dir must be a non-empty string")
    if float(resolved["alpha"]) <= 0.0:
        raise ValueError("demo.alpha must be positive")
    if int(resolved["polynomial_degree"]) != 1:
        raise ValueError("demo.polynomial_degree must be 1 for this toy circuit")
    singular_values = np.asarray(resolved["singular_values"], dtype=np.float64)
    if singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("demo.singular_values must be a non-empty numeric list")
    if np.any(singular_values < 0.0) or np.any(singular_values > 1.0):
        raise ValueError("demo.singular_values must lie in [0, 1]")
    resolved["singular_values"] = singular_values.tolist()
    return {"demo": resolved}


def run_qiskit_demo(config: dict[str, Any]) -> dict[str, Any]:
    resolved = validate_qiskit_demo_config(config)["demo"]
    output_dir = ensure_directory(Path(resolved["output_dir"]))
    logger = configure_run_logger(output_dir / "run.log")
    logger.info("Starting Qiskit QSP/QSVT-style demo %s", resolved["run_id"])

    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("Qiskit is required for this demo") from exc

    alpha = float(resolved["alpha"])
    singular_values = np.asarray(resolved["singular_values"], dtype=np.float64)
    exact_filter = singular_values / (singular_values**2 + alpha)
    coefficient = _best_linear_coefficient(singular_values, exact_filter)
    phases = _fit_scalar_circuit_phases(
        singular_values,
        coefficient * singular_values,
        n_phases=4,
    )

    simulation_rows = []
    representative_circuit = None
    for singular_value, target in zip(singular_values, exact_filter, strict=True):
        circuit = _build_scalar_qsp_circuit(float(singular_value), phases, QuantumCircuit)
        if representative_circuit is None:
            representative_circuit = circuit
        state = Statevector.from_instruction(circuit)
        amplitude = complex(state.data[0])
        transformed = float(np.real(amplitude))
        simulation_rows.append(
            {
                "singular_value": float(singular_value),
                "exact_filter": float(target),
                "linear_polynomial_filter": float(coefficient * singular_value),
                "qiskit_statevector_real_amplitude": transformed,
                "qiskit_statevector_imag_amplitude": float(np.imag(amplitude)),
                "abs_error_to_classical_filter": abs(transformed - float(target)),
            }
        )
    if representative_circuit is None:
        raise RuntimeError("failed to construct a representative Qiskit circuit")

    try:
        import qiskit_aer  # type: ignore[import-not-found]

        aer_available = True
        aer_version = getattr(qiskit_aer, "__version__", "unknown")
    except Exception:
        aer_available = False
        aer_version = None

    gate_counts = dict(representative_circuit.count_ops())
    comparison = pd.DataFrame(simulation_rows)
    summary = {
        "run_id": resolved["run_id"],
        "qiskit_available": True,
        "circuit_type": "toy scalar QSP/QSVT-style circuit",
        "is_full_matrix_qsvt": False,
        "n_qubits": int(representative_circuit.num_qubits),
        "circuit_depth": int(representative_circuit.depth()),
        "gate_counts": gate_counts,
        "gate_count_total": int(sum(gate_counts.values())),
        "simulation_backend": "qiskit.quantum_info.Statevector",
        "qiskit_aer_available": aer_available,
        "qiskit_aer_version": aer_version,
        "polynomial_coefficients": [0.0, coefficient],
        "n_phase_angles": int(phases.size),
        "phase_synthesis_method": "scipy_least_squares_qiskit_toy_scalar_circuit",
        "max_abs_error": float(comparison["abs_error_to_classical_filter"].max()),
        "mean_abs_error": float(comparison["abs_error_to_classical_filter"].mean()),
        "scope_note": (
            "Toy scalar QSP/QSVT-style circuit only; not production matrix QSVT and not "
            "used for IEEE benchmark matrices."
        ),
    }
    artifacts = _write_qiskit_artifacts(
        output_dir=output_dir,
        resolved_config={"demo": resolved},
        phases=phases,
        circuit=representative_circuit,
        gate_counts=gate_counts,
        comparison=comparison,
        summary=summary,
    )
    logger.info("Completed Qiskit QSP/QSVT-style demo %s", resolved["run_id"])
    return {"output_dir": output_dir, "artifacts": artifacts, "summary": summary}


def _best_linear_coefficient(singular_values: np.ndarray, exact_filter: np.ndarray) -> float:
    denominator = float(np.dot(singular_values, singular_values))
    if denominator <= 0.0:
        raise ValueError("singular values must contain a positive value")
    return float(np.clip(np.dot(singular_values, exact_filter) / denominator, -1.0, 1.0))


def _build_scalar_qsp_circuit(
    singular_value: float,
    phases: np.ndarray,
    quantum_circuit_type: Any,
) -> Any:
    circuit = quantum_circuit_type(1, name="toy_scalar_qsp")
    signal_angle = 2.0 * np.arccos(np.clip(singular_value, -1.0, 1.0))
    for index, phase in enumerate(phases):
        circuit.rz(2.0 * float(phase), 0)
        if index < len(phases) - 1:
            circuit.ry(signal_angle, 0)
    return circuit


def _fit_scalar_circuit_phases(
    singular_values: np.ndarray,
    targets: np.ndarray,
    *,
    n_phases: int,
) -> np.ndarray:
    if n_phases < 2:
        raise ValueError("n_phases must be at least 2")

    def residual(phases: np.ndarray) -> np.ndarray:
        return (
            np.array(
                [_scalar_circuit_amplitude(float(value), phases) for value in singular_values],
                dtype=np.float64,
            )
            - targets
        )

    guesses = [
        np.zeros(n_phases, dtype=np.float64),
        np.linspace(-np.pi / 4.0, np.pi / 4.0, n_phases, dtype=np.float64),
    ]
    best = None
    for guess in guesses:
        result = least_squares(residual, guess, max_nfev=2000)
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError("failed to fit toy Qiskit scalar circuit phases")
    return np.asarray(best.x, dtype=np.float64)


def _scalar_circuit_amplitude(singular_value: float, phases: np.ndarray) -> float:
    state = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    signal_angle = 2.0 * np.arccos(np.clip(singular_value, -1.0, 1.0))
    for index, phase in enumerate(phases):
        state = _rz(2.0 * float(phase)) @ state
        if index < len(phases) - 1:
            state = _ry(signal_angle) @ state
    return float(np.real(state[0]))


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _ry(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _write_qiskit_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    phases: np.ndarray,
    circuit: Any,
    gate_counts: dict[str, int],
    comparison: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    config_path = output_dir / "config_resolved.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=True)
    phase_path = output_dir / "phase_angles.csv"
    summary_path = output_dir / "circuit_summary.json"
    gate_counts_path = output_dir / "gate_counts.json"
    draw_path = output_dir / "circuit_draw.txt"
    simulation_path = output_dir / "qiskit_simulation_results.csv"
    comparison_path = output_dir / "comparison_to_classical.csv"

    pd.DataFrame({"phase_index": np.arange(len(phases)), "phase_angle": phases}).to_csv(
        phase_path,
        index=False,
    )
    write_json(summary_path, summary)
    write_json(gate_counts_path, gate_counts)
    draw_path.write_text(str(circuit.draw(output="text")), encoding="utf-8")
    comparison.to_csv(simulation_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return {
        "config_resolved": str(config_path),
        "circuit_summary": str(summary_path),
        "gate_counts": str(gate_counts_path),
        "circuit_draw": str(draw_path),
        "phase_angles": str(phase_path),
        "qiskit_simulation_results": str(simulation_path),
        "comparison_to_classical": str(comparison_path),
        "run_log": str(output_dir / "run.log"),
    }
