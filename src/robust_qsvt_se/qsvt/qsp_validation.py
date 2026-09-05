from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain


@dataclass(frozen=True, slots=True)
class PhaseValidationResult:
    frame: pd.DataFrame
    max_abs_error: float
    mean_abs_error: float
    max_scaled_abs_error: float
    mean_scaled_abs_error: float
    backend: str


def validate_pennylane_qsvt_phases(
    *,
    power_coefficients: np.ndarray,
    scale_factor: float,
    phases: np.ndarray,
    alpha: float,
    block_encoding_normalization: float,
    domain_min: float,
    domain_max: float,
    grid_size: int,
) -> PhaseValidationResult:
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    coefficients = np.asarray(power_coefficients, dtype=np.float64)
    phase_values = np.asarray(phases, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("power_coefficients must be a non-empty 1D array")
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a non-empty 1D array")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(phase_values)):
        raise ValueError("coefficients and phases must be finite")

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for phase validation") from exc

    scaled_coefficients = coefficients / scale_factor
    scaled_polynomial = Polynomial(scaled_coefficients)
    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    scaled_target = target / scale_factor
    scaled_polynomial_values = scaled_polynomial(grid)
    response = np.array(
        [_scalar_qsvt_response(float(value), phase_values, qml) for value in grid],
        dtype=np.float64,
    )
    unscaled_response = response * scale_factor
    frame = pd.DataFrame(
        {
            "normalized_singular_value": grid,
            "scaled_target": scaled_target,
            "scaled_polynomial": scaled_polynomial_values,
            "scaled_phase_response": response,
            "target": target,
            "phase_response": unscaled_response,
            "phase_abs_error": np.abs(unscaled_response - target),
            "phase_scaled_abs_error": np.abs(response - scaled_target),
            "phase_vs_polynomial_abs_error": np.abs(response - scaled_polynomial_values),
        }
    )
    return PhaseValidationResult(
        frame=frame,
        max_abs_error=float(frame["phase_abs_error"].max()),
        mean_abs_error=float(frame["phase_abs_error"].mean()),
        max_scaled_abs_error=float(frame["phase_scaled_abs_error"].max()),
        mean_scaled_abs_error=float(frame["phase_scaled_abs_error"].mean()),
        backend=f"pennylane-{qml.__version__}",
    )


def _scalar_qsvt_response(x_value: float, phases: np.ndarray, qml: object) -> float:
    block_encoding = qml.RX(2.0 * np.arccos(np.clip(x_value, -1.0, 1.0)), wires=0)
    projectors = [qml.PCPhase(float(phase), dim=1, wires=0) for phase in phases]
    operator = qml.QSVT(block_encoding, projectors)
    matrix = qml.matrix(operator, wire_order=[0])
    return float(np.real(matrix[0, 0]))
