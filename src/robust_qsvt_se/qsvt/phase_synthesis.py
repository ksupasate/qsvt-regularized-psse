from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from robust_qsvt_se.qsvt.polynomial import regularized_filter_on_normalized_domain
from robust_qsvt_se.utils.io import ensure_directory


@dataclass(frozen=True, slots=True)
class PhaseSynthesisResult:
    phases: np.ndarray
    degree: int
    target_scale: float
    max_abs_error: float
    mean_abs_error: float
    optimization_success: bool
    optimization_message: str
    n_function_evaluations: int

    def to_rows(self) -> list[dict[str, float | int]]:
        return [
            {"phase_index": index, "phase_angle": float(phase)}
            for index, phase in enumerate(self.phases)
        ]


@dataclass(frozen=True, slots=True)
class CachedPhaseResult:
    phases: np.ndarray
    metadata: dict[str, Any]
    cache_hit: bool


def validate_qsvt_polynomial(
    coefficients: np.ndarray,
    *,
    parity: str = "odd",
    grid_size: int = 4097,
    bound_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coefficients must be a non-empty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")
    if parity not in {"odd", "even", "any"}:
        raise ValueError("parity must be odd, even, or any")
    if parity == "odd" and np.any(np.abs(values[::2]) > bound_tolerance):
        raise ValueError("odd QSVT polynomial has nonzero even-power coefficients")
    if parity == "even" and np.any(np.abs(values[1::2]) > bound_tolerance):
        raise ValueError("even QSVT polynomial has nonzero odd-power coefficients")
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    polynomial_values = np.polynomial.Polynomial(values)(grid)
    max_abs = float(np.max(np.abs(polynomial_values)))
    if max_abs > 1.0 + bound_tolerance:
        raise ValueError(f"QSVT polynomial is not bounded by 1 on [-1, 1]: {max_abs}")
    return {
        "parity": parity,
        "max_abs_on_unit_interval": max_abs,
        "coefficient_count": int(values.size),
        "degree": int(values.size - 1),
    }


def synthesize_pennylane_phases_cached(
    coefficients: np.ndarray,
    *,
    angle_solver: str,
    cache_dir: str | Path = "outputs/qsvt_phase_cache",
    cache_metadata: dict[str, Any] | None = None,
) -> CachedPhaseResult:
    """Synthesize QSVT phases with PennyLane and cache by coefficient hash."""
    values = np.asarray(coefficients, dtype=np.float64)
    validation = validate_qsvt_polynomial(values, parity="odd")
    cache_root = ensure_directory(cache_dir)
    key = _phase_cache_key(values, angle_solver=angle_solver, metadata=cache_metadata or {})
    phase_path = cache_root / f"{key}_phase_angles.csv"
    metadata_path = cache_root / f"{key}_metadata.json"
    if phase_path.is_file() and metadata_path.is_file():
        phases = np.loadtxt(phase_path, delimiter=",", skiprows=1, usecols=1)
        if phases.ndim == 0:
            phases = np.asarray([float(phases)])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_key"] = key
        return CachedPhaseResult(
            phases=np.asarray(phases, dtype=np.float64), metadata=metadata, cache_hit=True
        )

    try:
        import pennylane as qml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError("PennyLane is required for QSVT phase synthesis") from exc

    phases = np.asarray(
        qml.poly_to_angles(values, "QSVT", angle_solver=angle_solver),
        dtype=np.float64,
    )
    if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
        raise RuntimeError("PennyLane produced invalid QSVT phases")
    if np.allclose(phases, 0.0):
        raise RuntimeError("PennyLane produced all-zero phases; refusing dummy phase output")
    metadata = {
        **validation,
        **(cache_metadata or {}),
        "phase_synthesis_method": "pennylane_poly_to_angles",
        "phase_synthesis_success": True,
        "phase_synthesis_message": "PennyLane poly_to_angles completed",
        "phase_synthesis_angle_solver": angle_solver,
        "phase_synthesis_backend": f"pennylane-{qml.__version__}",
        "n_phase_angles": int(phases.size),
        "cache_key": key,
        "cache_hit": False,
    }
    with phase_path.open("w", encoding="utf-8") as file:
        file.write("phase_index,phase_angle\n")
        for index, phase in enumerate(phases):
            file.write(f"{index},{float(phase):.17g}\n")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return CachedPhaseResult(phases=phases, metadata=metadata, cache_hit=False)


def qsp_response(normalized_sigma: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Return the real top-left QSP response for a scalar signal operator."""
    x_values = np.asarray(normalized_sigma, dtype=np.float64)
    phase_values = np.asarray(phases, dtype=np.float64)
    if x_values.ndim != 1:
        raise ValueError("normalized_sigma must be a 1D array")
    if phase_values.ndim != 1 or phase_values.size == 0:
        raise ValueError("phases must be a non-empty 1D array")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(phase_values)):
        raise ValueError("normalized_sigma and phases must be finite")
    if np.any(np.abs(x_values) > 1.0):
        raise ValueError("normalized_sigma values must lie in [-1, 1]")

    responses = np.empty_like(x_values, dtype=np.float64)
    for index, value in enumerate(x_values):
        off_diagonal = 1j * np.sqrt(max(0.0, 1.0 - float(value) ** 2))
        signal = np.array([[value, off_diagonal], [off_diagonal, value]], dtype=np.complex128)
        unitary = _phase_rotation(phase_values[0])
        for phase in phase_values[1:]:
            unitary = unitary @ signal @ _phase_rotation(float(phase))
        responses[index] = float(np.real(unitary[0, 0]))
    return responses


def synthesize_qsp_phases(
    *,
    alpha: float,
    block_encoding_normalization: float,
    degree: int,
    domain_min: float,
    domain_max: float,
    grid_size: int,
    seed: int,
    max_nfev: int = 4000,
) -> PhaseSynthesisResult:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if block_encoding_normalization <= 0.0:
        raise ValueError("block_encoding_normalization must be positive")
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if grid_size <= degree + 1:
        raise ValueError("grid_size must be greater than degree + 1")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if not 0.0 <= domain_min < domain_max <= 1.0:
        raise ValueError("domain must satisfy 0 <= domain_min < domain_max <= 1")

    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    target_scale = max(float(np.max(np.abs(target))), np.finfo(float).eps)
    scaled_target = target / target_scale
    rng = np.random.default_rng(seed)

    def residual(phases: np.ndarray) -> np.ndarray:
        return qsp_response(grid, phases) - scaled_target

    initial_points = [
        np.zeros(degree + 1, dtype=np.float64),
        rng.normal(0.0, 0.05, size=degree + 1),
        rng.normal(0.0, 0.2, size=degree + 1),
    ]
    best = None
    for initial in initial_points:
        result = least_squares(
            residual,
            initial,
            max_nfev=max_nfev,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        if best is None or result.cost < best.cost:
            best = result

    if best is None:
        raise RuntimeError("QSP phase optimization did not run")
    approximation = qsp_response(grid, best.x) * target_scale
    error = approximation - target
    return PhaseSynthesisResult(
        phases=np.asarray(best.x, dtype=np.float64),
        degree=int(degree),
        target_scale=target_scale,
        max_abs_error=float(np.max(np.abs(error))),
        mean_abs_error=float(np.mean(np.abs(error))),
        optimization_success=bool(best.success),
        optimization_message=str(best.message),
        n_function_evaluations=int(best.nfev),
    )


def _phase_rotation(phase: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(1j * phase), 0.0],
            [0.0, np.exp(-1j * phase)],
        ],
        dtype=np.complex128,
    )


def _phase_cache_key(
    coefficients: np.ndarray,
    *,
    angle_solver: str,
    metadata: dict[str, Any],
) -> str:
    payload = {
        "coefficients": [float(value) for value in np.asarray(coefficients, dtype=np.float64)],
        "angle_solver": angle_solver,
        "metadata": metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
