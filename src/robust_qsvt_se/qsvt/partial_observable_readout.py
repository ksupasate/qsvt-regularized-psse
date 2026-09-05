from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_state(vec: np.ndarray, eps: float = 1.0e-15) -> tuple[np.ndarray, float]:
    """Return a unit-norm state vector and the original Euclidean norm."""

    values = np.asarray(vec, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("vec must be a one-dimensional state vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("vec must contain finite values")
    norm = float(np.linalg.norm(values))
    if norm <= float(eps):
        raise ValueError("cannot normalize a zero or near-zero state vector")
    return values / norm, norm


def basis_probability(state: np.ndarray, index: int) -> float:
    """Return the computational-basis probability ``|<index|state>|^2``."""

    values = _as_state(state)
    selected = int(index)
    if selected < 0 or selected >= values.size:
        raise IndexError("basis index out of range")
    return float(np.abs(values[selected]) ** 2)


def subset_probability(state: np.ndarray, indices: Sequence[int]) -> float:
    """Return the probability mass on a selected index subset."""

    values = _as_state(state)
    selected = _validate_indices(indices, values.size)
    if selected.size == 0:
        return 0.0
    return float(np.sum(np.abs(values[selected]) ** 2))


def linear_functional_overlap(state: np.ndarray, coeffs: np.ndarray) -> complex:
    """Return ``<c|state>`` after normalizing the coefficient vector ``c``."""

    values = _as_state(state)
    coefficient_values = np.asarray(coeffs, dtype=np.complex128)
    if coefficient_values.ndim != 1:
        raise ValueError("coeffs must be one-dimensional")
    if coefficient_values.shape != values.shape:
        raise ValueError("coeffs and state must have the same shape")
    if not np.all(np.isfinite(coefficient_values)):
        raise ValueError("coeffs must contain finite values")
    coefficient_norm = float(np.linalg.norm(coefficient_values))
    if coefficient_norm <= 1.0e-15:
        raise ValueError("coeffs must have nonzero norm")
    normalized_coefficients = coefficient_values / coefficient_norm
    return complex(np.vdot(normalized_coefficients, values))


def estimate_bernoulli_probability(
    p: float,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Estimate a probability with Bernoulli samples and return standard error."""

    probability = _clip_probability(p)
    shot_count = _validate_shots(shots)
    successes = int(rng.binomial(shot_count, probability))
    estimate = successes / float(shot_count)
    standard_error = float(np.sqrt(probability * (1.0 - probability) / float(shot_count)))
    return float(estimate), standard_error


def estimate_overlap_from_hadamard_proxy(
    overlap: complex,
    shots: int,
    rng: np.random.Generator,
    component: str = "real",
) -> tuple[float, float]:
    """Simulate a Hadamard-test-style readout for one overlap component.

    This is a small sampling proxy. For the real component it samples
    ``p_+ = (1 + Re(<c|psi>)) / 2`` and returns ``2 p_+ - 1``. The imaginary
    component uses the analogous expression with ``Im(<c|psi>)``.
    """

    shot_count = _validate_shots(shots)
    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'")
    target = float(np.real(overlap) if component == "real" else np.imag(overlap))
    target = float(np.clip(target, -1.0, 1.0))
    plus_probability = _clip_probability((1.0 + target) / 2.0)
    plus_counts = int(rng.binomial(shot_count, plus_probability))
    estimate = 2.0 * (plus_counts / float(shot_count)) - 1.0
    standard_error = float(np.sqrt(max(0.0, 1.0 - target**2) / float(shot_count)))
    return float(estimate), standard_error


def _as_state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.complex128)
    if values.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("state must contain finite values")
    return values


def _validate_indices(indices: Sequence[int], dimension: int) -> np.ndarray:
    selected = np.asarray([int(index) for index in indices], dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= dimension):
        raise IndexError("subset index out of range")
    return np.unique(selected)


def _validate_shots(shots: int) -> int:
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError("shots must be positive")
    return shot_count


def _clip_probability(p: float) -> float:
    if not np.isfinite(p):
        raise ValueError("probability must be finite")
    return float(np.clip(float(p), 0.0, 1.0))
