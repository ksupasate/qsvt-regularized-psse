from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


@dataclass(frozen=True, slots=True)
class ChebyshevApproximationResult:
    degree: int
    max_error: float
    domain_min: float
    domain_max: float
    coefficients: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class OddPolynomialApproximation:
    degree: int
    alpha: float
    domain_min: float
    domain_max: float
    block_encoding_normalization: float
    scale_factor: float
    max_error: float
    mean_error: float
    scaled_max_error: float
    scaled_mean_error: float
    power_coefficients: tuple[float, ...]

    @property
    def polynomial(self) -> Polynomial:
        return Polynomial(self.power_coefficients)


def regularized_filter_on_normalized_domain(
    normalized_sigma: np.ndarray,
    *,
    alpha: float,
    block_encoding_normalization: float,
) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if block_encoding_normalization <= 0.0:
        raise ValueError("block_encoding_normalization must be positive")
    sigma = np.asarray(normalized_sigma, dtype=np.float64) * block_encoding_normalization
    return sigma / (sigma**2 + alpha)


def chebyshev_max_error(
    *,
    alpha: float,
    block_encoding_normalization: float,
    degree: int,
    domain_min: float,
    domain_max: float,
    grid_size: int,
) -> ChebyshevApproximationResult:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if grid_size <= degree + 1:
        raise ValueError("grid_size must be greater than degree + 1")
    if not 0.0 <= domain_min < domain_max <= 1.0:
        raise ValueError("domain must satisfy 0 <= domain_min < domain_max <= 1")

    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    polynomial = Chebyshev.fit(grid, target, deg=degree, domain=[domain_min, domain_max])
    approximation = polynomial(grid)
    return ChebyshevApproximationResult(
        degree=int(degree),
        max_error=float(np.max(np.abs(approximation - target))),
        domain_min=float(domain_min),
        domain_max=float(domain_max),
        coefficients=tuple(float(value) for value in polynomial.coef),
    )


def fit_regularized_chebyshev(
    *,
    alpha: float,
    block_encoding_normalization: float,
    degree: int,
    domain_min: float,
    domain_max: float,
    grid_size: int,
) -> Chebyshev:
    """Fit the regularized filter on a normalized singular-value interval."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if grid_size <= degree + 1:
        raise ValueError("grid_size must be greater than degree + 1")
    if not 0.0 <= domain_min < domain_max <= 1.0:
        raise ValueError("domain must satisfy 0 <= domain_min < domain_max <= 1")
    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    return Chebyshev.fit(grid, target, deg=degree, domain=[domain_min, domain_max])


def fit_odd_regularized_polynomial(
    *,
    alpha: float,
    block_encoding_normalization: float,
    degree: int,
    domain_min: float,
    domain_max: float,
    grid_size: int,
) -> OddPolynomialApproximation:
    """Fit an odd QSVT-compatible polynomial for sigma / (sigma^2 + alpha).

    The fit uses p(x) = x q(x^2), where x is the normalized singular value.
    This enforces odd parity while approximating the regularized inverse on
    ``[domain_min, domain_max]``.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if block_encoding_normalization <= 0.0:
        raise ValueError("block_encoding_normalization must be positive")
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if grid_size <= degree + 1:
        raise ValueError("grid_size must be greater than degree + 1")
    if not 0.0 < domain_min < domain_max <= 1.0:
        raise ValueError("domain must satisfy 0 < domain_min < domain_max <= 1")

    reduced_degree = (degree - 1) // 2
    y_min = domain_min**2
    y_max = domain_max**2
    y_grid = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    q_target = block_encoding_normalization / ((block_encoding_normalization**2) * y_grid + alpha)
    q_polynomial = Chebyshev.fit(
        y_grid,
        q_target,
        deg=reduced_degree,
        domain=[y_min, y_max],
    ).convert(kind=Polynomial)

    power_coefficients = np.zeros(degree + 1, dtype=np.float64)
    for index, coefficient in enumerate(q_polynomial.coef):
        target_index = 2 * index + 1
        if target_index <= degree:
            power_coefficients[target_index] = coefficient

    polynomial = Polynomial(power_coefficients)
    grid = np.linspace(domain_min, domain_max, grid_size, dtype=np.float64)
    target = regularized_filter_on_normalized_domain(
        grid,
        alpha=alpha,
        block_encoding_normalization=block_encoding_normalization,
    )
    approximation = polynomial(grid)
    error = np.abs(approximation - target)

    qsvt_grid_size = max(grid_size * 4, 8193, 256 * degree + 1)
    qsvt_grid = np.linspace(-1.0, 1.0, qsvt_grid_size, dtype=np.float64)
    scale_factor = (
        max(
            1.0,
            float(np.max(np.abs(polynomial(qsvt_grid)))),
            float(np.max(np.abs(target))),
        )
        * 2.0
    )
    scaled_error = error / scale_factor
    return OddPolynomialApproximation(
        degree=int(degree),
        alpha=float(alpha),
        domain_min=float(domain_min),
        domain_max=float(domain_max),
        block_encoding_normalization=float(block_encoding_normalization),
        scale_factor=scale_factor,
        max_error=float(np.max(error)),
        mean_error=float(np.mean(error)),
        scaled_max_error=float(np.max(scaled_error)),
        scaled_mean_error=float(np.mean(scaled_error)),
        power_coefficients=tuple(float(value) for value in power_coefficients),
    )
