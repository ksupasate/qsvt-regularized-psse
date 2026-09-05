from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_singular_values(sigma: ArrayLike) -> FloatArray:
    values = np.asarray(sigma, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("singular values must be finite")
    if np.any(values < 0.0):
        raise ValueError("singular values must be nonnegative")
    return values


def inverse_filter(sigma: ArrayLike, eps: float = 0.0) -> FloatArray:
    """Return the stable pseudoinverse filter `1 / sigma` above `eps`."""

    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    values = _as_singular_values(sigma)
    return np.divide(1.0, values, out=np.zeros_like(values), where=values > eps)


def qsvt_unregularized_inverse_filter(sigma: ArrayLike, cutoff: float) -> FloatArray:
    """Return an inverse-like QSVT ablation target with a numerical cutoff.

    This models the unstable unregularized inverse target `P_inv(sigma)=1/sigma`
    while clipping singular values below `cutoff` to avoid infinities. It is an
    ablation target, not the proposed regularized filter.
    """

    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    values = _as_singular_values(sigma)
    clipped = np.maximum(values, float(cutoff))
    return 1.0 / clipped


def ridge_filter(sigma: ArrayLike, alpha: float) -> FloatArray:
    """Return the Tikhonov spectral filter `sigma / (sigma^2 + alpha)`."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    values = _as_singular_values(sigma)
    return values / (values**2 + float(alpha))


def truncated_inverse_filter(sigma: ArrayLike, tau: float) -> FloatArray:
    """Return `1 / sigma` for singular values above `tau`, otherwise zero."""

    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    values = _as_singular_values(sigma)
    return np.divide(1.0, values, out=np.zeros_like(values), where=values > tau)


def qsvt_regularized_filter(sigma: ArrayLike, alpha: float) -> FloatArray:
    """Milestone-1 QSVT-inspired target filter.

    This is a classical simulation of the singular-value transformation target,
    not a hardware QSVT circuit or a phase-factor synthesis implementation.
    """

    return ridge_filter(sigma, alpha=alpha)
