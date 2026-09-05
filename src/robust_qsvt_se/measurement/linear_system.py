from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class WeightedSystem:
    """Weighted linearized state-estimation system.

    The benchmark solves the already-weighted system

        H_tilde x = r_tilde

    where any covariance weighting has already been folded into `H_tilde` and
    `r_tilde`. Residual metrics should therefore be computed on this weighted
    system and not weighted a second time.
    """

    H_tilde: ArrayLike
    r_tilde: ArrayLike
    x_true: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        H_tilde = np.asarray(self.H_tilde, dtype=np.float64)
        r_tilde = np.asarray(self.r_tilde, dtype=np.float64)
        x_true = None if self.x_true is None else np.asarray(self.x_true, dtype=np.float64)

        if H_tilde.ndim != 2:
            raise ValueError("H_tilde must be a 2D array")
        if r_tilde.ndim != 1:
            raise ValueError("r_tilde must be a 1D array")
        if H_tilde.shape[0] != r_tilde.shape[0]:
            raise ValueError(
                "H_tilde row count must match r_tilde length: "
                f"{H_tilde.shape[0]} != {r_tilde.shape[0]}"
            )
        if H_tilde.shape[1] == 0 or H_tilde.shape[0] == 0:
            raise ValueError("H_tilde must have nonzero dimensions")
        if x_true is not None and x_true.shape != (H_tilde.shape[1],):
            raise ValueError(
                "x_true must have shape (n_states,): "
                f"expected {(H_tilde.shape[1],)}, got {x_true.shape}"
            )
        if not np.all(np.isfinite(H_tilde)):
            raise ValueError("H_tilde must contain only finite values")
        if not np.all(np.isfinite(r_tilde)):
            raise ValueError("r_tilde must contain only finite values")
        if x_true is not None and not np.all(np.isfinite(x_true)):
            raise ValueError("x_true must contain only finite values")

        self.H_tilde = H_tilde
        self.r_tilde = r_tilde
        self.x_true = x_true
        self.metadata = dict(self.metadata)

    @property
    def n_measurements(self) -> int:
        return int(self.H_tilde.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.H_tilde.shape[1])

    def singular_values(self) -> FloatArray:
        return np.linalg.svd(self.H_tilde, compute_uv=False)

    def rank(self, tol: float | None = None) -> int:
        if tol is None:
            return int(np.linalg.matrix_rank(self.H_tilde))
        singular_values = self.singular_values()
        return int(np.count_nonzero(singular_values > tol))

    def effective_rank(self, rtol: float = 1e-10) -> int:
        singular_values = self.singular_values()
        if singular_values.size == 0:
            return 0
        threshold = float(rtol) * float(np.max(singular_values))
        return int(np.count_nonzero(singular_values > threshold))

    def condition_number(self, tol: float = 1e-15) -> float:
        singular_values = self.singular_values()
        if singular_values.size == 0:
            return float("inf")
        sigma_max = float(np.max(singular_values))
        sigma_min = float(np.min(singular_values))
        if sigma_max == 0.0 or sigma_min <= tol:
            return float("inf")
        return sigma_max / sigma_min

    def residual(self, x_hat: ArrayLike) -> FloatArray:
        x_hat_array = np.asarray(x_hat, dtype=np.float64)
        if x_hat_array.shape != (self.n_states,):
            raise ValueError(
                "x_hat must have shape (n_states,): "
                f"expected {(self.n_states,)}, got {x_hat_array.shape}"
            )
        return self.H_tilde @ x_hat_array - self.r_tilde

    def residual_norm(self, x_hat: ArrayLike) -> float:
        return float(np.linalg.norm(self.residual(x_hat), ord=2))

    def weighted_residual(self, x_hat: ArrayLike) -> float:
        return self.weighted_residual_norm(x_hat)

    def weighted_residual_norm(self, x_hat: ArrayLike) -> float:
        return self.residual_norm(x_hat)

    def weighted_residual_quadratic(self, x_hat: ArrayLike) -> float:
        norm = self.weighted_residual_norm(x_hat)
        return float(norm**2)

    def rmse(self, x_hat: ArrayLike) -> float | None:
        if self.x_true is None:
            return None
        x_hat_array = np.asarray(x_hat, dtype=np.float64)
        if x_hat_array.shape != self.x_true.shape:
            raise ValueError(
                f"x_hat shape {x_hat_array.shape} does not match x_true shape {self.x_true.shape}"
            )
        return float(np.sqrt(np.mean((x_hat_array - self.x_true) ** 2)))
