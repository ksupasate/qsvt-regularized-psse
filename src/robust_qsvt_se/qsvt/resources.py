from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from robust_qsvt_se.qsvt.polynomial import chebyshev_max_error

RESOURCE_NOTES = (
    "Proxy estimate only: Chebyshev target approximation, not QSVT phase synthesis, "
    "hardware execution, oracle construction, state preparation, or readout cost."
)


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    degree: int
    max_error: float
    target_error: float
    recommended_degree: int | None
    block_encoding_normalization: float
    normalized_domain_min: float
    normalized_domain_max: float
    effective_condition_number: float
    proxy_query_count: int
    notes: str = RESOURCE_NOTES

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "degree": self.degree,
            "max_error": self.max_error,
            "target_error": self.target_error,
            "recommended_degree": self.recommended_degree,
            "block_encoding_normalization": self.block_encoding_normalization,
            "normalized_domain_min": self.normalized_domain_min,
            "normalized_domain_max": self.normalized_domain_max,
            "effective_condition_number": self.effective_condition_number,
            "proxy_query_count": self.proxy_query_count,
            "notes": self.notes,
        }
        if extra:
            row.update(extra)
        return row


def estimate_qsvt_resources(
    singular_values: ArrayLike,
    *,
    alpha: float,
    degrees: list[int],
    grid_size: int,
    target_error: float,
) -> list[ResourceEstimate]:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if target_error <= 0.0:
        raise ValueError("target_error must be positive")
    if not degrees:
        raise ValueError("degrees must be non-empty")
    if any(degree < 0 for degree in degrees):
        raise ValueError("degrees must be nonnegative")
    if grid_size <= max(degrees) + 1:
        raise ValueError("grid_size must be greater than max(degrees) + 1")

    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty 1D array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and nonnegative")
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("singular_values must contain at least one positive value")

    sigma_max = float(np.max(positive))
    sigma_min = float(np.min(positive))
    domain_min = max(sigma_min / sigma_max, np.finfo(float).eps)
    domain_max = 1.0
    effective_condition_number = sigma_max / sigma_min

    sorted_degrees = sorted(set(int(degree) for degree in degrees))
    approximation_results = [
        chebyshev_max_error(
            alpha=alpha,
            block_encoding_normalization=sigma_max,
            degree=degree,
            domain_min=domain_min,
            domain_max=domain_max,
            grid_size=grid_size,
        )
        for degree in sorted_degrees
    ]
    recommended_degree = next(
        (result.degree for result in approximation_results if result.max_error <= target_error),
        None,
    )
    return [
        ResourceEstimate(
            degree=result.degree,
            max_error=result.max_error,
            target_error=float(target_error),
            recommended_degree=recommended_degree,
            block_encoding_normalization=sigma_max,
            normalized_domain_min=domain_min,
            normalized_domain_max=domain_max,
            effective_condition_number=effective_condition_number,
            proxy_query_count=2 * result.degree + 1,
        )
        for result in approximation_results
    ]
