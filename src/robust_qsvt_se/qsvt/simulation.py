from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev

from robust_qsvt_se.qsvt.phase_synthesis import PhaseSynthesisResult, qsp_response


def diagonal_demo_results(
    *,
    singular_values: np.ndarray,
    alpha: float,
    block_encoding_normalization: float,
    phase_result: PhaseSynthesisResult,
    chebyshev_polynomial: Chebyshev,
) -> pd.DataFrame:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if block_encoding_normalization <= 0.0:
        raise ValueError("block_encoding_normalization must be positive")
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty 1D array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and nonnegative")
    normalized = values / block_encoding_normalization
    if np.any(normalized > 1.0):
        raise ValueError("singular_values must not exceed block_encoding_normalization")

    exact = values / (values**2 + alpha)
    qsp_values = qsp_response(normalized, phase_result.phases) * phase_result.target_scale
    chebyshev_values = chebyshev_polynomial(normalized)
    return pd.DataFrame(
        {
            "singular_value": values,
            "normalized_singular_value": normalized,
            "exact_filter": exact,
            "qsp_filter": qsp_values,
            "chebyshev_filter": chebyshev_values,
            "qsp_abs_error": np.abs(qsp_values - exact),
            "chebyshev_abs_error": np.abs(chebyshev_values - exact),
        }
    )
