from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.norm_success import (
    NORM_RECOVERY_LIMITATION,
    compute_norm_success_diagnostics,
    estimate_success_probability_from_shots,
)


def test_norm_success_diagnostics_handle_small_success_probability() -> None:
    diagnostics = compute_norm_success_diagnostics(
        ridge_update=np.array([1.0, 0.0]),
        qsvt_vector=np.array([0.99, 0.01]),
        bounded_scaling_C=2.0,
        beta=3.0,
        residual_norm=1.0,
        success_probability_proxy=1.0e-6,
        norm_recovery_method="classical_simulator_metadata",
    )

    assert diagnostics["success_probability_proxy"] == pytest.approx(1.0e-6)
    assert diagnostics["diagnostic_rescaled_update_error"] > 0.0
    assert diagnostics["limitation"] == NORM_RECOVERY_LIMITATION


def test_success_probability_shot_proxy_is_reproducible() -> None:
    first = estimate_success_probability_from_shots(0.25, 1000, 123)
    second = estimate_success_probability_from_shots(0.25, 1000, 123)

    assert first == second
    assert first["standard_error"] == pytest.approx(np.sqrt(0.25 * 0.75 / 1000))
