from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_phase_synthesis_8x8_rescue import (
    ATTEMPT_COLUMNS,
    AttemptSpec,
    RescueTarget,
    _empty_attempt_row,
    attempt_meets_rescue_criteria,
    build_rescue_polynomial,
    enforce_odd_parity,
    normalized_singular_values_with_gamma,
    summarize_rescue_attempts,
    target_scaling_factor,
)


def test_rescue_attempt_schema() -> None:
    target = RescueTarget("mock", 8, "high_leverage", 1.0e-2, 1.0e-2, 5)
    attempt = AttemptSpec(11, 1.0, 0.95, "actual_singular_weighted", "unit")
    row = _empty_attempt_row(target, attempt)
    assert list(pd.DataFrame([row], columns=ATTEMPT_COLUMNS).columns) == ATTEMPT_COLUMNS


def test_target_scaling_boundedness() -> None:
    scale = target_scaling_factor(raw_max_abs_dense=10.0, target_scale_safety=0.8)
    assert np.isclose(scale, 0.08)
    scale_small = target_scaling_factor(raw_max_abs_dense=0.5, target_scale_safety=0.8)
    assert np.isclose(scale_small, 0.8)


def test_gamma_multiplier_normalized_singular_values() -> None:
    singular_values = np.array([4.0, 2.0])
    normalized = normalized_singular_values_with_gamma(
        singular_values,
        gamma_base=4.0,
        gamma_multiplier=2.0,
    )
    assert np.allclose(normalized, [0.5, 0.25])


def test_odd_parity_enforcement() -> None:
    coefficients, violation = enforce_odd_parity(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    assert np.allclose(coefficients, [0.0, 2.0, 0.0, 4.0])
    assert np.isclose(violation, np.linalg.norm([1.0, 3.0]))


def test_build_rescue_polynomial_respects_safety_cap() -> None:
    singular_values = np.array([1.0, 0.5])
    build = build_rescue_polynomial(
        singular_values=singular_values,
        alpha=1.0e-2,
        gamma=1.0,
        degree=3,
        target_scale_safety=0.5,
        approximation_mode="actual_singular_weighted",
        dense_grid_size=257,
    )
    grid = np.linspace(-1.0, 1.0, 257)
    assert np.max(np.abs(build.polynomial(grid))) <= 0.5 + 1.0e-12
    assert build.effective_C_alpha >= build.C_alpha


def test_rescue_success_criteria() -> None:
    row = {
        "phase_synthesis_status": "completed",
        "qsvt_circuit_status": "completed",
        "simulation_status": "completed",
        "circuit_vs_polynomial_fro_error": 1.0e-12,
        "circuit_vs_ridge_relative_update_error": 5.0e-3,
        "absolute_update_error": 1.0e-3,
        "residual_gap": 2.0e-2,
    }
    config = {
        "transform_tolerance": 1.0e-10,
        "relative_update_tolerance": 1.0e-2,
        "absolute_update_tolerance": 1.0e-6,
        "residual_gap_tolerance": 1.0e-2,
    }
    assert attempt_meets_rescue_criteria(row, config)
    row["circuit_vs_ridge_relative_update_error"] = 2.0e-2
    row["absolute_update_error"] = 5.0e-7
    row["residual_gap"] = 5.0e-3
    assert attempt_meets_rescue_criteria(row, config)


def test_failure_recording_and_summary_selection() -> None:
    target = RescueTarget("mock", 8, "high_leverage", 1.0e-2, 1.0e-2, 5)
    failed = _empty_attempt_row(
        target,
        AttemptSpec(11, 1.0, 0.95, "actual_singular_weighted", "unit"),
    )
    failed.update(
        {
            "phase_synthesis_status": "failed",
            "rescue_status": "attempt_failed",
            "failure_mode": "numerical_failure",
        }
    )
    rescued = _empty_attempt_row(
        target,
        AttemptSpec(15, 1.0, 0.95, "actual_singular_weighted", "unit"),
    )
    rescued.update(
        {
            "phase_synthesis_status": "completed",
            "qsvt_circuit_status": "completed",
            "simulation_status": "completed",
            "circuit_vs_ridge_relative_update_error": 1.0e-4,
            "absolute_update_error": 1.0e-8,
            "residual_gap": 1.0e-5,
            "success_probability": 0.9,
            "phase_count": 16,
            "rescue_status": "rescued",
        }
    )
    summary = summarize_rescue_attempts(pd.DataFrame([failed, rescued], columns=ATTEMPT_COLUMNS))
    assert bool(summary.iloc[0]["rescued"])
    assert int(summary.iloc[0]["best_degree"]) == 15
