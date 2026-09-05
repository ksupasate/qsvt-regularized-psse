from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    QsvtReadoutState,
    _error_propagation_row,
    _run_sampling_trials,
    _shot_norm_rows,
    qsvt_target_readout,
    sample_success_probability,
)

_CONTEXT = {
    "case": "toy",
    "subproblem_id": "toy_00",
    "subproblem_type": "high_leverage",
    "alpha": 1.0e-4,
    "degree": 15,
}


def _state() -> QsvtReadoutState:
    H = np.array(
        [
            [1.2, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.3, 1.1, 0.2],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    r = np.array([0.5, -0.3, 0.4, -0.2], dtype=np.float64)
    return qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)


def test_binomial_success_probability_is_unbiased_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    estimates = [sample_success_probability(0.3, shots=20_000, rng=rng) for _ in range(50)]
    assert abs(float(np.mean(estimates)) - 0.3) < 0.01


def test_norm_confidence_interval_covers_exact_norm() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=50_000, trials=40, base_seed=11)
    rows = _shot_norm_rows(trials, state.recovered_norm, _CONTEXT)
    assert all(row["status"] == "norm_sampled" for row in rows)
    covered = sum(
        1 for row in rows if row["norm_ci_low"] <= state.recovered_norm <= row["norm_ci_high"]
    )
    # 95% intervals should cover the exact norm for the large majority of trials.
    assert covered >= int(0.8 * len(rows))


def test_missing_success_probability_metadata_flagged() -> None:
    state = _state()
    broken = QsvtReadoutState(
        ridge_update=state.ridge_update,
        readout_state=state.readout_state,
        recovered_norm=state.recovered_norm,
        beta=state.beta,
        alpha=state.alpha,
        alpha_norm=state.alpha_norm,
        scale_factor_C=state.scale_factor_C,
        residual_norm=state.residual_norm,
        success_probability=float("nan"),
        singular_values=state.singular_values,
        polynomial_update=state.polynomial_update,
        polynomial_available=state.polynomial_available,
        bounded_target_ok=state.bounded_target_ok,
    )
    trials = _run_sampling_trials(broken, shots=1_000, trials=3, base_seed=1)
    rows = _shot_norm_rows(trials, broken.recovered_norm, _CONTEXT)
    assert all(row["status"] == "missing_success_probability_metadata" for row in rows)


def test_error_propagation_separates_exact_and_sampled_norm() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=100_000, trials=20, base_seed=5)
    row = _error_propagation_row(trials, 100_000, _CONTEXT)
    assert row["status"] == "propagated"
    assert np.isfinite(row["mean_vector_relative_error_with_exact_norm"])
    assert np.isfinite(row["mean_vector_relative_error_with_sampled_norm"])
    # Adding sampled-norm noise cannot reduce the error below the exact-norm reconstruction.
    assert (
        row["mean_vector_relative_error_with_sampled_norm"]
        >= row["mean_vector_relative_error_with_exact_norm"] - 1.0e-9
    )
