from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.amplitude_estimation_routines import (
    DEPLOYABILITY_CLASSES,
    SUMMARY_COLUMNS,
    bernoulli_amplitude_estimate,
    build_qsvt_amplitude_problem,
    evaluate_amplitude_estimation_point,
    exact_success_probability,
    iterative_amplitude_estimate,
    observable_amplitude_estimate,
    wilson_interval,
    write_amplitude_estimation_outputs,
)


def _ry(theta: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128
    )


def test_bernoulli_estimator_is_unbiased_within_tolerance() -> None:
    p_true = 0.37
    estimates = [bernoulli_amplitude_estimate(p_true, 20000, seed).estimate for seed in range(40)]
    assert abs(float(np.mean(estimates)) - p_true) < 0.01


def test_wilson_interval_contains_true_probability() -> None:
    covered = 0
    trials = 200
    p_true = 0.3
    for seed in range(trials):
        result = bernoulli_amplitude_estimate(p_true, 500, seed)
        if result.confidence_interval_low <= p_true <= result.confidence_interval_high:
            covered += 1
    # Wilson 95% interval should cover well above a conservative 0.9 in practice.
    assert covered / trials >= 0.9
    low, high, method = wilson_interval(150, 500)
    assert 0.0 <= low < high <= 1.0
    assert method in {"wilson_scipy", "wilson_normal_approx"}


def test_exact_success_probability_matches_direct_computation() -> None:
    state = np.array([0.6, 0.0, 0.0, 0.8j, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    state = state / np.linalg.norm(state)
    direct = float(np.sum(np.abs(state[:4]) ** 2))
    assert np.isclose(exact_success_probability(state, encoded_dimension=4), direct)


def test_iterative_mlae_recovers_known_amplitude_on_testbed() -> None:
    theta = 0.41
    unitary = _ry(theta)
    # Mark |1> as the success state: place it first so encoded_dimension=1 selects it.
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    unitary_good_first = swap @ unitary
    result = iterative_amplitude_estimate(
        unitary_good_first,
        encoded_dimension=1,
        powers=(0, 1, 2, 4),
        shots=4000,
        seed=7,
    )
    assert result.status == "succeeded"
    assert abs(result.estimate - np.sin(theta) ** 2) < 0.03
    assert result.confidence_interval_low <= result.confidence_interval_high


def test_build_problem_and_iterative_estimate_agree_with_exact() -> None:
    H = np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64)
    r = np.array([0.4, -0.2], dtype=np.float64)
    problem = build_qsvt_amplitude_problem(H, r, alpha=1.0e-2, degree=5)
    assert problem.encoded_dimension == 2
    reconstructed = float(np.sum(np.abs(problem.unitary_A[:, 0][:2]) ** 2))
    assert np.isclose(reconstructed, problem.exact_success_probability)
    result = iterative_amplitude_estimate(
        problem.unitary_A,
        problem.encoded_dimension,
        powers=(0, 1, 2),
        shots=8000,
        seed=11,
        exact_probability=problem.exact_success_probability,
    )
    assert result.status == "succeeded"
    assert abs(result.estimate - problem.exact_success_probability) < 0.05


def test_observable_estimator_reports_magnitude_only() -> None:
    state = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.complex128)
    estimate = observable_amplitude_estimate(
        state, encoded_dimension=2, component=0, shots=5000, seed=3
    )
    assert estimate["estimates_sign"] is False
    assert abs(estimate["estimate"] - 0.36) < 0.05


def test_point_rows_use_valid_labels_and_columns(tmp_path: Path) -> None:
    H = np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64)
    r = np.array([0.4, -0.2], dtype=np.float64)
    rows = evaluate_amplitude_estimation_point(
        H=H, r=r, alpha=1.0e-2, degree=5, shot_levels=[256], grover_powers=(0, 1, 2)
    )
    estimators = {row["estimator_type"] for row in rows}
    assert {"statevector_exact", "bernoulli_shots", "iterative_mlae"}.issubset(estimators)
    for row in rows:
        assert row["deployability_class"] in DEPLOYABILITY_CLASSES
        assert row["claim_disallowed"].startswith("QSVT beats Ridge")

    artifacts = write_amplitude_estimation_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)
    summary = pd.read_csv(artifacts["amplitude_estimation_summary"])
    for column in SUMMARY_COLUMNS:
        assert column in summary.columns
    assert artifacts["iterative_amplitude_estimation_attempts"].is_file()
