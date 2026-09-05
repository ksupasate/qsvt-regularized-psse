from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.qsvt.observable_measurement_circuits import (
    default_update_observables,
    exact_postselected_observable_values,
    observable_values_from_state,
    postselected_encoded_state,
    sample_statevector_counts,
    shot_observable_summary,
)


def test_postselected_encoded_state_and_exact_observables() -> None:
    state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    post, success = postselected_encoded_state(state, encoded_dimension=2)
    observables = default_update_observables(2)
    values = exact_postselected_observable_values(
        state,
        encoded_dimension=2,
        observables=observables,
    )

    assert success == pytest.approx(0.5)
    np.testing.assert_allclose(post, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    assert values["component_0_probability"] == pytest.approx(0.5)
    assert values["first_two_state_energy"] == pytest.approx(1.0)


def test_statevector_sampling_and_shot_summary_are_reproducible() -> None:
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    counts_a = sample_statevector_counts(state, shots=100, rng=rng_a)
    counts_b = sample_statevector_counts(state, shots=100, rng=rng_b)
    observables = default_update_observables(2)
    exact = {"success_probability": 1.0, "component_0_probability": 1.0}
    exact["component_1_probability"] = 0.0
    exact["first_two_state_energy"] = 1.0
    ridge = observable_values_from_state(np.array([1.0, 0.0]), observables)
    rows = shot_observable_summary(
        counts_a,
        encoded_dimension=2,
        observables=observables,
        exact_qsvt_values=exact,
        ridge_values=ridge,
        shots=100,
        seed=123,
    )

    assert counts_a == counts_b
    assert rows[0]["shot_estimate"] == pytest.approx(1.0)
    assert rows[1]["absolute_error_vs_ridge"] == pytest.approx(0.0)
