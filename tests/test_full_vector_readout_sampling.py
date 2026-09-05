from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    _small_probability_flag,
    sample_basis_magnitudes,
)


def _state(values: list[float]) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    return state / np.linalg.norm(state)


def test_sampling_estimates_probabilities_on_synthetic_state() -> None:
    state = _state([0.6, 0.4, 0.5, 0.48])
    rng = np.random.default_rng(0)
    out = sample_basis_magnitudes(state, shots=200_000, rng=rng)
    assert np.allclose(out["estimated_probability"], out["true_probability"], atol=5.0e-3)
    assert np.allclose(out["estimated_magnitude"], np.abs(state), atol=5.0e-3)
    assert int(out["counts"].sum()) == 200_000


def test_sampling_error_decreases_with_shots() -> None:
    state = _state([0.7, -0.4, 0.5, -0.3])
    shot_levels = (1_000, 10_000, 100_000)
    mean_errors: list[float] = []
    for shots in shot_levels:
        # Average over several seeds to assess the trend rather than one noisy draw.
        per_seed = []
        for seed in range(8):
            rng = np.random.default_rng(1000 + seed)
            out = sample_basis_magnitudes(state, shots=shots, rng=rng)
            per_seed.append(
                float(np.mean(np.abs(out["estimated_probability"] - out["true_probability"])))
            )
        mean_errors.append(float(np.mean(per_seed)))
    assert mean_errors[0] > mean_errors[1] > mean_errors[2]


def test_small_probability_flag_marks_tiny_coordinates() -> None:
    state = _state([0.9, 0.435, 0.02, 0.01])
    true_probability = np.abs(state) ** 2
    flags = [_small_probability_flag(float(p), 1_000) for p in true_probability]
    # The last coordinate (~1e-4 probability) is below the 5/shots detectability floor.
    assert flags[-1] is True
    assert flags[0] is False
