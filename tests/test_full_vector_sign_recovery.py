from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import recover_relative_signs


def _state(values: list[float]) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    return state / np.linalg.norm(state)


def test_sign_recovery_all_positive() -> None:
    state = _state([0.5, 0.5, 0.5, 0.5])
    rng = np.random.default_rng(0)
    out = recover_relative_signs(state, shots=100_000, rng=rng)
    assert np.array_equal(out["estimated_signs"], np.ones(4))
    assert all(status == "reliable" for status in out["reliability_status"])


def test_sign_recovery_mixed_sign() -> None:
    state = _state([0.6, -0.5, 0.4, -0.45])
    rng = np.random.default_rng(1)
    out = recover_relative_signs(state, shots=200_000, rng=rng)
    reliable = np.array([s == "reliable" for s in out["reliability_status"]])
    true_signs = np.sign(state)
    assert np.array_equal(out["estimated_signs"][reliable], true_signs[reliable])
    assert reliable.all()


def test_near_zero_coordinate_marked_unreliable() -> None:
    state = _state([0.7, 1.0e-5, 0.5, 0.5])
    rng = np.random.default_rng(2)
    out = recover_relative_signs(state, shots=100_000, rng=rng)
    assert out["reliability_status"][1] == "sign_unreliable_small_magnitude"


def test_unstable_reference_is_flagged() -> None:
    state = _state([0.7, 0.5, 0.5, 1.0e-4])
    rng = np.random.default_rng(3)
    out = recover_relative_signs(state, shots=100_000, rng=rng, reference_index=3)
    assert out["reference_unstable"] is True
    assert all(status == "reference_unstable" for status in out["reliability_status"])


def test_default_reference_is_largest_coordinate() -> None:
    state = _state([0.3, 0.9, 0.4, 0.2])
    rng = np.random.default_rng(4)
    out = recover_relative_signs(state, shots=50_000, rng=rng)
    assert out["reference_index"] == 1
