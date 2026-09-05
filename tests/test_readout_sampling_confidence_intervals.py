from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    _confidence_interval_row,
    _run_sampling_trials,
    _sampling_trial_rows,
    qsvt_target_readout,
)

_CONTEXT = {
    "case": "toy",
    "subproblem_id": "toy_00",
    "subproblem_type": "high_leverage",
    "alpha": 1.0e-4,
    "degree": 15,
}


def _state():
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


def test_trial_rows_record_seed_and_trial_id() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=2_000, trials=5, base_seed=3)
    rows = _sampling_trial_rows(trials, _CONTEXT)
    assert len(rows) == 5
    assert sorted(row["trial_id"] for row in rows) == [0, 1, 2, 3, 4]
    assert len({row["rng_seed"] for row in rows}) == 5


def test_confidence_interval_built_from_trials() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=10_000, trials=30, base_seed=9)
    row = _confidence_interval_row(trials, 10_000, _CONTEXT)
    assert row["trials"] == 30
    assert (
        row["p05_vector_relative_l2_error"]
        <= row["p50_vector_relative_l2_error"]
        <= row["p95_vector_relative_l2_error"]
    )
    assert row["std_vector_relative_l2_error"] >= 0.0


def test_error_decreases_with_shots() -> None:
    state = _state()
    means = []
    for shots in (1_000, 10_000, 100_000):
        trials = _run_sampling_trials(state, shots=shots, trials=20, base_seed=42)
        means.append(
            _confidence_interval_row(trials, shots, _CONTEXT)["mean_vector_relative_l2_error"]
        )
    assert means[0] > means[1] > means[2]


def test_same_seed_is_deterministic() -> None:
    state = _state()
    a = _run_sampling_trials(state, shots=5_000, trials=4, base_seed=7)
    b = _run_sampling_trials(state, shots=5_000, trials=4, base_seed=7)
    assert [r["rng_seed"] for r in a] == [r["rng_seed"] for r in b]
    assert [r["vector_relative_l2_error"] for r in a] == [r["vector_relative_l2_error"] for r in b]
