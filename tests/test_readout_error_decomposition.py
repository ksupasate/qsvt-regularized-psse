from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    _error_decomposition_row,
    _run_sampling_trials,
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


def _gate(state):
    return {
        "available": True,
        "gate_update": state.polynomial_update.copy(),  # gate faithfully realizes the polynomial
        "gate_state": state.readout_state,
        "statevector": np.zeros(2 * state.ridge_update.size),
        "ridge_update": state.ridge_update,
        "success_probability": 0.5,
        "synthesized_degree": 15,
        "state_error_vs_ridge": 0.05,
    }


def test_complete_decomposition_picks_dominant_source() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=100_000, trials=10, base_seed=2)
    row = _error_decomposition_row(state, _gate(state), trials, _CONTEXT)
    assert row["status"] == "decomposed"
    assert row["dominant_error_source"] in {
        "ridge_vs_exact_target",
        "exact_target_vs_polynomial",
        "polynomial_vs_gate",
        "statevector_vs_shot_reconstruction",
    }
    # Ridge vs exact target is machine precision; the polynomial/shot stages dominate.
    assert row["ridge_vs_exact_target_error"] < 1.0e-9


def test_missing_gate_is_marked_not_fabricated() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=1_000, trials=3, base_seed=2)
    row = _error_decomposition_row(state, None, trials, _CONTEXT)
    assert "missing_gate_artifact" in row["status"]
    assert np.isnan(row["polynomial_vs_gate_error"])
    assert row["dominant_error_source"] == ""


def test_readout_error_separate_from_polynomial_error() -> None:
    state = _state()
    trials = _run_sampling_trials(state, shots=100_000, trials=10, base_seed=2)
    row = _error_decomposition_row(state, _gate(state), trials, _CONTEXT)
    # The finite-shot reconstruction error is reported in its own column, not merged with
    # the QSVT polynomial-approximation error.
    assert (
        row["statevector_vs_shot_reconstruction_error"] != row["exact_target_vs_polynomial_error"]
    )
    assert np.isfinite(row["shot_reconstruction_vs_ridge_error"])
