from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    EXACT_TARGET_METHOD,
    GATE_LEVEL_METHOD,
    _gate_level_rows,
    _gate_level_summary_row,
    _gate_vs_target_row,
    gate_level_subproblem_readout,
    qsvt_target_readout,
)

_CONTEXT = {
    "case": "toy",
    "subproblem_id": "toy_00",
    "subproblem_type": "high_leverage",
    "alpha": 1.0e-4,
    "degree": 15,
}


def _toy() -> tuple[np.ndarray, np.ndarray]:
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
    return H, r


def _synthetic_gate(state, *, perturbation: float) -> dict:
    rng = np.random.default_rng(0)
    gate_update = state.ridge_update + perturbation * rng.standard_normal(state.ridge_update.size)
    norm = float(np.linalg.norm(gate_update))
    return {
        "available": True,
        "gate_update": gate_update,
        "gate_state": gate_update / norm,
        "statevector": np.zeros(2 * state.ridge_update.size),
        "ridge_update": state.ridge_update,
        "success_probability": 0.5,
        "synthesized_degree": 15,
        "state_error_vs_ridge": 0.05,
    }


def test_gate_level_method_is_distinct_from_exact_target() -> None:
    assert GATE_LEVEL_METHOD != EXACT_TARGET_METHOD
    H, r = _toy()
    state = qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)
    gate = _synthetic_gate(state, perturbation=0.0)
    rows = _gate_level_rows(gate, state, ["a", "b", "c", "d"], "gate_src", "target_src", _CONTEXT)
    assert rows and all(row["method"] == GATE_LEVEL_METHOD for row in rows)
    # The per-coordinate rows expose a gate amplitude column the exact-target rows do not.
    assert "gate_level_amplitude" in rows[0]


def test_gate_rows_require_a_gate_update_not_exact_target_alone() -> None:
    H, r = _toy()
    state = qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)
    # _gate_level_rows reads gate["gate_update"]; without an actual gate result it cannot run.
    import pytest

    with pytest.raises(KeyError):
        _gate_level_rows({"available": True}, state, ["a"], "g", "t", _CONTEXT)


def test_gate_vs_exact_target_consistent_with_gate_vs_ridge() -> None:
    H, r = _toy()
    state = qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)
    gate = _synthetic_gate(state, perturbation=0.01)
    row = _gate_vs_target_row(gate, state, _CONTEXT)
    # Exact target reproduces Ridge to machine precision, so gate-vs-target tracks gate-vs-ridge.
    assert row["exact_target_vs_ridge_l2"] < 1.0e-9
    assert abs(row["gate_vs_ridge_l2"] - row["gate_vs_exact_target_l2"]) < 1.0e-6


def test_missing_gate_level_statevector_recorded_not_substituted() -> None:
    H, r = _toy()
    # An even degree makes the gate solver raise immediately; the readout must mark it missing
    # rather than substituting the exact target.
    result = gate_level_subproblem_readout(H, r, alpha=1.0e-4, degree=14, seed=0)
    assert result["available"] is False
    assert result.get("reason")


def test_gate_summary_reports_availability_and_errors() -> None:
    H, r = _toy()
    state = qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)
    gate = _synthetic_gate(state, perturbation=0.02)
    summary = _gate_level_summary_row(gate, state, H, r, _CONTEXT)
    assert summary["gate_available"] is True
    assert summary["status"] == "gate_reconstructed"
    assert np.isfinite(summary["vector_relative_l2_error_vs_ridge"])
    assert 0.0 <= summary["sign_accuracy"] <= 1.0
