from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    QsvtReadoutState,
    _norm_scaling_row,
    _scaling_audit_row,
    qsvt_target_readout,
    ridge_update,
)

_CONTEXT = {
    "case": "toy",
    "subproblem_id": "toy_00",
    "subproblem_type": "high_leverage",
    "alpha": 1.0e-4,
    "degree": 15,
}


def _toy_state() -> QsvtReadoutState:
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


def test_scaling_chain_recovers_norm_with_matched_alpha() -> None:
    state = _toy_state()
    norm_row = _norm_scaling_row(state, "toy", _CONTEXT, "note")
    audit_row = _scaling_audit_row(state, "toy", _CONTEXT, "note")
    assert norm_row["status"] == "audited"
    assert norm_row["norm_relative_error"] < 1.0e-6
    assert audit_row["ridge_comparison_same_alpha"] is True
    assert audit_row["ridge_comparison_same_scaling"] is True
    assert audit_row["statevector_rescaling_ok"] is True
    assert audit_row["singular_value_normalization_ok"] is True


def test_scaling_audit_fails_when_constant_C_missing() -> None:
    state = _toy_state()
    broken = QsvtReadoutState(
        ridge_update=state.ridge_update,
        readout_state=state.readout_state,
        recovered_norm=state.recovered_norm,
        beta=state.beta,
        alpha=state.alpha,
        alpha_norm=state.alpha_norm,
        scale_factor_C=float("nan"),
        residual_norm=state.residual_norm,
        success_probability=float("nan"),
        singular_values=state.singular_values,
        polynomial_update=np.full_like(state.ridge_update, np.nan),
        polynomial_available=False,
        bounded_target_ok=False,
    )
    norm_row = _norm_scaling_row(broken, "toy", _CONTEXT, "note")
    audit_row = _scaling_audit_row(broken, "toy", _CONTEXT, "note")
    assert norm_row["status"] == "missing_scaling_metadata"
    assert audit_row["status"] == "missing_scaling_metadata"
    assert audit_row["statevector_rescaling_ok"] is False


def test_ridge_comparison_uses_matched_alpha() -> None:
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
    # A mismatched-alpha Ridge reference is materially different, confirming the
    # matched-alpha comparison is the meaningful one.
    state = qsvt_target_readout(H, r, alpha=1.0e-4, degree=15)
    matched = ridge_update(H, r, alpha=1.0e-4)
    mismatched = ridge_update(H, r, alpha=1.0e-1)
    reconstruction = state.readout_state * state.recovered_norm
    assert np.allclose(reconstruction, matched, atol=1.0e-10)
    assert not np.allclose(reconstruction, mismatched, atol=1.0e-6)
