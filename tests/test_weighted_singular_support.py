from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.weighted_singular_support import (
    compute_singular_support,
    evaluate_weighted_singular_support,
)


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    singular_values = np.array([4.0, 3.0, 2.0, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.3, -0.2, 0.5, 0.1]) + 0.01 * rng.standard_normal(4)
    metadata = {
        "state_index_mapping": [
            {"local_index": 0, "state_type": "angle", "bus_id": 4, "label": "theta_4"},
            {"local_index": 1, "state_type": "angle", "bus_id": 5, "label": "theta_5"},
        ]
    }
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata=metadata)


def test_singular_support_weights_are_nonnegative_and_normalized() -> None:
    subproblem = _subproblem()
    support = compute_singular_support(
        np.asarray(subproblem.H_tilde), np.asarray(subproblem.r_tilde), alpha=1.0e-3
    )

    for weights in (
        support.residual_projection_weight,
        support.update_magnitude_weight,
        support.residual_sensitivity_weight,
        support.combined_weight,
    ):
        assert np.all(weights >= 0.0)

    assert abs(float(np.sum(support.combined_weight)) - 1.0) <= 1.0e-9


def test_high_weight_direction_identification_is_deterministic() -> None:
    subproblem = _subproblem()
    H = np.asarray(subproblem.H_tilde)
    r = np.asarray(subproblem.r_tilde)
    first = compute_singular_support(H, r, alpha=1.0e-3).high_weight_direction
    second = compute_singular_support(H, r, alpha=1.0e-3).high_weight_direction

    assert np.array_equal(first, second)
    assert bool(first.any()) is True


def test_evaluate_emits_one_row_per_alpha_and_singular_index() -> None:
    subproblem = _subproblem()
    summary_rows, error_rows = evaluate_weighted_singular_support(
        subproblem=subproblem,
        alphas=[1.0e-4, 1.0e-2],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
        current_degree=15,
    )

    assert len(summary_rows) == 2 * 4
    assert len(error_rows) == 2 * 4
    cumulative = sum(1 for row in summary_rows if bool(row["high_weight_direction"]))
    assert cumulative >= 2
