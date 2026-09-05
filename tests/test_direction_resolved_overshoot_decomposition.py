from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.direction_resolved_overshoot_decomposition import (
    FAILURE_MECHANISMS,
    PER_DIRECTION_COLUMNS,
    SUMMARY_COLUMNS,
    TRANSITION_COLUMNS,
    _summary_frame,
    _transition_frame,
    classify_failure_mechanism,
    evaluate_direction_resolved,
)
from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(11)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    # Two near-degenerate leading directions near the boundary plus two small directions.
    singular_values = np.array([6.0, 5.85, 1.4, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.5, -0.2, 0.3, 0.4])
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata={})


def test_decomposition_produces_per_direction_rows() -> None:
    rows = evaluate_direction_resolved(
        subproblem=_subproblem(),
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
        selection_mode="high_leverage",
        alphas=[1.0e-3],
        degrees=[45, 47],
        target_families=["weighted_support_ls"],
    )
    # One row per (degree, family, singular direction): 2 degrees x 1 family x 4 directions.
    assert len(rows) == 8
    for row in rows:
        for column in PER_DIRECTION_COLUMNS:
            assert column in row
        assert row["failure_mechanism"] in FAILURE_MECHANISMS
    # Each config carries all four singular indices.
    indices = sorted({row["singular_index"] for row in rows})
    assert indices == [0, 1, 2, 3]


def test_degree_45_vs_47_transition_is_represented() -> None:
    rows = evaluate_direction_resolved(
        subproblem=_subproblem(),
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
        selection_mode="high_leverage",
        alphas=[1.0e-3],
        degrees=[45, 47],
        target_families=["weighted_support_ls"],
    )
    import pandas as pd

    summary = _summary_frame(pd.DataFrame(rows))
    assert list(summary.columns) == SUMMARY_COLUMNS
    transition = _transition_frame(summary)
    assert list(transition.columns) == TRANSITION_COLUMNS
    # The 45 vs 47 transition row exists and names both degrees' mechanisms.
    assert len(transition) == 1
    record = transition.iloc[0]
    assert record["degree_45_failure_mechanism"] in FAILURE_MECHANISMS
    assert record["degree_47_failure_mechanism"] in FAILURE_MECHANISMS


def test_dominant_failure_direction_is_deterministic() -> None:
    kwargs = dict(
        subproblem=_subproblem(),
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
        selection_mode="high_leverage",
        alphas=[1.0e-3, 1.0e-2],
        degrees=[45, 47, 49],
        target_families=["weighted_support_ls"],
    )
    first = evaluate_direction_resolved(**kwargs)
    second = evaluate_direction_resolved(**kwargs)
    dominant_first = [bool(row["dominant_failure_direction"]) for row in first]
    dominant_second = [bool(row["dominant_failure_direction"]) for row in second]
    mechanisms_first = [row["failure_mechanism"] for row in first]
    mechanisms_second = [row["failure_mechanism"] for row in second]
    assert dominant_first == dominant_second
    assert mechanisms_first == mechanisms_second


def test_classify_failure_mechanism_labels() -> None:
    sigma = np.array([6.0, 5.85, 1.4, 1.0])
    # Feasible direction -> no failure.
    assert (
        classify_failure_mechanism(
            qsvt_safe=True,
            direction_error=0.01,
            sigma=sigma,
            signed_filter_error=np.zeros(4),
            residual_error_contribution=np.zeros(4),
            sign_flip=np.array([False, False, False, False]),
            dominant_index=0,
        )
        == "no_failure"
    )
    # Boundedness breach dominates.
    assert (
        classify_failure_mechanism(
            qsvt_safe=False,
            direction_error=0.5,
            sigma=sigma,
            signed_filter_error=np.array([0.1, 0.1, 0.0, 0.0]),
            residual_error_contribution=np.array([1.0, 2.0, 0.0, 0.0]),
            sign_flip=np.array([False, False, False, False]),
            dominant_index=1,
        )
        == "boundedness_violation"
    )
    # Leading near-boundary direction overshoots (symmetric across the leading pair) -> amplitude.
    assert (
        classify_failure_mechanism(
            qsvt_safe=True,
            direction_error=0.3,
            sigma=sigma,
            signed_filter_error=np.array([3.0e-4, 3.0e-4, 0.0, 0.0]),
            residual_error_contribution=np.array([1.2, 13.0, 0.0, 0.0]),
            sign_flip=np.array([False, False, False, False]),
            dominant_index=1,
        )
        == "leading_direction_amplitude_distortion"
    )
    # A sign flip at the leading direction -> sign error.
    assert (
        classify_failure_mechanism(
            qsvt_safe=True,
            direction_error=1.4,
            sigma=sigma,
            signed_filter_error=np.array([-5.0e-3, -5.0e-3, 0.0, 0.0]),
            residual_error_contribution=np.array([5.0, 0.5, 0.0, 0.0]),
            sign_flip=np.array([True, False, False, False]),
            dominant_index=0,
        )
        == "leading_direction_sign_error"
    )
