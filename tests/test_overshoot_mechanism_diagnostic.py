from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.overshoot_mechanism_diagnostic import (
    OVERSHOOT_FAILURE_MODES,
    SUMMARY_COLUMNS,
    classify_overshoot_failure_mode,
    detect_off_support_peak,
    evaluate_overshoot_mechanism,
)


def _subproblem() -> SelectedSubproblem:
    rng = np.random.default_rng(11)
    u, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    v, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    singular_values = np.array([6.0, 5.5, 1.4, 1.0])
    H = u @ np.diag(singular_values) @ v.T
    r = H @ np.array([0.5, -0.2, 0.3, 0.4])
    return SelectedSubproblem(H_tilde=H, r_tilde=r, metadata={})


def test_detect_off_support_peak_flags_dominant_far_peak() -> None:
    # A response that peaks far from the support (at |x| = 0.15) and dominates the on-support
    # values at the support singular values {0.9, 1.0}.
    support = np.array([0.9, 1.0])
    bump = lambda x: 1.0 / (1.0 + 400.0 * (np.abs(x) - 0.15) ** 2)  # noqa: E731
    result = detect_off_support_peak(bump, support)
    assert result["detected"] is True
    assert result["distance"] > 0.02
    assert result["margin"] > 2.0


def test_detect_off_support_peak_not_flagged_on_support() -> None:
    # A polynomial whose maximum sits at a support singular value is not an off-support peak.
    support = np.array([0.3, 1.0])
    on_support = Polynomial([0.0, 1.0])  # |p| maximal at x = +/-1, a support value
    result = detect_off_support_peak(on_support, support)
    assert result["detected"] is False


def test_classify_failure_mode_distinguishes_support_fit_from_off_support() -> None:
    # Feasible, clean target.
    assert (
        classify_overshoot_failure_mode(
            qsvt_safe=True,
            deployable=True,
            success_probability=0.5,
            direction_error=0.001,
            residual_ratio_no_update=0.01,
            off_support_peak_detected=False,
            high_weight_singular_error=0.01,
        )
        == "no_overshoot_residual_feasible"
    )
    # Direction broke with no dominant off-support peak -> singular-support fit error.
    assert (
        classify_overshoot_failure_mode(
            qsvt_safe=True,
            deployable=True,
            success_probability=0.1,
            direction_error=0.5,
            residual_ratio_no_update=0.7,
            off_support_peak_detected=False,
            high_weight_singular_error=0.45,
        )
        == "singular_support_error_breaks_direction"
    )
    # Direction broke with a dominant off-support peak -> off-support peak breaks direction.
    assert (
        classify_overshoot_failure_mode(
            qsvt_safe=True,
            deployable=True,
            success_probability=0.1,
            direction_error=1.5,
            residual_ratio_no_update=10.0,
            off_support_peak_detected=True,
            high_weight_singular_error=5.0,
        )
        == "off_support_peak_breaks_direction"
    )
    # A bounded polynomial that exceeds the QSVT bound -> boundedness violation.
    assert (
        classify_overshoot_failure_mode(
            qsvt_safe=False,
            deployable=True,
            success_probability=0.1,
            direction_error=0.5,
            residual_ratio_no_update=0.7,
            off_support_peak_detected=False,
            high_weight_singular_error=0.45,
        )
        == "boundedness_violation"
    )
    # Phase synthesis failure dominates.
    assert (
        classify_overshoot_failure_mode(
            qsvt_safe=True,
            deployable=True,
            success_probability=0.5,
            direction_error=0.001,
            residual_ratio_no_update=0.01,
            off_support_peak_detected=False,
            high_weight_singular_error=0.01,
            phase_synthesis_failed=True,
        )
        == "phase_synthesis_limited"
    )


def test_degree_47_classified_from_actual_behaviour_and_deterministic() -> None:
    kwargs = dict(
        subproblem=_subproblem(),
        alphas=[1.0e-2],
        degrees=[15, 45, 47, 51],
        target_families=["weighted_support_ls"],
        case="toy",
        model="weighted",
        subproblem_id="toy_4x4",
    )
    first = evaluate_overshoot_mechanism(**kwargs)
    second = evaluate_overshoot_mechanism(**kwargs)

    modes_first = [row["overshoot_failure_mode"] for row in first]
    modes_second = [row["overshoot_failure_mode"] for row in second]
    assert modes_first == modes_second
    for row in first:
        assert row["overshoot_failure_mode"] in OVERSHOOT_FAILURE_MODES
        assert row["phase_synthesis_status_if_checked"] == "not_checked"
        for column in SUMMARY_COLUMNS:
            assert column in row
