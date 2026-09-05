from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.tqe_observable_first_readout import (
    RESULT_COLUMNS,
    build_observable_definitions,
    normalized_update_probabilities,
    run_observable_first_readout,
    single_coordinate_energy_estimate,
    subset_energy_estimate,
    theoretical_observable_std,
)


def test_update_probabilities_match_squared_amplitudes() -> None:
    probabilities = normalized_update_probabilities(np.array([3.0, -4.0]))

    np.testing.assert_allclose(probabilities, np.array([9.0 / 25.0, 16.0 / 25.0]))
    assert np.isclose(float(np.sum(probabilities)), 1.0)


def test_single_coordinate_energy_estimator_uses_norm_squared_probability() -> None:
    estimate = single_coordinate_energy_estimate(
        {0: 30, 1: 70},
        index=0,
        update_norm=5.0,
        shots=100,
    )

    assert estimate == 7.5


def test_subset_energy_estimator_uses_selected_probability_mass() -> None:
    estimate = subset_energy_estimate(
        {0: 20, 1: 50, 2: 30},
        indices=(0, 2),
        update_norm=4.0,
        shots=100,
    )

    assert estimate == 8.0


def test_signed_coordinate_difference_is_labeled_as_sign_access_required() -> None:
    observables = build_observable_definitions(
        np.array([1.0, -2.0, 0.5]),
        metadata={},
    )
    signed = [
        item for item in observables if item.observable_type == "signed_coordinate_difference"
    ]

    assert len(signed) == 1
    assert signed[0].signed_observable is True
    assert signed[0].sign_access_required is True
    assert "not_computational_basis_shot_accessible" in signed[0].shot_access_model


def test_binomial_style_uncertainty_is_finite_and_shrinks_with_shots() -> None:
    coarse = theoretical_observable_std(
        probability=0.25,
        shots=100,
        update_norm=2.0,
        estimator="subset_energy",
    )
    fine = theoretical_observable_std(
        probability=0.25,
        shots=10000,
        update_norm=2.0,
        estimator="subset_energy",
    )

    assert np.isfinite(coarse)
    assert np.isfinite(fine)
    assert fine < coarse


def test_observable_first_readout_output_schema_contains_required_columns(tmp_path: Path) -> None:
    run = _run_tiny_readout(tmp_path)
    frame = pd.read_csv(run["artifacts"]["results_csv"])

    assert set(RESULT_COLUMNS).issubset(frame.columns)
    assert len(frame) == 20
    assert run["artifacts"]["counts_json"].is_file()
    assert run["artifacts"]["summary_table_csv"].is_file()
    assert run["artifacts"]["error_vs_shots_figure"].is_file()
    assert run["artifacts"]["ci_width_figure"].is_file()
    assert run["artifacts"]["ridge_vs_qsvt_figure"].is_file()


def test_signed_readout_skip_is_recorded_without_failing_experiment(tmp_path: Path) -> None:
    run = _run_tiny_readout(tmp_path)
    frame = pd.read_csv(run["artifacts"]["results_csv"])
    skipped = frame[frame["simulation_status"] == "skipped_sign_access_required"]

    assert not skipped.empty
    assert set(skipped["observable_name"]) == {"branch_coordinate_difference_signed_proxy"}
    assert skipped["failure_or_skip_reason"].str.contains("phase/sign-aware readout").all()


def _run_tiny_readout(tmp_path: Path) -> dict[str, object]:
    return run_observable_first_readout(
        {
            "output_root": str(tmp_path),
            "case_name": "synthetic",
            "subproblem_size": 4,
            "alpha": 1.0e-2,
            "degree": 3,
            "ridge_update": [0.4, -0.2, 0.1, 0.05],
            "qsvt_update": [0.39, -0.21, 0.11, 0.04],
            "success_probability": 0.75,
            "metadata": {"selected_state_indices": [10, 11, 12, 13]},
            "shots_grid": [100, 1000],
            "seed_grid": [0, 1],
        }
    )
