from __future__ import annotations

import numpy as np

from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.readout_diagnostics import (
    basis_sampling_energy_estimate,
    observable_shot_sweep,
    shots_for_relative_error,
)
from robust_qsvt_se.qsvt.selected_observables import (
    energy_observable,
    voltage_magnitude_observable,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata
from robust_qsvt_se.utils.seed import make_rng


def _ieee14():
    system, _ = build_engineering_system(
        {"case_name": "ieee14", "case_source": "pypower", "matrix_source": "weighted_jacobian"}
    )
    metadata = build_state_metadata_from_system_metadata(system.metadata)
    update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=1e-4)
    return system, metadata, update


def test_sign_aware_error_decreases_with_shots_deterministic() -> None:
    system, metadata, update = _ieee14()
    bus = system.metadata["voltage_state_buses"][0]
    obs = voltage_magnitude_observable(metadata, bus)
    rows = observable_shot_sweep(
        obs, update, shots_grid=(100, 1_000, 10_000), trials=64, base_seed=11
    )
    mean_errors = [row["mean_abs_error"] for row in rows]
    # Deterministic seeds: mean absolute error must shrink monotonically with shots.
    assert mean_errors[0] > mean_errors[1] > mean_errors[2]
    # Repeating with the same seed reproduces the exact errors.
    rows_repeat = observable_shot_sweep(
        obs, update, shots_grid=(100, 1_000, 10_000), trials=64, base_seed=11
    )
    assert [r["mean_abs_error"] for r in rows_repeat] == mean_errors


def test_sign_aware_error_tracks_one_over_sqrt_shots() -> None:
    system, metadata, update = _ieee14()
    bus = system.metadata["voltage_state_buses"][0]
    obs = voltage_magnitude_observable(metadata, bus)
    rows = observable_shot_sweep(obs, update, shots_grid=(1_000, 100_000), trials=128, base_seed=3)
    ratio = rows[0]["mean_abs_error"] / rows[1]["mean_abs_error"]
    # 100x more shots -> ~10x lower error; allow a generous statistical band.
    assert 4.0 < ratio < 25.0


def test_basis_sampling_energy_is_unbiased() -> None:
    _system, metadata, update = _ieee14()
    obs = energy_observable(update.size, list(metadata.voltage_indices))
    exact = obs.exact_value(update)
    true_value, estimated = basis_sampling_energy_estimate(
        np.asarray(obs.support), update, shots=500_000, rng=make_rng(7)
    )
    assert true_value == exact
    dx_norm_sq = float(update @ update)
    # Binomial sampling band on the in-subspace probability estimate.
    assert abs(estimated - exact) <= 6.0 * dx_norm_sq / np.sqrt(500_000)


def test_shots_for_relative_error_selects_increasing_budget() -> None:
    system, metadata, update = _ieee14()
    bus = system.metadata["voltage_state_buses"][0]
    obs = voltage_magnitude_observable(metadata, bus)
    sweep = observable_shot_sweep(
        obs, update, shots_grid=(100, 1_000, 10_000, 100_000), trials=64, base_seed=5
    )
    loose = shots_for_relative_error(sweep, target_relative_error=0.2)
    tight = shots_for_relative_error(sweep, target_relative_error=0.02)
    assert loose is not None and tight is not None
    assert loose <= tight
