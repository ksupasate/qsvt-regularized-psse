from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_common import forbidden_in
from robust_qsvt_se.paper.selected_observable_workload import (
    MAP_COLUMNS,
    OBSERVABLE_COLUMNS,
    SWEEP_COLUMNS,
    run_selected_observable_workload,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution
from robust_qsvt_se.qsvt.selected_observables import (
    BASIS_SAMPLING_MODEL,
    SIGN_AWARE_MODEL,
    branch_angle_difference_observable,
    build_selected_observables,
    energy_observable,
    voltage_angle_observable,
    voltage_magnitude_observable,
)
from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata


def _ieee14():
    system, _ = build_engineering_system(
        {"case_name": "ieee14", "case_source": "pypower", "matrix_source": "weighted_jacobian"}
    )
    metadata = build_state_metadata_from_system_metadata(system.metadata)
    update = ridge_svd_solution(system.H_tilde, system.r_tilde, alpha=1e-4)
    return system, metadata, update


def test_exact_value_equals_direct_dot_product() -> None:
    system, metadata, update = _ieee14()
    voltage_bus = system.metadata["voltage_state_buses"][0]
    angle_bus = system.metadata["angle_state_buses"][0]

    v_obs = voltage_magnitude_observable(metadata, voltage_bus)
    a_obs = voltage_angle_observable(metadata, angle_bus)
    assert v_obs.exact_value(update) == pytest.approx(float(v_obs.vector @ update))
    assert a_obs.exact_value(update) == pytest.approx(float(a_obs.vector @ update))

    # Branch angle-difference equals difference of the two angle components.
    buses = system.metadata["angle_state_buses"]
    b_obs = branch_angle_difference_observable(metadata, buses[0], buses[1])
    i = metadata.index_for_angle_bus(buses[0])
    j = metadata.index_for_angle_bus(buses[1])
    assert b_obs.exact_value(update) == pytest.approx(float(update[i] - update[j]))


def test_energy_observable_is_squared_amplitude() -> None:
    _system, _metadata, update = _ieee14()
    support = [0, 2, 5]
    obs = energy_observable(update.size, support)
    assert obs.readout_model == BASIS_SAMPLING_MODEL
    assert obs.basis_sampling_accessible is True
    assert obs.sign_aware_required is False
    assert obs.exact_value(update) == pytest.approx(float(np.sum(update[support] ** 2)))


def test_signed_observables_are_sign_aware_not_full_vector() -> None:
    _system, metadata, update = _ieee14()
    observables = build_selected_observables(metadata, update)
    assert observables, "expected at least one observable"
    for obs in observables:
        assert obs.full_vector_required is False
        if obs.readout_model == SIGN_AWARE_MODEL:
            assert obs.sign_aware_required is True
            assert obs.basis_sampling_accessible is False


def test_fallback_is_labeled_when_metadata_missing() -> None:
    from robust_qsvt_se.qsvt.state_metadata import build_state_metadata_from_system_metadata

    metadata = build_state_metadata_from_system_metadata({"n_states": 6})
    update = np.array([0.3, -0.1, 0.5, -0.4, 0.05, 0.2])
    observables = build_selected_observables(metadata, update)
    assert observables
    types = {obs.observable_type for obs in observables}
    assert any("coordinate" in t or "block" in t or "energy" in t for t in types)
    for obs in observables:
        note = obs.notes
        assert "coordinate-level" in note or "block-level" in note or "energy" in note


def test_workload_csv_columns_and_safe_text(tmp_path: Path) -> None:
    run = run_selected_observable_workload(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee14"],
            "shots": [100, 1000],
            "trials": 20,
            "command": "test",
        }
    )
    assert set(OBSERVABLE_COLUMNS).issubset(run["observables"].columns)
    assert set(SWEEP_COLUMNS).issubset(run["shot_sweep"].columns)
    assert set(MAP_COLUMNS).issubset(run["readout_map"].columns)
    # full_vector_required is False for every selected observable.
    assert bool((~run["observables"]["full_vector_required"]).all())
    for name in (
        "selected_observables.csv",
        "readout_shot_sweep.csv",
        "readout_map.csv",
        "readout_summary.md",
        "selected_observable_manifest.json",
    ):
        assert (tmp_path / name).is_file()
    assert forbidden_in((tmp_path / "readout_summary.md").read_text(encoding="utf-8")) == []
