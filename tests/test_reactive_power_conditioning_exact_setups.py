"""Phase 1 hardening: exact S1-S3 reactive-power measurement setups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.paper._estimation import (
    DEFAULT_MEASUREMENT,
    SUBSET_INCLUDE_FLAGS,
    build_system,
    conditioning,
    subset_spec,
)
from robust_qsvt_se.paper.reactive_power_conditioning_table import (
    _subset_system,
    build_reactive_power_conditioning_table,
)

_EXACT_S1_S3 = (
    ("s1_voltage_plus_p_injection", {"voltage_magnitude", "p_injection"}),
    (
        "s2_voltage_plus_p_injection_plus_p_branch_flow",
        {"voltage_magnitude", "p_injection", "p_branch_flow"},
    ),
    (
        "s3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
        {"voltage_magnitude", "p_injection", "p_branch_flow", "q_injection"},
    ),
)


def test_s1_s3_subsets_are_exactly_constructible() -> None:
    for subset, expected_types in _EXACT_S1_S3:
        spec = subset_spec(subset)
        assert spec is not None
        assert set(spec.included_types) == expected_types


def test_s1_s3_match_direct_flag_build() -> None:
    # Row-masking the full system must equal building with the include flags directly.
    full = build_system(case="ieee14", measurement_config=dict(DEFAULT_MEASUREMENT), seed=0)
    for subset, _ in _EXACT_S1_S3:
        spec = subset_spec(subset)
        assert spec is not None
        masked = _subset_system(full, set(spec.included_types))
        direct = build_system(case="ieee14", measurement_config=spec.measurement_config, seed=0)
        assert masked.H_tilde.shape == direct.H_tilde.shape
        assert np.allclose(np.sort(masked.singular_values()), np.sort(direct.singular_values()))
        assert np.isclose(
            conditioning(masked)["condition_number"],
            conditioning(direct)["condition_number"],
        )


def test_no_setup_silently_approximated(tmp_path: Path) -> None:
    run = build_reactive_power_conditioning_table(
        {"cases": ["ieee14"], "seeds": [0], "output_dir": str(tmp_path / "rpc")}
    )
    table = pd.read_csv(run["artifacts"]["reactive_power_conditioning_table"])
    assert (table["mapping_status"] == "exact_subset").all()
    # All nine setups (S0-S4 plus the four drop views) are present.
    assert len(table) == 9


def test_missing_file_present_but_records_no_approximation(tmp_path: Path) -> None:
    run = build_reactive_power_conditioning_table(
        {"cases": ["ieee14"], "seeds": [0], "output_dir": str(tmp_path / "rpc")}
    )
    missing = pd.read_csv(run["artifacts"]["missing_reactive_power_rows"])
    # The schema is preserved even when nothing is missing.
    assert list(missing.columns) == ["case", "setup_id", "requested_subset", "reason", "status"]
    assert missing.empty


def test_preexisting_subsets_unchanged() -> None:
    # Adding S1-S3 must not alter the pre-existing measurement subsets.
    assert SUBSET_INCLUDE_FLAGS["voltage_only"] == {"include_voltage_magnitudes"}
    assert SUBSET_INCLUDE_FLAGS["full_ac_measurement_set"] == {
        "include_voltage_magnitudes",
        "include_p_injections",
        "include_q_injections",
        "include_p_branch_flows",
        "include_q_branch_flows",
    }
    assert "include_q_injections" not in SUBSET_INCLUDE_FLAGS["drop_q_injection_rows"]
