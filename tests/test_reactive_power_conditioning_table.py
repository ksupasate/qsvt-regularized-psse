from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.reactive_power_conditioning_table import (
    build_reactive_power_conditioning_table,
)


def _run(tmp_path: Path) -> dict:
    # ieee14 with two seeds keeps the real computation fast and deterministic.
    return build_reactive_power_conditioning_table(
        {"cases": ["ieee14"], "seeds": [0, 1], "output_dir": str(tmp_path / "rpc")}
    )


def test_no_claim_q_always_improves_conditioning(tmp_path: Path) -> None:
    run = _run(tmp_path)
    summary = Path(run["artifacts"]["reactive_power_conditioning_summary"]).read_text(
        encoding="utf-8"
    )
    assert "No monotonic improvement is claimed" in summary
    # The summary must not assert that reactive-power rows uniformly improve conditioning.
    assert "always lowers the condition number" not in summary.lower()


def test_exact_s1_s3_setups_computed(tmp_path: Path) -> None:
    run = _run(tmp_path)
    table = pd.read_csv(run["artifacts"]["reactive_power_conditioning_table"])
    setups = set(table["setup_id"])
    for setup_id in (
        "S1_voltage_plus_p_injection",
        "S2_voltage_plus_p_injection_plus_p_branch_flow",
        "S3_voltage_plus_p_injection_plus_p_branch_flow_plus_q_injection",
    ):
        assert setup_id in setups
        row = table[table["setup_id"] == setup_id].iloc[0]
        assert row["mapping_status"] == "exact_subset"
        # Real conditioning value, not a blank placeholder.
        assert pd.notna(row["condition_number"])
    # No setup is silently approximated and no S1-S3 row is recorded missing.
    missing = pd.read_csv(run["artifacts"]["missing_reactive_power_rows"])
    assert missing.empty or not missing["setup_id"].isin(setups).any()


def test_monotone_buildup_is_exact_not_approximated(tmp_path: Path) -> None:
    run = _run(tmp_path)
    table = pd.read_csv(run["artifacts"]["reactive_power_conditioning_table"])
    # The full S0->S4 build-up is present and every row is an exact subset.
    assert (table["mapping_status"] == "exact_subset").all()
    assert {"S0_voltage_only", "S4_full_pq_ac_measurement_set"}.issubset(set(table["setup_id"]))


def test_voltage_only_flagged_rank_deficient(tmp_path: Path) -> None:
    run = _run(tmp_path)
    table = pd.read_csv(run["artifacts"]["reactive_power_conditioning_table"])
    s0 = table[table["setup_id"] == "S0_voltage_only"].iloc[0]
    assert "rank_deficient" in str(s0["notes"])
