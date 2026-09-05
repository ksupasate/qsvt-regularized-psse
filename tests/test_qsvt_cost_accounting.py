from __future__ import annotations

import json
from pathlib import Path

import pytest

from robust_qsvt_se.paper.qsvt_cost_accounting import (
    COST_COMPONENTS,
    build_cost_accounting,
    query_count,
)
from robust_qsvt_se.paper.tqe_revision_support_common import find_forbidden

REQUIRED_COMPONENTS = {
    "access",
    "state_prep",
    "qsvt_signal_and_phase_sequence",
    "postselection",
    "amplitude_amplification",
    "readout",
    "full_vector_recovery",
}


def test_query_count_is_signal_unitary_call_count() -> None:
    assert query_count(51) == 51
    assert query_count(5) == 5
    assert query_count(0) == 0
    with pytest.raises(ValueError):
        query_count(-1)


def test_all_required_cost_terms_present() -> None:
    components = {component.component for component in COST_COMPONENTS}
    assert REQUIRED_COMPONENTS.issubset(components)


def test_build_cost_accounting_marks_excluded_and_provenance(tmp_path: Path) -> None:
    run = build_cost_accounting({"output_dir": str(tmp_path)})
    frame = run["component_frame"]

    assert REQUIRED_COMPONENTS.issubset(set(frame["component"]))

    excluded = frame[frame["component"] == "full_vector_recovery"].iloc[0]
    assert excluded["current_status"] == "excluded"
    assert bool(excluded["included_in_existing_resource_table"]) is False

    qsvt_row = frame[frame["component"] == "qsvt_signal_and_phase_sequence"].iloc[0]
    assert qsvt_row["current_status"] == "validated"
    assert int(qsvt_row["signal_unitary_calls"]) == int(qsvt_row["representative_value"])
    assert int(qsvt_row["projector_phase_operations"]) == int(qsvt_row["signal_unitary_calls"]) + 1
    assert int(qsvt_row["alternating_sequence_length"]) == (
        2 * int(qsvt_row["signal_unitary_calls"]) + 1
    )

    provenance = run["provenance_frame"]
    resolved = provenance[provenance["degree_d"] != ""]
    assert not resolved.empty
    for _, row in resolved.iterrows():
        degree = int(row["degree_d"])
        assert int(row["computed_signal_unitary_calls"]) == query_count(degree)
        assert int(row["computed_projector_phase_operations"]) == degree + 1
        assert int(row["computed_alternating_sequence_length"]) == 2 * degree + 1


def test_cost_accounting_outputs_and_safe_wording(tmp_path: Path) -> None:
    run = build_cost_accounting({"output_dir": str(tmp_path)})
    frame = run["component_frame"]

    for name in (
        "qsvt_cost_accounting.csv",
        "qsvt_cost_accounting_query_provenance.csv",
        "qsvt_cost_accounting.md",
        "qsvt_cost_accounting_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    markdown = (tmp_path / "qsvt_cost_accounting.md").read_text(encoding="utf-8")
    assert "not a speedup claim" in markdown.lower()

    # The manuscript-safe wording fields must not contain any forbidden overclaim,
    # even though the dedicated avoid_wording column intentionally does.
    safe_text = "\n".join(frame["manuscript_safe_wording"].astype(str).tolist())
    assert find_forbidden(safe_text) == []

    manifest = json.loads(
        (tmp_path / "qsvt_cost_accounting_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["speedup_claim"] is False
    assert manifest["hardware_execution"] is False
    assert manifest["excluded_components"] == ["full_vector_recovery"]
