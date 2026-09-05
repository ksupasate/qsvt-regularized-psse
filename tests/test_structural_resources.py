from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_structural_generalization"


def test_completed_resource_wrappers_have_measured_nonzero_costs_and_reconstruct() -> None:
    config = json.loads((ROOT / "configs/output_aware_structural_generalization.json").read_text())
    frame = pd.read_csv(OUT / "resource_registry.csv")
    completed = frame[frame["status"] == "completed"]
    assert not completed.empty
    for column in (
        "actual_nonzeros",
        "slot_count",
        "signal_unitary_gate_count",
        "signal_unitary_depth",
        "cx_count",
        "controlled_rotations",
    ):
        assert (completed[column] > 0).all()
    assert completed["slot_assignment_valid"].astype(bool).all()
    assert completed["real_edges_covered_exactly_once"].astype(bool).all()
    assert (
        completed["wrapper_reconstruction_error"]
        <= config["resources"]["wrapper_reconstruction_tolerance"]
    ).all()
    assert (~completed["missing_cost_is_zero"].astype(bool)).all()


def test_resource_failures_are_retained_and_group_summary_is_complete() -> None:
    frame = pd.read_csv(OUT / "resource_registry.csv")
    failed = frame[frame["status"] != "completed"]
    if not failed.empty:
        assert failed["failure_reason"].fillna("").str.len().gt(0).all()
    grouped = pd.read_csv(OUT / "resource_group_summary.csv")
    assert grouped["structural_group_id"].nunique() == 12
    assert set(grouped["ieee_case"]) == {"ieee14", "ieee30", "ieee57"}
