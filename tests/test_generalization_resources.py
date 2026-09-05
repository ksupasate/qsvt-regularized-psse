from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "output_aware_generalization"


def test_resource_registry_uses_actual_wrapper_circuits() -> None:
    resources = pd.read_csv(OUT / "resource_registry.csv")
    assert set(resources["resource_record_type"]) == {"executed_sparse_signal_unitary"}
    assert resources["resource_measurement"].str.startswith("actual_qiskit_transpile").all()
    assert (resources["signal_unitary_gate_count"] > 0).all()
    assert (resources["signal_unitary_depth"] > 0).all()
    assert (resources["controlled_rotations"] > 0).all()
    assert (~resources["missing_cost_is_zero"].astype(bool)).all()
    assert resources["slot_assignment_valid"].astype(bool).all()
    failures = resources[resources["status"] != "completed"]
    assert failures["failure_reason"].fillna("").str.len().gt(0).all()


def test_resource_and_support_fingerprints_match() -> None:
    resources = pd.read_csv(OUT / "resource_registry.csv")
    supports = pd.read_csv(OUT / "support_registry.csv")
    merged = resources.merge(
        supports[["support_id", "support_fingerprint", "sparse_matrix_fingerprint"]],
        on="support_id",
        suffixes=("_resource", "_support"),
    )
    assert (
        merged["support_fingerprint_resource"] == merged["support_fingerprint_support"]
    ).all()
    assert (
        merged["sparse_matrix_fingerprint_resource"]
        == merged["sparse_matrix_fingerprint_support"]
    ).all()


def test_gate_counts_are_not_substituted_by_nonzero_counts() -> None:
    completed = pd.read_csv(OUT / "resource_registry.csv").query("status == 'completed'")
    assert (completed["signal_unitary_gate_count"] != completed["actual_nonzeros"]).all()
    assert completed.groupby("actual_nonzeros")["signal_unitary_gate_count"].nunique().max() > 1
