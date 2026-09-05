"""Guards over mechanically derived QSVT resource accounting.

Resource counts must be derived from the configuration degree and recorded
probabilities, never hand-entered; absent costs must be marked, never zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"
FINAL = ROOT / "outputs" / "final_useful_overlap_validation"
TQE_REV = ROOT / "outputs" / "tqe_implementation_revision"


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    path = OUT / "resource_ledger.csv"
    assert path.exists(), "run scripts/build_tqe_evidence_registry.py first"
    return pd.read_csv(path)


def test_degree_dependent_counts_are_derived(ledger: pd.DataFrame) -> None:
    for _, row in ledger.iterrows():
        degree = int(row["degree"])
        n_u = int(row["signal_calls_per_attempt"])
        n_phi = int(row["phase_operations_per_attempt"])
        l_alt = int(row["alternating_sequence_length"])
        if row["configuration_id"] == "ieee14_fullrect_d255_isolated_readout_wpj":
            assert (n_u, n_phi, l_alt) == (0, 0, 0)
            continue
        assert n_u == degree, row["configuration_id"]
        assert n_phi == degree + 1, row["configuration_id"]
        assert l_alt == 2 * degree + 1, row["configuration_id"]


def test_degree255_rows_use_degree255_accounting(ledger: pd.DataFrame) -> None:
    row = ledger[ledger["configuration_id"] == "ieee14_fullrect_d255_useful_overlap"].iloc[0]
    assert int(row["degree"]) == 255
    assert int(row["signal_calls_per_attempt"]) == 255
    assert int(row["phase_operations_per_attempt"]) == 256


def test_expected_attempts_derived_from_quadrature_probability(
    ledger: pd.DataFrame,
) -> None:
    quantum = pd.read_csv(FINAL / "final_quantum_reproduction.csv").iloc[0]
    p_quad = float(quantum["target_quadrature_probability"])
    row = ledger[ledger["configuration_id"] == "ieee14_fullrect_d255_useful_overlap"].iloc[0]
    assert math.isclose(
        float(row["expected_attempts_per_success"]), 1.0 / p_quad, rel_tol=1e-9
    )
    recorded = pd.read_csv(FINAL / "degree255_resource_ledger.csv")
    ledger_row = recorded[
        recorded["item"] == "direct_rejection_expected_attempts_per_success"
    ].iloc[0]
    assert math.isclose(float(ledger_row["value"]), 1.0 / p_quad, rel_tol=1e-9)


def test_d31_sensitivity_numbers_recompute_from_artifact(ledger: pd.DataFrame) -> None:
    meta = json.loads(
        (TQE_REV / "full_rectangular_finite_shot_metadata.json").read_text("utf-8")
    )
    p_succ = float(meta["qsvt"]["postselection_probability"])
    degree = int(meta["qsvt"]["degree"])
    attempts = math.ceil(2500 / p_succ)
    assert attempts == 2947
    assert attempts * degree == 91_357
    assert attempts * (degree + 1) == 94_304
    row = ledger[
        ledger["configuration_id"] == "ieee14_fullrect_d31_integrated_30seed_lambda_0p068"
    ].iloc[0]
    text = str(row["signal_calls_total"])
    for token in (str(attempts), str(attempts * degree), str(attempts * (degree + 1))):
        assert token in text, text


def test_absent_costs_are_marked_not_estimated_never_zero(ledger: pd.DataFrame) -> None:
    for column in ("modeled_loader_cost", "modeled_value_rotation_cost"):
        for value in ledger[column]:
            assert str(value).strip() not in {"0", "0.0", ""}, column


def test_fault_tolerant_cost_is_excluded_everywhere(ledger: pd.DataFrame) -> None:
    assert set(ledger["fault_tolerant_cost"]) == {"excluded"}




