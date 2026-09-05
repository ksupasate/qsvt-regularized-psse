"""Guards over the TQE blocking-revision evidence registry.

The registry (outputs/tqe_blocking_revision/evidence_registry.csv) is the
single source of truth for manuscript evidence statuses. These tests fail when
a configuration ID carries contradictory values, when execution tiers are
conflated, or when registry rows lose their artifact grounding.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"

EVIDENCE_STATUSES = {
    "classical_exact",
    "classical_simulation",
    "polynomial_evaluation",
    "qsvt_matrix_action",
    "statevector_dense",
    "sampled_distribution",
    "sampled_simulator",
    "transpiled_partial",
    "transpiled_complete",
    "modeled",
    "excluded",
    "missing",
}


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    path = OUT / "evidence_registry.csv"
    assert path.exists(), "run scripts/build_tqe_evidence_registry.py first"
    return pd.read_csv(path)


def test_registry_statuses_are_from_the_enum(registry: pd.DataFrame) -> None:
    assert set(registry["evidence_status"]) <= EVIDENCE_STATUSES


def test_configuration_ids_are_unique(registry: pd.DataFrame) -> None:
    ids = registry["configuration_id"]
    assert ids.is_unique, ids[ids.duplicated()].tolist()


def test_no_conflicting_degree_or_regularization_within_an_id(
    registry: pd.DataFrame,
) -> None:
    # one row per ID by construction; conflicting values would appear as
    # lambda != alpha / beta^2 within a row.
    for _, row in registry.iterrows():
        try:
            alpha = float(row["alpha"])
            beta = float(row["beta"])
            lam = float(row["lambda"])
        except (TypeError, ValueError):
            continue
        if any(math.isnan(v) for v in (alpha, beta, lam)):
            continue
        assert math.isclose(lam, alpha / beta**2, rel_tol=1e-6), row["configuration_id"]


def test_counting_convention_nu_equals_d_nphi_equals_d_plus_1(
    registry: pd.DataFrame,
) -> None:
    for _, row in registry.iterrows():
        try:
            degree = int(float(row["polynomial_degree"]))
            n_u = int(float(row["signal_calls_per_attempt"]))
            n_phi = int(float(row["phase_operations_per_attempt"]))
        except (TypeError, ValueError):
            continue
        config_id = row["configuration_id"]
        if config_id == "ieee14_fullrect_d255_isolated_readout_wpj":
            # isolated readout: QSVT is not in the circuit, so 0 calls is the
            # only honest count.
            assert n_u == 0 and n_phi == 0
            continue
        assert n_u == degree, config_id
        assert n_phi == degree + 1, config_id


def test_degree_255_and_degree_31_configurations_are_distinct(
    registry: pd.DataFrame,
) -> None:
    d255 = registry[registry["configuration_id"] == "ieee14_fullrect_d255_useful_overlap"]
    d31 = registry[
        registry["configuration_id"] == "ieee14_fullrect_d31_integrated_30seed_lambda_0p068"
    ]
    assert len(d255) == 1 and len(d31) == 1
    assert int(d255["polynomial_degree"].iloc[0]) == 255
    assert int(d31["polynomial_degree"].iloc[0]) == 31
    assert not math.isclose(
        float(d255["lambda"].iloc[0]), float(d31["lambda"].iloc[0]), rel_tol=0.5
    )
    # the d=31 30-seed record is distribution sampling, never Aer backend shots
    assert d31["evidence_status"].iloc[0] == "sampled_distribution"


def test_sampled_rows_reference_shot_artifacts(registry: pd.DataFrame) -> None:
    shot_row = registry[registry["configuration_id"] == "ieee14_fullrect_d255_shot_readout"].iloc[0]
    assert "high_shot_backend" in shot_row["artifact_paths"]
    # a sampled row must not cite only statevector artifacts
    assert "final_quantum_reproduction.csv" not in shot_row["artifact_paths"]


def test_modeled_rows_are_not_described_as_executed(registry: pd.DataFrame) -> None:
    modeled = registry[registry["evidence_status"] == "modeled"]
    assert len(modeled) >= 2
    for _, row in modeled.iterrows():
        notes = str(row["notes"]).lower()
        assert "modeled" in notes or "model" in notes, row["configuration_id"]


def test_ieee57_has_single_matrix_action_status(registry: pd.DataFrame) -> None:
    ieee57 = registry[registry["case"] == "ieee57"]
    assert len(ieee57) == 1
    assert set(ieee57["evidence_status"]) == {"qsvt_matrix_action"}
    matrix = pd.read_csv(OUT / "ieee_case_evidence_matrix.csv")
    row57 = matrix[matrix["case"] == "IEEE-57"].iloc[0]
    assert "qsvt_matrix_action" in row57["status"]
    assert row57["sampled_readout"] == "no"
    assert str(row57["dense_statevector"]).startswith("no")


def test_matrix_fingerprints_are_dimension_consistent(registry: pd.DataFrame) -> None:
    with_fp = registry[
        registry["matrix_fingerprint"].notna() & (registry["matrix_fingerprint"] != "")
    ]
    for fingerprint, group in with_fp.groupby("matrix_fingerprint"):
        assert group["matrix_shape"].nunique() == 1, fingerprint




def test_convention_campaigns_are_not_merged(registry: pd.DataFrame) -> None:
    gen = registry[
        registry["configuration_id"] == "convention_validation_generalized_campaign"
    ].iloc[0]
    fin = registry[registry["configuration_id"] == "convention_validation_final_campaign"].iloc[0]
    assert "245" in gen["notes"]
    assert "150" in fin["notes"]
    assert gen["artifact_paths"] != fin["artifact_paths"]


def test_registry_matches_frozen_configuration_artifact(registry: pd.DataFrame) -> None:
    import json

    cfg = json.loads(
        (
            ROOT
            / "outputs"
            / "final_useful_overlap_validation"
            / "final_scientific_configuration.json"
        ).read_text("utf-8")
    )
    row = registry[registry["configuration_id"] == "ieee14_fullrect_d255_useful_overlap"].iloc[0]
    assert int(row["polynomial_degree"]) == int(cfg["degree"])
    assert math.isclose(float(row["alpha"]), float(cfg["alpha"]), rel_tol=1e-12)
    assert math.isclose(float(row["beta"]), float(cfg["beta"]), rel_tol=1e-12)
    assert math.isclose(float(row["contraction_C"]), float(cfg["contraction_C"]), rel_tol=1e-12)
    assert row["matrix_fingerprint"] == cfg["matrix_checksum"]
