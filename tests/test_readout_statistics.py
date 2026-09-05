"""Statistical guards for the blocking-revision readout registry."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    return pd.read_csv(OUT / "readout_registry.csv", keep_default_na=False)


def row(registry: pd.DataFrame, readout_id: str) -> pd.Series:
    found = registry[registry["readout_id"] == readout_id]
    assert len(found) == 1, readout_id
    return found.iloc[0]


def test_probability_events_are_separate(registry: pd.DataFrame) -> None:
    required = {
        "postselection_probability",
        "branch_probability",
        "quadrature_probability",
        "conditional_acceptance_probability",
    }
    assert required <= set(registry.columns)
    assert not any("success_probability" in column for column in registry.columns)
    markers = {"not_applicable", "not_estimated"}
    for column in required:
        for value in registry[column]:
            if str(value) in markers:
                continue
            assert 0.0 <= float(value) <= 1.0, (column, value)


def test_integrated_d255_counts_and_variance(registry: pd.DataFrame) -> None:
    integrated = row(registry, "d255_integrated_branch_hadamard")
    diagnostic = row(registry, "d255_encoded_prefix_diagnostic")
    assert int(integrated["attempted_shots"]) == 5_000_000
    assert int(integrated["readout_accepted_shots"]) == 5_000_000
    assert integrated["postselection_accepted_shots"] == "not_applicable"
    assert int(diagnostic["attempted_shots"]) == 5_000_000
    assert int(diagnostic["postselection_accepted_shots"]) == 3_014_620
    source = pd.read_csv(
        OUT.parent
        / "final_useful_overlap_validation"
        / "final_readout_variance_checks.csv"
    ).iloc[0]
    expected = float(source["variance_per_shot"]) / 5_000_000
    assert math.isclose(float(integrated["analytic_variance"]), expected, rel_tol=1e-12)
    assert math.isclose(float(integrated["standard_error"]), math.sqrt(expected), rel_tol=1e-12)


def test_isolated_wpj_pooled_uncertainty_is_not_single_seed_width(
    registry: pd.DataFrame,
) -> None:
    source = pd.read_csv(
        OUT.parent / "generalized_rectangular_qsvt" / "ieee14_high_precision_backend_summary.csv"
    )
    source = source[source["shots"] == 1_000_000].iloc[0]
    target = row(registry, "d255_isolated_wpj_1000000")
    expected_se = math.sqrt(
        float(source["theoretical_variance_per_shot"]) / int(source["n_seeds"])
    )
    assert math.isclose(float(target["standard_error"]), expected_se, rel_tol=1e-12)
    assert float(target["relative_ci_half_width"]) < float(
        source["aggregate_relative_ci_half_width"]
    )
    assert "single-seed" in target["notes"]


def test_d31_ci_is_for_mean_not_average_of_per_seed_endpoints(
    registry: pd.DataFrame,
) -> None:
    target = row(registry, "d31_integrated_30seed_distribution")
    seeds = pd.read_csv(
        OUT.parent / "tqe_implementation_revision" / "full_rectangular_finite_shot_seeds.csv"
    )
    empirical_variance = float(seeds["selected_output_estimate"].var(ddof=1))
    expected_se = math.sqrt(empirical_variance / 30)
    assert math.isclose(float(target["standard_error"]), expected_se, rel_tol=1e-12)
    assert "Student-t CI for the mean" in target["confidence_interval_scope"]
    assert "not a CI for the mean" in target["notes"]
    assert int(target["postselection_accepted_shots"]) == 2_544_038
    assert int(target["readout_accepted_shots"]) == 2_772_019


def test_variance_validation_compares_like_for_like() -> None:
    frame = pd.read_csv(OUT / "readout_variance_validation.csv", keep_default_na=False)
    assert set(frame["validation_status"]) == {"variance_consistent"}
    assert (frame["variance_ratio_empirical_to_analytic"] >= 0.1).all()
    assert (frame["variance_ratio_empirical_to_analytic"] <= 10.0).all()
    assert frame["comparison_scope"].str.contains("separate").all()


def test_imprecise_area_output_is_not_called_agreement(registry: pd.DataFrame) -> None:
    target = row(registry, "d255_multioutput_area_aggregate_angle")
    statevector = float(target["statevector_reference"])
    assert not (
        float(target["confidence_interval_lower"])
        <= statevector
        <= float(target["confidence_interval_upper"])
    )
    assert "not called" in target["notes"] and "agreement" in target["notes"]
