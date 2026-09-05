from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.physical_alignment.statistics import (
    aggregate_structure_units,
    bootstrap_paired_effect,
    case_stratified_bootstrap_effect,
    leave_one_case_out,
)


def _raw_rows() -> pd.DataFrame:
    rows = []
    cases = ("ieee14", "ieee30", "ieee57")
    selectors = ("global_magnitude", "noise_propagation_risk_mean_refined")
    for case_index, case in enumerate(cases):
        for group_index in range(4):
            group_id = f"{case}_group_{group_index}"
            for realization in (1, 2):
                instance_id = f"{group_id}_realization_{realization}"
                for selector in selectors:
                    selector_shift = 0.0 if selector == "global_magnitude" else -0.1
                    for seed in (100, 200):
                        for functional_id, family in (
                            ("angle", "coordinate_angle_update"),
                            ("voltage", "coordinate_voltage_magnitude_update"),
                        ):
                            base = 1.0 + 0.1 * case_index + 0.01 * group_index
                            value = (
                                base
                                + selector_shift
                                + 0.001 * realization
                                + 0.00001 * seed
                                + (0.002 if functional_id == "voltage" else 0.0)
                            )
                            rows.append(
                                {
                                    "structural_group_id": group_id,
                                    "instance_id": instance_id,
                                    "realization_order": realization,
                                    "ieee_case": case,
                                    "selector": selector,
                                    "functional_classification": "physical",
                                    "functional_family": family,
                                    "residual_seed": seed,
                                    "functional_id": functional_id,
                                    "E_physical": value,
                                    "A_physical": value / 10.0,
                                    "E_support": value / 20.0,
                                    "status": "completed",
                                }
                            )
    return pd.DataFrame(rows)


def _paired() -> pd.DataFrame:
    rows = []
    for case_index, case in enumerate(("ieee14", "ieee30", "ieee57")):
        for group_index in range(4):
            rows.append(
                {
                    "structural_group_id": f"{case}_{group_index}",
                    "ieee_case": case,
                    "candidate": 0.8 + 0.01 * group_index,
                    "baseline": 1.0 + 0.02 * case_index,
                    "effect": 0.2 + 0.02 * case_index - 0.01 * group_index,
                }
            )
    return pd.DataFrame(rows)


def test_two_stage_aggregation_has_exactly_twelve_independent_units() -> None:
    realization, structure = aggregate_structure_units(_raw_rows())
    primary = structure.loc[
        structure["metric"].eq("E_physical") & structure["summary_scope"].eq("all_families")
    ]
    assert primary["structural_group_id"].nunique() == 12
    assert primary.groupby("selector")["structural_group_id"].nunique().eq(12).all()
    assert structure["realization_count"].eq(2).all()
    assert realization["held_out_seed_count"].eq(2).all()
    assert (
        realization.loc[realization["summary_scope"].eq("all_families"), "held_out_row_count"]
        .eq(4)
        .all()
    )
    assert (
        realization.loc[
            realization["summary_scope"].str.startswith("family:"), "held_out_row_count"
        ]
        .eq(2)
        .all()
    )
    assert structure["independent_unit"].eq("structural_group_id").all()
    assert not structure["numerical_realizations_are_independent_units"].any()


def test_repeated_tasks_are_collapsed_before_structure_bootstrap() -> None:
    _realization, structure = aggregate_structure_units(_raw_rows())
    primary = structure.loc[
        structure["metric"].eq("E_physical")
        & structure["summary_scope"].eq("all_families")
        & structure["selector"].eq("global_magnitude")
    ]
    assert len(primary) == 12
    assert primary["held_out_row_count"].eq(8).all()


def test_ordinary_and_case_stratified_confidence_intervals_are_reproducible() -> None:
    paired = _paired()
    ordinary_a = bootstrap_paired_effect(paired, replicates=10_000, seed=44)
    ordinary_b = bootstrap_paired_effect(paired, replicates=10_000, seed=44)
    stratified_a = case_stratified_bootstrap_effect(paired, replicates=10_000, seed=55)
    stratified_b = case_stratified_bootstrap_effect(paired, replicates=10_000, seed=55)
    assert ordinary_a == ordinary_b
    assert stratified_a == stratified_b
    assert ordinary_a.resampling_unit == "structural_group_id"
    assert stratified_a.resampling_unit == "structural_group_id within ieee_case"
    assert ordinary_a.confidence_interval_low > 0.0
    assert stratified_a.confidence_interval_low > 0.0


def test_case_stratified_bootstrap_rejects_duplicate_structure_rows() -> None:
    paired = _paired()
    duplicated = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly once"):
        case_stratified_bootstrap_effect(duplicated, replicates=100, seed=1)


def test_leave_one_case_out_removes_every_structure_in_the_case() -> None:
    paired = _paired()
    result = leave_one_case_out(paired)
    assert set(result["omitted_case"]) == {"ieee14", "ieee30", "ieee57"}
    assert result["retained_structure_count"].eq(8).all()
    assert result["complete_case_removed"].all()


def test_bootstrap_uses_effect_sign_baseline_minus_candidate() -> None:
    paired = _paired()
    expected = float(np.mean(paired["baseline"] - paired["candidate"]))
    result = bootstrap_paired_effect(paired, replicates=10_000, seed=9)
    assert result.observed_effect == pytest.approx(expected)
    assert result.probability_effect_positive == 1.0
