"""Mechanical Pareto-dominance tests: deterministic, uncurated, failure-retaining."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.sparse_error_precision_study import (
    build_phase_rounding_sensitivity,
    build_value_precision_pareto_tables,
    pareto_nondominated,
    precision_key_to_numeric,
)


def _frame(points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["error", "cost"])


def test_dominated_points_are_excluded_deterministically():
    frame = _frame([(1.0, 1.0), (2.0, 2.0), (0.5, 3.0), (2.0, 0.5), (3.0, 3.0)])
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    assert flags.tolist() == [True, False, True, True, False]


def test_equal_points_dominate_nothing_and_survive_together():
    frame = _frame([(1.0, 1.0), (1.0, 1.0)])
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    assert flags.tolist() == [True, True]


def test_weakly_dominated_point_is_excluded():
    frame = _frame([(1.0, 1.0), (1.0, 2.0)])
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    assert flags.tolist() == [True, False]


def test_equal_cost_lower_error_dominates_higher_error():
    frame = _frame([(2.0, 3985.0), (1.0, 3985.0)])
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    assert flags.tolist() == [False, True]


def test_nan_rows_are_retained_in_registry_but_never_on_frontier():
    frame = _frame([(1.0, 1.0), (float("nan"), 2.0), (2.0, float("nan"))])
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    assert flags.tolist() == [True, False, False]
    assert len(frame) == 3  # failed points stay in the candidate registry


def test_dominance_is_permutation_invariant():
    rng = np.random.default_rng(7)
    points = [(float(e), float(c)) for e, c in rng.uniform(0.1, 5.0, size=(24, 2))]
    frame = _frame(points)
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order].reset_index(drop=True)
    shuffled_flags = pareto_nondominated(
        shuffled, error_column="error", cost_column="cost"
    )
    assert shuffled_flags.tolist() == flags.iloc[order].tolist()


def test_frontier_is_monotone_after_sorting():
    rng = np.random.default_rng(11)
    points = [(float(e), float(c)) for e, c in rng.uniform(0.1, 5.0, size=(40, 2))]
    frame = _frame(points)
    flags = pareto_nondominated(frame, error_column="error", cost_column="cost")
    frontier = frame[flags].sort_values("cost")
    errors = frontier["error"].to_numpy()
    assert np.all(np.diff(errors) < 0.0) or len(frontier) == 1


def test_precision_key_numeric_ordering_for_frontiers():
    assert precision_key_to_numeric("8") == 8.0
    assert precision_key_to_numeric("24") == 24.0
    assert precision_key_to_numeric("full") == 53.0
    assert precision_key_to_numeric("exact") == 53.0


def _synthetic_precision_inputs():
    rows = []
    for functional in ("ell_1", "ell_2"):
        for bits, sparse_signed, quant_signed in (
            ("6", -4.0, 3.0),
            ("12", -4.0, 1.0),
            ("exact", -4.0, 0.0),
        ):
            rows.append(
                {
                    "configuration_id": f"{functional}_{bits}_full",
                    "functional_id": functional,
                    "value_bits": bits,
                    "phase_bits": "full",
                    "phase_bits_numeric": 53.0,
                    "sparsification_signed_delta": sparse_signed,
                    "quantization_signed_delta": quant_signed,
                    "sparsification_absolute_error": abs(sparse_signed),
                    "quantization_absolute_error": abs(quant_signed),
                    "qsvt_absolute_error": 0.1,
                    "status": "completed",
                }
            )
        for phase_bits, error in (("8", 0.3), ("12", 0.1)):
            rows.append(
                {
                    "configuration_id": f"{functional}_6_{phase_bits}",
                    "functional_id": functional,
                    "value_bits": "6",
                    "phase_bits": phase_bits,
                    "phase_bits_numeric": float(phase_bits),
                    "sparsification_signed_delta": -4.0,
                    "quantization_signed_delta": 3.0,
                    "sparsification_absolute_error": 4.0,
                    "quantization_absolute_error": 3.0,
                    "qsvt_absolute_error": error,
                    "status": "completed",
                }
            )
    resources = pd.DataFrame(
        {
            "value_bits": ["6", "12", "exact"],
            "one_signal_unitary_gate_count": [3985, 3985, 3985],
        }
    )
    return pd.DataFrame(rows), resources


def test_value_precision_frontier_uses_no_signed_error_cancellation():
    grid, resources = _synthetic_precision_inputs()
    candidates, _frontier = build_value_precision_pareto_tables(grid, resources)
    six = candidates[(candidates["functional_id"] == "ell_1")
                     & (candidates["value_bits"] == "6")].iloc[0]
    # Signed increments would cancel to |-4+3|=1; the declared cost is |4|+|3|=7.
    assert six["error_value"] == 7.0
    assert six["error_semantics"] == (
        "componentwise_absolute_sum_no_signed_cancellation"
    )


def test_value_precision_frontier_includes_complete_candidate_set_and_is_deterministic():
    grid, resources = _synthetic_precision_inputs()
    first_candidates, first_frontier = build_value_precision_pareto_tables(grid, resources)
    second_candidates, second_frontier = build_value_precision_pareto_tables(
        grid.sample(frac=1.0, random_state=9).reset_index(drop=True), resources
    )
    assert len(first_candidates) == 6
    assert set(first_candidates["value_bits"]) == {"6", "12", "exact"}
    pd.testing.assert_frame_equal(first_candidates, second_candidates)
    pd.testing.assert_frame_equal(first_frontier, second_frontier)
    # Exact has the same measured cost and the lowest componentwise absolute error.
    assert set(first_frontier["value_bits"]) == {"exact"}


def test_phase_rounding_is_sensitivity_not_executed_resource_frontier():
    grid, _resources = _synthetic_precision_inputs()
    curve = build_phase_rounding_sensitivity(grid)
    assert set(curve["curve_kind"]) == {"phase_rounding_sensitivity"}
    assert not curve["executed_resource_frontier"].any()
    assert set(curve["resource_semantics"]) == {
        "rotation_parameters_only_no_discrete_gate_synthesis_cost"
    }


def test_generated_frontier_files_are_uncurated_if_present():
    try:
        candidates = pd.read_csv(
            "outputs/sparse_error_precision_study/pareto_candidates_accuracy_cost.csv",
            dtype={"value_bits": str, "phase_bits": str},
        )
        frontier = pd.read_csv(
            "outputs/sparse_error_precision_study/pareto_frontier_accuracy_cost.csv",
            dtype={"value_bits": str, "phase_bits": str},
        )
    except FileNotFoundError:
        pytest.skip("study outputs not generated yet")
    for (axis, functional), group in candidates.groupby(["cost_axis", "functional_id"]):
        recomputed = pareto_nondominated(
            group.reset_index(drop=True), error_column="error_value",
            cost_column="cost_value",
        )
        stored = group["nondominated"].reset_index(drop=True)
        assert recomputed.tolist() == stored.tolist(), (axis, functional)
    merge_columns = ["cost_axis", "functional_id", "configuration_id", "shots_attempted"]
    rebuilt = candidates[candidates["nondominated"]][merge_columns].reset_index(drop=True)
    stored_frontier = frontier[merge_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, stored_frontier)
