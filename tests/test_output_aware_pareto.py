"""Mechanical output-aware frontier and semantic-label tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.output_aware_sparse_selection import (
    deterministic_pareto_frontier,
)
from robust_qsvt_se.qsvt.sparse_error_precision_study import (
    build_phase_rounding_sensitivity,
)


def test_equal_cost_lower_absolute_error_dominates():
    frame = pd.DataFrame(
        {
            "support_id": ["high", "low"],
            "absolute_error": [0.2, 0.1],
            "gates": [100.0, 100.0],
        }
    )
    candidates, frontier = deterministic_pareto_frontier(
        frame,
        error_column="absolute_error",
        cost_column="gates",
        tie_columns=("support_id",),
    )
    assert candidates.set_index("support_id")["nondominated"].to_dict() == {
        "high": False,
        "low": True,
    }
    assert frontier["support_id"].tolist() == ["low"]


def test_frontier_is_deterministic_under_candidate_permutation():
    frame = pd.DataFrame(
        {
            "support_id": ["a", "b", "c", "d"],
            "absolute_error": [0.4, 0.3, 0.2, 0.1],
            "slots": [1.0, 2.0, 3.0, 4.0],
        }
    )
    first = deterministic_pareto_frontier(
        frame,
        error_column="absolute_error",
        cost_column="slots",
        tie_columns=("support_id",),
    )
    second = deterministic_pareto_frontier(
        frame.sample(frac=1.0, random_state=13),
        error_column="absolute_error",
        cost_column="slots",
        tie_columns=("support_id",),
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_failed_supports_remain_in_candidates_but_not_frontier():
    frame = pd.DataFrame(
        {
            "support_id": ["ok", "failed"],
            "status": ["completed", "failed"],
            "absolute_error": [0.1, np.nan],
            "nonzeros": [8.0, np.nan],
        }
    )
    candidates, frontier = deterministic_pareto_frontier(
        frame,
        error_column="absolute_error",
        cost_column="nonzeros",
        tie_columns=("support_id",),
    )
    assert len(candidates) == 2
    assert not candidates.loc[candidates["support_id"] == "failed", "nondominated"].item()
    assert frontier["support_id"].tolist() == ["ok"]


def test_phase_bits_are_labeled_sensitivity_not_resource_frontier():
    grid = pd.DataFrame(
        {
            "configuration_id": ["p8", "p12"],
            "functional_id": ["ell", "ell"],
            "value_bits": ["6", "6"],
            "phase_bits": ["8", "12"],
            "phase_bits_numeric": [8.0, 12.0],
            "qsvt_absolute_error": [0.2, 0.1],
            "status": ["completed", "completed"],
        }
    )
    sensitivity = build_phase_rounding_sensitivity(grid)
    assert set(sensitivity["curve_kind"]) == {"phase_rounding_sensitivity"}
    assert not sensitivity["executed_resource_frontier"].any()


@pytest.mark.parametrize(
    "stem",
    ["error_nnz", "error_slots", "error_gates"],
)
def test_generated_frontiers_use_absolute_heldout_error_if_present(stem):
    candidate_path = Path(
        f"outputs/output_aware_sparse_selection/pareto_candidates_{stem}.csv"
    )
    frontier_path = Path(
        f"outputs/output_aware_sparse_selection/pareto_frontier_{stem}.csv"
    )
    if not candidate_path.is_file() or not frontier_path.is_file():
        pytest.skip("output-aware Pareto campaign artifacts not generated yet")
    candidates = pd.read_csv(candidate_path)
    frontier = pd.read_csv(frontier_path)
    assert "signed_error" not in candidates.columns
    assert set(candidates["accuracy_semantics"]) == {
        "held_out_absolute_error_no_signed_cancellation"
    }
    rebuilt = candidates[candidates["nondominated"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, frontier, check_dtype=False)
