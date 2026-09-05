"""Physical-validity vs block-size study - determinism, semantics, and aggregation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import robust_qsvt_se.cross_case_validation.block_size_validity as bsv
from robust_qsvt_se.cross_case_validation.block_size_validity import (
    PHYSICAL_FLOOR,
    block_size_plan,
    build_block_context,
    evaluate_full_support_rows,
    resolve_alpha,
    summarize,
    trend_assessment,
)

# ----------------------------------------------------------------- plan / extraction


def test_block_size_plan_respects_state_dimension():
    sizes14, skipped14 = block_size_plan("ieee14")
    assert sizes14 == [8, 16]
    assert [s["size_label"] for s in skipped14] == ["32x32", "64x64"]
    assert all("structurally unavailable" in s["reason"] for s in skipped14)
    sizes30, skipped30 = block_size_plan("ieee30")
    assert sizes30 == [8, 16, 32]
    assert [s["size_label"] for s in skipped30] == ["64x64"]


def test_ieee14_8x8_block_matches_frozen_extractor_selection():
    ctx = build_block_context("ieee14", 8)
    assert ctx.selected_rows == (15, 17, 18, 29, 31, 32, 48, 68)
    assert ctx.selected_columns == (0, 2, 3, 7, 13, 14, 16, 17)
    families = {record.family for record in ctx.functional_records}
    assert families <= {"coordinate", "branch_angle_difference", "area_aggregate"}
    assert "legacy_predetermined" not in families
    assert ctx.design_alpha == pytest.approx(
        4.0 * ctx.conditioning["min_positive_singular_value"] ** 2
    )


def test_blocks_are_nested_for_ieee14():
    small = build_block_context("ieee14", 8)
    large = build_block_context("ieee14", 16)
    assert set(small.selected_columns) <= set(large.selected_columns)
    assert set(small.selected_rows) <= set(large.selected_rows)
    assert len(large.functional_records) >= len(small.functional_records)


# ----------------------------------------------------------------- alpha policies


def test_matched_policy_uses_frozen_alpha_only_for_primary_structure(tmp_path):
    frozen_alpha, frozen_matrix = bsv._frozen_manuscript_alpha(str(tmp_path / "cache"))
    assert frozen_alpha == pytest.approx(1134521.3658711074)
    ctx8 = build_block_context("ieee14", 8)
    assert resolve_alpha(
        ctx8, "matched_frozen_benchmark", frozen_alpha, frozen_matrix
    ) == pytest.approx(frozen_alpha)
    assert resolve_alpha(
        ctx8, "design_4sigma_min", frozen_alpha, frozen_matrix
    ) == pytest.approx(ctx8.design_alpha)
    ctx16 = build_block_context("ieee14", 16)
    assert resolve_alpha(
        ctx16, "matched_frozen_benchmark", frozen_alpha, frozen_matrix
    ) == pytest.approx(ctx16.design_alpha)
    with pytest.raises(ValueError):
        resolve_alpha(ctx8, "no_such_policy", frozen_alpha, frozen_matrix)


def test_matched_policy_guards_frozen_matrix_drift(tmp_path):
    frozen_alpha, frozen_matrix = bsv._frozen_manuscript_alpha(str(tmp_path / "cache"))
    ctx8 = build_block_context("ieee14", 8)
    with pytest.raises(RuntimeError, match="drifted"):
        resolve_alpha(
            ctx8, "matched_frozen_benchmark", frozen_alpha, frozen_matrix + 1.0
        )


# ----------------------------------------------------------------- row semantics


def test_full_support_rows_arithmetic_is_reproducible(monkeypatch):
    monkeypatch.setattr(bsv, "HELD_OUT_SEEDS", (2000, 2001))
    ctx = build_block_context("ieee14", 8)
    rows = evaluate_full_support_rows(ctx, ctx.design_alpha, "design_4sigma_min")
    assert len(rows) == 2 * len(ctx.functional_records)
    frame = pd.DataFrame(rows)
    assert set(frame["seed"]) == {2000, 2001}
    recomputed = np.abs(frame["y_full_ridge"] - frame["y_true"]) / np.maximum(
        np.abs(frame["y_true"]), PHYSICAL_FLOOR
    )
    assert np.allclose(frame["E_physical_norm"], recomputed)
    assert (frame["near_zero_y_true"] == (frame["y_true"].abs() < PHYSICAL_FLOOR)).all()
    # Full support at identical alpha is deterministic.
    again = pd.DataFrame(
        evaluate_full_support_rows(ctx, ctx.design_alpha, "design_4sigma_min")
    )
    assert np.allclose(frame["y_full_ridge"], again["y_full_ridge"])


# ----------------------------------------------------------------- aggregation


def _synthetic_raw() -> pd.DataFrame:
    rows = []
    for value, near_zero in [(0.1, False), (0.3, False), (0.5, False), (99.0, True)]:
        rows.append(
            {
                "case": "ieee14",
                "size_label": "8x8",
                "block_rows": 8,
                "block_cols": 8,
                "alpha_policy": "design_4sigma_min",
                "alpha": 1.0,
                "seed": 2000,
                "functional_id": f"f_{value}",
                "functional_family": "coordinate",
                "y_true": 1.0,
                "y_full_ridge": 1.0 + value,
                "E_full_abs": value,
                "E_physical_norm": value,
                "near_zero_y_true": near_zero,
            }
        )
    return pd.DataFrame(rows)


def test_stage1_median_excludes_near_zero_rows():
    summary, family = summarize(_synthetic_raw())
    assert len(summary) == 1
    assert summary["median_E_physical_norm"].iloc[0] == pytest.approx(0.3)
    assert summary["n_near_zero_y_true"].iloc[0] == 1
    assert family["median_E_physical_norm"].iloc[0] == pytest.approx(0.3)


def test_trend_assessment_flags_and_thresholds():
    summary = pd.DataFrame(
        [
            {"case": "c", "alpha_policy": "p", "size_label": "8x8",
             "median_E_physical_norm": 0.6},
            {"case": "c", "alpha_policy": "p", "size_label": "16x16",
             "median_E_physical_norm": 0.4},
            {"case": "c", "alpha_policy": "p", "size_label": "full",
             "median_E_physical_norm": 0.05},
        ]
    )
    trend = trend_assessment(summary)
    row = trend.iloc[0]
    assert row["size_sequence"] == "8x8>16x16>full"
    assert bool(row["monotonic_nonincreasing"]) and bool(row["strictly_decreasing"])
    assert row["first_size_median_le_0.5"] == "16x16"
    assert row["first_size_median_le_0.1"] == "full"
    assert row["ratio_full_to_8x8"] == pytest.approx(0.05 / 0.6)
