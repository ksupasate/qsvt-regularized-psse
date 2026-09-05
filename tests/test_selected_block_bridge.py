from __future__ import annotations

import numpy as np

from robust_qsvt_se.paper.selected_block_bridge import build_bridge_rows


def test_bridge_rows_are_deterministic_and_nonzero() -> None:
    first = build_bridge_rows(seed=123)
    second = build_bridge_rows(seed=123)
    assert first == second
    assert {row["block_shape"] for row in first} == {"4x4", "8x8", "16x16"}
    assert all(row["absolute_discrepancy_delta_l"] > 0.0 for row in first)
    assert all(row["relative_discrepancy_vs_full"] > 0.0 for row in first)


def test_bridge_matches_recorded_anchor_values() -> None:
    rows = {row["workload_id"]: row for row in build_bridge_rows(seed=123)}
    anchor = rows["ieee14_4x4_primary_anchor"]
    assert np.isclose(anchor["full_selected_functional"], -0.004838278412072782)
    assert np.isclose(anchor["block_selected_functional"], -0.0042590181430577)
    assert np.isclose(anchor["relative_discrepancy_vs_full"], 0.11972445975197898)
