from __future__ import annotations

from tests.final_useful_overlap_helpers import rows


def test_final_error_budget():
    data = rows("final_same_configuration_error_budget.csv")
    families = {row["error_family"] for row in data}
    assert {"application", "deterministic", "statistical", "modeled"} <= families
    assert any(row["term"] == "application_regularization_bias" for row in data)
    assert any(row["term"] == "shot_statistical_error" for row in data)
