from __future__ import annotations

import pytest

from tests.final_useful_overlap_helpers import rows


def test_final_application_reproduction():
    row = rows("final_application_reproduction.csv")[0]
    assert row["status"] == "pass"
    assert float(row["rmse_ratio_vs_benchmark"]) == pytest.approx(0.9654351225695081)
    assert row["application_useful"] == "True"
