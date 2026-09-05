from __future__ import annotations

import pytest

from tests.final_useful_overlap_helpers import rows


def test_final_quantum_reproduction():
    row = rows("final_quantum_reproduction.csv")[0]
    assert row["status"] == "pass"
    assert float(row["polynomial_max_after_repair"]) == pytest.approx(0.9999999900000403)
    assert float(row["boundedness_margin"]) > 0.0
    assert float(row["phase_reconstruction_error"]) <= 1.0e-12
    assert float(row["production_vs_reference_relative_error"]) <= 1.0e-8
    assert float(row["selected_relative_error_vs_ridge"]) <= 1.0e-3
