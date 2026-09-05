from __future__ import annotations

from tests.final_useful_overlap_helpers import rows


def test_final_readout_estimator():
    data = rows("final_readout_validation.csv")
    analytic = next(
        row for row in data if row["validation_method"] == "analytic_statevector_overlap"
    )
    assert analytic["status"] == "pass"
    assert float(analytic["absolute_error"]) <= 1.0e-12
    diagnostics = [
        row for row in data if row["validation_method"] == "distribution_monte_carlo_diagnostic"
    ]
    assert diagnostics
    assert all(row["evidence_label"] == "DISTRIBUTION_MONTE_CARLO" for row in diagnostics)
