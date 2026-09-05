from __future__ import annotations

from tests.final_useful_overlap_helpers import rows


def test_final_classical_baselines():
    data = rows("final_classical_baselines.csv")
    methods = {row["method"] for row in data}
    assert {
        "dense_ridge",
        "sparse_direct_ridge",
        "classical_adjoint_selected_output",
        "cg_normal_equations",
        "lsmr_augmented",
        "lsqr_augmented",
        "classical_degree255_polynomial",
    } <= methods
    assert all(row["status"] == "success" for row in data)
    assert max(float(row["selected_output_error"]) for row in data) <= 1.0e-5
