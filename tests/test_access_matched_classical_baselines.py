import numpy as np

from robust_qsvt_se.paper.tqe_revision_core import access_matched_classical_baselines


def test_access_matched_methods_compute_same_selected_output_or_report_approximation():
    rng = np.random.default_rng(44)
    H = rng.normal(size=(20, 7))
    r = rng.normal(size=20)
    c = np.zeros(7)
    c[2] = 1.0
    rows = access_matched_classical_baselines(H, r, c, alpha=0.3, repeats=3)
    assert {row["method"] for row in rows} == {
        "dense_ridge",
        "sparse_direct_ridge",
        "classical_adjoint",
        "matrix_free_cg_normal_equations",
        "lsmr_augmented_ridge",
        "fixed_8_step_krylov_filter",
    }
    exact = [row for row in rows if row["method"] != "fixed_8_step_krylov_filter"]
    assert max(row["selected_output_relative_error"] for row in exact) < 1e-7
