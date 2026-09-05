from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    fit_codesigned_bounded_polynomial,
    forbidden_in,
)
from robust_qsvt_se.paper.selected_observable_qsvt_demo import (
    build_block_observables,
    run_selected_observable_qsvt_demo,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system, ridge_svd_solution

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("selected_observable_qsvt_demo")
    cache_dir = tmp_path_factory.mktemp("phase_cache")
    sigma_min = _block_sigma_min()
    config = {
        "output_dir": str(output_dir),
        "phase_cache_dir": str(cache_dir),
        "readout_shots": [2_000, 20_000],
        "blocks": [{"size": 4, "alpha": 4.0 * sigma_min**2, "degree": 31, "headline": True}],
    }
    return run_selected_observable_qsvt_demo(config)


def _block_sigma_min() -> float:
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H_block, _r, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    return float(np.linalg.svd(H_block, compute_uv=False).min())


def test_block_and_residual_extraction_is_deterministic():
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H = np.asarray(system.H_tilde, dtype=np.float64)
    r = np.asarray(system.r_tilde, dtype=np.float64)
    first = select_deterministic_block(
        H, r, row_count=4, col_count=4, policy="largest_row_col_norms"
    )
    second = select_deterministic_block(
        H, r, row_count=4, col_count=4, policy="largest_row_col_norms"
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])


def test_ridge_reference_matches_normal_equations():
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    H_block, r_block, _rows, _cols = select_deterministic_block(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        row_count=4,
        col_count=4,
        policy="largest_row_col_norms",
    )
    alpha = 4.0 * float(np.linalg.svd(H_block, compute_uv=False).min()) ** 2
    direct = np.linalg.solve(
        H_block.T @ H_block + alpha * np.eye(H_block.shape[1]), H_block.T @ r_block
    )
    svd_solution = ridge_svd_solution(H_block, r_block, alpha=alpha)
    np.testing.assert_allclose(svd_solution, direct, atol=1e-10)


def test_bounded_target_is_bounded_by_one_on_domain():
    target = fit_codesigned_bounded_polynomial(
        beta=2300.0, alpha=4.0 * 300.0**2, domain_min=0.1, domain_max=1.0, degree=31
    )
    grid = np.linspace(-1.0, 1.0, 4001)
    assert np.max(np.abs(target.polynomial(grid))) <= 1.0 + 1.0e-3
    assert target.bounded_max_abs <= 1.0 + 1.0e-3


def test_selected_observables_are_predetermined_not_solution_dependent():
    labels = [
        {"block_column": i, "full_state_index": i, "state_type": "angle", "bus_id": None}
        for i in range(4)
    ]
    first = build_block_observables(4, labels)
    second = build_block_observables(4, labels)
    # Identical regardless of any solution; vectors are fixed e_0, e_0-e_1, area, energy.
    for obs_a, obs_b in zip(first, second, strict=True):
        np.testing.assert_array_equal(obs_a.vector, obs_b.vector)
    ids = {obs.observable_id for obs in first}
    assert "state_correction_0" in ids
    np.testing.assert_array_equal(first[0].vector, np.array([1.0, 0.0, 0.0, 0.0]))


def test_headline_block_passes_with_consistent_physical_rescaling(demo_run):
    headline = demo_run["headline"]
    assert headline.status_label == "pass"
    # Physical rescaling: QSVT update reproduces the matched Ridge update.
    assert headline.row_common["update_relative_error_vs_ridge"] < 0.05
    summary = demo_run["summary"]
    block_rows = summary[summary["block_shape"] == "4x4"]
    assert not block_rows.empty
    for _, row in block_rows.iterrows():
        if abs(row["ridge_reference_value"]) > 1e-6:
            assert row["relative_error"] < 0.05


def test_phase_synthesis_status_is_recorded(demo_run):
    headline = demo_run["headline"]
    assert headline.row_common["phase_synthesis_status"] == "completed"
    assert headline.row_common["phase_count"] == headline.row_common["degree"] + 1


def test_normalization_record_uses_single_C_over_beta_factor(demo_run):
    headline = demo_run["headline"]
    beta = headline.row_common["beta"]
    bound_c = headline.row_common["bound_C"]
    factor = headline.row_common["physical_recovery_factor_C_over_beta"]
    assert factor == pytest.approx(bound_c / beta, rel=1e-9)


def test_demo_outputs_have_no_forbidden_wording(demo_run):
    output_dir = demo_run["output_dir"]
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    for name in ("demo_summary.csv", "qsvt_pipeline_metadata.json", "normalization_record.json"):
        assert (output_dir / name).is_file()


def test_shot_readout_is_explicitly_isolated_from_qsvt(demo_run):
    output_dir = demo_run["output_dir"]
    metadata = json.loads((output_dir / "readout_metadata.json").read_text("utf-8"))
    assert metadata["integrated_qsvt_readout"] is False
    assert metadata["output_state_access_model"] == (
        "direct_StatePreparation_of_classically_computed_postselected_output"
    )
    assert metadata["status"] == "isolated_overlap_shot_experiment_completed"
