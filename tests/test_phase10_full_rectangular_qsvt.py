"""Tests for the Phase 10 WP B full rectangular selected-output QSVT execution."""

from __future__ import annotations

import numpy as np
import pytest

from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import (
    ALPHA_TIERS,
    build_padded_dilation,
    execute_case,
    run_full_rectangular_qsvt,
)
from robust_qsvt_se.qsvt.engineering_utils import build_engineering_system


@pytest.fixture(scope="module")
def ieee14_system():
    system, _ = build_engineering_system(
        {
            "case_name": "ieee14",
            "case_source": "pypower",
            "matrix_source": "weighted_jacobian",
            "seed": 123,
        }
    )
    return system


def test_executed_matrix_is_full_rectangular_not_submatrix(ieee14_system):
    H = np.asarray(ieee14_system.H_tilde)
    # The full generated IEEE 14 measurement system: 82 rows, 27 states.
    assert H.shape == (82, 27)
    beta = float(np.linalg.svd(H, compute_uv=False).max())
    dilation = build_padded_dilation(H, beta)
    # A = H^T/beta is 27x82, padded to 128 (max(82,27) -> 128).
    assert dilation["n_states"] == 27
    assert dilation["m_measurements"] == 82
    assert dilation["padded_dimension"] == 128
    assert dilation["unitary_dimension"] == 256
    assert dilation["top_left_block_error"] <= 1e-8
    assert dilation["unitarity_error"] <= 1e-7


def test_full_residual_prepared_and_recovery_uses_c_over_beta(ieee14_system):
    H = np.asarray(ieee14_system.H_tilde)
    r = np.asarray(ieee14_system.r_tilde)
    beta = float(np.linalg.svd(H, compute_uv=False).max())
    alpha = 0.068 * beta**2
    record = run_full_rectangular_qsvt(
        H,
        r,
        alpha=alpha,
        degree=31,
        margin=1.05,
        phase_cache_dir="outputs/phase10_full_rectangular_selected_output_qsvt/phase_cache",
        beta=beta,
        run_circuit_path=True,
    )
    assert record["status"] == "executed_pass"
    # Full residual dimension prepared (all 82 measurement rows).
    assert record["residual_dimension_prepared"] == 82
    # Physical recovery is exactly C/beta (Option B convention).
    assert record["physical_recovery_factor_C_over_beta"] == pytest.approx(
        record["bound_C"] / beta, rel=1e-12
    )
    # Circuit and matrix-vector paths agree; matches exact SVT on padded matrix.
    assert record["circuit_vs_matvec_error"] <= 1e-10
    assert record["matvec_vs_exact_svt_update_error"] <= 1e-6
    # Padding tail vanishes (odd polynomial preserves the zero padding).
    assert record["padding_tail_norm"] <= 1e-9
    assert record["update_relative_error_vs_full_ridge"] <= 0.05


def test_degree_limited_tiers_recorded_not_hidden(ieee14_system):
    H = np.asarray(ieee14_system.H_tilde)
    r = np.asarray(ieee14_system.r_tilde)
    beta = float(np.linalg.svd(H, compute_uv=False).max())
    # Canonical alpha=1e-4 gives a tiny lambda that the synthesis ceiling cannot
    # meet: it must still return a recorded status, never silently pass.
    record = run_full_rectangular_qsvt(
        H,
        r,
        alpha=1.0e-4,
        degree=31,
        margin=1.05,
        phase_cache_dir="outputs/phase10_full_rectangular_selected_output_qsvt/phase_cache",
        beta=beta,
        run_circuit_path=False,
    )
    assert record["status"] in {
        "executed_degree_limited",
        "bounded_polynomial_invalid",
    }
    if record["status"] == "executed_degree_limited":
        assert record["update_relative_error_vs_full_ridge"] > 0.05


@pytest.fixture(scope="module")
def executed_ieee14(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase10_full_rect_exec")
    # Only the anchor tier and degree 31 keeps this fast while still exercising
    # the full end-to-end executed path and the selected functionals.
    return execute_case(
        "ieee14",
        seed=123,
        phase_cache_dir=output_dir / "phase_cache",
        postselection_seed=20261,
        tiers=(ALPHA_TIERS[3],),
        degree_candidates=(31,),
    )


def test_selected_functionals_agree_with_full_system_ridge(executed_ieee14):
    rows = executed_ieee14["selected_rows"]
    assert rows
    names = {row["functional"] for row in rows}
    # First coordinate, a voltage magnitude, a branch-angle difference, an area aggregate.
    assert "first_state_coordinate" in names
    assert "first_voltage_magnitude" in names
    assert "area_aggregate_angle" in names
    for row in rows:
        # Selected outputs compared against the FULL-system Ridge update.
        assert row["absolute_error"] <= 5e-2 * (abs(row["selected_output_full_ridge"]) + 1e-3)


def test_postselection_sampled_at_finite_shots(executed_ieee14):
    records = executed_ieee14["postselection_records"]
    assert records
    for record in records:
        assert record["shots"] == 4096
        # Sampled acceptance is close to the exact postselection probability.
        assert abs(record["p_hat_succ"] - record["exact_p_succ"]) < 0.05


def test_full_run_outputs_and_claim_safe():
    import tempfile

    from robust_qsvt_se.paper.phase10_full_rectangular_qsvt import run_phase10_full_rectangular

    with tempfile.TemporaryDirectory() as tmp:
        run = run_phase10_full_rectangular(
            {
                "output_dir": tmp,
                "executed_cases": ["ieee14"],
                "modeled_cases": ["ieee57"],
            }
        )
        output_dir = run["output_dir"]
        for name in (
            "full_rectangular_cases_summary.csv",
            "full_rectangular_selected_outputs.csv",
            "full_rectangular_qsvt_vs_ridge.csv",
            "full_rectangular_resource_accounting.csv",
            "full_rectangular_block_encoding_metadata.json",
            "full_rectangular_postselection.json",
            "README.md",
            "manifest.json",
            "checksums.sha256",
            "command_log.txt",
        ):
            assert (output_dir / name).is_file(), name
        readme = (output_dir / "README.md").read_text(encoding="utf-8")
        assert forbidden_in(readme) == []
        # Modeled cases must be labeled not-executed.
        modeled = run["modeled_rows"]
        assert modeled
        for row in modeled:
            assert row["status"] == "resource_estimated_not_executed"
