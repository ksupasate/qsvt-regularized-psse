from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.paper.full_vector_readout import (
    qsvt_target_readout,
    ridge_update,
    run_full_vector_readout,
)


def _toy_problem() -> tuple[np.ndarray, np.ndarray]:
    H = np.array(
        [
            [1.2, 0.2, 0.1, 0.0],
            [0.1, 0.9, 0.2, 0.1],
            [0.0, 0.3, 1.1, 0.2],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    r = np.array([0.5, -0.3, 0.4, -0.2], dtype=np.float64)
    return H, r


_DISCOVERY = (
    "case,subproblem_id,selection_mode,alpha,degree,state_error_gate_vs_ridge,"
    "row_indices,col_indices\n"
    "ieee14,high_leverage_00,high_leverage,0.001,25,0.0222,17 31 48 68,2 3 16 17\n"
    "ieee14,metadata_mapped_02,metadata_mapped,0.0001,35,0.0100,15 29 42 62,0 1 13 14\n"
    "ieee14,residual_supported_03,residual_supported,0.01,35,0.0018,17 32 48 68,2 3 16 17\n"
)


def _write_discovery(input_root: Path) -> None:
    source = input_root / "qsvt_cross_case_solver_prototype"
    source.mkdir(parents=True)
    (source / "cross_case_gate_validated_results.csv").write_text(_DISCOVERY, encoding="utf-8")


def test_statevector_reconstructs_known_vector_exactly() -> None:
    H, r = _toy_problem()
    alpha = 1.0e-4
    state = qsvt_target_readout(H, r, alpha=alpha, degree=15)
    reconstruction = state.readout_state * state.recovered_norm
    ridge = ridge_update(H, r, alpha=alpha)
    assert np.allclose(reconstruction, ridge, atol=1.0e-10)
    relative = np.linalg.norm(reconstruction - ridge) / np.linalg.norm(ridge)
    assert relative < 1.0e-5


def test_qsvt_target_equals_ridge_for_matched_alpha() -> None:
    H, r = _toy_problem()
    for alpha in (1.0e-2, 1.0e-4):
        state = qsvt_target_readout(H, r, alpha=alpha, degree=11)
        ridge = ridge_update(H, r, alpha=alpha)
        ridge_direction = ridge / np.linalg.norm(ridge)
        # The exact QSVT singular-value transform is the Ridge filter for matched alpha.
        assert np.allclose(state.readout_state, ridge_direction, atol=1.0e-12)


def test_run_full_vector_readout_reconstructs_selected_subproblems(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_discovery(input_root)
    run = run_full_vector_readout(
        {
            "input_root": str(input_root),
            "output_dir": str(tmp_path / "full_vector_readout"),
            "cases": ["ieee14"],
            "subproblem_types": ["high_leverage", "metadata_mapped", "residual_supported"],
            "alpha": 1.0e-4,
            "shots": [1000, 4000],
            "seed": 123,
        }
    )
    assert run["subproblem_count"] == 3
    assert run["statevector_successes"] == 3
    assert run["best_statevector_error"] < 1.0e-5

    output_dir = Path(run["output_dir"])
    for name in (
        "artifact_discovery.csv",
        "statevector_full_vector_readout.csv",
        "statevector_full_vector_summary.csv",
        "sampling_magnitude_readout.csv",
        "sampling_error_vs_shots.csv",
        "sign_recovery_readout.csv",
        "norm_scaling_recovery.csv",
        "qsvt_full_vector_scaling_audit.csv",
        "signed_vector_reconstruction.csv",
        "signed_vector_reconstruction_summary.csv",
        "full_vector_readout_cost_model.csv",
        "full_vector_readout_scope_note.md",
    ):
        assert (output_dir / name).is_file(), name


def test_run_records_missing_combinations_honestly(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_discovery(input_root)
    run = run_full_vector_readout(
        {
            "input_root": str(input_root),
            "output_dir": str(tmp_path / "full_vector_readout"),
            "cases": ["ieee14", "ieee30", "ieee57", "ieee118"],
            "subproblem_types": ["high_leverage", "metadata_mapped", "residual_supported"],
            "alpha": 1.0e-4,
            "shots": [1000],
            "seed": 123,
        }
    )
    # Only ieee14 is provided in the synthetic discovery source; the rest are missing.
    assert run["subproblem_count"] == 3
    assert len(run["missing"]) == 9
