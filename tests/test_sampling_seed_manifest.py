from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.full_vector_readout import run_full_vector_readout

_DISCOVERY = (
    "case,subproblem_id,selection_mode,alpha,degree,state_error_gate_vs_ridge,"
    "row_indices,col_indices\n"
    "ieee14,high_leverage_00,high_leverage,0.001,15,0.0222,17 31 48 68,2 3 16 17\n"
)


def _run(tmp_path: Path) -> Path:
    input_root = tmp_path / "outputs"
    source = input_root / "qsvt_cross_case_solver_prototype"
    source.mkdir(parents=True)
    (source / "cross_case_gate_validated_results.csv").write_text(_DISCOVERY, encoding="utf-8")
    run = run_full_vector_readout(
        {
            "input_root": str(input_root),
            "output_dir": str(tmp_path / "fvr"),
            "cases": ["ieee14"],
            "subproblem_types": ["high_leverage"],
            "alpha": 1.0e-4,
            "shots": [500, 1000],
            "seed": 17,
            "trials": 3,
        }
    )
    return Path(run["output_dir"])


def test_sampling_artifacts_have_seed_and_trial_columns(tmp_path: Path) -> None:
    out = _run(tmp_path)
    for name in (
        "sampling_magnitude_readout.csv",
        "sign_recovery_readout.csv",
        "signed_vector_reconstruction.csv",
        "readout_sampling_trials.csv",
        "shot_based_norm_recovery.csv",
    ):
        frame = pd.read_csv(out / name)
        assert {"trial_id", "rng_seed"}.issubset(frame.columns), name


def test_seed_manifest_lists_sampling_artifacts(tmp_path: Path) -> None:
    out = _run(tmp_path)
    manifest = pd.read_csv(out / "sampling_seed_manifest.csv")
    assert {"artifact", "rng_seed", "trial_id", "deterministic_replay_command"}.issubset(
        manifest.columns
    )
    joined = " ".join(manifest["artifact"].astype(str))
    assert "readout_sampling_trials.csv" in joined
    # Trial-0 rows also seed the single-draw per-coordinate artifacts.
    assert "sampling_magnitude_readout.csv" in joined


def test_same_seed_reproduces_rows(tmp_path: Path) -> None:
    out_a = _run(tmp_path / "a")
    out_b = _run(tmp_path / "b")
    a = pd.read_csv(out_a / "readout_sampling_trials.csv")
    b = pd.read_csv(out_b / "readout_sampling_trials.csv")
    pd.testing.assert_series_equal(a["rng_seed"], b["rng_seed"])
    pd.testing.assert_series_equal(a["vector_relative_l2_error"], b["vector_relative_l2_error"])
