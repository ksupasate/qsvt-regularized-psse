from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.initializer_ablation import build_initializer_ablation


def _run(tmp_path: Path) -> dict:
    # Minimal real run: one small case, one seed, one estimator to bound runtime.
    return build_initializer_ablation(
        {
            "cases": ["ieee14"],
            "initializers": ["controlled_perturbed_true_state", "dc_warm_start"],
            "estimators": ["ridge_tikhonov"],
            "alpha": 1e-4,
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "init_ablation"),
        }
    )


def test_unknown_initializer_recorded_unsupported(tmp_path: Path) -> None:
    # A genuinely unknown initializer is still recorded as unsupported, not invented.
    run = build_initializer_ablation(
        {
            "cases": ["ieee14"],
            "initializers": ["ac_power_flow_warm_start"],
            "estimators": ["ridge_tikhonov"],
            "alpha": 1e-4,
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "init_ablation"),
        }
    )
    results = pd.read_csv(run["artifacts"]["initializer_ablation_results"])
    unknown = results[results["initializer"] == "ac_power_flow_warm_start"]
    assert not unknown.empty
    assert (unknown["result_status"] == "not_supported_by_current_code").all()
    missing = pd.read_csv(run["artifacts"]["missing_initializer_ablation_outputs"])
    assert "ac_power_flow_warm_start" in set(missing["initializer"])


def test_supported_initializer_computed_with_initial_rmse(tmp_path: Path) -> None:
    run = _run(tmp_path)
    results = pd.read_csv(run["artifacts"]["initializer_ablation_results"])
    computed = results[results["result_status"] == "computed"]
    assert not computed.empty
    # initial_rmse is a real finite distance of the start from the true state.
    assert (pd.to_numeric(computed["initial_rmse"], errors="coerce") >= 0).all()
    assert computed["final_rmse"].notna().all()


def test_qsvt_ridge_equivalent_flag(tmp_path: Path) -> None:
    run = build_initializer_ablation(
        {
            "cases": ["ieee14"],
            "initializers": ["flat_start"],
            "estimators": ["ridge_tikhonov", "qsvt_target_classical"],
            "alpha": 1e-4,
            "seeds": [0],
            "input_root": str(tmp_path),
            "output_dir": str(tmp_path / "init_ablation"),
        }
    )
    assert run["qsvt_ridge_equivalent"] is True
