from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.norm_recovery_from_amplitude import (
    SUMMARY_COLUMNS,
    apply_norm_recovery,
    evaluate_norm_recovery_point,
    write_norm_recovery_outputs,
)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    H = np.array([[1.0, 0.2], [0.15, 0.8]], dtype=np.float64)
    r = np.array([0.4, -0.2], dtype=np.float64)
    return H, r


def test_norm_recovery_is_deterministic_for_fixed_seed() -> None:
    H, r = _toy()
    first = evaluate_norm_recovery_point(
        H=H,
        r=r,
        alpha=1.0e-3,
        degree=5,
        shots=512,
        methods=["bernoulli_success_amplitude"],
        grover_powers=(0, 1, 2),
        seed=99,
    )
    second = evaluate_norm_recovery_point(
        H=H,
        r=r,
        alpha=1.0e-3,
        degree=5,
        shots=512,
        methods=["bernoulli_success_amplitude"],
        grover_powers=(0, 1, 2),
        seed=99,
    )
    assert first[0]["scale_factor"] == second[0]["scale_factor"]
    assert first[0]["residual_after_recovery"] == second[0]["residual_after_recovery"]


def test_best_scalar_upper_bound_is_diagnostic_only() -> None:
    H, r = _toy()
    rows = evaluate_norm_recovery_point(
        H=H,
        r=r,
        alpha=1.0e-3,
        degree=5,
        shots=512,
        methods=["best_scalar_upper_bound"],
        grover_powers=(0, 1),
        seed=1,
    )
    row = rows[0]
    assert row["deployability_class"] == "statevector_diagnostic_only"
    assert row["uses_exact_statevector"] is True


def test_best_scalar_upper_bound_minimizes_residual_over_deployable() -> None:
    H, r = _toy()
    rows = evaluate_norm_recovery_point(
        H=H,
        r=r,
        alpha=1.0e-3,
        degree=5,
        shots=4096,
        methods=["bernoulli_success_amplitude", "best_scalar_upper_bound"],
        grover_powers=(0, 1),
        seed=5,
    )
    by_method = {row["norm_recovery_method"]: row for row in rows}
    best_scalar = float(by_method["best_scalar_upper_bound"]["residual_after_recovery"])
    bernoulli = float(by_method["bernoulli_success_amplitude"]["residual_after_recovery"])
    assert best_scalar <= bernoulli + 1.0e-9


def test_exact_diagnostic_matches_known_c_magnitude() -> None:
    H, r = _toy()
    recovery = apply_norm_recovery(
        "exact_success_amplitude_diagnostic",
        unit=np.array([0.6, 0.8]),
        direction_norm=0.5,
        scale_factor_C=2.0,
        beta=4.0,
        H=H,
        r=r,
        ridge_update=np.array([0.1, 0.2]),
        shots=128,
        seed=0,
    )
    # magnitude == (C/beta) * direction_norm
    assert np.isclose(recovery.scale_factor, (2.0 / 4.0) * 0.5)
    assert recovery.routine_status == "succeeded"


def test_iterative_qae_alias_and_outputs(tmp_path: Path) -> None:
    H, r = _toy()
    rows = evaluate_norm_recovery_point(
        H=H,
        r=r,
        alpha=1.0e-3,
        degree=5,
        shots=1024,
        methods=["iterative_qae_if_available", "exact_success_amplitude_diagnostic"],
        grover_powers=(0, 1, 2),
        seed=3,
    )
    methods = {row["norm_recovery_method"] for row in rows}
    assert "iterative_qae" in methods  # alias normalized
    qae = next(row for row in rows if row["norm_recovery_method"] == "iterative_qae")
    assert qae["uses_qae"] is True

    artifacts = write_norm_recovery_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)
    summary = pd.read_csv(artifacts["norm_recovery_summary"])
    for column in SUMMARY_COLUMNS:
        assert column in summary.columns
    assert artifacts["scale_factor_comparison"].is_file()
