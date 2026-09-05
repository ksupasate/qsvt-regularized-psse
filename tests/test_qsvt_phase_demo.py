from __future__ import annotations

import json

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.phase_synthesis import qsp_response, synthesize_qsp_phases
from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.qsvt.run_phase_demo import run_phase_demo, validate_phase_demo_config


def test_qsp_response_and_phase_synthesis_are_finite() -> None:
    result = synthesize_qsp_phases(
        alpha=0.05,
        block_encoding_normalization=1.0,
        degree=3,
        domain_min=0.1,
        domain_max=1.0,
        grid_size=32,
        seed=7,
        max_nfev=500,
    )

    response = qsp_response(np.array([0.1, 0.5, 0.9]), result.phases)

    assert result.phases.shape == (4,)
    assert np.all(np.isfinite(result.phases))
    assert np.all(np.isfinite(response))
    assert np.isfinite(result.max_abs_error)


def test_phase_demo_writes_matrix_fallback_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = {
        "demo": {
            "run_id": "test_qsvt_phase_demo",
            "output_dir": str(tmp_path / "qsvt_phase_demo"),
            "alpha": 0.05,
            "degree": 3,
            "domain_min": 0.1,
            "domain_max": 1.0,
            "block_encoding_normalization": 1.0,
            "grid_size": 32,
            "seed": 9,
            "max_nfev": 500,
            "singular_values": [0.1, 0.5, 0.9],
        }
    }

    run = run_phase_demo(config)
    output_dir = run["output_dir"]

    for filename in (
        "config_resolved.yaml",
        "qsvt_demo_config_resolved.yaml",
        "phase_angles.csv",
        "polynomial_coefficients.csv",
        "approximation_error.csv",
        "circuit_summary.json",
        "qsvt_demo_results.csv",
        "qsvt_demo_plot.png",
        "run.log",
    ):
        assert (output_dir / filename).is_file()

    phases = pd.read_csv(output_dir / "phase_angles.csv")
    results = pd.read_csv(output_dir / "qsvt_demo_results.csv")
    with (output_dir / "circuit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert len(phases) == 4
    assert np.all(np.isfinite(results["polynomial_abs_error"]))
    assert "qiskit_available" in summary


def test_improved_odd_polynomial_meets_accuracy_target() -> None:
    approximation = fit_odd_regularized_polynomial(
        alpha=0.01,
        block_encoding_normalization=1.0,
        degree=35,
        domain_min=0.2,
        domain_max=1.0,
        grid_size=4096,
    )

    assert approximation.max_error <= 1.0e-2
    assert approximation.mean_error <= 1.0e-3
    assert approximation.scaled_max_error <= approximation.max_error


def test_phase_demo_validation_rejects_invalid_parameters(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = {
        "demo": {
            "output_dir": str(tmp_path / "bad"),
            "degree": 0,
        }
    }

    try:
        validate_phase_demo_config(config)
    except ValueError as exc:
        assert "degree" in str(exc)
    else:
        raise AssertionError("invalid QSVT phase demo config should fail")
