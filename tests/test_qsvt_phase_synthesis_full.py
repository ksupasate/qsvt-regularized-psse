from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.polynomial import fit_odd_regularized_polynomial
from robust_qsvt_se.qsvt.run_phase_demo import run_phase_demo


def test_degree35_scaled_polynomial_is_bounded_and_accurate() -> None:
    approximation = fit_odd_regularized_polynomial(
        alpha=0.01,
        block_encoding_normalization=1.0,
        degree=35,
        domain_min=0.2,
        domain_max=1.0,
        grid_size=4096,
    )
    polynomial = approximation.polynomial / approximation.scale_factor
    grid = np.linspace(-1.0, 1.0, 4096)

    assert np.max(np.abs(polynomial(grid))) <= 1.0 + 1.0e-10
    assert approximation.max_error <= 1.0e-2
    assert approximation.mean_error <= 1.0e-3


def test_pennylane_phase_synthesis_demo_writes_validated_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_phase_demo(
        {
            "demo": {
                "run_id": "test_qsvt_phase_synthesis",
                "output_dir": str(tmp_path / "phase"),
                "alpha": 0.05,
                "degree": 5,
                "domain_min": 0.2,
                "domain_max": 1.0,
                "block_encoding_normalization": 1.0,
                "grid_size": 256,
                "phase_validation_grid_size": 17,
                "phase_synthesis_method": "pennylane_poly_to_angles",
                "angle_solver": "iterative",
                "singular_values": [0.2, 0.5, 0.8],
            }
        }
    )
    output_dir = run["output_dir"]
    phases = pd.read_csv(output_dir / "phase_angles.csv")
    validation = pd.read_csv(output_dir / "phase_implemented_error.csv")
    with (output_dir / "circuit_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert len(phases) == 6
    assert np.all(np.isfinite(phases["phase_angle"]))
    assert not np.allclose(phases["phase_angle"], 0.0)
    assert np.isfinite(validation["phase_abs_error"]).all()
    assert summary["phase_synthesis_method"] == "pennylane_poly_to_angles"
    assert summary["phase_synthesis_success"] is True
    assert "phase_implemented_max_abs_error" in summary
