from __future__ import annotations

import json

import numpy as np
import pytest

from robust_qsvt_se.qsvt.run_phase_demo import run_phase_demo, validate_phase_demo_config


def test_phase_validation_rejects_required_none_method(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="phase_synthesis_method=none"):
        validate_phase_demo_config(
            {
                "demo": {
                    "output_dir": str(tmp_path / "bad"),
                    "phase_synthesis_method": "none",
                    "require_phase_validation": True,
                }
            }
        )


def test_phase_validation_report_passes_known_good_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pennylane")
    run = run_phase_demo(
        {
            "demo": {
                "run_id": "test_phase_validation",
                "output_dir": str(tmp_path / "phase_validation"),
                "alpha": 0.5,
                "degree": 9,
                "domain_min": 0.2,
                "domain_max": 1.0,
                "grid_size": 1024,
                "phase_validation_grid_size": 65,
                "phase_synthesis_method": "pennylane_poly_to_angles",
                "angle_solver": "iterative",
                "phase_cache_dir": str(tmp_path / "cache"),
                "require_phase_validation": True,
                "singular_values": [0.2, 0.5, 0.8],
            }
        }
    )
    report_path = run["output_dir"] / "phase_validation_report.json"
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["validation_passed"] is True
    assert report["parity_check_passed"] is True
    assert report["boundedness_check_passed"] is True
    assert report["dummy_phase_check_passed"] is True
    assert np.isfinite(report["max_phase_implemented_error"])
    assert (run["output_dir"] / "phase_validation_plot.png").is_file()
