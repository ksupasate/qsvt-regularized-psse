from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import run_preconditioned_variant_sweeps


def test_preconditioned_variant_sweeps_label_variants_separately(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = run_preconditioned_variant_sweeps(
        {
            "output_dir": str(tmp_path / "sweeps"),
            "cases": ["synthetic"],
            "alphas": [1.0e-2],
            "noise_stds": [0.0],
            "missing_ratios": [0.0],
            "bad_data_ratios": [0.0],
            "seeds": [10],
            "degree": 15,
            "grid_size": 64,
            "fallback_to_synthetic": True,
        }
    )
    output_dir = run["output_dir"]
    results = pd.read_csv(output_dir / "preconditioned_variant_sweep_results.csv")

    required_variants = {
        "unpreconditioned_ridge",
        "preconditioned_coordinate_ridge",
        "preconditioned_transformed_penalty_ridge",
        "unpreconditioned_qsvt_diagnostic",
        "preconditioned_qsvt_diagnostic",
    }
    assert required_variants.issubset(set(results["variant_name"]))
    assert not results["variant_name"].str.contains("coordinate.*transformed").any()
    assert np.isfinite(results["qsvt_full_interval_approx_error"]).all()
    assert (output_dir / "preconditioned_variant_manifest.json").is_file()
    assert (output_dir / "preconditioned_variant_failure_log.csv").is_file()
