from __future__ import annotations

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import build_preconditioning_resource_comparison


def test_preconditioning_resource_comparison_outputs_required_columns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = build_preconditioning_resource_comparison(
        {
            "output_dir": str(tmp_path / "resource"),
            "cases": ["synthetic"],
            "alphas": [1.0e-2],
            "degree": 15,
            "grid_size": 64,
            "method": "odd_chebyshev_ls",
            "fallback_to_synthetic": True,
        }
    )
    output_dir = run["output_dir"]
    frame = pd.read_csv(output_dir / "preconditioning_resource_comparison.csv")
    required = {
        "case_name",
        "variant",
        "sigma_min",
        "sigma_max",
        "kappa",
        "rank",
        "max_filter_gain",
        "bounded_scaling_C",
        "degree_used",
        "query_count",
        "full_interval_approx_error",
        "actual_singular_value_approx_error",
        "logical_qubits_proxy",
        "depth_proxy",
        "readout_caveat",
        "oracle_caveat",
    }
    assert required.issubset(frame.columns)
    assert {"unpreconditioned", "column_equilibrated"}.issubset(set(frame["variant"]))
    assert np.isfinite(frame["full_interval_approx_error"]).all()
    assert (output_dir / "manifest.json").is_file()
