from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.full_qsvt_ieee_pathway import run_alpha_degree_tradeoff


def test_alpha_degree_sweep_produces_required_csv_columns(tmp_path: Path) -> None:
    run = run_alpha_degree_tradeoff(
        {
            "output_dir": str(tmp_path),
            "case": "synthetic",
            "case_source": "builtin",
            "matrix_source": "synthetic",
            "submatrix_sizes": [2],
            "alphas": [1.0e-3],
            "degrees": [5],
            "seed": 123,
        }
    )

    summary_path = run["artifacts"]["alpha_degree_summary"]
    observable_path = run["artifacts"]["alpha_degree_observable_summary"]
    resource_path = run["artifacts"]["alpha_degree_resource_summary"]

    assert summary_path.is_file()
    assert observable_path.is_file()
    assert resource_path.is_file()

    summary = pd.read_csv(summary_path)
    required = {
        "alpha",
        "requested_degree",
        "synthesized_degree",
        "phase_count",
        "maximum_pointwise_filter_error",
        "qsvt_state_error_vs_ridge",
        "normalized_state_overlap",
        "observable_max_error",
        "success_probability_proxy",
        "query_count_estimate",
        "phase_synthesis_status",
    }
    assert required.issubset(summary.columns)
    assert len(summary) == 1
