from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.accuracy_success_tradeoff import (
    mark_pareto_and_classify,
    run_accuracy_success_tradeoff,
)


def test_accuracy_success_tradeoff_computes_pareto_frontier() -> None:
    rows = [
        {
            "subproblem_id": "a",
            "alpha": 1.0e-4,
            "degree": 3,
            "best_scalar_residual": 0.1,
            "residual_ratio_vs_no_update": 0.1,
            "residual_ratio_vs_ridge_if_defined": 10.0,
            "success_probability": 0.5,
            "postselection_cost_proxy": 2.0,
            "amplitude_amplification_cost_proxy": 2.0,
            "qsvt_query_count": 3.0,
            "amplified_query_cost_proxy": 6.0,
            "pareto_optimal": False,
            "tradeoff_classification": "unclassified",
        },
        {
            "subproblem_id": "b",
            "alpha": 1.0e-4,
            "degree": 3,
            "best_scalar_residual": 0.2,
            "residual_ratio_vs_no_update": 0.2,
            "residual_ratio_vs_ridge_if_defined": 20.0,
            "success_probability": 0.4,
            "postselection_cost_proxy": 2.5,
            "amplitude_amplification_cost_proxy": 2.5,
            "qsvt_query_count": 3.0,
            "amplified_query_cost_proxy": 7.5,
            "pareto_optimal": False,
            "tradeoff_classification": "unclassified",
        },
    ]

    marked = mark_pareto_and_classify(rows)

    assert marked[0]["pareto_optimal"] is True
    assert marked[1]["tradeoff_classification"] == "dominated_configuration"


def test_accuracy_success_tradeoff_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {
                "subproblem_id": "toy",
                "alpha": 1.0e-4,
                "requested_degree": 3,
                "residual_qsvt_best_scalar": 0.1,
                "residual_ratio_best_scalar_vs_no_update": 0.1,
                "residual_ridge": 0.01,
                "success_probability": 0.5,
                "qsvt_query_count": 3,
            }
        ]
    ).to_csv(input_dir / "alpha_degree_refinement_summary.csv", index=False)

    run = run_accuracy_success_tradeoff(
        {"input_dirs": [str(input_dir)], "output_dir": str(tmp_path / "out")}
    )

    assert len(run["rows"]) == 1
    assert run["artifacts"]["pareto_frontier"].is_file()
