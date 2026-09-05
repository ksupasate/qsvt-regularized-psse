from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.success_amplification_cost import (
    bottleneck_severity,
    cost_row,
    run_success_amplification_cost_study,
)
from robust_qsvt_se.utils.io import write_json


def test_success_amplification_cost_handles_small_probabilities() -> None:
    row = cost_row("unit", 1.0e-8, 10)

    assert row["bottleneck_severity"] == "severe"
    assert row["postselection_cost"] == 1.0e8
    assert row["amplified_qsvt_query_proxy"] > 0.0
    assert bottleneck_severity(1.0e-7) == "severe"


def test_success_amplification_cost_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    write_json(
        input_dir / "summary.json",
        {"success_probability": 0.25, "qsvt_query_count": 35, "alpha": 1.0e-4, "degree": 9},
    )

    run = run_success_amplification_cost_study(
        {"input_dirs": [str(input_dir)], "output_dir": str(tmp_path / "out")}
    )

    assert len(run["rows"]) == 1
    for name in [
        "manifest",
        "success_probability_sweep",
        "postselection_cost",
        "amplitude_amplification_cost",
        "alpha_degree_success_tradeoff",
        "success_bottleneck_interpretation",
    ]:
        assert run["artifacts"][name].is_file()
