from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robust_qsvt_se.paper.degree_aware_alpha import run_degree_aware_alpha_selection
from robust_qsvt_se.paper.integrated_small_selected_readout import (
    run_integrated_small_selected_readout,
)
from robust_qsvt_se.paper.selected_observable_workload import (
    run_selected_observable_workload,
)


def test_main_observables_are_selected_before_solving(tmp_path: Path) -> None:
    run = run_selected_observable_workload(
        {"output_dir": str(tmp_path), "cases": ["ieee14"], "shots": [100], "trials": 2}
    )
    frame = run["observables"]
    assert bool(frame["selected_before_solving"].all())
    assert bool((~frame["depends_on_solution"]).all())
    assert not frame["observable_id"].str.contains("topk|top_", case=False).any()


def test_ieee57_readout_target_miss_is_retained(tmp_path: Path) -> None:
    run = run_selected_observable_workload(
        {
            "output_dir": str(tmp_path),
            "cases": ["ieee57"],
            "shots": [100, 1_000, 10_000, 100_000],
            "trials": 200,
            "base_seed": 20240601,
        }
    )
    branch = run["shot_sweep"][
        run["shot_sweep"]["observable_id"].str.startswith("dtheta_diff_")
        & (run["shot_sweep"]["shots"] == 100_000)
    ]
    assert len(branch) == 1
    assert bool(branch.iloc[0]["target_met"]) is False
    assert branch.iloc[0]["target_status"] == "target_miss"
    assert float(branch.iloc[0]["relative_error"]) == pytest.approx(0.05589030178617602)


def test_revised_degree_rows_do_not_label_counts_realizable_without_phases(
    tmp_path: Path,
) -> None:
    run = run_degree_aware_alpha_selection(
        {"output_dir": str(tmp_path), "cases": ["ieee14"], "command": "test"}
    )
    revised = run["revised_summary"]
    assert {
        "spectrum_point_degree",
        "uniform_grid_degree",
        "phase_synthesis_available",
        "boundedness_verified",
        "parity_verified",
        "qsvt_query_count_realizable",
    }.issubset(revised.columns)
    assert bool((~revised["phase_synthesis_available"]).all())
    assert bool((~revised["qsvt_query_count_realizable"]).all())
    assert (revised["query_count_status"] == "spectrum_point_action_only").all()


def test_sparse_oracle_spec_has_required_interface_terms() -> None:
    text = Path("docs/SPARSE_ORACLE_BLOCK_ENCODING_SPEC.md").read_text(encoding="utf-8")
    for term in (
        "XOR c(i,k)",
        "XOR fix(H_tilde[i,j])",
        "Invalid-index behavior",
        "two's-complement fixed point",
        "Reversible `O_col`, `O_val` circuit",
        "excluded",
    ):
        assert term in text


def test_small_demo_is_predetermined_and_records_scaling(tmp_path: Path) -> None:
    run = run_integrated_small_selected_readout({"output_dir": str(tmp_path)})
    row = run["demo"].iloc[0]
    assert bool(row["selected_before_solving"])
    assert not bool(row["depends_on_solution"])
    assert row["phase_synthesis_status"] == "completed"
    assert row["qsvt_sequence_status"] == "completed"
    assert float(row["residual_norm"]) > 0.0
    assert 0.0 < float(row["postselection_success_probability"]) <= 1.0
    assert np.isfinite(float(row["qsvt_circuit_physical_observable"]))
    assert run["artifact"].is_file()


