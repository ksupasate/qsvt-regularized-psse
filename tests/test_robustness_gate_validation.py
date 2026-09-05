from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.robustness_gate_validation import (
    GATE_COLUMNS,
    select_robustness_gate_candidates,
    write_robustness_gate_outputs,
)


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subproblem_id": "high_leverage_00",
                "selection_mode": "high_leverage",
                "row_indices": "0 1 2 3",
                "col_indices": "0 1 2 3",
                "alpha": 1.0e-3,
                "degree": 25,
                "target_family": "residual_aware",
                "condition_number": 7.6,
                "residual_ratio_vs_no_update": 0.0002,
            },
            {
                "subproblem_id": "metadata_mapped_02",
                "selection_mode": "metadata_mapped",
                "row_indices": "1 2 3 4",
                "col_indices": "1 2 3 4",
                "alpha": 1.0e-4,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 18.4,
                "residual_ratio_vs_no_update": 0.007,
            },
            {
                "subproblem_id": "residual_supported_03",
                "selection_mode": "residual_supported",
                "row_indices": "2 3 4 5",
                "col_indices": "2 3 4 5",
                "alpha": 1.0e-2,
                "degree": 35,
                "target_family": "residual_aware",
                "condition_number": 8.7,
                "residual_ratio_vs_no_update": 0.0004,
            },
            {
                "subproblem_id": "worst_conditioned_control_09",
                "selection_mode": "worst_conditioned_control",
                "row_indices": "5 6 7 8",
                "col_indices": "5 6 7 8",
                "alpha": 1.0e-4,
                "degree": 25,
                "target_family": "residual_aware",
                "condition_number": 1.0e17,
                "residual_ratio_vs_no_update": 0.0001,
            },
        ]
    )


def test_select_excludes_control_and_prefers_diverse_candidates() -> None:
    selected = select_robustness_gate_candidates(_candidate_frame(), max_configs=3)
    modes = set(selected["selection_mode"].astype(str))

    assert "worst_conditioned_control" not in modes
    assert len(selected) <= 3
    # A metadata-mapped and a non-high-leverage candidate are represented.
    assert "metadata_mapped" in modes
    assert any(mode != "high_leverage" for mode in modes)


def test_select_returns_empty_for_empty_input() -> None:
    selected = select_robustness_gate_candidates(pd.DataFrame(), max_configs=3)
    assert selected.empty


def test_write_outputs_when_no_results(tmp_path) -> None:
    selected = select_robustness_gate_candidates(pd.DataFrame(), max_configs=3)
    artifacts = write_robustness_gate_outputs(tmp_path, {"output_dir": str(tmp_path)}, selected, [])
    assert artifacts["robustness_gate_results"].is_file()
    results = pd.read_csv(artifacts["robustness_gate_results"])
    assert results.empty or set(GATE_COLUMNS).issubset(set(results.columns))
