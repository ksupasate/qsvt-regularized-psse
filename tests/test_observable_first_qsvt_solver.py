from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_qsvt_se.qsvt.gate_level_state_estimation_solver import SelectedSubproblem
from robust_qsvt_se.qsvt.observable_first_qsvt_solver import (
    SUMMARY_COLUMNS,
    evaluate_observable_first_grid,
    write_observable_first_outputs,
)


def _subproblem() -> SelectedSubproblem:
    return SelectedSubproblem(
        H_tilde=np.array(
            [
                [1.0, 0.2, 0.1, 0.0],
                [0.15, 0.8, 0.0, 0.1],
                [0.1, 0.0, 0.9, 0.2],
                [0.0, 0.1, 0.2, 0.7],
            ],
            dtype=np.float64,
        ),
        r_tilde=np.array([0.4, -0.2, 0.3, 0.1], dtype=np.float64),
        metadata={
            "case": "toy",
            "state_index_mapping": [
                {"local_index": 0, "source_index": 0, "label": "theta_2", "state_type": "angle"},
                {"local_index": 1, "source_index": 1, "label": "theta_3", "state_type": "angle"},
                {"local_index": 2, "source_index": 2, "label": "V_2", "state_type": "voltage"},
                {"local_index": 3, "source_index": 3, "label": "V_3", "state_type": "voltage"},
            ],
        },
    )


def _rows() -> list[dict]:
    return evaluate_observable_first_grid(
        subproblem=_subproblem(),
        alphas=[1.0e-4],
        degrees=[5],
        shot_levels=[2000],
        target_tolerances=[1.0e-1],
        topk=2,
        seed=11,
    )


def test_practical_observables_do_not_require_full_vector_readout() -> None:
    rows = _rows()
    practical = [row for row in rows if row["practical_for_observable_first_solver"]]
    assert practical
    for row in practical:
        assert row["requires_full_vector_readout"] is False


def test_full_vector_observable_is_flagged_and_not_practical() -> None:
    rows = _rows()
    full_vector = [row for row in rows if row["requires_full_vector_readout"]]
    assert full_vector
    for row in full_vector:
        assert row["practical_for_observable_first_solver"] is False


def test_topk_observable_needs_no_norm_recovery() -> None:
    rows = _rows()
    topk = [row for row in rows if "topk" in row["observable_name"]]
    assert topk
    for row in topk:
        assert row["requires_norm_recovery"] is False
        assert row["norm_recovery_method"] == "none"


def test_outputs_have_required_columns(tmp_path: Path) -> None:
    rows = _rows()
    artifacts = write_observable_first_outputs(tmp_path, {"output_dir": str(tmp_path)}, rows)
    summary = pd.read_csv(artifacts["observable_first_solver_summary"])
    for column in SUMMARY_COLUMNS:
        assert column in summary.columns
    assert artifacts["observable_first_accuracy_cost"].is_file()
    assert artifacts["observable_first_interpretation"].is_file()
