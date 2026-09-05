from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.subproblem_selection_policy import (
    run_qsvt_subproblem_selection_policy,
)


def test_subproblem_selection_policy_is_deterministic(tmp_path: Path) -> None:
    config = {
        "output_dir": str(tmp_path / "first"),
        "candidate_modes": ["best_conditioned", "high_leverage", "random_seeded_pool"],
        "submatrix_size": 4,
        "seed": 123,
    }
    first = run_qsvt_subproblem_selection_policy(config)
    second = run_qsvt_subproblem_selection_policy(
        {**config, "output_dir": str(tmp_path / "second")}
    )

    first_rows = pd.read_csv(first["artifacts"]["candidate_subproblem_scores"])
    second_rows = pd.read_csv(second["artifacts"]["candidate_subproblem_scores"])

    assert first_rows["row_indices"].tolist() == second_rows["row_indices"].tolist()
    assert first_rows["col_indices"].tolist() == second_rows["col_indices"].tolist()


def test_subproblem_selection_policy_has_rejection_reasons_and_no_qsvt_score(
    tmp_path: Path,
) -> None:
    run = run_qsvt_subproblem_selection_policy(
        {
            "output_dir": str(tmp_path),
            "candidate_modes": ["high_leverage", "worst_conditioned_control"],
            "submatrix_size": 4,
            "seed": 123,
        }
    )
    candidates = pd.read_csv(run["artifacts"]["candidate_subproblem_scores"])
    rejected = pd.read_csv(run["artifacts"]["rejected_subproblems"])
    policy_text = run["artifacts"]["selection_policy"].read_text(encoding="utf-8")

    assert "qsvt_residual" not in candidates.columns
    assert "QSVT residuals are not used" in policy_text
    assert rejected["rejection_reason"].astype(str).str.len().gt(0).any()
