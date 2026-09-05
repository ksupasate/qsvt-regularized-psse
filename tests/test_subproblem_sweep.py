from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_qsvt_se.qsvt.subproblem_sweep import (
    _build_system,
    run_gate_level_qsvt_subproblem_sweep,
    select_subproblem,
)


def test_subproblem_selection_is_deterministic_with_fixed_seed() -> None:
    system, matrix_source = _build_system(
        case="ieee14",
        model="ac_linearized",
        case_source="pypower",
        seed=123,
    )
    first = select_subproblem(
        system=system,
        matrix_source=matrix_source,
        case="ieee14",
        model="ac_linearized",
        submatrix_size=4,
        selection_mode="random_seeded",
        seed=123,
    )
    second = select_subproblem(
        system=system,
        matrix_source=matrix_source,
        case="ieee14",
        model="ac_linearized",
        submatrix_size=4,
        selection_mode="random_seeded",
        seed=123,
    )

    np.testing.assert_allclose(first.H_tilde, second.H_tilde)
    np.testing.assert_allclose(first.r_tilde, second.r_tilde)


def test_subproblem_sweep_writes_outputs(tmp_path: Path) -> None:
    run = run_gate_level_qsvt_subproblem_sweep(
        {
            "output_dir": str(tmp_path),
            "selection_modes": ["high_leverage"],
            "degree": 9,
            "shots": 100,
            "seed": 123,
        }
    )

    assert len(run["summary_rows"]) == 1
    for name in [
        "manifest",
        "subproblem_summary",
        "residual_vs_conditioning",
        "state_error_vs_conditioning",
        "best_and_worst_cases",
        "subproblem_sweep_interpretation",
    ]:
        assert run["artifacts"][name].is_file()
