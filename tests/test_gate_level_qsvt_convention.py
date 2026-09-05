from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.gate_level_qsvt_convention import (
    CORRECT_STATE_EXTRACTION_RULE,
    run_gate_level_qsvt_convention_debug,
)


def test_gate_level_qsvt_convention_debug_writes_required_outputs(tmp_path: Path) -> None:
    run = run_gate_level_qsvt_convention_debug(tmp_path)
    summary = run["summary"]

    assert summary["scalar_test_error"] < 1.0e-10
    assert summary["diagonal_test_error"] < 1.0e-10
    assert summary["best_extraction_rule"] in {
        CORRECT_STATE_EXTRACTION_RULE,
        "best_sign_aligned_real_prefix_state",
    }
    assert summary["qiskit_vs_pennylane_operator_error"] < 1.0e-10
    for name in [
        "manifest",
        "convention_debug_summary",
        "scalar_diagonal_tests",
        "operator_block_errors",
        "state_extraction_errors",
        "qiskit_pennylane_operator_comparison",
    ]:
        assert run["artifacts"][name].is_file()
