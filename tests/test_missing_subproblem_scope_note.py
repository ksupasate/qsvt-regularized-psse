from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.paper.full_vector_readout import (
    _missing_subproblem_scope_markdown,
    run_full_vector_readout,
)

_DISCOVERY = (
    "case,subproblem_id,selection_mode,alpha,degree,state_error_gate_vs_ridge,"
    "row_indices,col_indices\n"
    "ieee14,high_leverage_00,high_leverage,0.001,15,0.0222,17 31 48 68,2 3 16 17\n"
)


def _write_discovery(input_root: Path) -> None:
    source = input_root / "qsvt_cross_case_solver_prototype"
    source.mkdir(parents=True)
    (source / "cross_case_gate_validated_results.csv").write_text(_DISCOVERY, encoding="utf-8")


def test_scope_note_lists_available_and_missing() -> None:
    evaluated = [{"case": "ieee14", "subproblem_type": "high_leverage"}]
    missing = [
        {"case": "ieee57", "subproblem_type": "metadata_mapped"},
        {"case": "ieee57", "subproblem_type": "residual_supported"},
    ]
    text = _missing_subproblem_scope_markdown(evaluated, missing)
    assert "ieee14/high_leverage" in text
    assert "ieee57/metadata_mapped: missing_phase_data" in text
    assert "not** readout failures" in text or "not readout failures" in text
    assert "excluded from the readout success denominator" in text


def test_missing_recorded_and_excluded_from_denominator(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    _write_discovery(input_root)
    run = run_full_vector_readout(
        {
            "input_root": str(input_root),
            "output_dir": str(tmp_path / "fvr"),
            "cases": ["ieee14", "ieee30", "ieee57", "ieee118"],
            "subproblem_types": ["high_leverage", "metadata_mapped", "residual_supported"],
            "alpha": 1.0e-4,
            "shots": [500],
            "seed": 1,
            "trials": 2,
        }
    )
    out = Path(run["output_dir"])
    # Only ieee14/high_leverage is available; the success denominator is the 1 available.
    assert run["subproblem_count"] == 1
    assert len(run["missing"]) == 11
    note = (out / "missing_subproblem_scope_note.md").read_text("utf-8")
    assert "ieee14/high_leverage" in note
    missing_csv = (out / "missing_full_vector_readout_cases.csv").read_text("utf-8")
    assert "missing_phase_data" in missing_csv
