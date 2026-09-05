from __future__ import annotations

from pathlib import Path

from robust_qsvt_se.qsvt.cross_case_solver_audit import (
    CROSS_CASE_SELECTION_MODES,
    reusable_pipeline_components,
    run_qsvt_cross_case_solver_audit,
)


def test_audit_identifies_reusable_cross_case_components() -> None:
    components = reusable_pipeline_components()
    cross_case = components[components["supports_cross_case"].astype(bool)]
    names = set(cross_case["component"].astype(str))

    # The case-routing entry point and the criteria-based selection must be reusable.
    assert "_build_system" in names
    assert "generate_candidate_subproblems" in names
    assert "build_codesigned_solution" in names
    # No component should claim cross-case support without a reuse plan.
    assert (cross_case["reuse_for_cross_case"].astype(str).str.len() > 0).all()


def test_audit_writes_outputs_and_marks_cross_case_cases(tmp_path: Path) -> None:
    run = run_qsvt_cross_case_solver_audit(
        {
            "input_root": str(tmp_path / "outputs"),
            "cases": ["ieee14", "ieee30", "ieee57"],
            "output_dir": str(tmp_path / "audit"),
        }
    )
    for name in [
        "manifest",
        "reusable_pipeline_components",
        "cross_case_candidate_cases",
        "cross_case_audit",
    ]:
        assert run["artifacts"][name].is_file()

    candidates = run["candidates"]
    roles = set(candidates["role"].astype(str))
    assert "reference" in roles
    assert "cross_case_test" in roles
    # IEEE30/IEEE57 are labelled as cross-case tests, not the reference case.
    cross = candidates[candidates["role"].astype(str) == "cross_case_test"]
    assert set(cross["case"].astype(str)) == {"ieee30", "ieee57"}


def test_audit_preserves_criteria_only_selection_modes() -> None:
    # The control mode is retained for honest failure reporting but stays criteria-based.
    assert "worst_conditioned_control" in CROSS_CASE_SELECTION_MODES
    assert "high_leverage" in CROSS_CASE_SELECTION_MODES
    assert "metadata_mapped" in CROSS_CASE_SELECTION_MODES
