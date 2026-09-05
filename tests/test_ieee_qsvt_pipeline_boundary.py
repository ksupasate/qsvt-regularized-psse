"""IEEE-derived quantum-pipeline boundary study: provenance, validation, and claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.experiments.tqe_revision_evidence import select_deterministic_block
from robust_qsvt_se.paper.ieee_qsvt_pipeline_boundary import (
    EQUIVALENCE_RELATIVE_TOLERANCE,
    SELECTION_POLICY,
    build_case_system,
    build_ieee_qsvt_pipeline_boundary,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

CASE = "ieee14"
SEED = 123
SIZES = [4, 8]


@pytest.fixture(scope="module")
def boundary_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output_dir = tmp_path_factory.mktemp("ieee_qsvt_pipeline_boundary")
    return build_ieee_qsvt_pipeline_boundary(
        {
            "output_dir": str(output_dir),
            "seed": SEED,
            "cases": [[CASE, SIZES]],
            "command": "pytest",
        }
    )


def _frame(boundary_run: dict[str, Any], name: str) -> pd.DataFrame:
    path = Path(boundary_run["output_dir"]) / name
    assert path.is_file(), f"missing required artifact {name}"
    return pd.read_csv(path)


def test_required_artifacts_exist(boundary_run: dict[str, Any]) -> None:
    output_dir = Path(boundary_run["output_dir"])
    for name in [
        "manifest.json",
        "selected_workload_summary.csv",
        "block_encoding_report.csv",
        "state_preparation_report.csv",
        "qsvt_target_equivalence_report.csv",
        "readout_report.csv",
        "complexity_report.csv",
        "summary.md",
    ]:
        assert (output_dir / name).is_file(), f"missing required artifact {name}"


def test_selected_block_is_ieee_derived_not_random(boundary_run: dict[str, Any]) -> None:
    """The stored blocks must be exact submatrices of the regenerated IEEE system."""

    system, matrix_source = build_case_system(CASE, seed=SEED)
    H_full = np.asarray(system.H_tilde, dtype=np.float64)
    r_full = np.asarray(system.r_tilde, dtype=np.float64)
    assert system.metadata.get("external_case") is True
    assert "pypower" in matrix_source

    output_dir = Path(boundary_run["output_dir"])
    for size in SIZES:
        expected_H, expected_r, _rows, _cols = select_deterministic_block(
            H_full, r_full, row_count=size, col_count=size, policy=SELECTION_POLICY
        )
        stored_H = np.load(output_dir / f"selected_block_{CASE}_{size}x{size}.npy")
        stored_r = np.load(output_dir / f"selected_residual_{CASE}_{size}x{size}.npy")
        np.testing.assert_array_equal(stored_H, expected_H)
        np.testing.assert_array_equal(stored_r, expected_r)

    workload = _frame(boundary_run, "selected_workload_summary.csv")
    assert workload["rows_generated_from_ieee_pypower"].all()
    assert workload["measurement_row_provenance"].str.contains("IEEE/PYPOWER", regex=False).all()
    assert workload["selection_policy"].eq(SELECTION_POLICY).all()


def test_block_encoding_reconstruction_below_tolerance(boundary_run: dict[str, Any]) -> None:
    report = _frame(boundary_run, "block_encoding_report.csv")
    executable = report[report["path"] == "selected_block_executable"]
    assert len(executable) == len(SIZES)
    assert executable["implementation_status"].eq("implemented").all()
    assert (executable["top_left_block_error"] <= 1.0e-7).all()
    assert (executable["unitarity_error"] <= 1.0e-7).all()
    assert executable["validation_passed"].all()

    modeled = report[report["path"] == "full_matrix_modeled"]
    assert modeled["implementation_status"].eq("modeled").all()
    assert modeled["implementation_detail"].str.contains("cost model", regex=False).all()


def test_residual_state_preparation_fidelity(boundary_run: dict[str, Any]) -> None:
    report = _frame(boundary_run, "state_preparation_report.csv")
    executable = report[report["path"] == "selected_block_executable"]
    implemented = executable[executable["implementation_status"] == "implemented"]
    assert not implemented.empty, "no implemented state-preparation rows"
    assert (implemented["state_preparation_fidelity"] >= 1.0 - 1.0e-9).all()
    assert (implemented["state_preparation_l2_error"] <= 1.0e-7).all()
    assert (executable["residual_norm"] > 0.0).all()

    modeled = report[report["path"] == "full_matrix_modeled"]
    assert modeled["implementation_status"].eq("modeled").all()


def test_matched_alpha_qsvt_target_ridge_equivalence(boundary_run: dict[str, Any]) -> None:
    report = _frame(boundary_run, "qsvt_target_equivalence_report.csv")
    assert len(report) == len(SIZES)
    assert report["equivalence_status"].eq("pass").all()
    assert (report["relative_difference"] <= EQUIVALENCE_RELATIVE_TOLERANCE).all()
    assert report["alpha"].nunique() == 1, "alpha must not be tuned per block"
    assert (
        report["framing"].str.contains("not a comparison of estimator quality", regex=False).all()
    )


def test_readout_report_states_tomography_non_claim(boundary_run: dict[str, Any]) -> None:
    report = _frame(boundary_run, "readout_report.csv")
    assert not report.empty
    non_claim = report["full_vector_tomography_non_claim"]
    assert non_claim.str.contains("tomography", regex=False).all()
    assert non_claim.str.contains("not claimed", regex=False).all()
    assert report["readout_status"].isin(["simulated", "modeled"]).all()
    assert (report["overlap_a"].abs() <= 1.0 + 1.0e-12).all()
    assert (report["shots_for_abs_precision_1e-3"] >= report["shots_for_abs_precision_1e-2"]).all()


def test_complexity_report_covers_all_cost_terms(boundary_run: dict[str, Any]) -> None:
    report = _frame(boundary_run, "complexity_report.csv")
    for column in [
        "T_BE_proxy",
        "T_prep_proxy",
        "T_readout_proxy",
        "qsvt_degree_d",
        "condition_number_kappa",
        "postselection_success_proxy",
        "q_selected_observables",
        "total_query_proxy",
        "classical_dense_svd_proxy_flops",
        "classical_normal_equations_proxy_flops",
    ]:
        assert column in report.columns, f"complexity report missing {column}"
    assert set(report["path"]) == {"selected_block_executable", "full_matrix_modeled"}
    assert report["T_QSVT_formula"].str.contains("T_prep + d * T_BE + T_readout", regex=False).all()
    assert report["T_readout_formula"].str.contains("O(q/epsilon^2)", regex=False).all()
    assert (report["classical_normal_equations_proxy_flops"] > 0).all()
    assert report["asymptotic_comparison_note"].str.contains("no asymptotic claim").all()


def test_generated_text_respects_claim_boundary(boundary_run: dict[str, Any]) -> None:
    output_dir = Path(boundary_run["output_dir"])
    for path in [
        *sorted(output_dir.glob("*.csv")),
        output_dir / "summary.md",
        output_dir / "manifest.json",
    ]:
        violations = forbidden_in(path.read_text(encoding="utf-8"))
        assert not violations, f"{path.name} contains forbidden wording: {violations}"
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    for heading in [
        "## What was implemented",
        "## What was simulated",
        "## What was modeled",
        "## What was not implemented",
        "## Safe manuscript wording",
        "## Claims to avoid",
    ]:
        assert heading in summary
