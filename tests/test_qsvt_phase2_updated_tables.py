from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import build_paper_ready_qsvt_tables


def test_phase2_updated_tables_include_required_rows(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_table_sources(tmp_path)
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "outputs/paper_ready_qsvt_tables"
    build_paper_ready_qsvt_tables({"output_dir": str(output_dir)})

    table5 = pd.read_csv(output_dir / "table_5_qsvt_approximation_by_case.csv")
    table6 = pd.read_csv(output_dir / "table_6_phase_validation_status.csv")
    table7 = pd.read_csv(output_dir / "table_7_ieee300_preconditioning_summary.csv")
    table8 = pd.read_csv(output_dir / "table_8_resource_readout_summary.csv")
    table9 = pd.read_csv(output_dir / "table_9_claim_boundary_matrix.csv")

    table5_text = " ".join(table5.astype(str).stack().tolist())
    assert "IEEE118" in table5_text
    assert "IEEE300" in table5_text
    assert "coordinate-preconditioned Ridge" in table5_text
    assert "transformed-penalty preconditioned" in table5_text
    assert "preconditioned QSVT diagnostic" in table5_text

    pyqsp = table6[table6["target"] == "bounded_ridge_tikhonov_pyqsp"].iloc[0]
    assert pyqsp["status"] == "passed_scalar_full_domain"
    assert pyqsp["backend"] == "pyqsp_sym_qsp"
    assert float(pyqsp["full_domain_max_error"]) == 4.668e-4

    assert {
        "original Ridge",
        "coordinate-preconditioned Ridge",
        "transformed-penalty Ridge",
        "original QSVT diagnostic",
        "preconditioned QSVT diagnostic",
    }.issubset(set(table7["variant"]))
    assert int(table8[table8["backend"] == "pyqsp_sym_qsp"]["phase_count"].iloc[0]) == 202
    assert table9["claim"].str.contains("Alpha selection is diagnostic").any()


def _write_table_sources(root: Path) -> None:
    outputs = root / "outputs"
    (outputs / "qsvt_phase2_complete_summary").mkdir(parents=True)
    (outputs / "qsvt_phase2_preconditioned_alpha_sweeps").mkdir(parents=True)
    (outputs / "qsvt_external_backend_phase_validation").mkdir(parents=True)
    (outputs / "qsvt_preconditioning_resource_comparison").mkdir(parents=True)
    (outputs / "qsvt_engineering_extension").mkdir(parents=True)

    pd.DataFrame(_complete_rows()).to_csv(
        outputs / "qsvt_phase2_complete_summary/phase2_complete_summary.csv",
        index=False,
    )
    pd.DataFrame(_sweep_summary_rows()).to_csv(
        outputs / "qsvt_phase2_preconditioned_alpha_sweeps/phase2_sweep_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "backend_name": "pyqsp_sym_qsp",
                "candidate_name": "coefficient_conditioned_chebyshev_degree_201_lambda_1e-04",
                "degree": 201,
                "phase_count": 202,
                "phase_response_max_error_full_domain": 4.668135e-4,
                "phase_response_max_error_actual_singular_values_if_available": 8.673e-5,
                "status": "passed",
            }
        ]
    ).to_csv(
        outputs
        / "qsvt_external_backend_phase_validation"
        / "external_backend_phase_validation_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "case_name": "ieee300",
                "variant": "column_equilibrated",
                "degree_used": 201,
                "query_count": 403,
                "logical_qubits_proxy": 12,
                "depth_proxy": 1000,
            }
        ]
    ).to_csv(
        outputs
        / "qsvt_preconditioning_resource_comparison"
        / "preconditioning_resource_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "claim": "Existing claim",
                "support_status": "supported",
                "recommended_wording": "controlled diagnostic",
                "avoid_wording": "unsupported claim",
                "supporting_outputs": "outputs/example.csv",
                "limitations": "proxy only",
            }
        ]
    ).to_csv(outputs / "qsvt_engineering_extension/claim_support_matrix.csv", index=False)


def _complete_rows() -> list[dict]:
    rows = []
    for case_name in ["ieee118", "ieee300"]:
        for variant in [
            "original_ridge",
            "coordinate_preconditioned_ridge",
            "transformed_penalty_preconditioned_ridge",
            "original_qsvt_diagnostic",
            "preconditioned_qsvt_diagnostic",
        ]:
            rows.append(
                {
                    "case_name": case_name,
                    "variant_name": variant,
                    "qsvt_degree": 201,
                    "qsvt_query_count": 403,
                    "qsvt_full_interval_error": 1.0e-5 if "preconditioned" in variant else 1.0e-1,
                    "qsvt_actual_singular_value_error": 1.0e-5
                    if "preconditioned" in variant
                    else 1.0e-1,
                    "phase_validation_status": "passed_scalar_full_domain",
                    "status": "ok",
                }
            )
    return rows


def _sweep_summary_rows() -> list[dict]:
    rows = []
    for variant in [
        "original_ridge",
        "coordinate_preconditioned_ridge",
        "transformed_penalty_preconditioned_ridge",
        "original_qsvt_diagnostic",
        "preconditioned_qsvt_diagnostic",
    ]:
        rows.append(
            {
                "case_name": "ieee300",
                "variant_name": variant,
                "alpha": 0.01,
                "mean_residual_norm": 2.0 if variant.startswith("coordinate") else 1.0,
                "mean_weighted_residual_norm": 2.0 if variant.startswith("coordinate") else 1.0,
                "mean_rmse_if_available": 0.2 if variant.startswith("coordinate") else 0.1,
                "mean_qsvt_full_interval_approx_error": 1.0e-5
                if "preconditioned" in variant
                else 1.0e-1,
                "mean_qsvt_degree": 201,
                "mean_qsvt_query_count": 403,
                "median_condition_number_original": 100.0,
                "median_condition_number_preconditioned_if_applicable": 10.0,
                "status": "ok",
                "interpretation": "phase2 row",
            }
        )
    return rows
