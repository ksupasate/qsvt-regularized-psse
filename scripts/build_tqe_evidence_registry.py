"""Build the TQE blocking-revision evidence registry and derived ledgers.

Single source of truth for manuscript evidence statuses. Every row is read
mechanically from existing result artifacts (never copied from manuscript
prose), assigned a stable configuration ID, and classified with an explicit
evidence-status enum that never collapses distinct execution tiers.

Outputs (all under outputs/tqe_blocking_revision/ unless noted):
  artifact_inventory.csv/json          Phase 0 inventory of evidence files
  evidence_registry.csv/json           Phase 1 registry (one row per configuration)
  resource_ledger.csv/json             Phase 3 mechanically derived resource rows
  resource_accounting_notes.md         Phase 3 formulas and separations
  ieee_case_evidence_matrix.csv/md     Phase 4 per-case evidence matrix
  convention_validation.csv            Phase 5 consolidated campaign results
  convention_validation_summary.md     Phase 5 summary
  readout_registry.csv/json            Phase 6 unified readout schema
  readout_variance_validation.csv      Phase 6 analytic vs empirical uncertainty
  readout_audit.md                     Phase 6 audit narrative
  manuscript/tables/registry_evidence_status.tex        generated table
  manuscript/tables/registry_final_configurations.tex   generated table
  manuscript/tables/final_degree255_resource_ledger.tex regenerated from CSV

Run: .venv/bin/python scripts/build_tqe_evidence_registry.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_qsvt_se.generalized.convention_api import predict_extraction
from robust_qsvt_se.qsvt.rectangular_convention import (
    apply_pcphase_qsvt_sequence,
    convert_pyqsp_sym_qsp_to_pcphase,
    extract_component,
    pcphase_qsvt_operator,
    pcphase_qsvt_top_block,
    production_scalar_response,
    scalar_julia_signal,
    validate_real_rectangular_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tqe_blocking_revision"
TABLES = ROOT / "manuscript" / "tables"

FINAL = ROOT / "outputs" / "final_useful_overlap_validation"
GEN = ROOT / "outputs" / "generalized_rectangular_qsvt"
TQE_REV = ROOT / "outputs" / "tqe_implementation_revision"
PHASE10 = ROOT / "outputs" / "phase10_full_rectangular_selected_output_qsvt"
PHASE8 = ROOT / "outputs" / "phase8_integrated_readout"
PHASE9 = ROOT / "outputs" / "phase9_integrated_8x8_readout"
SPARSE10 = ROOT / "outputs" / "phase10_sparse_wrapper_8x8_complete"
NONLIN10 = ROOT / "outputs" / "phase10_nonlinear_qsvt_in_loop"

# Explicit evidence-status enum. `sampled_distribution` extends the base enum to
# name multinomial sampling from a verified exact circuit distribution, which is
# weaker than backend shot execution and must never be conflated with it.
EVIDENCE_STATUSES = (
    "classical_exact",
    "classical_simulation",
    "polynomial_evaluation",
    "qsvt_matrix_action",
    "statevector_dense",
    "sampled_distribution",
    "sampled_simulator",
    "transpiled_partial",
    "transpiled_complete",
    "modeled",
    "excluded",
    "missing",
)

REGISTRY_COLUMNS = [
    "configuration_id",
    "case",
    "matrix_shape",
    "matrix_fingerprint",
    "selected_output",
    "alpha",
    "beta",
    "lambda",
    "contraction_C",
    "polynomial_degree",
    "signal_calls_per_attempt",
    "phase_operations_per_attempt",
    "polynomial_error_uniform",
    "error_on_singular_values",
    "matrix_action_error",
    "qsvt_ridge_error",
    "postselection_probability",
    "postselection_probability_kind",
    "shots_attempted",
    "shots_accepted",
    "readout_estimate",
    "confidence_interval",
    "evidence_status",
    "artifact_paths",
    "generated_by",
    "notes",
]

NOT_ESTIMATED = "not_estimated"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_rows_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=1, sort_keys=True, default=str) + "\n", "utf-8")


def ci_text(low: float, high: float) -> str:
    return f"[{low:.6g}, {high:.6g}]"


# --------------------------------------------------------------------------- #
# Phase 0: artifact inventory
# --------------------------------------------------------------------------- #

PRODUCERS = {
    FINAL: "scripts/run_final_useful_overlap_validation.py",
    GEN: "scripts/run_generalized_* (see outputs/generalized_rectangular_qsvt/commands_run.txt)",
    TQE_REV: "scripts/run_tqe_implementation_revision.py",
    PHASE10: "scripts/run_phase10_full_rectangular_qsvt.py",
    PHASE8: "scripts/run_phase8_integrated_readout.py",
    PHASE9: "scripts/run_phase9_integrated_8x8_readout.py",
    SPARSE10: "scripts/run_phase10_sparse_wrapper_8x8_complete.py",
    NONLIN10: "scripts/run_phase10_nonlinear_qsvt_loop.py",
}


def producer_for(path: Path) -> str:
    for base, producer in PRODUCERS.items():
        if path.is_relative_to(base):
            return producer
    if path.is_relative_to(ROOT / "manuscript"):
        return "manuscript source (hand-maintained or generated; see table header comments)"
    return ""


def build_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(paths)):
        exists = path.exists()
        row = {
            "artifact_id": rel(path).replace("/", "__"),
            "path": rel(path),
            "exists": exists,
            "type": path.suffix.lstrip(".") or "dir",
            "modified_time": (
                datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat() if exists else ""
            ),
            "sha256": sha256_file(path) if exists and path.is_file() else "",
            "producer": producer_for(path) if exists else "",
            "status": "present" if exists else "missing",
        }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Phase 1: evidence registry rows (each built from source artifacts)
# --------------------------------------------------------------------------- #


def registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # ---- IEEE-14 degree-255 frozen configuration -------------------------- #
    cfg = read_json(FINAL / "final_scientific_configuration.json")
    quantum = pd.read_csv(FINAL / "final_quantum_reproduction.csv").iloc[0]
    app = pd.read_csv(FINAL / "final_application_reproduction.csv").iloc[0]
    shot = pd.read_csv(FINAL / "high_shot_backend_summary.csv").iloc[0]
    d255 = int(cfg["degree"])
    base_d255 = {
        "case": cfg["case"],
        "matrix_shape": f"{cfg['matrix_shape'][0]}x{cfg['matrix_shape'][1]}",
        "matrix_fingerprint": cfg["matrix_checksum"],
        "selected_output": cfg["selected_output_definition"],
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "lambda": cfg["lambda"],
        "contraction_C": cfg["contraction_C"],
        "polynomial_degree": d255,
        "signal_calls_per_attempt": d255,
        "phase_operations_per_attempt": d255 + 1,
    }
    rows.append(
        {
            **base_d255,
            "configuration_id": "ieee14_fullrect_d255_useful_overlap",
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": float(quantum["phase_reconstruction_error"]),
            "matrix_action_error": float(quantum["production_vs_reference_relative_error"]),
            "qsvt_ridge_error": float(quantum["selected_relative_error_vs_ridge"]),
            "postselection_probability": float(quantum["target_quadrature_probability"]),
            "postselection_probability_kind": (
                "target-quadrature probability (encoded-prefix probability "
                f"{float(quantum['encoded_prefix_probability']):.4f} reported separately)"
            ),
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "statevector_dense",
            "artifact_paths": ";".join(
                [
                    rel(FINAL / "final_scientific_configuration.json"),
                    rel(FINAL / "final_quantum_reproduction.csv"),
                    rel(FINAL / "final_application_reproduction.csv"),
                ]
            ),
            "generated_by": "scripts/run_final_useful_overlap_validation.py",
            "notes": (
                "Dense float64 sequence action on the residual state plus full "
                "operator block vs exact-SVD reference; benchmark-anchored "
                "application ratio "
                f"{float(app['rmse_ratio_vs_benchmark']):.4f} <= 1.25 (classical criterion). "
                "Correctness experiment; not a scalable sparse block-encoding."
            ),
        }
    )
    rows.append(
        {
            **base_d255,
            "configuration_id": "ieee14_fullrect_d255_shot_readout",
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": float(quantum["phase_reconstruction_error"]),
            "matrix_action_error": float(quantum["production_vs_reference_relative_error"]),
            "qsvt_ridge_error": abs(
                float(shot["aggregate_selected_output_estimate"])
                - float(shot["ridge_selected_output"])
            )
            / abs(float(shot["ridge_selected_output"])),
            "postselection_probability": float(shot["encoded_prefix_rate"]),
            "postselection_probability_kind": (
                "measured encoded-prefix rate from the separate postselection-"
                "diagnostic circuit; branch Hadamard test itself has no acceptance step"
            ),
            "shots_attempted": int(shot["total_hadamard_test_shots"])
            + int(shot["total_postselection_shots"]),
            "shots_accepted": int(shot["encoded_prefix_accepted_samples"]),
            "readout_estimate": float(shot["aggregate_selected_output_estimate"]),
            "confidence_interval": ci_text(
                float(shot["aggregate_confidence_interval_low"]),
                float(shot["aggregate_confidence_interval_high"]),
            ),
            "evidence_status": "sampled_simulator",
            "artifact_paths": ";".join(
                [
                    rel(FINAL / "high_shot_backend_summary.csv"),
                    rel(FINAL / "high_shot_backend_runs.csv"),
                ]
            ),
            "generated_by": "scripts/run_final_useful_overlap_validation.py",
            "notes": (
                "Integrated branch Hadamard test: controlled degree-255 QSVT operator "
                "acts inside the sampled Aer circuit as ONE opaque UnitaryGate "
                "(transpiled_partial: layer depth 5 is opaque-instruction depth, not "
                "elementary-gate depth). 5 seeds x 1e6 Hadamard shots + 5 x 1e6 "
                "encoded-prefix diagnostic shots."
            ),
        }
    )

    # ---- WP-J isolated high-precision readout ----------------------------- #
    wpj = pd.read_csv(GEN / "ieee14_high_precision_backend_summary.csv")
    wpj_top = wpj[wpj["shots"] == wpj["shots"].max()].iloc[0]
    rows.append(
        {
            **base_d255,
            "configuration_id": "ieee14_fullrect_d255_isolated_readout_wpj",
            "signal_calls_per_attempt": 0,
            "phase_operations_per_attempt": 0,
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": abs(
                float(wpj_top["aggregate_y_estimate"]) - float(wpj_top["y_ridge"])
            )
            / abs(float(wpj_top["y_ridge"])),
            "postselection_probability": "",
            "postselection_probability_kind": "none (no in-circuit QSVT, no acceptance step)",
            "shots_attempted": int(wpj_top["total_shots"]),
            "shots_accepted": int(wpj_top["total_shots"]),
            "readout_estimate": float(wpj_top["aggregate_y_estimate"]),
            "confidence_interval": (
                f"rel95 half-width {float(wpj_top['aggregate_relative_ci_half_width']):.4f}"
            ),
            "evidence_status": "sampled_simulator",
            "artifact_paths": ";".join(
                [
                    rel(GEN / "ieee14_high_precision_backend_summary.csv"),
                    rel(GEN / "ieee14_high_precision_backend_runs.csv"),
                ]
            ),
            "generated_by": "scripts/run_generalized_high_precision_readout.py",
            "notes": (
                "ISOLATED overlap readout: state preparation loads the classically "
                "computed convention-validated update direction; the QSVT operator "
                "does NOT act in the sampled circuit. Signal/phase calls per attempt "
                "are 0 by construction. Must never be pooled with the integrated "
                "branch-circuit campaign."
            ),
        }
    )

    # ---- WP-F multioutput (matrix action + backend shots) ----------------- #
    multi_sv = pd.read_csv(GEN / "ieee14_multioutput_statevector.csv")
    multi_shot = pd.read_csv(GEN / "ieee14_multioutput_backend_shots.csv")
    rows.append(
        {
            **base_d255,
            "configuration_id": "ieee14_fullrect_d255_multioutput",
            "matrix_fingerprint": "",
            "selected_output": "; ".join(multi_sv["output"].tolist()),
            "alpha": float(multi_sv["lambda"].iloc[0]) * cfg["beta"] ** 2,
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": float(multi_sv["convention_block_error_vs_exact_svd"].max()),
            "qsvt_ridge_error": float(multi_sv["selected_rel_err_vs_ridge"].max()),
            "postselection_probability": float(multi_sv["postselection_probability_est"].iloc[0]),
            "postselection_probability_kind": "state-dependent quadrature estimate |B^T r_hat|^2",
            "shots_attempted": int(multi_shot["shots"].sum()),
            "shots_accepted": int(multi_shot["shots"].sum()),
            "readout_estimate": "; ".join(
                f"{o}={v:.6g}"
                for o, v in zip(multi_shot["output"], multi_shot["y_backend_shot"], strict=True)
            ),
            "confidence_interval": "; ".join(
                ci_text(lo, hi)
                for lo, hi in zip(multi_shot["ci_low"], multi_shot["ci_high"], strict=True)
            ),
            "evidence_status": "sampled_simulator",
            "artifact_paths": ";".join(
                [
                    rel(GEN / "ieee14_multioutput_statevector.csv"),
                    rel(GEN / "ieee14_multioutput_backend_shots.csv"),
                ]
            ),
            "generated_by": "scripts/run_generalized_ieee_application.py",
            "notes": (
                "Three preselected outputs; dense matrix-action values plus Aer "
                "Hadamard-test shots (200k/output). Area-aggregate output has poor "
                "RELATIVE shot precision (~0.59) because its value is small; retained."
            ),
        }
    )

    # ---- WP-G robustness sweep -------------------------------------------- #
    rob = pd.read_csv(GEN / "ieee14_robustness_results.csv")
    rows.append(
        {
            **base_d255,
            "configuration_id": "ieee14_fullrect_d255_robustness_sweep",
            "matrix_fingerprint": "",
            "selected_output": "full-update ground-truth RMSE ratio (no single functional)",
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": float(rob["convention_block_error_vs_exact_svd"].max()),
            "qsvt_ridge_error": float(rob["rmse_ratio"].max()),
            "postselection_probability": "",
            "postselection_probability_kind": "per-row quadrature estimates in artifact",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "qsvt_matrix_action",
            "artifact_paths": rel(GEN / "ieee14_robustness_results.csv"),
            "generated_by": "scripts/run_generalized_ieee_application.py",
            "notes": (
                f"{len(rob)} controlled perturbation configs (gaussian/bad-data/missing), "
                "all pass; qsvt_ridge_error column holds the MAX matched-lambda "
                "ground-truth RMSE ratio (execution-accuracy criterion), max 1.107."
            ),
        }
    )

    # ---- degree-31 boundary configurations (phase10 + tqe revision) -------- #
    p10 = pd.read_csv(PHASE10 / "full_rectangular_qsvt_vs_ridge.csv")
    be_meta = read_json(PHASE10 / "full_rectangular_block_encoding_metadata.json")
    fin31_meta = read_json(TQE_REV / "full_rectangular_finite_shot_metadata.json")
    fin31 = pd.read_csv(TQE_REV / "full_rectangular_finite_shot.csv").iloc[0]

    def phase10_row(case: str, tier: str, config_id: str, status: str, note: str) -> dict:
        row = p10[(p10["case"] == case) & (p10["tier"] == tier)].iloc[0]
        beta = float(be_meta[case]["beta_spectral_norm"])
        alpha = float(row["alpha"])
        degree = int(row["degree"])
        c_over_beta = float(row["physical_recovery_factor_C_over_beta"])
        return {
            "configuration_id": config_id,
            "case": case,
            "matrix_shape": "{}x{}".format(*be_meta[case]["full_matrix_shape_rows_cols"]),
            "matrix_fingerprint": "",
            "selected_output": "full postselected update (L2 diagnostic) + first coordinate",
            "alpha": alpha,
            "beta": beta,
            "lambda": alpha / beta**2,
            "contraction_C": c_over_beta * beta,
            "polynomial_degree": degree,
            "signal_calls_per_attempt": degree,
            "phase_operations_per_attempt": degree + 1,
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": float(row["relative_l2_error"]),
            "postselection_probability": float(row["postselection_probability"]),
            "postselection_probability_kind": "statevector postselection probability",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": status,
            "artifact_paths": ";".join(
                [
                    rel(PHASE10 / "full_rectangular_qsvt_vs_ridge.csv"),
                    rel(PHASE10 / "full_rectangular_block_encoding_metadata.json"),
                ]
            ),
            "generated_by": "scripts/run_phase10_full_rectangular_qsvt.py",
            "notes": note,
        }

    rows.append(
        phase10_row(
            "ieee14",
            "degree_aware_lambda_0.02",
            "ieee14_fullrect_d31_degree_aware_lambda_0p02",
            "statevector_dense",
            "Lower-degree boundary experiment; compiled-circuit statevector cross-"
            "checked against matrix-vector path. Distinct from the d=255 campaign.",
        )
    )
    rows.append(
        phase10_row(
            "ieee14",
            "anchor_lambda_0.068",
            "ieee14_fullrect_d31_degree_aware_lambda_0p068",
            "statevector_dense",
            "Lower-degree boundary experiment (anchor lambda=0.068, exact p_succ "
            "0.8483616). Basis of the 30-seed finite-shot record and the d=31 "
            "sensitivity/resource paragraphs.",
        )
    )
    rows.append(
        phase10_row(
            "ieee30",
            "degree_aware_lambda_0.02",
            "ieee30_fullrect_d39_degree_aware_lambda_0p02",
            "statevector_dense",
            "Lower-degree boundary experiment (degree 39).",
        )
    )
    rows.append(
        phase10_row(
            "ieee30",
            "anchor_lambda_0.068",
            "ieee30_fullrect_d31_degree_aware_lambda_0p068",
            "statevector_dense",
            "Lower-degree boundary experiment (anchor lambda=0.068).",
        )
    )

    q31 = fin31_meta["qsvt"]
    rows.append(
        {
            "configuration_id": "ieee14_fullrect_d31_integrated_30seed_lambda_0p068",
            "case": "ieee14",
            "matrix_shape": f"{q31['m_measurements']}x{q31['n_states']}",
            "matrix_fingerprint": "",
            "selected_output": str(fin31["output_name"]),
            "alpha": float(fin31_meta["alpha"]),
            "beta": float(fin31_meta["beta"]),
            "lambda": float(fin31_meta["lambda"]),
            "contraction_C": float(q31["bound_C"]),
            "polynomial_degree": int(q31["degree"]),
            "signal_calls_per_attempt": int(q31["signal_unitary_calls"]),
            "phase_operations_per_attempt": int(q31["projector_phase_operations"]),
            "polynomial_error_uniform": float(q31["target_fit_error"]),
            "error_on_singular_values": "",
            "matrix_action_error": float(q31["circuit_vs_matvec_error"]),
            "qsvt_ridge_error": float(fin31["mean_relative_error_vs_qsvt"]),
            "postselection_probability": float(q31["postselection_probability"]),
            "postselection_probability_kind": "exact statevector postselection probability",
            "shots_attempted": int(fin31["total_shots"]),
            "shots_accepted": round(
                float(fin31["mean_estimated_qsvt_postselection_accepted_shots"]) * 30
            ),
            "readout_estimate": float(fin31["mean_selected_output"]),
            "confidence_interval": ci_text(
                float(fin31["mean_ci95_low"]), float(fin31["mean_ci95_high"])
            ),
            "evidence_status": "sampled_distribution",
            "artifact_paths": ";".join(
                [
                    rel(TQE_REV / "full_rectangular_finite_shot.csv"),
                    rel(TQE_REV / "full_rectangular_finite_shot_metadata.json"),
                ]
            ),
            "generated_by": "scripts/run_tqe_implementation_revision.py",
            "notes": (
                "30 seeds x 100,000 samples drawn multinomially from the VERIFIED "
                "exact integrated-circuit distribution (max distribution error "
                "4.4e-15); one 1,000-shot Aer smoke run is the only backend shot "
                "execution. Not Aer backend sampling at scale; never pool with the "
                "d=255 Aer campaigns. qsvt_ridge_error is mean rel. error vs the "
                "exact QSVT output."
            ),
        }
    )

    # ---- Generalized IEEE-30 sweep and IEEE-57 escalation ------------------ #
    def action_sweep_row(
        df: pd.DataFrame, config_id: str, case: str, script: str, artifact: Path, note: str
    ) -> dict:
        passing = df[df["status"] == "STATEVECTOR_PASSED"]
        best = passing.loc[passing["rmse_ratio"].idxmin()] if len(passing) else df.iloc[-1]
        degree = int(best["degree"])
        return {
            "configuration_id": config_id,
            "case": case,
            "matrix_shape": str(best["matrix_shape"]),
            "matrix_fingerprint": "",
            "selected_output": "full-update ground-truth RMSE ratio (no single functional)",
            "alpha": float(best["alpha"]),
            "beta": float(best["beta"]),
            "lambda": float(best["lambda"]),
            "contraction_C": float(best["C_global"]),
            "polynomial_degree": degree,
            "signal_calls_per_attempt": degree,
            "phase_operations_per_attempt": degree + 1,
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": float(df["convention_block_error_vs_exact_svd"].max()),
            "qsvt_ridge_error": float(best["rmse_ratio"]),
            "postselection_probability": float(best["postselection_probability_est"]),
            "postselection_probability_kind": "state-dependent quadrature estimate |B^T r_hat|^2",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "qsvt_matrix_action",
            "artifact_paths": rel(artifact),
            "generated_by": script,
            "notes": note,
        }

    d30 = pd.read_csv(GEN / "ieee30_useful_overlap_search.csv")
    d30_full = pd.read_csv(GEN / "ieee30_statevector_results.csv")
    rows.append(
        action_sweep_row(
            d30_full,
            "ieee30_fullrect_generalized_sweep_seed123",
            "ieee30",
            "scripts/run_generalized_ieee_application.py",
            GEN / "ieee30_useful_overlap_search.csv",
            f"{len(d30)} candidates (degree x lambda grid), "
            f"{int((d30['status'] == 'STATEVECTOR_PASSED').sum())} pass the matched-lambda "
            "execution-accuracy criterion (QSVT/matched-Ridge ground-truth RMSE ratio "
            "<= 1.25); 9 failed candidates retained. Dense exact matrix action; NO "
            "state propagation, NO sampled shots, NO transpilation. Row shows the "
            "best passing configuration (d=255, lambda=1e-3).",
        )
    )

    d57 = pd.read_csv(GEN / "ieee57_escalation_results.csv")
    rows.append(
        action_sweep_row(
            d57,
            "ieee57_fullrect_escalation_seed123",
            "ieee57",
            "scripts/run_generalized_ieee_application.py",
            GEN / "ieee57_escalation_results.csv",
            f"{len(d57)} executed rows (d in {{31,63,127,255}}), "
            f"{int((d57['status'] == 'STATEVECTOR_PASSED').sum())} pass the matched-lambda "
            "execution-accuracy criterion at lambda=1e-3; 4 failed rows retained. "
            "Dense exact matrix action (10-qubit dilation, <=36 s); NO state "
            "propagation, NO sampled shots, NO transpilation. Row shows the best "
            "passing configuration (d=255, lambda=1e-3).",
        )
    )

    # ---- selected-submatrix integrated readouts ---------------------------- #
    for config_id, case_label, summary_path, script in (
        (
            "selected4x4_d31_integrated_readout",
            "ieee14 4x4 selected block",
            PHASE8 / "integrated_readout_summary.csv",
            "scripts/run_phase8_integrated_readout.py",
        ),
        (
            "selected8x8_d31_integrated_readout",
            "ieee14 8x8 selected block",
            PHASE9 / "integrated_readout_summary.csv",
            "scripts/run_phase9_integrated_8x8_readout.py",
        ),
    ):
        summ = pd.read_csv(summary_path)
        top = summ[
            (summ["observable_label"] == "state_correction_0")
            & (summ["shots"] == summ["shots"].max())
        ].iloc[0]
        rows.append(
            {
                "configuration_id": config_id,
                "case": case_label,
                "matrix_shape": "4x4" if "4x4" in config_id else "8x8",
                "matrix_fingerprint": "",
                "selected_output": "first-coordinate functional (state_correction_0)",
                "alpha": "",
                "beta": "",
                "lambda": "",
                "contraction_C": "",
                "polynomial_degree": int(top["signal_unitary_calls_per_attempt"]),
                "signal_calls_per_attempt": int(top["signal_unitary_calls_per_attempt"]),
                "phase_operations_per_attempt": int(top["projector_phase_operations_per_attempt"]),
                "polynomial_error_uniform": NOT_ESTIMATED,
                "error_on_singular_values": "",
                "matrix_action_error": "",
                "qsvt_ridge_error": float(top["mean_relative_error_vs_ridge"]),
                "postselection_probability": float(top["statevector_postselection_probability"]),
                "postselection_probability_kind": (
                    "statevector postselection probability (measured mean "
                    f"{float(top['mean_measured_postselection_probability']):.4f})"
                ),
                "shots_attempted": int(top["shots"]) * int(top["num_seeds"]),
                "shots_accepted": round(
                    float(top["mean_accepted_attempts"]) * int(top["num_seeds"])
                ),
                "readout_estimate": float(top["mean_recovered_physical_functional"]),
                "confidence_interval": (
                    f"mean SE {float(top['mean_signed_overlap_standard_error']):.3g} (overlap)"
                ),
                "evidence_status": "sampled_simulator",
                "artifact_paths": rel(summary_path),
                "generated_by": script,
                "notes": (
                    "Aer integrated-chain shot sampling "
                    "(backend label aer_integrated_circuit_shot_sampling); joint "
                    "postselection+sign estimator."
                ),
            }
        )

    # ---- sparse 8x8 wrapper ------------------------------------------------ #
    sw = pd.read_csv(SPARSE10 / "sparse_wrapper_8x8_qsvt_validation.csv").iloc[0]
    swv = pd.read_csv(SPARSE10 / "sparse_wrapper_8x8_validation.csv")
    swv_primary = swv[swv["is_primary"].astype(str).str.lower() == "true"]
    swv_row = swv_primary.iloc[0] if len(swv_primary) else swv.iloc[0]
    rows.append(
        {
            "configuration_id": "sparse8x8_d31_wrapper",
            "case": "ieee14 8x8 sparsified quantized block",
            "matrix_shape": "8x8",
            "matrix_fingerprint": "",
            "selected_output": "postselected update vs Ridge on the quantized block",
            "alpha": float(sw["alpha"]),
            "beta": float(sw["beta_effective"]),
            "lambda": float(sw["lambda_alpha_over_beta2"]),
            "contraction_C": float(sw["bound_C"]),
            "polynomial_degree": int(sw["degree"]),
            "signal_calls_per_attempt": int(sw["degree"]),
            "phase_operations_per_attempt": int(sw["degree"]) + 1,
            "polynomial_error_uniform": float(sw["target_fit_error"]),
            "error_on_singular_values": float(sw["sparse_vs_exact_svt_error"]),
            "matrix_action_error": float(sw["sparse_vs_dense_action_error"]),
            "qsvt_ridge_error": float(sw.get("sparse_update_relative_error", "nan"))
            if "sparse_update_relative_error" in sw
            else "",
            "postselection_probability": float(sw["sparse_postselection_probability"]),
            "postselection_probability_kind": "statevector postselection probability",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "statevector_dense",
            "artifact_paths": ";".join(
                [
                    rel(SPARSE10 / "sparse_wrapper_8x8_qsvt_validation.csv"),
                    rel(SPARSE10 / "sparse_wrapper_8x8_validation.csv"),
                ]
            ),
            "generated_by": "scripts/run_phase10_sparse_wrapper_8x8_complete.py",
            "notes": (
                "Complete sparse block-encoding wrapper for ONE quantized 8x8 block; "
                "wrapper circuit transpiled to elementary gates "
                f"(transpiled gate count {int(swv_row['transpiled_gate_count'])}, "
                f"depth {int(swv_row['transpiled_depth'])}, "
                f"cx {int(swv_row['transpiled_cx_count'])}) - transpiled_complete for "
                "the wrapper only; QSVT action statevector-validated. Not IEEE-scale."
            ),
        }
    )

    # ---- nonlinear loop ----------------------------------------------------- #
    nl = pd.read_csv(NONLIN10 / "nonlinear_qsvt_summary.csv")
    qsvt_nl = nl[nl["solver"].str.contains("qsvt", case=False)]
    rows.append(
        {
            "configuration_id": "ieee14_nonlinear_loop_d31_one_seed",
            "case": "ieee14 nonlinear AC Gauss-Newton",
            "matrix_shape": "82x27 (rebuilt per iteration)",
            "matrix_fingerprint": "",
            "selected_output": "full-state RMSE trajectory",
            "alpha": "",
            "beta": "recomputed per iteration",
            "lambda": "recomputed per iteration",
            "contraction_C": "",
            "polynomial_degree": 31,
            "signal_calls_per_attempt": 31,
            "phase_operations_per_attempt": 32,
            "polynomial_error_uniform": NOT_ESTIMATED,
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": float(qsvt_nl["max_update_error_vs_ridge"].max())
            if len(qsvt_nl)
            else "",
            "postselection_probability": "",
            "postselection_probability_kind": "per-iteration values in artifact",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "statevector_dense",
            "artifact_paths": ";".join(
                [
                    rel(NONLIN10 / "nonlinear_qsvt_summary.csv"),
                    rel(NONLIN10 / "nonlinear_qsvt_iteration_log.csv"),
                ]
            ),
            "generated_by": "scripts/run_phase10_nonlinear_qsvt_loop.py",
            "notes": (
                "One-seed interface and consistency check; degree-aware QSVT/Ridge "
                "trajectories reach the iteration cap without convergence (negative "
                "result retained)."
            ),
        }
    )

    # ---- mitigation, structured access, classical, modeled ------------------ #
    mlae = pd.read_csv(GEN / "postselection_mitigation_executed_results.csv")
    rows.append(
        {
            "configuration_id": "controlled_mlae_aer",
            "case": "controlled amplitude case (not IEEE)",
            "matrix_shape": "",
            "matrix_fingerprint": "",
            "selected_output": "amplitude estimate",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": int(mlae["shots"].sum()),
            "shots_accepted": "",
            "readout_estimate": "; ".join(
                f"{m}:{e:.5g}" for m, e in zip(mlae["method"], mlae["estimate"], strict=True)
            ),
            "confidence_interval": "",
            "evidence_status": "sampled_simulator",
            "artifact_paths": rel(GEN / "postselection_mitigation_executed_results.csv"),
            "generated_by": "scripts/run_generalized_postselection_mitigation.py",
            "notes": "MLAE executed on Aer for controlled amplitudes only.",
        }
    )
    rows.append(
        {
            "configuration_id": "ieee14_postselection_mitigation_model",
            "case": "ieee14",
            "matrix_shape": "82x27",
            "matrix_fingerprint": "",
            "selected_output": "modeled oracle-call comparison at epsilon=0.01",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "modeled",
            "artifact_paths": rel(GEN / "postselection_mitigation_cost_comparison.csv"),
            "generated_by": "scripts/run_generalized_postselection_mitigation.py",
            "notes": "IEEE-14 MLAE/IQAE cost reduction is MODELED, never executed.",
        }
    )
    rows.append(
        {
            "configuration_id": "structured_psse_access_classical",
            "case": "ieee14 + ieee30",
            "matrix_shape": "82x27; 172x59",
            "matrix_fingerprint": "",
            "selected_output": "row-wise sparse reconstruction of the weighted Jacobian",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": 0.0,
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "classical_exact",
            "artifact_paths": ";".join(
                [
                    rel(GEN / "structured_psse_access_ieee14.csv"),
                    rel(GEN / "structured_psse_access_ieee30.csv"),
                ]
            ),
            "generated_by": "scripts/run_generalized_structured_access.py",
            "notes": (
                "Classical structured row access reconstructs the dense weighted "
                "Jacobian exactly; the quantum QROM/reversible oracle remains modeled "
                "(see ieee_quantum_sparse_oracle_model row)."
            ),
        }
    )
    rows.append(
        {
            "configuration_id": "ieee_quantum_sparse_oracle_model",
            "case": "ieee14-ieee300",
            "matrix_shape": "case-dependent",
            "matrix_fingerprint": "",
            "selected_output": "T-count model for QROM lookup access",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "modeled",
            "artifact_paths": rel(GEN / "structured_psse_access_report.md"),
            "generated_by": "scripts/run_generalized_structured_access.py",
            "notes": (
                "Quantum sparse/QROM oracle access is modeled only; compiled 4x4/8x8 "
                "lookup circuits exist separately (revision sparse-oracle demo) but no "
                "IEEE-scale block encoding is compiled."
            ),
        }
    )
    rows.append(
        {
            "configuration_id": "classical_baselines_final_config",
            "case": "ieee14",
            "matrix_shape": "82x27",
            "matrix_fingerprint": cfg["matrix_checksum"],
            "selected_output": cfg["selected_output_definition"],
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "lambda": cfg["lambda"],
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "classical_exact",
            "artifact_paths": ";".join(
                [
                    rel(FINAL / "final_classical_baselines.csv"),
                    rel(GEN / "generalized_classical_baselines.csv"),
                ]
            ),
            "generated_by": (
                "scripts/run_final_useful_overlap_validation.py; "
                "scripts/run_generalized_classical_baselines.py"
            ),
            "notes": "Classical solvers are microsecond-fast and exact; no speedup claim.",
        }
    )
    rows.append(
        {
            "configuration_id": "ieee118_300_resource_models",
            "case": "ieee118; ieee300",
            "matrix_shape": "case-dependent",
            "matrix_fingerprint": "",
            "selected_output": "resource models only",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "modeled",
            "artifact_paths": rel(ROOT / "outputs" / "phase10_end_to_end_resource_ledger"),
            "generated_by": "scripts/run_phase10_resource_ledger.py",
            "notes": "No dense-rectangular execution for IEEE 118/300; models only.",
        }
    )

    # ---- convention-validation campaigns (controlled matrices) -------------- #
    heldout_gen = pd.read_csv(GEN / "heldout_rectangular_matrix_results.csv")
    degrees_gen = pd.read_csv(GEN / "degree_generalization_results.csv")
    heldout_fin = pd.read_csv(FINAL / "heldout_rectangular_validation.csv")
    rows.append(
        {
            "configuration_id": "convention_validation_generalized_campaign",
            "case": "controlled real rectangular matrices",
            "matrix_shape": "2x1 - 12x7 (7 dims x 7 spectral families)",
            "matrix_fingerprint": "",
            "selected_output": "rectangular block vs exact-SVD reference",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "odd {1,3,5,7,15,31,63,127,255}",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": float(
                pd.to_numeric(heldout_gen["convention_error_vs_encoded"], errors="coerce").max()
            ),
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "qsvt_matrix_action",
            "artifact_paths": ";".join(
                [
                    rel(GEN / "heldout_rectangular_matrix_results.csv"),
                    rel(GEN / "degree_generalization_results.csv"),
                    rel(GEN / "complex_rectangular_results.csv"),
                    rel(GEN / "rectangular_convention_symbolic_checks.csv"),
                ]
            ),
            "generated_by": "scripts/run_generalized_rectangular_convention.py",
            "notes": (
                f"{len(heldout_gen)} held-out real matrices pass; "
                f"{int((degrees_gen['parity'] == 'odd').sum())} odd rows over 9 distinct "
                "odd degrees pass; 6 even probes rejected; 245 complex probes "
                "unsupported. Seeds disjoint from development."
            ),
        }
    )
    rows.append(
        {
            "configuration_id": "convention_validation_final_campaign",
            "case": "controlled real rectangular matrices",
            "matrix_fingerprint": "",
            "matrix_shape": "held-out sweep (150 matrices)",
            "selected_output": "rectangular block vs exact-SVD reference",
            "alpha": "",
            "beta": "",
            "lambda": "",
            "contraction_C": "",
            "polynomial_degree": "8 degree rows + 5 evaluator rows",
            "signal_calls_per_attempt": "",
            "phase_operations_per_attempt": "",
            "polynomial_error_uniform": "",
            "error_on_singular_values": "",
            "matrix_action_error": "",
            "qsvt_ridge_error": "",
            "postselection_probability": "",
            "postselection_probability_kind": "",
            "shots_attempted": "",
            "shots_accepted": "",
            "readout_estimate": "",
            "confidence_interval": "",
            "evidence_status": "qsvt_matrix_action",
            "artifact_paths": ";".join(
                [
                    rel(FINAL / "heldout_rectangular_validation.csv"),
                    rel(FINAL / "degree_generalization_validation.csv"),
                    rel(FINAL / "independent_mapping_validation.csv"),
                ]
            ),
            "generated_by": "scripts/run_final_useful_overlap_validation.py",
            "notes": (
                f"{len(heldout_fin)} held-out matrices; separate earlier campaign - "
                "counts intentionally differ from the generalized campaign and the two "
                "are never merged."
            ),
        }
    )

    for row in rows:
        if row["evidence_status"] not in EVIDENCE_STATUSES:
            raise ValueError(f"invalid evidence status: {row['evidence_status']}")
    ids = [row["configuration_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate configuration IDs in registry")
    return rows


# --------------------------------------------------------------------------- #
# Phase 3: resource ledger, mechanically derived
# --------------------------------------------------------------------------- #


def resource_rows(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["configuration_id"]: row for row in registry}
    ledger: list[dict[str, Any]] = []

    def derived(config_id: str, **overrides: Any) -> dict[str, Any]:
        reg = by_id[config_id]
        degree = int(reg["polynomial_degree"])
        row = {
            "configuration_id": config_id,
            "degree": degree,
            "signal_calls_per_attempt": degree,
            "phase_operations_per_attempt": degree + 1,
            "alternating_sequence_length": 2 * degree + 1,
            "residual_preparations_per_attempt": 1,
            "block_encoding_calls_per_attempt": degree,
            "controlled_block_encoding_calls_per_attempt": NOT_ESTIMATED,
            "shots_attempted": reg.get("shots_attempted", ""),
            "shots_accepted": reg.get("shots_accepted", ""),
            "postselection_probability": reg.get("postselection_probability", ""),
            "expected_attempts_per_success": "",
            "signal_calls_total": NOT_ESTIMATED,
            "modeled_loader_cost": NOT_ESTIMATED,
            "modeled_value_rotation_cost": NOT_ESTIMATED,
            "fault_tolerant_cost": "excluded",
            "evidence_status": reg["evidence_status"],
            "notes": "",
        }
        row.update(overrides)
        p = row.get("postselection_probability")
        if isinstance(p, float) and 0.0 < p <= 1.0:
            row["expected_attempts_per_success"] = 1.0 / p
        # consistency guards: mechanically derived counts must match the formulas
        assert row["signal_calls_per_attempt"] == row["degree"]
        assert row["phase_operations_per_attempt"] == row["degree"] + 1
        assert row["alternating_sequence_length"] == 2 * row["degree"] + 1
        return row

    # d=255 statevector row
    ledger.append(
        derived(
            "ieee14_fullrect_d255_useful_overlap",
            notes=(
                "Executed dense statevector action; 8 logical qubits (dilation 256). "
                "Direct-rejection sampling would need 1/p_quadrature = "
                "350.19 attempts per accepted sample (modeled, not executed)."
            ),
        )
    )
    # d=255 shot campaign
    shot = pd.read_csv(FINAL / "high_shot_backend_summary.csv").iloc[0]
    run0 = pd.read_csv(FINAL / "high_shot_backend_runs.csv").iloc[0]
    ledger.append(
        derived(
            "ieee14_fullrect_d255_shot_readout",
            controlled_block_encoding_calls_per_attempt=(
                "1 opaque controlled QSVT-operator application per Hadamard shot "
                "(executed); equals d controlled signal calls per shot at "
                "decomposition level (modeled)"
            ),
            residual_preparations_per_attempt=1,
            signal_calls_total=(
                f"{int(shot['total_hadamard_test_shots']) * 255} decomposition-level "
                "signal calls over 5e6 Hadamard shots (modeled equivalence; executed "
                "circuits apply one opaque operator per shot)"
            ),
            modeled_loader_cost="dense initialize executed per shot; QROM not used in this run",
            notes=(
                "Two circuit families executed: branch Hadamard test "
                f"(depth {int(run0['transpiled_depth_hadamard'])} opaque-instruction "
                "layers - NOT elementary-gate depth) and encoded-prefix diagnostic "
                f"(depth {int(run0['transpiled_depth_postselection'])} layers). "
                "9 qubits (Hadamard) / 8 qubits (prefix)."
            ),
        )
    )
    # WP-J isolated readout: no QSVT in circuit
    wpj = by_id["ieee14_fullrect_d255_isolated_readout_wpj"]
    ledger.append(
        {
            "configuration_id": "ieee14_fullrect_d255_isolated_readout_wpj",
            "degree": 255,
            "signal_calls_per_attempt": 0,
            "phase_operations_per_attempt": 0,
            "alternating_sequence_length": 0,
            "residual_preparations_per_attempt": 1,
            "block_encoding_calls_per_attempt": 0,
            "controlled_block_encoding_calls_per_attempt": 0,
            "shots_attempted": wpj["shots_attempted"],
            "shots_accepted": wpj["shots_accepted"],
            "postselection_probability": "",
            "expected_attempts_per_success": "",
            "signal_calls_total": 0,
            "modeled_loader_cost": "dense StatePreparation of the classical update direction",
            "modeled_value_rotation_cost": NOT_ESTIMATED,
            "fault_tolerant_cost": "excluded",
            "evidence_status": wpj["evidence_status"],
            "notes": (
                "Isolated overlap readout: QSVT executed classically upstream, so "
                "in-circuit signal/phase counts are zero by construction."
            ),
        }
    )
    # d=31 integrated 30-seed campaign, plus the epsilon=1e-2 sensitivity model
    fin31_meta = read_json(TQE_REV / "full_rectangular_finite_shot_metadata.json")
    p31 = float(fin31_meta["qsvt"]["postselection_probability"])
    accepted_target = 2500
    attempts = math.ceil(accepted_target / p31)
    ledger.append(
        derived(
            "ieee14_fullrect_d31_integrated_30seed_lambda_0p068",
            controlled_block_encoding_calls_per_attempt=0,
            signal_calls_total=(
                f"epsilon=1e-2 model: {attempts} attempts x 31 = {attempts * 31} "
                f"signal calls, {attempts * 32} phase operations, {attempts} residual "
                "reloads (mechanically derived from p_succ "
                f"{p31:.7f} and 2,500 accepted samples)"
            ),
            modeled_loader_cost=(
                f"8-bit QROM loader model: 818 Toffolis (3,272 T) per load; "
                f"{attempts * 818} Toffolis ({attempts * 3272} T) over the modeled attempts"
            ),
            notes=(
                "30 seeds x 100,000 samples from the verified exact distribution; "
                "9 circuit qubits; opaque-instruction depth 68 (NOT elementary-gate "
                "depth). Distinct interference-branch acceptance (mean 92,400.6/seed) "
                "and estimated QSVT-postselection acceptance (mean 84,801.3/seed) are "
                "never merged."
            ),
        )
    )
    for config_id in (
        "ieee14_fullrect_d31_degree_aware_lambda_0p02",
        "ieee14_fullrect_d31_degree_aware_lambda_0p068",
        "ieee30_fullrect_d39_degree_aware_lambda_0p02",
        "ieee30_fullrect_d31_degree_aware_lambda_0p068",
        "ieee30_fullrect_generalized_sweep_seed123",
        "ieee57_fullrect_escalation_seed123",
        "selected4x4_d31_integrated_readout",
        "selected8x8_d31_integrated_readout",
        "sparse8x8_d31_wrapper",
    ):
        ledger.append(derived(config_id))
    return ledger


# --------------------------------------------------------------------------- #
# Phase 4: IEEE case evidence matrix
# --------------------------------------------------------------------------- #


def ieee_matrix(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def paths_for(*ids: str) -> str:
        return ";".join(row["artifact_paths"] for row in registry if row["configuration_id"] in ids)

    return [
        {
            "case": "IEEE-14",
            "exact_target": "yes (" + paths_for("classical_baselines_final_config") + ")",
            "polynomial": "yes (stable-chebyshev d=255 + d=31 fits; see registry rows)",
            "rectangular_qsvt_action": "yes ("
            + paths_for("ieee14_fullrect_d255_useful_overlap")
            + ")",
            "dense_statevector": "yes (d=255 sequence action on residual state; d=31 "
            "phase10 compiled-circuit statevector)",
            "sampled_readout": "yes (d=255: 5x1e6 Aer integrated branch test + WP-J "
            "isolated + multioutput; d=31: 30x100k exact-distribution sampling with "
            "1,000-shot Aer smoke)",
            "complete_transpilation": "no (opaque-unitary circuits only at full scale)",
            "sparse_oracle": "classical structured access executed; quantum oracle modeled",
            "status": "statevector_dense + sampled_simulator (d=255); "
            "sampled_distribution for the d=31 30-seed record",
        },
        {
            "case": "IEEE-30",
            "exact_target": "yes (matched Ridge in every sweep row)",
            "polynomial": "yes (d in {31,63,127,255} fits recorded)",
            "rectangular_qsvt_action": "yes ("
            + paths_for("ieee30_fullrect_generalized_sweep_seed123")
            + ")",
            "dense_statevector": "yes for the d=31/39 phase10 campaign ("
            + paths_for("ieee30_fullrect_d31_degree_aware_lambda_0p068")
            + "); generalized sweep rows are exact matrix action (no state propagation)",
            "sampled_readout": "no",
            "complete_transpilation": "no",
            "sparse_oracle": "classical structured access executed; quantum oracle modeled",
            "status": "qsvt_matrix_action (generalized sweep; 3/12 pass) + "
            "statevector_dense (d=31/39 boundary campaign)",
        },
        {
            "case": "IEEE-57",
            "exact_target": "yes (matched Ridge in every escalation row)",
            "polynomial": "yes (d in {31,63,127,255} fits recorded)",
            "rectangular_qsvt_action": "yes ("
            + paths_for("ieee57_fullrect_escalation_seed123")
            + ")",
            "dense_statevector": "no (no state propagation and no simulator framework "
            "for IEEE-57; exact dense operator action only)",
            "sampled_readout": "no",
            "complete_transpilation": "no",
            "sparse_oracle": "modeled only",
            "status": "qsvt_matrix_action (6 executed rows; 2 pass at lambda=1e-3, "
            "d in {127,255}); never 'modeled' and never 'sampled'",
        },
    ]


# --------------------------------------------------------------------------- #
# Phase 5/6: convention validation + readout registry
# --------------------------------------------------------------------------- #


def convention_validation() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for campaign, path in (
        ("generalized_heldout", GEN / "heldout_rectangular_matrix_results.csv"),
        ("generalized_degrees", GEN / "degree_generalization_results.csv"),
        ("generalized_complex", GEN / "complex_rectangular_results.csv"),
        ("generalized_symbolic", GEN / "rectangular_convention_symbolic_checks.csv"),
        ("final_heldout", FINAL / "heldout_rectangular_validation.csv"),
        ("final_degrees", FINAL / "degree_generalization_validation.csv"),
        ("final_evaluator", FINAL / "independent_mapping_validation.csv"),
    ):
        frame = pd.read_csv(path)
        frame.insert(0, "campaign", campaign)
        frame.insert(1, "source_artifact", rel(path))
        frames.append(frame)

    def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
        hermitian = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(hermitian)
        return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T

    def julia(matrix: np.ndarray) -> np.ndarray:
        matrix = validate_real_rectangular_matrix(matrix)
        rows, columns = matrix.shape
        pad = max(rows, columns)
        padded = np.zeros((pad, pad), dtype=np.float64)
        padded[:rows, :columns] = matrix
        identity = np.eye(pad)
        return np.block(
            [
                [padded, psd_sqrt(identity - padded @ padded.T)],
                [psd_sqrt(identity - padded.T @ padded), -padded.T],
            ]
        )

    def matrix_with_spectrum(
        shape: tuple[int, int], singular_values: np.ndarray, seed: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rows, columns = shape
        left = np.linalg.qr(rng.normal(size=(rows, rows)))[0]
        right = np.linalg.qr(rng.normal(size=(columns, columns)))[0]
        sigma = np.zeros(shape)
        rank = min(shape)
        sigma[:rank, :rank] = np.diag(singular_values[:rank])
        return left @ sigma @ right.T

    strict_rows: list[dict[str, Any]] = []
    strict_source = "scripts/build_tqe_evidence_registry.py::convention_validation"
    strict_cases = (
        ((6, 4), np.array([0.82, 0.82, 1.0e-8, 0.0]), "tall_repeated_near_zero"),
        ((4, 6), np.array([0.91, 0.43, 1.0e-10, 0.0]), "wide_near_zero"),
    )
    for degree in (1, 3, 5, 7):
        for shape, singular_values, label in strict_cases:
            rng = np.random.default_rng(81_000 + 101 * degree + shape[0])
            phases = rng.uniform(-np.pi, np.pi, degree + 1)
            matrix = matrix_with_spectrum(shape, singular_values, 91_000 + degree)
            rows, columns = shape
            pad = max(shape)
            component = predict_extraction(degree)[0]
            top = pcphase_qsvt_top_block(julia(matrix), phases, encoded_dimension=pad)
            extracted = extract_component(top, component)[:rows, :columns]
            left, values, right_t = np.linalg.svd(matrix, full_matrices=False)
            scalar = np.array(
                [production_scalar_response(value, phases, component=component) for value in values]
            )
            reference = (left * scalar) @ right_t
            error = float(np.max(np.abs(extracted - reference)))
            strict_rows.append(
                {
                    "campaign": "strict_live",
                    "source_artifact": strict_source,
                    "validation_case": label,
                    "degree": degree,
                    "shape": f"{rows}x{columns}",
                    "spectrum": "repeated+near_zero+exact_zero",
                    "component": component,
                    "max_error": error,
                    "expected_outcome": "pass_below_2e-12",
                    "observed_outcome": "pass" if error < 2.0e-12 else "fail",
                    "status": "pass" if error < 2.0e-12 else "fail",
                }
            )

    # Global-phase invariance is a state-level property, distinct from the
    # operator's convention-transfer phase.
    rng = np.random.default_rng(77)
    phases = rng.uniform(-np.pi, np.pi, 6)
    signal = scalar_julia_signal(0.37)
    state = rng.normal(size=2) + 1j * rng.normal(size=2)
    state /= np.linalg.norm(state)
    theta = 1.234
    base = apply_pcphase_qsvt_sequence(signal, phases, encoded_dimension=1, vector=state)
    phased = apply_pcphase_qsvt_sequence(
        signal, phases, encoded_dimension=1, vector=np.exp(1j * theta) * state
    )
    phase_error = float(np.max(np.abs(phased - np.exp(1j * theta) * base)))
    strict_rows.append(
        {
            "campaign": "strict_live",
            "source_artifact": strict_source,
            "validation_case": "residual_state_global_phase_invariance",
            "degree": 5,
            "shape": "scalar_signal",
            "max_error": phase_error,
            "expected_outcome": "pass_below_1e-12",
            "observed_outcome": "pass" if phase_error < 1.0e-12 else "fail",
            "status": "pass" if phase_error < 1.0e-12 else "fail",
        }
    )

    rejection_checks: list[tuple[str, Any, str]] = [
        ("even_degree_rejection", lambda: predict_extraction(2), "ConversionError"),
        (
            "complex_matrix_rejection",
            lambda: validate_real_rectangular_matrix(np.eye(2, dtype=np.complex128)),
            "ValueError",
        ),
        (
            "nonfinite_phase_rejection",
            lambda: pcphase_qsvt_operator(
                scalar_julia_signal(0.2), np.array([0.0, np.nan]), encoded_dimension=1
            ),
            "ValueError",
        ),
        (
            "invalid_phase_length_rejection",
            lambda: convert_pyqsp_sym_qsp_to_pcphase(np.zeros(3), degree=1),
            "ValueError",
        ),
    ]
    for label, operation, expected_exception in rejection_checks:
        try:
            operation()
        except (ValueError, RuntimeError) as exc:
            observed = type(exc).__name__
            passed = observed == expected_exception or (
                label == "even_degree_rejection" and observed == "ConversionError"
            )
        else:
            observed = "not_rejected"
            passed = False
        strict_rows.append(
            {
                "campaign": "strict_live",
                "source_artifact": strict_source,
                "validation_case": label,
                "degree": 2 if label == "even_degree_rejection" else "not_applicable",
                "shape": "2x2" if label == "complex_matrix_rejection" else "not_applicable",
                "max_error": "not_applicable",
                "expected_outcome": expected_exception,
                "observed_outcome": observed,
                "status": "pass" if passed else "fail",
            }
        )
    frames.append(pd.DataFrame(strict_rows))
    return pd.concat(frames, ignore_index=True, sort=False)


READOUT_COLUMNS = [
    "readout_id",
    "configuration_id",
    "polynomial_degree",
    "variant",
    "evidence_status",
    "backend",
    "attempted_shots",
    "postselection_accepted_shots",
    "readout_accepted_shots",
    "postselection_probability",
    "postselection_probability_definition",
    "branch_probability",
    "branch_probability_definition",
    "quadrature_probability",
    "quadrature_probability_definition",
    "conditional_acceptance_probability",
    "conditional_acceptance_probability_definition",
    "selected_output_estimate",
    "statevector_reference",
    "matched_ridge_reference",
    "standard_error",
    "standard_error_scope",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "confidence_interval_level",
    "confidence_interval_scope",
    "relative_ci_half_width",
    "analytic_variance",
    "analytic_variance_scope",
    "empirical_seed_variance",
    "empirical_seed_variance_scope",
    "empirical_mean_standard_error",
    "seed_count",
    "estimator_definition",
    "artifact_paths",
    "notes",
]


def readout_registry() -> list[dict[str, Any]]:
    not_applicable = "not_applicable"

    def base(**updates: Any) -> dict[str, Any]:
        row: dict[str, Any] = {column: NOT_ESTIMATED for column in READOUT_COLUMNS}
        row.update(
            {
                "postselection_accepted_shots": not_applicable,
                "readout_accepted_shots": not_applicable,
                "postselection_probability": not_applicable,
                "postselection_probability_definition": not_applicable,
                "branch_probability": not_applicable,
                "branch_probability_definition": not_applicable,
                "quadrature_probability": not_applicable,
                "quadrature_probability_definition": not_applicable,
                "conditional_acceptance_probability": not_applicable,
                "conditional_acceptance_probability_definition": not_applicable,
                "confidence_interval_level": 0.95,
                "empirical_seed_variance": NOT_ESTIMATED,
                "empirical_seed_variance_scope": NOT_ESTIMATED,
                "empirical_mean_standard_error": NOT_ESTIMATED,
            }
        )
        row.update(updates)
        return row

    rows: list[dict[str, Any]] = []
    shot = pd.read_csv(FINAL / "high_shot_backend_summary.csv").iloc[0]
    shot_runs = pd.read_csv(FINAL / "high_shot_backend_runs.csv")
    quantum = pd.read_csv(FINAL / "final_quantum_reproduction.csv").iloc[0]
    variance = pd.read_csv(FINAL / "final_readout_variance_checks.csv").iloc[0]
    n_hadamard = int(shot["total_hadamard_test_shots"])
    variance_per_shot = float(variance["variance_per_shot"])
    integrated_variance = variance_per_shot / n_hadamard
    integrated_seed_variance = float(shot_runs["selected_output_estimate"].var(ddof=1))
    rows.append(
        base(
            readout_id="d255_fullrect_statevector_diagnostic",
            configuration_id="ieee14_fullrect_d255_useful_overlap",
            polynomial_degree=255,
            variant="full-rectangular statevector selected-output diagnostic",
            evidence_status="statevector_dense",
            backend="dense float64 sequence action (no finite shots)",
            attempted_shots=not_applicable,
            postselection_probability=float(quantum["encoded_prefix_probability"]),
            postselection_probability_definition=(
                "statevector encoded-prefix norm after the production sequence"
            ),
            quadrature_probability=float(quantum["target_quadrature_probability"]),
            quadrature_probability_definition="statevector |selected quadrature|^2",
            selected_output_estimate=float(quantum["production_selected_output"]),
            statevector_reference=float(quantum["production_selected_output"]),
            matched_ridge_reference=float(quantum["ridge_selected_output"]),
            standard_error=not_applicable,
            standard_error_scope=not_applicable,
            confidence_interval_lower=not_applicable,
            confidence_interval_upper=not_applicable,
            confidence_interval_level=not_applicable,
            confidence_interval_scope=not_applicable,
            relative_ci_half_width=not_applicable,
            analytic_variance=not_applicable,
            analytic_variance_scope=not_applicable,
            seed_count=not_applicable,
            estimator_definition="direct statevector amplitude inspection; diagnostic only",
            artifact_paths=rel(FINAL / "final_quantum_reproduction.csv"),
            notes="No measurement-based full-vector recovery is performed.",
        )
    )
    rows.append(
        base(
            readout_id="d255_integrated_branch_hadamard",
            configuration_id="ieee14_fullrect_d255_shot_readout",
            polynomial_degree=255,
            variant="integrated branch Hadamard test (controlled QSVT acts in circuit)",
            evidence_status="sampled_simulator",
            backend=str(shot["backend_name"]) + " (Aer shots)",
            attempted_shots=n_hadamard,
            readout_accepted_shots=n_hadamard,
            branch_probability=(1.0 + float(variance["expectation"])) / 2.0,
            branch_probability_definition="Hadamard readout probability Pr[branch bit=0]",
            quadrature_probability=float(quantum["target_quadrature_probability"]),
            quadrature_probability_definition=(
                "statevector target-quadrature probability; not an acceptance rate"
            ),
            selected_output_estimate=float(shot["aggregate_selected_output_estimate"]),
            statevector_reference=float(shot["statevector_selected_output"]),
            matched_ridge_reference=float(shot["ridge_selected_output"]),
            standard_error=math.sqrt(integrated_variance),
            standard_error_scope="analytic SE of the pooled 5,000,000-shot estimate",
            confidence_interval_lower=float(shot["aggregate_confidence_interval_low"]),
            confidence_interval_upper=float(shot["aggregate_confidence_interval_high"]),
            confidence_interval_scope="normal-approximation CI for the pooled estimate",
            relative_ci_half_width=float(shot["relative_95ci_half_width"]),
            analytic_variance=integrated_variance,
            analytic_variance_scope="variance of the pooled selected-output estimate",
            empirical_seed_variance=integrated_seed_variance,
            empirical_seed_variance_scope=(
                "sample variance across five independent 1,000,000-shot estimates"
            ),
            empirical_mean_standard_error=math.sqrt(integrated_seed_variance / 5.0),
            seed_count=int(shot["seed_count"]),
            estimator_definition="y=(C/beta)||r|| (N0-N1)/N; no acceptance step",
            artifact_paths=";".join(
                [
                    rel(FINAL / "high_shot_backend_summary.csv"),
                    rel(FINAL / "high_shot_backend_runs.csv"),
                    rel(FINAL / "final_readout_variance_checks.csv"),
                ]
            ),
            notes=(
                "The encoded-prefix diagnostic is a separate circuit and separate "
                "registry row; its accepted counts are never used as Hadamard samples."
            ),
        )
    )
    n_post = int(shot["total_postselection_shots"])
    p_post_measured = float(shot["encoded_prefix_rate"])
    post_se = math.sqrt(p_post_measured * (1.0 - p_post_measured) / n_post)
    rows.append(
        base(
            readout_id="d255_encoded_prefix_diagnostic",
            configuration_id="ieee14_fullrect_d255_shot_readout",
            polynomial_degree=255,
            variant="separate encoded-prefix postselection diagnostic",
            evidence_status="sampled_simulator",
            backend=str(shot["backend_name"]) + " (Aer shots)",
            attempted_shots=n_post,
            postselection_accepted_shots=int(shot["encoded_prefix_accepted_samples"]),
            postselection_probability=p_post_measured,
            postselection_probability_definition=(
                "measured encoded-prefix frequency in the separate diagnostic circuit"
            ),
            selected_output_estimate=not_applicable,
            statevector_reference=float(quantum["encoded_prefix_probability"]),
            matched_ridge_reference=not_applicable,
            standard_error=post_se,
            standard_error_scope="binomial SE of the encoded-prefix frequency",
            confidence_interval_lower=max(0.0, p_post_measured - 1.96 * post_se),
            confidence_interval_upper=min(1.0, p_post_measured + 1.96 * post_se),
            confidence_interval_scope="normal-approximation CI for the frequency",
            relative_ci_half_width=1.96 * post_se / p_post_measured,
            analytic_variance=post_se**2,
            analytic_variance_scope="variance of the 5,000,000-shot frequency",
            empirical_seed_variance=float(shot_runs["encoded_prefix_rate"].var(ddof=1)),
            empirical_seed_variance_scope=(
                "sample variance across five independent 1,000,000-shot frequencies"
            ),
            empirical_mean_standard_error=math.sqrt(
                float(shot_runs["encoded_prefix_rate"].var(ddof=1)) / 5.0
            ),
            seed_count=int(shot["seed_count"]),
            estimator_definition="p_post_hat=N_encoded_prefix/N",
            artifact_paths=";".join(
                [
                    rel(FINAL / "high_shot_backend_summary.csv"),
                    rel(FINAL / "high_shot_backend_runs.csv"),
                ]
            ),
            notes="This probability is not the selected-output estimate.",
        )
    )

    wpj = pd.read_csv(GEN / "ieee14_high_precision_backend_summary.csv")
    wpj_runs = pd.read_csv(GEN / "ieee14_high_precision_backend_runs.csv")
    for _, source in wpj.iterrows():
        shots_per_seed = int(source["shots"])
        seed_count = int(source["n_seeds"])
        total_shots = int(source["total_shots"])
        single_seed_variance = float(source["theoretical_variance_per_shot"])
        pooled_variance = single_seed_variance / seed_count
        estimate = float(source["aggregate_y_estimate"])
        standard_error = math.sqrt(pooled_variance)
        run_block = wpj_runs[
            (wpj_runs["shots"] == shots_per_seed)
            & (wpj_runs["_row_type"].fillna("") != "aggregate")
        ]
        mu_exact = float(run_block["mu_exact"].iloc[0])
        rows.append(
            base(
                readout_id=f"d255_isolated_wpj_{shots_per_seed}",
                configuration_id="ieee14_fullrect_d255_isolated_readout_wpj",
                polynomial_degree=255,
                variant="isolated overlap Hadamard test (QSVT does not act in circuit)",
                evidence_status="sampled_simulator",
                backend="aer:AerSimulator (Aer shots)",
                attempted_shots=total_shots,
                readout_accepted_shots=total_shots,
                branch_probability=(1.0 + mu_exact) / 2.0,
                branch_probability_definition="Hadamard readout probability Pr[0]",
                selected_output_estimate=estimate,
                statevector_reference=float(source["y_statevector"]),
                matched_ridge_reference=float(source["y_ridge"]),
                standard_error=standard_error,
                standard_error_scope=(
                    f"analytic SE after pooling {seed_count} equal {shots_per_seed}-shot runs"
                ),
                confidence_interval_lower=estimate - 1.96 * standard_error,
                confidence_interval_upper=estimate + 1.96 * standard_error,
                confidence_interval_scope="normal-approximation CI for pooled counts",
                relative_ci_half_width=1.96 * standard_error / abs(estimate),
                analytic_variance=pooled_variance,
                analytic_variance_scope="variance of the pooled selected-output estimate",
                empirical_seed_variance=float(source["empirical_variance"]),
                empirical_seed_variance_scope=(
                    f"sample variance across {seed_count} separate seed estimates"
                ),
                empirical_mean_standard_error=math.sqrt(
                    float(source["empirical_variance"]) / seed_count
                ),
                seed_count=seed_count,
                estimator_definition="y=||dx|| mu_hat; no acceptance step",
                artifact_paths=";".join(
                    [
                        rel(GEN / "ieee14_high_precision_backend_summary.csv"),
                        rel(GEN / "ieee14_high_precision_backend_runs.csv"),
                    ]
                ),
                notes=(
                    "The source's aggregate_relative_ci_half_width is the analytic "
                    "single-seed width; this registry reports the pooled width and "
                    "keeps seed variation separate."
                ),
            )
        )

    fin31 = pd.read_csv(TQE_REV / "full_rectangular_finite_shot.csv").iloc[0]
    fin31_seeds = pd.read_csv(TQE_REV / "full_rectangular_finite_shot_seeds.csv")
    seed_count = int(fin31["seeds"])
    empirical_variance = float(fin31_seeds["selected_output_estimate"].var(ddof=1))
    mean_standard_error = math.sqrt(empirical_variance / seed_count)
    t29 = 2.045229642132703
    estimate = float(fin31["mean_selected_output"])
    total_shots = int(fin31["total_shots"])
    rows.append(
        base(
            readout_id="d31_integrated_30seed_distribution",
            configuration_id="ieee14_fullrect_d31_integrated_30seed_lambda_0p068",
            polynomial_degree=31,
            variant=(
                "integrated joint postselection+sign estimator; exact-distribution "
                "multinomial sampling (not backend shots at scale)"
            ),
            evidence_status="sampled_distribution",
            backend="NumPy multinomial from verified circuit distribution; 1,000-shot Aer smoke",
            attempted_shots=total_shots,
            postselection_accepted_shots=round(
                float(fin31["mean_estimated_qsvt_postselection_accepted_shots"]) * seed_count
            ),
            readout_accepted_shots=round(
                float(fin31["mean_interference_accepted_shots"]) * seed_count
            ),
            postselection_probability=float(fin31["mean_postselection_probability"]),
            postselection_probability_definition="inferred QSVT postselection p_hat=2f_hat-1",
            branch_probability=float(fin31["mean_interference_accepted_shots"])
            / float(fin31["shots_per_seed"]),
            branch_probability_definition="measured interference-branch acceptance f_hat",
            conditional_acceptance_probability=float(
                ((fin31_seeds["conditional_signed_mean"] + 1.0) / 2.0).mean()
            ),
            conditional_acceptance_probability_definition=(
                "mean Pr[positive sign | accepted interference branch]"
            ),
            selected_output_estimate=estimate,
            statevector_reference=float(fin31["exact_qsvt_output"]),
            matched_ridge_reference=float(fin31["matched_ridge_output"]),
            standard_error=mean_standard_error,
            standard_error_scope="empirical standard error of the 30-seed mean",
            confidence_interval_lower=estimate - t29 * mean_standard_error,
            confidence_interval_upper=estimate + t29 * mean_standard_error,
            confidence_interval_scope="Student-t CI for the mean across 30 simulator seeds",
            relative_ci_half_width=t29 * mean_standard_error / abs(estimate),
            analytic_variance=float(
                np.mean(fin31_seeds["selected_output_standard_error"] ** 2) / seed_count
            ),
            analytic_variance_scope="analytic shot variance of the 30-seed mean",
            empirical_seed_variance=empirical_variance,
            empirical_seed_variance_scope="sample variance across 30 seed estimates",
            empirical_mean_standard_error=mean_standard_error,
            seed_count=seed_count,
            estimator_definition=(
                "y=scale*z_hat; p_post_hat=2f_hat-1; per-run SE=scale*sqrt((f_hat-z_hat^2)/N)"
            ),
            artifact_paths=";".join(
                [
                    rel(TQE_REV / "full_rectangular_finite_shot.csv"),
                    rel(TQE_REV / "full_rectangular_finite_shot_seeds.csv"),
                    rel(TQE_REV / "full_rectangular_finite_shot_metadata.json"),
                ]
            ),
            notes=(
                "The source mean_ci95 columns average individual-run interval endpoints; "
                "they are not a CI for the mean and are not reused here."
            ),
        )
    )

    for readout_id, config_id, path in (
        ("d31_selected4x4_integrated", "selected4x4_d31_integrated_readout", PHASE8),
        ("d31_selected8x8_integrated", "selected8x8_d31_integrated_readout", PHASE9),
    ):
        summary = pd.read_csv(path / "integrated_readout_summary.csv")
        top = summary[
            (summary["observable_label"] == "state_correction_0")
            & (summary["shots"] == summary["shots"].max())
        ].iloc[0]
        per_seed = pd.read_csv(path / "integrated_readout_per_seed.csv")
        per_seed = per_seed[
            (per_seed["observable_label"] == "state_correction_0")
            & (per_seed["shots"] == int(top["shots"]))
        ]
        seed_count = int(top["num_seeds"])
        attempted = int(top["shots"]) * seed_count
        empirical_variance = float(per_seed["recovered_physical_functional"].var(ddof=1))
        standard_error = math.sqrt(empirical_variance / seed_count)
        estimate = float(top["mean_recovered_physical_functional"])
        rows.append(
            base(
                readout_id=readout_id,
                configuration_id=config_id,
                polynomial_degree=31,
                variant="integrated residual-QSVT-postselection-readout Aer chain",
                evidence_status="sampled_simulator",
                backend="aer_integrated_circuit_shot_sampling",
                attempted_shots=attempted,
                postselection_accepted_shots=round(
                    float(top["mean_measured_postselection_probability"]) * attempted
                ),
                readout_accepted_shots=int(per_seed["accepted_attempts"].sum()),
                postselection_probability=float(top["mean_measured_postselection_probability"]),
                postselection_probability_definition="measured QSVT p_hat=2f_hat-1",
                branch_probability=float(per_seed["accepted_attempts"].sum()) / attempted,
                branch_probability_definition="observed interference-branch acceptance",
                conditional_acceptance_probability=float(
                    ((per_seed["readout_sign_mean_accepted"] + 1.0) / 2.0).mean()
                ),
                conditional_acceptance_probability_definition=(
                    "mean Pr[positive sign | accepted interference branch]"
                ),
                selected_output_estimate=estimate,
                statevector_reference=float(top["exact_qsvt_statevector_functional"]),
                matched_ridge_reference=float(top["exact_ridge_functional"]),
                standard_error=standard_error,
                standard_error_scope="empirical standard error of the 30-seed mean",
                confidence_interval_lower=estimate - t29 * standard_error,
                confidence_interval_upper=estimate + t29 * standard_error,
                confidence_interval_scope="Student-t CI for the mean across 30 Aer seeds",
                relative_ci_half_width=t29 * standard_error / abs(estimate),
                analytic_variance=float(
                    np.mean(per_seed["recovered_physical_functional_standard_error"] ** 2)
                    / seed_count
                ),
                analytic_variance_scope="analytic shot variance of the 30-seed mean",
                empirical_seed_variance=empirical_variance,
                empirical_seed_variance_scope="sample variance across 30 Aer seed estimates",
                empirical_mean_standard_error=standard_error,
                seed_count=seed_count,
                estimator_definition="joint postselection/sign y=physical_scale*z_hat",
                artifact_paths=";".join(
                    [
                        rel(path / "integrated_readout_summary.csv"),
                        rel(path / "integrated_readout_per_seed.csv"),
                    ]
                ),
                notes="Postselection and interference-branch acceptance are distinct.",
            )
        )

    multi = pd.read_csv(GEN / "ieee14_multioutput_backend_shots.csv")
    for _, source in multi.iterrows():
        standard_error = (float(source["ci_high"]) - float(source["ci_low"])) / 3.92
        estimate = float(source["y_backend_shot"])
        rows.append(
            base(
                readout_id=f"d255_multioutput_{str(source['output']).replace(' ', '_')}",
                configuration_id="ieee14_fullrect_d255_multioutput",
                polynomial_degree=255,
                variant="isolated overlap Hadamard test per preselected output",
                evidence_status="sampled_simulator",
                backend=str(source["backend"]),
                attempted_shots=int(source["shots"]),
                readout_accepted_shots=int(source["shots"]),
                selected_output_estimate=estimate,
                statevector_reference=float(source["y_statevector"]),
                matched_ridge_reference=float(source["y_ridge"]),
                standard_error=standard_error,
                standard_error_scope="analytic SE of one finite-shot estimate",
                confidence_interval_lower=float(source["ci_low"]),
                confidence_interval_upper=float(source["ci_high"]),
                confidence_interval_scope="normal-approximation CI for one estimate",
                relative_ci_half_width=1.96 * standard_error / abs(estimate),
                analytic_variance=standard_error**2,
                analytic_variance_scope="variance of one selected-output estimate",
                seed_count=1,
                estimator_definition="y=||dx|| mu_hat; no acceptance step",
                artifact_paths=rel(GEN / "ieee14_multioutput_backend_shots.csv"),
                notes=(
                    "The area aggregate retains poor relative precision; broad-CI "
                    "containment is not called numerical agreement."
                ),
            )
        )

    mlae = pd.read_csv(GEN / "postselection_mitigation_executed_results.csv")
    for _, source in mlae.iterrows():
        rows.append(
            base(
                readout_id=f"controlled_{source['method']}_{source['true_amplitude']}",
                configuration_id="controlled_mlae_aer",
                polynomial_degree=not_applicable,
                variant=f"{source['method']} on a controlled amplitude",
                evidence_status="sampled_simulator",
                backend="Aer",
                attempted_shots=int(source["shots"]),
                readout_accepted_shots=int(source["shots"]),
                selected_output_estimate=float(source["estimate"]),
                statevector_reference=float(source["true_amplitude"]),
                matched_ridge_reference=not_applicable,
                standard_error=NOT_ESTIMATED,
                standard_error_scope=NOT_ESTIMATED,
                confidence_interval_lower=NOT_ESTIMATED,
                confidence_interval_upper=NOT_ESTIMATED,
                confidence_interval_scope=NOT_ESTIMATED,
                relative_ci_half_width=NOT_ESTIMATED,
                analytic_variance=NOT_ESTIMATED,
                analytic_variance_scope=NOT_ESTIMATED,
                seed_count=1,
                estimator_definition=str(source["method"]),
                artifact_paths=rel(GEN / "postselection_mitigation_executed_results.csv"),
                notes=(
                    str(source["note"])
                    + "; this controlled case is not integrated IEEE-14 mitigation"
                ),
            )
        )
    return rows


def readout_variance_validation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        *,
        readout_id: str,
        configuration_id: str,
        analytic_variance: float,
        empirical_variance: float,
        seed_count: int,
        estimate: float,
        statevector: float,
        ridge: float | str,
        artifact_paths: str,
        scope: str,
    ) -> None:
        ratio = empirical_variance / analytic_variance
        rows.append(
            {
                "readout_id": readout_id,
                "configuration_id": configuration_id,
                "analytic_single_run_variance": analytic_variance,
                "empirical_seed_variance": empirical_variance,
                "seed_count": seed_count,
                "variance_ratio_empirical_to_analytic": ratio,
                "analytic_single_run_standard_error": math.sqrt(analytic_variance),
                "empirical_seed_standard_deviation": math.sqrt(empirical_variance),
                "selected_output_estimate": estimate,
                "statevector_reference": statevector,
                "matched_ridge_reference": ridge,
                "absolute_error_vs_statevector": abs(estimate - statevector),
                "absolute_error_vs_ridge": (
                    abs(estimate - float(ridge)) if isinstance(ridge, (int, float)) else ridge
                ),
                "comparison_scope": scope,
                "artifact_paths": artifact_paths,
                "validation_status": (
                    "variance_consistent" if 0.1 <= ratio <= 10.0 else "variance_mismatch"
                ),
                "notes": (
                    "Variance consistency is assessed separately from estimate accuracy; "
                    "a broad CI alone is not called agreement."
                ),
            }
        )

    variance = pd.read_csv(FINAL / "final_readout_variance_checks.csv").iloc[0]
    shot = pd.read_csv(FINAL / "high_shot_backend_summary.csv").iloc[0]
    shot_runs = pd.read_csv(FINAL / "high_shot_backend_runs.csv")
    add(
        readout_id="d255_integrated_branch_hadamard",
        configuration_id="ieee14_fullrect_d255_shot_readout",
        analytic_variance=float(variance["variance_per_shot"]) / int(shot["shots_per_seed"]),
        empirical_variance=float(shot_runs["selected_output_estimate"].var(ddof=1)),
        seed_count=int(shot["seed_count"]),
        estimate=float(shot["aggregate_selected_output_estimate"]),
        statevector=float(shot["statevector_selected_output"]),
        ridge=float(shot["ridge_selected_output"]),
        artifact_paths=";".join(
            [
                rel(FINAL / "final_readout_variance_checks.csv"),
                rel(FINAL / "high_shot_backend_runs.csv"),
            ]
        ),
        scope="five separate 1,000,000-shot integrated Hadamard estimates",
    )

    wpj = pd.read_csv(GEN / "ieee14_high_precision_backend_summary.csv")
    for _, source in wpj.iterrows():
        add(
            readout_id=f"d255_isolated_wpj_{int(source['shots'])}",
            configuration_id="ieee14_fullrect_d255_isolated_readout_wpj",
            analytic_variance=float(source["theoretical_variance_per_shot"]),
            empirical_variance=float(source["empirical_variance"]),
            seed_count=int(source["n_seeds"]),
            estimate=float(source["aggregate_y_estimate"]),
            statevector=float(source["y_statevector"]),
            ridge=float(source["y_ridge"]),
            artifact_paths=rel(GEN / "ieee14_high_precision_backend_summary.csv"),
            scope=f"six separate {int(source['shots'])}-shot isolated estimates",
        )

    fin31 = pd.read_csv(TQE_REV / "full_rectangular_finite_shot.csv").iloc[0]
    fin31_seeds = pd.read_csv(TQE_REV / "full_rectangular_finite_shot_seeds.csv")
    add(
        readout_id="d31_integrated_30seed_distribution",
        configuration_id="ieee14_fullrect_d31_integrated_30seed_lambda_0p068",
        analytic_variance=float(np.mean(fin31_seeds["selected_output_standard_error"] ** 2)),
        empirical_variance=float(fin31_seeds["selected_output_estimate"].var(ddof=1)),
        seed_count=int(fin31["seeds"]),
        estimate=float(fin31["mean_selected_output"]),
        statevector=float(fin31["exact_qsvt_output"]),
        ridge=float(fin31["matched_ridge_output"]),
        artifact_paths=";".join(
            [
                rel(TQE_REV / "full_rectangular_finite_shot.csv"),
                rel(TQE_REV / "full_rectangular_finite_shot_seeds.csv"),
            ]
        ),
        scope="30 separate 100,000-draw exact-distribution estimates",
    )

    for readout_id, configuration_id, path in (
        ("d31_selected4x4_integrated", "selected4x4_d31_integrated_readout", PHASE8),
        ("d31_selected8x8_integrated", "selected8x8_d31_integrated_readout", PHASE9),
    ):
        summary = pd.read_csv(path / "integrated_readout_summary.csv")
        top = summary[
            (summary["observable_label"] == "state_correction_0")
            & (summary["shots"] == summary["shots"].max())
        ].iloc[0]
        per_seed = pd.read_csv(path / "integrated_readout_per_seed.csv")
        per_seed = per_seed[
            (per_seed["observable_label"] == "state_correction_0")
            & (per_seed["shots"] == int(top["shots"]))
        ]
        add(
            readout_id=readout_id,
            configuration_id=configuration_id,
            analytic_variance=float(
                np.mean(per_seed["recovered_physical_functional_standard_error"] ** 2)
            ),
            empirical_variance=float(per_seed["recovered_physical_functional"].var(ddof=1)),
            seed_count=int(top["num_seeds"]),
            estimate=float(top["mean_recovered_physical_functional"]),
            statevector=float(top["exact_qsvt_statevector_functional"]),
            ridge=float(top["exact_ridge_functional"]),
            artifact_paths=";".join(
                [
                    rel(path / "integrated_readout_summary.csv"),
                    rel(path / "integrated_readout_per_seed.csv"),
                ]
            ),
            scope=f"30 separate {int(top['shots'])}-shot integrated Aer estimates",
        )
    return rows


# --------------------------------------------------------------------------- #
# Generated manuscript tables
# --------------------------------------------------------------------------- #

STATUS_LABELS = {
    "classical_exact": "classical exact",
    "qsvt_matrix_action": "dense QSVT matrix action",
    "statevector_dense": "dense statevector action",
    "sampled_distribution": "exact-distribution sampling",
    "sampled_simulator": "Aer shot execution",
    "modeled": "modeled",
}


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_evidence_status_table(registry: list[dict[str, Any]]) -> None:
    keep = [
        "ieee14_fullrect_d255_useful_overlap",
        "ieee14_fullrect_d255_shot_readout",
        "ieee14_fullrect_d255_isolated_readout_wpj",
        "ieee14_fullrect_d255_robustness_sweep",
        "ieee30_fullrect_generalized_sweep_seed123",
        "ieee57_fullrect_escalation_seed123",
        "ieee14_fullrect_d31_degree_aware_lambda_0p068",
        "ieee14_fullrect_d31_integrated_30seed_lambda_0p068",
        "ieee_quantum_sparse_oracle_model",
        "ieee14_postselection_mitigation_model",
    ]
    by_id = {row["configuration_id"]: row for row in registry}
    lines = [
        "% GENERATED by scripts/build_tqe_evidence_registry.py -- do not hand-edit.",
        "% Source: outputs/tqe_blocking_revision/evidence_registry.csv",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Registry-generated evidence status for the headline configurations."
        r" Every row carries a stable configuration ID; statuses are never merged.}",
        r"\label{tab:registry_evidence_status}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}p{0.48\textwidth}p{0.10\textwidth}"
        r"p{0.05\textwidth}p{0.28\textwidth}@{}}",
        r"\toprule",
        r"Configuration ID & Case & $d$ & Evidence status \\",
        r"\midrule",
    ]
    for config_id in keep:
        row = by_id[config_id]
        short = (
            config_id.replace("ieee14_", "i14_")
            .replace("ieee30_", "i30_")
            .replace("ieee57_", "i57_")
        )
        degree = row["polynomial_degree"] if row["polynomial_degree"] != "" else "--"
        status = STATUS_LABELS.get(row["evidence_status"], row["evidence_status"])
        case = tex_escape(str(row["case"]).split(" ")[0])
        lines.append(rf"\path{{{short}}} & {case} & {degree} & {tex_escape(status)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "registry_evidence_status.tex").write_text("\n".join(lines), "utf-8")


def write_final_configurations_table(registry: list[dict[str, Any]]) -> None:
    keep = [
        ("ieee14_fullrect_d255_useful_overlap", "i14-d255-overlap"),
        ("ieee14_fullrect_d255_shot_readout", "i14-d255-shots"),
        ("ieee14_fullrect_d31_degree_aware_lambda_0p068", "i14-d31-0.068"),
        ("ieee14_fullrect_d31_integrated_30seed_lambda_0p068", "i14-d31-30seed"),
        ("ieee30_fullrect_generalized_sweep_seed123", "i30-sweep-best"),
        ("ieee57_fullrect_escalation_seed123", "i57-escal-best"),
    ]
    by_id = {row["configuration_id"]: row for row in registry}
    lines = [
        "% GENERATED by scripts/build_tqe_evidence_registry.py -- do not hand-edit.",
        "% Source: outputs/tqe_blocking_revision/evidence_registry.csv",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Registry-generated final configurations. Short IDs abbreviate the"
        r" registry configuration IDs (Table~\ref{tab:registry_evidence_status});"
        r" IEEE-30/57 rows show the best passing sweep configuration under the"
        r" matched-$\lambda$ execution-accuracy criterion, not the IEEE-14"
        r" benchmark-anchored application criterion.}",
        r"\label{tab:registry_final_configurations}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llllllllll@{}}",
        r"\toprule",
        r"ID & Case & Shape & $\lambda$ & $\alpha$ & $\beta$ & $C$ & $d$ & "
        r"Postselection/quadrature & Status \\",
        r"\midrule",
    ]
    for config_id, short in keep:
        row = by_id[config_id]

        def num(value: Any, fmt: str = "{:.4g}") -> str:
            try:
                return fmt.format(float(value))
            except (TypeError, ValueError):
                return "--"

        status = STATUS_LABELS.get(row["evidence_status"], row["evidence_status"])
        lines.append(
            " & ".join(
                [
                    tex_escape(short),
                    tex_escape(str(row["case"]).split(" ")[0]),
                    tex_escape(str(row["matrix_shape"]).split(" ")[0]),
                    num(row["lambda"]),
                    num(row["alpha"]),
                    num(row["beta"]),
                    num(row["contraction_C"]),
                    str(row["polynomial_degree"]),
                    num(row["postselection_probability"]),
                    tex_escape(status),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "registry_final_configurations.tex").write_text("\n".join(lines), "utf-8")


def write_degree255_ledger_table() -> None:
    ledger = pd.read_csv(FINAL / "degree255_resource_ledger.csv")
    cfg = read_json(FINAL / "final_scientific_configuration.json")
    degree = int(cfg["degree"])
    display = {
        "logical_qubits_statevector": "statevector qubits",
        "polynomial_degree": "polynomial degree",
        "projector_phases": "projector phases ($N_\\phi=d{+}1$)",
        "total_hadamard_shots": "Aer Hadamard shots",
        "transpiled_hadamard_depth": "opaque-unitary circuit depth (layers)",
        "direct_rejection_expected_attempts_per_success": (
            "direct-rejection attempts/$p_{\\rm quadrature}$"
        ),
        "qrom_proxy_status": "QROM residual loader",
        "fault_tolerant_physical_overhead": "fault-tolerant overhead",
    }
    lines = [
        "% GENERATED by scripts/build_tqe_evidence_registry.py from",
        "% outputs/final_useful_overlap_validation/degree255_resource_ledger.csv",
        "% (values unchanged; labels carry units). Do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Degree-255 resource ledger for configuration"
        r" \texttt{ieee14\_fullrect\_d255\_useful\_overlap} (executed,"
        r" transpiled, modeled, and"
        r" excluded entries separated). The transpiled depth counts opaque-unitary"
        r" instruction layers, not elementary gates; the modeled attempts row is"
        r" $1/p_{\rm quadrature}$ and is not executed.}",
        r"\label{tab:final_degree255_resources}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}p{0.18\columnwidth}p{0.49\columnwidth}"
        r"p{0.26\columnwidth}@{}}",
        r"\toprule",
        r"Category & Item & Value \\",
        r"\midrule",
        rf"EXECUTED & signal-unitary calls per attempt ($N_U=d$) & {degree} \\",
    ]
    for _, row in ledger.iterrows():
        item = display.get(str(row["item"]), str(row["item"]).replace("_", " "))
        value = row["value"]
        try:
            fvalue = float(value)
            value_text = (
                f"{fvalue:.4f}" if abs(fvalue - round(fvalue)) > 1e-9 else f"{round(fvalue)}"
            )
        except (TypeError, ValueError):
            value_text = tex_escape(str(value))
        lines.append(f"{row['category']} & {item} & {value_text} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "final_degree255_resource_ledger.tex").write_text("\n".join(lines), "utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    registry = registry_rows()
    write_rows_csv(OUT / "evidence_registry.csv", registry, REGISTRY_COLUMNS)
    write_json_rows(OUT / "evidence_registry.json", registry)

    inventory_paths: list[Path] = [
        ROOT / "manuscript" / "main.tex",
        ROOT / "manuscript" / "main.pdf",
        ROOT / "manuscript" / "supplementary_material.tex",
        ROOT / "manuscript" / "supplementary_material.pdf",
    ]
    for row in registry:
        for part in str(row["artifact_paths"]).split(";"):
            if part:
                inventory_paths.append(ROOT / part)
    for extra in (
        FINAL / "final_scientific_configuration.json",
        FINAL / "degree255_resource_ledger.csv",
        FINAL / "final_readout_variance_checks.csv",
        FINAL / "final_same_configuration_error_budget.csv",
        GEN / "preregistered_criteria.yaml",
        GEN / "generalized_resource_ledger.csv",
        GEN / "generalized_error_budget.csv",
        GEN / "rectangular_convention_derivation.md",
        GEN / "evidence_status_matrix.csv",
        TQE_REV / "end_to_end_resource_ledger.csv",
    ):
        inventory_paths.append(extra)
    inventory = build_inventory(inventory_paths)
    inventory_columns = [
        "artifact_id",
        "path",
        "exists",
        "type",
        "modified_time",
        "sha256",
        "producer",
        "status",
    ]
    write_rows_csv(OUT / "artifact_inventory.csv", inventory, inventory_columns)
    write_json_rows(OUT / "artifact_inventory.json", inventory)

    ledger = resource_rows(registry)
    ledger_columns = list(ledger[0].keys())
    write_rows_csv(OUT / "resource_ledger.csv", ledger, ledger_columns)
    write_json_rows(OUT / "resource_ledger.json", ledger)
    (OUT / "resource_accounting_notes.md").write_text(
        "\n".join(
            [
                "# Resource-Accounting Notes (Phase 3)",
                "",
                "All degree-dependent counts in `resource_ledger.csv` are DERIVED from",
                "the configuration degree, never hand-entered:",
                "",
                "    N_U = d        signal-unitary calls per attempt (U_A / U_A^dagger)",
                "    N_phi = d + 1  projector-phase operations per attempt",
                "    L_alt = 2d + 1 total alternating sequence length (NOT a query count)",
                "    N_attempt = ceil(N_accepted / p_succ)   (direct rejection sampling)",
                "    N_U_total = d * N_attempt",
                "",
                "Verified against the implementation "
                "(`robust_qsvt_se.qsvt.rectangular_convention.pcphase_qsvt_operator`).",
                "",
                "Separations that are never merged:",
                "- target-quadrature probability (d=255 Hadamard branch: 0.0028556; the",
                "  modeled 350.19 attempts/success row) vs encoded-prefix probability",
                "  (0.6028) vs interference-branch acceptance (d=31 chain: 0.924) vs",
                "  QSVT postselection probability (d=31: 0.8483616).",
                "- executed opaque-unitary applications vs decomposition-level signal",
                "  calls (the d=255 Aer circuits apply ONE opaque operator per shot).",
                "- 'not_estimated' marks absent costs; absent costs are never zero.",
                "- The epsilon=1e-2 sensitivity model (2,947 attempts, 91,357 signal",
                "  calls, 94,304 phases, 2,947 reloads) belongs to configuration",
                "  ieee14_fullrect_d31_integrated_30seed_lambda_0p068 (d=31,",
                "  lambda=0.068, p_succ=0.8483616) and is recomputed here from those",
                "  values; it must never be attributed to the d=255 configuration.",
                "- Fault-tolerant physical overhead is excluded everywhere, not zero.",
                "",
            ]
        ),
        "utf-8",
    )

    matrix = ieee_matrix(registry)
    matrix_columns = list(matrix[0].keys())
    write_rows_csv(OUT / "ieee_case_evidence_matrix.csv", matrix, matrix_columns)
    md_lines = ["# IEEE Case Evidence Matrix (Phase 4)", ""]
    header = "| " + " | ".join(matrix_columns) + " |"
    md_lines += [header, "|" + "---|" * len(matrix_columns)]
    for row in matrix:
        md_lines.append("| " + " | ".join(str(row[c]) for c in matrix_columns) + " |")
    md_lines += [
        "",
        "Every 'yes' cites artifact paths via the registry rows named in the cell;",
        "see `evidence_registry.csv` for the full paths and configuration IDs.",
        "",
    ]
    (OUT / "ieee_case_evidence_matrix.md").write_text("\n".join(md_lines), "utf-8")

    conv = convention_validation()
    conv.to_csv(OUT / "convention_validation.csv", index=False)
    gen_held = conv[conv["campaign"] == "generalized_heldout"]
    fin_held = conv[conv["campaign"] == "final_heldout"]
    strict = conv[conv["campaign"] == "strict_live"]
    if not (strict["status"] == "pass").all():
        raise AssertionError("strict live convention validation failed")
    (OUT / "convention_validation_summary.md").write_text(
        "\n".join(
            [
                "# Convention Validation Summary (Phase 5)",
                "",
                "Convention status: `formally_derived_and_independently_validated`.",
                "",
                "Two independent campaigns, kept separate (never merged):",
                "",
                f"- generalized campaign: {len(gen_held)} held-out real rectangular",
                "  matrices (7 dims x 7 spectral families, reserved seed range",
                "  [770000,779999]), 9 distinct odd degrees {1,3,5,7,15,31,63,127,255}",
                "  (11 odd rows), 6 even probes rejected by the API, 245 complex probes",
                "  unsupported, 5 symbolic identity checks.",
                f"- final campaign: {len(fin_held)} held-out matrices, 8 degree rows,",
                "  5 independent-evaluator rows.",
                f"- strict live boundary checks: {len(strict)} rows, including tall",
                "  and wide matrices, d=1 mod 4 and d=3 mod 4, repeated/near-zero/zero",
                "  singular values, random orthogonal singular vectors, residual-state",
                "  global-phase invariance, and explicit even/complex/nonfinite/length",
                "  rejection checks.",
                "",
                "Scope of the validated rule: odd degree, real rectangular matrices,",
                "PyQSP sym_qsp plus-i phases -> dense-Julia PCPhase, global +pi/2 phase",
                "offset, signed-imaginary top-left extraction with sign (-1)^((d+1)/2).",
                "Complex matrices and even degrees are explicitly unsupported.",
                "The formal derivation is `docs/RECTANGULAR_QSVT_CONVENTION_DERIVATION.md`.",
                "",
            ]
        ),
        "utf-8",
    )

    readouts = readout_registry()
    write_rows_csv(OUT / "readout_registry.csv", readouts, READOUT_COLUMNS)
    write_json_rows(OUT / "readout_registry.json", readouts)
    variance = readout_variance_validation()
    write_rows_csv(OUT / "readout_variance_validation.csv", variance, list(variance[0].keys()))
    (OUT / "readout_audit.md").write_text(
        "\n".join(
            [
                "# Readout Audit (Phase 6)",
                "",
                "Canonical records are in `readout_registry.csv` and JSON. The schema",
                "never uses an undifferentiated 'success probability':",
                "",
                "- `postselection_probability` is the QSVT encoded-subspace event;",
                "- `branch_probability` is a Hadamard/interference branch event;",
                "- `quadrature_probability` is the squared selected quadrature and is",
                "  not an acceptance rate; and",
                "- `conditional_acceptance_probability` names a conditional sign event",
                "  where that event is present.",
                "",
                "Variants:",
                "",
                "1. Degree-255 full-rectangular statevector diagnostic: no shots and no",
                "   measurement-based full-vector recovery.",
                "2. Degree-255 integrated branch Hadamard test: controlled QSVT acts as",
                "   one opaque unitary inside each of 5,000,000 Aer shots; pooled 95%",
                "   CI relative half-width 0.0832. The separate 5,000,000-shot",
                "   encoded-prefix diagnostic has its own accepted count (3,014,620).",
                "3. Degree-255 isolated WP-J readout: StatePreparation loads the",
                "   classically computed update direction and QSVT does not act in",
                "   circuit. Six runs are recorded at each shot level. The source",
                "   value 0.0096 is a one-million-shot single-run analytic relative CI",
                "   width; after pooling six equal runs the registry value is about",
                "   0.0039. Neither is relabeled as seed-to-seed variation.",
                "4. Degree-31 full-rectangular 30-seed record: 3,000,000 multinomial",
                "   draws from the verified exact circuit distribution, with one",
                "   1,000-shot Aer smoke run; status `sampled_distribution`, not",
                "   backend-shot execution at scale.",
                "5. Degree-31 4x4 and 8x8 integrated chains: 30,000,000 Aer shots each",
                "   at the largest shot level; QSVT-postselection and interference-",
                "   branch accepted counts are separate.",
                "6. Three degree-255 isolated multioutput rows retain the imprecise",
                "   area-aggregate result; a broad interval is not called agreement.",
                "7. MLAE and direct sampling are executed only on controlled amplitudes;",
                "   IEEE-14 mitigation remains modeled.",
                "",
                "Uncertainty separations enforced:",
                "- analytic variance of one finite-shot estimate;",
                "- analytic variance after pooling equal independent shot records;",
                "- empirical variance across simulator seeds; and",
                "- Student-t confidence intervals for a mean across seeds.",
                "The older degree-31 source columns average per-seed CI endpoints; that",
                "quantity is not a CI for the 30-seed mean and is not reused as one.",
                "`readout_variance_validation.csv` compares like-for-like single-run",
                "analytic and seed variances before any mean uncertainty is formed.",
                "",
            ]
        ),
        "utf-8",
    )

    write_evidence_status_table(registry)
    write_final_configurations_table(registry)
    write_degree255_ledger_table()

    print(f"registry rows: {len(registry)}")
    print(f"resource ledger rows: {len(ledger)}")
    print(f"readout rows: {len(readouts)}")
    print(f"inventory rows: {len(inventory)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
