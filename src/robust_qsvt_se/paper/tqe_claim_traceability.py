"""Task E: claim-to-output traceability manifest for the TQE revision.

Links each manuscript claim to the code, configs, outputs, and tests that
support it, with a controlled support status and the manuscript-safe vs.
to-avoid wording. The goal is reviewer-proofing: every claim shows where it
comes from and whether it is supported, supported-with-limitations, a scope
boundary, future work, or unsupported.

Referenced paths are existence-checked at build time, so ``evidence_exists``
reflects the repository state rather than an assertion. The dedicated
``avoid_wording`` column intentionally lists phrasing to avoid; the
``manuscript_safe_wording`` column is verified to contain none of it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper.tqe_revision_support_common import (
    REVISION_OUTPUT_ROOT,
    count_truthy,
    find_forbidden,
    write_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory

CLAIM_TRACEABILITY_DIR = REVISION_OUTPUT_ROOT / "claim_traceability"

SUPPORT_STATUSES = frozenset(
    {"supported", "supported_with_limitations", "unsupported", "future_work", "scope_boundary"}
)
EVIDENCE_LAYERS = frozenset(
    {
        "classical_benchmark",
        "qsvt_target",
        "block_encoding",
        "readout",
        "resource",
        "nonlinear_ac",
        "measurement_model",
        "limitation",
    }
)

CLAIM_COLUMNS = [
    "claim_id",
    "manuscript_claim",
    "support_status",
    "evidence_layer",
    "source_files",
    "source_outputs",
    "source_configs",
    "source_tests",
    "evidence_exists",
    "evidence_present_fraction",
    "manuscript_safe_wording",
    "avoid_wording",
    "notes",
]


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    manuscript_claim: str
    support_status: str
    evidence_layer: str
    manuscript_safe_wording: str
    avoid_wording: str
    notes: str
    source_files: tuple[str, ...] = field(default_factory=tuple)
    source_outputs: tuple[str, ...] = field(default_factory=tuple)
    source_configs: tuple[str, ...] = field(default_factory=tuple)
    source_tests: tuple[str, ...] = field(default_factory=tuple)


CLAIMS: tuple[Claim, ...] = (
    Claim(
        claim_id="CLAIM-01",
        manuscript_claim="IEEE/PYPOWER cases provide benchmark network models, not field "
        "PMU/SCADA data.",
        support_status="supported",
        evidence_layer="measurement_model",
        manuscript_safe_wording="IEEE/PYPOWER cases provide controlled benchmark network "
        "models (buses, branches, admittances); they are not real PMU/SCADA field "
        "measurement streams.",
        avoid_wording="validated on real PMU/SCADA field data",
        notes="Documented in the measurement-model section.",
        source_files=(
            "docs/EXPERIMENT_MEASUREMENT_MODEL.md",
            "src/robust_qsvt_se/data/cases.py",
            "src/robust_qsvt_se/data/real_cases.py",
        ),
        source_outputs=("outputs/measurement_inventory",),
        source_configs=("configs/real_ieee14.yaml",),
        source_tests=("tests/test_real_cases.py",),
    ),
    Claim(
        claim_id="CLAIM-02",
        manuscript_claim="Measurement rows are generated from network equations.",
        support_status="supported",
        evidence_layer="measurement_model",
        manuscript_safe_wording="Measurement rows are generated from the AC/DC network "
        "equations and Jacobians of the benchmark cases.",
        avoid_wording="measurements collected from deployed sensors",
        notes="Generated measurement rows; AC/DC measurement builders.",
        source_files=(
            "src/robust_qsvt_se/measurement/ac_linear.py",
            "src/robust_qsvt_se/measurement/dc_linear.py",
            "docs/EXPERIMENT_MEASUREMENT_MODEL.md",
        ),
        source_outputs=("outputs/measurement_inventory", "outputs/measurement_redundancy"),
        source_configs=("configs/real_ieee14.yaml",),
        source_tests=("tests/test_ac_linear_model.py", "tests/test_dc_linear_model.py"),
    ),
    Claim(
        claim_id="CLAIM-03",
        manuscript_claim="Nonlinear AC experiments perturb raw generated measurements.",
        support_status="supported",
        evidence_layer="nonlinear_ac",
        manuscript_safe_wording="Nonlinear AC experiments perturb the raw generated "
        "measurement vector z = h(x_true) + e + b before forming per-iteration weighted "
        "update systems.",
        avoid_wording="nonlinear AC perturbs real field measurement streams",
        notes="Raw-measurement perturbation path.",
        source_files=(
            "src/robust_qsvt_se/experiments/iterative_ac.py",
            "src/robust_qsvt_se/measurement/perturbations.py",
        ),
        source_outputs=("outputs/nonlinear_ac_ieee14_seed10",),
        source_configs=("configs/nonlinear_ac_ieee14_seed10.yaml",),
        source_tests=("tests/test_iterative_ac.py",),
    ),
    Claim(
        claim_id="CLAIM-04",
        manuscript_claim="Single-step experiments perturb weighted residuals.",
        support_status="supported",
        evidence_layer="measurement_model",
        manuscript_safe_wording="Synthetic, DC, and AC-linearized single-step experiments "
        "perturb the already-weighted residual r_tilde, distinct from the raw-z nonlinear "
        "path.",
        avoid_wording="single-step experiments use the same path as nonlinear AC",
        notes="Single-step weighted-residual perturbation path.",
        source_files=(
            "src/robust_qsvt_se/measurement/perturbations.py",
            "docs/EXPERIMENT_MEASUREMENT_MODEL.md",
        ),
        source_outputs=("outputs/real_ieee14_seed10",),
        source_configs=("configs/ieee14_dc_sweeps.yaml",),
        source_tests=("tests/test_measurement_type_ablation.py",),
    ),
    Claim(
        claim_id="CLAIM-05",
        manuscript_claim="Ridge/Tikhonov suppresses small singular-value directions.",
        support_status="supported",
        evidence_layer="classical_benchmark",
        manuscript_safe_wording="The Ridge/Tikhonov spectral filter sigma/(sigma^2 + alpha) "
        "damps small singular-value directions relative to the unregularized pseudoinverse.",
        avoid_wording="Ridge is universally optimal across all scenarios",
        notes="Spectral-filter damping; validated against pseudoinverse.",
        source_files=(
            "src/robust_qsvt_se/estimators/ridge.py",
            "src/robust_qsvt_se/qsvt/filters.py",
        ),
        source_outputs=(
            "outputs/full_alpha_sensitivity_classical",
            "outputs/alpha_sensitivity_summary",
        ),
        source_configs=("configs/alpha_sensitivity_real_ieee14.yaml",),
        source_tests=("tests/test_filters.py", "tests/test_tqe_revision_evidence.py"),
    ),
    Claim(
        claim_id="CLAIM-06",
        manuscript_claim="QSVT-target is numerically equivalent to Ridge/Tikhonov when alpha "
        "is matched.",
        support_status="supported_with_limitations",
        evidence_layer="qsvt_target",
        manuscript_safe_wording="In the classical simulator, the QSVT-compatible target "
        "implements the same regularized spectral filter as Ridge/Tikhonov for matched alpha; "
        "it is a reference equivalence, not a superiority result.",
        avoid_wording="QSVT beats Ridge/Tikhonov numerically",
        notes="Matched-alpha equivalence; no QSVT-over-Ridge advantage.",
        source_files=(
            "src/robust_qsvt_se/estimators/qsvt_spectral.py",
            "src/robust_qsvt_se/qsvt/filters.py",
        ),
        source_outputs=("outputs/tqe_qsvt_additional_experiments/end_to_end_qsvt_vs_ridge",),
        source_configs=(),
        source_tests=("tests/test_tqe_end_to_end_qsvt_vs_ridge.py",),
    ),
    Claim(
        claim_id="CLAIM-07",
        manuscript_claim="QSVT phase/block evidence supports implementation-pathway discussion "
        "only.",
        support_status="supported_with_limitations",
        evidence_layer="block_encoding",
        manuscript_safe_wording="Phase-synthesis and block-encoding results are "
        "implementation-pathway evidence for selected subproblems, not full-scale execution.",
        avoid_wording="full IEEE-scale QSVT circuit executed",
        notes="Selected-subproblem phase/block evidence.",
        source_files=(
            "src/robust_qsvt_se/qsvt/phase_synthesis.py",
            "src/robust_qsvt_se/qsvt/block_encoding.py",
        ),
        source_outputs=("outputs/qsvt_phase_validation_paper", "outputs/qsvt_block_encoding"),
        source_configs=("configs/qsvt_phase_validation_paper.yaml",),
        source_tests=("tests/test_qsvt_phase_validation.py", "tests/test_block_encoding.py"),
    ),
    Claim(
        claim_id="CLAIM-08",
        manuscript_claim="Resource estimates are proxies, not synthesized hardware execution.",
        support_status="supported_with_limitations",
        evidence_layer="resource",
        manuscript_safe_wording="Qubit, query, depth, and cost figures are resource-model "
        "proxies for feasibility discussion; they are not measured on synthesized hardware.",
        avoid_wording="resource estimates are measured hardware execution results",
        notes="Resource/cost proxies; see Task C cost accounting.",
        source_files=(
            "src/robust_qsvt_se/qsvt/resource_estimator.py",
            "src/robust_qsvt_se/qsvt/hardware_resource_estimator.py",
        ),
        source_outputs=(
            "outputs/qsvt_oracle_model_resources",
            "outputs/hardware_aware_oracle_cost_model",
            "outputs/tqe_revision_support/cost_accounting",
        ),
        source_configs=("configs/qsvt_resource_full_ieee.yaml",),
        source_tests=(
            "tests/test_qsvt_resources.py",
            "tests/test_oracle_model_resource_estimator.py",
        ),
    ),
    Claim(
        claim_id="CLAIM-09",
        manuscript_claim="Sparse-access oracle is modeled, not compiled as a reversible circuit.",
        support_status="scope_boundary",
        evidence_layer="block_encoding",
        manuscript_safe_wording="The sparse index/value oracle is a modeled sparse-access "
        "emulator with a resource ledger; it is not compiled into a full reversible circuit.",
        avoid_wording="scalable reversible sparse oracle circuit implemented",
        notes="Sparse-access model + assumption ledger.",
        source_files=(
            "src/robust_qsvt_se/qsvt/sparse_access_oracle.py",
            "src/robust_qsvt_se/qsvt/sparse_oracle_assumption_ledger.py",
        ),
        source_outputs=(
            "outputs/qsvt_sparse_access_oracle",
            "outputs/sparse_oracle_assumption_ledger",
        ),
        source_configs=(),
        source_tests=(
            "tests/test_sparse_access_oracle.py",
            "tests/test_sparse_oracle_assumption_ledger.py",
        ),
    ),
    Claim(
        claim_id="CLAIM-10",
        manuscript_claim="State preparation is assumed/modeled, not efficiently synthesized.",
        support_status="scope_boundary",
        evidence_layer="resource",
        manuscript_safe_wording="Preparation of the weighted-residual state is an "
        "assumed/modeled amplitude or qRAM-style loader; no efficient loader is synthesized.",
        avoid_wording="efficient qRAM state-preparation loader implemented",
        notes="State-preparation assumption/model.",
        source_files=(
            "src/robust_qsvt_se/qsvt/state_preparation_model.py",
            "src/robust_qsvt_se/qsvt/gate_state_preparation.py",
        ),
        source_outputs=("outputs/qsvt_state_preparation_model",),
        source_configs=(),
        source_tests=(
            "tests/test_state_preparation_model.py",
            "tests/test_gate_state_preparation.py",
        ),
    ),
    Claim(
        claim_id="CLAIM-11",
        manuscript_claim="Basis sampling supports energy-style observables, not signed "
        "full-vector PSSE readout.",
        support_status="supported_with_limitations",
        evidence_layer="readout",
        manuscript_safe_wording="Basis sampling supports energy-style (squared-amplitude) "
        "observables; signed full-vector PSSE update readout is not provided by sampling "
        "alone.",
        avoid_wording="full-vector readout solved",
        notes="Energy-style observable readout; full-vector recovery out of scope.",
        source_files=(
            "src/robust_qsvt_se/qsvt/gate_observable_readout.py",
            "src/robust_qsvt_se/qsvt/readout_analysis.py",
            "src/robust_qsvt_se/paper/readout_limitation_formalization.py",
        ),
        source_outputs=(
            "outputs/qsvt_gate_observable_readout",
            "outputs/readout_limitation_formalization",
        ),
        source_configs=(),
        source_tests=(
            "tests/test_gate_observable_readout.py",
            "tests/test_readout_limitation_formalization.py",
        ),
    ),
    Claim(
        claim_id="CLAIM-12",
        manuscript_claim="The selected signed-readout diagnostic supports selected signed "
        "observables only.",
        support_status="supported_with_limitations",
        evidence_layer="readout",
        manuscript_safe_wording="A Hadamard-test-style sign-aware diagnostic estimates "
        "selected signed linear functionals of the matched-alpha update; it is selected "
        "signed observables only, not full-vector tomography.",
        avoid_wording="signed full-vector readout solved on hardware",
        notes="New Task B diagnostic; selected signed observables only.",
        source_files=("src/robust_qsvt_se/paper/signed_readout_diagnostic.py",),
        source_outputs=("outputs/tqe_revision_support/signed_readout",),
        source_configs=(),
        source_tests=("tests/test_signed_readout_diagnostic.py",),
    ),
    Claim(
        claim_id="CLAIM-13",
        manuscript_claim="Nonlinear AC results are per-iteration consistency checks, not "
        "nonlinear QSVT advantage.",
        support_status="scope_boundary",
        evidence_layer="nonlinear_ac",
        manuscript_safe_wording="Nonlinear AC results are per-iteration feasibility/consistency "
        "checks of the linear update; they are not a nonlinear QSVT-in-the-loop solver or an "
        "advantage claim.",
        avoid_wording="nonlinear QSVT solver demonstrates advantage",
        notes="Per-iteration feasibility, classical loop.",
        source_files=(
            "src/robust_qsvt_se/qsvt/tqe_nonlinear_ac_per_iteration_feasibility.py",
            "src/robust_qsvt_se/experiments/iterative_ac.py",
        ),
        source_outputs=(
            "outputs/tqe_qsvt_additional_experiments/nonlinear_ac_per_iteration_feasibility",
        ),
        source_configs=("configs/nonlinear_ac_ieee14_seed10.yaml",),
        source_tests=("tests/test_tqe_nonlinear_ac_per_iteration_feasibility.py",),
    ),
    Claim(
        claim_id="CLAIM-14",
        manuscript_claim="No quantum speedup is claimed.",
        support_status="scope_boundary",
        evidence_layer="limitation",
        manuscript_safe_wording="The study reports implementation-pathway, cost-accounting, and "
        "boundary evidence under matched alpha; it makes no claim of computational speedup over "
        "classical solvers.",
        avoid_wording="quantum speedup or quantum advantage demonstrated",
        notes="Hard scope boundary; enforced by claim-safety audits.",
        source_files=(
            "docs/EXPERIMENT_MEASUREMENT_MODEL.md",
            "src/robust_qsvt_se/paper/claim_boundary_writer.py",
            "src/robust_qsvt_se/paper/tqe_revision_support_common.py",
        ),
        source_outputs=(
            "outputs/final_qsvt_claim_safety_audit",
            "outputs/tqe_revision_support/claim_traceability",
        ),
        source_configs=(),
        source_tests=("tests/test_final_qsvt_claim_safety_audit.py", "tests/test_claim_lint.py"),
    ),
    Claim(
        claim_id="CLAIM-15",
        manuscript_claim="No full IEEE-scale hardware execution is claimed.",
        support_status="scope_boundary",
        evidence_layer="limitation",
        manuscript_safe_wording="Resource and cost figures are simulator/model-level proxies "
        "for selected subproblems; the study does not claim execution on full-scale IEEE "
        "quantum hardware.",
        avoid_wording="full IEEE-scale quantum hardware execution achieved",
        notes="Hard scope boundary; resource proxies only.",
        source_files=(
            "docs/qsvt_implementation_scope.md",
            "src/robust_qsvt_se/paper/tqe_revision_support_common.py",
        ),
        source_outputs=(
            "outputs/full_qsvt_ieee_hardware_resources",
            "outputs/tqe_revision_support/cost_accounting",
        ),
        source_configs=("configs/qsvt_resource_full_ieee.yaml",),
        source_tests=(
            "tests/test_qsvt_hardware_aware.py",
            "tests/test_final_qsvt_claim_safety_audit.py",
        ),
    ),
)


def build_claim_traceability(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", CLAIM_TRACEABILITY_DIR)))
    repo_root = Path(resolved.get("repo_root", "."))

    _validate_vocabulary()
    rows = [_claim_row(claim, repo_root) for claim in CLAIMS]
    frame = pd.DataFrame(rows, columns=CLAIM_COLUMNS)
    _self_check_safe_fields(frame)

    claim_csv = output_dir / "tqe_claim_traceability.csv"
    summary_md = output_dir / "tqe_claim_traceability.md"
    frame.to_csv(claim_csv, index=False)
    summary_md.write_text(_summary_markdown(frame), encoding="utf-8")

    artifacts = {
        "tqe_claim_traceability_csv": claim_csv,
        "tqe_claim_traceability_md": summary_md,
    }
    manifest = write_manifest(
        output_dir=output_dir,
        artifact_name="tqe_claim_traceability",
        description=(
            "Claim-to-output traceability manifest linking manuscript claims to code, "
            "configs, outputs, and tests with controlled support statuses."
        ),
        artifacts=artifacts,
        extra={
            "claim_count": len(frame),
            "support_status_counts": frame["support_status"].value_counts().to_dict(),
            "fully_traceable_claims": count_truthy(frame["evidence_exists"]),
            "support_status_vocabulary": sorted(SUPPORT_STATUSES),
            "evidence_layer_vocabulary": sorted(EVIDENCE_LAYERS),
        },
        manifest_name="tqe_claim_traceability_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {"output_dir": output_dir, "frame": frame, "artifacts": artifacts}


def _claim_row(claim: Claim, repo_root: Path) -> dict[str, Any]:
    references = (
        claim.source_files + claim.source_outputs + claim.source_configs + claim.source_tests
    )
    present = sum(1 for ref in references if (repo_root / ref).exists())
    total = len(references)
    if total == 0:
        evidence_exists = "false"
    elif present == total:
        evidence_exists = "true"
    elif present == 0:
        evidence_exists = "false"
    else:
        evidence_exists = "partial"
    return {
        "claim_id": claim.claim_id,
        "manuscript_claim": claim.manuscript_claim,
        "support_status": claim.support_status,
        "evidence_layer": claim.evidence_layer,
        "source_files": "; ".join(claim.source_files),
        "source_outputs": "; ".join(claim.source_outputs),
        "source_configs": "; ".join(claim.source_configs),
        "source_tests": "; ".join(claim.source_tests),
        "evidence_exists": evidence_exists,
        "evidence_present_fraction": f"{present}/{total}",
        "manuscript_safe_wording": claim.manuscript_safe_wording,
        "avoid_wording": claim.avoid_wording,
        "notes": claim.notes,
    }


def _summary_markdown(frame: pd.DataFrame) -> str:
    status_counts = frame["support_status"].value_counts().to_dict()
    lines = [
        "# TQE Claim-to-Output Traceability",
        "",
        "Each manuscript claim is linked to the code, configs, outputs, and tests that "
        "support it, with a controlled support status. Referenced paths are existence-checked; "
        "`evidence_exists` reflects the repository state. The `avoid_wording` column is a "
        "do-not-use list, not a set of supported claims.",
        "",
        "## Support-Status Summary",
        "",
        "| Support status | Claims |",
        "| --- | --- |",
    ]
    for status in sorted(SUPPORT_STATUSES):
        lines.append(f"| {status} | {int(status_counts.get(status, 0))} |")
    lines += [
        "",
        "## Claims",
        "",
        "| ID | Claim | Status | Layer | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['claim_id']} | {row['manuscript_claim']} | {row['support_status']} | "
            f"{row['evidence_layer']} | {row['evidence_exists']} "
            f"({row['evidence_present_fraction']}) |"
        )
    lines += [
        "",
        "## Manuscript-Safe Wording",
        "",
    ]
    for _, row in frame.iterrows():
        lines.append(f"- **{row['claim_id']}**: {row['manuscript_safe_wording']}")
    lines += [
        "",
        "## Boundary",
        "",
        "Scope-boundary and supported-with-limitations claims are not promoted to unqualified "
        "support. No claim asserts quantum speedup, QSVT-over-Ridge superiority, full-scale "
        "hardware execution, solved full-vector readout, or field-data validation.",
        "",
    ]
    return "\n".join(lines)


def _validate_vocabulary() -> None:
    for claim in CLAIMS:
        if claim.support_status not in SUPPORT_STATUSES:
            raise ValueError(f"{claim.claim_id} has invalid support_status {claim.support_status}")
        if claim.evidence_layer not in EVIDENCE_LAYERS:
            raise ValueError(f"{claim.claim_id} has invalid evidence_layer {claim.evidence_layer}")


def _self_check_safe_fields(frame: pd.DataFrame) -> None:
    safe_text = "\n".join(frame["manuscript_safe_wording"].astype(str).tolist())
    violations = find_forbidden(safe_text)
    if violations:
        raise RuntimeError(f"claim safe wording contains forbidden phrases: {violations}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the TQE claim-traceability manifest")
    parser.add_argument("--output-dir", default=str(CLAIM_TRACEABILITY_DIR))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    run = build_claim_traceability({"output_dir": args.output_dir, "repo_root": args.repo_root})
    frame = run["frame"]
    fully = count_truthy(frame["evidence_exists"])
    print(
        f"Claim traceability complete ({len(frame)} claims, {fully} fully traceable) -> "
        f"{run['artifacts']['tqe_claim_traceability_csv']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
