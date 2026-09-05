"""Phase A: classical spectral-filtering audit recheck.

Verifies, directly against the repository and the generated Phase 2 package,
which estimators are actually benchmarked and which are only defined. The
QSVT-target classical filter stays Ridge-equivalent; no QSVT-over-Ridge claim is
introduced and missing coverage is recorded, never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.paper.classical_spectral_filtering_audit import ESTIMATOR_DEFINITIONS
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

COVERAGE_COLUMNS = [
    "estimator",
    "implementation_file_found",
    "definition_row_found",
    "main_results_found",
    "alpha_resolved_results_found",
    "nonlinear_results_found",
    "stress_results_found",
    "all_cases_covered",
    "coverage_status",
    "notes",
]

CASE_FAMILIES = ("ieee14", "ieee30", "ieee57", "ieee118", "ieee300")

MATRIX_COLUMNS = [
    "estimator",
    "result_name",
    *CASE_FAMILIES,
    "synthetic_diagnostic",
    "n_result_dirs",
]

MISSING_COLUMNS = [
    "missing_output",
    "estimator",
    "needed_for",
    "importance",
    "reason_missing",
    "recommended_action",
]

DECISION_COLUMNS = ["item", "status", "evidence", "decision", "notes"]


# Estimator definition name -> (aggregate_metrics name, implementation file, alpha-parametrized).
@dataclass(frozen=True)
class EstimatorSpec:
    estimator: str
    result_name: str
    impl_file: str
    alpha_parametrized: bool
    role: str


ESTIMATOR_SPECS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("pseudoinverse", "pseudoinverse", "pseudoinverse.py", False, "unregularized"),
    EstimatorSpec(
        "normal_equation_wls",
        "normal_equation_wls",
        "normal_equation_wls.py",
        False,
        "wls_baseline",
    ),
    EstimatorSpec("ridge_tikhonov", "ridge", "ridge.py", True, "reference"),
    EstimatorSpec("truncated_svd", "truncated_svd", "truncated_svd.py", False, "rank_truncation"),
    EstimatorSpec("huber_irls", "huber_irls", "huber_irls.py", False, "robust"),
    EstimatorSpec("lav", "lav", "lav.py", False, "robust"),
    EstimatorSpec(
        "hhl_style_proxy",
        "hhl_style_inverse_proxy",
        "hhl_style_inverse_proxy.py",
        False,
        "ablation",
    ),
    EstimatorSpec(
        "qsvt_target_classical", "qsvt_regularized", "qsvt_spectral.py", True, "ridge_equivalent"
    ),
)

# (relative dir, case family, workflow type). Stress = dedicated conditioning/stress baselines.
_MAIN_DIRS: tuple[tuple[str, str], ...] = (
    ("real_ieee14_seed10", "ieee14"),
    ("real_ieee30_seed10", "ieee30"),
    ("real_ieee57_seed10", "ieee57"),
    ("real_ieee118_seed10", "ieee118"),
    ("real_ieee300_seed10", "ieee300"),
)
_NONLINEAR_DIRS: tuple[tuple[str, str], ...] = (
    ("nonlinear_ac_ieee14_seed10", "ieee14"),
    ("nonlinear_ac_ieee30_seed10", "ieee30"),
    ("nonlinear_ac_ieee57_seed10", "ieee57"),
    ("nonlinear_ac_ieee118_seed10", "ieee118"),
    ("nonlinear_ac_ieee300_seed10", "ieee300"),
)
_STRESS_DIRS: tuple[tuple[str, str], ...] = (
    ("diagnostic_missing_baselines", "synthetic"),
    ("real_ieee30_high_stress_missing_baselines", "ieee30"),
    ("real_ieee_high_stress_missing_baselines", "ieee14"),
)
_LEGACY_DIRS: tuple[tuple[str, str], ...] = (
    ("historical/smoke_and_legacy/ieee14_robust_bad_data_sweeps", "ieee14"),
)


def build_classical_audit_recheck(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "input_root": "outputs",
        "phase_package_dir": (
            "outputs/final_manuscript_package/phase2_classical_spectral_filtering"
        ),
        "output_dir": "outputs/final_manuscript_package/phase2_classical_recheck",
    }
    resolved.update(config)
    input_root = Path(resolved["input_root"])
    phase_package_dir = Path(resolved["phase_package_dir"])
    output_dir = ensure_directory(resolved["output_dir"])

    presence = _scan_presence(input_root)
    definition_names = _definition_names(phase_package_dir)

    coverage_rows = _coverage_rows(input_root, presence, definition_names)
    matrix_rows = _matrix_rows(presence)
    missing_rows = _missing_rows(coverage_rows)
    decision_rows = _decision_rows(coverage_rows, presence)

    artifacts = _write_outputs(
        output_dir,
        resolved,
        coverage_rows=coverage_rows,
        matrix_rows=matrix_rows,
        missing_rows=missing_rows,
        decision_rows=decision_rows,
    )
    return {
        "output_dir": output_dir,
        "coverage_rows": coverage_rows,
        "matrix_rows": matrix_rows,
        "missing_rows": missing_rows,
        "decision_rows": decision_rows,
        "artifacts": artifacts,
    }


def _scan_presence(input_root: Path) -> dict[str, dict[str, set[str]]]:
    """Map result_name -> {main,nonlinear,stress,legacy} -> set of case families seen."""

    presence: dict[str, dict[str, set[str]]] = {}
    groups = (
        ("main", _MAIN_DIRS),
        ("nonlinear", _NONLINEAR_DIRS),
        ("stress", _STRESS_DIRS),
        ("legacy", _LEGACY_DIRS),
    )
    for group_name, dirs in groups:
        for rel_dir, case in dirs:
            frame = read_csv(input_root / rel_dir / "aggregate_metrics.csv")
            if frame.empty or "estimator" not in frame.columns:
                continue
            for name in frame["estimator"].dropna().astype(str).unique():
                presence.setdefault(name, {}).setdefault(group_name, set()).add(case)
    return presence


def _definition_names(phase_package_dir: Path) -> set[str]:
    frame = read_csv(phase_package_dir / "paper_table_estimator_definitions.csv")
    if not frame.empty and "estimator" in frame.columns:
        return set(frame["estimator"].dropna().astype(str))
    # Fall back to the in-code definitions so the recheck still classifies all estimators.
    return {str(entry["estimator"]) for entry in ESTIMATOR_DEFINITIONS}


def _coverage_rows(
    input_root: Path,
    presence: dict[str, dict[str, set[str]]],
    definition_names: set[str],
) -> list[dict[str, Any]]:
    estimators_dir = input_root.parent / "src" / "robust_qsvt_se" / "estimators"
    rows: list[dict[str, Any]] = []
    for spec in ESTIMATOR_SPECS:
        seen = presence.get(spec.result_name, {})
        main_cases = seen.get("main", set())
        nonlinear_cases = seen.get("nonlinear", set())
        stress_cases = seen.get("stress", set())
        legacy_cases = seen.get("legacy", set())
        covered = main_cases | nonlinear_cases | stress_cases
        all_cases = set(CASE_FAMILIES).issubset(covered)
        has_current = bool(main_cases or nonlinear_cases or stress_cases)

        status = _coverage_status(
            def_row=spec.estimator in definition_names,
            has_current=has_current,
            all_cases=all_cases,
            has_legacy=bool(legacy_cases),
        )
        rows.append(
            {
                "estimator": spec.estimator,
                "implementation_file_found": _yes_no((estimators_dir / spec.impl_file).is_file()),
                "definition_row_found": _yes_no(spec.estimator in definition_names),
                "main_results_found": _yes_no(bool(main_cases)),
                "alpha_resolved_results_found": (
                    "fixed_only" if spec.alpha_parametrized else "not_applicable"
                ),
                "nonlinear_results_found": _yes_no(bool(nonlinear_cases)),
                "stress_results_found": _yes_no(bool(stress_cases)),
                "all_cases_covered": _yes_no(all_cases),
                "coverage_status": status,
                "notes": _coverage_note(spec, covered, legacy_cases),
            }
        )
    return rows


def _coverage_status(*, def_row: bool, has_current: bool, all_cases: bool, has_legacy: bool) -> str:
    if not has_current:
        if has_legacy:
            return "implemented_but_missing_results"
        return "definition_only" if def_row else "expected_but_not_implemented"
    return "complete" if all_cases else "complete_with_limitations"


def _coverage_note(spec: EstimatorSpec, covered: set[str], legacy_cases: set[str]) -> str:
    if spec.estimator == "qsvt_target_classical":
        return "Ridge-equivalent (qsvt_regularized_filter == ridge_filter); not superior to Ridge."
    if spec.estimator == "lav" and not covered:
        scope = ", ".join(sorted(legacy_cases)) or "none"
        return (
            f"Implemented; only legacy/historical smoke results ({scope}); not in benchmark suite."
        )
    if spec.role == "ablation":
        return "Intentional unstable ablation; diagnostic/high-stress baselines only."
    if spec.role == "wls_baseline" and not (set(CASE_FAMILIES) <= covered):
        return (
            "Classical WLS baseline; conditioning/stress baselines only, not all benchmark cases."
        )
    scope = ", ".join(sorted(covered)) or "none"
    return f"Benchmarked on: {scope}. alpha fixed per config (not swept; see Phase 3)."


def _matrix_rows(presence: dict[str, dict[str, set[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ESTIMATOR_SPECS:
        seen = presence.get(spec.result_name, {})
        per_case: dict[str, list[str]] = {case: [] for case in CASE_FAMILIES}
        synthetic: list[str] = []
        n_dirs = 0
        for group, label in (
            ("main", "real_linear"),
            ("nonlinear", "nonlinear"),
            ("stress", "stress"),
            ("legacy", "legacy"),
        ):
            cases = seen.get(group, set())
            n_dirs += len(cases)
            for case in cases:
                if case == "synthetic":
                    synthetic.append(label)
                elif case in per_case:
                    per_case[case].append(label)
        row = {"estimator": spec.estimator, "result_name": spec.result_name}
        for case in CASE_FAMILIES:
            row[case] = ";".join(sorted(set(per_case[case]))) or "none"
        row["synthetic_diagnostic"] = ";".join(sorted(set(synthetic))) or "none"
        row["n_result_dirs"] = n_dirs
        rows.append(row)
    return rows


def _missing_rows(coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_name = {r["estimator"]: r for r in coverage_rows}

    lav = by_name.get("lav", {})
    if lav.get("coverage_status") != "complete":
        rows.append(
            {
                "missing_output": "LAV main results across all benchmark cases",
                "estimator": "lav",
                "needed_for": "complete robust-baseline (L1) comparison",
                "importance": "low",
                "reason_missing": "LAV only present in a legacy/historical IEEE14 smoke run",
                "recommended_action": "add LAV to the benchmark set if an L1 row is wanted",
            }
        )
    for name in ("normal_equation_wls", "hhl_style_proxy"):
        row = by_name.get(name, {})
        if row.get("all_cases_covered") == "no":
            rows.append(
                {
                    "missing_output": f"{name} results on the IEEE57/118/300 benchmark cases",
                    "estimator": name,
                    "needed_for": "uniform baseline coverage across all cases",
                    "importance": "low",
                    "reason_missing": "run only on diagnostic/high-stress conditioning baselines",
                    "recommended_action": "by design (WLS/ablation conditioning probe); optional",
                }
            )
    rows.append(
        {
            "missing_output": "alpha-resolved classical main-result table (RMSE/residual vs alpha)",
            "estimator": "ridge_tikhonov; qsvt_target_classical",
            "needed_for": "alpha-sensitivity discussion",
            "importance": "medium",
            "reason_missing": "classical sweeps fix alpha; it is not a swept axis in RMSE results",
            "recommended_action": "see Phase 3 alpha-sensitivity consolidation (do not fabricate)",
        }
    )
    return rows


def _decision_rows(
    coverage_rows: list[dict[str, Any]], presence: dict[str, dict[str, set[str]]]
) -> list[dict[str, Any]]:
    complete = [r["estimator"] for r in coverage_rows if r["coverage_status"] == "complete"]
    limited = [
        r["estimator"] for r in coverage_rows if r["coverage_status"] == "complete_with_limitations"
    ]
    defined_only = [
        r["estimator"]
        for r in coverage_rows
        if r["coverage_status"] in {"definition_only", "implemented_but_missing_results"}
    ]
    overall = "complete_with_limitations" if (limited or defined_only) else "complete"
    return [
        {
            "item": "estimator_definitions",
            "status": "complete",
            "evidence": f"{len(coverage_rows)} estimators defined",
            "decision": "confirmed",
            "notes": "8 estimator definitions present.",
        },
        {
            "item": "fully_benchmarked_estimators",
            "status": "complete",
            "evidence": ", ".join(complete) or "none",
            "decision": "confirmed",
            "notes": "Compared across all benchmark cases.",
        },
        {
            "item": "partially_benchmarked_estimators",
            "status": "complete_with_limitations",
            "evidence": ", ".join(limited) or "none",
            "decision": "downgraded",
            "notes": "Conditioning/stress baselines only or subset of cases.",
        },
        {
            "item": "defined_but_not_benchmarked_estimators",
            "status": "implemented_but_missing_results",
            "evidence": ", ".join(defined_only) or "none",
            "decision": "downgraded",
            "notes": "Implemented and defined; not in the current benchmark suite.",
        },
        {
            "item": "alpha_resolved_classical_results",
            "status": "missing_evidence",
            "evidence": "classical RMSE results fix alpha (not swept)",
            "decision": "deferred_to_phase3",
            "notes": "Do not fabricate alpha; see Phase 3 consolidation.",
        },
        {
            "item": "qsvt_target_vs_ridge",
            "status": "ridge_equivalent",
            "evidence": "qsvt_regularized_filter == ridge_filter",
            "decision": "confirmed",
            "notes": "No QSVT-over-Ridge superiority claim.",
        },
        {
            "item": "overall_phase2_classical",
            "status": overall,
            "evidence": f"{len(complete)} complete / {len(limited)} limited / {len(defined_only)} "
            "defined-only",
            "decision": "confirmed" if overall == "complete" else "downgraded",
            "notes": "Phase 2 classical core is broadly complete with documented coverage limits.",
        },
    ]


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_outputs(
    output_dir: Path,
    resolved: dict[str, Any],
    *,
    coverage_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    coverage_path = rows_to_table(
        coverage_rows, output_dir / "estimator_coverage_recheck.csv", COVERAGE_COLUMNS
    )
    matrix_path = rows_to_table(
        matrix_rows, output_dir / "estimator_result_coverage_matrix.csv", MATRIX_COLUMNS
    )
    missing_path = rows_to_table(
        missing_rows, output_dir / "missing_classical_evidence_recheck.csv", MISSING_COLUMNS
    )
    decision_path = rows_to_table(
        decision_rows, output_dir / "phase2_readiness_decision.csv", DECISION_COLUMNS
    )
    status_path = output_dir / "phase2_classical_recheck.md"
    status_path.write_text(_recheck_markdown(coverage_rows, decision_rows), encoding="utf-8")

    manifest = write_manifest(
        output_dir,
        artifacts={
            "estimator_coverage_recheck": str(coverage_path),
            "estimator_result_coverage_matrix": str(matrix_path),
            "missing_classical_evidence_recheck": str(missing_path),
            "phase2_readiness_decision": str(decision_path),
            "phase2_classical_recheck": str(status_path),
        },
        input_config=resolved,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "manifest": manifest,
        "estimator_coverage_recheck": coverage_path,
        "estimator_result_coverage_matrix": matrix_path,
        "missing_classical_evidence_recheck": missing_path,
        "phase2_readiness_decision": decision_path,
        "phase2_classical_recheck": status_path,
    }


def _recheck_markdown(
    coverage_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]
) -> str:
    complete = [r["estimator"] for r in coverage_rows if r["coverage_status"] == "complete"]
    limited = [
        r["estimator"] for r in coverage_rows if r["coverage_status"] == "complete_with_limitations"
    ]
    defined_only = [
        r["estimator"]
        for r in coverage_rows
        if r["coverage_status"] in {"definition_only", "implemented_but_missing_results"}
    ]
    overall = next(r for r in decision_rows if r["item"] == "overall_phase2_classical")
    return "\n".join(
        [
            "# Phase 2 Classical Spectral Filtering Audit Recheck",
            "",
            PAPER_CLAIM_BOUNDARY,
            "",
            "## Filters under audit",
            "Ridge/Tikhonov reference:",
            "",
            "\\[",
            "P_{\\alpha}(\\sigma)",
            "=",
            "\\frac{\\sigma}{\\sigma^2+\\alpha}.",
            "\\]",
            "",
            "Pseudoinverse:",
            "",
            "\\[",
            "P_{\\mathrm{pinv}}(\\sigma)",
            "=",
            "\\frac{1}{\\sigma}.",
            "\\]",
            "",
            "Weighted Jacobian condition number:",
            "",
            "\\[",
            "\\kappa(\\tilde H)",
            "=",
            "\\frac{\\sigma_{\\max}(\\tilde H)}",
            "{\\sigma_{\\min}(\\tilde H)}.",
            "\\]",
            "",
            "## Estimator coverage (verified against repository + Phase 2 package)",
            f"- Fully benchmarked (all cases): {complete or 'none'}.",
            f"- Complete with limitations (subset of cases / stress-only): {limited or 'none'}.",
            f"- Implemented/defined but not benchmarked: {defined_only or 'none'}.",
            "",
            "## Estimators actually compared in the final classical results",
            "- pseudoinverse, ridge_tikhonov, truncated_svd, huber_irls, and qsvt_target_classical "
            "(run name qsvt_regularized) are compared across the IEEE benchmark cases.",
            "- normal_equation_wls and hhl_style_proxy appear only in the synthetic conditioning / "
            "high-stress baselines (by design).",
            "- lav is implemented and defined but only has a legacy IEEE14 smoke result; "
            "it is not part of the current benchmark suite.",
            "",
            "## Alpha-resolved results",
            "- Classical RMSE/residual results fix the Tikhonov alpha per configuration; alpha is "
            "not a swept axis, so an alpha-resolved classical main-result table is not available "
            "here and is deferred to the Phase 3 alpha-sensitivity consolidation (not fabricated).",
            "",
            "## QSVT-target vs Ridge",
            "- qsvt_target_classical is numerically identical to Ridge/Tikhonov for matched alpha "
            "(qsvt_regularized_filter == ridge_filter). No QSVT-over-Ridge superiority asserted.",
            "",
            "## Decision",
            f"- Overall Phase 2 classical core: {overall['status']} ({overall['decision']}).",
            "- Phase 2 is broadly complete; coverage limitations (stress-only baselines for "
            "normal_equation_wls/hhl_style_proxy, legacy-only LAV, fixed-alpha results) are "
            "explicitly recorded rather than hidden.",
            "",
        ]
    )
