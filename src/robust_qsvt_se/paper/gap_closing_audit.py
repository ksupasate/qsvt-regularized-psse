"""Phase 0: gap and evidence-integrity triage for the gap-closing task.

Reads the current final manuscript package (``remaining_missing_evidence.csv`` and
``claim_support_matrix_final.csv``) and triages every recorded gap / weak claim into
one of three buckets: closeable from existing artifacts, requires a new fast run, or
must remain missing. The triage is a deterministic policy over the *real* on-disk
items; it never invents gaps and never marks a gap closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

TRIAGE_COLUMNS = [
    "gap_id",
    "gap_name",
    "related_claim_id",
    "current_status",
    "priority",
    "can_close_from_existing_artifacts",
    "requires_new_run",
    "expected_runtime_class",
    "risk_of_fabrication",
    "recommended_action",
    "source_artifacts_checked",
    "notes",
]

RUNTIME_CLASSES = (
    "existing_artifact_only",
    "fast_regeneration",
    "moderate_experiment",
    "expensive_experiment",
    "not_available",
)


@dataclass(frozen=True, slots=True)
class TriageRule:
    """A keyword-driven triage policy entry.

    ``keywords`` are matched (case-insensitive substring) against the gap text; the
    first matching rule decides the runtime class, action, and fabrication risk.
    """

    keywords: tuple[str, ...]
    runtime_class: str
    priority: str
    can_close: bool
    requires_new_run: bool
    risk: str
    action: str
    gap_phase: str


# Ordered most-specific first. Hard overclaims and field/hardware items are matched
# before the closeable regeneration buckets so they can never be triaged as closeable.
_RULES: tuple[TriageRule, ...] = (
    TriageRule(
        (
            "quantum speedup",
            "quantum advantage",
            "superiority over ridge",
            "beats ridge",
            "numerical superiority",
        ),
        "not_available",
        "low",
        False,
        False,
        "high",
        "do not claim; preserve as unsupported overclaim",
        "unsupported",
    ),
    TriageRule(
        ("pmu", "scada", "field data", "field-calibrated", "field calibrated"),
        "not_available",
        "low",
        False,
        False,
        "high",
        "do not claim; real field validation / field-calibrated statistics out of scope",
        "unsupported",
    ),
    TriageRule(
        (
            "full ieee-scale",
            "hardware execution",
            "deployment-ready",
            "full output-direction",
            "full-vector readout",
            "sparse-oracle pathway",
        ),
        "not_available",
        "low",
        False,
        False,
        "high",
        "keep as assumption-only / future work; not executed in this task",
        "assumption",
    ),
    TriageRule(
        (
            "per-measurement-type",
            "per measurement type",
            "per-type",
            "type drop",
            "measurement-type",
            "measurement type ablation",
        ),
        "fast_regeneration",
        "high",
        False,
        True,
        "low",
        "regenerate via measurement_type_ablation (Phase 1)",
        "phase1",
    ),
    TriageRule(
        (
            "alpha-resolved",
            "alpha sweep",
            "rmse-vs-alpha",
            "alpha-sensitivity",
            "alpha sensitivity",
            "tikhonov) selection",
        ),
        "fast_regeneration",
        "medium",
        False,
        True,
        "low",
        "regenerate via full_alpha_sweep_classical (Phase 2)",
        "phase2",
    ),
    TriageRule(
        (
            "compound",
            "weak-area",
            "weak area",
            "spatial",
            "contiguous",
            "structured_stress_measurement_ablation",
            "structured stress",
        ),
        "fast_regeneration",
        "medium",
        False,
        True,
        "medium",
        "regenerate controlled benchmark stress via compound_structured_stress (Phase 3)",
        "phase3",
    ),
    TriageRule(
        ("nonlinear",),
        "fast_regeneration",
        "medium",
        False,
        True,
        "low",
        "regenerate via nonlinear_ac_alpha_stress (Phase 4)",
        "phase4",
    ),
    TriageRule(
        ("lav", "normal_equation_wls", "normal-equation", "hhl"),
        "fast_regeneration",
        "low",
        False,
        True,
        "low",
        "extend or document via baseline_coverage_extension (Phase 5)",
        "phase5",
    ),
    TriageRule(
        ("readout", "observable", "norm recovery", "top-k", "amplitude estimation"),
        "existing_artifact_only",
        "medium",
        True,
        False,
        "low",
        "consolidate existing readout artifacts via readout_limitation_formalization (Phase 6)",
        "phase6",
    ),
    TriageRule(
        ("classical_main_results", "main-result table", "main results"),
        "existing_artifact_only",
        "high",
        True,
        False,
        "low",
        "already consolidated by the manuscript package from per-case aggregates",
        "phase2_pkg",
    ),
)

_DEFAULT_RULE = TriageRule(
    (),
    "fast_regeneration",
    "low",
    False,
    True,
    "low",
    "consolidate from existing artifacts where available; otherwise record as future work",
    "general",
)

# Claims that are intentionally unsupported / assumption-only regardless of new runs.
_HARD_UNSUPPORTED = {"C11", "C12", "C13"}
_ASSUMPTION_ONLY = {"C10"}
# supported_with_limitations claims that THIS task tries to strengthen with new fast runs.
_STRENGTHEN_CLAIMS = {
    "C14": _RULES[5],  # compound/weak-area structured stress (Phase 3)
    "C15": _RULES[4],  # full classical alpha sweep (Phase 2)
    "C16": _RULES[6],  # nonlinear alpha/weak-area stress (Phase 4)
}
# Documented supported_with_limitations claims whose evidence already exists on disk.
_DOCUMENTED_RULE = TriageRule(
    (),
    "existing_artifact_only",
    "low",
    True,
    False,
    "low",
    "documented limitation; supporting evidence already exists, optionally strengthened",
    "documented",
)


def build_gap_closing_audit(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_dir = Path(config.get("package_dir", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_dir / "gap_closing_audit"))

    missing = read_csv(package_dir / "remaining_missing_evidence.csv")
    claims = read_csv(package_dir / "claim_support_matrix_final.csv")

    rows: list[dict[str, Any]] = []
    rows.extend(_claim_gap_rows(claims, package_dir))
    rows.extend(_missing_gap_rows(missing, package_dir))
    for index, row in enumerate(rows, start=1):
        row["gap_id"] = f"G{index:02d}"

    closeable = [r for r in rows if r["can_close_from_existing_artifacts"] == "yes"]
    new_runs = [
        r
        for r in rows
        if r["requires_new_run"] == "yes" and r["expected_runtime_class"] != "not_available"
    ]
    must_remain = [r for r in rows if r["expected_runtime_class"] == "not_available"]

    return _write_outputs(
        output_dir=output_dir,
        triage_rows=rows,
        closeable_rows=closeable,
        new_run_rows=new_runs,
        must_remain_rows=must_remain,
        input_config={
            "input_root": str(input_root),
            "package_dir": str(package_dir),
            "output_dir": str(output_dir),
        },
    )


def _classify(text: str) -> TriageRule:
    lowered = text.lower()
    for rule in _RULES:
        if any(keyword in lowered for keyword in rule.keywords):
            return rule
    return _DEFAULT_RULE


def _triage_row(
    *,
    gap_name: str,
    related_claim_id: str,
    current_status: str,
    rule: TriageRule,
    source_artifacts_checked: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "gap_id": "",
        "gap_name": gap_name,
        "related_claim_id": related_claim_id,
        "current_status": current_status,
        "priority": rule.priority,
        "can_close_from_existing_artifacts": "yes" if rule.can_close else "no",
        "requires_new_run": "yes" if rule.requires_new_run else "no",
        "expected_runtime_class": rule.runtime_class,
        "risk_of_fabrication": rule.risk,
        "recommended_action": rule.action,
        "source_artifacts_checked": source_artifacts_checked,
        "notes": notes,
    }


def _claim_gap_rows(claims: Any, package_dir: Path) -> list[dict[str, Any]]:
    if claims.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, claim in claims.iterrows():
        status = str(claim.get("support_status", ""))
        if status == "supported":
            continue
        claim_id = str(claim.get("claim_id", ""))
        text = f"{claim.get('claim_text', '')} {claim.get('limitation_note', '')}"
        if claim_id in _HARD_UNSUPPORTED:
            rule = _RULES[0]
        elif claim_id in _ASSUMPTION_ONLY:
            rule = _RULES[2]
        elif claim_id in _STRENGTHEN_CLAIMS:
            rule = _STRENGTHEN_CLAIMS[claim_id]
        elif status == "missing_evidence":
            rule = _classify(text)
        elif status == "supported_with_limitations":
            rule = _DOCUMENTED_RULE
        else:
            rule = _classify(text)
        artifacts = str(claim.get("supporting_artifacts", "")) or "none"
        rows.append(
            _triage_row(
                gap_name=str(claim.get("claim_text", ""))[:160],
                related_claim_id=claim_id,
                current_status=status,
                rule=rule,
                source_artifacts_checked=_check_artifacts(artifacts, package_dir),
                notes=str(claim.get("limitation_note", ""))[:200],
            )
        )
    return rows


def _missing_gap_rows(missing: Any, package_dir: Path) -> list[dict[str, Any]]:
    if missing.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, item in missing.iterrows():
        gap_name = str(item.get("item", ""))
        rule = _classify(f"{gap_name} {item.get('recommended_action', '')}")
        rows.append(
            _triage_row(
                gap_name=gap_name[:160],
                related_claim_id="",
                current_status=str(item.get("status", "missing")),
                rule=rule,
                source_artifacts_checked=_check_artifacts(gap_name, package_dir),
                notes=(
                    f"source_phase={item.get('source_phase', '')}; "
                    f"importance={item.get('importance', '')}"
                ),
            )
        )
    return rows


def _check_artifacts(artifact_text: str, package_dir: Path) -> str:
    """Report which referenced artifact paths exist on disk (grounds the triage)."""

    candidates = [
        token.strip()
        for token in artifact_text.replace(";", " ").split()
        if "/" in token or token.endswith(".csv")
    ]
    if not candidates:
        return "no_path_referenced"
    checked = []
    repo_root = package_dir.parents[1] if len(package_dir.parents) >= 2 else Path(".")
    for candidate in candidates[:4]:
        path = Path(candidate)
        exists = path.exists() or (repo_root / candidate).exists()
        checked.append(f"{candidate}:{'present' if exists else 'absent'}")
    return "; ".join(checked)


def _audit_markdown(
    triage_rows: list[dict[str, Any]],
    closeable_rows: list[dict[str, Any]],
    new_run_rows: list[dict[str, Any]],
    must_remain_rows: list[dict[str, Any]],
) -> str:
    by_phase: dict[str, int] = {}
    for row in new_run_rows:
        action = row["recommended_action"]
        by_phase[action] = by_phase.get(action, 0) + 1
    lines = [
        "# Gap-Closing Evidence-Integrity Audit (Phase 0)",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "This triage reads the current `remaining_missing_evidence.csv` and "
        "`claim_support_matrix_final.csv` and classifies every recorded gap. It does not "
        "close any gap and does not fabricate evidence; the gap-closing phases generate the "
        "real artifacts that may later upgrade a claim.",
        "",
        "## Triage Totals",
        "",
        f"- Total gaps triaged: {len(triage_rows)}",
        f"- Closeable from existing artifacts: {len(closeable_rows)}",
        f"- Require a new fast run: {len(new_run_rows)}",
        f"- Must remain missing (not available in this task): {len(must_remain_rows)}",
        "",
        "## Gaps Attempted by New Fast Runs",
        "",
        "| Recommended action | Gap count |",
        "| --- | ---: |",
    ]
    for action, count in sorted(by_phase.items()):
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Must Remain Missing",
            "",
            "These gaps are preserved as unsupported or assumption-only; this task does not "
            "attempt them and does not claim them:",
            "",
        ]
    )
    for row in must_remain_rows:
        claim = f" ({row['related_claim_id']})" if row["related_claim_id"] else ""
        lines.append(f"- {row['gap_name']}{claim} — {row['recommended_action']}")
    lines.extend(
        [
            "",
            "## Runtime Classes",
            "",
            "Exact runtime-class labels used: "
            + ", ".join(f"`{label}`" for label in RUNTIME_CLASSES)
            + ".",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    triage_rows: list[dict[str, Any]],
    closeable_rows: list[dict[str, Any]],
    new_run_rows: list[dict[str, Any]],
    must_remain_rows: list[dict[str, Any]],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(output_dir)
    triage_path = output_dir / "remaining_gap_triage.csv"
    closeable_path = output_dir / "closeable_from_existing_artifacts.csv"
    new_runs_path = output_dir / "requires_new_fast_runs.csv"
    must_remain_path = output_dir / "must_remain_missing.csv"
    markdown_path = output_dir / "gap_closing_audit.md"

    rows_to_table(triage_rows, triage_path, TRIAGE_COLUMNS)
    rows_to_table(closeable_rows, closeable_path, TRIAGE_COLUMNS)
    rows_to_table(new_run_rows, new_runs_path, TRIAGE_COLUMNS)
    rows_to_table(must_remain_rows, must_remain_path, TRIAGE_COLUMNS)
    markdown_path.write_text(
        _audit_markdown(triage_rows, closeable_rows, new_run_rows, must_remain_rows),
        encoding="utf-8",
    )

    artifacts = {
        "remaining_gap_triage": str(triage_path),
        "closeable_from_existing_artifacts": str(closeable_path),
        "requires_new_fast_runs": str(new_runs_path),
        "must_remain_missing": str(must_remain_path),
        "gap_closing_audit_md": str(markdown_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "triage_rows": triage_rows,
        "closeable_rows": closeable_rows,
        "new_run_rows": new_run_rows,
        "must_remain_rows": must_remain_rows,
        "artifacts": artifacts,
    }
