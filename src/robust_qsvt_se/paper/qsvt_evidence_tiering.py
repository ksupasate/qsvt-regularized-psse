"""Task 6: tier QSVT polynomial/phase evidence by uniform admissibility + phase synthesis.

Audits existing QSVT evidence CSVs plus the new selected-observable demo and sorts
each configuration into one tier:

* ``uniform_admissible_phase_synthesized`` - bounded on the whole interval (not only
  at spectrum points), gate/feasibility-recommended, and phase-synthesized -> **main paper**,
* ``phase_synthesized``      - phases synthesized but not uniform-admissible/recommended,
* ``spectrum_point_only``    - admissibility/pointwise evidence, phases not synthesized,
* ``phase_unavailable``      - phase synthesis attempted and failed,
* ``degree_limited``         - degree/synthesis ceiling is the binding limitation,
* ``failed_boundedness``     - |p| exceeds 1 / overshoot,
* ``failed_parity``          - parity violation,
* ``failed_tolerance``       - residual/tolerance not met.

Only ``uniform_admissible_phase_synthesized`` rows are recommended as main-paper
QSVT evidence; everything else is labelled diagnostic and routed to the appendix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.paper.selected_observable_qsvt_common import (
    DEMO_DIR,
    EVIDENCE_TIERING_DIR,
    assert_safe,
    write_demo_manifest,
)
from robust_qsvt_se.utils.io import ensure_directory

MAIN_TIER = "uniform_admissible_phase_synthesized"
APPENDIX_TIERS = (
    "phase_synthesized",
    "spectrum_point_only",
    "phase_unavailable",
    "degree_limited",
    "failed_boundedness",
    "failed_parity",
    "failed_tolerance",
)
ALL_TIERS = (MAIN_TIER, *APPENDIX_TIERS)

NORMALIZED_COLUMNS = [
    "source",
    "case",
    "subproblem",
    "alpha",
    "degree",
    "phase_attempted",
    "phase_synthesized",
    "bounded_ok",
    "uniform_admissible",
    "recommended",
    "limitation",
    "tier",
    "main_paper_recommended",
]


@dataclass(frozen=True, slots=True)
class NormalizedEvidence:
    source: str
    case: str
    subproblem: str
    alpha: float | None
    degree: int | None
    phase_attempted: bool
    phase_synthesized: bool
    bounded_ok: bool | None
    uniform_admissible: bool
    recommended: bool
    limitation: str


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "synthesized", "completed", "success"}


def _get(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and not _is_missing(row[name]):
            return row[name]
    return default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def classify_tier(evidence: NormalizedEvidence) -> str:
    """Pure tier-classification rule (precedence: failures, then success grades)."""

    reason = (evidence.limitation or "").lower()
    if "parity" in reason:
        return "failed_parity"
    if evidence.bounded_ok is False or "overshoot" in reason or "bound" in reason:
        return "failed_boundedness"
    if "degree" in reason or "synthesis ceiling" in reason:
        return "degree_limited"
    if evidence.phase_attempted and not evidence.phase_synthesized:
        return "phase_unavailable"
    if evidence.phase_synthesized and evidence.uniform_admissible and evidence.recommended:
        return MAIN_TIER
    if evidence.phase_synthesized and evidence.uniform_admissible:
        return "phase_synthesized"
    if evidence.phase_synthesized:
        return "spectrum_point_only"
    if evidence.uniform_admissible:
        return "spectrum_point_only"
    if "toler" in reason or "residual" in reason or "infeasible" in reason:
        return "failed_tolerance"
    return "failed_tolerance"


def _normalize_demo_row(row: dict[str, Any]) -> NormalizedEvidence:
    status = str(row.get("status_label", "")).lower()
    phase_status = str(row.get("phase_synthesis_status", "")).lower()
    phase_synth = phase_status in {"completed", "synthesized", "success"}
    bounded = _as_bool(row.get("boundedness_ok", True))
    limitation = ""
    if status == "degree_limited":
        limitation = "degree-limited reconstruction (synthesis ceiling)"
    elif status == "phase_unavailable":
        limitation = "phase unavailable"
    elif status == "failed_boundedness":
        limitation = "failed boundedness"
    return NormalizedEvidence(
        source="selected_observable_qsvt_demo",
        case=str(row.get("case", "")),
        subproblem=str(row.get("block_shape", "")),
        alpha=_float_or_none(row.get("alpha")),
        degree=_int_or_none(row.get("degree")),
        phase_attempted=True,
        phase_synthesized=phase_synth,
        bounded_ok=bounded,
        uniform_admissible=bool(status == "pass"),
        recommended=bool(status == "pass"),
        limitation=limitation,
    )


def _normalize_generic_row(source: str, row: dict[str, Any]) -> NormalizedEvidence:
    phase_status = str(
        _get(row, "phase_synthesis_status", "phase_status", "status", default="")
    ).lower()
    phase_count = _int_or_none(_get(row, "phase_count"))
    phase_attempted = (
        "phase_synthesis_status" in row or "phase_count" in row or "phase_method" in row
    )
    phase_synth = phase_status in {"synthesized", "completed", "success"} or (
        phase_count is not None and phase_count > 0 and phase_status not in {"failed", "skipped"}
    )

    overshoot = _as_bool(_get(row, "overshoot_detected", default=False))
    max_abs = _float_or_none(
        _get(row, "max_abs_polynomial_on_grid", "polynomial_validation_max_abs", "bounded_max_abs")
    )
    bounded_ok: bool | None
    if overshoot:
        bounded_ok = False
    elif max_abs is not None:
        bounded_ok = bool(max_abs <= 1.0 + 1.0e-3)
    else:
        bounded_ok = None

    qsvt_safe = _as_bool(_get(row, "qsvt_safe", default=False))
    residual_feasible = _as_bool(
        _get(row, "residual_feasible_after_gate", "residual_feasible", default=False)
    )
    uniform_admissible = bool(qsvt_safe and (bounded_ok is not False))
    recommended = _as_bool(_get(row, "gate_validation_recommended", default=False)) or (
        residual_feasible and qsvt_safe
    )
    limitation = str(
        _get(
            row,
            "rejection_reason",
            "failure_reason",
            "dominant_limitation",
            "skip_reason",
            default="",
        )
    )
    if limitation.lower() in {"none", "nan", ""}:
        limitation = ""
    return NormalizedEvidence(
        source=source,
        case=str(_get(row, "case", "case_name", default="")),
        subproblem=str(_get(row, "subproblem_id", "subproblem", default="")),
        alpha=_float_or_none(_get(row, "alpha")),
        degree=_int_or_none(_get(row, "degree")),
        phase_attempted=bool(phase_attempted),
        phase_synthesized=bool(phase_synth),
        bounded_ok=bounded_ok,
        uniform_admissible=uniform_admissible,
        recommended=recommended,
        limitation=limitation,
    )


def _float_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    return None if parsed is None else int(parsed)


def _ingest(path: Path, source: str, is_demo: bool) -> list[NormalizedEvidence]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    records: list[NormalizedEvidence] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in frame.to_dict(orient="records"):
        evidence = _normalize_demo_row(raw) if is_demo else _normalize_generic_row(source, raw)
        # Collapse duplicate (case, subproblem, alpha, degree) configs per source.
        key = (evidence.source, evidence.case, evidence.subproblem, evidence.alpha, evidence.degree)
        if key in seen:
            continue
        seen.add(key)
        records.append(evidence)
    return records


def run_qsvt_evidence_tiering(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_dir = ensure_directory(Path(resolved["output_dir"]))

    records: list[NormalizedEvidence] = []
    records.extend(
        _ingest(Path(resolved["demo_summary_csv"]), "selected_observable_qsvt_demo", True)
    )
    for source_name, rel_path in resolved["sources"]:
        records.extend(_ingest(Path(rel_path), source_name, False))

    rows: list[dict[str, Any]] = []
    for evidence in records:
        tier = classify_tier(evidence)
        rows.append(
            {
                "source": evidence.source,
                "case": evidence.case,
                "subproblem": evidence.subproblem,
                "alpha": evidence.alpha,
                "degree": evidence.degree,
                "phase_attempted": evidence.phase_attempted,
                "phase_synthesized": evidence.phase_synthesized,
                "bounded_ok": evidence.bounded_ok,
                "uniform_admissible": evidence.uniform_admissible,
                "recommended": evidence.recommended,
                "limitation": evidence.limitation,
                "tier": tier,
                "main_paper_recommended": bool(tier == MAIN_TIER),
            }
        )

    frame = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
    main_frame = frame[frame["tier"] == MAIN_TIER].reset_index(drop=True)
    appendix_frame = frame[frame["tier"] != MAIN_TIER].reset_index(drop=True)

    status_counts = {tier: int((frame["tier"] == tier).sum()) for tier in ALL_TIERS}
    status_summary = pd.DataFrame(
        [
            {
                "tier": tier,
                "count": status_counts[tier],
                "destination": "main_paper" if tier == MAIN_TIER else "appendix_supplement",
            }
            for tier in ALL_TIERS
        ]
    )

    main_csv = output_dir / "main_paper_qsvt_evidence.csv"
    appendix_csv = output_dir / "appendix_qsvt_diagnostics.csv"
    status_csv = output_dir / "phase_synthesis_status_summary.csv"
    main_frame.to_csv(main_csv, index=False)
    appendix_frame.to_csv(appendix_csv, index=False)
    status_summary.to_csv(status_csv, index=False)

    readme = output_dir / "README.md"
    readme.write_text(_readme(frame, status_summary), encoding="utf-8")

    artifacts = {
        "main_paper_qsvt_evidence_csv": main_csv,
        "appendix_qsvt_diagnostics_csv": appendix_csv,
        "phase_synthesis_status_summary_csv": status_csv,
        "readme_md": readme,
    }
    manifest = write_demo_manifest(
        output_dir=output_dir,
        artifact_name="qsvt_evidence_tiering",
        description=(
            "Tiering of QSVT polynomial/phase evidence by uniform admissibility and phase "
            "synthesis. Only uniform-admissible phase-synthesized rows are recommended as "
            "main-paper QSVT evidence; spectrum-point-only and failed rows are diagnostic."
        ),
        command=resolved["command"],
        artifacts=artifacts,
        input_files=[resolved["demo_summary_csv"], *[path for _name, path in resolved["sources"]]],
        extra={"tier_counts": status_counts, "n_records": len(records)},
        manifest_name="tiering_manifest.json",
    )
    artifacts["manifest"] = manifest
    return {
        "output_dir": output_dir,
        "all_evidence": frame,
        "main_paper": main_frame,
        "appendix": appendix_frame,
        "status_summary": status_summary,
        "artifacts": artifacts,
    }


def _readme(frame: pd.DataFrame, status_summary: pd.DataFrame) -> str:
    lines = [
        "# QSVT Evidence Tiering",
        "",
        "Each QSVT configuration is tiered by **uniform admissibility** (bounded on the whole "
        "interval, not only at spectrum points) and **phase synthesis**. Only "
        f"`{MAIN_TIER}` rows are recommended as main-paper QSVT evidence; all other tiers are "
        "diagnostic and routed to the appendix/supplement.",
        "",
        "## Tier counts",
        "",
        "| Tier | Count | Destination |",
        "| --- | --- | --- |",
    ]
    for _, row in status_summary.iterrows():
        lines.append(f"| `{row['tier']}` | {int(row['count'])} | {row['destination']} |")
    lines += [
        "",
        f"Total configurations audited: {len(frame)}. Sources: "
        + ", ".join(sorted(frame["source"].unique()))
        + ".",
        "",
        "Main-paper recommended rows (uniform-admissible + phase-synthesized):",
        "",
    ]
    main_rows = frame[frame["tier"] == MAIN_TIER]
    if main_rows.empty:
        lines.append("- _none_")
    else:
        for _, row in main_rows.iterrows():
            lines.append(
                f"- `{row['source']}` {row['case']} {row['subproblem']} "
                f"(alpha={row['alpha']}, degree={row['degree']})"
            )
    lines.append("")
    text = "\n".join(lines)
    assert_safe(text)
    return text


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "output_dir": str(EVIDENCE_TIERING_DIR),
        "demo_summary_csv": str(DEMO_DIR / "demo_summary.csv"),
        "sources": [
            (
                "codesigned_gate_validation",
                "outputs/qsvt_codesigned_gate_validation/codesigned_gate_validation_results.csv",
            ),
            (
                "residual_feasible_deployable",
                "outputs/qsvt_residual_feasible_codesigned_search/deployable_residual_feasible_configs.csv",
            ),
            (
                "residual_feasible_diagnostic",
                "outputs/qsvt_residual_feasible_codesigned_search/diagnostic_feasible_configs.csv",
            ),
            (
                "residual_feasible_rejected",
                "outputs/qsvt_residual_feasible_codesigned_search/rejected_codesigned_configs.csv",
            ),
            (
                "cross_case_candidates",
                "outputs/qsvt_cross_case_codesigned_robustness/cross_case_gate_validation_candidates.csv",
            ),
            (
                "phase_synthesis_validation",
                "outputs/qsvt_optional_phase_synthesis_validation/phase_synthesis_summary.csv",
            ),
        ],
        "command": "run_qsvt_evidence_tiering",
    }
    if config:
        resolved.update({key: value for key, value in config.items() if value is not None})
    return resolved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tier QSVT evidence by admissibility + phase")
    parser.add_argument("--output-dir", default=str(EVIDENCE_TIERING_DIR))
    parser.add_argument("--demo-summary-csv", default=str(DEMO_DIR / "demo_summary.csv"))
    args = parser.parse_args(argv)
    run = run_qsvt_evidence_tiering(
        {
            "output_dir": args.output_dir,
            "demo_summary_csv": args.demo_summary_csv,
            "command": "scripts/tier_qsvt_evidence.py " + " ".join(argv or []),
        }
    )
    print(f"QSVT evidence tiering complete: {run['artifacts']['main_paper_qsvt_evidence_csv']}")
    counts = run["status_summary"].set_index("tier")["count"].to_dict()
    print(f"Tier counts: {counts}")


if __name__ == "__main__":  # pragma: no cover
    main()
