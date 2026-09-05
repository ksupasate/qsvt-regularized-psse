"""Phase 4 hardening: claim-boundary lint tool.

Scans manuscript drafts and generated text artifacts for risky overclaiming language and
classifies each hit so that genuine overclaims are flagged while negated statements,
limitation statements, and disallowed-wording citations (e.g. the DO_NOT_CLAIM list) are
recognized as safe. This tool fabricates nothing; it only reports what the text contains.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/lint_claim_boundaries.py"

REPORT_COLUMNS = [
    "file_path",
    "line_number",
    "matched_phrase",
    "risk_level",
    "classification",
    "recommended_action",
    "line_excerpt",
]

# High-risk positive overclaims (lower-cased substrings).
RISK_PHRASES = (
    "quantum speedup",
    "quantum advantage",
    "qsvt beats ridge",
    "qsvt outperforms ridge",
    "qsvt improves over ridge",
    "qsvt is better than ridge",
    "full ieee-scale qsvt",
    "full-scale quantum state estimator",
    "full nonlinear quantum ac state estimator",
    "real pmu validation",
    "real scada validation",
    "real pmu/scada validation",
    "field validated",
    "field-validated",
    "field-calibrated",
    "hardware execution",
    "deployment-ready",
    "full-vector readout solved",
    "hhl implementation",
)

# Context cues. A hit in any of these contexts is treated as safe (low risk).
_CITATION_CUES = (
    "do not claim",
    "do_not_claim",
    "must not",
    "disallowed",
    "forbidden",
    "avoid claiming",
    "not_supported",
    "unsupported",
    "do-not-claim",
)
_NEGATION_CUES = (
    "not ",
    "no ",
    "never",
    "without",
    "cannot",
    "can't",
    "does not",
    "do not",
    "is not",
    "are not",
    "n't ",
    "rather than",
    "neither",
    " nor ",
    "absent",
)
_LIMITATION_CUES = (
    "assumption",
    "limitation",
    "out of scope",
    "out-of-scope",
    "remains",
    "future work",
    "not demonstrated",
    "not claimed",
    "not shown",
    "missing",
    "selected-subproblem",
    "controlled benchmark",
)

_SCAN_SUFFIXES = {".md", ".csv", ".tex", ".txt"}
# Lint-report directories quote disallowed wording by design; never re-lint them.
_SELF_DIRS = frozenset({"claim_lint", "manuscript_claim_lint"})
# CSV columns that legitimately list disallowed/forbidden wording (cite, never assert).
_DISALLOWED_COLUMN_CUES = (
    "disallow",
    "do_not",
    "donot",
    "forbidden",
    "must_not",
    "not_claim",
)


def build_claim_lint(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs/final_manuscript_package"))
    output_dir = Path(config.get("output_dir", input_root / "claim_lint"))
    manuscript = config.get("manuscript")
    extra_paths = config.get("extra_paths", [])
    ensure_directory(output_dir)

    files = _gather_files(input_root, manuscript, extra_paths, output_dir)
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(_scan_file(file_path))

    return _write_outputs(
        output_dir=output_dir,
        rows=rows,
        scanned_files=files,
        input_config={
            "input_root": str(input_root),
            "manuscript": str(manuscript) if manuscript else None,
            "extra_paths": [str(p) for p in extra_paths],
            "output_dir": str(output_dir),
        },
    )


def _gather_files(
    input_root: Path, manuscript: Any, extra_paths: list[Any], output_dir: Path
) -> list[Path]:
    files: list[Path] = []
    if input_root.is_dir():
        for path in sorted(input_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _SCAN_SUFFIXES:
                if output_dir.resolve() in path.resolve().parents or path.parent.name in _SELF_DIRS:
                    continue
                files.append(path)
    elif input_root.is_file():
        files.append(input_root)
    if manuscript:
        manuscript_path = Path(manuscript)
        if manuscript_path.is_file():
            files.append(manuscript_path)
    for extra in extra_paths:
        extra_path = Path(extra)
        if extra_path.is_file():
            files.append(extra_path)
        elif extra_path.is_dir():
            files.extend(
                p for p in sorted(extra_path.rglob("*")) if p.suffix.lower() in _SCAN_SUFFIXES
            )
    return files


def scan_file(file_path: Path) -> list[dict[str, Any]]:
    """Public entry point: scan a single file and return classified risk-phrase findings."""

    return _scan_file(file_path)


def _scan_file(file_path: Path) -> list[dict[str, Any]]:
    if file_path.suffix.lower() == ".csv":
        return _scan_csv(file_path)
    return _scan_text(file_path)


def _scan_text(file_path: Path) -> list[dict[str, Any]]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for phrase in RISK_PHRASES:
            if phrase in lowered:
                classification, risk = classify_line(lowered, file_path.suffix.lower())
                rows.append(_finding(file_path, line_number, phrase, classification, risk, line))
    return rows


def _scan_csv(file_path: Path) -> list[dict[str, Any]]:
    """Scan a CSV cell-by-cell so disallowed-wording columns are recognized as citations."""

    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return []
            rows: list[dict[str, Any]] = []
            columns = [str(name).lower() for name in header]
            for record in reader:
                line_number = reader.line_num
                # The whole row provides context (e.g. a support_status of
                # ``unsupported_do_not_claim`` marks a quoted claim as a citation).
                row_text = " ".join(str(cell) for cell in record).lower()
                for index, cell in enumerate(record):
                    column = columns[index] if index < len(columns) else ""
                    lowered = cell.lower()
                    for phrase in RISK_PHRASES:
                        if phrase not in lowered:
                            continue
                        if any(cue in column for cue in _DISALLOWED_COLUMN_CUES):
                            classification, risk = "citation_or_table_entry", "low"
                        else:
                            classification, risk = classify_line(row_text, ".csv")
                        rows.append(
                            _finding(file_path, line_number, phrase, classification, risk, cell)
                        )
            return rows
    except (OSError, UnicodeDecodeError, csv.Error):
        return _scan_text(file_path)


def _finding(
    file_path: Path,
    line_number: int,
    phrase: str,
    classification: str,
    risk: str,
    excerpt: str,
) -> dict[str, Any]:
    return {
        "file_path": str(file_path),
        "line_number": line_number,
        "matched_phrase": phrase,
        "risk_level": risk,
        "classification": classification,
        "recommended_action": _recommended_action(classification),
        "line_excerpt": excerpt.strip()[:200],
    }


def classify_line(lowered: str, suffix: str) -> tuple[str, str]:
    """Classify a line that contains a risk phrase into one of four categories."""

    if any(cue in lowered for cue in _CITATION_CUES):
        return "citation_or_table_entry", "low"
    if any(cue in lowered for cue in _NEGATION_CUES):
        return "negated_safe_statement", "low"
    if any(cue in lowered for cue in _LIMITATION_CUES):
        return "limitation_statement", "low"
    return "positive_or_ambiguous_claim", "high"


def _recommended_action(classification: str) -> str:
    if classification == "positive_or_ambiguous_claim":
        return "revise or remove; assert only with explicit negation or a limitation qualifier"
    return "ok: negated / limitation / disallowed-wording citation"


def _summary_markdown(rows: list[dict[str, Any]], scanned_files: list[Path]) -> str:
    high = [r for r in rows if r["risk_level"] == "high"]
    by_class: dict[str, int] = {}
    for row in rows:
        by_class[row["classification"]] = by_class.get(row["classification"], 0) + 1
    lines = [
        "# Claim-Boundary Lint Report",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "## Scan",
        f"- Files scanned: {len(scanned_files)}.",
        f"- Risk-phrase hits: {len(rows)}.",
        f"- High-risk positive overclaims: {len(high)}.",
        "",
        "## Hits by classification",
        *([f"- {name}: {count}" for name, count in sorted(by_class.items())] or ["- none"]),
        "",
        "## High-risk findings",
        *(
            [
                f"- `{row['file_path']}`:{row['line_number']} - "
                f"`{row['matched_phrase']}` - {row['line_excerpt']}"
                for row in high
            ]
            or ["- none (no positive overclaims detected)"]
        ),
        "",
        "## Interpretation",
        '- Negated statements (e.g. "we do not claim quantum speedup"), limitation statements, '
        "and disallowed-wording citations (the DO_NOT_CLAIM list, claim-boundary table) are "
        "classified as safe and are not scientific claims.",
        "- A high-risk hit is a positive or ambiguous assertion of a disallowed claim and must be "
        "revised before submission.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    scanned_files: list[Path],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    report_path = rows_to_table(rows, output_dir / "claim_lint_report.csv", REPORT_COLUMNS)
    summary_path = output_dir / "claim_lint_summary.md"
    summary_path.write_text(_summary_markdown(rows, scanned_files), encoding="utf-8")

    artifacts = {
        "claim_lint_report": str(report_path),
        "claim_lint_summary": str(summary_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    high_risk = [r for r in rows if r["risk_level"] == "high"]
    return {
        "output_dir": output_dir,
        "rows": rows,
        "scanned_files": [str(p) for p in scanned_files],
        "high_risk_count": len(high_risk),
        "high_risk": high_risk,
        "artifacts": artifacts,
    }
