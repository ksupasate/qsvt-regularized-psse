from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from robust_qsvt_se.utils.io import ensure_directory

OUTPUT_ROOT = Path("outputs/tqe_qsvt_additional_experiments")

FLAG_COLUMNS = [
    "file_path",
    "line_number",
    "phrase",
    "severity",
    "surrounding_context",
    "suggested_replacement",
    "claim_boundary_category",
]


@dataclass(frozen=True, slots=True)
class ClaimPhrase:
    phrase: str
    severity: str
    category: str
    replacement: str


CLAIM_PHRASES = [
    ClaimPhrase(
        "quantum speedup",
        "critical",
        "speedup",
        "QSVT-compatible implementation-pathway evidence; no speedup claim.",
    ),
    ClaimPhrase(
        "speedup demonstrated",
        "critical",
        "speedup",
        "degree/resource diagnostics only; speedup is not demonstrated.",
    ),
    ClaimPhrase(
        "QSVT outperforms Ridge",
        "critical",
        "qsvt_over_ridge",
        "QSVT-compatible target is compared with matched Ridge/Tikhonov.",
    ),
    ClaimPhrase(
        "QSVT outperforms Tikhonov",
        "critical",
        "qsvt_over_ridge",
        "QSVT-compatible target is compared with matched Ridge/Tikhonov.",
    ),
    ClaimPhrase(
        "full-scale QSVT",
        "high",
        "scale",
        "selected-subproblem QSVT-compatible validation.",
    ),
    ClaimPhrase(
        "full IEEE-scale QSVT",
        "critical",
        "scale",
        "selected-subproblem and oracle-model evidence only.",
    ),
    ClaimPhrase(
        "scalable quantum circuit implemented",
        "critical",
        "scalable_circuit",
        "sparse-access model and dense selected-subproblem circuits.",
    ),
    ClaimPhrase(
        "hardware implementation",
        "high",
        "hardware",
        "simulator-level circuit construction and resource diagnostics.",
    ),
    ClaimPhrase(
        "hardware execution",
        "critical",
        "hardware",
        "statevector/operator simulation, not hardware execution.",
    ),
    ClaimPhrase(
        "solves nonlinear PSSE",
        "critical",
        "nonlinear",
        "nonlinear AC per-iteration feasibility diagnostics.",
    ),
    ClaimPhrase(
        "nonlinear QSVT solver",
        "high",
        "nonlinear",
        "classical nonlinear loop with QSVT-compatible feasibility analysis.",
    ),
    ClaimPhrase(
        "full-vector readout solved",
        "critical",
        "readout",
        "selected-observable readout; full-vector recovery remains out of scope.",
    ),
    ClaimPhrase(
        "readout solved",
        "high",
        "readout",
        "selected-observable readout diagnostics.",
    ),
    ClaimPhrase(
        "scalable sparse oracle implemented",
        "critical",
        "sparse_oracle",
        "classical sparse index/value oracle emulator and resource model.",
    ),
    ClaimPhrase(
        "reversible sparse oracle implemented",
        "high",
        "sparse_oracle",
        "tiny lookup prototype only; no full reversible value oracle.",
    ),
    ClaimPhrase(
        "field validated",
        "critical",
        "field_data",
        "validated on generated IEEE/PYPOWER benchmark artifacts.",
    ),
    ClaimPhrase(
        "production ready",
        "critical",
        "deployment",
        "research prototype and reproducibility package.",
    ),
]

NEGATION_CUES = (
    "do not",
    "does not",
    "not ",
    "no ",
    "never",
    "without",
    "not claimed",
    "not demonstrate",
    "not demonstrated",
    "not imply",
    "outside scope",
    "out of scope",
    "outside this",
    "remains",
    "remain outside",
    "future work",
    "limitation",
    "claims: 0",
    "would require",
    "rather than",
    "unsafe wording",
    "to avoid",
    "unsafe wording",
    "do-not-claim",
)

SCAN_SUFFIXES = {".md", ".tex", ".txt"}


def audit_claim_boundaries(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = _resolve_config(config)
    output_root = Path(resolved["output_root"])
    reports_dir = ensure_directory(output_root / "reports")
    tables_dir = ensure_directory(output_root / "tables")
    files = gather_scan_files(resolved)
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(scan_text_file(path))
    flags = pd.DataFrame(rows, columns=FLAG_COLUMNS)
    flags_csv = tables_dir / "claim_boundary_flags.csv"
    report_path = reports_dir / "claim_boundary_audit_report.md"
    flags.to_csv(flags_csv, index=False)
    report_path.write_text(
        claim_audit_report(flags=flags, scanned_files=files, flags_csv=flags_csv),
        encoding="utf-8",
    )
    return {
        "flags": flags,
        "scanned_files": files,
        "artifacts": {"flags_csv": flags_csv, "report": report_path},
    }


def gather_scan_files(config: dict[str, Any]) -> list[Path]:
    roots = [Path(path) for path in config["scan_roots"]]
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in SCAN_SUFFIXES:
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and path.suffix.lower() in SCAN_SUFFIXES
                and "tqe_submission_package" not in path.parts
            )
    return sorted(set(files))


def scan_text_file(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        lowered = line.lower()
        for phrase in CLAIM_PHRASES:
            if phrase.phrase.lower() not in lowered:
                continue
            severity = classify_severity(line, phrase)
            rows.append(
                {
                    "file_path": str(path),
                    "line_number": line_number,
                    "phrase": phrase.phrase,
                    "severity": severity,
                    "surrounding_context": line.strip()[:500],
                    "suggested_replacement": suggested_replacement(phrase, severity),
                    "claim_boundary_category": phrase.category,
                }
            )
    return rows


def classify_severity(line: str, phrase: ClaimPhrase) -> str:
    lowered = line.lower()
    safe_context = any(cue in lowered for cue in NEGATION_CUES)
    if safe_context:
        return "low"
    return phrase.severity


def suggested_replacement(phrase: ClaimPhrase, severity: str) -> str:
    if severity == "low":
        return "Verify the context is explicitly negated or framed as a limitation."
    return phrase.replacement


def claim_audit_report(*, flags: pd.DataFrame, scanned_files: list[Path], flags_csv: Path) -> str:
    counts = flags["severity"].value_counts().to_dict() if not flags.empty else {}
    critical = int(counts.get("critical", 0))
    high = int(counts.get("high", 0))
    manual = (
        sorted(flags.loc[flags["severity"].isin(["critical", "high"]), "file_path"].unique())
        if not flags.empty
        else []
    )
    manual_lines = [f"- `{path}`" for path in manual] or ["- None."]
    return "\n".join(
        [
            "# TQE Claim-Boundary Audit Report",
            "",
            "## Summary",
            "",
            f"- Files scanned: {len(scanned_files)}",
            f"- Total flags: {len(flags)}",
            f"- Critical flags: {critical}",
            f"- High flags: {high}",
            f"- Severity counts: `{counts}`",
            "",
            "## Files Requiring Manual Review",
            "",
            *manual_lines,
            "",
            "## Global Wording Rules",
            "",
            "- Say `QSVT-compatible target` rather than implying QSVT numerical superiority.",
            "- Say `selected-subproblem circuit-level validation` rather than "
            "full-scale execution.",
            "- Say `statevector/operator simulation` rather than hardware execution.",
            "- Say `sparse-access oracle model` rather than implemented scalable oracle circuit.",
            "- Say `selected-observable readout` rather than full-vector readout solved.",
            "",
            "## Output",
            "",
            f"- Flags CSV: `{flags_csv}`",
            "",
        ]
    )


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    output_root = Path((config or {}).get("output_root", OUTPUT_ROOT))
    scan_roots = (config or {}).get(
        "scan_roots",
        [
            output_root / "reports",
            Path("manuscript"),
            Path("paper"),
        ],
    )
    return {"output_root": str(output_root), "scan_roots": [str(path) for path in scan_roots]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TQE claim-boundary audit")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    run = audit_claim_boundaries({"output_root": args.output_root})
    print(f"Wrote claim-boundary audit to {run['artifacts']['report']}")


if __name__ == "__main__":
    main()
