"""Phase 5: claim-boundary table and DO_NOT_CLAIM checklist.

Distils the final claim-support matrix into a concise, citable claim-boundary table
and a writing-safety checklist for the manuscript. It reads the existing claim matrix
and limitations and never promotes an unsupported claim: QSVT-over-Ridge superiority,
quantum speedup/advantage, real PMU/SCADA validation stay unsupported, and full-vector
readout stays assumption-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robust_qsvt_se.paper import PAPER_CLAIM_BOUNDARY
from robust_qsvt_se.paper._common import read_csv, rows_to_table
from robust_qsvt_se.qsvt.engineering_io import write_manifest
from robust_qsvt_se.utils.io import ensure_directory

SOURCE_SCRIPT = "scripts/build_claim_boundary_docs.py"

BOUNDARY_COLUMNS = [
    "claim_id",
    "claim_text",
    "support_status",
    "category",
    "allowed_wording",
    "disallowed_wording",
    "manuscript_guidance",
]

_CATEGORIES = (
    "supported",
    "supported_with_limitations",
    "assumption_only",
    "unsupported_do_not_claim",
)
_GUIDANCE = {
    "supported": "State directly with the controlled-benchmark scope.",
    "supported_with_limitations": "State with the documented limitation attached.",
    "assumption_only": "Present as an explicit assumption / future work, not a result.",
    "unsupported_do_not_claim": "Do NOT claim; keep in limitations as out of scope.",
}

DO_NOT_CLAIM_ITEMS = (
    "Do not claim quantum speedup or quantum advantage.",
    "Do not claim QSVT numerically outperforms Ridge/Tikhonov.",
    "Do not claim full IEEE-scale gate-level QSVT execution.",
    "Do not claim full nonlinear AC QSVT solver.",
    "Do not claim real PMU/SCADA validation.",
    "Do not claim field-calibrated stress statistics.",
    "Do not claim full-vector readout is solved.",
    "Do not claim the HHL-style proxy is full HHL.",
    "Do not claim bad-data robustness is due to QSVT; robust baselines address outliers.",
    "Do not claim IEEE cases directly provide measurements; rows are generated from the model.",
)


def build_claim_boundary_docs(config: dict[str, Any]) -> dict[str, Any]:
    input_root = Path(config.get("input_root", "outputs"))
    package_root = Path(config.get("package_root", input_root / "final_manuscript_package"))
    output_dir = Path(config.get("output_dir", package_root / "claim_boundaries"))
    ensure_directory(output_dir)

    matrix = read_csv(package_root / "claim_support_matrix_final.csv")
    boundary_rows = _boundary_rows(matrix)
    counts = _counts(boundary_rows)

    return _write_outputs(
        output_dir=output_dir,
        boundary_rows=boundary_rows,
        counts=counts,
        input_config={
            "input_root": str(input_root),
            "package_root": str(package_root),
            "output_dir": str(output_dir),
        },
    )


def _boundary_rows(matrix: Any) -> list[dict[str, Any]]:
    if matrix.empty:
        return []
    rows = []
    for _, record in matrix.iterrows():
        status = str(record.get("support_status", ""))
        category = status if status in _CATEGORIES else "supported_with_limitations"
        rows.append(
            {
                "claim_id": record.get("claim_id", ""),
                "claim_text": record.get("claim_text", ""),
                "support_status": status,
                "category": category,
                "allowed_wording": record.get("allowed_wording", ""),
                "disallowed_wording": record.get("disallowed_wording", ""),
                "manuscript_guidance": _GUIDANCE.get(
                    category, _GUIDANCE["supported_with_limitations"]
                ),
            }
        )
    return rows


def _counts(boundary_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in _CATEGORIES}
    for row in boundary_rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def _summary_markdown(boundary_rows: list[dict[str, Any]], counts: dict[str, int]) -> str:
    lines = ["# Claim-Boundary Summary", "", PAPER_CLAIM_BOUNDARY, "", "## Category counts"]
    for category in _CATEGORIES:
        lines.append(f"- {category}: {counts.get(category, 0)}")
    lines.append("")
    for category in _CATEGORIES:
        lines.append(f"## {category}")
        members = [r for r in boundary_rows if r["category"] == category]
        if not members:
            lines.append("- none")
        for row in members:
            lines.append(
                f"- **{row['claim_id']}**: {row['claim_text']} ({row['manuscript_guidance']})"
            )
        lines.append("")
    lines.append(
        "Unsupported claims (QSVT-over-Ridge superiority, quantum speedup/advantage, real "
        "PMU/SCADA validation) are preserved as do-not-claim; full-vector readout is assumption "
        "only. See DO_NOT_CLAIM.md."
    )
    lines.append("")
    return "\n".join(lines)


def _do_not_claim_markdown() -> str:
    lines = [
        "# DO_NOT_CLAIM Checklist",
        "",
        PAPER_CLAIM_BOUNDARY,
        "",
        "Use this checklist while writing. None of the following may be claimed:",
        "",
    ]
    lines.extend(f"- [ ] {item}" for item in DO_NOT_CLAIM_ITEMS)
    lines.extend(
        [
            "",
            "## Allowed framing",
            "- Controlled IEEE benchmark with generated measurement rows.",
            "- AC-linearized weighted update; selected-(4x4) QSVT solver prototype.",
            "- QSVT-target classical equals Ridge/Tikhonov under matched alpha and the same scalar "
            "filter.",
            "- Observable-first readout of selected observables; full-vector readout remains an "
            "assumption.",
            "- Resource-model and gate-validated selected-subproblem evidence; supported with "
            "limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    boundary_rows: list[dict[str, Any]],
    counts: dict[str, int],
    input_config: dict[str, Any],
) -> dict[str, Any]:
    table_path = rows_to_table(
        boundary_rows, output_dir / "claim_boundary_table.csv", BOUNDARY_COLUMNS
    )
    summary_path = output_dir / "claim_boundary_summary.md"
    do_not_path = output_dir / "DO_NOT_CLAIM.md"
    summary_path.write_text(_summary_markdown(boundary_rows, counts), encoding="utf-8")
    do_not_path.write_text(_do_not_claim_markdown(), encoding="utf-8")

    artifacts = {
        "claim_boundary_table": str(table_path),
        "claim_boundary_summary": str(summary_path),
        "DO_NOT_CLAIM": str(do_not_path),
    }
    write_manifest(
        output_dir,
        artifacts=artifacts,
        input_config=input_config,
        claim_boundary=PAPER_CLAIM_BOUNDARY,
    )
    return {
        "output_dir": output_dir,
        "boundary_rows": boundary_rows,
        "counts": counts,
        "do_not_claim_items": list(DO_NOT_CLAIM_ITEMS),
        "artifacts": artifacts,
    }
