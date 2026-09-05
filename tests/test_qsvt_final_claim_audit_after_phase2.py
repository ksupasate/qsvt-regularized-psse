from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import run_final_claim_safety_audit


def test_final_claim_audit_after_phase2_has_no_unsafe_contexts(tmp_path: Path) -> None:
    doc = tmp_path / "phase2_summary.md"
    doc.write_text(
        "\n".join(
            [
                "This does not demonstrate quantum speedup.",
                "This does not demonstrate quantum advantage.",
                "This is not hardware validation.",
                "This is not full hardware execution.",
                "This is not real PMU/SCADA field-data validation.",
                "This does not claim QSVT outperforms Ridge.",
            ]
        ),
        encoding="utf-8",
    )
    run = run_final_claim_safety_audit(
        {
            "output_dir": str(tmp_path / "audit"),
            "scan_roots": [str(doc)],
            "unsafe_phrases": [
                "quantum speedup",
                "quantum advantage",
                "QSVT outperforms Ridge",
                "hardware validation",
                "full hardware execution",
                "real PMU/SCADA",
                "field-data validation",
            ],
        }
    )
    frame = pd.read_csv(run["output_dir"] / "claim_safety_audit.csv")

    assert "unsafe_context" not in set(frame["classification"])
    assert (run["output_dir"] / "claim_safety_audit_summary.md").is_file()
