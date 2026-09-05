from __future__ import annotations

import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import run_final_claim_safety_audit


def test_final_claim_safety_audit_classifies_contexts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    doc = tmp_path / "claims.md"
    unsafe_doc = tmp_path / "unsafe.md"
    doc.write_text(
        "\n".join(
            [
                "Avoid wording: quantum speedup.",
                "This does not demonstrate quantum advantage.",
            ]
        ),
        encoding="utf-8",
    )
    unsafe_doc.write_text("QSVT outperforms Ridge.\n", encoding="utf-8")
    run = run_final_claim_safety_audit(
        {
            "output_dir": str(tmp_path / "audit"),
            "scan_roots": [str(doc), str(unsafe_doc)],
            "unsafe_phrases": ["quantum speedup", "quantum advantage", "QSVT outperforms Ridge"],
        }
    )
    frame = pd.read_csv(run["output_dir"] / "claim_safety_audit.csv")

    assert "avoid_wording_context" in set(frame["classification"])
    assert "safe_context" in set(frame["classification"])
    assert "unsafe_context" in set(frame["classification"])
    assert (run["output_dir"] / "claim_safety_audit_summary.md").is_file()
