from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.qsvt.paper_finalization import run_final_claim_safety_audit


def test_phase2_final_claim_audit_has_no_unsafe_contexts(tmp_path: Path) -> None:
    root = tmp_path / "phase2"
    root.mkdir()
    (root / "alpha_selection_report.md").write_text(
        (
            "Alpha selection is diagnostic and controlled-benchmark-specific. "
            "It is not a field-calibrated operational rule.\n"
        ),
        encoding="utf-8",
    )
    (root / "claim_safe_wording.md").write_text(
        "\n".join(
            [
                "# Claim Boundary",
                "Avoid wording: quantum speedup.",
                "Do not claim hardware execution.",
                "This does not demonstrate quantum advantage.",
            ]
        ),
        encoding="utf-8",
    )

    run = run_final_claim_safety_audit(
        {
            "output_dir": str(tmp_path / "audit"),
            "scan_roots": [str(root)],
            "unsafe_phrases": [
                "quantum speedup",
                "quantum advantage",
                "hardware execution",
            ],
        }
    )
    frame = pd.read_csv(run["output_dir"] / "claim_safety_audit.csv")

    assert not (frame["classification"] == "unsafe_context").any()
    assert "avoid_wording_context" in set(frame["classification"])
    assert "safe_context" in set(frame["classification"])
    assert "field-calibrated operational rule" in (root / "alpha_selection_report.md").read_text(
        encoding="utf-8"
    )
