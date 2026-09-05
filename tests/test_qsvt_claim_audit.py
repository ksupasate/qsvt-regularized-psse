from __future__ import annotations

from pathlib import Path


def test_docs_do_not_claim_full_ieee300_hardware_qsvt_execution() -> None:
    files = [
        Path("README.md"),
        Path("docs/qsvt_implementation_scope.md"),
        Path("docs/dataset_strategy.md"),
    ]
    combined = " ".join(
        "\n".join(path.read_text(encoding="utf-8").lower() for path in files).split()
    )

    forbidden_claims = [
        "quantum speedup",
        "full ieee300 hardware qsvt complete",
        "full ieee300 hardware-level qsvt complete",
        "validated on real pmu/scada field data",
    ]
    assert "level 6" in combined
    assert "not complete and not claimed" in combined
    for phrase in forbidden_claims:
        if phrase == "quantum speedup":
            assert (
                "do not imply quantum speedup" in combined
                or "do not claim quantum speedup" in combined
                or "no quantum speedup" in combined
            )
            continue
        assert phrase not in combined
