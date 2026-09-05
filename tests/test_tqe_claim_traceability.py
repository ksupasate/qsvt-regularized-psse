from __future__ import annotations

import json
from pathlib import Path

from robust_qsvt_se.paper.tqe_claim_traceability import (
    CLAIMS,
    SUPPORT_STATUSES,
    build_claim_traceability,
)
from robust_qsvt_se.paper.tqe_revision_support_common import find_forbidden

REQUIRED_CLAIM_IDS = {f"CLAIM-{index:02d}" for index in range(1, 16)}


def test_required_claims_present() -> None:
    ids = {claim.claim_id for claim in CLAIMS}
    assert REQUIRED_CLAIM_IDS.issubset(ids)
    assert len(CLAIMS) == 15


def test_every_claim_has_valid_support_status() -> None:
    for claim in CLAIMS:
        assert claim.support_status in SUPPORT_STATUSES
        assert claim.evidence_layer != ""


def test_unsupported_or_future_work_not_marked_supported(tmp_path: Path) -> None:
    run = build_claim_traceability({"output_dir": str(tmp_path)})
    frame = run["frame"]

    # The two "supported" tiers must never be applied to an unsupported/future-work claim.
    flagged = frame[frame["support_status"].isin(["unsupported", "future_work"])]
    assert not flagged["support_status"].isin(["supported", "supported_with_limitations"]).any()
    assert set(frame["support_status"]).issubset(SUPPORT_STATUSES)


def test_safe_wording_has_no_forbidden_phrases(tmp_path: Path) -> None:
    run = build_claim_traceability({"output_dir": str(tmp_path)})
    frame = run["frame"]

    safe_text = "\n".join(frame["manuscript_safe_wording"].astype(str).tolist())
    assert find_forbidden(safe_text) == []


def test_outputs_created_and_manifest(tmp_path: Path) -> None:
    run = build_claim_traceability({"output_dir": str(tmp_path)})

    for name in (
        "tqe_claim_traceability.csv",
        "tqe_claim_traceability.md",
        "tqe_claim_traceability_manifest.json",
    ):
        assert (tmp_path / name).is_file()

    manifest = json.loads(
        (tmp_path / "tqe_claim_traceability_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["claim_count"] == 15
    # CLAIM-12 ties the new signed-readout diagnostic back to a manuscript claim.
    frame = run["frame"]
    signed = frame[frame["claim_id"] == "CLAIM-12"].iloc[0]
    assert "signed" in signed["manuscript_claim"].lower()
    assert signed["support_status"] == "supported_with_limitations"
