from __future__ import annotations

import json
from pathlib import Path

from robust_qsvt_se.physical_alignment.artifacts import (
    atomic_write_text,
    validate_checksums,
    write_artifact_manifest,
)

CONFIG_PATH = Path("configs/tqe_physical_alignment/campaign.json")












def test_manifest_and_checksum_round_trip(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "evidence.txt", "deterministic evidence\n")
    manifest = write_artifact_manifest(
        tmp_path,
        configuration_id="test",
        configuration_hash="0" * 64,
    )
    assert manifest["file_count_including_manifest_and_checksums"] == 3
    result = validate_checksums(tmp_path)
    assert result["status"] == "pass"
    payload = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert {row["path"] for row in payload["files"]} == {
        "artifact_manifest.json",
        "checksums.sha256",
        "evidence.txt",
    }
