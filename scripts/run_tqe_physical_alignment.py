#!/usr/bin/env python3
"""Reproduce the additive physical-alignment and nonlinear evidence campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from robust_qsvt_se.physical_alignment.artifacts import (
    atomic_write_json,
    compare_protected_hashes,
    validate_checksums,
    write_artifact_manifest,
)
from robust_qsvt_se.physical_alignment.config import load_campaign_config
from robust_qsvt_se.physical_alignment.nonlinear_ac import run_nonlinear_campaign
from robust_qsvt_se.physical_alignment.physical_audit import run_physical_audit
from robust_qsvt_se.physical_alignment.reporting import generate_reports
from robust_qsvt_se.physical_alignment.statistics import run_structure_statistics


def _stages(requested: str) -> list[str]:
    if requested == "all":
        return ["physical", "statistics", "nonlinear", "protected", "reports", "manifest"]
    return [requested]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["physical", "statistics", "nonlinear", "protected", "reports", "manifest", "all"],
    )
    parser.add_argument("--config", default="configs/tqe_physical_alignment/campaign.json")
    parser.add_argument("--skip-statevector", action="store_true")
    args = parser.parse_args(argv)
    config = load_campaign_config(args.config)
    root = Path(config["output_root"])
    for stage in _stages(args.stage):
        if stage == "physical":
            print(run_physical_audit(args.config))
        elif stage == "statistics":
            print(run_structure_statistics(args.config))
        elif stage == "nonlinear":
            print(
                run_nonlinear_campaign(
                    args.config,
                    run_statevector_boundary=not args.skip_statevector,
                )
            )
        elif stage == "protected":
            audit = compare_protected_hashes(Path.cwd(), config["protected_baseline_path"])
            atomic_write_json(root / "protected_hash_audit.json", audit)
            print(audit["status"])
        elif stage == "reports":
            print(generate_reports(args.config))
        elif stage == "manifest":
            manifest = write_artifact_manifest(
                root,
                configuration_id=config["configuration_id"],
                configuration_hash=config["configuration_hash"],
            )
            print(manifest)
            print(validate_checksums(root))


if __name__ == "__main__":
    main()
