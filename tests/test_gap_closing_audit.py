from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.gap_closing_audit import (
    RUNTIME_CLASSES,
    TRIAGE_COLUMNS,
    build_gap_closing_audit,
)


def _write_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "claim_id": ["C01", "C11", "C12", "C13", "C10", "C14", "C17"],
            "claim_text": [
                "Benchmark cases provide controlled measurement rows",
                "QSVT achieves numerical superiority over Ridge/Tikhonov",
                "QSVT demonstrates quantum speedup or quantum advantage",
                "Results are validated on real PMU/SCADA field data",
                "A full IEEE-scale QSVT sparse-oracle pathway is assumed but not executed",
                "Structured stress / measurement ablation sweeps degrade conditioning",
                "Per-measurement-type ablation identifies critical measurement types",
            ],
            "support_status": [
                "supported",
                "unsupported_do_not_claim",
                "unsupported_do_not_claim",
                "unsupported_do_not_claim",
                "assumption_only",
                "supported_with_limitations",
                "missing_evidence",
            ],
            "supporting_artifacts": [
                "outputs/measurement_inventory",
                "none",
                "none",
                "none",
                "outputs/sparse_oracle_assumption_ledger",
                "none",
                "outputs/final_manuscript_package/phase5_structured_stress_ablation",
            ],
            "limitation_note": ["", "", "", "", "", "compound/spatial stress future work", ""],
        }
    ).to_csv(package_dir / "claim_support_matrix_final.csv", index=False)
    pd.DataFrame(
        {
            "source_phase": ["phase2", "phase2_recheck"],
            "item": [
                "alpha-resolved main-result table (explicit alpha column per row)",
                "LAV estimator main results across cases",
            ],
            "category": ["missing_consolidation_or_future_work"] * 2,
            "importance": ["medium", "low"],
            "status": ["missing", "missing"],
            "recommended_action": ["consolidate in Phase 3", "add LAV"],
        }
    ).to_csv(package_dir / "remaining_missing_evidence.csv", index=False)


def test_triage_partitions_and_columns(tmp_path: Path) -> None:
    package_dir = tmp_path / "final_manuscript_package"
    _write_package(package_dir)
    run = build_gap_closing_audit(
        {
            "input_root": str(tmp_path),
            "package_dir": str(package_dir),
            "output_dir": str(tmp_path / "gap_closing_audit"),
        }
    )
    triage = pd.read_csv(run["artifacts"]["remaining_gap_triage"])
    assert list(triage.columns) == TRIAGE_COLUMNS
    # supported C01 is not a gap; every other claim + both missing items are triaged.
    assert "C01" not in set(triage["related_claim_id"].dropna())
    assert {"C11", "C12", "C13", "C10", "C14", "C17"} <= set(triage["related_claim_id"].dropna())
    assert triage["gap_id"].is_unique
    assert set(triage["expected_runtime_class"]) <= set(RUNTIME_CLASSES)


def test_overclaims_must_remain_missing(tmp_path: Path) -> None:
    package_dir = tmp_path / "final_manuscript_package"
    _write_package(package_dir)
    run = build_gap_closing_audit(
        {"package_dir": str(package_dir), "output_dir": str(tmp_path / "audit")}
    )
    must_remain = pd.read_csv(run["artifacts"]["must_remain_missing"])
    remaining_claims = set(must_remain["related_claim_id"].dropna())
    # Hard overclaims and the assumption-only pathway are never closeable here.
    assert {"C11", "C12", "C13", "C10"} <= remaining_claims
    assert (must_remain["expected_runtime_class"] == "not_available").all()
    assert (must_remain["can_close_from_existing_artifacts"] == "no").all()


def test_per_type_ablation_routed_to_fast_run(tmp_path: Path) -> None:
    package_dir = tmp_path / "final_manuscript_package"
    _write_package(package_dir)
    run = build_gap_closing_audit(
        {"package_dir": str(package_dir), "output_dir": str(tmp_path / "audit")}
    )
    new_runs = pd.read_csv(run["artifacts"]["requires_new_fast_runs"])
    c17 = new_runs[new_runs["related_claim_id"] == "C17"]
    assert not c17.empty
    assert c17.iloc[0]["expected_runtime_class"] == "fast_regeneration"
    assert "measurement_type_ablation" in c17.iloc[0]["recommended_action"]
    # The hard overclaims are excluded from the fast-run bucket.
    assert not {"C11", "C12", "C13"} & set(new_runs["related_claim_id"].dropna())
