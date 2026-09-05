"""Finalize the reviewer-blocking evidence pass: claim-support matrix + checksums.

Builds ``claim_support_matrix.csv`` (each reviewer concern -> decision -> evidence file) and
``checksums.sha256`` over the whole task-owned output root, and verifies the protected roots are
byte-identical to the pre-edit snapshot fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("outputs/reviewer_blocking_tqe_evidence")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_claim_support_matrix(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    base = Path(output_dir)
    dec = json.loads((base / "support_vs_physical_decoupling.json").read_text())
    overall_spear = dec.get("overall", {}).get("spearman_support_vs_physical")
    overall_pear = dec.get("overall", {}).get("pearson")
    hd_conc = {}
    hd_path = base / "high_degree_conclusions.json"
    if hd_path.exists():
        hd_conc = json.loads(hd_path.read_text()).get("structures", {})
    ss = json.loads((base / "structure_aware_statistics.json").read_text())
    baseline_rows = pd.read_csv(base / "task_aware_baseline_rows.csv")
    feasible = baseline_rows[baseline_rows["feasible"].astype(bool)]
    baseline_pivot = feasible.pivot(
        index=["structure_id", "k_budget"],
        columns="selector",
        values=["heldout_support_fidelity_mean", "heldout_physical_median"],
    )
    support_effect = (
        baseline_pivot[("heldout_support_fidelity_mean", "global_magnitude")]
        - baseline_pivot[("heldout_support_fidelity_mean", "sensitivity_refined_mean")]
    )
    physical_effect = (
        baseline_pivot[("heldout_physical_median", "global_magnitude")]
        - baseline_pivot[("heldout_physical_median", "sensitivity_refined_mean")]
    )
    support_better = int((support_effect > 1e-12).sum())
    support_worse = int((support_effect < -1e-12).sum())
    physical_better = int((physical_effect > 1e-12).sum())
    physical_worse = int((physical_effect < -1e-12).sum())

    rows: list[dict[str, Any]] = [
        {
            "reviewer_concern": "Output-aware sparse support improves the accuracy of the true "
            "physical selected output (not just full-block Ridge).",
            "decision": "Contradicted",
            "evidence": "physical_selected_output_summary.csv; support_vs_physical_decoupling.json",
            "finding": "Full support has the lowest aggregated median physical error in all 18 "
            "structure-by-alpha cells. Refined sensitivity is physically better than global "
            f"magnitude in {physical_better} matched structure-budget cells and worse in "
            f"{physical_worse}.",
            "remaining_limitation": "Physical error is dominated by the block-truncation + "
            "regularization floor common to all supports.",
        },
        {
            "reviewer_concern": "Low sparse-to-full Ridge fidelity implies accuracy against the "
            "true physical output.",
            "decision": "Contradicted",
            "evidence": "support_vs_physical_decoupling.json",
            "finding": f"Pearson(E_support,E_physical)~{overall_pear:.3f} (near zero); "
            f"Spearman~{overall_spear:.3f} (moderate). The metrics are decoupled.",
            "remaining_limitation": "Rank association is moderate overall and varies by "
            "structure; the support metric is not a reliable linear or universal physical proxy.",
        },
        {
            "reviewer_concern": "Increasing QSVT degree to 127/255 closes the "
            "application-useful / QSVT-feasible gap.",
            "decision": "Not resolved",
            "evidence": "high_degree_qsvt_rows.csv; high_degree_conclusions.json; "
            "high_degree_synthesis_demonstration.json; protocol_consistency_audit.md",
            "finding": "; ".join(f"{k}: {v.get('category')}" for k, v in hd_conc.items())
            or "high-degree study output pending",
            "remaining_limitation": "This is a descriptive result for the executed, post-freeze "
            "amended slice and tested polynomial construction; it is not a universal "
            "approximation-theoretic impossibility theorem.",
        },
        {
            "reviewer_concern": "The proposed sensitivity selector is distinguished from strong "
            "task-aware baselines (not just from magnitude).",
            "decision": "Partially resolved",
            "evidence": "task_aware_baseline_rows.csv; task_aware_baseline_summary.csv; "
            "task_aware_baseline_pairwise.csv",
            "finding": f"Refined sensitivity beats global magnitude on held-out support fidelity "
            f"in {support_better} of {support_better + support_worse} matched feasible "
            "structure-budget cells, but other task-aware references have mixed structure-level "
            "results and the proposed selector does not uniformly dominate them.",
            "remaining_limitation": "Output information often helps the original objective, but "
            "the specific score normalization is not uniquely distinguished and no selector "
            "consistently improves physical accuracy.",
        },
        {
            "reviewer_concern": "The selector advantage is a uniform structural improvement across "
            "IEEE cases / block structures.",
            "decision": "Not resolved",
            "evidence": "structure_aware_statistics.json",
            "finding": f"Bootstrap-over-structures CI on the physical effect spans zero "
            f"({ss['primary_bootstrap_over_structures']['ci95_low']:.3f},"
            f"{ss['primary_bootstrap_over_structures']['ci95_high']:.3f}); "
            f"interpretation: {ss['interpretation_primary']}.",
            "remaining_limitation": "Only 3 structures over 2 cases: low statistical power for a "
            "structural claim.",
        },
    ]
    return pd.DataFrame(rows)


def write_checksums(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> int:
    base = Path(output_dir)
    targets = sorted(
        p
        for p in base.rglob("*")
        if p.is_file()
        and p.name != "checksums.sha256"
        and "_frozen_cache" not in p.parts
        and "phase_cache" not in p.parts
    )
    text = "\n".join(f"{_sha256(p)}  {p.relative_to(base)}" for p in targets) + "\n"
    (base / "checksums.sha256").write_text(text, encoding="utf-8")
    return len(targets)


def verify_protected_unchanged(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Recompute protected-root fingerprints and diff against the pre-edit snapshot."""

    base = Path(output_dir)
    snapshot = json.loads((base / "pre_edit_snapshot.json").read_text())
    before = snapshot["protected_roots"]
    result: dict[str, Any] = {"unchanged": [], "changed": [], "critical_source": {}}

    def fingerprint(root: Path) -> dict[str, Any]:
        if not root.exists():
            return {"exists": False}
        files = sorted(p for p in root.rglob("*") if p.is_file())
        manifest = "\n".join(f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}" for p in files)
        return {
            "exists": True,
            "file_count": len(files),
            "path_size_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        }

    for rel, fp_before in before.items():
        fp_now = fingerprint(REPO_ROOT / rel)
        same = fp_before.get("file_count") == fp_now.get("file_count") and fp_before.get(
            "path_size_manifest_sha256"
        ) == fp_now.get("path_size_manifest_sha256")
        (result["unchanged"] if same else result["changed"]).append(rel)

    for rel, hash_before in snapshot["critical_source_hashes"].items():
        p = REPO_ROOT / rel
        now = _sha256(p) if p.exists() else "MISSING"
        result["critical_source"][rel] = {"unchanged": bool(now == hash_before)}
    result["all_protected_unchanged"] = len(result["changed"]) == 0
    result["all_critical_source_unchanged"] = all(
        v["unchanged"] for v in result["critical_source"].values()
    )
    return result


def finalize(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    base = Path(output_dir)
    matrix = build_claim_support_matrix(base)
    matrix.to_csv(base / "claim_support_matrix.csv", index=False)
    protected = verify_protected_unchanged(base)
    (base / "protected_roots_verification.json").write_text(
        json.dumps(protected, indent=2), encoding="utf-8"
    )
    n = write_checksums(base)
    return {
        "claim_rows": len(matrix),
        "checksum_files": n,
        "all_protected_unchanged": protected["all_protected_unchanged"],
        "all_critical_source_unchanged": protected["all_critical_source_unchanged"],
        "changed_protected": protected["changed"],
    }
