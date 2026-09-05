from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_qsvt_se.paper.classical_audit_recheck import build_classical_audit_recheck

_EXPECTED = {
    "pseudoinverse",
    "normal_equation_wls",
    "ridge_tikhonov",
    "truncated_svd",
    "huber_irls",
    "lav",
    "hhl_style_proxy",
    "qsvt_target_classical",
}


def _write_aggregate(path: Path, estimators: list[str], case: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"estimator": estimators, "case_name": [case] * len(estimators)})
    frame.to_csv(path, index=False)


def test_recheck_classifies_all_eight_estimators(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_classical_audit_recheck(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "recheck")}
    )
    coverage = pd.read_csv(run["artifacts"]["estimator_coverage_recheck"])
    assert set(coverage["estimator"]) == _EXPECTED
    assert len(coverage) == 8


def test_recheck_does_not_claim_lav_all_case_when_only_one_case(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    # LAV present only for IEEE14 -> must not be marked as covering all cases.
    _write_aggregate(
        input_root / "real_ieee14_seed10" / "aggregate_metrics.csv",
        ["lav", "ridge", "qsvt_regularized"],
        "ieee14",
    )
    run = build_classical_audit_recheck(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "recheck")}
    )
    coverage = pd.read_csv(run["artifacts"]["estimator_coverage_recheck"])
    lav = coverage[coverage["estimator"] == "lav"].iloc[0]
    assert lav["all_cases_covered"] == "no"
    assert lav["coverage_status"] != "complete"

    missing = pd.read_csv(run["artifacts"]["missing_classical_evidence_recheck"])
    assert (missing["estimator"].astype(str) == "lav").any()


def test_recheck_keeps_qsvt_ridge_equivalent_no_superiority(tmp_path: Path) -> None:
    input_root = tmp_path / "outputs"
    input_root.mkdir()
    run = build_classical_audit_recheck(
        {"input_root": str(input_root), "output_dir": str(tmp_path / "recheck")}
    )
    coverage = pd.read_csv(run["artifacts"]["estimator_coverage_recheck"])
    qsvt = coverage[coverage["estimator"] == "qsvt_target_classical"].iloc[0]
    assert "ridge-equivalent" in str(qsvt["notes"]).lower()

    text = Path(run["artifacts"]["phase2_classical_recheck"]).read_text(encoding="utf-8").lower()
    # The recheck affirms Ridge-equivalence and explicitly states no superiority is asserted.
    assert "no qsvt-over-ridge" in text
    assert "numerically identical to ridge" in text
    # No affirmative superiority phrasing (the boundary disclaimer may list disallowed wording).
    assert "qsvt outperforms ridge" not in text
    assert "qsvt is superior" not in text
