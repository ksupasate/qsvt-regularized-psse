"""Integration tests: run the four experiments minimally and aggregate readiness."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.tqe_revision_conditioning_boundary import run_conditioning_boundary
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in
from robust_qsvt_se.paper.tqe_revision_readiness import run_readiness
from robust_qsvt_se.paper.tqe_revision_readout_statistics import run_readout_statistics
from robust_qsvt_se.paper.tqe_revision_resource_ledger import run_resource_ledger
from robust_qsvt_se.paper.tqe_revision_sparse_oracle_demo import run_sparse_oracle_demo

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

_EXPECTED_ISSUES = {
    "W1_regime_mismatch",
    "W2_novelty_thin",
    "W3_no_end_to_end_resource_number",
    "W4_single_readout_draw",
    "W5_stateprep_blockencoding_literature_gap",
    "W6_overhedging",
    "W7_motivation_self_undercut",
}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("tqe_revision_experiments")
    readout_dir = root / "readout_statistics"
    boundary_dir = root / "conditioning_boundary"
    resource_dir = root / "end_to_end_resource_case"
    sparse_dir = root / "sparse_block_encoding_demo"
    readiness_dir = root / "revision_readiness"

    run_readout_statistics(
        {"output_dir": str(readout_dir), "num_seeds": 2, "shots_grid": [500, 5000]}
    )
    run_conditioning_boundary(
        {
            "output_dir": str(boundary_dir),
            "sizes": [4],
            "kappa_grid": [100.0],
            "matrix_seeds": [0],
            "lambda_grid": [1e-4, 6.9e-2],
            "degree_grid": [15, 31, 63],
            "ieee_cases": [],
            "ieee_sizes": [],
            "include_stressed": False,
            "heatmap_shape": "4x4",
        }
    )
    run_resource_ledger({"output_dir": str(resource_dir), "timing_repeats": 3})
    run_sparse_oracle_demo({"output_dir": str(sparse_dir), "sizes": [4]})
    readiness = run_readiness(
        {
            "output_dir": str(readiness_dir),
            "readout_dir": str(readout_dir),
            "boundary_dir": str(boundary_dir),
            "resource_dir": str(resource_dir),
            "sparse_dir": str(sparse_dir),
            "experiments_root": str(root),
        }
    )
    return readiness, root


def test_reviewer_matrix_covers_w1_to_w7(full_run):
    readiness, _ = full_run
    ids = set(readiness["reviewer_matrix"]["reviewer_issue_id"])
    assert ids >= _EXPECTED_ISSUES


def test_readiness_files_created(full_run):
    readiness, _ = full_run
    output_dir = readiness["output_dir"]
    for name in [
        "reviewer_issue_response_matrix.csv",
        "claim_boundary_audit.csv",
        "new_artifact_manifest.csv",
        "recommended_manuscript_changes.md",
        "final_readiness_report.md",
        "manifest.json",
    ]:
        assert (output_dir / name).is_file(), name


def test_claim_audit_has_no_unsupported_overclaim(full_run):
    readiness, _ = full_run
    claims = readiness["claim_audit"]
    # The conservative non-claims must be present and marked supported.
    for phrase in [
        "No quantum speed-up",
        "No PMU/SCADA field-measurement validation",
        "does not beat Ridge",
    ]:
        matched = claims[claims["claim"].str.contains(phrase, case=False, regex=False)]
        assert not matched.empty
        assert (matched["support_status"] == "supported").all()


def test_forbidden_wording_self_check_clean(full_run):
    readiness, _ = full_run
    assert readiness["forbidden_violations"] == []
    for name in ["recommended_manuscript_changes.md", "final_readiness_report.md"]:
        assert not forbidden_in((readiness["output_dir"] / name).read_text())


def test_all_experiment_manifests_present_and_safe(full_run):
    _, root = full_run
    manifests = list(root.rglob("manifest.json"))
    assert len(manifests) >= 5  # A, B, C, D, E
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["fabricates_results"] is False
        assert not forbidden_in(manifest.get("claim_boundary", ""))


def test_new_artifact_manifest_non_empty(full_run):
    readiness, _ = full_run
    frame = readiness["artifact_manifest"]
    assert not frame.empty
    assert "sha256" in frame.columns
    assert frame["sha256"].notna().all()
