"""Tests for the Phase 9 (optional) 8x8 sparse-access wrapper feasibility + validation."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.phase9_sparse_wrapper_8x8 import run_phase9_sparse_wrapper_8x8

pytest.importorskip("qiskit")


@pytest.fixture(scope="module")
def wrapper_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase9_sparse_wrapper_8x8")
    # 4-bit only and a short konig timeout keep the test fast; the blocker is
    # non-termination, so a small guard still detects it deterministically.
    return run_phase9_sparse_wrapper_8x8(
        {
            "output_dir": str(output_dir),
            "value_precision_bits": [4],
            "konig_timeout": 3.0,
        }
    )


def test_lookup_oracle_validates_at_8x8(wrapper_run):
    rows = wrapper_run["lookup_rows"]
    assert rows
    for row in rows:
        assert row["lookup_validation_passed"] is True
        assert row["max_column_lookup_error"] == 0
        assert row["max_value_register_error"] == 0
        assert row["n_qubits"] == 12  # 3 row + 1 local + 3 col + 4 value + 1 rotation
        assert row["lookup_calls_validated"] == 16


def test_block_encoding_wrapper_is_blocked_at_konig(wrapper_run):
    report = wrapper_run["report"]
    assert report["block_encoding_wrapper_status"] == "blocked"
    coloring = report["feasibility_audit"]["edge_coloring_requirement"]
    assert coloring["coloring_exists_by_theorem"] is True
    assert coloring["reused_implementation_terminates"] is False
    assert coloring["blocker"]  # a non-empty blocker string is recorded


def test_overall_status_and_files(wrapper_run):
    assert wrapper_run["report"]["overall_status"] == (
        "lookup_oracle_validated_block_encoding_wrapper_blocked"
    )
    output_dir = wrapper_run["output_dir"]
    for name in [
        "feasibility_audit.json",
        "lookup_oracle_validation.csv",
        "README.md",
        "checksums.sha256",
        "manifest.json",
    ]:
        assert (output_dir / name).is_file(), name


def test_feasibility_audit_documents_registers_and_scope(wrapper_run):
    audit = wrapper_run["report"]["feasibility_audit"]
    assert audit["block_shape"] == "8x8"
    assert audit["register_layout"]["row_qubits"] == 3
    assert audit["register_layout"]["column_qubits"] == 3
    assert audit["sparsity_pattern"]["max_row_sparsity"] <= 2
    manifest = json.loads((wrapper_run["output_dir"] / "manifest.json").read_text(encoding="utf-8"))
    assert "not an ieee-scale sparse block encoding" in manifest["claim_boundary"].lower()
