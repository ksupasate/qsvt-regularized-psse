"""Tests for the Phase 10 WP A complete 8x8 sparse block-encoding wrapper."""

from __future__ import annotations

import json

import pytest

from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.paper.phase10_sparse_wrapper_8x8_complete import (
    run_phase10_sparse_wrapper_complete,
)

TOLERANCE = 1.0e-9


@pytest.fixture(scope="module")
def wrapper_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase10_sparse_wrapper_8x8_complete")
    # Primary precision only and no transpilation keeps the test fast; the
    # degree-31 QSVT integration is the load-bearing check and is kept.
    return run_phase10_sparse_wrapper_complete(
        {
            "output_dir": str(output_dir),
            "value_precision_bits": [6],
            "qsvt_degrees": [31],
            "transpile": False,
        }
    )


def test_wrapper_statevector_validates_at_8x8(wrapper_run):
    rows = wrapper_run["validation_rows"]
    assert rows
    for row in rows:
        assert row["status"] == "statevector_validated"
        assert row["top_left_reconstruction_error"] <= TOLERANCE
        assert row["unitarity_error"] <= TOLERANCE
        assert row["statevector_max_error"] <= TOLERANCE
        assert row["lookup_value_max_error"] <= TOLERANCE
        assert row["slots"] == 3
        assert row["qubits"] == 6  # 3 index + 2 slot + 1 rotation ancilla
        assert row["nnz"] == 16


def test_edge_coloring_blocker_is_fixed_and_documented(wrapper_run):
    coloring_path = wrapper_run["output_dir"] / "edge_coloring_validation.json"
    report = json.loads(coloring_path.read_text(encoding="utf-8"))
    assert report["pattern"]["konig_minimum_slots"] == 3
    feasibility = report["phase9_blocker_resolution"]["layer_1_feasibility"]
    assert feasibility["two_slot_coloring_status"] == "infeasible"
    assert "maximum row/column degree 3" in feasibility["diagnosis"]
    assert report["validation"]["valid"] is True
    assert report["validation"]["real_edges_covered_exactly_once"] is True


def test_qsvt_integration_matches_dense_dilation_and_ridge(wrapper_run):
    rows = wrapper_run["qsvt_rows"]
    assert rows
    final = rows[-1]
    assert final["status"] == "statevector_validated"
    assert final["degree"] == 31
    assert final["sparse_vs_dense_action_error"] <= 1.0e-8
    assert final["sparse_vs_exact_svt_error"] <= 1.0e-6
    assert final["sparse_update_relative_error_vs_ridge"] <= 0.05
    assert final["sparse_operator_vs_statevector_error"] <= 1.0e-10
    # Physical recovery must be the single factor C/beta (Option B convention).
    expected = final["bound_C"] / final["beta_effective"]
    assert final["physical_recovery_factor_C_over_beta"] == pytest.approx(expected, rel=1e-12)


def test_outputs_exist_and_readme_is_claim_safe(wrapper_run):
    output_dir = wrapper_run["output_dir"]
    for name in (
        "sparse_wrapper_8x8_validation.csv",
        "sparse_wrapper_8x8_qsvt_validation.csv",
        "sparse_wrapper_8x8_circuit_metadata.json",
        "sparse_wrapper_8x8_block_reconstruction.json",
        "edge_coloring_validation.json",
        "README.md",
        "manifest.json",
        "checksums.sha256",
        "command_log.txt",
    ):
        assert (output_dir / name).is_file(), name
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "phase10_sparse_wrapper_8x8_complete"
    assert manifest["seed_provenance"]["seeds"] == {"system_seed": 123}
    assert manifest["changes_estimator_behavior"] is False
