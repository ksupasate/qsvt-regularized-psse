"""Focused guards for the sparse-chain integration verdict and manuscript claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    build_default_sparse_integrated_inputs,
    build_integrated_sparse_selected_output_circuit,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "sparse_chain_reconciliation"
CANONICAL = "ieee14_sparse_quantized_8x8_d31_selected_v1"


@pytest.fixture(scope="module")
def reconciled_bundle(tmp_path_factory):
    inputs = build_default_sparse_integrated_inputs(
        tmp_path_factory.mktemp("sparse_chain_reconciliation"),
        shot_counts=(1_000,),
        seeds=(0,),
    )
    bundle = build_integrated_sparse_selected_output_circuit(
        inputs.config,
        matrix=inputs.matrix_quantized,
        residual=inputs.residual,
        selected_functional=inputs.selected_functionals["coordinate_e0"],
        phases=inputs.phases,
    )
    return inputs, bundle


def _inventory() -> list[dict[str, str]]:
    with (AUDIT / "workload_inventory.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_binary_verdict_and_unique_workload_identity():
    verdict = (AUDIT / "integration_verdict.md").read_text(encoding="utf-8")
    assert "## Verdict A: Genuine single integrated sparse finite-shot circuit exists" in verdict
    rows = _inventory()
    ids = [row["workload_id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert ids.count(CANONICAL) == 1


def test_headline_resources_and_shots_share_the_canonical_row():
    row = next(item for item in _inventory() if item["workload_id"] == CANONICAL)
    assert row["configuration_id"] == CANONICAL
    assert row["logical_qubits"] == "8"
    assert row["transpiled_gate_count"] == "186191"
    assert row["transpiled_depth"] == "180380"
    assert row["Toffoli_count"] == "51898"
    assert row["controlled_rotation_count"] == "744"
    assert row["sparse_wrapper_present"] == "true"
    assert row["finite_shots_sampled"] == "true"

    wrapper_only = next(
        item
        for item in _inventory()
        if item["workload_id"] == "ieee14_sparse_wrapper_8x8_d31_statevector_v1"
    )
    assert wrapper_only["logical_qubits"] == "6"
    assert wrapper_only["finite_shots_sampled"] == "false"
    dense_shot = next(
        item for item in _inventory() if item["workload_id"] == "selected_8x8_dense_d31_shot_v1"
    )
    assert dense_shot["logical_qubits"] == "5"
    assert dense_shot["sparse_wrapper_present"] == "false"


def test_final_circuit_contains_sparse_lookup_value_uncomputation_and_readout(
    reconciled_bundle,
):
    inputs, bundle = reconciled_bundle
    assert inputs.config.configuration_id == CANONICAL
    assert bundle.circuit.num_qubits == 8
    labels = [str(item.operation.label or item.operation.name) for item in bundle.circuit.data]
    assert labels.count("c0_residual_prep") == 1
    assert labels.count("c0_sparse_U_A") == 16
    assert labels.count("c0_sparse_U_A_dagger") == 15
    assert labels.count("c0_PCPhase") == 32
    assert labels.count("c1_functional_prep") == 1
    assert labels.count("measure") == 2

    wrapper_ops = list(bundle.wrapper.circuit.data)
    wrapper_names = [item.operation.name for item in wrapper_ops]
    wrapper_labels = [str(item.operation.label or "") for item in wrapper_ops]
    assert sum(name.startswith("c5ry") for name in wrapper_names) == 24
    assert sum(name.startswith("c-unitary") for name in wrapper_names) == 3
    assert wrapper_labels.count("V_slot") == 1
    assert wrapper_labels.count("V_slot_dag") == 1
    assert not any(label in {"c0_U_A", "c0_U_A_dagger"} for label in labels)
    assert bundle.output_state_used_for_preparation is False


def test_rerun_resource_and_shot_files_do_not_cross_workloads():
    run = AUDIT / "end_to_end_run"
    resources = pd.read_csv(run / "resource_ledger.csv")
    sparse = resources.loc[
        resources["resource_category"] == "executed_small_scale_sparse_integrated"
    ].iloc[0]
    assert sparse["configuration_id"] == CANONICAL
    assert int(sparse["total_logical_qubits"]) == 8
    assert int(sparse["transpiled_gate_count"]) == 186_191
    assert int(sparse["toffoli_count"]) == 51_898

    shots = pd.read_csv(run / "finite_shot_results.csv")
    assert set(shots["configuration_id"]) == {CANONICAL}
    assert set(shots["chain_type"]) == {"dense", "sparse"}
    assert set(shots.loc[shots["chain_type"] == "sparse", "backend"]) == {
        "qiskit_aer_statevector_actual_shot_sampling"
    }
    assert not shots["output_state_used_for_preparation"].astype(bool).any()


def test_full_rectangular_evidence_tiers_are_not_conflated():
    rows = {row["workload_id"]: row for row in _inventory()}
    for workload in (
        "ieee30_fullrect_172x59_d255_exact_action_v1",
        "ieee57_fullrect_331x113_d255_exact_action_v1",
    ):
        row = rows[workload]
        assert row["QSVT_executed"] == "false"
        assert row["statevector_executed"] == "false"
        assert row["evidence_status"].startswith("exact dense matrix action")

    nonlinear = rows["nonlinear_ieee14_selected_8x8_d31_statevector_seed101_v1"]
    assert nonlinear["statevector_executed"] == "true"
    assert nonlinear["finite_shots_sampled"] == "false"




def test_trace_json_names_the_same_final_circuit():
    trace = json.loads((AUDIT / "component_trace.json").read_text(encoding="utf-8"))
    assert trace["canonical_workload_id"] == CANONICAL
    final = trace["final_circuit"]
    assert final["name"] == "sparse_integrated_qsvt_readout"
    assert final["qubits"] == 8
    assert final["same_object_compiled_and_sampled"] is True
    assert final["output_state_used_for_preparation"] is False
