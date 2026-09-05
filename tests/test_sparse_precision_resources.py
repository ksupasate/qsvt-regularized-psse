"""Executed-resource accounting tests: real circuits, explicit limitations, exact math."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("pennylane")
pytest.importorskip("pypower")

import pandas as pd

from robust_qsvt_se.qsvt.gate_level_qsvt import qsvt_sequence_operation_counts
from robust_qsvt_se.qsvt.sparse_error_precision_study import (
    DIRECT_ROTATION_LIMITATION,
    FULL_PHASE_KEY,
    FUNCTIONAL_IDS,
    SHOT_BUDGETS,
    build_frozen_design,
    build_point_config,
    make_context,
    stage_resources,
)
from robust_qsvt_se.qsvt.sparse_integrated_chain import (
    build_integrated_sparse_selected_output_circuit,
    compile_for_aer,
)


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    return build_frozen_design(tmp_path_factory.mktemp("design"))


@pytest.fixture(scope="module")
def compiled_baseline(design, tmp_path_factory):
    config = build_point_config(
        design, "6", FULL_PHASE_KEY, tmp_path_factory.mktemp("cfg")
    )
    bundle = build_integrated_sparse_selected_output_circuit(
        config,
        matrix=design.matrices_by_value_key["6"],
        residual=design.residual,
        selected_functional=design.functionals["coordinate_e0"],
        phases=design.phases_by_phase_key[FULL_PHASE_KEY],
    )
    compiled, _simulator = compile_for_aer(bundle.circuit)
    return bundle, compiled


def test_resource_records_come_from_actual_compiled_circuits(compiled_baseline):
    bundle, compiled = compiled_baseline
    ops = {str(key): int(value) for key, value in compiled.count_ops().items()}
    stored = pd.read_csv("outputs/sparse_integrated_chain/resource_ledger.csv")
    row = stored[
        stored["resource_category"] == "executed_small_scale_sparse_integrated"
    ].iloc[0]
    assert sum(ops.values()) == int(row["transpiled_gate_count"])
    assert int(compiled.depth()) == int(row["transpiled_depth"])
    assert ops.get("ccx", 0) == int(row["toffoli_count"])
    counts = bundle.operation_counts
    assert counts["signal_unitary_calls_per_attempt"] == 31
    assert counts["projector_phase_operations_per_attempt"] == 32
    assert counts["value_rotations_per_attempt"] == 744


def test_convention_counts_match_degree_31():
    convention = qsvt_sequence_operation_counts(32)
    assert convention["signal_unitary_calls"] == 31
    assert convention["projector_phase_operations"] == 32


def _write_synthetic_finite_shot_part(checkpoint, *, status: str = "completed") -> None:
    capture = {
        functional_id: {
            "operation_counts": {"cx": 100, "ccx": 50, "u3": 300},
            "transpiled_gate_count": 450,
            "transpiled_depth": 400,
            "toffoli_count": 50,
            "cx_count": 100,
            "total_logical_qubits": 8,
        }
        for functional_id in FUNCTIONAL_IDS
    }
    capture["direct_postselection"] = dict(capture["coordinate_e0"])
    payload = {
        "configuration_id": "synthetic_cfg",
        "label": "baseline",
        "value_bits": "6",
        "phase_bits": "full",
        "status": status,
        "rows": [],
        "resource_capture": capture if status == "completed" else {},
        "postselection_probability_statevector": 0.5,
    }
    if status != "completed":
        payload["failure_reason"] = "numerical_instability: synthetic"
    checkpoint.write_part("finite-shot", "bv6_bpfull", payload)


def test_stage_resources_math_and_limitation_flags(design, tmp_path, monkeypatch):
    import robust_qsvt_se.qsvt.sparse_error_precision_study as study

    context = make_context(output_dir=tmp_path)
    context._design = design
    _write_synthetic_finite_shot_part(context.checkpoint)
    for sampled in study.FINITE_SHOT_CONFIGURATIONS[1:]:
        context.checkpoint.write_part(
            "finite-shot",
            study._finite_shot_config_key(sampled),
            {
                "configuration_id": f"synthetic_{sampled['label']}",
                "label": sampled["label"],
                "value_bits": sampled["value_bits"],
                "phase_bits": sampled["phase_bits"],
                "status": "failed",
                "failure_reason": "finite_shot_runtime_limit: synthetic",
                "rows": [],
                "resource_capture": {},
            },
        )
    monkeypatch.setattr(
        study, "_wrapper_resource_registry", lambda _context: pd.DataFrame([{"ok": 1}])
    )
    monkeypatch.setattr(study, "_residual_preparation_gate_count", lambda _design: 15)
    stage_resources(context)
    frame = pd.read_csv(
        tmp_path / "resource_sweep.csv", dtype={"value_bits": str, "phase_bits": str}
    )
    completed = frame[frame["status"] == "completed"]
    assert len(completed) == len(FUNCTIONAL_IDS) * len(SHOT_BUDGETS)
    row = completed.iloc[0]
    assert row["attempts_per_accepted_direct_sample"] == pytest.approx(1.0 / 0.5)
    assert row["gates_per_accepted_direct_sample"] == pytest.approx(450 / 0.5)
    assert int(row["estimated_total_gate_applications"]) == (
        int(row["shots_attempted"]) * 450
    )
    assert row["value_precision_gate_count_scaling"] == DIRECT_ROTATION_LIMITATION
    assert row["phase_precision_gate_count_scaling"] == DIRECT_ROTATION_LIMITATION
    # failed configurations are retained, and their unknown costs are NaN, never zero
    failed = frame[frame["status"] == "failed"]
    assert len(failed) == len(study.FINITE_SHOT_CONFIGURATIONS) - 1
    assert failed["transpiled_gate_count_per_attempt"].isna().all()


def test_resource_sweep_artifact_contract_if_present():
    try:
        frame = pd.read_csv(
            "outputs/sparse_error_precision_study/resource_sweep.csv",
            dtype={"value_bits": str, "phase_bits": str},
        )
    except FileNotFoundError:
        pytest.skip("study outputs not generated yet")
    completed = frame[frame["status"] == "completed"]
    assert not completed.empty
    assert (
        completed["estimated_total_gate_applications"]
        == completed["shots_attempted"] * completed["transpiled_gate_count_per_attempt"]
    ).all()
    assert np.allclose(
        completed["gates_per_accepted_direct_sample"],
        completed["transpiled_gate_count_per_attempt"]
        / completed["postselection_probability"],
    )
    assert (completed["value_precision_gate_count_scaling"]
            == DIRECT_ROTATION_LIMITATION).all()
    assert (completed["transpiled_gate_count_per_attempt"] > 0).all()
