from __future__ import annotations

import json

from robust_qsvt_se.paper.resource_boundary_stack import (
    STACK_COLUMNS,
    build_resource_stack,
    run_resource_boundary_stack,
)
from robust_qsvt_se.paper.selected_observable_qsvt_common import forbidden_in

EXPECTED_STATUS = {
    "dense block encoding": "implemented",
    "QSVT phase synthesis": "implemented",
    "polynomial degree": "implemented",
    "residual state preparation": "proxy",
    "postselection": "implemented",
    "signed readout": "implemented",
    "sparse index/value access": "implemented",
    "amplitude amplification": "modeled",
    "full-vector recovery": "excluded",
}


def _write_demo_artifacts(demo_dir):
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "circuit_stats.json").write_text(
        json.dumps(
            {
                "num_qubits": 3,
                "raw_gate_count": 63,
                "raw_circuit_depth": 63,
                "transpiled_depth": 120,
                "phase_count": 32,
                "degree": 31,
            }
        )
    )
    (demo_dir / "normalization_record.json").write_text(
        json.dumps({"postselection_probability": 0.99})
    )
    (demo_dir / "qsvt_pipeline_metadata.json").write_text(
        json.dumps({"block_shape": "4x4", "degree": 31, "phase_count": 32})
    )
    (demo_dir / "readout_metadata.json").write_text(json.dumps({"status": "ok"}))
    (demo_dir / "readout_diagnostics.csv").write_text(
        "observable_id,shots,backend\nstate_correction_0,100000,aer:AerSimulator\n"
    )


def _write_oracle_artifacts(oracle_dir):
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (oracle_dir / "oracle_metadata.json").write_text(json.dumps({"status_label": "synthesized"}))
    (oracle_dir / "oracle_resource_summary.csv").write_text(
        "status,row_index_qubits,slot_index_qubits,column_register_qubits,"
        "col_oracle_synth_gate_count,col_oracle_synth_depth,total_t_count_qrom\n"
        "synthesized_small_scale,2,2,2,56,35,224\n"
        "modeled,11,5,10,0,0,345674\n"
    )


def test_resource_stack_labels_status_correctly(tmp_path):
    demo_dir = tmp_path / "demo"
    oracle_dir = tmp_path / "oracle"
    _write_demo_artifacts(demo_dir)
    _write_oracle_artifacts(oracle_dir)
    rows, provenance = build_resource_stack(demo_dir=demo_dir, oracle_dir=oracle_dir)
    by_component = {row["component"]: row for row in rows}
    for component, status in EXPECTED_STATUS.items():
        assert by_component[component]["implemented_status"] == status
    # Taxonomy coverage: implemented, proxy, modeled, excluded all present.
    statuses = {row["implemented_status"] for row in rows}
    assert {"implemented", "proxy", "modeled", "excluded"} <= statuses
    assert provenance["status_counts"]["excluded"] == 1
    assert provenance["status_counts"]["modeled"] == 1


def test_resource_stack_pulls_numbers_from_artifacts(tmp_path):
    demo_dir = tmp_path / "demo"
    oracle_dir = tmp_path / "oracle"
    _write_demo_artifacts(demo_dir)
    _write_oracle_artifacts(oracle_dir)
    rows, _ = build_resource_stack(demo_dir=demo_dir, oracle_dir=oracle_dir)
    by_component = {row["component"]: row for row in rows}
    assert by_component["dense block encoding"]["gates"] == 63
    assert by_component["signed readout"]["shots"] == 100000
    assert by_component["postselection"]["postselection_factor"] == 0.99
    # Sparse oracle qubits = row+slot+col+invalid = 2+2+2+1 = 7.
    assert by_component["sparse index/value access"]["qubits"] == 7


def test_full_stack_outputs_have_no_forbidden_wording(tmp_path):
    demo_dir = tmp_path / "demo"
    oracle_dir = tmp_path / "oracle"
    _write_demo_artifacts(demo_dir)
    _write_oracle_artifacts(oracle_dir)
    out_dir = tmp_path / "stack"
    run = run_resource_boundary_stack(
        {"output_dir": str(out_dir), "demo_dir": str(demo_dir), "oracle_dir": str(oracle_dir)}
    )
    assert list(run["stack"].columns) == STACK_COLUMNS
    for name in (
        "resource_boundary_stack.md",
        "resource_boundary_stack.tex",
        "resource_boundary_stack.csv",
    ):
        text = (out_dir / name).read_text(encoding="utf-8")
        assert forbidden_in(text) == []
