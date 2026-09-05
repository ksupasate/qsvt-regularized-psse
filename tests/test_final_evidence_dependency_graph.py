import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_contribution_evidence"


def test_dependency_graph_has_no_broken_or_orphan_nodes() -> None:
    graph = json.loads((OUT / "artifact_dependency_graph.json").read_text())
    node_ids = {node["node_id"] for node in graph["nodes"]}
    assert graph["status"] == "pass"
    assert graph["broken_paths"] == []
    assert graph["broken_edges"] == []
    assert graph["orphan_results"] == []
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])
    results = pd.read_csv(OUT / "canonical_result_registry.csv")
    assert all(f"result:{result_id}" in node_ids for result_id in results["result_id"])


def test_qsvt_finite_shot_and_resource_dependencies_are_explicit() -> None:
    graph = json.loads((OUT / "artifact_dependency_graph.json").read_text())
    relationships = {edge["relationship"] for edge in graph["edges"]}
    assert {"uses_polynomial", "uses_phases", "uses_support", "validated_against"}.issubset(
        relationships
    )
    finite = "result:res:integrated:finite_shot_coordinate_1e6"
    state = "result:res:integrated:selected_output:coordinate_e0"
    assert any(edge["source"] == state and edge["target"] == finite for edge in graph["edges"])
    assert any(
        node["type"] == "configuration" and ":resource:" in node["node_id"]
        for node in graph["nodes"]
    )
