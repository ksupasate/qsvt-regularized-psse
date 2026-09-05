"""Artifact dependency graph for canonical contribution evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical_registry import (
    _read_csv,
    atomic_write_json,
    sha256_file,
    stable_json_fingerprint,
)


class GraphBuilder:
    """Small deterministic graph builder with path and edge validation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: set[tuple[str, str, str]] = set()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        path: str | Path,
        configuration_id: str = "",
        fingerprint: str = "",
    ) -> str:
        path_value = Path(path).as_posix()
        full_path = self.root / path_value
        digest = sha256_file(full_path) if full_path.is_file() else ""
        candidate = {
            "node_id": node_id,
            "type": node_type,
            "path": path_value,
            "sha256": digest,
            "configuration_id": configuration_id,
            "fingerprint": fingerprint,
        }
        existing = self.nodes.get(node_id)
        if existing is not None and existing != candidate:
            raise ValueError(f"conflicting graph node: {node_id}")
        self.nodes[node_id] = candidate
        return node_id

    def add_edge(self, source: str, target: str, relationship: str) -> None:
        self.edges.add((source, target, relationship))

    def payload(self) -> dict[str, Any]:
        edges = [
            {"source": source, "target": target, "relationship": relationship}
            for source, target, relationship in sorted(self.edges)
        ]
        nodes = [self.nodes[key] for key in sorted(self.nodes)]
        broken_paths = [
            node["node_id"]
            for node in nodes
            if not node["path"] or not (self.root / node["path"]).is_file()
        ]
        broken_edges = [
            edge
            for edge in edges
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes
        ]
        result_nodes = {node["node_id"] for node in nodes if node["type"] == "canonical_result"}
        sourced_results = {
            edge["target"]
            for edge in edges
            if edge["relationship"] == "summarized_by" and edge["target"] in result_nodes
        }
        claimed_results = {
            edge["source"]
            for edge in edges
            if edge["relationship"] in {"supports_claim", "limited_by"}
            and edge["source"] in result_nodes
        }
        orphan_results = sorted(result_nodes - sourced_results | result_nodes - claimed_results)
        return {
            "schema_version": 1,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "broken_paths": broken_paths,
            "broken_edges": broken_edges,
            "orphan_results": orphan_results,
            "complete_claim_to_artifact_paths": len(result_nodes - set(orphan_results)),
            "required_relationship_types": [
                "derived_from",
                "uses_matrix",
                "uses_support",
                "uses_residual_split",
                "uses_functional",
                "uses_polynomial",
                "uses_phases",
                "validated_against",
                "summarized_by",
                "supports_claim",
                "limited_by",
            ],
            "status": (
                "pass"
                if not broken_paths and not broken_edges and not orphan_results
                else "blocking_failure"
            ),
        }


def _family_dir(root: Path, experiment_family: str) -> Path | None:
    if experiment_family in {
        "output_aware_sparse_selection",
        "output_aware_generalization",
        "output_aware_structural_generalization",
    }:
        return root / "outputs" / experiment_family
    return None


def _matrix_path_for_result(root: Path, family: Path | None, row: dict[str, Any]) -> Path:
    if family is None:
        if str(row["experiment_family"]) == "integrated_sparse_chain":
            return root / "outputs/sparse_integrated_chain/matrix_quantized.npy"
        if str(row["experiment_family"]) == "precision_sensitivity":
            registry = _read_csv(
                root / "outputs/sparse_error_precision_study/matrix_precision_registry.csv",
                dtype=str,
            )
            matches = registry[registry["matrix_fingerprint"] == str(row["matrix_fingerprint"])]
            if not matches.empty:
                return root / str(matches.iloc[0]["matrix_file"])
        return root / str(row["source_artifact"])
    instances = family / "instances"
    if instances.exists():
        for path in instances.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("matrix_fingerprint") == row["matrix_fingerprint"]:
                return path
    if family.name == "output_aware_sparse_selection":
        return family / "entry_scores.csv"
    return root / str(row["source_artifact"])


def build_artifact_dependency_graph(root: Path, output_dir: Path) -> dict[str, Any]:
    """Trace every canonical result and explicit QSVT stage dependency."""

    results = _read_csv(output_dir / "canonical_result_registry.csv", dtype=str)
    configurations = _read_csv(output_dir / "canonical_configuration_registry.csv", dtype=str)
    claims = _read_csv(output_dir / "canonical_claim_evidence_registry.csv", dtype=str)
    limitations = _read_csv(output_dir / "canonical_limitation_registry.csv", dtype=str)
    referenced_configurations = set(results["configuration_id"].astype(str))
    resource_configurations = set(
        configurations.loc[
            configurations["experiment_family"] == "resource_accounting",
            "configuration_id",
        ].astype(str)
    )
    configurations = configurations[
        configurations["configuration_id"].isin(referenced_configurations | resource_configurations)
    ]
    configuration_sources = {
        str(item["configuration_id"]): root / str(item["source_configuration_file"])
        for _, item in configurations.iterrows()
    }
    support_paths: dict[str, dict[str, Path]] = {}
    matrix_paths: dict[str, Path] = {}
    integrated_config = json.loads(
        (root / "outputs/sparse_integrated_chain/configuration.json").read_text(encoding="utf-8")
    )
    matrix_paths[str(integrated_config["matrix_fingerprint"])] = (
        root / "outputs/sparse_integrated_chain/matrix_quantized.npy"
    )
    precision_registry = _read_csv(
        root / "outputs/sparse_error_precision_study/matrix_precision_registry.csv",
        dtype=str,
    )
    for _, item in precision_registry.iterrows():
        value_bits = str(item["value_bits"])
        if str(item["matrix_file"]) == "nan":
            filename = (
                "matrix_original.npy" if value_bits == "original" else "matrix_sparse_exact.npy"
            )
            path = root / "outputs/sparse_error_precision_study" / filename
        else:
            path = root / str(item["matrix_file"])
        matrix_paths[str(item["matrix_fingerprint"])] = path
    for _, item in configurations.iterrows():
        fingerprint = str(item["matrix_fingerprint"])
        if str(item["configuration_id"]).endswith(":study") and fingerprint not in {"", "nan"}:
            matrix_paths.setdefault(fingerprint, root / str(item["source_configuration_file"]))
    for family_name in (
        "output_aware_sparse_selection",
        "output_aware_generalization",
        "output_aware_structural_generalization",
    ):
        family_path = root / "outputs" / family_name
        registry = _read_csv(family_path / "support_registry.csv", dtype=str)
        mapping: dict[str, Path] = {}
        for _, item in registry.drop_duplicates("support_fingerprint").iterrows():
            value = Path(str(item["support_file"]))
            mapping[str(item["support_fingerprint"])] = (
                value if value.is_absolute() else family_path / value
            )
        support_paths[family_name] = mapping
        instances = family_path / "instances"
        if instances.exists():
            for path in instances.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                matrix_paths[str(payload["matrix_fingerprint"])] = path
    builder = GraphBuilder(root)

    claim_path = (output_dir / "canonical_claim_evidence_registry.csv").relative_to(root)
    limitation_path = (output_dir / "canonical_limitation_registry.csv").relative_to(root)
    for _, claim in claims.iterrows():
        builder.add_node(
            f"claim:{claim['claim_family']}",
            "claim_family",
            claim_path,
            fingerprint=stable_json_fingerprint(claim.to_dict()),
        )
    for _, limitation in limitations.iterrows():
        builder.add_node(
            f"limitation:{limitation['limitation_id']}",
            "limitation",
            limitation_path,
            fingerprint=stable_json_fingerprint(limitation.to_dict()),
        )

    for _, config in configurations.iterrows():
        config_id = str(config["configuration_id"])
        path = str(config["source_configuration_file"])
        builder.add_node(
            f"configuration:{config_id}",
            "configuration",
            path,
            config_id,
            fingerprint=config_id,
        )
        support_fp = str(config["support_fingerprint"])
        if support_fp not in {"", "nan"}:
            config_family = next(
                (
                    name
                    for name in (
                        "output_aware_sparse_selection",
                        "output_aware_generalization",
                        "output_aware_structural_generalization",
                    )
                    if f"cfg:{name}:" in config_id
                ),
                str(config["experiment_family"]),
            )
            family = root / "outputs" / config_family
            support_path = support_paths.get(config_family, {}).get(support_fp)
            if support_path is not None:
                support_node = builder.add_node(
                    f"support:{support_fp}:{config_id}",
                    "support_selection",
                    support_path.relative_to(root),
                    config_id,
                    support_fp,
                )
                builder.add_edge(support_node, f"configuration:{config_id}", "uses_support")

    for _, series in results.iterrows():
        row = series.to_dict()
        result_id = str(row["result_id"])
        result_node = f"result:{result_id}"
        source_path = Path(str(row["source_artifact"]))
        source_node = builder.add_node(
            f"artifact:{sha256_file(root / source_path)[:20]}:{source_path.name}",
            "source_artifact",
            source_path,
            "",
            fingerprint=sha256_file(root / source_path),
        )
        builder.add_node(
            result_node,
            "canonical_result",
            source_path,
            str(row["configuration_id"]),
            fingerprint=result_id,
        )
        builder.add_edge(source_node, result_node, "summarized_by")
        config_node = f"configuration:{row['configuration_id']}"
        if config_node in builder.nodes:
            builder.add_edge(config_node, result_node, "derived_from")
        family = _family_dir(root, str(row["experiment_family"]))
        matrix_fp = str(row["matrix_fingerprint"])
        if matrix_fp not in {"", "nan"}:
            if matrix_fp in matrix_paths:
                matrix_path = matrix_paths[matrix_fp]
            elif str(row["configuration_id"]).endswith(":study"):
                matrix_path = configuration_sources[str(row["configuration_id"])]
            else:
                matrix_path = _matrix_path_for_result(root, family, row)
            matrix_node = builder.add_node(
                f"matrix:{matrix_fp}:{row['configuration_id']}",
                "source_matrix",
                matrix_path.relative_to(root),
                str(row["configuration_id"]),
                matrix_fp,
            )
            builder.add_edge(matrix_node, result_node, "uses_matrix")
        else:
            matrix_node = source_node
        support_fp = str(row["support_fingerprint"])
        support_node: str | None = None
        if support_fp not in {"", "nan"}:
            if family is None and str(row["experiment_family"]) == "integrated_sparse_chain":
                support_path = root / "outputs/sparse_integrated_chain/matrix_metadata.json"
            elif family is None and str(row["experiment_family"]) == "precision_sensitivity":
                support_path = root / "outputs/sparse_error_precision_study/sparse_support.json"
            else:
                support_path = (
                    support_paths.get(family.name, {}).get(support_fp)
                    if family is not None
                    else None
                )
            if support_path is not None:
                support_node = builder.add_node(
                    f"support:{support_fp}:{row['configuration_id']}",
                    "support_selection",
                    support_path.relative_to(root),
                    str(row["configuration_id"]),
                    support_fp,
                )
                builder.add_edge(matrix_node, support_node, "derived_from")
                builder.add_edge(support_node, result_node, "uses_support")
        residual_fp = str(row["residual_split_fingerprint"])
        if residual_fp not in {"", "nan", "multiple"}:
            configuration_id = str(row["configuration_id"])
            if configuration_id.startswith("cfg:integrated:"):
                residual_path = root / "outputs/sparse_integrated_chain/residual.npy"
            elif configuration_id.startswith("cfg:precision:"):
                residual_path = root / "outputs/sparse_error_precision_study/matrices/residual.npy"
            elif family is not None and (family / "residual_split.json").exists():
                residual_path = family / "residual_split.json"
            elif family is not None and (family / "residual_splits").exists():
                candidates = list((family / "residual_splits").glob("*.json"))
                residual_path = next(
                    (path for path in candidates if sha256_file(path) == residual_fp),
                    candidates[0] if candidates else source_path,
                )
            else:
                residual_path = source_path
            residual_node = builder.add_node(
                f"residual_split:{residual_fp}:{row['configuration_id']}",
                "residual_split",
                residual_path.relative_to(root) if residual_path.is_absolute() else residual_path,
                str(row["configuration_id"]),
                residual_fp,
            )
            builder.add_edge(residual_node, result_node, "uses_residual_split")
        functional_id = str(row["functional_id"])
        if functional_id not in {"", "nan"}:
            configuration_id = str(row["configuration_id"])
            if configuration_id.startswith(("cfg:integrated:", "cfg:precision:")):
                functional_path = Path("outputs/sparse_integrated_chain/selected_functionals.json")
            elif configuration_id.startswith("cfg:output_aware_sparse_selection:"):
                functional_path = Path(
                    "outputs/output_aware_sparse_selection/study_configuration.json"
                )
            elif family is not None and (family / "functional_registry.csv").exists():
                functional_path = (family / "functional_registry.csv").relative_to(root)
            else:
                functional_path = source_path
            functional_node = builder.add_node(
                f"functional:{functional_id}:{row['configuration_id']}",
                "selected_functional",
                functional_path,
                str(row["configuration_id"]),
                functional_id,
            )
            builder.add_edge(functional_node, result_node, "uses_functional")
        else:
            functional_node = source_node

        phase_fp = str(row["phase_fingerprint"])
        polynomial_fp = str(row["polynomial_fingerprint"])
        if phase_fp not in {"", "nan"} or polynomial_fp not in {"", "nan"}:
            if str(row["experiment_family"]) == "integrated_sparse_chain":
                phase_path = Path("outputs/sparse_integrated_chain/phases.npy")
                polynomial_path = Path(
                    "outputs/sparse_integrated_chain/polynomial_coefficients.npy"
                )
            elif family is not None and (family / "qsvt_instance_designs.json").exists():
                phase_path = (family / "qsvt_instance_designs.json").relative_to(root)
                polynomial_path = phase_path
            elif family is not None and (family / "common_qsvt_design.json").exists():
                phase_path = (family / "common_qsvt_design.json").relative_to(root)
                polynomial_path = phase_path
            else:
                phase_path = source_path
                polynomial_path = source_path
            polynomial_node = builder.add_node(
                f"polynomial:{polynomial_fp}:{row['configuration_id']}",
                "polynomial_design",
                polynomial_path,
                str(row["configuration_id"]),
                polynomial_fp,
            )
            phase_node = builder.add_node(
                f"phases:{phase_fp}:{row['configuration_id']}",
                "phase_sequence",
                phase_path,
                str(row["configuration_id"]),
                phase_fp,
            )
            qsvt_node = builder.add_node(
                f"qsvt:{result_id}",
                "qsvt_result",
                source_path,
                str(row["configuration_id"]),
                result_id,
            )
            preceding = support_node or matrix_node
            sparse_node = builder.add_node(
                f"sparse_matrix:{matrix_fp}:{support_fp}:{row['configuration_id']}",
                "sparse_matrix",
                source_path if support_node is None else builder.nodes[support_node]["path"],
                str(row["configuration_id"]),
                support_fp,
            )
            ridge_node = builder.add_node(
                f"ridge_reference:{result_id}",
                "ridge_reference",
                source_path,
                str(row["configuration_id"]),
                result_id,
            )
            builder.add_edge(preceding, sparse_node, "uses_support")
            builder.add_edge(sparse_node, ridge_node, "validated_against")
            builder.add_edge(ridge_node, polynomial_node, "uses_polynomial")
            builder.add_edge(polynomial_node, phase_node, "uses_polynomial")
            builder.add_edge(phase_node, qsvt_node, "uses_phases")
            builder.add_edge(functional_node, qsvt_node, "uses_functional")
            builder.add_edge(qsvt_node, result_node, "validated_against")
            if str(row["claim_family"]) in {
                "finite_shot_selected_output",
                "resource_accounting",
            }:
                post_node = builder.add_node(
                    f"postselection:{result_id}",
                    "postselection_readout",
                    source_path,
                    str(row["configuration_id"]),
                    result_id,
                )
                builder.add_edge(qsvt_node, post_node, "derived_from")
                builder.add_edge(post_node, result_node, "summarized_by")

        claim_node = f"claim:{row['claim_family']}"
        if claim_node in builder.nodes:
            relationship = (
                "supports_claim"
                if str(row["manuscript_eligible"]).lower() == "true"
                else "limited_by"
            )
            builder.add_edge(result_node, claim_node, relationship)
        for limitation_id in str(row["limitation_code"]).split(";"):
            limitation_node = f"limitation:{limitation_id}"
            if limitation_node in builder.nodes:
                builder.add_edge(result_node, limitation_node, "limited_by")

    # Explicit finite-shot ancestry: sampled output is derived from the matching
    # statevector reference, not mislabeled as statevector evidence itself.
    finite_node = "result:res:integrated:finite_shot_coordinate_1e6"
    state_node = "result:res:integrated:selected_output:coordinate_e0"
    if finite_node in builder.nodes and state_node in builder.nodes:
        builder.add_edge(state_node, finite_node, "validated_against")

    payload = builder.payload()
    atomic_write_json(output_dir / "artifact_dependency_graph.json", payload)
    if payload["status"] != "pass":
        raise ValueError(
            "artifact graph validation failed: "
            f"broken_paths={len(payload['broken_paths'])}, "
            f"broken_edges={len(payload['broken_edges'])}, "
            f"orphans={len(payload['orphan_results'])}"
        )
    return payload
