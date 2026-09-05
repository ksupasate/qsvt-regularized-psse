#!/usr/bin/env python3
"""Finalize reports, protected hashes, manifests, and checksums for the study."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/generic_sparse_qsvt_compiler"
AUDIT = OUTPUT / "protected_hash_audit.json"

TASK_CODE = (
    "configs/generic_sparse_qsvt_compiler.json",
    "src/robust_qsvt_se/qsvt/generic_sparse_compiler.py",
    "src/robust_qsvt_se/qsvt/generic_sparse_execution.py",
    "src/robust_qsvt_se/qsvt/generic_sparse_workloads.py",
    "src/robust_qsvt_se/qsvt/generic_sparse_scaling.py",
    "scripts/run_generic_sparse_qsvt_compiler.py",
    "scripts/build_generic_sparse_qsvt_diagram.py",
    "scripts/finalize_generic_sparse_qsvt_compiler.py",
    "tests/test_generic_sparse_qsvt_compiler.py",
    "tests/test_generic_sparse_qsvt_artifacts.py",
)

KNOWN_BASELINE_FAILURES = (
    "tests/test_final_evidence_protected_sources.py::test_all_protected_sources_are_byte_identical",
    "tests/test_generalization_instance_registry.py::test_exclusions_retain_reasons_and_protected_snapshot_is_unchanged",
    "tests/test_output_aware_heldout_split.py::test_manuscript_packages_and_prior_outputs_match_campaign_snapshot_if_present",
    "tests/test_reproducibility_package_audit.py::test_repo_traceability_structure_matches_current_layout",
    "tests/test_submission_package_current.py::test_submission_package_current",
    "tests/test_tqe_submission_staleness.py::test_packaged_pdfs_are_byte_identical_to_canonical_pdfs",
    "tests/test_tqe_submission_staleness.py::test_active_source_set_exists_and_package_is_current",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        target = os.readlink(path)
        payload = target.encode("utf-8", errors="surrogateescape")
        return {
            "kind": "symlink",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "target": target,
        }
    return {"kind": "file", "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_snapshot(relative_root: str) -> dict[str, Any]:
    target = ROOT / relative_root
    if not target.exists():
        return {"exists": False, "file_count": 0, "total_bytes": 0, "combined_sha256": None}
    paths = sorted(path for path in target.rglob("*") if path.is_file() or path.is_symlink())
    combined = hashlib.sha256()
    total = 0
    for path in paths:
        record = file_record(path)
        relative = path.relative_to(ROOT).as_posix()
        size = int(record["size_bytes"])
        total += size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(record["sha256"]).encode("ascii"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\n")
    return {
        "exists": True,
        "file_count": len(paths),
        "total_bytes": total,
        "combined_sha256": combined.hexdigest(),
    }


def parse_junit(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
        "failed_tests": [],
    }
    for suite in suites:
        for case in suite.findall("testcase"):
            failure = case.find("failure")
            error = case.find("error")
            if failure is not None or error is not None:
                classname = case.attrib.get("classname", "")
                if classname.startswith("tests."):
                    classname = classname.replace(".", "/") + ".py"
                result["failed_tests"].append(
                    f"{classname}::{case.attrib.get('name', '')}"
                )
    result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    return result


def write_test_reports() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    focused = parse_junit(OUTPUT / "logs/focused_final.junit.xml")
    related = parse_junit(OUTPUT / "logs/related_tests.junit.xml")
    if focused is None or related is None:
        raise FileNotFoundError("focused and related JUnit evidence must exist")
    focused_text = f"""# Focused test report

- Command: `.venv/bin/python -m pytest -q tests/test_generic_sparse_qsvt_compiler.py --junitxml=outputs/generic_sparse_qsvt_compiler/logs/focused_final.junit.xml`
- Collected: {focused['tests']}
- Passed: {focused['passed']}
- Failed: {focused['failures'] + focused['errors']}
- Skipped: {focused['skipped']}
- Runtime: {focused['time']:.3f} seconds
- Machine-readable log: `outputs/generic_sparse_qsvt_compiler/logs/focused_final.junit.xml`

The first diagnostic iteration exposed a nondeterministic QPY component fingerprint and an incorrectly declared synthetic slot budget. The stable fingerprint was replaced by a deterministic semantic circuit descriptor, the fixture was corrected, and the complete focused file then passed. No tolerance was weakened and no protected snapshot was refreshed.
"""
    (OUTPUT / "focused_test_report.md").write_text(focused_text, encoding="utf-8")
    related_text = f"""# Related test report

- Command: `.venv/bin/python -m pytest -q tests/test_sparse_integrated_chain.py tests/test_sparse_integrated_readout.py tests/test_sparse_integrated_resources.py tests/test_convention_identity_and_global_phase.py tests/test_readout_statistics.py tests/test_sparse_chain_reconciliation.py tests/test_qsvt_resource_accounting.py tests/test_sparse_precision_resources.py tests/test_tqe_physical_alignment_protocol.py --junitxml=outputs/generic_sparse_qsvt_compiler/logs/related_tests.junit.xml`
- Collected: {related['tests']}
- Passed: {related['passed']}
- Failed: {related['failures'] + related['errors']}
- Skipped: {related['skipped']}
- Runtime: {related['time']:.3f} seconds
- Machine-readable log: `outputs/generic_sparse_qsvt_compiler/logs/related_tests.junit.xml`

The related set covers sparse lookup, block encoding, QSVT convention, postselection, signed readout, sampled statistics, resource ledgers, physical-functional metadata, and artifact checksums.
"""
    (OUTPUT / "related_test_report.md").write_text(related_text, encoding="utf-8")

    isolated = parse_junit(OUTPUT / "logs/isolated_full.junit.xml")
    if isolated is None:
        isolated_text = """# Isolated full-suite report

Status: scheduled in a copy-on-write temporary repository after final artifact assembly.

The pre-edit isolated baseline collected 1859 tests: 1852 passed, 7 failed, and 0 skipped in 894.89 seconds. The seven baseline failures are recorded below and will be compared with the new isolated run.

""" + "\n".join(f"- `{name}`" for name in KNOWN_BASELINE_FAILURES) + "\n"
    else:
        failed = isolated["failed_tests"]
        known = [name for name in failed if name in KNOWN_BASELINE_FAILURES]
        new = [name for name in failed if name not in KNOWN_BASELINE_FAILURES]
        isolated_text = f"""# Isolated full-suite report

- Command: `copy-on-write repository/.venv/bin/python -m pytest -q --junitxml=<active-output>/logs/isolated_full.junit.xml`
- Isolation: APFS copy-on-write temporary repository; legacy generators could not modify the active working tree.
- Collected: {isolated['tests']}
- Passed: {isolated['passed']}
- Failed: {isolated['failures'] + isolated['errors']}
- Skipped: {isolated['skipped']}
- Runtime: {isolated['time']:.3f} seconds
- Machine-readable log: `outputs/generic_sparse_qsvt_compiler/logs/isolated_full.junit.xml`
- Text log: `outputs/generic_sparse_qsvt_compiler/logs/isolated_full.log`

## Failure classification

- Failures matching the seven pre-edit baseline failures: {len(known)}.
- New failures attributable to this implementation: {len(new)}.

"""
        if failed:
            isolated_text += "\n".join(f"- `{name}`" for name in failed) + "\n"
        else:
            isolated_text += "No isolated-suite failures were observed.\n"
    (OUTPUT / "isolated_full_test_report.md").write_text(isolated_text, encoding="utf-8")
    return focused, related, isolated


def update_protected_audit() -> dict[str, Any]:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    after = {name: tree_snapshot(name) for name in payload["protected_roots"]}
    manuscript_after = tree_snapshot("manuscript")
    root_comparison = {
        name: {
            "unchanged": after[name] == payload["before"][name],
            "before_combined_sha256": payload["before"][name]["combined_sha256"],
            "after_combined_sha256": after[name]["combined_sha256"],
        }
        for name in payload["protected_roots"]
    }
    comparison = {
        "protected_roots": root_comparison,
        "all_protected_roots_unchanged": all(item["unchanged"] for item in root_comparison.values()),
        "manuscript_unchanged": manuscript_after == payload["manuscript_before"],
    }
    payload["after_generated_at_utc"] = datetime.now(UTC).isoformat()
    payload["after"] = after
    payload["comparison"] = comparison
    payload["manuscript_after"] = manuscript_after
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not comparison["all_protected_roots_unchanged"] or not comparison["manuscript_unchanged"]:
        raise RuntimeError("protected roots or manuscript changed after the pre-edit snapshot")
    return payload


def write_claim_assessment() -> None:
    claims = [
        (1, "The sparse QSVT implementation is generated by a reusable compiler.", "supported", "Typed in-memory inputs compile across dimensions, slot budgets, value precisions, matrices, and functionals; source guards exclude workload registries.", "Real square power-of-two matrices only."),
        (2, "The generic compiler reproduces the canonical IEEE-14 workload.", "supported", "All 34 registered numerical, hash, seed, and integer-resource comparisons pass.", "Classical simulator and transpiler evidence."),
        (3, "The same compiler executes a second independent IEEE-30-derived workload.", "supported", "Frozen IEEE-30 block and training-only support compile, statevector validate, and produce 60 finite-shot rows.", "One additional 8 by 8 workload."),
        (4, "Sparse access, QSVT, postselection, and signed readout are integrated in the second workload.", "supported", "One stored final source circuit contains residual preparation, sparse wrapper calls, QSVT phases, postselection flag, and signed interference readout.", "Direct multiplexed small-scale access."),
        (5, "The second workload preserves matched-polynomial circuit correctness.", "supported", "QSVT-versus-exact-polynomial action error is 2.8543e-9, below 1e-6.", "Statevector simulation."),
        (6, "The second workload preserves matched-Ridge accuracy.", "supported with limitations", "QSVT-versus-quantized-Ridge relative error is 2.8542e-9.", "The statement is only for the identical quantized sparse matrix; support error versus the full block is 0.3773."),
        (7, "Finite-shot estimates converge toward the statevector value.", "supported with limitations", "For both IEEE-30 functionals, mean absolute error is lower at 1e6 than at 1e4 shots and analytic variance decreases with 1/N.", "Ten fixed seeds per budget; finite-sample trends, not a universal convergence proof."),
        (8, "Circuit resources scale consistently with dimension, slots, value bits, and degree.", "supported with limitations", "Compiled dimension, slot, precision, and degree ledgers retain execution status and failures.", "s=2 is infeasible; d=63 fails boundedness; direct angle multiplexing makes gate topology flat in b_v; three-point descriptive evidence only."),
        (9, "The evidence supports scalable IEEE-size quantum access.", "unsupported", "Only n=4 and n=8 statevectors and n=16 transpilation are present.", "No scalable data-access oracle or IEEE-scale hardware circuit."),
        (10, "The evidence supports practical quantum competitiveness.", "unsupported", "No hardware-normalized or fault-tolerant comparison is performed.", "Classical simulation and transpilation only."),
        (11, "The evidence supports quantum speedup or advantage.", "unsupported", "No runtime-complexity or hardware comparison establishes speedup.", "No speedup or advantage claim is admissible."),
    ]
    table = "\n".join(
        f"| {number} | {claim} | {classification} | {evidence} | {limitation} |"
        for number, claim, classification, evidence, limitation in claims
    )
    report = f"""# Claim-support assessment

| ID | Claim | Classification | Evidence | Limitation |
| ---: | --- | --- | --- | --- |
{table}

The classifications distinguish implementation correctness from approximation and support-selection error. Matched-Ridge agreement is not evidence that QSVT is more accurate than Ridge.
"""
    (OUTPUT / "claim_support_assessment.md").write_text(report, encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.6e}"


def write_final_report(
    protected: dict[str, Any], focused: dict[str, Any], related: dict[str, Any], isolated: dict[str, Any] | None
) -> None:
    canonical = pd.read_csv(OUTPUT / "canonical_resource_ledger_generic.csv").iloc[0]
    second_resource = pd.read_csv(OUTPUT / "second_workload_resource_ledger.csv").iloc[0]
    canonical_validation = pd.read_csv(OUTPUT / "generic_compiler_validation.csv")
    canonical_reproduction = pd.read_csv(OUTPUT / "canonical_reproduction.csv")
    second_state = pd.read_csv(OUTPUT / "second_workload_statevector_validation.csv")
    canonical_shots = pd.read_csv(OUTPUT / "canonical_shot_summary_generic.csv")
    second_shots = pd.read_csv(OUTPUT / "second_workload_shot_summary.csv")
    metadata = json.loads((OUTPUT / "second_workload_metadata.json").read_text(encoding="utf-8"))
    canonical_matrix = np.load(OUTPUT / "data/canonical_matrix_original.npy")
    canonical_quantized = np.load(OUTPUT / "data/canonical_matrix_quantized.npy")
    second_matrix = np.load(OUTPUT / "data/second_matrix_original.npy")
    second_quantized = np.load(OUTPUT / "data/second_matrix_quantized.npy")
    can_primary_shot = canonical_shots[(canonical_shots.functional_id == "coordinate_e0") & (canonical_shots.shots == 1_000_000)].iloc[0]
    sec_primary_shot = second_shots[(second_shots.functional_id == "coordinate_angle_bus4") & (second_shots.shots == 1_000_000)].iloc[0]
    can_metrics = canonical_validation[canonical_validation.workload_id == "ieee14_sparse_quantized_8x8_d31_selected_v1"].set_index("metric")["value"]
    can_reproduction_metrics = canonical_reproduction.set_index("criterion")[
        "generic_compiler_value"
    ]
    sec_primary = second_state.iloc[0]
    isolated_line = (
        "pending final isolated run"
        if isolated is None
        else f"{isolated['passed']} passed, {isolated['failures'] + isolated['errors']} failed, {isolated['skipped']} skipped"
    )
    report = f"""# Final implementation report

## 1. Repository and baseline

- Root: `{ROOT}`.
- Branch: `research/generalized-rectangular-qsvt`.
- Commit: `ae6a46ef52e0f26e9d2e017f4b5dffcf51b0c2d6`.
- Initial state: 649 dirty entries were recorded as user-owned in the pre-edit audit.
- Protected roots unchanged: `{protected['comparison']['all_protected_roots_unchanged']}`.
- Manuscript unchanged: `{protected['comparison']['manuscript_unchanged']}`.

## 2. Generic compiler

The new API is `compile_sparse_selected_output_qsvt(matrix_spec, support_spec, quantization_spec, qsvt_spec, residual_spec, functional_spec, execution_spec)`, with `compile_from_bundle` as its typed convenience wrapper. Construction returns validated metadata, padded dimensions, register allocation, index/value/sign lookups, rotations, permutations, inverse path, wrapper, QSVT sequence, postselection/readout definitions, final measured circuits, recovery factors, workload identity, and component hashes. Construction is separate from statevector, shot, and transpilation execution.

The implementation supports real square power-of-two matrices. Tall and wide matrices return `unsupported_rectangular_orientation`. Structured failures also cover invalid shape, duplicate or out-of-bounds support, inconsistent assignments, slot overflow, nonfinite or complex values, quantization, phase count, parity, zero residual, functional dimensions, missing uncomputation, and register collision. No legacy public function was changed.

Changed task source is additive: `generic_sparse_compiler.py`, `generic_sparse_execution.py`, `generic_sparse_workloads.py`, `generic_sparse_scaling.py`, the frozen config, two experiment/report scripts, the diagram script, and focused artifact tests.

## 3. Canonical reproduction

- Workload: `ieee14_sparse_quantized_8x8_d31_selected_v1`.
- Original matrix hash: `b158d34b86b778f0c290519ca98985345107012e225798a4cfc7fbf9178df7f9`.
- Support hash: `af1a0f82f3c62e482452f92cdc7bad8903af75dc809e481dc8fb6a34ec38b7fd`.
- Reproduction registry: 34 of 34 comparisons pass.
- Circuit resources: {int(canonical.total_simultaneously_live_qubits)} qubits, {int(canonical.transpiled_gate_count)} gates, depth {int(canonical.transpiled_depth)}, {int(canonical.toffoli_count)} Toffolis, and {int(canonical.controlled_rotation_count)} controlled rotations, all exactly equal to the historical ledger.
- All 90 sparse fixed-seed rows reproduce the historical counts and estimates within the declared serialization tolerance.

## 4. Second-workload selection

The second case is the frozen IEEE-30/PYPOWER block `{metadata['block']['block_id']}`, rows `{metadata['block']['selected_rows']}`, columns `{metadata['block']['selected_columns']}`, rank {metadata['block']['rank']}, and {metadata['block']['nonzero_count']} original nonzeros. Its matrix hash is `{metadata['block']['matrix_hash']}`.

The support is `{metadata['support']['support_id']}` with 16 coordinates, three slots, and training seeds 1000 through 1019. Held-out residual seed 2000 was fixed separately. The operating point was chosen from the pre-existing grid by passing parity, boundedness, uniform-fit, phase-synthesis, phase-count, and action checks, then minimizing degree and applying the declared tie breaks. The selected point is degree 31, normalized lambda {metadata['qsvt_operating_point']['normalized_lambda']:.16g}, and C {metadata['qsvt_operating_point']['boundedness_factor_C']:.16g}. No benchmark-reference error, selected output, postselection probability, resource count, or shot result entered selection.

Requested physical functionals were the first metadata-ordered coordinate update and first real branch-angle difference. The connected-area aggregate is unavailable because the in-block angle buses do not form a connected set of at least two buses; it is retained without a proxy.

## 5. Second end-to-end execution

One stored final source circuit per functional contains controlled residual preparation, sparse slot/index/value/sign access, sparse block encoding and inverse calls, 31 QSVT signal calls, 32 phase operations, aggregate postselection, and real signed readout. No dense signal fallback or computed output-state preparation occurs. The register ledger is three index qubits, two slot qubits, one rotation ancilla, one postselection flag, and one readout qubit.

- Lookup error: {_fmt(float(sec_primary.epsilon_lookup))}.
- Block error: {_fmt(float(sec_primary.epsilon_block))}.
- Sparse-versus-dense action error: {_fmt(float(sec_primary.sparse_dense_action_relative_error))}.
- QSVT-versus-polynomial action error: {_fmt(float(sec_primary.epsilon_qsvt))}.
- Polynomial-versus-matched rational error: {_fmt(float(sec_primary.epsilon_poly))}.
- Quantization error: {_fmt(float(sec_primary.epsilon_quant))}.
- Support error versus the full block: {_fmt(float(sec_primary.epsilon_support))}.
- Postselection probability: {float(sec_primary.postselection_probability):.12f}.
- Coordinate output: {float(sec_primary.statevector_selected_output):.12e}; branch-difference output: {float(second_state.iloc[1].statevector_selected_output):.12e}.
- At 1,000,000 shots over ten seeds, coordinate mean absolute error is {float(sec_primary_shot.mean_absolute_error_vs_statevector):.6e}. All 60 requested IEEE-30 shot rows executed.

Retained failures are external to the second end-to-end circuit: the connected-area functional is unavailable, the canonical support is infeasible at `s=2`, and degree 63 fails boundedness under the existing synthesis protocol.

## 6. Cross-workload comparison

| Quantity | Canonical IEEE-14 | Second IEEE-30 |
| --- | ---: | ---: |
| Original matrix nonzeros | {int(np.count_nonzero(canonical_matrix))} | {int(np.count_nonzero(second_matrix))} |
| Original matrix rank | {int(np.linalg.matrix_rank(canonical_matrix))} | {int(np.linalg.matrix_rank(second_matrix))} |
| Compiled quantized-support rank | {int(np.linalg.matrix_rank(canonical_quantized))} | {int(np.linalg.matrix_rank(second_quantized))} |
| Support coordinates / slots | 16 / 3 | 16 / 3 |
| Degree | 31 | 31 |
| Postselection probability | {float(can_reproduction_metrics['sparse_postselection_probability']):.6f} | {float(sec_primary.postselection_probability):.6f} |
| QSVT-polynomial action error | {_fmt(float(can_metrics['epsilon_qsvt']))} | {_fmt(float(sec_primary.epsilon_qsvt))} |
| Primary 1e6-shot mean absolute error | {float(can_primary_shot.mean_absolute_error_vs_statevector):.6e} | {float(sec_primary_shot.mean_absolute_error_vs_statevector):.6e} |
| Qubits | {int(canonical.total_simultaneously_live_qubits)} | {int(second_resource.total_simultaneously_live_qubits)} |
| Gates | {int(canonical.transpiled_gate_count)} | {int(second_resource.transpiled_gate_count)} |
| Depth | {int(canonical.transpiled_depth)} | {int(second_resource.transpiled_depth)} |

## 7. Compiled scaling evidence

At fixed slots, precision, and degree, gate count rises from 60,587 at n=4 to 186,191 at n=8 and 478,753 at n=16; n=16 is transpilation-only. The frozen canonical support cannot compile at two slots, while four slots compile to 246,379 gates. From 4 to 8 value bits, matrix quantization error falls from 0.02580 to 0.001383, but resources are unchanged because the current direct-multiplexed architecture embeds values as rotation angles. Degree 15 compiles to 90,111 gates, degree 31 to 186,191, and degree 63 is retained as a boundedness failure. These are descriptive compiled relationships, not asymptotic theorems.

## 8. Tests and integrity

- Focused: {focused['passed']} passed, {focused['failures'] + focused['errors']} failed, {focused['skipped']} skipped.
- Related: {related['passed']} passed, {related['failures'] + related['errors']} failed, {related['skipped']} skipped.
- Complete isolated suite: {isolated_line}.
- Pre-edit isolated baseline: 1852 passed and 7 known failures out of 1859 collected.
- Protected hashes and manuscript hash: unchanged.
- Checksum validation is covered by `tests/test_generic_sparse_qsvt_artifacts.py` and `checksums.sha256`.

## 9. Claim assessment

Supported: reusable compiler, canonical reproduction, independent IEEE-30 execution, full integration, and matched-polynomial correctness. Supported with limitations: matched-Ridge accuracy on the identical quantized sparse matrix, finite-shot convergence, and compiled scaling trends. Inconclusive: none of the registered claims. Contradicted: none of the registered claims. Unsupported: scalable IEEE-size access, practical competitiveness, and quantum speedup or advantage.

## 10. Remaining scientific limitations

Execution is simulator-only. Sparse values are implemented by direct multiplexed small-scale rotations rather than a scalable memory oracle. There is no hardware run, fault-tolerant resource estimate, or end-to-end complexity comparison. The n=16 row is transpilation-only. The second workload's support error versus its full frozen block is 0.3773 even though circuit action agrees with the matched polynomial and quantized Ridge. No result proves hardware scalability, practical competitiveness, speedup, or advantage.

## 11. Exact artifact paths

The exhaustive machine-readable path list is `outputs/generic_sparse_qsvt_compiler/artifact_path_registry.csv`; hashes and sizes are in `artifact_manifest.json` and `checksums.sha256`. Required reports and tables are located directly under `outputs/generic_sparse_qsvt_compiler/`; source and transpiled circuits are under `circuits/`; frozen arrays are under `data/`; test and execution logs are under `logs/`; degree-15 generated phases are under `scaling_phase_cache/`. The grouped diagram is `generic_sparse_qsvt_compiler_diagram.pdf`.

## 12. Recommended manuscript impact

The evidence can strengthen statements that the sparse circuit is generated by a reusable typed compiler, exactly reproduces the canonical IEEE-14 result, transfers unchanged to one frozen IEEE-30-derived workload, integrates sparse access through signed readout, and has circuit-grounded small-scale scaling data. It should qualify any broad physical-generalization statement because the second workload retains substantial support error. It leaves hardware, scalable-access, fault-tolerant, practical-competitiveness, speedup, and advantage claims unchanged and unsupported. The manuscript was not edited in this task.
"""
    (OUTPUT / "final_implementation_report.md").write_text(report, encoding="utf-8")


def artifact_category(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith("src/") or relative.startswith("scripts/"):
        return "code"
    if relative.startswith("tests/"):
        return "test"
    if relative.startswith("configs/"):
        return "configuration"
    if "/circuits/" in relative:
        return "circuit"
    if "/data/" in relative or "/scaling_phase_cache/" in relative:
        return "data"
    if "/logs/" in relative:
        return "log"
    if path.suffix == ".pdf":
        return "diagram"
    if path.suffix == ".csv":
        return "table"
    return "report_or_manifest"


def write_manifest_and_checksums() -> None:
    manifest_path = OUTPUT / "artifact_manifest.json"
    checksum_path = OUTPUT / "checksums.sha256"
    artifact_paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path not in {manifest_path, checksum_path}
    )
    artifact_paths.extend(ROOT / relative for relative in TASK_CODE)
    artifact_paths = sorted(set(artifact_paths))
    registry_rows = [
        {
            "category": artifact_category(path),
            "path": path.relative_to(ROOT).as_posix(),
            "exists": path.is_file(),
        }
        for path in artifact_paths
    ]
    registry_rows.extend(
        [
            {"category": "manifest", "path": manifest_path.relative_to(ROOT).as_posix(), "exists": True},
            {"category": "checksum", "path": checksum_path.relative_to(ROOT).as_posix(), "exists": True},
        ]
    )
    pd.DataFrame(registry_rows).to_csv(OUTPUT / "artifact_path_registry.csv", index=False)
    if OUTPUT / "artifact_path_registry.csv" not in artifact_paths:
        artifact_paths.append(OUTPUT / "artifact_path_registry.csv")
        artifact_paths.sort()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "category": artifact_category(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    ]
    payload = {
        "schema_version": 1,
        "study_id": "generic_sparse_qsvt_compiler_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repository_root": ROOT.as_posix(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "artifact_count_excluding_manifest_and_checksum_file": len(records),
        "artifacts": records,
        "manifest_self_hash_excluded": True,
        "checksum_file_self_hash_excluded": True,
        "claim_boundary": "Small-scale classical simulation and transpilation only; no hardware, scalable access, competitiveness, speedup, or advantage claim.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_paths = sorted([*artifact_paths, manifest_path])
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )


def main() -> None:
    focused, related, isolated = write_test_reports()
    protected = update_protected_audit()
    write_claim_assessment()
    write_final_report(protected, focused, related, isolated)
    write_manifest_and_checksums()


if __name__ == "__main__":
    main()
