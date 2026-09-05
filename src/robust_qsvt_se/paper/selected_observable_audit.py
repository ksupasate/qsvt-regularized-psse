"""Task A: repository audit and integration plan for the selected-observable workload.

Read-only except for writing the two audit artifacts. Records the existing
modules, configs, and outputs this workload builds on; which workload outputs are
present or still missing; the integration points; the reused functions; and which
parts are modeled rather than implemented. Nothing here changes solver behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from robust_qsvt_se.paper.selected_observable_common import (
    WORKLOAD_CLAIM_BOUNDARY,
    WORKLOAD_DIR,
    assert_safe,
)
from robust_qsvt_se.utils.io import ensure_directory, write_json

# Existing modules reused by the workload (module path -> what is reused).
REUSED_MODULES: dict[str, str] = {
    "robust_qsvt_se.qsvt.engineering_utils": "build_engineering_system, ridge_svd_solution",
    "robust_qsvt_se.qsvt.sparse_access_oracle": "SparseAccessOracle, build_sparse_access_oracle",
    "robust_qsvt_se.qsvt.state_metadata": "build_state_metadata_from_system_metadata",
    "robust_qsvt_se.paper.signed_readout_diagnostic": "hadamard_test_estimate",
    "robust_qsvt_se.qsvt.tqe_degree_alpha_precision_sweep": (
        "bounded_ridge_normalization_C, bounded_ridge_target, fit_bounded_ridge_polynomial"
    ),
    "robust_qsvt_se.measurement.linear_system": "WeightedSystem",
    "robust_qsvt_se.paper.tqe_revision_support_common": "find_forbidden, manifest helpers",
    "robust_qsvt_se.paper._common": "read_csv, rows_to_table",
}

# New workload modules and their task.
NEW_MODULES: dict[str, str] = {
    "robust_qsvt_se.qsvt.sparse_access": "Task B: SparseAccessModel API + qubit/precision metadata",
    "robust_qsvt_se.qsvt.selected_observables": "Task C: physical PSSE observable builders",
    "robust_qsvt_se.qsvt.readout_diagnostics": "Task C: sign-aware + basis-sampling shot sweeps",
    "robust_qsvt_se.paper.sparse_access_workload": "Task B workload tables",
    "robust_qsvt_se.paper.selected_observable_workload": "Task C workload tables",
    "robust_qsvt_se.paper.degree_aware_alpha": "Task D: degree-aware alpha selection",
    "robust_qsvt_se.paper.selected_observable_cost": "Task E: selected-observable cost accounting",
    "robust_qsvt_se.paper.selected_observable_consolidation": "Task F: consolidation + manifest",
}

# Existing configs/outputs reused or cross-referenced.
REUSED_CONFIGS = (
    "configs/qsvt_resource_full_ieee.yaml",
    "configs/alpha_sensitivity_real_ieee14.yaml",
)
REUSED_OUTPUTS = (
    "outputs/full_alpha_sensitivity_classical/alpha_sweep_summary_by_case.csv",
    "outputs/hardware_aware_oracle_cost_model/qsvt_total_cost_estimate.csv",
    "outputs/qsvt_oracle_model_resources/oracle_model_resource_summary.csv",
)

# Workload outputs this package generates (relative to WORKLOAD_DIR).
WORKLOAD_OUTPUTS = (
    "implementation_audit.md",
    "repo_integration_plan.json",
    "sparse_access_summary.csv",
    "sparse_access_summary.json",
    "sparse_access_validation.csv",
    "sparse_access_report.md",
    "selected_observables.csv",
    "readout_shot_sweep.csv",
    "readout_map.csv",
    "readout_summary.md",
    "degree_aware_alpha_grid.csv",
    "degree_aware_alpha_summary.csv",
    "degree_aware_alpha_report.md",
    "selected_observable_cost.csv",
    "selected_observable_cost_summary.md",
    "claim_boundary_update.md",
    "paper_ready_tables.md",
    "manifest.json",
)

# Modeled-vs-implemented ledger.
MODELED_ASSUMPTIONS = (
    (
        "sparse_access_oracle",
        "validated_emulator",
        "exact CSR index/value lookup, validated; reversible circuit not synthesized",
    ),
    ("state_preparation", "modeled", "residual-state loader assumed; not synthesized"),
    (
        "qsvt_query_count_2d_plus_1",
        "validated",
        "degree and 2d+1 query count computed from the bounded target",
    ),
    (
        "postselection_success_probability",
        "proxy",
        "amplitude-ratio / ingested proxy; no measured hardware yield",
    ),
    ("amplitude_amplification", "modeled", "optional O(1/sqrt(p)) overhead when applied"),
    (
        "selected_observable_readout",
        "proxy",
        "unbiased Monte-Carlo shot estimate of one selected functional",
    ),
    ("full_vector_recovery", "excluded", "one readout per state component; out of scope"),
    (
        "phase_synthesis",
        "feasibility_hint",
        "degree-range hint; not a per-row synthesis in this workload",
    ),
)


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(config or {})
    output_dir = ensure_directory(Path(resolved.get("output_dir", WORKLOAD_DIR)))
    repo_root = Path(resolved.get("repo_root", "."))

    reused_modules = {name: _module_present(name) for name in REUSED_MODULES}
    new_modules = {name: _module_present(name) for name in NEW_MODULES}
    reused_configs = {path: (repo_root / path).is_file() for path in REUSED_CONFIGS}
    reused_outputs = {path: (repo_root / path).is_file() for path in REUSED_OUTPUTS}
    workload_present = {name: (output_dir / name).is_file() for name in WORKLOAD_OUTPUTS}
    missing_outputs = [name for name, present in workload_present.items() if not present]

    plan = {
        "claim_boundary": WORKLOAD_CLAIM_BOUNDARY,
        "reused_modules": {
            name: {"reused": REUSED_MODULES[name], "importable": ok}
            for name, ok in reused_modules.items()
        },
        "new_modules": {
            name: {"task": NEW_MODULES[name], "importable": ok} for name, ok in new_modules.items()
        },
        "reused_configs": reused_configs,
        "reused_outputs": reused_outputs,
        "workload_outputs_present": workload_present,
        "missing_workload_outputs": missing_outputs,
        "integration_points": {
            "weighted_jacobian_source": "build_engineering_system (pypower AC-linearized)",
            "matched_update": "ridge_svd_solution (= (H~^T H~ + alpha I)^-1 H~^T r~)",
            "sparse_access": "wraps SparseAccessOracle (validated CSR lookup)",
            "qsvt_degree": "bounded-target convention from tqe_degree_alpha_precision_sweep",
            "readout": "hadamard_test_estimate (sign-aware) + basis-sampling energy",
        },
        "modeled_vs_implemented": [
            {"component": component, "status": status, "detail": detail}
            for component, status, detail in MODELED_ASSUMPTIONS
        ],
        "estimator_behavior_changed": False,
        "existing_outputs_overwritten": False,
    }

    audit_md = output_dir / "implementation_audit.md"
    plan_json = output_dir / "repo_integration_plan.json"
    audit_text = _audit_markdown(plan, reused_modules, new_modules, reused_outputs, missing_outputs)
    assert_safe(audit_text)
    audit_md.write_text(audit_text, encoding="utf-8")
    write_json(plan_json, plan)

    return {
        "output_dir": output_dir,
        "plan": plan,
        "artifacts": {"implementation_audit_md": audit_md, "repo_integration_plan_json": plan_json},
    }


def _module_present(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _audit_markdown(
    plan: dict[str, Any],
    reused_modules: dict[str, bool],
    new_modules: dict[str, bool],
    reused_outputs: dict[str, bool],
    missing_outputs: list[str],
) -> str:
    lines = [
        "# Selected-Observable Workload: Implementation Audit",
        "",
        WORKLOAD_CLAIM_BOUNDARY,
        "",
        "This audit is read-only except for writing this file and "
        "`repo_integration_plan.json`. It does not change estimator behavior or overwrite "
        "existing outputs.",
        "",
        "## Reused Existing Modules",
        "",
        "| Module | Reused | Importable |",
        "| --- | --- | --- |",
    ]
    for name, ok in reused_modules.items():
        lines.append(f"| `{name}` | {REUSED_MODULES[name]} | {ok} |")
    lines += [
        "",
        "## New Workload Modules",
        "",
        "| Module | Task | Importable |",
        "| --- | --- | --- |",
    ]
    for name, ok in new_modules.items():
        lines.append(f"| `{name}` | {NEW_MODULES[name]} | {ok} |")
    lines += [
        "",
        "## Reused Existing Outputs (cross-reference)",
        "",
        "| Output | Present |",
        "| --- | --- |",
    ]
    for path, present in reused_outputs.items():
        lines.append(f"| `{path}` | {present} |")
    lines += [
        "",
        "## Workload Outputs Still Missing",
        "",
    ]
    if missing_outputs:
        lines.extend(f"- `{name}`" for name in missing_outputs)
    else:
        lines.append("- None: all workload outputs are present.")
    lines += [
        "",
        "## Modeled vs Implemented Ledger",
        "",
        "| Component | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for entry in plan["modeled_vs_implemented"]:
        lines.append(f"| {entry['component']} | {entry['status']} | {entry['detail']} |")
    lines += [
        "",
        "## Integration Points",
        "",
    ]
    for key, value in plan["integration_points"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the selected-observable workload audit")
    parser.add_argument("--output-dir", default=str(WORKLOAD_DIR))
    args = parser.parse_args(argv)
    run = run_audit({"output_dir": args.output_dir})
    print(f"Audit complete: {run['artifacts']['implementation_audit_md']}")


if __name__ == "__main__":  # pragma: no cover
    main()
