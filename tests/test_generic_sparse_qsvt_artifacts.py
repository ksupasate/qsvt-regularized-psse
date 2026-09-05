from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/generic_sparse_qsvt_compiler"


REQUIRED = (
    "pre_edit_audit.md",
    "compiler_specification.md",
    "compiler_api.md",
    "implementation_change_log.md",
    "canonical_reproduction.csv",
    "canonical_reproduction_report.md",
    "second_workload_selection_protocol.md",
    "second_workload_metadata.json",
    "second_workload_functional_registry.csv",
    "second_workload_statevector_validation.csv",
    "second_workload_shot_rows.csv",
    "second_workload_shot_summary.csv",
    "second_workload_resource_ledger.csv",
    "generic_compiler_validation.csv",
    "dimension_scaling.csv",
    "slot_scaling.csv",
    "value_precision_scaling.csv",
    "degree_scaling.csv",
    "scaling_summary.md",
    "failure_registry.csv",
    "evidence_status_registry.csv",
    "focused_test_report.md",
    "related_test_report.md",
    "isolated_full_test_report.md",
    "protected_hash_audit.json",
    "artifact_manifest.json",
    "checksums.sha256",
    "claim_support_assessment.md",
    "final_implementation_report.md",
    "generic_sparse_qsvt_compiler_diagram.pdf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def test_canonical_reproduction_registry_has_no_mismatch():
    frame = pd.read_csv(OUTPUT / "canonical_reproduction.csv")
    assert len(frame) >= 30
    assert frame["pass"].all()
    assert "finite_shot_selected_output_all_seeds_max_difference" in set(frame["criterion"])


def test_shot_grids_retain_every_budget_seed_functional_and_near_zero_output():
    canonical = pd.read_csv(OUTPUT / "canonical_shot_rows_generic.csv")
    second = pd.read_csv(OUTPUT / "second_workload_shot_rows.csv")
    assert len(canonical) == 3 * 10 * 3
    assert len(second) == 2 * 10 * 3
    assert set(canonical["shots_attempted"]) == {10_000, 100_000, 1_000_000}
    assert set(second["shots_attempted"]) == {10_000, 100_000, 1_000_000}
    assert set(canonical["seed"]) == set(range(10))
    assert set(second["seed"]) == set(range(10))
    assert canonical["same_final_circuit_as_statevector_and_resources"].all()
    assert second["same_final_circuit_as_statevector_and_resources"].all()
    assert canonical["output_state_used_for_preparation"].eq(False).all()
    assert second["output_state_used_for_preparation"].eq(False).all()
    assert canonical["relative_error_numerically_stable"].eq(False).any()


def test_second_shot_absolute_error_converges_descriptively():
    summary = pd.read_csv(OUTPUT / "second_workload_shot_summary.csv")
    for _, group in summary.groupby("functional_id"):
        ordered = group.sort_values("shots")
        errors = ordered["mean_absolute_error_vs_statevector"].to_numpy()
        variances = ordered["mean_analytic_variance_estimate"].to_numpy()
        assert errors[-1] < errors[0]
        assert variances[-1] < variances[0]
        assert ordered["absolute_error_is_primary_metric"].all()


def test_resource_rows_come_from_final_compiler_circuits():
    canonical = pd.read_csv(OUTPUT / "canonical_resource_ledger_generic.csv").iloc[0]
    second = pd.read_csv(OUTPUT / "second_workload_resource_ledger.csv").iloc[0]
    for row in (canonical, second):
        assert row["register_sum"] == row["total_simultaneously_live_qubits"]
        assert bool(row["same_final_circuit_as_shots"])
        assert not bool(row["opaque_instructions_remain"])
        assert not bool(row["dense_fallback_used"])
    assert canonical["transpiled_gate_count"] == 186191
    assert second["transpiled_gate_count"] == 186006


def test_scaling_rows_use_compiled_statuses_and_retain_failures():
    dimension = pd.read_csv(OUTPUT / "dimension_scaling.csv")
    slots = pd.read_csv(OUTPUT / "slot_scaling.csv")
    precision = pd.read_csv(OUTPUT / "value_precision_scaling.csv")
    degree = pd.read_csv(OUTPUT / "degree_scaling.csv")
    assert set(dimension["level"]) == {4, 8, 16}
    assert dimension.loc[dimension.level == 16, "evidence_status"].item() == "transpiled only"
    assert slots.loc[slots.level == 2, "evidence_status"].item() == "failed"
    assert slots.loc[slots.level == 2, "failure_code"].item() == "slot_overflow"
    assert set(precision["level"]) == {4, 6, 8}
    assert precision["analytically_modeled"].eq(False).all()
    assert degree.loc[degree.level == 63, "evidence_status"].item() == "failed"
    assert degree.loc[degree.level == 15, "transpiled"].item()




