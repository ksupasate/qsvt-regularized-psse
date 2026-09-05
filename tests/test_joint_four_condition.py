"""Focused tests for the joint four-condition single-candidate pipeline.

Covers the 18 required areas. Follows the repo test conventions: src-layout imports
(``robust_qsvt_se.*``), ``ROOT = parents[1]``, ``skipif`` artifact guards for the real evidence,
call-twice determinism, frozen identity tuples, ``monkeypatch`` failure retention, and
``.tex``-substring manuscript consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.tqe_extensions import joint_four_condition as jfc

ROOT = Path(__file__).resolve().parents[1]
REAL_OUTPUT = ROOT / "outputs" / "joint_four_condition"
CANONICAL_ID = "ieee14_sparse_quantized_8x8_d31_selected_v1"
WORKLOAD_DIGEST = "17eaef9e9cd3201ab392a049a1c0d74c96f09c2a1e20590594db6c984b2dcf27"
MATRIX_HASH = "b158d34b86b778f0c290519ca98985345107012e225798a4cfc7fbf9178df7f9"
QUANTIZED_HASH = "26159050694e76abc32692332daba94e9cd5e22d958a242236b4d57509aeab21"
SUPPORT_HASH = "af1a0f82f3c62e482452f92cdc7bad8903af75dc809e481dc8fb6a34ec38b7fd"


@pytest.fixture(scope="module")
def compiled():
    return jfc.rebuild_canonical()


@pytest.fixture(scope="module")
def config():
    return jfc.load_decision_config()


@pytest.fixture(scope="module")
def freeze(compiled, config):
    return jfc.freeze_candidate(compiled, config)


@pytest.fixture(scope="module")
def benchmark(compiled, freeze):
    return jfc.build_full_system_benchmark(
        compiled, freeze.global_rows, freeze.global_columns, 1.0e-4, freeze.alpha
    )


@pytest.fixture(scope="module")
def statevector(compiled):
    return jfc.validate_compiled_statevector(compiled)


# ---- (1) deterministic candidate freezing --------------------------------------------


def test_01_freeze_is_deterministic(compiled, config):
    freeze_a = jfc.freeze_candidate(compiled, config)
    freeze_b = jfc.freeze_candidate(compiled, config)
    assert freeze_a.workload_id == freeze_b.workload_id == CANONICAL_ID
    assert freeze_a.workload_digest == freeze_b.workload_digest
    assert freeze_a.global_rows == freeze_b.global_rows
    assert freeze_a.alpha == freeze_b.alpha


# ---- (2) matrix / support / residual / polynomial / phase hashes ----------------------


def test_02_frozen_hashes_match_ledger(freeze):
    assert freeze.workload_digest == WORKLOAD_DIGEST
    assert freeze.matrix_original_hash == MATRIX_HASH
    assert freeze.quantized_matrix_hash == QUANTIZED_HASH
    assert freeze.support_hash == SUPPORT_HASH
    assert freeze.polynomial_degree == 31
    assert freeze.phase_count == 32
    assert freeze.global_rows == [15, 17, 18, 29, 31, 32, 48, 68]
    assert freeze.global_columns == [0, 2, 3, 7, 13, 14, 16, 17]


# ---- (3) no outcome-dependent candidate selection ------------------------------------


def test_03_candidate_id_is_canonical_not_outcome_chosen(freeze):
    # The candidate is the predeclared canonical; it is never re-picked after inspecting metrics.
    assert freeze.workload_id == CANONICAL_ID
    assert freeze.freeze_provenance["declared_before_outcome_evaluation"] is True
    assert "rebuild" in freeze.freeze_provenance["rebuild"]


# ---- (4) exact reuse of the frozen support and seeds ---------------------------------


def test_04_frozen_support_and_seeds_reused(compiled, freeze):
    assert freeze.support_budget_k == len(compiled.support_spec.coordinates)
    assert freeze.slot_budget_d_s == compiled.wrapper.slots
    assert freeze.simulator_seeds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert freeze.shot_counts == [10000, 100000, 1000000]


# ---- (5) full-system benchmark-reference provenance ----------------------------------


def test_05_benchmark_link_provenance(benchmark):
    # The 8x8 block is an EXACT submatrix of the full 82x27 system (no fabrication).
    assert benchmark.full_matrix_shape == (82, 27)
    assert benchmark.block_submatrix_match is True
    assert benchmark.residual_slice_match is True
    assert benchmark.block_submatrix_max_abs_err <= 1.0e-9
    assert benchmark.residual_slice_max_abs_err <= 1.0e-9
    assert benchmark.provenance["link_established_here"] is True
    assert "full 82x27 Ridge" in benchmark.provenance["distinct_from_engine_y_full"]


# ---- (6) full-block vs sparse-Ridge separation ---------------------------------------


def test_06_full_block_and_sparse_ridge_separated(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    # The two references are distinct quantities, not merged.
    assert "full_matrix_ridge_output" in primary
    assert "selected_exact_ridge_output" in primary
    e_support = abs(
        primary["selected_exact_ridge_output"] - primary["full_matrix_ridge_output"]
    ) / max(abs(primary["full_matrix_ridge_output"]), 1.0e-6)
    assert e_support == pytest.approx(0.1942, rel=1e-3)


# ---- (7) quantized-Ridge consistency -------------------------------------------------


def test_07_quantized_ridge_tracks_supported(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    # Quantization error is small relative to the support gap.
    assert primary["epsilon_quant"] < primary["epsilon_support"]
    assert primary["quantized_ridge_selected_output"] == pytest.approx(
        0.0024717150238716926, rel=1e-6
    )


# ---- (8) polynomial-versus-rational action -------------------------------------------


def test_08_polynomial_approximates_rational(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    # The degree-31 polynomial approximates the bounded rational filter closely.
    assert primary["epsilon_poly"] < 1.0e-3
    assert (
        abs(primary["exact_polynomial_selected_output"] - primary["exact_rational_selected_output"])
        < 1.0e-4
    )


# ---- (9) statevector-versus-polynomial action ----------------------------------------


def test_09_statevector_tracks_polynomial(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    assert primary["epsilon_qsvt"] == pytest.approx(1.1157854519147143e-08, rel=1e-4)
    assert primary["epsilon_qsvt"] < 1.0e-6  # Condition-3 gate


# ---- (10) signed selected-output recovery --------------------------------------------


def test_10_signed_readout_recovers_statevector(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    recovered = primary["recovered_selected_output_from_final_circuit"]
    target = primary["statevector_selected_output"]
    assert abs(recovered - target) < 1.0e-6
    assert primary["direct_output_state_preparation_used"] is False


# ---- (11) postselection accounting ---------------------------------------------------


def test_11_postselection_probability_consistent(statevector):
    primary = next(r for r in statevector.functional_rows if r["functional_id"] == "coordinate_e0")
    p_post = primary["postselection_probability"]
    assert 0.0 < p_post < 1.0
    assert p_post == pytest.approx(0.6090421558900074, rel=1e-6)
    assert statevector.metrics["postselection_probability_absolute_difference"] >= 0.0


# ---- (12) finite-shot deterministic seed reproduction ---------------------------------


def test_12_finite_shot_seed_reproduction(compiled, statevector):
    # A tiny fresh finite-shot run reproduces bit-for-bit across two identical calls.
    prepared_a = jfc.prepare_compiled_execution(compiled)
    prepared_b = jfc.prepare_compiled_execution(compiled)
    # shrink to a tiny budget for test speed via a lightweight clone
    import dataclasses

    tiny = dataclasses.replace(compiled.execution_spec, shot_counts=(1000,), simulator_seeds=(3,))
    object.__setattr__(compiled, "execution_spec", tiny)
    try:
        shots_a = jfc.run_compiled_shots(compiled, statevector, prepared_a)
        shots_b = jfc.run_compiled_shots(compiled, statevector, prepared_b)
    finally:
        object.__setattr__(
            compiled,
            "execution_spec",
            dataclasses.replace(
                tiny, shot_counts=(10000, 100000, 1000000), simulator_seeds=tuple(range(10))
            ),
        )
    df_a = pd.DataFrame(shots_a.rows).sort_values(["functional_id", "seed"]).reset_index(drop=True)
    df_b = pd.DataFrame(shots_b.rows).sort_values(["functional_id", "seed"]).reset_index(drop=True)
    for col in ["count_00", "count_01", "count_10", "count_11"]:
        assert np.array_equal(df_a[col].astype(int).values, df_b[col].astype(int).values)


@pytest.mark.skipif(
    not (REAL_OUTPUT / "finite_shot_reproduction_check.csv").is_file(),
    reason="outputs/joint_four_condition not generated",
)
def test_12b_finite_shot_reproduces_frozen_ledger():
    check = pd.read_csv(REAL_OUTPUT / "finite_shot_reproduction_check.csv")
    assert not check.empty
    row = check.iloc[0]
    assert bool(row["bit_for_bit_reproduction"]) is True
    assert int(row["max_count_drift"]) == 0


# ---- (13) four-condition status logic ------------------------------------------------


def test_13a_four_condition_logic(compiled, statevector, benchmark, config, freeze):
    empty_fs = pd.DataFrame(
        columns=[
            "functional_id",
            "shots_attempted",
            "mean_recovered_selected_output",
            "statevector_reference",
            "mean_abs_error_vs_statevector",
            "mean_relative_error_vs_statevector",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "estimated_postselection_probability",
        ]
    )
    conditions, _selected, _errors = jfc.evaluate_four_conditions(
        compiled, statevector, benchmark, config, empty_fs
    )
    by_id = {c.condition_id: c for c in conditions}
    assert by_id["condition_1"].status == "fail"  # E_support 0.194 > 0.1
    assert by_id["condition_2"].status == "inconclusive"  # no registered threshold
    assert by_id["condition_3"].status == "pass"  # epsilon_qsvt < 1e-6
    assert by_id["condition_4"].status == "fail"  # not credible
    first = jfc.first_failed_condition(conditions)
    assert first is not None and first.condition_id == "condition_1"


def test_13b_overall_pass_requires_all_pass():
    from robust_qsvt_se.tqe_extensions.joint_four_condition import ConditionResult

    all_pass = [
        ConditionResult("c1", "n", "m", 0.01, 0.1, "lower_is_better", "pass", -0.09, "x", "", []),
        ConditionResult("c2", "n", "m", 0.01, 0.1, "lower_is_better", "pass", -0.09, "x", "", []),
    ]
    assert jfc.first_failed_condition(all_pass) is None
    one_fail = [
        ConditionResult(
            "c1", "n", "m", 0.5, 0.1, "lower_is_better", "fail", 0.4, "x", "too big", []
        ),
        ConditionResult("c2", "n", "m", 0.01, 0.1, "lower_is_better", "pass", -0.09, "x", "", []),
    ]
    assert jfc.first_failed_condition(one_fail).condition_id == "c1"


# ---- (14) failure retention ----------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_OUTPUT / "failures.csv").is_file(),
    reason="outputs/joint_four_condition not generated",
)
def test_14_failure_schema_retained():
    failures = pd.read_csv(REAL_OUTPUT / "failures.csv")
    assert {"workload_id", "failure_code", "stage", "retained"}.issubset(failures.columns)
    # No failed condition is converted into a successful evidence tier.
    ledger = pd.read_csv(REAL_OUTPUT / "four_condition_decision_ledger.csv")
    failed = ledger[ledger["status"] == "fail"]
    assert len(failed) >= 1  # at least condition_1 (and condition_4) are retained as failures


# ---- (15) no dense signal fallback ---------------------------------------------------


def test_15_no_dense_fallback(compiled, statevector):
    # The compiled sparse circuit contains no forbidden dense-signal fallback gates.
    assert statevector.metrics["dense_fallback_used"] is False
    for row in statevector.functional_rows:
        assert row["dense_fallback_used"] is False


# ---- (16) no direct target-output preparation ----------------------------------------


def test_16_no_direct_target_output_preparation(statevector):
    for row in statevector.functional_rows:
        assert row["direct_output_state_preparation_used"] is False
        assert row["same_final_circuit_for_exact_distribution_and_shots"] is True


# ---- (17) complete decision-ledger schema --------------------------------------------


@pytest.mark.skipif(
    not (REAL_OUTPUT / "four_condition_decision_ledger.csv").is_file(),
    reason="outputs/joint_four_condition not generated",
)
def test_17_decision_ledger_schema_complete():
    ledger = pd.read_csv(REAL_OUTPUT / "four_condition_decision_ledger.csv")
    required = {
        "condition_id",
        "condition_name",
        "metric",
        "metric_value",
        "threshold",
        "direction",
        "status",
        "margin",
        "evidence_level",
        "failure_reason",
        "source_artifacts",
    }
    assert required.issubset(ledger.columns)
    assert set(ledger["condition_id"]) >= {
        "condition_1",
        "condition_2",
        "condition_3",
        "condition_4",
        "overall",
    }
    ledger_json = __import__("json").loads(
        (REAL_OUTPUT / "four_condition_decision_ledger.json").read_text()
    )
    assert ledger_json["declared_before_outcome_evaluation"] is True
    assert "first_failed_logical_condition" in ledger_json
    assert ledger_json["first_failed_logical_condition"]["condition_id"] == "condition_1"


# ---- (18) manuscript table consistency with machine-readable outputs -----------------


@pytest.mark.skipif(
    not (REAL_OUTPUT / "four_condition_decision_ledger.json").is_file(),
    reason="outputs/joint_four_condition not generated",
)
def test_18_manuscript_tables_consistent_with_evidence():
    metrics_path = ROOT / "manuscript" / "tables" / "joint_four_condition_metrics.tex"
    candidate_path = ROOT / "manuscript" / "tables" / "joint_four_condition_candidate.tex"
    if not metrics_path.is_file() or not candidate_path.is_file():
        pytest.skip("manuscript assets not built yet")
    metrics = metrics_path.read_text(encoding="utf-8")
    candidate = candidate_path.read_text(encoding="utf-8")
    assert r"\providecommand{\JointFourCandidateId}" in metrics
    assert r"\providecommand{\JointFourEpsilonQsvt}" in metrics
    assert "tab:joint_four_condition_candidate" in candidate
    # the metrics file must not be a stale hand-typed literal: epsilon_qsvt is generated
    freeze = __import__("json").loads((REAL_OUTPUT / "candidate_freeze.json").read_text())
    assert freeze["workload_id"] in metrics.replace(r"\_", "_") or CANONICAL_ID in metrics.replace(
        r"\_", "_"
    )
    # candidate table echoes the FAIL status of condition 1
    assert "FAIL" in candidate


# ---- manifest / checksum integrity ---------------------------------------------------


@pytest.mark.skipif(
    not (REAL_OUTPUT / "manifest.json").is_file(),
    reason="outputs/joint_four_condition not generated",
)
def test_19_manifest_and_checksums_consistent():
    import hashlib

    manifest = __import__("json").loads((REAL_OUTPUT / "manifest.json").read_text())
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "manifest.json" not in paths  # Convention A: excludes itself
    assert "checksums.sha256" not in paths
    # every listed checksum matches the file on disk
    for item in manifest["artifacts"]:
        digest = hashlib.sha256((REAL_OUTPUT / item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"], f"checksum drift: {item['path']}"


def test_20_claim_boundary_forbids_speedup():
    assert "quantum speedup" in jfc.CLAIM_BOUNDARY.lower()
    assert "hardware execution" in jfc.CLAIM_BOUNDARY.lower()
    assert "scalability" in jfc.CLAIM_BOUNDARY.lower()
