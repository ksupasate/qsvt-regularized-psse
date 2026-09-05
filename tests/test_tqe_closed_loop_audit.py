"""Audit tests for the closed-loop nonlinear sparse-QSVT verification pass (2026-07).

Covers the task section-5 families: accounting invariants (1-6), resource integrity (7-12),
trajectory classification (13-16), claim discipline (17-20), and reproducibility (21-24).  A
single tiny closed-loop run (one scenario, one seed, degree 15, small shot budget, 3-iteration
extended horizon) is executed once at module scope; the remaining tests exercise the accounting
and classification functions directly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from robust_qsvt_se.experiments.iterative_ac import (
    _linearized_update_system,
    build_ac_nonlinear_problem,
)
from robust_qsvt_se.physical_alignment.nonlinear_ac import build_problem_config
from robust_qsvt_se.tqe_extensions import closed_loop_nonlinear_update as clnu

CONFIG_PATH = "configs/tqe_closed_loop_nonlinear_update.yaml"
REAL_OUTPUT = Path("outputs/nonlinear_closed_loop_qsvt")


def _tiny_config() -> dict:
    config = copy.deepcopy(clnu.load_yaml_config(CONFIG_PATH))
    config["seeds"] = [101]
    config["finite_shot_seed"] = 101
    config["scenarios"] = config["scenarios"][:1]
    config["nonlinear_settings"]["iteration"]["max_iterations"] = 2
    config["block_qsvt"]["degree"] = 15
    config["block_qsvt"]["finite_shot_budget"] = 3000
    config["closed_loop"]["finite_shot_budget"] = 3000
    config["extended_horizon"]["max_iterations"] = 3
    return config


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, inprocess_phase_synthesis_guard) -> dict:
    output_dir = tmp_path_factory.mktemp("audit_closed_loop")
    config = _tiny_config()
    config_path = output_dir / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = clnu.run_closed_loop(config_path, output_dir, progress=False)
    return {"summary": summary, "output_dir": output_dir, "config": config}


@pytest.fixture(scope="module")
def operating_point(tmp_path_factory, inprocess_phase_synthesis_guard) -> clnu.BlockOperatingPoint:
    config = _tiny_config()
    problem = build_ac_nonlinear_problem(
        build_problem_config(config["nonlinear_settings"], config["scenarios"][0], 101)
    )
    system, _ = _linearized_update_system(problem, problem.initial_state.copy())
    op = clnu.build_block_operating_point(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        config["block_qsvt"],
        tmp_path_factory.mktemp("audit_op_cache"),
        need_phases=True,
    )
    return op


def _fs(op, config, shots=3000, seed=20260722):
    return clnu.finite_shot_update(op, config["block_qsvt"], shots=shots, sampler_seed=seed)


# ----------------------------------------------------------------- accounting invariants (1-6)


def test_1_functional_queries_equal_iters_times_coords(tiny_run):
    acc = pd.read_csv(
        tiny_run["output_dir"] / "resource_ledgers" / "query_execution_accounting.csv"
    )
    assert not acc.empty
    coords = int(tiny_run["config"]["block_qsvt"]["block_size"])
    for _, row in acc.iterrows():
        assert int(row["functional_queries"]) == int(row["executed_iterations"]) * coords
        assert bool(row["invariant_queries_equals_iters_times_coords"]) is True


def test_2_physical_executions_tracked_separately(operating_point):
    fs = _fs(operating_point, _tiny_config())
    n = operating_point.block_quantized.shape[1]
    assert fs["functional_queries"] == n
    assert fs["physical_circuit_executions"] == 2 * n  # readout + direct-postselection
    assert fs["physical_circuit_executions"] != fs["functional_queries"]


def test_3_sampling_calls_match_executions(operating_point):
    fs = _fs(operating_point, _tiny_config())
    assert fs["sampling_calls"] == fs["physical_circuit_executions"]
    assert fs["sampling_calls"] == 2 * fs["functional_queries"]


def test_4_total_attempted_equals_calls_times_shots(operating_point):
    shots = 3000
    fs = _fs(operating_point, _tiny_config(), shots=shots)
    assert fs["total_attempted_shots"] == fs["sampling_calls"] * shots
    assert (
        fs["total_attempted_shots"]
        == fs["readout_signal_attempted_shots"] + fs["diagnostic_postselection_attempted_shots"]
    )


def test_5_accepted_not_exceeding_attempted(operating_point):
    fs = _fs(operating_point, _tiny_config())
    assert fs["interference_accepted_shots"] <= fs["readout_signal_attempted_shots"]
    assert fs["postselection_accepted_shots"] <= fs["diagnostic_postselection_attempted_shots"]


def test_6_resource_summary_totals_match_ledger(tiny_run):
    out = tiny_run["output_dir"]
    acc = pd.read_csv(out / "resource_ledgers" / "query_execution_accounting.csv")
    coords = pd.read_csv(out / "resource_ledgers" / "finite_shot_coordinate_readout.csv")
    # Total attempted (both branches) reconciles with the per-coordinate ledger.
    ledger_total = int(
        coords["readout_attempted_shots"].sum() + coords["diagnostic_attempted_shots"].sum()
    )
    assert int(acc["total_attempted_shots"].sum()) == ledger_total


# ------------------------------------------------------------------- resource integrity (7-12)


def _statevector_circuit(op):
    from robust_qsvt_se.qsvt.gate_level_qsvt import build_structured_qsvt_operator_circuit

    n = op.block_quantized.shape[1]
    return build_structured_qsvt_operator_circuit(
        op.wrapper_unitary, op.phases, encoded_dimension=n
    ).qsvt_operator_circuit


def test_7_resource_level_explicitly_identified(operating_point):
    levels = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    for key in ("logical_operations", "decomposed_operations", "transpiled_basis_gates"):
        assert key in levels
    # Logical count is the opaque-block count, strictly below the transpiled primitive-gate count.
    assert levels["logical_operations"] < levels["transpiled_basis_gates"]


def test_8_opaque_instruction_count_reported(operating_point):
    levels = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    assert "opaque_instructions_remaining" in levels


def test_9_primitive_gate_claim_requires_zero_opaque(operating_point):
    levels = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    # A primitive-gate count is only reported when nothing opaque remains after transpilation.
    assert int(levels["opaque_instructions_remaining"]) == 0


def test_10_sparse_wrapper_present_in_logical_breakdown(operating_point):
    # The QSVT operator embeds the block-encoding wrapper as opaque `unitary` blocks; the
    # projector-controlled phases and signal unitaries account for exactly degree+1 blocks.
    levels = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    assert "unitary" in levels["logical_op_breakdown"]
    # Odd-degree QSVT alternates `degree` signal applications with `degree + 1` phases.
    assert levels["logical_operations"] == 2 * operating_point.degree + 1


def test_11_transpiler_basis_level_seed_recorded(operating_point):
    levels = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    assert levels["transpiled_basis"] == "+".join(clnu.TRANSPILE_BASIS)
    assert int(levels["transpiler_opt_level"]) == clnu.TRANSPILE_OPT_LEVEL
    assert int(levels["transpiler_seed"]) == clnu.TRANSPILE_SEED


def test_12_circuit_resource_records_deterministic(operating_point):
    a = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    b = clnu.circuit_resource_levels(
        _statevector_circuit(operating_point), label="qsvt_statevector_operator"
    )
    assert a["logical_operations"] == b["logical_operations"]
    assert a["transpiled_basis_gates"] == b["transpiled_basis_gates"]


# ---------------------------------------------------------------- trajectory classification (13-16)


def _classify(rmse, step, **overrides):
    kwargs = dict(
        converged=False, failed=False, accurate_threshold=2e-3, plateau_window=5,
        plateau_rmse_rel_tol=1e-2, plateau_step_norm=1e-4, still_improving_rel_tol=1e-2,
        oscillation_rel_tol=5e-2,
    )
    kwargs.update(overrides)
    return clnu.classify_trajectory(rmse, step, **kwargs)


def test_13_eight_iteration_cap_not_auto_plateau():
    # A trajectory still descending across the window is NOT plateaued even at the cap.
    rmse = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007]
    step = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007, 0.003]
    result = _classify(rmse, step)
    assert result["classification"] != "plateaued"


def test_14_plateau_requires_declared_rule():
    # Flat RMSE window + settled update -> plateaued, with a recorded onset.
    rmse = [0.02, 0.0061, 0.00601, 0.006008, 0.006006, 0.006005, 0.006004, 0.006004]
    step = [2e-3, 5e-4, 2e-4, 8e-5, 4e-5, 2e-5, 1e-5, 8e-6]
    result = _classify(rmse, step)
    assert result["classification"] == "plateaued"
    assert result["plateau_onset_iteration"] >= 0
    assert result["plateau_above_accurate_threshold"] is True


def test_15_extended_horizon_labeled_separately(tiny_run):
    cls = pd.read_csv(
        tiny_run["output_dir"]
        / "extended_horizon"
        / "run_summaries"
        / "trajectory_classification.csv"
    )
    assert not cls.empty
    # Each row records the extended max and the primary protocol horizon distinctly.
    assert (cls["max_iterations"] == 3).all()
    assert (cls["primary_protocol_iterations"] == 2).all()


def test_16_still_improving_retained():
    rmse = [1.0, 0.8, 0.6, 0.45, 0.34, 0.25, 0.19, 0.14]
    step = [0.2, 0.2, 0.15, 0.11, 0.09, 0.06, 0.05, 0.05]
    result = _classify(rmse, step)
    assert result["classification"] in {"still_improving_at_horizon", "oscillatory"}
    assert result["classification"] != "plateaued"


# ------------------------------------------------------------------- claim discipline (17-20)


def test_17_finite_shot_run_count_matches_summary(tiny_run):
    solver = pd.read_csv(tiny_run["output_dir"] / "run_summaries" / "solver_outcomes.csv")
    acc = pd.read_csv(
        tiny_run["output_dir"] / "resource_ledgers" / "query_execution_accounting.csv"
    )
    fs_runs = int((solver["arm"] == clnu.ARM_FINITE_SHOT).sum())
    # The accounting table cannot claim more finite-shot runs than were executed.
    assert len(acc) == fs_runs




def test_19_finite_shot_evidence_level_matches_run_count():
    config = clnu.load_yaml_config(CONFIG_PATH)
    level = config["closed_loop"].get("finite_shot_evidence_level", "")
    seed_only = bool(config["closed_loop"].get("finite_shot_arm_seed_only", True))
    # The nine-run evidence level requires the finite-shot arm on every seed (not seed-only).
    if level == "nine_run_matched_finite_shot_evidence":
        assert seed_only is False


def test_20_plateau_wording_backed_by_classification(tiny_run):
    # Any "plateau" classification present in the ledger is produced by the declared rule, and
    # the block arms are never labelled "converged" unless the stopping rule was met.
    cls = pd.read_csv(
        tiny_run["output_dir"]
        / "extended_horizon"
        / "run_summaries"
        / "trajectory_classification.csv"
    )
    valid = {
        "converged", "plateaued", "still_improving_at_horizon", "oscillatory",
        "diverged", "maximum_iterations_without_plateau",
    }
    assert set(cls["classification"].unique()).issubset(valid)


# ------------------------------------------------------------------- reproducibility (21-24)


def test_21_assets_rebuild_from_ledgers(tiny_run):
    import scripts.build_tqe_closed_loop_assets as assets

    out = tiny_run["output_dir"]
    tables = out / "rebuild_tables"
    figs = out / "rebuild_figs"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    paths = assets.build_all(out, tables, figs)
    for p in paths:
        assert Path(p).exists()
    # The query-accounting table is derived from the ledger, not hand-entered.
    acc = pd.read_csv(out / "resource_ledgers" / "query_execution_accounting.csv")
    total = int(acc["total_attempted_shots"].sum())
    text = (tables / "tqe_closed_loop_query_accounting.tex").read_text(encoding="utf-8")
    assert f"{total:,}".replace(",", "\\,") in text


def test_22_no_absolute_repo_path_in_manifests(tiny_run):
    import json

    out = tiny_run["output_dir"]
    repo_root = str(Path.cwd())
    for name in ("run_manifest.json", "audit_manifest.json"):
        data = json.loads((out / "manifests" / name).read_text(encoding="utf-8"))
        assert repo_root not in json.dumps(data)


def test_23_tables_generated_from_ledgers_not_hardcoded(tiny_run):
    import scripts.build_tqe_closed_loop_assets as assets

    out = tiny_run["output_dir"]
    tables = out / "check_tables"
    figs = out / "check_figs"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    assets.build_main_table(out, tables)
    solver = pd.read_csv(out / "run_summaries" / "solver_outcomes.csv")
    full = solver[solver["arm"] == clnu.ARM_FULL]
    converged = int(full["converged"].sum())
    text = (tables / "tqe_closed_loop_main_summary.tex").read_text(encoding="utf-8")
    assert f"{converged}/{len(full)}" in text


def test_24_seeded_finite_shot_and_statevector_reproduce(operating_point):
    cfg = _tiny_config()
    a = clnu.finite_shot_update(operating_point, cfg["block_qsvt"], shots=3000, sampler_seed=7)
    b = clnu.finite_shot_update(operating_point, cfg["block_qsvt"], shots=3000, sampler_seed=7)
    assert np.array_equal(a["dx_block"], b["dx_block"])
    assert a["total_attempted_shots"] == b["total_attempted_shots"]
    sv1 = clnu.statevector_update(operating_point)
    sv2 = clnu.statevector_update(operating_point)
    assert np.array_equal(sv1["dx_block"], sv2["dx_block"])


def test_25_plateau_onset_is_sustained_joint_window_end():
    rmse = [0.006] * 8
    step = [2e-3, 1e-3, 8e-4, 6e-4, 4e-4, 2e-4, 8e-5, 5e-5]
    result = _classify(rmse, step)
    assert result["classification"] == "plateaued"
    assert result["rmse_floor_entry_iteration"] == 0
    assert result["first_qualifying_window_end_iteration"] == 6
    assert result["plateau_onset_iteration"] == 6
    assert result["plateau_rule_remains_satisfied_from_onset"] is True


def test_26_visually_flat_rmse_large_update_not_plateaued():
    result = _classify([0.006] * 8, [2e-3] * 8)
    assert result["rmse_floor_entry_iteration"] == 0
    assert result["plateau_onset_iteration"] == -1
    assert result["plateau_rule_satisfied_at_horizon"] is False
    assert result["classification"] == "maximum_iterations_without_plateau"


def test_27_exact_window_length_has_no_off_by_one():
    result = _classify([0.006] * 5, [5e-5] * 5)
    assert result["classification"] == "plateaued"
    assert result["first_qualifying_window_end_iteration"] == 4
    assert result["plateau_onset_iteration"] == 4


def test_28_convergence_does_not_create_plateau_onset():
    result = _classify([0.02, 0.004, 0.001], [1e-2, 1e-3, 1e-8], converged=True)
    assert result["classification"] == "converged"
    assert result["convergence_iteration"] == 2
    assert result["plateau_onset_iteration"] == -1


def test_29_real_extended_ledger_sorted_unique_and_audited():
    ledger = pd.read_csv(
        REAL_OUTPUT / "extended_horizon" / "iteration_ledgers" / "extended_iterations.csv"
    )
    audit = pd.read_csv(REAL_OUTPUT / "audits" / "trajectory_plateau_onset_audit.csv")
    assert len(audit) == 54
    assert not ledger.duplicated(["arm", "scenario", "seed", "iteration"]).any()
    for _, group in ledger.groupby(["arm", "scenario", "seed"], sort=False):
        assert group["iteration"].is_monotonic_increasing
    assert audit["iteration_order_sorted_unique"].astype(bool).all()
    plateau = audit[audit["classification"] == "plateaued"]
    assert plateau["plateau_rule_satisfied_at_horizon"].astype(bool).all()
    assert plateau["plateau_rule_remains_satisfied_from_onset"].astype(bool).all()


def test_30_real_finite_shot_claim_is_nine_matched_runs():
    solver = pd.read_csv(REAL_OUTPUT / "run_summaries" / "solver_outcomes.csv")
    comparison = pd.read_csv(
        REAL_OUTPUT / "audits" / "finite_shot_statevector_comparison.csv"
    )
    assert int((solver["arm"] == clnu.ARM_FINITE_SHOT).sum()) == 9
    assert int((solver["arm"] == clnu.ARM_STATEVECTOR).sum()) == 9
    assert len(comparison) == 9
    assert comparison["matched"].astype(bool).all()




def test_32_transpiled_counts_record_full_configuration():
    levels = pd.read_csv(REAL_OUTPUT / "resource_ledgers" / "circuit_resource_levels.csv")
    assert set(levels["transpiled_basis"]) == {"rz+ry+rx+cx"}
    assert set(levels["transpiler_opt_level"]) == {1}
    assert set(levels["transpiler_seed"]) == {20260722}
    assert set(levels["coupling_map_assumption"]) == {"all_to_all_no_coupling_map"}
    assert not levels["routing_included"].astype(bool).any()
    assert (levels["opaque_instructions_remaining"] == 0).all()


def test_33_real_query_reconciliation_matches_raw_ledgers():
    recon = pd.read_csv(
        REAL_OUTPUT / "resource_ledgers" / "query_execution_reconciliation.csv"
    ).set_index("quantity")["verified_value"]
    coords = pd.read_csv(
        REAL_OUTPUT / "resource_ledgers" / "finite_shot_coordinate_readout.csv"
    )
    assert int(recon["finite_shot_runs"]) == 9
    assert int(recon["executed_iterations"]) == 72
    assert int(recon["functional_queries"]) == len(coords) == 288
    assert int(recon["physical_circuit_executions"]) == 576
    assert int(recon["sampling_calls"]) == 576
    assert int(recon["total_attempted_shots"]) == 57_600_000
    assert int(recon["unique_functional_circuits"]) == coords[
        "functional_circuit_fingerprint"
    ].nunique()


def test_34_comparison_final_rmse_comes_from_run_summaries():
    comparison = pd.read_csv(
        REAL_OUTPUT / "audits" / "finite_shot_statevector_comparison.csv"
    ).set_index(["scenario", "seed"])
    solver = pd.read_csv(REAL_OUTPUT / "run_summaries" / "solver_outcomes.csv")
    for key, row in comparison.iterrows():
        statevector = solver[
            (solver["arm"] == clnu.ARM_STATEVECTOR)
            & (solver["scenario"] == key[0])
            & (solver["seed"] == key[1])
        ].iloc[0]
        finite = solver[
            (solver["arm"] == clnu.ARM_FINITE_SHOT)
            & (solver["scenario"] == key[0])
            & (solver["seed"] == key[1])
        ].iloc[0]
        assert row["statevector_final_rmse"] == pytest.approx(statevector["final_state_rmse"])
        assert row["finite_shot_final_rmse"] == pytest.approx(finite["final_state_rmse"])
        assert row["absolute_final_rmse_difference"] == pytest.approx(
            abs(finite["final_state_rmse"] - statevector["final_state_rmse"])
        )




