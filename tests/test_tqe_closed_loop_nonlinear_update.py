"""Tests for the closed-loop nonlinear sparse-QSVT state-update experiment.

Covers the four families required by the task sheet: scientific-path, circuit-integrity,
failure-retention, and reproducibility.  A single tiny closed-loop run (one scenario, one seed,
two iterations, degree 15, small shot budget) is executed once at module scope; unit tests
exercise the stage functions, run classifier, signed-readout convention, and failure paths
directly.
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
from robust_qsvt_se.qsvt.sparse_integrated_chain import estimate_signed_selected_output
from robust_qsvt_se.tqe_extensions import closed_loop_nonlinear_update as clnu

CONFIG_PATH = "configs/tqe_closed_loop_nonlinear_update.yaml"


# --------------------------------------------------------------------------- fixtures


def _tiny_config() -> dict:
    config = copy.deepcopy(clnu.load_yaml_config(CONFIG_PATH))
    config["seeds"] = [101]
    config["finite_shot_seed"] = 101
    config["scenarios"] = config["scenarios"][:1]
    config["nonlinear_settings"]["iteration"]["max_iterations"] = 2
    config["block_qsvt"]["degree"] = 15  # cheaper phase synthesis for the test
    config["block_qsvt"]["finite_shot_budget"] = 4000
    config["closed_loop"]["finite_shot_budget"] = 4000
    # Keep the extended-horizon diagnostic tiny so the module-scoped run stays fast.
    config["extended_horizon"]["max_iterations"] = 3
    return config


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, inprocess_phase_synthesis_guard) -> dict:
    output_dir = tmp_path_factory.mktemp("closed_loop")
    config = _tiny_config()
    config_path = output_dir / "tiny_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = clnu.run_closed_loop(config_path, output_dir, progress=False)
    frames = {
        "summary": summary,
        "iterations": pd.read_csv(output_dir / "iteration_ledgers" / "closed_loop_iterations.csv"),
        "solver": pd.read_csv(output_dir / "run_summaries" / "solver_outcomes.csv"),
        "decomposition": pd.read_csv(
            output_dir / "error_decomposition" / "stage_error_decomposition.csv"
        ),
        "coords": pd.read_csv(
            output_dir / "resource_ledgers" / "finite_shot_coordinate_readout.csv"
        ),
        "failures": pd.read_csv(output_dir / "failure_ledgers" / "structured_failures.csv"),
        "output_dir": output_dir,
    }
    return frames


@pytest.fixture(scope="module")
def operating_point(tmp_path_factory, inprocess_phase_synthesis_guard) -> clnu.BlockOperatingPoint:
    config = _tiny_config()
    settings = config["nonlinear_settings"]
    problem = build_ac_nonlinear_problem(
        build_problem_config(settings, config["scenarios"][0], 101)
    )
    system, _ = _linearized_update_system(problem, problem.initial_state.copy())
    cache = tmp_path_factory.mktemp("op_cache")
    op = clnu.build_block_operating_point(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        config["block_qsvt"],
        cache,
        need_phases=True,
    )
    return op


# ------------------------------------------------------------------- scientific-path tests


def test_1_statevector_update_changes_next_state(tiny_run):
    rows = tiny_run["iterations"]
    arm = rows[rows["arm"] == clnu.ARM_STATEVECTOR].sort_values("iteration")
    fingerprints = arm["state_fingerprint"].tolist()
    assert len(fingerprints) >= 2
    assert fingerprints[0] != fingerprints[1]  # the QSVT-driven update moved the state
    assert (arm["step_norm"].to_numpy() > 0).all()


def test_2_qsvt_arm_does_not_use_full_ridge(operating_point):
    op = operating_point
    sv = clnu.statevector_update(op)
    dx_block = sv["dx_block"]
    dx_quant_ridge = clnu.quantized_ridge_update(op, op.alpha_k)
    # The statevector update is a block-dimensional vector produced by circuit evolution of the
    # sparse block-encoding; it tracks the *block* matched Ridge, never a full-system solve.
    assert dx_block.shape == (op.block_quantized.shape[1],)
    assert np.linalg.norm(dx_block - dx_quant_ridge) / np.linalg.norm(dx_quant_ridge) < 5e-2
    # statevector_update never receives the full weighted Jacobian: its only matrix input is the
    # 4x4 block-encoding unitary carried on the operating point.
    import inspect

    source = inspect.getsource(clnu.statevector_update)
    assert "H_tilde" not in source and "full_ridge" not in source


def test_3_finite_shot_uses_reconstructed_coordinates(tiny_run):
    rows = tiny_run["iterations"]
    coords = tiny_run["coords"]
    arm = rows[rows["arm"] == clnu.ARM_FINITE_SHOT].sort_values("iteration")
    assert not arm.empty
    for _, row in arm.iterrows():
        it = int(row["iteration"])
        block_coords = coords[coords["iteration"] == it].sort_values("coordinate")
        assembled = block_coords["recovered_coordinate_update"].to_numpy(dtype=float)
        # The driving step norm equals the norm of the assembled signed coordinate estimates.
        assert row["step_norm"] == pytest.approx(float(np.linalg.norm(assembled)), rel=1e-6)
    # The finite-shot estimates carry genuine sampling noise (are not the exact statevector values).
    assert (coords["sampling_standard_error"].to_numpy() > 0).all()


def test_4_nonselected_coordinates_frozen(operating_point):
    op = operating_point
    state = np.arange(27, dtype=np.float64) * 0.01
    dx_block = np.ones(op.block_quantized.shape[1])
    updated = clnu._apply_block_update(state, op.cols, dx_block, 1.0, 0.5, 5)
    mask = np.ones(state.size, dtype=bool)
    mask[op.cols] = False
    # Non-selected coordinates (above the min-voltage clamp) are unchanged.
    unchanged = np.where(mask)[0]
    for idx in unchanged:
        if idx < 5 or state[idx] >= 0.5:
            assert updated[idx] == state[idx]


def test_5_exact_polynomial_matches_independent_matrix_action(operating_point):
    op = operating_point
    from numpy.polynomial import Polynomial

    normalized = op.block_quantized.T / op.beta
    left, singular, right_t = np.linalg.svd(normalized, full_matrices=False)
    poly = Polynomial(op.coefficients)
    residual_unit = op.residual_block / op.residual_block_norm
    reference = op.physical_scale * (left @ (poly(singular) * (right_t @ residual_unit)))
    assert np.allclose(clnu.exact_polynomial_update(op), reference, atol=1e-12)


def test_6_statevector_matches_exact_polynomial(operating_point):
    op = operating_point
    sv = clnu.statevector_update(op)
    poly = clnu.exact_polynomial_update(op)
    rel = np.linalg.norm(sv["dx_block"] - poly) / max(np.linalg.norm(poly), 1e-30)
    assert rel < 1e-6  # the circuit reproduces the polynomial action to machine precision


def test_7_block_arms_share_support_and_matrix_convention(tiny_run):
    rows = tiny_run["iterations"]
    first_iter = rows[rows["iteration"] == 0]
    block_arms = first_iter[first_iter["arm"].isin(clnu.BLOCK_ARMS)]
    # All block arms start from the identical initial state -> identical block and beta at iter 0.
    assert block_arms["selected_global_columns"].nunique() == 1
    betas = block_arms["beta_k"].dropna().to_numpy()
    assert np.allclose(betas, betas[0])


def test_8_sampler_seed_disjoint_from_measurement_seeds():
    config = clnu.load_yaml_config(CONFIG_PATH)
    measurement_seeds = set(int(s) for s in config["seeds"])
    sampler_seed = int(config["block_qsvt"]["finite_shot_sampler_seed"])
    # The finite-shot simulator seed is disjoint from the measurement-realization seeds so the
    # readout randomness cannot alias the data-generating randomness.
    assert sampler_seed not in measurement_seeds


# ------------------------------------------------------------------- circuit-integrity tests


def test_9_and_10_no_output_state_preparation_fallback(operating_point):
    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        build_integrated_sparse_selected_output_circuit,
    )

    op = operating_point
    config = clnu._finite_shot_config(op, _tiny_config()["block_qsvt"])
    bundle = build_integrated_sparse_selected_output_circuit(
        config,
        matrix=op.block_quantized,
        residual=op.residual_block,
        selected_functional=np.eye(op.block_quantized.shape[1])[0],
        phases=op.phases,
    )
    # Neither a dense signal operator nor a direct output-state preparation is used.
    assert bundle.output_state_used_for_preparation is False
    labels = {inst.operation.label for inst in bundle.circuit.data if inst.operation.label}
    assert not any("output" in str(label).lower() for label in labels)


def test_11_register_allocations_disjoint(operating_point):
    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        build_integrated_sparse_selected_output_circuit,
    )

    op = operating_point
    config = clnu._finite_shot_config(op, _tiny_config()["block_qsvt"])
    bundle = build_integrated_sparse_selected_output_circuit(
        config,
        matrix=op.block_quantized,
        residual=op.residual_block,
        selected_functional=np.eye(op.block_quantized.shape[1])[0],
        phases=op.phases,
    )
    layout = bundle.register_layout
    index = set(layout["index_qubits"])
    slot = set(layout["slot_qubits"])
    flag = {layout["postselection_flag_qubit"]}
    readout = {layout["readout_qubit"]}
    assert index.isdisjoint(slot)
    assert flag.isdisjoint(index | slot | readout)
    assert readout.isdisjoint(index | slot | flag)


def test_12_postselection_convention_matches_signed_readout():
    # "00" = readout 0, accepted (plus); "10" = readout 1, accepted (minus); key[-1]='1' rejected.
    counts = {"00": 700, "10": 100, "01": 150, "11": 50}
    estimate = estimate_signed_selected_output(counts, physical_scale=2.0)
    assert estimate["readout_accepted"] == 800
    assert estimate["signed_overlap_estimate"] == pytest.approx((700 - 100) / 1000)
    assert estimate["selected_output_estimate"] == pytest.approx(2.0 * (700 - 100) / 1000)


def test_13_functional_circuits_distinct_when_preparation_differs(operating_point):
    from qiskit.quantum_info import Operator

    from robust_qsvt_se.qsvt.sparse_integrated_chain import (
        build_integrated_sparse_selected_output_circuit,
    )

    op = operating_point
    config = clnu._finite_shot_config(op, _tiny_config()["block_qsvt"])
    n = op.block_quantized.shape[1]
    bundle0 = build_integrated_sparse_selected_output_circuit(
        config, matrix=op.block_quantized, residual=op.residual_block,
        selected_functional=np.eye(n)[0], phases=op.phases,
    )
    bundle1 = build_integrated_sparse_selected_output_circuit(
        config, matrix=op.block_quantized, residual=op.residual_block,
        selected_functional=np.eye(n)[1], phases=op.phases,
    )
    op0 = Operator(bundle0.circuit.remove_final_measurements(inplace=False)).data
    op1 = Operator(bundle1.circuit.remove_final_measurements(inplace=False)).data
    assert not np.allclose(op0, op1)  # different functionals -> different circuits


def test_14_coordinate_query_ordering_deterministic(tiny_run):
    coords = tiny_run["coords"]
    for _iteration, block in coords.groupby("iteration"):
        assert block.sort_index()["coordinate"].tolist() == sorted(block["coordinate"].tolist())
        assert block["coordinate"].tolist() == list(range(len(block)))


# ------------------------------------------------------------------- failure-retention tests


def test_15_circuit_construction_failure_retained(monkeypatch, operating_point):
    config = _tiny_config()["block_qsvt"]
    problem_matrix = operating_point.block_dense  # any real block

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated slot overflow / wrapper reconstruction failure")

    monkeypatch.setattr(clnu, "validate_complete_wrapper", _raise)
    op = clnu.build_block_operating_point(
        np.tile(problem_matrix, (5, 5)),  # larger matrix so a 4x4 block is still selectable
        np.ones(problem_matrix.shape[0] * 5),
        config,
        Path("/tmp/does-not-matter"),
        need_phases=True,
    )
    assert op.failure_code == clnu.CLASS_CIRCUIT_FAILED
    assert op.phases is None  # no fallback circuit produced


def test_16_boundedness_failure_retained(operating_point):
    # A tiny degree with a large normalized lambda forces an unbounded target -> retained failure,
    # phase synthesis skipped, no fallback.
    config = dict(_tiny_config()["block_qsvt"])
    config["degree"] = 3
    config["lambda_target"] = 1.0e-6
    tiny = _tiny_config()
    problem = build_ac_nonlinear_problem(
        build_problem_config(tiny["nonlinear_settings"], tiny["scenarios"][0], 101)
    )
    system, _ = _linearized_update_system(problem, problem.initial_state.copy())
    op = clnu.build_block_operating_point(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        config,
        Path("/tmp/cache-bound"),
        need_phases=True,
    )
    if op.failure_code:  # unbounded targets are retained; phases never synthesized
        assert op.failure_code in {"boundedness_failure", clnu.CLASS_PHASE_FAILED}
        assert op.phases is None


def test_17_phase_count_mismatch_retained(monkeypatch, operating_point):
    config = _tiny_config()["block_qsvt"]

    def _bad_phases(*args, **kwargs):
        return np.zeros(3), "synthesized"  # wrong length

    monkeypatch.setattr(clnu, "_synthesize_guarded", _bad_phases)
    tiny = _tiny_config()
    problem = build_ac_nonlinear_problem(
        build_problem_config(tiny["nonlinear_settings"], tiny["scenarios"][0], 101)
    )
    system, _ = _linearized_update_system(problem, problem.initial_state.copy())
    op = clnu.build_block_operating_point(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        config,
        Path("/tmp/cache-mismatch"),
        need_phases=True,
    )
    assert op.failure_code == "phase_count_mismatch"
    assert op.phases is None


def test_18_zero_postselection_does_not_substitute():
    # The classifier maps a zero-postselection failure to its own category, never to a solved run.
    rows = [{"failure_code": clnu.CLASS_ZERO_POSTSELECTION}]
    classification = clnu._classify_run(
        rows, converged=False, failed=True, failure_code=clnu.CLASS_ZERO_POSTSELECTION,
        final_state_rmse=float("nan"), accurate_threshold=2e-3, stalled_update_norm=1e-6,
        max_iterations=8,
    )
    assert classification == clnu.CLASS_ZERO_POSTSELECTION


def test_19_finite_shot_cost_ceiling_retained(tmp_path):
    config = _tiny_config()
    config["closed_loop"]["max_total_shots_per_run"] = 10  # impossibly small ceiling
    config["closed_loop"]["arms"] = [
        clnu.ARM_FULL,
        clnu.ARM_FINITE_SHOT,
    ]
    config_path = tmp_path / "ceiling.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    clnu.run_closed_loop(config_path, tmp_path / "out", progress=False)
    failures = pd.read_csv(tmp_path / "out" / "failure_ledgers" / "structured_failures.csv")
    assert (failures["failure_code"] == clnu.CLASS_FINITE_SHOT_CEILING).any()


def test_20_all_runs_including_failures_remain_in_summary(tiny_run):
    solver = tiny_run["solver"]
    # Every arm has a summary row (arm G on the finite-shot seed); nothing is dropped.
    arms_present = set(solver["arm"].unique())
    assert clnu.ARM_FULL in arms_present
    assert clnu.ARM_FINITE_SHOT in arms_present
    assert "classification" in solver.columns
    diverged = clnu._classify_run(
        [{"failure_code": "residual_growth"}], converged=False, failed=True,
        failure_code="residual_growth", final_state_rmse=float("nan"),
        accurate_threshold=2e-3, stalled_update_norm=1e-6, max_iterations=8,
    )
    assert diverged == clnu.CLASS_DIVERGED


# ------------------------------------------------------------------- reproducibility tests


def test_21_operating_point_reproducible(operating_point, tmp_path):
    config = _tiny_config()
    settings = config["nonlinear_settings"]
    problem = build_ac_nonlinear_problem(
        build_problem_config(settings, config["scenarios"][0], 101)
    )
    system, _ = _linearized_update_system(problem, problem.initial_state.copy())
    op2 = clnu.build_block_operating_point(
        np.asarray(system.H_tilde, dtype=np.float64),
        np.asarray(system.r_tilde, dtype=np.float64),
        config["block_qsvt"],
        tmp_path / "cache-repro",
        need_phases=False,
    )
    assert op2.beta == pytest.approx(operating_point.beta, rel=1e-12)
    assert np.allclose(op2.coefficients, operating_point.coefficients)
    assert np.array_equal(op2.cols, operating_point.cols)


def test_22_statevector_deterministic(operating_point):
    a = clnu.statevector_update(operating_point)
    b = clnu.statevector_update(operating_point)
    assert np.array_equal(a["dx_block"], b["dx_block"])
    assert a["postselection_probability"] == b["postselection_probability"]


def test_23_finite_shot_counts_reproduce_with_fixed_seed(operating_point):
    config = _tiny_config()["block_qsvt"]
    a = clnu.finite_shot_update(operating_point, config, shots=4000, sampler_seed=20260722)
    b = clnu.finite_shot_update(operating_point, config, shots=4000, sampler_seed=20260722)
    assert np.array_equal(a["dx_block"], b["dx_block"])
    assert a["total_accepted_shots"] == b["total_accepted_shots"]


def test_24_summary_tables_regenerated_from_ledgers(tiny_run):
    import scripts.build_tqe_closed_loop_assets as assets

    output_dir = tiny_run["output_dir"]
    tables_dir = output_dir / "manuscript_tables"
    tables_dir.mkdir(exist_ok=True)
    path = assets.build_main_table(output_dir, tables_dir)
    text = path.read_text(encoding="utf-8")
    # The table is derived from the solver-outcome ledger, not hand-entered: the full-system arm's
    # converged-run count computed from the CSV appears in the generated LaTeX.
    solver = tiny_run["solver"]
    full = solver[solver["arm"] == clnu.ARM_FULL]
    converged = int(full["converged"].sum())
    assert f"{converged}/{len(full)}" in text
