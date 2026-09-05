"""Tests for the Phase 8 integrated 4x4 finite-shot QSVT chain."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import robust_qsvt_se.paper.phase8_integrated_readout as phase8
from robust_qsvt_se.paper.phase8_integrated_readout import (
    PER_SEED_COLUMNS,
    SUMMARY_COLUMNS,
    build_integrated_readout_circuit,
    estimate_from_integrated_counts,
    run_phase8_integrated_readout,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

_NUM_SEEDS = 3
_SHOTS = [500, 4000]


@pytest.fixture(scope="module")
def readout_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase8_integrated_readout")
    return run_phase8_integrated_readout(
        {
            "output_dir": str(output_dir),
            "num_seeds": _NUM_SEEDS,
            "shots_grid": _SHOTS,
        }
    )


def test_no_direct_output_state_preparation_in_source():
    source = Path(phase8.__file__).read_text(encoding="utf-8")
    assert "StatePreparation(output_state" not in source
    builder = inspect.getsource(build_integrated_readout_circuit)
    assert "output_state" not in builder
    chain_builder = inspect.getsource(phase8.build_direct_chain_circuit)
    assert "output_state" not in chain_builder


def test_no_output_state_preparation_flagged_in_rows(readout_run):
    per_seed = readout_run["per_seed"]
    assert not per_seed["output_state_used_for_preparation"].any()
    assert readout_run["circuit_metadata"]["output_state_used_for_preparation"] is False


def test_signal_unitary_call_count_is_degree(readout_run):
    metadata = readout_run["circuit_metadata"]
    degree = int(metadata["degree"])
    assert metadata["signal_unitary_calls_per_attempt"] == degree
    assert metadata["projector_phase_operations_per_attempt"] == degree + 1
    assert metadata["alternating_sequence_length"] == 2 * degree + 1
    for record in metadata["per_observable_circuits"].values():
        assert record["signal_unitary_calls_per_attempt"] == degree
        labels = record["gate_label_counts"]
        controlled_signal = sum(
            count for name, count in labels.items() if name.startswith("c0_U_A")
        )
        assert controlled_signal == degree
        assert labels.get("c0_PCPhase", 0) == degree + 1
        assert labels.get("c0_residual_prep", 0) == 1
        assert labels.get("c1_functional_prep", 0) == 1


def test_measured_postselection_recorded_and_plausible(readout_run):
    per_seed = readout_run["per_seed"]
    p_sv = float(readout_run["references"]["statevector_postselection_probability"])
    measured = per_seed["measured_postselection_probability"].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(measured))
    assert np.all(measured > 0.5)
    largest = per_seed[per_seed["shots"] == max(_SHOTS)]
    assert abs(largest["measured_postselection_probability"].mean() - p_sv) < 0.05
    direct = largest["direct_chain_measured_postselection_probability"]
    assert abs(direct.mean() - p_sv) < 0.05


def test_recovery_uses_measured_statistics(readout_run):
    per_seed = readout_run["per_seed"]
    f_hat = per_seed["acceptance_frequency"].to_numpy(dtype=np.float64)
    x_bar = per_seed["readout_sign_mean_accepted"].to_numpy(dtype=np.float64)
    z_hat = per_seed["signed_overlap_estimate_z"].to_numpy(dtype=np.float64)
    scale = per_seed["physical_recovery_scale"].to_numpy(dtype=np.float64)
    recovered = per_seed["recovered_physical_functional"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(z_hat, f_hat * x_bar, atol=1.0e-12)
    np.testing.assert_allclose(recovered, scale * z_hat, rtol=1.0e-12)
    p_hat = per_seed["measured_postselection_probability"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(p_hat, 2.0 * f_hat - 1.0, atol=1.0e-12)


def test_estimator_algebra_on_synthetic_counts():
    counts = {"00": 600, "10": 200, "01": 150, "11": 50}
    estimate = estimate_from_integrated_counts(counts, physical_scale=2.0)
    assert estimate["total_attempts"] == 1000.0
    assert estimate["accepted_attempts"] == 800.0
    assert estimate["acceptance_frequency"] == pytest.approx(0.8)
    assert estimate["measured_postselection_probability"] == pytest.approx(0.6)
    assert estimate["signed_overlap_estimate_z"] == pytest.approx(0.4)
    assert estimate["readout_sign_mean_accepted"] == pytest.approx(0.5)
    assert estimate["recovered_physical_functional"] == pytest.approx(0.8)


def test_required_output_files_created(readout_run):
    output_dir = readout_run["output_dir"]
    for name in [
        "README.md",
        "integrated_readout_summary.csv",
        "integrated_readout_per_seed.csv",
        "integrated_readout_reference_values.json",
        "integrated_readout_circuit_metadata.json",
        "integrated_readout_counts_summary.json",
        "checksums.sha256",
        "manifest.json",
    ]:
        assert (output_dir / name).is_file(), name


def test_output_frames_have_required_columns(readout_run):
    per_seed = readout_run["per_seed"]
    summary = readout_run["summary"]
    for column in PER_SEED_COLUMNS:
        assert column in per_seed.columns, column
    for column in SUMMARY_COLUMNS:
        assert column in summary.columns, column
    n_signed = per_seed["observable_label"].nunique()
    assert len(per_seed) == n_signed * len(_SHOTS) * _NUM_SEEDS
    assert len(summary) == n_signed * len(_SHOTS)


def test_integrated_estimates_converge_toward_ridge(readout_run):
    summary = readout_run["summary"]
    primary = summary[summary["observable_label"] == "state_correction_0"]
    ordered = primary.sort_values("shots")["mean_relative_error_vs_ridge"].to_numpy()
    assert ordered[-1] < ordered[0]
    assert ordered[-1] < 0.2


def test_deterministic_seeds_reproduce(readout_run, tmp_path):
    rerun = run_phase8_integrated_readout(
        {
            "output_dir": str(tmp_path / "repeat"),
            "num_seeds": _NUM_SEEDS,
            "shots_grid": _SHOTS,
        }
    )
    first = readout_run["per_seed"]["recovered_physical_functional"].to_numpy()
    second = rerun["per_seed"]["recovered_physical_functional"].to_numpy()
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_manifest_and_wording_are_claim_safe(readout_run):
    output_dir = readout_run["output_dir"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_state_used_for_preparation"] is False
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    assert "does not imply scalable residual loading" in readme.lower()
