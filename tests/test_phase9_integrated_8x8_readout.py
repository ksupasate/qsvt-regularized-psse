"""Tests for the Phase 9 integrated 8x8 finite-shot QSVT chain."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import robust_qsvt_se.paper.phase9_integrated_8x8_readout as phase9
from robust_qsvt_se.paper.phase9_integrated_8x8_readout import (
    build_anchor_context_8x8,
    run_phase9_integrated_8x8_readout,
)
from robust_qsvt_se.paper.tqe_revision_experiments_common import forbidden_in

pytest.importorskip("pennylane")
pytest.importorskip("qiskit")

_NUM_SEEDS = 3
_SHOTS = [500, 8000]


@pytest.fixture(scope="module")
def readout_run(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("phase9_integrated_8x8_readout")
    cache_dir = tmp_path_factory.mktemp("phase9_8x8_phase_cache")
    return run_phase9_integrated_8x8_readout(
        {
            "output_dir": str(output_dir),
            "num_seeds": _NUM_SEEDS,
            "shots_grid": _SHOTS,
            "phase_cache_dir": str(cache_dir),
        }
    )


def test_anchor_is_the_lambda_matched_8x8_pass(readout_run):
    references = readout_run["references"]
    assert references["block_shape"] == "8x8"
    # lambda matched to the 4x4 anchor (~0.069) and a genuine statevector pass.
    assert references["lambda_normalized"] == pytest.approx(0.069018, abs=1.0e-4)
    assert references["condition_number"] == pytest.approx(132.84, abs=0.5)
    assert references["statevector_update_relative_error_vs_ridge"] < 0.05
    assert 0.90 < references["statevector_postselection_probability"] < 1.0


def test_no_direct_output_state_preparation_in_source():
    source = Path(phase9.__file__).read_text(encoding="utf-8")
    assert "StatePreparation(output_state" not in source
    anchor_builder = inspect.getsource(build_anchor_context_8x8)
    # The anchor builder may reference the demo output_state only as a validation
    # reference; it must never feed it into a state-preparation of a sampled circuit.
    assert "StatePreparation" not in anchor_builder
    run_source = inspect.getsource(phase9.run_phase9_integrated_8x8_readout)
    assert "StatePreparation" not in run_source


def test_no_output_state_preparation_flagged_in_rows(readout_run):
    per_seed = readout_run["per_seed"]
    assert not per_seed["output_state_used_for_preparation"].any()
    assert readout_run["circuit_metadata"]["output_state_used_for_preparation"] is False


def test_integrated_circuit_is_five_qubits(readout_run):
    per_seed = readout_run["per_seed"]
    assert int(per_seed["circuit_qubits"].iloc[0]) == 5


def test_signal_unitary_call_count_is_degree(readout_run):
    metadata = readout_run["circuit_metadata"]
    degree = int(metadata["degree"])
    assert degree == 31
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


def test_measured_postselection_matches_statevector_at_high_shots(readout_run):
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


def test_integrated_estimates_converge_toward_ridge_like_sqrt_n(readout_run):
    summary = readout_run["summary"]
    primary = summary[summary["observable_label"] == "state_correction_0"].sort_values("shots")
    ordered = primary["mean_relative_error_vs_ridge"].to_numpy(dtype=np.float64)
    # Error must fall as shots grow (finite-shot N^-1/2-like behavior).
    assert ordered[-1] < ordered[0]
    assert ordered[-1] < 0.2


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


def test_deterministic_seeds_reproduce(readout_run, tmp_path):
    rerun = run_phase9_integrated_8x8_readout(
        {
            "output_dir": str(tmp_path / "repeat"),
            "num_seeds": _NUM_SEEDS,
            "shots_grid": _SHOTS,
            "phase_cache_dir": str(tmp_path / "cache"),
        }
    )
    first = readout_run["per_seed"]["recovered_physical_functional"].to_numpy()
    second = rerun["per_seed"]["recovered_physical_functional"].to_numpy()
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_deterministic_block_checksums_stable(readout_run):
    references = readout_run["references"]
    # The deterministic IEEE-14-derived selection is fixed by (case, seed) only.
    assert references["block_checksum"]
    assert references["residual_checksum"]
    assert references["selected_cols"].split()[0:2] == ["0", "2"]


def test_manifest_and_wording_are_claim_safe(readout_run):
    output_dir = readout_run["output_dir"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_state_used_for_preparation"] is False
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert forbidden_in(readme) == []
    assert "does not imply scalable residual loading" in readme.lower()
