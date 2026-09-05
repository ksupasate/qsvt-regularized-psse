"""Noise-model boundary - model construction, estimator arithmetic, and sweep artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robust_qsvt_se.qsvt.noise_model_boundary import (
    DEPOLARIZING_SWEEP,
    NOISE_BASIS,
    build_depolarizing_model,
    noise_basis_gate_counts,
    reduce_to_noise_basis,
    selected_output_from_distribution,
)

ROOT = Path(__file__).resolve().parents[1]
EXACT_PATH = ROOT / "outputs/noise_model_boundary/exact_noise_rows.csv"
SAMPLED_PATH = ROOT / "outputs/noise_model_boundary/sampled_noise_rows.csv"


def test_depolarizing_model_construction():
    assert build_depolarizing_model(0.0) is None
    model = build_depolarizing_model(1.0e-4)
    noisy = set(model.noise_instructions)
    assert {"rz", "sx", "x", "cx"} <= noisy


def test_selected_output_arithmetic():
    distribution = {"00": 0.30, "01": 0.25, "10": 0.10, "11": 0.35}
    result = selected_output_from_distribution(distribution, physical_scale=2.0)
    assert result["signed_contrast"] == pytest.approx(0.20)
    assert result["selected_output"] == pytest.approx(0.40)
    assert result["interference_acceptance_probability"] == pytest.approx(0.40)


def test_reduction_to_noise_basis_counts_only_declared_gates():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.crz(0.3, 0, 1)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    reduced = reduce_to_noise_basis(circuit)
    counts = noise_basis_gate_counts(reduced)
    assert counts["total_noisy_gates"] == (
        counts["noisy_one_qubit_gates"] + counts["noisy_two_qubit_gates"]
    )
    named = {k for k in counts if k in NOISE_BASIS}
    assert named <= set(NOISE_BASIS)
    assert counts["noisy_two_qubit_gates"] >= 2  # crz decomposes into >= 2 cx


@pytest.mark.skipif(not EXACT_PATH.is_file(), reason="noise sweep not generated")
def test_exact_rows_show_monotone_signal_loss():
    exact = pd.read_csv(EXACT_PATH).sort_values("depolarizing_p")
    assert list(exact["depolarizing_p"]) == sorted(DEPOLARIZING_SWEEP)
    ideal = exact[exact["depolarizing_p"] == 0.0].iloc[0]
    assert ideal["absolute_error_vs_ideal"] < 1.0e-9
    assert ideal["signal_retention_fraction"] == pytest.approx(1.0, abs=1e-6)
    noisy = exact[exact["depolarizing_p"] > 0.0]
    retention = noisy["signal_retention_fraction"].to_numpy()
    assert (np.diff(retention) <= 1e-12).all()  # retention never increases with p
    heavy = noisy.iloc[-1]
    assert heavy["depolarizing_p"] == pytest.approx(1.0e-3)
    # Fully mixed work register: acceptance collapses to 2^-3 and the signal to ~0.
    assert heavy["direct_postselection_acceptance"] == pytest.approx(0.125, abs=0.01)
    assert abs(heavy["signal_retention_fraction"]) < 1.0e-3


@pytest.mark.skipif(not SAMPLED_PATH.is_file(), reason="noise sweep not generated")
def test_sampled_rows_are_seed_complete_and_consistent():
    sampled = pd.read_csv(SAMPLED_PATH)
    for p in DEPOLARIZING_SWEEP:
        group = sampled[sampled["depolarizing_p"] == p]
        assert set(group["seed"]) == {0, 1, 2}
        assert (group["shots"] == 100_000).all()
        total = group[["count_00", "count_01", "count_10", "count_11"]].sum(axis=1)
        assert (total == 100_000).all()
