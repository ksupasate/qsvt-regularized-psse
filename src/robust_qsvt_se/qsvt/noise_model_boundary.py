"""Depolarizing noise-model boundary for the smallest executed sparse-QSVT circuit.

The identity-verified IEEE-14 4x4, degree-31 dimension anchor (the smallest statevector-executed
circuit: 7 qubits, 60,587 default-target transpiled gates) is reduced to the standard
``rz/sx/x/cx`` basis and simulated on the Aer density-matrix method under a uniform depolarizing
model: one-qubit depolarizing channels with parameter ``p`` on every ``rz``/``sx``/``x`` gate
and a two-qubit depolarizing channel with the same ``p`` on every ``cx``.

For each ``p`` the study records the exact noisy joint readout distribution (no shot noise),
the recovered signed selected output, the postselection acceptance, and genuine sampled Aer
counts at fixed seeds.  The expected outcome at realistic ``p`` is total signal loss through
the ~10^5-gate circuit; that boundary is the finding.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robust_qsvt_se.qsvt.generic_sparse_compiler import CompiledSparseQSVT

NOISE_BASIS = ("rz", "sx", "x", "cx")
ONE_QUBIT_NOISY_GATES = ("rz", "sx", "x")
TWO_QUBIT_NOISY_GATES = ("cx",)
DEPOLARIZING_SWEEP = (0.0, 1.0e-5, 1.0e-4, 1.0e-3)
SHOTS_PER_SEED = 100_000
# Three seeds: each sampled point costs one full ~4.5-minute density-matrix evolution; the
# exact rows (no shot noise) carry the finding and the sampled rows are a consistency layer.
SAMPLING_SEEDS = (0, 1, 2)


def _require_trivial_layout(circuit: Any) -> None:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return
    initial = layout.initial_index_layout(filter_ancillas=True)
    if list(initial) != list(range(len(initial))):
        raise RuntimeError("basis reduction permuted qubits; qubit-indexed readout is invalid")


def reduce_to_noise_basis(circuit: Any) -> Any:
    """Transpile a circuit to the declared rz/sx/x/cx basis with the identity layout."""

    from qiskit import transpile

    reduced = transpile(circuit, basis_gates=list(NOISE_BASIS), optimization_level=0)
    _require_trivial_layout(reduced)
    return reduced


def noise_basis_gate_counts(reduced: Any) -> dict[str, int]:
    counts = {str(k): int(v) for k, v in reduced.count_ops().items()}
    unexpected = sorted(set(counts) - {*NOISE_BASIS, "measure", "barrier"})
    if unexpected:
        raise RuntimeError(f"gates outside the declared noise basis remain: {unexpected}")
    one = sum(counts.get(name, 0) for name in ONE_QUBIT_NOISY_GATES)
    two = sum(counts.get(name, 0) for name in TWO_QUBIT_NOISY_GATES)
    return {
        **counts,
        "noisy_one_qubit_gates": int(one),
        "noisy_two_qubit_gates": int(two),
        "total_noisy_gates": int(one + two),
        "depth": int(reduced.depth()),
    }


def build_depolarizing_model(p: float) -> Any | None:
    """Uniform depolarizing noise model; ``None`` for the ideal reference."""

    if p == 0.0:
        return None
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel(basis_gates=list(NOISE_BASIS))
    model.add_all_qubit_quantum_error(depolarizing_error(float(p), 1),
                                      list(ONE_QUBIT_NOISY_GATES))
    model.add_all_qubit_quantum_error(depolarizing_error(float(p), 2),
                                      list(TWO_QUBIT_NOISY_GATES))
    return model


def _density_matrix_backend(noise_model: Any | None) -> Any:
    from qiskit_aer import AerSimulator

    kwargs = {
        "method": "density_matrix",
        "max_parallel_threads": 1,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
    }
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)


def exact_noisy_joint_distribution(
    reduced_measured: Any,
    noise_model: Any | None,
    *,
    postselection_flag_qubit: int,
    readout_qubit: int,
) -> dict[str, float]:
    """Exact noisy c1c0 probabilities from the final density matrix (no shot noise)."""

    rho = _final_density_matrix(reduced_measured, noise_model)
    probabilities = np.asarray(
        rho.probabilities([postselection_flag_qubit, readout_qubit]), dtype=np.float64
    )
    return {
        "00": float(probabilities[0]),
        "01": float(probabilities[1]),
        "10": float(probabilities[2]),
        "11": float(probabilities[3]),
    }


def _final_density_matrix(reduced_measured: Any, noise_model: Any | None) -> Any:
    """Evolve the measurement-free circuit once and return the exact final density matrix."""

    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer.library import SaveDensityMatrix

    measurement_free = reduced_measured.remove_final_measurements(inplace=False)
    measurement_free.append(
        SaveDensityMatrix(measurement_free.num_qubits, label="density_matrix"),
        list(range(measurement_free.num_qubits)),
    )
    backend = _density_matrix_backend(noise_model)
    result = backend.run(measurement_free, shots=1).result()
    return DensityMatrix(result.data(0)["density_matrix"])


def exact_noisy_marginal_zero(
    reduced_measured: Any, noise_model: Any | None, *, qubit: int
) -> float:
    """Exact noisy probability that one qubit reads 0 (for the direct postselection circuit)."""

    rho = _final_density_matrix(reduced_measured, noise_model)
    probabilities = np.asarray(rho.probabilities([int(qubit)]), dtype=np.float64)
    return float(probabilities[0])


def sampled_noisy_counts(
    reduced_measured: Any, noise_model: Any | None, *, shots: int, seed: int
) -> dict[str, int]:
    """Genuine Aer sampling of the noisy final measured circuit (density-matrix method)."""

    backend = _density_matrix_backend(noise_model)
    result = backend.run(
        reduced_measured, shots=int(shots), seed_simulator=int(seed)
    ).result()
    return {str(k).replace(" ", ""): int(v) for k, v in result.get_counts().items()}


def selected_output_from_distribution(
    distribution: dict[str, float], physical_scale: float
) -> dict[str, float]:
    """Signed selected output and acceptance from a (possibly noisy) c1c0 distribution."""

    plus = float(distribution["00"])
    minus = float(distribution["10"])
    return {
        "selected_output": float(physical_scale) * (plus - minus),
        "interference_acceptance_probability": plus + minus,
        "signed_contrast": plus - minus,
    }


def run_noise_sweep(
    compiled: CompiledSparseQSVT,
    *,
    sweep: tuple[float, ...] = DEPOLARIZING_SWEEP,
    shots_per_seed: int = SHOTS_PER_SEED,
    seeds: tuple[int, ...] = SAMPLING_SEEDS,
    log=print,
) -> dict[str, Any]:
    """Full sweep for the primary functional of one compiled workload."""

    from robust_qsvt_se.qsvt.generic_sparse_execution import validate_compiled_statevector
    from robust_qsvt_se.qsvt.sparse_integrated_chain import estimate_signed_selected_output

    primary = compiled.functional_spec.primary_functional_id
    bundle = compiled.functional_circuits[primary]
    layout = bundle.register_layout
    flag_qubit = int(layout["postselection_flag_qubit"])
    readout_qubit = int(layout["readout_qubit"])
    physical_scale = float(compiled.recovery_factors[primary])

    statevector = validate_compiled_statevector(compiled)
    reference_row = next(
        row for row in statevector.functional_rows if row["functional_id"] == primary
    )
    reference_output = float(reference_row["statevector_selected_output"])
    reference_postselection = float(statevector.metrics["sparse_postselection_probability"])

    reduced_measured = reduce_to_noise_basis(bundle.circuit)
    reduced_direct = reduce_to_noise_basis(compiled.direct_postselection_circuit)
    gate_counts = noise_basis_gate_counts(reduced_measured)
    log(
        f"[noise] {compiled.workload_id}: noise basis {gate_counts['total_noisy_gates']} "
        f"noisy gates ({gate_counts['noisy_one_qubit_gates']} 1q, "
        f"{gate_counts['noisy_two_qubit_gates']} 2q), depth {gate_counts['depth']}"
    )

    exact_rows: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    for p in sweep:
        started = time.perf_counter()
        model = build_depolarizing_model(p)
        distribution = exact_noisy_joint_distribution(
            reduced_measured,
            model,
            postselection_flag_qubit=flag_qubit,
            readout_qubit=readout_qubit,
        )
        derived = selected_output_from_distribution(distribution, physical_scale)
        direct_acceptance = exact_noisy_marginal_zero(
            reduced_direct,
            model,
            qubit=int(compiled.register_allocation["postselection_flag"][0]),
        )
        exact_rows.append(
            {
                "workload_id": compiled.workload_id,
                "functional_id": primary,
                "depolarizing_p": float(p),
                "estimator": "exact_noisy_density_matrix",
                **{f"p_{k}": v for k, v in distribution.items()},
                "selected_output": derived["selected_output"],
                "statevector_reference": reference_output,
                "absolute_error_vs_ideal": abs(
                    derived["selected_output"] - reference_output
                ),
                "relative_error_vs_ideal": abs(
                    derived["selected_output"] - reference_output
                )
                / abs(reference_output),
                "signal_retention_fraction": derived["selected_output"]
                / reference_output,
                "interference_acceptance_probability": derived[
                    "interference_acceptance_probability"
                ],
                "direct_postselection_acceptance": direct_acceptance,
                "ideal_postselection_probability": reference_postselection,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        log(
            f"[noise] p={p:g}: exact y={derived['selected_output']:.3e} "
            f"(ideal {reference_output:.3e}), direct acceptance "
            f"{direct_acceptance:.4f}, {time.perf_counter() - started:.1f}s"
        )
        for seed in seeds:
            counts = sampled_noisy_counts(
                reduced_measured, model, shots=shots_per_seed, seed=seed
            )
            estimate = estimate_signed_selected_output(
                counts, physical_scale=physical_scale
            )
            if seed == seeds[0]:
                # One direct-circuit sampling per noise level (each is a full evolution).
                direct_sampled = sampled_noisy_counts(
                    reduced_direct, model, shots=shots_per_seed, seed=seed
                )
                accepted = int(direct_sampled.get("0", 0))
            else:
                accepted = -1  # not sampled at this seed; see the seeds[0] row
            sampled_rows.append(
                {
                    "workload_id": compiled.workload_id,
                    "functional_id": primary,
                    "depolarizing_p": float(p),
                    "estimator": "sampled_noisy_counts",
                    "shots": int(shots_per_seed),
                    "seed": int(seed),
                    "count_00": int(counts.get("00", 0)),
                    "count_01": int(counts.get("01", 0)),
                    "count_10": int(counts.get("10", 0)),
                    "count_11": int(counts.get("11", 0)),
                    "selected_output": float(estimate["selected_output_estimate"]),
                    "analytic_standard_error": float(estimate["analytic_standard_error"]),
                    "statevector_reference": reference_output,
                    "absolute_error_vs_ideal": abs(
                        float(estimate["selected_output_estimate"]) - reference_output
                    ),
                    "direct_postselection_accepted": accepted,
                    "direct_postselection_acceptance": accepted / float(shots_per_seed),
                }
            )
    return {
        "exact_rows": exact_rows,
        "sampled_rows": sampled_rows,
        "gate_counts": gate_counts,
        "reference": {
            "workload_id": compiled.workload_id,
            "functional_id": primary,
            "statevector_selected_output": reference_output,
            "ideal_postselection_probability": reference_postselection,
            "physical_scale": physical_scale,
            "postselection_flag_qubit": flag_qubit,
            "readout_qubit": readout_qubit,
        },
    }


__all__ = [
    "DEPOLARIZING_SWEEP",
    "NOISE_BASIS",
    "ONE_QUBIT_NOISY_GATES",
    "SAMPLING_SEEDS",
    "SHOTS_PER_SEED",
    "TWO_QUBIT_NOISY_GATES",
    "build_depolarizing_model",
    "exact_noisy_joint_distribution",
    "exact_noisy_marginal_zero",
    "noise_basis_gate_counts",
    "reduce_to_noise_basis",
    "run_noise_sweep",
    "sampled_noisy_counts",
    "selected_output_from_distribution",
]
