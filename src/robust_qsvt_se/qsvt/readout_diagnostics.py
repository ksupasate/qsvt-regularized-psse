"""Shot-budget readout diagnostics for selected observables.

Two unbiased Monte-Carlo readout models are provided, matched to the two
observable classes in :mod:`robust_qsvt_se.qsvt.selected_observables`:

* **sign-aware** (``y_l = l^T dx``): a Hadamard-test-style estimator. With the
  normalized inner product ``mu = <l_hat | dx_hat>`` the ancilla measures |0>
  with probability ``p = (1 + mu)/2``; the estimate ``||l|| ||dx|| (2 p_hat - 1)``
  is unbiased. Reused from the existing signed-readout diagnostic.
* **basis-sampling** (``||Pi_A dx||^2``): computational-basis sampling of the
  normalized output state estimates the in-subspace probability
  ``p_A = ||Pi_A dx||^2 / ||dx||^2``; the estimate ``||dx||^2 (successes/shots)``
  is unbiased.

Both estimators have absolute error scaling on the order of ``1/sqrt(shots)``.
Each call estimates **one** selected functional, not the full update vector. This
is a measurement-model diagnostic, not a synthesized circuit and not a
quantum-hardware run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.paper.signed_readout_diagnostic import hadamard_test_estimate
from robust_qsvt_se.qsvt.selected_observables import (
    BASIS_SAMPLING_MODEL,
    SIGN_AWARE_MODEL,
    SelectedObservable,
)
from robust_qsvt_se.utils.seed import make_rng

DEFAULT_SHOTS = (100, 1_000, 10_000, 100_000)
DEFAULT_TRIALS = 200
_RELATIVE_ERROR_FLOOR = 1.0e-12


def basis_sampling_energy_estimate(
    support: np.ndarray | list[int],
    update: np.ndarray,
    *,
    shots: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return ``(true_value, estimated_value)`` for ``||Pi_A dx||^2`` by basis sampling.

    The normalized output state has amplitudes ``dx / ||dx||``; the probability of
    measuring inside ``A`` is ``p_A = ||Pi_A dx||^2 / ||dx||^2``. The estimate
    ``||dx||^2 (successes/shots)`` with ``successes ~ Binomial(shots, p_A)`` is
    unbiased for the energy observable.
    """

    dx = np.asarray(update, dtype=np.float64)
    idx = list(support)
    dx_norm_sq = float(dx @ dx)
    true_value = float(np.sum(dx[idx] ** 2))
    if dx_norm_sq <= 0.0:
        return true_value, 0.0
    p_a = float(np.clip(true_value / dx_norm_sq, 0.0, 1.0))
    successes = int(rng.binomial(int(shots), p_a))
    estimated = dx_norm_sq * (successes / int(shots))
    return true_value, float(estimated)


def _trial_seed(base_seed: int, shot_index: int, trial: int) -> int:
    return int(base_seed) * 1_000_000 + int(shot_index) * 10_000 + int(trial)


def observable_shot_sweep(
    observable: SelectedObservable,
    update: np.ndarray,
    *,
    shots_grid: tuple[int, ...] = DEFAULT_SHOTS,
    trials: int = DEFAULT_TRIALS,
    base_seed: int = 20240601,
) -> list[dict[str, Any]]:
    """Run a shot sweep for ``observable`` and return per-shot-budget error rows.

    For each shot budget, ``trials`` independent deterministic-seed estimates are
    averaged into mean/max/std absolute error and a relative error. Works for both
    the sign-aware and basis-sampling readout models.
    """

    dx = np.asarray(update, dtype=np.float64)
    exact = observable.exact_value(dx)
    rows: list[dict[str, Any]] = []
    for shot_index, shots in enumerate(shots_grid):
        errors = np.empty(int(trials), dtype=np.float64)
        for trial in range(int(trials)):
            rng = make_rng(_trial_seed(base_seed, shot_index, trial))
            if observable.readout_model == SIGN_AWARE_MODEL:
                _true, estimated, _mu = hadamard_test_estimate(
                    observable.vector, dx, shots=int(shots), rng=rng
                )
            elif observable.readout_model == BASIS_SAMPLING_MODEL:
                _true, estimated = basis_sampling_energy_estimate(
                    np.asarray(observable.support, dtype=np.int64),
                    dx,
                    shots=int(shots),
                    rng=rng,
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown readout model {observable.readout_model!r}")
            errors[trial] = abs(estimated - exact)
        mean_abs = float(errors.mean())
        relative = mean_abs / abs(exact) if abs(exact) > _RELATIVE_ERROR_FLOOR else float("nan")
        rows.append(
            {
                "observable_id": observable.observable_id,
                "shots": int(shots),
                "trials": int(trials),
                "mean_abs_error": mean_abs,
                "max_abs_error": float(errors.max()),
                "std_abs_error": float(errors.std(ddof=0)),
                "relative_error": relative,
                "readout_model": observable.readout_model,
                "status": observable.status,
                "exact_value": float(exact),
            }
        )
    return rows


def shots_for_relative_error(
    sweep_rows: list[dict[str, Any]], *, target_relative_error: float
) -> int | None:
    """Smallest shot budget in ``sweep_rows`` meeting ``target_relative_error``.

    Returns ``None`` when no budget meets the target or when the relative error is
    undefined (exact value ~0).
    """

    target = float(target_relative_error)
    for row in sorted(sweep_rows, key=lambda r: int(r["shots"])):
        rel = row["relative_error"]
        if np.isfinite(rel) and rel <= target:
            return int(row["shots"])
    return None
