from __future__ import annotations

import math

import numpy as np

from robust_qsvt_se.estimators.base import EstimatorResult, timed_solve
from robust_qsvt_se.measurement.linear_system import WeightedSystem
from robust_qsvt_se.qsvt.filters import qsvt_unregularized_inverse_filter


class HHLStyleInverseProxyEstimator:
    name = "hhl_style_inverse_proxy"

    def __init__(
        self,
        *,
        cutoff: float = 1.0e-8,
        precision: float = 1.0e-3,
        instability_condition_threshold: float = 1.0e8,
        fail_on_instability: bool = False,
    ) -> None:
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if precision <= 0.0:
            raise ValueError("precision must be positive")
        if instability_condition_threshold <= 0.0:
            raise ValueError("instability_condition_threshold must be positive")
        self.cutoff = float(cutoff)
        self.precision = float(precision)
        self.instability_condition_threshold = float(instability_condition_threshold)
        self.fail_on_instability = bool(fail_on_instability)

    def solve(self, system: WeightedSystem) -> EstimatorResult:
        def _solve() -> tuple[np.ndarray, dict[str, bool | float | int | str]]:
            U, singular_values, Vt = np.linalg.svd(system.H_tilde, full_matrices=False)
            positive = singular_values[singular_values > 0.0]
            sigma_max = float(np.max(positive)) if positive.size else 0.0
            sigma_min_effective = max(
                float(np.min(positive)) if positive.size else self.cutoff,
                self.cutoff,
            )
            effective_condition = math.inf if sigma_max == 0.0 else sigma_max / sigma_min_effective
            below_cutoff = int(np.count_nonzero(singular_values < self.cutoff))
            instability = (
                bool(below_cutoff)
                or not np.isfinite(effective_condition)
                or effective_condition > self.instability_condition_threshold
            )
            if instability and self.fail_on_instability:
                raise np.linalg.LinAlgError(
                    "HHL-style inverse proxy marked unstable at effective condition number "
                    f"{effective_condition:.3e}"
                )

            filter_values = qsvt_unregularized_inverse_filter(
                singular_values,
                cutoff=self.cutoff,
            )
            x_hat = Vt.T @ (filter_values * (U.T @ system.r_tilde))
            hhl_resource_proxy = _hhl_resource_proxy(
                effective_condition=effective_condition,
                precision=self.precision,
                n_states=system.n_states,
            )
            return x_hat, {
                "cutoff": self.cutoff,
                "precision": self.precision,
                "hhl_effective_condition_number": effective_condition,
                "hhl_resource_proxy": hhl_resource_proxy,
                "hhl_instability_flag": instability,
                "singular_values_below_cutoff": below_cutoff,
                "max_filter_value": float(np.max(filter_values)) if filter_values.size else 0.0,
                "simulation_scope": "diagnostic HHL-style inverse proxy; no circuit execution",
                "output_readout_caveat": (
                    "Proxy reports a classical state-vector estimate and does not model "
                    "quantum state preparation or output tomography/readout costs."
                ),
            }

        result = timed_solve(self.name, system, _solve)
        if result.failed and "hhl_instability_flag" not in result.extra_diagnostics:
            result.extra_diagnostics.update(
                {
                    "cutoff": self.cutoff,
                    "precision": self.precision,
                    "hhl_instability_flag": True,
                    "simulation_scope": "diagnostic HHL-style inverse proxy; no circuit execution",
                }
            )
        return result


def _hhl_resource_proxy(*, effective_condition: float, precision: float, n_states: int) -> float:
    if not np.isfinite(effective_condition):
        return math.inf
    dimension_factor = math.log2(max(2, n_states))
    return float((effective_condition**2) * dimension_factor / precision)
