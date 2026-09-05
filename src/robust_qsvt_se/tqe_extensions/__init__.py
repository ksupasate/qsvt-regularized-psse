"""Reviewer-blocking TQE evidence extensions (Workstreams A, B, C).

Additive, isolated extensions requested by a strict TQE-style review:

* Workstream A - :mod:`.degree_lambda_scaling`: systematic degree / normalized-regularization /
  target-tolerance feasibility map for the bounded QSVT spectral filter.
* Workstream B - :mod:`.nonlinear_circuit_loop`: small nonlinear AC QSVT circuit-in-the-loop
  executed in classical statevector simulation.
* Workstream C - :mod:`.row_sparsification`: physically implementable measurement-row selection
  compared against arbitrary weighted-Jacobian entry selection.

Every module reuses the frozen QSVT / Ridge / PSSE / support-selection primitives by import and
adds only bookkeeping, scope wiring, and reporting.  No estimator or QSVT definition is changed.
No claim of quantum speedup, quantum advantage, or hardware execution is made; IEEE/PYPOWER cases
provide benchmark network models, not field PMU/SCADA data.
"""

from __future__ import annotations

CLAIM_BOUNDARY = (
    "Controlled IEEE/PYPOWER benchmark and QSVT feasibility study for regularized spectral "
    "filtering in ill-conditioned power-system state estimation. IEEE/PYPOWER cases provide "
    "benchmark network models; measurement rows are generated from the network equations (not "
    "field PMU/SCADA data). The QSVT component is a QSVT-compatible implementation pathway for the "
    "same Ridge/Tikhonov regularized spectral filter evaluated at the same regularization. This "
    "work makes no claim of quantum speedup, quantum advantage, QSVT-over-Ridge numerical "
    "superiority, full-vector quantum readout, physically optimal support, or IEEE-scale hardware "
    "execution."
)
