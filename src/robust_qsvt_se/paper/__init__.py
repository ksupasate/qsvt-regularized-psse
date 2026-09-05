"""Paper-level manuscript assembly package.

Audits and consolidates existing classical, measurement-model, nonlinear-AC, and
QSVT evidence into manuscript-ready tables, figure-ready data, claim-support
matrices, and reproducibility documentation. This package does not run new
experiments and does not change solver behavior; it only reads artifacts that
already exist on disk and never fabricates results.
"""

from __future__ import annotations

__all__ = ["PAPER_CLAIM_BOUNDARY"]

PAPER_CLAIM_BOUNDARY = (
    "Manuscript assembly package auditing and consolidating existing evidence for a "
    "controlled IEEE benchmark and QSVT feasibility/prototype study of regularized "
    "spectral filtering in ill-conditioned power-system state estimation. IEEE/PYPOWER "
    "cases provide benchmark network models and generated measurement rows; QSVT is a "
    "QSVT-compatible implementation pathway for the same regularized spectral filter "
    "with Ridge/Tikhonov as the reference. No full IEEE-scale quantum state estimator, "
    "QSVT-over-Ridge superiority, quantum speedup, quantum advantage, real PMU/SCADA "
    "validation, field data, deployment-ready quantum solver, full nonlinear AC QSVT "
    "loop, or hardware execution is claimed."
)
