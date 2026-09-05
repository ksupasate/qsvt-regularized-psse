"""Reviewer-blocking TQE experiments (additive, evidence-status-tagged).

This subpackage implements four reviewer-blocking extensions on top of the frozen
8x8 IEEE-14-derived output-aware QSVT workload without modifying any existing solver,
Ridge/Tikhonov target, QSVT phase convention, frozen residual bank, support registry,
or manuscript artifact:

1. physically meaningful selected-output functionals (:mod:`physical_functionals`);
2. strong exact-loss support-selection baselines (:mod:`exact_loss_baselines`);
3. joint application-utility and QSVT-feasibility experiments (:mod:`joint_feasibility`);
4. quantum resource-accuracy Pareto analysis (:mod:`resource_pareto`).

No result here claims quantum speedup, quantum advantage, QSVT-over-Ridge numerical
superiority, field PMU/SCADA validation, IEEE-scale hardware execution, or practical
competitiveness against access-matched classical solvers.
"""

from __future__ import annotations

STUDY_NAMESPACE = "tqe_reviewer_blocking_experiments"
