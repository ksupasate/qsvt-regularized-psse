"""Reviewer-blocking TQE evidence pass (task-owned, additive).

Produces the minimum new evidence for four strict-reviewer questions, reusing the frozen
reviewer_blocking and cross_case_validation primitives unchanged:

* physical selected-output accuracy vs the true update ``y_true = ell^T Delta_x_true`` (Q1);
* high-degree QSVT feasibility slice at degrees 31/63/127/255 (Q2);
* strong task-aware support-selection baselines (adjoint-unnormalized, exact single-removal) (Q3);
* structure- and case-aware statistical reanalysis (Q4).

Nothing here modifies core solver, QSVT, Ridge, rectangular-convention, manuscript, or any frozen
canonical output. Outputs live only under ``outputs/reviewer_blocking_tqe_evidence/``.
"""

from __future__ import annotations

OUTPUT_ROOT = "outputs/reviewer_blocking_tqe_evidence"
PROTOCOL_ID = "reviewer_blocking_tqe_evidence_v1"
