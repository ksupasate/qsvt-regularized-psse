"""Cross-case and larger-block validation of the output-aware QSVT-oriented PSSE protocol.

Additive, evidence-status-tagged transfer study built entirely on top of the frozen
reviewer-blocking implementation (:mod:`robust_qsvt_se.reviewer_blocking`) and the protected
output-aware selection core (:mod:`robust_qsvt_se.qsvt.output_aware_sparse_selection`).  It
generalizes exactly three hardcoded IEEE-14 ``8x8`` couplings — case name, the physical-functional
score set, and frozen-design acquisition — and reuses every scientific primitive unchanged.

No result here claims broad cross-system generalization, scalable quantum advantage, IEEE-scale
hardware execution, measured speedup, or QSVT-over-Ridge numerical superiority.  Evidence is a
controlled transfer test on one additional IEEE-30-derived structure and one larger ``16x16``
IEEE-14 block.
"""

from __future__ import annotations

STUDY_NAMESPACE = "cross_case_larger_block_validation"
