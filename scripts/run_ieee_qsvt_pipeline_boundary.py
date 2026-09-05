"""Run the IEEE-derived quantum-pipeline boundary study.

Builds IEEE/PYPOWER-derived weighted Jacobians, extracts deterministic selected
blocks, validates dense block encoding and residual-state preparation at small
scale, checks matched-alpha bounded-QSVT-target/Ridge equivalence, and writes
selected-observable readout and complexity boundary accounting to
``outputs/ieee_qsvt_pipeline_boundary/``. Boundary evidence only: no speedup,
no QSVT-over-Ridge, no full-vector readout, and no quantum-device run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_qsvt_se.paper.ieee_qsvt_pipeline_boundary import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
