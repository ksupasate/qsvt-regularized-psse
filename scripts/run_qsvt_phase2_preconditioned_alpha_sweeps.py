from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_qsvt_se.qsvt.phase2_preconditioned_alpha import (  # noqa: E402
    run_phase2_preconditioned_alpha_sweeps,
)

if __name__ == "__main__":
    run = run_phase2_preconditioned_alpha_sweeps()
    print(f"QSVT Phase 2 preconditioned alpha sweeps complete: {run['output_dir']}")
