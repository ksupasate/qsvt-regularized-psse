from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_qsvt_se.qsvt.phase2_preconditioned_alpha import (  # noqa: E402
    build_phase2_alpha_selection_report,
)

if __name__ == "__main__":
    run = build_phase2_alpha_selection_report()
    print(f"QSVT Phase 2 alpha-selection report complete: {run['output_dir']}")
