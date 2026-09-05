from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_qsvt_se.qsvt.phase1_finalization import finalize_qsvt_phase1_artifacts  # noqa: E402

if __name__ == "__main__":
    run = finalize_qsvt_phase1_artifacts()
    print(f"QSVT Phase 1 finalization complete: {run['output_dir']}")
