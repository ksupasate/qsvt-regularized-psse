from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_qsvt_se.qsvt.paper_finalization import main_claim_audit  # noqa: E402,I001


if __name__ == "__main__":
    main_claim_audit()
