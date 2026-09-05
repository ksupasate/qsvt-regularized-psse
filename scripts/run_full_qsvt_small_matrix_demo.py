from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _main() -> None:
    from robust_qsvt_se.qsvt.full_matrix_qsvt_demo import main

    main()


if __name__ == "__main__":
    _main()
