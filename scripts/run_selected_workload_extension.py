"""Goal A entry point: larger selected-output QSVT workload attempts (8x8)."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.selected_workload_extension import main

if __name__ == "__main__":
    main(sys.argv[1:])
