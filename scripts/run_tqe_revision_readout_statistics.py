#!/usr/bin/env python
"""Run Experiment A: seed-resolved readout statistics for the 4x4 QSVT demo."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.tqe_revision_readout_statistics import main

if __name__ == "__main__":
    main(sys.argv[1:])
