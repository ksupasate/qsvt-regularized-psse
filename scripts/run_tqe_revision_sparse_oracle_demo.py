#!/usr/bin/env python
"""Run Experiment D: compiled sparse-access oracle demonstration."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.tqe_revision_sparse_oracle_demo import main

if __name__ == "__main__":
    main(sys.argv[1:])
