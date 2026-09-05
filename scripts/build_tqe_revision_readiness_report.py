#!/usr/bin/env python
"""Build Experiment E: reviewer-issue response matrix, claim audit, readiness report."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.tqe_revision_readiness import main

if __name__ == "__main__":
    main(sys.argv[1:])
