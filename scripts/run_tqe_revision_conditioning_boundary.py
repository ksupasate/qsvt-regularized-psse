#!/usr/bin/env python
"""Run Experiment B: conditioning / alpha / degree / phase / postselection boundary."""

from __future__ import annotations

import sys

from robust_qsvt_se.paper.tqe_revision_conditioning_boundary import main

if __name__ == "__main__":
    main(sys.argv[1:])
