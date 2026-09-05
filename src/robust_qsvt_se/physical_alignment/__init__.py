"""Additive physical-accuracy alignment and structural-generalization evidence.

This namespace is deliberately separate from the frozen selector, Ridge, QSVT,
measurement, and historical evidence implementations.  It reads the frozen
structural registry but writes only to the task-owned evidence root.
"""

from robust_qsvt_se.physical_alignment.config import (
    DEFAULT_CONFIG_PATH,
    configuration_hash,
    load_campaign_config,
)

__all__ = ["DEFAULT_CONFIG_PATH", "configuration_hash", "load_campaign_config"]
