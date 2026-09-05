from __future__ import annotations

import logging
from pathlib import Path


def configure_run_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("robust_qsvt_se.run")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
