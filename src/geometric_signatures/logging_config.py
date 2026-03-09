"""Structured logging configuration for the geometric-signatures pipeline.

Provides a single ``setup_logging`` function that configures the root logger
with a consistent format, optional file output, and level control. All pipeline
modules should use ``logging.getLogger(__name__)`` to inherit this configuration.

Usage::

    from geometric_signatures.logging_config import setup_logging
    logger = setup_logging(level="INFO", log_dir=Path("runs/logs"))
    logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Default format: timestamp, level, module, message
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_filename: str = "geometric_signatures.log",
) -> logging.Logger:
    """Configure the package-level logger.

    Args:
        level: Logging level string (e.g., "DEBUG", "INFO", "WARNING").
        log_dir: If provided, also writes logs to a file in this directory.
        log_filename: Name of the log file (only used if log_dir is provided).

    Returns:
        The configured logger for the ``geometric_signatures`` package.
    """
    logger = logging.getLogger("geometric_signatures")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler → stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
