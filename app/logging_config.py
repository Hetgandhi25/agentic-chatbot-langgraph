"""
Centralized logging configuration.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with consistent formatting."""
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Obtain a logger instance for a given module."""
    return logging.getLogger(name)

