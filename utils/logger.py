"""
logger.py — Configurable logging utility for UCGA.

Author: Aman Singh
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str = "ucga",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a named logger with console (and optional file) output.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (default ``INFO``).
    log_file : str, optional
        If provided, also log to this file path.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
