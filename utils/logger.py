"""Logging utilities"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_file: str = "experiment.log"
) -> logger:
    """
    Setup loguru logger with console and file handlers

    Args:
        log_dir: Directory for log files
        level: Logging level
        log_file: Log file name

    Returns:
        Configured logger
    """
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if log_dir specified
    if log_dir:
        log_path = Path(log_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="500 MB",
            retention="10 days",
            compression="zip"
        )

    return logger
