"""
Logging utility for Ares Optimizer.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Logging level
        format: Log message format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir is provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger 