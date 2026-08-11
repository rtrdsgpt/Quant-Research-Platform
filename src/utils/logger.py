"""
Logging configuration module for the Multi-Dimensional Return Forecasting system.

This module provides a centralized logging setup that creates both file and
console handlers with configurable log levels and formatting.

Example:
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger(__name__)
    >>> logger.info("Pipeline started")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml


def setup_logger(
    name: str,
    config_path: Optional[str] = "config/config.yaml",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.

    Creates a logger instance with both a RotatingFileHandler for persistent
    logging and a StreamHandler for console output. The logger configuration
    is read from the config.yaml file, with optional overrides.

    Args:
        name: Logger name, typically __name__ of the calling module.
            This creates a hierarchical logger structure.
        config_path: Path to the configuration YAML file.
            Defaults to "config/config.yaml".
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            If None, uses the level from config.yaml.
        log_file: Override log file path.
            If None, uses the path from config.yaml.

    Returns:
        logging.Logger: Configured logger instance with:
            - RotatingFileHandler (default: logs/pipeline.log)
            - StreamHandler (console output)

    Raises:
        FileNotFoundError: If config_path doesn't exist and no overrides provided.
        yaml.YAMLError: If config file is malformed.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting data collection")
        >>> logger.warning("Missing values detected")
        >>> logger.error("Failed to fetch data", exc_info=True)

    Note:
        The logs directory is created automatically if it doesn't exist.
        Log files are rotated when they reach 10MB, with up to 5 backup files.
    """
    # Create logger instance
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Load configuration
    config = _load_logging_config(config_path)

    # Apply overrides if provided
    level_str = log_level or config.get("level", "INFO")
    log_file_path = log_file or config.get("file", "logs/pipeline.log")
    log_format = config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Convert level string to logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup file handler with rotation
    file_handler = _create_file_handler(log_file_path, formatter, config)
    logger.addHandler(file_handler)

    # Setup console handler
    console_handler = _create_console_handler(formatter, level)
    logger.addHandler(console_handler)

    return logger


def _load_logging_config(config_path: Optional[str]) -> dict:
    """
    Load logging configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        dict: Logging configuration dictionary with default fallbacks.
    """
    default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/pipeline.log",
        "rotate_mb": 10,
        "backup_count": 5,
    }

    if config_path is None:
        return default_config

    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return default_config

        with open(config_file, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        logging_config = full_config.get("logging", {})
        # Merge with defaults
        return {**default_config, **logging_config}

    except (yaml.YAMLError, IOError) as e:
        # Log to stderr since logger isn't set up yet
        print(f"Warning: Could not load logging config from {config_path}: {e}")
        return default_config


def _create_file_handler(
    log_file_path: str,
    formatter: logging.Formatter,
    config: dict,
) -> RotatingFileHandler:
    """
    Create a rotating file handler for persistent logging.

    Args:
        log_file_path: Path to the log file.
        formatter: Logging formatter instance.
        config: Logging configuration dictionary.

    Returns:
        RotatingFileHandler: Configured file handler with rotation.
    """
    # Ensure log directory exists
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure rotation parameters
    max_bytes = config.get("rotate_mb", 10) * 1024 * 1024  # Convert MB to bytes
    backup_count = config.get("backup_count", 5)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    return file_handler


def _create_console_handler(
    formatter: logging.Formatter,
    level: int,
) -> logging.StreamHandler:
    """
    Create a console handler for stdout logging.

    Args:
        formatter: Logging formatter instance.
        level: Logging level.

    Returns:
        StreamHandler: Configured console handler.
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    This is a convenience function for modules that need a logger
    without explicitly setting up configuration.

    Args:
        name: Logger name, typically __name__.

    Returns:
        logging.Logger: Logger instance.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing complete")
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger
