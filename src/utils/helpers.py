"""
Utility helper functions for the Multi-Dimensional Return Forecasting system.

This module provides common utility functions used throughout the pipeline,
including configuration loading, directory management, DataFrame I/O,
and timing utilities.

Example:
    >>> from src.utils.helpers import load_config, ensure_directories
    >>> config = load_config()
    >>> ensure_directories(config)
"""

import functools
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

# Type variable for generic function decorator
F = TypeVar("F", bound=Callable)

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load and return configuration from a YAML file.

    Reads the configuration file and returns a dictionary containing
    all configuration parameters for the pipeline.

    Args:
        config_path: Path to the configuration YAML file.
            Defaults to "config/config.yaml".

    Returns:
        dict: Parsed configuration dictionary with all pipeline settings.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> config = load_config()
        >>> print(config['stocks']['tickers'])
        ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', ...]
        >>> print(config['dates']['start'])
        '2020-01-01'
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please ensure config/config.yaml exists."
        )

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def ensure_directories(config: dict) -> None:
    """
    Create all required data, model, and log directories.

    Reads directory paths from the configuration and creates them
    if they don't already exist. This ensures the pipeline has
    all necessary directories before execution.

    Args:
        config: Configuration dictionary containing 'paths' section
            with directory specifications.

    Example:
        >>> config = load_config()
        >>> ensure_directories(config)
        # Creates: data/raw, data/processed, data/features, models, reports, logs

    Note:
        This function is idempotent - it's safe to call multiple times.
        Existing directories are not modified.
    """
    paths_config = config.get("paths", {})

    # Define all directories to create
    directories = [
        paths_config.get("raw_data", "data/raw"),
        paths_config.get("processed_data", "data/processed"),
        paths_config.get("features_data", "data/features"),
        paths_config.get("models_dir", "models"),
        paths_config.get("reports_dir", "reports"),
        paths_config.get("logs_dir", "logs"),
        # Additional subdirectories for raw data
        os.path.join(paths_config.get("raw_data", "data/raw"), "market"),
        os.path.join(paths_config.get("raw_data", "data/raw"), "fundamental"),
        os.path.join(paths_config.get("raw_data", "data/raw"), "macro"),
        os.path.join(paths_config.get("raw_data", "data/raw"), "sentiment"),
        # Reports subdirectory for figures
        os.path.join(paths_config.get("reports_dir", "reports"), "figures"),
    ]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.debug(f"Directory already exists: {directory}")


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    filename: str,
    file_format: str = "parquet",
) -> None:
    """
    Save a DataFrame to disk with error handling.

    Saves the DataFrame to the specified path and filename,
    creating the directory if it doesn't exist.

    Args:
        df: The pandas DataFrame to save.
        path: Directory path where the file will be saved.
        filename: Name of the file (without extension unless specified).
        file_format: Output format - 'parquet' (default) or 'csv'.

    Raises:
        ValueError: If an unsupported file format is specified.
        IOError: If the file cannot be written.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> save_dataframe(df, 'data/processed', 'market_data')
        # Saves to data/processed/market_data.parquet

        >>> save_dataframe(df, 'data/raw', 'prices', file_format='csv')
        # Saves to data/raw/prices.csv
    """
    # Ensure directory exists
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if not filename.endswith(f".{file_format}"):
        filename = f"{filename}.{file_format}"

    file_path = dir_path / filename

    try:
        if file_format == "parquet":
            df.to_parquet(file_path, index=True)
        elif file_format == "csv":
            df.to_csv(file_path, index=True)
        else:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                "Use 'parquet' or 'csv'."
            )

        logger.info(f"Saved DataFrame to {file_path} ({len(df)} rows)")

    except IOError as e:
        logger.error(f"Failed to save DataFrame to {file_path}: {e}")
        raise


def load_dataframe(
    path: str,
    filename: str,
    file_format: str = "parquet",
) -> pd.DataFrame:
    """
    Load a DataFrame from disk.

    Loads a DataFrame from the specified path and filename,
    with automatic format detection based on extension.

    Args:
        path: Directory path where the file is located.
        filename: Name of the file (with or without extension).
        file_format: Expected format - 'parquet' (default) or 'csv'.
            Used only if filename has no extension.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.

    Example:
        >>> df = load_dataframe('data/processed', 'market_data')
        >>> print(df.shape)
        (1000, 10)
    """
    # Add extension if not present
    if "." not in filename:
        filename = f"{filename}.{file_format}"

    file_path = Path(path) / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Detect format from extension
        extension = file_path.suffix.lower()

        if extension == ".parquet":
            df = pd.read_parquet(file_path)
        elif extension == ".csv":
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                "Use '.parquet' or '.csv'."
            )

        logger.info(f"Loaded DataFrame from {file_path} ({len(df)} rows)")
        return df

    except Exception as e:
        logger.error(f"Failed to load DataFrame from {file_path}: {e}")
        raise


def get_trading_days(
    start: str,
    end: str,
    exchange: str = "NSE",
) -> pd.DatetimeIndex:
    """
    Get trading days between start and end dates.

    Returns a DatetimeIndex of trading days for the specified
    exchange, excluding weekends and holidays.

    Args:
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        exchange: Exchange name - 'NSE' (default) for National Stock Exchange.
            Currently falls back to business days if calendar unavailable.

    Returns:
        pd.DatetimeIndex: Index of trading days between start and end.

    Example:
        >>> trading_days = get_trading_days('2024-01-01', '2024-01-31')
        >>> print(len(trading_days))
        22

    Note:
        This function attempts to use exchange-specific calendars via
        pandas_market_calendars. If unavailable, it falls back to
        standard business days (Monday-Friday).
    """
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)

    try:
        # Try to use pandas_market_calendars for accurate trading days
        import pandas_market_calendars as mcal

        # NSE uses 'XNSE' in pandas_market_calendars
        calendar_map = {"NSE": "XNSE", "BSE": "XBOM"}
        calendar_name = calendar_map.get(exchange.upper(), "XNSE")

        calendar = mcal.get_calendar(calendar_name)
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)
        trading_days = pd.DatetimeIndex(schedule.index)

        logger.debug(
            f"Retrieved {len(trading_days)} trading days from {calendar_name}"
        )
        return trading_days

    except ImportError:
        logger.warning(
            "pandas_market_calendars not available, using business days"
        )
    except Exception as e:
        logger.warning(
            f"Could not get {exchange} calendar: {e}. Using business days."
        )

    # Fallback to business days
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    logger.debug(f"Generated {len(trading_days)} business days as fallback")

    return trading_days


def timer_decorator(func: F) -> F:
    """
    Decorator to measure and log function execution time.

    Wraps a function to log its execution time in seconds.
    Useful for profiling pipeline stages and identifying bottlenecks.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that logs execution time.

    Example:
        >>> @timer_decorator
        ... def slow_function():
        ...     time.sleep(2)
        ...     return "done"
        >>> result = slow_function()
        # Logs: "slow_function completed in 2.00 seconds"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time

        logger.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
        return result

    return wrapper  # type: ignore


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets the random seed for Python's random module, NumPy,
    and optionally PyTorch for consistent results.

    Args:
        seed: Random seed value. Defaults to 42.

    Example:
        >>> set_random_seed(123)
        # All random number generators now produce reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"PyTorch random seed set to {seed}")
    except ImportError:
        pass  # PyTorch not available

    logger.info(f"Random seed set to {seed}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame",
) -> bool:
    """
    Validate that a DataFrame has required columns and is not empty.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        name: Name for logging purposes.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.

    Example:
        >>> df = pd.DataFrame({'date': [...], 'close': [...]})
        >>> validate_dataframe(df, ['date', 'close', 'volume'], 'Market Data')
        ValueError: Market Data missing required columns: ['volume']
    """
    if df.empty:
        raise ValueError(f"{name} is empty")

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{name} missing required columns: {sorted(missing_cols)}"
        )

    logger.debug(f"{name} validated: {len(df)} rows, {len(df.columns)} columns")
    return True


def date_to_string(date: Union[pd.Timestamp, str]) -> str:
    """
    Convert a date to string format 'YYYY-MM-DD'.

    Args:
        date: Date as Timestamp or string.

    Returns:
        str: Date in 'YYYY-MM-DD' format.

    Example:
        >>> date_to_string(pd.Timestamp('2024-01-15'))
        '2024-01-15'
    """
    if isinstance(date, str):
        return date
    return date.strftime("%Y-%m-%d")


def string_to_date(date_str: str) -> pd.Timestamp:
    """
    Convert a string to pandas Timestamp.

    Args:
        date_str: Date string in 'YYYY-MM-DD' format.

    Returns:
        pd.Timestamp: Parsed timestamp.

    Example:
        >>> string_to_date('2024-01-15')
        Timestamp('2024-01-15 00:00:00')
    """
    return pd.Timestamp(date_str)
