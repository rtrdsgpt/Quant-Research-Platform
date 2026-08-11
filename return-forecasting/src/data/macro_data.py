"""
Macroeconomic indicators fetcher for the Multi-Dimensional Return Forecasting system.

This module provides the ``MacroDataFetcher`` class which downloads macro-economic
indicators (USD/INR, crude oil, gold, NIFTY 50, India VIX, bond yields, and
inflation) from Yahoo Finance and synthetic fallback sources, merges them into
a single daily-frequency DataFrame, and persists the result.

Example:
    >>> from src.data.macro_data import MacroDataFetcher
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> fetcher = MacroDataFetcher(config)
    >>> macro_df = fetcher.fetch_all_macro()
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MacroDataFetcher:
    """Fetches macroeconomic indicators from Yahoo Finance and other sources.

    For each indicator the class first attempts a *yfinance* download.  If
    the download fails or returns insufficient data a synthetic fallback with
    realistic ranges is used.  All indicators are aligned to daily frequency
    via forward-fill and combined into a single DataFrame.

    Attributes:
        config: Parsed ``config.yaml`` dictionary.
        start_date: Start of the data window.
        end_date: End of the data window.
        raw_data_path: Directory for persisting the merged parquet file.
        max_retries: Number of download attempts per indicator.
        retry_delay: Initial back-off delay in seconds (doubled per attempt).
    """

    def __init__(self, config: dict) -> None:
        """Initialise the macro fetcher from the master configuration.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config

        dates_cfg = config.get("dates", {})
        self.start_date: str = dates_cfg.get("start", "2020-01-01")
        self.end_date: str = dates_cfg.get("end", "2025-12-31")

        paths_cfg = config.get("paths", {})
        self.raw_data_path: str = paths_cfg.get("raw_data", "data/raw")
        self.raw_dir: str = paths_cfg.get("raw_macro", str(Path(self.raw_data_path) / "macro"))
        self.processed_path: str = paths_cfg.get("processed_data", "data/processed")

        # Macro ticker symbols from config (with sensible defaults)
        macro_cfg = config.get("macro", {})
        self.symbol_usdinr: str = macro_cfg.get("usdinr", "INR=X")
        self.symbol_crude: str = macro_cfg.get("crude_oil", "CL=F")
        self.symbol_gold: str = macro_cfg.get("gold", "GC=F")
        self.symbol_nifty: str = macro_cfg.get("nifty50", "^NSEI")
        self.symbol_vix: str = macro_cfg.get("vix_india", "^INDIAVIX")
        self.symbol_bond: str = macro_cfg.get("india_10y", "^IRX")

        self.max_retries: int = 3
        self.retry_delay: float = 2.0

        logger.info(
            "MacroDataFetcher initialised – range %s to %s",
            self.start_date,
            self.end_date,
        )

    # ------------------------------------------------------------------
    # Individual indicator fetchers
    # ------------------------------------------------------------------

    def fetch_forex(self) -> pd.DataFrame:
        """Fetch USD/INR exchange rate using *yfinance*.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``USD_INR``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_usdinr,
            column_name="USD_INR",
            fallback_range=(73.0, 85.0),
            description="USD/INR exchange rate",
        )

    def fetch_crude_oil(self) -> pd.DataFrame:
        """Fetch Crude Oil futures (WTI) using *yfinance*.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``Crude_Oil``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_crude,
            column_name="Crude_Oil",
            fallback_range=(40.0, 120.0),
            description="Crude Oil futures (CL=F)",
        )

    def fetch_gold(self) -> pd.DataFrame:
        """Fetch Gold futures using *yfinance*.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``Gold``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_gold,
            column_name="Gold",
            fallback_range=(1700.0, 2400.0),
            description="Gold futures (GC=F)",
        )

    def fetch_nifty(self) -> pd.DataFrame:
        """Fetch NIFTY 50 index close using *yfinance*.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``Nifty50``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_nifty,
            column_name="Nifty50",
            fallback_range=(11000.0, 25000.0),
            description="NIFTY 50 index",
        )

    def fetch_vix(self) -> pd.DataFrame:
        """Fetch India VIX using *yfinance*.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``India_VIX``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_vix,
            column_name="India_VIX",
            fallback_range=(10.0, 35.0),
            description="India VIX",
        )

    def fetch_bond_yields(self) -> pd.DataFrame:
        """Fetch India 10-year bond yield proxy.

        Tries *yfinance* first (using the configured proxy symbol).
        Falls back to synthetic data based on realistic Indian
        10-year government bond yields (5.5 % – 7.5 % over 2020-2025).

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column
            ``Bond_Yield_10Y``.
        """
        return self._fetch_yfinance_indicator(
            symbol=self.symbol_bond,
            column_name="Bond_Yield_10Y",
            fallback_range=(5.5, 7.5),
            description="India 10Y bond yield proxy",
        )

    def fetch_inflation_proxy(self) -> pd.DataFrame:
        """Generate a monthly CPI / inflation proxy forward-filled to daily.

        Uses a realistic Indian CPI inflation range of 3 % – 8 % with
        smooth trend and seasonal variation.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` (daily) with column
            ``Inflation``.
        """
        logger.info("Generating synthetic inflation proxy data")
        rng = np.random.RandomState(42)

        # Monthly dates
        monthly_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="MS"
        )
        n_months = len(monthly_dates)

        # Smooth trend + seasonal
        trend = np.linspace(4.5, 5.5, n_months)
        seasonal = 1.2 * np.sin(2 * np.pi * np.arange(n_months) / 12)
        noise = rng.normal(0, 0.3, n_months)

        inflation = trend + seasonal + noise
        inflation = np.clip(inflation, 3.0, 8.0)

        monthly_df = pd.DataFrame(
            {"Inflation": inflation}, index=monthly_dates
        )
        monthly_df.index.name = "Date"

        # Forward-fill to daily
        daily_dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="D"
        )
        daily_df = monthly_df.reindex(daily_dates).ffill()
        daily_df.index.name = "Date"

        logger.info(
            "Generated inflation proxy – %d daily records", len(daily_df)
        )
        return daily_df

    # ------------------------------------------------------------------
    # Combined fetch
    # ------------------------------------------------------------------

    def fetch_all_macro(self) -> pd.DataFrame:
        """Fetch all macro indicators and merge into a single DataFrame.

        The resulting DataFrame is indexed by ``Date`` at daily frequency
        with columns: ``USD_INR``, ``Crude_Oil``, ``Gold``, ``Nifty50``,
        ``India_VIX``, ``Bond_Yield_10Y``, ``Inflation``.

        Missing values are forward-filled, then back-filled at the start.
        The merged data is saved to ``data/raw/macro_indicators.parquet``.

        Returns:
            Merged daily-frequency ``pd.DataFrame`` of all macro indicators.
        """
        indicator_fetchers = [
            ("USD_INR", self.fetch_forex),
            ("Crude_Oil", self.fetch_crude_oil),
            ("Gold", self.fetch_gold),
            ("Nifty50", self.fetch_nifty),
            ("India_VIX", self.fetch_vix),
            ("Bond_Yield_10Y", self.fetch_bond_yields),
            ("Inflation", self.fetch_inflation_proxy),
        ]

        frames = []
        for name, fetcher_fn in tqdm(
            indicator_fetchers, desc="Fetching macro indicators"
        ):
            try:
                df = fetcher_fn()
                if df is not None and not df.empty:
                    frames.append(df)
                    logger.info("Collected %s – %d rows", name, len(df))
                else:
                    logger.warning("No data for %s", name)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", name, exc)

        if not frames:
            logger.error("No macro indicators could be fetched")
            return pd.DataFrame()

        # Merge all on Date index
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.join(df, how="outer")

        # Sort chronologically
        merged = merged.sort_index()

        # Forward-fill then back-fill remaining gaps at the start
        merged = merged.ffill().bfill()

        # Persist
        macro_dir = Path(self.raw_dir)
        macro_dir.mkdir(parents=True, exist_ok=True)
        out_path = macro_dir / "macro_indicators.parquet"
        merged.to_parquet(out_path, index=True)
        logger.info(
            "Saved merged macro indicators → %s (%d rows, %d cols)",
            out_path,
            len(merged),
            len(merged.columns),
        )

        # Save to processed directory too (it's already merged)
        proc_dir = Path(self.processed_path)
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / "macro_processed.parquet"
        merged.to_parquet(proc_path, index=True)
        logger.info("Saved processed macro data → %s", proc_path)

        return merged

    def load_cached(self) -> pd.DataFrame:
        """Load previously fetched macro indicators from ``data/raw/macro/``.

        Reads the ``macro_indicators.parquet`` file.

        Returns:
            Merged daily-frequency ``pd.DataFrame`` of all macro indicators.
            Returns empty DataFrame if file not found.
        """
        macro_dir = Path(self.raw_dir)
        # Try new path first, fall back to old path
        file_path = macro_dir / "macro_indicators.parquet"
        if not file_path.exists():
            file_path = Path(self.raw_data_path) / "macro_indicators.parquet"

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                logger.info(
                    "Loaded cached macro indicators (%d rows, %d cols) from %s",
                    len(df),
                    len(df.columns),
                    file_path,
                )
                return df
            except Exception as exc:
                logger.warning(
                    "Failed to load cached macro indicators: %s", exc
                )
                return pd.DataFrame()
        else:
            logger.warning("No cached macro indicators found at %s", file_path)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_yfinance_indicator(
        self,
        symbol: str,
        column_name: str,
        fallback_range: tuple,
        description: str,
    ) -> pd.DataFrame:
        """Download a single indicator from *yfinance* with retry logic.

        If the download fails after all retries, synthetic data is generated
        using the provided ``fallback_range``.

        Args:
            symbol: Yahoo Finance symbol, e.g. ``'INR=X'``.
            column_name: Desired output column name.
            fallback_range: ``(low, high)`` tuple for synthetic fallback.
            description: Human-readable label for logging.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with a single column
            ``column_name``.
        """
        delay = self.retry_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching %s via %s (attempt %d/%d)",
                    description,
                    symbol,
                    attempt,
                    self.max_retries,
                )
                raw_df: pd.DataFrame = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )

                if raw_df.empty:
                    raise ValueError(
                        f"Empty DataFrame returned for {symbol}"
                    )

                # Flatten MultiIndex columns if present
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)

                # Use Close (or Adj Close) as the indicator value
                if "Close" in raw_df.columns:
                    series = raw_df["Close"]
                elif "Adj Close" in raw_df.columns:
                    series = raw_df["Adj Close"]
                else:
                    raise ValueError(
                        f"No Close column found for {symbol}"
                    )

                df = pd.DataFrame({column_name: series})
                df.index.name = "Date"

                logger.info(
                    "Successfully fetched %s – %d rows", description, len(df)
                )
                return df

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d for %s failed: %s",
                    attempt,
                    self.max_retries,
                    description,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2

        # Fallback to synthetic data
        logger.warning(
            "All attempts failed for %s – generating synthetic fallback "
            "(last error: %s)",
            description,
            last_exc,
        )
        return self._generate_synthetic_indicator(
            column_name, fallback_range
        )

    def _generate_synthetic_indicator(
        self,
        column_name: str,
        value_range: tuple,
    ) -> pd.DataFrame:
        """Generate synthetic daily indicator data.

        Produces a smooth random walk within the specified range.

        Args:
            column_name: Output column name.
            value_range: ``(low, high)`` bounds for the indicator.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with column ``column_name``.
        """
        rng = np.random.RandomState(abs(hash(column_name)) % (2**31))

        daily_dates = pd.bdate_range(
            start=self.start_date, end=self.end_date
        )
        n_days = len(daily_dates)
        lo, hi = value_range
        mid = (lo + hi) / 2.0
        amplitude = (hi - lo) / 2.0

        # Random walk with drift
        steps = rng.normal(0, amplitude * 0.01, n_days)
        values = mid + np.cumsum(steps)

        # Add gentle trend
        trend = np.linspace(-amplitude * 0.3, amplitude * 0.3, n_days)
        values = values + trend

        # Clip to range
        values = np.clip(values, lo, hi)

        df = pd.DataFrame({column_name: values}, index=daily_dates)
        df.index.name = "Date"

        logger.info(
            "Generated synthetic %s – %d business days", column_name, n_days
        )
        return df
