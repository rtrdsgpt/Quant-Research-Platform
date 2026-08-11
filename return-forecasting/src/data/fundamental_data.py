"""
Fundamental data provider for the Multi-Dimensional Return Forecasting system.

This module provides the ``FundamentalDataFetcher`` class which retrieves
quarterly fundamental metrics (PE Ratio, Debt-to-Equity, ROE, EPS, etc.)
for every stock in the universe.  It first tries *yfinance* and falls back
to realistic synthetic data generation when the API does not supply
sufficient fundamentals.

Example:
    >>> from src.data.fundamental_data import FundamentalDataFetcher
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> fetcher = FundamentalDataFetcher(config)
    >>> fundamentals = fetcher.fetch_all_fundamentals()
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Realistic per-stock fundamental ranges used for synthetic data generation
# ---------------------------------------------------------------------------
_STOCK_FUNDAMENTAL_RANGES: Dict[str, Dict[str, tuple]] = {
    "RELIANCE.NS": {
        "PE_Ratio": (20, 35),
        "Debt_to_Equity": (0.3, 0.8),
        "ROE": (8, 15),
        "EPS": (60, 120),
        "Book_Value": (900, 1500),
        "Market_Cap": (14e12, 20e12),
        "Dividend_Yield": (0.3, 0.6),
    },
    "HDFCBANK.NS": {
        "PE_Ratio": (15, 25),
        "Debt_to_Equity": (6, 9),
        "ROE": (14, 20),
        "EPS": (60, 90),
        "Book_Value": (350, 550),
        "Market_Cap": (8e12, 13e12),
        "Dividend_Yield": (0.8, 1.5),
    },
    "INFY.NS": {
        "PE_Ratio": (20, 35),
        "Debt_to_Equity": (0.05, 0.15),
        "ROE": (25, 35),
        "EPS": (50, 80),
        "Book_Value": (180, 280),
        "Market_Cap": (5e12, 8e12),
        "Dividend_Yield": (1.5, 3.0),
    },
    "TATAMOTORS.NS": {
        "PE_Ratio": (5, 50),
        "Debt_to_Equity": (0.8, 2.5),
        "ROE": (-10, 25),
        "EPS": (-20, 40),
        "Book_Value": (100, 250),
        "Market_Cap": (1e12, 4e12),
        "Dividend_Yield": (0.0, 0.5),
    },
    "BHARTIARTL.NS": {
        "PE_Ratio": (30, 80),
        "Debt_to_Equity": (1.0, 3.0),
        "ROE": (5, 20),
        "EPS": (15, 50),
        "Book_Value": (150, 300),
        "Market_Cap": (4e12, 9e12),
        "Dividend_Yield": (0.3, 0.8),
    },
    "HINDUNILVR.NS": {
        "PE_Ratio": (50, 80),
        "Debt_to_Equity": (0.1, 0.3),
        "ROE": (80, 120),
        "EPS": (30, 50),
        "Book_Value": (30, 60),
        "Market_Cap": (5e12, 7e12),
        "Dividend_Yield": (1.2, 2.0),
    },
}


class FundamentalDataFetcher:
    """Fetches quarterly fundamental data for the stock universe.

    The class employs a *try-yfinance-first, synthetic-fallback* strategy so
    that the downstream pipeline always receives complete fundamental data
    regardless of API availability.

    Attributes:
        config: Parsed ``config.yaml`` dictionary.
        tickers: List of Yahoo Finance ticker symbols.
        ticker_names: Mapping of ticker symbol → company name.
        start_date: Start of the data window.
        end_date: End of the data window.
        raw_data_path: Directory for persisting raw parquet files.
    """

    # Required output columns
    FUNDAMENTAL_COLUMNS = [
        "PE_Ratio",
        "Debt_to_Equity",
        "ROE",
        "EPS",
        "Book_Value",
        "Market_Cap",
        "Dividend_Yield",
    ]

    def __init__(self, config: dict) -> None:
        """Initialise the fetcher from the master configuration dictionary.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config

        self.tickers: list[str] = config.get("stocks", {}).get("tickers", [])
        self.ticker_names: dict[str, str] = config.get("stocks", {}).get(
            "names", {}
        )

        dates_cfg = config.get("dates", {})
        self.start_date: str = dates_cfg.get("start", "2020-01-01")
        self.end_date: str = dates_cfg.get("end", "2025-12-31")

        paths_cfg = config.get("paths", {})
        self.raw_data_path: str = paths_cfg.get("raw_data", "data/raw")
        self.raw_dir: str = paths_cfg.get("raw_fundamental", str(Path(self.raw_data_path) / "fundamental"))
        self.processed_path: str = paths_cfg.get("processed_data", "data/processed")

        logger.info(
            "FundamentalDataFetcher initialised – %d tickers", len(self.tickers)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Fetch fundamental data for a single stock.

        The method first attempts to retrieve fundamentals via *yfinance*.
        If the result is empty or incomplete it falls back to synthetic data
        generation with realistic per-stock ranges.

        Args:
            ticker: Yahoo Finance symbol, e.g. ``'RELIANCE.NS'``.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` (quarter-end dates) with
            columns ``PE_Ratio``, ``Debt_to_Equity``, ``ROE``, ``EPS``,
            ``Book_Value``, ``Market_Cap``, ``Dividend_Yield``.
        """
        logger.info("Fetching fundamentals for %s", ticker)

        # Try yfinance first
        df = self._get_yfinance_fundamentals(ticker)

        if df is not None and not df.empty and len(df) >= 4:
            logger.info(
                "yfinance provided %d quarterly records for %s",
                len(df),
                ticker,
            )
            return df

        # Fallback to synthetic
        logger.info(
            "Falling back to synthetic fundamental data for %s", ticker
        )
        return self._generate_synthetic_fundamentals(ticker)

    def fetch_all_fundamentals(self) -> Dict[str, pd.DataFrame]:
        """Fetch fundamentals for every stock in the universe.

        Saves each stock's data to ``data/raw/fundamental/{ticker}_fundamentals.parquet``.

        Returns:
            Dictionary mapping ticker → fundamental ``pd.DataFrame``.
        """
        all_data: Dict[str, pd.DataFrame] = {}
        fund_dir = Path(self.raw_dir)
        fund_dir.mkdir(parents=True, exist_ok=True)

        for ticker in tqdm(self.tickers, desc="Fetching fundamentals"):
            try:
                df = self.fetch_fundamentals(ticker)
                all_data[ticker] = df

                out_path = fund_dir / f"{ticker}_fundamentals.parquet"
                df.to_parquet(out_path, index=True)
                logger.info("Saved %s fundamentals → %s", ticker, out_path)

            except Exception as exc:
                logger.error(
                    "Failed to fetch fundamentals for %s: %s", ticker, exc
                )

        logger.info(
            "Fetched fundamentals for %d / %d tickers",
            len(all_data),
            len(self.tickers),
        )

        # Save processed (merged) version
        if all_data:
            proc_dir = Path(self.processed_path)
            proc_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for t, df in all_data.items():
                tmp = df.copy()
                tmp["ticker"] = t
                frames.append(tmp)
            merged = pd.concat(frames)
            proc_path = proc_dir / "fundamental_processed.parquet"
            merged.to_parquet(proc_path, index=True)
            logger.info("Saved processed fundamental data → %s", proc_path)

        return all_data

    def load_cached(self) -> Dict[str, pd.DataFrame]:
        """Load previously fetched fundamental data from ``data/raw/fundamental/``.

        Reads parquet files matching the pattern
        ``{ticker}_fundamentals.parquet`` for every ticker in the configured
        universe.

        Returns:
            A dictionary mapping each ticker to its cached
            ``pd.DataFrame``.  Tickers whose cache files are missing
            are silently skipped.
        """
        cached_data: Dict[str, pd.DataFrame] = {}
        fund_dir = Path(self.raw_dir)

        for ticker in self.tickers:
            # Try new path first, fall back to old path
            file_path = fund_dir / f"{ticker}_fundamentals.parquet"
            if not file_path.exists():
                file_path = Path(self.raw_data_path) / f"{ticker}_fundamentals.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    cached_data[ticker] = df
                    logger.info(
                        "Loaded cached %s fundamentals (%d rows) from %s",
                        ticker,
                        len(df),
                        file_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load cached fundamentals for %s: %s", ticker, exc
                    )
            else:
                logger.warning("No cached fundamentals found for %s at %s", ticker, file_path)

        logger.info(
            "Loaded cached fundamentals for %d / %d tickers",
            len(cached_data),
            len(self.tickers),
        )
        return cached_data

    # ------------------------------------------------------------------
    # yfinance approach
    # ------------------------------------------------------------------

    def _get_yfinance_fundamentals(
        self, ticker: str
    ) -> Optional[pd.DataFrame]:
        """Attempt to retrieve fundamentals from *yfinance*.

        Uses ``Ticker.quarterly_financials``, ``Ticker.quarterly_balance_sheet``,
        and ``Ticker.info`` to assemble the required metrics.

        Args:
            ticker: Yahoo Finance symbol.

        Returns:
            A ``pd.DataFrame`` with the standard columns if data is
            available, or ``None`` if insufficient data is retrieved.
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info or {}

            # Try quarterly financials and balance sheet
            qf = stock.quarterly_financials
            qbs = stock.quarterly_balance_sheet

            if qf is None or qf.empty:
                logger.debug("No quarterly financials available for %s", ticker)
                return None

            records = []
            dates = qf.columns  # columns are quarter-end dates

            for date in dates:
                record: Dict[str, object] = {"Date": date}

                # EPS – try to compute from Net Income / shares outstanding
                try:
                    net_income = qf.loc["Net Income", date] if "Net Income" in qf.index else np.nan
                    shares = info.get("sharesOutstanding", np.nan)
                    record["EPS"] = (
                        net_income / shares if pd.notna(net_income) and pd.notna(shares) and shares > 0 else np.nan
                    )
                except Exception:
                    record["EPS"] = np.nan

                # PE Ratio from info (snapshot, not historical)
                record["PE_Ratio"] = info.get("trailingPE", np.nan)

                # Debt-to-Equity
                try:
                    if qbs is not None and not qbs.empty and date in qbs.columns:
                        total_debt_key = next(
                            (k for k in qbs.index if "debt" in k.lower() and "total" in k.lower()),
                            None,
                        )
                        equity_key = next(
                            (k for k in qbs.index if "stockholder" in k.lower() or "equity" in k.lower()),
                            None,
                        )
                        if total_debt_key and equity_key:
                            debt = qbs.loc[total_debt_key, date]
                            equity = qbs.loc[equity_key, date]
                            record["Debt_to_Equity"] = (
                                debt / equity if pd.notna(equity) and equity != 0 else np.nan
                            )
                        else:
                            record["Debt_to_Equity"] = np.nan
                    else:
                        record["Debt_to_Equity"] = np.nan
                except Exception:
                    record["Debt_to_Equity"] = np.nan

                # ROE
                try:
                    if pd.notna(record.get("EPS")) and pd.notna(info.get("bookValue")):
                        bv = info.get("bookValue", 1)
                        record["ROE"] = (record["EPS"] / bv * 100) if bv != 0 else np.nan
                    else:
                        record["ROE"] = np.nan
                except Exception:
                    record["ROE"] = np.nan

                record["Book_Value"] = info.get("bookValue", np.nan)
                record["Market_Cap"] = info.get("marketCap", np.nan)
                record["Dividend_Yield"] = info.get("dividendYield", np.nan)
                if pd.notna(record["Dividend_Yield"]):
                    record["Dividend_Yield"] = record["Dividend_Yield"] * 100  # pct

                records.append(record)

            if not records:
                return None

            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            # Check if we have enough non-NaN data
            non_nan_frac = df[self.FUNDAMENTAL_COLUMNS].notna().mean().mean()
            if non_nan_frac < 0.3:
                logger.debug(
                    "yfinance data for %s has only %.0f%% coverage — insufficient",
                    ticker,
                    non_nan_frac * 100,
                )
                return None

            return df[self.FUNDAMENTAL_COLUMNS]

        except Exception as exc:
            logger.debug(
                "yfinance fundamentals failed for %s: %s", ticker, exc
            )
            return None

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def _generate_synthetic_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Generate realistic synthetic quarterly fundamental data.

        Creates quarter-end data points between ``start_date`` and
        ``end_date`` with smooth trends, seasonal patterns, and
        Gaussian noise calibrated to each stock's historical range.

        Args:
            ticker: Yahoo Finance symbol.

        Returns:
            ``pd.DataFrame`` indexed by ``Date`` with the standard
            fundamental columns.
        """
        rng = np.random.RandomState(hash(ticker) % (2**31))

        # Generate quarter-end dates
        quarter_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq="QE",  # Quarter End
        )

        if len(quarter_dates) == 0:
            quarter_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq="Q",
            )

        n_quarters = len(quarter_dates)
        ranges = _STOCK_FUNDAMENTAL_RANGES.get(
            ticker,
            _STOCK_FUNDAMENTAL_RANGES.get("RELIANCE.NS"),  # default fallback
        )

        data: Dict[str, list] = {col: [] for col in self.FUNDAMENTAL_COLUMNS}

        for col in self.FUNDAMENTAL_COLUMNS:
            lo, hi = ranges[col]  # type: ignore[index]
            mid = (lo + hi) / 2.0
            amplitude = (hi - lo) / 2.0

            # Create a smooth trend with noise
            trend = np.linspace(0, 1, n_quarters)
            # Gentle upward or downward drift
            drift_direction = rng.choice([-1, 1])
            drift = drift_direction * amplitude * 0.3 * trend

            # Seasonal component (annual cycle)
            seasonal = amplitude * 0.15 * np.sin(
                2 * np.pi * np.arange(n_quarters) / 4
            )

            # Random noise
            noise = rng.normal(0, amplitude * 0.1, n_quarters)

            values = mid + drift + seasonal + noise

            # Clip to realistic range
            values = np.clip(values, lo, hi)
            data[col] = values.tolist()

        df = pd.DataFrame(data, index=quarter_dates)
        df.index.name = "Date"

        logger.info(
            "Generated synthetic fundamentals for %s – %d quarters "
            "(%s → %s)",
            ticker,
            n_quarters,
            quarter_dates[0].strftime("%Y-%m-%d"),
            quarter_dates[-1].strftime("%Y-%m-%d"),
        )
        return df
