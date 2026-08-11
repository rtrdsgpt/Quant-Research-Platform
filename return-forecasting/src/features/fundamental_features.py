"""
Fundamental feature engineering from quarterly financial data.

This module transforms quarterly fundamental metrics (P/E ratio,
Debt-to-Equity, ROE, EPS) into daily-frequency model features with
appropriate lagging to prevent look-ahead bias.

Example:
    >>> from src.features.fundamental_features import FundamentalFeatureEngineer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> engineer = FundamentalFeatureEngineer(config)
    >>> features = engineer.generate_all_features(fundamentals_df, daily_dates)
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FundamentalFeatureEngineer:
    """Engineers features from quarterly fundamental data.

    Quarterly data is aligned to daily frequency via forward-fill with
    a safety lag of 45 calendar days (approximate time for quarterly
    results to be publicly released after quarter-end).

    Attributes:
        config: Full pipeline configuration dictionary.
        lag_days: Number of calendar days to lag fundamental data
            for look-ahead bias prevention.
        fundamental_cols: List of raw fundamental metric column names
            expected in the input DataFrame.
    """

    # Expected raw fundamental columns
    EXPECTED_COLS: List[str] = ["PE_Ratio", "Debt_to_Equity", "ROE", "EPS"]

    def __init__(self, config: dict) -> None:
        """Initialize with configuration dictionary.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config

        # Safety lag in calendar days (results released ~45 days after quarter end)
        self.lag_days: int = 45

        logger.info(
            "FundamentalFeatureEngineer initialised – lag %d days",
            self.lag_days,
        )

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_to_daily(
        self,
        fundamentals_df: pd.DataFrame,
        daily_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Align quarterly fundamental data to daily frequency.

        Strategy:
            1. Shift each quarterly observation forward by
               ``self.lag_days`` calendar days so the value is only
               available well after the quarter-end filing date.
            2. Reindex to the daily trading calendar.
            3. Forward-fill so each day carries the most recent
               (lagged) quarterly value.

        Args:
            fundamentals_df: DataFrame indexed by quarter-end date with
                fundamental metric columns (e.g. PE_Ratio, EPS).
            daily_dates: Target daily DatetimeIndex to align to.

        Returns:
            DataFrame reindexed to ``daily_dates`` with forward-filled
            fundamental values.
        """
        df = fundamentals_df.copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Apply calendar-day lag to prevent look-ahead bias
        df.index = df.index + pd.Timedelta(days=self.lag_days)

        # Reindex to daily dates and forward-fill
        aligned = df.reindex(daily_dates, method="ffill")

        logger.info(
            "Aligned quarterly data to daily – %d rows, lag %d days",
            len(aligned.dropna(how="all")),
            self.lag_days,
        )
        return aligned

    # ------------------------------------------------------------------
    # Quarter-over-quarter changes
    # ------------------------------------------------------------------

    def compute_fundamental_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute quarter-over-quarter percentage changes.

        Because the input is daily-frequency (forward-filled quarterly
        values), the method detects points where the underlying
        quarterly value *changed* and computes percentage differences.
        Between change points the QoQ value is carried forward.

        Features created:
            - ``PE_Change_QoQ``
            - ``EPS_Growth_QoQ``
            - ``ROE_Change_QoQ``
            - ``Debt_to_Equity_Change_QoQ``

        Args:
            df: Daily-aligned fundamental DataFrame.

        Returns:
            DataFrame with QoQ change columns.
        """
        result = pd.DataFrame(index=df.index)

        metric_map = {
            "PE_Ratio": "PE_Change_QoQ",
            "EPS": "EPS_Growth_QoQ",
            "ROE": "ROE_Change_QoQ",
            "Debt_to_Equity": "Debt_to_Equity_Change_QoQ",
        }

        for src_col, dst_col in metric_map.items():
            if src_col not in df.columns:
                logger.warning("Column %s not found; skipping %s", src_col, dst_col)
                continue

            series = df[src_col]

            # Detect quarterly change points (value changed from previous row)
            changed = series.diff().abs() > 1e-10
            prev_val = series.where(changed).ffill().shift(1)

            # Safe percentage change avoiding division by zero
            safe_prev = prev_val.replace(0, np.nan)
            result[dst_col] = ((series - prev_val) / safe_prev.abs()).ffill()

        logger.debug("Computed %d QoQ change features", len(result.columns))
        return result

    # ------------------------------------------------------------------
    # Derived ratios
    # ------------------------------------------------------------------

    def compute_fundamental_ratios(
        self,
        df: pd.DataFrame,
        close_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute derived fundamental ratios.

        Features created:
            - ``Earnings_Yield``: ``EPS / Close`` (requires price data).
            - ``PE_Relative_Mean``: ``PE / rolling_mean(PE, 252 days ≈ 4 quarters)``.
            - ``Quality_Score``: ``z(ROE) - z(Debt_to_Equity)`` where
              *z* denotes rolling z-score normalisation.

        Args:
            df: Daily-aligned fundamental DataFrame.
            close_prices: Optional Adj_Close Series for
                Earnings_Yield calculation.

        Returns:
            DataFrame with derived ratio columns.
        """
        result = pd.DataFrame(index=df.index)

        # Earnings Yield
        if close_prices is not None and "EPS" in df.columns:
            safe_close = close_prices.replace(0, np.nan)
            result["Earnings_Yield"] = df["EPS"] / safe_close
        elif "EPS" in df.columns:
            logger.debug("No close prices provided; skipping Earnings_Yield")

        # PE relative to rolling mean (~4 quarters ≈ 252 trading days)
        if "PE_Ratio" in df.columns:
            pe = df["PE_Ratio"]
            pe_ma = pe.rolling(window=252, min_periods=63).mean()
            result["PE_Relative_Mean"] = pe / pe_ma.replace(0, np.nan)

        # Quality Score = normalised ROE - normalised Debt_to_Equity
        if "ROE" in df.columns and "Debt_to_Equity" in df.columns:
            window = 252

            roe = df["ROE"]
            roe_mean = roe.rolling(window=window, min_periods=63).mean()
            roe_std = roe.rolling(window=window, min_periods=63).std().replace(0, np.nan)
            roe_z = (roe - roe_mean) / roe_std

            de = df["Debt_to_Equity"]
            de_mean = de.rolling(window=window, min_periods=63).mean()
            de_std = de.rolling(window=window, min_periods=63).std().replace(0, np.nan)
            de_z = (de - de_mean) / de_std

            result["Quality_Score"] = roe_z - de_z

        logger.debug("Computed %d fundamental ratio features", len(result.columns))
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all_features(
        self,
        fundamentals_df: pd.DataFrame,
        daily_dates: pd.DatetimeIndex,
        close_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Generate all fundamental features aligned to daily dates.

        Args:
            fundamentals_df: Quarterly fundamental DataFrame indexed by
                quarter-end date.
            daily_dates: Target daily DatetimeIndex.
            close_prices: Optional Adj_Close price Series for
                Earnings_Yield computation.

        Returns:
            DataFrame indexed by ``daily_dates`` with all fundamental
            features.
        """
        logger.info("Generating fundamental features")

        # Step 1: Align quarterly → daily with lag
        daily_df = self.align_to_daily(fundamentals_df, daily_dates)

        # Step 2: Raw metrics as features (already forward-filled)
        raw_features = daily_df.copy()
        raw_features.columns = [f"Fund_{c}" for c in raw_features.columns]

        # Step 3: QoQ changes
        changes_df = self.compute_fundamental_changes(daily_df)

        # Step 4: Derived ratios
        ratios_df = self.compute_fundamental_ratios(daily_df, close_prices)

        # Combine all
        all_features = pd.concat([raw_features, changes_df, ratios_df], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        logger.info(
            "Fundamental features generated – %d features, %d rows",
            len(all_features.columns),
            len(all_features),
        )
        return all_features
