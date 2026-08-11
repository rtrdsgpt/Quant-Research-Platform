"""
Macro feature engineering from macroeconomic indicators.

This module derives features from macro indicators such as USD/INR
exchange rate, crude oil prices, gold, bond yields, Nifty 50, and
India VIX.  All features are lagged by at least 1 trading day to
prevent look-ahead bias.

Example:
    >>> from src.features.macro_features import MacroFeatureEngineer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> engineer = MacroFeatureEngineer(config)
    >>> features = engineer.generate_all_features(macro_df)
"""

from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MacroFeatureEngineer:
    """Engineers features from macroeconomic indicators.

    All generated features are lagged by ``lag_days`` trading days
    (default 1) so that macro information available on day *T* uses
    only values published on or before day *T - lag_days*.

    Attributes:
        config: Full pipeline configuration dictionary.
        lag_days: Number of trading days to lag features.
        ma_window: Moving-average window for deviation features.
        regime_window: Window for regime indicator moving averages.
    """

    def __init__(self, config: dict) -> None:
        """Initialize with configuration dictionary.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config
        macro_cfg = config.get("features", {}).get("macro", {})

        self.lag_days: int = macro_cfg.get("lag_days", 1)
        self.ma_window: int = 20
        self.regime_window: int = 50

        logger.info(
            "MacroFeatureEngineer initialised – lag %d day(s)", self.lag_days
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        """Return list of numeric column names in *df*."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # Changes / returns
    # ------------------------------------------------------------------

    def compute_macro_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute changes and returns for each macro indicator.

        For every numeric column *Var* in the input:
            - ``{Var}_Return``: daily log return.
            - ``{Var}_Change_5D``: 5-day absolute change.
            - ``{Var}_Change_21D``: 21-day (≈ monthly) absolute change.
            - ``{Var}_MA_20``: 20-day simple moving average.
            - ``{Var}_Deviation``: current value / MA_20.

        Args:
            df: Macro indicator DataFrame indexed by Date.

        Returns:
            DataFrame with change / return features.
        """
        result = pd.DataFrame(index=df.index)

        for col in self._numeric_columns(df):
            series = df[col]
            safe = series.replace(0, np.nan)

            # Daily log return
            result[f"{col}_Return"] = np.log(series / series.shift(1))

            # Multi-day absolute changes
            result[f"{col}_Change_5D"] = series - series.shift(5)
            result[f"{col}_Change_21D"] = series - series.shift(21)

            # Moving average and deviation
            ma = series.rolling(window=self.ma_window).mean()
            result[f"{col}_MA_20"] = ma
            result[f"{col}_Deviation"] = series / ma.replace(0, np.nan)

        logger.debug("Computed macro change features for %d indicators", len(self._numeric_columns(df)))
        return result

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def compute_macro_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility of macro indicators.

        Features:
            - ``{Var}_Vol_21D``: 21-day rolling std of daily log
              returns for each indicator.
            - ``VIX_Level``: raw India VIX value (if present).
            - ``VIX_Change``: daily change in VIX.

        Args:
            df: Macro indicator DataFrame indexed by Date.

        Returns:
            DataFrame with volatility features.
        """
        result = pd.DataFrame(index=df.index)

        for col in self._numeric_columns(df):
            log_ret = np.log(df[col] / df[col].shift(1))
            result[f"{col}_Vol_21D"] = log_ret.rolling(window=21).std() * np.sqrt(252)

        # VIX-specific features
        vix_cols = [c for c in df.columns if "vix" in c.lower()]
        for vix_col in vix_cols:
            result["VIX_Level"] = df[vix_col]
            result["VIX_Change"] = df[vix_col].diff()

        logger.debug("Computed macro volatility features")
        return result

    # ------------------------------------------------------------------
    # Regime indicators
    # ------------------------------------------------------------------

    def compute_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market regime indicators.

        Features:
            - ``Risk_On_Off``: +1 if Nifty above 50-day MA, else -1.
            - ``Oil_Regime``: +1 if crude oil above 50-day MA, else -1.
            - ``Currency_Stress``: 1 if USD/INR deviation > 1 std, else 0.
            - ``Rate_Regime``: +1 if bond yield above 50-day MA, else -1.

        Args:
            df: Macro indicator DataFrame indexed by Date.

        Returns:
            DataFrame with regime indicator columns.
        """
        result = pd.DataFrame(index=df.index)
        w = self.regime_window

        # Helper: detect column by keyword (case-insensitive)
        def _find_col(keywords: List[str]) -> str | None:
            for kw in keywords:
                matches = [c for c in df.columns if kw.lower() in c.lower()]
                if matches:
                    return matches[0]
            return None

        # Risk On/Off (Nifty)
        nifty_col = _find_col(["nifty", "nsei"])
        if nifty_col is not None:
            ma = df[nifty_col].rolling(window=w).mean()
            result["Risk_On_Off"] = np.where(df[nifty_col] > ma, 1, -1)

        # Oil Regime
        oil_col = _find_col(["oil", "crude", "cl=f"])
        if oil_col is not None:
            ma = df[oil_col].rolling(window=w).mean()
            result["Oil_Regime"] = np.where(df[oil_col] > ma, 1, -1)

        # Currency Stress (USD/INR)
        fx_col = _find_col(["usd", "inr", "usdinr"])
        if fx_col is not None:
            ma = df[fx_col].rolling(window=w).mean()
            std = df[fx_col].rolling(window=w).std()
            deviation = (df[fx_col] - ma) / std.replace(0, np.nan)
            result["Currency_Stress"] = np.where(deviation.abs() > 1.0, 1, 0)

        # Rate Regime (Bond Yield)
        rate_col = _find_col(["bond", "yield", "10y", "irx", "tnx"])
        if rate_col is not None:
            ma = df[rate_col].rolling(window=w).mean()
            result["Rate_Regime"] = np.where(df[rate_col] > ma, 1, -1)

        logger.debug("Computed %d regime indicators", len(result.columns))
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all_features(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all macro features with look-ahead bias prevention.

        All features are shifted forward by ``self.lag_days`` trading
        days after computation so that the model at time *T* only
        sees macro data from *T - lag_days* or earlier.

        Args:
            macro_df: DataFrame indexed by Date with macro indicator
                columns (e.g. ``USDINR``, ``Crude_Oil``, ``Nifty50``).

        Returns:
            DataFrame indexed by Date with all macro features, lagged.
        """
        logger.info("Generating macro features – %d rows", len(macro_df))

        # Ensure DatetimeIndex
        df = macro_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Compute feature groups
        changes_df = self.compute_macro_changes(df)
        vol_df = self.compute_macro_volatility(df)
        regime_df = self.compute_regime_indicators(df)

        # Also keep raw macro levels as features
        raw_features = df.copy()
        raw_features.columns = [f"Macro_{c}" for c in raw_features.columns]

        # Combine all
        all_features = pd.concat(
            [raw_features, changes_df, vol_df, regime_df], axis=1
        )
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # CRITICAL: Apply lag to prevent look-ahead bias
        all_features = all_features.shift(self.lag_days)

        logger.info(
            "Macro features generated – %d features, %d rows, lag %d day(s)",
            len(all_features.columns),
            len(all_features),
            self.lag_days,
        )
        return all_features
