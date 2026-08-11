"""
Technical feature engineering from OHLCV price data.

This module computes price-based technical indicators including returns,
moving averages, momentum, volatility, volume features, and candlestick
price patterns from raw OHLCV data.

Example:
    >>> from src.features.technical_features import TechnicalFeatureEngineer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> engineer = TechnicalFeatureEngineer(config)
    >>> features_df = engineer.generate_all_features(ohlcv_df)
"""

from typing import List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalFeatureEngineer:
    """Generates technical indicators from OHLCV price data.

    All calculations use ``Adj_Close`` for price-based features to
    account for stock splits and dividends.  Features are computed
    using only past data to prevent look-ahead bias.  The target
    variable is the *future* 1-day log return (``shift(-1)``).

    Attributes:
        config: Full pipeline configuration dictionary.
        sma_windows: SMA window sizes from config.
        ema_windows: EMA window sizes from config.
        rsi_window: RSI lookback period.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal: MACD signal line EMA period.
        bollinger_window: Bollinger Band window.
        bollinger_std: Bollinger Band standard deviation multiplier.
        atr_window: Average True Range window.
        volume_ma_window: Volume moving average window.
        log_return_lags: Lag periods for return features.
        volatility_windows: Rolling volatility window sizes.
    """

    def __init__(self, config: dict) -> None:
        """Initialize with configuration dictionary.

        Args:
            config: Parsed ``config.yaml`` dictionary containing at
                least the ``features`` section with ``technical`` and
                ``returns`` sub-sections.
        """
        self.config = config
        tech_cfg = config.get("features", {}).get("technical", {})
        ret_cfg = config.get("features", {}).get("returns", {})

        # Moving average windows
        self.sma_windows: List[int] = tech_cfg.get("sma_windows", [5, 10, 20, 50])
        self.ema_windows: List[int] = tech_cfg.get("ema_windows", [12, 26])

        # Momentum parameters
        self.rsi_window: int = tech_cfg.get("rsi_window", 14)
        self.macd_fast: int = tech_cfg.get("macd_fast", 12)
        self.macd_slow: int = tech_cfg.get("macd_slow", 26)
        self.macd_signal: int = tech_cfg.get("macd_signal", 9)

        # Volatility parameters
        self.bollinger_window: int = tech_cfg.get("bollinger_window", 20)
        self.bollinger_std: float = tech_cfg.get("bollinger_std", 2.0)
        self.atr_window: int = tech_cfg.get("atr_window", 14)

        # Volume parameters
        self.volume_ma_window: int = tech_cfg.get("volume_ma_window", 20)

        # Return parameters
        self.log_return_lags: List[int] = ret_cfg.get(
            "log_return_lags", [1, 2, 3, 5, 10, 21]
        )
        self.volatility_windows: List[int] = ret_cfg.get(
            "volatility_windows", [5, 10, 21, 63]
        )

        logger.info(
            "TechnicalFeatureEngineer initialised – SMA %s, EMA %s, "
            "RSI %d, lags %s",
            self.sma_windows,
            self.ema_windows,
            self.rsi_window,
            self.log_return_lags,
        )

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns and lagged returns.

        Features created:
            - ``Log_Return``: ``ln(Close_t / Close_{t-1})``
            - ``Log_Return_Lag_{n}``: lagged log returns for each
              configured lag period.
            - ``Target``: future 1-day log return (``Log_Return.shift(-1)``).

        Args:
            df: DataFrame with ``Adj_Close`` column indexed by Date.

        Returns:
            DataFrame with return and target columns.
        """
        result = pd.DataFrame(index=df.index)

        close = df["Adj_Close"]
        result["Log_Return"] = np.log(close / close.shift(1))

        # Lagged log returns (all use past data only)
        for lag in self.log_return_lags:
            result[f"Log_Return_Lag_{lag}"] = result["Log_Return"].shift(lag)

        # CRITICAL: Target is FUTURE log return (shift by -1)
        result["Target"] = result["Log_Return"].shift(-1)

        logger.debug(
            "Computed returns – %d lag features + Target", len(self.log_return_lags)
        )
        return result

    # ------------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------------

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute SMA and EMA indicators.

        Features created:
            - ``SMA_{w}``: simple moving average of Adj_Close for each
              configured window *w*.
            - ``EMA_{w}``: exponential moving average for each window.
            - ``Price_to_SMA_{w}``: Close / SMA ratio for the shortest
              and a longer SMA window.
            - ``SMA_{short}_{long}_Cross``: +1 when short SMA > long
              SMA, else -1.

        Args:
            df: DataFrame with ``Adj_Close`` column indexed by Date.

        Returns:
            DataFrame with moving-average feature columns.
        """
        result = pd.DataFrame(index=df.index)
        close = df["Adj_Close"]

        # Simple Moving Averages
        for w in self.sma_windows:
            result[f"SMA_{w}"] = close.rolling(window=w).mean()
            result[f"Price_to_SMA_{w}"] = close / result[f"SMA_{w}"]

        # Exponential Moving Averages
        for w in self.ema_windows:
            result[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

        # Cross signal between shortest and longest SMA
        if len(self.sma_windows) >= 2:
            short_w = min(self.sma_windows)
            long_w = max(self.sma_windows)
            sma_short = result[f"SMA_{short_w}"]
            sma_long = result[f"SMA_{long_w}"]
            result[f"SMA_{short_w}_{long_w}_Cross"] = np.where(
                sma_short > sma_long, 1, -1
            )

        logger.debug(
            "Computed moving averages – %d SMA, %d EMA windows",
            len(self.sma_windows),
            len(self.ema_windows),
        )
        return result

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators.

        Features created:
            - ``RSI_{rsi_window}``: Relative Strength Index.
            - ``MACD``, ``MACD_Signal``, ``MACD_Histogram``.
            - ``ROC_5``, ``ROC_10``, ``ROC_21``: rate of change.
            - ``Momentum_5``, ``Momentum_10``: absolute price momentum.

        Args:
            df: DataFrame with ``Adj_Close`` column indexed by Date.

        Returns:
            DataFrame with momentum feature columns.
        """
        result = pd.DataFrame(index=df.index)
        close = df["Adj_Close"]

        # ----- RSI -----
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window, min_periods=self.rsi_window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result[f"RSI_{self.rsi_window}"] = 100.0 - (100.0 / (1.0 + rs))

        # ----- MACD -----
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        result["MACD"] = ema_fast - ema_slow
        result["MACD_Signal"] = result["MACD"].ewm(
            span=self.macd_signal, adjust=False
        ).mean()
        result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]

        # ----- Rate of Change -----
        for period in [5, 10, 21]:
            result[f"ROC_{period}"] = (close - close.shift(period)) / close.shift(
                period
            )

        # ----- Momentum (absolute) -----
        for period in [5, 10]:
            result[f"Momentum_{period}"] = close - close.shift(period)

        logger.debug("Computed momentum indicators")
        return result

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility indicators.

        Features created:
            - ``Volatility_{w}``: annualised rolling std of log returns
              for each configured volatility window *w*.
            - ``ATR_{atr_window}``: Average True Range.
            - ``Bollinger_Upper``, ``Bollinger_Lower``,
              ``Bollinger_Width``, ``Bollinger_Pct``.
            - ``Parkinson_Volatility``: high-low based volatility
              estimator.

        Args:
            df: DataFrame with Adj_Close, High, Low columns.

        Returns:
            DataFrame with volatility feature columns.
        """
        result = pd.DataFrame(index=df.index)
        close = df["Adj_Close"]
        high = df["High"]
        low = df["Low"]

        log_ret = np.log(close / close.shift(1))

        # Rolling volatility (annualised)
        for w in self.volatility_windows:
            result[f"Volatility_{w}"] = log_ret.rolling(window=w).std() * np.sqrt(252)

        # ----- ATR -----
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result[f"ATR_{self.atr_window}"] = true_range.rolling(
            window=self.atr_window
        ).mean()

        # ----- Bollinger Bands -----
        sma_bb = close.rolling(window=self.bollinger_window).mean()
        std_bb = close.rolling(window=self.bollinger_window).std()
        result["Bollinger_Upper"] = sma_bb + self.bollinger_std * std_bb
        result["Bollinger_Lower"] = sma_bb - self.bollinger_std * std_bb
        result["Bollinger_Width"] = (
            result["Bollinger_Upper"] - result["Bollinger_Lower"]
        ) / sma_bb
        bb_range = result["Bollinger_Upper"] - result["Bollinger_Lower"]
        result["Bollinger_Pct"] = (close - result["Bollinger_Lower"]) / bb_range.replace(
            0, np.nan
        )

        # ----- Parkinson Volatility (high-low estimator) -----
        log_hl = np.log(high / low.replace(0, np.nan))
        result["Parkinson_Volatility"] = (
            log_hl.rolling(window=self.bollinger_window).apply(
                lambda x: np.sqrt((1 / (4 * len(x) * np.log(2))) * np.sum(x**2)),
                raw=True,
            )
            * np.sqrt(252)
        )

        logger.debug("Computed volatility indicators")
        return result

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features.

        Features created:
            - ``Volume_MA_20``: 20-day moving average of volume.
            - ``Volume_Ratio``: Volume / Volume_MA_20.
            - ``OBV``: On-Balance Volume.
            - ``Volume_Price_Trend``: Volume * daily return.
            - ``VWAP_Proxy``: rolling 20-day volume-weighted average price.

        Args:
            df: DataFrame with Adj_Close and Volume columns.

        Returns:
            DataFrame with volume feature columns.
        """
        result = pd.DataFrame(index=df.index)
        close = df["Adj_Close"]
        volume = df["Volume"].astype(float)

        # Volume moving average
        w = self.volume_ma_window
        result[f"Volume_MA_{w}"] = volume.rolling(window=w).mean()
        result["Volume_Ratio"] = volume / result[f"Volume_MA_{w}"].replace(0, np.nan)

        # On-Balance Volume
        direction = np.sign(close.diff())
        result["OBV"] = (volume * direction).cumsum()

        # Volume Price Trend
        pct_change = close.pct_change()
        result["Volume_Price_Trend"] = volume * pct_change

        # VWAP proxy (rolling 20-day)
        vol_price = volume * close
        result["VWAP_Proxy"] = (
            vol_price.rolling(window=w).sum()
            / volume.rolling(window=w).sum().replace(0, np.nan)
        )

        logger.debug("Computed volume features")
        return result

    # ------------------------------------------------------------------
    # Price patterns
    # ------------------------------------------------------------------

    def compute_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price pattern features from OHLC candles.

        Features created:
            - ``Daily_Range``: ``(High - Low) / Close``
            - ``Daily_Body``: ``|Close - Open| / Close``
            - ``Upper_Shadow``: ``(High - max(Open, Close)) / Close``
            - ``Lower_Shadow``: ``(min(Open, Close) - Low) / Close``
            - ``Gap_Up``: 1 if ``Open > prev_High`` else 0
            - ``Gap_Down``: 1 if ``Open < prev_Low`` else 0
            - ``Higher_High``: 1 if ``High > prev_High`` else 0
            - ``Lower_Low``: 1 if ``Low < prev_Low`` else 0

        Args:
            df: DataFrame with Open, High, Low, Close columns.

        Returns:
            DataFrame with price-pattern feature columns.
        """
        result = pd.DataFrame(index=df.index)
        o = df["Open"]
        h = df["High"]
        low = df["Low"]
        c = df["Close"]

        safe_close = c.replace(0, np.nan)

        result["Daily_Range"] = (h - low) / safe_close
        result["Daily_Body"] = (c - o).abs() / safe_close
        result["Upper_Shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / safe_close
        result["Lower_Shadow"] = (
            pd.concat([o, c], axis=1).min(axis=1) - low
        ) / safe_close

        prev_high = h.shift(1)
        prev_low = low.shift(1)

        result["Gap_Up"] = (o > prev_high).astype(int)
        result["Gap_Down"] = (o < prev_low).astype(int)
        result["Higher_High"] = (h > prev_high).astype(int)
        result["Lower_Low"] = (low < prev_low).astype(int)

        logger.debug("Computed price pattern features")
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical features for a single stock.

        Calls every compute method, concatenates the results, and
        returns a single DataFrame.  The ``Target`` column (future
        log return) is included.

        Args:
            ohlcv_df: DataFrame with columns
                ``Date, Open, High, Low, Close, Adj_Close, Volume``.
                Must be indexed by ``Date`` (DatetimeIndex).

        Returns:
            DataFrame indexed by Date with all technical features
            and the ``Target`` column.
        """
        logger.info(
            "Generating technical features – %d rows", len(ohlcv_df)
        )

        # Ensure DatetimeIndex
        df = ohlcv_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        # Compute each feature group
        returns_df = self.compute_returns(df)
        ma_df = self.compute_moving_averages(df)
        momentum_df = self.compute_momentum(df)
        volatility_df = self.compute_volatility(df)
        volume_df = self.compute_volume_features(df)
        patterns_df = self.compute_price_patterns(df)

        # Concatenate all feature groups
        all_features = pd.concat(
            [returns_df, ma_df, momentum_df, volatility_df, volume_df, patterns_df],
            axis=1,
        )

        # Remove duplicate columns if any
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        logger.info(
            "Technical features generated – %d features, %d rows",
            len(all_features.columns),
            len(all_features),
        )
        return all_features
