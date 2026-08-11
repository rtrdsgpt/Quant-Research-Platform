"""
Sentiment feature engineering from FinBERT sentiment scores.

This module transforms daily sentiment scores (positive, negative,
neutral) into model-ready features including moving averages, momentum,
extreme indicators, and headline activity metrics.  All features are
lagged by 1 trading day so that news sentiment from day *T* is used
to predict returns on day *T+1*.

Example:
    >>> from src.features.sentiment_features import SentimentFeatureEngineer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> engineer = SentimentFeatureEngineer(config)
    >>> features = engineer.generate_all_features(sentiment_df)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentFeatureEngineer:
    """Engineers features from daily sentiment scores.

    Input sentiment DataFrames are expected to have at minimum a
    ``Sentiment_Score`` column (positive − negative) and optionally
    ``Sentiment_Positive``, ``Sentiment_Negative``,
    ``Sentiment_Neutral``, and ``Headline_Count`` columns.

    All generated features are shifted by ``lag_days`` (default 1)
    trading days before being returned, preventing look-ahead bias.

    Attributes:
        config: Full pipeline configuration dictionary.
        lag_days: Number of trading days to lag sentiment features.
    """

    def __init__(self, config: dict) -> None:
        """Initialize with configuration dictionary.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config
        sent_cfg = config.get("features", {}).get("sentiment", {})

        self.lag_days: int = sent_cfg.get("lag_days", 1)

        logger.info(
            "SentimentFeatureEngineer initialised – lag %d day(s)",
            self.lag_days,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_score_col(df: pd.DataFrame) -> str:
        """Identify the sentiment score column in *df*.

        Checks for ``Sentiment_Score``, ``sentiment_score``, and
        ``Score`` in that order.

        Args:
            df: Sentiment DataFrame.

        Returns:
            Column name string.

        Raises:
            KeyError: If no recognisable score column is found.
        """
        for candidate in ["Sentiment_Score", "sentiment_score", "Score"]:
            if candidate in df.columns:
                return candidate
        raise KeyError(
            "No sentiment score column found. "
            "Expected one of: Sentiment_Score, sentiment_score, Score"
        )

    # ------------------------------------------------------------------
    # Sentiment momentum
    # ------------------------------------------------------------------

    def compute_sentiment_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment momentum features.

        Features created:
            - ``Sentiment_MA_5``: 5-day simple moving average.
            - ``Sentiment_MA_10``: 10-day simple moving average.
            - ``Sentiment_Trend``: ``MA_5 − MA_10``.
            - ``Sentiment_Change_5D``: 5-day absolute change.
            - ``Sentiment_Vol_10D``: 10-day rolling std of sentiment.

        Args:
            df: Sentiment DataFrame with a score column.

        Returns:
            DataFrame with sentiment momentum features.
        """
        result = pd.DataFrame(index=df.index)
        score_col = self._get_score_col(df)
        score = df[score_col]

        result["Sentiment_MA_5"] = score.rolling(window=5, min_periods=1).mean()
        result["Sentiment_MA_10"] = score.rolling(window=10, min_periods=1).mean()
        result["Sentiment_Trend"] = result["Sentiment_MA_5"] - result["Sentiment_MA_10"]
        result["Sentiment_Change_5D"] = score - score.shift(5)
        result["Sentiment_Vol_10D"] = score.rolling(window=10, min_periods=3).std()

        logger.debug("Computed sentiment momentum features")
        return result

    # ------------------------------------------------------------------
    # Sentiment extremes
    # ------------------------------------------------------------------

    def compute_sentiment_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment extreme indicators.

        Features created:
            - ``Sentiment_Zscore``: ``(score − MA_20) / Std_20``.
            - ``High_Sentiment``: 1 if z-score > 1.5 else 0.
            - ``Low_Sentiment``: 1 if z-score < −1.5 else 0.
            - ``Sentiment_Positive_Ratio``: ``Positive / (Positive + Negative)``
              (requires component columns).

        Args:
            df: Sentiment DataFrame with score and optionally
                ``Sentiment_Positive`` / ``Sentiment_Negative`` columns.

        Returns:
            DataFrame with extreme indicator features.
        """
        result = pd.DataFrame(index=df.index)
        score_col = self._get_score_col(df)
        score = df[score_col]

        ma_20 = score.rolling(window=20, min_periods=5).mean()
        std_20 = score.rolling(window=20, min_periods=5).std().replace(0, np.nan)

        result["Sentiment_Zscore"] = (score - ma_20) / std_20
        result["High_Sentiment"] = (result["Sentiment_Zscore"] > 1.5).astype(int)
        result["Low_Sentiment"] = (result["Sentiment_Zscore"] < -1.5).astype(int)

        # Positive ratio (if component columns exist)
        pos_col = None
        neg_col = None
        for candidate in ["Sentiment_Positive", "sentiment_positive", "Positive"]:
            if candidate in df.columns:
                pos_col = candidate
                break
        for candidate in ["Sentiment_Negative", "sentiment_negative", "Negative"]:
            if candidate in df.columns:
                neg_col = candidate
                break

        if pos_col is not None and neg_col is not None:
            denom = (df[pos_col] + df[neg_col]).replace(0, np.nan)
            result["Sentiment_Positive_Ratio"] = df[pos_col] / denom
        else:
            # Derive from score: map score ∈ [-1,1] → ratio ∈ [0,1]
            result["Sentiment_Positive_Ratio"] = (score + 1) / 2.0

        logger.debug("Computed sentiment extreme features")
        return result

    # ------------------------------------------------------------------
    # Headline features
    # ------------------------------------------------------------------

    def compute_headline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute headline count-based features.

        Features created:
            - ``Headline_Count_MA_5``: 5-day moving average of daily
              headline count.
            - ``High_Activity``: 1 if headline count > 2 × MA_5 else 0.

        Args:
            df: Sentiment DataFrame, optionally containing a
                ``Headline_Count`` column.

        Returns:
            DataFrame with headline activity features.
        """
        result = pd.DataFrame(index=df.index)

        # Detect headline count column
        count_col = None
        for candidate in ["Headline_Count", "headline_count", "Count"]:
            if candidate in df.columns:
                count_col = candidate
                break

        if count_col is not None:
            count = df[count_col].astype(float)
            ma5 = count.rolling(window=5, min_periods=1).mean()
            result["Headline_Count_MA_5"] = ma5
            result["High_Activity"] = (count > 2.0 * ma5).astype(int)
        else:
            logger.debug("No Headline_Count column found; skipping headline features")

        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all sentiment features with look-ahead bias lag.

        All features are shifted forward by ``self.lag_days`` trading
        days after computation so that sentiment from day *T* is used
        to predict returns on day *T + lag_days*.

        Args:
            sentiment_df: DataFrame indexed by Date with at least a
                sentiment score column (see ``_get_score_col``).

        Returns:
            DataFrame indexed by Date with all sentiment features,
            lagged.
        """
        logger.info("Generating sentiment features – %d rows", len(sentiment_df))

        # Ensure DatetimeIndex
        df = sentiment_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Compute feature groups
        momentum_df = self.compute_sentiment_momentum(df)
        extremes_df = self.compute_sentiment_extremes(df)
        headline_df = self.compute_headline_features(df)

        # Raw sentiment columns (renamed with prefix)
        raw_cols = {}
        score_col = self._get_score_col(df)
        raw_cols["Sent_Score"] = df[score_col]
        for candidate_name, prefix in [
            ("Sentiment_Positive", "Sent_Positive"),
            ("sentiment_positive", "Sent_Positive"),
            ("Sentiment_Negative", "Sent_Negative"),
            ("sentiment_negative", "Sent_Negative"),
            ("Sentiment_Neutral", "Sent_Neutral"),
            ("sentiment_neutral", "Sent_Neutral"),
        ]:
            if candidate_name in df.columns:
                raw_cols[prefix] = df[candidate_name]

        raw_df = pd.DataFrame(raw_cols, index=df.index)

        # Combine all
        all_features = pd.concat(
            [raw_df, momentum_df, extremes_df, headline_df], axis=1
        )
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # CRITICAL: Lag by 1 day to prevent look-ahead bias
        all_features = all_features.shift(self.lag_days)

        logger.info(
            "Sentiment features generated – %d features, %d rows, lag %d day(s)",
            len(all_features.columns),
            len(all_features),
            self.lag_days,
        )
        return all_features
