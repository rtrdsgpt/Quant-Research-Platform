"""
Unified feature engineering pipeline.

This module orchestrates all feature engineering sub-modules —
technical, fundamental, macro, and sentiment — into a single
``FeaturePipeline`` class.  It merges the resulting DataFrames,
applies robust scaling, handles missing values, and produces
train/test splits ready for model consumption.

Example:
    >>> from src.features.feature_pipeline import FeaturePipeline
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> pipeline = FeaturePipeline(config)
    >>> all_features = pipeline.build_all_feature_matrices(
    ...     ohlcv_data, fundamental_data, macro_data, sentiment_data
    ... )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.features.fundamental_features import FundamentalFeatureEngineer
from src.features.macro_features import MacroFeatureEngineer
from src.features.sentiment_features import SentimentFeatureEngineer
from src.features.technical_features import TechnicalFeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Combines all feature engineering modules into a unified pipeline.

    The pipeline performs the following steps for each ticker:

    1. Generate technical features from OHLCV data.
    2. Generate and align fundamental features from quarterly data.
    3. Generate macro features (common across tickers).
    4. Generate sentiment features.
    5. Merge all feature groups on the Date index.
    6. Apply ``RobustScaler`` normalisation.
    7. Handle missing values (forward-fill then drop remaining NaN rows).

    Attributes:
        config: Full pipeline configuration dictionary.
        technical_engineer: Technical feature builder.
        fundamental_engineer: Fundamental feature builder.
        macro_engineer: Macro feature builder.
        sentiment_engineer: Sentiment feature builder.
        scaler: Fitted ``RobustScaler`` instance (available after
            :meth:`scale_features` has been called with ``fit=True``).
        feature_names_: List of feature column names (excluding Target).
        train_end: Last date of training period.
        test_start: First date of test period.
        test_end: Last date of test period.
        features_path: Directory to persist feature parquet files.
    """

    def __init__(self, config: dict) -> None:
        """Initialize with config and all feature engineers.

        Args:
            config: Parsed ``config.yaml`` dictionary containing
                ``features``, ``dates``, and ``paths`` sections.
        """
        self.config = config

        # Sub-builders
        self.technical_engineer = TechnicalFeatureEngineer(config)
        self.fundamental_engineer = FundamentalFeatureEngineer(config)
        self.macro_engineer = MacroFeatureEngineer(config)
        self.sentiment_engineer = SentimentFeatureEngineer(config)

        # Scaler (fitted during training)
        self.scaler: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []

        # Date boundaries
        dates_cfg = config.get("dates", {})
        self.train_end: str = dates_cfg.get("train_end", "2025-09-30")
        self.test_start: str = dates_cfg.get("test_start", "2025-10-01")
        self.test_end: str = dates_cfg.get("test_end", "2025-12-31")

        # Paths
        paths_cfg = config.get("paths", {})
        self.features_path: str = paths_cfg.get("features_data", "data/features")

        logger.info(
            "FeaturePipeline initialised – train end %s, test %s → %s",
            self.train_end,
            self.test_start,
            self.test_end,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        """Strip timezone info from DatetimeIndex for consistent merging.

        Args:
            df: DataFrame whose index may be timezone-aware.

        Returns:
            DataFrame with timezone-naive DatetimeIndex.
        """
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df

    # ------------------------------------------------------------------
    # Single-ticker feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        fundamental_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame,
        sentiment_data: Dict[str, pd.DataFrame],
        ticker: str,
    ) -> pd.DataFrame:
        """Build the complete feature matrix for a single stock.

        Steps:
            1. Generate technical features from OHLCV.
            2. Generate and align fundamental features.
            3. Generate macro features (shared across tickers).
            4. Generate sentiment features.
            5. Merge all on Date index.
            6. Forward-fill then back-fill intermittent gaps.
            7. Drop leading rows that are fully NaN due to rolling
               warm-up, then fill any remaining NaN with 0.

        Args:
            ohlcv_data: Dict mapping ticker → OHLCV DataFrame.
            fundamental_data: Dict mapping ticker → quarterly
                fundamental DataFrame.
            macro_data: Macro indicator DataFrame (shared across
                tickers).
            sentiment_data: Dict mapping ticker → sentiment DataFrame.
            ticker: Ticker symbol to build features for.

        Returns:
            DataFrame indexed by Date with all feature columns and a
            ``Target`` column.
        """
        logger.info("Building feature matrix for %s", ticker)

        # --- 1. Technical features ---
        ohlcv_df = ohlcv_data.get(ticker)
        if ohlcv_df is None:
            raise ValueError(f"No OHLCV data found for ticker {ticker}")

        # Normalize timezone on OHLCV before generating features
        ohlcv_df = self._normalize_index(ohlcv_df)

        tech_df = self.technical_engineer.generate_all_features(ohlcv_df)
        tech_df = self._normalize_index(tech_df)

        # Separate Target (created by technical engineer)
        target = tech_df[["Target"]].copy() if "Target" in tech_df.columns else None
        tech_features = tech_df.drop(columns=["Target"], errors="ignore")

        # --- 2. Fundamental features ---
        fund_features = pd.DataFrame(index=tech_features.index)
        fund_df = fundamental_data.get(ticker)
        if fund_df is not None and not fund_df.empty:
            fund_df = self._normalize_index(fund_df)
            close_prices = ohlcv_df["Adj_Close"] if "Adj_Close" in ohlcv_df.columns else None
            fund_features = self.fundamental_engineer.generate_all_features(
                fund_df,
                daily_dates=tech_features.index,
                close_prices=close_prices,
            )
            fund_features = self._normalize_index(fund_features)
        else:
            logger.warning("No fundamental data for %s; skipping", ticker)

        # --- 3. Macro features ---
        macro_features = pd.DataFrame(index=tech_features.index)
        if macro_data is not None and not macro_data.empty:
            macro_data_clean = self._normalize_index(macro_data)
            macro_all = self.macro_engineer.generate_all_features(macro_data_clean)
            macro_all = self._normalize_index(macro_all)
            # Align to technical feature dates
            macro_features = macro_all.reindex(tech_features.index, method="ffill")
        else:
            logger.warning("No macro data provided; skipping")

        # --- 4. Sentiment features ---
        sent_features = pd.DataFrame(index=tech_features.index)
        sent_df = sentiment_data.get(ticker)
        if sent_df is not None and not sent_df.empty:
            sent_df = self._normalize_index(sent_df)
            sent_all = self.sentiment_engineer.generate_all_features(sent_df)
            sent_all = self._normalize_index(sent_all)
            sent_features = sent_all.reindex(tech_features.index, method="ffill")
        else:
            logger.warning("No sentiment data for %s; skipping", ticker)

        # --- 5. Merge all features ---
        merged = pd.concat(
            [tech_features, fund_features, macro_features, sent_features],
            axis=1,
        )

        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # Attach target
        if target is not None:
            merged["Target"] = target["Target"]

        # --- 6. Handle missing values ---
        # Forward-fill then back-fill to handle both leading and
        # trailing NaN (e.g. sparse sentiment data)
        merged = merged.ffill().bfill()

        # Drop leading warm-up rows where a majority of features are NaN
        feature_cols = [c for c in merged.columns if c != "Target"]
        nan_ratio = merged[feature_cols].isna().mean(axis=1)
        first_valid_idx = nan_ratio[nan_ratio < 0.5].index.min()
        if first_valid_idx is not None:
            merged = merged.loc[first_valid_idx:]

        # Fill any remaining NaN with 0 (safe for normalised features)
        merged[feature_cols] = merged[feature_cols].fillna(0)

        logger.info(
            "Feature matrix for %s – %d features, %d rows",
            ticker,
            len(feature_cols),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # All-tickers feature matrices
    # ------------------------------------------------------------------

    def build_all_feature_matrices(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        fundamental_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame,
        sentiment_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Build feature matrices for all stocks.

        Iterates over every ticker in ``ohlcv_data``, builds the
        feature matrix, and saves each to
        ``data/features/{ticker}_features.parquet``.

        Args:
            ohlcv_data: Dict mapping ticker → OHLCV DataFrame.
            fundamental_data: Dict mapping ticker → quarterly
                fundamental DataFrame.
            macro_data: Macro indicator DataFrame.
            sentiment_data: Dict mapping ticker → sentiment DataFrame.

        Returns:
            Dict mapping ticker → feature DataFrame (with Target).
        """
        all_matrices: Dict[str, pd.DataFrame] = {}
        tickers = list(ohlcv_data.keys())

        # Ensure output directory exists
        out_dir = Path(self.features_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for ticker in tqdm(tickers, desc="Building feature matrices"):
            try:
                feature_df = self.build_feature_matrix(
                    ohlcv_data=ohlcv_data,
                    fundamental_data=fundamental_data,
                    macro_data=macro_data,
                    sentiment_data=sentiment_data,
                    ticker=ticker,
                )
                all_matrices[ticker] = feature_df

                # Persist to parquet
                out_path = out_dir / f"{ticker}_features.parquet"
                feature_df.to_parquet(out_path, index=True)
                logger.info("Saved features for %s → %s", ticker, out_path)

            except Exception as exc:
                logger.error("Failed to build features for %s: %s", ticker, exc)

        logger.info(
            "Built feature matrices for %d / %d tickers",
            len(all_matrices),
            len(tickers),
        )
        return all_matrices

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        scaler: Optional[RobustScaler] = None,
    ) -> Tuple[pd.DataFrame, RobustScaler]:
        """Apply robust scaling to features (excluding Target).

        Uses ``sklearn.preprocessing.RobustScaler`` which centres on
        the median and scales by the inter-quartile range, making it
        less sensitive to financial-data outliers.

        Args:
            df: Feature DataFrame with optional ``Target`` column.
            fit: If ``True`` (training), fit a new scaler.  If
                ``False`` (inference), use the provided *scaler*.
            scaler: Pre-fitted ``RobustScaler`` for inference mode.

        Returns:
            Tuple of (scaled DataFrame, fitted RobustScaler).
        """
        feature_cols = [c for c in df.columns if c != "Target"]
        self.feature_names_ = feature_cols

        if fit:
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(df[feature_cols])
            self.scaler = scaler
            logger.info(
                "Fitted RobustScaler on %d features, %d samples",
                len(feature_cols),
                len(df),
            )
        else:
            if scaler is None:
                scaler = self.scaler
            if scaler is None:
                raise ValueError(
                    "No fitted scaler available.  Call with fit=True first."
                )
            scaled_values = scaler.transform(df[feature_cols])

        scaled_df = pd.DataFrame(
            scaled_values, index=df.index, columns=feature_cols
        )

        # Preserve Target column unscaled
        if "Target" in df.columns:
            scaled_df["Target"] = df["Target"].values

        return scaled_df, scaler

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------

    def split_train_test(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets by date.

        Train period: ``start_date`` → ``train_end`` (inclusive).
        Test period: ``test_start`` → ``test_end`` (inclusive).

        Args:
            df: Feature DataFrame indexed by DatetimeIndex.

        Returns:
            Tuple of ``(train_df, test_df)``.
        """
        train_end_ts = pd.Timestamp(self.train_end)
        test_start_ts = pd.Timestamp(self.test_start)
        test_end_ts = pd.Timestamp(self.test_end)

        train_df = df.loc[df.index <= train_end_ts].copy()
        test_df = df.loc[
            (df.index >= test_start_ts) & (df.index <= test_end_ts)
        ].copy()

        logger.info(
            "Train/test split – train %d rows (%s → %s), test %d rows (%s → %s)",
            len(train_df),
            train_df.index.min() if len(train_df) else "N/A",
            train_df.index.max() if len(train_df) else "N/A",
            len(test_df),
            test_df.index.min() if len(test_df) else "N/A",
            test_df.index.max() if len(test_df) else "N/A",
        )
        return train_df, test_df

    # ------------------------------------------------------------------
    # Feature name accessor
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names (excluding Target).

        Returns:
            List of feature column name strings.  Only available
            after :meth:`scale_features` or
            :meth:`build_feature_matrix` has been called.
        """
        return list(self.feature_names_)

    # ------------------------------------------------------------------
    # Convenience: full run
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        fundamental_data: Dict[str, pd.DataFrame],
        macro_data: pd.DataFrame,
        sentiment_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full feature pipeline end-to-end.

        Steps:
            1. Build feature matrices for all tickers.
            2. Stack vertically with a ``Ticker`` column.
            3. Scale features (fit on training data only).
            4. Split into train and test sets.
            5. Persist to ``data/features/``.

        Args:
            ohlcv_data: Dict mapping ticker → OHLCV DataFrame.
            fundamental_data: Dict mapping ticker → quarterly
                fundamental DataFrame.
            macro_data: Macro indicator DataFrame.
            sentiment_data: Dict mapping ticker → sentiment DataFrame.

        Returns:
            Tuple of ``(train_df, test_df)`` — scaled feature
            DataFrames with ``Target`` and ``Ticker`` columns.
        """
        logger.info("Running full feature pipeline")

        # Step 1: Build per-ticker feature matrices
        all_matrices = self.build_all_feature_matrices(
            ohlcv_data, fundamental_data, macro_data, sentiment_data
        )

        if not all_matrices:
            raise RuntimeError("No feature matrices were built for any ticker")

        # Step 2: Stack with ticker identifier
        frames = []
        for ticker, feat_df in all_matrices.items():
            tmp = feat_df.copy()
            tmp["Ticker"] = ticker
            frames.append(tmp)

        stacked = pd.concat(frames, axis=0)
        stacked = stacked.sort_index()

        # Record feature names (before scaling)
        self.feature_names_ = [
            c for c in stacked.columns if c not in ("Target", "Ticker")
        ]

        # Step 3: Train/test split BEFORE scaling (fit scaler on train only)
        train_raw, test_raw = self.split_train_test(stacked)

        # Step 4: Scale features
        # Fit on training data
        ticker_col_train = train_raw["Ticker"].copy()
        target_col_train = train_raw["Target"].copy() if "Target" in train_raw.columns else None
        train_features_only = train_raw.drop(columns=["Ticker", "Target"], errors="ignore")
        train_scaled, fitted_scaler = self.scale_features(
            train_features_only, fit=True
        )
        train_scaled["Ticker"] = ticker_col_train.values
        if target_col_train is not None:
            train_scaled["Target"] = target_col_train.values

        # Transform test data with fitted scaler
        ticker_col_test = test_raw["Ticker"].copy()
        target_col_test = test_raw["Target"].copy() if "Target" in test_raw.columns else None
        test_features_only = test_raw.drop(columns=["Ticker", "Target"], errors="ignore")
        test_scaled, _ = self.scale_features(
            test_features_only, fit=False, scaler=fitted_scaler
        )
        test_scaled["Ticker"] = ticker_col_test.values
        if target_col_test is not None:
            test_scaled["Target"] = target_col_test.values

        # Step 5: Persist
        out_dir = Path(self.features_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "features_train.parquet"
        test_path = out_dir / "features_test.parquet"
        train_scaled.to_parquet(train_path, index=True)
        test_scaled.to_parquet(test_path, index=True)

        logger.info(
            "Feature pipeline complete – train %d rows → %s, test %d rows → %s",
            len(train_scaled),
            train_path,
            len(test_scaled),
            test_path,
        )
        return train_scaled, test_scaled
