"""
Feature selection module for the Multi-Dimensional Return Forecasting system.

Provides Recursive Feature Elimination (RFE), importance-based
selection, and collinearity filtering to reduce overfitting and
improve model generalisation on financial return data.

Example:
    >>> from src.models.feature_selection import FeatureSelector
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> selector = FeatureSelector(config)
    >>> selected, ranking = selector.select_features_rfe(X, y)
    >>> print(selected[:5])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Feature selection using RFE, importance ranking, and collinearity filtering.

    The selector combines three complementary strategies:

    1.  **Recursive Feature Elimination (RFE)** — iteratively removes
        the least important features according to a tree-based estimator.
    2.  **Importance-based top-N selection** — extracts
        ``feature_importances_`` (or ``coef_``) from a fitted model and
        returns the most predictive features.
    3.  **Collinearity removal** — drops features whose pair-wise
        Pearson correlation exceeds a configurable threshold.

    Attributes:
        n_features_to_select: Target number of features after RFE.
        step: Number of features to eliminate per RFE iteration.
        collinearity_threshold: Maximum allowed absolute Pearson
            correlation between any two retained features.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise from the full pipeline configuration dictionary.

        Reads from ``config["model"]["feature_selection"]`` (actual
        ``config.yaml``) and ``config["feature_selection"]``
        (``ARCHITECTURE.md`` spec).

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        fs_cfg = config.get("model", {}).get("feature_selection", {})
        arch_cfg = config.get("feature_selection", {})

        self.n_features_to_select: int = (
            fs_cfg.get("n_features_to_select")
            or arch_cfg.get("max_features", 30)
        )
        self.step: int = fs_cfg.get("step") or arch_cfg.get("step", 5)
        self.collinearity_threshold: float = 0.9
        self.min_features: int = arch_cfg.get("min_features", 10)

        logger.info(
            "FeatureSelector initialised — target_n=%d, step=%d, "
            "collinearity_threshold=%.2f",
            self.n_features_to_select,
            self.step,
            self.collinearity_threshold,
        )

    # ------------------------------------------------------------------
    # RFE
    # ------------------------------------------------------------------

    def select_features_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[object] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Perform Recursive Feature Elimination.

        When no *estimator* is supplied a low-complexity LightGBM
        regressor is used as the base model.  The function removes
        ``self.step`` features per iteration until
        ``self.n_features_to_select`` remain.

        Args:
            X: Feature DataFrame of shape ``(n_samples, n_features)``.
            y: Target Series of shape ``(n_samples,)``.
            estimator: Sklearn-compatible estimator with
                ``feature_importances_`` or ``coef_`` after fitting.
                Defaults to a lightweight LightGBM regressor.

        Returns:
            Tuple of ``(selected_feature_names, feature_ranking)``
            where *feature_ranking* is an integer array where ``1``
            marks selected features.
        """
        if estimator is None:
            estimator = self._default_rfe_estimator()

        n_to_select = min(self.n_features_to_select, X.shape[1])

        logger.info(
            "Running RFE: %d features → %d, step=%d, estimator=%s",
            X.shape[1],
            n_to_select,
            self.step,
            type(estimator).__name__,
        )

        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_to_select,
            step=self.step,
            verbose=0,
        )

        rfe.fit(X.values, y.values)

        selected_mask: np.ndarray = rfe.support_
        ranking: np.ndarray = rfe.ranking_
        selected_names = [
            col for col, keep in zip(X.columns, selected_mask) if keep
        ]

        logger.info(
            "RFE complete — selected %d / %d features",
            len(selected_names),
            X.shape[1],
        )

        return selected_names, ranking

    # ------------------------------------------------------------------
    # Feature importance extraction
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_importance(
        model: object, feature_names: List[str]
    ) -> pd.DataFrame:
        """Extract and rank feature importances from a fitted model.

        Works with tree-based models (LightGBM, XGBoost) that expose
        ``feature_importances_`` and with linear models (Ridge, Lasso)
        that expose ``coef_``.

        Args:
            model: A fitted sklearn-compatible estimator.
            feature_names: Feature column names in the order used
                during training.

        Returns:
            DataFrame with columns ``Feature``, ``Importance``, and
            ``Rank`` sorted by descending importance.

        Raises:
            AttributeError: If the model does not expose importance
                or coefficient attributes.
        """
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        else:
            raise AttributeError(
                f"Model {type(model).__name__} does not expose "
                "feature_importances_ or coef_."
            )

        # Normalise to [0, 1] for comparability
        total = importances.sum()
        if total > 0:
            importances_norm = importances / total
        else:
            importances_norm = importances

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances_norm}
        )
        importance_df = importance_df.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)
        importance_df["Rank"] = np.arange(1, len(importance_df) + 1)

        return importance_df

    # ------------------------------------------------------------------
    # Top-N selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_top_features(
        importance_df: pd.DataFrame, n: int
    ) -> List[str]:
        """Select the top *n* features by normalised importance.

        Args:
            importance_df: DataFrame returned by
                :meth:`get_feature_importance`.
            n: Number of features to retain.

        Returns:
            List of feature name strings.
        """
        n = min(n, len(importance_df))
        top = importance_df.nlargest(n, "Importance")
        selected = top["Feature"].tolist()
        logger.info("Selected top %d features by importance", len(selected))
        return selected

    # ------------------------------------------------------------------
    # Collinearity removal
    # ------------------------------------------------------------------

    def remove_collinear_features(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Drop features whose pair-wise correlation exceeds *threshold*.

        For every pair of features with ``|ρ| > threshold`` the feature
        with the **smaller** mean absolute correlation to all other
        features is retained (i.e. the more "unique" one is kept).

        Args:
            X: Feature DataFrame.
            threshold: Maximum allowed absolute Pearson correlation.
                Defaults to ``self.collinearity_threshold`` (0.9).

        Returns:
            List of feature names to **keep** (i.e. the non-collinear
            subset).
        """
        if threshold is None:
            threshold = self.collinearity_threshold

        logger.info(
            "Removing collinear features with |ρ| > %.2f from %d features",
            threshold,
            X.shape[1],
        )

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        to_drop: set = set()
        for col in upper_tri.columns:
            high_corr = upper_tri.index[upper_tri[col] > threshold].tolist()
            for hc in high_corr:
                if hc in to_drop:
                    continue
                # Drop the feature with the higher mean correlation
                mean_corr_col = corr_matrix[col].mean()
                mean_corr_hc = corr_matrix[hc].mean()
                drop_candidate = col if mean_corr_col > mean_corr_hc else hc
                to_drop.add(drop_candidate)

        features_to_keep = [c for c in X.columns if c not in to_drop]
        logger.info(
            "Collinearity filter: kept %d / %d features (dropped %d)",
            len(features_to_keep),
            X.shape[1],
            len(to_drop),
        )

        return features_to_keep

    # ------------------------------------------------------------------
    # Cross-symbol robustness filter
    # ------------------------------------------------------------------

    def cross_symbol_filter(
        self,
        feature_matrices: Dict[str, pd.DataFrame],
        target_col: str = "Target",
        min_importance: float = 0.01,
    ) -> List[str]:
        """Retain only features that are important across **all** tickers.

        For each ticker a lightweight LightGBM model is fitted and its
        feature importances are extracted.  A feature is kept only if
        its normalised importance exceeds *min_importance* in **every**
        ticker — this prevents the model from overfitting to a single
        high-beta stock.

        Args:
            feature_matrices: Dictionary mapping ticker → feature
                DataFrame (with ``Target`` column).
            target_col: Name of the target column.
            min_importance: Minimum normalised importance threshold.

        Returns:
            List of cross-symbol robust feature names.
        """
        logger.info(
            "Running cross-symbol importance filter (threshold=%.4f) "
            "across %d tickers",
            min_importance,
            len(feature_matrices),
        )

        estimator = self._default_rfe_estimator()

        importance_per_ticker: Dict[str, pd.DataFrame] = {}

        for ticker, df in tqdm(
            feature_matrices.items(), desc="Cross-symbol filter"
        ):
            feature_cols = [c for c in df.columns if c != target_col]
            if not feature_cols or target_col not in df.columns:
                logger.warning(
                    "Skipping %s — no features or missing target", ticker
                )
                continue

            df_clean = df.dropna(subset=[target_col])
            X_t = df_clean[feature_cols]
            y_t = df_clean[target_col]

            if len(X_t) < 50:
                logger.warning(
                    "Skipping %s — only %d samples", ticker, len(X_t)
                )
                continue

            try:
                from sklearn.base import clone

                est = clone(estimator)
            except Exception:
                est = estimator.__class__(**estimator.get_params())

            est.fit(X_t.values, y_t.values)
            imp_df = self.get_feature_importance(est, feature_cols)
            importance_per_ticker[ticker] = imp_df

        if not importance_per_ticker:
            logger.warning("No ticker data available for cross-symbol filter")
            return []

        # Keep features above threshold in ALL tickers
        common_features: Optional[set] = None
        for ticker, imp_df in importance_per_ticker.items():
            above_thresh = set(
                imp_df.loc[imp_df["Importance"] >= min_importance, "Feature"]
            )
            if common_features is None:
                common_features = above_thresh
            else:
                common_features &= above_thresh

        robust_features = sorted(common_features) if common_features else []
        logger.info(
            "Cross-symbol filter: %d features pass threshold across all tickers",
            len(robust_features),
        )

        return robust_features

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[object] = None,
        remove_collinear: bool = True,
    ) -> List[str]:
        """Execute the full feature-selection pipeline on a single dataset.

        Steps:
            1. Remove highly collinear features.
            2. Run RFE on the remaining features.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            estimator: Optional estimator for RFE.
            remove_collinear: Whether to apply collinearity removal
                before RFE.

        Returns:
            List of selected feature names.
        """
        feature_cols = list(X.columns)

        if remove_collinear:
            feature_cols = self.remove_collinear_features(X[feature_cols])
            X = X[feature_cols]

        selected, _ = self.select_features_rfe(X, y, estimator=estimator)
        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_rfe_estimator():
        """Create a lightweight LightGBM regressor for RFE.

        The model uses low complexity to keep RFE fast and to avoid
        overfitting the importance rankings themselves.

        Returns:
            Unfitted ``LGBMRegressor`` with conservative parameters.
        """
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1,
                random_state=42,
            )
        except ImportError:
            logger.warning(
                "LightGBM not available; falling back to "
                "sklearn GradientBoostingRegressor for RFE"
            )
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
