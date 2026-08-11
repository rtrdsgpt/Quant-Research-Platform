"""
Return prediction model for the Multi-Dimensional Return Forecasting system.

This module provides the :class:`ReturnForecaster` — the central class
that orchestrates model creation, walk-forward training, feature
selection, ensemble prediction, and model persistence for per-stock
and multi-stock return forecasting.

Supported model types:
    - **LightGBM** (primary) — fast histogram-based GBDT
    - **XGBoost** — secondary ensemble member
    - **Ridge** — highly-regularised linear baseline

Example:
    >>> from src.models.forecaster import ReturnForecaster
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> forecaster = ReturnForecaster(config)
    >>> results = forecaster.train_all_stocks(feature_matrices)
    >>> predictions = forecaster.predict_all_stocks(feature_matrices)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.feature_selection import FeatureSelector
from src.models.walk_forward import WalkForwardValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReturnForecaster:
    """Manages training, prediction, and persistence for stock return models.

    The forecaster supports three workflows:

    1.  **Single-stock training** — train one model type for a single
        ticker via :meth:`train_single_stock`.
    2.  **Ensemble training** — train LightGBM + XGBoost + Ridge and
        combine predictions with :meth:`train_ensemble`.
    3.  **Batch training** — iterate over all 6 tickers with
        :meth:`train_all_stocks`.

    Internally every training call performs:
        a. Feature selection (RFE via :class:`FeatureSelector`).
        b. Walk-forward cross-validation (via :class:`WalkForwardValidator`).
        c. Final re-fit on the full training data.

    Attributes:
        config: Full pipeline configuration dictionary.
        walk_forward: Walk-forward validator instance.
        feature_selector: Feature selector instance.
        models: Dictionary of **unfitted** model templates keyed by
            model type name.
        fitted_models: Dictionary keyed by ``(ticker, model_type)``
            holding fitted model objects.
        scalers: Per-ticker :class:`StandardScaler` instances.
        selected_features: Per-ticker lists of selected feature names.
        ensemble_weights: Per-ticker ensemble weight dictionaries.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise from the full pipeline configuration.

        Creates internal helper objects (:class:`WalkForwardValidator`,
        :class:`FeatureSelector`) and registers base model factories.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config
        self.walk_forward = WalkForwardValidator(config)
        self.feature_selector = FeatureSelector(config)

        # Model storage
        self.models: Dict[str, Any] = {}
        self.fitted_models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.selected_features: Dict[str, List[str]] = {}
        self.ensemble_weights: Dict[str, Dict[str, float]] = {}

        # Configuration shortcuts
        model_cfg = config.get("model", {})
        self.primary_model_type: str = model_cfg.get("type", "lightgbm")
        self.use_ensemble: bool = model_cfg.get("ensemble", False)
        self.ensemble_model_names: List[str] = model_cfg.get(
            "ensemble_models", ["lightgbm", "xgboost", "ridge"]
        )

        # Paths
        paths_cfg = config.get("paths", {})
        self.models_dir: str = paths_cfg.get("models_dir", "models")

        # Early stopping
        self.early_stopping_rounds: int = model_cfg.get("lightgbm", {}).get(
            "early_stopping_rounds", 50
        )

        logger.info(
            "ReturnForecaster initialised — primary=%s, ensemble=%s, "
            "models=%s",
            self.primary_model_type,
            self.use_ensemble,
            self.ensemble_model_names,
        )

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------

    def _create_lightgbm(self) -> Any:
        """Create a LightGBM regressor from config hyperparameters.

        Returns:
            Unfitted ``LGBMRegressor``.
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required. Install with: pip install lightgbm"
            ) from exc

        lgb_cfg = self.config.get("model", {}).get("lightgbm", {})
        return lgb.LGBMRegressor(
            n_estimators=lgb_cfg.get("n_estimators", 500),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            max_depth=lgb_cfg.get("max_depth", 6),
            num_leaves=lgb_cfg.get("num_leaves", 31),
            min_child_samples=lgb_cfg.get("min_child_samples", 20),
            subsample=lgb_cfg.get("subsample", 0.8),
            colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
            reg_alpha=lgb_cfg.get("reg_alpha", 0.1),
            reg_lambda=lgb_cfg.get("reg_lambda", 0.1),
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        )

    def _create_xgboost(self) -> Any:
        """Create an XGBoost regressor from config hyperparameters.

        Returns:
            Unfitted ``XGBRegressor``.
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            ) from exc

        xgb_cfg = self.config.get("model", {}).get("xgboost", {})
        return xgb.XGBRegressor(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            max_depth=xgb_cfg.get("max_depth", 6),
            min_child_weight=xgb_cfg.get("min_child_weight", 5),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            reg_alpha=xgb_cfg.get("reg_alpha", 0.1),
            reg_lambda=xgb_cfg.get("reg_lambda", 0.1),
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )

    def _create_ridge(self) -> Ridge:
        """Create a Ridge regression model from config hyperparameters.

        Returns:
            Unfitted ``Ridge`` estimator.
        """
        ridge_cfg = self.config.get("model", {}).get("ridge", {})
        return Ridge(
            alpha=ridge_cfg.get("alpha", 1.0),
            fit_intercept=True,
            random_state=42,
        )

    def create_model(self, model_type: str = "lightgbm") -> Any:
        """Factory method returning a fresh unfitted model.

        Args:
            model_type: One of ``'lightgbm'``, ``'xgboost'``, or
                ``'ridge'``.

        Returns:
            Unfitted sklearn-compatible estimator.

        Raises:
            ValueError: If *model_type* is not recognised.
        """
        model_type = model_type.lower().strip()
        if model_type == "lightgbm":
            return self._create_lightgbm()
        if model_type == "xgboost":
            return self._create_xgboost()
        if model_type == "ridge":
            return self._create_ridge()
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Supported: lightgbm, xgboost, ridge"
        )

    # ------------------------------------------------------------------
    # Single-stock training
    # ------------------------------------------------------------------

    def train_single_stock(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        ticker: str,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train a single model for one stock.

        Steps:
            1. Feature selection via RFE.
            2. Walk-forward cross-validation for evaluation.
            3. Final fit on all training data.
            4. Store fitted model, scaler, and selected features.

        Args:
            X_train: Feature DataFrame.
            y_train: Target Series.
            ticker: Ticker symbol (e.g. ``'RELIANCE.NS'``).
            model_type: Model type to use.  Defaults to the primary
                model from config.

        Returns:
            Dictionary with keys:
                - ``model``: fitted model object
                - ``features``: list of selected feature names
                - ``cv_scores``: walk-forward CV metric dictionary
                - ``feature_importance``: importance DataFrame
        """
        if model_type is None:
            model_type = self.primary_model_type

        logger.info(
            "Training %s model for %s — %d samples, %d features",
            model_type,
            ticker,
            len(X_train),
            X_train.shape[1],
        )

        # --- 1. Feature selection ---
        selected_features = self._run_feature_selection(X_train, y_train)
        X_selected = X_train[selected_features]

        # --- 2. Scale features ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        y_arr = y_train.values

        # --- 3. Walk-forward cross-validation ---
        cv_model = self.create_model(model_type)
        cv_scores = self.walk_forward.cross_validate(
            model=cv_model,
            X=X_scaled,
            y=y_arr,
            metrics=["mse", "mae", "rmse", "r2", "direction_accuracy", "ic"],
            early_stopping_rounds=self.early_stopping_rounds,
        )

        # --- 4. Final model training on all data ---
        final_model = self.create_model(model_type)
        final_model = self._fit_with_early_stopping(
            final_model, X_scaled, y_arr, model_type
        )

        # --- 5. Feature importance ---
        importance_df = self.feature_selector.get_feature_importance(
            final_model, selected_features
        )

        # --- 6. Store artefacts ---
        self.fitted_models.setdefault(ticker, {})[model_type] = final_model
        self.scalers[ticker] = scaler
        self.selected_features[ticker] = selected_features

        logger.info(
            "Finished training %s for %s — CV R²=%.4f, "
            "Direction Acc=%.2f%%",
            model_type,
            ticker,
            np.nanmean(cv_scores.get("r2", [0.0])),
            np.nanmean(cv_scores.get("direction_accuracy", [0.0])) * 100,
        )

        return {
            "model": final_model,
            "features": selected_features,
            "cv_scores": cv_scores,
            "feature_importance": importance_df,
        }

    # ------------------------------------------------------------------
    # Ensemble training
    # ------------------------------------------------------------------

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        ticker: str,
    ) -> Dict[str, Any]:
        """Train an ensemble of models for one stock.

        Trains each model type listed in ``config.model.ensemble_models``
        and computes per-model weights based on walk-forward CV
        performance (inverse MSE weighting).

        Args:
            X_train: Feature DataFrame.
            y_train: Target Series.
            ticker: Ticker symbol.

        Returns:
            Dictionary with keys:
                - ``models``: dict mapping model_type → fitted model
                - ``weights``: dict mapping model_type → ensemble weight
                - ``features``: list of selected feature names
                - ``cv_scores``: dict mapping model_type → CV scores
        """
        logger.info(
            "Training ensemble for %s — models: %s",
            ticker,
            self.ensemble_model_names,
        )

        # --- 1. Feature selection (shared across ensemble members) ---
        selected_features = self._run_feature_selection(X_train, y_train)
        X_selected = X_train[selected_features]

        # --- 2. Scale features ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        y_arr = y_train.values

        models_fitted: Dict[str, Any] = {}
        all_cv_scores: Dict[str, Dict[str, List[float]]] = {}
        cv_mse_means: Dict[str, float] = {}

        for mtype in self.ensemble_model_names:
            logger.info("  Training ensemble member: %s", mtype)
            try:
                # Walk-forward CV
                cv_model = self.create_model(mtype)
                cv_scores = self.walk_forward.cross_validate(
                    model=cv_model,
                    X=X_scaled,
                    y=y_arr,
                    metrics=["mse", "mae", "r2", "direction_accuracy", "ic"],
                    early_stopping_rounds=(
                        self.early_stopping_rounds
                        if mtype != "ridge"
                        else None
                    ),
                )
                all_cv_scores[mtype] = cv_scores

                mse_vals = [v for v in cv_scores.get("mse", [1.0]) if np.isfinite(v)]
                cv_mse_means[mtype] = np.mean(mse_vals) if mse_vals else 1.0

                # Final fit
                final_model = self.create_model(mtype)
                final_model = self._fit_with_early_stopping(
                    final_model, X_scaled, y_arr, mtype
                )
                models_fitted[mtype] = final_model

            except Exception as exc:
                logger.error(
                    "Failed to train %s for %s: %s",
                    mtype,
                    ticker,
                    exc,
                    exc_info=True,
                )

        # --- 3. Compute ensemble weights (inverse-MSE) ---
        weights = self._compute_ensemble_weights(cv_mse_means)

        # --- 4. Store artefacts ---
        self.fitted_models[ticker] = models_fitted
        self.scalers[ticker] = scaler
        self.selected_features[ticker] = selected_features
        self.ensemble_weights[ticker] = weights

        logger.info(
            "Ensemble for %s complete — weights: %s",
            ticker,
            {k: f"{v:.3f}" for k, v in weights.items()},
        )

        return {
            "models": models_fitted,
            "weights": weights,
            "features": selected_features,
            "cv_scores": all_cv_scores,
        }

    # ------------------------------------------------------------------
    # Batch training
    # ------------------------------------------------------------------

    def train_all_stocks(
        self, feature_matrices: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Train models for every stock in the universe.

        For each ticker the method:
            1. Splits into ``X`` (features) and ``y`` (``Target``).
            2. Runs feature selection.
            3. Trains either a single model or an ensemble (per config).
            4. Stores fitted artefacts.

        Args:
            feature_matrices: Dictionary mapping ticker → feature
                DataFrame.  Each DataFrame must contain a ``Target``
                column.

        Returns:
            Dictionary mapping ticker → training result dictionary.
        """
        results: Dict[str, Dict] = {}

        for ticker, df in tqdm(
            feature_matrices.items(),
            desc="Training models",
        ):
            logger.info(
                "=" * 60 + "\nTraining for %s (%d rows)\n" + "=" * 60,
                ticker,
                len(df),
            )

            try:
                X_train, y_train = self._split_X_y(df)

                if self.use_ensemble:
                    result = self.train_ensemble(X_train, y_train, ticker)
                else:
                    result = self.train_single_stock(
                        X_train, y_train, ticker
                    )

                results[ticker] = result

            except Exception as exc:
                logger.error(
                    "Failed to train model for %s: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )
                results[ticker] = {"error": str(exc)}

        logger.info(
            "Completed training for %d / %d tickers",
            sum(1 for v in results.values() if "error" not in v),
            len(feature_matrices),
        )
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, ticker: str) -> np.ndarray:
        """Predict returns for a single stock.

        Uses the fitted model(s) for *ticker* and applies the same
        feature selection and scaling that were used during training.

        Args:
            X: Feature DataFrame (may contain extra columns).
            ticker: Ticker symbol whose model to use.

        Returns:
            1-D array of predicted returns.

        Raises:
            KeyError: If no model has been trained for *ticker*.
        """
        if ticker not in self.fitted_models:
            raise KeyError(
                f"No fitted model for ticker '{ticker}'. "
                "Call train_all_stocks() first."
            )

        # Select the same features used during training
        features = self.selected_features[ticker]
        X_sub = X[features].copy()

        # Scale
        scaler = self.scalers.get(ticker)
        if scaler is not None:
            X_scaled = scaler.transform(X_sub)
        else:
            X_scaled = X_sub.values

        models = self.fitted_models[ticker]

        # Ensemble prediction
        if isinstance(models, dict) and len(models) > 1:
            weights = self.ensemble_weights.get(ticker, {})
            return self._predict_ensemble(models, X_scaled, weights)

        # Single model prediction
        if isinstance(models, dict):
            model = next(iter(models.values()))
        else:
            model = models

        return model.predict(X_scaled)

    def predict_all_stocks(
        self, feature_matrices: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Generate predictions for every stock in the universe.

        Args:
            feature_matrices: Dictionary mapping ticker → feature
                DataFrame with a ``Target`` column.

        Returns:
            Dictionary mapping ticker → DataFrame with columns
            ``Date``, ``Predicted_Return``, ``Actual_Return``.
        """
        predictions: Dict[str, pd.DataFrame] = {}

        for ticker, df in feature_matrices.items():
            try:
                X, y = self._split_X_y(df)
                y_pred = self.predict(X, ticker)

                pred_df = pd.DataFrame(
                    {
                        "Predicted_Return": y_pred,
                        "Actual_Return": y.values,
                    },
                    index=df.index[: len(y_pred)],
                )
                pred_df.index.name = "Date"
                predictions[ticker] = pred_df

                logger.info(
                    "Predictions for %s — %d rows", ticker, len(pred_df)
                )

            except Exception as exc:
                logger.error(
                    "Prediction failed for %s: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )

        return predictions

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute a comprehensive set of evaluation metrics.

        Metrics computed:
            - **MSE**: Mean Squared Error
            - **RMSE**: Root Mean Squared Error
            - **MAE**: Mean Absolute Error
            - **R2**: R-squared
            - **Direction_Accuracy**: Hit ratio (fraction of correct
              sign predictions)
            - **IC**: Information Coefficient (Spearman ρ)

        Args:
            y_true: Actual target values.
            y_pred: Predicted values.

        Returns:
            Dictionary mapping metric names to scalar values.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # Direction accuracy (hit ratio)
        if len(y_true) > 0:
            direction_acc = float(
                np.mean(np.sign(y_true) == np.sign(y_pred))
            )
        else:
            direction_acc = 0.0

        # Information Coefficient (Spearman correlation)
        if len(y_true) >= 3:
            ic, _ = stats.spearmanr(y_true, y_pred)
            ic = float(ic) if np.isfinite(ic) else 0.0
        else:
            ic = 0.0

        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Direction_Accuracy": direction_acc,
            "IC": ic,
        }

        logger.info(
            "Evaluation — MSE=%.6f, RMSE=%.6f, MAE=%.6f, R²=%.4f, "
            "DirAcc=%.2f%%, IC=%.4f",
            mse,
            rmse,
            mae,
            r2,
            direction_acc * 100,
            ic,
        )

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self, path: Optional[str] = None) -> None:
        """Serialise all fitted models, scalers, and metadata to disk.

        Creates the directory structure::

            {path}/
            ├── {ticker}/
            │   ├── {model_type}.joblib
            │   ├── scaler.joblib
            │   └── metadata.joblib
            └── ...

        Args:
            path: Root directory for model artefacts.  Defaults to
                ``config.paths.models_dir``.
        """
        if path is None:
            path = self.models_dir

        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        for ticker in self.fitted_models:
            ticker_dir = base / ticker.replace(".", "_")
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Save models
            models = self.fitted_models[ticker]
            if isinstance(models, dict):
                for mtype, model in models.items():
                    model_path = ticker_dir / f"{mtype}.joblib"
                    joblib.dump(model, model_path)
                    logger.info("Saved %s model → %s", mtype, model_path)
            else:
                model_path = ticker_dir / "model.joblib"
                joblib.dump(models, model_path)

            # Save scaler
            if ticker in self.scalers:
                scaler_path = ticker_dir / "scaler.joblib"
                joblib.dump(self.scalers[ticker], scaler_path)

            # Save metadata (selected features, ensemble weights)
            metadata = {
                "selected_features": self.selected_features.get(ticker, []),
                "ensemble_weights": self.ensemble_weights.get(ticker, {}),
            }
            meta_path = ticker_dir / "metadata.joblib"
            joblib.dump(metadata, meta_path)

        logger.info("All models saved to %s", base)

    def load_models(self, path: Optional[str] = None) -> None:
        """Load previously fitted models, scalers, and metadata.

        Args:
            path: Root directory containing model artefacts.
                Defaults to ``config.paths.models_dir``.
        """
        if path is None:
            path = self.models_dir

        base = Path(path)
        if not base.exists():
            raise FileNotFoundError(f"Models directory not found: {base}")

        for ticker_dir in base.iterdir():
            if not ticker_dir.is_dir():
                continue

            # Reconstruct ticker name
            ticker = ticker_dir.name.replace("_", ".")

            # Load models
            models: Dict[str, Any] = {}
            for model_file in ticker_dir.glob("*.joblib"):
                if model_file.stem in ("scaler", "metadata"):
                    continue
                mtype = model_file.stem
                models[mtype] = joblib.load(model_file)
                logger.info("Loaded %s model ← %s", mtype, model_file)

            if models:
                self.fitted_models[ticker] = models

            # Load scaler
            scaler_path = ticker_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scalers[ticker] = joblib.load(scaler_path)

            # Load metadata
            meta_path = ticker_dir / "metadata.joblib"
            if meta_path.exists():
                metadata = joblib.load(meta_path)
                self.selected_features[ticker] = metadata.get(
                    "selected_features", []
                )
                self.ensemble_weights[ticker] = metadata.get(
                    "ensemble_weights", {}
                )

        logger.info(
            "Loaded models for %d tickers from %s",
            len(self.fitted_models),
            base,
        )

    # ------------------------------------------------------------------
    # Feature importance (aggregate)
    # ------------------------------------------------------------------

    def get_all_feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importances across all trained tickers.

        For each ticker the method extracts feature importances from
        the primary (or first available) fitted model and concatenates
        them into a single DataFrame.

        Returns:
            DataFrame with columns ``Ticker``, ``Feature``,
            ``Importance``.
        """
        frames: List[pd.DataFrame] = []

        for ticker, models in self.fitted_models.items():
            features = self.selected_features.get(ticker, [])
            if not features:
                continue

            # Pick the primary or first model
            if isinstance(models, dict):
                model = models.get(
                    self.primary_model_type, next(iter(models.values()))
                )
            else:
                model = models

            try:
                imp_df = self.feature_selector.get_feature_importance(
                    model, features
                )
                imp_df["Ticker"] = ticker
                frames.append(imp_df[["Ticker", "Feature", "Importance"]])
            except AttributeError as exc:
                logger.warning(
                    "Cannot extract importance for %s: %s", ticker, exc
                )

        if not frames:
            return pd.DataFrame(columns=["Ticker", "Feature", "Importance"])

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Aggregated feature importance — %d rows across %d tickers",
            len(combined),
            combined["Ticker"].nunique(),
        )
        return combined

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _split_X_y(
        df: pd.DataFrame, target_col: str = "Target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate feature columns from the target.

        Args:
            df: Feature DataFrame with a ``Target`` column.
            target_col: Name of the target column.

        Returns:
            Tuple ``(X, y)`` where ``X`` is the feature DataFrame
            and ``y`` is the target Series.  Rows with NaN in the
            target are dropped.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame."
            )

        df_clean = df.dropna(subset=[target_col])
        feature_cols = [c for c in df_clean.columns if c != target_col]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        return X, y

    def _run_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[str]:
        """Run the full feature selection pipeline.

        Removes collinear features first, then applies RFE.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            List of selected feature names.
        """
        try:
            selected = self.feature_selector.select(
                X, y, remove_collinear=True
            )
            if len(selected) < self.feature_selector.min_features:
                logger.warning(
                    "RFE selected only %d features; falling back to "
                    "top-%d by importance",
                    len(selected),
                    self.feature_selector.n_features_to_select,
                )
                selected = list(X.columns)[
                    : self.feature_selector.n_features_to_select
                ]
        except Exception as exc:
            logger.warning(
                "Feature selection failed (%s); using all %d features",
                exc,
                X.shape[1],
            )
            selected = list(X.columns)

        return selected

    def _fit_with_early_stopping(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
    ) -> Any:
        """Fit a model, using early stopping for tree-based learners.

        For LightGBM and XGBoost the last 20 % of samples are used as
        an evaluation set.  Ridge regression is fitted directly.

        Args:
            model: Unfitted estimator.
            X: Scaled feature matrix.
            y: Target array.
            model_type: Model name string.

        Returns:
            Fitted model.
        """
        is_tree = model_type in ("lightgbm", "xgboost")

        if is_tree and len(X) > 50:
            split = int(len(X) * 0.8)
            X_tr, X_es = X[:split], X[split:]
            y_tr, y_es = y[:split], y[split:]

            fit_kwargs: dict = {
                "eval_set": [(X_es, y_es)],
            }

            if model_type == "lightgbm":
                try:
                    import lightgbm as lgb

                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(
                            stopping_rounds=self.early_stopping_rounds,
                            verbose=False,
                        ),
                        lgb.log_evaluation(period=-1),
                    ]
                except (ImportError, AttributeError):
                    pass
            elif model_type == "xgboost":
                fit_kwargs["verbose"] = False

            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
                return model
            except TypeError:
                pass  # Fall through to plain fit

        model.fit(X, y)
        return model

    @staticmethod
    def _compute_ensemble_weights(
        cv_mse_means: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute inverse-MSE ensemble weights.

        Models with **lower** CV MSE receive **higher** weights.

        Args:
            cv_mse_means: Dictionary mapping model_type → mean CV MSE.

        Returns:
            Dictionary mapping model_type → normalised weight in [0, 1]
            summing to 1.
        """
        if not cv_mse_means:
            return {}

        # Inverse MSE
        inv = {k: 1.0 / max(v, 1e-10) for k, v in cv_mse_means.items()}
        total = sum(inv.values())
        weights = {k: v / total for k, v in inv.items()}

        return weights

    @staticmethod
    def _predict_ensemble(
        models: Dict[str, Any],
        X: np.ndarray,
        weights: Dict[str, float],
    ) -> np.ndarray:
        """Generate weighted ensemble predictions.

        Args:
            models: Dictionary mapping model_type → fitted model.
            X: Scaled feature matrix.
            weights: Dictionary mapping model_type → weight.

        Returns:
            1-D array of ensemble predictions.
        """
        preds: List[np.ndarray] = []
        w_list: List[float] = []

        for mtype, model in models.items():
            try:
                y_hat = model.predict(X)
                preds.append(y_hat)
                w_list.append(weights.get(mtype, 1.0 / len(models)))
            except Exception as exc:
                logger.warning(
                    "Ensemble member %s prediction failed: %s", mtype, exc
                )

        if not preds:
            raise RuntimeError("All ensemble members failed to predict.")

        # Normalise weights
        w_arr = np.array(w_list)
        w_arr /= w_arr.sum()

        # Weighted average
        stacked = np.column_stack(preds)
        ensemble_pred = stacked @ w_arr

        return ensemble_pred
