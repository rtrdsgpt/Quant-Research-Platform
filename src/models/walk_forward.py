"""
Walk-forward (time-series) cross-validation engine.

This module implements expanding-window and sliding-window walk-forward
validation that preserves temporal ordering — the **only** valid
cross-validation strategy for time-series data.  Standard K-Fold
cross-validation is strictly prohibited because it violates the
temporal dependency structure of financial return data.

Walk-forward validation procedure::

    Split 1: train [0 : T0],        val [T0+purge : T0+purge+V]
    Split 2: train [0 : T0+step],   val [T0+step+purge : T0+step+purge+V]
    ...
    (expanding window — training set grows; sliding window — fixed size)

Example:
    >>> from src.models.walk_forward import WalkForwardValidator
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> wfv = WalkForwardValidator(config)
    >>> splits = list(wfv.generate_splits(n_samples=1200))
    >>> print(len(splits))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """Represents a single walk-forward train / validation split.

    Attributes:
        fold_id: Zero-based fold index.
        train_indices: Integer array of training row positions.
        val_indices: Integer array of validation row positions.
    """

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray


class WalkForwardValidator:
    """Walk-forward (expanding / sliding window) cross-validation for time series.

    This validator generates chronologically ordered train/validation
    splits so that training data **always precedes** validation data.
    An optional *purge gap* prevents information leakage between the
    end of training and the start of validation.

    Attributes:
        initial_train_days: Number of observations in the first training
            window (default ``504`` ≈ 2 years of trading days).
        validation_days: Number of observations in each validation window
            (default ``63`` ≈ 3 months).
        step_days: Number of observations to advance the split boundary
            between consecutive folds (default ``21`` ≈ 1 month).
        expanding: If ``True`` the training window grows; if ``False`` a
            fixed-width sliding window is used.
        purge_days: Gap (in observations) inserted between the end of
            training and the start of validation to avoid data leakage
            from overlapping feature lags.

    Example:
        >>> validator = WalkForwardValidator(config)
        >>> for train_idx, val_idx in validator.generate_splits(1200):
        ...     print(f"train {len(train_idx)}, val {len(val_idx)}")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise from the full pipeline configuration dictionary.

        The constructor reads parameters from both ``config["model"]["walk_forward"]``
        (actual ``config.yaml``) and ``config["validation"]`` (``ARCHITECTURE.md``
        spec), with sensible defaults so either structure works.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        # Try the actual config.yaml structure first
        wf_cfg = config.get("model", {}).get("walk_forward", {})

        # Fall back to ARCHITECTURE.md "validation" key
        val_cfg = config.get("validation", {})

        self.initial_train_days: int = (
            wf_cfg.get("initial_train_days")
            or val_cfg.get("initial_train_days", 504)
        )
        self.validation_days: int = (
            wf_cfg.get("validation_days")
            or val_cfg.get("min_val_days", 63)
        )
        self.step_days: int = (
            wf_cfg.get("step_days")
            or val_cfg.get("step_size_days", 21)
        )
        self.expanding: bool = (
            wf_cfg.get("expanding")
            if wf_cfg.get("expanding") is not None
            else val_cfg.get("expanding_window", True)
        )
        self.purge_days: int = val_cfg.get("purge_days", 5)

        logger.info(
            "WalkForwardValidator initialised — initial_train=%d, "
            "val=%d, step=%d, expanding=%s, purge=%d",
            self.initial_train_days,
            self.validation_days,
            self.step_days,
            self.expanding,
            self.purge_days,
        )

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def generate_splits(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train / validation index arrays in temporal order.

        For an **expanding window** the training set starts at index ``0``
        and its right boundary moves forward.  For a **sliding window**
        the left boundary also advances to keep the window size constant.

        Yields:
            ``(train_indices, val_indices)`` — 1-D integer ``np.ndarray``
            pairs suitable for direct fancy-indexing into ``X`` and ``y``.

        Raises:
            ValueError: If *n_samples* is too small to produce at least
                one valid split.

        Example (expanding window, no purge)::

            Split 0: train [0 : 504],  val [504 : 567]
            Split 1: train [0 : 525],  val [525 : 588]
            ...
        """
        if n_samples < self.initial_train_days + self.purge_days + self.validation_days:
            raise ValueError(
                f"Need at least {self.initial_train_days + self.purge_days + self.validation_days} "
                f"samples for one split, but got {n_samples}."
            )

        all_indices = np.arange(n_samples)
        fold_id = 0
        train_end = self.initial_train_days  # exclusive upper bound

        while train_end + self.purge_days + self.validation_days <= n_samples:
            # Training indices
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.initial_train_days)

            train_idx = all_indices[train_start:train_end]

            # Validation indices (after purge gap)
            val_start = train_end + self.purge_days
            val_end = min(val_start + self.validation_days, n_samples)
            val_idx = all_indices[val_start:val_end]

            logger.debug(
                "Fold %d — train [%d:%d] (%d), purge %d, val [%d:%d] (%d)",
                fold_id,
                train_start,
                train_end,
                len(train_idx),
                self.purge_days,
                val_start,
                val_end,
                len(val_idx),
            )

            yield train_idx, val_idx

            fold_id += 1
            train_end += self.step_days

    def get_n_splits(self, n_samples: int) -> int:
        """Return the total number of walk-forward folds.

        Args:
            n_samples: Total number of observations.

        Returns:
            Integer count of folds that :meth:`generate_splits` would
            yield for the given sample size.
        """
        count = 0
        train_end = self.initial_train_days
        while train_end + self.purge_days + self.validation_days <= n_samples:
            count += 1
            train_end += self.step_days
        return count

    # ------------------------------------------------------------------
    # Full cross-validation loop
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run walk-forward cross-validation and return per-fold metrics.

        At each fold a **fresh clone** of *model* is fitted on the
        training slice and evaluated on the validation slice.  The model
        must expose a scikit-learn–compatible ``fit`` / ``predict`` API.

        For LightGBM and XGBoost models, when *early_stopping_rounds*
        is set, 20 % of the training fold is held out as an eval set
        to enable early stopping.

        Args:
            model: An sklearn-compatible estimator (must have ``fit``
                and ``predict`` methods).
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of shape ``(n_samples,)``.
            metrics: Metric names to compute (see
                :meth:`_compute_metric`).  Defaults to
                ``['mse', 'mae', 'r2', 'direction_accuracy']``.
            early_stopping_rounds: If not ``None``, use the last
                portion of training data as an eval set for early
                stopping in tree-based models.

        Returns:
            Dictionary mapping each metric name to a list of per-fold
            scores, e.g. ``{'mse': [0.001, 0.002, ...], ...}``.
        """
        if metrics is None:
            metrics = ["mse", "mae", "r2", "direction_accuracy"]

        n_samples = X.shape[0]
        n_splits = self.get_n_splits(n_samples)

        results: Dict[str, List[float]] = {m: [] for m in metrics}

        logger.info(
            "Starting walk-forward CV — %d folds, %d samples, metrics=%s",
            n_splits,
            n_samples,
            metrics,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(
                self.generate_splits(n_samples),
                total=n_splits,
                desc="Walk-forward CV",
            )
        ):
            try:
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                X_val_fold, y_val_fold = X[val_idx], y[val_idx]

                # Clone the model to avoid leaking state between folds
                cloned = _clone_model(model)

                # Fit — with optional early stopping
                cloned = _fit_model(
                    cloned,
                    X_train_fold,
                    y_train_fold,
                    early_stopping_rounds=early_stopping_rounds,
                )

                y_pred = cloned.predict(X_val_fold)

                for metric_name in metrics:
                    score = self._compute_metric(y_val_fold, y_pred, metric_name)
                    results[metric_name].append(score)

                logger.debug(
                    "Fold %d — %s",
                    fold_idx,
                    {m: f"{results[m][-1]:.6f}" for m in metrics},
                )

            except Exception as exc:
                logger.error("Fold %d failed: %s", fold_idx, exc, exc_info=True)
                for metric_name in metrics:
                    results[metric_name].append(np.nan)

        # Log summary statistics
        for m in metrics:
            vals = [v for v in results[m] if not np.isnan(v)]
            if vals:
                logger.info(
                    "CV metric %-20s — mean=%.6f  std=%.6f",
                    m,
                    np.mean(vals),
                    np.std(vals),
                )

        return results

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metric(
        y_true: np.ndarray, y_pred: np.ndarray, metric: str
    ) -> float:
        """Compute a single evaluation metric.

        Supported metrics:
            - **mse**: Mean Squared Error
            - **rmse**: Root Mean Squared Error
            - **mae**: Mean Absolute Error
            - **r2**: R-squared (coefficient of determination)
            - **direction_accuracy**: Fraction of observations where
              ``sign(y_pred) == sign(y_true)`` (hit ratio).
            - **ic**: Information Coefficient — Spearman rank
              correlation between predictions and actuals.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            metric: Name of the metric (case-insensitive).

        Returns:
            Scalar metric value.

        Raises:
            ValueError: If *metric* is not recognised.
        """
        metric = metric.lower().strip()

        if metric == "mse":
            return float(mean_squared_error(y_true, y_pred))

        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        if metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))

        if metric == "r2":
            return float(r2_score(y_true, y_pred))

        if metric in ("direction_accuracy", "hit_ratio"):
            if len(y_true) == 0:
                return 0.0
            correct = np.sign(y_true) == np.sign(y_pred)
            return float(np.mean(correct))

        if metric == "ic":
            if len(y_true) < 3:
                return 0.0
            corr, _ = stats.spearmanr(y_true, y_pred)
            return float(corr) if np.isfinite(corr) else 0.0

        raise ValueError(
            f"Unknown metric '{metric}'. Supported: "
            "mse, rmse, mae, r2, direction_accuracy, ic"
        )


# ======================================================================
# Module-level helpers
# ======================================================================


def _clone_model(model):
    """Return a fresh, unfitted copy of *model*.

    Attempts ``sklearn.base.clone`` first; falls back to constructing
    a new instance from the class and ``get_params()``.

    Args:
        model: An sklearn-compatible estimator.

    Returns:
        A new estimator instance with the same hyper-parameters.
    """
    try:
        from sklearn.base import clone

        return clone(model)
    except Exception:
        # Manual fallback
        params = model.get_params()
        return model.__class__(**params)


def _fit_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    early_stopping_rounds: Optional[int] = None,
):
    """Fit *model* with optional early stopping for tree-based learners.

    When *early_stopping_rounds* is set and the model is a LightGBM or
    XGBoost regressor, the last 20 % of ``X_train`` / ``y_train`` is
    carved out as an evaluation set.

    Args:
        model: Unfitted estimator.
        X_train: Training features.
        y_train: Training targets.
        early_stopping_rounds: Patience parameter for early stopping.

    Returns:
        Fitted model.
    """
    model_type = type(model).__name__.lower()
    is_tree = "lgbm" in model_type or "xgb" in model_type

    if early_stopping_rounds and is_tree and len(X_train) > 50:
        split_point = int(len(X_train) * 0.8)
        X_tr, X_es = X_train[:split_point], X_train[split_point:]
        y_tr, y_es = y_train[:split_point], y_train[split_point:]

        fit_kwargs: dict = {}

        if "lgbm" in model_type:
            fit_kwargs["eval_set"] = [(X_es, y_es)]
            fit_kwargs["callbacks"] = [
                _lgb_early_stopping(early_stopping_rounds),
                _lgb_log_evaluation(-1),
            ]
        elif "xgb" in model_type:
            fit_kwargs["eval_set"] = [(X_es, y_es)]
            fit_kwargs["verbose"] = False

        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except TypeError:
            # If callbacks/eval_set not supported, fall back to plain fit
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    return model


def _lgb_early_stopping(stopping_rounds: int):
    """Return a LightGBM ``early_stopping`` callback.

    Args:
        stopping_rounds: Patience.

    Returns:
        LightGBM callback object.
    """
    try:
        import lightgbm as lgb

        return lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)
    except (ImportError, AttributeError):
        return None


def _lgb_log_evaluation(period: int):
    """Return a LightGBM ``log_evaluation`` callback.

    Args:
        period: Print frequency (``-1`` to suppress).

    Returns:
        LightGBM callback object.
    """
    try:
        import lightgbm as lgb

        return lgb.log_evaluation(period=period)
    except (ImportError, AttributeError):
        return None
