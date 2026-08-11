"""SARIMAX baseline wrapped as a scikit-learn-compatible regressor.

The return-forecasting ensemble (LightGBM/XGBoost/Ridge) is entirely
feature-driven; this project had no classical time-series baseline to
compare it against. This module adds one: a univariate SARIMAX model
fitted on the target return series alone (no exogenous features), so it
can drop into :class:`src.models.walk_forward.WalkForwardValidator`'s
existing ``fit``/``predict``/``clone`` walk-forward CV loop unchanged
and be benchmarked fold-for-fold against the ML ensemble.

It deliberately ignores the feature matrix ``X`` — SARIMAX forecasts
from its own fitted series, not from cross-sectional features — but
still accepts and uses ``X`` for shape (``len(X)`` = number of steps to
forecast) so its ``fit``/``predict`` signatures match every other model
in the walk-forward harness.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SARIMAXRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around ``statsmodels`` SARIMAX.

    Attributes:
        order: ``(p, d, q)`` ARIMA order.
        seasonal_order: ``(P, D, Q, s)`` seasonal order.
        trend: Trend term passed to ``SARIMAX`` (``'c'`` = constant).
        purge_gap: Number of steps to forecast and discard before the
            returned prediction window, matching the walk-forward
            validator's purge gap so SARIMAX is not compared unfairly
            against models evaluated across the same gap.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c",
        purge_gap: int = 5,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.purge_gap = purge_gap

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SARIMAXRegressor":
        """Fit SARIMAX on the target series ``y`` (``X`` is unused).

        Args:
            X: Feature matrix — accepted for API compatibility only.
            y: 1-D target return series, in chronological order.

        Returns:
            ``self``.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.asarray(y, dtype=float).ravel()
        self.n_train_ = len(y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            try:
                self.results_ = model.fit(disp=False)
            except Exception:
                # Numerically unstable fit (short/degenerate series) — fall
                # back to a plain constant-mean model rather than raising,
                # so a single bad fold doesn't kill the whole CV loop.
                model = SARIMAX(y, order=(0, 0, 0), trend="c")
                self.results_ = model.fit(disp=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast ``len(X)`` steps ahead, skipping the purge gap.

        Args:
            X: Feature matrix — only its row count is used.

        Returns:
            1-D array of ``len(X)`` forecasted values.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError("SARIMAXRegressor.predict() called before fit().")

        n_out = len(X)
        n_ahead = self.purge_gap + n_out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.results_.get_forecast(steps=n_ahead).predicted_mean
        forecast = np.asarray(forecast).ravel()
        return forecast[-n_out:] if n_out > 0 else forecast
