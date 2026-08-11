"""
Portfolio weight optimization for the Multi-Dimensional Return Forecasting system.

This module provides :class:`PortfolioOptimizer` which implements several
portfolio construction strategies:

1. **Equal Weight** — 1/N allocation across all assets.
2. **Inverse Volatility** — weight inversely proportional to realised volatility.
3. **Mean-Variance (Markowitz)** — maximise Sharpe ratio or minimise variance.
4. **Risk Parity** — equal risk contribution from each asset.
5. **Signal-Weighted** — weight proportional to predicted return signal.

Example:
    >>> from src.portfolio.optimizer import PortfolioOptimizer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> optimizer = PortfolioOptimizer(config)
    >>> weights = optimizer.optimize(returns_history, predicted_returns)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """Implements various portfolio weight optimization strategies.

    Strategies supported:
        1. Equal Weight — ``1/N`` allocation.
        2. Inverse Volatility — weight inversely proportional to volatility.
        3. Mean-Variance (Markowitz) — maximise Sharpe ratio.
        4. Risk Parity — equal risk contribution from each asset.
        5. Signal-Weighted — weight proportional to predicted return signal.

    Attributes:
        method: Portfolio optimisation method name.
        risk_free_rate: Annualised risk-free rate (default 6.5 %).
        max_weight: Maximum weight per stock.
        min_weight: Minimum weight per stock.
        lookback_window: Trading days used for covariance estimation.
        transaction_cost: Transaction cost rate (fraction).
        rebalance_frequency: Rebalancing cadence (daily/weekly/monthly).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise from the full pipeline configuration.

        Args:
            config: Parsed ``config.yaml`` dictionary.  The ``portfolio``
                section is used for all tuneable parameters.

        Config parameters (under ``portfolio`` key):
            - ``method``: Portfolio optimisation method.
            - ``risk_free_rate``: 0.065 (India 10Y bond yield).
            - ``max_weight``: 0.40 (maximum weight per stock).
            - ``min_weight``: 0.05 (minimum weight per stock).
            - ``lookback_window``: 63 (trading days for covariance).
            - ``transaction_cost``: 0.001 (10 bps).
            - ``rebalance_frequency``: ``'monthly'``.
        """
        self.config = config
        port_cfg = config.get("portfolio", {})

        self.method: str = port_cfg.get("method", "inverse_volatility")
        self.risk_free_rate: float = port_cfg.get("risk_free_rate", 0.065)
        self.max_weight: float = port_cfg.get("max_weight", 0.40)
        self.min_weight: float = port_cfg.get("min_weight", 0.05)
        self.lookback_window: int = port_cfg.get("lookback_window", 63)
        self.transaction_cost: float = port_cfg.get("transaction_cost", 0.001)
        self.rebalance_frequency: str = port_cfg.get(
            "rebalance_frequency", "monthly"
        )

        logger.info(
            "PortfolioOptimizer initialised — method=%s, "
            "max_weight=%.2f, min_weight=%.2f, lookback=%d",
            self.method,
            self.max_weight,
            self.min_weight,
            self.lookback_window,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def equal_weight(self, n_assets: int) -> np.ndarray:
        """Equal weight allocation.

        Args:
            n_assets: Number of assets in the portfolio.

        Returns:
            Array of weights (all equal to ``1 / n_assets``).
        """
        if n_assets <= 0:
            raise ValueError("n_assets must be positive.")
        weights = np.full(n_assets, 1.0 / n_assets)
        logger.debug("Equal weight: %s", weights)
        return weights

    def inverse_volatility(self, returns: pd.DataFrame) -> np.ndarray:
        """Inverse volatility weighting.

        .. math::
            w_i = \\frac{1 / \\sigma_i}{\\sum_j 1 / \\sigma_j}

        Args:
            returns: DataFrame of historical returns (columns = tickers).

        Returns:
            Array of weights.
        """
        vols = returns.std()

        # Guard against zero or NaN volatility
        vols = vols.replace(0.0, np.nan)
        if vols.isna().all():
            logger.warning(
                "All volatilities are zero/NaN; falling back to equal weight."
            )
            return self.equal_weight(len(returns.columns))

        inv_vol = 1.0 / vols
        inv_vol = inv_vol.fillna(0.0)
        total = inv_vol.sum()

        if total == 0:
            return self.equal_weight(len(returns.columns))

        weights = (inv_vol / total).values
        weights = self.apply_constraints(weights)

        logger.debug("Inverse-vol weights: %s", weights)
        return weights

    def mean_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target: str = "max_sharpe",
    ) -> np.ndarray:
        """Mean-Variance (Markowitz) optimisation.

        Uses :func:`scipy.optimize.minimize` with SLSQP.

        Targets:
            - ``'max_sharpe'``: Maximise Sharpe ratio.
            - ``'min_variance'``: Minimise portfolio variance.

        Constraints:
            - Sum of weights equals 1.
            - ``min_weight <= w_i <= max_weight`` for each asset.

        Args:
            expected_returns: Array of expected returns per asset.
            cov_matrix: Covariance matrix of returns.
            target: Optimisation target (``'max_sharpe'`` or
                ``'min_variance'``).

        Returns:
            Array of optimal weights.
        """
        n = len(expected_returns)
        x0 = np.full(n, 1.0 / n)
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        daily_rf = self.risk_free_rate / 252.0

        if target == "max_sharpe":

            def neg_sharpe(w: np.ndarray) -> float:
                port_ret = w @ expected_returns
                port_vol = np.sqrt(w @ cov_matrix @ w)
                if port_vol < 1e-12:
                    return 1e6
                return -(port_ret - daily_rf) / port_vol

            objective = neg_sharpe

        elif target == "min_variance":

            def portfolio_var(w: np.ndarray) -> float:
                return w @ cov_matrix @ w

            objective = portfolio_var

        else:
            raise ValueError(
                f"Unknown target '{target}'. Use 'max_sharpe' or 'min_variance'."
            )

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(
                "Mean-variance optimisation did not converge: %s. "
                "Falling back to equal weight.",
                result.message,
            )
            return self.equal_weight(n)

        weights = self.apply_constraints(result.x)
        logger.debug("Mean-variance (%s) weights: %s", target, weights)
        return weights

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity allocation.

        Each asset contributes equally to total portfolio risk.  Uses
        numerical optimisation to solve for the weight vector that
        equalises marginal risk contributions.

        Args:
            cov_matrix: Covariance matrix of returns.

        Returns:
            Array of weights.
        """
        n = cov_matrix.shape[0]
        x0 = np.full(n, 1.0 / n)
        target_risk = 1.0 / n

        def risk_parity_objective(w: np.ndarray) -> float:
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-12:
                return 1e6
            # Marginal risk contribution
            mrc = cov_matrix @ w
            rc = w * mrc / port_vol
            # Objective: minimise sum of squared deviations from target
            return np.sum((rc - target_risk * port_vol) ** 2)

        bounds = tuple((1e-6, 1.0) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-14},
        )

        if not result.success:
            logger.warning(
                "Risk parity optimisation did not converge: %s. "
                "Falling back to equal weight.",
                result.message,
            )
            return self.equal_weight(n)

        weights = self.apply_constraints(result.x)
        logger.debug("Risk parity weights: %s", weights)
        return weights

    def signal_weighted(
        self,
        predicted_returns: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Weight based on predicted return signals.

        Method:
            1. Clip predicted returns to avoid extreme weights.
            2. Shift negative predictions to ensure long-only constraint.
            3. Normalise to sum to 1.
            4. Optionally scale by confidence.
            5. Apply min/max weight constraints.

        Args:
            predicted_returns: Array of predicted returns per asset.
            confidence: Optional confidence scores for each prediction.

        Returns:
            Array of weights.
        """
        n = len(predicted_returns)
        if n == 0:
            raise ValueError("predicted_returns must be non-empty.")

        # Clip extreme predictions (within 3 std devs)
        pred = np.array(predicted_returns, dtype=np.float64).copy()
        std = np.std(pred)
        if std > 0:
            pred = np.clip(pred, pred.mean() - 3 * std, pred.mean() + 3 * std)

        # Apply confidence scaling if provided
        if confidence is not None:
            conf = np.array(confidence, dtype=np.float64)
            conf = np.clip(conf, 0.0, 1.0)
            pred = pred * conf

        # Shift to ensure all values are positive (long-only)
        min_pred = pred.min()
        if min_pred <= 0:
            pred = pred - min_pred + 1e-8

        # Normalise to sum to 1
        total = pred.sum()
        if total < 1e-12:
            return self.equal_weight(n)

        weights = pred / total
        weights = self.apply_constraints(weights)

        logger.debug("Signal-weighted weights: %s", weights)
        return weights

    # ------------------------------------------------------------------
    # Constraint application
    # ------------------------------------------------------------------

    def apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max weight constraints and renormalise.

        Steps:
            1. Clip weights to ``[min_weight, max_weight]``.
            2. Renormalise to sum to 1.
            3. Repeat until convergence (max 50 iterations).

        Args:
            weights: Raw weight array.

        Returns:
            Constrained weights summing to 1.
        """
        w = np.array(weights, dtype=np.float64).copy()
        n = len(w)

        # Handle edge case: fewer assets than needed for min_weight
        if n * self.min_weight > 1.0:
            return np.full(n, 1.0 / n)

        for _ in range(50):
            w = np.clip(w, self.min_weight, self.max_weight)
            total = w.sum()
            if total < 1e-12:
                w = np.full(n, 1.0 / n)
                break
            w = w / total
            # Check convergence
            if np.all(w >= self.min_weight - 1e-10) and np.all(
                w <= self.max_weight + 1e-10
            ):
                break

        return w

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns_history: pd.DataFrame,
        predicted_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Main optimisation method — dispatches to the configured strategy.

        Args:
            returns_history: Historical returns for covariance estimation.
                Columns correspond to tickers.
            predicted_returns: Predicted future returns (optional, required
                for ``'signal_weighted'`` and ``'mean_variance'``).

        Returns:
            Array of optimal weights.
        """
        n_assets = returns_history.shape[1]
        method = self.method.lower().strip()

        # Use lookback window
        lookback = min(self.lookback_window, len(returns_history))
        recent_returns = returns_history.iloc[-lookback:]

        logger.info(
            "Optimising portfolio — method=%s, n_assets=%d, "
            "lookback=%d days",
            method,
            n_assets,
            lookback,
        )

        if method == "equal_weight":
            return self.equal_weight(n_assets)

        if method == "inverse_volatility":
            return self.inverse_volatility(recent_returns)

        if method in ("mean_variance", "max_sharpe"):
            if predicted_returns is None:
                logger.warning(
                    "mean_variance requires predicted_returns; "
                    "using historical mean instead."
                )
                predicted_returns = recent_returns.mean().values
            cov = recent_returns.cov().values
            return self.mean_variance(predicted_returns, cov, target="max_sharpe")

        if method == "min_variance":
            cov = recent_returns.cov().values
            if predicted_returns is None:
                predicted_returns = recent_returns.mean().values
            return self.mean_variance(predicted_returns, cov, target="min_variance")

        if method == "risk_parity":
            cov = recent_returns.cov().values
            return self.risk_parity(cov)

        if method == "signal_weighted":
            if predicted_returns is None:
                logger.warning(
                    "signal_weighted requires predicted_returns; "
                    "falling back to equal weight."
                )
                return self.equal_weight(n_assets)
            return self.signal_weighted(predicted_returns)

        logger.warning(
            "Unknown method '%s'; falling back to equal weight.", method
        )
        return self.equal_weight(n_assets)

    # ------------------------------------------------------------------
    # Rebalancing schedule
    # ------------------------------------------------------------------

    def compute_rebalance_dates(
        self, start_date: str, end_date: str
    ) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on the configured frequency.

        Frequencies:
            - ``'daily'``: Every trading day.
            - ``'weekly'``: Every Monday (or first trading day of the week).
            - ``'monthly'``: First trading day of each month.

        Args:
            start_date: Start date string (``YYYY-MM-DD``).
            end_date: End date string (``YYYY-MM-DD``).

        Returns:
            Sorted list of :class:`pd.Timestamp` rebalancing dates.
        """
        # Generate business-day range (Mon–Fri)
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        freq = self.rebalance_frequency.lower().strip()

        if freq == "daily":
            rebal_dates = list(all_dates)

        elif freq == "weekly":
            # First business day of each week
            df = pd.DataFrame({"date": all_dates})
            df["week"] = df["date"].dt.isocalendar().week.values
            df["year"] = df["date"].dt.year
            rebal_dates = (
                df.groupby(["year", "week"])["date"]
                .first()
                .sort_values()
                .tolist()
            )

        elif freq == "monthly":
            # First business day of each month
            df = pd.DataFrame({"date": all_dates})
            df["month"] = df["date"].dt.month
            df["year"] = df["date"].dt.year
            rebal_dates = (
                df.groupby(["year", "month"])["date"]
                .first()
                .sort_values()
                .tolist()
            )

        else:
            logger.warning(
                "Unknown rebalance frequency '%s'; defaulting to monthly.",
                freq,
            )
            return self.compute_rebalance_dates(start_date, end_date)

        logger.info(
            "Generated %d rebalance dates (%s) from %s to %s",
            len(rebal_dates),
            freq,
            start_date,
            end_date,
        )
        return rebal_dates
