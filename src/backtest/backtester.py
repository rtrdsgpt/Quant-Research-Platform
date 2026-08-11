"""
Portfolio backtesting engine for the Multi-Dimensional Return Forecasting system.

This module provides :class:`PortfolioBacktester` which simulates portfolio
strategies on historical data, accounting for transaction costs and weight
constraints.  It orchestrates :class:`PortfolioOptimizer` and
:class:`PortfolioMetrics` to produce full backtest results.

Example:
    >>> from src.portfolio.backtester import PortfolioBacktester
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> bt = PortfolioBacktester(config)
    >>> results = bt.run_backtest(predictions, prices, '2025-10-01', '2025-12-31')
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.portfolio.metrics import PortfolioMetrics
from src.portfolio.optimizer import PortfolioOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioBacktester:
    """Backtests portfolio strategies using predicted returns.

    Simulates:
        - Periodic portfolio rebalancing.
        - Transaction costs.
        - Weight constraints (min/max per asset).

    Attributes:
        config: Full pipeline configuration dictionary.
        optimizer: :class:`PortfolioOptimizer` instance.
        metrics: :class:`PortfolioMetrics` instance.
        initial_capital: Default starting capital in INR.
        reports_dir: Directory for saving reports and figures.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise with config and create helper instances.

        Args:
            config: Parsed ``config.yaml`` dictionary.
        """
        self.config = config
        self.optimizer = PortfolioOptimizer(config)
        self.metrics = PortfolioMetrics(config)

        port_cfg = config.get("portfolio", {})
        self.transaction_cost_rate: float = port_cfg.get(
            "transaction_cost", 0.001
        )
        self.reports_dir: str = config.get("paths", {}).get(
            "reports_dir", "reports"
        )

        # Date configuration
        dates_cfg = config.get("dates", {})
        self.test_start: str = dates_cfg.get("test_start", "2025-10-01")
        self.test_end: str = dates_cfg.get("test_end", "2025-12-31")

        logger.info(
            "PortfolioBacktester initialised — transaction_cost=%.4f, "
            "test_period=%s to %s",
            self.transaction_cost_rate,
            self.test_start,
            self.test_end,
        )

    # ------------------------------------------------------------------
    # Main backtest
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        predictions: Dict[str, pd.DataFrame],
        prices: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
    ) -> Dict:
        """Run portfolio backtest over a specified date range.

        Args:
            predictions: Dict mapping ticker → DataFrame with columns
                ``Predicted_Return`` (and optionally ``Actual_Return``),
                indexed by date.
            prices: Dict mapping ticker → DataFrame with column
                ``Adj_Close``, indexed by date.
            start_date: Backtest start date (``YYYY-MM-DD``).
            end_date: Backtest end date (``YYYY-MM-DD``).
            initial_capital: Starting capital (default ₹10,00,000).

        Returns:
            Dictionary with keys:
                - ``portfolio_values``: :class:`pd.Series` of daily values.
                - ``returns``: :class:`pd.Series` of daily returns.
                - ``weights_history``: :class:`pd.DataFrame` of daily
                  weights per ticker.
                - ``trades``: :class:`pd.DataFrame` of all rebalance events.
                - ``metrics``: dictionary of performance metrics.
                - ``predicted_returns``: aggregated predicted returns.
        """
        tickers = sorted(predictions.keys())
        n_assets = len(tickers)

        if n_assets == 0:
            raise ValueError("predictions dict is empty — no tickers.")

        logger.info(
            "Running backtest: %s → %s, %d tickers, "
            "initial_capital=₹%.0f",
            start_date,
            end_date,
            n_assets,
            initial_capital,
        )

        # --- Build aligned return matrices ---
        aligned_returns, aligned_predictions = self._align_data(
            predictions, prices, tickers, start_date, end_date
        )

        if aligned_returns.empty:
            raise ValueError(
                "No overlapping dates in the backtest period."
            )

        trading_dates = aligned_returns.index
        n_days = len(trading_dates)

        logger.info("Backtest covers %d trading days.", n_days)

        # --- Rebalancing schedule ---
        rebal_dates = set(
            self.optimizer.compute_rebalance_dates(start_date, end_date)
        )

        # --- Simulation state ---
        portfolio_value = initial_capital
        current_weights = np.full(n_assets, 1.0 / n_assets)

        # Storage
        values = []
        daily_returns_list = []
        weights_rows = []
        trades_rows = []
        pred_returns_rows = []

        for i, date in enumerate(trading_dates):
            stock_returns = aligned_returns.loc[date].values
            pred_rets = aligned_predictions.loc[date].values

            # Check for rebalancing
            is_rebal = (date in rebal_dates) or (i == 0)

            if is_rebal:
                # Use history up to current date for covariance
                hist_end = i + 1
                hist_start = max(0, hist_end - self.optimizer.lookback_window)
                returns_history = aligned_returns.iloc[hist_start:hist_end]

                new_weights = self.optimizer.optimize(
                    returns_history, pred_rets
                )

                # Transaction costs
                tc = self._apply_transaction_costs(
                    current_weights, new_weights, portfolio_value
                )
                portfolio_value -= tc

                trades_rows.append(
                    {
                        "Date": date,
                        "Old_Weights": current_weights.copy(),
                        "New_Weights": new_weights.copy(),
                        "Turnover": float(
                            np.sum(np.abs(new_weights - current_weights)) / 2
                        ),
                        "Transaction_Cost": tc,
                        "Portfolio_Value": portfolio_value,
                    }
                )

                current_weights = new_weights.copy()

            # Compute daily portfolio return
            daily_ret = self._compute_daily_return(
                current_weights, stock_returns
            )
            portfolio_value *= 1 + daily_ret

            # Drift weights based on individual stock returns
            if i < n_days - 1:
                current_weights = self._drift_weights(
                    current_weights, stock_returns
                )

            values.append(portfolio_value)
            daily_returns_list.append(daily_ret)
            weights_rows.append(
                dict(zip(tickers, current_weights))
            )
            pred_returns_rows.append(
                dict(zip(tickers, pred_rets))
            )

        # --- Build result DataFrames ---
        portfolio_values = pd.Series(
            values, index=trading_dates, name="Portfolio_Value"
        )
        returns_series = pd.Series(
            daily_returns_list, index=trading_dates, name="Daily_Return"
        )
        weights_df = pd.DataFrame(weights_rows, index=trading_dates)
        trades_df = pd.DataFrame(trades_rows) if trades_rows else pd.DataFrame()
        pred_returns_df = pd.DataFrame(pred_returns_rows, index=trading_dates)

        # --- Compute aggregated predicted return (weighted) ---
        port_pred_returns = (pred_returns_df * weights_df).sum(axis=1)
        port_pred_returns.name = "Predicted_Return"

        # --- Compute metrics ---
        all_metrics = self.metrics.compute_all_metrics(
            portfolio_values=portfolio_values,
            returns=returns_series,
            predicted_returns=port_pred_returns,
            benchmark_returns=None,
        )

        logger.info(
            "Backtest complete — Final Value=₹%.0f, "
            "Total Return=%.4f, Sharpe=%.4f, MDD=%.4f",
            portfolio_values.iloc[-1],
            all_metrics["Total_Return"],
            all_metrics["Sharpe_Ratio"],
            all_metrics["Maximum_Drawdown"],
        )

        return {
            "portfolio_values": portfolio_values,
            "returns": returns_series,
            "weights_history": weights_df,
            "trades": trades_df,
            "metrics": all_metrics,
            "predicted_returns": port_pred_returns,
        }

    # ------------------------------------------------------------------
    # Daily return computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_daily_return(
        weights: np.ndarray, stock_returns: np.ndarray
    ) -> float:
        """Compute portfolio return for a single day.

        .. math::
            r_p = \\sum_i w_i \\cdot r_i

        Args:
            weights: Current portfolio weights.
            stock_returns: Daily returns per stock.

        Returns:
            Portfolio return for the day.
        """
        # Replace NaN stock returns with 0
        clean_returns = np.nan_to_num(stock_returns, nan=0.0)
        return float(np.dot(weights, clean_returns))

    # ------------------------------------------------------------------
    # Transaction costs
    # ------------------------------------------------------------------

    def _apply_transaction_costs(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """Compute transaction costs for rebalancing.

        .. math::
            \\text{Cost} = \\text{turnover} \\times
            \\text{cost\\_rate} \\times \\text{portfolio\\_value}

        Where turnover is:

        .. math::
            \\text{turnover} = \\frac{1}{2} \\sum_i |w^{\\text{new}}_i
            - w^{\\text{old}}_i|

        Args:
            old_weights: Pre-rebalance weights.
            new_weights: Post-rebalance weights.
            portfolio_value: Current portfolio value.

        Returns:
            Transaction cost in INR.
        """
        turnover = float(np.sum(np.abs(new_weights - old_weights))) / 2.0
        cost = turnover * self.transaction_cost_rate * portfolio_value
        return cost

    # ------------------------------------------------------------------
    # Weight drift
    # ------------------------------------------------------------------

    @staticmethod
    def _drift_weights(
        weights: np.ndarray, stock_returns: np.ndarray
    ) -> np.ndarray:
        """Drift weights by one day's stock returns.

        After one day the weight of each asset changes proportionally
        to its return.  This method computes the new weights without
        rebalancing.

        Args:
            weights: Current weights.
            stock_returns: Day's stock returns.

        Returns:
            Drifted weights (summing to 1).
        """
        clean_returns = np.nan_to_num(stock_returns, nan=0.0)
        new_values = weights * (1 + clean_returns)
        total = new_values.sum()
        if total < 1e-12:
            return weights.copy()
        return new_values / total

    # ------------------------------------------------------------------
    # Forward test
    # ------------------------------------------------------------------

    def forward_test(
        self,
        predictions: Dict[str, pd.DataFrame],
        prices: Dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000.0,
    ) -> Dict:
        """Run forward test on the held-out period.

        Uses dates from config (``test_start``, ``test_end``).

        Args:
            predictions: Dict mapping ticker → prediction DataFrame.
            prices: Dict mapping ticker → price DataFrame.
            initial_capital: Starting capital (default ₹10,00,000).

        Returns:
            Same structure as :meth:`run_backtest`.
        """
        logger.info(
            "Running forward test: %s → %s",
            self.test_start,
            self.test_end,
        )
        return self.run_backtest(
            predictions,
            prices,
            start_date=self.test_start,
            end_date=self.test_end,
            initial_capital=initial_capital,
        )

    # ------------------------------------------------------------------
    # Strategy comparison
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        predictions: Dict[str, pd.DataFrame],
        prices: Dict[str, pd.DataFrame],
        strategies: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 1_000_000.0,
    ) -> pd.DataFrame:
        """Compare different portfolio strategies.

        Runs the backtest with each strategy and returns a comparison
        table of metrics.

        Args:
            predictions: Dict mapping ticker → prediction DataFrame.
            prices: Dict mapping ticker → price DataFrame.
            strategies: List of strategy names.  Defaults to
                ``['equal_weight', 'inverse_volatility', 'signal_weighted']``.
            start_date: Backtest start.  Defaults to ``test_start``.
            end_date: Backtest end.  Defaults to ``test_end``.
            initial_capital: Starting capital.

        Returns:
            DataFrame comparing metrics across strategies.
        """
        if strategies is None:
            strategies = [
                "equal_weight",
                "inverse_volatility",
                "signal_weighted",
            ]

        if start_date is None:
            start_date = self.test_start
        if end_date is None:
            end_date = self.test_end

        logger.info(
            "Comparing strategies: %s over %s → %s",
            strategies,
            start_date,
            end_date,
        )

        results_table: Dict[str, Dict[str, float]] = {}

        original_method = self.optimizer.method

        for strategy in strategies:
            logger.info("  Running strategy: %s", strategy)
            try:
                self.optimizer.method = strategy
                result = self.run_backtest(
                    predictions,
                    prices,
                    start_date,
                    end_date,
                    initial_capital,
                )
                results_table[strategy] = result["metrics"]
            except Exception as exc:
                logger.error(
                    "Strategy %s failed: %s", strategy, exc, exc_info=True
                )
                results_table[strategy] = {"error": str(exc)}

        # Restore original method
        self.optimizer.method = original_method

        comparison_df = pd.DataFrame(results_table).T
        comparison_df.index.name = "Strategy"

        # Save comparison
        reports_dir = Path(self.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = reports_dir / "strategy_comparison.csv"
        comparison_df.to_csv(comparison_path)
        logger.info("Strategy comparison saved to %s", comparison_path)

        return comparison_df

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self, backtest_results: Dict
    ) -> Tuple[str, List[plt.Figure]]:
        """Generate comprehensive backtest report.

        Produces:
            1. Text performance report.
            2. Equity curve figure.
            3. Weights allocation figure.

        Args:
            backtest_results: Dictionary returned by :meth:`run_backtest`.

        Returns:
            Tuple of ``(text_report, list_of_figures)``.
        """
        portfolio_values = backtest_results["portfolio_values"]
        returns = backtest_results["returns"]
        metrics = backtest_results["metrics"]
        weights_history = backtest_results["weights_history"]

        figures: List[plt.Figure] = []

        # --- 1. Text report ---
        text_report = self.metrics.create_performance_report(
            metrics, portfolio_values, returns
        )

        # --- 2. Equity curve ---
        fig_equity = self.metrics.create_equity_curve(
            portfolio_values=portfolio_values,
            benchmark_values=None,
            metrics=metrics,
            title="Portfolio Equity Curve — Backtest",
            save_path=str(Path(self.reports_dir) / "equity_curve.png"),
        )
        figures.append(fig_equity)

        # --- 3. Weights allocation over time ---
        fig_weights = self._plot_weights(weights_history)
        figures.append(fig_weights)

        logger.info(
            "Report generated — %d figures, text report length=%d",
            len(figures),
            len(text_report),
        )

        return text_report, figures

    # ------------------------------------------------------------------
    # Data alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_index(df_or_series):
        """Strip timezone info from DatetimeIndex for consistent comparison."""
        obj = df_or_series.copy()
        if isinstance(obj.index, pd.DatetimeIndex) and obj.index.tz is not None:
            obj.index = obj.index.tz_localize(None)
        return obj

    @staticmethod
    def _align_data(
        predictions: Dict[str, pd.DataFrame],
        prices: Dict[str, pd.DataFrame],
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build aligned return and prediction matrices.

        Aligns all tickers to a common date index within the backtest
        window.

        Args:
            predictions: Ticker → prediction DataFrame.
            prices: Ticker → price DataFrame.
            tickers: Sorted list of tickers.
            start_date: Start date string.
            end_date: End date string.

        Returns:
            Tuple of ``(returns_df, predictions_df)`` both indexed by
            date with one column per ticker.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        returns_dict: Dict[str, pd.Series] = {}
        pred_dict: Dict[str, pd.Series] = {}

        for ticker in tickers:
            # --- Returns from prices ---
            if ticker in prices:
                price_df = prices[ticker].copy()
                # Normalize timezone
                if isinstance(price_df.index, pd.DatetimeIndex) and price_df.index.tz is not None:
                    price_df.index = price_df.index.tz_localize(None)

                if "Adj_Close" in price_df.columns:
                    price_col = "Adj_Close"
                elif "Close" in price_df.columns:
                    price_col = "Close"
                else:
                    # Use first numeric column
                    numeric_cols = price_df.select_dtypes(
                        include=[np.number]
                    ).columns
                    if len(numeric_cols) == 0:
                        logger.warning(
                            "No numeric price column for %s; skipping.",
                            ticker,
                        )
                        continue
                    price_col = numeric_cols[0]

                price_series = price_df[price_col]
                if not isinstance(price_series.index, pd.DatetimeIndex):
                    if "Date" in price_df.columns:
                        price_series.index = pd.to_datetime(
                            price_df["Date"]
                        )
                    else:
                        price_series.index = pd.to_datetime(
                            price_series.index
                        )

                ret = price_series.pct_change().dropna()
                returns_dict[ticker] = ret

            # --- Predicted returns ---
            if ticker in predictions:
                pred_df = predictions[ticker].copy()
                # Normalize timezone
                if isinstance(pred_df.index, pd.DatetimeIndex) and pred_df.index.tz is not None:
                    pred_df.index = pred_df.index.tz_localize(None)

                if "Predicted_Return" in pred_df.columns:
                    pred_series = pred_df["Predicted_Return"]
                else:
                    # Use first column
                    pred_series = pred_df.iloc[:, 0]

                if not isinstance(pred_series.index, pd.DatetimeIndex):
                    if "Date" in pred_df.columns:
                        pred_series.index = pd.to_datetime(pred_df["Date"])
                    else:
                        pred_series.index = pd.to_datetime(pred_series.index)

                pred_dict[ticker] = pred_series

        if not returns_dict:
            return pd.DataFrame(), pd.DataFrame()

        # Combine into DataFrames
        returns_df = pd.DataFrame(returns_dict)
        pred_df = pd.DataFrame(pred_dict)

        # Filter to date range
        mask_ret = (returns_df.index >= start) & (returns_df.index <= end)
        returns_df = returns_df.loc[mask_ret]

        mask_pred = (pred_df.index >= start) & (pred_df.index <= end)
        pred_df = pred_df.loc[mask_pred]

        # Align to common dates
        common_dates = returns_df.index.intersection(pred_df.index)
        returns_df = returns_df.loc[common_dates].fillna(0.0)
        pred_df = pred_df.loc[common_dates].fillna(0.0)

        # Ensure same column order
        returns_df = returns_df[tickers]
        pred_df = pred_df[tickers]

        logger.info(
            "Aligned data: %d dates, %d tickers",
            len(common_dates),
            len(tickers),
        )

        return returns_df, pred_df

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _plot_weights(self, weights_history: pd.DataFrame) -> plt.Figure:
        """Plot portfolio weight allocation over time as a stacked area chart.

        Args:
            weights_history: DataFrame of daily weights (columns = tickers).

        Returns:
            :class:`matplotlib.figure.Figure` object.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.stackplot(
            weights_history.index,
            [weights_history[col].values for col in weights_history.columns],
            labels=weights_history.columns.tolist(),
            alpha=0.8,
        )

        ax.set_title(
            "Portfolio Weight Allocation Over Time",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Weight", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(6, len(weights_history.columns)),
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        reports_dir = Path(self.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        save_path = reports_dir / "weight_allocation.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Weight allocation chart saved to %s", save_path)

        return fig
