"""
Portfolio management modules for the Multi-Dimensional Return Forecasting system.

This package contains modules for portfolio optimization, performance metrics,
and backtesting:

Modules:
    optimizer: Portfolio weight optimization (equal weight, inverse volatility,
        mean-variance, risk parity, signal-weighted).
    metrics: Performance metrics (Sharpe ratio, max drawdown, hit ratio,
        equity curve, Sortino, Calmar, information ratio).
    backtester: Portfolio backtesting engine with transaction costs and
        strategy comparison.

Example:
    >>> from src.portfolio import PortfolioOptimizer, PortfolioMetrics, PortfolioBacktester
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> optimizer = PortfolioOptimizer(config)
    >>> weights = optimizer.optimize(returns_history, predicted_returns)
    >>> metrics = PortfolioMetrics(config)
    >>> performance = metrics.compute_all_metrics(portfolio_values, returns)
    >>> backtester = PortfolioBacktester(config)
    >>> results = backtester.run_backtest(predictions, prices, start, end)
"""

from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.metrics import PortfolioMetrics
from src.portfolio.backtester import PortfolioBacktester

__all__ = [
    "PortfolioOptimizer",
    "PortfolioMetrics",
    "PortfolioBacktester",
]
