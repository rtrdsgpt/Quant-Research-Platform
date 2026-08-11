"""
Backtesting and evaluation modules for the forecast -> construct -> backtest
platform.

Modules:
    metrics: Performance metrics (Sharpe ratio, max drawdown, hit ratio,
        equity curve, Sortino, Calmar, information ratio).
    backtester: Portfolio backtesting engine with transaction costs and
        strategy comparison.
    replication_evaluation / replication_report / replication_visualization:
        rolling-CV tracking-error evaluation inherited from the legacy
        index-replication mode (see src/construction/selection.py).
    benchmarks: runs the forecast-driven portfolio against equal-weight and
        mean-variance baselines and reports a comparison table.

Example:
    >>> from src.backtest import PortfolioMetrics, PortfolioBacktester
    >>> from src.construction.optimizer import PortfolioOptimizer
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> optimizer = PortfolioOptimizer(config)
    >>> weights = optimizer.optimize(returns_history, predicted_returns)
    >>> metrics = PortfolioMetrics(config)
    >>> performance = metrics.compute_all_metrics(portfolio_values, returns)
    >>> backtester = PortfolioBacktester(config)
    >>> results = backtester.run_backtest(predictions, prices, start, end)
"""

from src.construction.optimizer import PortfolioOptimizer
from src.backtest.metrics import PortfolioMetrics
from src.backtest.backtester import PortfolioBacktester

__all__ = [
    "PortfolioOptimizer",
    "PortfolioMetrics",
    "PortfolioBacktester",
]
