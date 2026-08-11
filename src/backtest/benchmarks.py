"""Benchmarks the forecast-driven portfolio against equal-weight and
mean-variance baselines (todo.md #1: "Benchmark the forecast-driven
portfolio against equal-weight and mean-variance baselines").

Runs the same forward-test backtest under several `portfolio.method`
values and reports metrics side by side, so the alpha-construction methods
(`alpha_hrp` / `alpha_markowitz`, wired to the forecasting ensemble via
`src.construction.alpha_portfolio`) can be judged against simple baselines
rather than in isolation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import pandas as pd

from src.backtest.backtester import PortfolioBacktester
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METHODS: List[str] = [
    "equal_weight",
    "mean_variance",
    "alpha_hrp",
    "alpha_markowitz",
]


def run_benchmark_comparison(
    config: dict,
    predictions: Dict[str, pd.DataFrame],
    ohlcv_data: Dict[str, pd.DataFrame],
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the same forward-test backtest under several portfolio methods.

    Args:
        config: Base pipeline configuration (deep-copied per method run,
            never mutated).
        predictions: Per-ticker prediction DataFrames, e.g. from
            ``ReturnForecaster.predict_all_stocks()``.
        ohlcv_data: Per-ticker OHLCV DataFrames (for prices).
        methods: Portfolio construction methods to compare. Defaults to
            ``DEFAULT_METHODS`` (equal_weight, mean_variance, alpha_hrp,
            alpha_markowitz).

    Returns:
        DataFrame indexed by method name, one row of performance metrics
        (Sharpe, Sortino, Max Drawdown, Calmar, Hit Ratio, ...) each.
    """
    methods = methods or DEFAULT_METHODS
    rows: Dict[str, Dict[str, float]] = {}

    for method in methods:
        logger.info("Benchmark run — portfolio.method=%s", method)
        run_cfg = copy.deepcopy(config)
        run_cfg.setdefault("portfolio", {})["method"] = method
        try:
            backtester = PortfolioBacktester(run_cfg)
            results = backtester.forward_test(predictions, ohlcv_data)
            rows[method] = results["metrics"]
        except Exception as exc:
            logger.error("Benchmark run failed for method=%s: %s", method, exc, exc_info=True)
            rows[method] = {"error": str(exc)}

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "method"
    return comparison


def format_comparison_report(comparison: pd.DataFrame) -> str:
    """Render the comparison table as fixed-width text.

    Args:
        comparison: Output of :func:`run_benchmark_comparison`.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        "  PORTFOLIO CONSTRUCTION METHOD COMPARISON",
        "  (forward-test period; see config.dates.test_start/test_end)",
        "=" * 70,
        "",
        comparison.round(4).to_string(),
        "",
    ]
    return "\n".join(lines)
