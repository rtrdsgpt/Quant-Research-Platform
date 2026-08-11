"""
Portfolio performance metrics for the Multi-Dimensional Return Forecasting system.

This module provides :class:`PortfolioMetrics` which computes comprehensive
performance analytics required by the assignment:

Required metrics:
    - **Sharpe Ratio** — annualised risk-adjusted return.
    - **Maximum Drawdown** — peak-to-trough decline.
    - **Hit Ratio** — directional accuracy of predictions.
    - **Equity Curve** — visual plot of portfolio value over time.

Additional metrics:
    - Sortino Ratio, Calmar Ratio, Information Ratio,
      Win Rate, Profit Factor, Total / Annualised Return, Volatility.

Example:
    >>> from src.portfolio.metrics import PortfolioMetrics
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> pm = PortfolioMetrics(config)
    >>> all_metrics = pm.compute_all_metrics(portfolio_values, returns)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioMetrics:
    """Computes comprehensive portfolio performance metrics.

    Required metrics per assignment:
        - Sharpe Ratio
        - Maximum Drawdown
        - Hit Ratio (directional accuracy)
        - Equity Curve

    Additional metrics:
        - Sortino Ratio
        - Calmar Ratio
        - Information Ratio
        - Win Rate
        - Profit Factor
        - Total Return
        - Annualised Return
        - Annualised Volatility

    Attributes:
        risk_free_rate: Annualised risk-free rate.
        trading_days: Number of trading days per year (252).
        reports_dir: Directory for saving figures.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise from the full pipeline configuration.

        Args:
            config: Parsed ``config.yaml`` dictionary.

        Config parameters used (under ``portfolio``):
            - ``risk_free_rate``: Annualised risk-free rate (default 0.065).
        """
        self.config = config
        port_cfg = config.get("portfolio", {})

        self.risk_free_rate: float = port_cfg.get("risk_free_rate", 0.065)
        self.trading_days: int = 252
        self.reports_dir: str = config.get("paths", {}).get(
            "reports_dir", "reports"
        )

        logger.info(
            "PortfolioMetrics initialised — rf=%.4f, trading_days=%d",
            self.risk_free_rate,
            self.trading_days,
        )

    # ------------------------------------------------------------------
    # Return computation
    # ------------------------------------------------------------------

    def compute_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Compute portfolio returns from a value series.

        Args:
            portfolio_values: Series of portfolio values indexed by date.

        Returns:
            Series of daily simple returns (first value is ``NaN``-dropped).
        """
        returns = portfolio_values.pct_change().dropna()
        return returns

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Compute annualised Sharpe Ratio.

        .. math::
            \\text{Sharpe} = \\frac{\\bar{r} - r_f / 252}{\\sigma_r}
            \\times \\sqrt{252}

        Args:
            returns: Series of daily returns.

        Returns:
            Annualised Sharpe Ratio.
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for Sharpe ratio.")
            return 0.0

        excess = returns.mean() - self.risk_free_rate / self.trading_days
        std = returns.std(ddof=1)

        if std < 1e-12:
            logger.warning("Zero volatility — Sharpe ratio undefined.")
            return 0.0

        sharpe = (excess / std) * np.sqrt(self.trading_days)
        return float(sharpe)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Compute annualised Sortino Ratio.

        Like Sharpe but only penalises downside volatility.

        .. math::
            \\text{Sortino} = \\frac{\\bar{r} - r_f / 252}
            {\\sigma_{\\text{down}}} \\times \\sqrt{252}

        Args:
            returns: Series of daily returns.

        Returns:
            Annualised Sortino Ratio.
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for Sortino ratio.")
            return 0.0

        excess = returns.mean() - self.risk_free_rate / self.trading_days
        downside = returns[returns < 0]

        if len(downside) < 2:
            logger.warning("No downside returns — Sortino ratio undefined.")
            return 0.0

        downside_std = downside.std(ddof=1)
        if downside_std < 1e-12:
            return 0.0

        sortino = (excess / downside_std) * np.sqrt(self.trading_days)
        return float(sortino)

    def maximum_drawdown(self, portfolio_values: pd.Series) -> float:
        """Compute Maximum Drawdown.

        .. math::
            \\text{MDD} = \\max_t \\frac{\\text{peak}_t - \\text{value}_t}
            {\\text{peak}_t}

        Args:
            portfolio_values: Series of portfolio values (not returns).

        Returns:
            Maximum Drawdown as a positive decimal (e.g. 0.15 for 15 %).
        """
        if len(portfolio_values) < 2:
            return 0.0

        cummax = portfolio_values.cummax()
        drawdown = (cummax - portfolio_values) / cummax
        drawdown = drawdown.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        mdd = float(drawdown.max())
        return mdd

    def drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """Compute drawdown at each point in time.

        Args:
            portfolio_values: Series of portfolio values.

        Returns:
            Series of drawdown values (positive decimals).
        """
        cummax = portfolio_values.cummax()
        dd = (cummax - portfolio_values) / cummax
        dd = dd.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return dd

    def calmar_ratio(
        self, returns: pd.Series, portfolio_values: pd.Series
    ) -> float:
        """Compute Calmar Ratio.

        .. math::
            \\text{Calmar} = \\frac{\\text{Annualised Return}}
            {\\text{Maximum Drawdown}}

        Args:
            returns: Series of daily returns.
            portfolio_values: Series of portfolio values.

        Returns:
            Calmar Ratio.
        """
        ann_ret = self.annualized_return(returns)
        mdd = self.maximum_drawdown(portfolio_values)

        if mdd < 1e-12:
            logger.warning("Zero MDD — Calmar ratio undefined.")
            return 0.0

        return float(ann_ret / mdd)

    # ------------------------------------------------------------------
    # Directional / trade metrics
    # ------------------------------------------------------------------

    def hit_ratio(
        self, predicted_returns: pd.Series, actual_returns: pd.Series
    ) -> float:
        """Compute Hit Ratio (directional accuracy).

        .. math::
            \\text{Hit} = \\frac{1}{N} \\sum_{t}
            \\mathbb{1}[\\text{sign}(\\hat{r}_t) = \\text{sign}(r_t)]

        Args:
            predicted_returns: Series of predicted returns.
            actual_returns: Series of actual returns.

        Returns:
            Hit Ratio as decimal (e.g. 0.55 for 55 %).
        """
        # Align indices
        common = predicted_returns.index.intersection(actual_returns.index)
        if len(common) == 0:
            logger.warning("No overlapping dates for hit ratio.")
            return 0.0

        pred = predicted_returns.loc[common]
        actual = actual_returns.loc[common]

        correct = (np.sign(pred) == np.sign(actual)).sum()
        ratio = float(correct / len(common))
        return ratio

    def win_rate(self, returns: pd.Series) -> float:
        """Compute Win Rate (% of days with positive return).

        Args:
            returns: Series of daily returns.

        Returns:
            Win Rate as decimal.
        """
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))

    def profit_factor(self, returns: pd.Series) -> float:
        """Compute Profit Factor.

        .. math::
            \\text{PF} = \\frac{\\sum r_{r>0}}{|\\sum r_{r<0}|}

        Args:
            returns: Series of daily returns.

        Returns:
            Profit Factor (> 1 is profitable).
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses < 1e-12:
            if gains > 0:
                logger.info("No losses — profit factor is infinite.")
                return float("inf")
            return 0.0

        return float(gains / losses)

    def information_ratio(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Compute Information Ratio.

        .. math::
            \\text{IR} = \\frac{\\bar{r}_p - \\bar{r}_b}
            {\\sigma(r_p - r_b)} \\times \\sqrt{252}

        Args:
            returns: Portfolio daily returns.
            benchmark_returns: Benchmark (e.g. Nifty 50) daily returns.

        Returns:
            Information Ratio (annualised).
        """
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) < 2:
            logger.warning("Insufficient data for information ratio.")
            return 0.0

        excess = returns.loc[common] - benchmark_returns.loc[common]
        tracking_error = excess.std(ddof=1)

        if tracking_error < 1e-12:
            return 0.0

        ir = (excess.mean() / tracking_error) * np.sqrt(self.trading_days)
        return float(ir)

    # ------------------------------------------------------------------
    # Return / volatility aggregates
    # ------------------------------------------------------------------

    def total_return(self, portfolio_values: pd.Series) -> float:
        """Compute total return over the period.

        Args:
            portfolio_values: Series of portfolio values.

        Returns:
            Total return as decimal (e.g. 0.25 for 25 %).
        """
        if len(portfolio_values) < 2:
            return 0.0

        first = portfolio_values.iloc[0]
        last = portfolio_values.iloc[-1]

        if abs(first) < 1e-12:
            return 0.0

        return float((last - first) / first)

    def annualized_return(self, returns: pd.Series) -> float:
        """Compute annualised return.

        Uses geometric compounding:

        .. math::
            r_{\\text{ann}} = (1 + r_{\\text{total}})^{252 / N} - 1

        Args:
            returns: Series of daily returns.

        Returns:
            Annualised return.
        """
        if len(returns) == 0:
            return 0.0

        total = (1 + returns).prod() - 1
        n_days = len(returns)

        if n_days < 1:
            return 0.0

        ann = (1 + total) ** (self.trading_days / n_days) - 1
        return float(ann)

    def annualized_volatility(self, returns: pd.Series) -> float:
        """Compute annualised volatility.

        .. math::
            \\sigma_{\\text{ann}} = \\sigma_{\\text{daily}}
            \\times \\sqrt{252}

        Args:
            returns: Series of daily returns.

        Returns:
            Annualised standard deviation.
        """
        if len(returns) < 2:
            return 0.0
        return float(returns.std(ddof=1) * np.sqrt(self.trading_days))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def compute_all_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        predicted_returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute all performance metrics.

        Args:
            portfolio_values: Series of portfolio values.
            returns: Series of daily returns.
            predicted_returns: Optional predicted returns for hit ratio.
            benchmark_returns: Optional benchmark returns for IR.

        Returns:
            Dictionary mapping metric names to scalar values.
        """
        metrics: Dict[str, float] = {}

        metrics["Total_Return"] = self.total_return(portfolio_values)
        metrics["Annualized_Return"] = self.annualized_return(returns)
        metrics["Annualized_Volatility"] = self.annualized_volatility(returns)
        metrics["Sharpe_Ratio"] = self.sharpe_ratio(returns)
        metrics["Sortino_Ratio"] = self.sortino_ratio(returns)
        metrics["Maximum_Drawdown"] = self.maximum_drawdown(portfolio_values)
        metrics["Calmar_Ratio"] = self.calmar_ratio(returns, portfolio_values)
        metrics["Win_Rate"] = self.win_rate(returns)
        metrics["Profit_Factor"] = self.profit_factor(returns)

        if predicted_returns is not None:
            actual = returns.copy()
            metrics["Hit_Ratio"] = self.hit_ratio(predicted_returns, actual)
        else:
            metrics["Hit_Ratio"] = float("nan")

        if benchmark_returns is not None:
            metrics["Information_Ratio"] = self.information_ratio(
                returns, benchmark_returns
            )
        else:
            metrics["Information_Ratio"] = float("nan")

        logger.info(
            "All metrics computed — Sharpe=%.4f, MDD=%.4f, "
            "Total Return=%.4f",
            metrics["Sharpe_Ratio"],
            metrics["Maximum_Drawdown"],
            metrics["Total_Return"],
        )

        return metrics

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def create_equity_curve(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        metrics: Optional[Dict[str, float]] = None,
        title: str = "Portfolio Equity Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create equity curve visualisation.

        The plot includes:
            - **Top panel**: Portfolio value over time (line chart).
            - **Top panel (optional)**: Benchmark comparison.
            - **Bottom panel**: Drawdown area chart.
            - **Annotations**: Key metrics (Sharpe, MDD, Total Return).

        Args:
            portfolio_values: Series of portfolio values.
            benchmark_values: Optional benchmark values for comparison.
            metrics: Optional pre-computed metrics dictionary for
                annotation.
            title: Plot title.
            save_path: File path to save the figure.  If ``None``, saves
                to ``reports/equity_curve.png``.

        Returns:
            :class:`matplotlib.figure.Figure` object.
        """
        fig, (ax_equity, ax_dd) = plt.subplots(
            2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # --- Equity curve ---
        ax_equity.plot(
            portfolio_values.index,
            portfolio_values.values,
            label="Portfolio",
            color="#1f77b4",
            linewidth=1.5,
        )

        if benchmark_values is not None:
            # Normalise benchmark to same starting value
            bm = benchmark_values / benchmark_values.iloc[0] * portfolio_values.iloc[0]
            ax_equity.plot(
                bm.index,
                bm.values,
                label="Benchmark (Nifty 50)",
                color="#ff7f0e",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )

        ax_equity.set_title(title, fontsize=14, fontweight="bold")
        ax_equity.set_ylabel("Portfolio Value (INR)", fontsize=11)
        ax_equity.legend(loc="upper left", fontsize=10)
        ax_equity.grid(True, alpha=0.3)
        ax_equity.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
        )

        # --- Metrics annotation ---
        if metrics:
            annotation_lines = []
            if "Sharpe_Ratio" in metrics:
                annotation_lines.append(
                    f"Sharpe: {metrics['Sharpe_Ratio']:.2f}"
                )
            if "Maximum_Drawdown" in metrics:
                annotation_lines.append(
                    f"Max DD: {metrics['Maximum_Drawdown']:.2%}"
                )
            if "Total_Return" in metrics:
                annotation_lines.append(
                    f"Total Return: {metrics['Total_Return']:.2%}"
                )
            if "Hit_Ratio" in metrics and not np.isnan(
                metrics.get("Hit_Ratio", float("nan"))
            ):
                annotation_lines.append(
                    f"Hit Ratio: {metrics['Hit_Ratio']:.2%}"
                )

            if annotation_lines:
                text = "\n".join(annotation_lines)
                ax_equity.text(
                    0.02,
                    0.97,
                    text,
                    transform=ax_equity.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="gray",
                        alpha=0.9,
                    ),
                )

        # --- Drawdown chart ---
        dd = self.drawdown_series(portfolio_values)
        ax_dd.fill_between(
            dd.index,
            0,
            -dd.values,
            color="#d62728",
            alpha=0.4,
            label="Drawdown",
        )
        ax_dd.plot(dd.index, -dd.values, color="#d62728", linewidth=0.8)
        ax_dd.set_ylabel("Drawdown", fontsize=11)
        ax_dd.set_xlabel("Date", fontsize=11)
        ax_dd.legend(loc="lower left", fontsize=9)
        ax_dd.grid(True, alpha=0.3)
        ax_dd.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1%}")
        )

        # Date formatting
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_dir = Path(self.reports_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / "equity_curve.png")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Equity curve saved to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def create_performance_report(
        self,
        metrics: Dict[str, float],
        portfolio_values: pd.Series,
        returns: pd.Series,
    ) -> str:
        """Generate a text report of performance metrics.

        Args:
            metrics: Dictionary of computed metrics.
            portfolio_values: Series of portfolio values.
            returns: Series of daily returns.

        Returns:
            Formatted string with all metrics.
        """
        lines = [
            "=" * 60,
            "  PORTFOLIO PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"  Period: {portfolio_values.index[0].strftime('%Y-%m-%d')} "
            f"→ {portfolio_values.index[-1].strftime('%Y-%m-%d')}",
            f"  Trading Days: {len(returns)}",
            f"  Initial Value: ₹{portfolio_values.iloc[0]:,.2f}",
            f"  Final Value:   ₹{portfolio_values.iloc[-1]:,.2f}",
            "",
            "-" * 60,
            "  RETURN METRICS",
            "-" * 60,
            f"  Total Return:       {metrics.get('Total_Return', 0):.4%}",
            f"  Annualised Return:  {metrics.get('Annualized_Return', 0):.4%}",
            f"  Annualised Vol:     {metrics.get('Annualized_Volatility', 0):.4%}",
            "",
            "-" * 60,
            "  RISK-ADJUSTED METRICS",
            "-" * 60,
            f"  Sharpe Ratio:       {metrics.get('Sharpe_Ratio', 0):.4f}",
            f"  Sortino Ratio:      {metrics.get('Sortino_Ratio', 0):.4f}",
            f"  Calmar Ratio:       {metrics.get('Calmar_Ratio', 0):.4f}",
            "",
            "-" * 60,
            "  RISK METRICS",
            "-" * 60,
            f"  Maximum Drawdown:   {metrics.get('Maximum_Drawdown', 0):.4%}",
            "",
            "-" * 60,
            "  TRADE METRICS",
            "-" * 60,
            f"  Win Rate:           {metrics.get('Win_Rate', 0):.4%}",
            f"  Profit Factor:      {metrics.get('Profit_Factor', 0):.4f}",
        ]

        hit = metrics.get("Hit_Ratio", float("nan"))
        if not np.isnan(hit):
            lines.append(f"  Hit Ratio:          {hit:.4%}")

        ir = metrics.get("Information_Ratio", float("nan"))
        if not np.isnan(ir):
            lines.append(f"  Information Ratio:  {ir:.4f}")

        lines.extend(["", "=" * 60])

        report = "\n".join(lines)

        # Save report
        reports_dir = Path(self.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "performance_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Performance report saved to %s", report_path)

        return report
