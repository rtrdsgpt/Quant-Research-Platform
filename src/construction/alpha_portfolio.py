"""Glue layer: turns `src.models.forecaster`'s per-ticker Predicted_Return
signal into portfolio weights via `src.construction.weighting`.

This is the forecast -> construct integration point: it wires the
return-forecasting ensemble's (LightGBM/XGBoost/Ridge, benchmarked against
a SARIMAX baseline) predicted returns into the portfolio-replication
project's cvxpy mean-variance / Hierarchical Risk Parity weighting layer,
replacing that project's original "replicate the S&P 500" tracking-error
objective with an alpha-maximizing one.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.construction import sectors, weighting


def build_alpha_weights(
    returns_history: pd.DataFrame,
    alpha: Dict[str, float],
    universe: Optional[List[str]] = None,
    method: str = "hrp",
    risk_aversion: Optional[float] = None,
) -> Dict[str, float]:
    """Construct a portfolio from a forecasted alpha signal.

    Args:
        returns_history: Historical daily returns (date x ticker), used
            to estimate the covariance/correlation structure.
        alpha: Mapping ticker -> forecasted return (e.g. the ensemble's
            latest ``Predicted_Return`` per ticker).
        universe: Candidate tickers. Defaults to the intersection of
            ``returns_history.columns`` and ``alpha``'s keys.
        method: ``"markowitz"`` (alpha-maximizing mean-variance) or
            ``"hrp"`` (Hierarchical Risk Parity over the positive-alpha
            subset).
        risk_aversion: Passed through to
            ``weighting.alpha_markowitz_weights`` when
            ``method="markowitz"``.

    Returns:
        Dict[ticker, float] weights over the resolved universe,
        non-negative, summing to 1. Tickers outside that universe are
        absent (treat as 0 weight).
    """
    if universe is None:
        universe = [t for t in returns_history.columns if t in alpha]
    else:
        universe = [t for t in universe if t in returns_history.columns and t in alpha]
    if not universe:
        raise ValueError("No overlap between returns_history columns and alpha's tickers.")

    benchmark_sector_weights = sectors.compute_benchmark_sector_weights(universe)

    method = method.lower().strip()
    if method in ("markowitz", "alpha_markowitz"):
        return weighting.alpha_markowitz_weights(
            returns_history,
            alpha,
            universe,
            benchmark_sector_weights,
            risk_aversion=risk_aversion,
        )

    if method in ("hrp", "alpha_hrp"):
        # HRP ignores alpha magnitude by construction -- it only uses risk
        # structure -- so alpha is used here to pick the candidate set
        # (long only the tickers the ensemble forecasts a positive return
        # for); HRP then allocates risk-parity weights within that set.
        positive = [t for t in universe if float(alpha[t]) > 0]
        candidates = positive if len(positive) >= 2 else universe
        return weighting.hrp_weights(returns_history, candidates, benchmark_sector_weights)

    raise ValueError(f"Unknown alpha weighting method: '{method}'. Supported: markowitz, hrp")


def latest_alpha_from_predictions(predictions: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract the most recent ``Predicted_Return`` per ticker.

    Args:
        predictions: Output of ``ReturnForecaster.predict_all_stocks()`` --
            a dict mapping ticker -> DataFrame with a ``Predicted_Return``
            column indexed by date.

    Returns:
        Dict[ticker, float] of each ticker's latest predicted return.
    """
    alpha: Dict[str, float] = {}
    for ticker, df in predictions.items():
        if df is None or df.empty or "Predicted_Return" not in df.columns:
            continue
        alpha[ticker] = float(df["Predicted_Return"].iloc[-1])
    return alpha
