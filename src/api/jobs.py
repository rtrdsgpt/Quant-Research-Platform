"""In-memory job store + pipeline stage execution for the API service.

Reuses the same staged pipeline pieces the CLI uses via --data-only /
--train-only / --backtest-only / --step N (main.py) as the service's
internal job stages: this module loads the pipeline's own cached
artifacts (feature parquet files, saved models, cached OHLCV) rather
than re-implementing forecasting/construction/backtesting logic, and
calls straight into src.models / src.construction / src.backtest.

This service does not fetch data or train models inline -- both are
long-running (network calls, model fits) and belong in the batch
pipeline (`python main.py --full`), not an HTTP request. Endpoints here
assume that pipeline has been run at least once so cached artifacts
exist; if not, they fail with a clear 4xx rather than silently
triggering a slow background fetch.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = "config/config.yaml"

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def get_config() -> dict:
    """Load config.yaml fresh each call -- cheap, and keeps the service
    honest about config edits without needing a restart."""
    return load_config(_CONFIG_PATH)


def _get_cached_forecaster():
    """Load trained models once per process and reuse the instance."""
    with _cache_lock:
        if "forecaster" not in _cache:
            from src.models import ReturnForecaster

            config = get_config()
            forecaster = ReturnForecaster(config)
            forecaster.load_models()
            _cache["forecaster"] = forecaster
        return _cache["forecaster"]


def _load_cached_feature_matrix(ticker: str) -> pd.DataFrame:
    """Load the cached feature-engineering output for one ticker."""
    config = get_config()
    features_dir = Path(config["paths"]["features_data"])
    path = features_dir / f"{ticker}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached feature matrix for '{ticker}' at {path}. "
            "Run `python main.py --data-only && python main.py --train-only` first."
        )
    return pd.read_parquet(path)


def forecast_ticker(ticker: str) -> Dict[str, Any]:
    """POST /forecast/{ticker}: latest predicted return for one ticker,
    using the cached trained model + cached feature matrix."""
    forecaster = _get_cached_forecaster()
    if ticker not in forecaster.fitted_models:
        raise KeyError(
            f"No trained model for ticker '{ticker}'. "
            f"Available: {sorted(forecaster.fitted_models)}"
        )

    features = _load_cached_feature_matrix(ticker)
    X, y = forecaster._split_X_y(features)
    y_pred = forecaster.predict(X, ticker)

    return {
        "ticker": ticker,
        "predicted_return": float(y_pred[-1]),
        "as_of": str(X.index[-1].date()),
    }


def _latest_predictions_for_universe(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    forecaster = _get_cached_forecaster()
    predictions: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            features = _load_cached_feature_matrix(ticker)
        except FileNotFoundError:
            continue
        if ticker not in forecaster.fitted_models:
            continue
        X, y = forecaster._split_X_y(features)
        y_pred = forecaster.predict(X, ticker)
        predictions[ticker] = pd.DataFrame(
            {"Predicted_Return": y_pred, "Actual_Return": y.values},
            index=X.index[: len(y_pred)],
        )
    return predictions


def construct_portfolio(
    alpha: Optional[Dict[str, float]],
    universe: Optional[List[str]],
    method: str,
    lookback_days: int,
) -> Dict[str, float]:
    """POST /portfolio/construct: build weights from a supplied or
    latest-cached alpha signal, via src.construction.alpha_portfolio."""
    from src.construction.alpha_portfolio import build_alpha_weights, latest_alpha_from_predictions
    from src.data import MarketDataFetcher

    config = get_config()
    tickers = universe or config["stocks"]["tickers"]

    if alpha is None:
        predictions = _latest_predictions_for_universe(tickers)
        if not predictions:
            raise FileNotFoundError(
                "No cached predictions available and no `alpha` supplied. "
                "Run `python main.py --full` first, or pass `alpha` explicitly."
            )
        alpha = latest_alpha_from_predictions(predictions)

    market_fetcher = MarketDataFetcher(config)
    ohlcv = market_fetcher.load_cached()
    if not ohlcv:
        raise FileNotFoundError("No cached OHLCV data. Run `python main.py --data-only` first.")

    returns = {}
    for ticker, df in ohlcv.items():
        if ticker not in alpha:
            continue
        price_col = "Adj_Close" if "Adj_Close" in df.columns else "Close"
        returns[ticker] = df[price_col].pct_change()
    returns_df = pd.DataFrame(returns).dropna(how="all").tail(lookback_days)

    return build_alpha_weights(returns_df, alpha, universe=universe, method=method)


def start_backtest_job() -> str:
    """POST /backtest: starts a forward-test backtest in a background
    thread (reusing cached models + cached data) and returns immediately."""
    run_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    thread = threading.Thread(target=_run_backtest_job, args=(run_id,), daemon=True)
    thread.start()
    return run_id


def _run_backtest_job(run_id: str) -> None:
    try:
        from src.backtest import PortfolioBacktester
        from src.data import MarketDataFetcher

        config = get_config()
        tickers = config["stocks"]["tickers"]
        predictions = _latest_predictions_for_universe(tickers)
        if not predictions:
            raise FileNotFoundError(
                "No cached predictions available -- run `python main.py --full` first."
            )

        market_fetcher = MarketDataFetcher(config)
        ohlcv = market_fetcher.load_cached()

        backtester = PortfolioBacktester(config)
        results = backtester.forward_test(predictions, ohlcv)

        with _JOBS_LOCK:
            _JOBS[run_id].update(status="completed", metrics=results["metrics"])
    except Exception as exc:
        logger.error("Backtest job %s failed: %s", run_id, exc, exc_info=True)
        with _JOBS_LOCK:
            _JOBS[run_id].update(status="failed", error=str(exc))


def get_job(run_id: str) -> Optional[Dict[str, Any]]:
    """GET /backtest/{run_id}: poll a backtest job's status/result."""
    with _JOBS_LOCK:
        job = _JOBS.get(run_id)
        return dict(job) if job is not None else None
