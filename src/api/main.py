"""FastAPI service wrapping the forecast -> construct -> backtest pipeline.

Endpoints (todo.md's API layer spec):
    POST /forecast/{ticker}      - latest predicted return for one ticker
    POST /portfolio/construct    - build weights from an alpha signal
    GET  /backtest/{run_id}      - poll a backtest job's status/result
    POST /backtest                - start an async forward-test backtest job
                                     (needed to obtain a run_id for the
                                     endpoint above -- not in the original
                                     spec list but required to use it)

Requires the pipeline to have been run at least once (`python main.py
--full`, or `--data-only` + `--train-only`) so cached feature matrices,
OHLCV, and trained models exist on disk -- this service reads those
staged artifacts rather than fetching data or training inline.

Run with: uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api import jobs
from src.api.schemas import (
    BacktestJobResponse,
    BacktestResultResponse,
    ForecastResponse,
    PortfolioConstructRequest,
    PortfolioConstructResponse,
)

app = FastAPI(
    title="Quant Research Platform API",
    description="Forecast -> construct -> backtest, served over HTTP.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast/{ticker}", response_model=ForecastResponse)
def forecast(ticker: str) -> ForecastResponse:
    """Latest predicted return for `ticker`, from the cached trained
    ensemble model (see src.models.forecaster.ReturnForecaster)."""
    try:
        result = jobs.forecast_ticker(ticker)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ForecastResponse(**result)


@app.post("/portfolio/construct", response_model=PortfolioConstructResponse)
def construct_portfolio(request: PortfolioConstructRequest) -> PortfolioConstructResponse:
    """Build portfolio weights from a forecasted alpha signal via
    src.construction.alpha_portfolio (the forecast -> construct wiring)."""
    try:
        weights = jobs.construct_portfolio(
            alpha=request.alpha,
            universe=request.universe,
            method=request.method,
            lookback_days=request.lookback_days,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PortfolioConstructResponse(method=request.method, weights=weights)


@app.post("/backtest", response_model=BacktestJobResponse, status_code=202)
def start_backtest() -> BacktestJobResponse:
    """Start an async forward-test backtest job (see src.backtest.backtester
    .PortfolioBacktester) and return its run_id immediately."""
    run_id = jobs.start_backtest_job()
    return BacktestJobResponse(run_id=run_id, status="running")


@app.get("/backtest/{run_id}", response_model=BacktestResultResponse)
def get_backtest(run_id: str) -> BacktestResultResponse:
    """Poll a backtest job started via POST /backtest."""
    job = jobs.get_job(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No backtest job with run_id '{run_id}'.")
    return BacktestResultResponse(
        run_id=run_id,
        status=job["status"],
        metrics=job.get("metrics"),
        error=job.get("error"),
    )
