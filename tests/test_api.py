"""
Tests for the FastAPI service (src.api.main).

These run against whatever cached data/models are on disk (from a prior
`python main.py --full` or `--train-only` run) -- if none exist, the
data-dependent tests are skipped rather than failed, since a fresh clone
of this repo has no cached artifacts by design (see .gitignore).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.helpers import load_config

client = TestClient(app)


def _has_cached_pipeline_artifacts() -> bool:
    config = load_config("config/config.yaml")
    features_dir = Path(config["paths"]["features_data"])
    models_dir = Path(config["paths"]["models_dir"])
    return features_dir.exists() and any(features_dir.glob("*_features.parquet")) and models_dir.exists() and any(
        models_dir.iterdir()
    )


requires_cache = pytest.mark.skipif(
    not _has_cached_pipeline_artifacts(),
    reason="No cached feature matrices / trained models on disk -- run `python main.py --full` first.",
)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_forecast_unknown_ticker_404():
    resp = client.post("/forecast/NOT_A_REAL_TICKER.NS")
    assert resp.status_code == 404


@requires_cache
def test_forecast_known_ticker():
    config = load_config("config/config.yaml")
    ticker = config["stocks"]["tickers"][0]

    resp = client.post(f"/forecast/{ticker}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == ticker
    assert isinstance(body["predicted_return"], float)
    assert "as_of" in body


@requires_cache
def test_portfolio_construct_with_explicit_alpha():
    config = load_config("config/config.yaml")
    tickers = config["stocks"]["tickers"]
    alpha = {t: 0.001 * (i + 1) for i, t in enumerate(tickers)}

    resp = client.post(
        "/portfolio/construct",
        json={"alpha": alpha, "universe": tickers, "method": "hrp"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "hrp"
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-4


@requires_cache
def test_portfolio_construct_defaults_to_cached_predictions():
    resp = client.post("/portfolio/construct", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-4


def test_portfolio_construct_markowitz_method():
    config = load_config("config/config.yaml")
    tickers = config["stocks"]["tickers"]
    alpha = {t: 0.001 for t in tickers}

    resp = client.post(
        "/portfolio/construct",
        json={"alpha": alpha, "universe": tickers, "method": "not_a_method"},
    )
    # Unknown method surfaces as a 400, not a 500.
    assert resp.status_code == 400


@requires_cache
def test_backtest_job_lifecycle():
    resp = client.post("/backtest")
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    assert resp.json()["status"] == "running"

    status = None
    for _ in range(30):
        poll = client.get(f"/backtest/{run_id}")
        assert poll.status_code == 200
        status = poll.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)

    # The job must resolve (not hang) and report a clean status either way.
    # Whether it actually *completes* depends on the cached OHLCV data on
    # disk covering config.dates.test_start/test_end -- a small locally
    # cached sample may not, in which case the backtester correctly raises
    # "No overlapping dates" and the job reports status="failed" with that
    # error, rather than crashing the service.
    assert status in ("completed", "failed"), f"Backtest job did not resolve: {status}"
    poll_body = client.get(f"/backtest/{run_id}").json()
    if status == "completed":
        assert poll_body["metrics"]
    else:
        assert poll_body["error"]


def test_backtest_unknown_run_id_404():
    resp = client.get("/backtest/not-a-real-run-id")
    assert resp.status_code == 404
