"""Pydantic request/response models for the API service."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastResponse(BaseModel):
    ticker: str
    predicted_return: float
    as_of: str


class PortfolioConstructRequest(BaseModel):
    alpha: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Ticker -> forecasted return. If omitted, the latest cached "
            "model predictions are used for `universe` (or the full "
            "configured stock universe)."
        ),
    )
    universe: Optional[List[str]] = Field(
        default=None, description="Candidate tickers. Defaults to the configured stock universe."
    )
    method: str = Field(default="hrp", description="'hrp' or 'markowitz'")
    lookback_days: int = Field(default=252, description="Trading days of history used to estimate covariance.")


class PortfolioConstructResponse(BaseModel):
    method: str
    weights: Dict[str, float]


class BacktestJobResponse(BaseModel):
    run_id: str
    status: str


class BacktestResultResponse(BaseModel):
    run_id: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
