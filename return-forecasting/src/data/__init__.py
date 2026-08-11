"""
Data collection modules for the Multi-Dimensional Return Forecasting system.

This package contains modules for fetching and preprocessing data from
various sources:

Modules:
    market_data: Yahoo Finance OHLCV data fetcher
    fundamental_data: Fundamental data provider (yfinance + synthetic fallback)
    macro_data: Macro indicators fetcher (USD/INR, crude oil, gold, etc.)
    sentiment_data: News headlines + FinBERT sentiment pipeline

Example:
    >>> from src.data import MarketDataFetcher, FundamentalDataFetcher
    >>> from src.data import MacroDataFetcher, SentimentDataFetcher
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> market_fetcher = MarketDataFetcher(config)
    >>> data = market_fetcher.fetch_all_stocks()
"""

from src.data.fundamental_data import FundamentalDataFetcher
from src.data.macro_data import MacroDataFetcher
from src.data.market_data import MarketDataFetcher
from src.data.sentiment_data import SentimentDataFetcher

__all__ = [
    "MarketDataFetcher",
    "FundamentalDataFetcher",
    "MacroDataFetcher",
    "SentimentDataFetcher",
]
