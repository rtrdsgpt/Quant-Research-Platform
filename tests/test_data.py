"""
Tests for data fetching modules.

Tests cover:
    - MarketDataFetcher initialisation and fetching
    - FundamentalDataFetcher initialisation and fetching
    - MacroDataFetcher initialisation and fetching
    - SentimentDataFetcher initialisation and fetching
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.helpers import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load the default configuration for tests."""
    return load_config("config/config.yaml")


@pytest.fixture
def sample_ohlcv():
    """Create a sample OHLCV DataFrame for testing."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Adj_Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# MarketDataFetcher
# ---------------------------------------------------------------------------

class TestMarketDataFetcher:
    """Tests for the MarketDataFetcher class."""

    def test_init(self, config):
        """MarketDataFetcher should initialise without errors."""
        from src.data.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher(config)
        assert fetcher.tickers is not None
        assert len(fetcher.tickers) == 6

    def test_tickers_from_config(self, config):
        """Tickers should match those defined in config.yaml."""
        from src.data.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher(config)
        expected = config["stocks"]["tickers"]
        assert fetcher.tickers == expected

    @patch("src.data.market_data.yf.download")
    def test_fetch_single_stock(self, mock_download, config, sample_ohlcv):
        """fetch_single_stock should return a DataFrame with OHLCV columns."""
        from src.data.market_data import MarketDataFetcher

        mock_download.return_value = sample_ohlcv
        fetcher = MarketDataFetcher(config)

        result = fetcher.fetch_single_stock(
            "RELIANCE.NS", fetcher.start_date, fetcher.end_date
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch("src.data.market_data.yf.download")
    def test_fetch_all_stocks_returns_dict(self, mock_download, config, sample_ohlcv, tmp_path):
        """fetch_all_stocks should return a dict keyed by ticker.

        Redirects paths.raw_data/raw_market/processed_data to a pytest
        tmp_path *before* constructing MarketDataFetcher -- fetch_all_stocks()
        persists its (here, mocked) result to disk via those paths, and
        without this override it silently overwrites the real
        data/raw/market/*.parquet cache with 100 rows of this synthetic
        fixture every time the suite runs (this is exactly what happened
        and corrupted a real local training run -- see DECISIONS.md).
        """
        from src.data.market_data import MarketDataFetcher

        isolated_config = {
            **config,
            "paths": {
                **config["paths"],
                "raw_data": str(tmp_path / "raw"),
                "raw_market": str(tmp_path / "raw" / "market"),
                "processed_data": str(tmp_path / "processed"),
            },
        }

        mock_download.return_value = sample_ohlcv
        fetcher = MarketDataFetcher(isolated_config)

        result = fetcher.fetch_all_stocks()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_cached_file_not_found(self, config):
        """load_cached should skip missing files gracefully (not raise)."""
        from src.data.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher(config)
        # Point both the primary and fallback paths at a non-existent directory
        fetcher.raw_dir = "data/raw/nonexistent_market_test/"
        fetcher.raw_data_path = "data/raw/nonexistent_market_test/"
        result = fetcher.load_cached()
        assert result == {}


# ---------------------------------------------------------------------------
# FundamentalDataFetcher
# ---------------------------------------------------------------------------

class TestFundamentalDataFetcher:
    """Tests for the FundamentalDataFetcher class."""

    def test_init(self, config):
        """FundamentalDataFetcher should initialise without errors."""
        from src.data.fundamental_data import FundamentalDataFetcher

        fetcher = FundamentalDataFetcher(config)
        assert fetcher is not None

    def test_tickers_available(self, config):
        """Fetcher should have access to the stock universe."""
        from src.data.fundamental_data import FundamentalDataFetcher

        fetcher = FundamentalDataFetcher(config)
        tickers = config["stocks"]["tickers"]
        assert len(tickers) == 6


# ---------------------------------------------------------------------------
# MacroDataFetcher
# ---------------------------------------------------------------------------

class TestMacroDataFetcher:
    """Tests for the MacroDataFetcher class."""

    def test_init(self, config):
        """MacroDataFetcher should initialise without errors."""
        from src.data.macro_data import MacroDataFetcher

        fetcher = MacroDataFetcher(config)
        assert fetcher is not None

    def test_macro_symbols_configured(self, config):
        """Macro section should contain expected indicator keys."""
        macro_cfg = config.get("macro", {})
        assert "usdinr" in macro_cfg or "crude_oil" in macro_cfg


# ---------------------------------------------------------------------------
# SentimentDataFetcher
# ---------------------------------------------------------------------------

class TestSentimentDataFetcher:
    """Tests for the SentimentDataFetcher class."""

    def test_init(self, config):
        """SentimentDataFetcher should initialise without errors."""
        from src.data.sentiment_data import SentimentDataFetcher

        fetcher = SentimentDataFetcher(config)
        assert fetcher is not None

    def test_sentiment_config(self, config):
        """Sentiment config should specify FinBERT model name."""
        sentiment_cfg = config.get("sentiment", {})
        assert "model_name" in sentiment_cfg
        assert "finbert" in sentiment_cfg["model_name"].lower()


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for configuration loading."""

    def test_load_config(self):
        """load_config should return a non-empty dictionary."""
        config = load_config("config/config.yaml")
        assert isinstance(config, dict)
        assert "stocks" in config
        assert "dates" in config
        assert "paths" in config

    def test_load_config_missing_file(self):
        """load_config should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config("config/nonexistent.yaml")

    def test_config_stocks(self):
        """Config should contain exactly 6 tickers."""
        config = load_config("config/config.yaml")
        tickers = config["stocks"]["tickers"]
        assert len(tickers) == 6

    def test_config_dates(self):
        """Config should define start, end, train_end, test_start."""
        config = load_config("config/config.yaml")
        dates = config["dates"]
        assert "start" in dates
        assert "end" in dates
        assert "train_end" in dates
        assert "test_start" in dates
