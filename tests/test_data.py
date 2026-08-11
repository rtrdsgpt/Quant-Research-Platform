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

        result = fetcher.fetch_single_stock("RELIANCE.NS")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch("src.data.market_data.yf.download")
    def test_fetch_all_stocks_returns_dict(self, mock_download, config, sample_ohlcv):
        """fetch_all_stocks should return a dict keyed by ticker."""
        from src.data.market_data import MarketDataFetcher

        mock_download.return_value = sample_ohlcv
        fetcher = MarketDataFetcher(config)

        result = fetcher.fetch_all_stocks()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_load_cached_file_not_found(self, config):
        """load_cached should handle missing files gracefully."""
        from src.data.market_data import MarketDataFetcher

        fetcher = MarketDataFetcher(config)
        # Point to a non-existent directory
        fetcher.raw_data_path = "data/raw/nonexistent_market_test/"
        with pytest.raises((FileNotFoundError, Exception)):
            fetcher.load_cached()


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
