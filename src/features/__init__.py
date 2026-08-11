"""
Feature engineering modules for the Multi-Dimensional Return Forecasting system.

This package contains modules for building features from raw data:

Modules:
    technical_features: Price-based technical indicators (RSI, MACD, etc.)
    fundamental_features: Fundamental ratio features (P/E, D/E, ROE, EPS)
    macro_features: Macro indicator features (USD/INR returns, bond yields)
    sentiment_features: Sentiment score features from FinBERT
    feature_pipeline: Orchestrates all feature builders into unified pipeline

Example:
    >>> from src.features import FeaturePipeline
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> pipeline = FeaturePipeline(config)
    >>> train_features, test_features = pipeline.run(
    ...     market_data, fundamental_data, macro_data, sentiment_data
    ... )
"""

from src.features.fundamental_features import FundamentalFeatureEngineer
from src.features.macro_features import MacroFeatureEngineer
from src.features.sentiment_features import SentimentFeatureEngineer
from src.features.technical_features import TechnicalFeatureEngineer
from src.features.feature_pipeline import FeaturePipeline

__all__ = [
    "TechnicalFeatureEngineer",
    "FundamentalFeatureEngineer",
    "MacroFeatureEngineer",
    "SentimentFeatureEngineer",
    "FeaturePipeline",
]
