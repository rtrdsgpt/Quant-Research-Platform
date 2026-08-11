"""
Multi-Dimensional Return Forecasting and Portfolio Management System.

A comprehensive ML pipeline for predicting stock returns and managing
a portfolio of 6 Indian equities using market data, fundamentals,
macro indicators, and NLP sentiment analysis.

Modules:
    data: Data fetching and preprocessing modules
    features: Feature engineering pipelines
    models: Machine learning models and validation
    portfolio: Portfolio optimization and metrics
    utils: Utility functions and logging

Example:
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> print(config['stocks']['tickers'])
"""

__version__ = "1.0.0"
__author__ = "Assignment 02 Team"
