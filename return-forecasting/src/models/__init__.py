"""
Machine learning model modules for the Multi-Dimensional Return Forecasting system.

This package contains modules for model training, validation, and prediction:

Modules:
    walk_forward: Walk-forward (time series) cross-validation engine
    feature_selection: Recursive Feature Elimination and importance analysis
    forecaster: Return prediction models (LightGBM, XGBoost, ensemble)

Example:
    >>> from src.models import ReturnForecaster, WalkForwardValidator, FeatureSelector
    >>> from src.utils.helpers import load_config
    >>> config = load_config()
    >>> forecaster = ReturnForecaster(config)
    >>> results = forecaster.train_all_stocks(feature_matrices)
"""

from src.models.feature_selection import FeatureSelector
from src.models.forecaster import ReturnForecaster
from src.models.walk_forward import WalkForwardValidator

__all__ = [
    "WalkForwardValidator",
    "FeatureSelector",
    "ReturnForecaster",
]
