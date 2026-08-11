"""
Tests for the forecast -> construct merge logic and the SARIMAX baseline.

Covers:
    - src.construction.weighting.alpha_markowitz_weights / hrp_weights
    - src.construction.alpha_portfolio (the forecaster -> weighting glue)
    - src.construction.optimizer's alpha_markowitz / alpha_hrp methods
    - src.models.sarimax_model.SARIMAXRegressor (sklearn-compatibility,
      walk-forward CV integration)
    - src.models.forecaster.ReturnForecaster's sarimax/benchmark_models wiring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.helpers import load_config

TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "M&M.NS", "BHARTIARTL.NS", "HINDUNILVR.NS"]


@pytest.fixture
def config():
    return load_config("config/config.yaml")


@pytest.fixture
def sample_returns():
    """Daily returns for the 6-stock universe, uncorrelated, low vol."""
    np.random.seed(7)
    dates = pd.bdate_range("2024-01-01", periods=250)
    return pd.DataFrame(
        np.random.normal(0, 0.01, size=(250, len(TICKERS))),
        index=dates,
        columns=TICKERS,
    )


@pytest.fixture
def sample_alpha():
    """A forecasted-return signal: some positive, some negative."""
    np.random.seed(11)
    return {t: float(v) for t, v in zip(TICKERS, np.random.normal(0, 0.002, len(TICKERS)))}


# ---------------------------------------------------------------------------
# weighting.alpha_markowitz_weights / hrp_weights
# ---------------------------------------------------------------------------

class TestAlphaWeighting:
    def test_alpha_markowitz_weights_sum_to_one_and_respect_cap(self, sample_returns, sample_alpha):
        from src.construction import sectors, weighting

        benchmark_sector_weights = sectors.compute_benchmark_sector_weights(TICKERS)
        weights = weighting.alpha_markowitz_weights(
            sample_returns, sample_alpha, TICKERS, benchmark_sector_weights
        )
        assert set(weights) == set(TICKERS)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in weights.values())
        assert all(w <= weighting.config.MAX_STOCK_WEIGHT + 1e-6 for w in weights.values())

    def test_alpha_markowitz_weights_handles_single_row_lookback(self, sample_alpha):
        """A 1-row returns history (e.g. the first rebalance date in a
        short backtest window, before any lookback has accumulated)
        makes np.cov produce an all-NaN matrix. Ridge epsilon alone
        doesn't fix that (NaN + anything is NaN); this used to blow up
        with cvxpy's "Quadratic form matrices must be symmetric/
        Hermitian" the first time the real backtest hit it."""
        from src.construction import sectors, weighting

        single_row = pd.DataFrame([[0.01, -0.02, 0.005, 0.0, 0.003, -0.001]], columns=TICKERS)
        benchmark_sector_weights = sectors.compute_benchmark_sector_weights(TICKERS)

        weights = weighting.alpha_markowitz_weights(
            single_row, sample_alpha, TICKERS, benchmark_sector_weights
        )
        assert set(weights) == set(TICKERS)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(np.isfinite(w) for w in weights.values())

    def test_hrp_weights_sum_to_one(self, sample_returns):
        from src.construction import sectors, weighting

        benchmark_sector_weights = sectors.compute_benchmark_sector_weights(TICKERS)
        weights = weighting.hrp_weights(sample_returns, TICKERS, benchmark_sector_weights)
        assert set(weights) == set(TICKERS)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in weights.values())

    def test_higher_alpha_gets_more_markowitz_weight(self, sample_returns):
        """All else equal, the ticker with the highest forecasted alpha
        should not receive the smallest weight."""
        from src.construction import sectors, weighting

        alpha = {t: 0.0 for t in TICKERS}
        alpha["INFY.NS"] = 0.05  # dominant positive signal
        benchmark_sector_weights = sectors.compute_benchmark_sector_weights(TICKERS)
        weights = weighting.alpha_markowitz_weights(
            sample_returns, alpha, TICKERS, benchmark_sector_weights
        )
        assert weights["INFY.NS"] == max(weights.values())


# ---------------------------------------------------------------------------
# alpha_portfolio (forecaster -> weighting glue)
# ---------------------------------------------------------------------------

class TestAlphaPortfolio:
    def test_build_alpha_weights_hrp(self, sample_returns, sample_alpha):
        from src.construction.alpha_portfolio import build_alpha_weights

        weights = build_alpha_weights(sample_returns, sample_alpha, TICKERS, method="hrp")
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in weights.values())

    def test_build_alpha_weights_markowitz(self, sample_returns, sample_alpha):
        from src.construction.alpha_portfolio import build_alpha_weights

        weights = build_alpha_weights(sample_returns, sample_alpha, TICKERS, method="markowitz")
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_hrp_restricts_to_positive_alpha_subset(self, sample_returns):
        from src.construction.alpha_portfolio import build_alpha_weights

        alpha = {t: -0.01 for t in TICKERS}
        alpha["RELIANCE.NS"] = 0.02
        alpha["INFY.NS"] = 0.03
        weights = build_alpha_weights(sample_returns, alpha, TICKERS, method="hrp")
        # Only the two positive-alpha tickers should receive weight.
        nonzero = {t for t, w in weights.items() if w > 1e-9}
        assert nonzero == {"RELIANCE.NS", "INFY.NS"}

    def test_unknown_method_raises(self, sample_returns, sample_alpha):
        from src.construction.alpha_portfolio import build_alpha_weights

        with pytest.raises(ValueError):
            build_alpha_weights(sample_returns, sample_alpha, TICKERS, method="not_a_method")

    def test_latest_alpha_from_predictions(self):
        from src.construction.alpha_portfolio import latest_alpha_from_predictions

        dates = pd.bdate_range("2025-01-01", periods=3)
        predictions = {
            "RELIANCE.NS": pd.DataFrame({"Predicted_Return": [0.001, 0.002, 0.003]}, index=dates),
            "INFY.NS": pd.DataFrame({"Predicted_Return": [-0.001, -0.002, -0.0005]}, index=dates),
        }
        alpha = latest_alpha_from_predictions(predictions)
        assert alpha == {"RELIANCE.NS": 0.003, "INFY.NS": -0.0005}


# ---------------------------------------------------------------------------
# optimizer.py's alpha_markowitz / alpha_hrp dispatch
# ---------------------------------------------------------------------------

class TestOptimizerAlphaMethods:
    def test_optimize_alpha_hrp(self, config, sample_returns, sample_alpha):
        from src.construction.optimizer import PortfolioOptimizer

        run_cfg = dict(config)
        run_cfg["portfolio"] = dict(config["portfolio"], method="alpha_hrp")
        optimizer = PortfolioOptimizer(run_cfg)
        predicted = np.array([sample_alpha[t] for t in TICKERS])
        weights = optimizer.optimize(sample_returns, predicted)
        assert weights.shape == (len(TICKERS),)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_optimize_alpha_markowitz(self, config, sample_returns, sample_alpha):
        from src.construction.optimizer import PortfolioOptimizer

        run_cfg = dict(config)
        run_cfg["portfolio"] = dict(config["portfolio"], method="alpha_markowitz")
        optimizer = PortfolioOptimizer(run_cfg)
        predicted = np.array([sample_alpha[t] for t in TICKERS])
        weights = optimizer.optimize(sample_returns, predicted)
        assert weights.shape == (len(TICKERS),)
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_alpha_methods_fall_back_to_equal_weight_without_predictions(self, config, sample_returns):
        from src.construction.optimizer import PortfolioOptimizer

        run_cfg = dict(config)
        run_cfg["portfolio"] = dict(config["portfolio"], method="alpha_hrp")
        optimizer = PortfolioOptimizer(run_cfg)
        weights = optimizer.optimize(sample_returns, predicted_returns=None)
        assert np.allclose(weights, 1.0 / len(TICKERS))


# ---------------------------------------------------------------------------
# SARIMAX baseline
# ---------------------------------------------------------------------------

class TestSARIMAXRegressor:
    def test_is_sklearn_compatible(self):
        from sklearn.base import clone

        from src.models.sarimax_model import SARIMAXRegressor

        model = SARIMAXRegressor(order=(1, 0, 0), purge_gap=5)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        assert not hasattr(cloned, "results_")

    def test_fit_predict_shape(self):
        from src.models.sarimax_model import SARIMAXRegressor

        np.random.seed(3)
        y = np.random.normal(0, 0.01, 200)
        X = np.random.normal(0, 1, size=(200, 4))

        model = SARIMAXRegressor(order=(1, 0, 0), purge_gap=5)
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:170])
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        from src.models.sarimax_model import SARIMAXRegressor

        model = SARIMAXRegressor()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 2)))

    def test_walk_forward_cv_integration(self, config):
        from src.models.sarimax_model import SARIMAXRegressor
        from src.models.walk_forward import WalkForwardValidator

        run_cfg = dict(config)
        run_cfg["model"] = dict(
            config["model"],
            walk_forward={
                "initial_train_days": 150,
                "validation_days": 20,
                "step_days": 20,
                "expanding": True,
            },
        )
        validator = WalkForwardValidator(run_cfg)

        np.random.seed(4)
        y = np.random.normal(0, 0.01, 250)
        X = np.random.normal(0, 1, size=(250, 3))

        model = SARIMAXRegressor(order=(1, 0, 0), purge_gap=5)
        scores = validator.cross_validate(model, X, y, metrics=["mse", "direction_accuracy"])
        assert len(scores["mse"]) >= 1
        assert all(np.isfinite(v) for v in scores["mse"])


class TestForecasterSarimaxWiring:
    def test_create_model_sarimax(self, config):
        from src.models.forecaster import ReturnForecaster
        from src.models.sarimax_model import SARIMAXRegressor

        forecaster = ReturnForecaster(config)
        model = forecaster.create_model("sarimax")
        assert isinstance(model, SARIMAXRegressor)

    def test_create_model_unknown_raises(self, config):
        from src.models.forecaster import ReturnForecaster

        forecaster = ReturnForecaster(config)
        with pytest.raises(ValueError):
            forecaster.create_model("not_a_real_model")

    def test_benchmark_models_not_blended_into_ensemble(self, config):
        """sarimax listed in benchmark_models should not appear in
        ensemble_model_names (it's CV'd for comparison, not blended)."""
        from src.models.forecaster import ReturnForecaster

        forecaster = ReturnForecaster(config)
        assert "sarimax" in forecaster.benchmark_model_names
        assert "sarimax" not in forecaster.ensemble_model_names
