"""Tests for src.tracking (MLflow logging of walk-forward CV results)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.helpers import load_config


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config/config.yaml")
    cfg["mlflow"] = {
        "tracking_uri": f"sqlite:///{tmp_path}/mlflow_test.db",
        "experiment_name": "pytest-quant-research-platform",
    }
    cfg["paths"] = {**cfg["paths"], "models_dir": str(tmp_path / "models")}
    return cfg


@pytest.fixture
def training_results():
    return {
        "RELIANCE.NS": {
            "weights": {"lightgbm": 0.6, "ridge": 0.4},
            "cv_scores": {
                "lightgbm": {"mse": [0.0001, 0.00012, np.nan], "direction_accuracy": [0.55, 0.52, 0.6]},
                "ridge": {"mse": [0.00015, 0.00014], "direction_accuracy": [0.51, 0.5]},
                "sarimax": {"mse": [0.0002, 0.00019], "direction_accuracy": [0.5, 0.48]},
            },
        },
        "INFY.NS": {"error": "training failed"},
    }


def test_log_training_results_creates_runs(config, training_results):
    import mlflow

    from src.tracking import log_training_results

    log_training_results(config, training_results)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[config["mlflow"]["experiment_name"]])

    run_names = set(runs["tags.mlflow.runName"])
    assert "train_RELIANCE.NS" in run_names
    assert "RELIANCE.NS_lightgbm" in run_names
    assert "RELIANCE.NS_ridge" in run_names
    assert "RELIANCE.NS_sarimax" in run_names
    # The errored ticker must not produce any run.
    assert not any("INFY" in name for name in run_names)


def test_log_training_results_skips_nan_folds(config, training_results):
    import mlflow

    from src.tracking import log_training_results

    log_training_results(config, training_results)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string="tags.model_type = 'lightgbm'",
    )
    assert len(runs) == 1
    # 3 fold values for lightgbm mse, one NaN -> mean over the 2 finite ones.
    expected_mean = float(np.mean([0.0001, 0.00012]))
    assert runs.iloc[0]["metrics.cv_mse_mean"] == pytest.approx(expected_mean)
