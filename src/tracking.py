"""MLflow tracking integration.

Logs each walk-forward CV fold, for every model variant (LightGBM /
XGBoost / Ridge, and the SARIMAX baseline when listed in
`model.benchmark_models`), as an MLflow run nested under one parent run
per ticker -- todo.md's MLOps section: "log each walk-forward fold + model
variant ... as an MLflow run; existing .joblib artifacts per stock/model
map directly onto MLflow's artifact store."

Enabled via `python main.py --train-only --mlflow` (or `--full --mlflow`).
Not required for normal runs -- training works identically without it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tracking_uri(config: dict) -> str:
    return config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")


def _experiment_name(config: dict) -> str:
    return config.get("mlflow", {}).get("experiment_name", "quant-research-platform")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(value)
    except TypeError:
        return False


def log_training_results(config: dict, training_results: Dict[str, Dict[str, Any]]) -> None:
    """Log `ReturnForecaster.train_all_stocks()`'s output to MLflow.

    Structure: one parent run per ticker (tagged `ticker`, logging the
    final inverse-MSE ensemble weights), with one nested run per model
    variant logging that variant's per-fold walk-forward CV metrics
    (`cv_{metric}` per fold as a step series, plus `cv_{metric}_mean`)
    and, if the fitted model was saved to disk, its `.joblib` file as an
    MLflow artifact.

    Args:
        config: Full pipeline configuration (reads the `mlflow` section
            for `tracking_uri` / `experiment_name`, both optional).
        training_results: Output of `ReturnForecaster.train_all_stocks()`.
    """
    import mlflow

    mlflow.set_tracking_uri(_tracking_uri(config))
    mlflow.set_experiment(_experiment_name(config))

    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    n_logged = 0

    for ticker, result in training_results.items():
        if "error" in result:
            continue

        with mlflow.start_run(run_name=f"train_{ticker}"):
            mlflow.set_tag("ticker", ticker)

            for model_type, weight in result.get("weights", {}).items():
                mlflow.log_metric(f"ensemble_weight_{model_type}", weight)

            cv_scores = result.get("cv_scores", {})
            for model_type, scores in cv_scores.items():
                with mlflow.start_run(run_name=f"{ticker}_{model_type}", nested=True):
                    mlflow.set_tags({"ticker": ticker, "model_type": model_type})

                    for metric_name, fold_values in scores.items():
                        finite_values = []
                        for fold_idx, value in enumerate(fold_values):
                            if not _is_finite(value):
                                continue
                            mlflow.log_metric(f"cv_{metric_name}", float(value), step=fold_idx)
                            finite_values.append(float(value))
                        if finite_values:
                            mlflow.log_metric(
                                f"cv_{metric_name}_mean",
                                sum(finite_values) / len(finite_values),
                            )

                    artifact_path = models_dir / ticker.replace(".", "_") / f"{model_type}.joblib"
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path), artifact_path="model")

            n_logged += 1

    logger.info(
        "Logged training results for %d/%d tickers to MLflow (tracking_uri=%s, experiment=%s)",
        n_logged,
        len(training_results),
        _tracking_uri(config),
        _experiment_name(config),
    )
