#!/usr/bin/env python
"""
Main Pipeline Orchestrator — Quant Research Platform.

Coordinates the full forecast -> construct -> backtest pipeline: data
collection, feature engineering, ensemble model training (walk-forward
CV'd against a SARIMAX baseline), and portfolio construction + backtest
(forecast-driven alpha construction via src/construction, benchmarked
against equal-weight / mean-variance baselines).

Usage:
    python main.py --full          # Run complete pipeline
    python main.py --data-only     # Only fetch data
    python main.py --train-only    # Only train models (requires data)
    python main.py --backtest-only # Only run backtest (requires trained models)
    python main.py --step 3        # Resume from step 3
    python main.py --backtest-only --benchmark  # Also compare vs baselines

Steps:
    1. Data collection (market, fundamental, macro, sentiment)
    2. Feature engineering (technical, fundamental, macro, sentiment)
    3. Model training (walk-forward CV + ensemble, SARIMAX benchmark)
    4. Portfolio backtest (forward test period, per config.dates)
    5. (optional, --benchmark) Compare the configured portfolio.method
       against equal-weight / mean-variance / alpha_hrp / alpha_markowitz
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Optional, Tuple

from src.utils.helpers import load_config, ensure_directories, set_random_seed
from src.utils.logger import setup_logger


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with pipeline control flags.
    """
    parser = argparse.ArgumentParser(
        description="Quant Research Platform — forecast -> construct -> backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --full\n"
            "  python main.py --data-only\n"
            "  python main.py --train-only\n"
            "  python main.py --backtest-only\n"
            "  python main.py --step 3\n"
            "  python main.py --backtest-only --benchmark\n"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete pipeline (steps 1-4)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only fetch and cache data (step 1)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only build features and train models (steps 2-3, requires data)",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Only run the portfolio backtest (step 4, requires trained models)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Resume from a specific step (1-4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "After step 4, also run the forward-test backtest under "
            "equal_weight / mean_variance / alpha_hrp / alpha_markowitz "
            "and print a comparison table (step 5)"
        ),
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log walk-forward CV folds and model variants to MLflow during training",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def fetch_all_data(
    config: dict, logger: Any
) -> Tuple[Dict, Dict, Any, Dict]:
    """Stage 1 — Fetch all data from various sources.

    Downloads OHLCV market data, fundamental data, macro indicators,
    and sentiment scores for the full stock universe.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple of ``(ohlcv_data, fundamental_data, macro_data, sentiment_data)``.
    """
    from src.data import (
        MarketDataFetcher,
        FundamentalDataFetcher,
        MacroDataFetcher,
        SentimentDataFetcher,
    )

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA COLLECTION")
    logger.info("=" * 60)

    logger.info("Fetching market data...")
    t0 = time.perf_counter()
    market_fetcher = MarketDataFetcher(config)
    ohlcv_data = market_fetcher.fetch_all_stocks()
    logger.info(
        "Market data fetched for %d tickers in %.1fs",
        len(ohlcv_data),
        time.perf_counter() - t0,
    )

    logger.info("Fetching fundamental data...")
    t0 = time.perf_counter()
    fundamental_fetcher = FundamentalDataFetcher(config)
    fundamental_data = fundamental_fetcher.fetch_all_fundamentals()
    logger.info(
        "Fundamental data fetched for %d tickers in %.1fs",
        len(fundamental_data),
        time.perf_counter() - t0,
    )

    logger.info("Fetching macro indicators...")
    t0 = time.perf_counter()
    macro_fetcher = MacroDataFetcher(config)
    macro_data = macro_fetcher.fetch_all_macro()
    logger.info(
        "Macro data fetched (%d rows) in %.1fs",
        len(macro_data) if hasattr(macro_data, "__len__") else 0,
        time.perf_counter() - t0,
    )

    logger.info("Fetching sentiment data...")
    t0 = time.perf_counter()
    sentiment_fetcher = SentimentDataFetcher(config)
    sentiment_data = sentiment_fetcher.fetch_all_sentiment()
    logger.info(
        "Sentiment data fetched for %d tickers in %.1fs",
        len(sentiment_data),
        time.perf_counter() - t0,
    )

    logger.info("Data collection complete!")
    return ohlcv_data, fundamental_data, macro_data, sentiment_data


def build_features(
    config: dict,
    logger: Any,
    ohlcv_data: Dict,
    fundamental_data: Dict,
    macro_data: Any,
    sentiment_data: Dict,
) -> Dict:
    """Stage 2 — Build feature matrices for all stocks.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        ohlcv_data: Dict mapping ticker -> OHLCV DataFrame.
        fundamental_data: Dict mapping ticker -> fundamental DataFrame.
        macro_data: Macro indicator DataFrame.
        sentiment_data: Dict mapping ticker -> sentiment DataFrame.

    Returns:
        Dictionary mapping ticker -> feature DataFrame (with ``Target``).
    """
    from src.features import FeaturePipeline

    logger.info("=" * 60)
    logger.info("STAGE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    pipeline = FeaturePipeline(config)
    feature_matrices = pipeline.build_all_feature_matrices(
        ohlcv_data, fundamental_data, macro_data, sentiment_data
    )

    total_rows = sum(len(df) for df in feature_matrices.values())
    logger.info(
        "Feature engineering complete — %d tickers, %d total rows in %.1fs",
        len(feature_matrices),
        total_rows,
        time.perf_counter() - t0,
    )
    return feature_matrices


def train_models(
    config: dict, logger: Any, feature_matrices: Dict, use_mlflow: bool = False
) -> Tuple[Any, Dict]:
    """Stage 3 — Train forecasting models for all stocks.

    Uses walk-forward cross-validation internally (also CV'ing the
    SARIMAX baseline listed in ``config.model.benchmark_models``) and
    saves fitted models, scalers, and metadata to disk.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        feature_matrices: Dict mapping ticker -> feature DataFrame.
        use_mlflow: If True, log per-ticker/per-model-variant CV results
            to MLflow (see src/tracking.py).

    Returns:
        Tuple of ``(forecaster, training_results)``.
    """
    from src.models import ReturnForecaster

    logger.info("=" * 60)
    logger.info("STAGE 3: MODEL TRAINING")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    forecaster = ReturnForecaster(config)
    training_results = forecaster.train_all_stocks(feature_matrices)
    forecaster.save_models()

    if use_mlflow:
        from src.tracking import log_training_results

        log_training_results(config, training_results)

    n_success = sum(
        1 for v in training_results.values() if "error" not in v
    )
    logger.info(
        "Model training complete — %d/%d tickers succeeded in %.1fs",
        n_success,
        len(training_results),
        time.perf_counter() - t0,
    )
    return forecaster, training_results


def run_backtest(
    config: dict,
    logger: Any,
    forecaster: Any,
    feature_matrices: Dict,
    ohlcv_data: Dict,
) -> Tuple[Dict, str, list]:
    """Stage 4 — Run portfolio backtest on the forward-test period.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        forecaster: Fitted :class:`ReturnForecaster`.
        feature_matrices: Dict mapping ticker -> feature DataFrame.
        ohlcv_data: Dict mapping ticker -> OHLCV DataFrame (for prices).

    Returns:
        Tuple of ``(results_dict, text_report, list_of_figures)``.
    """
    from src.backtest import PortfolioBacktester

    logger.info("=" * 60)
    logger.info(
        "STAGE 4: PORTFOLIO CONSTRUCTION + BACKTEST (method=%s)",
        config.get("portfolio", {}).get("method"),
    )
    logger.info("=" * 60)

    t0 = time.perf_counter()

    logger.info("Generating return predictions...")
    predictions = forecaster.predict_all_stocks(feature_matrices)
    logger.info(
        "Predictions generated for %d tickers", len(predictions)
    )

    logger.info("Running forward-test backtest...")
    backtester = PortfolioBacktester(config)
    results = backtester.forward_test(predictions, ohlcv_data)

    logger.info("Generating performance report...")
    report, figures = backtester.generate_report(results)

    logger.info(
        "Backtest complete in %.1fs", time.perf_counter() - t0
    )
    return results, report, figures


def run_benchmark_stage(
    config: dict,
    logger: Any,
    forecaster: Any,
    feature_matrices: Dict,
    ohlcv_data: Dict,
) -> str:
    """Stage 5 (optional) — Compare the configured method against baselines.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        forecaster: Fitted :class:`ReturnForecaster`.
        feature_matrices: Dict mapping ticker -> feature DataFrame.
        ohlcv_data: Dict mapping ticker -> OHLCV DataFrame.

    Returns:
        Formatted comparison report string.
    """
    from pathlib import Path

    from src.backtest.benchmarks import format_comparison_report, run_benchmark_comparison

    logger.info("=" * 60)
    logger.info("STAGE 5: BENCHMARK COMPARISON")
    logger.info("=" * 60)

    predictions = forecaster.predict_all_stocks(feature_matrices)
    comparison = run_benchmark_comparison(config, predictions, ohlcv_data)
    report_text = format_comparison_report(comparison)

    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "benchmark_comparison.txt"
    out_path.write_text(report_text)
    logger.info("Benchmark comparison written to %s", out_path)

    return report_text


# ---------------------------------------------------------------------------
# Data loading helpers (for partial pipeline runs)
# ---------------------------------------------------------------------------

def _load_cached_data(config: dict, logger: Any) -> Tuple[Dict, Dict, Any, Dict]:
    """Load previously cached data from disk.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.

    Returns:
        Tuple of ``(ohlcv_data, fundamental_data, macro_data, sentiment_data)``.
    """
    from src.data import (
        MarketDataFetcher,
        FundamentalDataFetcher,
        MacroDataFetcher,
        SentimentDataFetcher,
    )

    logger.info("Loading cached data from disk...")

    market_fetcher = MarketDataFetcher(config)
    ohlcv_data = market_fetcher.load_cached()

    fundamental_fetcher = FundamentalDataFetcher(config)
    try:
        fundamental_data = fundamental_fetcher.load_cached()
    except (AttributeError, FileNotFoundError):
        logger.warning("Fundamental data cache not found; re-fetching...")
        fundamental_data = fundamental_fetcher.fetch_all_fundamentals()

    macro_fetcher = MacroDataFetcher(config)
    try:
        macro_data = macro_fetcher.load_cached()
    except (AttributeError, FileNotFoundError):
        logger.warning("Macro data cache not found; re-fetching...")
        macro_data = macro_fetcher.fetch_all_macro()

    sentiment_fetcher = SentimentDataFetcher(config)
    try:
        sentiment_data = sentiment_fetcher.load_cached()
    except (AttributeError, FileNotFoundError):
        logger.warning("Sentiment data cache not found; re-fetching...")
        sentiment_data = sentiment_fetcher.fetch_all_sentiment()

    logger.info("Cached data loaded successfully.")
    return ohlcv_data, fundamental_data, macro_data, sentiment_data


def _load_models(config: dict, logger: Any) -> Any:
    """Load previously trained models from disk.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.

    Returns:
        Fitted :class:`ReturnForecaster`.
    """
    from src.models import ReturnForecaster

    logger.info("Loading trained models from disk...")
    forecaster = ReturnForecaster(config)
    forecaster.load_models()
    logger.info("Models loaded for %d tickers.", len(forecaster.fitted_models))
    return forecaster


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the pipeline based on command-line arguments."""
    args = parse_args()

    if not any([args.full, args.data_only, args.train_only,
                args.backtest_only, args.step]):
        args.full = True

    config = load_config(args.config)
    ensure_directories(config)
    set_random_seed(42)
    logger = setup_logger("main", args.config)

    logger.info("=" * 60)
    logger.info("QUANT RESEARCH PLATFORM — FORECAST -> CONSTRUCT -> BACKTEST")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info(
        "Tickers: %s",
        ", ".join(config.get("stocks", {}).get("tickers", [])),
    )
    logger.info(
        "Date range: %s -> %s",
        config.get("dates", {}).get("start", "N/A"),
        config.get("dates", {}).get("end", "N/A"),
    )
    logger.info(
        "Portfolio construction method: %s",
        config.get("portfolio", {}).get("method", "N/A"),
    )

    pipeline_start = time.perf_counter()

    try:
        start_step = args.step or 1
        end_step = 4

        if args.data_only:
            start_step, end_step = 1, 1
        elif args.train_only:
            start_step, end_step = 2, 3
        elif args.backtest_only:
            start_step, end_step = 4, 4

        ohlcv_data: Optional[Dict] = None
        fundamental_data: Optional[Dict] = None
        macro_data: Any = None
        sentiment_data: Optional[Dict] = None
        feature_matrices: Optional[Dict] = None
        forecaster: Any = None
        training_results: Optional[Dict] = None

        # STEP 1: Data Collection
        if start_step <= 1 <= end_step:
            (
                ohlcv_data,
                fundamental_data,
                macro_data,
                sentiment_data,
            ) = fetch_all_data(config, logger)

        # STEP 2: Feature Engineering
        if start_step <= 2 <= end_step:
            if ohlcv_data is None:
                (
                    ohlcv_data,
                    fundamental_data,
                    macro_data,
                    sentiment_data,
                ) = _load_cached_data(config, logger)

            feature_matrices = build_features(
                config, logger,
                ohlcv_data, fundamental_data, macro_data, sentiment_data,
            )

        # STEP 3: Model Training
        if start_step <= 3 <= end_step:
            if feature_matrices is None:
                logger.error(
                    "Feature matrices not available. "
                    "Run with --full or --train-only first."
                )
                sys.exit(1)

            forecaster, training_results = train_models(
                config, logger, feature_matrices, use_mlflow=args.mlflow
            )

        # STEP 4: Portfolio Construction + Backtest
        if start_step <= 4 <= end_step:
            if forecaster is None:
                forecaster = _load_models(config, logger)

            if ohlcv_data is None:
                (
                    ohlcv_data,
                    fundamental_data,
                    macro_data,
                    sentiment_data,
                ) = _load_cached_data(config, logger)

            if feature_matrices is None:
                feature_matrices = build_features(
                    config, logger,
                    ohlcv_data, fundamental_data, macro_data, sentiment_data,
                )

            results, report, figures = run_backtest(
                config, logger, forecaster, feature_matrices, ohlcv_data
            )

            print("\n" + "=" * 60)
            print("FINAL PORTFOLIO PERFORMANCE REPORT")
            print("=" * 60)
            print(report)

            # STEP 5 (optional): Benchmark vs equal-weight / mean-variance
            if args.benchmark:
                benchmark_report = run_benchmark_stage(
                    config, logger, forecaster, feature_matrices, ohlcv_data
                )
                print(benchmark_report)

        elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully in %.1f seconds!", elapsed)
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
