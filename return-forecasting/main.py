#!/usr/bin/env python
"""
Main Pipeline Orchestrator for Multi-Dimensional Return Forecasting.

Coordinates the full end-to-end pipeline from data collection through
feature engineering, model training, and portfolio backtesting.

Usage:
    python main.py --full          # Run complete pipeline
    python main.py --data-only     # Only fetch data
    python main.py --train-only    # Only train models (requires data)
    python main.py --backtest-only # Only run backtest (requires trained models)
    python main.py --step 3        # Resume from step 3

Steps:
    1. Data collection (market, fundamental, macro, sentiment)
    2. Feature engineering (technical, fundamental, macro, sentiment)
    3. Model training (walk-forward validation + final fit)
    4. Portfolio backtest (forward test period Oct–Dec 2025)
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
        description="Multi-Dimensional Return Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --full\n"
            "  python main.py --data-only\n"
            "  python main.py --train-only\n"
            "  python main.py --backtest-only\n"
            "  python main.py --step 3\n"
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

    # --- Market data (Yahoo Finance OHLCV) ---
    logger.info("Fetching market data...")
    t0 = time.perf_counter()
    market_fetcher = MarketDataFetcher(config)
    ohlcv_data = market_fetcher.fetch_all_stocks()
    logger.info(
        "Market data fetched for %d tickers in %.1fs",
        len(ohlcv_data),
        time.perf_counter() - t0,
    )

    # --- Fundamental data ---
    logger.info("Fetching fundamental data...")
    t0 = time.perf_counter()
    fundamental_fetcher = FundamentalDataFetcher(config)
    fundamental_data = fundamental_fetcher.fetch_all_fundamentals()
    logger.info(
        "Fundamental data fetched for %d tickers in %.1fs",
        len(fundamental_data),
        time.perf_counter() - t0,
    )

    # --- Macro indicators ---
    logger.info("Fetching macro indicators...")
    t0 = time.perf_counter()
    macro_fetcher = MacroDataFetcher(config)
    macro_data = macro_fetcher.fetch_all_macro()
    logger.info(
        "Macro data fetched (%d rows) in %.1fs",
        len(macro_data) if hasattr(macro_data, "__len__") else 0,
        time.perf_counter() - t0,
    )

    # --- Sentiment data (FinBERT) ---
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

    Runs the :class:`FeaturePipeline` which generates technical,
    fundamental, macro, and sentiment features, then merges and
    scales them.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        ohlcv_data: Dict mapping ticker → OHLCV DataFrame.
        fundamental_data: Dict mapping ticker → fundamental DataFrame.
        macro_data: Macro indicator DataFrame.
        sentiment_data: Dict mapping ticker → sentiment DataFrame.

    Returns:
        Dictionary mapping ticker → feature DataFrame (with ``Target``).
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
    config: dict, logger: Any, feature_matrices: Dict
) -> Tuple[Any, Dict]:
    """Stage 3 — Train forecasting models for all stocks.

    Uses walk-forward cross-validation internally and saves fitted
    models, scalers, and metadata to disk.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        feature_matrices: Dict mapping ticker → feature DataFrame.

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

    Generates return predictions for Oct–Dec 2025, constructs a
    portfolio via the configured optimisation strategy, and computes
    performance metrics.

    Args:
        config: Parsed configuration dictionary.
        logger: Logger instance.
        forecaster: Fitted :class:`ReturnForecaster`.
        feature_matrices: Dict mapping ticker → feature DataFrame.
        ohlcv_data: Dict mapping ticker → OHLCV DataFrame (for prices).

    Returns:
        Tuple of ``(results_dict, text_report, list_of_figures)``.
    """
    from src.portfolio import PortfolioBacktester

    logger.info("=" * 60)
    logger.info("STAGE 4: PORTFOLIO BACKTEST")
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # --- Generate predictions ---
    logger.info("Generating return predictions...")
    predictions = forecaster.predict_all_stocks(feature_matrices)
    logger.info(
        "Predictions generated for %d tickers", len(predictions)
    )

    # --- Run backtest ---
    logger.info("Running forward-test backtest...")
    backtester = PortfolioBacktester(config)
    results = backtester.forward_test(predictions, ohlcv_data)

    # --- Generate report ---
    logger.info("Generating performance report...")
    report, figures = backtester.generate_report(results)

    logger.info(
        "Backtest complete in %.1fs", time.perf_counter() - t0
    )
    return results, report, figures


# ---------------------------------------------------------------------------
# Data loading helpers (for partial pipeline runs)
# ---------------------------------------------------------------------------

def _load_cached_data(config: dict, logger: Any) -> Tuple[Dict, Dict, Any, Dict]:
    """Load previously cached data from disk.

    Used when running ``--train-only`` or ``--backtest-only`` without
    re-fetching.

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
        logger.warning(
            "Fundamental data cache not found; "
            "re-fetching fundamental data..."
        )
        fundamental_data = fundamental_fetcher.fetch_all_fundamentals()

    macro_fetcher = MacroDataFetcher(config)
    try:
        macro_data = macro_fetcher.load_cached()
    except (AttributeError, FileNotFoundError):
        logger.warning(
            "Macro data cache not found; re-fetching macro data..."
        )
        macro_data = macro_fetcher.fetch_all_macro()

    sentiment_fetcher = SentimentDataFetcher(config)
    try:
        sentiment_data = sentiment_fetcher.load_cached()
    except (AttributeError, FileNotFoundError):
        logger.warning(
            "Sentiment data cache not found; "
            "re-fetching sentiment data..."
        )
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
    logger.info(
        "Models loaded for %d tickers.", len(forecaster.fitted_models)
    )
    return forecaster


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the pipeline based on command-line arguments."""
    args = parse_args()

    # -- Resolve which steps to run --
    # Default to --full if no flag is provided
    if not any([args.full, args.data_only, args.train_only,
                args.backtest_only, args.step]):
        args.full = True

    # -- Load config and initialise --
    config = load_config(args.config)
    ensure_directories(config)
    set_random_seed(42)
    logger = setup_logger("main", args.config)

    logger.info("=" * 60)
    logger.info("MULTI-DIMENSIONAL RETURN FORECASTING PIPELINE")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info(
        "Tickers: %s",
        ", ".join(config.get("stocks", {}).get("tickers", [])),
    )
    logger.info(
        "Date range: %s → %s",
        config.get("dates", {}).get("start", "N/A"),
        config.get("dates", {}).get("end", "N/A"),
    )

    pipeline_start = time.perf_counter()

    try:
        # Determine step range
        start_step = args.step or 1
        end_step = 4

        if args.data_only:
            start_step, end_step = 1, 1
        elif args.train_only:
            start_step, end_step = 2, 3
        elif args.backtest_only:
            start_step, end_step = 4, 4

        # Shared state across steps
        ohlcv_data: Optional[Dict] = None
        fundamental_data: Optional[Dict] = None
        macro_data: Any = None
        sentiment_data: Optional[Dict] = None
        feature_matrices: Optional[Dict] = None
        forecaster: Any = None
        training_results: Optional[Dict] = None

        # ------------------------------------------------------------------
        # STEP 1: Data Collection
        # ------------------------------------------------------------------
        if start_step <= 1 <= end_step:
            (
                ohlcv_data,
                fundamental_data,
                macro_data,
                sentiment_data,
            ) = fetch_all_data(config, logger)

        # ------------------------------------------------------------------
        # STEP 2: Feature Engineering
        # ------------------------------------------------------------------
        if start_step <= 2 <= end_step:
            # Load cached data if not yet available
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

        # ------------------------------------------------------------------
        # STEP 3: Model Training
        # ------------------------------------------------------------------
        if start_step <= 3 <= end_step:
            if feature_matrices is None:
                logger.error(
                    "Feature matrices not available. "
                    "Run with --full or --train-only first."
                )
                sys.exit(1)

            forecaster, training_results = train_models(
                config, logger, feature_matrices
            )

        # ------------------------------------------------------------------
        # STEP 4: Portfolio Backtest
        # ------------------------------------------------------------------
        if start_step <= 4 <= end_step:
            # Load models if not trained in this run
            if forecaster is None:
                forecaster = _load_models(config, logger)

            # Load data if not available
            if ohlcv_data is None:
                (
                    ohlcv_data,
                    fundamental_data,
                    macro_data,
                    sentiment_data,
                ) = _load_cached_data(config, logger)

            # Build features if not available
            if feature_matrices is None:
                feature_matrices = build_features(
                    config, logger,
                    ohlcv_data, fundamental_data, macro_data, sentiment_data,
                )

            results, report, figures = run_backtest(
                config, logger, forecaster, feature_matrices, ohlcv_data
            )

            # Print summary to console
            print("\n" + "=" * 60)
            print("FINAL PORTFOLIO PERFORMANCE REPORT")
            print("=" * 60)
            print(report)

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 60)
        logger.info(
            "Pipeline completed successfully in %.1f seconds!", elapsed
        )
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
