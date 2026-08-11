# Multi-Dimensional Return Forecasting and Portfolio Management

A production-grade ML pipeline for predicting stock returns using multi-dimensional features (technical, fundamental, macro, sentiment) and constructing optimised portfolios with walk-forward validation.

---

## Stock Universe

| Ticker         | Company              | Sector       |
|----------------|----------------------|--------------|
| RELIANCE.NS    | Reliance Industries  | Energy       |
| HDFCBANK.NS    | HDFC Bank            | Banking      |
| INFY.NS        | Infosys              | IT Services  |
| TATAMOTORS.NS  | Tata Motors          | Automotive   |
| BHARTIARTL.NS  | Bharti Airtel        | Telecom      |
| HINDUNILVR.NS  | Hindustan Unilever   | FMCG         |

**Data period:** January 2020 – December 2025  
**Training pool:** January 2020 – September 2025  
**Forward test:** October 2025 – December 2025

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Assignment02

# Create virtual environment (recommended)
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate

# Install dependencies (first run can take several minutes; torch is large)
pip install -r requirements.txt
```

**One-liner to run the app** (creates venv and installs deps if needed, then runs `main.py`):

```bash
./run.sh [args]   # e.g. ./run.sh --full  or  ./run.sh --data-only
```

python main.py 
### Dependencies

Key libraries used:

| Category       | Libraries                                    |
|----------------|----------------------------------------------|
| Data           | `yfinance`, `requests`, `beautifulsoup4`, `gnews` |
| Processing     | `pandas`, `numpy`, `scipy`                   |
| ML             | `scikit-learn`, `lightgbm`, `xgboost`        |
| NLP            | `transformers`, `torch` (FinBERT)            |
| Visualisation  | `matplotlib`, `seaborn`, `plotly`            |
| Config         | `pyyaml`                                     |
| Testing        | `pytest`, `pytest-cov`                       |

### GNews.io API key// Finnhub API Key (optional)

For **real news headlines** in the sentiment pipeline (instead of the scraper or synthetic data), use the [GNews.io](https://gnews.io) API or Finnhub API:

1. Sign up at [gnews.io](https://gnews.io/register) and copy your API key.
2. Either:
   - **Environment variable** (recommended, so you don’t commit the key):
     ```bash
     export GNEWS_API_KEY="api key here"
     export FINNHUB_API_KEY='api key here' 
     ```
   - **Config file**: in `config/config.yaml`, under `sentiment`, set:
     ```yaml
     sentiment:
       gnews_api_key: "api key here"
       finnhub_api_key: "api key here"
     ```

If the key is set, the pipeline uses the Finnhub API first then GNews.io API; if it’s missing or the request fails, it falls back to the free GNews scraper and then to synthetic headlines.

---

## Quick Start

### Run the complete pipeline

```bash
python main.py --full
```

This will execute all four stages:
1. **Data Collection** — download OHLCV, fundamentals, macro, sentiment
2. **Feature Engineering** — build ~54 features with proper lagging
3. **Model Training** — walk-forward CV + ensemble (LightGBM, XGBoost, Ridge)
4. **Portfolio Backtest** — forward test Oct–Dec 2025

### Run individual stages

```bash
# Fetch data only
python main.py --data-only

# Train models only (requires cached data)
python main.py --train-only

# Run backtest only (requires trained models)
python main.py --backtest-only

# Resume from a specific step (1-4)
python main.py --step 3
```

### Use a custom config file

```bash
python main.py --full --config config/my_config.yaml
```

---

## Project Structure

```
Assignment02/
├── config/
│   └── config.yaml                 # Master configuration file
├── data/
│   ├── raw/                        # Raw downloaded data
│   │   ├── market/                 # OHLCV parquet files per ticker
│   │   ├── fundamental/            # Quarterly fundamental CSVs
│   │   ├── macro/                  # Macro indicator CSVs
│   │   └── sentiment/              # Sentiment score CSVs
│   ├── processed/                  # Cleaned, aligned intermediate data
│   └── features/                   # Final feature matrices (parquet)
├── src/
│   ├── data/                       # Data collection modules
│   │   ├── market_data.py          # Yahoo Finance OHLCV fetcher
│   │   ├── fundamental_data.py     # Fundamental data provider
│   │   ├── macro_data.py           # Macro indicators fetcher
│   │   └── sentiment_data.py       # Google News + FinBERT pipeline
│   ├── features/                   # Feature engineering modules
│   │   ├── technical_features.py   # RSI, MACD, Bollinger, ATR, etc.
│   │   ├── fundamental_features.py # P/E, D/E, ROE, EPS features
│   │   ├── macro_features.py       # Macro indicator features
│   │   ├── sentiment_features.py   # Sentiment score features
│   │   └── feature_pipeline.py     # Orchestrates all feature builders
│   ├── models/                     # Machine learning modules
│   │   ├── walk_forward.py         # Walk-forward validation engine
│   │   ├── feature_selection.py    # RFE and importance analysis
│   │   └── forecaster.py           # Return prediction (LightGBM/XGBoost/Ridge)
│   ├── portfolio/                  # Portfolio management modules
│   │   ├── optimizer.py            # Weight optimization (inv-vol, MVO, equal)
│   │   ├── metrics.py              # Sharpe, MDD, hit ratio, equity curve
│   │   └── backtester.py           # Portfolio backtesting engine
│   └── utils/                      # Utilities
│       ├── logger.py               # Rotating file + console logger
│       └── helpers.py              # Config loader, I/O, timing, seeding
├── tests/                          # Automated test suite
│   ├── test_data.py                # Data fetcher tests
│   ├── test_features.py            # Feature engineering tests
│   ├── test_models.py              # Model and validation tests
│   └── test_portfolio.py           # Portfolio optimizer tests
├── reports/                        # Output reports and figures
│   └── figures/                    # Generated charts (equity curve, etc.)
├── models/                         # Serialised model artefacts (.joblib)
├── logs/                           # Pipeline logs (rotating)
├── main.py                         # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
├── ARCHITECTURE.md                 # Detailed architecture document
└── README.md                       # This file
```

---

## Configuration

All parameters are centralised in [`config/config.yaml`](config/config.yaml):

### Key Sections

| Section     | Description                                            |
|-------------|--------------------------------------------------------|
| `stocks`    | Ticker symbols and display names for the 6-stock universe |
| `dates`     | Data window, training cutoff, and forward-test period  |
| `paths`     | Directory paths for data, models, reports, and logs    |
| `features`  | Technical indicator windows, return lags, volatility windows |
| `model`     | Model type, ensemble settings, hyperparameters         |
| `portfolio` | Optimization method, rebalance frequency, constraints  |
| `sentiment` | FinBERT model name, batch size                         |
| `macro`     | Yahoo Finance symbols for macro indicators             |
| `logging`   | Log level, format, rotation settings                   |

### Example: Change the portfolio method

```yaml
portfolio:
  method: "mean_variance"  # Options: equal_weight, inverse_volatility, mean_variance
  rebalance_frequency: "weekly"
  max_weight: 0.35
```

---

## Usage Examples

### Python API

```python
from src.utils.helpers import load_config
from src.data import MarketDataFetcher
from src.features import FeaturePipeline
from src.models import ReturnForecaster
from src.portfolio import PortfolioBacktester

# Load configuration
config = load_config("config/config.yaml")

# Fetch market data
fetcher = MarketDataFetcher(config)
ohlcv_data = fetcher.fetch_all_stocks()

# Build features
pipeline = FeaturePipeline(config)
features = pipeline.build_all_feature_matrices(
    ohlcv_data, fundamental_data, macro_data, sentiment_data
)

# Train models
forecaster = ReturnForecaster(config)
results = forecaster.train_all_stocks(features)

# Generate predictions
predictions = forecaster.predict_all_stocks(features)

# Run backtest
backtester = PortfolioBacktester(config)
bt_results = backtester.forward_test(predictions, ohlcv_data)
report, figures = backtester.generate_report(bt_results)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_data.py -v
```

---

## Pipeline Architecture

```
Data Collection     Feature Engineering     Model Training        Portfolio Backtest
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────────────┐
│ Yahoo OHLCV │───>│ Technical (RSI,  │    │ Feature Select │    │ Forward-test    │
│ Fundamentals│───>│ MACD, Bollinger) │───>│ (RFE)          │───>│ Predictions     │
│ Macro Data  │───>│ Fund/Macro/Sent  │    │ Walk-Forward CV│    │ Portfolio Optim. │
│ Sentiment   │───>│ + RobustScaler   │    │ Ensemble Train │    │ Metrics & Report│
└─────────────┘    └──────────────────┘    └────────────────┘    └─────────────────┘
```

### Key Design Decisions

- **Walk-forward validation** (no K-Fold) — prevents temporal data leakage
- **Expanding window** — captures evolving market regimes
- **5-day purge gap** — eliminates feature-lag contamination
- **RobustScaler** — handles financial data outliers
- **Inverse-MSE ensemble weights** — better models contribute more
- **Cross-symbol feature filter** — ensures robustness across all 6 stocks

---

## Results Summary

> **Note:** Fill in after running `python main.py --full`

| Metric                 | Value   |
|------------------------|---------|
| Annualised Return      |28.0916% |
| Annualised Volatility  |8.0649%  |
| Sharpe Ratio           |2.3049   |
| Sortino Ratio          |5.3842   |
| Maximum Drawdown       |2.6950%  |
| Hit Ratio              |54.0984% |
| Calmar Ratio           |10.4238  |

See [`reports/report_template.md`](reports/report_template.md) for the full report template.

---



## References

- **LightGBM**: Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
- **XGBoost**: Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.*
- **FinBERT**: Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*
- **Walk-Forward Validation**: Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies.*
# Return-Forecasting-and-Portfolio-Management
