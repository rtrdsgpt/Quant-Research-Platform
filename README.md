# Quant Research Platform

[![CI](https://github.com/rtrdsgpt/Quant-Research-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rtrdsgpt/Quant-Research-Platform/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Forecast → construct → backtest: a walk-forward ML return-forecasting
ensemble (LightGBM/XGBoost/Ridge, benchmarked against a SARIMAX baseline)
feeding a cvxpy/HRP portfolio-construction and backtesting layer, served
over a FastAPI API and tracked with MLflow/DVC.

Merge of two prior academic group projects — **Return Forecasting and
Portfolio Management** and **Portfolio Replication** (see
[`docs/original-coursework/`](docs/original-coursework/) for the originals and
group rosters) — into one pipeline. This repo carries no shared git
history with either; the merge, API layer, and MLOps additions are
independent work, done after both courses concluded.

## Table of Contents

- [Results](#results)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Scope](#scope)
- [License](#license)

## Results

Forward-test, Oct–Dec 2025 (61 trading days, 3 monthly rebalances) — real
output of `src/backtest/benchmarks.py`, checked in at
[`reports/benchmark_comparison.txt`](reports/benchmark_comparison.txt):

| Method | Total Return | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Hit Ratio |
|---|---|---|---|---|---|---|---|---|
| `equal_weight` | 6.20% | 31.34% | 8.27% | 2.554 | 5.907 | 2.31% | 13.57 | 49.18% |
| `mean_variance` | 9.71% | 50.49% | 10.39% | 3.363 | 7.760 | 1.97% | 25.60 | 54.10% |
| `alpha_hrp` (current default) | 9.82% | 49.41% | 11.19% | 3.064 | 5.853 | 2.88% | 17.17 | **65.57%** |
| `alpha_markowitz` | 9.75% | 48.84% | 9.75% | **3.463** | 7.789 | 2.28% | 21.38 | 59.02% |

**Read honestly, not spun**: the forecast-driven methods (`alpha_hrp`,
`alpha_markowitz`) don't cleanly dominate the naive baselines here.
`alpha_hrp` — the current default — has the best hit ratio of the four
but a *worse* Sharpe than plain `mean_variance`; `alpha_markowitz` has
the best Sharpe but a middling hit ratio. Three rebalances over 61 days
is a small sample — this is one forward-test window on 6 stocks, not a
statistically powered claim that either forecast-driven method beats
equal-weight/mean-variance in general. Reproduce or extend it:

```bash
./run.sh --backtest-only --benchmark   # writes reports/benchmark_comparison.txt
```

A separate, older forward-test under the original `inverse_volatility`
construction method (no longer the default) is also checked in at
[`reports/performance_report.txt`](reports/performance_report.txt): Sharpe 2.3049,
Sortino 5.3842, Max Drawdown 2.6950%, Hit Ratio 54.0984%, Calmar 10.4238.
(The source README this came from had a copy-paste bug — Max Drawdown and
Hit Ratio were both shown as `2.3049`, the Sharpe ratio's value. Fixed
here; see [`DECISIONS.md`](DECISIONS.md).)

## Architecture

```
Data Collection      Feature Engineering    Model Training (walk-forward CV)     Portfolio Construction    Backtest
┌──────────────┐    ┌──────────────────┐   ┌───────────────────────────────┐   ┌────────────────────┐   ┌─────────────┐
│ Yahoo OHLCV  │───>│ Technical (RSI,  │   │ LightGBM ┐                     │   │ alpha_hrp /         │  │ Forward-test│
│ Fundamentals │───>│ MACD, Bollinger) │──>│ XGBoost  ├─> inverse-MSE       │──>│ alpha_markowitz     │─>│ vs. equal-  │
│ Macro data   │───>│ Fund/Macro/Sent  │   │ Ridge    ┘   ensemble          │   │ (cvxpy / HRP,        │  │ weight & MV │
│ Sentiment    │───>│ + RobustScaler   │   │ SARIMAX (CV'd for comparison,  │   │  src/construction/)  │  │ baselines   │
└──────────────┘    └──────────────────┘   │ not blended into the ensemble)│   └────────────────────┘   └─────────────┘
                                            └───────────────────────────────┘
```

- **Forecast**: [`src/models/forecaster.py`](src/models/forecaster.py) trains a per-ticker inverse-MSE-weighted
  ensemble (LightGBM/XGBoost/Ridge) with 5-day-purge expanding-window
  walk-forward CV ([`src/models/walk_forward.py`](src/models/walk_forward.py)), plus a SARIMAX
  baseline ([`src/models/sarimax_model.py`](src/models/sarimax_model.py)) CV'd on the same folds for direct
  comparison (not blended into the ensemble). Trained on 134 features
  per ticker (135th column is `Target`, the label — next-day log
  return), built by [`src/features/feature_pipeline.py`](src/features/feature_pipeline.py):

  | Category | Count | Examples |
  |---|---|---|
  | Technical | 50 | Log returns (+6 lags), SMA/EMA (5/10/20/50) + price ratios, RSI-14, MACD/signal/histogram, ROC/momentum, realized vol (5/10/21/63d), ATR-14, Bollinger bands, Parkinson vol, volume (OBV, VWAP proxy, volume ratio), candlestick shape |
  | Fundamental | 14 | P/E, D/E, ROE, EPS, book value, market cap, dividend yield, + QoQ changes in each, earnings yield, P/E-relative-to-mean, a composite quality score |
  | Macro | 55 | 7 raw indicators (USD/INR, crude, gold, Nifty50, India VIX, 10Y yield, inflation) × {return, 5d/21d change, 20d MA, deviation-from-MA, 21d vol} = 35, + 6 regime flags (risk-on/off, oil regime, currency stress, rate regime, VIX level/change) |
  | Sentiment | 15 | FinBERT score + pos/neg/neutral probabilities, 5d/10d sentiment MA, sentiment trend/z-score/5d change/10d vol, high/low-sentiment flags, headline-count MA, high-activity flag |

  RFE then cuts this down to the top 30 per ticker before training (see
  [`src/models/feature_selection.py`](src/models/feature_selection.py)).
- **Construct**: [`src/construction/alpha_portfolio.py`](src/construction/alpha_portfolio.py) turns the ensemble's
  predicted returns into portfolio weights via a mean-variance
  alpha-maximizing objective or Hierarchical Risk Parity
  ([`src/construction/weighting.py`](src/construction/weighting.py)) — replacing the original
  portfolio-replication project's "track the S&P 500" objective with
  alpha maximization on the 6-stock forecasting universe. **Nothing is
  being replicated here or in the backtest step below** — there's no
  benchmark index anywhere in this path, forecast-driven or `--benchmark`
  baseline alike. The only place index replication still happens is the
  separate, disconnected legacy mode — see [Scope](#scope).
- **Backtest**: [`src/backtest/backtester.py`](src/backtest/backtester.py) simulates day-by-day rebalancing with
  transaction costs and reports Sharpe/Sortino/Calmar/Max-Drawdown/Hit-Ratio
  ([`src/backtest/metrics.py`](src/backtest/metrics.py)); [`src/backtest/benchmarks.py`](src/backtest/benchmarks.py) reruns it
  across construction methods for the comparison in [Results](#results).
- **API**: [`src/api/main.py`](src/api/main.py) (FastAPI) — `POST /forecast/{ticker}`, `POST /portfolio/construct`,
  `POST /backtest` + `GET /backtest/{run_id}` — reads the pipeline's own
  cached staged artifacts rather than fetching/training inline.
- **MLOps**: `Dockerfile` + `run.sh` entrypoint · GitHub Actions CI
  (`.github/workflows/ci.yml`) · MLflow fold-level logging
  (`src/tracking.py`, `--mlflow`) · DVC-tracked `data/{raw,features,processed}`
  · a 3-task Airflow DAG (`airflow/dags/`) mirroring the CLI's own
  staging. Full rationale (including a segfault and a CI disk-space
  failure hit and fixed along the way) in [`DECISIONS.md`](DECISIONS.md).

## Setup

```bash
git clone https://github.com/rtrdsgpt/Quant-Research-Platform.git
cd Quant-Research-Platform

./run.sh --full          # creates a venv, installs deps, runs the full pipeline
```

`requirements.txt` is the lean core (what `run.sh`, CI, and the test
suite need). Two more files layer on top — neither is required for the
pipeline to run correctly, only for its designed synthetic/PCA fallbacks
to be replaced with the real thing:

```bash
pip install -r requirements.txt                                # core (default)
pip install -r requirements.txt -r requirements-optional.txt   # + real FinBERT sentiment, real autoencoder selection
pip install -r requirements.txt -r requirements-dev.txt        # + jupyter, dvc
```

Or with Docker (installs core + optional, CPU-only torch build):

```bash
docker build -t quant-research-platform .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  quant-research-platform --full
```

For real news sentiment (instead of synthetic headlines), set
`GNEWS_API_KEY` / `FINNHUB_API_KEY` as environment variables, or under
`sentiment:` in [`config/config.yaml`](config/config.yaml).

## Usage

```bash
./run.sh --data-only
./run.sh --train-only        # rebuilds features from cached raw data, retrains -- skips re-fetching
./run.sh --train-only --mlflow   # + log every walk-forward fold/model variant to MLflow
./run.sh --backtest-only     # requires --train-only (or --full) to have completed first
./run.sh --backtest-only --benchmark   # + compare vs. equal-weight/mean-variance baselines
./run.sh --step 3            # resume from a specific step (1-4)

./run.sh api                  # start the FastAPI service on :8000
```

```bash
curl -X POST localhost:8000/forecast/RELIANCE.NS
curl -X POST localhost:8000/portfolio/construct -H 'Content-Type: application/json' -d '{}'
```

Config (stock universe, date ranges, model hyperparameters including the
SARIMAX baseline's `order`/`seasonal_order`, `portfolio.method`, MLflow
tracking URI) all lives in [`config/config.yaml`](config/config.yaml).

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

88 tests, offline/mocked where the real thing would need network or a
multi-hour training run: unit coverage of every data/feature/model/
construction/backtest module, the forecast → construct wiring
(`tests/test_construction.py`), the SARIMAX baseline (walk-forward CV
integration, sklearn-compatibility), MLflow logging
(`tests/test_tracking.py`), and the API (`tests/test_api.py` — the
data-dependent tests skip cleanly on a fresh clone with no cached
pipeline artifacts, rather than failing).

CI runs the same suite on every push/PR — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Scope

Deliberately bounded, on purpose:

- **No RAG / agentic / MCP** — this platform's job is forecast → construct
  → backtest, not retrieval or tool-calling. That's covered by sibling
  projects in the portfolio (`Financial Anomaly Detection Using RAG`,
  `Patent Prior-Art Agent`); pulling either capability in here would be
  scope creep, not a feature.
- **The original portfolio-replication project's large-universe
  "replicate an index" mode is kept, but as a separate, optional
  entrypoint** (`scripts/legacy_replication.py`), not merged into the
  default forecast → construct path. It needs a completely different,
  disconnected ~409-ticker S&P 500 universe (with user-supplied price
  CSVs never included in this repo) and has no relationship to the
  6-stock sentiment/forecasting pipeline — kept alive because it was
  working, tested functionality, not because it belongs in the main
  story.
- **FinBERT sentiment and the legacy mode's autoencoder selection both
  have a designed fallback** (synthetic headlines + rule-based scoring;
  PCA) when `transformers`/`torch`/`tensorflow` aren't installed — see
  `requirements-optional.txt`. CI and the test suite deliberately run
  against the fallback path, not the real models.
- **DVC's configured remote is a placeholder** (`s3://your-bucket/...`
  in `.dvc/config`) — point it at real credentials before `dvc push`/`pull`
  do anything in production.
- **The Airflow DAG was written and syntax-checked, not run against a
  live scheduler** in this repo's dev environment — standing one up
  (metadata DB, webserver, scheduler) was judged out of scope for
  verifying a 3-task `BashOperator` DAG whose correctness mostly rides on
  `main.py`'s CLI flags, which are tested.
- **No fabricated backtest numbers, ever** — see [Results](#results).

## License

[MIT](LICENSE)
