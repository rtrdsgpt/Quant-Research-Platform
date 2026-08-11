# Quant Research Platform

Forecast → construct → backtest: a walk-forward ML return-forecasting
ensemble (LightGBM/XGBoost/Ridge, benchmarked against a SARIMAX baseline)
feeding a cvxpy/HRP portfolio-construction and backtesting layer, served
over a FastAPI API and tracked with MLflow/DVC.

This is a merge of two prior academic group projects — **Return
Forecasting and Portfolio Management** and **Portfolio Replication**
(see [`docs/original-coursework/`](docs/original-coursework/) for the
originals and group rosters) — into one flagship pipeline, extended
independently afterward with the API/MLOps layer described below. See
[`DECISIONS.md`](DECISIONS.md) for the full log of merge decisions,
problems found, and how they were resolved.

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
  comparison (not blended into the ensemble).
- **Construct**: [`src/construction/alpha_portfolio.py`](src/construction/alpha_portfolio.py) turns the ensemble's
  predicted returns into portfolio weights via a mean-variance
  alpha-maximizing objective or Hierarchical Risk Parity
  ([`src/construction/weighting.py`](src/construction/weighting.py)), replacing the original
  portfolio-replication project's "track the S&P 500" objective with
  alpha maximization. [`src/backtest/benchmarks.py`](src/backtest/benchmarks.py) compares it against
  equal-weight and mean-variance baselines.
- **Backtest**: [`src/backtest/backtester.py`](src/backtest/backtester.py) simulates day-by-day rebalancing with
  transaction costs and reports Sharpe/Sortino/Calmar/Max-Drawdown/Hit-Ratio
  ([`src/backtest/metrics.py`](src/backtest/metrics.py)).
- A large-universe "replicate an index" mode from the original
  portfolio-replication project (LASSO/autoencoder sparse selection vs. a
  benchmark, rolling-CV sparsity search) is preserved as an optional,
  separate entrypoint: [`scripts/legacy_replication.py`](scripts/legacy_replication.py).

## Quick start

```bash
git clone <this-repo-url>
cd Quant-Research-Platform

./run.sh --full          # creates a venv, installs deps, runs the full pipeline
# or, staged:
./run.sh --data-only
./run.sh --train-only        # add --mlflow to log walk-forward CV to MLflow
./run.sh --backtest-only     # add --benchmark to compare vs. baselines

./run.sh api              # start the FastAPI service on :8000
```

Or with Docker (see [Docker](#docker) below):

```bash
docker build -t quant-research-platform .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  quant-research-platform --full
```

### Configuration

Everything is in [`config/config.yaml`](config/config.yaml): stock universe, date ranges, model
hyperparameters (including the SARIMAX baseline's `order`/`seasonal_order`),
portfolio construction method (`portfolio.method`, e.g. `alpha_hrp`,
`alpha_markowitz`, `equal_weight`, `mean_variance`, ...), MLflow tracking
URI, and the `construction:` section governing the forecast → weighting
integration.

For real news sentiment (instead of synthetic headlines), set
`GNEWS_API_KEY` / `FINNHUB_API_KEY` as environment variables, or under
`sentiment:` in the config.

## API

[`src/api/main.py`](src/api/main.py) — FastAPI service reusing the pipeline's cached, staged
artifacts (feature matrices, trained models, OHLCV cache) as its
"internal job stages", per the pipeline's own `--data-only`/`--train-only`/
`--backtest-only` staging. Requires the pipeline to have been run at
least once so those artifacts exist.

```bash
uvicorn src.api.main:app --reload
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/forecast/{ticker}` | POST | Latest predicted return for one ticker |
| `/portfolio/construct` | POST | Build weights from an alpha signal (defaults to the latest cached predictions) |
| `/backtest` | POST | Start an async forward-test backtest job, returns a `run_id` |
| `/backtest/{run_id}` | GET | Poll a backtest job's status/result |

```bash
curl -X POST localhost:8000/forecast/RELIANCE.NS
curl -X POST localhost:8000/portfolio/construct -H 'Content-Type: application/json' -d '{}'
```

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

88 tests: unit coverage of every data/feature/model/construction/backtest
module, the forecast → construct wiring (`tests/test_construction.py`),
the SARIMAX baseline (walk-forward CV integration, sklearn-compatibility),
MLflow logging (`tests/test_tracking.py`), and the API
(`tests/test_api.py` — data-dependent tests skip cleanly if no cached
pipeline artifacts exist yet, rather than failing on a fresh clone).

## MLOps

- **Docker**: [`Dockerfile`](Dockerfile) + [`run.sh`](run.sh) as the entrypoint (`./run.sh <main.py args>`
  locally, `./run.sh api` for the service). CPU-only torch build (the
  default PyPI wheel pulls in several GB of CUDA libraries that are dead
  weight here — FinBERT sentiment scoring never touches a GPU in this
  pipeline).
- **CI**: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the full `pytest`/`pytest-cov` suite on
  every push/PR to `main`.
- **MLflow**: [`src/tracking.py`](src/tracking.py) logs every walk-forward CV fold, for every
  model variant (LightGBM/XGBoost/Ridge/SARIMAX), as a nested MLflow run,
  plus the final `.joblib` artifacts. Opt in with `--mlflow` on
  `--train-only`/`--full`. Tracking store: `sqlite:///mlflow.db` (MLflow
  3.x deprecated the plain `./mlruns` file store).
- **DVC**: `data/raw`, `data/features`, `data/processed` are DVC-tracked
  (`data/*.dvc`) rather than git-committed. The configured remote
  (`.dvc/config`) is a placeholder S3 URL — point it at real bucket
  credentials before `dvc push`/`pull` in production.
- **Airflow**: [`airflow/dags/quant_research_platform_pipeline.py`](airflow/dags/quant_research_platform_pipeline.py) — a 3-task
  DAG (`collect_data` → `engineer_features_and_train` → `backtest`)
  shelling out to the same staged CLI. Written and syntax-checked, not
  run against a live Airflow scheduler in this repo's dev environment.

## Results

A historical example forward-test run (Oct–Dec 2025, under the original
`inverse_volatility` construction method, before the merge changed the
default to `alpha_hrp`) is checked in at
[`reports/performance_report.txt`](reports/performance_report.txt):

| Metric | Value |
|---|---|
| Annualised Return | 28.0916% |
| Annualised Volatility | 8.0649% |
| Sharpe Ratio | 2.3049 |
| Sortino Ratio | 5.3842 |
| Maximum Drawdown | 2.6950% |
| Hit Ratio | 54.0984% |
| Calmar Ratio | 10.4238 |

(The original README this came from had a copy-paste bug — Max Drawdown
and Hit Ratio were both shown as `2.3049`, the Sharpe ratio's value. Fixed
here; see `DECISIONS.md`.)

**This table has not been regenerated under the new default
(`alpha_hrp`) construction method in this session** — run
`./run.sh --full --benchmark` (or `--backtest-only --benchmark` if models
are already trained) to produce a current comparison across
`equal_weight` / `mean_variance` / `alpha_hrp` / `alpha_markowitz`,
written to `reports/benchmark_comparison.txt`.

## Project structure

```
config/           config.yaml, sector_map.csv
src/
  data/           market/fundamental/macro/sentiment fetchers + replication_loader.py
  features/       technical/fundamental/macro/sentiment feature engineers
  models/         forecaster.py, walk_forward.py, feature_selection.py, sarimax_model.py
  construction/   optimizer.py, weighting.py, selection.py, sectors.py, alpha_portfolio.py
  backtest/       backtester.py, metrics.py, benchmarks.py, replication_*.py
  api/            FastAPI service
  utils/          logger.py, helpers.py
  tracking.py     MLflow logging
scripts/          legacy_replication.py (optional large-universe mode)
airflow/dags/     Airflow DAG
tests/            pytest suite (88 tests)
docs/original-coursework/   the two source projects' original READMEs, reports, group rosters
main.py           CLI entrypoint (--full / --data-only / --train-only / --backtest-only / --step N)
DECISIONS.md      running log: decisions, problems found, how they were resolved
todo.md           the original merge plan this repo was built from
```

## Credit

Originally two group coursework projects (Group 34) — see
[`docs/original-coursework/`](docs/original-coursework/) for the original READMEs and
[`docs/original-coursework/*/Group_Details.*`](docs/original-coursework/) for the group rosters filed
with each course. The merge, API layer, and MLOps additions in this
repository (everything from the first commit onward — see `git log`)
are Aritra Dasgupta's own independent work, built after the courses
concluded.
