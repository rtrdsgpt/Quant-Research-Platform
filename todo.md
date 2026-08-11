# Quant Research Platform — TODO

Merge of `return-forecasting/` (was "Return Forecasting and Portfolio Management") and
`portfolio-replication/` (was "Portfolio Replication") into one flagship: forecast → construct →
backtest. Both subfolders were copied here with `.git` stripped — this becomes its own repo.

See `Project Plan.md` (Projects root) section 2 for full rationale.

## 0. Repo setup
- [ ] `git init` in this folder, first commit with both subfolders as-is (baseline before merge)
- [ ] Decide final top-level layout: merge into one `src/` package, or keep `return-forecasting/`
      + `portfolio-replication/` as two internal modules under a shared `src/`
- [ ] Check `portfolio-replication/docs/Group_Details.txt` for the original contribution split
      before claiming full authorship of the optimization-layer code
- [ ] Re-run and fix the README's Sharpe/Max-Drawdown/Hit-Ratio numbers — currently identical
      values (2.3049) across all three metrics, looks like a copy-paste bug, must be corrected
      before this goes on a CV or a live demo

## 1. Core merge: forecast → construct → backtest
- [ ] Replace `portfolio-replication`'s "replicate the S&P 500" objective with "construct a
      portfolio from `return-forecasting`'s ML-forecasted alpha"
- [ ] Wire `return-forecasting`'s ensemble (LightGBM/XGBoost/Ridge) predicted returns as the
      input signal to `portfolio-replication`'s weighting layer (cvxpy tracking-error min / HRP)
- [ ] Benchmark the forecast-driven portfolio against equal-weight and mean-variance baselines
- [ ] Add a **SARIMAX/statsmodels baseline** into the existing walk-forward CV harness, benchmarked
      against the LightGBM/XGBoost/Ridge ensemble (closes the ARIMA/SARIMA gap — no project in the
      portfolio currently has this)

## 2. API layer
- [ ] FastAPI service wrapping the pipeline:
  - [ ] `POST /forecast/{ticker}`
  - [ ] `POST /portfolio/construct`
  - [ ] `GET /backtest/{run_id}`
- [ ] Reuse the existing staged CLI flags (`--data-only`, `--train-only`, `--backtest-only`,
      `--step N`) as the service's internal job stages

## 3. MLOps
- [ ] **Docker**: containerize; existing `run.sh` becomes the entrypoint
- [ ] **CI**: GitHub Actions workflow running the existing `pytest`/`pytest-cov` suite on push
      (currently no `.github/workflows` on either source project)
- [ ] **MLflow**: log each walk-forward fold + model variant (LightGBM/XGBoost/Ridge/SARIMAX) as
      an MLflow run; existing `.joblib` artifacts per stock/model map directly onto MLflow's
      artifact store
- [ ] **DVC**: version the OHLCV/fundamentals/sentiment datasets and engineered feature matrices
- [ ] **Airflow**: model the pipeline as a DAG — collect → engineer features → train/CV → backtest
      (the staged structure already exists in code, just not orchestrated)

## 4. Testing / evaluation
- [ ] Extend the existing `tests/` (`test_data.py`, `test_features.py`, `test_models.py`,
      `test_portfolio.py`) to cover the new merge logic and SARIMAX baseline
- [ ] Confirm walk-forward CV (expanding window, 5-day purge gap) still holds after merge
- [ ] Document backtest metrics honestly (Sharpe, Calmar, max drawdown, hit ratio) once the
      copy-paste bug above is fixed

## Explicitly out of scope
- RAG / Agentic / MCP — this project doesn't need them, that's covered by other projects in the
  portfolio (Patent Prior-Art Agent, F.A.I.L_OOPS v2, DatoScope V2)
