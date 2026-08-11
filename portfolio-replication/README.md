# Portfolio Replication

An S&P 500 index-replication pipeline that selects a sparse subset of
constituent stocks and weights them to minimize tracking error against the
index, comparing two selection methods and two weighting schemes across
train / validation / holdout splits.

Built for **Assignment 4 (Group 34)** — see [`docs/`](docs/) for the group
roster and full report.

## Overview

Given daily prices for the S&P 500 and its constituents (2020–2025), the
pipeline:

1. **Loads** daily close prices and converts them to returns, split into
   train (2020–H1 2024), validation (H2 2024–H1 2025), and holdout
   (H2 2025) windows, plus three rolling train/validation folds for
   cross-validated model selection.
2. **Selects** a sparse subset of stocks via two methods:
   - **LASSO regression** — coefficient-magnitude ranking against the
     index's daily return.
   - **Autoencoder** — a dense autoencoder trained on standardized returns,
     ranked either by input-to-latent path importance for the
     benchmark-correlated latent dimension (`latent` mode) or by
     reconstruction error (`communality` mode). Falls back to PCA if
     TensorFlow is unavailable.
3. **Weights** the selected stocks via two schemes, each capped at 8% per
   stock and bounded per GICS-style sector relative to the benchmark
   universe's sector composition:
   - **Markowitz** — minimizes tracking-error variance (`Cov(portfolio -
     benchmark)`) via convex optimization (falls back to SLSQP if `cvxpy`
     is unavailable).
   - **HRP (Hierarchical Risk Parity)** — recursive bisection over a
     correlation-distance dendrogram, allocating inversely to cluster
     variance.
4. **Chooses sparsity (k)** per of the 6 method combinations independently,
   via rolling cross-validated tracking error, preferring the sparsest k
   within `0.30` percentage points of the best observed CV tracking error.
5. **Evaluates** all 6 models (2 selection variants × 3 modes → 6 combined
   with 2 weighting schemes) on train/validation/holdout annualized
   tracking error, information ratio, correlation, and R².
6. **Plots** cumulative returns, sparsity-vs-tracking-error sweeps, sector
   drift, information ratios, and a final metrics summary table.

## Project structure

```
.
├── config.yaml                        # all tunable parameters (dates, k ranges, weight constraints, AE params)
├── main.py                            # pipeline entry point
├── pyproject.toml                     # package metadata / dependencies
├── requirements.txt                   # dependencies for a plain `pip install`
├── data/
│   ├── sector_map.csv                 # ticker -> GICS-style sector mapping (409 tickers, 11 sectors)
│   └── raw/                           # daily price CSVs, not committed (see Setup)
├── results/                           # generated plots (created on run)
├── src/
│   └── portfolio_replication/
│       ├── config.py                  # loads config.yaml into typed constants
│       ├── sectors.py                 # sector map loading + benchmark sector-weight computation
│       ├── data_loader.py             # price loading, returns, and train/valid/holdout/rolling splits
│       ├── selection.py               # LASSO and autoencoder stock selection
│       ├── weighting.py               # Markowitz and HRP weighting + constraint enforcement
│       ├── evaluation.py              # metrics, rolling-CV evaluation, sparsity sweeps, sector drift
│       ├── visualization.py           # all plotting functions
│       └── report.py                  # metrics & information-ratio summary tables
├── notebooks/
│   └── Portfolio_Replication.ipynb    # original exploratory notebook
└── docs/
    ├── Group_Details.txt              # group roster & contribution split
    └── Report.pdf                     # assignment report
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .                  # installs the package + dependencies
# or: pip install -r requirements.txt
```

### Data

Daily price CSVs are not included in this repository. Place them under
`data/raw/` before running:

- `data/raw/^GSPC.csv` — S&P 500 index
- `data/raw/<TICKER>.csv` — one file per constituent stock

Each CSV needs a date index and a `Close` column. `TensorFlow` and `cvxpy`
are optional — the autoencoder falls back to PCA and the Markowitz solver
falls back to SLSQP if either is unavailable.

## Usage

```bash
python main.py
```

This runs rolling cross-validation to pick a sparsity level for each of 6
models, evaluates them on train/validation/holdout tracking error, and
writes the following plots to `results/`:

- `fig1_cumulative_returns.png` — cumulative returns of all 6 models vs. the S&P 500, across train/validation/holdout
- `fig2_sparsity_vs_te.png` — tracking error vs. number of selected stocks (k), for each model
- `fig3_sector_drift.png` — sector-weight drift of the best validation model vs. the benchmark universe
- `fig4_information_ratio.png` — validation vs. holdout information ratio, across all 6 models
- `fig5_metrics_table.png` — full metrics summary table (tracking error, information ratio, correlation, R²)

## Configuration

All parameters live in [`config.yaml`](config.yaml):

| Key | Description | Default |
|---|---|---|
| `dates.*` | Train/validation/holdout window boundaries and rolling-CV fold boundaries | see file |
| `selection.k_max_final` / `k_min_opt` | Search range for the optimal sparsity level per model | `50`, `10` |
| `selection.sweep_min_k` / `sweep_max_k` / `sweep_step` | Range swept for the sparsity-vs-TE plot | `10`, `100`, `5` |
| `selection.selection_tol` | TE tolerance (pct pts) for preferring a sparser k | `0.30` |
| `weighting.max_stock_weight` | Per-stock weight cap | `0.08` |
| `weighting.sector_cap_buffer` | Allowed sector overweight vs. benchmark | `0.03` |
| `autoencoder.max_latent` / `dropout` / `epochs` | Autoencoder architecture/training params | `12`, `0.20`, `10` |
| `market.trading_days_per_year` | Used to annualize daily returns | `252` |
| `simulation.random_seed` | Seed for reproducible selection/weighting | `42` |

## Method notes

- **Sector mapping**: `data/sector_map.csv` maps each constituent to a
  GICS-style sector, used only to benchmark the pipeline's own selected
  universe against the mapped constituent universe — the dataset does not
  include official index sector weights or market caps.
- **k selection**: chosen independently per model via 3 rolling
  train/validation folds, not on the holdout window, to avoid holdout
  leakage into the model-selection process.
- **Weighting caps**: both Markowitz and HRP weights are passed through
  the same constraint-enforcement step (8% per-stock cap, sector caps
  relative to the benchmark universe) so the two schemes are compared on
  equal footing.

## Group 34

See [`docs/Group_Details.txt`](docs/Group_Details.txt) for the full roster
and contribution breakdown.
