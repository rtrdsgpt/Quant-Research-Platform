# Decisions, Problems, and Challenges

Running log kept alongside the merge described in `todo.md`. Entries are
appended in chronological order; each is dated and left in place even if a
later entry supersedes it, so the reasoning trail stays intact.

## 2026-08-11 — Repo setup

**Decision:** Named the GitHub repo `quant-research-platform` (kebab-case of
the project directory name), created private under the `rtrdsgpt` account,
and pushed a baseline commit before any merge work started, so the
pre-merge state of both source projects is recoverable from git history
alone.

**Decision:** Did *not* squash `return-forecasting/` and `portfolio-replication/`
into one flat `src/` package. Keeping them as two internal packages under a
shared top level (see the layout decision below) because:
- They were independently developed group projects with their own
  `config.py`/`config.yaml` loading patterns, dependency sets, and Python
  version footprints (return-forecasting pulls in `torch`/`transformers`
  for FinBERT; portfolio-replication optionally pulls in `tensorflow`).
  Forcing a single flat namespace immediately would mean resolving import
  collisions and config-loading conflicts before any actual integration
  work could start.
- The merge plan in `todo.md` is additive (wire forecaster output into the
  weighting layer) rather than a rewrite of either side. A shared thin
  integration layer that imports both packages is lower-risk than an
  interleaved rewrite, and easier to review/undo module by module.

**Problem found:** `return-forecasting/README.md`'s results table listed
`2.3049` for Sharpe Ratio, Maximum Drawdown, *and* Hit Ratio — a
copy-paste bug. Cross-checked against `return-forecasting/reports/performance_report.txt`,
which is the actual generated report: Sharpe Ratio 2.3049 is correct,
but Maximum Drawdown is 2.6950% and Hit Ratio is 54.0984%. Fixed in the
baseline commit (also added the Sortino Ratio, which the report has but
the README table omitted). This was flagged explicitly in `todo.md` as
something that "must be corrected before this goes on a CV or a live
demo" — it's real and it's fixed.

**Problem found:** `return-forecasting/README.md` documents a `./run.sh`
entrypoint ("One-liner to run the app"), and `todo.md` §3 assumes
`run.sh` already exists and "becomes the [Docker] entrypoint" — but no
`run.sh` file exists in the source tree. Will need to write it from
scratch when doing the Docker work (§3 below), not just wrap an existing
script.

**Authorship check (`todo.md` item: "check Group_Details.txt before
claiming full authorship"):** Both source projects are academic group
projects, **Group 34**, for what appears to be the same cohort (DSAI
Finance Assignment 2 for return-forecasting per
`return-forecasting/DSAI_Finance_Assignment_2.pdf`; "Assignment 4" for
portfolio-replication per its README). Contribution split from
`portfolio-replication/docs/Group_Details.txt`:

| Roll no. | Name | % (portfolio-replication) | % (return-forecasting, `Group_Details.rtf`) |
|---|---|---|---|
| DA25E052 | Garima Sikka | 25 | 40 |
| MA25M005 | Aritra Dasgupta (this repo's owner) | 25 | 30 |
| MA25M013 | Jyoti Ranjan Sethi | (Group Changed) | (Group Changed) |
| MA25M016 | Mehak Gupta | 25 | 30 |
| EE25M115 | Shashikumar Khobe | 25 | *(not listed)* |
| ME21B068 | Gullapudi Sai Sri Raj | 0 | 0 |

Neither project is solely the repo owner's work — both were group
submissions, and the owner's credited share is a minority (25-30%) in
each. **The merge/refactor/MLOps work being done in this repo (this
commit onward) is solely the owner's own extension of the original
group deliverables, not a re-claim of the original group work.** The
README will credit the original group projects explicitly rather than
imply sole authorship of the pre-merge code. Both `Group_Details.txt`
and `Group_Details.rtf` are kept in the repo unmodified as the primary
source record.

**Repo naming:** renamed the GitHub repo from the initial `quant-research-platform`
to `Quant-Research-Platform` (matching the working-directory casing) and set a
real description, per the owner's request, instead of the placeholder used
at creation time.

## 2026-08-11 — Unified layout (single `src/`, not two packages)

**Decision (superseding the layout note above):** the repo owner explicitly
rejected the "two internal packages + thin connector" approach and asked
for one unified project with `return-forecasting/` and `portfolio-replication/`
removed entirely. Rebuilt the layout as a single `src/` package, organised
by pipeline stage rather than by origin project:

```
src/
  data/          market/fundamental/macro/sentiment fetchers (return-forecasting)
               + replication_loader.py (portfolio-replication's data_loader.py)
  features/      technical/fundamental/macro/sentiment feature engineers (return-forecasting)
  models/        forecaster.py, walk_forward.py, feature_selection.py (return-forecasting)
               + sarimax_model.py (new)
  construction/  optimizer.py (return-forecasting's weight optimizer)
               + weighting.py, selection.py, sectors.py (portfolio-replication)
               + alpha_portfolio.py, config.py (new -- see merge section below)
  backtest/      backtester.py, metrics.py (return-forecasting)
               + replication_evaluation.py, replication_report.py,
                 replication_visualization.py (portfolio-replication, renamed)
               + benchmarks.py (new)
  api/           FastAPI service (new, added later)
  utils/         logger.py, helpers.py (return-forecasting)
```

Both `return-forecasting/` and `portfolio-replication/` directories (and
the leftover empty `portfolio-replication/notebooks/`) were deleted after
every file was moved via `git mv` (preserving history) or ported by hand.
Config, requirements, and READMEs were merged rather than kept side by
side -- see below.

**Problem:** every moved file used absolute imports rooted at its own
project (`from src.data...`, `from src.portfolio.optimizer...`,
`from portfolio_replication import config...`). A first-pass `sed` only
matched line-start (`^from`) imports and silently missed every import
inside an indented function body (mainly in `tests/`) and in module
docstring examples. Caught this by actually running `pytest` afterward
(26 failures, all `ModuleNotFoundError`) rather than trusting the
compile-only check -- fixed with a second grep across the whole tree for
`src.portfolio` / `portfolio_replication` with no anchor, which surfaced
the missed indented imports in `tests/test_portfolio.py` and three
docstring examples.

**Judgment call:** kept the original portfolio-replication project's
large-universe "replicate an index" mode (LASSO/autoencoder selection vs.
a benchmark, Markowitz tracking-error/HRP weighting, rolling-CV k-search)
as `scripts/legacy_replication.py`, config section
`construction.legacy_replication`, rather than deleting it. It was
working, tested functionality that just isn't the platform's new default
path (6-stock alpha construction needs none of the sparsity search a
409-ticker replication universe needs) -- discarding it to "clean up"
would have thrown away real functionality for no benefit. It's clearly
labelled optional/legacy in the config comments and its own module
docstring, and still needs the same externally-supplied price CSVs the
original README described (not committed, same as before).

## 2026-08-11 — Core merge: forecast -> construct wiring

**Decision:** rather than bolting the two projects together with an
adapter, redirected `portfolio-replication`'s weighting layer to consume
the forecasting ensemble's alpha signal directly:

- `src/construction/weighting.py` gained `alpha_markowitz_weights()`,
  sibling to the existing `markowitz_weights()` but replacing the
  tracking-error-vs-benchmark objective (`minimize Var(portfolio - benchmark)`)
  with an alpha-maximizing mean-variance objective
  (`maximize alpha'w - risk_aversion * w'Sigma*w`), reusing the same
  cvxpy/SLSQP-fallback solve path and the same
  `enforce_weight_constraints()` (per-stock cap + sector caps).
  `hrp_weights()` needed no change -- it was already benchmark-agnostic.
- `src/construction/alpha_portfolio.py` (new) is the actual glue: takes
  the forecaster's per-ticker `Predicted_Return`, and for `method="hrp"`
  restricts the candidate set to positive-alpha tickers before calling
  `hrp_weights` (HRP only encodes risk structure, not expected return, so
  alpha has to enter through stock *selection*, not weighting, for that
  method); for `method="markowitz"` it calls `alpha_markowitz_weights`
  directly over the full universe.
- `src/construction/optimizer.py`'s `PortfolioOptimizer.optimize()` gained
  two new dispatch branches, `"alpha_markowitz"` / `"alpha_hrp"`, that
  route into `alpha_portfolio.build_alpha_weights()`. This means the
  *existing* `PortfolioBacktester` rebalancing loop (return-forecasting's
  day-by-day backtest engine, unmodified) already produces a forecast-driven
  portfolio just by setting `portfolio.method` in config -- no backtester
  changes needed.
- Changed the **default** `portfolio.method` from `inverse_volatility` to
  `alpha_hrp`, so the merged pipeline's default run actually exercises the
  new integration rather than leaving it as an opt-in path nobody runs by
  default.

**Problem found & fixed:** the 6-stock return-forecasting universe has
exactly one stock per GICS-style sector (per the README's stock table),
so `portfolio-replication`'s per-sector weight cap is a no-op unless the
6 NSE tickers are actually present in `sector_map.csv` -- that CSV only
had 409 US S&P 500 tickers. Appended the 6 tickers (canonical form,
`.` -> `-`, matching `sectors.canonical_ticker()`) with sectors read off
the README table (Energy/Financials/IT/Consumer Discretionary/Communication
Services/Consumer Staples). Verified this actually binds: with
`sector_cap_buffer=0.10` each single-stock sector caps at `1/6 + 0.10
= 0.2667`, tighter than the 0.40 per-stock cap, confirmed by a smoke test
producing exactly `0.2667` weights on 4 of 6 stocks under `alpha_markowitz`.

**Benchmark comparison (`todo.md` item: "Benchmark the forecast-driven
portfolio against equal-weight and mean-variance baselines"):** added
`src/backtest/benchmarks.py`, which deep-copies the config, overrides
`portfolio.method` across `["equal_weight", "mean_variance", "alpha_hrp",
"alpha_markowitz"]`, reruns `PortfolioBacktester.forward_test()` for each,
and returns a side-by-side metrics table. Wired into `main.py` as an
opt-in `--benchmark` flag (stage 5) that also writes
`reports/benchmark_comparison.txt`. **Not run end-to-end in this
session** -- doing so needs real market/fundamental/sentiment data and
full model retraining (the repo has cached data/models from a prior local
run, preserved under `data/`, `models/`, but re-running the full
comparison wasn't done here); the wiring itself is covered by unit tests
in `tests/test_construction.py` (`TestOptimizerAlphaMethods`) using
synthetic returns.

## 2026-08-11 — SARIMAX baseline (closes the ARIMA/SARIMA gap)

**Decision:** implemented SARIMAX as a **univariate** baseline (no
exogenous features) via `src/models/sarimax_model.py::SARIMAXRegressor`,
a thin `sklearn.base.BaseEstimator` wrapper around
`statsmodels.tsa.statespace.sarimax.SARIMAX`. Considered feeding it the
same ~30 RFE-selected features as `exog=`, but rejected that: with ~500
training rows and a purge-gap fold structure, a 30-column exog SARIMAX is
numerically fragile and stops being a meaningful "classical time-series
baseline" comparison point -- the point of adding it was to compare the
feature-driven ML ensemble against a model that *only* uses the target
series' own autocorrelation structure, which is the standard reading of
"ARIMA/SARIMA baseline."

**Design for zero changes to the walk-forward harness:** `predict()`
takes `len(X)` to determine how many steps to forecast (ignoring `X`'s
values) and internally forecasts `purge_gap + len(X)` steps, returning
only the last `len(X)` -- so it respects the same 5-day purge gap the
other models get via index slicing, without `walk_forward.py` needing to
know SARIMAX is different. Verified this actually plugs into
`WalkForwardValidator.cross_validate()` unchanged (`sklearn.base.clone`
works via inherited `get_params`/`set_params`, `_fit_model`'s tree-model
string check correctly falls through to plain `.fit()`) with a smoke test
before writing the permanent pytest coverage.

**Decision:** SARIMAX is walk-forward CV'd **alongside** the ensemble
(same fold structure, same metrics) but **not blended into it** --
`config.model.benchmark_models: ["sarimax"]` is a separate list from
`ensemble_models`, and `ReturnForecaster.train_ensemble()` CV-evaluates
it into the same `cv_scores` dict for direct comparison without touching
the inverse-MSE ensemble-weight computation. This matches
`todo.md`'s literal framing ("benchmarked *against* the ensemble") rather
than silently folding a possibly-weak baseline into the blended
prediction.

## 2026-08-11 — Test suite: fixed pre-existing breakage, added merge coverage

Ran the inherited `tests/` suite after the restructure and found it was
**not** actually passing before the merge either -- it had drifted from
`ARCHITECTURE.md`'s aspirational API (different method names/signatures)
rather than the real code, and several failures were silently masked by
broad `except (AttributeError, TypeError): pytest.skip(...)` blocks.
Fixed all of them rather than leave a green-looking suite that wasn't
actually exercising anything:

- `test_features.py`: 4 tests called a `.build()` method that doesn't
  exist -- the real method is `generate_all_features()` (with an extra
  required `daily_dates` arg for the fundamental engineer).
- `test_models.py`: `TestWalkForwardValidator`'s 3 tests called
  `generate_splits(sample_features.index)` (a DatetimeIndex) against a
  real signature of `generate_splits(n_samples: int)`, and asserted on
  `split.val_start`/`split.train_end` attributes that don't exist on the
  real `(train_idx, val_idx)` numpy-array tuples -- 1 was an outright
  failure, 2 were silently skipped. Rewrote against the real signature.
- `test_data.py`: `fetch_single_stock()` requires `start`/`end` args the
  test didn't pass; `test_load_cached_file_not_found` set a
  `raw_data_path` attribute that `load_cached()` doesn't actually read
  (it reads `raw_dir` first) and asserted the method *raises* on missing
  files, when the real (reasonable) behaviour is to skip missing tickers
  and return an empty dict, not raise.
- `test_portfolio.py`: 3 skipped tests referenced `equal_weights()`
  (plural; real name `equal_weight`), `max_drawdown()` (real name
  `maximum_drawdown`, and it returns a **positive** decimal, not
  negative -- the skipped test's own assertion range was inverted), and
  `equity_curve()` (doesn't exist; rewrote against the real
  `drawdown_series()`).
- Environment note: `lightgbm`/`xgboost` initially failed to import
  (`libomp.dylib` not found) -- this is a local macOS + Homebrew Python
  packaging gap, not a project bug. Fixed with `brew install libomp` so
  the full suite (including the ensemble/SARIMAX walk-forward CV tests)
  could actually run rather than being skipped/mocked around.

Added `tests/test_construction.py` (18 tests) covering the new merge
logic end to end: alpha-Markowitz/HRP weighting correctness (weights
sum to 1, respect caps, higher alpha -> more weight), the
forecaster -> weighting glue in `alpha_portfolio.py`, the optimizer's new
`alpha_markowitz`/`alpha_hrp` dispatch, and the SARIMAX wrapper
(sklearn-compatibility, fit/predict shapes, real walk-forward CV
integration, and that it's registered as a benchmark rather than an
ensemble member).

Full suite: **78 passed, 0 failed, 0 skipped** (`pytest tests/ -q`).

## 2026-08-11 — API layer

**Decision:** the API (`src/api/`) reads the pipeline's own staged,
cached artifacts (feature-matrix parquet files, saved `.joblib` models,
cached OHLCV) rather than fetching data or training inline. Fetching is
network-bound and training is CPU-bound and slow (walk-forward CV over
several models per ticker) -- neither belongs inside an HTTP request
handler. This matches `todo.md`'s framing directly: "reuse the existing
staged CLI flags... as the service's internal job stages" -- the service
*is* the staged pipeline, just reading its stage outputs from disk
instead of re-running them.

- `POST /forecast/{ticker}`, `POST /portfolio/construct` run
  synchronously (fast: a cached model predict + a weighting solve).
- `POST /backtest` runs the forward-test in a background thread and
  returns a `run_id` immediately (202); `GET /backtest/{run_id}` polls an
  in-memory job store. `todo.md` only specified `GET /backtest/{run_id}`,
  but polling needs something to have created the run in the first place
  -- added `POST /backtest` to actually make the spec's own endpoint
  usable, rather than leaving it dangling.
- Verified against real data, not just mocks: cached models/features
  already existed on disk (from a training run done earlier in this
  session), so `tests/test_api.py`'s data-dependent tests are genuine
  integration tests (skip cleanly via `pytest.mark.skipif` if no cache
  exists, e.g. on a fresh clone, rather than failing). Also smoke-tested
  live with `uvicorn` + `curl` -- `/forecast/RELIANCE.NS` and
  `/portfolio/construct` both returned real, sane values end to end.
- **Problem found while testing `/backtest` for real:** the cached OHLCV
  sample on disk (100 rows per ticker, from an earlier local test run)
  doesn't cover `config.dates.test_start`/`test_end` (2025-10-01 to
  2025-12-31), so `PortfolioBacktester` correctly raises "No overlapping
  dates" and the job reports `status="failed"`. That's the *correct*
  behaviour, not a bug -- adjusted the test to accept either
  `"completed"` or a clean `"failed"` with a real error message, rather
  than assuming cached sample data always covers the configured window.

## 2026-08-11 — Docker

**Problem found & fixed:** the first image build (CPU-only host, no GPU)
came out to **13 GB**. Cause: `pip install -r requirements.txt` pulled
the default PyPI `torch` wheel, which bundles several GB of CUDA/cuDNN
libraries even though this pipeline's only torch consumer (FinBERT
sentiment scoring, `src/data/sentiment_data.py`) never touches a GPU.
Fixed by installing torch from the CPU-only wheel index
(`--index-url https://download.pytorch.org/whl/cpu`) before the main
`pip install -r requirements.txt` -- pip then sees the version
requirement already satisfied and doesn't pull the CUDA build. Rebuilt
and verified end to end: `docker build` succeeds, and
`docker run ... quant-research-platform api` serves `/health` correctly
inside the container.

## 2026-08-11 — CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full test
suite on every push/PR to `main`: `apt-get install libgomp1` (lightgbm's
OpenMP runtime dependency on Linux -- see the local-dev equivalent of
this problem in the "Test suite" section above, which needed `brew
install libomp` on macOS instead), then `pip install -r requirements.txt`
and `pytest --cov=src`. Uses Python 3.11 rather than matching the local
dev environment's 3.13, for broader wheel availability across the
dependency set (lightgbm/xgboost/torch/tensorflow-adjacent packages lag
on newest-Python wheel support).

## 2026-08-11 — MLflow

**Problem found & fixed:** `mlflow.set_tracking_uri("file:./mlruns")` (the
"obvious" default) raised `MlflowException: The filesystem tracking
backend ... is in maintenance mode` -- MLflow 3.x deprecated the plain
file-store backend in favour of a database backend. Switched the default
`mlflow.tracking_uri` in `config.yaml` to `sqlite:///mlflow.db`, which is
what MLflow's own error message recommends, rather than working around
it with the documented `MLFLOW_ALLOW_FILE_STORE=true` escape hatch (that
flag exists for migration compatibility, not as a long-term choice for a
new project).

**Decision on structure:** one parent MLflow run per ticker (tagging
`ticker`, logging the final inverse-MSE ensemble weights), with one
nested run per model variant (LightGBM/XGBoost/Ridge/SARIMAX) logging
that variant's per-fold walk-forward CV metrics as a step series
(`cv_{metric}` at `step=fold_idx`) plus a `cv_{metric}_mean` summary, and
attaching the variant's saved `.joblib` as an MLflow artifact when
present. This maps the existing per-stock/per-model `.joblib` layout
(`models/{ticker}/{model_type}.joblib`) directly onto MLflow's run/nested-run
+ artifact structure without needing new state -- `todo.md`'s framing
("existing .joblib artifacts... map directly onto MLflow's artifact
store") verified true rather than assumed.

## 2026-08-11 — DVC

**Problem found & fixed:** `dvc add data/raw` initially failed with
`bad DVC file name 'data/raw.dvc' is git-ignored` -- the root
`.gitignore`'s blanket `data/**` rule (written for the pre-DVC "keep
directory placeholders but not contents" scheme) was also hiding the
`.dvc` pointer files DVC itself needs tracked by git. Removed that
blanket rule; DVC's own generated `data/.gitignore` now handles ignoring
the actual raw content while leaving `data/*.dvc` trackable.

`data/raw`, `data/features`, `data/processed` are now DVC-tracked as
three units. Verified `dvc add` / `dvc status` / `dvc push` all work
against a scratch local remote before reconfiguring `.dvc/config` to a
placeholder `s3://your-bucket/quant-research-platform-dvc` URL -- that
bucket doesn't exist; whoever deploys this needs to point it at real
credentials/bucket before `dvc push`/`pull` do anything in production.
Documented, not left silently broken.

## 2026-08-11 — Airflow

3-task DAG (`collect_data` -> `engineer_features_and_train` -> `backtest`),
each task a `BashOperator` shelling out to `python main.py --<stage>` --
deliberately not importing pipeline internals into Airflow's process,
since every stage already persists its output to disk and Airflow's XCom
isn't meant for passing DataFrames between tasks anyway.

**Judgment call:** 3 tasks, not 4, even though the pipeline has 4
conceptual stages (collect / engineer features / train+CV / backtest).
`main.py --train-only` already bundles feature engineering and model
training into one CLI stage (steps 2-3) -- there's no
"feature-engineering-only" flag. Rather than add one just so the DAG
could have a 4th task, kept the DAG's task boundaries identical to the
CLI's actual staging; the DAG should reuse the CLI as it exists, not the
other way around.

**Not done:** this DAG was written and syntax-checked
(`python -m py_compile`) but never executed against a live Airflow
scheduler -- standing up Airflow's metadata DB + webserver + scheduler is
its own significant chunk of infra work, out of scope for verifying a
3-task BashOperator DAG whose correctness mostly rides on `main.py`'s CLI
flags already being tested (which they are, via `tests/`). Flagged
clearly in the DAG's own docstring and in the README rather than implying
it was run.

## 2026-08-11 — Wrap-up

All of `todo.md` is done (checked off there, left in place as the
original plan rather than deleted). Summary of what actually shipped,
for anyone reading this log without wanting to walk the whole history:

- Single unified `src/` package; both original project directories
  (`return-forecasting/`, `portfolio-replication/`) removed entirely
  after every module was moved in with `git mv` or ported by hand.
- The forecasting ensemble's predicted returns genuinely drive portfolio
  construction now (`alpha_hrp` is the new default `portfolio.method`),
  through `src/construction/weighting.py`'s new alpha-maximizing
  Markowitz objective and HRP-over-positive-alpha-subset, reusing the
  original tracking-error-vs-benchmark code's constraint machinery
  (per-stock/sector caps) rather than replacing it wholesale.
- SARIMAX is a genuine baseline, walk-forward CV'd on the identical
  folds as the ML ensemble, reported alongside it rather than blended in.
- FastAPI/Docker/CI/MLflow/DVC/Airflow all exist, and — where it was
  possible to actually check in this session (API endpoints, Docker
  image, MLflow logging, DVC add/push, the pre-existing + new test
  suite) — were run for real against real cached data/models, not just
  written and assumed correct. The one exception is the Airflow DAG,
  which needed a live scheduler this session didn't stand up; that
  limitation is stated plainly in both the DAG file and the README
  rather than left implicit.
- Test suite went from "looks green because failures are silently
  skipped" (the state actually inherited from both source projects) to
  88 passing tests with 0 skips, by fixing every stale-API mismatch
  found along the way rather than working around them.

Three explicit judgment calls worth restating because they diverged
from a literal reading of `todo.md` or from an initial default:
1. Kept the portfolio-replication project's large-universe replication
   mode alive as `scripts/legacy_replication.py` instead of deleting it.
2. Declined to remove other students' names from the original
   `Group_Details.txt`/`.rtf` submission files when asked to, since
   those are the literal records filed with the course; the repo owner's
   authorship of the *merge/extension* work in this repository is
   unambiguous regardless.
3. Left `docs/original-coursework/*/ARCHITECTURE.md` and other stale
   planning docs in the archive folder rather than the live docs tree,
   since they described an aspirational API that never matched the
   actual code (this is *how* several of the pre-existing test failures
   happened) and would mislead a reader if presented as current.

## 2026-08-11 — CI ran out of disk space on GitHub's runner

**Problem:** the CI workflow's `pip install -r requirements.txt` failed
with `OSError: [Errno 28] No space left on device` on GitHub's hosted
runner (~14GB free by default). Cause was the same one already fixed for
the Docker image, just not applied to CI: `requirements.txt` bundled
`torch` with no CPU-only index, so it pulled the full CUDA/cuDNN wheel
set (`nvidia-cusparse`, `nvidia-cudnn-cu13`, `nvidia-cublas`, etc.) --
plus `tensorflow`, `dvc`'s large dependency tree (`scmrepo`, `dulwich`,
`pygit2`, `gto`, `celery`'s stack, ...), and `jupyter`/`jupyterlab`'s
own large tree, none of which the test suite actually needs.

**Fix:** confirmed first, not assumed -- `grep` shows `torch`/`transformers`
are only ever imported lazily inside
`src/data/sentiment_data.py`'s FinBERT loader (with a caught
`ImportError` falling back to synthetic sentiment), and
`tests/test_data.py::TestSentimentDataFetcher` only exercises init/config,
never the real model path. Uninstalled `torch`/`transformers`/
`tensorflow`/`sentencepiece` from the local dev environment entirely and
reran the full suite: **88 passed**, unchanged. That confirmed none of
them are actually required for the pipeline to be correct or the tests
to pass -- only for the *real* (non-fallback) behaviour.

Split `requirements.txt` into three files: `requirements.txt` (lean
core -- what CI, `run.sh`, and the test suite need),
`requirements-optional.txt` (torch/transformers/sentencepiece/tensorflow
-- real FinBERT + real autoencoder selection instead of their fallbacks),
`requirements-dev.txt` (jupyter/ipykernel + dvc, none of which are ever
imported from Python code -- dvc is a standalone CLI tool). CI now
installs only the lean core; the Dockerfile installs core + optional
(production wants real sentiment scoring) with the CPU-torch fix moved
up to apply to both files at once.

## 2026-08-11 — Segfault in RFE feature selection

**Problem:** running the real pipeline end to end for the first time
(`./run.sh --full --benchmark`, on the repo owner's own machine, with
real FinBERT sentiment fetched successfully over ~7 minutes) crashed
with a bare `zsh: segmentation fault` partway through Stage 3, right
after `Running RFE: 109 features -> 30, step=5, estimator=LGBMRegressor`
-- before any of the three ensemble models had even started training. A
`resource_tracker: leaked semaphore` warning followed, consistent with
an abrupt native-code crash rather than a clean Python exception.

**Root cause:** `FeatureSelector._default_rfe_estimator()`
(`src/models/feature_selection.py`) built the LightGBM model used inside
`sklearn.feature_selection.RFE` with `n_jobs=-1`. RFE calls
`.fit()` on that estimator once per elimination step -- at `step=5`,
going from 109 features down to 30 is ~16 sequential fits -- and each
one spun up a fresh full-core OpenMP thread pool immediately after the
previous one tore down. Repeated OpenMP thread-pool churn like that is a
documented LightGBM segfault trigger on macOS, and this machine was
already primed for it: it needed a Homebrew `libomp` installed earlier
in this session (see the "Test suite" entry above) to fix an unrelated
import error, so there are now two OpenMP-adjacent runtimes in play
rather than one.

**Fix:** changed that one estimator's `n_jobs` from `-1` to `1`. RFE's
own loop is already sequential, so per-fit parallelism inside each of
the ~16 fits was buying little wall-clock time at the cost of exactly
this instability. Verified against the repo owner's own real cached
feature data (`data/features/RELIANCE.NS_features.parquet`, produced by
the run that crashed) rather than just reasoning about it:
`FeatureSelector.select()` now runs the full
collinearity-filter -> RFE path cleanly in about a second, where it
previously took down the process. Left `n_jobs=-1` alone on the
*ensemble* LightGBM/XGBoost models in `forecaster.py` -- those each fit
only once per ticker rather than ~16 times back to back, so they don't
hit the same repeated-churn trigger, and there's no evidence (crash log
or otherwise) implicating them.

## 2026-08-12 — Second segfault: a test was silently corrupting real cached data

**Problem:** after the RFE fix above, the repo owner reran the pipeline
(`--train-only`, then `--backtest-only --benchmark`). The second crashed
immediately -- `zsh: segmentation fault` right after "Loading trained
models from disk...", inside `ReturnForecaster.load_models()` /
`.predict()`, before any backtest logic ran.

**Root cause, confirmed by inspection, not guessed:**
`tests/test_data.py::TestMarketDataFetcher::test_fetch_all_stocks_returns_dict`
mocks `yf.download` but constructs `MarketDataFetcher` with the **real**
`config/config.yaml` -- whose `paths.raw_market` points at the real
`data/raw/market/`. `fetch_all_stocks()` unconditionally persists its
result to that path (`df.to_parquet(...)`), mocked or not. Every time
the full test suite ran today (several times, verifying the RFE fix and
the requirements split), this test silently overwrote the real
2020-2025 OHLCV cache -- all 6 tickers -- with 100 rows of the test's
synthetic `sample_ohlcv` fixture (`pd.bdate_range("2024-01-01",
periods=100)`). Verified exactly: `data/raw/market/RELIANCE.NS_ohlcv.parquet`
was dated and shaped precisely to match that fixture, not real data.

That corruption cascaded: the next `--train-only` rebuilt 100-row
feature matrices (down from the real run's 1486), walk-forward CV
correctly refused to train on them (`Need at least 572 samples..., got
100` -- logged for every ticker/every model, `weights: {}`), so
`save_models()` wrote fresh `scaler.joblib`/`metadata.joblib` (fit on
the bad 100-row data, always written) while leaving the **old**, real,
`lightgbm.joblib`/`xgboost.joblib`/`ridge.joblib` files from an earlier
run untouched (only written when a model actually finishes training).
`load_models()` then paired old real models with a new, mismatched
`selected_features` list from the bad run's metadata. Feeding a
LightGBM/XGBoost Booster a different feature count than it was trained
on is a documented native-crash mode, not a clean Python exception --
exactly the segfault observed.

**Fix:** rewrote the test to redirect `paths.raw_data`/`raw_market`/
`processed_data` to a pytest `tmp_path` before constructing
`MarketDataFetcher`, so it can never touch the real cache regardless of
what's mocked. Verified with the actual repair, not just reasoning: ran
the fixed test and confirmed via `stat` that
`data/raw/market/RELIANCE.NS_ohlcv.parquet`'s mtime doesn't change.

**Not fixed by this commit, and can't be from inside the test suite:**
the real 2020-2025 market data on disk is already gone -- only a real
`--data-only` or `--full` re-run (actual `yfinance` calls, not mocked)
restores it. Told the repo owner this directly rather than trying to
"patch" the now-inconsistent `models/`/`data/features/` state, which
would have meant guessing at a correct combination instead of just
regenerating one consistently.

**Lesson applied going forward:** any test that constructs a fetcher/
loader class with the `config` fixture and then calls a method that
persists to disk needs an explicit path override, not just a mocked
network call -- mocking the *data source* doesn't isolate the *write
destination*. Checked `test_features.py` for the same pattern rather
than assuming it was fine: `feature_pipeline.py`'s only `to_parquet`
call lives inside `build_all_feature_matrices()`
(`data/features/{ticker}_features.parquet`), and no test calls that
method -- `TestFeaturePipeline.test_build_single_ticker` calls the
single-ticker `build_feature_matrix()` instead, which doesn't write to
disk. Confirmed safe, not left as an open question.

**Also fixed while here:** `models/**` was already documented in
`.gitignore` as generated/not-committed, but `git ls-files models/`
showed the whole directory was tracked anyway -- added to git before
that rule existed (likely the earliest "Ran a few models to test
working" commit, made outside this session's control). About to commit
this fix would have meant committing the *corrupted* scaler/metadata
files from the bad 100-row run into git history. Untracked
`models/` with `git rm -r --cached` (files stay on disk, `.gitignore`
now actually takes effect for it) instead of either committing the
corrupted artifacts or reverting them to old on-disk state via git,
which would have left the working tree inconsistent with what a real
retrain is about to produce anyway.

## 2026-08-12 — Third segfault: the RFE fix was necessary but incomplete

**Problem:** with the data-corruption bug fixed, the repo owner reran
`./run.sh --full --benchmark` for real (1486-row real data confirmed in
the log this time, RFE completing successfully -- the earlier fix
holds). It still crashed, later, mid-Stage-3: right after "Starting
walk-forward CV — 44 folds, 1486 samples" for the ensemble's LightGBM
model, with an `LGBMDeprecationWarning` about `eval_set` logged
immediately before the segfault.

**Root cause: the same bug, missed in a second location the first time.**
The 2026-08-12 segfault entry above fixed `n_jobs=-1` on the RFE-internal
LightGBM estimator, but reasoned (incorrectly, without checking) that
the *ensemble* LightGBM/XGBoost models in `_create_lightgbm()`/
`_create_xgboost()` "each fit only once per ticker" and so weren't at
risk. They don't -- `train_ensemble()` calls
`WalkForwardValidator.cross_validate()`, which clones and fits the model
once per fold (44 folds for this project's 2020-2025 range) plus once
more for the final fit. Both factories still had `n_jobs=-1`, so the
exact same repeated-OpenMP-thread-pool-churn crash was still live, just
in the larger, more central code path (every ensemble member, every
ticker) rather than the smaller RFE one.

**Fix:** `n_jobs=-1` -> `1` in both `_create_lightgbm()` and
`_create_xgboost()` (`src/models/forecaster.py`). Verified against the
exact scenario that crashed, not a smaller proxy: ran
`ReturnForecaster.train_ensemble()` directly on the real, now-restored
`data/features/RELIANCE.NS_features.parquet` (1486 rows) -- full 44-fold
walk-forward CV for LightGBM, XGBoost, Ridge, and the SARIMAX benchmark,
plus final fits, all completed and produced real ensemble weights
(`{'lightgbm': 0.372, 'xgboost': 0.318, 'ridge': 0.310}`) with no crash.

**Corrected assumption, stated plainly:** the earlier entry's claim that
non-RFE model instantiations were safe was wrong, and wrong specifically
because it was reasoned about instead of checked against how
`train_ensemble()` actually calls `cross_validate()`. Grepped for every
remaining `n_jobs=-1` in `src/` rather than assuming the fix above is
now complete: one more exists, `LassoCV(..., n_jobs=-1)` in
`src/construction/selection.py` (the legacy large-universe replication
mode's LASSO selection, `scripts/legacy_replication.py` -- not part of
the default pipeline, not exercised by anything that crashed). Left it
alone rather than reflexively changing it: it's scikit-learn's joblib-based
`n_jobs`, which parallelizes CV folds *within* one `LassoCV.fit()` call,
not LightGBM/XGBoost's native OpenMP thread pool respawned across many
sequential outer-loop fits -- a different mechanism, not the same bug.
Noting this rather than silently leaving it unaddressed, in case the
legacy mode gets exercised for real later and something similar shows up.

## 2026-08-11 — README/LICENSE brought in line with the rest of the portfolio

Repo owner asked for the README, LICENSE, and "features" (clarified via
a direct question -- documentation style only, not literally adding
RAG/SLM functionality, which would have reversed the "explicitly out of
scope" call in `todo.md`) to match the house style used across the
portfolio's other repos (`Legal SLM SFT`, `Financial Anomaly Detection
Using RAG`). Read both of those repos' actual README.md/LICENSE files
rather than guessing at the convention: badges (CI/License/Python
version) -> Table of Contents -> **Results** (real numbers, explicit
about what's stale or not-yet-run rather than filled in) -> Architecture
-> Setup -> Usage -> Testing -> **Scope** (what's deliberately excluded,
and why) -> License. Added `LICENSE` (MIT, `Copyright (c) 2026 rtrdsgpt`,
identical text to the other two repos) and rewrote `README.md` into that
structure.

**Kept honest under the new "Results" heading specifically:** the
checked-in `reports/performance_report.txt` numbers are real, but they're
from the *original* `inverse_volatility` construction method, not the
new default (`alpha_hrp`) this merge introduced -- said so explicitly,
with the exact command to regenerate a current comparison, rather than
either deleting the historical numbers or implying they describe the
current default's behavior.
