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

## Next entries

Further decisions (layout, forecaster->weighting wiring, SARIMAX
baseline, API/MLOps additions) are logged below as they're made.
