"""Airflow DAG: collect -> engineer features + train/CV -> backtest.

Orchestrates the existing staged CLI (`main.py --data-only` /
`--train-only` / `--backtest-only`) as an Airflow DAG -- todo.md's MLOps
section: "model the pipeline as a DAG ... the staged structure already
exists in code, just not orchestrated."

Each task shells out to `python main.py --<stage>` rather than importing
pipeline internals directly: every stage already persists its output to
disk (cached data/, data/features/, models/), so Airflow doesn't need to
pass DataFrames through XCom -- a task just needs the previous stage's
on-disk artifacts to exist, which BashOperator + `>>` dependencies give
for free, and it's the exact same entrypoint a human would run by hand.

Only 3 tasks, not 4, because `--train-only` already covers *both*
feature engineering and model training (main.py's step range for
`--train-only` is steps 2-3) -- there's no standalone
"feature-engineering-only" CLI flag, and adding one just to get a 4th
Airflow task would mean changing the CLI's staging to fit the DAG,
rather than the DAG faithfully reusing the CLI as it already exists.

Deployment note: this file was written and syntax-checked against the
stable Airflow 2.x BashOperator API, but was not run against a live
Airflow scheduler in this repo's dev environment (standing one up --
metadata DB, webserver, scheduler -- is out of scope here). To use it,
copy/symlink into an existing Airflow deployment's `dags/` folder and
set the `QRP_PROJECT_ROOT` Airflow Variable (or env var) to this repo's
absolute path.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("QRP_PROJECT_ROOT", "/opt/airflow/quant-research-platform")

default_args = {
    "owner": "quant-research-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="quant_research_platform_pipeline",
    description="forecast -> construct -> backtest: collect -> engineer features + train/CV -> backtest",
    default_args=default_args,
    schedule_interval="0 6 * * 1-5",  # weekdays, 06:00 -- `schedule_interval`
    # rather than the newer `schedule` alias, for compatibility with
    # Airflow >=2.2 (this DAG doesn't need anything from 2.4+'s dataset
    # scheduling or timetables).
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["quant-research-platform"],
) as dag:

    collect_data = BashOperator(
        task_id="collect_data",
        bash_command=f"cd {PROJECT_ROOT} && python main.py --data-only",
        doc_md="Stage 1: fetch OHLCV, fundamentals, macro, sentiment.",
    )

    engineer_features_and_train = BashOperator(
        task_id="engineer_features_and_train",
        bash_command=f"cd {PROJECT_ROOT} && python main.py --train-only --mlflow",
        doc_md=(
            "Stages 2-3: build feature matrices, then walk-forward CV + "
            "fit the LightGBM/XGBoost/Ridge ensemble (SARIMAX benchmarked "
            "alongside), logging fold-level results to MLflow."
        ),
    )

    backtest = BashOperator(
        task_id="backtest",
        bash_command=f"cd {PROJECT_ROOT} && python main.py --backtest-only --benchmark",
        doc_md=(
            "Stage 4 (+5): forward-test the configured portfolio "
            "construction method and compare it against equal-weight / "
            "mean-variance / alpha_hrp / alpha_markowitz baselines."
        ),
    )

    collect_data >> engineer_features_and_train >> backtest
