#!/usr/bin/env bash
# Quant Research Platform entrypoint.
#
# Local use: creates/reuses a venv, installs deps if needed, then runs
# main.py with whatever args are passed through.
#   ./run.sh --full
#   ./run.sh --data-only
#
# Container use (see Dockerfile): the image already has deps installed
# system-wide, so this script skips the venv step and just execs main.py.
#   docker run quant-research-platform --backtest-only
#
# Special case: `./run.sh api` starts the FastAPI service instead of the
# batch pipeline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# torch, scikit-learn, and lightgbm/xgboost (via Homebrew libomp on macOS)
# each bundle/link their own copy of libomp.dylib. Once real FinBERT
# sentiment scoring (src/data/sentiment_data.py) imports torch, its
# libomp.dylib is resident in the process for good -- when lightgbm/xgboost
# later spin up their own OpenMP thread pools during walk-forward CV, two
# different runtime copies are active at once, which segfaults (reproduced
# and root-caused; see DECISIONS.md). KMP_DUPLICATE_LIB_OK alone silences
# the friendlier abort but still crashed in testing -- OMP_NUM_THREADS=1
# is required too (forces every OpenMP-using library, not just sklearn's
# n_jobs, to skip thread-pool creation entirely). Harmless on Linux/Docker,
# where this doesn't come up, and on machines with only one libomp.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

if [ "${1:-}" = "api" ]; then
    shift
    exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
fi

if [ -z "${QRP_IN_CONTAINER:-}" ]; then
    if [ ! -d venv ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate

    if [ ! -f venv/.deps_installed ] || [ requirements.txt -nt venv/.deps_installed ]; then
        echo "Installing dependencies (first run can take several minutes; torch is large)..."
        pip install --quiet --upgrade pip
        pip install --quiet -r requirements.txt
        touch venv/.deps_installed
    fi
fi

exec python main.py "$@"
