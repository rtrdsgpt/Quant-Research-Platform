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
