"""FastAPI service wrapping the forecast -> construct -> backtest pipeline.

See src.api.main for the route definitions and src.api.jobs for the
underlying stage execution (reusing the same staged pipeline pieces as
main.py's --data-only / --train-only / --backtest-only / --step N CLI
flags, per todo.md's API layer spec).
"""
