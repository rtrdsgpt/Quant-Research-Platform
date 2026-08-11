"""Portfolio construction: turns a forecasted-return (alpha) signal or a
benchmark-tracking objective into asset weights.

Modules:
    optimizer: equal-weight / inverse-volatility / mean-variance / risk-parity
        / signal-weighted construction (from the return-forecasting project).
    weighting: cvxpy mean-variance (tracking-error *or* alpha-maximizing) and
        Hierarchical Risk Parity (HRP) weighting, plus per-stock/per-sector
        constraint enforcement (from the portfolio-replication project).
    alpha_portfolio: glue layer — turns src.models.forecaster's per-ticker
        Predicted_Return signal into weights via `weighting.py`, this is the
        forecast -> construct integration point.
    selection / sectors: sparse stock selection (LASSO / autoencoder) and
        sector-map utilities, used by the optional legacy replication mode.
"""
