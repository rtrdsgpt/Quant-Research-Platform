"""Portfolio weighting: Markowitz tracking-error minimization and
Hierarchical Risk Parity (HRP), both followed by max-stock-weight and
sector-cap constraint enforcement.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from src.construction import config, sectors

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

WEIGHT_CACHE = {}


def _dataset_cache_key(X):
    return (str(X.index[0].date()), str(X.index[-1].date()), X.shape[0], X.shape[1])


def normalize_weights(weights, assets):
    w = pd.Series(weights, index=assets, dtype=float).fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    return w.to_dict()


def portfolio_returns(X, weights):
    w = pd.Series(weights, dtype=float)
    cols = [c for c in w.index if c in X.columns]
    if not cols:
        return pd.Series(index=X.index, data=0.0)
    w = w.loc[cols]
    w = w / w.sum()
    return X[cols].mul(w, axis=1).sum(axis=1)


def sector_cap_lookup(benchmark_sector_weights):
    caps = benchmark_sector_weights.to_dict()
    return {sector: min(1.0, weight + config.SECTOR_CAP_BUFFER) for sector, weight in caps.items()}


def enforce_weight_constraints(weights, benchmark_sector_weights):
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    w = w / w.sum()

    caps = sector_cap_lookup(benchmark_sector_weights)
    sector_labels = pd.Series({ticker: sectors.get_sector_full_name(ticker) for ticker in w.index})

    for _ in range(10):
        w = w.clip(upper=config.MAX_STOCK_WEIGHT)
        if w.sum() <= 0:
            w[:] = 1.0 / len(w)
        w = w / w.sum()

        sector_sums = w.groupby(sector_labels).sum()
        changed = False
        for sector, total in sector_sums.items():
            cap = caps.get(sector, 1.0)
            if total > cap + 1e-10:
                members = sector_labels[sector_labels == sector].index
                w.loc[members] *= cap / total
                changed = True

        remaining = 1.0 - w.sum()
        if remaining <= 1e-8:
            w = w / w.sum()
            if not changed:
                break
            continue

        sector_sums = w.groupby(sector_labels).sum()
        headroom = pd.Series(config.MAX_STOCK_WEIGHT, index=w.index) - w
        for asset in headroom.index:
            sector = sector_labels[asset]
            headroom[asset] = min(headroom[asset], caps.get(sector, 1.0) - sector_sums.get(sector, 0.0))
        eligible = headroom[headroom > 1e-8]
        if eligible.empty:
            break
        add = remaining * (eligible / eligible.sum())
        w.loc[eligible.index] += add

    w = w.clip(lower=0.0)
    w = w / w.sum()
    return w.to_dict()


def markowitz_weights(X_train, y_train, selected, benchmark_sector_weights):
    X_sel = X_train[selected].values
    y = y_train.values.reshape(-1, 1)
    # Compute covariance of (r_i - r_b) for each stock i.
    # w' Cov(R - r_b) w = Var(w'R - r_b) = TE^2  (since sum(w)=1).
    diff = X_sel - y
    te_cov = np.cov(diff, rowvar=False) + config.RIDGE_EPS * np.eye(len(selected))

    sector_names = [sectors.get_sector_full_name(t) for t in selected]
    sector_caps = sector_cap_lookup(benchmark_sector_weights)

    if HAS_CVXPY:
        w = cp.Variable(len(selected))
        objective = cp.Minimize(cp.quad_form(w, te_cov) + config.WEIGHT_REG * cp.sum_squares(w))
        constraints = [cp.sum(w) == 1, w >= 0, w <= config.MAX_STOCK_WEIGHT]
        for sector in sorted(set(sector_names)):
            idx = [i for i, s in enumerate(sector_names) if s == sector]
            cap = sector_caps.get(sector, 1.0)
            constraints.append(cp.sum(w[idx]) <= cap)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            return enforce_weight_constraints(pd.Series(w.value, index=selected), benchmark_sector_weights)

    x0 = np.full(len(selected), 1.0 / len(selected))

    def objective_fn(w_):
        return float(w_ @ te_cov @ w_ + config.WEIGHT_REG * np.sum(w_**2))

    constraints = {"type": "eq", "fun": lambda w_: np.sum(w_) - 1.0}
    bounds = [(0.0, config.MAX_STOCK_WEIGHT)] * len(selected)
    res = minimize(objective_fn, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if res.success:
        return enforce_weight_constraints(pd.Series(res.x, index=selected), benchmark_sector_weights)
    return enforce_weight_constraints(pd.Series(x0, index=selected), benchmark_sector_weights)


def alpha_markowitz_weights(X_train, alpha, selected, benchmark_sector_weights, risk_aversion=None):
    """Mean-variance construction driven by a forecasted alpha vector.

    This is the alpha-construction counterpart to :func:`markowitz_weights`:
    instead of minimizing tracking error against a benchmark return series,
    it maximizes ``alpha' w - risk_aversion * w' Sigma w`` -- the standard
    mean-variance objective, using the ML ensemble's per-ticker predicted
    return as the expected-return vector.

    Args:
        X_train: Historical returns DataFrame (date x ticker), used only
            to estimate the covariance matrix `Sigma`.
        alpha: Mapping (dict or Series) of ticker -> forecasted return.
        selected: List of tickers to include (the candidate universe).
        benchmark_sector_weights: Series of sector -> weight, used only
            for sector-cap constraints (see `enforce_weight_constraints`).
        risk_aversion: Risk-aversion coefficient. Defaults to
            `config.ALPHA_RISK_AVERSION`.

    Returns:
        Dict[ticker, float] weights, non-negative, summing to 1.
    """
    X_sel = X_train[selected].values
    cov = np.cov(X_sel, rowvar=False) + config.RIDGE_EPS * np.eye(len(selected))
    alpha_vec = np.array([float(alpha[t]) for t in selected])
    risk_aversion = config.ALPHA_RISK_AVERSION if risk_aversion is None else risk_aversion

    sector_names = [sectors.get_sector_full_name(t) for t in selected]
    sector_caps = sector_cap_lookup(benchmark_sector_weights)

    if HAS_CVXPY:
        w = cp.Variable(len(selected))
        objective = cp.Maximize(
            alpha_vec @ w - risk_aversion * cp.quad_form(w, cov) - config.WEIGHT_REG * cp.sum_squares(w)
        )
        constraints = [cp.sum(w) == 1, w >= 0, w <= config.MAX_STOCK_WEIGHT]
        for sector in sorted(set(sector_names)):
            idx = [i for i, s in enumerate(sector_names) if s == sector]
            cap = sector_caps.get(sector, 1.0)
            constraints.append(cp.sum(w[idx]) <= cap)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            return enforce_weight_constraints(pd.Series(w.value, index=selected), benchmark_sector_weights)

    x0 = np.full(len(selected), 1.0 / len(selected))

    def objective_fn(w_):
        return float(-(alpha_vec @ w_) + risk_aversion * (w_ @ cov @ w_) + config.WEIGHT_REG * np.sum(w_**2))

    constraints = {"type": "eq", "fun": lambda w_: np.sum(w_) - 1.0}
    bounds = [(0.0, config.MAX_STOCK_WEIGHT)] * len(selected)
    res = minimize(objective_fn, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if res.success:
        return enforce_weight_constraints(pd.Series(res.x, index=selected), benchmark_sector_weights)
    return enforce_weight_constraints(pd.Series(x0, index=selected), benchmark_sector_weights)


def _get_quasi_diag(link):
    num_items = int(link[-1, 3])
    sort_ix = [int(link[-1, 0]), int(link[-1, 1])]
    while any(i >= num_items for i in sort_ix):
        new_sort_ix = []
        for item in sort_ix:
            if item < num_items:
                new_sort_ix.append(int(item))
            else:
                left = int(link[item - num_items, 0])
                right = int(link[item - num_items, 1])
                new_sort_ix.extend([left, right])
        sort_ix = new_sort_ix
    return sort_ix


def _cluster_var(cov, items):
    cov_slice = cov.loc[items, items]
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp = ivp / ivp.sum()
    return float(ivp @ cov_slice.values @ ivp)


def hrp_weights(X_train, selected, benchmark_sector_weights):
    corr = X_train[selected].corr().fillna(0.0)
    cov = pd.DataFrame(
        X_train[selected].cov().fillna(0.0).values + config.RIDGE_EPS * np.eye(len(selected)),
        index=selected,
        columns=selected,
    )
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    # "single" linkage is the canonical choice
    link = linkage(squareform(dist.values, checks=False), method="single")
    sort_ix = corr.index[_get_quasi_diag(link)].tolist()

    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            c1 = clusters[i]
            c2 = clusters[i + 1]
            var1 = _cluster_var(cov, c1)
            var2 = _cluster_var(cov, c2)
            alpha = 1.0 - var1 / (var1 + var2)
            weights[c1] *= alpha
            weights[c2] *= 1.0 - alpha
    return enforce_weight_constraints(weights, benchmark_sector_weights)


def build_weights(X_train, y_train, selected, weighting, benchmark_sector_weights):
    cache_key = (_dataset_cache_key(X_train), weighting, tuple(selected))
    if cache_key in WEIGHT_CACHE:
        return WEIGHT_CACHE[cache_key]
    if weighting == "Markowitz":
        weights = markowitz_weights(X_train, y_train, selected, benchmark_sector_weights)
    elif weighting == "HRP":
        weights = hrp_weights(X_train, selected, benchmark_sector_weights)
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")
    WEIGHT_CACHE[cache_key] = weights
    return weights
