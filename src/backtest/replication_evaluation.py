"""Performance metrics, rolling-fold cross-validation, sparsity (k) sweeps,
and sector-drift computation.
"""

import numpy as np
import pandas as pd

from src.construction import config, sectors, selection, weighting


def annual_tracking_error(port_ret, bench_ret):
    active = port_ret - bench_ret
    return active.std(ddof=1) * np.sqrt(config.TRADING_DAYS) * 100


def compute_metrics(port_ret, bench_ret, label):
    active = (port_ret - bench_ret).dropna()
    aligned_bench = bench_ret.loc[active.index]
    ann_te = active.std(ddof=1) * np.sqrt(config.TRADING_DAYS) * 100
    ann_active = active.mean() * config.TRADING_DAYS * 100
    ann_ir = ann_active / ann_te if ann_te > 1e-12 else np.nan
    corr = port_ret.loc[aligned_bench.index].corr(aligned_bench)
    r2 = 1.0 - ((active**2).sum() / ((aligned_bench - aligned_bench.mean()) ** 2).sum())
    return {
        "Label": label,
        "Ann TE (%)": ann_te,
        "Ann IR": ann_ir,
        "Corr": corr,
        "R²": r2,
        "Ann Active(%)": ann_active,
    }


def evaluate_on_folds(method, mode, weighting_scheme, k, folds, benchmark_sector_weights):
    te_scores = []
    for X_tr, y_tr, X_va, y_va in folds:
        if method == "LASSO":
            selected, _ = selection.select_lasso(X_tr, y_tr, k=k, verbose=False)
        else:
            selected, _ = selection.select_autoencoder(X_tr, y_tr, k=k, mode=mode, verbose=False)
        weights = weighting.build_weights(X_tr, y_tr, selected, weighting_scheme, benchmark_sector_weights)
        ret_valid = weighting.portfolio_returns(X_va, weights)
        te_scores.append(annual_tracking_error(ret_valid, y_va))
    return float(np.mean(te_scores)), float(np.std(te_scores))


def choose_sparse_k(candidate_rows):
    df = pd.DataFrame(candidate_rows).sort_values("k")
    best_te = df["cv_te_mean"].min()
    eligible = df[df["cv_te_mean"] <= best_te + config.SELECTION_TOL]
    chosen = eligible.sort_values(["k", "cv_te_std"]).iloc[0]
    return int(chosen["k"]), df


def sparsity_sweep(
    preholdout_returns,
    preholdout_bench,
    X_hold,
    y_hold,
    folds,
    method,
    weighting_scheme,
    benchmark_sector_weights,
    mode=None,
    k_values=None,
):
    if k_values is None:
        max_k = min(config.SWEEP_MAX_K, preholdout_returns.shape[1])
        k_values = list(range(max_k, config.SWEEP_MIN_K - 1, -config.SWEEP_STEP))

    print(f"  Sweep: {method} {mode or ''} + {weighting_scheme} | k = {k_values[0]}..{k_values[-1]}")
    rows = []
    for k in k_values:
        cv_te_mean, cv_te_std = evaluate_on_folds(method, mode, weighting_scheme, k, folds, benchmark_sector_weights)
        if method == "LASSO":
            selected, _ = selection.select_lasso(preholdout_returns, preholdout_bench, k=k, verbose=False)
        else:
            selected, _ = selection.select_autoencoder(preholdout_returns, preholdout_bench, k=k, mode=mode, verbose=False)
        weights = weighting.build_weights(preholdout_returns, preholdout_bench, selected, weighting_scheme, benchmark_sector_weights)
        ret_hold = weighting.portfolio_returns(X_hold, weights)
        te_holdout = annual_tracking_error(ret_hold, y_hold)
        print(f"    k={k:3d}  CV TE={cv_te_mean:.3f}% ± {cv_te_std:.3f}  Holdout TE={te_holdout:.3f}%")
        rows.append(
            {
                "k": k,
                "TE_valid": cv_te_mean,
                "TE_valid_std": cv_te_std,
                "TE_holdout": te_holdout,
            }
        )
    return pd.DataFrame(rows).sort_values("k", ascending=False).reset_index(drop=True)


def sector_drift(selected, weights, benchmark_sector_weights, label):
    port_weights = pd.Series(weights, dtype=float)
    sector_codes = pd.Series({ticker: sectors.get_sector_label(ticker) for ticker in port_weights.index})
    mapped = sector_codes[sector_codes != sectors.UNKNOWN_SECTOR].map(sectors.SECTOR_FULL)
    port_weights = port_weights.loc[mapped.index]
    portfolio_sector_weights = port_weights.groupby(mapped).sum()
    portfolio_sector_weights = portfolio_sector_weights / portfolio_sector_weights.sum()
    all_sectors = sorted(set(benchmark_sector_weights.index).union(portfolio_sector_weights.index))
    benchmark = benchmark_sector_weights.reindex(all_sectors, fill_value=0.0)
    portfolio = portfolio_sector_weights.reindex(all_sectors, fill_value=0.0)
    return pd.DataFrame(
        {
            "Sector": all_sectors,
            "Benchmark Weight": benchmark.values,
            "Portfolio Weight": portfolio.values,
            "Drift": (portfolio - benchmark).values,
            "Label": label,
        }
    )
