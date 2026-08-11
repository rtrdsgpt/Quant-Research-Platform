"""Loads daily close prices from data/raw/, converts them to returns, and
builds the train/validation/holdout and rolling-CV splits used by the
pipeline.

Data layout expected under `paths.raw_data_dir` (see config.yaml):
    ^GSPC.csv    - S&P 500 index
    <TICKER>.csv - one file per constituent stock

Splits:
    Train+Test: Jan 2020 - Jun 2025
    Holdout:    Jul 2025 - Dec 2025
"""

import os

import numpy as np
import pandas as pd

from src.construction import config


def _numeric_close_frame(path, series_name):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False).normalize()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close_cols = [c for c in df.columns if "close" in c.lower() or c == "Close"]
    if not close_cols:
        close_cols = [df.columns[-1]]
    s = df[close_cols[0]].copy()
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(series_name)


def load_data(raw_data_dir=None):
    raw_data_dir = raw_data_dir or config.RAW_DATA_DIR
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    gspc = _numeric_close_frame(os.path.join(raw_data_dir, "^GSPC.csv"), "^GSPC").to_frame()

    files = sorted(
        f for f in os.listdir(raw_data_dir)
        if f.endswith(".csv") and f not in {"^GSPC.csv", "sp500_tickers_cache.csv"}
    )
    print(f"  Found {len(files)} constituent files")

    prices = {}
    for f in files:
        ticker = f.replace(".csv", "")
        try:
            prices[ticker] = _numeric_close_frame(os.path.join(raw_data_dir, f), ticker)
        except Exception:
            continue

    prices_df = pd.DataFrame(prices).sort_index()
    common_idx = gspc.index.intersection(prices_df.index)
    prices_df = prices_df.loc[common_idx].sort_index().ffill()
    gspc = gspc.loc[common_idx].sort_index()

    returns_df = prices_df.pct_change().replace([np.inf, -np.inf], np.nan).iloc[1:]
    gspc_ret = gspc["^GSPC"].pct_change().replace([np.inf, -np.inf], np.nan).iloc[1:]

    common_idx = returns_df.index.intersection(gspc_ret.index)
    returns_df = returns_df.loc[common_idx]
    gspc_ret = gspc_ret.loc[common_idx]

    missing_mask = returns_df.isnull().mean() < 0.10
    returns_df = returns_df.loc[:, missing_mask].fillna(0.0)

    print(f"  Benchmark dates    : {gspc_ret.index[0].date()} -> {gspc_ret.index[-1].date()}")
    print(f"  Usable constituents: {returns_df.shape[1]}")
    return returns_df, gspc_ret


def split_data(
    returns_df,
    gspc_ret,
    train_start=config.TRAIN_START,
    train_end=config.TRAIN_END,
    valid_start=config.VALID_START,
    valid_end=config.VALID_END,
    holdout_start=config.HOLDOUT_START,
    holdout_end=config.HOLDOUT_END,
):
    train_mask = (returns_df.index >= train_start) & (returns_df.index <= train_end)
    valid_mask = (returns_df.index >= valid_start) & (returns_df.index <= valid_end)
    hold_mask = (returns_df.index >= holdout_start) & (returns_df.index <= holdout_end)

    X_train = returns_df.loc[train_mask]
    y_train = gspc_ret.loc[train_mask]
    X_valid = returns_df.loc[valid_mask]
    y_valid = gspc_ret.loc[valid_mask]
    X_hold = returns_df.loc[hold_mask]
    y_hold = gspc_ret.loc[hold_mask]
    return X_train, y_train, X_valid, y_valid, X_hold, y_hold


def build_rolling_folds(returns_df, gspc_ret):
    folds = []
    for tr_start, tr_end, va_start, va_end in config.ROLLING_FOLDS:
        train_mask = (returns_df.index >= tr_start) & (returns_df.index <= tr_end)
        valid_mask = (returns_df.index >= va_start) & (returns_df.index <= va_end)
        X_tr = returns_df.loc[train_mask]
        y_tr = gspc_ret.loc[train_mask]
        X_va = returns_df.loc[valid_mask]
        y_va = gspc_ret.loc[valid_mask]
        if len(X_tr) > 0 and len(X_va) > 0:
            folds.append((X_tr, y_tr, X_va, y_va))
    return folds
