"""Ticker -> GICS-style sector mapping, loaded from data/sector_map.csv.

The map is only used to aggregate the provided constituent universe by
sector. The dataset does not include official GICS weights or market caps,
so sector drift is benchmarked against the mapped constituent universe
rather than live index weights.
"""

import pandas as pd

from src.construction import config

UNKNOWN_SECTOR = "Unknown"
UNKNOWN_SECTOR_FULL = "Unknown / Unmapped"

_sector_df = pd.read_csv(config.SECTOR_MAP_PATH, dtype=str)
SECTOR_MAP = dict(zip(_sector_df["ticker"], _sector_df["sector_code"]))
SECTOR_FULL = dict(zip(_sector_df["sector_code"], _sector_df["sector_name"]))
SECTOR_FULL[UNKNOWN_SECTOR] = UNKNOWN_SECTOR_FULL


def canonical_ticker(ticker):
    return ticker.replace(".", "-")


def get_sector_label(ticker):
    return SECTOR_MAP.get(canonical_ticker(ticker), UNKNOWN_SECTOR)


def get_sector_full_name(ticker):
    return SECTOR_FULL.get(get_sector_label(ticker), UNKNOWN_SECTOR_FULL)


def compute_benchmark_sector_weights(universe_tickers):
    universe = pd.Index(sorted(set(map(canonical_ticker, universe_tickers))))
    sectors = pd.Series([get_sector_label(t) for t in universe], index=universe, name="Sector")
    sectors = sectors[sectors != UNKNOWN_SECTOR]
    sector_weights = sectors.value_counts(normalize=True).sort_index()
    sector_weights.index = [SECTOR_FULL.get(s, s) for s in sector_weights.index]
    return sector_weights.sort_values(ascending=False)
