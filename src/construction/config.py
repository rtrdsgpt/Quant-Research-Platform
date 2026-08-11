"""Loads the `construction:` section of the unified config.yaml and exposes
it as module-level constants, used by selection.py/weighting.py/sectors.py
(carried over from the original portfolio-replication project).

Two use cases share this config:
  - **Alpha construction** (default): weighting.py's Markowitz-style and
    HRP weighting, applied to the forecast-driven universe from
    src.models.forecaster — only the `weighting`/`market`/`simulation` keys
    are used.
  - **Legacy index replication** (optional, unused by the default
    pipeline): selection.py's LASSO/autoencoder sparse selection and
    evaluation.py's rolling-CV tracking-error evaluation against a
    benchmark index, which need a large external universe of price CSVs
    not included in this repo (see legacy_replication.paths.raw_data_dir).
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_construction = _cfg["construction"]
_legacy = _construction["legacy_replication"]

# Shared weighting params (alpha construction + legacy replication)
RIDGE_EPS = _construction["weighting"]["ridge_eps"]
WEIGHT_REG = _construction["weighting"]["weight_reg"]
MAX_STOCK_WEIGHT = _construction["weighting"]["max_stock_weight"]
SECTOR_CAP_BUFFER = _construction["weighting"]["sector_cap_buffer"]
MIN_SECTOR_WEIGHT_FOR_QUOTA = _construction["weighting"]["min_sector_weight_for_quota"]
SECTOR_QUOTA_RELAX = _construction["weighting"]["sector_quota_relax"]

# Alpha construction (forecast -> weights)
ALPHA_RISK_AVERSION = _construction["alpha_risk_aversion"]

# Market / simulation
TRADING_DAYS = _construction["market"]["trading_days_per_year"]
RANDOM_SEED = _construction["simulation"]["random_seed"]

# Sector map (shared: used by both alpha construction and legacy replication)
SECTOR_MAP_PATH = PROJECT_ROOT / _construction["sector_map_file"]

# --- Legacy large-universe replication mode -------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / _legacy["paths"]["raw_data_dir"]
OUTPUT_DIR = PROJECT_ROOT / _legacy["paths"]["output_dir"]

TRAIN_START = _legacy["dates"]["train_start"]
TRAIN_END = _legacy["dates"]["train_end"]
VALID_START = _legacy["dates"]["valid_start"]
VALID_END = _legacy["dates"]["valid_end"]
HOLDOUT_START = _legacy["dates"]["holdout_start"]
HOLDOUT_END = _legacy["dates"]["holdout_end"]
PREHOLDOUT_END = VALID_END

ROLLING_FOLDS = [tuple(fold) for fold in _legacy["dates"]["rolling_folds"]]

K_DEFAULT = _legacy["selection"]["k_default"]
K_MAX_FINAL = _legacy["selection"]["k_max_final"]
K_OPT_RANGE = range(K_MAX_FINAL, _legacy["selection"]["k_min_opt"], -1)
SWEEP_MIN_K = _legacy["selection"]["sweep_min_k"]
SWEEP_MAX_K = _legacy["selection"]["sweep_max_k"]
SWEEP_STEP = _legacy["selection"]["sweep_step"]
SELECTION_TOL = _legacy["selection"]["selection_tol"]

AE_MAX_LATENT = _legacy["autoencoder"]["max_latent"]
AE_DROPOUT = _legacy["autoencoder"]["dropout"]
AE_EPOCHS = _legacy["autoencoder"]["epochs"]
