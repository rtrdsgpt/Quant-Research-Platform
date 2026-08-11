"""Loads settings from the root-level config.yaml and exposes them as
module-level constants used throughout the pipeline.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Paths
DATA_DIR = PROJECT_ROOT / _cfg["paths"]["data_dir"]
RAW_DATA_DIR = PROJECT_ROOT / _cfg["paths"]["raw_data_dir"]
OUTPUT_DIR = PROJECT_ROOT / _cfg["paths"]["output_dir"]

SECTOR_MAP_PATH = DATA_DIR / _cfg["data"]["sector_map_file"]

# Date splits
TRAIN_START = _cfg["dates"]["train_start"]
TRAIN_END = _cfg["dates"]["train_end"]
VALID_START = _cfg["dates"]["valid_start"]
VALID_END = _cfg["dates"]["valid_end"]
HOLDOUT_START = _cfg["dates"]["holdout_start"]
HOLDOUT_END = _cfg["dates"]["holdout_end"]
PREHOLDOUT_END = VALID_END

ROLLING_FOLDS = [tuple(fold) for fold in _cfg["dates"]["rolling_folds"]]

# Stock selection
K_DEFAULT = _cfg["selection"]["k_default"]
K_MAX_FINAL = _cfg["selection"]["k_max_final"]
K_OPT_RANGE = range(K_MAX_FINAL, _cfg["selection"]["k_min_opt"], -1)
SWEEP_MIN_K = _cfg["selection"]["sweep_min_k"]
SWEEP_MAX_K = _cfg["selection"]["sweep_max_k"]
SWEEP_STEP = _cfg["selection"]["sweep_step"]
SELECTION_TOL = _cfg["selection"]["selection_tol"]

# Portfolio weighting
RIDGE_EPS = _cfg["weighting"]["ridge_eps"]
WEIGHT_REG = _cfg["weighting"]["weight_reg"]
MAX_STOCK_WEIGHT = _cfg["weighting"]["max_stock_weight"]
SECTOR_CAP_BUFFER = _cfg["weighting"]["sector_cap_buffer"]
MIN_SECTOR_WEIGHT_FOR_QUOTA = _cfg["weighting"]["min_sector_weight_for_quota"]
SECTOR_QUOTA_RELAX = _cfg["weighting"]["sector_quota_relax"]

# Autoencoder
AE_MAX_LATENT = _cfg["autoencoder"]["max_latent"]
AE_DROPOUT = _cfg["autoencoder"]["dropout"]
AE_EPOCHS = _cfg["autoencoder"]["epochs"]

# Market assumptions
TRADING_DAYS = _cfg["market"]["trading_days_per_year"]

# Simulation
RANDOM_SEED = _cfg["simulation"]["random_seed"]
