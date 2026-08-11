"""Stock selection methods: LASSO coefficient ranking and autoencoder
feature-importance ranking (latent-correlation or reconstruction-communality
based), each optionally rebalanced to match benchmark sector weights.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from portfolio_replication import config, sectors

try:
    import tensorflow as tf
    from tensorflow import keras

    HAS_TF = True
except ImportError:
    HAS_TF = False

SELECTION_CACHE = {}


def _dataset_cache_key(X):
    return (str(X.index[0].date()), str(X.index[-1].date()), X.shape[0], X.shape[1])


def select_stocks(X_train, y_train, method, mode, k, verbose=False):
    key = (_dataset_cache_key(X_train), method, mode or "none", k)
    if key in SELECTION_CACHE:
        return SELECTION_CACHE[key]
    if method == "LASSO":
        selected, _ = select_lasso(X_train, y_train, k=k, verbose=verbose)
    else:
        selected, _ = select_autoencoder(X_train, y_train, k=k, mode=mode, verbose=verbose)
    SELECTION_CACHE[key] = selected
    return selected


def sector_balanced_subset(columns, importance, k, benchmark_sector_weights):
    columns = list(columns)
    importance = pd.Series(np.asarray(importance, dtype=float), index=columns)
    ranked = importance.sort_values(ascending=False)

    if benchmark_sector_weights is None:
        return ranked.head(k).index.tolist()

    sector_map = pd.Series({c: sectors.get_sector_full_name(c) for c in columns})
    mapped = sector_map[sector_map != sectors.UNKNOWN_SECTOR_FULL]
    if mapped.empty:
        return ranked.head(k).index.tolist()

    sector_targets = (benchmark_sector_weights * k).round().astype(int)
    active_sectors = benchmark_sector_weights[
        benchmark_sector_weights >= config.MIN_SECTOR_WEIGHT_FOR_QUOTA
    ].index.tolist()
    for sector in active_sectors:
        if sector in mapped.values and sector_targets.get(sector, 0) == 0:
            sector_targets.loc[sector] = 1

    available = mapped.value_counts()
    sector_targets = sector_targets.reindex(available.index, fill_value=0)
    sector_targets = pd.concat([sector_targets, available], axis=1)
    sector_targets.columns = ["target", "available"]
    sector_targets["target"] = np.minimum(sector_targets["target"], sector_targets["available"])

    current = int(sector_targets["target"].sum())
    if current > k:
        for sector in sector_targets.sort_values("target", ascending=False).index:
            while sector_targets.loc[sector, "target"] > 0 and current > k:
                sector_targets.loc[sector, "target"] -= 1
                current -= 1
    elif current < k:
        deficits = (
            benchmark_sector_weights.reindex(sector_targets.index).fillna(0).sort_values(ascending=False).index.tolist()
        )
        changed = True
        while current < k and changed:
            changed = False
            for sector in deficits:
                if current >= k:
                    break
                if sector_targets.loc[sector, "target"] < sector_targets.loc[sector, "available"]:
                    sector_targets.loc[sector, "target"] += 1
                    current += 1
                    changed = True

    selected = []
    for sector, row in sector_targets.iterrows():
        n_take = int(row["target"])
        if n_take <= 0:
            continue
        sector_names = ranked.loc[mapped[mapped == sector].index].head(n_take).index.tolist()
        selected.extend(sector_names)

    selected = list(dict.fromkeys(selected))
    if len(selected) < k:
        remaining = [name for name in ranked.index.tolist() if name not in selected]
        selected.extend(remaining[: k - len(selected)])

    return selected[:k]


def select_lasso(X_train, y_train, k=config.K_DEFAULT, verbose=True):
    if verbose:
        print(f"\n[LASSO] Selecting top-{k} stocks")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    # TimeSeriesSplit preserves temporal order in cross-validation. Standard
    # KFold shuffles data randomly, which breaks the time-series structure
    # and allows future data to leak into training folds.
    tscv = TimeSeriesSplit(n_splits=5)
    lcv = LassoCV(cv=tscv, max_iter=5000, random_state=config.RANDOM_SEED, n_jobs=-1)
    lcv.fit(X_sc, y_train.values)
    coef_abs = np.abs(lcv.coef_)

    # Stronger regularisation: do not relax alpha aggressively.
    if np.sum(coef_abs > 0) < k:
        for alpha in np.linspace(max(lcv.alpha_ * 0.35, 1e-5), lcv.alpha_, 15):
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_sc, y_train.values)
            if np.sum(np.abs(model.coef_) > 0) >= k:
                coef_abs = np.abs(model.coef_)
                break

    top_idx = np.argsort(coef_abs)[-k:]
    selected = X_train.columns[top_idx].tolist()
    return selected, coef_abs


def select_autoencoder(X_train, y_train, k=config.K_DEFAULT, mode="latent", verbose=True):
    if verbose:
        print(f"\n[AUTOENCODER {mode}] Selecting top-{k} stocks")

    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)
    mu = X_np.mean(axis=0, keepdims=True)
    std = X_np.std(axis=0, keepdims=True) + 1e-8
    X_sc = (X_np - mu) / std

    if HAS_TF:
        tf.random.set_seed(config.RANDOM_SEED)
        n_features = X_sc.shape[1]
        latent_dim = max(6, min(config.AE_MAX_LATENT, n_features // 10))

        inp = keras.Input(shape=(n_features,))
        enc = keras.layers.Dense(96, activation="relu")(inp)
        enc = keras.layers.Dropout(config.AE_DROPOUT)(enc)
        enc = keras.layers.Dense(32, activation="relu")(enc)
        lat = keras.layers.Dense(latent_dim, activation="linear", name="latent")(enc)
        dec = keras.layers.Dense(32, activation="relu")(lat)
        dec = keras.layers.Dense(96, activation="relu")(dec)
        out = keras.layers.Dense(n_features, activation="linear")(dec)

        ae = keras.Model(inp, out)
        ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        # Keras validation_split takes the LAST fraction of the sequence,
        # which preserves temporal order for time-series data.
        X_noisy = X_sc + 0.04 * np.random.randn(*X_sc.shape).astype(np.float32)
        ae.fit(
            X_noisy,
            X_sc,
            epochs=config.AE_EPOCHS,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )

        if mode == "latent":
            encoder = keras.Model(inp, lat)
            Z = encoder.predict(X_sc, verbose=0)
            corrs = np.array([
                0.0 if np.std(Z[:, j]) < 1e-12 else np.corrcoef(Z[:, j], y_np)[0, 1]
                for j in range(Z.shape[1])
            ])
            best_dim = int(np.nanargmax(np.abs(np.nan_to_num(corrs, nan=0.0))))
            w1 = np.abs(ae.layers[1].get_weights()[0])  # Input -> Dense(96): (n_features, 96)
            w2 = np.abs(ae.layers[3].get_weights()[0])  # Dense(96) -> Dense(32): (96, 32)
            w3 = np.abs(ae.get_layer("latent").get_weights()[0][:, best_dim])  # Dense(32) -> latent dim: (32,)
            importance = w1 @ (w2 @ w3)  # (n_features,) path-importance to best latent dim
        else:
            X_recon = ae.predict(X_sc, verbose=0)
            mse_per_stock = np.mean((X_sc - X_recon) ** 2, axis=0)
            importance = 1.0 - mse_per_stock
    else:
        from sklearn.decomposition import PCA

        n_components = max(5, min(12, X_sc.shape[1] // 12))
        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        Z = pca.fit_transform(X_sc)
        if mode == "latent":
            corrs = np.array([
                0.0 if np.std(Z[:, j]) < 1e-12 else np.corrcoef(Z[:, j], y_np)[0, 1]
                for j in range(Z.shape[1])
            ])
            best_dim = int(np.nanargmax(np.abs(np.nan_to_num(corrs, nan=0.0))))
            importance = np.abs(pca.components_[best_dim])
        else:
            X_recon = pca.inverse_transform(Z)
            mse_per_stock = np.mean((X_sc - X_recon) ** 2, axis=0)
            importance = 1.0 - mse_per_stock

    top_idx = np.argsort(np.asarray(importance))[-k:]
    selected = X_train.columns[top_idx].tolist()
    return selected, importance
