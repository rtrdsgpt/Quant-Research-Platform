"""End-to-end index-replication pipeline.

Selects a sparse subset of S&P 500 constituents via LASSO or autoencoder
feature importance, weights them via Markowitz tracking-error minimization
or Hierarchical Risk Parity, picks the sparsity level (k) per model via
rolling cross-validation, and evaluates all six method combinations
(2 selection methods x 3 modes collapsed to 2 selection variants x 2
weighting schemes = 6 models) on train/validation/holdout tracking error.
Plots and a metrics summary are written to results/.

Requires daily price CSVs under `paths.raw_data_dir` (see config.yaml) —
not included in this repository. See README.md for the expected layout.

Usage:
    python main.py
"""

from portfolio_replication import config, data_loader, evaluation, report, sectors, selection, visualization
from portfolio_replication.weighting import build_weights, portfolio_returns

MODELS = [
    ("LASSO", None, "Markowitz"),
    ("LASSO", None, "HRP"),
    ("Autoencoder", "latent", "Markowitz"),
    ("Autoencoder", "latent", "HRP"),
    ("Autoencoder", "communality", "Markowitz"),
    ("Autoencoder", "communality", "HRP"),
]


def run_model_selection(X_prehold, y_prehold, X_train, y_train, X_valid, y_valid, X_hold, y_hold, rolling_folds, benchmark_sector_weights):
    results_all = {}
    for method, mode, weighting_scheme in MODELS:
        print(f"\n--- Rolling validation for {method} {mode or ''} + {weighting_scheme} ---")
        candidates = []
        for k in config.K_OPT_RANGE:
            cv_te_mean, cv_te_std = evaluation.evaluate_on_folds(
                method, mode, weighting_scheme, k, rolling_folds, benchmark_sector_weights
            )
            candidates.append({"k": k, "cv_te_mean": cv_te_mean, "cv_te_std": cv_te_std})

        chosen_k, selection_df = evaluation.choose_sparse_k(candidates)

        if method == "LASSO":
            selected, _ = selection.select_lasso(X_prehold, y_prehold, k=chosen_k, verbose=False)
        else:
            selected, _ = selection.select_autoencoder(X_prehold, y_prehold, k=chosen_k, mode=mode, verbose=False)

        weights = build_weights(X_prehold, y_prehold, selected, weighting_scheme, benchmark_sector_weights)
        ret_prehold = portfolio_returns(X_prehold, weights)
        ret_hold = portfolio_returns(X_hold, weights)
        ret_train = ret_prehold.loc[X_train.index]
        ret_valid = ret_prehold.loc[X_valid.index]

        method_label = f"{method}_{mode}" if mode else method
        cv_te_mean = float(selection_df.loc[selection_df["k"] == chosen_k, "cv_te_mean"].iloc[0])
        cv_te_std = float(selection_df.loc[selection_df["k"] == chosen_k, "cv_te_std"].iloc[0])
        results_all[(method_label, weighting_scheme)] = {
            "k": chosen_k,
            "selected": selected,
            "weights": weights,
            "ret_train": ret_train,
            "ret_valid": ret_valid,
            "ret_hold": ret_hold,
            "cv_te_mean": cv_te_mean,
            "cv_te_std": cv_te_std,
            "metrics_train": evaluation.compute_metrics(ret_train, y_train, f"{method_label}+{weighting_scheme} (Train)"),
            "metrics_valid": evaluation.compute_metrics(ret_valid, y_valid, f"{method_label}+{weighting_scheme} (Validation Diagnostic)"),
            "metrics_hold": evaluation.compute_metrics(ret_hold, y_hold, f"{method_label}+{weighting_scheme} (Holdout)"),
        }
        print(f"  Chosen k={chosen_k} | mean rolling TE={cv_te_mean:.3f}% +/- {cv_te_std:.3f}")

    return results_all


def run_sparsity_sweeps(X_prehold, y_prehold, X_hold, y_hold, rolling_folds, benchmark_sector_weights):
    print("Starting sparsity sweeps (k = 100 down to 10, step 5)...")
    print("Each sweep runs 3 rolling CV folds per k value.\n")

    sweep_args = dict(
        preholdout_returns=X_prehold, preholdout_bench=y_prehold,
        X_hold=X_hold, y_hold=y_hold, folds=rolling_folds,
        benchmark_sector_weights=benchmark_sector_weights,
    )
    sweeps = {
        "LASSO + Markowitz": evaluation.sparsity_sweep(method="LASSO", weighting_scheme="Markowitz", **sweep_args),
        "LASSO + HRP": evaluation.sparsity_sweep(method="LASSO", weighting_scheme="HRP", **sweep_args),
        "Autoencoder Latent + Markowitz": evaluation.sparsity_sweep(method="Autoencoder", mode="latent", weighting_scheme="Markowitz", **sweep_args),
        "Autoencoder Latent + HRP": evaluation.sparsity_sweep(method="Autoencoder", mode="latent", weighting_scheme="HRP", **sweep_args),
        "Autoencoder Communality + Markowitz": evaluation.sparsity_sweep(method="Autoencoder", mode="communality", weighting_scheme="Markowitz", **sweep_args),
        "Autoencoder Communality + HRP": evaluation.sparsity_sweep(method="Autoencoder", mode="communality", weighting_scheme="HRP", **sweep_args),
    }
    print("\nAll sweeps complete.")
    return sweeps


def main():
    # 1. Load data and build splits
    returns_df, gspc_ret = data_loader.load_data()
    X_train, y_train, X_valid, y_valid, X_hold, y_hold = data_loader.split_data(returns_df, gspc_ret)

    prehold_mask = returns_df.index <= config.PREHOLDOUT_END
    X_prehold = returns_df.loc[prehold_mask]
    y_prehold = gspc_ret.loc[prehold_mask]
    rolling_folds = data_loader.build_rolling_folds(X_prehold, y_prehold)

    print(f"\nTrain shape     : {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")
    print(f"Holdout shape   : {X_hold.shape}")
    print(f"Rolling folds   : {len(rolling_folds)}")

    benchmark_sector_weights = sectors.compute_benchmark_sector_weights(returns_df.columns.tolist())

    # 2. Choose k and fit each of the 6 models via rolling validation
    results_all = run_model_selection(
        X_prehold, y_prehold, X_train, y_train, X_valid, y_valid, X_hold, y_hold, rolling_folds, benchmark_sector_weights
    )

    best_model_key = min(results_all, key=lambda key: results_all[key]["cv_te_mean"])
    print(f"\nBest validation model: {best_model_key[0]} + {best_model_key[1]}")
    print(f"Best rolling-val TE  : {results_all[best_model_key]['cv_te_mean']:.3f}%")
    print(f"Holdout TE for best  : {results_all[best_model_key]['metrics_hold']['Ann TE (%)']:.3f}%")

    # 3. Cumulative return plot
    visualization.plot_cumulative_returns(y_train, y_valid, y_hold, results_all)

    # 4. Sparsity sweeps and sector drift
    sweeps = run_sparsity_sweeps(X_prehold, y_prehold, X_hold, y_hold, rolling_folds, benchmark_sector_weights)

    sector_frames = {
        f"{method}+{weighting_scheme}": evaluation.sector_drift(
            result["selected"], result["weights"], benchmark_sector_weights, f"{method}+{weighting_scheme}"
        )
        for (method, weighting_scheme), result in results_all.items()
    }
    best_sector_df = sector_frames[f"{best_model_key[0]}+{best_model_key[1]}"].copy()

    # 5. Report plots
    visualization.plot_sparsity_sweeps(list(sweeps.items()))
    visualization.plot_sector_drift(best_sector_df, f"{best_model_key[0]} + {best_model_key[1]}")

    df_ir = report.build_information_ratio_dataframe(results_all)
    visualization.plot_information_ratio(df_ir)

    # 6. Final metrics summary
    df_metrics = report.build_metrics_dataframe(results_all)
    visualization.plot_metrics_table(report.build_display_metrics(df_metrics))

    print("\nMetrics summary:")
    print(df_metrics.to_string(index=False))
    print("\nBest model constituents:")
    print(results_all[best_model_key]["selected"])

    print(f"\nPlots written to {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
