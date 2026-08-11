"""Assembles the final metrics summary and information-ratio tables from
the per-model results produced by the pipeline.
"""

import numpy as np
import pandas as pd


def build_metrics_dataframe(results_all):
    rows = []
    for (method, weighting_scheme), result in results_all.items():
        for split_name, metrics_key in [("Train", "metrics_train"), ("Validation", "metrics_valid"), ("Holdout", "metrics_hold")]:
            row = result[metrics_key].copy()
            row["Method"] = method
            row["Weighting"] = weighting_scheme
            row["Split"] = split_name
            row["Selected k"] = result["k"]
            rows.append(row)

    df_metrics = pd.DataFrame(rows)[
        ["Method", "Weighting", "Split", "Selected k", "Ann TE (%)", "Ann IR", "Corr", "R²", "Ann Active(%)"]
    ].sort_values(["Method", "Weighting", "Split"])
    return df_metrics


def build_display_metrics(df_metrics):
    display_df = df_metrics.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    return display_df


def build_information_ratio_dataframe(results_all):
    ir_rows = []
    for (method, weighting_scheme), result in results_all.items():
        ir_rows.append(
            {
                "Model": f"{method}+{weighting_scheme}",
                "Rolling Val TE": result["cv_te_mean"],
                "Holdout IR": result["metrics_hold"]["Ann IR"],
                "Validation IR": result["metrics_valid"]["Ann IR"],
            }
        )
    return pd.DataFrame(ir_rows).sort_values("Holdout IR", ascending=False)
