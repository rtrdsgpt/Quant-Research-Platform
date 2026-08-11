"""Plotting: cumulative returns, sparsity-vs-TE sweeps, sector drift, and
the information-ratio and metrics-summary charts. All figures are written
to `paths.output_dir` (see config.yaml).
"""

import numpy as np
import matplotlib.pyplot as plt

from portfolio_replication import config

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

COLORS = {
    "LASSO+Markowitz": "#1f77b4",
    "LASSO+HRP": "#6baed6",
    "Autoencoder_latent+Markowitz": "#ff7f0e",
    "Autoencoder_latent+HRP": "#fdae6b",
    "Autoencoder_communality+Markowitz": "#2ca02c",
    "Autoencoder_communality+HRP": "#74c476",
    "Benchmark": "#d62728",
}


def _output_path(filename):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return config.OUTPUT_DIR / filename


def plot_cumulative_returns(y_train, y_valid, y_hold, results_all, filename="fig1_cumulative_returns.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    panels = [
        ("Train (2020-2024)", y_train, "ret_train"),
        ("Validation (H1 2025)", y_valid, "ret_valid"),
        ("Holdout (H2 2025)", y_hold, "ret_hold"),
    ]

    for ax, (title, benchmark, ret_key) in zip(axes, panels):
        ax.plot(benchmark.index, (1 + benchmark).cumprod(), label="S&P 500", color=COLORS["Benchmark"], lw=2)
        for (method, weighting_scheme), result in results_all.items():
            series = result[ret_key]
            ax.plot(
                series.index,
                (1 + series).cumprod(),
                label=f"{method}+{weighting_scheme}",
                color=COLORS.get(f"{method}+{weighting_scheme}", None),
                lw=1.3,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_ylabel("Cumulative Return")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=1)

    plt.tight_layout()
    path = _output_path(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return path


def plot_sparsity_sweeps(sweeps, filename="fig2_sparsity_vs_te.png"):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, (sw, title) in zip(axes.ravel(), sweeps):
        ax.plot(sw["k"], sw["TE_valid"], "o-", lw=2, color="#1f77b4", label="Rolling Validation TE")
        ax.plot(sw["k"], sw["TE_holdout"], "s--", lw=2, color="#ff7f0e", label="Holdout TE")
        ax.fill_between(
            sw["k"], sw["TE_valid"] - sw["TE_valid_std"], sw["TE_valid"] + sw["TE_valid_std"],
            color="#1f77b4", alpha=0.12,
        )
        ax.set_title(title)
        ax.set_xlabel("Number of Stocks (k)")
        ax.set_ylabel("Annualised Tracking Error (%)")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
        ax.legend()
    plt.tight_layout()
    path = _output_path(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return path


def plot_sector_drift(best_sector_df, best_model_label, filename="fig3_sector_drift.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(best_sector_df))
    width = 0.38
    ax.bar(x - width / 2, best_sector_df["Benchmark Weight"] * 100, width=width, label="Benchmark Universe")
    ax.bar(x + width / 2, best_sector_df["Portfolio Weight"] * 100, width=width, label=best_model_label)
    ax.set_xticks(x)
    ax.set_xticklabels(best_sector_df["Sector"], rotation=45, ha="right")
    ax.set_ylabel("Weight (%)")
    ax.set_title("Sector Drift for Best Validation Model (Mapped Sectors)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = _output_path(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return path


def plot_information_ratio(df_ir, filename="fig4_information_ratio.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_ir))
    ax.bar(x - 0.2, df_ir["Validation IR"], width=0.4, label="Validation IR", color="#4c78a8")
    ax.bar(x + 0.2, df_ir["Holdout IR"], width=0.4, label="Holdout IR", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(df_ir["Model"], rotation=30, ha="right")
    ax.set_ylabel("Information Ratio")
    ax.set_title("Information Ratio Across Portfolio Replication Methods")
    ax.axhline(0, color="black", lw=1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = _output_path(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return path


def plot_metrics_table(display_df, filename="fig5_metrics_table.png"):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.15, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    fig.suptitle("Performance Metrics Summary", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = _output_path(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return path
