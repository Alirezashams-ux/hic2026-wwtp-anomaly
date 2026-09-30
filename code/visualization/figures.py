from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_climate_conditioned_rates(
    cs_rates: pd.DataFrame,
    gc_rates: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = ["Normal", "Hottest 10%", "Wettest 10%"]
    cs_series = cs_rates.set_index("ClimateGroup").reindex(order)["anomaly_rate_percent"].fillna(0.0)
    gc_series = gc_rates.set_index("ClimateGroup").reindex(order)["anomaly_rate_percent"].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Figure 1. Climate-conditioned anomaly rates", fontsize=12)

    axes[0].bar(order, cs_series.values)
    axes[0].set_title("(a) CS-LSTM AE")
    axes[0].set_ylabel("Anomaly rate (%)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(order, gc_series.values)
    axes[1].set_title("(b) Gaussian Copula")
    axes[1].tick_params(axis="x", rotation=20)

    for ax, series in zip(axes, [cs_series, gc_series]):
        for idx, value in enumerate(series.values):
            ax.text(idx, value + 0.1, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
