from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "Hottest 10%": "#E69F00",
    "Wettest 10%": "#56B4E9",
}


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def _pick_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in norm:
            return norm[cand.lower().strip()]
    raise ValueError(f"Missing '{label}'. Tried {candidates}. Available columns: {list(df.columns)}")


def _norm_dataset(v: str) -> str:
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    if "tancheon" in s:
        return "P1"
    if "seoul3" in s or s == "seoul_3":
        return "P2"
    return str(v).strip()


def _norm_model(v: str) -> str:
    s = str(v).strip().lower()
    if "lstm" in s:
        return "CS-LSTM AE"
    if "copula" in s or "gaussian" in s:
        return "Gaussian Copula"
    return str(v).strip()


def _norm_climate(v: str) -> str:
    s = str(v).strip().lower()
    if "normal" in s:
        return "Normal"
    if "hottest" in s or "hot" in s:
        return "Hottest 10%"
    if "wettest" in s or "wet" in s:
        return "Wettest 10%"
    return str(v).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_q = _pick_column(df, ["threshold_quantile", "q", "threshold", "quantile"], "threshold quantile")
    col_dataset = _pick_column(df, ["dataset", "dataset_name", "plant", "site", "wwtp"], "dataset")
    col_model = _pick_column(df, ["model", "model_name", "detector"], "model")
    col_climate = _pick_column(df, ["ClimateGroup", "climate_group", "climate", "group"], "climate_group")
    col_rate = _pick_column(df, ["anomaly_rate_percent", "anomaly_rate", "rate"], "anomaly_rate")
    col_lo = _pick_column(df, ["ci95_low_percent", "ci95_low", "ci_low", "lower_ci"], "ci_low")
    col_hi = _pick_column(df, ["ci95_high_percent", "ci95_high", "ci_high", "upper_ci"], "ci_high")
    col_n = _pick_column(df, ["n_days", "n", "sample_size", "n_samples"], "n_days")

    out = pd.DataFrame(
        {
            "threshold_q": pd.to_numeric(df[col_q], errors="coerce"),
            "dataset": df[col_dataset].map(_norm_dataset),
            "model": df[col_model].map(_norm_model),
            "climate_group": df[col_climate].map(_norm_climate),
            "anomaly_rate": pd.to_numeric(df[col_rate], errors="coerce"),
            "ci_low": pd.to_numeric(df[col_lo], errors="coerce"),
            "ci_high": pd.to_numeric(df[col_hi], errors="coerce"),
            "n_days": pd.to_numeric(df[col_n], errors="coerce").fillna(0).astype(int),
        }
    )

    vmax = np.nanmax(out[["anomaly_rate", "ci_low", "ci_high"]].to_numpy())
    if vmax <= 1.5:
        out[["anomaly_rate", "ci_low", "ci_high"]] = out[["anomaly_rate", "ci_low", "ci_high"]] * 100.0

    out = out.dropna(subset=["threshold_q", "dataset", "model", "climate_group"]).copy()
    return out


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = sorted(df["threshold_q"].dropna().unique().tolist())
    if len(thresholds) < 3:
        print(f"Warning: only {len(thresholds)} thresholds found: {thresholds}")

    recs = []
    required_groups = {"Normal", "Hottest 10%", "Wettest 10%"}

    for dataset in ["P1", "P2"]:
        for model in ["CS-LSTM AE", "Gaussian Copula"]:
            for q in thresholds:
                sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (np.isclose(df["threshold_q"], q))]
                groups_present = set(sub["climate_group"].unique().tolist())
                if not required_groups.issubset(groups_present):
                    missing = sorted(required_groups - groups_present)
                    raise ValueError(
                        f"Missing climate groups for dataset={dataset}, model={model}, q={q}: {missing}"
                    )

                sub = sub.set_index("climate_group")
                normal = sub.loc["Normal"]
                hot = sub.loc["Hottest 10%"]
                wet = sub.loc["Wettest 10%"]

                recs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "threshold_q": float(q),
                        "group": "Hottest 10%",
                        "delta": float(hot["anomaly_rate"] - normal["anomaly_rate"]),
                        "delta_low": float(hot["ci_low"] - normal["ci_high"]),
                        "delta_high": float(hot["ci_high"] - normal["ci_low"]),
                        "n_group": int(hot["n_days"]),
                        "normal_rate": float(normal["anomaly_rate"]),
                        "n_normal": int(normal["n_days"]),
                    }
                )
                recs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "threshold_q": float(q),
                        "group": "Wettest 10%",
                        "delta": float(wet["anomaly_rate"] - normal["anomaly_rate"]),
                        "delta_low": float(wet["ci_low"] - normal["ci_high"]),
                        "delta_high": float(wet["ci_high"] - normal["ci_low"]),
                        "n_group": int(wet["n_days"]),
                        "normal_rate": float(normal["anomaly_rate"]),
                        "n_normal": int(normal["n_days"]),
                    }
                )

    delta_df = pd.DataFrame(recs)
    return delta_df


def plot_fig4(delta_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    datasets = ["P1", "P2"]
    models = ["CS-LSTM AE", "Gaussian Copula"]
    groups = ["Hottest 10%", "Wettest 10%"]
    thresholds = sorted(delta_df["threshold_q"].unique().tolist())

    if len(thresholds) < 1:
        raise ValueError("No thresholds available to plot.")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=True)

    # shared y-limits by model column
    y_lims: Dict[str, tuple] = {}
    for model in models:
        sub = delta_df[delta_df["model"] == model]
        y_min = float(np.nanmin(sub["delta_low"]))
        y_max = float(np.nanmax(sub["delta_high"]))
        span = max(4.0, y_max - y_min)
        y_lims[model] = (y_min - 0.12 * span, y_max + 0.18 * span)

    for r, dataset in enumerate(datasets):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = delta_df[(delta_df["dataset"] == dataset) & (delta_df["model"] == model)].copy()

            for grp in groups:
                ss = sub[sub["group"] == grp].sort_values("threshold_q")
                x = ss["threshold_q"].to_numpy(dtype=float)
                y = ss["delta"].to_numpy(dtype=float)
                y_low = ss["delta_low"].to_numpy(dtype=float)
                y_high = ss["delta_high"].to_numpy(dtype=float)

                ax.plot(x, y, marker="o", lw=2, color=PALETTE[grp], label=grp)
                yerr = np.vstack([y - y_low, y_high - y])
                ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=PALETTE[grp], capsize=3, lw=1.2)

                # rightmost point annotation
                xr = x[-1]
                yr = y[-1]
                ax.annotate(
                    f"{yr:+.0f} pp",
                    xy=(xr, yr),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=PALETTE[grp],
                    fontsize=9,
                    va="center",
                )

            ax.axhline(0, color="#666666", lw=1.0)
            ax.grid(axis="y", alpha=0.2)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_xticks(thresholds)
            ax.set_xticklabels([f"{q:g}" for q in thresholds])
            ax.set_ylim(*y_lims[model])

            if r == 1:
                ax.set_xlabel("Threshold quantile")
            else:
                ax.set_xlabel("")

            if c == 0:
                ax.set_ylabel("Δ vs Normal (pp)")
            else:
                ax.set_ylabel("")

            ax.set_title(f"{dataset} | {model}")

    fig.suptitle("Figure 4 — Threshold-sensitivity robustness of climate stress amplification", fontsize=15)
    fig.text(0.5, 0.955, "Robustness: stress amplification remains elevated across thresholds", ha="center", fontsize=10)

    handles = [
        mlines.Line2D([], [], color=PALETTE["Hottest 10%"], marker="o", lw=2, label="Hottest 10%"),
        mlines.Line2D([], [], color=PALETTE["Wettest 10%"], marker="o", lw=2, label="Wettest 10%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 4 threshold-sensitivity delta robustness plot.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/two_real_datasets/comparison_climate_rates_all_thresholds.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("figures/fig4_threshold_sensitivity_delta.png"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figures/fig4_threshold_sensitivity_delta.pdf"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/two_real_datasets/fig4_threshold_sensitivity_delta_data.csv"),
    )
    args = parser.parse_args()

    raw = load_results(args.input)
    norm = normalize_columns(raw)
    delta_df = compute_deltas(norm)
    plot_fig4(delta_df, args.out_png, args.out_pdf)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    delta_df.to_csv(args.out_csv, index=False)

    thresholds = sorted(delta_df["threshold_q"].unique().tolist())
    print(f"Thresholds used: {thresholds}")
    print("Saved:")
    print(args.out_png)
    print(args.out_pdf)
    print(args.out_csv)


if __name__ == "__main__":
    main()
