from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "Normal": "#7F7F7F",
    "Hottest 10%": "#E69F00",
    "Wettest 10%": "#56B4E9",
}


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    return pd.read_csv(path)


def _pick_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    available = ", ".join(df.columns)
    expected = ", ".join(candidates)
    raise ValueError(
        f"Missing required field for '{label}'. Tried columns: [{expected}]. Available columns: [{available}]"
    )


def _normalize_dataset(name: str) -> str:
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if "tancheon" in s:
        return "Seoul_Tancheon_WWTP"
    if "seoul3" in s or s == "seoul_3":
        return "Seoul3"
    return str(name).strip()


def _normalize_model(name: str) -> str:
    s = str(name).strip().lower()
    if "lstm" in s:
        return "CS-LSTM AE"
    if "copula" in s or s == "gc" or "gaussian" in s:
        return "Gaussian Copula"
    return str(name).strip()


def _normalize_climate(name: str) -> str:
    s = str(name).strip().lower().replace("%", "")
    if "normal" in s:
        return "Normal"
    if "hottest" in s or "hot" in s:
        return "Hottest 10%"
    if "wettest" in s or "wet" in s:
        return "Wettest 10%"
    return str(name).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    threshold_col = _pick_column(
        df,
        candidates=["threshold_quantile", "q", "threshold", "quantile"],
        label="threshold quantile",
    )
    dataset_col = _pick_column(df, ["dataset", "dataset_name", "site", "plant", "wwtp"], "dataset")
    model_col = _pick_column(df, ["model", "model_name", "detector"], "model")
    climate_col = _pick_column(df, ["ClimateGroup", "climate_group", "climate", "group"], "climate_group")
    rate_col = _pick_column(
        df,
        ["anomaly_rate_percent", "anomaly_rate", "rate", "anomaly_rate_pct"],
        "anomaly_rate",
    )
    ci_low_col = _pick_column(df, ["ci95_low_percent", "ci_low", "ci95_low", "lower_ci"], "ci_low")
    ci_high_col = _pick_column(df, ["ci95_high_percent", "ci_high", "ci95_high", "upper_ci"], "ci_high")
    n_days_col = _pick_column(df, ["n_days", "n", "sample_size", "n_samples"], "n_days")

    out = pd.DataFrame(
        {
            "threshold_quantile": pd.to_numeric(df[threshold_col], errors="coerce"),
            "dataset": df[dataset_col].map(_normalize_dataset),
            "model": df[model_col].map(_normalize_model),
            "climate_group": df[climate_col].map(_normalize_climate),
            "anomaly_rate": pd.to_numeric(df[rate_col], errors="coerce"),
            "ci_low": pd.to_numeric(df[ci_low_col], errors="coerce"),
            "ci_high": pd.to_numeric(df[ci_high_col], errors="coerce"),
            "n_days": pd.to_numeric(df[n_days_col], errors="coerce"),
        }
    )

    out = out[np.isclose(out["threshold_quantile"], 0.99, atol=1e-9)].copy()
    if out.empty:
        raise ValueError("No rows found for threshold quantile == 0.99 (or q == 0.99).")

    max_rate = np.nanmax(out[["anomaly_rate", "ci_low", "ci_high"]].to_numpy())
    if max_rate <= 1.5:
        out[["anomaly_rate", "ci_low", "ci_high"]] = out[["anomaly_rate", "ci_low", "ci_high"]] * 100.0

    out["n_days"] = out["n_days"].fillna(0).astype(int)
    return out


def _assert_complete(df: pd.DataFrame) -> None:
    datasets_expected = ["Seoul_Tancheon_WWTP", "Seoul3"]
    models_expected = ["CS-LSTM AE", "Gaussian Copula"]
    climates_expected = ["Normal", "Hottest 10%", "Wettest 10%"]

    observed = set(zip(df["dataset"], df["model"], df["climate_group"]))
    expected = {(d, m, c) for d in datasets_expected for m in models_expected for c in climates_expected}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing or extra:
        msg = ["Data completeness check failed after filtering q=0.99."]
        if missing:
            msg.append(f"Missing combinations: {missing}")
        if extra:
            msg.append(f"Unexpected combinations: {extra}")
        raise ValueError(" ".join(msg))

    if len(df) != 12:
        raise ValueError(f"Expected exactly 12 rows (2×2×3), got {len(df)}. Check duplicates or missing rows.")


def plot_fig2(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    _assert_complete(df)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    datasets = ["Seoul_Tancheon_WWTP", "Seoul3"]
    models = ["CS-LSTM AE", "Gaussian Copula"]
    climates = ["Normal", "Hottest 10%", "Wettest 10%"]
    y_positions = np.array([2, 1, 0], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)

    col_limits: Dict[str, tuple] = {}
    for model_name in models:
        sub = df[df["model"] == model_name]
        c_min = float(np.nanmin(sub["ci_low"]))
        c_max = float(np.nanmax(sub["ci_high"]))
        span = max(5.0, c_max - c_min)
        left = max(0.0, c_min - 0.08 * span)
        right = c_max + 0.45 * span
        col_limits[model_name] = (left, right)

    for r, dataset_name in enumerate(datasets):
        for c, model_name in enumerate(models):
            ax = axes[r, c]
            sub = (
                df[(df["dataset"] == dataset_name) & (df["model"] == model_name)]
                .set_index("climate_group")
                .reindex(climates)
                .reset_index()
            )

            x = sub["anomaly_rate"].to_numpy(dtype=float)
            lo = sub["ci_low"].to_numpy(dtype=float)
            hi = sub["ci_high"].to_numpy(dtype=float)
            n_days = sub["n_days"].to_numpy(dtype=int)

            for idx, climate in enumerate(climates):
                xi = x[idx]
                l = lo[idx]
                h = hi[idx]
                y = y_positions[idx]
                color = OKABE_ITO[climate]

                ax.errorbar(
                    xi,
                    y,
                    xerr=np.array([[xi - l], [h - xi]]),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=3,
                    lw=1.5,
                    ms=6,
                    zorder=3,
                )

                x_offset = 0.6
                x_text = h + x_offset
                x_left, x_right = col_limits[model_name]
                if x_text > x_right - 0.5:
                    x_text = max(x_left + 0.5, l - (1.8 + x_offset))
                    ha = "right"
                else:
                    ha = "left"
                ax.text(x_text, y, f"{xi:.1f}% (n={int(n_days[idx])})", va="center", ha=ha, fontsize=9)

            normal_rate = float(sub.loc[sub["climate_group"] == "Normal", "anomaly_rate"].iloc[0])
            ax.axvline(normal_rate, color="#666666", lw=1.0, ls="--", alpha=0.8, zorder=1)

            ax.set_ylim(-0.6, 2.6)
            ax.set_xlim(*col_limits[model_name])
            ax.grid(axis="x", alpha=0.25)
            ax.grid(axis="y", visible=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if c == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(climates)
            else:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([])

            if r == 1:
                ax.set_xlabel("Anomaly rate (%)")
            else:
                ax.set_xlabel("")

            ds_short = "Seoul_Tancheon_WWTP" if dataset_name == "Seoul_Tancheon_WWTP" else "Seoul3"
            ax.set_title(f"{ds_short} | {model_name}")

    fig.suptitle("Climate-conditioned anomaly rates (q=0.99, rolling-origin; 95% CI)", fontsize=15)

    legend_handles = [
        mlines.Line2D([], [], color=OKABE_ITO[g], marker="o", linestyle="None", markersize=6, label=g)
        for g in climates
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 2 forest/lollipop plot for q=0.99 climate-conditioned anomaly rates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/two_real_datasets/comparison_climate_rates_all_thresholds.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("figures/fig2_climate_rates_q99_forest.png"),
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("figures/fig2_climate_rates_q99_forest.pdf"),
    )
    args = parser.parse_args()

    raw = load_results(args.input)
    normalized = normalize_columns(raw)
    plot_fig2(normalized, out_png=args.out_png, out_pdf=args.out_pdf)

    print("Saved:")
    print(args.out_png)
    print(args.out_pdf)


if __name__ == "__main__":
    main()
