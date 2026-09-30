from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from models.cs_lstm_autoencoder import train_cs_lstm_ae
from models.gaussian_copula_detector import GaussianCopulaDetector
from preprocessing.data_pipeline import FEATURE_COLUMNS, assign_climate_group, build_sequences


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(r"^,", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        df[col] = clean_numeric(df[col])
    return df


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rolling_scores(
    df: pd.DataFrame,
    min_train: int = 365,
    horizon: int = 90,
    seq_len: int = 14,
    thresholds: Tuple[float, ...] = (0.95, 0.975, 0.99),
    seed: int = 42,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    n = len(df)
    fold_id = 0

    for train_end in range(min_train, n - horizon + 1, horizon):
        test_end = min(train_end + horizon, n)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)

        train_x = df.loc[train_idx, FEATURE_COLUMNS].copy()
        test_x = df.loc[test_idx, FEATURE_COLUMNS].copy()

        imputer = KNNImputer(n_neighbors=5, weights="distance")
        train_imp = imputer.fit_transform(train_x)
        test_imp = imputer.transform(test_x)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imp)
        test_scaled = scaler.transform(test_imp)

        gc_model = GaussianCopulaDetector(epsilon=1e-6, regularization=1e-6)
        gc_model.fit(train_scaled)
        gc_train_scores = gc_model.score_samples(train_scaled)
        gc_test_scores = gc_model.score_samples(test_scaled)
        gc_thresholds = {q: float(np.quantile(gc_train_scores, q)) for q in thresholds}

        train_seq, _ = build_sequences(train_scaled, seq_len=seq_len)
        combined = np.vstack([train_scaled, test_scaled])
        combined_seq, combined_end_idx = build_sequences(combined, seq_len=seq_len)

        test_mask = (combined_end_idx >= len(train_scaled)) & (combined_end_idx < len(combined))
        test_seq = combined_seq[test_mask]
        test_seq_end_idx = combined_end_idx[test_mask]

        cs_result = train_cs_lstm_ae(
            train_sequences=train_seq,
            test_sequences=test_seq,
            input_dim=len(FEATURE_COLUMNS),
            seed=seed,
            hidden_dim=64,
            latent_dim=32,
            lr=1e-3,
            batch_size=64,
            epochs=70,
        )
        cs_thresholds = {q: float(np.quantile(cs_result.train_scores, q)) for q in thresholds}

        cs_score_map = {}
        for score, end_idx in zip(cs_result.test_scores, test_seq_end_idx):
            local_test_day = int(end_idx - len(train_scaled))
            cs_score_map[local_test_day] = float(score)

        for j, global_idx in enumerate(test_idx):
            row = {
                "fold": fold_id,
                "Date": df.loc[global_idx, "Date"],
                "Average Temperature": float(df.loc[global_idx, "Average Temperature"]),
                "Precipitation": float(df.loc[global_idx, "Precipitation"]),
                "GC_Score": float(gc_test_scores[j]),
                "CS_Score": cs_score_map.get(j, np.nan),
            }
            for q in thresholds:
                row[f"GC_thr_{q}"] = gc_thresholds[q]
                row[f"CS_thr_{q}"] = cs_thresholds[q]
            rows.append(row)

        fold_id += 1

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


def climate_rate_table(scored: pd.DataFrame, quantile: float, climate_thresholds: Dict[str, float]) -> pd.DataFrame:
    df = scored.copy()
    df["ClimateGroup"] = assign_climate_group(
        avg_temp=df["Average Temperature"],
        precipitation=df["Precipitation"],
        temp_q90=climate_thresholds["temp_q90"],
        precip_q90=climate_thresholds["precip_q90"],
    )

    df["GC_Anomaly"] = (df["GC_Score"] > df[f"GC_thr_{quantile}"]).astype(int)
    df["CS_Anomaly"] = (df["CS_Score"] > df[f"CS_thr_{quantile}"]).astype(int)

    parts = []
    for model_name, anomaly_col, score_col in [
        ("CS-LSTM AE", "CS_Anomaly", "CS_Score"),
        ("Gaussian Copula", "GC_Anomaly", "GC_Score"),
    ]:
        sub = (
            df.groupby("ClimateGroup", dropna=False)
            .agg(n_days=(anomaly_col, "size"), n_anomalies=(anomaly_col, "sum"), mean_score=(score_col, "mean"))
            .reset_index()
        )
        full = pd.DataFrame({"ClimateGroup": ["Normal", "Hottest 10%", "Wettest 10%"]})
        sub = full.merge(sub, on="ClimateGroup", how="left")
        sub["n_days"] = sub["n_days"].fillna(0).astype(int)
        sub["n_anomalies"] = sub["n_anomalies"].fillna(0).astype(int)
        sub["anomaly_rate_percent"] = np.where(sub["n_days"] > 0, 100.0 * sub["n_anomalies"] / sub["n_days"], np.nan)

        ci_lo = []
        ci_hi = []
        for k, n in zip(sub["n_anomalies"], sub["n_days"]):
            lo, hi = wilson_ci(int(k), int(n))
            ci_lo.append(np.nan if np.isnan(lo) else 100.0 * lo)
            ci_hi.append(np.nan if np.isnan(hi) else 100.0 * hi)
        sub["ci95_low_percent"] = ci_lo
        sub["ci95_high_percent"] = ci_hi
        sub["model"] = model_name
        sub["threshold_quantile"] = quantile
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    return out


def threshold_sensitivity_table(scored: pd.DataFrame, climate_thresholds: Dict[str, float], quantiles=(0.95, 0.975, 0.99)) -> pd.DataFrame:
    records = []
    for q in quantiles:
        tab = climate_rate_table(scored, quantile=q, climate_thresholds=climate_thresholds)
        for model_name in ["CS-LSTM AE", "Gaussian Copula"]:
            sub = tab[tab["model"] == model_name].set_index("ClimateGroup")
            normal = float(sub.loc["Normal", "anomaly_rate_percent"]) if "Normal" in sub.index else np.nan
            hot = float(sub.loc["Hottest 10%", "anomaly_rate_percent"]) if "Hottest 10%" in sub.index else np.nan
            wet = float(sub.loc["Wettest 10%", "anomaly_rate_percent"]) if "Wettest 10%" in sub.index else np.nan
            records.append(
                {
                    "model": model_name,
                    "threshold_quantile": q,
                    "normal_rate_percent": normal,
                    "hottest_rate_percent": hot,
                    "wettest_rate_percent": wet,
                    "wet_minus_normal_pp": (wet - normal) if not np.isnan(wet) and not np.isnan(normal) else np.nan,
                    "hot_minus_normal_pp": (hot - normal) if not np.isnan(hot) and not np.isnan(normal) else np.nan,
                }
            )
    return pd.DataFrame(records)


def plot_climate_ci(table_q99: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = ["Normal", "Hottest 10%", "Wettest 10%"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for i, model_name in enumerate(["CS-LSTM AE", "Gaussian Copula"]):
        sub = table_q99[table_q99["model"] == model_name].set_index("ClimateGroup").reindex(order)
        vals = sub["anomaly_rate_percent"].to_numpy(dtype=float)
        lo = sub["ci95_low_percent"].to_numpy(dtype=float)
        hi = sub["ci95_high_percent"].to_numpy(dtype=float)
        n_days = sub["n_days"].to_numpy(dtype=int)

        x = np.arange(len(order))
        valid = ~np.isnan(vals)
        axes[i].bar(x[valid], vals[valid])

        yerr_low = np.where(valid, vals - lo, np.nan)
        yerr_high = np.where(valid, hi - vals, np.nan)
        axes[i].errorbar(x[valid], vals[valid], yerr=[yerr_low[valid], yerr_high[valid]], fmt="none", capsize=4)

        for j, (v, n) in enumerate(zip(vals, n_days)):
            if np.isnan(v):
                axes[i].text(j, 0.2, "N/A\n(n=0)", ha="center", va="bottom", fontsize=9)
            else:
                axes[i].text(j, v + 0.2, f"{v:.2f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(order, rotation=20)
        axes[i].set_title(model_name)
        axes[i].set_ylabel("Anomaly rate (%)")

    fig.suptitle("Figure 1 (Revised). Climate-conditioned anomaly rates with 95% CI (rolling-origin, q=0.99)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(sens: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for i, model_name in enumerate(["CS-LSTM AE", "Gaussian Copula"]):
        sub = sens[sens["model"] == model_name].sort_values("threshold_quantile")
        axes[i].plot(sub["threshold_quantile"], sub["normal_rate_percent"], marker="o", label="Normal")
        axes[i].plot(sub["threshold_quantile"], sub["hottest_rate_percent"], marker="o", label="Hottest 10%")
        axes[i].plot(sub["threshold_quantile"], sub["wettest_rate_percent"], marker="o", label="Wettest 10%")
        axes[i].set_title(model_name)
        axes[i].set_xlabel("Threshold quantile")
        axes[i].set_ylabel("Anomaly rate (%)")
        axes[i].set_xticks([0.95, 0.975, 0.99])
        axes[i].legend(fontsize=8)
    fig.suptitle("Figure 2 (Revised). Threshold sensitivity (95 / 97.5 / 99%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main(project_root: Path, seed: int = 42) -> None:
    csv_path = project_root / "data" / "Seoul_Tancheon_WWTP.csv"
    df = load_clean(csv_path)

    climate_thresholds = {
        "temp_q90": float(df["Average Temperature"].quantile(0.9)),
        "precip_q90": float(df["Precipitation"].quantile(0.9)),
    }

    scored = rolling_scores(df=df, min_train=365, horizon=90, seq_len=14, thresholds=(0.95, 0.975, 0.99), seed=seed)

    table_q99 = climate_rate_table(scored, quantile=0.99, climate_thresholds=climate_thresholds)
    sens = threshold_sensitivity_table(scored, climate_thresholds, quantiles=(0.95, 0.975, 0.99))

    table_dir = project_root / "tables"
    fig_dir = project_root / "figures"
    log_dir = project_root / "results" / "logs"
    for d in [table_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scored.to_csv(project_root / "results" / "anomaly_scores" / "rolling_real_daily_scores.csv", index=False)
    table_q99.to_csv(table_dir / "table_2_real_climate_novelty_metrics_rolling_q99.csv", index=False)
    sens.to_csv(table_dir / "table_3_threshold_sensitivity_rolling.csv", index=False)

    plot_climate_ci(table_q99, fig_dir / "figure_1_climate_rates_with_ci_rolling_q99.png")
    plot_threshold_sensitivity(sens, fig_dir / "figure_2_threshold_sensitivity_rolling.png")

    with open(log_dir / "real_climate_robustness_summary.txt", "w", encoding="utf-8") as f:
        f.write("Rolling-origin real-data climate robustness summary\n")
        f.write(f"Rows scored out-of-sample: {len(scored)}\n")
        f.write(f"Temperature q90: {climate_thresholds['temp_q90']:.3f}\n")
        f.write(f"Precipitation q90: {climate_thresholds['precip_q90']:.3f}\n")

    print("Saved revised outputs:")
    print(table_dir / "table_2_real_climate_novelty_metrics_rolling_q99.csv")
    print(table_dir / "table_3_threshold_sensitivity_rolling.csv")
    print(fig_dir / "figure_1_climate_rates_with_ci_rolling_q99.png")
    print(fig_dir / "figure_2_threshold_sensitivity_rolling.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run revised robust climate evaluation with rolling-origin validation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(project_root=args.project_root, seed=args.seed)
