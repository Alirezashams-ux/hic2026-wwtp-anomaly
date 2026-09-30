from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from models.cs_lstm_autoencoder import train_cs_lstm_ae
from models.gaussian_copula_detector import GaussianCopulaDetector
from preprocessing.data_pipeline import assign_climate_group, build_sequences


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "Seoul_Tancheon_WWTP": {
        "path": "data/Seoul_Tancheon_WWTP.csv",
        "feature_cols": [
            "1TP",
            "2TP",
            "BODout1",
            "SSout1",
            "TNout1",
            "TPout1",
            "BODout2",
            "SSout2",
            "TNout2",
            "TPout2",
            "Precipitation",
            "Max Precipitation in 1 hour",
            "Average Temperature",
            "Gap of daily temperature",
        ],
        "temp_col": "Average Temperature",
        "precip_col": "Precipitation",
    },
    "Seoul3": {
        "path": "data/Seoul3.csv",
        "feature_cols": [
            "Inflow",
            "Outflow",
            "BODout",
            "SSout",
            "TNout",
            "TPout",
            "BODin",
            "SSin",
            "TNin",
            "TPin",
            "precip_total_mm",
            "precip_max_1h_mm",
            "temp_mean_c",
            "temp_range_c",
        ],
        "temp_col": "temp_mean_c",
        "precip_col": "precip_total_mm",
    },
}


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(r"^,", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"(?<=\d)\s+(?=\d)", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_dataset(path: Path, feature_cols: List[str], temp_col: str, precip_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} has no Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    required = list(feature_cols) + [temp_col, precip_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    for col in required:
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


def rolling_score_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    temp_col: str,
    precip_col: str,
    thresholds: Tuple[float, ...] = (0.95, 0.975, 0.99),
    min_train: int = 365,
    horizon: int = 90,
    seq_len: int = 14,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    rows: List[Dict[str, float | int | str]] = []
    fold_rows: List[Dict[str, float | int]] = []
    fold_id = 0

    for train_end in range(min_train, n - horizon + 1, horizon):
        test_end = min(train_end + horizon, n)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)

        train_x = df.loc[train_idx, feature_cols].copy()
        test_x = df.loc[test_idx, feature_cols].copy()

        imputer = KNNImputer(n_neighbors=5, weights="distance")
        x_train_imp = imputer.fit_transform(train_x)
        x_test_imp = imputer.transform(test_x)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_imp)
        x_test = scaler.transform(x_test_imp)

        gc_model = GaussianCopulaDetector(epsilon=1e-6, regularization=1e-6)
        gc_model.fit(x_train)
        gc_train = gc_model.score_samples(x_train)
        gc_test = gc_model.score_samples(x_test)
        gc_thr = {q: float(np.quantile(gc_train, q)) for q in thresholds}

        train_seq, _ = build_sequences(x_train, seq_len=seq_len)
        combined = np.vstack([x_train, x_test])
        combined_seq, combined_end = build_sequences(combined, seq_len=seq_len)
        mask_test_seq = (combined_end >= len(x_train)) & (combined_end < len(combined))
        test_seq = combined_seq[mask_test_seq]
        test_seq_end = combined_end[mask_test_seq]

        cs = train_cs_lstm_ae(
            train_sequences=train_seq,
            test_sequences=test_seq,
            input_dim=len(feature_cols),
            seed=seed,
            hidden_dim=64,
            latent_dim=32,
            lr=1e-3,
            batch_size=64,
            epochs=70,
        )
        cs_thr = {q: float(np.quantile(cs.train_scores, q)) for q in thresholds}

        cs_map = {}
        for score, end_idx in zip(cs.test_scores, test_seq_end):
            local_day = int(end_idx - len(x_train))
            cs_map[local_day] = float(score)

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start_idx": int(train_idx[0]),
                "train_end_idx": int(train_idx[-1]),
                "test_start_idx": int(test_idx[0]),
                "test_end_idx": int(test_idx[-1]),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "gc_thr_0.99": gc_thr[0.99],
                "cs_thr_0.99": cs_thr[0.99],
            }
        )

        for j, g_idx in enumerate(test_idx):
            row = {
                "fold": fold_id,
                "Date": df.loc[g_idx, "Date"],
                "temp_value": float(df.loc[g_idx, temp_col]),
                "precip_value": float(df.loc[g_idx, precip_col]),
                "GC_Score": float(gc_test[j]),
                "CS_Score": cs_map.get(j, np.nan),
            }
            for q in thresholds:
                row[f"GC_thr_{q}"] = gc_thr[q]
                row[f"CS_thr_{q}"] = cs_thr[q]
                row[f"GC_Anomaly_{q}"] = int(gc_test[j] > gc_thr[q])
                row[f"CS_Anomaly_{q}"] = int((cs_map.get(j, np.nan) > cs_thr[q]) if not np.isnan(cs_map.get(j, np.nan)) else 0)
            rows.append(row)

        fold_id += 1

    scored = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    return scored, fold_df


def climate_tables(
    scored: pd.DataFrame,
    temp_q90: float,
    precip_q90: float,
    quantiles: Tuple[float, ...] = (0.95, 0.975, 0.99),
) -> pd.DataFrame:
    out = scored.copy()
    out["ClimateGroup"] = assign_climate_group(
        avg_temp=out["temp_value"],
        precipitation=out["precip_value"],
        temp_q90=temp_q90,
        precip_q90=precip_q90,
    )

    recs = []
    for q in quantiles:
        for model_name, anomaly_col, score_col in [
            ("CS-LSTM AE", f"CS_Anomaly_{q}", "CS_Score"),
            ("Gaussian Copula", f"GC_Anomaly_{q}", "GC_Score"),
        ]:
            grp = (
                out.groupby("ClimateGroup", dropna=False)
                .agg(n_days=(anomaly_col, "size"), n_anomalies=(anomaly_col, "sum"), mean_score=(score_col, "mean"))
                .reset_index()
            )
            full = pd.DataFrame({"ClimateGroup": ["Normal", "Hottest 10%", "Wettest 10%"]})
            grp = full.merge(grp, on="ClimateGroup", how="left")
            grp["n_days"] = grp["n_days"].fillna(0).astype(int)
            grp["n_anomalies"] = grp["n_anomalies"].fillna(0).astype(int)
            grp["anomaly_rate_percent"] = np.where(grp["n_days"] > 0, 100.0 * grp["n_anomalies"] / grp["n_days"], np.nan)

            ci_lo = []
            ci_hi = []
            for k, n in zip(grp["n_anomalies"], grp["n_days"]):
                lo, hi = wilson_ci(int(k), int(n))
                ci_lo.append(np.nan if np.isnan(lo) else 100.0 * lo)
                ci_hi.append(np.nan if np.isnan(hi) else 100.0 * hi)

            grp["ci95_low_percent"] = ci_lo
            grp["ci95_high_percent"] = ci_hi
            grp["model"] = model_name
            grp["threshold_quantile"] = q

            normal = grp.loc[grp["ClimateGroup"] == "Normal", "anomaly_rate_percent"].iloc[0]
            wet = grp.loc[grp["ClimateGroup"] == "Wettest 10%", "anomaly_rate_percent"].iloc[0]
            hot = grp.loc[grp["ClimateGroup"] == "Hottest 10%", "anomaly_rate_percent"].iloc[0]
            grp["wet_minus_normal_pp"] = wet - normal if not np.isnan(wet) and not np.isnan(normal) else np.nan
            grp["hot_minus_normal_pp"] = hot - normal if not np.isnan(hot) and not np.isnan(normal) else np.nan

            recs.append(grp)

    return pd.concat(recs, ignore_index=True)


def dataset_summary(scored: pd.DataFrame, climate: pd.DataFrame, dataset_name: str, temp_q90: float, precip_q90: float) -> pd.DataFrame:
    q99 = climate[climate["threshold_quantile"] == 0.99]
    rows = []
    for model_name in ["CS-LSTM AE", "Gaussian Copula"]:
        sub = q99[q99["model"] == model_name].set_index("ClimateGroup")
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "n_scored_days": int(len(scored)),
                "temp_q90": float(temp_q90),
                "precip_q90": float(precip_q90),
                "normal_rate_q99": float(sub.loc["Normal", "anomaly_rate_percent"]) if "Normal" in sub.index else np.nan,
                "hot_rate_q99": float(sub.loc["Hottest 10%", "anomaly_rate_percent"]) if "Hottest 10%" in sub.index else np.nan,
                "wet_rate_q99": float(sub.loc["Wettest 10%", "anomaly_rate_percent"]) if "Wettest 10%" in sub.index else np.nan,
                "wet_minus_normal_pp_q99": float(sub["wet_minus_normal_pp"].iloc[0]) if len(sub) else np.nan,
                "hot_minus_normal_pp_q99": float(sub["hot_minus_normal_pp"].iloc[0]) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_all(project_root: Path, seed: int = 42) -> None:
    out_dir = project_root / "results" / "two_real_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_climate = []
    all_scored = []

    for dataset_name, cfg in DATASET_CONFIGS.items():
        path = project_root / str(cfg["path"])
        feature_cols = list(cfg["feature_cols"])
        temp_col = str(cfg["temp_col"])
        precip_col = str(cfg["precip_col"])

        df = load_dataset(path, feature_cols, temp_col, precip_col)
        temp_q90 = float(df[temp_col].quantile(0.9))
        precip_q90 = float(df[precip_col].quantile(0.9))

        scored, folds = rolling_score_dataset(
            df=df,
            feature_cols=feature_cols,
            temp_col=temp_col,
            precip_col=precip_col,
            thresholds=(0.95, 0.975, 0.99),
            min_train=365,
            horizon=90,
            seq_len=14,
            seed=seed,
        )
        scored["dataset"] = dataset_name
        climate = climate_tables(scored, temp_q90=temp_q90, precip_q90=precip_q90, quantiles=(0.95, 0.975, 0.99))
        climate["dataset"] = dataset_name
        summary = dataset_summary(scored, climate, dataset_name, temp_q90=temp_q90, precip_q90=precip_q90)

        # Visualization-ready CSV bundle (dataset-specific)
        scored.to_csv(out_dir / f"{dataset_name}_daily_scores_rolling.csv", index=False)
        folds.to_csv(out_dir / f"{dataset_name}_fold_thresholds.csv", index=False)
        climate.to_csv(out_dir / f"{dataset_name}_climate_rates_threshold_sweep.csv", index=False)

        q99 = climate[climate["threshold_quantile"] == 0.99].copy()
        q99.to_csv(out_dir / f"{dataset_name}_climate_rates_q99.csv", index=False)

        sensitivity = (
            climate[["dataset", "model", "threshold_quantile", "ClimateGroup", "anomaly_rate_percent", "n_days", "ci95_low_percent", "ci95_high_percent"]]
            .sort_values(["model", "threshold_quantile", "ClimateGroup"])
            .reset_index(drop=True)
        )
        sensitivity.to_csv(out_dir / f"{dataset_name}_sensitivity_long.csv", index=False)

        all_summary.append(summary)
        all_climate.append(climate)
        all_scored.append(scored)

    summary_all = pd.concat(all_summary, ignore_index=True)
    climate_all = pd.concat(all_climate, ignore_index=True)
    scored_all = pd.concat(all_scored, ignore_index=True)

    summary_all.to_csv(out_dir / "comparison_summary_q99.csv", index=False)
    climate_all.to_csv(out_dir / "comparison_climate_rates_all_thresholds.csv", index=False)
    scored_all.to_csv(out_dir / "comparison_daily_scores_all.csv", index=False)

    novelty = summary_all.copy()
    novelty["novelty_stress_index_q99"] = np.where(
        (~novelty["wet_minus_normal_pp_q99"].isna()) & (~novelty["hot_minus_normal_pp_q99"].isna()),
        novelty["wet_minus_normal_pp_q99"].abs() + novelty["hot_minus_normal_pp_q99"].abs(),
        novelty["wet_minus_normal_pp_q99"].abs(),
    )
    novelty.to_csv(out_dir / "comparison_novelty_indices_q99.csv", index=False)

    with open(out_dir / "README_outputs.txt", "w", encoding="utf-8") as f:
        f.write("Two real-dataset experiments with same methods (CS-LSTM AE + Gaussian Copula).\n")
        f.write("Evaluation: rolling-origin out-of-sample; thresholds: 0.95/0.975/0.99.\n\n")
        f.write("Dataset-specific outputs:\n")
        for name in DATASET_CONFIGS.keys():
            f.write(f"- {name}_daily_scores_rolling.csv\n")
            f.write(f"- {name}_fold_thresholds.csv\n")
            f.write(f"- {name}_climate_rates_threshold_sweep.csv\n")
            f.write(f"- {name}_climate_rates_q99.csv\n")
            f.write(f"- {name}_sensitivity_long.csv\n")
        f.write("\nCross-dataset outputs:\n")
        f.write("- comparison_summary_q99.csv\n")
        f.write("- comparison_climate_rates_all_thresholds.csv\n")
        f.write("- comparison_daily_scores_all.csv\n")
        f.write("- comparison_novelty_indices_q99.csv\n")

    print("Saved two-real-dataset experiment outputs in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full same-method experiments on Seoul3 and Seoul_Tancheon_WWTP.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all(project_root=args.project_root, seed=args.seed)
