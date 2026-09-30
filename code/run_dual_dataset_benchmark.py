from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from evaluation.evaluate import climate_conditioned_anomaly_rates, label_anomalies
from models.cs_lstm_autoencoder import train_cs_lstm_ae
from models.gaussian_copula_detector import GaussianCopulaDetector
from preprocessing.data_pipeline import assign_climate_group, build_sequences, load_and_prepare_data


SEOUL2_FEATURES = [
    "Inflow",
    "Outflow",
    "BODin",
    "BODout",
    "TOCin",
    "TOCout",
    "SSin",
    "SSout",
    "TNin",
    "TNout",
    "TPin",
    "TPout",
    "Rainfall",
    "Temperature",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(",", "", regex=False)
        .str.replace(r"(?<=\d)\s+(?=\d)", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_seoul2_synth(path: Path, feature_cols: List[str], train_ratio: float = 0.6) -> Dict[str, object]:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in feature_cols:
        df[col] = _clean_numeric(df[col])

    labels = df["is_anomaly"].fillna(0).astype(int).to_numpy()

    split_idx = int(np.floor(len(df) * train_ratio))
    split_idx = max(100, min(split_idx, len(df) - 100))

    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, len(df))

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    x_train = imputer.fit_transform(df.loc[train_idx, feature_cols])
    x_test = imputer.transform(df.loc[test_idx, feature_cols])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {
        "df": df,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "split_idx": split_idx,
    }


def evaluate_synthetic_file(path: Path, seq_len: int = 14, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data = prepare_seoul2_synth(path, SEOUL2_FEATURES, train_ratio=0.6)
    train_seq, _ = build_sequences(data["x_train_scaled"], seq_len=seq_len)
    test_seq, test_seq_idx = build_sequences(data["x_test_scaled"], seq_len=seq_len)

    cs = train_cs_lstm_ae(
        train_sequences=train_seq,
        test_sequences=test_seq,
        input_dim=len(SEOUL2_FEATURES),
        seed=seed,
        hidden_dim=64,
        latent_dim=32,
        lr=1e-3,
        batch_size=64,
        epochs=90,
    )

    gc_detector = GaussianCopulaDetector(epsilon=1e-6, regularization=1e-6)
    gc = gc_detector.fit_and_score(data["x_train_scaled"], data["x_test_scaled"])

    labels_test = data["labels"][data["test_idx"]]

    cs_scores_daily = np.full(len(labels_test), np.nan)
    cs_scores_daily[test_seq_idx] = cs.test_scores
    valid_mask = ~np.isnan(cs_scores_daily)

    cs_seq_labels = []
    for end_idx in test_seq_idx:
        start_idx = end_idx - seq_len + 1
        start_idx = max(0, start_idx)
        cs_seq_labels.append(int(np.max(labels_test[start_idx : end_idx + 1])))
    cs_labels_eval = np.asarray(cs_seq_labels, dtype=int)
    cs_scores_eval = cs.test_scores
    cs_pred = (cs_scores_eval > cs.threshold_99).astype(int)

    gc_pred = (gc.test_scores > gc.threshold_99).astype(int)

    scenario_name = path.stem.replace("Seoul_2_synth_", "")

    metrics = []
    for model_name, y_true, y_score, y_pred in [
        ("CS-LSTM AE", cs_labels_eval, cs_scores_eval, cs_pred),
        ("Gaussian Copula", labels_test, gc.test_scores, gc_pred),
    ]:
        metrics.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
                "pr_auc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "threshold_99": float(cs.threshold_99 if model_name == "CS-LSTM AE" else gc.threshold_99),
                "n_test": int(len(y_true)),
                "n_anomaly_test": int(np.sum(y_true)),
            }
        )

    curve_payload = {
        "scenario": scenario_name,
        "cs_true": cs_labels_eval,
        "cs_score": cs_scores_eval,
        "gc_true": labels_test,
        "gc_score": gc.test_scores,
    }

    return pd.DataFrame(metrics), curve_payload


def run_real_dataset_analysis(project_root: Path, seq_len: int = 14, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
    prepared = load_and_prepare_data(project_root / "data" / "Seoul_Tancheon_WWTP.csv", test_fraction=0.2, n_neighbors=5)

    train_seq, _ = build_sequences(prepared.train_features_scaled, seq_len=seq_len)
    test_seq, test_seq_idx = build_sequences(prepared.test_features_scaled, seq_len=seq_len)

    cs = train_cs_lstm_ae(
        train_sequences=train_seq,
        test_sequences=test_seq,
        input_dim=len(prepared.feature_columns),
        seed=seed,
        hidden_dim=64,
        latent_dim=32,
        lr=1e-3,
        batch_size=64,
        epochs=90,
    )

    gc_detector = GaussianCopulaDetector(epsilon=1e-6, regularization=1e-6)
    gc = gc_detector.fit_and_score(prepared.train_features_scaled, prepared.test_features_scaled)

    test_df = prepared.clean_df.iloc[prepared.test_index].copy().reset_index(drop=True)
    test_df["ClimateGroup"] = assign_climate_group(
        avg_temp=test_df["Average Temperature"],
        precipitation=test_df["Precipitation"],
        temp_q90=prepared.climate_thresholds["temp_q90"],
        precip_q90=prepared.climate_thresholds["precip_q90"],
    )

    test_df["GC_Score"] = gc.test_scores
    test_df["GC_Anomaly"] = label_anomalies(test_df["GC_Score"], gc.threshold_99)

    cs_scores_daily = np.full(len(test_df), np.nan)
    cs_scores_daily[test_seq_idx] = cs.test_scores
    test_df["CSLSTM_Score"] = cs_scores_daily
    test_df["CSLSTM_Anomaly"] = 0
    valid = ~test_df["CSLSTM_Score"].isna()
    test_df.loc[valid, "CSLSTM_Anomaly"] = (
        test_df.loc[valid, "CSLSTM_Score"] > cs.threshold_99
    ).astype(int)

    cs_rates = climate_conditioned_anomaly_rates(
        df=test_df[valid],
        score_col="CSLSTM_Score",
        anomaly_col="CSLSTM_Anomaly",
        climate_col="ClimateGroup",
    )
    cs_rates["model"] = "CS-LSTM AE"

    gc_rates = climate_conditioned_anomaly_rates(
        df=test_df,
        score_col="GC_Score",
        anomaly_col="GC_Anomaly",
        climate_col="ClimateGroup",
    )
    gc_rates["model"] = "Gaussian Copula"

    out = pd.concat([cs_rates, gc_rates], ignore_index=True)

    novelty = {}
    for model_name, sub in out.groupby("model"):
        normal_rate = float(sub.loc[sub["ClimateGroup"] == "Normal", "anomaly_rate_percent"].iloc[0])
        wet_rate = float(sub.loc[sub["ClimateGroup"] == "Wettest 10%", "anomaly_rate_percent"].iloc[0])
        heat_rate = float(sub.loc[sub["ClimateGroup"] == "Hottest 10%", "anomaly_rate_percent"].iloc[0])
        normal_days = int(sub.loc[sub["ClimateGroup"] == "Normal", "n_days"].iloc[0])
        wet_days = int(sub.loc[sub["ClimateGroup"] == "Wettest 10%", "n_days"].iloc[0])
        heat_days = int(sub.loc[sub["ClimateGroup"] == "Hottest 10%", "n_days"].iloc[0])
        normal_score = float(sub.loc[sub["ClimateGroup"] == "Normal", "mean_score"].fillna(0.0).iloc[0])
        wet_score = float(sub.loc[sub["ClimateGroup"] == "Wettest 10%", "mean_score"].fillna(0.0).iloc[0])
        heat_score = float(sub.loc[sub["ClimateGroup"] == "Hottest 10%", "mean_score"].fillna(0.0).iloc[0])

        novelty[f"{model_name}_wet_score_ratio"] = (
            (wet_score + 1e-6) / (normal_score + 1e-6) if (normal_days > 0 and wet_days > 0) else np.nan
        )
        novelty[f"{model_name}_heat_score_ratio"] = (
            (heat_score + 1e-6) / (normal_score + 1e-6) if (normal_days > 0 and heat_days > 0) else np.nan
        )
        novelty[f"{model_name}_wet_anomaly_excess_pp"] = wet_rate - normal_rate if (normal_days > 0 and wet_days > 0) else np.nan
        novelty[f"{model_name}_heat_anomaly_excess_pp"] = (
            heat_rate - normal_rate if (normal_days > 0 and heat_days > 0) else np.nan
        )

    return out, novelty


def plot_figure_1_climate(real_rates: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = ["Normal", "Hottest 10%", "Wettest 10%"]
    models = ["CS-LSTM AE", "Gaussian Copula"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for i, model_name in enumerate(models):
        sub = real_rates[real_rates["model"] == model_name].set_index("ClimateGroup")
        vals = sub.reindex(order)["anomaly_rate_percent"].fillna(0.0)
        axes[i].bar(order, vals.values)
        axes[i].set_title(f"({chr(97+i)}) {model_name}")
        axes[i].set_ylabel("Anomaly rate (%)")
        axes[i].tick_params(axis="x", rotation=20)
        for idx, v in enumerate(vals.values):
            axes[i].text(idx, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 1. Climate-conditioned anomaly rates (Seoul_Tancheon_WWTP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_2_roc(curves: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, payload in zip(axes, curves):
        scenario = payload["scenario"]
        cs_true = payload["cs_true"]
        cs_score = payload["cs_score"]
        gc_true = payload["gc_true"]
        gc_score = payload["gc_score"]

        if len(np.unique(cs_true)) > 1:
            fpr_cs, tpr_cs, _ = roc_curve(cs_true, cs_score)
            auc_cs = auc(fpr_cs, tpr_cs)
            ax.plot(fpr_cs, tpr_cs, label=f"CS-LSTM AE (AUC={auc_cs:.3f})")
        if len(np.unique(gc_true)) > 1:
            fpr_gc, tpr_gc, _ = roc_curve(gc_true, gc_score)
            auc_gc = auc(fpr_gc, tpr_gc)
            ax.plot(fpr_gc, tpr_gc, label=f"Gaussian Copula (AUC={auc_gc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)

    fig.suptitle("Figure 2. ROC curves on synthesized Seoul_2 scenarios", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure_3_concept(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")

    boxes = [
        (0.03, 0.55, 0.2, 0.3, "Dataset A\nSeoul_2\nSynthesized labels"),
        (0.03, 0.15, 0.2, 0.3, "Dataset B\nSeoul_Tancheon\nClimate regimes"),
        (0.31, 0.35, 0.18, 0.3, "Dependency-aware\nmodels\nCS-LSTM AE + GC"),
        (0.56, 0.55, 0.18, 0.3, "Labeled benchmark\nROC/PR/F1\n(model validity)"),
        (0.56, 0.15, 0.18, 0.3, "Unlabeled climate\nstratification\n(resilience signal)"),
        (0.8, 0.35, 0.17, 0.3, "Novelty:\nCross-domain\ndependency stress\nprofiling"),
    ]

    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.23, 0.7), (0.31, 0.55)),
        ((0.23, 0.3), (0.31, 0.45)),
        ((0.49, 0.55), (0.56, 0.7)),
        ((0.49, 0.45), (0.56, 0.3)),
        ((0.74, 0.7), (0.8, 0.52)),
        ((0.74, 0.3), (0.8, 0.48)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Figure 3. Conceptual and methodological novelty framework", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(project_root: Path, seed: int = 42) -> None:
    synthetic_dir = project_root / "data" / "processed" / "synthetic"
    synth_files = sorted(synthetic_dir.glob("Seoul_2_synth_*.csv"))
    synth_files = [p for p in synth_files if "summary" not in p.name]
    if not synth_files:
        raise FileNotFoundError("No synthesized Seoul_2 files found. Run synthesis step first.")

    metrics_frames = []
    curves = []
    for file_path in synth_files:
        m, c = evaluate_synthetic_file(file_path, seq_len=14, seed=seed)
        metrics_frames.append(m)
        curves.append(c)

    synthetic_metrics = pd.concat(metrics_frames, ignore_index=True)
    real_rates, novelty = run_real_dataset_analysis(project_root, seq_len=14, seed=seed)

    table_dir = project_root / "tables"
    figure_dir = project_root / "figures"
    result_dir = project_root / "results" / "model_outputs"
    logs_dir = project_root / "results" / "logs"
    for d in [table_dir, figure_dir, result_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    table1 = synthetic_metrics.sort_values(["scenario", "model"]).reset_index(drop=True)
    table1.to_csv(table_dir / "table_1_synthetic_performance_metrics.csv", index=False)

    table2 = real_rates.copy()
    table2["novelty_wet_score_ratio"] = np.nan
    table2["novelty_heat_score_ratio"] = np.nan
    table2["novelty_wet_anomaly_excess_pp"] = np.nan
    table2["novelty_heat_anomaly_excess_pp"] = np.nan

    for model_name in ["CS-LSTM AE", "Gaussian Copula"]:
        table2.loc[table2["model"] == model_name, "novelty_wet_score_ratio"] = novelty[f"{model_name}_wet_score_ratio"]
        table2.loc[table2["model"] == model_name, "novelty_heat_score_ratio"] = novelty[f"{model_name}_heat_score_ratio"]
        table2.loc[table2["model"] == model_name, "novelty_wet_anomaly_excess_pp"] = novelty[
            f"{model_name}_wet_anomaly_excess_pp"
        ]
        table2.loc[table2["model"] == model_name, "novelty_heat_anomaly_excess_pp"] = novelty[
            f"{model_name}_heat_anomaly_excess_pp"
        ]
    table2.to_csv(table_dir / "table_2_real_climate_novelty_metrics.csv", index=False)

    plot_figure_1_climate(real_rates, figure_dir / "figure_1_real_climate_conditioned_rates.png")
    plot_figure_2_roc(curves, figure_dir / "figure_2_synthetic_roc_benchmark.png")
    plot_figure_3_concept(figure_dir / "figure_3_conceptual_methodological_novelty.png")

    with open(logs_dir / "dual_dataset_benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write("Dual-dataset benchmark summary\n")
        f.write("Methodology: CS-LSTM AE + Gaussian Copula (99th percentile thresholds)\n")
        f.write("Synthetic scenarios evaluated:\n")
        for sc in sorted(table1["scenario"].unique()):
            f.write(f"- {sc}\n")
        f.write("\nBest synthetic ROC AUC by model:\n")
        for model_name in ["CS-LSTM AE", "Gaussian Copula"]:
            best = table1[table1["model"] == model_name]["roc_auc"].max()
            f.write(f"- {model_name}: {best:.4f}\n")
        f.write("\nNovelty indices (real dataset):\n")
        for key, val in novelty.items():
            f.write(f"- {key}: {val:.4f}\n")

    print("Saved:")
    print(table_dir / "table_1_synthetic_performance_metrics.csv")
    print(table_dir / "table_2_real_climate_novelty_metrics.csv")
    print(figure_dir / "figure_1_real_climate_conditioned_rates.png")
    print(figure_dir / "figure_2_synthetic_roc_benchmark.png")
    print(figure_dir / "figure_3_conceptual_methodological_novelty.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dual-dataset anomaly detection benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(project_root=args.project_root, seed=args.seed)
