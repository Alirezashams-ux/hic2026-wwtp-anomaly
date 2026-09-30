from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.evaluate import (
    climate_conditioned_anomaly_rates,
    label_anomalies,
    model_summary_table,
    paper_ready_climate_table,
)
from models.cs_lstm_autoencoder import train_cs_lstm_ae
from models.gaussian_copula_detector import GaussianCopulaDetector
from preprocessing.data_pipeline import assign_climate_group, build_sequences, load_and_prepare_data
from visualization.figures import plot_climate_conditioned_rates


def run_pipeline(project_root: Path, sequence_length: int = 14, seed: int = 42) -> None:
    data_path = project_root / "data" / "Seoul_Tancheon_WWTP.csv"
    prepared = load_and_prepare_data(data_path, test_fraction=0.2, n_neighbors=5)

    train_seq, _ = build_sequences(prepared.train_features_scaled, seq_len=sequence_length)
    test_seq, test_seq_idx = build_sequences(prepared.test_features_scaled, seq_len=sequence_length)

    cs_result = train_cs_lstm_ae(
        train_sequences=train_seq,
        test_sequences=test_seq,
        input_dim=len(prepared.feature_columns),
        seed=seed,
        hidden_dim=64,
        latent_dim=32,
        lr=1e-3,
        batch_size=64,
        epochs=80,
    )

    gc_detector = GaussianCopulaDetector(epsilon=1e-6, regularization=1e-6)
    gc_result = gc_detector.fit_and_score(prepared.train_features_scaled, prepared.test_features_scaled)

    test_df = pd.DataFrame(prepared.clean_df.iloc[prepared.test_index].copy()).reset_index(drop=True)
    test_df["ClimateGroup"] = assign_climate_group(
        avg_temp=test_df["Average Temperature"],
        precipitation=test_df["Precipitation"],
        temp_q90=prepared.climate_thresholds["temp_q90"],
        precip_q90=prepared.climate_thresholds["precip_q90"],
    )

    test_df["GC_Score"] = gc_result.test_scores
    test_df["GC_Anomaly"] = label_anomalies(test_df["GC_Score"], gc_result.threshold_99)

    cs_score_col = np.full(len(test_df), np.nan, dtype=float)
    cs_score_col[test_seq_idx] = cs_result.test_scores
    test_df["CSLSTM_Score"] = cs_score_col
    test_df["CSLSTM_Anomaly"] = 0

    valid_cs_mask = ~test_df["CSLSTM_Score"].isna()
    test_df.loc[valid_cs_mask, "CSLSTM_Anomaly"] = (
        test_df.loc[valid_cs_mask, "CSLSTM_Score"] > cs_result.threshold_99
    ).astype(int)

    cs_rates = climate_conditioned_anomaly_rates(
        df=test_df[valid_cs_mask],
        score_col="CSLSTM_Score",
        anomaly_col="CSLSTM_Anomaly",
        climate_col="ClimateGroup",
    )
    gc_rates = climate_conditioned_anomaly_rates(
        df=test_df,
        score_col="GC_Score",
        anomaly_col="GC_Anomaly",
        climate_col="ClimateGroup",
    )

    anomaly_dir = project_root / "results" / "anomaly_scores"
    tables_dir = project_root / "tables"
    figures_dir = project_root / "figures"
    model_dir = project_root / "results" / "model_outputs"
    logs_dir = project_root / "results" / "logs"

    for folder in [anomaly_dir, tables_dir, figures_dir, model_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    test_df.to_csv(anomaly_dir / "test_daily_scores_and_flags.csv", index=False)

    max_len = max(len(cs_result.train_scores), len(gc_result.train_scores))
    train_score_df = pd.DataFrame(
        {
            "CSLSTM_TrainScore": np.pad(
                cs_result.train_scores,
                (0, max_len - len(cs_result.train_scores)),
                mode="constant",
                constant_values=np.nan,
            ),
            "GC_TrainScore": np.pad(
                gc_result.train_scores,
                (0, max_len - len(gc_result.train_scores)),
                mode="constant",
                constant_values=np.nan,
            ),
        }
    )
    train_score_df.to_csv(anomaly_dir / "train_scores_summary.csv", index=False)

    torch.save(cs_result.model.state_dict(), model_dir / "cs_lstm_ae_model.pt")
    if gc_detector.covariance is None or gc_detector.inv_covariance is None or gc_detector.log_det_covariance is None:
        raise RuntimeError("Gaussian copula model parameters are unavailable after fitting.")

    np.savez(
        model_dir / "gaussian_copula_model.npz",
        covariance=gc_detector.covariance,
        inv_covariance=gc_detector.inv_covariance,
        log_det_covariance=gc_detector.log_det_covariance,
        threshold_99=gc_result.threshold_99,
    )

    model_summary_table().to_csv(tables_dir / "table_1_model_summary.csv", index=False)
    paper_ready_climate_table(cs_rates, gc_rates).to_csv(
        tables_dir / "table_2_climate_conditioned_anomaly_rates.csv", index=False
    )

    plot_climate_conditioned_rates(
        cs_rates=cs_rates,
        gc_rates=gc_rates,
        out_path=figures_dir / "figure_1_climate_conditioned_anomaly_rates.png",
    )

    with open(logs_dir / "pipeline_run_summary.txt", "w", encoding="utf-8") as f:
        f.write("HIC 2026 anomaly detection pipeline run summary\n")
        f.write(f"Rows total: {len(prepared.clean_df)}\n")
        f.write(f"Rows train: {len(prepared.train_index)}\n")
        f.write(f"Rows test: {len(prepared.test_index)}\n")
        f.write(f"CS-LSTM AE threshold (99th): {cs_result.threshold_99:.6f}\n")
        f.write(f"GC threshold (99th): {gc_result.threshold_99:.6f}\n")
        f.write(f"Temperature q90: {prepared.climate_thresholds['temp_q90']:.3f}\n")
        f.write(f"Precipitation q90: {prepared.climate_thresholds['precip_q90']:.3f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HIC 2026 climate-conditioned cross-sectional anomaly detection pipeline."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to project root containing data and output folders.",
    )
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        project_root=args.project_root,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
