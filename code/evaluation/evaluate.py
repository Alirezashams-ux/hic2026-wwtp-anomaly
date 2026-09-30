from __future__ import annotations

from typing import Dict

import pandas as pd


def label_anomalies(scores: pd.Series, threshold: float) -> pd.Series:
    return (scores > threshold).astype(int)


def climate_conditioned_anomaly_rates(
    df: pd.DataFrame,
    score_col: str,
    anomaly_col: str,
    climate_col: str = "ClimateGroup",
) -> pd.DataFrame:
    ordered_groups = ["Normal", "Hottest 10%", "Wettest 10%"]
    grouped = (
        df.groupby(climate_col, dropna=False)
        .agg(
            n_days=(anomaly_col, "size"),
            n_anomalies=(anomaly_col, "sum"),
            mean_score=(score_col, "mean"),
        )
        .reset_index()
    )
    grouped["anomaly_rate_percent"] = 100.0 * grouped["n_anomalies"] / grouped["n_days"]

    full = pd.DataFrame({climate_col: ordered_groups})
    grouped = full.merge(grouped, on=climate_col, how="left")
    grouped["n_days"] = grouped["n_days"].fillna(0).astype(int)
    grouped["n_anomalies"] = grouped["n_anomalies"].fillna(0).astype(int)
    grouped["anomaly_rate_percent"] = grouped["anomaly_rate_percent"].fillna(0.0)
    grouped[climate_col] = pd.Categorical(grouped[climate_col], categories=ordered_groups, ordered=True)
    return grouped.sort_values(climate_col).reset_index(drop=True)


def model_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "M4",
                "Model": "CS-LSTM AE",
                "Input level": "14-day sequences",
                "Cross-sectional?": "Yes",
                "Temporal?": "Yes",
                "Score / threshold": "Mean-squared reconstruction error; 99th percentile",
            },
            {
                "ID": "M5",
                "Model": "GC detector",
                "Input level": "Daily multivariate",
                "Cross-sectional?": "Yes",
                "Temporal?": "No",
                "Score / threshold": "Negative log-likelihood; 99th percentile",
            },
        ]
    )


def paper_ready_climate_table(cs_rates: pd.DataFrame, gc_rates: pd.DataFrame) -> pd.DataFrame:
    merged = cs_rates[["ClimateGroup", "anomaly_rate_percent"]].merge(
        gc_rates[["ClimateGroup", "anomaly_rate_percent"]],
        on="ClimateGroup",
        suffixes=("_CSLSTM", "_GC"),
    )
    merged = merged.rename(
        columns={
            "ClimateGroup": "Climate Pattern",
            "anomaly_rate_percent_CSLSTM": "CS-LSTM AE anomaly rate (%)",
            "anomaly_rate_percent_GC": "GC anomaly rate (%)",
        }
    )
    return merged
