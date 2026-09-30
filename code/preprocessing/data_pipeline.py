from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
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
]


@dataclass
class PreparedData:
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    feature_columns: List[str]
    train_index: np.ndarray
    test_index: np.ndarray
    train_features_scaled: np.ndarray
    test_features_scaled: np.ndarray
    scaler: StandardScaler
    imputer: KNNImputer
    climate_thresholds: Dict[str, float]


def _to_clean_numeric(series: pd.Series) -> pd.Series:
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


def load_and_prepare_data(
    csv_path: Path,
    test_fraction: float = 0.2,
    n_neighbors: int = 5,
) -> PreparedData:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    if "Date" not in raw_df.columns:
        raise ValueError("Dataset must contain a 'Date' column.")

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    for col in FEATURE_COLUMNS:
        df[col] = _to_clean_numeric(df[col])

    feature_df = df[FEATURE_COLUMNS].copy()

    n_total = len(feature_df)
    if n_total < 200:
        raise ValueError("Dataset is too small for robust train/test unsupervised modeling.")

    split_idx = int(np.floor((1.0 - test_fraction) * n_total))
    split_idx = max(100, min(split_idx, n_total - 50))

    train_index = np.arange(0, split_idx)
    test_index = np.arange(split_idx, n_total)

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    train_imputed = imputer.fit_transform(feature_df.iloc[train_index])
    test_imputed = imputer.transform(feature_df.iloc[test_index])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    complete_imputed = np.zeros((n_total, len(FEATURE_COLUMNS)), dtype=float)
    complete_imputed[train_index] = train_imputed
    complete_imputed[test_index] = test_imputed
    clean_df = df.copy()
    clean_df[FEATURE_COLUMNS] = complete_imputed

    climate_thresholds = {
        "temp_q90": float(clean_df["Average Temperature"].quantile(0.9)),
        "precip_q90": float(clean_df["Precipitation"].quantile(0.9)),
    }

    return PreparedData(
        raw_df=raw_df,
        clean_df=clean_df,
        feature_columns=FEATURE_COLUMNS,
        train_index=train_index,
        test_index=test_index,
        train_features_scaled=train_scaled,
        test_features_scaled=test_scaled,
        scaler=scaler,
        imputer=imputer,
        climate_thresholds=climate_thresholds,
    )


def build_sequences(data: np.ndarray, seq_len: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    if len(data) < seq_len:
        raise ValueError("Not enough rows to create sequences.")

    sequences = []
    target_indices = []
    for end_idx in range(seq_len - 1, len(data)):
        start_idx = end_idx - seq_len + 1
        sequences.append(data[start_idx : end_idx + 1])
        target_indices.append(end_idx)

    return np.asarray(sequences, dtype=np.float32), np.asarray(target_indices, dtype=np.int64)


def assign_climate_group(
    avg_temp: pd.Series,
    precipitation: pd.Series,
    temp_q90: float,
    precip_q90: float,
) -> pd.Series:
    hottest = avg_temp >= temp_q90
    wettest = precipitation >= precip_q90
    normal = ~(hottest | wettest)

    group = pd.Series(index=avg_temp.index, dtype="object")
    group.loc[normal] = "Normal"
    group.loc[hottest] = "Hottest 10%"
    group.loc[wettest] = "Wettest 10%"
    return group
