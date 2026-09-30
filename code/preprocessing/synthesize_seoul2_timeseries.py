from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class SplitWindow:
    s_idx: int
    e_idx: int


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Input dataset must contain 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c != "Date"]
    numeric = []
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() >= 0.95:
            numeric.append(col)
    return numeric


def clean_numeric_series(series: pd.Series) -> pd.Series:
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


def zscore_fit_transform(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = pd.DataFrame({col: clean_numeric_series(df[col]) for col in cols})
    mu = x.mean()
    sigma = x.std(ddof=0).replace(0.0, 1.0)
    z = (x - mu) / sigma
    return z, mu, sigma


def inv_zscore(z_df: pd.DataFrame, mu: pd.Series, sigma: pd.Series) -> pd.DataFrame:
    return z_df * sigma + mu


def default_windows(n_rows: int, train_ratio: float = 0.6, test_end_ratio: float = 0.8) -> SplitWindow:
    s_idx = int(np.floor(n_rows * train_ratio))
    e_idx = int(np.floor(n_rows * test_end_ratio))
    s_idx = max(10, min(s_idx, n_rows - 3))
    e_idx = max(s_idx + 1, min(e_idx, n_rows - 2))
    return SplitWindow(s_idx=s_idx, e_idx=e_idx)


def synthesize_mean_shift(
    base_z: pd.DataFrame,
    target_col: str,
    window: SplitWindow,
    delta: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    z = base_z.copy()
    anomaly_mask = np.zeros(len(z), dtype=int)
    anomaly_mask[window.s_idx : window.e_idx + 1] = 1
    z.loc[window.s_idx : window.e_idx, target_col] = z.loc[window.s_idx : window.e_idx, target_col] + delta
    return z, anomaly_mask


def fit_child_models(
    z_df: pd.DataFrame,
    train_end_idx: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    dependency_map = {
        "Outflow": ["Inflow", "Rainfall"],
        "BODout": ["BODin", "Inflow", "Temperature"],
        "TOCout": ["TOCin", "Inflow"],
        "SSout": ["SSin", "Inflow", "Rainfall"],
        "TNout": ["TNin", "Temperature"],
        "TPout": ["TPin", "Rainfall"],
        "Coliformout": ["Coliformin", "Temperature", "Humidity"],
        "Sludge_volume": ["Inflow", "SSin"],
    }

    params: Dict[str, Dict[str, np.ndarray]] = {}
    residual_std: Dict[str, float] = {}

    train_slice = slice(0, train_end_idx)
    for child, parents in dependency_map.items():
        if child not in z_df.columns or any(p not in z_df.columns for p in parents):
            continue
        x = z_df.loc[train_slice, parents].to_numpy()
        y = z_df.loc[train_slice, child].to_numpy()
        model = LinearRegression()
        model.fit(x, y)
        y_hat = model.predict(x)
        resid = y - y_hat
        params[child] = {
            "parents": np.array(parents, dtype=object),
            "coef": model.coef_.copy(),
            "intercept": np.array([model.intercept_], dtype=float),
        }
        residual_std[child] = float(np.std(resid, ddof=0)) + 1e-6

    return params, residual_std


def synthesize_structural_change(
    base_z: pd.DataFrame,
    window: SplitWindow,
    drift_per_step: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    rng = np.random.default_rng(seed)
    z = base_z.copy()
    anomaly_mask = np.zeros(len(z), dtype=int)
    anomaly_mask[window.s_idx : window.e_idx + 1] = 1

    params, residual_std = fit_child_models(z, train_end_idx=window.s_idx)
    if not params:
        raise ValueError("No child dependency models could be fitted from Seoul_2 columns.")

    targets = [k for k in ["Outflow", "SSout", "TPout"] if k in params]
    chosen_parent = {
        "Outflow": "Inflow",
        "SSout": "Rainfall",
        "TPout": "Rainfall",
    }

    changed_edges: Dict[str, str] = {}
    for t in range(window.s_idx, window.e_idx + 1):
        step = t - window.s_idx + 1
        for child in targets:
            parent_names = list(params[child]["parents"])
            coef = params[child]["coef"].copy()
            intercept = float(params[child]["intercept"][0])

            target_parent = chosen_parent[child]
            if target_parent in parent_names:
                parent_idx = parent_names.index(target_parent)
                coef[parent_idx] = coef[parent_idx] + drift_per_step * step
                changed_edges[child] = target_parent

            parent_vals = z.loc[t, parent_names].to_numpy(dtype=float)
            noise = rng.normal(0.0, residual_std[child])
            z.loc[t, child] = float(intercept + np.dot(parent_vals, coef) + noise)

    return z, anomaly_mask, changed_edges


def save_synthetic(
    original_df: pd.DataFrame,
    z_df: pd.DataFrame,
    mu: pd.Series,
    sigma: pd.Series,
    anomaly_mask: np.ndarray,
    out_path: Path,
    scenario: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed = inv_zscore(z_df, mu, sigma)
    out = original_df.copy()
    for col in reconstructed.columns:
        out[col] = reconstructed[col].values
    out["is_anomaly"] = anomaly_mask
    out["anomaly_scenario"] = scenario
    out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Seoul_2 multivariate time-series anomalies.")
    parser.add_argument("--input", type=Path, required=True, help="Path to Seoul_2.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/synthetic"),
        help="Directory to save synthetic datasets",
    )
    parser.add_argument("--mean-shift-target", type=str, default="Inflow")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.input)
    cols = numeric_columns(df)
    z_df, mu, sigma = zscore_fit_transform(df, cols)
    z_df = z_df.fillna(0.0)

    window = default_windows(len(df), train_ratio=0.6, test_end_ratio=0.8)

    if args.mean_shift_target not in z_df.columns:
        raise ValueError(f"Mean shift target '{args.mean_shift_target}' not found in numeric columns: {cols}")

    for delta in [1.0, 5.0, 10.0]:
        shifted_z, mask = synthesize_mean_shift(
            base_z=z_df,
            target_col=args.mean_shift_target,
            window=window,
            delta=delta,
        )
        save_synthetic(
            original_df=df,
            z_df=shifted_z,
            mu=mu,
            sigma=sigma,
            anomaly_mask=mask,
            out_path=args.output_dir / f"Seoul_2_synth_mean_shift_delta_{int(delta)}.csv",
            scenario=f"mean_shift_delta_{delta}",
        )

    structural_z, struct_mask, changed_edges = synthesize_structural_change(
        base_z=z_df,
        window=window,
        drift_per_step=0.01,
        seed=args.seed,
    )
    save_synthetic(
        original_df=df,
        z_df=structural_z,
        mu=mu,
        sigma=sigma,
        anomaly_mask=struct_mask,
        out_path=args.output_dir / "Seoul_2_synth_structural_change.csv",
        scenario="structural_change_drift_0.01",
    )

    summary = pd.DataFrame(
        {
            "n_rows": [len(df)],
            "window_start_index": [window.s_idx],
            "window_end_index": [window.e_idx],
            "window_start_date": [df.loc[window.s_idx, "Date"]],
            "window_end_date": [df.loc[window.e_idx, "Date"]],
            "mean_shift_target": [args.mean_shift_target],
            "structural_changed_edges": [", ".join([f"{k}<-{v}" for k, v in changed_edges.items()])],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "Seoul_2_synth_summary.csv", index=False)

    print("Synthetic datasets generated in:", args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
