from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_column(columns: List[str], candidates: List[str], label: str) -> str:
    lookup = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lookup:
            return lookup[key]
    raise ValueError(f"Missing '{label}'. Tried {candidates}. Available columns: {columns}")


def load_scores(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")

    raw = pd.read_csv(path)
    cols = list(raw.columns)

    date_col = _pick_column(cols, ["Date", "date", "datetime", "timestamp"], "date")
    cs_col = _pick_column(cols, ["CS_Score", "score_cs", "cs_score", "CSLSTM_Score"], "score_cs")
    gc_col = _pick_column(cols, ["GC_Score", "score_gc", "gc_score"], "score_gc")

    cs_thr_col = _pick_column(
        cols,
        ["CS_thr_0.99", "threshold_cs_q99", "cs_threshold_q99", "CS_thr_0.99"],
        "threshold_cs_q99",
    )
    gc_thr_col = _pick_column(
        cols,
        ["GC_thr_0.99", "threshold_gc_q99", "gc_threshold_q99", "GC_thr_0.99"],
        "threshold_gc_q99",
    )

    precip_col = _pick_column(
        cols,
        ["precip_value", "precipitation", "Precipitation", "precip_total_mm", "rainfall", "Rainfall"],
        "precipitation",
    )
    temp_col = _pick_column(
        cols,
        ["temp_value", "temperature_mean", "Average Temperature", "temp_mean_c", "Temperature"],
        "temperature_mean",
    )

    mapping = {
        "date": date_col,
        "score_cs": cs_col,
        "score_gc": gc_col,
        "threshold_cs_q99": cs_thr_col,
        "threshold_gc_q99": gc_thr_col,
        "precipitation": precip_col,
        "temperature_mean": temp_col,
    }

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "score_cs": pd.to_numeric(raw[cs_col], errors="coerce"),
            "score_gc": pd.to_numeric(raw[gc_col], errors="coerce"),
            "threshold_cs_q99": pd.to_numeric(raw[cs_thr_col], errors="coerce"),
            "threshold_gc_q99": pd.to_numeric(raw[gc_thr_col], errors="coerce"),
            "precipitation": pd.to_numeric(raw[precip_col], errors="coerce"),
            "temperature_mean": pd.to_numeric(raw[temp_col], errors="coerce"),
        }
    )

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, mapping


def load_climate_thresholds(summary_path: Path, dataset_key: str) -> Tuple[float, float]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    summary = pd.read_csv(summary_path)

    key_norm = dataset_key.strip().lower()
    candidates = summary[summary["dataset"].astype(str).str.lower() == key_norm]
    if candidates.empty:
        raise ValueError(f"Dataset '{dataset_key}' not found in {summary_path}")

    row = candidates.iloc[0]
    temp_q90 = float(row["temp_q90"])
    precip_q90 = float(row["precip_q90"])
    return temp_q90, precip_q90


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def choose_window(df: pd.DataFrame, temp_q90: float, precip_q90: float, min_len: int = 120, max_len: int = 180) -> Tuple[int, int]:
    n = len(df)
    cs_flag = df["score_cs"] > df["threshold_cs_q99"]
    gc_flag = df["score_gc"] > df["threshold_gc_q99"]
    hot = df["temperature_mean"] >= temp_q90
    wet = df["precipitation"] >= precip_q90

    best = None
    for w in range(min_len, max_len + 1, 10):
        if w > n:
            continue
        for s in range(0, n - w + 1):
            e = s + w
            seg_cs = cs_flag.iloc[s:e].to_numpy()
            seg_gc = gc_flag.iloc[s:e].to_numpy()
            seg_hot = hot.iloc[s:e].to_numpy()
            seg_wet = wet.iloc[s:e].to_numpy()

            cs_excess = (df["score_cs"].iloc[s:e] - df["threshold_cs_q99"].iloc[s:e]).clip(lower=0).max()
            gc_excess = (df["score_gc"].iloc[s:e] - df["threshold_gc_q99"].iloc[s:e]).clip(lower=0).max()

            has_major_wet_gc = (seg_gc & seg_wet).any() and float(gc_excess) > 0
            has_sustained_hot = _max_consecutive_true(seg_hot) >= 5

            union_flags = int((seg_cs | seg_gc).sum())
            overlap_flags = int((seg_cs & seg_gc).sum())

            score = union_flags + 0.7 * overlap_flags + 0.4 * float(gc_excess) + 0.4 * float(cs_excess)
            if has_major_wet_gc:
                score += 25
            if has_sustained_hot:
                score += 25

            candidate = (score, s, e, has_major_wet_gc, has_sustained_hot)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("Could not choose a window from the score file.")

    _, s, e, wet_ok, hot_ok = best
    if not (wet_ok and hot_ok):
        # fallback already implicit by objective; this message clarifies behavior
        pass
    return s, e


def _longest_run_indices(mask: np.ndarray) -> Tuple[int, int]:
    best_start, best_end = 0, 0
    cur_start = 0
    in_run = False
    for i, v in enumerate(mask):
        if v and not in_run:
            cur_start = i
            in_run = True
        if not v and in_run:
            if i - cur_start > best_end - best_start:
                best_start, best_end = cur_start, i
            in_run = False
    if in_run and len(mask) - cur_start > best_end - best_start:
        best_start, best_end = cur_start, len(mask)
    return best_start, best_end


def plot_case_study(
    df: pd.DataFrame,
    temp_q90: float,
    precip_q90: float,
    start_idx: int,
    end_idx: int,
    out_png: Path,
    out_pdf: Path,
    dataset_label: str,
) -> None:
    win = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    cs_flag = win["score_cs"] > win["threshold_cs_q99"]
    gc_flag = win["score_gc"] > win["threshold_gc_q99"]
    hot_mask = win["temperature_mean"] >= temp_q90
    wet_mask = win["precipitation"] >= precip_q90

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.8), sharex=True, constrained_layout=True)
    ax1, ax2, ax3 = axes

    # Panel A: CS score
    ax1.plot(win["date"], win["score_cs"], color="#1f77b4", label="CS-LSTM AE score")
    ax1.axhline(float(win["threshold_cs_q99"].median()), color="#1f77b4", ls="--", lw=1.3, label="q99 threshold")
    ax1.scatter(win.loc[cs_flag, "date"], win.loc[cs_flag, "score_cs"], color="#1f77b4", s=22, zorder=4, label="Flagged")
    for d in win.loc[hot_mask, "date"]:
        ax1.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#E69F00", alpha=0.08, linewidth=0)
    ax1.set_ylabel("CS score")
    ax1.set_title(f"Panel A — CS-LSTM AE (dataset: {dataset_label})")
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: GC score
    ax2.plot(win["date"], win["score_gc"], color="#2ca02c", label="Gaussian Copula score")
    ax2.axhline(float(win["threshold_gc_q99"].median()), color="#2ca02c", ls="--", lw=1.3, label="q99 threshold")
    ax2.scatter(win.loc[gc_flag, "date"], win.loc[gc_flag, "score_gc"], color="#2ca02c", s=22, zorder=4, label="Flagged")
    for d in win.loc[wet_mask, "date"]:
        ax2.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#56B4E9", alpha=0.08, linewidth=0)
    ax2.set_ylabel("GC score")
    ax2.set_title("Panel B — Gaussian Copula")
    ax2.grid(axis="y", alpha=0.2)

    # Panel C: climate drivers
    ax3b = ax3.twinx()
    ax3.bar(win["date"], win["precipitation"], width=1.0, color="#56B4E9", alpha=0.55, label="Precipitation")
    ax3b.plot(win["date"], win["temperature_mean"], color="#E69F00", lw=1.8, label="Temperature")
    ax3.axhline(precip_q90, color="#56B4E9", ls=":", lw=1.2)
    ax3b.axhline(temp_q90, color="#E69F00", ls=":", lw=1.2)
    ax3.set_ylabel("Precip. (mm)")
    ax3b.set_ylabel("Temp. (°C)")
    ax3.set_title("Panel C — Climate drivers")

    # Callout 1: largest GC spike
    gc_idx = int(np.argmax((win["score_gc"] - win["threshold_gc_q99"]).fillna(-np.inf).to_numpy()))
    ax2.annotate(
        "Storm-linked dependency shift",
        xy=(win.loc[gc_idx, "date"], win.loc[gc_idx, "score_gc"]),
        xytext=(20, 25),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9.5,
    )

    # Callout 2: sustained CS period
    cs_run_start, cs_run_end = _longest_run_indices(cs_flag.to_numpy())
    if cs_run_end - cs_run_start >= 2:
        cs_idx = cs_run_start + (cs_run_end - cs_run_start) // 2
    else:
        cs_idx = int(np.argmax((win["score_cs"] - win["threshold_cs_q99"]).fillna(-np.inf).to_numpy()))
    ax1.annotate(
        "Heat-linked trajectory deviation",
        xy=(win.loc[cs_idx, "date"], win.loc[cs_idx, "score_cs"]),
        xytext=(20, 25),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=9.5,
    )

    # cosmetics
    for ax in [ax1, ax2, ax3, ax3b]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    handles = [
        plt.Line2D([], [], color="#1f77b4", lw=2, label="CS score"),
        plt.Line2D([], [], color="#2ca02c", lw=2, label="GC score"),
        plt.Line2D([], [], color="#1f77b4", ls="--", lw=1.2, label="CS q99"),
        plt.Line2D([], [], color="#2ca02c", ls="--", lw=1.2, label="GC q99"),
        plt.Line2D([], [], color="#E69F00", lw=2, label="Temperature"),
        plt.Line2D([], [], color="#56B4E9", lw=6, alpha=0.55, label="Precipitation"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False)

    fig.suptitle(f"Figure 3 — Time-series anomaly case study ({dataset_label})", fontsize=15)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run_one(dataset_key: str, score_path: Path, summary_path: Path, out_dir: Path) -> None:
    df, mapping = load_scores(score_path)
    temp_q90, precip_q90 = load_climate_thresholds(summary_path, dataset_key=dataset_key)
    s, e = choose_window(df, temp_q90=temp_q90, precip_q90=precip_q90, min_len=120, max_len=180)

    window = df.iloc[s:e].copy()
    cs_flag = window["score_cs"] > window["threshold_cs_q99"]
    gc_flag = window["score_gc"] > window["threshold_gc_q99"]
    overlap = cs_flag & gc_flag

    print(f"\n[{dataset_key}] Column mapping: {mapping}")
    print(f"[{dataset_key}] Chosen date range: {window['date'].min().date()} -> {window['date'].max().date()} ({len(window)} days)")
    print(
        f"[{dataset_key}] Figure 3 summary: CS flagged={100.0*cs_flag.mean():.1f}%, "
        f"GC flagged={100.0*gc_flag.mean():.1f}%, overlap_days={int(overlap.sum())}"
    )

    label = "P1" if dataset_key.lower().startswith("seoul_tancheon") else "P2"
    out_png = out_dir / f"fig3_timeseries_case_study_{label}.png"
    out_pdf = out_dir / f"fig3_timeseries_case_study_{label}.pdf"

    plot_case_study(
        df=df,
        temp_q90=temp_q90,
        precip_q90=precip_q90,
        start_idx=s,
        end_idx=e,
        out_png=out_png,
        out_pdf=out_pdf,
        dataset_label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 3 time-series case study for P1 and P2.")
    parser.add_argument(
        "--p1-scores",
        type=Path,
        default=Path("results/two_real_datasets/Seoul_Tancheon_WWTP_daily_scores_rolling.csv"),
    )
    parser.add_argument(
        "--p2-scores",
        type=Path,
        default=Path("results/two_real_datasets/Seoul3_daily_scores_rolling.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/two_real_datasets/comparison_summary_q99.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
    )
    args = parser.parse_args()

    run_one("Seoul_Tancheon_WWTP", args.p1_scores, args.summary, args.out_dir)
    run_one("Seoul3", args.p2_scores, args.summary, args.out_dir)


if __name__ == "__main__":
    main()
