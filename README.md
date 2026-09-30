# Machine Learning Anomaly Detection Under Climate-Driven Cross-Sectional Dependency in Wastewater Systems

[![DOI](https://img.shields.io/badge/DOI-10.64697%2Fiahr.proc.hic2026.295-blue)](https://doi.org/10.64697/iahr.proc.hic2026.295)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code, data and results for the paper:

> Mahvelati Shamsabadi, A., & Kyung, D. (2026). *Machine Learning Anomaly Detection Under Climate-Driven Cross-Sectional Dependency in Wastewater Systems.* Proceedings of the 16th International Conference on Hydroinformatics (HIC 2026), Zaragoza, Spain, 22–26 June 2026. IAHR, pp. 323–330. **https://doi.org/10.64697/iahr.proc.hic2026.295**

**Topics:** anomaly-detection · wastewater-treatment · lstm-autoencoder · gaussian-copula · climate-resilience · hydroinformatics · time-series · unsupervised-learning

## Overview

Climate extremes can stress wastewater treatment plants (WWTPs) through coupled flow–quality interactions that univariate alarms miss. The study evaluates two unsupervised, dependency-aware detectors on two Seoul municipal plants (P1, P2):

- **CS-LSTM AE** – cross-sectional LSTM autoencoder on 14-day joint trajectories (score = reconstruction error).
- **GC** – Gaussian copula on same-day multivariate vectors (score = negative log-likelihood).

Evaluation uses rolling-origin out-of-sample scoring, normal / hot / wet climate-regime stratification (90th-percentile temperature and precipitation), and threshold robustness (q = 0.95, 0.975, 0.99). Main finding: CS-LSTM AE is more sensitive to sustained heat-linked trajectory deviations, GC to wet-event dependency shifts, and the amplification persists across thresholds.

## Repository layout

```
code/
  models/           cs_lstm_autoencoder.py, gaussian_copula_detector.py
  preprocessing/    data_pipeline.py (KNN imputation, sequences, climate groups),
                    synthesize_seoul2_timeseries.py (synthetic benchmark)
  evaluation/       evaluate.py (thresholding, climate-conditioned rates)
  visualization/    make_fig2.py, make_fig3_case_study.py, make_fig4_threshold_sensitivity.py, figures.py
  run_two_real_datasets_experiments.py   main paper experiments (P1 + P2)
  run_hic2026_pipeline.py                single split (80/20) pipeline
  run_real_climate_robustness.py         rolling-origin robustness on P1
  run_dual_dataset_benchmark.py          synthetic benchmark (mean shift / structural change)
data/               NOT included (place input CSVs here; see below)
results/            score tables, per-fold thresholds, logs, trained models
tables/             CSV tables
figures/            paper figures (PNG/PDF)
```

## Setup

Python 3.10+ (developed on 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper results

Steps 1 needs the (unshared) input data; step 2 works from the included results. Run everything from the `code/` directory. Seeds are fixed (`--seed 42`); a CPU is sufficient, but PyTorch results can differ slightly across hardware and library versions.

```bash
cd code

# 1. Main experiments: both plants, rolling-origin, thresholds 0.95/0.975/0.99
python run_two_real_datasets_experiments.py       # -> results/two_real_datasets/

# 2. Paper figures (run from the repository root)
cd ..
python code/visualization/make_fig2.py            # climate-conditioned rates, q = 0.99
python code/visualization/make_fig3_case_study.py # time-series case study (P1, P2)
python code/visualization/make_fig4_threshold_sensitivity.py
```

Optional analyses (all require the data files above):

```bash
cd code
python run_hic2026_pipeline.py                    # single chronological 80/20 split
python run_real_climate_robustness.py             # rolling-origin robustness check
python run_dual_dataset_benchmark.py              # synthetic benchmark + ROC/PR metrics
python preprocessing/synthesize_seoul2_timeseries.py --input ../data/Seoul_2.csv --output-dir ../data/processed/synthetic
```

Pre-computed model outputs (scores, thresholds, rates) are included under `results/`, `tables/` and `figures/`, so the figures can be regenerated without retraining.

## Methods in brief

- Missing/malformed values → NaN → distance-weighted KNN imputation.
- Z-score scaling fitted on each training window only (no temporal leakage).
- Unsupervised thresholds: quantile *q* of training scores.
- Hot / Wet days: above the 90th percentile of daily mean temperature / precipitation; all others Normal.

## Data note

**The plant datasets are not distributed in this repository.** To re-run the experiments from scratch, place the daily plant/climate CSVs at `data/Seoul_Tancheon_WWTP.csv` (P1) and `data/Seoul3.csv` (P2), and optionally `data/Seoul_2.csv` for the synthetic benchmark, with the column names expected in `code/run_two_real_datasets_experiments.py`. Data are available from the corresponding author on reasonable request, subject to the data provider's permission.

## Citation

Use GitHub's **"Cite this repository"** button (from [`CITATION.cff`](CITATION.cff)) or:

```bibtex
@inproceedings{mahvelati2026wwtp,
  author    = {Mahvelati Shamsabadi, Alireza and Kyung, Daeseung},
  title     = {Machine Learning Anomaly Detection Under Climate-Driven Cross-Sectional Dependency in Wastewater Systems},
  booktitle = {Proceedings of the 16th International Conference on Hydroinformatics (HIC 2026)},
  publisher = {IAHR},
  address   = {Zaragoza, Spain},
  year      = {2026},
  pages     = {323--330},
  doi       = {10.64697/iahr.proc.hic2026.295}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper itself is © IAHR 2026 and is not redistributed here.

## Contact

Alireza Mahvelati Shamsabadi · Daeseung Kyung (corresponding author, dkyung@ulsan.ac.kr), School of Civil & Environmental Engineering, University of Ulsan.
