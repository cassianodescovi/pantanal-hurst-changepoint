# pantanal-hurst-changepoint

Code and processed data for:

> **Regime shifts or long-range dependence? Disentangling hydrological persistence in 126 years of Paraguay River stage data**
> Cassiano Sampaio Descovi, Antonio Carlos Zuffo, SeyedMehdi Mohammadizadeh
> *Journal of Hydrology* 

This repository contains the analysis pipeline used to test whether the regime shifts reported in the Pantanal hydrological literature can be statistically distinguished from long-range dependence (the Hurst phenomenon), using the daily river-stage record of the Paraguay River at Ladário (Mato Grosso do Sul, Brazil), 1900–2025.

## What this repository reproduces

- Preprocessing of the Hidroweb stage record, deseasonalization, and annual aggregation
- Change-point detection: recursive Pettitt test (with Bonferroni correction) and PELT with penalty sensitivity; BIC comparison of candidate break sets
- Hurst-exponent estimation: rescaled range (R/S), detrended fluctuation analysis (DFA-1 and DFA-2), and the GPH spectral estimator with bandwidth sensitivity
- Bidirectional Monte Carlo experiments: false-positive rate of the Pettitt test under fractional Gaussian noise, shift-only null distributions of H, and small-sample calibration of regime-level estimates
- Confidence intervals: moving block bootstrap and parametric (fGn) bootstrap
- Walk-forward neural-network residual-anomaly detector, with sensitivity to architecture, lag window, and flagging threshold, and a linear AR(p) baseline
- All figures and tables of the manuscript

## Repository structure

```
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   ├── README.md          data origin, period, and terms of use
│   ├── raw/               download instructions (raw files not redistributed unless permitted)
│   └── processed/         cleaned daily series and annual series
├── scripts/
│   ├── 01_preprocessing
│   ├── 02_changepoints
│   ├── 03_hurst
│   ├── 04_montecarlo
│   ├── 05_bootstrap
│   ├── 06_nn_detector
│   └── 07_figures_tables
├── results/
│   ├── tables/            tables as CSV
│   └── intermediate/      saved outputs of long-running steps
└── figures/               final figures (PNG and PDF)
```

## Data

The daily stage series for the Paraguay River at Ladário (ANA station code 66825000) comes from the Hidroweb system of the Brazilian National Water and Sanitation Agency (ANA): <https://www.snirh.gov.br/hidroweb/serieshistoricas>.

- Period: 1 January 1900 to 31 December 2025 (46,021 daily observations)
- Preprocessing: only daily-summary records are kept; where raw and consistency-reviewed versions of the same month exist, the more complete one is used; the 27 missing days (0.06%) are filled by linear interpolation (gaps of up to three days)
- Quality: 99.5% of days carry the ANA "real" flag; the remainder are mostly recent, not-yet-reviewed records (2024–2025)

Please cite ANA as the data source. See `data/README.md` for the redistribution status of the raw files.

## Installation

```
git clone https://github.com/cassianodescovi/pantanal-hurst-changepoint.git
cd pantanal-hurst-changepoint
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python version: 3.13.16. Exact package versions are pinned in `requirements.txt`. Main dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, nolds, hurst, fbm, pyhomogeneity, ruptures.

## Usage

Run the scripts in numerical order, from the repository root:

1. `01_preprocessing`: builds the cleaned daily series and the annual series in `data/processed/`
2. `02_changepoints`: Pettitt, Bonferroni, PELT, penalty sensitivity, BIC
3. `03_hurst`: R/S, DFA, GPH and their sensitivity analyses
4. `04_montecarlo`: false-positive curve, shift-only nulls, regime-level calibration
5. `05_bootstrap`: block and parametric bootstrap intervals
6. `06_nn_detector`: walk-forward neural-network detector, sensitivity grid, threshold sensitivity, AR baseline
7. `07_figures_tables`: regenerates all figures and tables from the saved results

## Reproducibility notes

- Random seeds are fixed at the top of each script. Monte Carlo and bootstrap results reproduce up to sampling variation; small differences in the last digit of reported rates are expected across platforms and package versions.
- The slowest step is the walk-forward neural-network detector (W = 60, R = 50 runs). Its outputs are stored in `results/intermediate/` so that figures and tables can be regenerated without re-running it.
- Two estimators have known sample-size behavior discussed in the paper: R/S is biased downward at n = 126, and DFA results are sensitive to the detrending order when the scaling exponent is near 1.

## License

The code is released under the MIT License (see `LICENSE`). Processed data derived from ANA records are shared under the terms stated in `data/README.md`; please cite ANA and this repository.

## How to cite

If you use this code, please cite the paper above and the archived repository: [Zenodo DOI to be added]. Citation metadata is provided in `CITATION.cff`.

## Contact

Cassiano Sampaio Descovi — [cassianodescovi@fec.unicamp.br]
School of Civil Engineering, Architecture and Urban Planning (FECFAU), University of Campinas (UNICAMP), Brazil