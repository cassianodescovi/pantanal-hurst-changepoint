"""
03_hurst.py

Hurst-exponent estimates for the Ladario stage record, with their sensitivity analyses.

Input : data/processed/annual_mean_stage_ladario.csv     (from 01_preprocessing)
        data/processed/daily_stage_ladario.csv            (daily-scale DFA only)
Output: results/tables/hurst_estimates.csv         DFA (daily and annual) and R/S estimates
        results/tables/gph_sensitivity.csv         GPH estimates for each bandwidth gamma
        results/tables/bootstrap_block.csv         moving-block bootstrap of the R/S estimate
        results/tables/bootstrap_parametric.csv    parametric (fGn) bootstrap of the R/S estimate
        results/tables/dfa_scaling.csv             log n, log F(n) of the annual DFA-1 (for figures)
        results/tables/rs_scaling.csv              window, R/S of the annual series (for figures)

Estimators
  R/S  : hurst.compute_Hc(kind="change", simplified=True)
  DFA  : nolds.dfa; alpha is H for alpha < 1 and H = alpha - 1 for 1 < alpha < 2
  GPH  : log-periodogram regression on the m = int(n**gamma) lowest Fourier frequencies, H = d + 0.5

Usage
  python scripts/03_hurst.py [--processed DIR] [--out DIR]
"""
import argparse
from importlib.metadata import version
from pathlib import Path

import nolds
import numpy as np
import pandas as pd
from fbm import FBM
from hurst import compute_Hc
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
GPH_GAMMAS = (0.4, 0.5, 0.6, 0.7)
GPH_VALID_MAX_GAMMA = 0.6            # gamma = 0.7 violates m << n and is excluded in the manuscript
BLOCK_LENGTHS = (5, 10, 15)
N_BOOT_BLOCK = 500
N_BOOT_PARAM = 1000

# Values reported in the manuscript, printed next to the results for comparison only.
PAPER = {
    "DFA daily raw": 1.486, "DFA daily anomaly": 1.533, "DFA-1 annual": 0.937, "DFA-2 annual": 1.030,
    "R/S annual": 0.828, "GPH 0.4": 0.966, "GPH 0.5": 0.911, "GPH 0.6": 0.830, "GPH 0.7": 1.051,
    "block b=5": 0.914, "block b=10": 0.872, "block b=15": 0.869, "parametric": 0.712,
}


# ------------------------------------------------------------------ estimators
def rs_hurst(x):
    """Classical R/S. Returns H and the (window, R/S) scaling data."""
    h, _, (windows, rs) = compute_Hc(np.asarray(x, float), kind="change", simplified=True)
    return float(h), np.asarray(windows), np.asarray(rs)


def dfa_alpha(x, order=1, fit_exp="RANSAC"):
    """DFA scaling exponent and its log-log data."""
    alpha, (log_n, log_f, _) = nolds.dfa(np.asarray(x, float), order=order, fit_exp=fit_exp, debug_data=True)
    return float(alpha), np.asarray(log_n), np.asarray(log_f)


def gph_hurst(x, gamma):
    """Geweke-Porter-Hudak estimator. Returns d, H, and the number of frequencies used."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    freqs, power = periodogram(x)
    freqs, power = freqs[1:], power[1:]               # drop the zero frequency
    m = int(n ** gamma)
    lam = 2 * np.pi * freqs[:m]
    slope, _ = np.polyfit(np.log(4 * np.sin(lam / 2) ** 2), np.log(power[:m]), 1)
    d = -slope
    return float(d), float(d + 0.5), m


# ------------------------------------------------------------------ bootstraps
def block_bootstrap(x, block, n_boot, rng):
    """Moving-block bootstrap of the R/S estimate."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        try:
            out.append(rs_hurst(sample)[0])
        except Exception:
            continue
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), len(out)


def parametric_bootstrap(h_imposed, n, n_boot):
    """Re-estimate R/S on fGn series with a known, imposed H (Davies-Harte)."""
    out = []
    for _ in range(n_boot):
        series = FBM(n=n, hurst=h_imposed, length=n, method="daviesharte").fgn()
        try:
            out.append(rs_hurst(series)[0])
        except Exception:
            continue
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), len(out)


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Hurst-exponent estimation for the annual stage series.")
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()

    annual_path = args.processed / "annual_mean_stage_ladario.csv"
    daily_path = args.processed / "daily_stage_ladario.csv"
    for p in (annual_path, daily_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run scripts/01_preprocessing.py first.")
    args.out.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(annual_path)["stage_cm"].to_numpy(dtype=float)
    daily = pd.read_csv(daily_path, parse_dates=["date"], index_col="date")
    n = len(annual)
    print(f"nolds {version('nolds')}, hurst {version('hurst')}, fbm {version('fbm')}; annual series n={n}\n")

    rows = []

    # --- daily-scale DFA (non-stationary range: alpha > 1)
    np.random.seed(SEED)
    for label, col in (("DFA daily raw", "stage_cm"), ("DFA daily anomaly", "anomaly_cm")):
        a, _, _ = dfa_alpha(daily[col].dropna().to_numpy())
        rows.append({"estimate": label, "value": a, "note": "alpha; > 1: non-stationary range"})

    # --- annual DFA-1 and DFA-2
    np.random.seed(SEED)
    a1, log_n, log_f = dfa_alpha(annual, order=1)
    a1_poly, _, _ = dfa_alpha(annual, order=1, fit_exp="poly")
    a2, _, _ = dfa_alpha(annual, order=2)
    rows += [
        {"estimate": "DFA-1 annual", "value": a1, "note": "alpha = H if alpha < 1"},
        {"estimate": "DFA-1 annual (least-squares fit)", "value": a1_poly, "note": "check of the default RANSAC fit"},
        {"estimate": "DFA-2 annual", "value": a2,
         "note": "crosses alpha = 1" if a2 >= 1 else "alpha = H if alpha < 1"},
    ]
    pd.DataFrame({"log_n": log_n, "log_F": log_f}).to_csv(args.out / "dfa_scaling.csv", index=False)

    # --- annual R/S
    h_rs, windows, rs = rs_hurst(annual)
    rows.append({"estimate": "R/S annual", "value": h_rs, "note": "H"})
    pd.DataFrame({"window": windows, "rs": rs}).to_csv(args.out / "rs_scaling.csv", index=False)

    pd.DataFrame(rows).to_csv(args.out / "hurst_estimates.csv", index=False)

    # --- GPH sensitivity to the bandwidth parameter
    gph_rows = []
    for g in GPH_GAMMAS:
        d, h, m = gph_hurst(annual, g)
        gph_rows.append({"gamma": g, "m": m, "m_over_n_pct": 100 * m / n, "d": d, "H": h,
                         "valid": g <= GPH_VALID_MAX_GAMMA})
    gph = pd.DataFrame(gph_rows)
    gph.to_csv(args.out / "gph_sensitivity.csv", index=False)

    # --- bootstraps
    rng = np.random.default_rng(SEED)
    block_rows = []
    for b in BLOCK_LENGTHS:
        mean, lo, hi, ok = block_bootstrap(annual, b, N_BOOT_BLOCK, rng)
        block_rows.append({"block": b, "H_mean": mean, "ci_low": lo, "ci_high": hi, "n_valid": ok})
    block = pd.DataFrame(block_rows)
    block.to_csv(args.out / "bootstrap_block.csv", index=False)

    h_imposed = round(h_rs, 3)
    np.random.seed(SEED)
    mean, lo, hi, ok = parametric_bootstrap(h_imposed, n, N_BOOT_PARAM)
    param = pd.DataFrame([{"H_imposed": h_imposed, "H_mean": mean, "ci_low": lo, "ci_high": hi, "n_valid": ok}])
    param.to_csv(args.out / "bootstrap_parametric.csv", index=False)

    # --- summary next to the manuscript values
    def line(label, value, paper_key):
        ref = PAPER.get(paper_key)
        print(f"  {label:<34s}{value:8.3f}   (paper: {ref:.3f})" if ref is not None else f"  {label:<34s}{value:8.3f}")

    print("Estimates (this run, manuscript value in parentheses):")
    for r in rows:
        line(r["estimate"], r["value"], r["estimate"])
    for _, r in gph.iterrows():
        tag = "" if r["valid"] else "  [excluded: m << n violated]"
        print(f"  GPH gamma={r['gamma']:.1f}  m={int(r['m']):2d} ({r['m_over_n_pct']:.1f}% of n)  "
              f"d={r['d']:.3f}  H={r['H']:.3f}   (paper: {PAPER['GPH ' + str(r['gamma'])]:.3f}){tag}")
    for _, r in block.iterrows():
        print(f"  block bootstrap b={int(r['block']):2d}  H={r['H_mean']:.3f}  95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
              f"   (paper: {PAPER['block b=' + str(int(r['block']))]:.3f})")
    r = param.iloc[0]
    print(f"  parametric bootstrap (H imposed {r['H_imposed']:.3f})  mean={r['H_mean']:.3f}  "
          f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]   (paper: {PAPER['parametric']:.3f}, CI [0.427, 0.940])")
    print(f"\nTables written to {args.out}")


if __name__ == "__main__":
    main()