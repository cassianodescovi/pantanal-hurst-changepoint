"""
04_montecarlo.py

Monte Carlo experiments on the confound between structural breaks and long-range dependence.

  1. False-positive rate of the Pettitt test on fractional Gaussian noise with NO break
     (manuscript Section 2.5.1). Reported with two decision rules: the Eq. 3 approximation
     (as in the manuscript) and the exact critical value (the rule behind the Monte Carlo
     p-values used on the real series).
  2. "Shift-only" null (Section 2.5.2): independent Gaussian noise inside each regime, with
     the regime means and SDs of the real series, so that the only structure is the level
     shifts. Run for three regime structures: sequential Pettitt (Table 2), Pettitt on both
     sides, and PELT.
  3. Regime-level DFA, autocorrelations and white-noise calibration (Section 2.5.3).

Input : data/processed/annual_mean_stage_ladario.csv           (01_preprocessing)
        results/tables/regimes_pettitt_sequential.csv           (02_changepoints)
        results/tables/pettitt_binary_segmentation.csv          (02_changepoints)
        results/tables/pelt_sensitivity.csv                     (02_changepoints)
Output: results/tables/fp_rate.csv, shift_only_null.csv, regime_level.csv

Usage
  python scripts/04_montecarlo.py [--annual PATH] [--tables DIR] [--out DIR]
                                  [--m-fp N] [--m-shift N] [--m-cal N]
"""
import argparse
import warnings
from importlib.metadata import version
from pathlib import Path

import nolds
import numpy as np
import pandas as pd
from fbm import FBM
from hurst import compute_Hc
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
ALPHA = 0.05
H_GRID = (0.50, 0.65, 0.75, 0.85)
N_PERM_CRIT = 100000          # permutations for the exact critical value
MIN_N_REGIME_DFA = 15         # shortest regime on which DFA is attempted

# Values reported in the manuscript, printed for comparison only.
PAPER_FP = {0.50: 4.0, 0.65: 29.8, 0.75: 56.2, 0.85: 74.1}


# ------------------------------------------------------------------ helpers
def rs_h(x):
    return float(compute_Hc(np.asarray(x, float), kind="change", simplified=True)[0])


def dfa_alpha(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # nolds warns when its RANSAC fit falls back to polyfit
        return float(nolds.dfa(np.asarray(x, float)))


def pettitt_k(x):
    """Pettitt statistic K = max |U_t| computed from ranks."""
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    return np.max(np.abs(u))


def critical_k_eq3(n, alpha=ALPHA):
    """K at which the approximation p = 2 exp(-6 K^2 / (n^3 + n^2)) equals alpha."""
    return np.sqrt(-np.log(alpha / 2) * (n ** 3 + n ** 2) / 6)


def critical_k_exact(n, rng, alpha=ALPHA):
    """(1 - alpha) quantile of K under no change (permutations of the ranks)."""
    ks = np.empty(N_PERM_CRIT)
    for i in range(N_PERM_CRIT):
        r = rng.permutation(n) + 1
        ks[i] = np.max(np.abs(2 * np.cumsum(r) - np.arange(1, n + 1) * (n + 1)))
    return np.quantile(ks, 1 - alpha)


def regimes_from_breaks(values, years, break_idx):
    edges = [0] + sorted(break_idx) + [len(values)]
    rows = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        seg = values[a:b]
        rows.append({"regime": f"R{i}", "start": years[a], "end": years[b - 1], "n": b - a,
                     "mean_cm": seg.mean(), "sd_cm": seg.std(ddof=1) if len(seg) > 1 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ experiment 1
def false_positive_rates(n, m, rng):
    k_eq3 = critical_k_eq3(n)
    k_exact = critical_k_exact(n, rng)
    rows = []
    for h in H_GRID:
        ks = np.array([pettitt_k(FBM(n=n, hurst=h, length=n, method="daviesharte").fgn()) for _ in range(m)])
        for rule, kc in (("Eq. 3 approximation", k_eq3), ("exact critical value", k_exact)):
            p = float(np.mean(ks > kc))
            rows.append({"H_imposed": h, "decision_rule": rule, "critical_K": kc, "rate_pct": 100 * p,
                         "se_pct": 100 * np.sqrt(p * (1 - p) / m), "M": m})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ experiment 2
def shift_only_null(regimes, m, rng):
    """H estimates on independent Gaussian noise with the regimes' means and SDs."""
    alphas, hs = [], []
    for _ in range(m):
        x = np.concatenate([rng.normal(r.mean_cm, r.sd_cm, int(r.n)) for r in regimes.itertuples()])
        alphas.append(dfa_alpha(x))
        hs.append(rs_h(x))
    return np.array(alphas), np.array(hs)


def structures(annual_values, years, tables):
    """Three regime structures: sequential Pettitt, Pettitt on both sides, PELT (jump=1)."""
    out = {"Pettitt sequential": pd.read_csv(tables / "regimes_pettitt_sequential.csv")}

    both = pd.read_csv(tables / "pettitt_binary_segmentation.csv")
    idx = sorted(both.loc[both["significant"], "break_index"].astype(int).unique())
    out["Pettitt both sides"] = regimes_from_breaks(annual_values, years, idx)

    pelt = pd.read_csv(tables / "pelt_sensitivity.csv")
    pelt = pelt[(pelt["jump"] == 1) & (pelt["n_breaks"] > 0)]
    modal = pelt["break_years"].mode().iloc[0]                   # break set found over the widest range of c
    idx = [int(np.where(years == int(y))[0][0]) for y in modal.split(";")]
    out["PELT (jump=1)"] = regimes_from_breaks(annual_values, years, idx)
    return out


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo experiments.")
    parser.add_argument("--annual", type=Path, default=ROOT / "data" / "processed" / "annual_mean_stage_ladario.csv")
    parser.add_argument("--tables", type=Path, default=ROOT / "results" / "tables", help="tables written by 02_changepoints")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: --tables)")
    parser.add_argument("--m-fp", type=int, default=1000)
    parser.add_argument("--m-shift", type=int, default=500)
    parser.add_argument("--m-cal", type=int, default=1000)
    args = parser.parse_args()
    out = args.out or args.tables

    needed = [args.annual] + [args.tables / f for f in
              ("regimes_pettitt_sequential.csv", "pettitt_binary_segmentation.csv", "pelt_sensitivity.csv")]
    for p in needed:
        if not p.exists():
            raise SystemExit(f"{p} not found. Run scripts/01_preprocessing.py and scripts/02_changepoints.py first.")
    out.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(args.annual)
    values, years, n = annual["stage_cm"].to_numpy(float), annual["year"].to_numpy(int), len(annual)
    print(f"nolds {version('nolds')}, hurst {version('hurst')}, fbm {version('fbm')}; n={n}\n")

    # ---- 1. false-positive rates
    rng = np.random.default_rng(SEED); np.random.seed(SEED)
    fp = false_positive_rates(n, args.m_fp, rng)
    fp.to_csv(out / "fp_rate.csv", index=False)
    print(f"1. Pettitt false-positive rate on fGn without a break (M={args.m_fp})")
    for h in H_GRID:
        a = fp[(fp.H_imposed == h) & (fp.decision_rule == "Eq. 3 approximation")].iloc[0]
        b = fp[(fp.H_imposed == h) & (fp.decision_rule == "exact critical value")].iloc[0]
        print(f"   H={h:.2f}  Eq.3: {a.rate_pct:5.1f}% (SE {a.se_pct:.1f})  exact: {b.rate_pct:5.1f}% (SE {b.se_pct:.1f})"
              f"   (paper, Eq.3: {PAPER_FP[h]:.1f}%)")

    # ---- 2. shift-only null
    obs_dfa, obs_rs = dfa_alpha(values), rs_h(values)
    rows = []
    print(f"\n2. Shift-only null (M={args.m_shift}); observed: DFA alpha {obs_dfa:.3f}, R/S H {obs_rs:.3f}")
    for name, regimes in structures(values, years, args.tables).items():
        rng = np.random.default_rng(SEED); np.random.seed(SEED)
        alphas, hs = shift_only_null(regimes, args.m_shift, rng)
        breaks = "; ".join(str(y) for y in regimes["start"].iloc[1:])
        for est, obs, null in (("DFA (alpha)", obs_dfa, alphas), ("R/S (H)", obs_rs, hs)):
            lo, hi = np.percentile(null, [5, 95])
            rows.append({"structure": name, "breaks": breaks, "n_regimes": len(regimes), "estimator": est,
                         "observed": obs, "null_mean": null.mean(), "null_p05": lo, "null_p95": hi,
                         "observed_above_p95": bool(obs > hi), "share_alpha_above_1": float(np.mean(null > 1)) if est.startswith("DFA") else np.nan})
            print(f"   {name:<20s} ({breaks})\n      {est:<12s} null mean {null.mean():.3f}, 90% [{lo:.3f}, {hi:.3f}]"
                  f"  -> observed {'ABOVE' if obs > hi else 'inside'}")
    pd.DataFrame(rows).to_csv(out / "shift_only_null.csv", index=False)

    # ---- 3. regime-level estimates and white-noise calibration
    regimes = pd.read_csv(args.tables / "regimes_pettitt_sequential.csv")
    rng = np.random.default_rng(SEED); np.random.seed(SEED)
    rows = []
    print(f"\n3. Regime-level DFA and calibration against white noise (M={args.m_cal})")
    for r in regimes.itertuples():
        seg = values[(years >= r.start) & (years <= r.end)]
        row = {"regime": r.regime, "start": r.start, "end": r.end, "n": len(seg)}
        if len(seg) < MIN_N_REGIME_DFA:
            row["note"] = f"n < {MIN_N_REGIME_DFA}: not estimated"
            print(f"   {r.regime} {r.start}-{r.end} n={len(seg)}: too short, not estimated")
        else:
            ac = acf(seg, nlags=2, fft=False)
            null = np.array([dfa_alpha(rng.normal(size=len(seg))) for _ in range(args.m_cal)])
            lo, hi = np.percentile(null, [5, 95])
            row.update(dfa_alpha=dfa_alpha(seg), acf1=ac[1], acf2=ac[2], null_mean=null.mean(), null_p05=lo, null_p95=hi)
            row["within_null_ci"] = bool(lo <= row["dfa_alpha"] <= hi)
            print(f"   {r.regime} {r.start}-{r.end} n={len(seg)}: DFA {row['dfa_alpha']:.3f}, ACF(1) {ac[1]:.3f}, ACF(2) {ac[2]:.3f}; "
                  f"white noise {null.mean():.3f} [{lo:.3f}, {hi:.3f}] -> {'inside' if row['within_null_ci'] else 'OUTSIDE'}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "regime_level.csv", index=False)
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()