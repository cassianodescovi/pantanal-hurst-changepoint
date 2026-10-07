"""
02_changepoints.py

Change-point analysis of the annual mean stage series (Ladario, 1900-2025).

Input : data/processed/annual_mean_stage_ladario.csv      (from 01_preprocessing)
Output: results/tables/pettitt_sequential.csv             Pettitt tests behind Table 2
        results/tables/regimes_pettitt_sequential.csv     regime means/SDs (used by 04_montecarlo)
        results/tables/pettitt_binary_segmentation.csv    Pettitt applied to BOTH sub-segments
        results/tables/pelt_sensitivity.csv               PELT breaks for each penalty factor c
        results/tables/pelt_plateaus.csv                  ranges of c with identical break sets
        results/tables/bic_comparison.csv                 BIC of competing break sets

Conventions
  * Break year = year of the FIRST observation of the new regime, for every method
    (a 1974 break means the regimes are 1900-1973 and 1974-...).
  * Pettitt p-values are Monte Carlo p-values (pyhomogeneity default, 20,000 simulations),
    reported next to the asymptotic approximation of Eq. 3 (p_eq3) for comparison.
  * PELT is run on the standardized series with penalty lambda = c * log(n), cost "l2".
    It is run with jump=1 (every year is a candidate break) and, for comparison, with the
    ruptures default jump=5, which restricts breaks to years 1900 + 5k.

Usage
  python scripts/02_changepoints.py [--annual PATH] [--out DIR]
"""
import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyhomogeneity as hg
import ruptures as rpt

ROOT = Path(__file__).resolve().parent.parent

ALPHA = 0.05
SIM = 20000              # Monte Carlo simulations per Pettitt p-value
SEED = 42                # re-seeded before every Pettitt call
MIN_SEGMENT_LEN = 10     # shortest segment that is tested (value not stated in the manuscript)
PELT_C_GRID = sorted(set(np.round(np.arange(1.0, 5.01, 0.1), 1).tolist() + [6, 8, 10, 15, 20, 30]))
PELT_JUMPS = (1, 5)


# ----------------------------------------------------------------------------- Pettitt
def pettitt(values):
    """Pettitt test. `cp` is the 0-based index of the first observation of the new regime."""
    values = np.asarray(values, dtype=float)      # numpy input: pyhomogeneity then returns loc
    n = len(values)
    np.random.seed(SEED)
    res = hg.pettitt_test(values, alpha=ALPHA, sim=SIM)
    return {
        "cp": int(res.cp),
        "U": float(res.U),
        "p_mc": float(res.p),
        "p_eq3": float(min(1.0, 2 * np.exp(-6 * res.U**2 / (n**3 + n**2)))),
        "mean_before": float(res.avg.mu1),
        "mean_after": float(res.avg.mu2),
    }


def segmentation(values, years, mode):
    """
    Recursive (binary) segmentation with the Pettitt test.
      mode="right": after a significant break, test only the segment that starts at the break
                    (the sequence of tests behind Table 2).
      mode="both" : test both sub-segments (full binary segmentation).
    """
    rows, queue = [], [(0, len(values))]
    while queue:
        start, end = queue.pop(0)
        n = end - start
        if n < MIN_SEGMENT_LEN:
            continue
        r = pettitt(values[start:end])
        brk = start + r["cp"]
        significant = r["p_mc"] < ALPHA
        rows.append({
            "test": len(rows) + 1, "segment_start": years[start], "segment_end": years[end - 1],
            "n": n, "break_year": years[brk], "break_index": brk, "U": r["U"],
            "p_mc": r["p_mc"], "p_eq3": r["p_eq3"],
            "mean_before": r["mean_before"], "mean_after": r["mean_after"],
            "significant": significant,
        })
        if significant:
            if mode == "both":
                queue.append((start, brk))
            queue.append((brk, end))
    df = pd.DataFrame(rows)
    m = len(df)
    df["bonferroni_alpha"] = ALPHA / m
    df["significant_bonferroni_mc"] = df["p_mc"] < ALPHA / m
    df["significant_bonferroni_eq3"] = df["p_eq3"] < ALPHA / m
    return df


def regimes_from_breaks(values, years, break_idx):
    edges = [0] + sorted(break_idx) + [len(values)]
    rows = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        seg = values[a:b]
        rows.append({"regime": f"R{i}", "start": years[a], "end": years[b - 1], "n": b - a,
                     "mean_cm": seg.mean(), "sd_cm": seg.std(ddof=1) if len(seg) > 1 else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- PELT
def pelt_break_indices(values, c, jump):
    """0-based indices of the first observation of each new segment."""
    z = (values - values.mean()) / values.std()          # population SD, as in StandardScaler
    algo = rpt.Pelt(model="l2", min_size=2, jump=jump).fit(z.reshape(-1, 1))
    return algo.predict(pen=c * np.log(len(z)))[:-1]


def pelt_sensitivity(values, years):
    rows = []
    for jump in PELT_JUMPS:
        for c in PELT_C_GRID:
            idx = pelt_break_indices(values, c, jump)
            rows.append({"jump": jump, "c": c, "n_breaks": len(idx),
                         "break_years": "; ".join(str(years[i]) for i in idx),
                         "break_indices": tuple(int(i) for i in idx)})
    return pd.DataFrame(rows)


def plateaus(sens):
    """Contiguous ranges of c that return the same set of breaks."""
    out = []
    for jump, g in sens.groupby("jump"):
        g = g.sort_values("c")
        run_id = (g["break_years"] != g["break_years"].shift()).cumsum()
        for _, run in g.groupby(run_id):
            out.append({"jump": jump, "break_years": run["break_years"].iloc[0] or "(none)",
                        "n_breaks": int(run["n_breaks"].iloc[0]),
                        "c_min": run["c"].min(), "c_max": run["c"].max(), "n_grid_points": len(run)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- BIC
def bic_segmentation(values, break_idx):
    """BIC of a piecewise-constant mean model; k counts the segment means only."""
    n = len(values)
    edges = [0] + sorted(break_idx) + [n]
    rss = sum(((values[a:b] - values[a:b].mean()) ** 2).sum() for a, b in zip(edges[:-1], edges[1:]))
    return n * np.log(rss / n) + (len(edges) - 1) * np.log(n)


def bic_table(values, years, pettitt_idx, sens):
    rows = [{"method": "Pettitt (sequential)", "jump": "", "break_years": "; ".join(str(years[i]) for i in pettitt_idx),
             "n_breaks": len(pettitt_idx), "BIC": bic_segmentation(values, pettitt_idx)}]
    seen = set()
    for _, row in sens.iterrows():
        key = (row["jump"], row["break_indices"])
        if row["n_breaks"] == 0 or key in seen:
            continue
        seen.add(key)
        rows.append({"method": "PELT", "jump": row["jump"], "break_years": row["break_years"],
                     "n_breaks": row["n_breaks"], "BIC": bic_segmentation(values, row["break_indices"])})
    df = pd.DataFrame(rows)
    df["delta_vs_pettitt"] = df["BIC"] - df.loc[0, "BIC"]
    return df


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Change-point analysis of the annual stage series.")
    parser.add_argument("--annual", type=Path, default=ROOT / "data" / "processed" / "annual_mean_stage_ladario.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()

    if not args.annual.exists():
        raise SystemExit(f"{args.annual} not found. Run scripts/01_preprocessing.py first.")
    args.out.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(args.annual)
    values = annual["stage_cm"].to_numpy(dtype=float)
    years = annual["year"].to_numpy(dtype=int)
    if np.isnan(values).any():
        raise SystemExit("The annual series contains missing values.")
    if "n_days" in annual and (annual["n_days"] < 360).any():
        print("WARNING: incomplete years in the annual series (e.g., a partial final year):")
        print(annual.loc[annual["n_days"] < 360, ["year", "n_days"]].to_string(index=False), "\n")
    print(f"pyhomogeneity {version('pyhomogeneity')}, ruptures {version('ruptures')}; "
          f"series {years[0]}-{years[-1]} (n={len(values)})\n")

    # Pettitt: sequential (Table 2) and full binary segmentation
    seq = segmentation(values, years, mode="right")
    seq.to_csv(args.out / "pettitt_sequential.csv", index=False)
    pettitt_idx = seq.loc[seq["significant"], "break_index"].astype(int).tolist()
    regimes_from_breaks(values, years, pettitt_idx).to_csv(args.out / "regimes_pettitt_sequential.csv", index=False)
    full = segmentation(values, years, mode="both")
    full.to_csv(args.out / "pettitt_binary_segmentation.csv", index=False)

    # PELT sensitivity and BIC comparison
    sens = pelt_sensitivity(values, years)
    sens.drop(columns="break_indices").to_csv(args.out / "pelt_sensitivity.csv", index=False)
    plateaus(sens).to_csv(args.out / "pelt_plateaus.csv", index=False)
    bic = bic_table(values, years, pettitt_idx, sens)
    bic.to_csv(args.out / "bic_comparison.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    show = ["test", "segment_start", "segment_end", "n", "break_year", "U", "p_mc", "p_eq3"]
    print("Pettitt, sequential (Table 2):")
    print(seq[show + ["significant_bonferroni_mc", "significant_bonferroni_eq3"]].round(5).to_string(index=False))
    print(f"  Bonferroni alpha = {ALPHA}/{len(seq)} = {ALPHA / len(seq):.4f}\n")
    print(f"Pettitt, binary segmentation on both sides: {len(full)} tests, "
          f"{int(full['significant'].sum())} significant breaks")
    print(full.loc[full["significant"], show].round(5).to_string(index=False), "\n")
    print("PELT plateaus (c = penalty factor):")
    print(plateaus(sens).to_string(index=False), "\n")
    print("BIC comparison:")
    print(bic.round(2).to_string(index=False))
    print(f"\nTables written to {args.out}")


if __name__ == "__main__":
    main()