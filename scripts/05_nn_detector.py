"""
05_nn_detector.py

Walk-forward residual-anomaly detector (a shallow neural network) used as an independent
cross-check of the change points, with its sensitivity analyses and a linear AR(p) baseline.

For each year t >= W the model is trained only on years before t and predicts year t; the
out-of-sample residual e_t is stored. A year is flagged in a run when |e_t| exceeds k times
the standard deviation of that run's residuals. Years before the first prediction (t < W)
are NOT evaluated and are reported as missing, never as "not flagged".

Input : data/processed/annual_mean_stage_ladario.csv           (01_preprocessing)
Output: results/tables/nn_flagged_by_year.csv       proportion of runs flagging each year (1.5, 2, 3 sigma)
        results/tables/nn_sensitivity_grid.csv      same, for each architecture (Hu) and lag window (p)
        results/tables/nn_ar_baseline.csv           agreement between the network and an AR(p) baseline
        results/intermediate/nn_residuals_W{W}_R{R}.npy   saved residuals of the main run (slow step)

Usage
  python scripts/05_nn_detector.py [--annual PATH] [--tables DIR] [--intermediate DIR]
                                   [--w 60] [--r 50] [--r-grid 20] [--jobs -1] [--skip-grid] [--force]
"""
import argparse
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

HU, P = 3, 5                       # main architecture: hidden units, lag window
L2_PENALTY, MAX_ITER = 1.0, 5000   # strong regularization: the series is short
GRID_HU, GRID_P = (2, 3, 5, 8), (3, 5, 7)
THRESHOLDS = (1.5, 2.0, 3.0)       # in units of the residual standard deviation
MAIN_K = 2.0
KEY_YEARS = (1962, 1964, 1966, 1967, 1974, 1999, 2019, 2020, 2024)

# Proportions reported in the manuscript for the main run (2 sigma), for comparison only.
PAPER = {1974: 1.00, 1964: 0.62, 2020: 0.36, 1967: 0.26, 1966: 0.16, 2024: 0.14, 1999: 0.00, 2019: 0.00}


# ------------------------------------------------------------------ walk-forward
def build_lagged(data, p):
    x = np.array([data[i - p:i] for i in range(p, len(data))])
    return x, np.asarray(data[p:])


def nn_run(y, w, p, hu, seed):
    """One walk-forward run; returns the residual series (NaN for t < w) and the number of
    convergence warnings raised during training."""
    n = len(y)
    res = np.full(n, np.nan)
    n_warn = 0
    for t in range(w, n):
        x, target = build_lagged(y[:t], p)
        if len(x) < 5:
            continue
        sx, sy = StandardScaler().fit(x), StandardScaler().fit(target.reshape(-1, 1))
        model = MLPRegressor(hidden_layer_sizes=(hu,), activation="tanh", solver="lbfgs",
                             alpha=L2_PENALTY, max_iter=MAX_ITER, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(sx.transform(x), sy.transform(target.reshape(-1, 1)).ravel())
        n_warn += sum(issubclass(c.category, ConvergenceWarning) for c in caught)
        pred = sy.inverse_transform(model.predict(sx.transform(y[t - p:t].reshape(1, -1))).reshape(-1, 1))
        res[t] = y[t] - pred.ravel()[0]
    return res, n_warn


def nn_residuals(y, w, p, hu, runs, jobs):
    out = Parallel(n_jobs=jobs)(delayed(nn_run)(y, w, p, hu, r) for r in range(runs))
    return np.vstack([o[0] for o in out]), sum(o[1] for o in out)


def ar_residuals(y, w, p):
    """Walk-forward linear AR(p) fitted by ordinary least squares."""
    res = np.full(len(y), np.nan)
    for t in range(w, len(y)):
        x, target = build_lagged(y[:t], p)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), target, rcond=None)
        res[t] = y[t] - (coef[0] + y[t - p:t] @ coef[1:])
    return res


def flagged_share(res, k):
    """Proportion of runs flagging each year; NaN where the year was not evaluated."""
    res = np.atleast_2d(res)
    sigma = np.nanstd(res, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        flags = np.abs(res) > k * sigma
    share = flags.mean(axis=0).astype(float)
    share[np.isnan(res).all(axis=0)] = np.nan
    return share


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Neural-network residual-anomaly detector.")
    ap.add_argument("--annual", type=Path, default=ROOT / "data" / "processed" / "annual_mean_stage_ladario.csv")
    ap.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--intermediate", type=Path, default=ROOT / "results" / "intermediate")
    ap.add_argument("--w", type=int, default=60, help="initial training window (years)")
    ap.add_argument("--r", type=int, default=50, help="runs of the main analysis")
    ap.add_argument("--r-grid", type=int, default=20, help="runs per configuration in the sensitivity grid")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1: all cores)")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if saved residuals exist")
    args = ap.parse_args()

    if not args.annual.exists():
        raise SystemExit(f"{args.annual} not found. Run scripts/01_preprocessing.py first.")
    args.tables.mkdir(parents=True, exist_ok=True)
    args.intermediate.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(args.annual)
    y, years, n = annual["stage_cm"].to_numpy(float), annual["year"].to_numpy(int), len(annual)
    w = args.w
    print(f"scikit-learn {version('scikit-learn')}; n={n}, first evaluated year {years[w]} "
          f"(years {years[0]}-{years[w - 1]} are not evaluated)\n")

    # ---- main run (W, R)
    saved = args.intermediate / f"nn_residuals_W{w}_R{args.r}.npy"
    if saved.exists() and not args.force:
        res = np.load(saved)
        print(f"Main run: using saved residuals {saved.name} (use --force to recompute)")
    else:
        print(f"Main run: Hu={HU}, p={P}, W={w}, R={args.r} ...")
        res, n_warn = nn_residuals(y, w, P, HU, args.r, args.jobs)
        np.save(saved, res)
        print(f"   convergence warnings: {n_warn}; residuals saved to {saved.name}")

    by_year = pd.DataFrame({"year": years, "evaluated": ~np.isnan(res).all(axis=0)})
    for k in THRESHOLDS:
        by_year[f"nn_share_{k}sigma"] = flagged_share(res, k)

    # ---- linear AR(p) baseline, same scheme and threshold
    ar_res = ar_residuals(y, w, P)
    ar_flag = np.abs(ar_res) > MAIN_K * np.nanstd(ar_res)
    by_year["ar_flagged_2sigma"] = np.where(by_year["evaluated"], ar_flag, np.nan)
    by_year.to_csv(args.tables / "nn_flagged_by_year.csv", index=False)

    ev = by_year[by_year["evaluated"]].copy()
    nn_flag = ev[f"nn_share_{MAIN_K}sigma"] > 0.5
    ar_f = ev["ar_flagged_2sigma"].astype(bool)
    agree = int((nn_flag == ar_f).sum())
    only_ar, only_nn = ev.loc[ar_f & ~nn_flag, "year"].tolist(), ev.loc[~ar_f & nn_flag, "year"].tolist()
    pd.DataFrame([{"n_evaluated_years": len(ev), "n_agree": agree, "agreement_pct": 100 * agree / len(ev),
                   "only_ar_flagged": "; ".join(map(str, only_ar)), "only_nn_flagged": "; ".join(map(str, only_nn))}]
                 ).to_csv(args.tables / "nn_ar_baseline.csv", index=False)

    # ---- sensitivity grid (architecture x lag window)
    if not args.skip_grid:
        rows = []
        for hu in GRID_HU:
            for p in GRID_P:
                print(f"Grid: Hu={hu}, p={p}, R={args.r_grid} ...")
                r_grid, _ = nn_residuals(y, w, p, hu, args.r_grid, args.jobs)
                share = flagged_share(r_grid, MAIN_K)
                rows += [{"Hu": hu, "p": p, "year": int(yr), "share": s}
                         for yr, s in zip(years, share) if not np.isnan(s)]
        grid = pd.DataFrame(rows)
        grid.to_csv(args.tables / "nn_sensitivity_grid.csv", index=False)
        print()
        key = grid[grid["year"].isin([1974, 1964, 1999, 2019])].pivot_table(index=["Hu", "p"], columns="year", values="share")
        print("Sensitivity grid (share of runs flagging the year):")
        print(key.round(2).to_string(), "\n")

    # ---- summary
    print(f"Main run, share of runs flagging each year (paper, 2 sigma, in parentheses):")
    cols = ["year"] + [f"nn_share_{k}sigma" for k in THRESHOLDS] + ["ar_flagged_2sigma"]
    show = by_year[by_year["year"].isin(KEY_YEARS)][cols].copy()
    show["paper_2sigma"] = show["year"].map(PAPER)
    print(show.round(2).to_string(index=False))
    print(f"\nAR({P}) baseline: agrees with the network in {agree} of {len(ev)} evaluated years "
          f"({100 * agree / len(ev):.1f}%); only AR flags: {only_ar or 'none'}; only network flags: {only_nn or 'none'}")
    print(f"\nTables written to {args.tables}")


if __name__ == "__main__":
    main()