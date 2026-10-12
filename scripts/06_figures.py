"""
06_figures.py

Regenerates the manuscript figures from the tables written by scripts 01-05.
Each figure is saved as PNG (300 dpi) and PDF in figures/. The study-area map (Figure 1)
is made in QGIS and is not produced here.

Input : data/processed/annual_mean_stage_ladario.csv
        results/tables/*.csv                              (scripts 02-05)
Output: figures/fig_annual_regimes, fig_dfa_loglog, fig_rs_loglog, fig_gph_sensitivity,
        fig_forest_H, fig_falsepositive_curve, fig_envelope_combined, fig_pelt_elbow,
        fig_nn_crosscheck, fig_nn_sensitivity_heatmap, graphical_abstract (also .tiff)

Usage
  python scripts/06_figures.py [--annual PATH] [--tables DIR] [--figures DIR]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd
from hurst import compute_Hc

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
M_ENVELOPE = 500
LITERATURE_H = 0.69                    # Coelho et al. (2025), same Ladario stage record
LITERATURE_PERIODS = [(1900, 1962), (1963, 1974), (1974, 2009)]   # periods reported in the literature
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]   # Okabe-Ito

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})


# ------------------------------------------------------------------ helpers
def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, figdir, name, tiff=False):
    for ext in ("png", "pdf") + (("tiff",) if tiff else ()):
        fig.savefig(figdir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def estimate(tables, key):
    est = pd.read_csv(tables / "hurst_estimates.csv").set_index("estimate")["value"]
    return float(est[key])


# ------------------------------------------------------------------ figures
def fig_annual_regimes(annual, regimes, figdir):
    fig, (ax_lit, ax) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [0.12, 1], "hspace": 0.05})
    for (a, b), c in zip(LITERATURE_PERIODS, ["#B0B0B0", "#7A7A7A", "#B0B0B0"]):
        ax_lit.axvspan(a, b, color=c, alpha=0.6)
    ax_lit.set_yticks([]); ax_lit.set_ylim(0, 1); ax_lit.set_xlim(annual.year.min(), annual.year.max())
    for sp in ax_lit.spines.values():
        sp.set_visible(False)
    ax_lit.text(annual.year.min(), 1.3, "Literature-reported periods", fontsize=7.5, color="dimgray", va="bottom")

    starts = regimes["start"].tolist() + [int(regimes["end"].iloc[-1])]
    for i, r in enumerate(regimes.itertuples()):
        x0, x1, color = starts[i], starts[i + 1], COLORS[i % len(COLORS)]
        ax.axvspan(x0, x1, color=color, alpha=0.15, zorder=0)
        ax.hlines(r.mean_cm, x0, x1, color=color, linestyle="--", linewidth=1.5, zorder=3)
        ax.annotate(f"{r.regime}\n{r.mean_cm:.0f} cm", xy=((x0 + x1) / 2, r.mean_cm), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
    ax.plot(annual.year, annual.stage_cm, color="black", linewidth=1, marker="o", markersize=3, zorder=2,
            label="Annual mean stage (this study)")
    ax.set_xlabel("Year"); ax.set_ylabel("Annual mean stage (cm)")
    ax.set_xlim(annual.year.min(), annual.year.max())
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    clean_axes(ax)
    save(fig, figdir, "fig_annual_regimes")


def fig_loglog(tables, figdir):
    dfa = pd.read_csv(tables / "dfa_scaling.csv")
    alpha = estimate(tables, "DFA-1 annual")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dfa.log_n, dfa.log_F, s=25, color="#0072B2", zorder=3, label="$F(n)$")
    xs = np.array([dfa.log_n.min(), dfa.log_n.max()])
    ax.plot(xs, alpha * xs + (dfa.log_F.mean() - alpha * dfa.log_n.mean()), color="black", linestyle="--",
            linewidth=1.5, label=f"Fitted slope ($\\alpha$ = $H$) = {alpha:.3f}")
    ax.set_xlabel("log($n$)"); ax.set_ylabel("log $F(n)$"); ax.legend(frameon=False, fontsize=9); clean_axes(ax)
    save(fig, figdir, "fig_dfa_loglog")

    rs = pd.read_csv(tables / "rs_scaling.csv")
    h = estimate(tables, "R/S annual")
    lw, lr = np.log(rs.window), np.log(rs.rs)
    slope, intercept = np.polyfit(lw, lr, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lw, lr, s=25, color="#E69F00", zorder=3, label="$R/S(n)$")
    xs = np.array([lw.min(), lw.max()])
    ax.plot(xs, intercept + slope * xs, color="black", linestyle="--", linewidth=1.5,
            label=f"Fitted slope ($H$) = {h:.3f}")
    ax.set_xlabel("log($n$)"); ax.set_ylabel("log($R/S$)"); ax.legend(frameon=False, fontsize=9); clean_axes(ax)
    save(fig, figdir, "fig_rs_loglog")
    if abs(slope - h) > 0.01:
        print(f"  WARNING: refitted R/S slope {slope:.3f} differs from the reported H {h:.3f}")


def fig_gph_sensitivity(tables, figdir):
    g = pd.read_csv(tables / "gph_sensitivity.csv")
    valid = g["valid"].astype(bool)
    lo, hi = sorted([estimate(tables, "R/S annual"), estimate(tables, "DFA-1 annual")])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(g.gamma[valid], g.H[valid], color="#0072B2", marker="o", markersize=8, linewidth=1.5, zorder=3,
            label=r"Valid ($\gamma \leq 0.6$, $m \ll n$)")
    ax.scatter(g.gamma[~valid], g.H[~valid], color="#D55E00", marker="X", s=100, zorder=4,
               label=r"Excluded ($\gamma = 0.7$, violates $m \ll n$)")
    ax.plot(g.gamma.iloc[-2:], g.H.iloc[-2:], color="gray", linestyle="--", linewidth=1, zorder=2)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="$H = 1$ (theoretical ceiling)")
    ax.axhspan(lo, hi, color="#009E73", alpha=0.12, zorder=0, label="Triangulated range (R/S, DFA)")
    for r in g.itertuples():
        ax.annotate(f"$m$={int(r.m)}", (r.gamma, r.H), textcoords="offset points", xytext=(8, -3), fontsize=8, color="dimgray")
    ax.set_xlabel(r"Bandwidth parameter ($\gamma$)"); ax.set_ylabel("GPH Hurst estimate ($H = d + 0.5$)")
    ax.set_xlim(g.gamma.min() - 0.05, g.gamma.max() + 0.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); clean_axes(ax)
    save(fig, figdir, "fig_gph_sensitivity")


def fig_forest(tables, figdir):
    blocks = pd.read_csv(tables / "bootstrap_block.csv")
    g = pd.read_csv(tables / "gph_sensitivity.csv")
    g = g[g["valid"].astype(bool)]
    glo, ghi = g.H.min(), g.H.max()
    rows = [("Literature (Coelho et al., 2025)", LITERATURE_H, None, None, "point"),
            ("R/S (single estimate)", estimate(tables, "R/S annual"), None, None, "point")]
    rows += [(f"R/S bootstrap (block={int(b.block)})", b.H_mean, b.ci_low, b.ci_high, "ci") for b in blocks.itertuples()]
    rows += [("DFA (annual)", estimate(tables, "DFA-1 annual"), None, None, "point"),
             (f"GPH ($\\gamma$={g.gamma.min():.1f}\u2013{g.gamma.max():.1f} range)", (glo + ghi) / 2, glo, ghi, "range")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, v, lo, hi, kind) in enumerate(rows):
        y = len(rows) - i
        if kind == "point":
            ax.scatter(v, y, color="#0072B2", s=50, zorder=3)
        elif kind == "ci":
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="#E69F00", capsize=4, markersize=6, zorder=3)
        else:
            ax.plot([lo, hi], [y, y], color="#009E73", linewidth=6, alpha=0.4, zorder=2)
            ax.scatter(v, y, color="#009E73", s=50, zorder=3, marker="D")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="$H = 0.5$ (no memory)")
    ax.set_yticks(range(1, len(rows) + 1)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlabel("Hurst exponent ($H$)"); ax.set_xlim(0.4, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="upper right"); clean_axes(ax)
    save(fig, figdir, "fig_forest_H")


def fig_falsepositive(tables, figdir):
    fp = pd.read_csv(tables / "fp_rate.csv")
    fp = fp[fp.decision_rule == "Eq. 3 approximation"]
    lo, hi = sorted([estimate(tables, "R/S annual"), estimate(tables, "DFA-1 annual")])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(fp.H_imposed, fp.rate_pct, yerr=1.96 * fp.se_pct, color="#0072B2", linewidth=1.5, marker="o",
                markersize=7, capsize=4, zorder=3,
                label=f"Simulated false-positive rate (fGn, $M$={int(fp.M.iloc[0])}, 95% CI)")
    ax.axhline(5, color="gray", linestyle=":", linewidth=1, zorder=1, label="Nominal significance level (5%)")
    ax.axvspan(lo, hi, color="#D55E00", alpha=0.15, zorder=0, label=f"Observed $H$, this study ({lo:.2f}\u2013{hi:.2f})")
    ax.axvline(LITERATURE_H, color="#009E73", linestyle="--", linewidth=1.5, zorder=2, label=f"Literature $H$ = {LITERATURE_H} (Coelho et al., 2025)")
    ax.set_xlabel("Imposed Hurst exponent ($H$)"); ax.set_ylabel("False-positive rate (%)")
    ax.set_xlim(0.45, 0.90); ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); clean_axes(ax)
    save(fig, figdir, "fig_falsepositive_curve")


def fig_envelope(tables, regimes, figdir):
    """Observed DFA and R/S scaling curves against the envelope of shift-only synthetic series."""
    rng = np.random.default_rng(SEED); np.random.seed(SEED)
    dfa_y, rs_y = [], []
    for _ in range(M_ENVELOPE):
        x = np.concatenate([rng.normal(r.mean_cm, r.sd_cm, int(r.n)) for r in regimes.itertuples()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, (log_n, log_f, _) = nolds.dfa(x, debug_data=True)
        dfa_y.append(log_f)
        _, _, (w, rs) = compute_Hc(x, kind="change", simplified=True)
        rs_y.append(np.log(rs))
    dfa_y, rs_y = np.array(dfa_y), np.array(rs_y)
    log_w = np.log(np.asarray(w, float))

    dfa_obs, rs_obs = pd.read_csv(tables / "dfa_scaling.csv"), pd.read_csv(tables / "rs_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = [(axes[0], log_n, dfa_y, dfa_obs.log_n, dfa_obs.log_F, f"Observed ($\\alpha$ = {estimate(tables, 'DFA-1 annual'):.3f})",
               "log($n$)", "log $F(n)$", "(a) DFA"),
              (axes[1], log_w, rs_y, np.log(rs_obs.window), np.log(rs_obs.rs), f"Observed ($H$ = {estimate(tables, 'R/S annual'):.3f})",
               "log($n$)", "log($R/S$)", "(b) R/S")]
    for ax, xe, ye, xo, yo, label, xl, yl, title in panels:
        ax.fill_between(xe, np.percentile(ye, 5, axis=0), np.percentile(ye, 95, axis=0), color="gray", alpha=0.3,
                        label=f"Shift-only null (90% envelope, $M$={M_ENVELOPE})")
        ax.plot(xo, yo, color="#D55E00", linewidth=1.5, marker="o", markersize=4, label=label)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8); clean_axes(ax)
    plt.tight_layout()
    save(fig, figdir, "fig_envelope_combined")


def fig_pelt_elbow(tables, figdir):
    s = pd.read_csv(tables / "pelt_sensitivity.csv")
    s = s[(s["jump"] == 1) & (s["c"] <= 5.0)].sort_values("c").copy()
    s["break_years"] = s["break_years"].fillna("")
    run = (s["break_years"] != s["break_years"].shift()).cumsum()
    runs = [g for _, g in s.groupby(run) if g["n_breaks"].iloc[0] > 0]
    plateau = max(runs, key=len)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(plateau.c.min(), plateau.c.max(), color="#009E73", alpha=0.15,
               label=f"Stable plateau ({plateau.break_years.iloc[0].replace('; ', ', ')})")
    ax.plot(s.c, s.n_breaks, marker="o", markersize=4, color="#0072B2", linewidth=1.2)
    ax.set_xlabel("Penalty factor ($c$)"); ax.set_ylabel("Number of detected breaks")
    ax.set_xlim(s.c.min() - 0.1, s.c.max() + 0.1); ax.set_ylim(-0.3, s.n_breaks.max() + 0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right"); clean_axes(ax)
    save(fig, figdir, "fig_pelt_elbow")


def fig_nn_crosscheck(tables, regimes, figdir):
    d = pd.read_csv(tables / "nn_flagged_by_year.csv")
    ev = d[d["evaluated"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvspan(d.year.min() - 0.5, ev.year.min() - 0.5, color="lightgray", alpha=0.5, zorder=0)
    ax.text((d.year.min() + ev.year.min()) / 2, 0.5, "initial training window\n(not evaluated)", ha="center",
            va="center", fontsize=8, color="dimgray")
    ax.bar(ev.year, ev["nn_share_2.0sigma"], color="#0072B2", width=0.8, zorder=3)
    for yr in regimes["start"].iloc[1:]:
        ax.vlines(yr, 0, 1.0, color="#D55E00", linestyle="--", linewidth=1.2, zorder=2)   # stops below the label
        ax.annotate(str(yr), xy=(yr, 1.04), ha="center", fontsize=8, color="#D55E00")
    ax.set_xlabel("Year"); ax.set_ylabel("Proportion of runs flagged")
    ax.set_xlim(d.year.min() - 0.5, d.year.max() + 0.5); ax.set_ylim(0, 1.15); clean_axes(ax)
    save(fig, figdir, "fig_nn_crosscheck")


def fig_nn_heatmap(tables, figdir, years=(1974, 1964, 1999, 2019)):
    grid = pd.read_csv(tables / "nn_sensitivity_grid.csv")
    fig, axes = plt.subplots(1, len(years), figsize=(13, 3.2))
    for ax, yr in zip(axes, years):
        m = grid[grid.year == yr].pivot_table(index="Hu", columns="p", values="share")
        im = ax.imshow(m.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
        ax.set_xlabel("$p$ (lag)"); ax.set_title(str(yr), fontsize=11)
        if yr == years[0]:
            ax.set_ylabel("$H_u$ (hidden units)")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m.values[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Proportion flagged")
    save(fig, figdir, "fig_nn_sensitivity_heatmap")


def graphical_abstract(tables, regimes, figdir):
    fp = pd.read_csv(tables / "fp_rate.csv")
    fp85 = fp[(fp.decision_rule == "Eq. 3 approximation") & (fp.H_imposed == fp.H_imposed.max())].iloc[0]
    lo, hi = sorted([estimate(tables, "R/S annual"), estimate(tables, "DFA-1 annual")])
    breaks = " \u00b7 ".join(str(y) for y in regimes["start"].iloc[1:])

    fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1], wspace=0.05)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 400)

    ax1 = fig.add_subplot(gs[0]); ax1.axis("off")
    wave = np.sin(2 * np.pi * 8 * t) * (0.3 + 0.2 * np.sin(2 * np.pi * 1.5 * t)) + np.cumsum(rng.normal(0, 0.02, 400))
    ax1.plot(t, wave, color="#185FA5", linewidth=1.2)
    ax1.text(0.5, 1.15, "126-year stage record", ha="center", fontsize=13, fontweight="bold", transform=ax1.transAxes)
    ax1.text(0.5, 1.02, "Lad\u00e1rio, Pantanal, Brazil", ha="center", fontsize=10, color="#5F5E5A", transform=ax1.transAxes)
    ax1.text(0.0, -0.08, "1900", ha="left", fontsize=9, color="#5F5E5A", transform=ax1.transAxes)
    ax1.text(1.0, -0.08, "2025", ha="right", fontsize=9, color="#5F5E5A", transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.text(0.5, 1.15, "Can we tell them apart?", ha="center", fontsize=13, fontweight="bold", transform=ax2.transAxes)
    tx = np.linspace(0, 1, 100)
    ax2.plot(tx * 0.35 + 0.05, np.where(tx < 0.5, 0.3, 0.7) * 0.3 + 0.55, color="#993C1D", linewidth=2, transform=ax2.transAxes)
    ax2.text(0.22, 0.45, "Candidate breaks", ha="center", fontsize=10, color="#993C1D", transform=ax2.transAxes)
    ax2.text(0.22, 0.38, breaks, ha="center", fontsize=9, color="#5F5E5A", transform=ax2.transAxes)
    persist = np.cumsum(rng.normal(0, 0.03, 200)); persist = (persist - persist.min()) / (persist.max() - persist.min())
    ax2.plot(np.linspace(0, 1, 200) * 0.35 + 0.60, persist * 0.3 + 0.55, color="#0F6E56", linewidth=1.5, transform=ax2.transAxes)
    ax2.text(0.775, 0.45, "Long-range dependence", ha="center", fontsize=10, color="#0F6E56", transform=ax2.transAxes)
    ax2.text(0.775, 0.38, f"H \u2248 {lo:.2f}\u2013{hi:.2f}", ha="center", fontsize=9, color="#5F5E5A", transform=ax2.transAxes)
    ax2.text(0.5, 0.55, "?", ha="center", va="center", fontsize=48, fontweight="bold", color="#2C2C2A", transform=ax2.transAxes)

    ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
    ax3.text(0.5, 1.15, "Key finding", ha="center", fontsize=13, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.5, 0.62, f"{fp85.rate_pct:.0f}%", ha="center", va="center", fontsize=40, fontweight="bold", color="#185FA5", transform=ax3.transAxes)
    ax3.text(0.5, 0.38, f"false-positive rate of the Pettitt test\nunder strong persistence (H = {fp85.H_imposed:.2f}),\nwith no structural break imposed",
             ha="center", va="top", fontsize=10, color="#5F5E5A", transform=ax3.transAxes)
    save(fig, figdir, "graphical_abstract", tiff=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Regenerate the manuscript figures.")
    ap.add_argument("--annual", type=Path, default=ROOT / "data" / "processed" / "annual_mean_stage_ladario.csv")
    ap.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    needed = ["regimes_pettitt_sequential", "pelt_sensitivity", "hurst_estimates", "gph_sensitivity", "bootstrap_block",
              "dfa_scaling", "rs_scaling", "fp_rate", "nn_flagged_by_year", "nn_sensitivity_grid"]
    missing = [p for p in [args.annual] + [args.tables / f"{n}.csv" for n in needed] if not p.exists()]
    if missing:
        raise SystemExit("Missing inputs (run scripts 01-05 first):\n  " + "\n  ".join(map(str, missing)))
    args.figures.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(args.annual)
    regimes = pd.read_csv(args.tables / "regimes_pettitt_sequential.csv")
    print(f"Writing figures to {args.figures}")
    fig_annual_regimes(annual, regimes, args.figures)
    fig_loglog(args.tables, args.figures)
    fig_gph_sensitivity(args.tables, args.figures)
    fig_forest(args.tables, args.figures)
    fig_falsepositive(args.tables, args.figures)
    fig_envelope(args.tables, regimes, args.figures)
    fig_pelt_elbow(args.tables, args.figures)
    fig_nn_crosscheck(args.tables, regimes, args.figures)
    fig_nn_heatmap(args.tables, args.figures)
    graphical_abstract(args.tables, regimes, args.figures)
    print("Done. Figure 1 (study-area map) is made in QGIS.")


if __name__ == "__main__":
    main()