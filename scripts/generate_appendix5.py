#!/usr/bin/env python3
"""
Appendix 5: Should the complexity index be normalised by heart rate?

The published figures divide rcMSE nAUC by mean HR. This appendix tests whether
that is justified. Four criteria, applied to all three cohorts:

  1. Is there an HR dependency to correct?      -> no (all rho n.s.)
  2. Does dividing remove one, or create one?   -> creates one (all rho p<0.05)
  3. Is a RATIO the right functional form?      -> no (log-log slope excludes 1)
  4. Does it improve discrimination?            -> yes, but the gain is HR's own
                                                   group signal, not a removed confound

Criterion 4 is the decisive one. If HR normalisation corrected a genuine confound
it should help most in the cohort where the HR difference is largest. Nagoya has
the largest HR difference and is the one cohort where normalisation makes
discrimination worse — the apparent gain elsewhere is the ratio smuggling in
heart rate, which itself differs between groups, rather than removing a confound.

Conclusion: report the UNNORMALISED index, and handle HR as a covariate where a
correction is wanted (ANCOVA shows it contributes nothing in CETRAM or Nagoya).

Output: figures/Appendix/FigureAppendix5.{png,svg} + appendix5_stats.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle as fs
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Appendix")
os.makedirs(OUT, exist_ok=True)

COH = ["CETRAM", "Cruces", "Nagoya"]
COL = {"CETRAM": "#1565C0", "Cruces": "#2E7D32", "Nagoya": "#E65100"}
GCOL = {"Control": "#2E86AB", "PD": "#D62828"}
HRV = ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_SD1", "HRV_SD2", "HRV_DFA_alpha1"]


def load(c):
    """per-subject complexity index, HR, and available HRV metrics"""
    if c == "CETRAM":
        m = pd.read_csv(os.path.join(DATA, "chile_mse.csv"))
        met = pd.read_csv(os.path.join(DATA, "chile_metrics.csv"))
    elif c == "Cruces":
        m = pd.read_csv(os.path.join(DATA, "spain_mse.csv"))
        met = pd.read_csv(os.path.join(DATA, "spain_metrics.csv"))
    else:
        m = pd.read_csv(os.path.join(DATA, "japan_window_mse.csv"))
        met = pd.read_csv(os.path.join(DATA, "japan_window_features.csv"))
    m = m[m.Group.isin(["Control", "PD"])]
    idx = (m[m.Scales.between(1, 5)].sort_values("Scales")
             .groupby(["Subject", "Group"])
             .apply(lambda g: np.trapezoid(g.MSE.values, np.arange(1, 6)) / 5,
                    include_groups=False)
             .rename("cx").reset_index())
    met = met.copy()
    if "HRV_MeanNN" in met.columns:
        met["HR"] = 60000.0 / met["HRV_MeanNN"]
    keep = ["Subject", "HR"] + [h for h in HRV if h in met.columns]
    d = idx.merge(met[keep], on="Subject", how="inner").dropna(subset=["HR"])
    d["cxhr"] = d.cx / d.HR
    d["y"] = (d.Group == "PD").astype(int)
    return d


def main():
    fs.apply()
    D = {c: load(c) for c in COH}
    stats = []

    fig = plt.figure(figsize=(17.5, 10.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def lab(ax, s):
        ax.text(-0.17, 1.08, s, transform=ax.transAxes, fontsize=17, fontweight="normal")

    # ---- A: the HR difference each cohort actually has ----
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    pos, tick = [], []
    for i, c in enumerate(COH):
        d = D[c]
        for j, g in enumerate(("Control", "PD")):
            v = d[d.Group == g].HR
            bp = ax.boxplot([v], positions=[i * 3 + j], widths=.75,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(GCOL[g]); bp["boxes"][0].set_alpha(.75)
            ax.scatter(np.random.normal(i * 3 + j, .07, len(v)), v, s=9, c="k", alpha=.4, zorder=3)
        p = mannwhitneyu(d[d.y == 0].HR, d[d.y == 1].HR)[1]
        ax.text(i * 3 + .5, ax.get_ylim()[1] * .99, f"p={p:.3f}", ha="center", fontsize=8.5)
        tick.append(i * 3 + .5)
        stats.append(dict(panel="A", cohort=c, metric="HR_p", value=p))
    ax.set_xticks(tick); ax.set_xticklabels(COH)
    ax.set_ylabel("mean heart rate (bpm)")
    ax.set_title("Mean heart rate by group",
                 fontweight="normal", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GCOL[g], alpha=.75, label=g) for g in GCOL],
              fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=.2, axis="y")

    # ---- B: dividing CREATES an HR dependency ----
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    x = np.arange(len(COH))
    raw = [spearmanr(D[c].cx, D[c].HR) for c in COH]
    div = [spearmanr(D[c].cxhr, D[c].HR) for c in COH]
    ax.bar(x - .2, [r[0] for r in raw], .4, color="#7F8C8D", label="unnormalized")
    ax.bar(x + .2, [r[0] for r in div], .4, color="#8E44AD", label="$\\div$ mean HR")
    for i, (r, dd) in enumerate(zip(raw, div)):
        ax.text(i - .2, r[0] - .05, "n.s." if r[1] > .05 else f"p={r[1]:.3f}",
                ha="center", fontsize=8)
        ax.text(i + .2, dd[0] - .06, f"p={dd[1]:.0e}" if dd[1] < .001 else f"p={dd[1]:.3f}",
                ha="center", fontsize=8, fontweight="normal")
        stats += [dict(panel="B", cohort=COH[i], metric="rho_raw_HR", value=r[0]),
                  dict(panel="B", cohort=COH[i], metric="rho_div_HR", value=dd[0])]
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(COH)
    ax.set_ylabel("Spearman $\\rho$ with mean HR"); ax.set_ylim(-0.92, 0.25)
    ax.set_title("Correlation of index with heart rate",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.grid(alpha=.2, axis="y")

    # ---- C: is a ratio the right form? ----
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    for i, c in enumerate(COH):
        d = D[c]
        m = sm.OLS(np.log(d.cx), sm.add_constant(np.log(d.HR))).fit()
        b = m.params.iloc[1]; ci = m.conf_int().iloc[1]
        ax.errorbar([i], [b], yerr=[[b - ci[0]], [ci[1] - b]], fmt="o", ms=9,
                    capsize=5, lw=2.2, color=COL[c])
        stats.append(dict(panel="C", cohort=c, metric="loglog_slope", value=b))
    ax.axhline(1, ls="--", lw=1.2, color="#C44E52")
    ax.axhline(0, ls=":", lw=0.9, color="0.4")
    ax.set_xticks(range(3)); ax.set_xticklabels(COH); ax.set_xlim(-.6, 2.6)
    ax.set_ylabel("log-log slope of index on HR")
    ax.set_title("Log–log scaling of index on heart rate",
                 fontweight="normal", fontsize=12)
    ax.grid(alpha=.2, axis="y")

    # ---- D: the decisive panel — where does it help? ----
    ax = fig.add_subplot(gs[1, 0]); lab(ax, "D")
    w = .21
    for i, c in enumerate(COH):
        d = D[c]
        a_raw = roc_auc_score(d.y, -d.cx)
        a_div = roc_auc_score(d.y, -d.cxhr)
        a_hr  = max(roc_auc_score(d.y, d.HR), roc_auc_score(d.y, -d.HR))
        mdl = sm.OLS(d.cx, sm.add_constant(d[["y", "HR"]].rename(columns={"y": "g"}))).fit()
        resid = d.cx - mdl.params["HR"] * d.HR
        a_anc = roc_auc_score(d.y, -resid)
        for j, (v, cc, nm) in enumerate([(a_hr,  "#D35400", "HR alone"),
                                         (a_raw, "#7F8C8D", "unnormalized"),
                                         (a_div, "#8E44AD", "$\\div$ HR"),
                                         (a_anc, "#16A085", "HR as covariate")]):
            ax.bar(i + (j - 1.5) * w, v, w, color=cc, label=nm if i == 0 else None)
            ax.text(i + (j - 1.5) * w, v + .006, f"{v:.3f}", ha="center", fontsize=7)
        stats += [dict(panel="D", cohort=c, metric="auc_hr_alone", value=a_hr),
                  dict(panel="D", cohort=c, metric="auc_raw", value=a_raw),
                  dict(panel="D", cohort=c, metric="auc_div", value=a_div),
                  dict(panel="D", cohort=c, metric="auc_ancova", value=a_anc)]
        ax.text(i, max(a_hr, a_raw, a_div, a_anc) + .030,
                f"$\\Delta$AUC {a_div - a_raw:+.3f}", ha="center", fontsize=8,
                fontweight="normal", color="#8E44AD")
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(COH)
    ax.set_ylim(.5, 1.02); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Discrimination under each HR adjustment")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.grid(alpha=.2, axis="y")

    # ---- E: scatter, one cohort, showing the injection ----
    ax = fig.add_subplot(gs[1, 1]); lab(ax, "E")
    d = D["Nagoya"]
    ax2 = ax.twinx()
    for g in ("Control", "PD"):
        s = d[d.Group == g]
        ax.scatter(s.HR, s.cx, s=30, color=GCOL[g], alpha=.75, marker="o")
        ax2.scatter(s.HR, s.cxhr, s=30, color=GCOL[g], alpha=.45, marker="x")
    for arr, axx, cc, ls in ((d.cx, ax, "#7F8C8D", "-"), (d.cxhr, ax2, "#8E44AD", "--")):
        z = np.polyfit(d.HR, arr, 1)
        xs = np.linspace(d.HR.min(), d.HR.max(), 50)
        axx.plot(xs, np.polyval(z, xs), ls=ls, lw=2.4, color=cc)
    ax.set_xlabel("mean heart rate (bpm)")
    ax.set_ylabel("unnormalized index  (o, grey fit)", color="#555")
    ax2.set_ylabel("index $\\div$ HR  (x, purple fit)", color="#8E44AD")
    ax2.tick_params(axis="y", labelcolor="#8E44AD")
    ax.set_title("Nagoya: index vs heart rate")
    ax.grid(alpha=.2)

    # ---- F: collateral distortion of HRV correlations ----
    ax = fig.add_subplot(gs[1, 2]); lab(ax, "F")
    for c in COH:
        d = D[c]
        mets = [h for h in HRV if h in d.columns]
        xs, ys = [], []
        for h in mets:
            ok = np.isfinite(d[h])
            if ok.sum() < 15:
                continue
            xs.append(spearmanr(d.HR[ok], d[h][ok])[0])
            ys.append(spearmanr(d.cxhr[ok], d[h][ok])[0] - spearmanr(d.cx[ok], d[h][ok])[0])
        ax.scatter(xs, ys, s=48, color=COL[c], alpha=.85, label=c)
        stats += [dict(panel="F", cohort=c, metric=f"shift_{m_}", value=v)
                  for m_, v in zip(mets, ys)]
    ax.axhline(0, ls=":", c="k"); ax.axvline(0, ls=":", c="k")
    ax.set_xlabel("metric's own $\\rho$ with HR")
    ax.set_ylabel("$\\Delta\\rho$ induced by dividing")
    ax.set_title("Induced change in HRV correlations",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)

    p = os.path.join(OUT, "FigureAppendix5.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats).to_csv(os.path.join(OUT, "appendix5_stats.csv"), index=False)

    print(f"{'cohort':9s}{'AUC raw':>9s}{'AUC /HR':>9s}{'AUC ANCOVA':>12s}"
          f"{'rho raw':>10s}{'rho /HR':>10s}{'slope':>8s}")
    print("-" * 68)
    for c in COH:
        d = D[c]
        m = sm.OLS(np.log(d.cx), sm.add_constant(np.log(d.HR))).fit()
        mdl = sm.OLS(d.cx, sm.add_constant(d[["y", "HR"]].rename(columns={"y": "g"}))).fit()
        resid = d.cx - mdl.params["HR"] * d.HR
        print(f"{c:9s}{roc_auc_score(d.y,-d.cx):>9.3f}{roc_auc_score(d.y,-d.cxhr):>9.3f}"
              f"{roc_auc_score(d.y,-resid):>12.3f}{spearmanr(d.cx,d.HR)[0]:>+10.2f}"
              f"{spearmanr(d.cxhr,d.HR)[0]:>+10.2f}{m.params.iloc[1]:>+8.2f}")
    print("\n ->", p)


if __name__ == "__main__":
    main()
