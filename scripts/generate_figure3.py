#!/usr/bin/env python3
"""
Figure 3: Temporal and multiscale structure of cardiac complexity (Nagoya)
==========================================================================
Rebuilt 2026-08-17. Replaces the previous 4-hour-window version.

What changed and why
--------------------
1. Circadian panels now use 15-min windows instead of 4-h blocks. 15 min is the
   shortest window that supports rcMSE at tau<=5 (N/tau>=200 needs ~1000 beats),
   gives ~91 windows/subject for a distributional summary, and matches the CETRAM
   recording length so the same index is computable in all three cohorts.
2. Added the scale-resolved panel out to tau=60. The 4-h window holds ~14 000
   beats and permits tau<=70; the old analysis stopped at tau=20 and therefore
   missed the peak. Discrimination is maximal at tau~15 (~12 s, baroreflex band),
   not at scale 1 — the core multiscale result.
3. Added distributional summaries. The minimum across the day discriminates far
   better than the median, mirroring Suzuki et al.'s SDNN-min.
4. REMOVED the "flattened circadian profile" claim. It is not supported. The
   across-day range of the HR-normalised index gives AUC 0.56, and of the
   unnormalised index 0.33 (i.e. PD show the LARGER range). Either way PD have a
   lower floor with a broadly comparable ceiling — not a compressed profile.

No window is selected by group separation anywhere in this figure.

Inputs  : data/japan_15min_windows.csv, data/japan_scale_profile.csv
          (both from scripts/compute_nagoya_windows.py)
Outputs : figures/Figure3/Figure3.{png,svg} + figure3_stats.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Figure3")
os.makedirs(OUT, exist_ok=True)

COL = {"Control": "#2E86AB", "PD": "#D62828"}
BANDS = [(0.0, 2.5, "sub-resp.", "#F2F2F2"),
         (2.5, 6.7, "HF / resp.", "#DCEEF5"),
         (6.7, 25.0, "LF / baroreflex", "#FBE3E3"),
         (25.0, 60.0, "VLF", "#EFEAF5")]


def auc_cp(v, y):
    v = np.asarray(v, float); y = np.asarray(y)
    ok = np.isfinite(v)
    return roc_auc_score(y[ok], -v[ok]) if ok.sum() > 10 else np.nan


def main():
    w = pd.read_csv(os.path.join(DATA, "japan_15min_windows.csv"))
    p = pd.read_csv(os.path.join(DATA, "japan_scale_profile.csv"))
    w = w[w.Group.isin(["Control", "PD"])]
    print(f"  windows: {len(w)} from {w.Subject.nunique()} subjects")
    print(f"  scale profiles: {len(p)} subjects")

    stats = []

    # ---- per-subject distributional summaries ---------------------------
    g = w.groupby(["Subject", "Group"])
    summ = g["cx_HR"].agg(min="min", p10=lambda s: np.percentile(s, 10),
                          p25=lambda s: np.percentile(s, 25), med="median",
                          p75=lambda s: np.percentile(s, 75),
                          p90=lambda s: np.percentile(s, 90), max="max").reset_index()
    summ["range"] = summ["max"] - summ["min"]
    summ["n_win"] = g.size().values
    ysum = (summ.Group == "PD").astype(int).values

    # ---- clock-aligned ensemble -----------------------------------------
    w["hour_bin"] = w.clock.astype(int)
    ens = (w.groupby(["Group", "hour_bin"])
             .agg(cx=("cx_HR", "mean"), cx_se=("cx_HR", "sem"),
                  hr=("HR", "mean"), hr_se=("HR", "sem"),
                  sdnn=("SDNN", "mean"), sdnn_se=("SDNN", "sem"),
                  drift=("drift", "mean"), drift_se=("drift", "sem"),
                  n=("cx_HR", "size")).reset_index())

    # ---- scale profile ---------------------------------------------------
    taus = [t for t in range(1, 61) if f"t{t}" in p.columns]
    mrr = p.meanRR.mean() / 1000.0
    yp = (p.Group == "PD").astype(int).values
    prof = []
    for t in taus:
        v = p[f"t{t}"].values
        ok = np.isfinite(v)
        if ok.sum() < 20:
            prof.append(dict(tau=t, sec=t * mrr, auc=np.nan, p=np.nan,
                             c_mean=np.nan, c_se=np.nan, p_mean=np.nan, p_se=np.nan))
            continue
        a = v[ok][yp[ok] == 0]; b = v[ok][yp[ok] == 1]
        prof.append(dict(tau=t, sec=t * mrr, auc=auc_cp(v, yp),
                         p=mannwhitneyu(a, b)[1],
                         c_mean=a.mean(), c_se=a.std(ddof=1) / np.sqrt(len(a)),
                         p_mean=b.mean(), p_se=b.std(ddof=1) / np.sqrt(len(b))))
    prof = pd.DataFrame(prof)
    peak = prof.loc[prof.auc.idxmax()]

    # ================= PLOT =================
    fig = plt.figure(figsize=(17.5, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.30)

    def lab(ax, s):
        ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=17, fontweight="bold")

    # --- Row 1: circadian at 15-min resolution
    for i, (col, se, ttl, yl) in enumerate([
            ("cx", "cx_se", "Complexity index", "rcMSE nAUC(1-5) / HR"),
            ("hr", "hr_se", "Heart rate", "HR (bpm)"),
            ("sdnn", "sdnn_se", "SDNN", "SDNN (ms)"),
            ("drift", "drift_se", "Activity proxy", "1-min drift (ms)")]):
        ax = fig.add_subplot(gs[0, i]); lab(ax, "ABCD"[i])
        for grp in ("Control", "PD"):
            s = ens[ens.Group == grp].sort_values("hour_bin")
            ax.plot(s.hour_bin, s[col], color=COL[grp], lw=2.4, marker="o", ms=3.5,
                    label=f"{grp}")
            ax.fill_between(s.hour_bin, s[col] - s[se], s[col] + s[se],
                            color=COL[grp], alpha=.22)
        ax.axvspan(0, 6, color="#EDEDED", zorder=0)
        ax.axvspan(22, 24, color="#EDEDED", zorder=0)
        ax.set_xlim(0, 23); ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Clock hour"); ax.set_ylabel(yl)
        ax.set_title(ttl, fontweight="bold", fontsize=12)
        ax.grid(alpha=.2)
        if i == 0:
            ax.legend(fontsize=9, frameon=False)
    fig.text(0.5, 0.632, "15-minute windows across the full 24 h — descriptive; no window is "
             "selected by group separation", ha="center", fontsize=10, style="italic", color="#555")

    # --- Row 2: multiscale
    ax = fig.add_subplot(gs[1, :2]); lab(ax, "E")
    smax = float(prof.sec.max())
    for bi, (lo, hi, nm, cl) in enumerate(BANDS):
        if lo >= smax:
            continue
        ax.axvspan(lo, min(hi, smax), color=cl, zorder=0)
        ax.text((lo + min(hi, smax)) / 2, 0.965 - 0.055 * (bi % 2),
                nm, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#555")
    ax.set_xlim(0, smax)
    for grp, mc, sc in (("Control", "c_mean", "c_se"), ("PD", "p_mean", "p_se")):
        ax.plot(prof.sec, prof[mc], color=COL[grp], lw=2.6, label=grp)
        ax.fill_between(prof.sec, prof[mc] - prof[sc], prof[mc] + prof[sc],
                        color=COL[grp], alpha=.22)
    ax.set_xlabel("Timescale  $\\tau\\times$ mean RR  (s)")
    ax.set_ylabel("Sample entropy")
    ax.set_title("Scale-resolved entropy, 16-20 h window ($\\tau$ = 1-60)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=.2)

    ax = fig.add_subplot(gs[1, 2:]); lab(ax, "F")
    for lo, hi, nm, cl in BANDS:
        if lo < smax:
            ax.axvspan(lo, min(hi, smax), color=cl, zorder=0)
    ax.set_xlim(0, smax)
    ax.plot(prof.sec, prof.auc, color="#6C3483", lw=2.6, marker="o", ms=3.5)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.plot(peak.sec, peak.auc, marker="*", ms=20, color="#F1C40F",
            markeredgecolor="k", zorder=6)
    ax.annotate(f"peak AUC {peak.auc:.3f}\n$\\tau$={int(peak.tau)}  ({peak.sec:.1f} s)",
                (peak.sec, peak.auc), xytext=(46, -46), textcoords="offset points",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("Timescale  $\\tau\\times$ mean RR  (s)")
    ax.set_ylabel("AUC (Control > PD)")
    ax.set_ylim(.45, .95)
    ax.set_title("Discrimination peaks in the baroreflex band — not at scale 1",
                 fontweight="bold", fontsize=12)
    ax.grid(alpha=.2)

    # --- Row 3: distribution across the day
    ax = fig.add_subplot(gs[2, :2]); lab(ax, "G")
    order = ["min", "p10", "p25", "med", "p75", "p90", "max", "range"]
    aucs = [auc_cp(summ[c], ysum) for c in order]
    cols = ["#1B7A3D" if a >= .8 else ("#7F8C8D" if a >= .55 else "#C0392B") for a in aucs]
    ax.bar(range(len(order)), aucs, color=cols, alpha=.9)
    for i, a in enumerate(aucs):
        ax.text(i, a + .012, f"{a:.3f}", ha="center", fontsize=9)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["min", "p10", "p25", "median", "p75", "p90", "max", "range"],
                       fontsize=9)
    ax.set_ylim(.25, .95); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Summaries of the 15-min complexity distribution across 24 h\n"
                 "all information is in the low tail; range is uninformative",
                 fontweight="bold", fontsize=12)
    ax.grid(alpha=.2, axis="y")

    ax = fig.add_subplot(gs[2, 2:]); lab(ax, "H")
    dat, pos, tick = [], [], []
    for j, stat in enumerate(["min", "med", "max"]):
        for k, grp in enumerate(("Control", "PD")):
            dat.append(summ[summ.Group == grp][stat].dropna().values)
            pos.append(j * 3 + k)
        tick.append(j * 3 + 0.5)
    bp = ax.boxplot(dat, positions=pos, widths=.75, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(COL["Control" if i % 2 == 0 else "PD"]); b.set_alpha(.75)
    for i, d_ in enumerate(dat):
        ax.scatter(np.random.normal(pos[i], .07, len(d_)), d_, s=11,
                   color="k", alpha=.45, zorder=3)
    for j, stat in enumerate(["min", "med", "max"]):
        a = summ[summ.Group == "Control"][stat].dropna()
        b = summ[summ.Group == "PD"][stat].dropna()
        pv = mannwhitneyu(a, b)[1]
        txt = "p<0.001" if pv < .001 else f"p={pv:.3f}"
        ax.text(j * 3 + .5, ax.get_ylim()[1] * .97, txt, ha="center", fontsize=9,
                fontweight="bold" if pv < .05 else "normal")
    ax.set_xticks(tick); ax.set_xticklabels(["minimum", "median", "maximum"])
    ax.set_ylabel("rcMSE nAUC(1-5) / HR")
    ax.set_title("Complexity floor separates the groups strongly; ceiling only weakly",
                 fontweight="bold", fontsize=12)
    ax.grid(alpha=.2, axis="y")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COL[g], alpha=.75, label=g)
                       for g in ("Control", "PD")], fontsize=9, frameon=False,
              loc="lower right")

    fig.suptitle("Figure 3 — Temporal and multiscale structure of cardiac complexity "
                 "(Nagoya, 24-h Holter)", fontsize=16, fontweight="bold", y=.955)

    png = os.path.join(OUT, "Figure3.png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)

    # ---- stats out -------------------------------------------------------
    for c, a in zip(order, aucs):
        a_, b_ = summ[summ.Group == "Control"][c], summ[summ.Group == "PD"][c]
        stats.append(dict(panel="G", metric=f"cx_HR_{c}", auc=a,
                          p=mannwhitneyu(a_.dropna(), b_.dropna())[1],
                          control=a_.median(), pd=b_.median()))
    for _, r in prof.iterrows():
        stats.append(dict(panel="EF", metric=f"tau{int(r.tau)}", auc=r.auc, p=r.p,
                          control=r.c_mean, pd=r.p_mean))
    pd.DataFrame(stats).to_csv(os.path.join(OUT, "figure3_stats.csv"), index=False)

    print(f"  peak: tau={int(peak.tau)} ({peak.sec:.1f}s) AUC={peak.auc:.3f}")
    print(f"  min AUC={aucs[0]:.3f}  median AUC={aucs[3]:.3f}  "
          f"max AUC={aucs[6]:.3f}  range AUC={aucs[7]:.3f}")
    print("  ->", png)


if __name__ == "__main__":
    main()
