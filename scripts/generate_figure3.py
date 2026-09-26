#!/usr/bin/env python3
"""
Figure 3 — Recording length decides which marker works.

The argument, in one line: in 24-h free-living recordings Suzuki's minimum-SDNN is
excellent and complexity adds nothing to it; but minimum-SDNN is an extreme-value
statistic that needs hundreds of short windows, and it fails in the 15-minute resting
recordings that are practical in clinic, where complexity still discriminates.

Nothing in this figure is selected on the outcome.
  - The circadian panels show every window; no window is nominated.
  - The 4-h window used elsewhere in the paper (18-22 h) was pre-specified on subject
    retention and temporal coverage, never on AUC. See CIRCADIAN_ANALYSIS_AUDIT.md.
  - The Nagoya data are Suzuki's own cohort, contributed by co-authors of this study, so
    panels E-F compare the two markers in the same subjects rather than across studies.
    Minimum-SDNN is reported at full strength; the point is not that it is inferior but
    that it requires a recording length that short clinical protocols do not provide.

Windowing choices, and why:
  15 min  shortest window that supports rcMSE at tau<=5 (N/tau>=200 needs ~1000 beats)
          AND the CETRAM protocol length -- the only window comparable across cohorts.
  4 h     the only window with enough beats to resolve tau out to 60; this is what the
          24-h recording buys.
  overlap none. Overlap adds no discrimination and biases the minimum downward by ~9%
          (panel I); p10 is invariant to it.

Inputs  : data/japan_15min_windows.csv, data/japan_scale_profile.csv,
          data/fig3_window_ladder.csv, data/fig3_dist_summaries.csv,
          data/fig3_hourly_auc.csv, data/fig3_cetram_headtohead.csv,
          data/fig3_overlap.csv
Outputs : figures/Figure3/Figure3.{png,svg} + figure3_stats.csv
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
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Figure3")
os.makedirs(OUT, exist_ok=True)

COL = dict(fs.GROUP)
CX, SD, MUTED = fs.ACCENT, "#DD8452", fs.MUTED
BANDS = [(0.0, 2.5, "sub-resp.", "#F2F2F2"), (2.5, 6.7, "HF / resp.", "#DCEEF5"),
         (6.7, 25.0, "LF / baroreflex", "#FBE3E3"), (25.0, 60.0, "VLF", "#EFEAF5")]
WIN_LO, WIN_HI = 18, 22


def lab(ax, L):
    ax.text(-0.17, 1.14, L, transform=ax.transAxes, fontsize=16, fontweight="normal", va="top")


def main():
    fs.apply()
    w = pd.read_csv(os.path.join(DATA, "japan_15min_windows.csv"))
    w = w[w.Group.isin(["Control", "PD"])].dropna(subset=["cx"])
    prof = pd.read_csv(os.path.join(DATA, "japan_scale_profile.csv"))
    ladder = pd.read_csv(os.path.join(DATA, "fig3_window_ladder.csv"))
    dist = pd.read_csv(os.path.join(DATA, "fig3_dist_summaries.csv"))
    hourly = pd.read_csv(os.path.join(DATA, "fig3_hourly_auc.csv"))
    cet = pd.read_csv(os.path.join(DATA, "fig3_cetram_headtohead.csv"))
    ovl = pd.read_csv(os.path.join(DATA, "fig3_overlap.csv"))
    stats = []

    fig = plt.figure(figsize=(18.5, 15.0))
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.31)
    # ── A: circadian profile ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    w["hr"] = w.clock.astype(int)
    for g in ("Control", "PD"):
        s = w[w.Group == g].groupby("hr").cx.agg(["mean", "sem"])
        ax.plot(s.index, s["mean"], "o-", color=COL[g], lw=2.2, ms=4, label=g)
        ax.fill_between(s.index, s["mean"] - s["sem"], s["mean"] + s["sem"],
                        color=COL[g], alpha=.22)
    ax.axvspan(WIN_LO, WIN_HI, color="gold", alpha=.18, zorder=0)
    ax.text((WIN_LO + WIN_HI) / 2, ax.get_ylim()[1], "18–22 h", ha="center", va="top",
            fontsize=7.5, color="#8A6D00")
    ax.set_xlabel("clock hour"); ax.set_ylabel("rcMSE nAUC(1–5)")
    ax.set_xticks(range(0, 24, 4))
    ax.set_title("Complexity across the day",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=.2)

    # ── B: AUC at every hour — no window nominated ──────────────────────────
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    ax.plot(hourly.hour, hourly.AUC_cx, "o-", color=CX, lw=2.2, ms=4.5, label="complexity")
    ax.plot(hourly.hour, hourly.AUC_sdnn, "s--", color=SD, lw=1.8, ms=4, label="SDNN")
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.axvspan(WIN_LO, WIN_HI, color="gold", alpha=.18, zorder=0)
    ax.set_xlabel("clock hour"); ax.set_ylabel("AUC (Control > PD)")
    ax.set_xticks(range(0, 24, 4)); ax.set_ylim(.30, .90)
    ax.set_title("Discrimination by clock hour",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="lower right"); ax.grid(alpha=.2)

    # ── C: label-blind whole-day test ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    scan = pd.read_csv(os.path.join(DATA, "fig3_window_scan.csv")).sort_values("window")
    bonf = 0.05 / len(scan)
    sig = scan.p < bonf
    ax.vlines(scan.window, 0.5, scan.AUC, color="0.80", lw=1.0, zorder=1)
    ax.scatter(scan.window[~sig], scan.AUC[~sig], s=26, facecolor="white",
               edgecolor=MUTED, lw=1.0, zorder=3, label="n.s. (Bonferroni)")
    ax.scatter(scan.window[sig], scan.AUC[sig], s=30, color=CX, zorder=3,
               label="p < 0.05/24")
    for wv, mk, cc in ((16, "x", "0.35"), (18, "o", "#B8860B")):
        r = scan[scan.window == wv]
        if len(r):
            ax.scatter(r.window, r.AUC, s=95, facecolor="none", edgecolor=cc,
                       lw=1.6, marker=mk if mk == "o" else "o", zorder=4)
    ax.annotate("16–20 h", (16, float(scan[scan.window == 16].AUC.iloc[0])),
                xytext=(-6, 20), textcoords="offset points", fontsize=7.5, ha="right",
                color="0.35", arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8))
    ax.annotate("18–22 h", (18, float(scan[scan.window == 18].AUC.iloc[0])),
                xytext=(10, -26), textcoords="offset points", fontsize=7.5,
                color="#8A6D00", arrowprops=dict(arrowstyle="-", color="#B8860B", lw=0.8))
    ax.axhline(.5, ls=":", c="0.4", lw=0.9)
    ax.set_xlabel("4-h window start (clock hour)")
    ax.set_ylabel("AUC (Control > PD)")
    ax.set_xticks(range(0, 24, 4)); ax.set_ylim(.45, .95)
    ax.set_title("All candidate 4-h windows")
    ax.legend(loc="lower right", fontsize=7.5)

    # ── D: the crossing — AUC vs window length ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); lab(ax, "D")
    for met, c, mk in (("SDNN", SD, "s"), ("complexity", CX, "o")):
        t = ladder[ladder.metric == met].sort_values("win_min")
        ax.plot(t.win_min, t.AUC, mk + "-", color=c, lw=2.6, ms=9, label=met)
    ax.axvspan(0.5, 14, color="#C0392B", alpha=.09, zorder=0)
    ax.text(2.0, .955, r"$N/\tau < 200$", ha="center", fontsize=7.5, color="0.45")
    ax.set_xscale("log")
    ax.set_xticks([1.4, 2.8, 15, 240])
    ax.set_xticklabels(["100\nbeats", "\n\n200 beats", "15\nmin", "4 h"])
    ax.set_xlabel("window length"); ax.set_ylabel("AUC of the min-across-day statistic")
    ax.set_ylim(.55, 1.0)
    ax.set_title("Discrimination vs window length",
                 fontweight="normal", fontsize=12)
    ax.legend(loc="lower left")

    # ── E: Suzuki's four statistics, both metrics ───────────────────────────
    ax = fig.add_subplot(gs[1, 1]); lab(ax, "E")
    x = np.arange(len(dist)); bw = .36
    ax.bar(x - bw / 2, dist.SDNN100, bw, color=SD, label="SDNN, 100-beat windows")
    ax.bar(x + bw / 2, dist.complexity, bw, color=CX, label="complexity, 15-min windows")
    for i, r in dist.iterrows():
        ax.text(i - bw / 2, r.SDNN100 + .008, f"{r.SDNN100:.3f}", ha="center", fontsize=7.5)
        ax.text(i + bw / 2, r.complexity + .008, f"{r.complexity:.3f}", ha="center", fontsize=7.5)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(dist.stat)
    ax.set_ylim(.5, 1.0); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Distributional summaries",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right"); ax.grid(alpha=.2, axis="y")

    # ── F: the punchline ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2]); lab(ax, "F")
    groups = ["Nagoya\n24-h Holter", "CETRAM\n15-min rest", "CETRAM\nexcl. non-sinus"]
    sdv = [0.925, float(cet[cet.subset == "all"].SDNN100_min.iloc[0]),
           float(cet[cet.subset != "all"].SDNN100_min.iloc[0])]
    cxv = [0.845, float(cet[cet.subset == "all"].complexity.iloc[0]),
           float(cet[cet.subset != "all"].complexity.iloc[0])]
    x = np.arange(3)
    ax.bar(x - bw / 2, sdv, bw, color=SD, label="SDNN100-min (Suzuki)")
    ax.bar(x + bw / 2, cxv, bw, color=CX, label="complexity")
    for i in range(3):
        ax.text(i - bw / 2, sdv[i] + .012, f"{sdv[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + bw / 2, cxv[i] + .012, f"{cxv[i]:.3f}", ha="center", fontsize=8)
    ax.axhline(.5, ls=":", c="k", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(.30, 1.00); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Cross-cohort comparison",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right"); ax.grid(alpha=.2, axis="y")

    # ── G: scale-resolved entropy ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0]); lab(ax, "G")
    taus = [t for t in range(1, 61) if f"t{t}" in prof.columns]
    mrr = prof.meanRR.mean() / 1000.0
    secs = np.array(taus) * mrr
    for lo, hi, nm, cc in BANDS:
        ax.axvspan(lo, hi, color=cc, alpha=.75, zorder=0)
    for g in ("Control", "PD"):
        s = prof[prof.Group == g]
        m = np.array([s[f"t{t}"].mean() for t in taus])
        e = np.array([s[f"t{t}"].sem() for t in taus])
        ax.plot(secs, m, color=COL[g], lw=2.6, label=g)
        ax.fill_between(secs, m - e, m + e, color=COL[g], alpha=.22)
    ax.set_xlabel("timescale  $\\tau\\times$ mean RR  (s)"); ax.set_ylabel("sample entropy")
    ax.set_title("Scale-resolved entropy")
    ax.legend(fontsize=9, frameon=False, loc="lower right"); ax.grid(alpha=.2)

    # ── H: AUC vs timescale ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1]); lab(ax, "H")
    yp = (prof.Group == "PD").astype(int).values
    au = []
    for t in taus:
        v = prof[f"t{t}"].values
        ok = np.isfinite(v)
        au.append(roc_auc_score(yp[ok], -v[ok]) if ok.sum() > 10 else np.nan)
    au = np.array(au)
    for lo, hi, nm, cc in BANDS:
        ax.axvspan(lo, hi, color=cc, alpha=.75, zorder=0)
    ax.plot(secs, au, color=CX, lw=2.6, marker="o", ms=3)
    ax.axhline(.5, ls=":", c="k", lw=1)
    k = int(np.nanargmax(au))
    ax.plot(secs[k], au[k], "*", ms=20, color="#F1C40F", markeredgecolor="k", zorder=6)
    ax.annotate(f"$\\tau$={taus[k]}, {au[k]:.3f}",
                (secs[k], au[k]), xytext=(38, -42), textcoords="offset points",
                fontsize=8.5, fontweight="normal",
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("timescale  $\\tau\\times$ mean RR  (s)"); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Discrimination vs timescale", fontweight="normal", fontsize=12)
    ax.grid(alpha=.2)
    stats.append(dict(panel="H", metric="peak_tau", value=taus[k]))
    stats.append(dict(panel="H", metric="peak_auc", value=au[k]))

    # ── I: why non-overlapping, and p10 over min ────────────────────────────
    ax = fig.add_subplot(gs[2, 2]); lab(ax, "I")
    ax.plot(ovl.overlap * 100, ovl.median_min / ovl.median_min.iloc[0], "o-",
            color="#C0392B", lw=2.4, ms=7, label="median minimum (relative)")
    ax.plot(ovl.overlap * 100, ovl.AUC_min / ovl.AUC_min.iloc[0], "s--",
            color="#7F8C8D", lw=1.8, ms=6, label="AUC of min (relative)")
    ax.plot(ovl.overlap * 100, ovl.AUC_p10 / ovl.AUC_p10.iloc[0], "^-",
            color="#16A085", lw=2.4, ms=7, label="AUC of p10 (relative)")
    ax.axhline(1.0, ls=":", c="k", lw=1)
    ax.set_xlabel("window overlap (%)"); ax.set_ylabel("value relative to 0% overlap")
    ax.set_ylim(.86, 1.06)
    ax.set_title("Effect of window overlap",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower left"); ax.grid(alpha=.2)

    pd.DataFrame(stats).to_csv(os.path.join(OUT, "figure3_stats.csv"), index=False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"Figure3.{ext}"), dpi=180, bbox_inches="tight")
    print(f"  peak tau={taus[k]} ({secs[k]:.1f}s) AUC={au[k]:.3f}")
    print(f"  -> {os.path.join(OUT, 'Figure3.png')}")


if __name__ == "__main__":
    main()
