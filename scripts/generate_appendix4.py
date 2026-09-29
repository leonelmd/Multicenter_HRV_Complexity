#!/usr/bin/env python3
"""
Appendix 4: Spectral characterisation, comparison against noise, and the nature
            of the RR-interval time base.

Three things this figure establishes.

1. THE SIGNALS ARE NOT WHITE NOISE. All three cohorts have a 1/f-like spectral
   exponent. CETRAM's rcMSE curve *falls* with scale, which superficially
   resembles Costa's white-noise example, but its exponent places it far from
   white noise; the falling direction comes from the limited low-frequency
   content of a 15-minute recording, not from an absence of structure.

2. THERE IS REAL PHYSIOLOGY BEYOND THE SPECTRUM. Comparing each subject against
   1/f^beta noise matched on length, exponent and standard deviation: real RR
   series are significantly LESS entropic than their spectrally matched
   surrogates at every scale in every cohort. A power-law spectrum alone does
   not reproduce the data.

3. THE RR SERIES IS NOT A UNIFORMLY SAMPLED SIGNAL. Its index is beat number,
   not time, and the interval between successive samples *is* the value being
   measured. Coarse-graining by tau therefore averages tau BEATS, spanning
   tau x meanRR seconds — a span that differs between subjects and, because
   heart rate is elevated in PD, systematically between groups. Courtiol et al.
   (2016) showed for uniformly sampled neural data that MSE at fine scales is
   dominated by broadband low-frequency power; for RR series the mapping from
   scale to frequency is additionally subject-dependent.

Reference
  Courtiol J, Perdikis D, Petkoski S, Muller V, Huys R, Sleimen-Malkoun R,
  Jirsa VK. The multiscale entropy: Guidelines for use and interpretation in
  brain signal analysis. J Neurosci Methods. 2016;273:175-190.

Input : data/spectral_analysis.csv, data/spectral_psd_curves.csv,
        data/chile_mse.csv, data/spain_mse.csv, data/japan_window_mse.csv
Output: figures/Appendix/FigureAppendix4.{png,svg} + appendix4_stats.csv
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
from scipy.stats import wilcoxon, mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Appendix")
os.makedirs(OUT, exist_ok=True)

COH = ["CETRAM", "Cruces", "Nagoya"]
COL = {"CETRAM": "#1565C0", "Cruces": "#2E7D32", "Nagoya": "#E65100"}
GCOL = {"Control": "#2E86AB", "PD": "#D62828"}
NOISE = {"white_b0": ("white noise  $\\beta$=0", "#9E9E9E"),
         "pink_b1": ("1/f  $\\beta$=1", "#7E57C2"),
         "brown_b2": ("brown  $\\beta$=2", "#6D4C41")}


def main():
    fs.apply()
    S = pd.read_csv(os.path.join(DATA, "spectral_analysis.csv"))
    S = S[S.Group.isin(["Control", "PD"])]
    P = pd.read_csv(os.path.join(DATA, "spectral_psd_curves.csv"))
    stats = []

    fig = plt.figure(figsize=(17.5, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    def lab(ax, s):
        ax.text(-0.17, 1.08, s, transform=ax.transAxes, fontsize=17, fontweight="normal")

    # ---------- A: PSD in the beat domain ----------
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    f = P.freq_cycles_per_beat.values
    for k, (nm, c) in NOISE.items():
        if k in P:
            y = P[k].values
            ax.loglog(f, y / y[0], ls="--", lw=1.6, color=c, label=nm, alpha=.85)
    for c in COH:
        if c in P:
            y = P[c].values
            ax.loglog(f, y / y[0], lw=2.6, color=COL[c],
                      label=f"{c}  $\\beta$={S[S.Cohort==c].beta_beat.median():.2f}")
    ax.set_xlabel("frequency (cycles per beat)")
    ax.set_ylabel("normalised PSD")
    ax.axvspan(0.25, 0.5, color="#F5F5F5", zorder=0)
    ax.annotate("spectra flatten toward Nyquist:\nthe measurement noise floor\n"
                "(uncorrelated, white)", xy=(0.35, 0.05), xytext=(0.022, 0.006),
                fontsize=7.5, color="#444",
                arrowprops=dict(arrowstyle="->", lw=1, color="#444"))
    ax.set_title("Spectra in the BEAT domain — what MSE sees",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.2, which="both")

    # ---------- B: beta distributions vs noise ----------
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    for i, c in enumerate(COH):
        v = S[S.Cohort == c].beta_beat.dropna()
        bp = ax.boxplot([v], positions=[i], widths=.6, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(COL[c]); bp["boxes"][0].set_alpha(.75)
        ax.scatter(np.random.normal(i, .07, len(v)), v, s=10, color="k", alpha=.4, zorder=3)
        stats.append(dict(panel="B", cohort=c, metric="beta_beat",
                          median=v.median(), q1=v.quantile(.25), q3=v.quantile(.75)))
    for b, nm, cc in [(0, "white noise", "#9E9E9E"), (1, "1/f", "#7E57C2"), (2, "brown", "#6D4C41")]:
        ax.axhline(b, ls="--", lw=1.5, color=cc)
        ax.text(2.42, b, nm, color=cc, fontsize=7.5, va="bottom", ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(COH)
    ax.set_ylabel("spectral exponent $\\beta$  (beat domain)")
    ax.set_xlim(-0.6, 2.5); ax.set_ylim(-0.45, 2.35)
    ax.set_title("All three cohorts are 1/f-like,\nnone is white noise",
                 fontweight="normal", fontsize=12)
    ax.grid(alpha=.2, axis="y")

    # ---------- C: beta depends on the domain ----------
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    for c in COH:
        s = S[S.Cohort == c]
        ax.scatter(s.beta_beat, s.beta_time, s=26, alpha=.7, color=COL[c], label=c)
    lo, hi = -0.2, 2.0
    ax.plot([lo, hi], [lo, hi], ls=":", c="k", lw=1.2)
    ax.text(1.1, 1.02, "identity", fontsize=8, rotation=38, color="#555")
    for c in COH:
        s = S[S.Cohort == c]
        ax.annotate("", xy=(s.beta_beat.median(), s.beta_time.median()),
                    xytext=(s.beta_beat.median(), s.beta_beat.median()),
                    arrowprops=dict(arrowstyle="-|>", color=COL[c], lw=2.6,
                                    mutation_scale=18))
        stats.append(dict(panel="C", cohort=c, metric="beta_beat_vs_time",
                          median=s.beta_beat.median(), q1=s.beta_time.median(), q3=np.nan))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("$\\beta$ indexed by BEAT  (MSE domain)")
    ax.set_ylabel("$\\beta$ indexed by TIME  (Hz, interpolated)")
    ax.set_title("The exponent depends on the time base\n"
                 "points above identity: $\\beta$ looks steeper in Hz",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(alpha=.2)

    # ---------- D: real vs spectrally matched noise ----------
    ax = fig.add_subplot(gs[1, 0]); lab(ax, "D")
    taus = np.arange(1, 11)
    for c in COH:
        s = S[S.Cohort == c]
        rc = s[[f"real_s{i}" for i in taus]].to_numpy(float)
        mc = s[[f"matched_s{i}" for i in taus]].to_numpy(float)
        ax.plot(taus, np.nanmean(rc, 0), lw=2.6, color=COL[c], marker="o", ms=4, label=f"{c} — real")
        ax.plot(taus, np.nanmean(mc, 0), lw=1.8, ls="--", color=COL[c], alpha=.75,
                label=f"{c} — matched 1/f$^\\beta$")
        for t in taus:
            a, b = rc[:, t - 1], mc[:, t - 1]
            ok = np.isfinite(a) & np.isfinite(b)
            stats.append(dict(panel="D", cohort=c, metric=f"tau{t}",
                              median=np.median(a[ok]), q1=np.median(b[ok]),
                              q3=wilcoxon(a[ok], b[ok])[1]))
    ax.set_xlabel("scale $\\tau$ (beats)"); ax.set_ylabel("sample entropy")
    ax.set_title("Real RR is LESS entropic than noise\nwith the same spectrum "
                 "(all p<0.05, mostly p<1e-8)", fontweight="normal", fontsize=12)
    ax.legend(fontsize=7, frameon=False, ncol=1); ax.grid(alpha=.2)

    # ---------- E: the RR time base ----------
    ax = fig.add_subplot(gs[1, 1]); lab(ax, "E")
    t = np.arange(1, 21)
    for g in ("Control", "PD"):
        mrr = S[(S.Cohort == "CETRAM") & (S.Group == g)].meanRR.median() / 1000.0
        ax.plot(t, t * mrr, lw=2.6, color=GCOL[g], marker="o", ms=3.5,
                label=f"{g}  (mean RR {mrr*1000:.0f} ms)")
    ax.axhspan(2.5, 6.7, color="#DCEEF5", zorder=0)
    ax.axhspan(6.7, 25, color="#FBE3E3", zorder=0)
    ax.text(20, 4.3, "HF / resp.", fontsize=8, ha="right", color="#555")
    ax.text(20, 14, "LF / baroreflex", fontsize=8, ha="right", color="#555")
    mc = S[(S.Cohort == "CETRAM") & (S.Group == "Control")].meanRR.median() / 1000
    mp = S[(S.Cohort == "CETRAM") & (S.Group == "PD")].meanRR.median() / 1000
    ax.annotate(f"at $\\tau$=20 the same scale spans\n{20*mc:.1f} s vs {20*mp:.1f} s "
                f"({100*(mc-mp)/mc:.0f}% apart)",
                xy=(20, 20 * mp), xytext=(8.5, 18.5), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("scale $\\tau$ (beats)"); ax.set_ylabel("physiological span  $\\tau\\times$mean RR  (s)")
    ax.set_title("RR is indexed by BEAT, not time\nthe same $\\tau$ is a different timescale per group",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)

    # ---------- F: curves on a beat axis vs a time axis ----------
    ax = fig.add_subplot(gs[1, 2]); lab(ax, "F")
    mse = pd.read_csv(os.path.join(DATA, "chile_mse.csv"))
    mse = mse[mse.Group.isin(["Control", "PD"])]
    hr = S[S.Cohort == "CETRAM"].set_index("Subject")["meanRR"].to_dict()
    for g in ("Control", "PD"):
        sub = mse[mse.Group == g]
        m = sub.groupby("Scales").MSE.mean()
        ax.plot(m.index, m.values, lw=1.6, ls="--", color=GCOL[g], alpha=.6,
                label=f"{g} — beat axis")
        mrr = np.median([hr[s] for s in sub.Subject.unique() if s in hr]) / 1000.0
        ax.plot(m.index * mrr, m.values, lw=2.8, color=GCOL[g], marker="o", ms=3.5,
                label=f"{g} — time axis")
    ax.set_xlabel("scale $\\tau$ (beats, dashed)   /   $\\tau\\times$mean RR (s, solid)")
    ax.set_ylabel("sample entropy")
    ax.set_title("Re-expressing the x-axis in seconds\nshifts the groups relative to each other",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.2)

    p = os.path.join(OUT, "FigureAppendix4.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats).to_csv(os.path.join(OUT, "appendix4_stats.csv"), index=False)

    print("beta (beat domain), median per cohort:")
    for c in COH:
        s = S[S.Cohort == c]
        print(f"  {c:8s} beta_beat={s.beta_beat.median():.3f}  "
              f"beta_time={s.beta_time.median():.3f}  beta_var={s.beta_var.median():.3f}")
    print("\nreal vs matched noise, tau=1:")
    for c in COH:
        s = S[S.Cohort == c]
        a = s.real_s1.to_numpy(float); b = s.matched_s1.to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"  {c:8s} real={np.median(a[ok]):.3f}  matched={np.median(b[ok]):.3f}  "
              f"p={wilcoxon(a[ok],b[ok])[1]:.2e}")
    print("\n ->", p)


if __name__ == "__main__":
    main()
