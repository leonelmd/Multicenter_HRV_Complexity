#!/usr/bin/env python3
"""
Appendix 3: Sensitivity of rcMSE to recording length
====================================================

Each subject's RR series is truncated to a ladder of beat counts (250 … 4000)
and rcMSE nAUC(1–5) recomputed at each. Nagoya spans the whole ladder and so
acts as an internal reference: truncating it to CETRAM's length or to Cruces'
length shows what those cohorts could possibly achieve with their recordings,
independent of any cohort difference.

Three questions this answers:

  1. Is the Cruces result weak because PPG is a poor signal, or because 7.4 min
     is short?  ->  Nagoya truncated to 500 beats performs like Cruces.
  2. Should CETRAM use its full recording or a fixed beat count?
     ->  discrimination rises monotonically with N; use the full record.
  3. Can a short-recording estimate be extrapolated to the long-recording value?
     ->  No. nAUC is essentially unbiased in N (paired differences non-significant);
         what shortness costs is variance, and no post-hoc transform recovers it.

Input : data/length_ladder.csv   (scripts/compute_length_ladder.py)
Output: figures/Appendix/FigureAppendix3.{png,svg} + appendix3_stats.csv
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
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Appendix")
os.makedirs(OUT, exist_ok=True)

COL = {"CETRAM": "#1565C0", "Cruces": "#2E7D32", "Nagoya": "#E65100"}
MIN_N = 15          # minimum subjects before a ladder point is plotted
OPERATING = {"Cruces": 498, "CETRAM": 1088}   # median beats actually analysed


def auc_cp(v, y):
    v = np.asarray(v, float); ok = np.isfinite(v)
    return roc_auc_score(np.asarray(y)[ok], -v[ok]) if ok.sum() > 10 else np.nan


def main():
    fs.apply()
    d = pd.read_csv(os.path.join(DATA, "length_ladder.csv"))
    d = d[d.Group.isin(["Control", "PD"])]
    lad = sorted(d.N.unique())
    stats = []

    # ---------- aggregate per cohort x N ----------
    # Restrict each cohort to the subjects present at EVERY one of its ladder
    # points, so the within-cohort trend is not confounded by attrition. Without
    # this, Cruces appears to get *noisier* with length simply because only the
    # longer recordings survive past N=500.
    keep = {}
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        dc = d[d.Cohort == c]
        usable = [N for N in lad if dc[(dc.N == N) & np.isfinite(dc.nAUC)].Subject.nunique() >= MIN_N]
        if not usable:
            keep[c] = (set(), [])
            continue
        Nmax = max(usable)
        subs = set(dc[(dc.N == Nmax) & np.isfinite(dc.nAUC)].Subject)
        keep[c] = (subs, [N for N in lad if N <= Nmax])
        print(f"  {c}: common subset n={len(subs)}, ladder {min(keep[c][1])}-{Nmax}")

    agg = []
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        subs, ladc = keep[c]
        for N in ladc:
            s = d[(d.Cohort == c) & (d.N == N) & d.Subject.isin(subs) & np.isfinite(d.nAUC)]
            if len(s) < MIN_N:
                continue
            y = (s.Group == "PD").astype(int).values
            ok = np.isfinite(s.h1) & np.isfinite(s.h2)
            agg.append(dict(
                Cohort=c, N=N, n=len(s),
                mean_nauc=s.nAUC.mean(), sd=s.nAUC.std(ddof=1),
                se=s.nAUC.std(ddof=1) / np.sqrt(len(s)),
                auc=auc_cp(s.nAUC, y),
                split_half=spearmanr(s.h1[ok], s.h2[ok])[0] if ok.sum() > 5 else np.nan,
                roughness=s.roughness.mean()))
    A = pd.DataFrame(agg)
    stats += A.to_dict("records")

    # ---------- Nagoya paired bias test ----------
    NG = d[d.Cohort == "Nagoya"].pivot_table(index="Subject", columns="N", values="nAUC")
    ref_N = max(c for c in NG.columns)
    bias = []
    for N in lad:
        if N not in NG.columns or N == ref_N:
            continue          # selecting [ref_N, ref_N] would duplicate the column
        pair = NG[[N, ref_N]].dropna()
        if len(pair) < 10:
            continue
        a = pair[N].to_numpy(float); b = pair[ref_N].to_numpy(float)
        diff = a - b
        bias.append(dict(N=N, n=len(pair), mean_diff=float(diff.mean()),
                         p=float(wilcoxon(a, b)[1]),
                         rho=float(spearmanr(a, b)[0]),
                         rmse=float(np.sqrt(np.mean(diff ** 2)))))
    B = pd.DataFrame(bias)

    # ================= PLOT =================
    fig = plt.figure(figsize=(17, 10.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.34)

    def lab(ax, s):
        ax.text(-0.17, 1.08, s, transform=ax.transAxes, fontsize=17, fontweight="normal")

    def marks(ax):
        for c, N in OPERATING.items():
            ax.axvline(N, ls="--", lw=1.3, color=COL[c], alpha=.8)
            ax.text(N, ax.get_ylim()[1], f" {c}\n {N}", color=COL[c], fontsize=7.5,
                    va="top", ha="left")

    # --- A: nAUC vs N (the estimate is flat) ---
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        s = A[A.Cohort == c]
        if s.empty: continue
        ax.errorbar(s.N, s["mean_nauc"], yerr=s["se"], marker="o", ms=5, lw=2,
                    capsize=3, color=COL[c], label=f"{c} (n={s.n.max()})")
    ax.set_xscale("log"); ax.set_xlabel("beats analysed"); ax.set_ylabel("rcMSE nAUC(1–5)")
    ax.set_title("Index vs analysed length", fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)
    marks(ax)

    # --- B: precision ---
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        s = A[A.Cohort == c]
        if s.empty: continue
        ax.plot(s.N, s.sd, marker="o", ms=5, lw=2, color=COL[c], label=c)
    ng = A[A.Cohort == "Nagoya"]
    if len(ng) > 2:
        k = ng.sd.iloc[0] * np.sqrt(ng.N.iloc[0])
        ax.plot(ng.N, k / np.sqrt(ng.N), ls=":", color="k", lw=1.5, label="$1/\\sqrt{N}$")
    ax.set_xscale("log"); ax.set_xlabel("beats analysed")
    ax.set_ylabel("between-subject SD of nAUC")
    ax.set_title("Estimator precision vs length", fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)
    marks(ax)

    # --- C: discrimination (the key panel) ---
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        s = A[A.Cohort == c]
        if s.empty: continue
        ax.plot(s.N, s.auc, marker="o", ms=6, lw=2.4, color=COL[c], label=c)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("beats analysed")
    ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Discrimination vs length",
                 fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)
    marks(ax)

    # --- D: paired bias, Nagoya ---
    ax = fig.add_subplot(gs[1, 0]); lab(ax, "D")
    ax.axhline(0, color="k", lw=1)
    ax.errorbar(B.N, B.mean_diff,
                yerr=[B.rmse / np.sqrt(B.n), B.rmse / np.sqrt(B.n)],
                marker="o", ms=6, lw=2, capsize=3, color=COL["Nagoya"])
    for _, r in B.iterrows():
        if np.isfinite(r.p):
            ax.text(r.N, r.mean_diff + 0.006, "n.s." if r.p > .05 else f"p={r.p:.3f}",
                    ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(f"beats analysed"); ax.set_ylabel(f"nAUC(N) − nAUC({ref_N})")
    ax.set_title(f"No length bias to extrapolate away\n(paired, within subject, Nagoya n={B.n.max()})",
                 fontweight="normal", fontsize=12)
    ax.grid(alpha=.2)

    # --- E: agreement with the long-recording value ---
    ax = fig.add_subplot(gs[1, 1]); lab(ax, "E")
    ax.plot(B.N, B.rho, marker="o", ms=6, lw=2, color="#6C3483", label="Spearman $\\rho$")
    ax.set_xscale("log"); ax.set_xlabel("beats analysed")
    ax.set_ylabel(f"Spearman $\\rho$ with nAUC({ref_N})", color="#6C3483", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#6C3483")
    ax2 = ax.twinx()
    ax2.plot(B.N, B.rmse, marker="s", ms=5, lw=2, ls="--", color="#C0392B", label="RMSE")
    ax2.set_ylabel("RMSE", color="#C0392B", labelpad=2); ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax.set_title("A short segment is a noisy\nestimate of the long-segment value",
                 fontweight="normal", fontsize=12)
    ax.grid(alpha=.2)
    marks(ax)

    # --- F: curve roughness ---
    ax = fig.add_subplot(gs[1, 2]); lab(ax, "F")
    for c in ["CETRAM", "Cruces", "Nagoya"]:
        s = A[A.Cohort == c]
        if s.empty: continue
        ax.plot(s.N, s.roughness, marker="o", ms=5, lw=2, color=COL[c], label=c)
    ax.set_xscale("log"); ax.set_xlabel("beats analysed")
    ax.set_ylabel("curve roughness")
    ax.set_title("Curve smoothness is set by length", fontweight="normal", fontsize=12)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.2)
    marks(ax)

    p = os.path.join(OUT, "FigureAppendix3.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)

    pd.concat([A.assign(panel="ABCF"), B.assign(panel="DE", Cohort="Nagoya")],
              ignore_index=True).to_csv(os.path.join(OUT, "appendix3_stats.csv"), index=False)

    print(A.to_string(index=False))
    print("\npaired bias vs N =", ref_N)
    print(B.to_string(index=False))
    print("\n ->", p)


if __name__ == "__main__":
    main()
