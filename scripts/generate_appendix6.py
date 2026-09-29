#!/usr/bin/env python3
"""
Appendix 6 — Sensitivity of the complexity index to the entropy parameters.

Sweeps the two free parameters of sample entropy, crossed:

  tolerance   r = k x SD(subject),  k = 0.05 .. 0.50   (published: k = 0.2)
  dimension   m = 1, 2, 3                              (published: m = 2)

r is always scaled to each subject's own SD — a fixed absolute tolerance is not
considered here, because the index must be blind to amplitude to be a complexity
measure (the amplitude-sensitivity of an absolute r is documented separately in
CRUCES_NULL_AND_TOLERANCE_CONFOUND.md).

IMPORTANT — the parameters must NOT be chosen to maximise AUC. That is tuning a
hyperparameter on the outcome. The legitimate criteria are estimator validity (no
undefined values) and split-half reliability, both shown here. AUC is displayed only
to demonstrate that the published conclusions do not depend on the choice.

Input : data/r_sweep.csv  (from scripts/compute_r_sweep.py)
Output: figures/Appendix/FigureAppendix6.{png,svg} + appendix6_stats.csv
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
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Appendix")
os.makedirs(OUT, exist_ok=True)

COH = ["CETRAM", "Cruces", "Nagoya"]
CC = {"CETRAM": "#2E86AB", "Cruces": "#1B8A5A", "Nagoya": "#D35400"}
PUB_K, PUB_M = 0.20, 2
MSTYLE = {1: (":", 4), 2: ("-", 6), 3: ("--", 4)}


def summarise(g):
    n = len(g)
    ok = g.dropna(subset=["cx"])
    out = dict(n=len(ok), undef=100 * (n - len(ok)) / n, AUC=np.nan, p=np.nan, rel=np.nan)
    if len(ok) < 10 or ok.Group.nunique() < 2:
        return pd.Series(out)
    y = (ok.Group == "PD").astype(int)
    out["AUC"] = roc_auc_score(y, -ok.cx)
    out["p"] = mannwhitneyu(ok[ok.Group == "Control"].cx, ok[ok.Group == "PD"].cx).pvalue
    hh = ok.dropna(subset=["h1", "h2"])
    if len(hh) > 10:
        out["rel"] = spearmanr(hh.h1, hh.h2).statistic
    return pd.Series(out)


def lab(ax, L):
    ax.text(-0.22, 1.13, L, transform=ax.transAxes, fontsize=16, fontweight="normal", va="top")


def main():
    fs.apply()
    d = pd.read_csv(os.path.join(DATA, "r_sweep.csv"))
    d = d[d["mode"] == "k"]                      # per-subject tolerance only
    S = (d.groupby(["Cohort", "m", "level"])
           .apply(summarise, include_groups=False).reset_index())
    S.to_csv(os.path.join(OUT, "appendix6_stats.csv"), index=False)

    fig = plt.figure(figsize=(19.5, 9.6))
    gs = fig.add_gridspec(2, 4, hspace=0.46, wspace=0.34)
    # ---- A: AUC vs k, all m -----------------------------------------------
    ax = fig.add_subplot(gs[0, 0]); lab(ax, "A")
    for c in COH:
        for mm in (1, 2, 3):
            t = S[(S.Cohort == c) & (S.m == mm)].sort_values("level")
            ls, ms = MSTYLE[mm]
            ax.plot(t.level, t.AUC, ls, color=CC[c], lw=2.4 if mm == 2 else 1.3,
                    alpha=1.0 if mm == 2 else .5, marker="o", ms=ms if mm == 2 else 0)
    ax.axvline(PUB_K, ls="--", c="k", lw=1.3); ax.axhline(.5, ls=":", c="k", lw=1)
    for c in COH:
        ax.plot([], [], "-", color=CC[c], lw=2.4, label=c)
    ax.plot([], [], "-", color="grey", lw=2.4, label="$m$=2 (bold)")
    ax.plot([], [], ":", color="grey", lw=1.3, label="$m$=1, 3 (faint)")
    ax.set_xlabel("$k$  in  $r = k \\times$ SD")
    ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Discrimination vs tolerance", fontweight="normal", fontsize=12)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.grid(alpha=.2)

    # ---- B: reliability vs k ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); lab(ax, "B")
    for c in COH:
        for mm in (1, 2, 3):
            t = S[(S.Cohort == c) & (S.m == mm)].sort_values("level")
            ls, ms = MSTYLE[mm]
            ax.plot(t.level, t.rel, ls, color=CC[c], lw=2.4 if mm == 2 else 1.3,
                    alpha=1.0 if mm == 2 else .5, marker="o", ms=ms if mm == 2 else 0)
            bad = t[t.undef > 1]
            ax.scatter(bad.level, bad.rel, s=130, facecolors="none", edgecolors="red",
                       lw=1.8, zorder=5)
    ax.axvline(PUB_K, ls="--", c="k", lw=1.3)
    ax.set_xlabel("$k$  in  $r = k \\times$ SD")
    ax.set_ylabel("split-half reliability ($\\rho$)")
    ax.set_title("Split-half reliability vs tolerance",
                 fontweight="normal", fontsize=12)
    ax.grid(alpha=.2)

    # ---- C: AUC vs m at k=0.2 ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); lab(ax, "C")
    w = .26
    for i, c in enumerate(COH):
        for j, mm in enumerate((1, 2, 3)):
            v = S[(S.Cohort == c) & (S.m == mm) & (np.isclose(S.level, PUB_K))].AUC
            if not len(v):
                continue
            ax.bar(i + (j - 1) * w, v.iloc[0], w, color=CC[c],
                   alpha=[.45, 1.0, .7][j], edgecolor="k" if mm == PUB_M else "none",
                   lw=1.6 if mm == PUB_M else 0)
            ax.text(i + (j - 1) * w, v.iloc[0] + (.030 if j == 1 else .008),
                    f"{v.iloc[0]:.3f}", ha="center", fontsize=7)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(COH)
    ax.set_ylim(.45, .88); ax.set_ylabel("AUC (Control > PD)")
    ax.set_title("Discrimination vs embedding dimension",
                 fontweight="normal", fontsize=11.5)
    ax.grid(alpha=.2, axis="y")

    # ---- D: reliability + validity vs m -----------------------------------
    ax = fig.add_subplot(gs[0, 3]); lab(ax, "D")
    for c in COH:
        t = S[(np.isclose(S.level, PUB_K)) & (S.Cohort == c)].sort_values("m")
        ax.plot(t.m, t.rel, "o-", color=CC[c], lw=2.2, ms=7, label=c)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("embedding dimension $m$")
    ax.set_ylabel("split-half reliability ($\\rho$)")
    ax.set_title("Reliability vs embedding dimension", fontweight="normal", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False); ax.grid(alpha=.2)

    # ---- E-G: AUC heatmaps over (k, m) -------------------------------------
    ks = np.sort(S.level.unique())
    for i, c in enumerate(COH):
        ax = fig.add_subplot(gs[1, i]); lab(ax, "EFG"[i])
        M = np.full((3, len(ks)), np.nan); U = np.zeros((3, len(ks)))
        for a, mm in enumerate((1, 2, 3)):
            for b, kk in enumerate(ks):
                row = S[(S.Cohort == c) & (S.m == mm) & (np.isclose(S.level, kk))]
                if len(row):
                    M[a, b] = row.AUC.iloc[0]; U[a, b] = row.undef.iloc[0]
        ax.grid(False)
        im = ax.imshow(M, aspect="auto", cmap="RdYlBu_r", vmin=.45, vmax=.85, origin="lower")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k:g}" for k in ks], rotation=90, fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(["1", "2", "3"])
        ax.set_xlabel("$k$"); ax.set_ylabel("$m$")
        for a in range(3):
            for b in range(len(ks)):
                if not np.isfinite(M[a, b]):
                    continue
                if U[a, b] > 1:      # estimator undefined for part of the cohort
                    ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=True,
                                               facecolor="white", alpha=.72, zorder=3))
                    ax.text(b, a, "x", ha="center", va="center", fontsize=9,
                            color="red", fontweight="normal", zorder=4)
                else:
                    ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center", fontsize=6.2,
                            color="white" if (M[a, b] > .78 or M[a, b] < .52) else "black")
        bi = int(np.where(np.isclose(ks, PUB_K))[0][0])
        ax.add_patch(plt.Rectangle((bi - .5, PUB_M - 1 - .5), 1, 1, fill=False,
                                   edgecolor="k", lw=2.4))
        ax.set_title(f"{c}", fontweight="normal", fontsize=11.5,
                     color=CC[c])
        if i == 2:
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03).set_label("AUC", fontsize=8)

    # ---- H: verdict --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3]); lab(ax, "H"); ax.axis("off")
    rows = []
    for c in COH:
        t = S[S.Cohort == c]
        pub = t[(np.isclose(t.level, PUB_K)) & (t.m == PUB_M)].AUC.iloc[0]
        g = t[t.level >= 0.125]
        rows.append([c, f"{pub:.3f}", f"{g.AUC.min():.3f}–{g.AUC.max():.3f}"])
    tb = ax.table(cellText=rows,
                  colLabels=["cohort", "published\n$k$=0.2, $m$=2", "range,\nusable grid"],
                  cellLoc="center", loc="upper center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.85)
    for j in range(3):
        tb[0, j].set_facecolor("#E8E8E8"); tb[0, j].set_text_props(fontweight="normal")

    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"FigureAppendix6.{ext}"), dpi=200, bbox_inches="tight")
    print(f" -> {os.path.join(OUT, 'FigureAppendix6.png')}")

    print(f"\n{'cohort':9s}{'m':>3s}{'k':>7s}{'AUC':>8s}{'rel':>8s}{'undef%':>8s}")
    for _, r in S[np.isclose(S.level, PUB_K)].sort_values(["Cohort", "m"]).iterrows():
        print(f"{r.Cohort:9s}{int(r.m):>3d}{r.level:>7.2f}{r.AUC:>8.3f}{r.rel:>8.3f}{r.undef:>8.1f}")


if __name__ == "__main__":
    main()
