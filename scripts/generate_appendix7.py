#!/usr/bin/env python3
"""
Appendix 7 — The Cruces null in the context of the source study.

The Cruces cohort is the cohort of Iniguez et al. (npj Parkinsons Dis 2022;8:64), whose
reported group effect was a breakdown in synchronization between HRV and the BOLD signal
across the central autonomic network, not a difference in HRV itself. The HRV metrics in
that work were inputs to the synchronization analysis.

This figure asks whether our complexity null is specific to our index. It is not: the
full resting HRV metric set of the source study, recomputed in the same participants,
also fails to separate the groups, while the clinical autonomic measures reported in that
paper separate them clearly.

Panel C reproduces PUBLISHED values from the source paper. Those were not recomputed here
and are plotted on their own axis, clearly marked, so that nothing implies we had access
to their BOLD or autonomic-test data.

Input : data/cruces_iniguez_replication.csv  (from the replication in Results)
Output: figures/Appendix/FigureAppendix7.{png,svg}
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures", "Appendix")

FAMCOL = {"time": "#4C72B0", "nonlin": "#55A868", "spec": "#DD8452", "ours": "#8172B3"}
FAMLAB = {"time": "time domain", "nonlin": "non-linear", "spec": "spectral",
          "ours": "this study"}

# Published values from Iniguez et al. 2022, Results section. NOT recomputed here.
PUBLISHED = [
    ("SCOPA-AUT total",                 0.05,  "higher in PD (2x control)"),
    ("Orthostatic hypotension",         0.017, "70% PD vs 15.4% control"),
    ("Valsalva pressure recovery time", 0.021, "sympathetic"),
    ("Valsalva $\\Delta$SBP phase IV",  0.002, "sympathetic"),
    ("Deep breathing E/I ratio",        0.051, "cardiovagal"),
]


def main():
    fs.apply()
    d = pd.read_csv(os.path.join(DATA, "cruces_iniguez_replication.csv")).sort_values("AUC")

    fig = plt.figure(figsize=(15.0, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.42)

    # ── A: forest plot of every resting metric ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); fs.panel(ax, "a", dx=-0.42)
    ypos = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        c = FAMCOL[r.family]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2.0, solid_capstyle="round", alpha=.85)
        ax.plot(r.AUC, i, "o", color=c, ms=7 if r.family == "ours" else 5.5,
                markeredgecolor="k" if r.family == "ours" else "none",
                markeredgewidth=1.1 if r.family == "ours" else 0, zorder=3)
    ax.axvline(.5, ls="--", c="0.35", lw=1.0)
    ax.set_yticks(ypos); ax.set_yticklabels(d.metric)
    ax.set_xlabel("AUC (Control > PD), 95% CI")
    ax.set_xlim(.22, .88)
    ax.set_title("Resting HRV, recomputed in the same participants")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=FAMCOL[k], marker="o", ls="", label=FAMLAB[k])
               for k in ["time", "nonlin", "spec", "ours"]]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)

    # ── B: p values on a log axis ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); fs.panel(ax, "b", dx=-0.30)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([0.03, r.p], [i, i], color=FAMCOL[r.family], lw=1.0, alpha=.45)
        ax.plot(r.p, i, "o", color=FAMCOL[r.family],
                ms=7 if r.family == "ours" else 5.5,
                markeredgecolor="k" if r.family == "ours" else "none",
                markeredgewidth=1.1 if r.family == "ours" else 0, zorder=3)
    ax.axvline(.05, ls="--", c="#C44E52", lw=1.2)
    ax.text(.052, len(d) - 0.4, "p = 0.05", fontsize=7.5, color="#C44E52", va="top")
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([])
    ax.set_xscale("log"); ax.set_xlim(.03, 1.0)
    ax.set_xlabel("Mann-Whitney $p$")
    ax.set_title("Group comparison $p$ values")
    ax.grid(axis="y", visible=False)

    # ── C: published clinical autonomic findings ────────────────────────────
    ax = fig.add_subplot(gs[0, 2]); fs.panel(ax, "c", dx=-0.30)
    names = [n for n, _, _ in PUBLISHED][::-1]
    pv = [p for _, p, _ in PUBLISHED][::-1]
    notes = [s for _, _, s in PUBLISHED][::-1]
    yy = np.arange(len(names))
    for i, v in enumerate(pv):
        ax.plot([0.0015, v], [i, i], color="#937860", lw=1.0, alpha=.45)
    ax.plot(pv, yy, "o", color="#937860", ms=6, zorder=3)
    ax.axvline(.05, ls="--", c="#C44E52", lw=1.2)
    for i, (v, s) in enumerate(zip(pv, notes)):
        ax.text(max(v, 0.0022) * 1.35, i, s, va="center", fontsize=7, color="0.35")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(.0015, 0.9)
    ax.set_xlabel("$p$ reported by Iniguez et al.")
    ax.set_title("Clinical autonomic tests, as published")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, -0.20, "values from the source paper; not recomputed here",
            transform=ax.transAxes, ha="right", fontsize=7, style="italic", color="0.45")

    fs.save(fig, OUT, "FigureAppendix7")
    print(f"  metrics with CI excluding 0.5: {((d.lo > .5) | (d.hi < .5)).sum()} of {len(d)}")
    print(f"  -> {os.path.join(OUT, 'FigureAppendix7.png')}")


if __name__ == "__main__":
    main()
