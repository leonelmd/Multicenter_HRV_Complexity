#!/usr/bin/env python3
"""Data-driven elements for Figure 1 (overview), to be placed in the hand-drawn layout.

Every trace and curve is drawn from the released data, so the schematic cannot
contradict the paper. Text is kept as live text (svg.fonttype = none) so labels can
be edited in Affinity/Inkscape; backgrounds are transparent.

Outputs (figures/Fig1_overview/elements/):
  rr_input.svg            RR-interval series (CETRAM control, 150 beats)       -> input
  A_variability.svg       RR histogram with mean and +-1 SD                     -> branch A
  B_coarse_graining.svg   the same series coarse-grained at tau = 1, 3, 5       -> branch B
  B_entropy_curve.svg     sample entropy against scale for that series, 1-5     -> branch B
  result_cetram.svg       group-mean rcMSE curves, CETRAM (42 C / 27 PD)        -> outcome
  result_three_cohorts.svg  the three cohorts side by side (as Fig. 3A-D)       -> outcome
  rr_control.svg, rr_pd.svg  representative CETRAM series (closest to group median index)

Needs CETRAM/public_release/results/cleaned_detections (raw beat times) for the traces.
Run: python scripts/fig1_elements.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rcmse_fast import rcmse
BASE = os.path.dirname(HERE); D = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "figures", "Fig1_overview", "elements"); os.makedirs(OUT, exist_ok=True)
from paths import CETRAM_PEAKS as DET, require
require(DET, "the original CETRAM beat series (two example recordings)")
C_COL, P_COL, INK, GREY = "#4C72B0", "#C44E52", "#2B2B2B", "#8C8C8C"
NEUTRAL = "#9A9A9A"   # schematic elements: never the Control/PD colours
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

def nauc(c): c = np.asarray(c, float); return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c))
def index_table(f, hi):
    m = pd.read_csv(os.path.join(D, f)); m = m[(m.Scales <= hi) & (m.Group.isin(["Control", "PD"]))]
    return m.sort_values("Scales").groupby(["Subject", "Group"]).MSE.apply(lambda s: nauc(s.values)).reset_index(name="idx")
def rr(subject, group):
    s = pd.read_csv(os.path.join(DET, group, f"{subject}_cleaned.csv"))["sample"].to_numpy(float)
    x = np.diff(s); return x[(x > 300) & (x < 2000)]
def representative(tab, g):
    t = tab[tab.Group == g]; return t.iloc[(t.idx - t.idx.median()).abs().argsort().iloc[0]].Subject
def clean_window(x, n, start=60):
    # first window of n beats with no interval deviating >20% from its 31-beat running median
    med = pd.Series(x).rolling(31, center=True, min_periods=5).median().to_numpy()
    bad = np.abs(x - med) > 0.2 * med
    for a in range(start, len(x) - n):
        if not bad[a:a + n].any(): return x[a:a + n]
    return x[start:start + n]
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), transparent=True, bbox_inches="tight"); plt.close(fig); print("wrote", name)
def bare(ax):
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

tab = index_table("chile_mse.csv", 5)
c_id, p_id = representative(tab, "Control"), representative(tab, "PD")
xc_full, xp_full = rr(c_id, "Control"), rr(p_id, "PD")
xc, xp = clean_window(xc_full, 150), clean_window(xp_full, 150)
print("representative control", c_id, "PD", p_id)

# 1. input series
fig, ax = plt.subplots(figsize=(2.6, 0.9))
ax.plot(np.arange(len(xc)), xc, color=INK, lw=0.9); bare(ax)
ax.set_xlabel("beat number", fontsize=8); ax.set_ylabel("RR (ms)", fontsize=8)
save(fig, "rr_input.svg")

# 2. branch A: distribution of the intervals
fig, ax = plt.subplots(figsize=(1.9, 1.3))
mu, sd = xc_full.mean(), xc_full.std(ddof=1)
ax.axvspan(mu - sd, mu + sd, color=NEUTRAL, alpha=0.18, lw=0)
ax.hist(xc_full, bins=30, color=NEUTRAL, edgecolor="white", lw=0.4)
ax.axvline(mu, color=INK, lw=0.8)
ax.annotate("", xy=(mu + sd, ax.get_ylim()[1] * 0.92), xytext=(mu - sd, ax.get_ylim()[1] * 0.92),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK))
ax.text(mu, ax.get_ylim()[1] * 0.97, "±1 SD (SDNN)", ha="center", va="bottom", fontsize=7.5)
ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("RR interval (ms)", fontsize=8)
save(fig, "A_variability.svg")

# 3. branch B: coarse-graining of the same series (non-overlapping block means)
fig, axs = plt.subplots(3, 1, figsize=(2.3, 2.2), sharex=True)
for ax, tau in zip(axs, (1, 3, 5)):
    L = len(xc) // tau; cg = xc[:L * tau].reshape(L, tau).mean(1)
    ax.step(np.arange(L) * tau, cg, where="post", color=INK, lw=0.9)
    ax.set_ylim(xc.min() - 10, xc.max() + 10); bare(ax)
    ax.text(-0.02, 0.5, f"τ = {tau}", transform=ax.transAxes, ha="right", va="center", fontsize=8)
save(fig, "B_coarse_graining.svg")

# 4. branch B: entropy against scale for that series (scales 1-5, r = 0.2 SD fixed)
e = rcmse(xc_full, 5)
fig, ax = plt.subplots(figsize=(1.7, 1.3))
ax.plot(range(1, 6), e, "o-", color=INK, ms=3.5, lw=1)
ax.set_xticks(range(1, 6)); ax.set_xlabel("scale τ", fontsize=8); ax.set_ylabel("sample entropy", fontsize=8)
ax.set_ylim(0, max(e) * 1.15)
ax.fill_between(range(1, 6), e, 0, color=NEUTRAL, alpha=0.25, lw=0)
ax.text(3, max(e) * 0.35, "area =\ncomplexity index", ha="center", va="center", fontsize=7)
save(fig, "B_entropy_curve.svg")

# 5. outcome: group-mean curves
def curves(ax, f, hi, title, nlab):
    m = pd.read_csv(os.path.join(D, f)); m = m[(m.Scales <= hi) & (m.Group.isin(["Control", "PD"]))]
    for g, col in (("Control", C_COL), ("PD", P_COL)):
        s = m[m.Group == g].groupby("Scales").MSE; mu, se = s.mean(), s.std() / np.sqrt(s.count())
        ax.plot(mu.index, mu.values, color=col, lw=1.4, label=g)
        ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=0.2, lw=0)
    ax.set_title(title, fontsize=8.5); ax.set_xlabel("scale τ", fontsize=8)
    n = m.groupby("Group").Subject.nunique()
    ax.text(0.5, -0.42, f"{n['Control']} {nlab} / {n['PD']} PD", transform=ax.transAxes, ha="center", va="top", fontsize=7, color=GREY)
fig, ax = plt.subplots(figsize=(1.9, 1.5))
curves(ax, "chile_mse.csv", 5, "", "C"); ax.set_ylabel("sample entropy", fontsize=8)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "result_cetram.svg")
fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.6))
curves(axs[0], "chile_mse.csv", 5, "Santiago · ECG", "C")
curves(axs[1], "spain_mse.csv", 5, "Bilbao · PPG", "C")
curves(axs[2], "japan_window_mse.csv", 20, "Nagoya · chest strap", "DC")
axs[0].set_ylabel("sample entropy", fontsize=8)
axs[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.55))
fig.subplots_adjust(wspace=0.35)
save(fig, "result_three_cohorts.svg")

# 6. representative series per group, same axes
lo, hi = min(xc.min(), xp.min()) - 20, max(xc.max(), xp.max()) + 20
for x, col, name in ((xc, C_COL, "rr_control.svg"), (xp, P_COL, "rr_pd.svg")):
    fig, ax = plt.subplots(figsize=(2.4, 0.7)); ax.plot(x, color=col, lw=0.9); ax.set_ylim(lo, hi); bare(ax); save(fig, name)
