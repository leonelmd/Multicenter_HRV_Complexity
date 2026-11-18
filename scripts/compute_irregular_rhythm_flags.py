#!/usr/bin/env python3
"""
Irregular beat-to-beat rhythm flag (Methods, interbeat-interval processing, step 4).

A recording is flagged when >= 0.30 of its detrended interval variance (31-beat centred
moving mean removed; Welch, nperseg = min(128, N/2)) lies at 0.35-0.50 cycles/beat.
Computed on the analysed series (loaders of compute_length_ladder.py; Nagoya
18:00-22:00 window as in compute_japan_window_features.py). Labels are not read
until the sensitivity AUC is computed at the end.

Output: data/irregular_rhythm_flags.csv ; prints the sensitivity analysis.
Run: python scripts/compute_irregular_rhythm_flags.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_spectral_analysis import series
import compute_length_ladder as L

THR = 0.30


def hf_frac(x):
    y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().values
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y) // 2))
    return p[f >= 0.35].sum() / p[f > 0.02].sum()


def nauc_idx(path, hi):
    d = pd.read_csv(path); d = d[(d.Scales >= 1) & (d.Scales <= hi)].sort_values("Scales")
    return d.groupby("Subject").MSE.apply(
        lambda c: np.trapezoid(c.values, np.arange(1, len(c) + 1)) / len(c))


def main():
    rows = [dict(Cohort=c, Subject=s, Group=g, n=len(x), hf=hf_frac(x)) for c, s, g, x in series()]
    R = pd.DataFrame(rows); R["flag"] = R.hf >= THR
    R.to_csv(os.path.join(L.DATA, "irregular_rhythm_flags.csv"), index=False)
    idx = {"CETRAM": nauc_idx(os.path.join(L.DATA, "chile_mse.csv"), 5),
           "Cruces": nauc_idx(os.path.join(L.DATA, "spain_mse.csv"), 5),
           "Nagoya": nauc_idx(os.path.join(L.DATA, "japan_window_mse.csv"), 20)}
    for c, g in R.groupby("Cohort", sort=False):
        g = g.assign(idx=g.Subject.map(idx[c]))
        k = g[~g.flag]
        a, b = k[k.Group == "Control"].idx, k[k.Group == "PD"].idx
        u = mannwhitneyu(a, b, alternative="two-sided")
        print(f"{c:7s} n={len(g)} flagged={g.flag.sum()} ({(g.flag & (g.Group=='PD')).sum()} PD) "
              f"{sorted(g[g.flag].Subject)}  median share {g.hf.median():.3f}  "
              f"AUC without flagged {u.statistic/(len(a)*len(b)):.3f} p={u.pvalue:.3f}")


if __name__ == "__main__":
    main()
