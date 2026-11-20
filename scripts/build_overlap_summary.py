#!/usr/bin/env python3
"""Build data/fig7_overlap.csv (panel I) from the dense sliding grid.

Reads data/japan_overlap_grid.csv, produced by scripts/compute_overlap_grid.py, which
holds rcMSE recomputed from the raw RR series on a 0.75-min step. Each overlap level is
an exact subset of that grid, taken by striding: step 15 min = every 20th window (0 %
overlap), 7.5 min = every 10th (50 %), 3.75 min = every 5th (75 %), 1.5 min = every 2nd
(90 %). These are real overlapping windows, not a resampling of the consecutive series.

Run:  python3 scripts/build_overlap_summary.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
BASE_STEP_MIN = 0.75
LEVELS = ((0.0, 15.0), (0.5, 7.5), (0.75, 3.75), (0.9, 1.5))


def auc(c, p):
    c, p = np.asarray(c, float), np.asarray(p, float)
    u = stats.mannwhitneyu(c, p, alternative="two-sided")
    return u.statistic / (len(c) * len(p))


def main():
    g = pd.read_csv(os.path.join(D, "japan_overlap_grid.csv"))
    g = g.sort_values(["Subject", "start_s"])
    g["k"] = (g.start_s / (BASE_STEP_MIN * 60)).round().astype(int)
    print(f"  grid: {len(g)} windows, {g.Subject.nunique()} subjects")

    rows = []
    for frac, step_min in LEVELS:
        stride = int(round(step_min / BASE_STEP_MIN))
        s = g[g.k % stride == 0]
        per = s.groupby(["Subject", "Group"]).cx
        mins = per.min().reset_index()
        p10 = per.quantile(.10).reset_index()
        med = per.median().reset_index()
        nwin = int(round(s.groupby("Subject").size().median()))
        rows.append((frac, step_min, nwin, round(float(mins.cx.median()), 3),
                     round(auc(mins[mins.Group == "Control"].cx, mins[mins.Group == "PD"].cx), 3),
                     round(auc(p10[p10.Group == "Control"].cx, p10[p10.Group == "PD"].cx), 3),
                     round(auc(med[med.Group == "Control"].cx, med[med.Group == "PD"].cx), 3)))
        print(f"    overlap {frac:4.0%}  step {step_min:5.2f} min  {nwin:4d} windows/subject")

    out = pd.DataFrame(rows, columns=["overlap", "step_min", "windows", "median_min",
                                      "AUC_min", "AUC_p10", "AUC_med"])
    out.to_csv(os.path.join(D, "fig7_overlap.csv"), index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
