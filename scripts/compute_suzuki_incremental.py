#!/usr/bin/env python3
"""
Nagoya: does complexity add to the published minimum-SDNN index? (Results, Comparison
with the published minimum-SDNN benchmark.)

Participants: the 47 analysed in the 18:00-22:00 window.
Complexity variant: minimum across the day of the 15-min complexity index
(japan_15min_windows.csv), the best complexity variant in Fig. 7E.
Reports Spearman rho between the two minima and a logistic model
PD ~ z(SDNN100_min) + z(complexity_min), with the likelihood-ratio test for
adding complexity to SDNN100_min alone.
Run: python scripts/compute_suzuki_incremental.py
"""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, chi2

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    keep = set(pd.read_csv(os.path.join(D, "japan_window_features.csv")).Subject)
    w = pd.read_csv(os.path.join(D, "japan_15min_windows.csv"))
    cx = w.groupby(["Subject", "Group"]).cx.min().rename("cx_min").reset_index()
    s = pd.read_csv(os.path.join(D, "sdnn_min_all_centers.csv"))[["Subject", "SDNN100_min"]]
    d = cx.merge(s, on="Subject"); d = d[d.Subject.isin(keep)].dropna()
    d["y"] = (d.Group.str.lower() == "pd").astype(int)
    for c in ("cx_min", "SDNN100_min"):
        d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
    rho, p_rho = spearmanr(d.cx_min, d.SDNN100_min)
    m1 = sm.Logit(d.y, sm.add_constant(d[["z_SDNN100_min"]])).fit(disp=0)
    m2 = sm.Logit(d.y, sm.add_constant(d[["z_SDNN100_min", "z_cx_min"]])).fit(disp=0)
    lr = 2 * (m2.llf - m1.llf)
    print(f"n={len(d)} ({d.y.sum()} PD)  rho={rho:+.2f} (p={p_rho:.1e})")
    print(f"complexity in joint model: Wald p={m2.pvalues['z_cx_min']:.2f}; "
          f"LR test p={chi2.sf(lr, 1):.2f}")


if __name__ == "__main__":
    main()
