#!/usr/bin/env python3
"""Build the data/fig7_*.csv inputs for panels D, E and F of fig7_recording_length.py.

All six fig7_*.csv files previously had no generator in the release: they were produced
by a scratch script that was never committed, on a 45-subject Nagoya set, and could not
be reproduced. This script rebuilds five of them from the committed intermediates, on the 47-subject
analysis set defined by data/japan_window_features.csv:

    fig7_window_ladder.csv      panel D
    fig7_dist_summaries.csv     panel E
    fig7_hourly_auc.csv         panel B
    fig7_window_scan.csv        panel C
    fig7_cetram_headtohead.csv  panel F

The sixth, fig7_overlap.csv (panel I), cannot be built from these intermediates, because
overlapping windows are genuinely different windows and cannot be obtained by
subsampling the non-overlapping series. It is produced by scripts/compute_overlap_grid.py
followed by scripts/build_overlap_summary.py, which recompute rcMSE from the raw RR
series on a dense sliding grid.

Sources, all of which carry all 50 Nagoya subjects in consistent units:
    data/japan_15min_windows.csv   consecutive 15-min windows (cx, SDNN, ms)
    data/japan_evolution.csv       4-h windows at 1-h steps (nAUC_1_20, SDNN, seconds)
    data/sdnn_min_all_centers.csv  100/200/300-beat window statistics, all cohorts
    data/chile_mse.csv             CETRAM rcMSE curves

Units note: japan_evolution.csv stores SDNN in seconds and japan_15min_windows.csv in
milliseconds. Both are used only for within-file group comparisons, and AUC is invariant
to a positive scale factor, so no conversion is applied; do not pool the two columns.

Run:  python3 scripts/compute_fig7_inputs.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")

# Mean RR across the Nagoya cohort, used only to place the beat-count windows on the
# minutes axis of Fig. 3D.
MEAN_RR_S = 60.0 / 70.6


def auc_p(control, pd_):
    c = np.asarray(control, float); p = np.asarray(pd_, float)
    c, p = c[np.isfinite(c)], p[np.isfinite(p)]
    if len(c) < 2 or len(p) < 2:
        return np.nan, np.nan
    u = stats.mannwhitneyu(c, p, alternative="two-sided")
    return u.statistic / (len(c) * len(p)), u.pvalue


def grouped(df, col, how):
    g = df.groupby(["Subject", "Group"])[col].agg(how).reset_index()
    return auc_p(g[g.Group == "Control"][col], g[g.Group == "PD"][col])


def main():
    keep = set(pd.read_csv(os.path.join(D, "japan_window_features.csv")).Subject)
    print(f"  analysis set: {len(keep)} Nagoya subjects")

    sdm = pd.read_csv(os.path.join(D, "sdnn_min_all_centers.csv"))
    jp = sdm[(sdm.Center == "Japan") & sdm.Subject.isin(keep)]
    w15 = pd.read_csv(os.path.join(D, "japan_15min_windows.csv"))
    w15 = w15[w15.Subject.isin(keep)]
    ev = pd.read_csv(os.path.join(D, "japan_evolution.csv"))
    ev = ev[ev.Subject.isin(keep)]
    for nm, d in (("sdnn_min", jp), ("15-min windows", w15), ("4-h windows", ev)):
        print(f"    {nm:<16} {d.Subject.nunique()} subjects")

    # --- D: AUC of the min-across-day statistic vs window length -------------
    lad = []
    for nb in (100, 200):
        a, _ = auc_p(jp[jp.Group == "Control"][f"SDNN{nb}_min"],
                     jp[jp.Group == "PD"][f"SDNN{nb}_min"])
        lad.append(("SDNN", nb * MEAN_RR_S / 60.0, a))
    lad.append(("SDNN", 15.0, grouped(w15, "SDNN", "min")[0]))
    lad.append(("SDNN", 240.0, grouped(ev, "SDNN", "min")[0]))
    lad.append(("complexity", 15.0, grouped(w15, "cx", "min")[0]))
    lad.append(("complexity", 240.0, grouped(ev, "nAUC_1_20", "min")[0]))
    pd.DataFrame(lad, columns=["metric", "win_min", "AUC"]).to_csv(
        os.path.join(D, "fig7_window_ladder.csv"), index=False)

    # --- E: distributional summaries, Suzuki's four statistics ---------------
    rows = []
    for stat, how, col in (("min", "min", "SDNN100_min"),
                           ("p10", lambda s: s.quantile(.10), "SDNN100_p10"),
                           ("p25", lambda s: s.quantile(.25), "SDNN100_p25"),
                           ("median", "median", "SDNN100_p50")):
        a_sd, _ = auc_p(jp[jp.Group == "Control"][col], jp[jp.Group == "PD"][col])
        a_cx, _ = grouped(w15, "cx", how)
        rows.append((stat, a_cx, a_sd))
    pd.DataFrame(rows, columns=["stat", "complexity", "SDNN100"]).to_csv(
        os.path.join(D, "fig7_dist_summaries.csv"), index=False)

    # --- B: discrimination by clock hour -------------------------------------
    # Coverage varies by clock hour because recordings began in the afternoon or
    # evening; n is written out per hour so this is stated rather than hidden.
    wh = w15.assign(hour=w15.clock.astype(float).mod(24).astype(int))
    hr = []
    for h, g in wh.groupby("hour"):
        sj = g.groupby(["Subject", "Group"])[["cx", "SDNN"]].mean().reset_index()
        a_cx, _ = auc_p(sj[sj.Group == "Control"].cx, sj[sj.Group == "PD"].cx)
        a_sd, _ = auc_p(sj[sj.Group == "Control"].SDNN, sj[sj.Group == "PD"].SDNN)
        hr.append((int(h), sj.Subject.nunique(), a_cx, a_sd))
    pd.DataFrame(hr, columns=["hour", "n", "AUC_cx", "AUC_sdnn"]).to_csv(
        os.path.join(D, "fig7_hourly_auc.csv"), index=False)

    # --- C: every candidate 4-h window ---------------------------------------
    sc = []
    for w, g in ev.groupby("Window_start_h"):
        a, p = auc_p(g[g.Group == "Control"].nAUC_1_20, g[g.Group == "PD"].nAUC_1_20)
        sc.append((int(w), g.Subject.nunique(), a, p))
    pd.DataFrame(sc, columns=["window", "n", "AUC", "p"]).to_csv(
        os.path.join(D, "fig7_window_scan.csv"), index=False)

    # --- F: CETRAM head-to-head ----------------------------------------------
    ce = sdm[sdm.Center == "Chile"]
    mse = pd.read_csv(os.path.join(D, "chile_mse.csv"))
    mse = mse[mse.Scales <= 5].sort_values("Scales")
    cx = (mse.groupby(["Subject", "Group"]).MSE
             .apply(lambda c: float(np.trapezoid(c.values, np.arange(1, len(c) + 1)) / len(c)))
             .reset_index().rename(columns={"MSE": "cx"}))
    met = pd.read_csv(os.path.join(D, "chile_metrics.csv"))
    hi = set(met[met.HRV_SDNN > 100].Subject)
    rows = []
    for lab, drop in (("all", set()), ("excl. SDNN>100 ms", hi)):
        c = cx[~cx.Subject.isin(drop)]
        s = ce[~ce.Subject.isin(drop)]
        sw = met[~met.Subject.isin(drop)]
        a_min, _ = auc_p(s[s.Group == "Control"].SDNN100_min, s[s.Group == "PD"].SDNN100_min)
        a_wh, _ = auc_p(sw[sw.Group == "Control"].HRV_SDNN, sw[sw.Group == "PD"].HRV_SDNN)
        a_cx, _ = auc_p(c[c.Group == "Control"].cx, c[c.Group == "PD"].cx)
        rows.append((lab, c.Subject.nunique(), a_min, a_wh, a_cx))
    pd.DataFrame(rows, columns=["subset", "n", "SDNN100_min", "SDNN_whole",
                                "complexity"]).to_csv(
        os.path.join(D, "fig7_cetram_headtohead.csv"), index=False)

    print("\n  wrote 5 files:")
    for f in ("fig7_window_ladder", "fig7_dist_summaries", "fig7_hourly_auc",
              "fig7_window_scan", "fig7_cetram_headtohead"):
        print(f"    data/{f}.csv")


if __name__ == "__main__":
    main()
