#!/usr/bin/env python3
"""Compute the complexity index on a dense sliding grid of 15-min Nagoya windows.

Panel I of Figure 3 asks what window overlap does to a minimum-based statistic. That
cannot be emulated by subsampling the non-overlapping series: overlapping windows are
genuinely different windows, with their own entropy values. This script therefore
recomputes rcMSE from the raw RR series on a 0.75-min step grid, which contains the
0 %, 50 %, 75 % and 90 % overlap grids as exact subsets (step 15, 7.5, 3.75 and 1.5 min
are all multiples of 0.75), so one pass serves all four overlap levels.

Restricted to the 47-subject analysis set in data/japan_window_features.csv.
Entropy settings match compute_nagoya_windows.py exactly: refined-composite MSE,
tau = 1..5, m = 2, r = 0.2 x SD of the window, nAUC = trapz/n.

Output: data/japan_overlap_grid.csv   (Subject, Group, start_s, clock, n_beats, cx)

Run:  python3 scripts/compute_overlap_grid.py          (~50 min on 4 cores)
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
NAGOYA = os.environ.get(
    "NAGOYA_ROOT",
    os.path.abspath(os.path.join(BASE, "..", "..", "Nagoya", "public_release")))
RRI_DIR = os.path.join(NAGOYA, "data", "processed_rri")
META = os.path.join(NAGOYA, "data", "metadata", "metadata.csv")
OUT = os.path.join(DATA, "japan_overlap_grid.csv")
PART = os.path.join(DATA, "_overlap_parts")

WIN_S = 900.0        # 15-min window
BASE_STEP_S = 45.0   # 0.75 min: common divisor of the 15/7.5/3.75/1.5-min steps
RRI_MIN, RRI_MAX = 300.0, 2000.0
R_FACTOR, M = 0.2, 2
TAUMAX = 5
MIN_BEATS = 600


def _AB(s, m, r):
    """Match counts for SampEn. Equivalent to the triu formulation in
    compute_nagoya_windows.py (verified identical) but ~8x faster: it counts over
    the full matrix and removes the diagonal, avoiding triu_indices and the
    fancy-indexed copy, and reuses one buffer for both templates. m is fixed at 2."""
    n = len(s) - m
    if n < 3:
        return 0.0, 0.0
    s = np.asarray(s, np.float32)
    a, b, c = s[:n], s[1:n + 1], s[m:n + m]
    d = np.maximum(np.abs(a[:, None] - a[None, :]), np.abs(b[:, None] - b[None, :]))
    B = (d <= r).sum()
    np.maximum(d, np.abs(c[:, None] - c[None, :]), out=d)
    A = (d <= r).sum()
    return float((A - n) / 2.0), float((B - n) / 2.0)


def rcmse(sig, r, taumax=TAUMAX):
    N = len(sig)
    out = []
    for t in range(1, taumax + 1):
        A = B = 0.0
        for k in range(t):
            L = (N - k) // t
            if L <= M:
                continue
            c = sig[k:k + L * t].reshape(L, t).mean(1)
            a, b = _AB(c, M, r)
            A += a; B += b
        out.append(-np.log(A / B) if (A > 0 and B > 0) else np.nan)
    return np.array(out)


def nauc(curve):
    c = np.asarray(curve, float)
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c)) if np.isfinite(c).all() else np.nan


def one_subject(args):
    f, group, start_time = args
    sub = os.path.basename(f).replace("_RRi.txt", "")
    part = os.path.join(PART, sub + ".csv")
    if os.path.exists(part):
        return part
    d = np.loadtxt(f)
    x = (d[:, 1] if d.ndim == 2 else d) * 1000.0
    t = d[:, 0] if d.ndim == 2 else np.cumsum(x) / 1000.0
    keep = (x > RRI_MIN) & (x < RRI_MAX)
    x, t = x[keep], t[keep]
    if len(x) < 5000:
        return []
    hh, mm, ss = [int(v) for v in str(start_time).split(":")]
    clk0 = hh + mm / 60 + ss / 3600
    rows, a = [], t[0]
    while a + WIN_S <= t[-1]:
        s = x[(t >= a) & (t < a + WIN_S)]
        if len(s) >= MIN_BEATS:
            v = nauc(rcmse(s, R_FACTOR * s.std(ddof=1)))
            if np.isfinite(v):
                rows.append(dict(Subject=sub, Group=group, start_s=a - t[0],
                                 clock=(clk0 + (a - t[0]) / 3600) % 24,
                                 n_beats=len(s), cx=v))
        a += BASE_STEP_S
    pd.DataFrame(rows).to_csv(part, index=False)
    return part


def main():
    keep = set(pd.read_csv(os.path.join(DATA, "japan_window_features.csv")).Subject)
    meta = pd.read_csv(META)
    grp = meta.set_index("Subject_ID")["Group"].to_dict()
    start = meta.set_index("Subject_ID")["Start_Time"].to_dict()

    jobs = []
    for f in sorted(glob.glob(os.path.join(RRI_DIR, "*_RRi.txt"))):
        sub = os.path.basename(f).replace("_RRi.txt", "")
        if sub not in keep or sub not in grp:
            continue
        jobs.append((f, "PD" if str(grp[sub]).lower() == "pd" else "Control", start[sub]))
    print(f"  {len(jobs)} subjects, step {BASE_STEP_S/60:.2f} min", flush=True)

    os.makedirs(PART, exist_ok=True)
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(PART, os.path.basename(j[0])
                                               .replace("_RRi.txt", "") + ".csv"))]
    print(f"  {len(jobs)-len(todo)} already done, {len(todo)} to do", flush=True)
    t0 = time.time()
    if todo:
        with Pool(min(4, os.cpu_count() or 1)) as pool:
            for i, _ in enumerate(pool.imap_unordered(one_subject, todo), 1):
                print(f"  [{i}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
    parts = sorted(glob.glob(os.path.join(PART, "*.csv")))
    if len(parts) == len(jobs):
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df.to_csv(OUT, index=False)
        print(f"\n  wrote {OUT}  ({len(df)} rows, {df.Subject.nunique()} subjects)", flush=True)
    else:
        print(f"\n  {len(parts)}/{len(jobs)} subjects done; re-run to continue", flush=True)


if __name__ == "__main__":
    main()
