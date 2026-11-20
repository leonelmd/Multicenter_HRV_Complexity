#!/usr/bin/env python3
"""Nagoya: 24-h RR series -> rcMSE in 4-h windows, and HR, SDNN, RMSSD across the day.

Input: $HRV_DATA_ROOT/nagoya/rri/<ID>_RRi.txt (two columns: time since start of recording
in s, RR interval in s), with the recording start time from data/japan_metadata.csv.
Intervals outside 0.3-2.0 s are dropped (in these exports such values mark recording gaps).

1. rcMSE (m = 2, r = 0.2 SD of the window, scales 1-20; scripts/rcmse_fast.py, which
   reproduces scripts/julia/entropy.jl to machine precision) in 24 clock-aligned 4-h
   windows, one starting every hour; windows with fewer than 4000 beats are skipped.
2. HR, SDNN and RMSSD in 30-min windows sliding by 1 min (at least 400 beats), and their
   median inside each 4-h window.

Writes
  data/japan_window_mse.csv      curves, 18:00-22:00 (the pre-specified primary window)
  data/japan_afternoon_mse.csv   curves, 16:00-20:00
  data/japan_morning_mse.csv     curves, 07:00-11:00
  data/japan_evolution.csv       per 4-h window: beats, nAUC(1-20), median HR, SDNN, RMSSD
Resumable (one part file per subject in data/_nagoya_parts/).

    python scripts/cohorts/nagoya_windows.py [--workers 4]
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA, NAGOYA_META, NAGOYA_RRI, require  # noqa: E402
from rcmse_fast import rcmse  # noqa: E402

SCALES, MIN_BEATS_4H, MIN_BEATS_30MIN = 20, 4000, 400
PARTS = os.path.join(DATA, "_nagoya_parts")
WINDOWS = {18: "japan_window_mse.csv", 16: "japan_afternoon_mse.csv", 7: "japan_morning_mse.csv"}
DAY = 86400


def load(fname, start):
    d = pd.read_csv(os.path.join(NAGOYA_RRI, fname), sep=r"\s+", header=None, names=["t", "rri"])
    d = d[(d.rri > 0.3) & (d.rri < 2.0)]
    h, m, s = (int(v) for v in str(start).split(":"))
    return (d.t.values + h * 3600 + m * 60 + s) % DAY, d.rri.values


def in_window(tod, t0, length):
    t1 = t0 + length
    return (tod >= t0) & (tod < t1) if t1 <= DAY else (tod >= t0) | (tod < t1 % DAY)


def one(args):
    sid, fname, group, start = args
    part = os.path.join(PARTS, f"{sid}.csv")
    if os.path.exists(part):
        return sid
    tod, rri = load(fname, start)
    # 30-min windows, 1-min step: HR (bpm), SDNN and RMSSD (s), as in the original analysis
    short = []
    for t0 in range(0, DAY, 60):
        x = rri[in_window(tod, t0, 1800)]
        if len(x) >= MIN_BEATS_30MIN:
            short.append((t0 / 3600, 60.0 / x.mean(), x.std(ddof=1), np.sqrt(np.mean(np.diff(x) ** 2))))
    short = pd.DataFrame(short, columns=["Time_h", "HR", "SDNN", "RMSSD"])
    rows = []
    for t0 in range(0, DAY, 3600):
        x = rri[in_window(tod, t0, 4 * 3600)]           # s, as in the original analysis
        if len(x) < MIN_BEATS_4H:
            continue
        ws = t0 / 3600
        inside = short[(short.Time_h >= ws) & (short.Time_h < ws + 4)]
        for sc, v in enumerate(rcmse(x, SCALES), 1):
            rows.append(dict(Subject=sid, Group=group, Window_start_h=ws, Scale=sc, MSE=v, n_beats=len(x),
                             HR=inside.HR.median(), SDNN=inside.SDNN.median(), RMSSD=inside.RMSSD.median()))
    pd.DataFrame(rows).to_csv(part + ".tmp", index=False)
    os.replace(part + ".tmp", part)
    return sid


def nauc(v):
    return np.trapezoid(v, np.arange(1, len(v) + 1)) / len(v)


def main():
    require(NAGOYA_RRI, "the Nagoya RR series")
    os.makedirs(PARTS, exist_ok=True)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    meta = pd.read_csv(NAGOYA_META)
    todo = [(r.Subject_ID, r.Filename, r.Group, r.Start_Time) for r in meta.itertuples()]
    with Pool(workers) as pool:
        for sid in pool.imap_unordered(one, todo):
            print("  ", sid, flush=True)
    c = pd.concat([pd.read_csv(os.path.join(PARTS, f"{s}.csv")) for s, *_ in todo
                   if os.path.getsize(os.path.join(PARTS, f"{s}.csv")) > 1], ignore_index=True)
    for start, name in WINDOWS.items():
        (c[c.Window_start_h == start].rename(columns={"Scale": "Scales"})[["Subject", "Group", "Scales", "MSE"]]
         .sort_values(["Subject", "Scales"]).to_csv(os.path.join(DATA, name), index=False))
    keys = ["Subject", "Group", "Window_start_h", "n_beats", "HR", "SDNN", "RMSSD"]
    evo = (c.sort_values("Scale").groupby(keys, dropna=False).MSE.apply(lambda v: nauc(v.values))
           .rename("nAUC_1_20").reset_index())
    evo[["Subject", "Group", "Window_start_h", "n_beats", "nAUC_1_20", "HR", "SDNN", "RMSSD"]] \
        .sort_values(["Subject", "Window_start_h"]).to_csv(os.path.join(DATA, "japan_evolution.csv"), index=False)
    print("wrote", ", ".join(WINDOWS.values()), "and japan_evolution.csv")


if __name__ == "__main__":
    main()
