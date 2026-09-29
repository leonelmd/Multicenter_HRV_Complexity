#!/usr/bin/env python3
"""
Length-sensitivity ladder for Appendix 3.

Truncates every subject's RR series to a ladder of beat counts and recomputes
rcMSE nAUC(1-5) at each, plus split-half halves and curve roughness. Nagoya spans
the whole ladder and therefore acts as an internal reference for what the shorter
cohorts could achieve with their recordings.

The tolerance r is recomputed from each truncated segment — deliberately, because
the question is what an investigator with a recording of that length would obtain,
and they would have no access to the longer series.

Requires the per-centre raw/cleaned RR data, which is not in this repository;
the output it produces (data/length_ladder.csv) is shipped so Appendix 3 can be
regenerated without it.

Resumable: re-run to continue after an interruption. --force starts over.

Run:  python scripts/compute_length_ladder.py [--force]
"""
from __future__ import annotations

import os
import sys
import glob
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))

CETRAM_DET = os.path.join(ROOT, "CETRAM", "public_release", "results", "cleaned_detections")
CRUCES_RRI = os.path.join(ROOT, "Cruces", "public_release", "data", "processed", "RRi")
NAGOYA_RRI = os.path.join(ROOT, "Nagoya", "public_release", "data", "processed_rri")
NAGOYA_META = os.path.join(ROOT, "Nagoya", "public_release", "data", "metadata", "metadata.csv")
SPAIN_METRICS = os.path.join(DATA, "spain_metrics.csv")

LADDER = [250, 350, 500, 700, 1000, 1400, 2000, 3000, 4000]
NSCALES, M, R_FACTOR = 5, 2, 0.2
RRI_MIN, RRI_MAX = 300.0, 2000.0
OUT = os.path.join(DATA, "length_ladder.csv")
FORCE = "--force" in sys.argv


def _AB(s, m, r):
    n = len(s) - m
    if n < 3:
        return 0.0, 0.0
    X = s[np.arange(n)[:, None] + np.arange(m)[None, :]].astype(np.float32)
    d = np.abs(X[:, None, :] - X[None, :, :]).max(2)
    xm = s[np.arange(n) + m].astype(np.float32)
    d1 = np.maximum(d, np.abs(xm[:, None] - xm[None, :]))
    iu = np.triu_indices(n, 1)
    return float((d1[iu] <= r).sum()), float((d[iu] <= r).sum())


def curve(sig, nscales=NSCALES):
    """rcMSE with fixed r (Wu 2014 shifts), r from this segment."""
    N = len(sig)
    r = R_FACTOR * sig.std(ddof=1)
    if not np.isfinite(r) or r <= 0:
        return np.full(nscales, np.nan)
    out = []
    for t in range(1, nscales + 1):
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


def nauc(c):
    c = np.asarray(c, float)
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c)) if np.isfinite(c).all() else np.nan


def roughness(c):
    c = np.asarray(c, float)
    if not np.isfinite(c).all() or len(c) < 3:
        return np.nan
    amp = np.nanmax(c) - np.nanmin(c)
    return float(np.mean(np.abs(np.diff(c, n=2))) / max(amp, 1e-9))


def clean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size and np.nanmax(x) < 20:          # seconds -> ms
        x = x * 1000.0
    return x[(x > RRI_MIN) & (x < RRI_MAX)]


def subjects():
    for g in ("Control", "PD"):
        for f in sorted(glob.glob(os.path.join(CETRAM_DET, g, "*_cleaned.csv"))):
            pk = pd.read_csv(f)["sample"].dropna().astype(np.int64).to_numpy()
            yield "CETRAM", os.path.basename(f).replace("_cleaned.csv", ""), g, clean(np.diff(pk))
    if os.path.exists(SPAIN_METRICS):
        grp = pd.read_csv(SPAIN_METRICS).set_index("Subject")["Group"].to_dict()
        for f in sorted(glob.glob(os.path.join(CRUCES_RRI, "*.csv"))):
            s = os.path.basename(f)[:-4]
            if grp.get(s) in ("Control", "PD"):
                yield "Cruces", s, grp[s], clean(pd.read_csv(f, header=None).iloc[:, -1].to_numpy())
    if os.path.exists(NAGOYA_META):
        md = pd.read_csv(NAGOYA_META)
        gp = {r.Subject_ID: ("PD" if str(r.Group).strip().lower() == "pd" else "Control")
              for r in md.itertuples()}
        for f in sorted(glob.glob(os.path.join(NAGOYA_RRI, "*_RRi.txt"))):
            s = os.path.basename(f).replace("_RRi.txt", "")
            if s in gp:
                d = np.loadtxt(f)
                yield "Nagoya", s, gp[s], clean(d[:, 1] if d.ndim == 2 else d)


def main():
    rows, done = [], set()
    if os.path.exists(OUT) and not FORCE:
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev.Cohort, prev.Subject))

    t0 = time.time()
    for cohort, sub, group, x in subjects():
        if (cohort, sub) in done:
            continue
        for N in LADDER:
            if len(x) < N:
                continue
            seg = x[:N]
            h = N // 2
            cv = curve(seg)
            rows.append(dict(Cohort=cohort, Subject=sub, Group=group, N=N,
                             N_full=len(x), nAUC=nauc(cv),
                             h1=nauc(curve(seg[:h])), h2=nauc(curve(seg[h:2 * h])),
                             roughness=roughness(cv),
                             **{f"MSE_s{i+1}": cv[i] for i in range(NSCALES)}))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  {cohort:7s} {sub:8s} N_full={len(x):6d}  {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > 145:
            print("PAUSE — re-run to continue")
            return
    d = pd.DataFrame(rows)
    print(f"\nwrote {OUT}  ({len(d)} rows)")
    print(d.groupby(["Cohort", "N"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
