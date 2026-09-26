#!/usr/bin/env python3
"""
Nagoya per-window complexity, for Figure 3.

Produces two tables:

  data/japan_15min_windows.csv   complexity + HRV in every consecutive 15-min
                                 window across 24 h, with clock time. Drives the
                                 circadian panels and the distributional summaries.

  data/japan_scale_profile.csv   scale-resolved MSE out to tau=60 in the 18-22 h
                                 window. Drives the multiscale panel.

Why 15 min: complexity at tau<=5 needs ~1000 beats under N/tau>=200, i.e. ~13 min
at 75 bpm. It is the shortest window that supports the index, gives ~91 windows
per subject for a distributional summary, and matches the CETRAM protocol so the
same index is computable in all three cohorts. 4-h windows give only ~6 per
subject — too few to define a minimum — and 100-beat windows cannot support MSE
at all. See Multicenter/NAGOYA_WINDOW_AND_MULTISCALE.md.

Resumable: re-run to continue after an interruption. Pass --force to start over.

Run:  python scripts/compute_nagoya_windows.py [--force]
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
NAGOYA = os.environ.get(
    "NAGOYA_ROOT",
    os.path.abspath(os.path.join(BASE, "..", "..", "Nagoya", "public_release")))
RRI_DIR = os.path.join(NAGOYA, "data", "processed_rri")
META = os.path.join(NAGOYA, "data", "metadata", "metadata.csv")

WIN_S = 900.0          # 15 min
STEP_S = 900.0         # non-overlapping; see memo re. overlap and p10 vs min
RRI_MIN, RRI_MAX = 300.0, 2000.0
R_FACTOR, M = 0.2, 2
MAX_TAU_PROFILE = 60   # 4-h window supports tau<=70 under N/tau>=200

OUT_WIN = os.path.join(DATA, "japan_15min_windows.csv")
OUT_PROF = os.path.join(DATA, "japan_scale_profile.csv")
FORCE = "--force" in sys.argv

# ── Scale-profile fidelity options ────────────────────────────────────────────
# The 15-min windows always use full refined-composite MSE (they are short enough
# that it is cheap). The tau=1..60 scale profile is the expensive part: cost is
# ~N^2 * H(60) ~ 4.7*N^2 pair comparisons per subject at N~14000 beats.
#
#   default            plain MSE (single coarse-graining, k=1) + 900-point cap.
#                      Fast; what the published Figure 3 used.
#   --rcmse            full refined-composite (all tau shifts pooled), matching
#                      the 15-min windows and entropy_toolbox.jl. ~tau x slower.
#   --cap N            points retained per coarse-grained series (0 = no cap).
#
# Recommended confirmation run on a normal machine:
#     python scripts/compute_nagoya_windows.py --force --rcmse --cap 0
# Expect roughly 30-60 min for the cohort single-threaded.
PROFILE_RCMSE = "--rcmse" in sys.argv
CAP = 900
if "--cap" in sys.argv:
    CAP = int(sys.argv[sys.argv.index("--cap") + 1])
if CAP <= 0:
    CAP = 10 ** 9


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


def rcmse(sig, r, taumax):
    """Refined-composite MSE, fixed r, Wu (2014) shifts."""
    N = len(sig)
    out = []
    for t in range(1, taumax + 1):
        A = B = 0.0
        for k in range(t):
            L = (N - k) // t
            if L <= M:
                continue
            c = sig[k:k + L * t].reshape(L, t).mean(1)
            if len(c) > CAP:
                c = c[:CAP]
            a, b = _AB(c, M, r)
            A += a; B += b
        out.append(-np.log(A / B) if (A > 0 and B > 0) else np.nan)
    return np.array(out)


def sampen(s, r):
    a, b = _AB(s, M, r)
    return -np.log(a / b) if (a > 0 and b > 0) else np.nan


def nauc(curve):
    c = np.asarray(curve, float)
    if not np.isfinite(c).all():
        return np.nan
    return float(np.trapezoid(c, np.arange(1, len(c) + 1)) / len(c))


def load_subject(f, meta_grp, meta_start):
    sub = os.path.basename(f).replace("_RRi.txt", "")
    if sub not in meta_grp:
        return None
    d = np.loadtxt(f)
    x = (d[:, 1] if d.ndim == 2 else d) * 1000.0
    t = d[:, 0] if d.ndim == 2 else np.cumsum(x) / 1000.0
    keep = (x > RRI_MIN) & (x < RRI_MAX)
    x, t = x[keep], t[keep]
    if len(x) < 5000:
        return None
    hh, mm, ss = [int(v) for v in str(meta_start[sub]).split(":")]
    clk0 = hh + mm / 60 + ss / 3600
    grp = "PD" if str(meta_grp[sub]).lower() == "pd" else "Control"
    return sub, grp, x, t, clk0


def main():
    meta = pd.read_csv(META)
    grp = meta.set_index("Subject_ID")["Group"].to_dict()
    start = meta.set_index("Subject_ID")["Start_Time"].to_dict()
    files = sorted(glob.glob(os.path.join(RRI_DIR, "*_RRi.txt")))

    win_rows, prof_rows, done_w, done_p = [], [], set(), set()
    if os.path.exists(OUT_WIN) and not FORCE:
        prev = pd.read_csv(OUT_WIN); win_rows = prev.to_dict("records"); done_w = set(prev.Subject)
    if os.path.exists(OUT_PROF) and not FORCE:
        prev = pd.read_csv(OUT_PROF); prof_rows = prev.to_dict("records"); done_p = set(prev.Subject)

    t0 = time.time()
    for f in files:
        got = load_subject(f, grp, start)
        if got is None:
            continue
        sub, group, x, t, clk0 = got
        if sub in done_w and sub in done_p:
            continue

        # ---- 15-min windows across the full record ----
        if sub not in done_w:
            a = t[0]
            while a + WIN_S <= t[-1]:
                sel = (t >= a) & (t < a + WIN_S)
                s = x[sel]; tt = t[sel]
                if len(s) >= 600:
                    r = R_FACTOR * s.std(ddof=1)
                    cur = rcmse(s, r, 5)
                    v = nauc(cur)
                    if np.isfinite(v):
                        mins = [s[(tt >= a + 60 * i) & (tt < a + 60 * (i + 1))].mean()
                                for i in range(15)]
                        mins = [q for q in mins if np.isfinite(q)]
                        win_rows.append(dict(
                            Subject=sub, Group=group,
                            clock=(clk0 + (a - t[0]) / 3600) % 24,
                            hours_in=(a - t[0]) / 3600,
                            n_beats=len(s), cx=v, HR=60000.0 / s.mean(),
                            cx_HR=v / (60000.0 / s.mean()),
                            SDNN=s.std(ddof=1),
                            RMSSD=float(np.sqrt(np.mean(np.diff(s) ** 2))),
                            drift=float(np.std(mins, ddof=1)) if len(mins) > 3 else np.nan))
                a += STEP_S
            pd.DataFrame(win_rows).to_csv(OUT_WIN, index=False)

        # ---- scale profile in the 18-22 h window (pre-specified; see audit) ----
        if sub not in done_p:
            clk = (clk0 + (t - t[0]) / 3600) % 24
            seg = x[(clk >= 18) & (clk < 22)]
            if len(seg) >= 3000:
                r = R_FACTOR * seg.std(ddof=1)
                rec = dict(Subject=sub, Group=group, N=len(seg), meanRR=seg.mean(),
                           profile_method="rcMSE" if PROFILE_RCMSE else "MSE",
                           cap=(0 if CAP >= 10 ** 9 else CAP))
                for tau in range(1, MAX_TAU_PROFILE + 1):
                    L = len(seg) // tau
                    if L < 250:                      # N/tau >= 250, conservative
                        rec[f"t{tau}"] = np.nan
                        continue
                    if PROFILE_RCMSE:
                        # refined composite: pool A and B over all tau shifts
                        A = B = 0.0
                        for k in range(tau):
                            Lk = (len(seg) - k) // tau
                            if Lk <= M:
                                continue
                            ck = seg[k:k + Lk * tau].reshape(Lk, tau).mean(1)
                            if len(ck) > CAP:
                                ck = ck[:CAP]
                            a, b = _AB(ck, M, r)
                            A += a; B += b
                        rec[f"t{tau}"] = -np.log(A / B) if (A > 0 and B > 0) else np.nan
                    else:
                        c = seg[:L * tau].reshape(L, tau).mean(1)
                        if len(c) > CAP:
                            c = c[:CAP]
                        rec[f"t{tau}"] = sampen(c, r)
                prof_rows.append(rec)
                pd.DataFrame(prof_rows).to_csv(OUT_PROF, index=False)

        print(f"  {sub} {group:8s}  {time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > 145:
            print("PAUSE — re-run to continue")
            return
    print(f"done: {len(set(r['Subject'] for r in win_rows))} subjects windowed, "
          f"{len(prof_rows)} scale profiles")


if __name__ == "__main__":
    main()
