#!/usr/bin/env python3
"""
Compute HRV Features for Nagoya 16–20h Window
==============================================
Version: 1.1  (2026-03)
Authors: NeuroEng@Usach

Reads the raw Nagoya RRi files (one per subject), filters each to the 16:00–20:00
clock-time window (the best PD-discrimination window, see Figure 3), and computes
HRV features using the same pipeline as the other centers.

Features computed
-----------------
  Time-domain (via neurokit2 hrv_time):
    HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, ...
  Nonlinear DFA:
    DFA_alpha1  — short-range (n=4–16), computed directly via numpy (fast)
    DFA_alpha2  — long-range (n=16–64), computed directly via numpy (fast)

Output
------
  data/japan_afternoon_features.csv
  Key columns: Subject, Group, Age, Gender, n_beats, HR,
               HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50,
               DFA_alpha1, DFA_alpha2

This file is the single authoritative source of Nagoya HRV features for the
16–20h window and should be used by generate_figure5.py and generate_appendix.py
instead of mixing japan_evolution.csv (partial) and japan_recalc_metrics.csv (24h).

Note: japan_recalc_metrics.csv (full 24h) is still used by traditional_hrv_metrics.py
for the Figure 7 correlation analysis, where whole-day HRV is appropriate.

Requires raw RRi data (not included in public_release — available on request):
  NAGOYA_RRI_DIR  : folder with *_RRi.txt files (space-sep: rel_time_s  rri_s)
  NAGOYA_META_CSV : metadata with Subject_ID, Group, Gender, Age, Start_Time
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import neurokit2 as nk

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAGOYA_RRI_DIR  = "/Users/leo/HRV-Complexity/Nagoya/public_release/data/processed_rri"
NAGOYA_META_CSV = "/Users/leo/HRV-Complexity/Nagoya/public_release/data/metadata/metadata.csv"
OUT_CSV         = os.path.join(BASE, 'data', 'japan_afternoon_features.csv')

# ── Settings ──────────────────────────────────────────────────────────────────

WIN_START_H = 16   # 16:00 clock time
WIN_END_H   = 20   # 20:00 clock time
RRI_MIN_S   = 0.3  # physiological bounds (seconds)
RRI_MAX_S   = 2.0

# DFA scale ranges (beats)
DFA_ALPHA1_SCALES = (4, 16)    # short-range
DFA_ALPHA2_SCALES = (16, 64)   # long-range


# ── DFA implementation ─────────────────────────────────────────────────────────

def _dfa_fluctuation(x, n):
    """
    Compute DFA root-mean-square fluctuation F(n) for window size n.
    Segments are non-overlapping; each is locally detrended (linear fit).
    """
    N = len(x)
    # Number of complete windows
    n_segs = N // n
    if n_segs < 2:
        return np.nan
    x = x[:n_segs * n]           # trim to fit complete windows
    segs = x.reshape(n_segs, n)  # (n_segs, n)
    t = np.arange(n)
    # Fit linear trend per segment and compute residual variance
    # Using least-squares via matrix solve (fast)
    A = np.column_stack([t, np.ones(n)])
    coefs, _, _, _ = np.linalg.lstsq(A, segs.T, rcond=None)  # (2, n_segs)
    trend = (A @ coefs).T   # (n_segs, n)
    residuals = segs - trend
    F = np.sqrt(np.mean(residuals ** 2))
    return F


def dfa_alpha(rri_ms, scale_min, scale_max, n_scales=10):
    """
    Compute DFA scaling exponent alpha for [scale_min, scale_max] beat-range.

    Parameters
    ----------
    rri_ms    : array of RRi in milliseconds
    scale_min : minimum window size (beats)
    scale_max : maximum window size (beats)
    n_scales  : number of log-spaced window sizes to evaluate

    Returns
    -------
    alpha : float scaling exponent (NaN if insufficient data)
    """
    # Integrate (mean-detrend then cumsum) — standard DFA pre-processing
    y = np.cumsum(rri_ms - np.mean(rri_ms))

    scales = np.unique(np.round(
        np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
    ).astype(int))
    scales = scales[scales >= 4]

    log_n, log_F = [], []
    for n in scales:
        F = _dfa_fluctuation(y, n)
        if np.isfinite(F) and F > 0:
            log_n.append(np.log10(n))
            log_F.append(np.log10(F))

    if len(log_n) < 3:
        return np.nan

    alpha = np.polyfit(log_n, log_F, 1)[0]
    return float(alpha)


# ── Processing ────────────────────────────────────────────────────────────────

def extract_window_rri(fpath, start_time_str):
    """
    Load RRi file, convert relative timestamps to clock time,
    return RRi values (ms) within WIN_START_H–WIN_END_H.
    """
    try:
        st  = datetime.strptime(start_time_str.strip(), '%H:%M:%S')
        off = st.hour * 3600 + st.minute * 60 + st.second
    except Exception:
        off = 9 * 3600  # fallback 09:00

    df    = pd.read_csv(fpath, sep=r'\s+', header=None, names=['rel_time', 'rri'])
    abs_h = ((df['rel_time'].values + off) % (24 * 3600)) / 3600.0
    mask  = (abs_h >= WIN_START_H) & (abs_h < WIN_END_H)
    rri_s = df.loc[mask, 'rri'].values

    # Physiological filter
    rri_s = rri_s[(rri_s > RRI_MIN_S) & (rri_s < RRI_MAX_S)]
    return rri_s * 1000.0  # → milliseconds for NK2


def compute_hrv(rri_ms):
    """
    Compute HRV features from RRi in milliseconds.

    Strategy:
      - Time-domain: nk.hrv_time() — fast (<1 s per subject)
      - DFA alpha1/2: direct numpy implementation — fast (<0.1 s per subject)
        (NK2's nk.hrv_nonlinear() / nk.hrv() is extremely slow for 4h windows
        because it computes many nonlinear metrics beyond DFA; we skip those.)

    Returns a flat dict of HRV metrics, or None if too few beats.
    """
    if len(rri_ms) < 300:
        return None
    try:
        peaks = nk.intervals_to_peaks(rri_ms)
        td    = nk.hrv_time(peaks, sampling_rate=1000, show=False)
        result = td.iloc[0].to_dict()
    except Exception as e:
        print(f"    NK2 time-domain error: {e}")
        return None

    # DFA via numpy (fast)
    result['DFA_alpha1'] = dfa_alpha(rri_ms, *DFA_ALPHA1_SCALES)
    result['DFA_alpha2'] = dfa_alpha(rri_ms, *DFA_ALPHA2_SCALES)

    return result


def main():
    meta = pd.read_csv(NAGOYA_META_CSV)
    meta_d = meta.set_index('Subject_ID').to_dict('index')

    files = sorted(glob.glob(os.path.join(NAGOYA_RRI_DIR, '*_RRi.txt')))
    print(f"Found {len(files)} RRi files")
    print(f"Extracting {WIN_START_H:02d}:00-{WIN_END_H:02d}:00 window...\n")

    rows = []
    failed = []

    for fpath in files:
        sid = os.path.basename(fpath).replace('_RRi.txt', '')
        if sid not in meta_d:
            print(f"  SKIP {sid} -- not in metadata")
            continue

        info       = meta_d[sid]
        start_time = str(info.get('Start_Time', '09:00:00'))
        group      = str(info.get('Group', '')).strip()
        age        = info.get('Age', np.nan)
        gender     = info.get('Gender', np.nan)

        rri_ms  = extract_window_rri(fpath, start_time)
        n_beats = len(rri_ms)

        if n_beats < 300:
            print(f"  SKIP {sid} -- only {n_beats} beats in window")
            failed.append(sid)
            continue

        hrv = compute_hrv(rri_ms)
        if hrv is None:
            print(f"  SKIP {sid} -- HRV computation failed")
            failed.append(sid)
            continue

        row = {'Subject': sid, 'Group': group, 'Age': age, 'Gender': gender,
               'n_beats': n_beats}
        row.update(hrv)
        rows.append(row)
        print(f"  OK   {sid}  ({group})  n_beats={n_beats}"
              f"  DFA_alpha1={hrv.get('DFA_alpha1', float('nan')):.3f}")

    df = pd.DataFrame(rows)

    # Add derived HR column for convenience
    if 'HRV_MeanNN' in df.columns:
        df.insert(4, 'HR', 60000.0 / df['HRV_MeanNN'])

    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*55}")
    print(f"Saved {len(df)} subjects -> {OUT_CSV}")
    if failed:
        print(f"Skipped ({len(failed)}): {failed}")

    # Quick sanity check on key metrics
    key   = ['Subject', 'Group', 'HR', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50',
             'DFA_alpha1', 'DFA_alpha2']
    avail = [c for c in key if c in df.columns]
    print(f"\nKey metrics preview:")
    print(df[avail].groupby('Group').mean(numeric_only=True).round(3).to_string())


if __name__ == '__main__':
    main()
