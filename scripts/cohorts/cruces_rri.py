#!/usr/bin/env python3
"""Cruces: raw finger PPG -> RR intervals.

For each recording in $HRV_DATA_ROOT/cruces/raw/ppg_signals.mat (500 Hz):
  1. NeuroKit2 ppg_clean; pulses by the Elgendi detector; each beat timed at the maximum
     of the first derivative on the upstroke, with parabolic sub-sample refinement.
  2. Ectopic and misdetection correction (Lipponen & Tarvainen 2019), as in CETRAM.
  3. Physiological guard 300-2000 ms, offending intervals replaced by linear interpolation.
  4. Irregular-rhythm share (variance at 0.35-0.5 cycles/beat) reported, not used to exclude.
Writes $HRV_DATA_ROOT/cruces/rri/<ID>.csv (ms) and cruces/rri_cleaning_summary.csv.

    python scripts/cohorts/cruces_rri.py
"""
import os
import sys

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CRUCES_PPG, CRUCES_RRI, require  # noqa: E402

FIDUCIAL = "maxd1"
RRI_MIN_MS, RRI_MAX_MS = 300, 2000
CLEANING_LOG = []


def load_data(path):
    data = scipy.io.loadmat(path)["ppg_signals"]
    subjects = []
    for i in range(data.shape[1]):
        v = data[0, i]
        sid = v.item() if isinstance(v, np.ndarray) and v.size else str(v)
        group = "Control" if sid.startswith("E") else "PD" if sid[0] in "BC" else "Other"
        subjects.append(dict(ID=sid, Group=group, Signal=data[1, i].flatten().astype(float)))
    return sorted(subjects, key=lambda s: s["ID"])


def _parab(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(i) if den == 0 else i + 0.5 * (a - c) / den


def beat_times(cleaned, peaks, fs, fiducial=FIDUCIAL):
    """Beat times in ms at the chosen fiducial point."""
    if fiducial == "apex":
        return np.asarray(peaks, float) / fs * 1000.0
    d1 = np.gradient(cleaned)
    t = []
    for i, p in enumerate(peaks):
        lo = max(int(p - 0.4 * fs), int(peaks[i - 1]) + 1 if i else 0)
        if p - lo < 10:
            continue
        ft = lo + int(np.argmin(cleaned[lo:p + 1]))          # foot of this pulse
        if p - ft < 5:
            continue
        md = ft + int(np.argmax(d1[ft:p + 1]))                # steepest point of the upstroke
        t.append(_parab(d1, md))
    return np.asarray(t) / fs * 1000.0


def alternating_share(x):
    from scipy.signal import welch
    y = x - np.convolve(np.pad(x, 15, mode="edge"), np.ones(31) / 31, mode="valid")
    f, p = welch(y, fs=1.0, nperseg=min(128, len(y) // 2))
    return float(p[f >= 0.35].sum() / p[f > 0.02].sum())


def process_subject(subject, output_dir_rri):
    sid = subject['ID']
    signal = subject['Signal']
    sampling_rate = 500
    try:
        cleaned = nk.ppg_clean(signal, sampling_rate=sampling_rate)
        peaks = nk.ppg_findpeaks(cleaned, sampling_rate=sampling_rate, method='elgendi')['PPG_Peaks']
        if len(peaks) < 2:
            print(f"  Warning: Not enough peaks for {sid}")
            return None

        # 1. fiducial
        t = beat_times(cleaned, peaks, sampling_rate)
        # 2. Lipponen & Tarvainen, on 0.1-ms integer times to keep sub-sample resolution
        pk = np.round((t - t[0]) * 10).astype(np.int64)
        info, pk2 = nk.signal_fixpeaks(pk, sampling_rate=10000, method="Kubios", iterative=True)
        n_lt = sum(len(info.get(k, [])) for k in ("ectopic", "missed", "extra", "longshort"))
        rri = np.diff(np.asarray(pk2, float)) / 10.0
        # 3. physiological guard by interpolation
        ok = (rri >= RRI_MIN_MS) & (rri <= RRI_MAX_MS)
        n_guard = int((~ok).sum())
        if n_guard and ok.sum() >= 10:
            rri[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), rri[ok])
        # 4. reported flag
        share = alternating_share(rri)
        CLEANING_LOG.append(dict(Subject=sid, Group=subject['Group'], Fiducial=FIDUCIAL,
                                 Pulses=len(peaks), LT_Fixes=n_lt, Physio_Guarded=n_guard,
                                 Alternating_Share=round(share, 4), Irregular_Flag=share >= 0.30))

        rri_path = os.path.join(output_dir_rri, f"{sid}.csv")
        np.savetxt(rri_path, rri, delimiter=',', fmt='%.3f')
        return rri

    except Exception as e:
        print(f"  Error processing {sid}: {e}")
        return None


def main():
    os.makedirs(CRUCES_RRI, exist_ok=True)
    for subj in load_data(require(CRUCES_PPG, "the raw Cruces PPG")):
        print(f"Processing {subj['ID']} ({subj['Group']})")
        process_subject(subj, CRUCES_RRI)
    pd.DataFrame(CLEANING_LOG).to_csv(os.path.join(os.path.dirname(CRUCES_RRI), "rri_cleaning_summary.csv"), index=False)
    print(f"wrote {len(CLEANING_LOG)} RR series to {CRUCES_RRI}")


if __name__ == "__main__":
    main()
