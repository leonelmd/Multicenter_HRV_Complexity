#!/usr/bin/env python3
"""
Spectral characterisation of the analysed interval series (Supplementary Fig. S2).

Recomputes data/spectral_analysis.csv and data/spectral_psd_curves.csv from the
series that enter the primary analysis:
  CETRAM  cleaned Brammer detections (trace-review exclusions applied), 300-2000 ms
  Cruces  standardized pulse-interval series (Cruces/public_release/.../RRi)
  Nagoya  18:00-22:00 window, 300-2000 ms (as compute_japan_window_features.py)
Loaders are those of compute_length_ladder.py, which reproduce chile_mse.csv and
spain_mse.csv exactly.

Per subject:
  beta_beat  -slope of log10 Welch PSD (beat index, nperseg=1024) over 0.01-0.4 cycles/beat
  beta_time  same, after linear interpolation at 4 Hz, over 0.01-0.4 Hz
  beta_var   -slope of log10 SD of non-overlapping block means vs log10 block size, 1..10
  real_s1..s10     rcMSE (m=2, r=0.2 SD fixed) of the series
  matched_s1..s10  rcMSE of 1/f^beta_beat Gaussian noise of the same length and SD
                   (Fourier synthesis, random phases, seed fixed per subject)
Run: python scripts/compute_spectral_analysis.py
"""
import os, sys, zlib
import numpy as np, pandas as pd
from scipy.signal import welch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_length_ladder as L
import compute_japan_window_features as JW
from rcmse_fast import rcmse

NS = 10
FGRID = np.logspace(np.log10(0.004), np.log10(0.5), 120)


def beta_beat(x):
    f, p = welch(x - x.mean(), fs=1.0, nperseg=min(1024, len(x)))
    k = (f >= 0.01) & (f <= 0.4)
    return -np.polyfit(np.log10(f[k]), np.log10(p[k]), 1)[0], f, p


def beta_time(x):
    t = np.cumsum(x) / 1000.0
    tt = np.arange(t[0], t[-1], 0.25)
    y = np.interp(tt, t, x)
    f, p = welch(y - y.mean(), fs=4.0, nperseg=min(1024, len(y)))
    k = (f >= 0.01) & (f <= 0.4)
    return -np.polyfit(np.log10(f[k]), np.log10(p[k]), 1)[0]


def beta_var(x):
    tl = np.arange(1, 11)
    sd = [x[:len(x) // t * t].reshape(-1, t).mean(1).std(ddof=1) for t in tl]
    return -np.polyfit(np.log10(tl), np.log10(sd), 1)[0]


def powerlaw_noise(n, beta, sd, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n)
    amp = np.zeros_like(f); amp[1:] = f[1:] ** (-beta / 2.0)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    y = np.fft.irfft(amp * np.exp(1j * ph), n)
    y = (y - y.mean()) / y.std(ddof=1)
    return y * sd


def series():
    keep = set(pd.read_csv(os.path.join(L.DATA, "chile_mse.csv")).Subject)
    for coh, s, g, x in L.subjects():
        if coh == "CETRAM" and s in keep:
            yield coh, s, g, x
        elif coh == "Cruces":
            yield coh, s, g, x
    meta = pd.read_csv(L.NAGOYA_META).set_index("Subject_ID")
    keepj = set(pd.read_csv(os.path.join(L.DATA, "japan_window_mse.csv")).Subject)
    for s in sorted(keepj):
        f = os.path.join(L.NAGOYA_RRI, f"{s}_RRi.txt")
        x = JW.extract_window_rri(f, str(meta.loc[s, "Start_Time"]))
        g = "PD" if str(meta.loc[s, "Group"]).strip().lower() == "pd" else "Control"
        yield "Nagoya", s, g, np.asarray(x, float)


def main():
    rows, psd = [], {"CETRAM": [], "Cruces": [], "Nagoya": []}
    for coh, s, g, x in series():
        bb, f, p = beta_beat(x)
        r = dict(Cohort=coh, Subject=s, Group=g, n_beats=len(x), meanRR=x.mean(),
                 HR=60000.0 / x.mean(), SD=x.std(ddof=1), beta_beat=bb,
                 beta_time=beta_time(x), beta_var=beta_var(x))
        re = rcmse(x, NS)
        me = rcmse(powerlaw_noise(len(x), bb, x.std(ddof=1), zlib.crc32(s.encode())), NS)
        for i in range(NS):
            r[f"real_s{i+1}"] = re[i]; r[f"matched_s{i+1}"] = me[i]
        rows.append(r)
        psd[coh].append(np.interp(FGRID, f[1:], p[1:] / np.trapezoid(p, f)))
        print(f"{coh:7s} {s:8s} N={len(x):6d} beta={bb:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(L.DATA, "spectral_analysis.csv"), index=False)
    P = pd.DataFrame({"freq_cycles_per_beat": FGRID})
    for c, v in psd.items():
        P[c] = np.exp(np.mean(np.log(np.array(v)), 0))
    for b, nm in ((0, "white_b0"), (1, "pink_b1"), (2, "brown_b2")):
        P[nm] = FGRID ** (-b) / np.trapezoid(FGRID ** (-b), FGRID)
    P.to_csv(os.path.join(L.DATA, "spectral_psd_curves.csv"), index=False)
    print("wrote spectral_analysis.csv, spectral_psd_curves.csv")


if __name__ == "__main__":
    main()
