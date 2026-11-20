#!/usr/bin/env python3
"""
Why Cruces rcMSE curves fall from tau=1 to tau=5 (Results, Cruces shape, point 5).

For every analysed Cruces recording, and for every analysed CETRAM recording truncated
to the Cruces length of 500 beats:
  * whether the real rcMSE curve falls (SampEn at tau=5 below tau=1);
  * the same for IAAFT surrogates (Schreiber & Schmitz 1996), which keep the power
    spectrum and value distribution; mean curve of 3 surrogates, fixed seed;
  * the share of spectral power in the respiratory band (0.1-0.4 cycles/beat, Welch on
    the series after removing a 31-beat moving mean) and the sharpness of the
    respiratory peak (largest bin / band power).

Series are loaded and cleaned exactly as in compute_length_ladder.py. Requires the
original data (paths.py); writes data/cruces_surrogates.csv (per-recording summary
values only, no interval series) and prints the numbers quoted in the text.

Run:  python scripts/compute_cruces_surrogates.py
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from paths import DATA
from compute_length_ladder import subjects, curve

N_CETRAM = 500        # Cruces recording length, in beats
N_SURR = 3
OUT = os.path.join(DATA, "cruces_surrogates.csv")
rng = np.random.default_rng(0)


def iaaft(x, it=50):
    amp = np.abs(np.fft.rfft(x - x.mean()))
    srt = np.sort(x)
    y = rng.permutation(x)
    for _ in range(it):
        Y = np.fft.rfft(y - y.mean())
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(x)) + x.mean()
        y = srt[np.argsort(np.argsort(y))]
    return y


def resp_band(x):
    y = x - pd.Series(x).rolling(31, center=True, min_periods=1).mean().to_numpy()
    f, p = welch(y, fs=1.0, nperseg=64)
    band = (f >= 0.1) & (f <= 0.4)
    return p[band].sum() / p[f > 0.02].sum(), p[band].max() / p[band].sum()


def main():
    analysed = {"CETRAM": set(pd.read_csv(os.path.join(DATA, "chile_mse.csv")).Subject),
                "Cruces": set(pd.read_csv(os.path.join(DATA, "spain_mse.csv")).Subject)}
    rows = []
    for coh, s, g, x in subjects():
        if coh not in analysed or s not in analysed[coh]:
            continue
        if coh == "CETRAM":
            if len(x) < N_CETRAM:
                continue
            x = x[:N_CETRAM]
        a = curve(x)
        b = np.mean([curve(iaaft(x)) for _ in range(N_SURR)], axis=0)
        share, peak = resp_band(x)
        rows.append(dict(Cohort=coh, Subject=s, Group=g, n_beats=len(x),
                         t1=a[0], t5=a[4], t1_surr=b[0], t5_surr=b[4],
                         resp_share=share, resp_peak=peak))
    R = pd.DataFrame(rows)
    R["falling"] = R.t5 < R.t1
    R["falling_surr"] = R.t5_surr < R.t1_surr
    R["surr_gap_t5"] = R.t5_surr - R.t5
    R.round(5).to_csv(OUT, index=False)

    for c, d in R.groupby("Cohort"):
        print(f"{c:7s} n={len(d):2d}  falling: real {100*d.falling.mean():.0f}%, "
              f"surrogate {100*d.falling_surr.mean():.0f}%  | resp-band share {d.resp_share.median():.2f}")
    d = R[R.Cohort == "Cruces"]
    print(f"Cruces resp-band share, falling vs rising curves: "
          f"{d[d.falling].resp_share.median():.2f} vs {d[~d.falling].resp_share.median():.2f}")
    rho, p = spearmanr(d.resp_peak, d.surr_gap_t5)
    print(f"Cruces rho(respiratory-peak sharpness, surrogate gap at tau=5) = {rho:+.2f} (p={p:.2f})")


if __name__ == "__main__":
    main()
