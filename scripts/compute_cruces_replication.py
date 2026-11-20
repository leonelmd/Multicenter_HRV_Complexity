#!/usr/bin/env python3
"""Cruces: the resting HRV metrics of the source study (Iniguez et al. 2022), recomputed in
the analysed participants, against our complexity index (Table S-Iniguez, Suppl. Fig. S3).

Time-domain and Poincare indices come from data/spain_metrics.csv. Spectral powers are
recomputed from the original interval series: cubic-spline resampling at 4 Hz, Welch
spectrum (512-sample segments), trapezoidal band power, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz.
AUC is oriented so that values above 0.5 mean lower in PD; 95% CI by percentile bootstrap
(2000 resamples within group, seed 0); p by two-sided Mann-Whitney.

Writes data/cruces_iniguez_replication.csv and data/cruces_iniguez_medians.csv.
Needs the original Cruces series (paths.py).
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch
from scipy.stats import mannwhitneyu

from paths import DATA, CRUCES_RRI, require


def nauc(v):
    v = np.asarray(v, float)
    return np.trapezoid(v, np.arange(1, len(v) + 1)) / len(v)


def band_powers(rri_ms):
    t = np.cumsum(rri_ms) / 1000.0
    y = CubicSpline(t, rri_ms)(np.arange(t[0], t[-1], 0.25))
    f, p = welch(y, fs=4.0, nperseg=min(512, len(y)))
    bp = lambda lo, hi: np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
    return bp(0.04, 0.15), bp(0.15, 0.40)


def auc_ci_p(c, p, rng, n_boot=2000):
    c, p = np.asarray(c, float), np.asarray(p, float)
    a = lambda x, y: mannwhitneyu(x, y).statistic / len(x) / len(y)
    boot = [a(rng.choice(c, len(c)), rng.choice(p, len(p))) for _ in range(n_boot)]
    return a(c, p), *np.percentile(boot, [2.5, 97.5]), mannwhitneyu(c, p).pvalue


def main():
    met = pd.read_csv(os.path.join(DATA, "spain_metrics.csv"))
    met = met[met.Group.isin(["Control", "PD"])].set_index("Subject")
    spec = {}
    for f in sorted(glob.glob(os.path.join(require(CRUCES_RRI, "the original Cruces series"), "*.csv"))):
        s = os.path.basename(f)[:-4]
        if s in met.index:
            x = np.loadtxt(f).astype(float)
            spec[s] = band_powers(x[(x > 300) & (x < 2000)])
    met["LF"] = pd.Series({s: v[0] for s, v in spec.items()})
    met["HF"] = pd.Series({s: v[1] for s, v in spec.items()})
    met["LFHF"] = met.LF / met.HF
    met["SD1SD2"] = met.HRV_SD1 * met.HRV_SD2
    mse = pd.read_csv(os.path.join(DATA, "spain_mse.csv"))
    met["Complexity"] = mse[mse.Scales <= 5].sort_values("Scales").groupby("Subject").MSE.apply(nauc)

    rows = [("SDNN", "time", "HRV_SDNN"), ("RMSSD", "time", "HRV_RMSSD"), ("NNiqr", "time", "HRV_IQRNN"),
            ("SD1", "nonlin", "HRV_SD1"), ("SD2", "nonlin", "HRV_SD2"), ("SD1$\\times$SD2", "nonlin", "SD1SD2"),
            ("HF power", "spec", "HF"), ("LF power", "spec", "LF"), ("LF/HF", "spec", "LFHF"),
            ("mean NN", "time", "HRV_MeanNN"), ("rcMSE nAUC(1-5)", "ours", "Complexity")]
    rng = np.random.default_rng(0)
    out, med = [], []
    for label, fam, col in rows:
        c, p = met.loc[met.Group == "Control", col].dropna(), met.loc[met.Group == "PD", col].dropna()
        a, lo, hi, pv = auc_ci_p(c, p, rng)
        out.append(dict(metric=label, family=fam, AUC=a, lo=lo, hi=hi, p=pv))
        med.append(dict(metric=label, Control=c.median(), PD=p.median(), n_Control=len(c), n_PD=len(p)))
    pd.DataFrame(out).to_csv(os.path.join(DATA, "cruces_iniguez_replication.csv"), index=False)
    pd.DataFrame(med).to_csv(os.path.join(DATA, "cruces_iniguez_medians.csv"), index=False)
    print(pd.DataFrame(out).merge(pd.DataFrame(med), on="metric").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
