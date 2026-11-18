#!/usr/bin/env python3
"""
Nagoya beat coverage of a 4-h window (Methods, Window selection; Supplement, Nagoya beat
coverage). Coverage = summed retained intervals (300-2000 ms) / 4 h.
Reported for the pre-specified 18:00-22:00 window (the 47 analysed) and for the 16:00-20:00 alternative (the 45 analysed there).
Run: python scripts/compute_nagoya_coverage.py
"""
import os, glob
from datetime import datetime
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data")
RRI = os.path.abspath(os.path.join(BASE, "..", "..", "Nagoya", "public_release", "data", "processed_rri"))


def coverage(a, b):
    meta = pd.read_csv(os.path.join(D, "japan_metadata.csv")).set_index("Subject_ID")
    out = {}
    for f in sorted(glob.glob(os.path.join(RRI, "*_RRi.txt"))):
        s = os.path.basename(f)[:-8]
        st = datetime.strptime(str(meta.loc[s, "Start_Time"]).strip(), "%H:%M:%S")
        off = st.hour * 3600 + st.minute * 60 + st.second
        d = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "rri"])
        h = ((d.t + off) % 86400) / 3600
        r = d.rri[(h >= a) & (h < b)]; r = r[(r > 0.3) & (r < 2.0)]
        if len(r) >= 300:
            out[s] = float(r.sum() / ((b - a) * 3600) * 100)
    return pd.Series(out, name="coverage")


def auc(df):
    c, p = df[df.Group == "Control"].idx, df[df.Group == "PD"].idx
    u = mannwhitneyu(c, p, alternative="two-sided")
    return u.statistic / (len(c) * len(p)), u.pvalue


def main():
    for a, b, f in ((18, 22, "japan_window_mse.csv"), (16, 20, "japan_afternoon_mse.csv")):
        keep = set(pd.read_csv(os.path.join(D, f)).Subject)
        c = coverage(a, b); c = c[c.index.isin(keep)]
        print(f"{a:02d}-{b:02d} h: n={len(c)}  median {c.median():.1f} %  below 80 %: {(c < 80).sum()}")
    c = coverage(18, 22)
    m = pd.read_csv(os.path.join(D, "japan_window_mse.csv"))
    m = m[(m.Scales >= 1) & (m.Scales <= 20)].sort_values("Scales")
    idx = m.groupby(["Subject", "Group"]).MSE.apply(
        lambda v: np.trapezoid(v.values, np.arange(1, len(v) + 1)) / len(v)).rename("idx").reset_index()
    idx["coverage"] = idx.Subject.map(c)
    g = mannwhitneyu(idx[idx.Group == "Control"].coverage, idx[idx.Group == "PD"].coverage).pvalue
    r = spearmanr(idx.coverage, idx.idx)
    print(f"18-22 h: coverage vs group p={g:.2f}; rho(coverage, index)={r[0]:+.2f} (p={r[1]:.2f})")
    for t in (50, 60, 70, 80, 90, 95):
        k = idx[idx.coverage >= t]
        a_, p_ = auc(k)
        print(f"  coverage >= {t}%: n={len(k)}  AUC {a_:.3f}  p={p_:.4f}")


if __name__ == "__main__":
    main()
