#!/usr/bin/env python3
"""Cruces: RR intervals -> HRV indices (NeuroKit2 hrv, entropy indices dropped).

Reads $HRV_DATA_ROOT/cruces/rri/<ID>.csv; writes data/spain_metrics.csv. Group from the
study code: E = Control, B or C = PD, anything else = Other (excluded from all analyses).
"""
import os
import sys

import neurokit2 as nk
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CRUCES_RRI, DATA, require  # noqa: E402


def group_of(sid):
    return "Control" if sid.startswith("E") else "PD" if sid[0] in "BC" else "Other"


def main():
    rows = []
    for f in sorted(x for x in os.listdir(require(CRUCES_RRI, "the Cruces RR series")) if x.endswith(".csv")):
        rri = np.loadtxt(os.path.join(CRUCES_RRI, f))
        if len(rri) < 10:
            continue
        # peaks on a 1-ms grid, so that diff(peaks) returns the intervals with sub-sample timing
        peaks = np.round(np.concatenate(([0], np.cumsum(rri)))).astype(int)
        h = nk.hrv(peaks, sampling_rate=1000, show=False)
        h = h.drop(columns=[c for c in h.columns if "En" in c or "MSE" in c], errors="ignore")
        sid = f[:-4]
        h["Subject"], h["Group"] = sid, group_of(sid)
        rows.append(h)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(DATA, "spain_metrics.csv"), index=False)
    print(f"wrote data/spain_metrics.csv ({len(out)} recordings)")


if __name__ == "__main__":
    main()
