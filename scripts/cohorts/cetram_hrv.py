#!/usr/bin/env python3
"""CETRAM: beat times -> HRV indices (NeuroKit2 hrv, entropy indices dropped).

Reads $HRV_DATA_ROOT/cetram/beats/{Control,PD}/<ID>_cleaned.csv; writes data/chile_metrics.csv.
"""
import os
import sys

import neurokit2 as nk
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CETRAM_PEAKS, DATA, require  # noqa: E402

FS = 1000


def main():
    rows = []
    for group in ("Control", "PD"):
        folder = os.path.join(require(CETRAM_PEAKS, "the CETRAM beat files"), group)
        for f in sorted(x for x in os.listdir(folder) if x.endswith("_cleaned.csv")):
            peaks = pd.read_csv(os.path.join(folder, f))["sample"].dropna().astype(int).to_numpy()
            if len(peaks) < 3:
                continue
            h = nk.hrv(peaks, sampling_rate=FS, show=False)
            h = h.drop(columns=[c for c in h.columns if "En" in c])   # entropy is rcMSE's job
            h["Subject"], h["Group"] = f.replace("_cleaned.csv", ""), group
            rows.append(h)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(DATA, "chile_metrics.csv"), index=False)
    print(f"wrote data/chile_metrics.csv ({len(out)} recordings)")


if __name__ == "__main__":
    main()
