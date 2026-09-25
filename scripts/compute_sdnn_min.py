#!/usr/bin/env python3
"""
Compute the Suzuki et al. (2022) minimum-value HRV indices for all three cohorts.

Suzuki M et al., J Neural Transm 129:1299-1306 (the source publication for the
Nagoya dataset) split the record into consecutive fixed-length beat windows,
compute SDNN and CVRR in each, then take the minimum / 1st decile / 1st quartile
/ median across windows. SDNN100-min reached AUC 0.90 in their cohort — higher
than our complexity index — yet it was never included in our Figure 5 benchmark,
which compares only whole-record metrics. This script supplies it.

Computed for every cohort, not just Nagoya, because the comparison is only
meaningful if the same metric is available everywhere. Note the important
caveat this exposes: the minimum over N windows is a different statistic
depending on N. Nagoya has ~560 windows of 100 beats to search; CETRAM has ~10
and Cruces ~5. SDNN-min is intrinsically a long-recording metric, and
`n_windows` is written out so this can be stated rather than glossed.

Output: data/sdnn_min_all_centers.csv

Run:  python scripts/compute_sdnn_min.py
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))

CETRAM_DET = os.path.join(ROOT, "CETRAM", "public_release", "results", "cleaned_detections")
CRUCES_RRI = os.path.join(ROOT, "Cruces", "public_release", "data", "processed", "RRi")
NAGOYA_RRI = os.path.join(ROOT, "Nagoya", "public_release", "data", "processed_rri")
NAGOYA_META = os.path.join(ROOT, "Nagoya", "public_release", "data", "metadata", "metadata.csv")

RRI_MIN, RRI_MAX = 300.0, 2000.0
WINDOWS = (100, 200, 300)


def window_indices(rri: np.ndarray, nb: int) -> dict:
    """min / 1st decile / 1st quartile / median of SDNN and CVRR across windows."""
    L = len(rri) // nb
    out = {f"n_windows_{nb}": L}
    if L < 2:
        for nm in ("SDNN", "CVRR"):
            for lab in ("min", "p10", "p25", "p50"):
                out[f"{nm}{nb}_{lab}"] = np.nan
        return out
    W = rri[:L * nb].reshape(L, nb)
    sd = W.std(axis=1, ddof=1)
    cv = 100.0 * sd / W.mean(axis=1)
    for nm, v in (("SDNN", sd), ("CVRR", cv)):
        out[f"{nm}{nb}_min"] = float(v.min())
        out[f"{nm}{nb}_p10"] = float(np.percentile(v, 10))
        out[f"{nm}{nb}_p25"] = float(np.percentile(v, 25))
        out[f"{nm}{nb}_p50"] = float(np.percentile(v, 50))
    return out


def clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size and np.nanmax(x) < 20:      # seconds -> ms
        x = x * 1000.0
    return x[(x > RRI_MIN) & (x < RRI_MAX)]


def load_cetram():
    for g in ("Control", "PD"):
        for f in sorted(glob.glob(os.path.join(CETRAM_DET, g, "*_cleaned.csv"))):
            s = pd.read_csv(f)["sample"].dropna().astype(np.int64).to_numpy()
            yield os.path.basename(f).replace("_cleaned.csv", ""), "Chile", g, clean(np.diff(s))


def cruces_group(sub: str) -> str:
    # Same convention as Cruces/public_release/scripts/entropy.jl
    if sub.startswith("E"):
        return "Control"
    if sub.startswith("B") or sub.startswith("C"):
        return "PD"
    return "Other"


def load_cruces():
    for f in sorted(glob.glob(os.path.join(CRUCES_RRI, "*.csv"))):
        sub = os.path.basename(f).replace(".csv", "")
        d = pd.read_csv(f, header=None)
        yield sub, "Spain", cruces_group(sub), clean(d.iloc[:, -1].to_numpy())


def load_nagoya():
    meta = pd.read_csv(NAGOYA_META)
    # metadata.csv uses 'PD' / 'control'; .capitalize() would mangle 'PD' -> 'Pd'
    norm = {"pd": "PD", "control": "Control"}
    grp = {k: norm.get(str(v).strip().lower(), str(v))
           for k, v in zip(meta.Subject_ID, meta.Group)}
    for f in sorted(glob.glob(os.path.join(NAGOYA_RRI, "*_RRi.txt"))):
        sub = os.path.basename(f).replace("_RRi.txt", "")
        if sub not in grp:
            continue
        d = np.loadtxt(f)
        yield sub, "Japan", grp[sub], clean(d[:, 1] if d.ndim == 2 else d)


def main():
    rows = []
    for loader, name in ((load_cetram, "CETRAM"), (load_cruces, "Cruces"), (load_nagoya, "Nagoya")):
        n = 0
        for sub, centre, group, rri in loader():
            if len(rri) < 200:
                continue
            rec = dict(Subject=sub, Center=centre, Group=group,
                       n_beats=len(rri), HR=60000.0 / rri.mean())
            for nb in WINDOWS:
                rec.update(window_indices(rri, nb))
            rows.append(rec)
            n += 1
        print(f"  {name:8s} {n} subjects")

    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "sdnn_min_all_centers.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    print("\nwindows available per subject (median):")
    print(df.groupby("Center")[["n_beats", "n_windows_100"]].median().to_string())
    print("\nSDNN100_min by centre and group (median):")
    print(df[df.Group.isin(["Control", "PD"])]
          .pivot_table(index="Center", columns="Group", values="SDNN100_min",
                       aggfunc="median").round(2).to_string())


if __name__ == "__main__":
    main()
