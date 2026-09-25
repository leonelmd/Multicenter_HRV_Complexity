#!/usr/bin/env python3
"""
Apply the Suzuki et al. (2022) adjacent-difference exclusion to the Nagoya
RMSSD / pNN50 columns of japan_recalc_metrics.csv.

Why a patch rather than a full recompute: `nk.hrv()` cannot process a 24-h RRi
series — it attempts an N x N allocation (~23.7 GiB for a 56k-beat recording) —
so compute_japan_fullday_hrv.py cannot be re-run end to end on the long records.
Only two columns are affected by the defect and both are computable directly
from the RRi in seconds, so this script rewrites just those.

Suzuki et al., J Neural Transm (2022) 129:1299-1306, p.1301: for RMSSD and
PNN50, "RRIs with an adjacent RRIs difference of 100 ms or more were excluded
from the calculation to avoid the effects of extrasystoles and artifacts."

Without it, our 24-h values were RMSSD 81.9/55.1 ms and pNN50 7.8/3.4 % against
their published 22.2/15.0 ms and 3.98/1.14 % — inflated roughly 3.7x and 2x by
extrasystoles and movement artifact. Those values fed Figures 5 and 7.

Original values are preserved as HRV_RMSSD_unfiltered / HRV_pNN50_unfiltered.
Idempotent: re-running is a no-op once the convention column is present.

See Multicenter/SUZUKI_CONSISTENCY_CHECK.md sec. 1.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAGOYA = os.environ.get(
    "NAGOYA_ROOT",
    os.path.abspath(os.path.join(BASE, "..", "..", "Nagoya", "public_release")))
RRI_DIR = os.path.join(NAGOYA, "data", "processed_rri")
TARGET = os.path.join(BASE, "data", "japan_recalc_metrics.csv")
EXCL_MS = 100.0
RRI_MIN, RRI_MAX = 300.0, 2000.0


def load_rri(sub):
    p = os.path.join(RRI_DIR, f"{sub}_RRi.txt")
    if not os.path.exists(p):
        return None
    d = np.loadtxt(p)
    x = (d[:, 1] if d.ndim == 2 else d) * 1000.0
    return x[(x > RRI_MIN) & (x < RRI_MAX)]


def main():
    df = pd.read_csv(TARGET)
    if "vagal_metric_convention" in df.columns:
        print("Already patched — nothing to do.")
        return

    df["HRV_RMSSD_unfiltered"] = df["HRV_RMSSD"]
    df["HRV_pNN50_unfiltered"] = df["HRV_pNN50"]

    n = 0
    for i, sub in df["Subject"].items():
        x = load_rri(sub)
        if x is None or len(x) < 100:
            print(f"  {sub}: RRi not found — left unchanged")
            continue
        d = np.diff(x)
        d = d[np.abs(d) < EXCL_MS]
        if d.size < 2:
            continue
        df.at[i, "HRV_RMSSD"] = float(np.sqrt(np.mean(d ** 2)))
        df.at[i, "HRV_pNN50"] = float(100.0 * np.mean(np.abs(d) > 50.0))
        n += 1

    df["vagal_metric_convention"] = f"Suzuki2022_adjdiff_lt_{int(EXCL_MS)}ms"
    df.to_csv(TARGET, index=False)
    print(f"Patched {n}/{len(df)} subjects -> {TARGET}\n")
    print(df.groupby("Group")[["HRV_RMSSD_unfiltered", "HRV_RMSSD",
                               "HRV_pNN50_unfiltered", "HRV_pNN50"]]
            .mean().round(2).to_string())
    print("\nSuzuki published (24 h): RMSSD DC 22.2 / PD 15.0 ms;"
          "  pNN50 DC 3.98 / PD 1.14 %")


if __name__ == "__main__":
    main()
