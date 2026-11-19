#!/usr/bin/env python3
"""Recompute only the Poincare columns of data/japan_window_features.csv with the
>= 100 ms successive-difference exclusion that RMSSD/pNN50 already use (2026-11-18).

Re-running compute_japan_window_features.py does the same, but also recomputes every
NeuroKit2 column; this patch leaves those untouched so that Table 2 is unchanged.
Run: NAGOYA_ROOT=<Nagoya/public_release> python scripts/patch_japan_poincare.py
"""
import importlib.util, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("w", os.path.join(HERE, "compute_japan_window_features.py"))
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
meta = pd.read_csv(w.NAGOYA_META_CSV).set_index("Subject_ID")
df = pd.read_csv(w.OUT_CSV)
for i, sid in enumerate(df.Subject):
    rri = w.extract_window_rri(os.path.join(w.NAGOYA_RRI_DIR, sid + "_RRi.txt"), str(meta.loc[sid, "Start_Time"]))
    assert len(rri) == df.loc[i, "n_beats"], sid
    d0 = np.diff(rri); d0f = d0[np.abs(d0) < w.SUZUKI_DIFF_EXCL_MS]
    sd1 = np.sqrt(0.5) * np.std(d0f, ddof=1)
    sd2 = np.sqrt(max(2 * np.var(rri, ddof=1) - 0.5 * np.var(d0f, ddof=1), 0.0))
    df.loc[i, "HRV_SD1_unfiltered"] = np.sqrt(0.5) * np.std(d0, ddof=1)
    df.loc[i, ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2"]] = [sd1, sd2, sd1 / sd2]
df.to_csv(w.OUT_CSV, index=False)
print("patched", len(df), "rows")
