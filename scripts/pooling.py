"""Pooled analyses across cohorts use values standardized within cohort.

Absolute entropy and HRV values are not comparable across cohorts (different devices,
lengths and scale ranges), so a correlation over raw pooled values mostly measures the
offsets between cohorts. Every 'Pooled' analysis in the Figure 8 pipeline therefore
z-scores the complexity index and each HRV metric within cohort first (Methods).
"""
import numpy as np
import pandas as pd


def within_center_z(df: pd.DataFrame, cols, center_col: str = "Center") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        g = out.groupby(center_col)[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std")
    return out
