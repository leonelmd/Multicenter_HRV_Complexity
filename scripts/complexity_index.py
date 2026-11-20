"""
Complexity index of an rcMSE curve.

The study's index is the trapezoidal area under the entropy-scale curve over scales
a..b divided by the number of scales (Methods); the figure scripts compute it as
`subject_index_table(..., method="trapz")` divided by the number of scales.
The other methods ("mean", "nauc" = area / span, "sum") are kept for sensitivity
checks only.

The index is not divided by heart rate (Fig. S1). `divide_by_hr=True` gives the
ratio index / HR, used only in the analyses that test that normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["complexity_index", "subject_index_table", "DEFAULT_METHOD"]

DEFAULT_METHOD = "mean"


def complexity_index(scales, mse, method: str = DEFAULT_METHOD) -> float:
    """Collapse one subject's entropy-vs-scale curve to a single number.

    Parameters
    ----------
    scales, mse : array-like
        Scale factors and matching entropy values. Sorted internally.
    method : {"mean", "trapz", "nauc", "sum"}
        "mean"  - arithmetic mean across scales (default; historical behaviour)
        "trapz" - trapezoidal area under the curve
        "nauc"  - trapezoidal area divided by (tau_max - tau_min); the literal
                  "normalized AUC" of the manuscript text
        "sum"   - simple sum
    """
    scales = np.asarray(scales, dtype=float)
    mse = np.asarray(mse, dtype=float)
    ok = np.isfinite(mse)
    if ok.sum() == 0:
        return np.nan
    scales, mse = scales[ok], mse[ok]
    order = np.argsort(scales)
    scales, mse = scales[order], mse[order]

    if method == "mean":
        return float(np.mean(mse))
    if method == "sum":
        return float(np.sum(mse))
    area = float(np.trapezoid(mse, scales))
    if method == "trapz":
        return area
    if method == "nauc":
        span = scales[-1] - scales[0]
        return area / span if span > 0 else np.nan
    raise ValueError(f"unknown method {method!r}")


def subject_index_table(df_mse: pd.DataFrame,
                        scale_range,
                        method: str = DEFAULT_METHOD,
                        hr: pd.Series | None = None,
                        divide_by_hr: bool = False) -> pd.DataFrame:
    """Per-subject complexity index from a long MSE table.

    `df_mse` must have columns Subject, Group, Scales, MSE.
    `hr` is an optional Series indexed by Subject, in bpm.
    Returns columns: Subject, Group, Complexity (and HR if supplied).
    """
    sub = df_mse[df_mse["Scales"].isin(list(scale_range))]
    out = (sub.groupby(["Subject", "Group"], sort=False)
              .apply(lambda g: complexity_index(g["Scales"], g["MSE"], method),
                     include_groups=False)
              .rename("Complexity")
              .reset_index())
    if hr is not None:
        out["HR"] = out["Subject"].map(hr)
        if divide_by_hr:
            out["Complexity"] = out["Complexity"] / out["HR"]
    elif divide_by_hr:
        raise ValueError("divide_by_hr=True requires hr=")
    return out
