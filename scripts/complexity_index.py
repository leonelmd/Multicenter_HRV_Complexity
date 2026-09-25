"""
Canonical definition of the rcMSE complexity index.

Why this module exists
----------------------
Before 2026-08-16 the complexity index was computed inline in four places
(generate_figure4/5/6.py and entropy.jl) as `MSE.mean()` over the scale range,
while the README, both manuscript drafts and MulticohortRCMSE_Paper all described
it as a *normalized area under the entropy-scale curve* (nAUC). Those are different
formulas, and no single place in the codebase stated which was actually used.

Empirically the choice does not matter (CETRAM, scales 1-5, n=73):

    definition                          p        AUC     p (/HR)   AUC (/HR)
    mean over scales                    0.0045   0.697   0.0004    0.743
    trapezoidal area                    0.0045   0.697   0.0005    0.740
    trapezoid / (tau_max - tau_min)     0.0045   0.697   0.0005    0.740
    simple sum                          0.0045   0.697   0.0004    0.743

`mean` is retained as the default so that published numbers do not shift, but it is
now named, documented and called from one place. Report it in Methods as
"mean refined-composite multiscale entropy across scales tau_min..tau_max", or use
`method="nauc"` if you prefer the literal normalized-area wording — they agree to
within 0.003 AUC.

Normalization by heart rate
---------------------------
`divide_by_hr=True` reproduces the primary endpoint used throughout the study.
Note that on CETRAM the *unnormalized* index is already orthogonal to HR
(rho = -0.05, p = 0.68) and the division injects a dependency (rho = -0.44), which
propagates into correlation analyses. Use `divide_by_hr=False` for Figures 6 and 7
and for any mechanistic claim. See Multicenter/CETRAM_METHODS_AUDIT.md sec. 2b.3.
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
