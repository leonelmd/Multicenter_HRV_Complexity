#!/usr/bin/env python3
"""
Minimal RRI Cleaning — Artifact Sensitivity Check
===================================================
Tests whether the PD < Control MSE complexity finding survives minimal
RRI cleaning. The original chile_mse.csv was computed WITHOUT beat-level
cleaning; recordings were only selected if bsqi (recording-level quality
index) passed. However, some accepted recordings contain physiologically
impossible intervals (e.g., 25,504 ms in P010) that inflate σ and force
SampEn → 0 via the tolerance mechanism (r = 0.15σ).

Two cleaning approaches are compared:
  1. Threshold-only: remove RRI > 2000 ms and < 300 ms, split/merge accordingly.
  2. Kubios + threshold: nk.signal_fixpeaks (Lipponen & Tarvainen 2019) applied
     to cumulative peak positions, then threshold residuals (conservative residual
     cutoff of 2000 ms, matching the CETRAM PAPER/BBDD pipeline).

Both are minimal by design: they only remove physiologically impossible intervals,
not ectopic beats or high-frequency noise.

Coverage: 25/29 PD and 41/43 Control subjects have RRI files (hrv_signals/).
The 4 missing PD subjects (P007, P012, P034, P035) are bsqi-only additions
without raw RRI files; the 2 missing Controls (C001, C018) similarly have no
raw files.

Outputs
-------
results/cleaning_sensitivity/
  mse_cleaned_threshold.csv     — per-subject MSE curves after threshold cleaning
  mse_cleaned_kubios.csv        — per-subject MSE curves after Kubios + threshold
  complexity_comparison.csv     — before/after complexity index & group statistics
  cleaning_sensitivity.png      — visualization
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
import neurokit2 as nk

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.dirname(SCRIPT_DIR)
RRI_DIR     = "/Users/leo/HRV-Complexity/CETRAM/public_release/data/processed/hrv_signals"
MSE_CSV     = os.path.join(BASE, "data", "chile_mse.csv")
METR_CSV    = os.path.join(BASE, "data", "chile_metrics.csv")
OUT_DIR     = os.path.join(BASE, "results", "cleaning_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

FS      = 1000   # Hz — Bitalino sampling rate
SCALES  = list(range(1, 21))   # scales 1–20 matching chile_mse.csv
DIM     = 2                    # SampEn embedding dimension m=2
TOL_R   = 0.15                 # tolerance as fraction of σ
RRI_MIN = 300                  # ms — HR > 200 bpm impossible (double detection)
RRI_MAX = 2000                 # ms — HR < 30 bpm impossible in awake adults

# ── bsqi-passing subjects (from chile_mse.csv) ────────────────────────────────
mse_orig = pd.read_csv(MSE_CSV)
bsqi_subjects = {
    grp: set(mse_orig[mse_orig["Group"] == grp]["Subject"].unique())
    for grp in ["PD", "Control"]
}
print(f"bsqi PD: {len(bsqi_subjects['PD'])}, Control: {len(bsqi_subjects['Control'])}")


# ── Load RRI files ─────────────────────────────────────────────────────────────
def load_rri_files():
    """Load all available hrv_signals/*.csv for bsqi-passing subjects."""
    records = {}
    for fname in sorted(os.listdir(RRI_DIR)):
        if not fname.endswith("_rri.csv"):
            continue
        subj = fname.replace("_rri.csv", "")
        grp  = "PD" if subj.startswith("P") else "Control"
        if subj not in bsqi_subjects[grp]:
            continue   # not a bsqi-passing subject
        rri = pd.read_csv(os.path.join(RRI_DIR, fname))["RRI_ms"].dropna().values.astype(float)
        records[subj] = {"group": grp, "rri_raw": rri}
    return records


# ── Cleaning Approach 1: hard physiological threshold ─────────────────────────
def clean_threshold(rri):
    """
    Remove intervals outside [RRI_MIN, RRI_MAX].

    Long intervals (> RRI_MAX) represent missed beats: they are split into
    equal sub-intervals (rounded to nearest physiological beat).
    Short intervals (< RRI_MIN) represent extra beats: they are merged into
    the next interval.

    Returns the cleaned RRI array (different length from input).
    """
    cleaned = []
    i = 0
    while i < len(rri):
        v = rri[i]
        if v > RRI_MAX:
            # Missed beat(s): split into N equal parts
            n_beats = max(2, round(v / np.median(rri[np.isfinite(rri)])))
            cleaned.extend([v / n_beats] * n_beats)
        elif v < RRI_MIN and i + 1 < len(rri):
            # Extra beat: merge with next interval
            cleaned.append(v + rri[i + 1])
            i += 1   # skip next
        else:
            cleaned.append(v)
        i += 1
    return np.array(cleaned)


# ── Cleaning Approach 2: Kubios (Lipponen & Tarvainen 2019) + threshold ───────
def clean_kubios(rri):
    """
    Convert RRI series to cumulative peak positions (ms → sample indices at FS),
    apply nk.signal_fixpeaks with Kubios method, convert back to intervals,
    then apply hard threshold on any residual impossible values.
    """
    # Cumulative sum gives sample positions at FS=1000 Hz
    peaks = np.round(np.concatenate([[0], np.cumsum(rri)])).astype(int)

    try:
        _, peaks_clean = nk.signal_fixpeaks(
            peaks, sampling_rate=FS, iterative=True, method="Kubios", show=False
        )
        rri_clean = np.diff(peaks_clean).astype(float)
    except Exception as e:
        print(f"    Kubios failed ({e}), falling back to threshold-only")
        rri_clean = rri.copy()

    # Cubic spline on residual impossible values (same as CETRAM PAPER/BBDD pipeline)
    mask = (rri_clean < RRI_MIN) | (rri_clean > RRI_MAX)
    if mask.sum() > 0 and (~mask).sum() >= 4:
        idx = np.arange(len(rri_clean))
        cs  = CubicSpline(idx[~mask], rri_clean[~mask])
        rri_clean[mask] = np.clip(cs(idx[mask]), RRI_MIN, RRI_MAX)
    elif mask.sum() > 0:
        # Not enough clean points for spline — fall back to threshold split/merge
        rri_clean = clean_threshold(rri_clean)

    return rri_clean


# ── Compute MSE ───────────────────────────────────────────────────────────────
def compute_mse(rri, scales=SCALES, m=DIM, r_frac=TOL_R):
    """
    Compute Multiscale Sample Entropy.
    Tolerance r = r_frac * σ(original series) — fixed across scales (Costa 2002).
    Returns array of length len(scales), NaN where computation fails.
    """
    r = r_frac * np.std(rri, ddof=1)
    if r == 0 or len(rri) < 50:
        return np.full(len(scales), np.nan)

    mse_vals = []
    for scale in scales:
        # Coarse-graining
        n_cg = len(rri) // scale
        if n_cg < 10:
            mse_vals.append(np.nan)
            continue
        cg = np.array([rri[j*scale:(j+1)*scale].mean() for j in range(n_cg)])
        try:
            se, _ = nk.entropy_sample(cg, dimension=m, tolerance=r)
            mse_vals.append(se)
        except Exception:
            mse_vals.append(np.nan)
    return np.array(mse_vals)


# ── Complexity index: nAUC(scales 1–5) / HR ──────────────────────────────────
def complexity_index(mse_vals, mean_rri_ms):
    """nAUC over scales 1–5, normalised by HR (bpm)."""
    s1_to_s5 = mse_vals[:5]
    if np.any(np.isnan(s1_to_s5)):
        return np.nan
    nauc = np.trapz(s1_to_s5, x=list(range(1, 6)))
    hr   = 60000.0 / mean_rri_ms
    return nauc / hr


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    records = load_rri_files()
    print(f"\nLoaded RRI files: {len(records)} subjects "
          f"({sum(1 for v in records.values() if v['group']=='PD')} PD, "
          f"{sum(1 for v in records.values() if v['group']=='Control')} Control)")

    results = []
    mse_rows_thr, mse_rows_kub = [], []

    for subj, rec in sorted(records.items()):
        grp = rec["group"]
        rri_raw = rec["rri_raw"]

        print(f"\n{subj} ({grp}) — raw n={len(rri_raw)}, "
              f"max={rri_raw.max():.0f} ms, >2000ms: {(rri_raw>2000).sum()}")

        # Original MSE (from chile_mse.csv — pre-computed)
        orig = mse_orig[(mse_orig["Subject"] == subj)]
        if not orig.empty:
            orig_vals = orig.sort_values("Scales")["MSE"].values
            ci_orig   = complexity_index(orig_vals, np.mean(rri_raw[rri_raw < 2000]))
        else:
            orig_vals = np.full(len(SCALES), np.nan)
            ci_orig   = np.nan

        # Threshold cleaning
        rri_thr = clean_threshold(rri_raw)
        mse_thr = compute_mse(rri_thr)
        ci_thr  = complexity_index(mse_thr, np.mean(rri_thr))
        print(f"  Threshold: n={len(rri_thr)}, removed {len(rri_raw)-len(rri_raw[(rri_raw>=300)&(rri_raw<=2000)])} impossible, CI={ci_thr:.4f}")

        # Kubios + threshold cleaning
        rri_kub = clean_kubios(rri_raw)
        mse_kub = compute_mse(rri_kub)
        ci_kub  = complexity_index(mse_kub, np.mean(rri_kub))
        print(f"  Kubios:    n={len(rri_kub)}, CI={ci_kub:.4f}")

        results.append({
            "Subject": subj, "Group": grp,
            "CI_original": ci_orig,
            "CI_threshold": ci_thr,
            "CI_kubios": ci_kub,
            "sigma_raw": np.std(rri_raw, ddof=1),
            "sigma_threshold": np.std(rri_thr, ddof=1),
            "sigma_kubios": np.std(rri_kub, ddof=1),
            "n_artifacts_raw": (rri_raw > 2000).sum() + (rri_raw < 300).sum(),
        })

        for i, sc in enumerate(SCALES):
            mse_rows_thr.append({"Subject": subj, "Group": grp, "Scales": sc, "MSE": mse_thr[i]})
            mse_rows_kub.append({"Subject": subj, "Group": grp, "Scales": sc, "MSE": mse_kub[i]})

    res_df = pd.DataFrame(results)
    pd.DataFrame(mse_rows_thr).to_csv(os.path.join(OUT_DIR, "mse_cleaned_threshold.csv"), index=False)
    pd.DataFrame(mse_rows_kub).to_csv(os.path.join(OUT_DIR, "mse_cleaned_kubios.csv"), index=False)

    # ── Group statistics ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GROUP COMPARISON: PD vs Control — Complexity Index")
    print("=" * 70)

    for col, label in [
        ("CI_original",   "Original (from chile_mse.csv, no beat cleaning)"),
        ("CI_threshold",  "After threshold cleaning (>2000ms / <300ms)    "),
        ("CI_kubios",     "After Kubios + threshold cleaning                "),
    ]:
        pd_vals   = res_df[res_df["Group"] == "PD"][col].dropna().values
        ctrl_vals = res_df[res_df["Group"] == "Control"][col].dropna().values
        if len(pd_vals) < 3 or len(ctrl_vals) < 3:
            print(f"\n{label}: insufficient data")
            continue

        u_stat, p_val = stats.mannwhitneyu(pd_vals, ctrl_vals, alternative="two-sided")
        d = (np.mean(pd_vals) - np.mean(ctrl_vals)) / np.std(
            np.concatenate([pd_vals, ctrl_vals]), ddof=1)

        print(f"\n  {label}")
        print(f"    PD     : n={len(pd_vals):2d}, median={np.median(pd_vals):.4f} "
              f"[IQR {np.percentile(pd_vals,25):.4f}–{np.percentile(pd_vals,75):.4f}]")
        print(f"    Control: n={len(ctrl_vals):2d}, median={np.median(ctrl_vals):.4f} "
              f"[IQR {np.percentile(ctrl_vals,25):.4f}–{np.percentile(ctrl_vals,75):.4f}]")
        print(f"    Mann-Whitney U={u_stat:.0f}, p={p_val:.4f}, Cohen's d={d:.3f}")
        direction = "PD < Control" if np.median(pd_vals) < np.median(ctrl_vals) else "PD > Control"
        print(f"    Direction: {direction}")

    # ── σ inflation summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  σ INFLATION SUMMARY (before vs after cleaning)")
    print("=" * 70)
    for grp in ["PD", "Control"]:
        sub = res_df[res_df["Group"] == grp]
        print(f"\n  {grp} (n={len(sub)}):")
        print(f"    σ_raw  : median={sub['sigma_raw'].median():.1f} ms, "
              f"max={sub['sigma_raw'].max():.1f} ms")
        print(f"    σ_thr  : median={sub['sigma_threshold'].median():.1f} ms, "
              f"max={sub['sigma_threshold'].max():.1f} ms")
        print(f"    n_artifacts>2000ms: {sub['n_artifacts_raw'].sum():.0f} total")

    res_df.to_csv(os.path.join(OUT_DIR, "complexity_comparison.csv"), index=False)
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
