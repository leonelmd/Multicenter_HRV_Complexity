"""
Traditional HRV Metrics Consolidation
======================================
Consolidates pre-computed HRV metrics from all three cohorts into a single
unified CSV. Does NOT recompute from raw signals — uses existing outputs from
the validated per-cohort pipelines to ensure consistency with all downstream
figures and benchmarks.

Preprocessing applied upstream (per cohort):
  - Chile (CETRAM)  : ECG → Pan-Tompkins QRS detection → Malik 25% local deviation
                      filter (±10-beat window), bounds 300–2500 ms.
                      Recording duration: ~15 min rest (sample signals confirm 901–910 s).
                      HRV computed with neurokit2 nk.hrv() (Lomb-Scargle freq domain).
  - Spain (Cruces)  : PPG → Elgendi peak detection → NK2 ppg_clean(), bounds 300–2500 ms.
                      Recording duration: 5–15 min rest.
                      HRV computed with neurokit2 nk.hrv() (Lomb-Scargle freq domain).
                      *** PPG-derived RRi: frequency-domain metrics subject to PPG bias. ***
  - Japan (Nagoya)  : Pre-cleaned 24-h Holter ECG RRi, bounds 300–2000 ms only.
                      HRV computed with custom Julia pipeline + neurokit2 recalculation.

SampEn is extracted as MSE at Scale=1 from the RC-MSE toolbox outputs (*_mse.csv),
which used m=2, r=0.2×σ_per_scale.

Outputs:
  figures/Figure7/traditional_hrv_metrics.csv  — one row per subject
  Printed summary table with Mann-Whitney U p-values (BH FDR-corrected)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "figures" / "Figure7"

# ---------------------------------------------------------------------------
# Column mapping: source column → unified output column
# ---------------------------------------------------------------------------
# Common neurokit2 columns present in all three cohorts
NK2_MAP = {
    "HRV_MeanNN": "MeanNN",
    "HRV_SDNN":   "SDNN",
    "HRV_RMSSD":  "RMSSD",
    "HRV_pNN50":  "pNN50",
    "HRV_TINN":   "TINN",
    "HRV_SD1":    "SD1",
    "HRV_SD2":    "SD2",
    "HRV_SD1SD2": "SD1SD2",
    "HRV_VLF":    "VLF_power",
    "HRV_LF":     "LF_power",
    "HRV_HF":     "HF_power",
    "HRV_TP":     "Total_power",
    "HRV_LFHF":   "LF_HF",
    "HRV_LFn":    "LF_norm",
    "HRV_HFn":    "HF_norm",
}

# Columns only in neurokit2 output (Chile/Spain); not in Japan recalc file
NK2_ONLY_MAP = {
    "HRV_DFA_alpha1": "DFA_alpha1",
    "HRV_DFA_alpha2": "DFA_alpha2",
}

# Japan recalc uses a different column name for DFA
JAPAN_DFA_COL = "DFA_alpha1"

OUTPUT_COLS = [
    "Subject", "Center", "Group",
    "MeanNN", "SDNN", "RMSSD", "pNN50", "SDANN",
    "TINN", "SD1", "SD2", "SD1SD2",
    "VLF_power", "LF_power", "HF_power", "Total_power",
    "LF_HF", "LF_norm", "HF_norm",
    "DFA_alpha1", "DFA_alpha2",
    "SampEn",
    "PPG_bias", "ectopic_method",
]


def load_mse_scale1(path: Path, scale_col: str = "Scales") -> pd.DataFrame:
    """Return Subject → SampEn (MSE at Scale=1) mapping."""
    df = pd.read_csv(path)
    s1 = df[df[scale_col] == 1][["Subject", "MSE"]].copy()
    s1 = s1.rename(columns={"MSE": "SampEn"})
    return s1.reset_index(drop=True)


def normalize_group(series: pd.Series) -> pd.Series:
    """Map group labels to 'PD' / 'Control'."""
    return series.str.strip().str.lower().map(
        lambda g: "PD" if g == "pd" else "Control"
    )


# ---------------------------------------------------------------------------
# Load Chile
# ---------------------------------------------------------------------------
chile_m = pd.read_csv(DATA / "chile_metrics.csv")
chile_mse = load_mse_scale1(DATA / "chile_mse.csv")

chile = chile_m.rename(columns={**NK2_MAP, **NK2_ONLY_MAP})
chile["Center"] = "Chile"
chile["Group"] = normalize_group(chile["Group"])
chile["SDANN"] = np.nan          # Not meaningful for ~15-min recordings
chile["PPG_bias"] = False        # ECG-derived
chile["ectopic_method"] = "Malik25pct_NK2"

chile = chile.merge(chile_mse[["Subject", "SampEn"]], on="Subject", how="left")

# ---------------------------------------------------------------------------
# Load Spain
# ---------------------------------------------------------------------------
spain_m = pd.read_csv(DATA / "spain_metrics.csv")
spain_mse = load_mse_scale1(DATA / "spain_mse.csv")

spain = spain_m.rename(columns={**NK2_MAP, **NK2_ONLY_MAP})
spain["Center"] = "Spain"
# 'Other' treated as Control per project convention
spain["Group"] = spain["Group"].str.strip().str.lower().map(
    lambda g: "PD" if g == "pd" else "Control"
)
spain["SDANN"] = np.nan          # Not meaningful for 5–15 min recordings
spain["PPG_bias"] = True         # PPG-derived RRi
spain["ectopic_method"] = "NK2_PPG_clean"

spain = spain.merge(spain_mse[["Subject", "SampEn"]], on="Subject", how="left")

# ---------------------------------------------------------------------------
# Load Japan
# ---------------------------------------------------------------------------
japan_m = pd.read_csv(DATA / "japan_recalc_metrics.csv")
japan_mse = load_mse_scale1(DATA / "japan_afternoon_mse.csv")

japan = japan_m.rename(columns=NK2_MAP)
# Japan recalc DFA column differs
japan["DFA_alpha1"] = japan_m[JAPAN_DFA_COL]
japan["DFA_alpha2"] = np.nan     # Not available in recalc file
japan["SD1SD2"] = np.nan         # Not in recalc file
japan["SDANN"] = japan_m.get("HRV_SDANN5", np.nan)  # 24-h recording: valid
japan["Center"] = "Japan"
japan["Group"] = normalize_group(japan["Group"])
japan["PPG_bias"] = False        # ECG-derived
japan["ectopic_method"] = "Bounds300-2000ms"

japan = japan.merge(japan_mse[["Subject", "SampEn"]], on="Subject", how="left")

# ---------------------------------------------------------------------------
# Concatenate and finalise
# ---------------------------------------------------------------------------
combined = pd.concat([chile, spain, japan], ignore_index=True, sort=False)
combined = combined[OUTPUT_COLS]
combined = combined.reset_index(drop=True)

out_path = RESULTS / "traditional_hrv_metrics.csv"
combined.to_csv(out_path, index=False)
print(f"Saved {len(combined)} rows → {out_path}")

# ---------------------------------------------------------------------------
# Summary table: mean ± SD per metric per (Center × Group) + Mann-Whitney U
# ---------------------------------------------------------------------------
METRIC_COLS = [
    "SDNN", "RMSSD", "pNN50", "SDANN", "TINN",
    "SD1", "SD2", "SD1SD2",
    "VLF_power", "LF_power", "HF_power", "Total_power",
    "LF_HF", "LF_norm", "HF_norm",
    "DFA_alpha1", "DFA_alpha2", "SampEn",
]

def stars(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def bh_fdr(p_values: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (manual, works on any scipy version)."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = p[order] * n / (np.arange(1, n + 1))
    # Enforce monotonicity (cumulative min from the right)
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.clip(q, 0, 1)


rows = []
centers = ["Chile", "Spain", "Japan"]

for center in centers:
    cdf = combined[combined["Center"] == center]
    hc = cdf[cdf["Group"] == "Control"]
    pd_ = cdf[cdf["Group"] == "PD"]

    p_vals = []
    for col in METRIC_COLS:
        a = hc[col].dropna()
        b = pd_[col].dropna()
        if len(a) >= 3 and len(b) >= 3:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        else:
            p = np.nan
        p_vals.append(p)

    valid_mask = [not np.isnan(p) for p in p_vals]
    valid_p = [p for p, v in zip(p_vals, valid_mask) if v]
    q_vals_valid = bh_fdr(valid_p) if valid_p else []

    q_vals = []
    vi = 0
    for v in valid_mask:
        if v:
            q_vals.append(q_vals_valid[vi])
            vi += 1
        else:
            q_vals.append(np.nan)

    for col, p, q in zip(METRIC_COLS, p_vals, q_vals):
        a = hc[col].dropna()
        b = pd_[col].dropna()
        row = {
            "Center":    center,
            "Metric":    col,
            "HC_n":      len(a),
            "PD_n":      len(b),
            "HC_mean":   a.mean(),
            "HC_sd":     a.std(),
            "PD_mean":   b.mean(),
            "PD_sd":     b.std(),
            "MW_p":      p,
            "BH_q":      q,
            "sig":       stars(q) if not np.isnan(q) else "—",
        }
        rows.append(row)

summary = pd.DataFrame(rows)

# Print readable table
print("\n=== Mann-Whitney U summary (BH-FDR corrected) ===")
print(f"{'Center':<8} {'Metric':<12} {'HC mean±SD':>18} {'PD mean±SD':>18} {'p':>8} {'q':>8} {'sig':>5}")
print("-" * 82)
for _, r in summary.iterrows():
    hc_str = f"{r.HC_mean:.3f}±{r.HC_sd:.3f}" if not np.isnan(r.HC_mean) else "—"
    pd_str = f"{r.PD_mean:.3f}±{r.PD_sd:.3f}" if not np.isnan(r.PD_mean) else "—"
    p_str  = f"{r.MW_p:.4f}" if not np.isnan(r.MW_p) else "—"
    q_str  = f"{r.BH_q:.4f}" if not np.isnan(r.BH_q) else "—"
    print(f"{r.Center:<8} {r.Metric:<12} {hc_str:>18} {pd_str:>18} {p_str:>8} {q_str:>8} {r.sig:>5}")

summary_path = RESULTS / "traditional_hrv_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\nSummary saved → {summary_path}")

# ---------------------------------------------------------------------------
# Verification counts
# ---------------------------------------------------------------------------
print("\n=== Verification ===")
for center in centers:
    cdf = combined[combined["Center"] == center]
    ppg = cdf["PPG_bias"].iloc[0]
    sdann_ok = cdf["SDANN"].notna().sum()
    dfa2_ok  = cdf["DFA_alpha2"].notna().sum()
    sampen_ok = cdf["SampEn"].notna().sum()
    print(f"{center}: n={len(cdf)}  PPG_bias={ppg}  SDANN_populated={sdann_ok}"
          f"  DFA_alpha2_populated={dfa2_ok}  SampEn_populated={sampen_ok}")
