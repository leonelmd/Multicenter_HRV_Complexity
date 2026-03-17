#!/usr/bin/env python3
"""
Spain (Cruces) Complexity Significance Exploration
====================================================
Investigates why HR-normalized rcMSE nAUC fails to reach significance
in the Spain/Cruces PPG cohort, and tests alternative metrics.

Tests:
  1. Per-scale (1–5) AUC and Mann-Whitney U
  2. Raw nAUC(1–5) vs HR-normalized
  3. MSE curve slope (linear regression over scales 1–5)
  4. Scale-dropping: nAUC(2–5) — exclude scale 1 (most PPG-noise-affected)
  5. All traditional HRV metrics from spain_metrics.csv (AUC ranking)
  6. MFDFA, CD, HFD, KFD, LZC from spain_metrics.csv
  7. "Other" inclusion sensitivity: PD vs Control only vs PD vs (Control+Other)
  8. PPG modality context summary

Output: printed table + results/spain_exploration.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
RES_DIR  = os.path.join(BASE, 'results')
os.makedirs(RES_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
mse  = pd.read_csv(os.path.join(DATA_DIR, 'spain_mse.csv'))
metr = pd.read_csv(os.path.join(DATA_DIR, 'spain_metrics.csv'))
demo = pd.read_csv(os.path.join(DATA_DIR, 'spain_demographics.csv'))

# Merge metrics + demographics on Subject
if 'Subject' in metr.columns:
    metr = metr.merge(demo[['Subject', 'Group', 'Age']], on='Subject', how='left', suffixes=('', '_d'))
    if 'Group' not in metr.columns and 'Group_d' in metr.columns:
        metr['Group'] = metr['Group_d']
else:
    # metrics file may not have Subject; use demo order
    metr = metr.copy()
    metr['Group'] = demo['Group'].values[:len(metr)]
    metr['Age']   = demo['Age'].values[:len(metr)]

# Pivot MSE wide: one row per subject
mse_wide = mse.pivot_table(index=['Subject', 'Group'], columns='Scales', values='MSE').reset_index()
mse_wide.columns = ['Subject', 'Group'] + [f'S{int(c)}' for c in mse_wide.columns[2:]]

# Merge with demographics for HR
# spain_metrics has HRV_MeanNN → derive HR
if 'HRV_MeanNN' in metr.columns and 'Subject' in metr.columns:
    hr_df = metr[['Subject', 'HRV_MeanNN']].copy()
    hr_df['HR'] = 60000.0 / hr_df['HRV_MeanNN']
    mse_wide = mse_wide.merge(hr_df[['Subject', 'HR']], on='Subject', how='left')
else:
    mse_wide['HR'] = np.nan

# Scale columns available in short recording (reliable range: scales 1–5)
SCALE_COLS = [f'S{i}' for i in range(1, 6)]


# ── Helper functions ───────────────────────────────────────────────────────────

def mwu_auc(df, feature, include_other_as_control=True):
    """Mann-Whitney U test and AUC between PD and Control."""
    if include_other_as_control:
        ctrl = df[df['Group'].isin(['Control', 'Other'])][feature].dropna()
        pd_  = df[df['Group'] == 'PD'][feature].dropna()
    else:
        ctrl = df[df['Group'] == 'Control'][feature].dropna()
        pd_  = df[df['Group'] == 'PD'][feature].dropna()

    if len(ctrl) < 3 or len(pd_) < 3:
        return np.nan, np.nan, len(ctrl), len(pd_)

    stat, p = stats.mannwhitneyu(ctrl, pd_, alternative='two-sided')

    # AUC: label 1=Control, 0=PD (complexity higher in Control)
    vals  = np.concatenate([ctrl, pd_])
    labs  = np.concatenate([np.ones(len(ctrl)), np.zeros(len(pd_))])
    try:
        auc = roc_auc_score(labs, vals)
    except Exception:
        auc = np.nan

    return p, auc, len(ctrl), len(pd_)


def cohens_d(df, feature, include_other_as_control=True):
    if include_other_as_control:
        ctrl = df[df['Group'].isin(['Control', 'Other'])][feature].dropna()
        pd_  = df[df['Group'] == 'PD'][feature].dropna()
    else:
        ctrl = df[df['Group'] == 'Control'][feature].dropna()
        pd_  = df[df['Group'] == 'PD'][feature].dropna()
    pooled_sd = np.sqrt(((len(ctrl)-1)*ctrl.std()**2 + (len(pd_)-1)*pd_.std()**2) /
                        (len(ctrl)+len(pd_)-2))
    if pooled_sd == 0:
        return np.nan
    return (ctrl.mean() - pd_.mean()) / pooled_sd


# ── Section 1: Per-scale AUC and MWU (scales 1–5) ─────────────────────────────

print("\n" + "="*70)
print("SECTION 1: Per-scale Mann-Whitney U and AUC (Spain, scales 1–5)")
print("="*70)
print(f"{'Scale':<8}{'MeanCtrl':>10}{'MeanPD':>10}{'Delta':>8}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
print("-"*70)

scale_results = []
for s in range(1, 6):
    col = f'S{s}'
    ctrl_vals = mse_wide[mse_wide['Group'].isin(['Control','Other'])][col].dropna()
    pd_vals   = mse_wide[mse_wide['Group'] == 'PD'][col].dropna()
    p, auc, nc, np_ = mwu_auc(mse_wide, col)
    d = cohens_d(mse_wide, col)
    delta = ctrl_vals.mean() - pd_vals.mean()
    print(f"  S{s}     {ctrl_vals.mean():>10.4f}{pd_vals.mean():>10.4f}{delta:>8.4f}{p:>10.4f}{auc:>8.3f}{d:>10.3f}")
    scale_results.append({'Scale': s, 'Mean_Ctrl': ctrl_vals.mean(), 'Mean_PD': pd_vals.mean(),
                           'Delta': delta, 'p_val': p, 'AUC': auc, 'Cohens_d': d,
                           'N_Ctrl': nc, 'N_PD': np_})

# ── Section 2: nAUC variants ──────────────────────────────────────────────────

print("\n" + "="*70)
print("SECTION 2: nAUC variants — raw vs HR-normalized, scale subsets")
print("="*70)

# Compute trapz AUC over different scale ranges
for label, cols in [('nAUC(1-5)', [f'S{i}' for i in range(1,6)]),
                     ('nAUC(2-5)', [f'S{i}' for i in range(2,6)]),
                     ('nAUC(3-5)', [f'S{i}' for i in range(3,6)])]:
    available = [c for c in cols if c in mse_wide.columns]
    scales_x  = [int(c[1:]) for c in available]
    mse_wide[label] = mse_wide[available].apply(
        lambda row: np.trapz(row.values, x=scales_x), axis=1)
    mse_wide[f'{label}/HR'] = mse_wide[label] / mse_wide['HR']

print(f"\n{'Metric':<20}{'MeanCtrl':>10}{'MeanPD':>10}{'Delta':>8}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
print("-"*70)
nauc_results = []
for col in ['nAUC(1-5)', 'nAUC(1-5)/HR', 'nAUC(2-5)', 'nAUC(2-5)/HR',
             'nAUC(3-5)', 'nAUC(3-5)/HR']:
    ctrl_vals = mse_wide[mse_wide['Group'].isin(['Control','Other'])][col].dropna()
    pd_vals   = mse_wide[mse_wide['Group'] == 'PD'][col].dropna()
    p, auc, nc, np_ = mwu_auc(mse_wide, col)
    d = cohens_d(mse_wide, col)
    delta = ctrl_vals.mean() - pd_vals.mean()
    print(f"  {col:<18}{ctrl_vals.mean():>10.4f}{pd_vals.mean():>10.4f}{delta:>8.4f}{p:>10.4f}{auc:>8.3f}{d:>10.3f}")
    nauc_results.append({'Metric': col, 'Mean_Ctrl': ctrl_vals.mean(), 'Mean_PD': pd_vals.mean(),
                          'Delta': delta, 'p_val': p, 'AUC': auc, 'Cohens_d': d})

# ── Section 3: MSE curve slope (LRS-style) ────────────────────────────────────

print("\n" + "="*70)
print("SECTION 3: MSE curve slope (linear regression over scales 1–5)")
print("  Positive slope = complexity increases with scale (healthy)")
print("  Negative/flat = pathological rigidity")
print("="*70)

def mse_slope(row, cols, xs):
    vals = row[cols].values.astype(float)
    mask = ~np.isnan(vals)
    if mask.sum() < 3:
        return np.nan
    return np.polyfit(np.array(xs)[mask], vals[mask], 1)[0]

scale_xs   = list(range(1, 6))
mse_wide['MSE_slope_1_5'] = mse_wide.apply(lambda r: mse_slope(r, [f'S{i}' for i in range(1,6)], scale_xs), axis=1)
mse_wide['MSE_slope_2_5'] = mse_wide.apply(lambda r: mse_slope(r, [f'S{i}' for i in range(2,6)], list(range(2,6))), axis=1)
# Slope as: (S5 - S1) / 4
mse_wide['MSE_rise'] = mse_wide['S5'] - mse_wide['S1']

print(f"\n{'Metric':<22}{'MeanCtrl':>10}{'MeanPD':>10}{'Delta':>8}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
print("-"*70)
slope_results = []
for col in ['MSE_slope_1_5', 'MSE_slope_2_5', 'MSE_rise']:
    ctrl_vals = mse_wide[mse_wide['Group'].isin(['Control','Other'])][col].dropna()
    pd_vals   = mse_wide[mse_wide['Group'] == 'PD'][col].dropna()
    p, auc, nc, np_ = mwu_auc(mse_wide, col)
    d = cohens_d(mse_wide, col)
    delta = ctrl_vals.mean() - pd_vals.mean()
    print(f"  {col:<20}{ctrl_vals.mean():>10.4f}{pd_vals.mean():>10.4f}{delta:>8.4f}{p:>10.4f}{auc:>8.3f}{d:>10.3f}")
    slope_results.append({'Metric': col, 'Mean_Ctrl': ctrl_vals.mean(), 'Mean_PD': pd_vals.mean(),
                           'Delta': delta, 'p_val': p, 'AUC': auc, 'Cohens_d': d})

# ── Section 4: "Other" group sensitivity ─────────────────────────────────────

print("\n" + "="*70)
print("SECTION 4: 'Other' group sensitivity (nAUC(1-5)/HR)")
print("="*70)

for label, inc_other in [('Control+Other vs PD', True), ('Control-only vs PD', False)]:
    col = 'nAUC(1-5)/HR'
    ctrl_vals = (mse_wide[mse_wide['Group'].isin(['Control','Other'])][col].dropna()
                 if inc_other else mse_wide[mse_wide['Group'] == 'Control'][col].dropna())
    pd_vals   = mse_wide[mse_wide['Group'] == 'PD'][col].dropna()
    stat, p = stats.mannwhitneyu(ctrl_vals, pd_vals, alternative='two-sided')
    vals = np.concatenate([ctrl_vals, pd_vals])
    labs = np.concatenate([np.ones(len(ctrl_vals)), np.zeros(len(pd_vals))])
    auc  = roc_auc_score(labs, vals)
    print(f"  {label}: N_ctrl={len(ctrl_vals)}, N_PD={len(pd_vals)}, p={p:.4f}, AUC={auc:.3f}")

# ── Section 5: All traditional HRV metrics AUC ranking ───────────────────────

print("\n" + "="*70)
print("SECTION 5: Traditional HRV metrics AUC ranking (spain_metrics.csv)")
print("  Top 20 by AUC (Control+Other vs PD)")
print("="*70)

# Build combined dataframe
if 'Subject' in metr.columns:
    combined = mse_wide[['Subject','Group']].merge(metr.drop(columns=['Group'], errors='ignore'),
                                                     on='Subject', how='left')
else:
    combined = mse_wide[['Subject','Group']].copy()
    for c in metr.columns:
        if c not in combined.columns:
            combined[c] = metr[c].values[:len(combined)]

hrv_numeric_cols = [c for c in metr.columns
                    if c not in ['Subject', 'Group', 'Age', 'Gender', 'Center']
                    and pd.api.types.is_numeric_dtype(metr[c])]

hrv_results = []
for col in hrv_numeric_cols:
    if col not in combined.columns:
        combined[col] = metr[col].values[:len(combined)] if col in metr.columns else np.nan
    p, auc, nc, np_ = mwu_auc(combined, col)
    d = cohens_d(combined, col)
    hrv_results.append({'Metric': col, 'p_val': p, 'AUC': auc, 'Cohens_d': d, 'N_Ctrl': nc, 'N_PD': np_})

hrv_df = pd.DataFrame(hrv_results).dropna(subset=['AUC']).sort_values('AUC', ascending=False)
print(f"\n{'Metric':<35}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
print("-"*65)
for _, row in hrv_df.head(25).iterrows():
    marker = '  ***' if row['p_val'] < 0.001 else ('  **' if row['p_val'] < 0.01 else
             ('  *' if row['p_val'] < 0.05 else ''))
    print(f"  {row['Metric']:<33}{row['p_val']:>10.4f}{row['AUC']:>8.3f}{row['Cohens_d']:>10.3f}{marker}")

# Also show bottom 5 (reverse direction: PD > Control)
print("\n  ... Bottom 5 (PD > Control direction):")
for _, row in hrv_df.tail(5).iterrows():
    print(f"  {row['Metric']:<33}{row['p_val']:>10.4f}{row['AUC']:>8.3f}{row['Cohens_d']:>10.3f}")

# ── Section 6: Nonlinear metrics spotlight ────────────────────────────────────

print("\n" + "="*70)
print("SECTION 6: Nonlinear complexity metrics (CD, HFD, KFD, LZC, MFDFA)")
print("="*70)

nonlin_cols = [c for c in hrv_numeric_cols
               if any(k in c.upper() for k in ['LZC','CD','HFD','KFD','MFDFA','CORR','PERM','APP','SAMP'])]
nonlin_df = hrv_df[hrv_df['Metric'].isin(nonlin_cols)].sort_values('AUC', ascending=False)
print(f"\n{'Metric':<40}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
print("-"*70)
for _, row in nonlin_df.iterrows():
    marker = '  ***' if row['p_val'] < 0.001 else ('  **' if row['p_val'] < 0.01 else
             ('  *' if row['p_val'] < 0.05 else ''))
    print(f"  {row['Metric']:<38}{row['p_val']:>10.4f}{row['AUC']:>8.3f}{row['Cohens_d']:>10.3f}{marker}")

# ── Section 7: PPG vs ECG — scale-1 noise hypothesis ─────────────────────────

print("\n" + "="*70)
print("SECTION 7: PPG modality — scale-1 noise hypothesis")
print("  If PPG adds high-freq noise, scale-1 entropy is artificially elevated")
print("  in ALL subjects, masking group differences. Prediction:")
print("  - S1 should show SMALLEST group difference (noise-dominated)")
print("  - S2-S5 should show LARGER/cleaner group difference")
print("="*70)

print(f"\n  Scale-1 absolute SampEn values:")
ctrl_s1 = mse_wide[mse_wide['Group'].isin(['Control','Other'])]['S1'].dropna()
pd_s1   = mse_wide[mse_wide['Group'] == 'PD']['S1'].dropna()
print(f"    Control+Other: {ctrl_s1.mean():.4f} ± {ctrl_s1.std():.4f}  (n={len(ctrl_s1)})")
print(f"    PD:            {pd_s1.mean():.4f} ± {pd_s1.std():.4f}  (n={len(pd_s1)})")
print(f"    Delta (Ctrl-PD): {ctrl_s1.mean()-pd_s1.mean():.4f}")

print(f"\n  Compare to Chile (ECG) scale-1:")
try:
    chile_mse = pd.read_csv(os.path.join(DATA_DIR, 'chile_mse.csv'))
    chile_wide = chile_mse.pivot_table(index=['Subject', 'Group'], columns='Scales', values='MSE').reset_index()
    chile_wide.columns = ['Subject', 'Group'] + [f'S{int(c)}' for c in chile_wide.columns[2:]]
    ctrl_c = chile_wide[chile_wide['Group'] == 'Control']['S1'].dropna()
    pd_c   = chile_wide[chile_wide['Group'] == 'PD']['S1'].dropna()
    print(f"    Chile Control: {ctrl_c.mean():.4f} ± {ctrl_c.std():.4f}  (n={len(ctrl_c)})")
    print(f"    Chile PD:      {pd_c.mean():.4f} ± {pd_c.std():.4f}  (n={len(pd_c)})")
    print(f"    Chile Delta:   {ctrl_c.mean()-pd_c.mean():.4f}")
    p_chile, auc_chile, _, _ = mwu_auc(chile_wide, 'S1')
    p_spain, auc_spain, _, _ = mwu_auc(mse_wide, 'S1')
    print(f"\n  S1 AUC: Chile={auc_chile:.3f} (p={p_chile:.4f})  |  Spain={auc_spain:.3f} (p={p_spain:.4f})")
    print(f"\n  >>> {'Scale-1 higher in Spain (PPG noise elevates baseline entropy)' if ctrl_s1.mean() > ctrl_c.mean() else 'Scale-1 similar or lower in Spain'}")
    ppg_diff = ctrl_s1.mean() - ctrl_c.mean()
    print(f"  >>> Spain S1 mean - Chile S1 mean = {ppg_diff:+.4f}")
except Exception as e:
    print(f"  Could not load Chile data: {e}")

# ── Section 8: Correlation between scale-1 and group (noise masking) ─────────

print("\n" + "="*70)
print("SECTION 8: Scale correlations to detect noise masking")
print("  If S1 is noise-dominated, it correlates less with other scales")
print("  and less with vagal metrics (RMSSD, pNN50)")
print("="*70)

if 'HRV_RMSSD' in combined.columns:
    scale_metric_corr = {}
    for s in range(1, 6):
        col = f'S{s}'
        if col not in combined.columns:
            # merge from mse_wide
            combined = combined.merge(mse_wide[['Subject', col]], on='Subject', how='left', suffixes=('','_mse'))
        sub = combined[['Subject', col, 'HRV_RMSSD', 'HRV_pNN50']].dropna()
        r_rmssd, p_rmssd = stats.spearmanr(sub[col], sub['HRV_RMSSD'])
        r_pnn50, p_pnn50 = stats.spearmanr(sub[col], sub['HRV_pNN50'])
        print(f"  S{s}: ρ(RMSSD)={r_rmssd:+.3f} (p={p_rmssd:.4f})   ρ(pNN50)={r_pnn50:+.3f} (p={p_pnn50:.4f})")
        scale_metric_corr[s] = {'rho_RMSSD': r_rmssd, 'p_RMSSD': p_rmssd,
                                  'rho_pNN50': r_pnn50, 'p_pNN50': p_pnn50}
else:
    print("  HRV_RMSSD not available in combined dataframe")

# ── Section 9: Signal length sensitivity ─────────────────────────────────────

print("\n" + "="*70)
print("SECTION 9: Signal length sensitivity")
print("  Spain has variable recording length (5–15 min).")
print("  Check if shorter recordings produce different nAUC")
print("="*70)

if 'HRV_MeanNN' in combined.columns and 'HRV_SDNN' in combined.columns:
    # Use n_beats as proxy (not directly available; approximate from MeanNN)
    # If we have subject-level data, check correlation of complexity with signal quality proxies
    print("  Note: Signal length not directly stored in spain_metrics.csv")
    print("  Proxy: check if SDNN (overall variability) correlates with complexity in Spain")
    sub = combined[['S1', 'nAUC(1-5)/HR', 'HRV_SDNN']].dropna() if 'nAUC(1-5)/HR' in combined.columns else pd.DataFrame()
    if not sub.empty and len(sub) > 5:
        r, p = stats.spearmanr(sub['nAUC(1-5)/HR'], sub['HRV_SDNN'])
        print(f"  ρ(nAUC/HR, SDNN) in Spain pooled: {r:+.3f} (p={p:.4f})")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY — Best discriminators in Spain (all metrics, AUC > 0.62)")
print("="*70)

all_results = []

# Per-scale MSE
for r in scale_results:
    all_results.append({'Category': 'MSE per-scale', 'Metric': f"SampEn S{r['Scale']}",
                         'p_val': r['p_val'], 'AUC': r['AUC'], 'Cohens_d': r['Cohens_d']})

# nAUC variants
for r in nauc_results:
    all_results.append({'Category': 'nAUC variants', 'Metric': r['Metric'],
                         'p_val': r['p_val'], 'AUC': r['AUC'], 'Cohens_d': r['Cohens_d']})

# Slope
for r in slope_results:
    all_results.append({'Category': 'MSE slope', 'Metric': r['Metric'],
                         'p_val': r['p_val'], 'AUC': r['AUC'], 'Cohens_d': r['Cohens_d']})

# HRV top
for _, row in hrv_df.head(30).iterrows():
    all_results.append({'Category': 'Traditional HRV', 'Metric': row['Metric'],
                         'p_val': row['p_val'], 'AUC': row['AUC'], 'Cohens_d': row['Cohens_d']})

all_df = pd.DataFrame(all_results).sort_values('AUC', ascending=False)
significant = all_df[all_df['p_val'] < 0.05]

print(f"\n  Significant metrics (p < 0.05): {len(significant)}")
if len(significant) > 0:
    print(f"\n  {'Category':<20}{'Metric':<35}{'p-val':>10}{'AUC':>8}{'Cohen_d':>10}")
    print("  " + "-"*83)
    for _, row in significant.sort_values('AUC', ascending=False).iterrows():
        print(f"  {row['Category']:<20}{row['Metric']:<35}{row['p_val']:>10.4f}{row['AUC']:>8.3f}{row['Cohens_d']:>10.3f}")

best_mse = all_df[all_df['Category'].isin(['MSE per-scale','nAUC variants','MSE slope'])].sort_values('AUC', ascending=False).head(1)
if not best_mse.empty:
    print(f"\n  Best MSE-based metric: {best_mse.iloc[0]['Metric']} — AUC={best_mse.iloc[0]['AUC']:.3f}, p={best_mse.iloc[0]['p_val']:.4f}")

print(f"\n  Total metrics tested: {len(all_df)}")

# Save
all_df.to_csv(os.path.join(RES_DIR, 'spain_exploration.csv'), index=False)
print(f"\n  Full results saved → results/spain_exploration.csv")
print()
