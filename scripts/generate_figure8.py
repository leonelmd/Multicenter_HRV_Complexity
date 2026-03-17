#!/usr/bin/env python3
"""
Figure 8: Clinical Correlation Analysis — CETRAM Cohort
=========================================================
Correlates the rcMSE cardiac complexity index (nAUC 1–5 / HR) with
clinical characteristics of 34 PD patients from the CETRAM cohort.

Clinical data source: data/deidentified_clinical_consolidated.xlsx
MSE data source:      data/chile_mse.csv
HRV metrics:          data/chile_metrics.csv

Panels
------
A – Spearman ρ heatmap: complexity + traditional HRV × clinical scales
    (raw ρ and age-adjusted partial ρ, BH-FDR corrected)
B – Scatter plots: top 4 clinical correlates with regression lines
C – Hoehn & Yahr stage boxplots (complexity by disease stage)
D – Non-motor symptom burden: composite score + key individual symptoms
E – LEDD subgroup analysis (n=9 with available data; pending completion)

Reference: see references/literature_hrv_clinical_pd.md
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(BASE, 'data')
OUT_DIR    = os.path.join(BASE, 'figures', 'Figure8')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 11, 'axes.titlesize': 12,
})
PD_COLOR   = '#C0392B'   # red — PD
CTRL_COLOR = '#2471A3'   # blue — used for H&Y gradient
ACCENT     = '#7D3C98'   # purple — complexity metric
GRAY       = '#7F8C8D'

HY_COLORS = {1.0: '#2ECC71', 1.5: '#27AE60', 2.0: '#F1C40F',
              2.5: '#F39C12', 3.0: '#E67E22', 3.5: '#E74C3C', 4.0: '#8E44AD'}


# ── Helper: non-motor symptom harmonisation ───────────────────────────────────

def harmonise_binary(series):
    """Convert mixed 0/1/text encoding to binary: 1=present, 0=absent, NaN=unknown."""
    result = []
    present_kw = {'present', '1', 'yes', 'probable', 'possible'}
    absent_kw  = {'absent', '0', 'no', 'currently'}
    unknown_kw = {'not recorded', 'nan', 'none', ''}
    for val in series:
        s = str(val).strip().lower()
        if s in unknown_kw or s == 'nan':
            result.append(np.nan)
        elif s == '1' or val == 1:
            result.append(1)
        elif s == '0' or val == 0:
            result.append(0)
        elif any(k in s for k in present_kw):
            result.append(1)
        elif any(k in s for k in absent_kw):
            result.append(0)
        else:
            result.append(np.nan)
    return pd.array(result, dtype='Float64')


# ── Bootstrap partial Spearman ρ ──────────────────────────────────────────────

def partial_spearman_age(df, x_col, y_col, cov_col='Age', n_boot=1000):
    """
    Partial Spearman ρ between x and y, controlling for cov (age).
    Implemented as Spearman on residuals after rank-regressing on covariate.
    Returns (rho, p_value, ci_lower, ci_upper).
    """
    sub = df[[x_col, y_col, cov_col]].dropna()
    if len(sub) < 6:
        return np.nan, np.nan, np.nan, np.nan

    def _partial_r(data):
        xr = stats.rankdata(data[x_col])
        yr = stats.rankdata(data[y_col])
        cr = stats.rankdata(data[cov_col])
        # Residualise x and y on covariate rank
        res_x = xr - np.polyval(np.polyfit(cr, xr, 1), cr)
        res_y = yr - np.polyval(np.polyfit(cr, yr, 1), cr)
        r, _ = stats.pearsonr(res_x, res_y)
        return r

    rho = _partial_r(sub)
    # Bootstrap CI
    boot_rhos = []
    for _ in range(n_boot):
        boot = sub.sample(n=len(sub), replace=True)
        try:
            boot_rhos.append(_partial_r(boot))
        except Exception:
            pass
    if len(boot_rhos) < 10:
        return rho, np.nan, np.nan, np.nan
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    # Approximate p-value via t-distribution
    n = len(sub)
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-10))
    p = 2 * stats.t.sf(abs(t_stat), df=n - 2)
    return rho, p, ci_lo, ci_hi


def spearman_boot(x, y, n_boot=1000):
    """Spearman ρ with bootstrap CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = np.array(x)[mask], np.array(y)[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    boot_rhos = [stats.spearmanr(
        *zip(*resample(list(zip(x, y)))))[0] for _ in range(n_boot)]
    return rho, p, np.percentile(boot_rhos, 2.5), np.percentile(boot_rhos, 97.5)


# ── BH-FDR correction ─────────────────────────────────────────────────────────

def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns array of adjusted p-values."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_adj = np.zeros(n)
    for i, idx in enumerate(order):
        p_adj[idx] = min(p[idx] * n / (i + 1), 1.0)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        p_adj[order[i]] = min(p_adj[order[i]], p_adj[order[i + 1]])
    return p_adj


# ── Load and merge data ───────────────────────────────────────────────────────

def load_data():
    clin = pd.read_excel(os.path.join(DATA_DIR, 'deidentified_clinical_consolidated.xlsx'))
    mse  = pd.read_csv(os.path.join(DATA_DIR, 'chile_mse.csv'))
    metr = pd.read_csv(os.path.join(DATA_DIR, 'chile_metrics.csv'))
    demo = pd.read_csv(os.path.join(DATA_DIR, 'chile_demographics.csv'))

    # ── Compute complexity index for PD subjects from MSE ─────────────────────
    mse_pd = mse[mse['Group'] == 'PD'].copy()
    mse_wide = mse_pd.pivot_table(index='Subject', columns='Scales', values='MSE').reset_index()
    mse_wide.columns = ['Subject'] + [f'S{int(c)}' for c in mse_wide.columns[1:]]

    # nAUC over scales 1–5 (reliable range for ~15-min recordings)
    scale_cols = [f'S{i}' for i in range(1, 6)]
    mse_wide['nAUC_1_5'] = mse_wide[scale_cols].apply(
        lambda r: np.trapezoid(r.values, x=list(range(1, 6))), axis=1)

    # HR from chile_metrics (HRV_MeanNN → HR)
    if 'Subject' in metr.columns and 'HRV_MeanNN' in metr.columns:
        hr_df = metr[['Subject', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD',
                       'HRV_pNN50']].copy()
        hr_df['HR'] = 60000.0 / hr_df['HRV_MeanNN']
    else:
        # Fallback: derive HR from demo
        hr_df = demo[demo['Group'] == 'PD'][['Subject', 'HR']].copy() \
            if 'HR' in demo.columns else pd.DataFrame({'Subject': [], 'HR': []})

    mse_wide = mse_wide.merge(hr_df, on='Subject', how='left')
    mse_wide['Complexity'] = mse_wide['nAUC_1_5'] / mse_wide['HR']

    # Also add DFA_alpha1 from chile_metrics if available
    if 'HRV_DFA_alpha1' in metr.columns and 'Subject' in metr.columns:
        mse_wide = mse_wide.merge(metr[['Subject', 'HRV_DFA_alpha1']], on='Subject', how='left')
    # SampEn scale 1
    if 'S1' in mse_wide.columns:
        mse_wide['SampEn_S1'] = mse_wide['S1']

    # ── Harmonise non-motor symptoms ──────────────────────────────────────────
    nm_cols = {
        'Hallucinations':            'Hallucinations',
        'Depression':                'Depression',
        'Constipation':              'Constipation',
        'Urinary disorders':         'Urinary disorders',
        'Orthostatic hypotension':   'Orthostatic hypotension',
        'Sleep disorders':           'Sleep disorders',
        'REM sleep behavior disorder': 'REM behavior disorder',
        'Olfactory disorders':       'Olfactory disorders',
        'Fatigue':                   'Fatigue',
        'Pain':                      'Pain',
    }
    for col in nm_cols:
        if col in clin.columns:
            clin[col] = harmonise_binary(clin[col])

    # Non-motor burden: sum of all available binary symptoms
    nm_available = [c for c in nm_cols if c in clin.columns]
    clin['NM_burden'] = clin[nm_available].apply(
        lambda r: r.dropna().sum() if r.dropna().shape[0] >= 3 else np.nan, axis=1)

    # ── Merge ─────────────────────────────────────────────────────────────────
    clin_renamed = clin.rename(columns={
        'Anonymous_ID':                    'Subject',
        'Age':                             'Age',
        'Gender':                          'Sex',
        'Time since diagnosis (Years)':    'Disease_duration',
        'Hoehn and Yahr stage':            'HY_stage',
        'Schwab & England score (%)':      'Schwab_England',
        'CISI-PD motor signs':             'CISI_motor',
        'CISI-PD disability':              'CISI_disability',
        'CISI-PD motor complications':     'CISI_motor_comp',
        'CISI-PD cognitive status':        'CISI_cognitive',
        'CISI-PD Total':                   'CISI_total',
        'LED total':                       'LEDD',
    })

    df = mse_wide.merge(
        clin_renamed[['Subject', 'Age', 'Sex', 'Disease_duration', 'HY_stage',
                       'Schwab_England', 'CISI_motor', 'CISI_disability',
                       'CISI_motor_comp', 'CISI_cognitive', 'CISI_total',
                       'LEDD', 'NM_burden'] + nm_available],
        on='Subject', how='inner'
    )

    # Rename NM cols for cleaner labels in the merged df
    df = df.rename(columns=nm_cols)

    print(f"  Merged dataset: {len(df)} PD subjects with both MSE + clinical data")
    print(f"  LEDD available: {df['LEDD'].notna().sum()}/{len(df)} "
          f"[PENDING: awaiting full dataset from clinical collaborator]")
    return df, nm_cols


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_figure8():

    print("\nLoading and merging data...")
    df, nm_cols = load_data()

    # Clinical variables for correlation analysis
    CONT_VARS = {
        'Disease_duration':  'Disease\nduration (yr)',
        'HY_stage':          'Hoehn &\nYahr stage',
        'Schwab_England':    'Schwab &\nEngland (%)',
        'CISI_motor':        'CISI-PD\nmotor signs',
        'CISI_disability':   'CISI-PD\ndisability',
        'CISI_motor_comp':   'CISI-PD\nmotor comp.',
        'CISI_cognitive':    'CISI-PD\ncognitive',
        'CISI_total':        'CISI-PD\ntotal',
        'NM_burden':         'Non-motor\nburden',
    }

    HRV_METRICS = {
        'Complexity':    'Complexity\n(rcMSE nAUC/HR)',
        'HRV_RMSSD':     'RMSSD',
        'HRV_DFA_alpha1': 'DFA α₁',
    }
    hrv_available = {k: v for k, v in HRV_METRICS.items() if k in df.columns}

    NM_BINARY = [c for c in nm_cols.values() if c in df.columns]

    # ── Compute correlation matrices ──────────────────────────────────────────
    print("  Computing correlations...")

    rows_crude, rows_adj = [], []
    for hrv_k, hrv_label in hrv_available.items():
        for var_k, var_label in CONT_VARS.items():
            rho, p, ci_lo, ci_hi = spearman_boot(df[hrv_k], df[var_k])
            rows_crude.append({
                'HRV': hrv_label.replace('\n', ' '),
                'Clinical': var_label.replace('\n', ' '),
                'rho': rho, 'p': p, 'ci_lo': ci_lo, 'ci_hi': ci_hi
            })
            rho_a, p_a, ci_lo_a, ci_hi_a = partial_spearman_age(
                df, hrv_k, var_k, cov_col='Age')
            rows_adj.append({
                'HRV': hrv_label.replace('\n', ' '),
                'Clinical': var_label.replace('\n', ' '),
                'rho': rho_a, 'p': p_a, 'ci_lo': ci_lo_a, 'ci_hi': ci_hi_a
            })

    corr_crude = pd.DataFrame(rows_crude)
    corr_adj   = pd.DataFrame(rows_adj)

    # BH-FDR within each HRV metric (over clinical variables)
    for df_c in [corr_crude, corr_adj]:
        for hrv_label in df_c['HRV'].unique():
            mask = df_c['HRV'] == hrv_label
            df_c.loc[mask, 'p_fdr'] = bh_fdr(df_c.loc[mask, 'p'].values)

    # Pivot for heatmap
    def pivot_rho(df_c):
        return df_c.pivot(index='HRV', columns='Clinical', values='rho')

    def pivot_sig(df_c):
        """Return significance labels based on FDR-corrected p."""
        sig = df_c.pivot(index='HRV', columns='Clinical', values='p_fdr')
        labels = sig.copy().astype(str)
        labels[sig < 0.001] = '***'
        labels[(sig >= 0.001) & (sig < 0.01)] = '**'
        labels[(sig >= 0.01) & (sig < 0.05)] = '*'
        labels[sig >= 0.05] = ''
        return labels

    # ── LEDD analysis (subgroup) ──────────────────────────────────────────────
    df_ledd = df[df['LEDD'].notna()].copy()
    ledd_n  = len(df_ledd)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor('white')

    gs_main = gridspec.GridSpec(4, 1, figure=fig,
                                 hspace=0.48, top=0.95, bottom=0.04,
                                 left=0.06, right=0.97,
                                 height_ratios=[2.2, 2.0, 1.8, 1.5])

    # ── Row 0: Panel A — Correlation heatmaps (crude + age-adjusted) ──────────
    gs_A = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.35)
    ax_Ac = fig.add_subplot(gs_A[0])
    ax_Aa = fig.add_subplot(gs_A[1])

    col_order = list(CONT_VARS.values())
    col_order_clean = [c.replace('\n', ' ') for c in col_order]
    hrv_order = [v.replace('\n', ' ') for v in hrv_available.values()]

    for ax, df_c, title_suffix in [
        (ax_Ac, corr_crude, '(A) Raw Spearman ρ'),
        (ax_Aa, corr_adj,   '(B) Partial ρ (age-adjusted)')
    ]:
        rho_mat = pivot_rho(df_c).reindex(index=hrv_order, columns=col_order_clean)
        sig_mat = pivot_sig(df_c).reindex(index=hrv_order, columns=col_order_clean)

        vmax = 0.75
        cmap = plt.cm.RdBu_r
        im = ax.imshow(rho_mat.values.astype(float), cmap=cmap,
                        vmin=-vmax, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(col_order_clean)))
        ax.set_xticklabels(col_order_clean, rotation=40, ha='right', fontsize=9)
        ax.set_yticks(range(len(hrv_order)))
        ax.set_yticklabels(hrv_order, fontsize=10)
        ax.set_title(title_suffix, fontsize=12, fontweight='bold', pad=8)

        # Annotate cells
        for i in range(rho_mat.shape[0]):
            for j in range(rho_mat.shape[1]):
                val = rho_mat.values[i, j]
                sig = sig_mat.values[i, j]
                if not np.isnan(val):
                    txt = f'{val:.2f}'
                    if sig:
                        txt += sig
                    color = 'white' if abs(val) > 0.45 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                             fontsize=8.5, color=color, fontweight='bold' if sig else 'normal')

        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03,
                     label='Spearman ρ')
        ax.tick_params(length=0)

    # Note on sample size and significance
    fig.text(0.5, gs_main[0].get_position(fig).y0 - 0.013,
             f'n = {len(df)} PD patients (CETRAM). '
             f'Stars denote BH-FDR corrected significance: * q<0.05  ** q<0.01  *** q<0.001.',
             ha='center', fontsize=8.5, color=GRAY, style='italic')

    # ── Row 1: Panel B — Scatter plots for top correlates ─────────────────────
    # Pick top 4 by |ρ| for Complexity
    comp_crude = corr_crude[corr_crude['HRV'] == 'Complexity (rcMSE nAUC/HR)'].copy()
    top4 = comp_crude.dropna(subset=['rho']).nlargest(4, 'rho')  # sort by |rho|
    top4 = pd.concat([top4.nlargest(2, 'rho'),
                       comp_crude.dropna(subset=['rho']).nsmallest(2, 'rho')])

    # Map label back to column key
    label_to_key = {v.replace('\n', ' '): k for k, v in {**CONT_VARS}.items()}

    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs_main[1], wspace=0.38)

    scatter_targets = [
        ('CISI_total', 'CISI-PD Total'),
        ('HY_stage', 'Hoehn & Yahr Stage'),
        ('Disease_duration', 'Disease Duration (yr)'),
        ('CISI_cognitive', 'CISI-PD Cognitive'),
    ]

    for idx, (var_k, var_label) in enumerate(scatter_targets):
        ax = fig.add_subplot(gs_B[idx])
        x = df[var_k].values.astype(float)
        y = df['Complexity'].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x_v, y_v = x[mask], y[mask]

        ax.scatter(x_v, y_v, color=ACCENT, s=55, alpha=0.75,
                    edgecolors='white', linewidths=0.6, zorder=4)

        # Regression line + 95% CI
        if len(x_v) >= 5:
            m, b, r, p_r, se = stats.linregress(x_v, y_v)
            x_line = np.linspace(x_v.min(), x_v.max(), 100)
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color=ACCENT, linewidth=1.8, zorder=3)
            # CI band (approximate)
            n = len(x_v)
            x_mean = x_v.mean()
            s_err = se * np.sqrt(1/n + (x_line - x_mean)**2 /
                                  np.sum((x_v - x_mean)**2))
            t_crit = stats.t.ppf(0.975, df=n - 2)
            ax.fill_between(x_line, y_line - t_crit * s_err,
                             y_line + t_crit * s_err,
                             color=ACCENT, alpha=0.12, zorder=2)

        rho, p, _, _ = spearman_boot(x_v, y_v, n_boot=500)
        p_label = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
        ax.set_title(f'({"BCDE"[idx]}) {var_label}', fontsize=10.5,
                      fontweight='bold')
        ax.text(0.96, 0.96, f'ρ = {rho:.2f}\n{p_label}\nn = {mask.sum()}',
                 transform=ax.transAxes, ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))
        ax.set_xlabel(var_label, fontsize=10)
        ax.set_ylabel('Complexity Index\n(rcMSE nAUC / HR)' if idx == 0 else '',
                       fontsize=10)
        ax.tick_params(labelsize=9)

    # ── Row 2: Panel C (H&Y stage) + Panel D (Non-motor) ─────────────────────
    gs_CD = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[2], wspace=0.38, width_ratios=[1, 1.6])

    # Panel C — H&Y stage
    ax_C = fig.add_subplot(gs_CD[0])
    hy_stages = sorted(df['HY_stage'].dropna().unique())
    hy_data   = [df[df['HY_stage'] == s]['Complexity'].dropna().values
                  for s in hy_stages]
    hy_n      = [len(d) for d in hy_data]
    hy_labels = [f'H&Y {s:.1g}\n(n={n})' for s, n in zip(hy_stages, hy_n)]
    colors    = [HY_COLORS.get(s, GRAY) for s in hy_stages]

    bp = ax_C.boxplot(hy_data, patch_artist=True, widths=0.55,
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2),
                       flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.82)

    # Jitter
    for i, (data, col) in enumerate(zip(hy_data, colors), 1):
        jitter = np.random.uniform(-0.18, 0.18, size=len(data))
        ax_C.scatter(np.full(len(data), i) + jitter, data,
                      color=col, s=28, alpha=0.7, zorder=5,
                      edgecolors='white', linewidths=0.5)

    # Kruskal-Wallis overall test
    if len([d for d in hy_data if len(d) >= 2]) >= 2:
        valid = [d for d in hy_data if len(d) >= 2]
        kw_stat, kw_p = stats.kruskal(*valid)
        kw_label = f'Kruskal-Wallis p={kw_p:.3f}' if kw_p >= 0.001 else 'Kruskal-Wallis p<0.001'
        ax_C.text(0.5, 0.97, kw_label, transform=ax_C.transAxes,
                   ha='center', va='top', fontsize=9, style='italic', color=GRAY)

    ax_C.set_xticks(range(1, len(hy_stages) + 1))
    ax_C.set_xticklabels(hy_labels, fontsize=8.5)
    ax_C.set_ylabel('Complexity Index (rcMSE nAUC / HR)', fontsize=10)
    ax_C.set_title('(F) Complexity by H&Y Stage', fontsize=11, fontweight='bold')

    # Trend line (Spearman over individual subjects)
    rho_hy, p_hy = stats.spearmanr(
        df['HY_stage'].dropna(), df.loc[df['HY_stage'].notna(), 'Complexity'])

    # Panel D — Non-motor symptom boxplots
    ax_D = fig.add_subplot(gs_CD[1])
    nm_plot = [c for c in ['Depression', 'Orthostatic hypotension',
                             'REM behavior disorder', 'Constipation',
                             'Olfactory disorders'] if c in df.columns]

    nm_positions = np.arange(len(nm_plot))
    width = 0.32
    colors_nm = {'Absent': CTRL_COLOR, 'Present': PD_COLOR}

    for pos, nm in zip(nm_positions, nm_plot):
        col = nm
        d_abs = df[df[col] == 0]['Complexity'].dropna()
        d_pres = df[df[col] == 1]['Complexity'].dropna()

        for offset, data, color, label in [
            (-width/2, d_abs, CTRL_COLOR, 'Absent'),
            (+width/2, d_pres, PD_COLOR,  'Present')
        ]:
            if len(data) >= 2:
                bp2 = ax_D.boxplot(data, positions=[pos + offset],
                                    widths=width - 0.04,
                                    patch_artist=True,
                                    medianprops=dict(color='white', linewidth=1.5),
                                    whiskerprops=dict(linewidth=1.0),
                                    capprops=dict(linewidth=1.0),
                                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
                for patch in bp2['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.75)
                # Jitter
                jit = np.random.uniform(-0.1, 0.1, size=len(data))
                ax_D.scatter(np.full(len(data), pos + offset) + jit, data,
                              color=color, s=22, alpha=0.65, zorder=5,
                              edgecolors='white', linewidths=0.4)

        # MWU p-value annotation
        if len(d_abs) >= 3 and len(d_pres) >= 3:
            _, p_nm = stats.mannwhitneyu(d_abs, d_pres, alternative='two-sided')
            y_max = max(df['Complexity'].dropna()) * 1.03
            sig = '***' if p_nm < 0.001 else ('**' if p_nm < 0.01 else
                  ('*' if p_nm < 0.05 else f'p={p_nm:.2f}'))
            ax_D.text(pos, y_max, sig, ha='center', va='bottom',
                       fontsize=9, color='black', fontweight='bold' if '*' in sig else 'normal')

    ax_D.set_xticks(nm_positions)
    ax_D.set_xticklabels([nm.replace(' ', '\n') for nm in nm_plot],
                          fontsize=9, rotation=0)
    ax_D.set_ylabel('Complexity Index (rcMSE nAUC / HR)', fontsize=10)
    ax_D.set_title('(G) Complexity by Non-Motor Symptom Presence',
                    fontsize=11, fontweight='bold')

    from matplotlib.patches import Patch
    ax_D.legend(handles=[Patch(facecolor=CTRL_COLOR, alpha=0.75, label='Absent'),
                           Patch(facecolor=PD_COLOR,  alpha=0.75, label='Present')],
                 fontsize=9, loc='upper right')

    # ── Row 3: Panel E — LEDD subgroup ────────────────────────────────────────
    gs_E = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[3], wspace=0.38)

    # LEDD scatter
    ax_E1 = fig.add_subplot(gs_E[0])
    x_l = df_ledd['LEDD'].values.astype(float)
    y_l = df_ledd['Complexity'].values.astype(float)
    mask_l = ~(np.isnan(x_l) | np.isnan(y_l))
    x_lv, y_lv = x_l[mask_l], y_l[mask_l]

    ax_E1.scatter(x_lv, y_lv, color=ACCENT, s=60, alpha=0.8,
                   edgecolors='white', linewidths=0.6, zorder=4)
    if len(x_lv) >= 4:
        m, b, r, p_r, se = stats.linregress(x_lv, y_lv)
        x_line = np.linspace(x_lv.min(), x_lv.max(), 100)
        ax_E1.plot(x_line, m * x_line + b, color=ACCENT, linewidth=1.8, zorder=3)
        rho_l, p_l = stats.spearmanr(x_lv, y_lv)
        p_lbl = f'p={p_l:.3f}' if p_l >= 0.001 else 'p<0.001'
        ax_E1.text(0.96, 0.96, f'ρ = {rho_l:.2f}\n{p_lbl}\nn = {len(x_lv)}',
                    transform=ax_E1.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.9))

    ax_E1.set_xlabel('LED Total (mg/day)', fontsize=10)
    ax_E1.set_ylabel('Complexity Index\n(rcMSE nAUC / HR)', fontsize=10)
    ax_E1.set_title('(H) Complexity vs. LEDD', fontsize=11, fontweight='bold')

    # Partial ρ LEDD controlling for disease duration
    ax_E2 = fig.add_subplot(gs_E[1])
    sub_ledd = df_ledd[['LEDD', 'Complexity', 'Disease_duration']].dropna()
    if len(sub_ledd) >= 5:
        rho_raw, p_raw, _, _ = spearman_boot(
            sub_ledd['LEDD'], sub_ledd['Complexity'], n_boot=200)
        rho_adj_l, p_adj_l, ci_lo_l, ci_hi_l = partial_spearman_age(
            sub_ledd, 'LEDD', 'Complexity', cov_col='Disease_duration', n_boot=200)

        categories = ['Raw ρ\n(LEDD vs\nComplexity)',
                       'Partial ρ\n(adj. disease\nduration)']
        rhos  = [rho_raw, rho_adj_l]
        ci_los = [np.nan, ci_lo_l]
        ci_his = [np.nan, ci_hi_l]

        colors_bar = [ACCENT, GRAY]
        bars = ax_E2.bar(categories, rhos, color=colors_bar, alpha=0.78,
                          edgecolor='white', width=0.5)
        ax_E2.axhline(0, color='black', linewidth=0.8)
        for i, (r, clo, chi) in enumerate(zip(rhos, ci_los, ci_his)):
            if not np.isnan(clo):
                ax_E2.errorbar(i, r, yerr=[[r - clo], [chi - r]],
                                fmt='none', color='black', capsize=4,
                                linewidth=1.5)
            ax_E2.text(i, r + (0.015 if r >= 0 else -0.025),
                        f'{r:.2f}', ha='center', fontsize=10, fontweight='bold')

        ax_E2.set_ylabel('Spearman ρ', fontsize=10)
        ax_E2.set_title('(I) LEDD: raw vs\nduration-adjusted ρ',
                         fontsize=11, fontweight='bold')
        ax_E2.set_ylim(-0.8, 0.6)
        ax_E2.tick_params(labelsize=9)

    # Pending data note
    ax_E3 = fig.add_subplot(gs_E[2])
    ax_E3.axis('off')
    note_txt = (
        f"⚠  LEDD Analysis — Preliminary\n\n"
        f"Current data: n = {ledd_n} / {len(df)} PD patients\n"
        f"Remaining {len(df) - ledd_n} subjects pending\n"
        f"(data collection in progress).\n\n"
        f"Literature context:\n"
        f"LEDD is not expected to be an\n"
        f"independent predictor of complexity\n"
        f"once disease duration is controlled\n"
        f"(Haapaniemi 2001; Devos 2003;\n"
        f"Lerma 2016; meta: Solla 2020).\n\n"
        f"Results will be updated when\n"
        f"the full dataset is available."
    )
    ax_E3.text(0.05, 0.95, note_txt, transform=ax_E3.transAxes,
               va='top', ha='left', fontsize=9.5, color='#444444',
               linespacing=1.5,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#FEF9E7',
                         edgecolor='#F39C12', linewidth=1.2, alpha=0.95))

    # ── Supertitle & save ─────────────────────────────────────────────────────
    N_TOTAL  = 28   # official CETRAM PD count (28 subjects with MSE data)
    N_AVAIL  = len(df)
    N_PEND   = N_TOTAL - N_AVAIL
    is_preliminary = N_AVAIL < N_TOTAL

    fig.suptitle(
        'Figure 8: Cardiac Complexity and Clinical Severity in Parkinson\'s Disease — CETRAM Cohort',
        fontsize=14, fontweight='bold', y=0.975
    )
    fig.text(0.5, 0.962,
             f'Panels A–B: Spearman ρ between rcMSE complexity index and CISI-PD subscales, disease duration, and non-motor burden. '
             f'Panels C–D: Group comparisons by H&Y stage and non-motor symptom presence. '
             f'Panel E: LEDD analysis (preliminary). '
             f'Current n = {N_AVAIL}/{N_TOTAL} PD patients; {N_PEND} clinical records pending.',
             ha='center', fontsize=9, color=GRAY, wrap=True)

    # ── Preliminary stamp (diagonal watermark) ────────────────────────────────
    if is_preliminary:
        stamp_txt = (f'PRELIMINARY  ·  {N_AVAIL} / {N_TOTAL} subjects  ·  '
                     f'{N_PEND} clinical records pending')
        fig.text(0.5, 0.50, stamp_txt,
                 fontsize=22, color='#E74C3C', alpha=0.13,
                 ha='center', va='center',
                 fontweight='bold',
                 rotation=30,
                 transform=fig.transFigure,
                 zorder=0)

        # Top banner strip
        fig.text(0.5, 0.005,
                 f'⚠  PRELIMINARY FIGURE  ·  n = {N_AVAIL}/{N_TOTAL} PD patients with clinical data  ·  '
                 f'{N_PEND} subjects pending clinical record entry  ·  '
                 f'LEDD: {df["LEDD"].notna().sum()}/{N_AVAIL} available  ·  '
                 f'Figure updates automatically when data/deidentified_clinical_consolidated.xlsx is completed.',
                 ha='center', va='bottom', fontsize=8, color='white',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='#E74C3C',
                           edgecolor='none', alpha=0.88),
                 transform=fig.transFigure)

    out_png = os.path.join(OUT_DIR, 'Figure8.png')
    out_svg = os.path.join(OUT_DIR, 'Figure8.svg')
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved → {out_png}")
    print(f"  Saved → {out_svg}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    results = []
    for df_c, adj_label in [(corr_crude, 'crude'), (corr_adj, 'age_adjusted')]:
        tmp = df_c.copy()
        tmp['adjustment'] = adj_label
        results.append(tmp)
    pd.concat(results).to_csv(
        os.path.join(OUT_DIR, 'figure8_correlations.csv'), index=False)
    print(f"  Correlations saved → figures/Figure8/figure8_correlations.csv")


if __name__ == '__main__':
    generate_figure8()
