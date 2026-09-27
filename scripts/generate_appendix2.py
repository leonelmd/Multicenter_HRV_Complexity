#!/usr/bin/env python3
"""
Appendix Figure 2: Per-scale AUC across all recording contexts
==============================================================
Supplementary figure illustrating the per-scale discriminative power
(AUC, Mann-Whitney U) of MSE SampEn across scales 1–5 (short recordings)
and 1–20 (Nagoya Holter), comparing ECG vs PPG modalities.

Purpose: Provides mechanistic context for the reduced significance of the
complexity index in the Cruces (PPG) cohort.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import figstyle as fs
fs.apply()
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(BASE, 'data')
OUT_DIR    = os.path.join(BASE, 'figures', 'Appendix')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dataset definitions ───────────────────────────────────────────────────────
DATASETS = [
    {
        'label':    'CETRAM',
        'subtitle': 'ECG · 15 min · Santiago, Chile',
        'file':     'chile_mse.csv',
        'modality': 'ECG',
        'ctrl_groups': ['Control'],
        'pd_group':    'PD',
        'max_scale_reliable': 5,
        'max_scale_show':     20,
    },
    {
        'label':    'Cruces',
        'subtitle': 'PPG · 7.40 min · Bilbao, Spain',
        'file':     'spain_mse.csv',
        'modality': 'PPG',
        'ctrl_groups': ['Control'],
        'pd_group':    'PD',
        'max_scale_reliable': 5,
        'max_scale_show':     20,
    },
    {
        'label':    'Nagoya (07–11h)',
        'subtitle': 'ECG · 4 h wearable · Nagoya, Japan',
        'file':     'japan_morning_mse.csv',
        'modality': 'ECG',
        'ctrl_groups': ['Control'],
        'pd_group':    'PD',
        'max_scale_reliable': 20,
        'max_scale_show':     20,
    },
    {
        'label':    'Nagoya (18–22 h)',
        'subtitle': 'ECG · 4 h wearable · Nagoya, Japan',
        'file':     'japan_window_mse.csv',
        'modality': 'ECG',
        'ctrl_groups': ['Control'],
        'pd_group':    'PD',
        'max_scale_reliable': 20,
        'max_scale_show':     20,
    },
]

# ── Color scheme ──────────────────────────────────────────────────────────────
# ECG: blue family; PPG: orange
MODALITY_COLOR = {
    'ECG': '#2166AC',   # strong blue
    'PPG': '#D6604D',   # muted red-orange
}
UNRELIABLE_ALPHA   = 0.18   # fill alpha for unreliable zone
UNRELIABLE_LINEWIDTH = 1.2

# ── Compute per-scale AUC + p-value ──────────────────────────────────────────

def compute_scale_stats(mse_wide, ctrl_groups, pd_group, scales):
    results = []
    for s in scales:
        col = f'S{s}'
        if col not in mse_wide.columns:
            continue
        ctrl = mse_wide[mse_wide['Group'].isin(ctrl_groups)][col].dropna()
        pd_  = mse_wide[mse_wide['Group'] == pd_group][col].dropna()
        if len(ctrl) < 3 or len(pd_) < 3:
            results.append({'scale': s, 'auc': np.nan, 'p': np.nan,
                             'mean_ctrl': np.nan, 'mean_pd': np.nan,
                             'n_ctrl': len(ctrl), 'n_pd': len(pd_)})
            continue
        _, p = stats.mannwhitneyu(ctrl, pd_, alternative='two-sided')
        vals = np.concatenate([ctrl, pd_])
        labs = np.concatenate([np.ones(len(ctrl)), np.zeros(len(pd_))])
        try:
            auc = roc_auc_score(labs, vals)
        except Exception:
            auc = np.nan
        results.append({'scale': s, 'auc': auc, 'p': p,
                         'mean_ctrl': ctrl.mean(), 'mean_pd': pd_.mean(),
                         'n_ctrl': len(ctrl), 'n_pd': len(pd_)})
    return pd.DataFrame(results)


def load_wide(fpath, ctrl_groups, pd_group):
    df = pd.read_csv(fpath)
    wide = df.pivot_table(index=['Subject', 'Group'], columns='Scales', values='MSE').reset_index()
    wide.columns = ['Subject', 'Group'] + [f'S{int(c)}' for c in wide.columns[2:]]
    return wide


# ── Significance annotation ───────────────────────────────────────────────────

def sig_label(p):
    if np.isnan(p):  return ''
    if p < 0.001:    return '***'
    if p < 0.01:     return '**'
    if p < 0.05:     return '*'
    return ''


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=False)
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.28, wspace=0.38)

for ax, ds in zip(axes, DATASETS):
    wide  = load_wide(os.path.join(DATA_DIR, ds['file']),
                      ds['ctrl_groups'], ds['pd_group'])
    scales_all = list(range(1, ds['max_scale_show'] + 1))
    stats_df   = compute_scale_stats(wide, ds['ctrl_groups'], ds['pd_group'], scales_all)

    reliable_mask   = stats_df['scale'] <= ds['max_scale_reliable']
    unreliable_mask = stats_df['scale'] >  ds['max_scale_reliable']

    color = MODALITY_COLOR[ds['modality']]

    # ── Reliable range ──────────────────────────────────────────────────────
    r = stats_df[reliable_mask]
    ax.plot(r['scale'], r['auc'], color=color, linewidth=2.4, zorder=4,
            marker='o', markersize=5.5, markerfacecolor=color, markeredgecolor='white',
            markeredgewidth=0.8)
    ax.fill_between(r['scale'], 0.5, r['auc'], alpha=0.15, color=color, zorder=2)

    # ── Unreliable range (greyed out) ───────────────────────────────────────
    if unreliable_mask.any():
        u = stats_df[unreliable_mask]
        # Bridge from last reliable point
        bridge_scale = [r['scale'].iloc[-1]] + u['scale'].tolist()
        bridge_auc   = [r['auc'].iloc[-1]]   + u['auc'].tolist()
        ax.plot(bridge_scale, bridge_auc, color=color, linewidth=UNRELIABLE_LINEWIDTH,
                linestyle='--', alpha=0.35, zorder=3)
        ax.fill_between(u['scale'],
                        ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.4,
                        1.0,
                        color='#CCCCCC', alpha=UNRELIABLE_ALPHA, zorder=1,
                        label='Unreliable range\n(N/τ < 200)')
        # Vertical dashed separator
        ax.axvline(ds['max_scale_reliable'] + 0.5, color='#999999',
                   linewidth=1.0, linestyle=':', zorder=3)

    # ── Significance stars at top of each scale ─────────────────────────────
    for _, row in stats_df[reliable_mask].iterrows():
        lbl = sig_label(row['p'])
        if lbl:
            ax.text(row['scale'], row['auc'] + 0.018, lbl,
                    ha='center', va='bottom', fontsize=11,
                    color=color, fontweight='normal', zorder=5)

    # ── Chance line ─────────────────────────────────────────────────────────
    ax.axhline(0.5, color='#666666', linewidth=1.0, linestyle='--', zorder=2, alpha=0.7)

    # ── Axes formatting ─────────────────────────────────────────────────────
    ax.set_xlim(0.3, ds['max_scale_show'] + 0.7)
    ax.set_ylim(0.38, 0.88)
    ax.set_xlabel('MSE Timescale (τ)', fontsize=11)
    ax.set_ylabel('AUC (Controls vs PD)', fontsize=11) if ax == axes[0] else None
    ax.tick_params(labelsize=10)

    if ds['max_scale_show'] == 5:
        ax.set_xticks([1, 2, 3, 4, 5])
    elif ds['max_scale_show'] <= 10:
        ax.set_xticks(range(1, ds['max_scale_show'] + 1))
    else:
        ax.set_xticks([1, 5, 10, 15, 20])

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Modality badge ───────────────────────────────────────────────────────
    badge_color = MODALITY_COLOR[ds['modality']]
    badge_text  = ds['modality']
    ax.text(0.97, 0.96, badge_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=11.5, fontweight='normal',
            color='white',
            bbox=dict(boxstyle='round,pad=0.28', facecolor=badge_color,
                      edgecolor='none', alpha=0.92))

    # ── N label ─────────────────────────────────────────────────────────────
    n_ctrl = stats_df[reliable_mask]['n_ctrl'].iloc[0] if not stats_df[reliable_mask].empty else '?'
    n_pd   = stats_df[reliable_mask]['n_pd'].iloc[0]   if not stats_df[reliable_mask].empty else '?'
    ax.text(0.03, 0.04, f'HC n={n_ctrl}  PD n={n_pd}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color='#555555')

    # ── Panel title ──────────────────────────────────────────────────────────
    ax.set_title(f"{ds['label']}\n{ds['subtitle']}", fontsize=11,
                 fontweight='normal', linespacing=1.5, pad=8)

# ── Figure-level legend ───────────────────────────────────────────────────────
ecg_patch = mpatches.Patch(color=MODALITY_COLOR['ECG'], label='ECG (CETRAM, Nagoya)')
ppg_patch = mpatches.Patch(color=MODALITY_COLOR['PPG'], label='PPG (Cruces)')
chance_line = Line2D([0], [0], color='#666666', linewidth=1.0,
                     linestyle='--', label='AUC = 0.50 (chance)')
unreliable_patch = mpatches.Patch(color='#CCCCCC', alpha=0.55,
                                   label='Unreliable scales (N/τ < 200)')
sig_line = Line2D([0], [0], color='none', label='*p<0.05  **p<0.01  ***p<0.001')

fig.legend(handles=[ecg_patch, ppg_patch, chance_line, unreliable_patch, sig_line],
           loc='lower center', ncol=5,
           fontsize=9.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

# (modality note moved to the figure legend; see FIGURE_LEGENDS.md)

out_png = os.path.join(OUT_DIR, 'FigureAppendix2.png')
out_svg = os.path.join(OUT_DIR, 'FigureAppendix2.svg')

fig.savefig(out_png, dpi=180, bbox_inches='tight')
fig.savefig(out_svg, bbox_inches='tight')
plt.close()

print(f'Saved → {out_png}')
print(f'Saved → {out_svg}')
