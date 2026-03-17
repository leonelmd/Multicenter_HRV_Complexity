#!/usr/bin/env python3
"""
Figure 7: Cardiac Complexity as an Autonomic Biomarker
Merged composite of 7 panels (A–G).

Panel A – Spearman ρ correlation heatmap (rcMSE-AUC vs HRV, per center × group)
Panel B – Cross-dataset consistency forest plot
Panel C – Scale physiology heatmap: CETRAM | Cruces | Nagoya (16–20h)   [regenerated]
Panel D – McFadden R² variance decomposition
Panel E – Age/sex confound correction lollipop                           [regenerated]
Panel F – Autonomic synthesis (raw ρ | partial ρ | unique variance)      [regenerated]
Panel G – Scale anatomy: Nagoya (16–20h) & CETRAM                        [regenerated]

Panels A, B, D loaded from pre-computed PNGs.
Panels C, E, F, G re-drawn inline for label consistency.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG8_DIR = os.path.join(BASE, 'figures', 'Figure7')
RESULTS  = FIG8_DIR  # intermediate CSVs live alongside the figure outputs
OUT_DIR  = FIG8_DIR

# ── shared style helpers ───────────────────────────────────────────────────────
groups_physio = {
    'Vagal':          ['pNN50', 'RMSSD', 'SD1', 'HF_power', 'HF_norm'],
    'Sympathovagal':  ['SDNN', 'SD2', 'Total_power'],
    'Sym. balance':   ['LF_norm', 'LF_power', 'LF_HF'],
    'Slow/circadian': ['VLF_power', 'SDANN'],
    'Fractal':        ['DFA_alpha1', 'DFA_alpha2'],
}
group_colors = {
    'Vagal':          '#3b82f6',
    'Sympathovagal':  '#7c3aed',
    'Sym. balance':   '#f59e0b',
    'Slow/circadian': '#16a34a',
    'Fractal':        '#7f4f2a',
}

def _ordered_metrics(index):
    out, spans = [], []
    for grp, mets in groups_physio.items():
        start = len(out)
        for m in mets:
            if m in index:
                out.append(m)
        end = len(out)
        spans.append((start, end, (start + end - 1) / 2.0, grp))
    return out, spans

def add_panel_label(ax, label, fontsize=20):
    ax.text(-0.06, 1.08, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='bottom', ha='right')

def show_png(ax, path):
    """Display a PNG in an axis, removing ticks."""
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img, aspect='auto', interpolation='lanczos')
    else:
        ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(path)}',
                ha='center', va='center', transform=ax.transAxes, color='red')
    ax.axis('off')


# ── Panel C : Scale physiology heatmap (3 columns) ───────────────────────────
def draw_panel_C(axes_row):
    """axes_row: list of 3 Axes."""
    sc = pd.read_csv(os.path.join(RESULTS, 'scale_metric_correlations.csv'))

    hrv_order = ['RMSSD', 'pNN50', 'SD1', 'HF_power', 'HF_norm',
                 'SDNN', 'SD2', 'Total_power',
                 'LF_norm', 'LF_power', 'LF_HF',
                 'VLF_power', 'SDANN',
                 'DFA_alpha1', 'DFA_alpha2']

    configs = [
        ('Chile',          'CETRAM\n(15-min rest)', 5),
        ('Spain',          'Cruces\n(15-min rest)', 5),
        ('Japan-afternoon','Nagoya (16–20h)\n(4-hour block)', 20),
    ]

    for ax, (ds_key, title, max_rel) in zip(axes_row, configs):
        sub = sc[sc['Dataset'] == ds_key]
        scales  = sorted(sub['Scale'].unique())
        metrics = [m for m in hrv_order if m in sub['Metric'].unique()]

        mat = pd.DataFrame(np.nan, index=metrics, columns=scales)
        for _, row in sub.iterrows():
            if row['Metric'] in metrics and row['Scale'] in scales:
                mat.loc[row['Metric'], row['Scale']] = row['rho']

        # grey-out unreliable scales (short recordings)
        reliable_mat   = mat.copy()
        unreliable_mat = mat.copy()
        reliable_mat.iloc[:, [i for i, s in enumerate(scales) if s > max_rel]]   = np.nan
        unreliable_mat.iloc[:, [i for i, s in enumerate(scales) if s <= max_rel]] = np.nan

        sns.heatmap(reliable_mat,   ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    cbar=False, linewidths=0.3, linecolor='#dddddd',
                    xticklabels=2, yticklabels=True)
        if unreliable_mat.notna().any().any():
            sns.heatmap(unreliable_mat, ax=ax, cmap='Greys', center=0, vmin=-1, vmax=1,
                        cbar=False, linewidths=0.3, linecolor='#dddddd',
                        xticklabels=2, yticklabels=False, alpha=0.35)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=5)
        ax.set_xlabel('MSE Scale', fontsize=10)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = axes_row[-1].figure.colorbar(sm, ax=axes_row, shrink=0.85,
                                         fraction=0.015, pad=0.02)
    cbar.set_label('Spearman ρ', fontsize=9)


# ── Panel E : Confound correction lollipop ────────────────────────────────────
def draw_panel_E(ax):
    adj    = pd.read_csv(os.path.join(RESULTS, 'age_sex_adjusted_correlations.csv'))
    pooled = adj[adj['Dataset'] == 'Pooled'].set_index('Metric')
    metrics, spans = _ordered_metrics(pooled.index)
    n   = len(metrics)
    ys  = np.arange(n)
    raw = [pooled.loc[m, 'rho_unadj'] for m in metrics]
    adj_r = [pooled.loc[m, 'rho_adj'] for m in metrics]

    for i, (m, r, a) in enumerate(zip(metrics, raw, adj_r)):
        grp = next(g for g, mets in groups_physio.items() if m in mets)
        c   = group_colors[grp]
        if not (pd.isna(r) or pd.isna(a)):
            ax.plot([r, a], [i, i], color=c, linewidth=1.4, alpha=0.6, zorder=2)
        if not pd.isna(r):
            ax.plot(r, i, 'o', ms=9, mec=c, mfc='white', mew=2, zorder=3)
        if not pd.isna(a):
            ax.plot(a, i, 'o', ms=8, color=c, zorder=4)

    for start, end, mid, grp in spans:
        ax.text(0.72, mid, grp, transform=ax.get_yaxis_transform(),
                color=group_colors[grp], fontsize=8, fontstyle='italic',
                va='center', ha='left')
        if start > 0:
            ax.axhline(start - 0.5, color='#cccccc', lw=0.7, ls='--', zorder=1)

    ax.axvline(0, color='#444', lw=1, ls='--', zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels(metrics, fontsize=9)
    ax.set_xlabel('Spearman ρ  (rcMSE-AUC vs HRV,  Pooled n=152)', fontsize=9)
    ax.set_title('Confound Correction\n(Age & Sex Adjustment)', fontsize=11, fontweight='bold')
    h1 = plt.Line2D([0],[0], marker='o', color='gray', mfc='white', mew=2, ms=8, ls='')
    h2 = plt.Line2D([0],[0], marker='o', color='gray', mfc='gray',  ms=8, ls='')
    ax.legend([h1, h2], ['Raw ρ', 'Partial ρ (adj.)'],
              loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.55, 0.75); ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)


# ── Panel F : Autonomic synthesis 3-panel ─────────────────────────────────────
def draw_panel_F(axes_3):
    phys     = pd.read_csv(os.path.join(RESULTS, 'physiological_interpretation.csv')).set_index('metric')
    metrics, spans = _ordered_metrics(phys.index)
    n        = len(metrics)
    ys       = np.arange(n)
    raw_rho  = [phys.loc[m, 'rho_with_complexity_pooled'] for m in metrics]
    part_rho = [phys.loc[m, 'partial_r_after_age_sex']    for m in metrics]
    uvar     = [phys.loc[m, 'unique_variance_%']           for m in metrics]
    bar_cols = [group_colors[next(g for g, mets in groups_physio.items() if m in mets)]
                for m in metrics]

    for ax, vals, xlabel, title in [
        (axes_3[0], raw_rho,  'Raw Spearman ρ\n(pooled n=152)', 'Raw correlation'),
        (axes_3[1], part_rho, 'Partial ρ\n(age + sex adj.)',    'Age/sex-adjusted'),
    ]:
        for i, (v, c) in enumerate(zip(vals, bar_cols)):
            ax.barh(i, v, color=c, alpha=0.75, height=0.65)
            ax.text(v + (0.015 if v >= 0 else -0.015), i, f'{v:+.2f}',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=7)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(ys); ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xlim(-0.6, 0.7); ax.grid(axis='x', alpha=0.3)
        ax.set_title(title, fontsize=9)

    axes_3[0].set_yticklabels(metrics, fontsize=9)
    axes_3[1].set_yticklabels([])

    ax = axes_3[2]
    for i, (u, c) in enumerate(zip(uvar, bar_cols)):
        ax.plot([0, u], [i, i], color=c, lw=1.2, alpha=0.6)
        ax.plot(u, i, 'o', color=c, ms=7, zorder=3)
        ax.text(u + 0.2, i, f'{u:.1f}%', va='center', fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels([])
    ax.set_xlabel('Unique variance %\n(indep. of pNN50)', fontsize=8)
    ax.set_xlim(-0.5, 16); ax.grid(axis='x', alpha=0.3)
    ax.set_title('Unique information', fontsize=9)

    for start, end, mid, grp in spans:
        for a in axes_3:
            if start > 0:
                a.axhline(start - 0.5, color='#cccccc', lw=0.7, ls='--', zorder=1)
        axes_3[2].text(15.5, mid, grp, color=group_colors[grp], fontsize=8,
                       fontstyle='italic', va='center', ha='left')
    for a in axes_3:
        a.invert_yaxis(); a.set_ylim(n - 0.5, -0.5)


# ── Panel G : Scale anatomy line plots ────────────────────────────────────────
def draw_panel_G(axes_2):
    sc = pd.read_csv(os.path.join(RESULTS, 'scale_metric_correlations.csv'))
    key_metrics = ['pNN50', 'RMSSD', 'LF_norm', 'SDNN', 'DFA_alpha1']
    labels = {'pNN50': 'pNN50',
              'RMSSD': 'RMSSD',
              'LF_norm': 'LF norm',
              'SDNN':  'SDNN',
              'DFA_alpha1': 'DFA alpha1'}
    colors = {'pNN50':'#e03030','RMSSD':'#e07030','LF_norm':'#208060',
              'SDNN':'#303090','DFA_alpha1':'#606060'}
    ls_map = {'pNN50':'-','RMSSD':'--','LF_norm':'-.','SDNN':':'
              ,'DFA_alpha1':(0,(3,1,1,1))}

    datasets = [('Japan-afternoon', 'Nagoya (16–20h)  n=38 — all scales reliable', None),
                ('Chile',           'CETRAM  n=71 — reliable up to scale 5',       5)]

    for ax, (ds, title, max_rel) in zip(axes_2, datasets):
        sub = sc[sc['Dataset'] == ds]
        if max_rel is not None:
            ax.axvspan(max_rel + 0.5, 20.5, facecolor='#dddddd', alpha=0.5,
                       zorder=0, label='Unreliable (short recording)')
        for m in key_metrics:
            d   = sub[sub['Metric'] == m].sort_values('Scale')
            if d.empty: continue
            sig = d['q_BH'].apply(
                lambda x: (str(x) != 'nan' and float(x) < 0.05) if pd.notna(x) else False)
            ax.plot(d['Scale'], d['rho'], color=colors[m], linestyle=ls_map[m],
                    lw=1.8, label=labels[m], zorder=3)
            if sig.any():
                ax.scatter(d.loc[sig,'Scale'], d.loc[sig,'rho'],
                           marker='D', s=35, color=colors[m], zorder=4)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlim(0.5, 20.5); ax.set_xticks(range(1, 21, 2))
        ax.set_xlabel('MSE Scale', fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)

    axes_2[0].set_ylabel('Spearman ρ\n(entropy at scale vs HRV)', fontsize=10)
    axes_2[0].set_ylim(-1.0, 1.0)
    axes_2[0].text(1.5, 0.85, '* Significant (FDR q<0.05)', fontsize=8, color='#555')
    handles, lbls = axes_2[0].get_legend_handles_labels()
    axes_2[1].legend(handles, lbls, loc='upper right', fontsize=8, framealpha=0.9)


# ── Compose ───────────────────────────────────────────────────────────────────
def generate_figure7():
    fig = plt.figure(figsize=(26, 32))
    sns.set_style('ticks')

    # Layout (7 panels, A–G):
    # Row 0: A (col 0)      | B (cols 1–2)
    # Row 1: C (3 sub-axes, all cols)
    # Row 2: D (cols 0–1)   | E / lollipop (col 2)
    # Row 3: F / synthesis  (3 sub-panels, all cols) + stub col
    # Row 4: G / anatomy    (2 sub-panels, all cols)

    outer = gridspec.GridSpec(5, 3, figure=fig,
                              hspace=0.52, wspace=0.32,
                              height_ratios=[4, 3.5, 3.5, 3.5, 4])

    # Row 0: A (col 0) | B (cols 1-2)
    ax_A = fig.add_subplot(outer[0, 0])
    ax_B = fig.add_subplot(outer[0, 1:])
    show_png(ax_A, os.path.join(FIG8_DIR, 'Fig7_A_correlation_heatmap.png'))
    show_png(ax_B, os.path.join(FIG8_DIR, 'Fig7_B_cross_dataset_consistency.png'))
    add_panel_label(ax_A, 'A')
    add_panel_label(ax_B, 'B')

    # Row 1: C — 3 sub-axes side by side
    c_axes = [fig.add_subplot(outer[1, j]) for j in range(3)]
    draw_panel_C(c_axes)
    add_panel_label(c_axes[0], 'C')

    # Row 2: D (cols 0-1, wider) | E lollipop (col 2)
    ax_D = fig.add_subplot(outer[2, :2])
    ax_E = fig.add_subplot(outer[2, 2])
    show_png(ax_D, os.path.join(FIG8_DIR, 'Fig7_D_variance_decomposition.png'))
    draw_panel_E(ax_E)
    add_panel_label(ax_D, 'D')
    add_panel_label(ax_E, 'E')

    # Row 3: F — synthesis (3 sub-panels, cols 0-1) + stub
    f_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3, :2],
                                               wspace=0.05)
    f_axes = [fig.add_subplot(f_inner[0, j]) for j in range(3)]
    draw_panel_F(f_axes)
    add_panel_label(f_axes[0], 'F')
    ax_stub = fig.add_subplot(outer[3, 2])
    ax_stub.axis('off')

    # Row 4: G — scale anatomy (2 sub-panels, full width)
    g_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4, :],
                                               wspace=0.08)
    g_axes = [fig.add_subplot(g_inner[0, j]) for j in range(2)]
    draw_panel_G(g_axes)
    add_panel_label(g_axes[0], 'G')

    plt.suptitle(
        'Figure 7: Cardiac Complexity as a Novel Autonomic Biomarker\n'
        'Pooled cohort: CETRAM + Cruces + Nagoya  ·  rcMSE nAUC(1–20) / HR',
        fontsize=20, fontweight='bold', y=0.995)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'Figure7.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 7 (merged) saved to {out_path}")


if __name__ == '__main__':
    generate_figure7()
