#!/usr/bin/env python3
"""
Figure 3: Circadian Dynamics of Cardiac Complexity (Japan Cohort, N=50)
Refined Composite MSE · Scales 1-20 · 4-hour clock-aligned windows · Minimum 4000 heartbeats

Panels A-D : Circadian evolution curves (HR, nAUC, SDNN, RMSSD)
Panel E    : Discrimination per 4h window — -log10(p) bar chart
Panel F    : Multi-window circadian comparison (key time windows, PD vs Control)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_ind

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
JAPAN_EVO   = os.path.join(DATA_DIR, "japan_evolution.csv")

COLORS = {'Control': '#2E86AB', 'PD': '#D62828'}

def add_panel_label(ax, label):
    ax.text(-0.05, 1.10, label, transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='bottom', ha='right')

def get_sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def generate_figure3():
    sns.set_style("ticks")

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(JAPAN_EVO)
    df['Group'] = df['Group'].str.strip().str.lower().replace(
        {'control': 'Control', 'pd': 'PD'})
    df = df[df['Group'].isin(['Control', 'PD'])]
    # Window centre for plotting (mid-point of 4h block)
    df['Win_centre'] = df['Window_start_h'] + 2.0

    fig = plt.figure(figsize=(24, 20))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28)

    # ── Panels A–D : Circadian evolution curves ────────────────────────────────
    metric_cfg = [
        ('HR',       'Heart Rate — 24h Circadian Profile',       'HR (bpm)'),
        ('nAUC_1_20','Cardiac Complexity — rcMSE nAUC(1–20)',     'nAUC(1–20)'),
        ('SDNN',     'SDNN — Sympatho-vagal HRV',                 'SDNN (s)'),
        ('RMSSD',    'RMSSD — Vagal HRV',                         'RMSSD (s)'),
    ]
    for idx, (metric, title, ylabel) in enumerate(metric_cfg):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        add_panel_label(ax, chr(65 + idx))

        # subject-level mean per window, then group mean ± SEM
        subj_mean = (df.groupby(['Group', 'Subject', 'Win_centre'])[metric]
                     .mean().reset_index())
        summary   = (subj_mean.groupby(['Group', 'Win_centre'])[metric]
                     .agg(['mean', 'sem']).reset_index())

        for grp in ['Control', 'PD']:
            d = summary[summary['Group'] == grp].sort_values('Win_centre')
            # light smoothing over 3 points
            ym = d['mean'].rolling(3, center=True, min_periods=1).mean()
            ys = d['sem'].rolling(3,  center=True, min_periods=1).mean()
            ax.plot(d['Win_centre'], ym, color=COLORS[grp], linewidth=3, label=grp)
            ax.fill_between(d['Win_centre'], ym - ys, ym + ys,
                            color=COLORS[grp], alpha=0.15)

        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Time of day', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlim(0, 26)
        ax.set_xticks([2, 6, 10, 14, 18, 22])
        ax.set_xticklabels(['02:00','06:00','10:00','14:00','18:00','22:00'],
                           fontsize=10)
        ax.grid(True, alpha=0.18)
        sns.despine(ax=ax)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=12)

    # ── Panel E : -log10(p) per window (absolute PD vs Control) ───────────────
    ax_e = fig.add_subplot(gs[2, 0])
    add_panel_label(ax_e, 'E')

    wins   = sorted(df['Window_start_h'].unique())
    logps  = []
    sig_flags = []
    for w in wins:
        sub = df[df['Window_start_h'] == w]
        c   = sub[sub['Group'] == 'Control']['nAUC_1_20'].dropna()
        p   = sub[sub['Group'] == 'PD']['nAUC_1_20'].dropna()
        if len(c) > 5 and len(p) > 5:
            _, pv = ttest_ind(c, p)
            lp = -np.log10(pv)
        else:
            lp = 0.0
            pv = 1.0
        logps.append(lp)
        sig_flags.append(pv < 0.05)

    bar_colors = ['#D62828' if s else '#AAAAAA' for s in sig_flags]
    x_labels   = [f'{int(w):02d}h' for w in wins]
    ax_e.bar(x_labels, logps, color=bar_colors, edgecolor='white', linewidth=0.4)
    ax_e.axhline(-np.log10(0.05),  color='black', ls='--', alpha=0.5, lw=1.2,
                 label='p = 0.05')
    ax_e.axhline(-np.log10(0.001), color='black', ls=':',  alpha=0.5, lw=1.2,
                 label='p = 0.001')
    ax_e.set_title('PD vs Control: nAUC(1–20) Discrimination Per Window\n'
                   '(t-test, 4-hour windows, minimum 4 000 beats)',
                   fontsize=16, fontweight='bold')
    ax_e.set_xlabel('Window start (4-hour block)', fontsize=13)
    ax_e.set_ylabel('−log₁₀(p)', fontsize=13)
    ax_e.set_xticks(range(len(x_labels)))
    ax_e.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax_e.legend(fontsize=10, loc='upper left')
    red_patch  = mpatches.Patch(color='#D62828', label='Significant (p<0.05)')
    gray_patch = mpatches.Patch(color='#AAAAAA', label='n.s.')
    ax_e.legend(handles=[red_patch, gray_patch], fontsize=10, loc='upper left')
    sns.despine(ax=ax_e)

    # ── Panel F : Multi-window circadian boxplot (5 key windows) ─────────────
    ax_f = fig.add_subplot(gs[2, 1])
    add_panel_label(ax_f, 'F')

    # Choose 5 representative windows spanning the day
    key_windows = [0, 7, 11, 16, 20]
    key_labels  = ['00–04h\n(Night)', '07–11h\n(Morning)',
                   '11–15h\n(Midday)', '16–20h\n(Afternoon★)',
                   '20–00h\n(Evening)']

    plot_rows = []
    for w, lbl in zip(key_windows, key_labels):
        sub = df[df['Window_start_h'] == w][['Group', 'nAUC_1_20']].dropna()
        sub = sub.copy()
        sub['Window'] = lbl
        plot_rows.append(sub)
    df_plot = pd.concat(plot_rows, ignore_index=True)

    # Dodge boxplots: Control and PD side-by-side within each window
    sns.boxplot(data=df_plot, x='Window', y='nAUC_1_20', hue='Group',
                palette=COLORS, hue_order=['Control', 'PD'],
                order=key_labels, showfliers=False, width=0.55, ax=ax_f,
                linewidth=1.2)
    sns.stripplot(data=df_plot, x='Window', y='nAUC_1_20', hue='Group',
                  palette=COLORS, hue_order=['Control', 'PD'],
                  order=key_labels, dodge=True, alpha=0.45,
                  size=4, ax=ax_f, legend=False)

    # Significance brackets above each window
    y_top = df_plot['nAUC_1_20'].quantile(0.97) * 1.04
    h     = df_plot['nAUC_1_20'].std() * 0.07
    for xi, (w, lbl) in enumerate(zip(key_windows, key_labels)):
        sub  = df[df['Window_start_h'] == w]
        cval = sub[sub['Group'] == 'Control']['nAUC_1_20'].dropna()
        pval = sub[sub['Group'] == 'PD']['nAUC_1_20'].dropna()
        if len(cval) > 3 and len(pval) > 3:
            _, pv = ttest_ind(cval, pval)
            sig   = get_sig(pv)
            color = '#D62828' if pv < 0.05 else '#888888'
            ax_f.text(xi, y_top + h * 0.5, sig, ha='center', va='bottom',
                      fontsize=13, fontweight='bold', color=color)

    ax_f.set_title('Circadian Variation in Cardiac Complexity\nPD vs Control by Time Window',
                   fontsize=16, fontweight='bold')
    ax_f.set_xlabel('Time window (4-hour block)', fontsize=13)
    ax_f.set_ylabel('nAUC(1–20) / HR', fontsize=13)
    ax_f.set_ylim(bottom=0)

    # Highlight best window (16-20h) with subtle background
    ax_f.axvspan(2.55, 3.45, color='#FFF0AA', alpha=0.55, zorder=0)
    ax_f.text(3, ax_f.get_ylim()[1] * 0.97, '★ best', ha='center',
              fontsize=9, color='#B8860B', style='italic')

    handles, labels = ax_f.get_legend_handles_labels()
    ax_f.legend(handles[:2], labels[:2], loc='upper left', fontsize=12)
    sns.despine(ax=ax_f)

    # ── Save ──────────────────────────────────────────────────────────────────
    n_pd  = df[df['Group'] == 'PD']['Subject'].nunique()
    n_con = df[df['Group'] == 'Control']['Subject'].nunique()
    plt.suptitle(
        f'Figure 3: Circadian Dynamics of Cardiac Complexity (Japan Cohort, N={n_pd+n_con})\n'
        'Refined Composite MSE · Scales 1–20 · 4-hour clock-aligned windows · '
        'Minimum 4 000 heartbeats per window',
        fontsize=22, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir  = os.path.join(FIGURES_DIR, "Figure3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Figure3.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 3 saved to {out_path}")

if __name__ == "__main__":
    generate_figure3()
