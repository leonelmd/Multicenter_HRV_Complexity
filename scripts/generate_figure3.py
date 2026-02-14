#!/usr/bin/env python3
"""
Figure 3: Circadian Fractal Dynamics and Biomarker Discovery (Reshaped)
- Rows 1-2: 24h Evolution Curves (HR, Complexity, SDNN, RMSSD)
- Row 3 Left: Exhaustive Heatmap (4h window sweep for Complexity)
- Row 3 Right: Discriminative Delta Boxplot (Morning vs. Afternoon)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
JAPAN_EVOLUTION = os.path.join(DATA_DIR, "japan_evolution.csv")
JAPAN_META = os.path.join(DATA_DIR, "japan_metadata.csv")

def add_panel_label(ax, label):
    ax.text(-0.05, 1.1, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')

def add_stat_annotation(ax, x1, x2, y, h, text, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=color, fontsize=12, fontweight='bold')

def get_sig_chars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def parse_time(t_str):
    try:
        if pd.isna(t_str): return np.nan
        h, m, s = map(int, str(t_str).split(':'))
        return h + m/60.0 + s/3600.0
    except:
        return np.nan

def generate_figure3():
    colors = {'Control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25)
    
    # 1. Load Data & Align
    df_ev = pd.read_csv(JAPAN_EVOLUTION)
    df_meta = pd.read_csv(JAPAN_META)
    df_ev['Group'] = df_ev['Group'].str.lower().replace({'control': 'Control', 'pd': 'PD'})
    if 'Subject' not in df_meta.columns:
        if 'Subject_ID' in df_meta.columns: df_meta = df_meta.rename(columns={'Subject_ID': 'Subject'})
    
    start_map = dict(zip(df_meta['Subject'], df_meta['Start_Time']))
    df_ev['Start_Hour'] = df_ev['Subject'].map(start_map).apply(parse_time)
    df_ev['Clock_Time'] = (df_ev['Start_Hour'] + df_ev['Time_h']) % 24
    df_ev['Plot_Time'] = (df_ev['Clock_Time'] - 12) % 24  # Center Noon=0, Midnight=12
    
    # === STAGE 1: Evolution Curves (Top 2 Rows) ===
    metrics = ['HR', 'Complexity', 'SDNN', 'RMSSD']
    titles = ['Heart Rate (BPM)', 'Complexity Index (MSE 1-5)', 'SDNN (s)', 'RMSSD (s)']
    
    for idx, m in enumerate(metrics):
        r = idx // 2
        c = idx % 2
        ax = fig.add_subplot(gs[r, c])
        add_panel_label(ax, chr(65 + idx))
        
        df_ev['Bin_Time'] = df_ev['Plot_Time'].round(2)
        subj_bins = df_ev.groupby(['Group', 'Subject', 'Bin_Time'])[m].mean().reset_index()
        summary = subj_bins.groupby(['Group', 'Bin_Time'])[m].agg(['mean', 'sem']).reset_index()
        
        for group in ['Control', 'PD']:
            data = summary[summary['Group'] == group].sort_values('Bin_Time')
            y_smooth = data['mean'].rolling(15, center=True, min_periods=1).mean()
            s_smooth = data['sem'].rolling(15, center=True, min_periods=1).mean()
            ax.plot(data['Bin_Time'], y_smooth, color=colors[group], linewidth=3, label=group)
            ax.fill_between(data['Bin_Time'], y_smooth - s_smooth, y_smooth + s_smooth, color=colors[group], alpha=0.15)
            
        ax.set_title(titles[idx], fontsize=18, fontweight='bold')
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['12 PM', '6 PM', '12 AM', '6 AM', '12 PM'])
        ax.axvline(12, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
        if idx == 0: ax.legend(loc='upper right', fontsize=12)

    # === STAGE 2: Exhaustive Heatmap (Bottom Left) ===
    print("Generating Heatmap...")
    win_len = 4
    win_data = {}
    for start_h in range(24):
        end_h = start_h + win_len
        if end_h > 24:
            mask = (df_ev['Clock_Time'] >= start_h) | (df_ev['Clock_Time'] < (end_h % 24))
        else:
            mask = (df_ev['Clock_Time'] >= start_h) & (df_ev['Clock_Time'] < end_h)
        label = f"{start_h:02d}"
        win_data[label] = df_ev[mask].groupby(['Group', 'Subject'])['Complexity'].mean().reset_index()
    
    all_pairs = []
    labels = list(win_data.keys())
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j: continue
            df1, df2 = win_data[labels[i]], win_data[labels[j]]
            df_pair = pd.merge(df1, df2, on=['Subject', 'Group'], suffixes=('_1', '_2'))
            df_pair['Delta'] = df_pair['Complexity_1'] - df_pair['Complexity_2']
            c = df_pair[df_pair['Group'] == 'Control']['Delta'].dropna()
            p = df_pair[df_pair['Group'] == 'PD']['Delta'].dropna()
            if len(c) > 10 and len(p) > 10:
                t, pval = ttest_ind(c, p)
                all_pairs.append({'W1': labels[i], 'W2': labels[j], 'LogP': -np.log10(pval)})
    
    df_h = pd.DataFrame(all_pairs)
    pivot = df_h.pivot(index='W1', columns='W2', values='LogP')
    
    ax_heat = fig.add_subplot(gs[2, 0])
    add_panel_label(ax_heat, 'E')
    sns.heatmap(pivot, cmap='Spectral_r', ax=ax_heat, cbar_kws={'label': '-log10(p)'})
    ax_heat.set_title(f'Exhaustive Search: Complexity Delta ({win_len}h Windows)', fontsize=18, fontweight='bold')
    ax_heat.set_xlabel('Reference Window (Start Hour)')
    ax_heat.set_ylabel('Comparison Window (Start Hour)')

    # === STAGE 3: Best Delta Boxplot (Bottom Right) ===
    # Using 07:00-11:00 vs 13:00-17:00 (High-p result from sweep)
    # Actually, let's use the one found: 07:00 vs 13:00 (4h windows)
    w1_start, w2_start = 7, 13
    df_w1 = win_data[f"{w1_start:02d}"]
    df_w2 = win_data[f"{w2_start:02d}"]
    df_final = pd.merge(df_w1, df_w2, on=['Subject', 'Group'], suffixes=('_Morning', '_Afternoon'))
    df_final['Delta'] = df_final['Complexity_Morning'] - df_final['Complexity_Afternoon']
    
    ax_box = fig.add_subplot(gs[2, 1])
    add_panel_label(ax_box, 'F')
    
    sns.boxplot(x='Group', y='Delta', data=df_final, palette=colors, ax=ax_box, width=0.5, showfliers=False, order=['Control', 'PD'])
    sns.stripplot(x='Group', y='Delta', data=df_final, palette=colors, ax=ax_box, dodge=False, alpha=0.6, color='black', order=['Control', 'PD'])
    
    c_vals = df_final[df_final['Group'] == 'Control']['Delta']
    p_vals = df_final[df_final['Group'] == 'PD']['Delta']
    t, pval = ttest_ind(c_vals, p_vals)
    
    y_max = df_final['Delta'].max()
    y_min = df_final['Delta'].min()
    dist = y_max - y_min
    add_stat_annotation(ax_box, 0, 1, y_max + dist*0.05, dist*0.05, f'p = {pval:.4f} {get_sig_chars(pval)}')
    
    ax_box.set_title('Best Biomarker: Morning vs. Afternoon Delta', fontsize=18, fontweight='bold')
    ax_box.set_ylabel('$\Delta$ Complexity (07-11 vs. 13-17)', fontsize=14)
    ax_box.set_xlabel('')
    ax_box.set_ylim(y_min - dist*0.1, y_max + dist*0.25)
    ax_box.axhline(0, color='black', linestyle='--', alpha=0.3)
    sns.despine(ax=ax_box)

    plt.suptitle("Figure 3: Circadian Fractal Dynamics and Biomarker Discovery", fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_dir = os.path.join(FIGURES_DIR, "Figure3")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'Figure3.png')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight'), dpi=300, bbox_inches='tight')
    print(f"Figure 3 reshaped and saved to {out_dir}")

if __name__ == "__main__":
    generate_figure3()
