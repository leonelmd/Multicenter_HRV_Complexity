#!/usr/bin/env python3
"""
Figure 4: Multicenter Cardiac Complexity Validation (Standardized by Signal Length)
Determines scale range based on available heartbeats (N/Tau >= 200 rule).
- Chile/Spain (15min, N~1100): Scales 1-5
- Japan (4hr, N~15000): Scales 1-20
All indices are HR-Normalized.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# PORTABLE PATH RESOLUTION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# DATA PATHS
CHILE_MSE = os.path.join(DATA_DIR, "chile_mse.csv")
CHILE_METRICS = os.path.join(DATA_DIR, "chile_metrics.csv")
SPAIN_MSE = os.path.join(DATA_DIR, "spain_mse.csv")
SPAIN_METRICS = os.path.join(DATA_DIR, "spain_metrics.csv")
JAPAN_EVO = os.path.join(DATA_DIR, "japan_evolution.csv")
JAPAN_MORNING_MSE = os.path.join(DATA_DIR, "japan_morning_mse.csv")
JAPAN_AFTERNOON_MSE = os.path.join(DATA_DIR, "japan_afternoon_mse.csv")

def add_panel_label(ax, label):
    ax.text(-0.05, 1.15, label, transform=ax.transAxes, fontsize=28, fontweight='bold', va='bottom', ha='right')

def get_sig_chars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'

def generate_figure4():
    colors = {'Control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)
    
    # Configuration for each column
    configs = [
        ('CETRAM (Chile)', CHILE_MSE, 'CHILE_METRICS', range(1, 6), '15m Rest'),
        ('Cruces (Spain)', SPAIN_MSE, 'SPAIN_METRICS', range(1, 6), '15m Slice'),
        ('Nagoya (Morning)', JAPAN_MORNING_MSE, 'JAPAN_EVO_MORNING', range(1, 21), '4h Block'),
        ('Nagoya (Afternoon)', JAPAN_AFTERNOON_MSE, 'JAPAN_EVO_AFTERNOON', range(1, 21), '4h Block')
    ]

    # Pre-calculate Japan HR per window
    df_evo = pd.read_csv(JAPAN_EVO)
    df_evo['Group'] = df_evo['Group'].str.lower().replace({'control': 'Control', 'pd': 'PD'})
    df_meta_j = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    if 'Subject_ID' in df_meta_j.columns: df_meta_j = df_meta_j.rename(columns={'Subject_ID': 'Subject'})
    
    def parse_t(s): 
        try:
            parts = str(s).split(':')
            return int(parts[0]) + int(parts[1])/60.0 + int(parts[2])/3600.0
        except: return np.nan
        
    start_map = dict(zip(df_meta_j['Subject'], df_meta_j['Start_Time'].apply(parse_t)))
    df_evo['Clock_T'] = (df_evo['Subject'].map(start_map) + df_evo['Time_h']) % 24
    
    hr_j_m = df_evo[(df_evo['Clock_T'] >= 7) & (df_evo['Clock_T'] < 11)].groupby('Subject')['HR'].mean().reset_index()
    hr_j_a = df_evo[(df_evo['Clock_T'] >= 13) & (df_evo['Clock_T'] < 17)].groupby('Subject')['HR'].mean().reset_index()

    for col_idx, (name, mse_file, met_type, index_range, length_desc) in enumerate(configs):
        if not os.path.exists(mse_file):
            continue
            
        df_mse = pd.read_csv(mse_file)
        df_mse['Group'] = df_mse['Group'].str.strip().str.lower().replace({'control': 'Control', 'pd': 'PD', 'parkinson': 'PD', 'other': 'Control'})
        df_mse = df_mse[df_mse['Group'].isin(['Control', 'PD'])]
        
        # Subject Counts for Check
        n_controls = df_mse[df_mse['Group']=='Control']['Subject'].nunique()
        n_pd = df_mse[df_mse['Group']=='PD']['Subject'].nunique()
        
        # --- TOP ROW: MSE Curves (Full 1-20 context) ---
        ax1 = fig.add_subplot(gs[0, col_idx])
        add_panel_label(ax1, chr(65 + col_idx))
        
        for group in ['Control', 'PD']:
            group_data = df_mse[df_mse['Group'] == group]
            summary = group_data.groupby('Scales')['MSE'].agg(['mean', 'sem']).reset_index()
            ax1.plot(summary['Scales'], summary['mean'], color=colors[group], linewidth=3, marker='o', markersize=4, label=f"{group} (N={n_controls if group=='Control' else n_pd})")
            ax1.fill_between(summary['Scales'], summary['mean']-summary['sem'], summary['mean']+summary['sem'], color=colors[group], alpha=0.15)
            
        ax1.set_title(f"{name}\nMSE Spectrum", fontsize=20, fontweight='bold')
        ax1.set_xlabel("Scale Factor", fontsize=14)
        ax1.set_ylabel("Sample Entropy", fontsize=14)
        ax1.set_xlim(1, 20)
        ax1.set_xticks([1, 5, 10, 15, 20])
        ax1.grid(True, alpha=0.15)
        
        # Highlight range used
        ax1.axvspan(min(index_range), max(index_range), color='gray', alpha=0.1)
        ax1.legend(fontsize=9, loc='upper right')

        # --- BOTTOM ROW: HR-Normalized Complexity Index ---
        ax2 = fig.add_subplot(gs[1, col_idx])
        add_panel_label(ax2, chr(69 + col_idx))
        
        sub_index = df_mse[df_mse['Scales'].isin(index_range)].groupby(['Subject', 'Group'])['MSE'].mean().reset_index()
        
        if met_type == 'CHILE_METRICS':
            df_met = pd.read_csv(CHILE_METRICS)
            df_met['HR'] = 60000.0 / df_met['HRV_MeanNN']
            df_final = pd.merge(sub_index, df_met[['Subject', 'HR']], on='Subject')
        elif met_type == 'SPAIN_METRICS':
            df_met = pd.read_csv(SPAIN_METRICS)
            df_met['HR'] = 60000.0 / df_met['HRV_MeanNN']
            df_final = pd.merge(sub_index, df_met[['Subject', 'HR']], on='Subject')
        elif met_type == 'JAPAN_EVO_MORNING':
            df_final = pd.merge(sub_index, hr_j_m, on='Subject')
        elif met_type == 'JAPAN_EVO_AFTERNOON':
            df_final = pd.merge(sub_index, hr_j_a, on='Subject')
            
        df_final['Norm'] = df_final['MSE'] / df_final['HR']
        
        sns.boxplot(x='Group', y='Norm', data=df_final, palette=colors, ax=ax2, order=['Control', 'PD'], showfliers=False, width=0.5)
        sns.stripplot(x='Group', y='Norm', data=df_final, palette=colors, ax=ax2, order=['Control', 'PD'], alpha=0.6, color='black', edgecolor='white', linewidth=0.5)
        
        c_vals = df_final[df_final['Group'] == 'Control']['Norm'].dropna()
        p_vals = df_final[df_final['Group'] == 'PD']['Norm'].dropna()
        t, p = ttest_ind(c_vals, p_vals)
        
        range_str = f"$\Sigma$MSE({min(index_range)}-{max(index_range)}) / HR"
        ax2.set_title(f"HR-Normalized Complexity\n{range_str}", fontsize=18, fontweight='bold')
        ax2.set_ylabel("Complexity Index (normalized)", fontsize=14)
        ax2.set_xlabel(f"({length_desc})", fontsize=12, style='italic')
        ax2.grid(True, axis='y', alpha=0.15)
        
        ax2.text(0.5, 0.92, f"p = {p:.4f} {get_sig_chars(p)}", transform=ax2.transAxes, ha='center', fontsize=16, fontweight='bold')
        sns.despine(ax=ax2)

    # RULE TEXT LABEL (Rule: N/Tau >= 200)
    rule_text = r"Rule: $N/\tau \geq 200$ for stable entropy estimation"
    fig.text(0.5, 0.02, rule_text, ha='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.suptitle("Figure 4: Global Validation using Signal-Length Adjusted Complexity Metrics", fontsize=32, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    out_path = os.path.join(FIGURES_DIR, "Figure4", "Figure4.png")
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 4 updated and saved to {out_path}")

if __name__ == "__main__":
    generate_figure4()
