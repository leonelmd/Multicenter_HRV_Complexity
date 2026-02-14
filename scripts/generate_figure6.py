#!/usr/bin/env python3
"""
Figure 6: Global Validation of Age-Independency of Cardiac Complexity
- Analysis of HR-Normalized Complexity vs Age across validation centers.
- Standardized range rule applied: Chile/Spain (1-5), Japan (1-20).
- Includes Morning and Afternoon Japanese cohorts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# PATH RESOLUTION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def add_panel_label(ax, label):
    ax.text(-0.08, 1.12, label, transform=ax.transAxes, fontsize=28, fontweight='bold', va='bottom', ha='right')

def format_corr_text(r, p):
    if np.isnan(r) or np.isnan(p):
        return "ρ=n/a (p=n/a)"
    return f"ρ={r:.2f} (p={p:.2f})"

def generate_figure6():
    colors = {'Control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # LOAD JAPANESE METADATA ONCE
    df_j_meta = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    if 'Subject_ID' in df_j_meta.columns: df_j_meta = df_j_meta.rename(columns={'Subject_ID': 'Subject'})
    df_j_meta['Group'] = df_j_meta['Group'].str.upper().replace({'CONTROL':'Control', 'PD':'PD'})
    
    def parse_t(s): 
        try: parts = str(s).split(':'); return int(parts[0]) + int(parts[1])/60.0 + int(parts[2])/3600.0
        except: return np.nan
    start_map = dict(zip(df_j_meta['Subject'], df_j_meta['Start_Time'].apply(parse_t)))
    df_j_evo = pd.read_csv(os.path.join(DATA_DIR, "japan_evolution.csv"))
    df_j_evo['Clock_T'] = (df_j_evo['Subject'].map(start_map) + df_j_evo['Time_h']) % 24

    # PANELS
    configs = [
        ('A', 'CETRAM (Chile)', 'chile_mse.csv', 'chile_demographics.csv', 'chile_metrics.csv', range(1, 6), '15m Rest (MSE 1-5)', 'CHILE'),
        ('B', 'Cruces (Spain)', 'spain_mse.csv', 'spain_demographics.csv', 'spain_metrics.csv', range(1, 6), '15m Slice (MSE 1-5)', 'SPAIN'),
        ('C', 'Nagoya (Morning)', 'japan_morning_mse.csv', None, None, range(1, 21), '4h Block (MSE 1-20)', 'JAPAN_M'),
        ('D', 'Nagoya (Afternoon)', 'japan_afternoon_mse.csv', None, None, range(1, 21), '4h Block (MSE 1-20)', 'JAPAN_A')
    ]

    for i, (label, name, mse_file, dem_file, met_file, scale_range, scale_desc, ds_type) in enumerate(configs):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        add_panel_label(ax, label)
        
        if ds_type == 'CHILE':
            df_mse = pd.read_csv(os.path.join(DATA_DIR, mse_file))
            df_dem = pd.read_csv(os.path.join(DATA_DIR, dem_file)).rename(columns={'Anon_ID':'Subject'})
            df_met = pd.read_csv(os.path.join(DATA_DIR, met_file))
            hr = (60000.0 / df_met.set_index('Subject')['HRV_MeanNN']).reset_index().rename(columns={'HRV_MeanNN':'HR'})
            comp = df_mse[df_mse.Scales.isin(scale_range)].groupby('Subject').MSE.mean().reset_index()
            df = pd.merge(pd.merge(comp, df_dem, on='Subject'), hr, on='Subject')
            df['Group'] = df['Group'].str.upper().replace({'CONTROL':'Control', 'PD':'PD', 'PARKINSON':'PD'})

        elif ds_type == 'SPAIN':
            df_mse = pd.read_csv(os.path.join(DATA_DIR, mse_file))
            df_dem = pd.read_csv(os.path.join(DATA_DIR, dem_file))
            df_met = pd.read_csv(os.path.join(DATA_DIR, met_file))
            hr = (60000.0 / df_met.set_index('Subject')['HRV_MeanNN']).reset_index().rename(columns={'HRV_MeanNN':'HR'})
            comp = df_mse[df_mse.Scales.isin(scale_range)].groupby('Subject').MSE.mean().reset_index()
            # In Spain, df_dem has Group. comp doesn't have it unless we add it. 
            df = pd.merge(pd.merge(comp, df_dem, on='Subject'), hr, on='Subject')
            df['Group'] = df['Group'].str.upper().replace({'CONTROL':'Control', 'PD':'PD', 'OTHER':'Control'})

        elif 'JAPAN' in ds_type:
            start_h, end_h = (7, 11) if 'M' in ds_type else (13, 17)
            hr_j = df_j_evo[(df_j_evo['Clock_T'] >= start_h) & (df_j_evo['Clock_T'] < end_h)].groupby('Subject')['HR'].mean().reset_index()
            df_mse = pd.read_csv(os.path.join(DATA_DIR, mse_file))
            comp = df_mse[df_mse.Scales.isin(scale_range)].groupby('Subject').MSE.mean().reset_index()
            df = pd.merge(pd.merge(comp, df_j_meta[['Subject','Group','Age']], on='Subject'), hr_j, on='Subject')
            df['Group'] = df['Group'].replace({'CONTROL':'Control', 'PD':'PD'})

        df['Norm'] = df['MSE'] / df['HR']
        df = df.dropna(subset=['Age', 'Norm'])

        for grp in ['Control', 'PD']:
            grp_data = df[df['Group'] == grp]
            if len(grp_data) < 2: continue
            
            sns.regplot(x='Age', y='Norm', data=grp_data, ax=ax, 
                        color=colors[grp], label=f"{grp} (N={len(grp_data)})", 
                        scatter_kws={'s': 100, 'alpha': 0.6, 'edgecolor': 'white'},
                        line_kws={'linewidth': 3})
            
            r, p = spearmanr(grp_data['Age'], grp_data['Norm'])
            ax.text(0.05, 0.95 - (0.08 if grp=='PD' else 0), 
                    f"{grp}: {format_corr_text(r, p)}", 
                    transform=ax.transAxes, color=colors[grp], fontweight='bold', fontsize=16)

        ax.set_title(f"{name}\n{scale_desc}", fontsize=22, fontweight='bold')
        ax.set_xlabel("Age (years)", fontsize=18)
        ax.set_ylabel("HR-Normalized Complexity", fontsize=18)
        ax.grid(True, alpha=0.15)
        ax.legend(loc='lower left', fontsize=12)
        sns.despine(ax=ax)

    plt.suptitle("Figure 6: Multi-Center Validation of Age-Independency", fontsize=36, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path = os.path.join(FIGURES_DIR, "Figure6", "Figure6.png")
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 6 finalized and saved to {out_path}")

if __name__ == "__main__":
    generate_figure6()
