#!/usr/bin/env python3
"""
Figure 2: Signal Archetypes - Multi-Center Data Characterization
Top Row: RRi traces (overlaid examples with group averages)
Second Row: Mean HR distributions (Violin)
Third Row: Age distributions (Violin)
Bottom Row: Poincaré plot examples
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind

# PORTABLE PATH RESOLUTION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # HRV-Complexity/

# Data paths
CHILE_METRICS = os.path.join(DATA_DIR, "chile_metrics.csv")
CHILE_MSE = os.path.join(DATA_DIR, "chile_mse.csv")
SPAIN_METRICS = os.path.join(DATA_DIR, "spain_metrics.csv")
JAPAN_DAY_METRICS = os.path.join(DATA_DIR, "japan_day_metrics.csv")
JAPAN_EVOLUTION = os.path.join(DATA_DIR, "japan_evolution.csv")
JAPAN_META = os.path.join(DATA_DIR, "japan_metadata.csv")

# Source data paths (for RRi traces)
CETRAM_RRI_DIR = os.path.join(PROJECT_ROOT, "CETRAM/public_release/data/processed/hrv_signals")
CRUCES_RRI_DIR = os.path.join(PROJECT_ROOT, "Cruces/public_release/data/processed/RRi")
NAGOYA_RRI_DIR = os.path.join(PROJECT_ROOT, "Nagoya/public_release/data/processed_rri")

def add_panel_label(ax, label):
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')

def get_poincare_stats(rri):
    """Calculate SD1 and SD2 for Poincaré plot"""
    if len(rri) < 2:
        return np.nan, np.nan
    diff_rri = np.diff(rri)
    sd1 = np.std(diff_rri) / np.sqrt(2)
    sd2_vals = [(rri[i] + rri[i+1]) for i in range(len(rri)-1)]
    sd2 = np.std(sd2_vals) / np.sqrt(2)
    return sd1, sd2

def generate_figure2():
    colors = {'Control': '#2E86AB', 'PD': '#D62828'}
    sns.set_style("ticks")
    
    # Increased height for 4 rows
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1.2, 1, 1, 1.2])
    
    centers = [
        ('CETRAM\n(Chile)', CHILE_METRICS, CETRAM_RRI_DIR, 'cetram'),
        ('Cruces\n(Spain)', SPAIN_METRICS, CRUCES_RRI_DIR, 'cruces'),
        ('Nagoya\n(Japan)', JAPAN_META, NAGOYA_RRI_DIR, 'nagoya')
    ]
    
    for col_idx, (name, metrics_file, rri_dir, center_type) in enumerate(centers):
        # Load metrics/metadata
        df_metrics = pd.read_csv(metrics_file)
        # Clean column names
        df_metrics.columns = df_metrics.columns.str.strip()
        
        if 'Group' in df_metrics.columns:
            df_metrics['Group'] = df_metrics['Group'].astype(str).str.strip()
            df_metrics['Group'] = df_metrics['Group'].str.replace('control', 'Control', case=False).str.replace('pd', 'PD', case=False)
            df_metrics = df_metrics[df_metrics['Group'].isin(['Control', 'PD'])]
            
        if 'Subject' not in df_metrics.columns:
            if 'Subject_ID' in df_metrics.columns:
                df_metrics = df_metrics.rename(columns={'Subject_ID': 'Subject'})
            elif 'Anon_ID' in df_metrics.columns:
                df_metrics = df_metrics.rename(columns={'Anon_ID': 'Subject'})
            else:
                print(f"ERROR: 'Subject' column not found in {metrics_file}")
                continue
            
        ctrl_subjects = df_metrics[df_metrics['Group'] == 'Control']['Subject'].tolist()
        pd_subjects = df_metrics[df_metrics['Group'] == 'PD']['Subject'].tolist()
        
        # Load Function
        def load_rri(subject, center_type, rri_dir):
            try:
                if center_type == 'cetram':
                    rri_file = os.path.join(rri_dir, f"{subject}_rri.csv")
                    if os.path.exists(rri_file):
                        df = pd.read_csv(rri_file)
                        if 'RRI_ms' in df.columns: return df['RRI_ms'].values
                        return df.values.flatten()
                elif center_type == 'nagoya':
                    rri_file = os.path.join(rri_dir, f"{subject}_RRi.txt")
                    if os.path.exists(rri_file):
                        df = pd.read_csv(rri_file, sep=' ', header=None, names=['timestamp', 'rri'])
                        return df['rri'].values * 1000
                else:
                    rri_file = os.path.join(rri_dir, f"{subject}.csv")
                    if os.path.exists(rri_file):
                        df = pd.read_csv(rri_file, header=None)
                        return df.values.flatten()
            except: return None
            return None

        def get_start_hour(subj):
            try:
                if 'Start_Time' in df_metrics.columns:
                    row = df_metrics[df_metrics['Subject'] == subj]
                    if not row.empty:
                        t_str = str(row.iloc[0]['Start_Time'])
                        h, m, s = map(int, t_str.split(':'))
                        return h + m/60.0 + s/3600.0
            except: pass
            return 0.0

        # === ROW 1: RRi Traces ===
        ax1 = fig.add_subplot(gs[0, col_idx])
        add_panel_label(ax1, chr(65 + col_idx))
        
        if center_type == 'nagoya':
            common_len = 1000 
            x_common = np.linspace(8, 40, common_len) 
        else:
            common_len = 1000
            if center_type == 'cetram': x_common = np.linspace(0, 300, common_len)
            else: x_common = np.linspace(0, 450, common_len)
            
        group_averages = {'Control': [], 'PD': []}
        
        for grp, subjs, c in [('Control', ctrl_subjects, colors['Control']), ('PD', pd_subjects, colors['PD'])]:
            n_plotted = 0
            for subj in subjs:
                rri = load_rri(subj, center_type, rri_dir)
                if rri is not None and len(rri) > 50: 
                    valid_mask = (rri >= 300) & (rri <= 2000)
                    rri_clean = rri[valid_mask]
                    if len(rri_clean) < 50: continue

                    if center_type == 'nagoya':
                        start_h = get_start_hour(subj)
                        step = max(1, len(rri) // 4000)
                        rri_plot = rri[::step]
                        duration_h_raw = (np.sum(rri) / 1000.0) / 3600.0
                        if duration_h_raw < 1: duration_h_raw = 24.0
                        t_elapsed = np.linspace(0, duration_h_raw, len(rri_plot))
                        t_plot = start_h + t_elapsed
                        
                        ax1.plot(t_plot, rri_plot, color=c, alpha=0.1, linewidth=0.5)
                        
                        duration_h_clean = (np.sum(rri_clean) / 1000.0) / 3600.0
                        t_full_clean = start_h + np.linspace(0, duration_h_clean, len(rri_clean))
                        rri_interp = np.interp(x_common, t_full_clean, rri_clean, left=np.nan, right=np.nan)
                        if len(t_full_clean) > 0:
                            rri_interp[x_common < t_full_clean[0]] = np.nan
                            rri_interp[x_common > t_full_clean[-1]] = np.nan
                        group_averages[grp].append(rri_interp)
                    else:
                        rri_plot = rri[:min(1200, len(rri))] 
                        t = np.cumsum(rri_plot) / 1000.0
                        ax1.plot(t, rri_plot, color=c, alpha=0.1, linewidth=0.5)
                        
                        t_full_clean = np.cumsum(rri_clean) / 1000.0
                        rri_interp = np.interp(x_common, t_full_clean, rri_clean, left=np.nan, right=np.nan)
                        if len(t_full_clean) > 0:
                            rri_interp[x_common > t_full_clean[-1]] = np.nan
                        else: rri_interp[:] = np.nan
                        group_averages[grp].append(rri_interp)
                    n_plotted += 1

        for grp, c in colors.items():
            if len(group_averages[grp]) > 0:
                stack = np.array(group_averages[grp])
                with np.errstate(invalid='ignore'):
                    avg_trace = np.nanmean(stack, axis=0)
                ax1.plot(x_common, avg_trace, color=c, linewidth=2.5)

        ax1.set_title(f"{name}\n(Control={len(ctrl_subjects)}, PD={len(pd_subjects)})", fontsize=14, fontweight='bold')
        ax1.set_ylabel('RR Interval (ms)', fontsize=12)
        
        if center_type == 'nagoya':
            ax1.set_xlabel('Time of Day', fontsize=11)
            ax1.set_xlim(8, 40)
            ax1.set_xticks([12, 18, 24, 30, 36])
            ax1.set_xticklabels(['12pm', '6pm', '12am', '6am', '12pm'])
            ax1.axvline(24, color='black', linestyle='--', alpha=0.3)
        else:
            ax1.set_xlabel('Time (s)', fontsize=11)
            if center_type == 'cetram': ax1.set_xlim(0, 300)
            else: ax1.set_xlim(0, 450)
        ax1.set_ylim(400, 1400)
        ax1.grid(True, alpha=0.2)
        sns.despine(ax=ax1)

        # Calculate MeanRRi_ms (Global)
        if 'MeanRRi_ms' not in df_metrics.columns:
            if 'HRV_MeanNN' in df_metrics.columns:
                df_metrics['MeanRRi_ms'] = df_metrics['HRV_MeanNN']
            elif 'MeanNN' in df_metrics.columns:
                df_metrics['MeanRRi_ms'] = df_metrics['MeanNN']
            elif 'Mean_RRi' in df_metrics.columns:
                df_metrics['MeanRRi_ms'] = df_metrics['Mean_RRi']
            elif 'Avg_RRi' in df_metrics.columns:
                df_metrics['MeanRRi_ms'] = df_metrics['Avg_RRi'] * 1000
            elif 'HR' in df_metrics.columns:
                df_metrics['MeanRRi_ms'] = 60000.0 / df_metrics['HR']
            else:
                means = []
                for _, row in df_metrics.iterrows():
                    rri = load_rri(row['Subject'], center_type, rri_dir)
                    means.append(np.mean(rri) if rri is not None else np.nan)
                df_metrics['MeanRRi_ms'] = means
        
        # Calculate HR for plotting
        if 'HR' not in df_metrics.columns:
             df_metrics['HR'] = 60000.0 / df_metrics['MeanRRi_ms']

        # === ROW 2: Mean HR Distribution ===
        ax2 = fig.add_subplot(gs[1, col_idx])
        add_panel_label(ax2, chr(68 + col_idx))
        
        if 'HR' in df_metrics.columns and df_metrics['HR'].notna().sum() > 0:
            sns.violinplot(x='Group', y='HR', data=df_metrics, palette=colors,
                          order=['Control', 'PD'], inner='points', ax=ax2)
            
            c_v = df_metrics[df_metrics['Group'] == 'Control']['HR'].dropna()
            p_v = df_metrics[df_metrics['Group'] == 'PD']['HR'].dropna()
            if len(c_v) > 1 and len(p_v) > 1:
                t, p = ttest_ind(c_v, p_v)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                ax2.text(0.5, 0.9, f'p={p:.3f} ({sig})', transform=ax2.transAxes, ha='center', fontweight='bold', fontsize=12)
                
            ax2.set_ylabel('Mean HR (bpm)', fontsize=12)
            ax2.set_title('Heart Rate Distribution', fontsize=13, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'HR Data Unavailable', ha='center', va='center', transform=ax2.transAxes)
        # Standardize HR Scale
        ax2.set_ylim(40, 130)
        ax2.grid(True, axis='y', alpha=0.2)
        sns.despine(ax=ax2)

        # === ROW 3: Age Distribution ===
        ax3 = fig.add_subplot(gs[2, col_idx])
        add_panel_label(ax3, chr(71 + col_idx))
        
        age_col = 'Age'
        try:
            if center_type == 'cetram':
                dem_path = os.path.join(DATA_DIR, "chile_demographics.csv")
                if os.path.exists(dem_path):
                    dem = pd.read_csv(dem_path)
                    if 'Age' not in df_metrics.columns:
                        df_metrics = pd.merge(df_metrics, dem[['Anon_ID', 'Age']], left_on='Subject', right_on='Anon_ID', how='left')
            elif center_type == 'cruces':
                dem_path = os.path.join(DATA_DIR, "spain_demographics.csv")
                if os.path.exists(dem_path):
                    dem = pd.read_csv(dem_path)
                    if 'Age' not in df_metrics.columns:
                        df_metrics = pd.merge(df_metrics, dem[['Subject', 'Age']], on='Subject', how='left')
            elif center_type == 'nagoya':
                dem_path = os.path.join(DATA_DIR, "japan_metadata.csv")
                if os.path.exists(dem_path):
                    dem = pd.read_csv(dem_path)
                    if 'Age' not in df_metrics.columns:
                        df_metrics = pd.merge(df_metrics, dem[['Subject_ID', 'Age']], left_on='Subject', right_on='Subject_ID', how='left')
        except: pass

        if age_col in df_metrics.columns and df_metrics[age_col].notna().sum() > 0:
            sns.violinplot(x='Group', y=age_col, data=df_metrics, palette=colors,
                          order=['Control', 'PD'], inner='points', ax=ax3)
            
            c_v = df_metrics[df_metrics['Group'] == 'Control'][age_col].dropna()
            p_v = df_metrics[df_metrics['Group'] == 'PD'][age_col].dropna()
            
            if len(c_v) > 1 and len(p_v) > 1:
                t, p = ttest_ind(c_v, p_v)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                ax3.text(0.5, 0.9, f'p={p:.3f} ({sig})', transform=ax3.transAxes, ha='center', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Age (years)', fontsize=12)
            ax3.set_title('Age Distribution', fontsize=13, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Age Data Unavailable', ha='center', va='center', transform=ax3.transAxes)
        # Standardize Age Scale
        ax3.set_ylim(20, 95)
        ax3.grid(True, axis='y', alpha=0.2)
        sns.despine(ax=ax3)

        # === ROW 4: Poincaré Examples ===
        ax4 = fig.add_subplot(gs[3, col_idx])
        add_panel_label(ax4, chr(74 + col_idx)) # J, K, L
        
        poincare_data = []
        for group, subjects, color in [('Control', ctrl_subjects, colors['Control']), 
                                       ('PD', pd_subjects, colors['PD'])]:
            if len(subjects) > 0:
                grp_mean_rri = df_metrics[df_metrics['Group'] == group]['MeanRRi_ms'].mean()
                best_sub = None
                min_diff = 9999
                for subj in subjects: 
                    try:
                        sub_mean = df_metrics[df_metrics['Subject'] == subj]['MeanRRi_ms'].values[0]
                        if abs(sub_mean - grp_mean_rri) < min_diff:
                            rri = load_rri(subj, center_type, rri_dir)
                            if rri is not None and len(rri) > 50:
                                min_diff = abs(sub_mean - grp_mean_rri)
                                best_sub = subj
                    except: continue
                if best_sub:
                    rri = load_rri(best_sub, center_type, rri_dir)
                    rri_clean = rri[(rri >= 400) & (rri <= 1400)]
                    poincare_data.append((rri_clean, color, group))
        
        for rri_clean, color, group in poincare_data:
            if len(rri_clean) > 2000: rri_plot = rri_clean[::len(rri_clean)//2000]
            else: rri_plot = rri_clean
            ax4.scatter(rri_plot[:-1], rri_plot[1:], c=color, s=8, alpha=0.3, label=group)
            sd1, sd2 = get_poincare_stats(rri_plot)
            mean_rri = np.mean(rri_plot)
            if not np.isnan(sd1):
                ellipse = Ellipse((mean_rri, mean_rri), width=sd2*4, height=sd1*4, angle=45,
                                 edgecolor=color, facecolor='none', linewidth=2.5)
                ax4.add_patch(ellipse)
        
        ax4.set_title('Poincaré Plot', fontsize=13, fontweight='bold')
        ax4.set_xlabel('RRi[n] (ms)', fontsize=11)
        ax4.set_ylabel('RRi[n+1] (ms)', fontsize=11)
        ax4.set_xlim(400, 1400); ax4.set_ylim(400, 1400)
        ax4.plot([400, 1400], [400, 1400], 'k--', alpha=0.3)
        ax4.set_aspect('equal')
        sns.despine(ax=ax4)
    
    plt.suptitle('Figure 2: Multi-Center Signal Archetypes & Demographics', fontsize=26, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_dir = os.path.join(FIGURES_DIR, "Figure2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Figure2.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Figure 2 saved to {out_path}")

if __name__ == "__main__":
    generate_figure2()
