#!/usr/bin/env python3
"""
Figure 5: Comparative Diagnostic Performance and Feature Independence
- AUC Comparison across centers (Complexity vs Standard Metrics)
- Multi-center ROC curves for the primary biomarker
- Correlation analysis showing the distinct nature of fractal complexity
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def add_panel_label(ax, label):
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='bottom', ha='right')

def get_auc(df, metric, group_col='Group', pos_label='PD'):
    try:
        sub = df[[group_col, metric]].dropna()
        y_true = (sub[group_col].str.lower() == pos_label.lower()).astype(int)
        y_score = sub[metric]
        auc = roc_auc_score(y_true, y_score)
        if auc < 0.5: auc = 1 - auc
        return auc
    except:
        return np.nan

def generate_figure5():
    sns.set_style("ticks")
    fig = plt.figure(figsize=(24, 26))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
    
    datasets = {}
    
    # 1. LOAD AND PREPARE DATA
    # --- CHILE ---
    df_c_mse = pd.read_csv(os.path.join(DATA_DIR, "chile_mse.csv"))
    df_c_met = pd.read_csv(os.path.join(DATA_DIR, "chile_metrics.csv"))
    df_c_mse['Group'] = df_c_mse['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    df_c_met['Group'] = df_c_met['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD'})
    comp_c = df_c_mse[df_c_mse.Scales.isin(range(1,6))].groupby('Subject').MSE.mean().reset_index()
    hr_c = 60000.0 / df_c_met.set_index('Subject')['HRV_MeanNN']
    df_c = pd.merge(comp_c, df_c_met[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_DFA_alpha1']], on='Subject')
    df_c['HR'] = df_c['Subject'].map(hr_c)
    df_c['Complexity Index (HR-Norm)'] = df_c['MSE'] / df_c['HR']
    df_c['Sample Entropy (S1)'] = df_c['Subject'].map(df_c_mse[df_c_mse.Scales==1].set_index('Subject')['MSE'].to_dict())
    datasets['CETRAM'] = df_c

    # --- SPAIN ---
    df_s_mse = pd.read_csv(os.path.join(DATA_DIR, "spain_mse.csv"))
    df_s_met = pd.read_csv(os.path.join(DATA_DIR, "spain_metrics.csv"))
    df_s_mse['Group'] = df_s_mse['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    df_s_met['Group'] = df_s_met['Group'].str.lower().replace({'pd':'PD','control':'Control','parkinson':'PD','other':'Control'})
    comp_s = df_s_mse[df_s_mse.Scales.isin(range(1,6))].groupby('Subject').MSE.mean().reset_index()
    hr_s = 60000.0 / df_s_met.set_index('Subject')['HRV_MeanNN']
    df_s = pd.merge(comp_s, df_s_met[['Subject','Group','HRV_SDNN','HRV_RMSSD','HRV_DFA_alpha1']], on='Subject')
    df_s['HR'] = df_s['Subject'].map(hr_s)
    df_s['Complexity Index (HR-Norm)'] = df_s['MSE'] / df_s['HR']
    df_s['Sample Entropy (S1)'] = df_s['Subject'].map(df_s_mse[df_s_mse.Scales==1].groupby('Subject').MSE.mean().to_dict())
    datasets['Cruces'] = df_s

    # --- JAPAN ---
    df_j_evo = pd.read_csv(os.path.join(DATA_DIR, "japan_evolution.csv"))
    df_j_meta = pd.read_csv(os.path.join(DATA_DIR, "japan_metadata.csv"))
    if 'Subject_ID' in df_j_meta.columns: df_j_meta = df_j_meta.rename(columns={'Subject_ID': 'Subject'})
    # DFA_alpha1 from full 24h recording (per-subject, window-independent)
    df_j_recalc = pd.read_csv(os.path.join(DATA_DIR, "japan_recalc_metrics.csv"))[['Subject','DFA_alpha1']]
    # New evolution file uses Window_start_h (4h windows, proper nAUC)
    for win_name, win_h, mse_file in [
        ('Nagoya (07-11h)', 7,  'japan_morning_mse.csv'),
        ('Nagoya (16-20h)', 16, 'japan_afternoon_mse.csv')
    ]:
        df_j_mse = pd.read_csv(os.path.join(DATA_DIR, mse_file))
        df_j_met = df_j_evo[df_j_evo['Window_start_h'] == win_h][['Subject','Group','HR','SDNN','RMSSD']].copy()
        comp_j = df_j_mse[df_j_mse.Scales.isin(range(1,21))].groupby('Subject').MSE.mean().reset_index()
        df_j = pd.merge(comp_j, df_j_met, on='Subject')
        df_j = pd.merge(df_j, df_j_recalc, on='Subject', how='left')
        df_j['Complexity Index (HR-Norm)'] = df_j['MSE'] / df_j['HR']
        df_j['Sample Entropy (S1)'] = df_j['Subject'].map(df_j_mse[df_j_mse.Scales==1].set_index('Subject')['MSE'].to_dict())
        df_j = df_j.rename(columns={'SDNN':'HRV_SDNN', 'RMSSD':'HRV_RMSSD', 'DFA_alpha1':'HRV_DFA_alpha1'})
        datasets[win_name] = df_j

    # 2. AUC BAR CHARTS (4 Panels)
    metrics_to_compare = ['Complexity Index (HR-Norm)', 'HRV_DFA_alpha1', 'Sample Entropy (S1)', 'HRV_SDNN', 'HRV_RMSSD']
    metric_labels = ['Complexity (HR-Norm)', 'DFA Alpha 1', 'SampEn (S1)', 'SDNN', 'RMSSD']
    panel_order = ['CETRAM', 'Cruces', 'Nagoya (07-11h)', 'Nagoya (16-20h)']
    
    for i, ds_key in enumerate(panel_order):
        ax = fig.add_subplot(gs[0 if i < 3 else 1, i % 3])
        add_panel_label(ax, chr(65 + i))
        aucs = [get_auc(datasets[ds_key], m) for m in metrics_to_compare]
        # Filter out NaN (metric not available for this dataset)
        valid = [(a, metric_labels[j]) for j, a in enumerate(aucs) if not np.isnan(a)]
        valid.sort(key=lambda x: x[0], reverse=True)
        sorted_aucs = [v[0] for v in valid]
        sorted_labels = [v[1] for v in valid]
        
        # Color Logic: Complexity=Purple, Nonlinear=Blue, Linear=Green
        variable_colors = {
            'Complexity (HR-Norm)': '#8E44AD', # Purple
            'DFA Alpha 1': '#2980B9',          # Dark Blue
            'SampEn (S1)': '#5DADE2',          # Light Blue
            'SDNN': '#27AE60',                 # Dark Green
            'RMSSD': '#58D68D'                 # Light Green
        }
        colors = [variable_colors.get(l, '#95A5A6') for l in sorted_labels]
        color_map = dict(zip(sorted_labels, colors))
        sns.barplot(x=sorted_aucs, y=sorted_labels, hue=sorted_labels,
                    palette=color_map, legend=False, ax=ax)
        
        # Emphasize "Complexity" label
        for label in ax.get_yticklabels():
            if 'Complexity' in label.get_text():
                label.set_fontweight('bold')
                label.set_fontsize(15)
                label.set_color('#8E44AD')
        
        ax.set_title(f"{ds_key}", fontsize=18, fontweight='bold')
        ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color='black', ls='--', alpha=0.5)
        ax.set_xlabel("AUC")

    # 3. CONSOLIDATED ROC CURVES (Panel E) — spans cols 1–2 in row 1
    ax_roc = fig.add_subplot(gs[1, 1:])
    add_panel_label(ax_roc, 'E')
    roc_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C77DFF']
    for ds_key, color in zip(panel_order, roc_colors):
        df = datasets[ds_key]
        y_true = (df['Group'].str.lower() == 'pd').astype(int)
        scores = df['Complexity Index (HR-Norm)']
        if roc_auc_score(y_true, scores) < 0.5: scores = -scores
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = get_auc(df, 'Complexity Index (HR-Norm)')
        ax_roc.plot(fpr, tpr, label=f"{ds_key} ({auc:.2f})", color=color, linewidth=3)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title("Multi-center Primary ROC", fontsize=18, fontweight='bold')
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc='lower right', fontsize=10)

    # 4. CORRELATION HEATMAPS (Panels F–H) — one per center, row 2
    heatmap_centers = ['CETRAM', 'Cruces', 'Nagoya (16-20h)']
    panel_letters = ['F', 'G', 'H']
    for col, (center, letter) in enumerate(zip(heatmap_centers, panel_letters)):
        ax_corr = fig.add_subplot(gs[2, col])
        add_panel_label(ax_corr, letter)
        corr_df = datasets[center][metrics_to_compare].copy()
        corr_df.columns = metric_labels
        corr_matrix = corr_df.corr(method='spearman')
        show_cbar = (col == 2)
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax_corr, fmt='.2f',
                    cbar=show_cbar, annot_kws={'size': 9})
        ax_corr.set_title(f"Feature Orthogonality ({center})", fontsize=16, fontweight='bold')
        if col > 0:
            ax_corr.set_yticklabels([])

    plt.suptitle("Figure 5: Multi-center Diagnostic Performance and Biomarker Independence", fontsize=28, fontweight='bold', y=1.0)
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.08, right=0.97, hspace=0.45, wspace=0.3)
    
    out_path = os.path.join(FIGURES_DIR, "Figure5", "Figure5.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print("Figure 5 finalized.")

if __name__ == "__main__":
    generate_figure5()
