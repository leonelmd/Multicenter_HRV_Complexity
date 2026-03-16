#!/usr/bin/env python3
"""
Multicenter Cardiac Autonomic Complexity Study
Figure Generation Pipeline

Regenerates all manuscript figures (1–8) from data in data/ and figures/Figure8/.
Figure 8 requires statistical pre-processing; this pipeline runs those steps
automatically (steps 1–5) before generating the figure.

Usage (from the public_release/ root):
    python scripts/run_pipeline.py

What this does NOT do:
    - Compute MSE/HRV features from raw RRI signals (raw signals are not shared).
"""

import os
import subprocess
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # public_release/


def run(command, description):
    print(f"\n>>> {description}...")
    try:
        subprocess.check_call(command, shell=True, cwd=PROJECT_ROOT)
        print(f"    OK")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! FAILED: {description}\n    {e}")
        sys.exit(1)


def main():
    print("=" * 70)
    print("  MULTICENTER CARDIAC AUTONOMIC COMPLEXITY — FIGURE PIPELINE")
    print("=" * 70)
    print(f"\n  Working directory : {PROJECT_ROOT}")
    print(f"  Figures will be saved to: {os.path.join(PROJECT_ROOT, 'figures')}\n")
    print("=" * 70)

    steps = [
        # ── Figures 1–7: read directly from data/ ─────────────────────────────
        ("python scripts/generate_figure1.py",
         "Figure 1 — Study design & cohort demographics"),
        ("python scripts/generate_figure2.py",
         "Figure 2 — Signal archetypes & HR/age distributions  "
         "[NOTE: RRi trace panels require raw data — see README]"),
        ("python scripts/generate_figure3.py",
         "Figure 3 — Circadian dynamics of cardiac complexity (Nagoya)"),
        ("python scripts/generate_figure4.py",
         "Figure 4 — Multiscale entropy comparison across centers"),
        ("python scripts/generate_figure5.py",
         "Figure 5 — Diagnostic performance & biomarker independence"),
        ("python scripts/generate_figure6.py",
         "Figure 6 — Age-independency validation"),
        ("python scripts/generate_figure7.py",
         "Figure 7 — ML validation: handcrafted features vs deep learning  "
         "[NOTE: runs LOCO cross-validation, takes ~1 min]"),

        # ── Figure 8 pre-processing: statistical analyses → figures/Figure8/ ──
        ("python scripts/traditional_hrv_metrics.py",
         "Figure 8 pre-step 1/5 — Consolidate HRV metrics across centers"),
        ("python scripts/multiscale_decomposition.py",
         "Figure 8 pre-step 2/5 — Scale-metric correlations & MSE curves"),
        ("python scripts/complexity_correlation_analysis.py",
         "Figure 8 pre-step 3/5 — Spearman correlations + bootstrap CIs"),
        ("python scripts/incremental_value_analysis.py",
         "Figure 8 pre-step 4/5 — Hierarchical regression & variance decomposition"),
        ("python scripts/cross_dataset_consistency.py",
         "Figure 8 pre-step 5/5 — Cross-dataset consistency & physiological interpretation"),

        # ── Figure 8: composite autonomic physiology figure ────────────────────
        ("python scripts/generate_figure8.py",
         "Figure 8 — Autonomic physiology of cardiac complexity (composite)"),
    ]

    for cmd, desc in steps:
        run(cmd, desc)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated figures:")
    for i in range(1, 10):
        figdir = os.path.join(PROJECT_ROOT, "figures", f"Figure{i}")
        if os.path.isdir(figdir):
            pngs = [f for f in os.listdir(figdir) if f.endswith(".png")]
            print(f"  Figure {i}: {', '.join(sorted(pngs))}")
    print()


if __name__ == "__main__":
    main()
