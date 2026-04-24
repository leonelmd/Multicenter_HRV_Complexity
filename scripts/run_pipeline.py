#!/usr/bin/env python3
"""
Multicenter Cardiac Autonomic Complexity Study
Figure Generation Pipeline

Regenerates all manuscript figures (1–7) from data in data/ and figures/Figure7/.
Figure 7 requires statistical pre-processing; this pipeline runs those steps
automatically (steps 1–5) before generating the figure.

Usage (from the public_release/ root):
    python scripts/run_pipeline.py           # sync data then regenerate all figures
    python scripts/run_pipeline.py --no-sync # skip sync (use existing data/ as-is)

What this does NOT do:
    - Compute MSE/HRV features from raw RRI signals (raw signals are not shared).
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip data sync (use existing data/ as-is)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MULTICENTER CARDIAC AUTONOMIC COMPLEXITY — FIGURE PIPELINE")
    print("=" * 70)
    print(f"\n  Working directory : {PROJECT_ROOT}")
    print(f"  Figures will be saved to: {os.path.join(PROJECT_ROOT, 'figures')}\n")
    print("=" * 70)

    if not args.no_sync:
        print("\n>>> Syncing data from center pipelines...")
        try:
            subprocess.check_call(
                f"{sys.executable} scripts/sync_center_data.py --apply",
                shell=True, cwd=PROJECT_ROOT
            )
        except subprocess.CalledProcessError as e:
            print(f"\n!!! Sync failed: {e}")
            print("    Run with --no-sync to skip sync and use existing data/.")
            sys.exit(1)
    else:
        print("\n  [--no-sync] Skipping data sync, using existing data/ files.")

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
         "Figure 5 — Diagnostic performance, ML validation & feature independence  "
         "[NOTE: runs LOCO cross-validation, takes ~1 min]"),
        ("python scripts/generate_figure6.py",
         "Figure 6 — Age-independency validation"),

        # ── Figure 7 pre-processing: statistical analyses → figures/Figure7/ ──
        ("python scripts/traditional_hrv_metrics.py",
         "Figure 7 pre-step 1/5 — Consolidate HRV metrics across centers"),
        ("python scripts/multiscale_decomposition.py",
         "Figure 7 pre-step 2/5 — Scale-metric correlations & MSE curves"),
        ("python scripts/complexity_correlation_analysis.py",
         "Figure 7 pre-step 3/5 — Spearman correlations + bootstrap CIs"),
        ("python scripts/incremental_value_analysis.py",
         "Figure 7 pre-step 4/5 — Hierarchical regression & variance decomposition"),
        ("python scripts/cross_dataset_consistency.py",
         "Figure 7 pre-step 5/5 — Cross-dataset consistency & physiological interpretation"),

        # ── Figure 7: composite autonomic physiology figure ────────────────────
        ("python scripts/generate_figure7.py",
         "Figure 7 — Autonomic physiology of cardiac complexity (composite)"),

        # ── Figure 8: clinical correlation (CETRAM PD cohort) ─────────────────
        ("python scripts/generate_figure8.py",
         "Figure 8 — Clinical correlation: complexity vs CISI-PD, H&Y, non-motor symptoms"),

        # ── Appendix: cross-center generalization matrix ───────────────────────
        ("python scripts/generate_appendix.py",
         "Appendix 1 — Cross-center generalization matrix (RF, LOCO)"),

        # ── Appendix 2: per-scale AUC, ECG vs PPG modality ───────────────────
        ("python scripts/generate_appendix2.py",
         "Appendix 2 — Per-scale AUC across modalities (ECG vs PPG)"),
    ]

    for cmd, desc in steps:
        run(cmd, desc)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated figures:")
    for i in range(1, 9):
        figdir = os.path.join(PROJECT_ROOT, "figures", f"Figure{i}")
        if os.path.isdir(figdir):
            pngs = [f for f in os.listdir(figdir) if f.endswith(".png")]
            print(f"  Figure {i}: {', '.join(sorted(pngs))}")
    print()


if __name__ == "__main__":
    main()
