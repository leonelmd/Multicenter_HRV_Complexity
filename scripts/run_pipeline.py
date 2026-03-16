#!/usr/bin/env python3
"""
Multicenter Cardiac Autonomic Complexity Study
Figure Generation Pipeline

Regenerates all manuscript figures (1–9) from the pre-computed data in data/
and the statistical results in results/.

Usage (from the public_release/ root):
    python scripts/run_pipeline.py

What this does NOT do:
    - Compute MSE/HRV features from raw RRI signals (raw signals are not shared).
    - Re-run statistical analyses. To regenerate results/ CSVs, run the
      scripts in analysis/ (see README.md for details and ordering).
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
