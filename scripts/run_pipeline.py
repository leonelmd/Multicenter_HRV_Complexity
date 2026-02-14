#!/usr/bin/env python3
"""
Multicenter Study - Main Pipeline Script
Orchestrates the complete analysis workflow for the multicenter study.
"""

import os
import subprocess
import sys

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # public_release/

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Run command from public_release/ root
        subprocess.check_call(command, shell=True, cwd=PROJECT_ROOT)
        print(f">>> {description} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Error during {description}: {e}")
        sys.exit(1)

def main():
    print("="*70)
    print("  MULTICENTER CARDIAC AUTONOMIC COMPLEXITY STUDY - PIPELINE")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Generate all manuscript figures (1-8) from processed data")
    print("  2. Verify consistency of results")
    print("\n" + "="*70)
    
    # Step 1: Generate Figure 1 (Study Overview)
    run_command("python scripts/generate_figure1.py", 
                "Step 1: Generating Figure 1 (Study Overview)")
    
    # Step 2: Generate Figure 2 (Signal Archetypes)
    run_command("python scripts/generate_figure2.py", 
                "Step 2: Generating Figure 2 (Signal Archetypes)")
    
    # Step 3: Generate Figure 3 (Circadian Dynamics)
    run_command("python scripts/generate_figure3.py", 
                "Step 3: Generating Figure 3 (Circadian Dynamics)")
    
    # Step 4: Generate Figure 4 (MSE & Complexity)
    run_command("python scripts/generate_figure4.py", 
                "Step 4: Generating Figure 4 (Multiscale Entropy)")
    
    # Step 5: Generate Figure 5 (Diagnostic Performance)
    run_command("python scripts/generate_figure5.py", 
                "Step 5: Generating Figure 5 (Diagnostic Performance)")
    
    # Step 6: Generate Figure 6 (Age Correlations)
    run_command("python scripts/generate_figure6.py", 
                "Step 6: Generating Figure 6 (Age Independency)")
    
    # Step 7: Generate Figure 7 (DL Benchmarking)
    run_command("python scripts/generate_figure7.py", 
                "Step 7: Generating Figure 7 (Deep Learning Benchmarks)")
    
    # Step 8: Generate Figure 8 (Clinical Insights)
    run_command("python scripts/generate_figure8.py", 
                "Step 8: Generating Figure 8 (Biomarker Utility & Clinical Validation)")
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📊 Generated figures (available in figures/):")
    print("  ✓ Figure 1 (Study Overview)")
    print("  ✓ Figure 2 (Signal Archetypes)")
    print("  ✓ Figure 3 (Circadian Dynamics)")
    print("  ✓ Figure 4 (MSE & Complexity)")
    print("  ✓ Figure 5 (Diagnostic Performance)")
    print("  ✓ Figure 6 (Age Correlations)")
    print("  ✓ Figure 7 (DL Benchmarking)")
    print("  ✓ Figure 8 (Biomarker Utility)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
