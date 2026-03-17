# Multicenter Parkinson's Disease — Cardiac Autonomic Complexity Study

Public code and data repository for the multicenter study investigating cardiac autonomic
complexity as a biomarker for Parkinson's Disease (PD).

---

## Study Overview

We measure Heart Rate Variability (HRV) and multiscale entropy complexity (rcMSE nAUC)
from ECG/PPG recordings across three international centers to identify robust physiological
biomarkers for Parkinson's Disease.

**Cohorts:**

| Center | Abbrev. | Signal | Recording | N subjects | Groups |
|--------|---------|--------|-----------|------------|--------|
| CETRAM, Santiago de Chile | CETRAM | ECG | ~15 min rest | 71 | 35 PD, 36 Control |
| Hospital Universitario Cruces, Spain | Cruces | PPG | 5–15 min | 58 | 29 PD, 29 Control |
| Nagoya University, Japan | Nagoya | ECG Holter | 24 h ambulatory | 45* | 24 PD, 21 Control |

\* Nagoya: up to 45 subjects depending on the time window analyzed (16–20h best window).

**Primary biomarker:** `rcMSE nAUC(1–20) / HR` — refined Composite Multiscale Entropy,
area under the curve over scales 1–20, normalized by mean heart rate.

---

## Repository Structure

```
public_release/
├── data/                          # Pre-computed features (SHARED)
│   ├── chile_mse.csv              # CETRAM: rcMSE per scale, 71 subjects × 20 scales
│   ├── chile_metrics.csv          # CETRAM: full HRV feature set (neurokit2)
│   ├── chile_demographics.csv     # CETRAM: Age, Sex
│   ├── spain_mse.csv              # Cruces: rcMSE per scale, 58 subjects × 20 scales
│   ├── spain_metrics.csv          # Cruces: full HRV feature set
│   ├── spain_demographics.csv     # Cruces: Age
│   ├── japan_afternoon_mse.csv    # Nagoya 16–20h: rcMSE, 45 subjects × 20 scales (best window)
│   ├── japan_morning_mse.csv      # Nagoya 07–11h: rcMSE, 39 subjects × 20 scales
│   ├── japan_evolution.csv        # Nagoya: nAUC(1–20) + HRV metrics per 4h window (full 24h)
│   ├── japan_recalc_metrics.csv   # Nagoya: HRV metrics (full 24h, neurokit2 recalculation)
│   ├── japan_metadata.csv         # Nagoya: Age, Sex
│   ├── deidentified_clinical_consolidated.xlsx  # CETRAM clinical data (de-identified)
│   ├── sample_signals/            # 5 anonymized ECG RRi traces (CETRAM) for Figure 2 demo
│   └── benchmarks/                # ML cross-validation results (leave-one-cohort-out)
│
├── scripts/                       # All scripts: figure generation + statistical analysis
│   ├── run_pipeline.py                    # Run everything end-to-end
│   ├── generate_figure1.py        # Figure 1: study design & demographics
│   ├── generate_figure2.py        # Figure 2: signal archetypes [partial — see below]
│   ├── generate_figure3.py        # Figure 3: circadian dynamics (Nagoya)
│   ├── generate_figure4.py        # Figure 4: MSE comparison across centers
│   ├── generate_figure5.py        # Figure 5: diagnostic performance & AUC comparison
│   ├── generate_figure6.py        # Figure 6: age-independency validation
│   ├── generate_figure7.py        # Figure 7: autonomic physiology composite
│   ├── generate_appendix.py       # Appendix: cross-center generalization matrix
│   │
│   │   # Statistical pre-processing for Figure 7 (run automatically by run_pipeline.py)
│   ├── traditional_hrv_metrics.py         # Step 1: consolidate HRV across centers
│   ├── multiscale_decomposition.py        # Step 2: scale-metric correlations, MSE curves
│   ├── complexity_correlation_analysis.py # Step 3: Spearman ρ + bootstrap CIs
│   ├── incremental_value_analysis.py      # Step 4: hierarchical regression, variance decomp
│   └── cross_dataset_consistency.py       # Step 5: physiological interpretation, report
│   ├── generate_figure8.py        # Figure 8: clinical correlations (CETRAM PD cohort)
│   ├── generate_appendix.py       # Appendix 1: cross-center generalization matrix
│   └── generate_appendix2.py      # Appendix 2: per-scale AUC, ECG vs PPG modality
│
├── figures/                       # Output directory for all figures
│   ├── Figure1/ … Figure6/        # PNG + SVG per manuscript figure
│   ├── Figure7/                   # Figure 7 PNG/SVG + all statistical intermediate CSVs
│   └── Appendix/                  # Appendix 1 (generalization matrix) + Appendix 2 (per-scale AUC)
│
└── requirements.txt               # Python dependencies
```

---

## What Can and Cannot Be Run

### Can be run (no raw data needed)

**Figures 1–7** can all be regenerated with one command:

```bash
cd /path/to/public_release
python scripts/run_pipeline.py        # regenerates all figures 1–7 + appendix
```

Figures 1–6 read only from `data/*.csv`. Figure 7 requires statistical
pre-processing (5 steps); `run_pipeline.py` runs these automatically before
generating Figure 7.

To run individual figures:
```bash
python scripts/generate_figure3.py   # any of figures 1–6 directly
python scripts/generate_figure7.py   # requires pre-processing steps first (see below)
```

To run the Figure 7 statistical pre-processing steps individually:
```bash
# Run in order — each step depends on the previous:
python scripts/traditional_hrv_metrics.py          # → figures/Figure7/traditional_hrv_metrics.csv
python scripts/multiscale_decomposition.py         # → figures/Figure7/scale_metric_correlations.csv
python scripts/complexity_correlation_analysis.py  # → figures/Figure7/spearman_correlations.csv
python scripts/incremental_value_analysis.py       # → figures/Figure7/hierarchical_regression.csv
python scripts/cross_dataset_consistency.py        # → figures/Figure7/physiological_interpretation.csv
```

> **Note:** All statistical outputs are pre-computed in `figures/Figure7/*.csv`. You only
> need to re-run the analysis scripts if you want to verify or modify the analysis.

### Cannot be run (raw data not shared)

**Figure 2 (RRi traces):** The top row of Figure 2 overlays raw ECG RRi time series.
The raw RRI signal files are not included for privacy and data-sharing agreement reasons.
Five anonymized sample traces are provided in `data/sample_signals/` for demonstration.
The remaining panels of Figure 2 (HR distributions, age distributions, Poincaré plots)
will render from `data/*.csv` but the RRi trace panels will be empty/blank.

**MSE/HRV feature computation:** The per-scale entropy values in `data/*_mse.csv` were
computed from raw RRI signals using our Julia RC-MSE toolbox. Re-running this computation
requires the raw RRI data, which is available upon reasonable request.

---

## Setup

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`,
`python-pptx`.

---

## Key Methods

### Complexity metric

**rcMSE nAUC(1–20)** — Refined Composite Multiscale Sample Entropy, computed via
the Julia `rcmse` toolbox. For each subject and time window:
1. Compute SampEn at scales 1–20 (coarse-graining by averaging)
2. Take the trapezoidal area under the entropy-vs-scale curve (nAUC)
3. Normalize by mean heart rate (bpm) to remove confounding by heart rate

### Recording windows

| Center | Window | Scales used | Notes |
|--------|--------|-------------|-------|
| CETRAM | Full ~15 min | 1–5 interpretable | Scales 6–20 unreliable (short recording) |
| Cruces | Full 5–15 min | 1–5 interpretable | Scales 6–20 unreliable |
| Nagoya | 16–20h (best) | 1–20 | Full 24h Holter; afternoon window optimal for PD discrimination |
| Nagoya | 07–11h (morning) | 1–20 | Secondary window for comparison |

### Statistical framework

- **Group comparisons:** Mann-Whitney U or Student's t-test; BH-FDR correction
- **Correlations:** Spearman ρ with bootstrap 95% CIs (n=1000); BH-FDR correction
- **Partial correlations:** Residualize on age (and sex where available) before Spearman ρ
- **Incremental value:** Hierarchical logistic regression (Block 1: confounders →
  Block 2: +pNN50 → Block 3: +complexity); Nagelkerke R², Δ log-likelihood test
- **Variance partitioning:** Commonality analysis (unique vs shared variance)
- **ML validation:** Leave-One-Cohort-Out (LOCO) cross-validation with Random Forest

---

## Figure Guide

| Figure | File | Description |
|--------|------|-------------|
| 1 | `figures/Figure1/Figure1.png` | Study design map + demographics |
| 2 | `figures/Figure2/Figure2.png` | RRi traces, HR & age distributions, Poincaré plots |
| 3 | `figures/Figure3/Figure3.png` | Nagoya 24h circadian profile of complexity (Panels A–F) |
| 4 | `figures/Figure4/Figure4.png` | MSE curves + nAUC comparison across centers |
| 5 | `figures/Figure5/Figure5.png` | Diagnostic performance, ML validation & feature independence (11 panels) |
| 6 | `figures/Figure6/Figure6.png` | Age-independence: scatter + partial ρ across centers |
| 7 | `figures/Figure7/Figure7.png` | Composite: autonomic correlates, confound correction, scale anatomy |
| 8 | `figures/Figure8/Figure8.png` | Clinical correlations: complexity vs CISI-PD, H&Y, disease duration, non-motor symptoms (CETRAM PD cohort, n=22; LEDD pending) |
| App. 1 | `figures/Appendix/FigureAppendix.png` | Cross-center generalization matrix (RF, LOCO) |
| App. 2 | `figures/Appendix/FigureAppendix2.png` | Per-scale AUC across recording modalities (ECG vs PPG) |

### Figure 7 panel guide

| Panel | Description | Source |
|-------|-------------|--------|
| A | Spearman ρ heatmap: complexity vs traditional HRV × center × group | `scripts/complexity_correlation_analysis.py` |
| B | Forest plot: cross-dataset consistency of key correlates | `scripts/cross_dataset_consistency.py` |
| C | Scale physiology heatmap: ρ per scale × metric (CETRAM / Cruces / Nagoya 16–20h) | `generate_figure7.py` inline |
| D | McFadden R² variance decomposition (incremental value) | `scripts/incremental_value_analysis.py` |
| E | Annotated MSE curves: PD vs HC + Mann-Whitney per scale | `scripts/multiscale_decomposition.py` |
| F | Confound correction: raw vs age/sex-adjusted ρ (Pooled n=152) | `generate_figure7.py` inline |
| G | Autonomic synthesis: raw ρ / partial ρ / unique variance per metric | `generate_figure7.py` inline |
| H | Scale anatomy: ρ vs timescale for key HRV metrics (Nagoya 16–20h & CETRAM) | `generate_figure7.py` inline |

---

## Data Dictionary

### `data/*_mse.csv` (all centers)

| Column | Description |
|--------|-------------|
| `Subject` | Anonymized subject ID |
| `Group` | `PD` or `Control` |
| `Scales` | MSE timescale (1–20) |
| `MSE` | Sample entropy at that scale (rcMSE toolbox) |

### `data/japan_evolution.csv`

Per-subject, per-4h-window summary for the full Nagoya 24h recording.

| Column | Description |
|--------|-------------|
| `Subject` | Anonymized subject ID |
| `Group` | `PD` or `Control` |
| `Window_start_h` | Start hour of 4h window (0, 4, 8, …, 20) |
| `n_beats` | Number of RR intervals in window |
| `nAUC_1_20` | rcMSE nAUC over scales 1–20 (primary complexity metric) |
| `HR` | Mean heart rate (bpm) |
| `SDNN` | SDNN (s) |
| `RMSSD` | RMSSD (s) |

### `figures/Figure7/traditional_hrv_metrics.csv`

Pooled HRV table (166 subjects: CETRAM 71 + Cruces 58 + Nagoya 37 with full HRV).

Key columns: `Subject`, `Center`, `Group`, `Age`, `Sex`, `MeanNN`, `SDNN`, `RMSSD`,
`pNN50`, `HF_power`, `LF_power`, `LF_HF`, `LF_norm`, `VLF_power`, `Total_power`,
`DFA_alpha1`, `DFA_alpha2`, `SD1`, `SD2`, `SDANN`, `SampEn_S1`, `rcMSE_AUC`,
`HR`, `PPG_bias`, `ectopic_method`.

---

## Label Conventions

Throughout all figures and results files:

| Internal CSV key | Figure label |
|-----------------|--------------|
| `Chile` | CETRAM |
| `Spain` | Cruces |
| `Japan-afternoon` | Nagoya (16–20h) |
| `Japan-morning` | Nagoya (07–11h) |
| `Japan` | Nagoya (center-level, e.g., in `traditional_hrv_metrics.csv`) |

---

## Contact

NeuroEng@Usach group — please open an issue or contact the corresponding author
for data access requests or questions about the methodology.
