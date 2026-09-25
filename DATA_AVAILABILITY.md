# Data availability

This repository contains **derived data only**. Every file in `data/` is a
pre-computed feature table — entropy curves, HRV metrics, de-identified
demographics — from which all figures and statistics can be regenerated.

Raw physiological recordings and individual-level clinical records are **not
shared here**. They remain subject to the ethics approvals and data-sharing
agreements of the contributing centres.

---

## What is included

| Path | Contents |
|---|---|
| `data/*_mse.csv` | rcMSE curves, scales 1–20, per subject |
| `data/*_metrics.csv` | HRV feature tables (time, frequency, nonlinear) |
| `data/*_demographics.csv`, `japan_metadata.csv` | age, sex, group |
| `data/japan_15min_windows.csv` | per-window complexity across 24 h (Nagoya) |
| `data/japan_scale_profile.csv` | scale-resolved entropy, τ = 1–60 (Nagoya) |
| `data/sdnn_min_all_centers.csv` | minimum-value HRV indices, all cohorts |
| `data/benchmarks/` | machine-learning cross-validation results |
| `data/sample_signals/` | five anonymised CETRAM RR traces, for demonstration |
| `figures/**/*.csv` | every number plotted in every figure |

## What is **not** included, and why

### Clinical data — CETRAM (affects Figure 8)

`deidentified_clinical_consolidated.xlsx` holds **28 PD patients × 65 clinical
variables**: Hoehn & Yahr stage, CISI-PD subscales, Schwab & England score,
disease duration, levodopa-equivalent dose, genetic findings, comorbidities and
non-motor symptom flags.

Although direct identifiers have been removed, the combination of age, sex,
disease duration and clinical scores across 28 individuals yields very small cell
sizes, and a formal k-anonymity assessment has not been carried out. Publishing
it could permit re-identification. **It is therefore withheld.**

**Consequence: Figure 8 (clinical correlations) cannot be regenerated from this
repository.** The figure and its underlying correlation table
(`figures/Figure8/figure8_correlations.csv`) are released so the results are fully
inspectable; the subject-level clinical data behind them are not.

Researchers who need the clinical data should **contact the corresponding author**.
Access requires a data-sharing agreement and approval from the CETRAM ethics
committee.

### Raw ECG / PPG / Holter recordings

| Cohort | Signal | Held by |
|---|---|---|
| CETRAM | raw ECG, 1000 Hz, ~15 min | CETRAM, Santiago de Chile |
| Cruces | finger PPG, 500 Hz, 7.40 min | Hospital Universitario Cruces, Bilbao |
| Nagoya | 24-h Holter RR series | Nagoya University Hospital |

**Consequences:** the raw-trace panels of Figure 2 render empty (five anonymised
CETRAM traces are provided in `data/sample_signals/` for demonstration); rcMSE
cannot be recomputed from signal, though the resulting curves are shipped; and the
deep-learning benchmark (`scripts/benchmark_dl_loco.py`) cannot be run.

The Cruces and Nagoya cohorts were first described in their source publications,
which should be consulted for cohort-level detail and cited alongside this work:

- Iniguez M, et al. *Heart-brain synchronization breakdown in Parkinson's disease.* npj Parkinsons Dis. 2022;8:64. doi:10.1038/s41531-022-00323-w
- Suzuki M, et al. *Wearable sensor device-based detection of decreased heart rate variability in Parkinson's disease.* J Neural Transm. 2022;129:1299–1306. doi:10.1007/s00702-022-02528-y

---

## Requesting access

Contact the corresponding author (see `CITATION.cff`) stating the intended
analysis. Requests are considered subject to:

1. a data-sharing agreement with the centre holding the data;
2. ethics approval covering the proposed use;
3. for clinical data, approval from the CETRAM ethics committee.

## Licence

Derived data in this repository: **CC BY 4.0**. Code: **MIT**. See `LICENSE`.
