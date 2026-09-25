# Multicenter Parkinson's Disease — Cardiac Autonomic Complexity

Code and derived data for a multicenter study of cardiac autonomic complexity in
Parkinson's disease (PD), measured by refined composite multiscale entropy (rcMSE)
across three independent cohorts and three recording modalities.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Cohorts

| Centre | Signal | Recording | Analysed n | Controls | Source publication |
|---|---|---|---|---|---|
| **CETRAM**, Santiago, Chile | ECG, 1000 Hz | ~15 min supine rest | **73** (43 C / 30 PD) | healthy | — |
| **Cruces**, Bilbao, Spain | finger PPG, 500 Hz | **7.40 min**, during resting-state fMRI | **52** (21 C / 31 PD) | healthy | Iniguez et al. 2022 |
| **Nagoya**, Japan | ECG Holter (POLAR) | ~24 h ambulatory | **45–50** (21–23 C / 24–27 PD) | **disease controls** | Suzuki et al. 2022 |

Notes that matter for interpretation:

- **Nagoya controls are disease controls**, not healthy volunteers — 12 essential tremor and 11 investigated for numbness/dizziness/light-headedness with no abnormality found. Nagoya effect sizes are therefore against a harder comparator than the other two cohorts.
- **Cruces carries six records labelled `Other`** (`A01_1, A03_2, A05_2, A09, D01, D05`) with no diagnosis. They are **excluded from all primary analyses**; the published cohort of Iniguez et al. is exactly 31 PD + 21 healthy controls, which matches our Control/PD counts. A sensitivity analysis including them is reported in the manuscript.
- Nagoya n varies by analysis: 50 subjects have a full record, 45 have a usable 16–20 h window, 43 have complete 24-h HRV.

---

## The complexity index

**rcMSE with a fixed tolerance.** For each subject:

1. Compute refined composite multiscale entropy (Wu et al. 2014) on the RR-interval series, scales τ = 1…20.
2. The tolerance is `r = 0.2 × SD(RRi)`, computed **once from the original series and held constant across every scale**.
3. Summarise with `nAUC = trapz(1..n, curve) / n` over the reliable scale range, optionally divided by mean HR.

### Why fixed r is not a detail

Coarse-graining reduces variance with scale. Holding `r` fixed lets that reduction
express itself in the entropy — which is what MSE exists to measure. Recomputing
`r` from each coarse-grained series cancels exactly that term: on synthetic
signals the per-scale variant inverts the canonical Costa contrast, with
white-noise entropy no longer decaying and 1/f noise rising.

An earlier version of this pipeline recomputed `r` per scale. It has been
corrected throughout and a regression test (`scripts/julia/test_entropy.jl`)
now fails loudly if the defect is reintroduced.

### Scale ranges

Reliability limit `N/τ ≥ 200`, validated independently by split-half reliability:

| Cohort | beats | reliable τ | index used |
|---|---|---|---|
| CETRAM | ~1088 | ≤ 5 | nAUC(1–5) / HR |
| Cruces | ~498 | ≤ 2 | nAUC(1–5) / HR — see caveat |
| Nagoya | ~14 000 (4 h window) | ≤ 20 (up to 70 permitted) | nAUC(1–5) / HR and nAUC(1–20) / HR |

A fixed τ = 1–5 is applied to CETRAM and Cruces so the cohorts remain directly
comparable. **Caveat:** at 498 beats Cruces exceeds its own reliability limit at
τ > 2 (split-half reliability of nAUC(1–5) is 0.347, against 0.494 in CETRAM and
0.670 in Nagoya). Cruces should be read as supporting the *direction* of effect,
not as independent confirmation of magnitude.

`nAUC` divides the trapezoidal area by the number of points `n`, not by the span
`n−1`, so it carries an n-dependent factor (0.80 at n=5, 0.95 at n=20). Within a
cohort this is a constant rescaling and changes nothing; **across cohorts with
different scale ranges the raw values are not comparable** — use effect sizes.

---

## Repository layout

```
public_release/
├── scripts/
│   ├── julia/                     rcMSE — the core computation
│   │   ├── entropy.jl               canonical toolbox (fixed r, Wu shifts)
│   │   ├── run_rcmse.jl             generic driver, any cohort
│   │   ├── test_entropy.jl          regression tests
│   │   └── Project.toml             pinned Julia environment
│   ├── run_pipeline.py            runs everything below, in order
│   ├── sync_center_data.py        pulls per-centre outputs into data/
│   ├── generate_figure2..8.py     manuscript figures (Figure 1 is not scripted —
│   │                                see figures/Figure1/ below)
│   ├── generate_appendix*.py      supplementary figures
│   ├── generate_validation_figure.py   methods-validation figure (S1)
│   ├── complexity_index.py        single canonical definition of the index
│   ├── traditional_hrv_metrics.py ┐
│   ├── multiscale_decomposition.py│ Figure 7 statistical pre-steps,
│   ├── complexity_correlation_analysis.py │ run in this order
│   ├── incremental_value_analysis.py      │
│   └── cross_dataset_consistency.py       ┘
├── data/                          derived features (see PROVENANCE.md)
├── figures/                       SVG output + every plotted number as CSV
├── requirements.txt               Python dependencies
├── requirements-lock.txt          exact versions used for the released figures
├── LICENSE                        MIT (code) / CC BY 4.0 (derived data)
└── CITATION.cff
```

---

## Reproducing

### 1. Environments

```bash
pip install -r requirements-lock.txt          # exact released versions
cd scripts/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Verify the entropy implementation

```bash
cd scripts/julia
julia --project=. test_entropy.jl
```

Checks SampEn against known limits, confirms rcMSE(τ=1) equals plain SampEn,
verifies amplitude invariance, and asserts the Costa signature that fixed `r`
must reproduce.

### 3. Recompute rcMSE (optional — curves are shipped)

```bash
cd scripts/julia
julia --project=. run_rcmse.jl \
      --input <cleaned_peaks_dir> --out ../../data/chile_mse.csv --scales 20
```

See the header of `run_rcmse.jl` for all options. Requires the per-centre cleaned
peak files, which are not in this repository.

### 4. Figures

```bash
python scripts/run_pipeline.py
```

Regenerates Figures 1–8 and both appendices from `data/`. Runs in a few minutes.

---

## What can and cannot be reproduced here

**Can** — everything from the shipped derived data: Figures 1, 3–7, both
appendices, the validation figure, and all statistics.

**Cannot:**

| Item | Why |
|---|---|
| Figure 1 | A graphical abstract authored in Affinity Designer. The source is tracked as `figures/Figure1/Figure1.af`; there is no scripted stage for it. |
| Figure 2 RRi trace panels | Raw RR series are not shared. Five anonymised CETRAM traces are in `data/sample_signals/` for demonstration; the trace panels render empty without the rest. |
| **Figure 8** | **Depends on individual-level clinical records that are not shared.** `deidentified_clinical_consolidated.xlsx` holds 28 PD patients × 65 clinical variables; the cell sizes make re-identification a real risk and no k-anonymity assessment has been done. The figure and its correlation table are released so the results are inspectable, but the underlying data are not. **Request access from the corresponding author** — see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). |
| rcMSE from raw signal | Requires the per-centre raw RRi. The curves in `data/*_mse.csv` are the shipped starting point. |
| Deep-learning benchmark | `benchmark_dl_loco.py` needs raw RRi and PyTorch. |

**Raw physiological recordings and individual-level clinical data are not shared
in this repository.** They are available from the corresponding author on
reasonable request, subject to a data-sharing agreement and the contributing
centres' ethics approvals. See **[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)**
for exactly what is and is not included, and how to request access.

---

## Known limitations

Recorded here rather than buried, because each affects how results should be read.

1. **Complexity does not outperform simpler metrics on long recordings.** In Nagoya the minimum SDNN over 100-beat windows (Suzuki et al.) reaches AUC 0.913 against 0.838 for the complexity index. At matched window counts complexity wins (0.853 vs 0.811), but SDNN-min subsumes it: after partialling out SDNN-min, complexity retains no independent discrimination (AUC 0.572, p=0.39).
2. **The group difference is not carried by nonlinearity.** IAAFT surrogates, which preserve the power spectrum and amplitude distribution while destroying nonlinear structure, reproduce almost the entire separation (AUC 0.689 vs 0.699 for real data), and there is no group difference in the nonlinearity index itself (p=0.60).
3. **Cruces is underpowered for a multiscale index** — see the scale-range caveat above.
4. **Nagoya PD are less physically active** (Suzuki et al. report standing time 2.1 vs 4.6 h/day). Any window-selection rule based on signal variability is therefore entangled with the disease; window selection here is never based on group separation.
5. **Absolute entropy is not comparable across cohorts**, because `r = 0.2 × SD` is set by each recording's low-frequency content. Within-cohort comparisons are unaffected.

---

## Data provenance

`data/PROVENANCE.md` records the source of every file and the sync log.
`sync_center_data.py` reproduces the copy from the per-centre repositories.

## Contact

NeuroEng@Usach — Universidad de Santiago de Chile.
Open an issue, or contact the corresponding author for data access.
