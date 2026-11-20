# Cardiac autonomic complexity in Parkinson's disease — three cohorts

Code and derived data for *Reduced multiscale complexity of RR intervals in Parkinson's
disease: effects of acquisition modality and recording length* (L. E. Medina et al.,
NeuroEng@Usach, Universidad de Santiago de Chile).

Complexity is refined composite multiscale entropy (rcMSE) of the RR-interval series,
with a tolerance fixed from the original series (r = 0.2 SD), summarised as the
normalised area under the curve, nAUC = trapz(curve, τ = 1…n) / n.

| Cohort | Signal | Recording | Analysed |
|---|---|---|---|
| CETRAM, Santiago | ECG, 1 kHz | ~15 min at rest | 42 control / 27 PD |
| Cruces, Bilbao | finger PPG | 7.4 min, during resting fMRI | 21 control / 31 PD |
| Nagoya | chest-strap ECG | ~24 h in hospital; 18:00–22:00 analysed | 21 disease control / 26 PD |

## Quick start

```bash
pip install -r requirements-lock.txt      # Python 3.10-3.12; exact versions used
python reproduce.py                       # every figure, then the check of every number in the text
```

This takes about two minutes. It needs nothing beyond this repository. Figures are
written to `figures/<figure>/<figure>.svg` (and `.png`). The last step,
`scripts/verify_manuscript_claims.py`, recomputes each number quoted in the manuscript
from `data/` and stops with an error if any one differs.

Four outputs need inputs that are not distributed and are skipped with a message:
Fig. 6 and Suppl. Fig. S9 (raw ECG), Fig. 9 (clinical table), and the data-driven
panels of Fig. 1. The trace panels of Fig. 2 are drawn only when the original series
are available. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Figures

Names follow the manuscript. Each script reads only `data/` unless stated.

| Manuscript | Script | Output folder |
|---|---|---|
| Tables 1–4 | `tables.py` | `Tables/` |
| Fig. 1 | `fig1_map.py`, `fig1_icons.py`, `fig1_ecg_traces.py`, `fig1_elements.py`*, `fig1_layout.py` | `Fig1_overview/` (final figure assembled in Affinity Designer: `Fig1_overview.af`) |
| Fig. 2 | `fig2_archetypes.py` (trace panels*) | `Fig2_archetypes/` |
| Fig. 3 | `fig3_curves_index.py` | `Fig3_curves_index/` |
| Fig. 4 | `fig4_age.py` | `Fig4_age/` |
| Fig. 5 | `fig5_parameters.py` | `Fig5_parameters/` |
| Fig. 6 | `fig6_detector.py`† | `Fig6_detector/` |
| Fig. 7 | `fig7_recording_length.py` | `Fig7_recording_length/` |
| Fig. 8 | `fig8_step1…5_*.py`, then `fig8_hrv_relationships.py` | `Fig8_hrv_relationships/` |
| Fig. 9 | `fig9_clinical.py`‡ | `Fig9_clinical/` |
| Suppl. S1–S5 | `figS1_hr_normalization.py`, `figS2_spectral.py`, `figS3_cruces_context.py`, `figS4_length.py`, `figS5_ppg_scales.py` | `FigS1_…` to `FigS5_…` |
| Suppl. S6, S7 | `figS6_S7_cross_cohort.py` | `FigS6_S7_cross_cohort/` |
| Suppl. S8 | `figS8_validation.py` | `FigS8_validation/` |
| Suppl. S9 | `figS9_traces.py`† | `FigS9_traces/` |

\* needs the original beat series; † needs the raw ECG; ‡ needs the clinical table.
Next to each figure are the tables it plots (`*_stats.csv` and the like). The legends
are in [FIGURE_LEGENDS.md](FIGURE_LEGENDS.md).

## Three levels of reproduction

**1. Figures and numbers, from the shipped tables** — `python reproduce.py`. Everything
in `data/` is described in [data/README.md](data/README.md).

**2. Everything else, from the original recordings** — needs the raw ECG, PPG and RR
recordings (available on reasonable request, not distributed; see
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)) and Julia ≥ 1.9. Point `HRV_DATA_ROOT` to
them; the expected layout is in [scripts/paths.py](scripts/paths.py), the only place where
locations are defined.

```bash
export HRV_DATA_ROOT=/path/to/original_data
cd scripts/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ../..
python reproduce.py cohorts     # raw signal -> RR series, rcMSE curves, HRV indices  (~20 min)
python reproduce.py derived     # the remaining tables, from the RR series            (~40 min)
python reproduce.py             # figures and the claim check from the rebuilt tables
```

`cohorts` runs one pipeline per cohort (`scripts/cohorts/`):

| Cohort | Raw signal to RR series | Tables |
|---|---|---|
| CETRAM | `cetram_beats.py`: R peaks (NeuroKit2), two-detector quality gate (bSQI ≥ 0.80), ectopic/missed-beat correction (Lipponen & Tarvainen), 300–2000 ms guard, trace-review exclusions (`cetram_exclusions.csv`) | `cetram_hrv.py`, rcMSE in Julia |
| Cruces | `cruces_rri.py`: pulses (Elgendi), beat at the steepest upstroke, same correction and guard | `cruces_hrv.py`, rcMSE in Julia |
| Nagoya | RR intervals as exported by the chest strap | `nagoya_windows.py`: rcMSE in 4-h windows, HR/SDNN/RMSSD across the day |

rcMSE is `scripts/julia/` (regression tests: `julia --project=scripts/julia
scripts/julia/test_entropy.jl`). `scripts/rcmse_fast.py` is a Python port of the same
algorithm (KD-tree counting) for the long Nagoya windows and the length, split-half and
parameter analyses; it reproduces the Julia values to machine precision.

Age, sex and recording start times (`chile_demographics.csv`, `spain_demographics.csv`,
`japan_metadata.csv`) were extracted from the clinical records, which are not distributed,
and are shipped as given.

## Layout

```
reproduce.py          entry point
scripts/
  paths.py            all file locations; HRV_DATA_ROOT
  fig*.py             one script per manuscript figure
  cohorts/            raw signal -> RR series, rcMSE curves, HRV indices, per cohort
  compute_*.py        the remaining tables in data/, from the RR series
  verify_manuscript_claims.py   every number in the text, recomputed
  julia/              rcMSE (Julia) with regression tests
  rcmse_fast.py       rcMSE (Python port)
  figstyle.py, pooling.py, complexity_index.py   shared helpers
  provenance/         one-off analyses quoted in the text (see its README)
  sync_manuscript.py  maintainer tool: figures and legends into the manuscript
data/                 derived tables (data/README.md)
figures/              SVG figures and the tables behind them
```

Reruns are deterministic: random draws are seeded and SVG output is fixed by
`matplotlibrc` and `SOURCE_DATE_EPOCH`, so `git status` stays clean after
`python reproduce.py`.

## Citation and licence

See [CITATION.cff](CITATION.cff). Code: MIT. Derived data: CC BY 4.0 ([LICENSE](LICENSE)).
The Cruces and Nagoya cohorts were first described by Iniguez et al., *npj Parkinson's
Disease* 2022;8:64, and Suzuki et al., *J Neural Transm* 2022;129:1299–1306; please cite
them alongside this work.
