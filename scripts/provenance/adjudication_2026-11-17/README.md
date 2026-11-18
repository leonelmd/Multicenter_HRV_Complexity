# CETRAM trace-review exclusions (2026-11-17)

Decision (Leo, 2026-11-17): exclude four CETRAM recordings after visual review of the raw ECG
(Resultados 2.pdf, P. Chaná, 2026-11-09) and make n = 69 (42 C / 27 PD) the primary CETRAM sample.

| Subject | Group | Criterion |
|---|---|---|
| P004 | PD | frequent ectopy (~18 % of intervals premature) |
| C050 | Control | frequent ectopy (~18 %, mostly supraventricular) |
| P012 | PD | ~5 min without QRS at the start |
| P023 | PD | ~2 min without QRS at the start |

Retained after review: P003, C018 (sinus, partial movement/tremor artefact), C033 (sinus, slow HR drift).

## Where the exclusion lives
- `CETRAM/public_release/data/metadata/adjudicated_exclusions.csv` — the list.
- `CETRAM/public_release/scripts/clean_data.py` — applies it after the bSQI gate (Status FAIL,
  Reason "trace review: …"), so a full re-run reproduces it.
- Existing outputs were updated equivalently (every metric is per subject): the four detection
  files moved to `results/excluded_adjudicated/`, rows dropped from entropy/metrics tables,
  `cleaning_summary.csv` marked. Pre-exclusion copies: `results/_pre_adjudication_2026-11-17/`.
- Multicenter `data/`: chile_* re-synced; per-subject derived files (fr_CETRAM, length_ladder,
  r_sweep, sdnn_min_all_centers, spectral_analysis) filtered; fig3_* recomputed with
  `compute_fig3_inputs.py`. Pre-exclusion copies: `_pre_adjudication_2026-11-17_data/`.

## Clinical workbook
`data/deidentified_clinical_consolidated.xlsx`: LED total added for P001, P003, P005, P009, P011,
P015, P017, P020, P021, P025, P026 (Resultados 1.pdf, Table 1; regimen at the ECG date);
`Cardiac problems` = 0 for P019 and P023 (chart: no cardiopathy). Detail incl. cardioactive drugs:
`CETRAM/data/clinical/clinical_update_2026-11-16_deid.csv`. Note: P009 is on carvedilol.

## Numbers and text
- `cetram_manuscript_numbers.py` — every CETRAM number in the manuscript. Run on the pre-exclusion
  data it reproduces the 2026-11-01 text (numbers_pre.csv) except the bootstrap CI (±0.002 MC noise)
  and the OH rank regression (original specification unknown; now OLS on ranks, age + HR, HC3).
- `apply_manuscript_edits.py` — the 68 text edits, each asserted to match exactly once.
- `scripts/verify_manuscript_claims.py` — claims updated; 58/58 pass.

## Not done
- Figure 2 was regenerated with the Cruces/Nagoya RRi (2026-11-17) and copied to the manuscript.
- `fig1_overview.pdf` (Affinity): Santiago should read 27 PD vs 42 HC.
- FIGURE_LEGENDS.md files were not updated (already flagged as stale in the manuscript README).
