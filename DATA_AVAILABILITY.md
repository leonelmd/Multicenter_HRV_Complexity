# Data availability

## In this repository

`data/` holds derived, de-identified tables: rcMSE curves, HRV indices, age, sex and
group per participant, and the intermediate tables behind each figure. Participants
appear only under study codes (C001, P001, B01, SUB-001, …). Every file is described in
[data/README.md](data/README.md). Every figure and every number in the manuscript can be
regenerated from these tables (`python reproduce.py`), except the four items below.

## Not in this repository

| Input | Needed for | Held by | Request from |
|---|---|---|---|
| CETRAM raw ECG and beat series | Figs. 6 and S9; the data-driven panels of Fig. 1; rebuilding `data/` | CETRAM, Santiago | the corresponding author (leonel.medina@usach.cl) |
| Cruces pulse-interval series | rebuilding `data/` | Hospital Universitario Cruces, Bilbao | the authors of Iniguez et al. 2022 |
| Nagoya 24-h RR series | rebuilding `data/` | Nagoya University Graduate School of Medicine | M. Suzuki or M. Katsuno, Nagoya University |
| CETRAM clinical table (de-identified) | Fig. 9 | CETRAM, Santiago | the corresponding author |

The trace panels of Fig. 2 are drawn only when the beat series are available.

The clinical table is withheld because the combination of age, sex, disease duration and
clinical scores in 25 participants gives cell sizes small enough to permit
re-identification. The correlations it yields are released
(`figures/Fig9_clinical/Fig9_clinical_correlations.csv`).

Access is subject to a data-sharing agreement and to the approvals of the centre that
holds the data.

## Using the original data

Place the data as described in [scripts/paths.py](scripts/paths.py) and set
`HRV_DATA_ROOT` to that folder; put the clinical table in `data/private/`, which git
ignores. Then

```bash
python reproduce.py derived    # rebuild data/ from the series
python reproduce.py            # all figures, including those otherwise skipped
```

## Source publications

- Iniguez M, et al. Heart-brain synchronization breakdown in Parkinson's disease.
  *npj Parkinson's Disease* 2022;8:64. doi:10.1038/s41531-022-00323-w
- Suzuki M, et al. Wearable sensor device-based detection of decreased heart rate
  variability in Parkinson's disease. *J Neural Transm* 2022;129:1299–1306.
  doi:10.1007/s00702-022-02528-y
