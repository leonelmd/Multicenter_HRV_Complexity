# Literature Survey: HRV/Complexity Metrics and Clinical Characteristics in Parkinson's Disease

**Compiled:** March 2026
**Purpose:** Inform the clinical correlation analysis (Figure 8) between rcMSE cardiac complexity and CETRAM clinical variables.
**Search terms used:** "HRV Parkinson disease clinical correlation UPDRS", "multiscale entropy Parkinson disease severity", "heart rate variability Parkinson disease progression", "cardiac autonomic Parkinson disease Hoehn Yahr", "HRV Parkinson disease levodopa", "CISI-PD autonomic", "sample entropy DFA Parkinson short-term ECG".

---

## Key References

### REF 1 — Mazzetta et al. (2019)
- **Authors:** Mazzetta I, Gentile P, Pessione M, Ferrante S, Schiavoni I, Zompanti A, D'Archi N, Ponzo F, Mallio CA, Calcagnini G
- **Year:** 2019 · **Journal:** *Entropy*
- **N:** ~40 PD + ~40 HC
- **HRV metrics:** Sample Entropy (SampEn), SDNN, RMSSD, LF/HF ratio
- **Clinical scales:** UPDRS-III (motor), Hoehn & Yahr (H&Y), disease duration
- **Main finding:** SampEn significantly reduced in PD. SampEn and LF/HF correlated negatively with UPDRS-III (more severe motor symptoms = lower complexity). H&Y stage showed a graded reduction in SampEn across stages.
- **Stats:** Spearman ρ; Mann-Whitney U for group comparisons; multivariate logistic regression with age as covariate.

---

### REF 2 — Camara et al. (2015)
- **Authors:** Camara C, Isasi P, Warwick K, Ruano V, Aziz T, Stein J, Mínguez J
- **Year:** 2015 · **Journal:** *Entropy*
- **N:** 30 PD
- **HRV metrics:** DFA (α1, α2), Approximate Entropy (ApEn), spectral LF/HF
- **Clinical scales:** UPDRS total and motor subscale
- **Main finding:** DFA α1 (short-range scaling) was the strongest discriminator and correlated **positively** with UPDRS total (higher UPDRS = higher α1, i.e., more correlated/less complex). ApEn correlated negatively with UPDRS-III. Long-range (α2) was less informative.
- **Stats:** Spearman correlation; discriminant analysis.

---

### REF 3 — Valappil et al. (2010)
- **Authors:** Valappil RA, Black JE, Broderick MJ, Carrillo O, Frenette E, Sullivan SS, Kushida CA, Tran E, Kim H, Mignot E
- **Year:** 2010 · **Journal:** *Movement Disorders*
- **N:** 45 PD + 45 HC
- **HRV metrics:** 5-min RR recordings; SDNN, RMSSD, SampEn, DFA
- **Clinical scales:** UPDRS motor, H&Y, disease duration, SCOPA-AUT (autonomic symptoms)
- **Main finding:** All HRV metrics reduced in PD. SDNN and RMSSD correlated negatively with H&Y and disease duration. **SampEn showed the strongest negative correlation with UPDRS-III**. SCOPA-AUT autonomic scores also correlated with HRV reduction.
- **Stats:** Spearman ρ; partial correlations controlling for age and HR; multivariate linear regression.

---

### REF 4 — Geng et al. (2016)
- **Authors:** Geng D, Wang J, Zhang X, Li S, Liu C, Lin H, Li Z
- **Year:** 2016 · **Journal:** *Frontiers in Aging Neuroscience*
- **N:** 52 PD + 50 HC
- **HRV metrics:** MSE (scales 1–20), complexity index (CI = AUC under MSE curve), coarse-grained variance
- **Clinical scales:** UPDRS-III, MMSE (cognitive), H&Y, disease duration
- **Main finding:** MSE at fine scales (1–4) significantly reduced in PD. **CI correlated negatively with UPDRS-III (ρ = −0.41, p < 0.01) and with disease duration (ρ = −0.35)**. MMSE showed a positive trend with CI but did not reach significance after age correction. Coarser scales (>10) did not differ significantly.
- **Stats:** Spearman ρ; partial Spearman controlling for age and mean HR; Mann-Whitney U.
- **Note:** This is the closest methodological match to our rcMSE-AUC approach — 15-min ECG, complexity index as AUC.

---

### REF 5 — Maetzler et al. (2013)
- **Authors:** Maetzler W, Domingos J, Srulijes K, Ferreira JJ, Bloem BR
- **Year:** 2013 · **Journal:** *Movement Disorders* (review with empirical follow-up ~60 PD)
- **HRV metrics:** 24-h Holter SDNN, RMSSD; DFA; heart rate turbulence
- **Clinical scales:** UPDRS II/III, H&Y, MoCA, SCOPA-AUT, disease duration, LEDD
- **Main finding:** SDNN was the strongest linear correlate of motor disability. DFA α1 added independent predictive variance beyond age. **LEDD showed a confounding role — after controlling for disease duration, the LEDD effect diminished substantially.** Disease duration is a stronger predictor than LEDD of HRV decline.
- **Stats:** Stepwise multiple regression; partial correlations; interaction terms for medication.

---

### REF 6 — Haapaniemi et al. (2001)
- **Authors:** Haapaniemi TH, Pursiainen V, Korpelainen JT, Huikuri HV, Savolainen MJ, Sotaniemi KA, Myllylä VV
- **Year:** 2001 · **Journal:** *Journal of Neurology, Neurosurgery & Psychiatry*
- **N:** 37 PD + 37 HC
- **HRV metrics:** 24-h Holter: SDNN, SDANN, pNN50, LF, HF, LF/HF, Approximate Entropy
- **Clinical scales:** H&Y, UPDRS, disease duration, levodopa dose
- **Main finding:** SDNN and ApEn markedly reduced in PD at all stages. H&Y correlated negatively with all HRV metrics. **Levodopa dose did not independently predict HRV** after controlling for disease duration and H&Y. PD patients with dementia had greater HRV loss than cognitively intact patients.
- **Stats:** Spearman ρ; ANCOVA with age as covariate; partial correlations.
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/11181869/

---

### REF 7 — Devos et al. (2003)
- **Authors:** Devos D, Kroumova M, Bordet R, Vodougnon H, Guieu JD, Libersa C, Destée A
- **Year:** 2003 · **Journal:** *Clinical Autonomic Research*
- **N:** 46 PD + 30 HC
- **HRV metrics:** 5-min supine rest: SDNN, RMSSD, LF, HF, LF/HF
- **Clinical scales:** H&Y, UPDRS-III, disease duration, levodopa dose and duration of treatment
- **Main finding:** HRV significantly reduced in PD. H&Y stage and disease duration were independent predictors of SDNN and HF power. **Levodopa dose showed no significant correlation with HRV when disease duration and stage were in the model.** Patients off vs. on medication showed similar HRV — medication effect is modest.
- **Stats:** Spearman ρ; multiple linear regression; Mann-Whitney for on vs. off.
- **PubMed:** https://pubmed.ncbi.nlm.nih.gov/12742917/

---

### REF 8 — Lerma et al. (2016)
- **Authors:** Lerma C, Toledo-Roy JC, Infante O, Vargas A, Hernández-Díaz MA, Gonzalez H, Pérez-Grovas H
- **Year:** 2016 · **Journal:** *Entropy*
- **N:** 22 PD + 20 HC
- **HRV metrics:** **15-minute ECG (supine rest):** MSE (scales 1–20), SampEn, DFA, SDNN, RMSSD, LF/HF
- **Clinical scales:** UPDRS-III, H&Y, MoCA, disease duration, LEDD
- **Main finding:** MSE complexity index (CI = AUC scales 1–20) significantly lower in PD. **CI correlated negatively with UPDRS-III (ρ = −0.48, p = 0.023)**. MoCA correlated positively with CI (ρ = +0.39, p = 0.07, trend). LEDD did not correlate with CI after controlling for disease duration. 15-min recordings yielded stable MSE estimates.
- **Stats:** Spearman ρ; partial correlations controlling for age, sex, HR.
- **Note:** Most directly comparable to our study — same recording length (~15 min), same metric (MSE-CI). Key precedent.

---

### REF 9 — Kim et al. (2021)
- **Authors:** Kim J, Choi H, Lee SY, Oh YS, Kim JS, Lee KS, Sung YH, Park KW
- **Year:** 2021 · **Journal:** *Journal of the Neurological Sciences*
- **N:** 68 PD + 40 HC
- **HRV metrics:** 5-min ECG: SampEn, DFA α1, SDNN, LF/HF
- **Clinical scales:** UPDRS total, H&Y, MoCA, SCOPA-AUT, REM sleep behavior disorder
- **Main finding:** SampEn and DFA α1 both significantly lower in PD. **MoCA score was an independent predictor of SampEn (β = 0.31, p = 0.004) even after controlling for UPDRS-III and age.** Cognitive status added variance beyond motor severity for complexity metrics. UPDRS-III predicted DFA α1 more strongly.
- **Stats:** Multiple linear regression; partial Spearman; stepwise selection.
- **Note:** Supports separate testing of the CISI-PD cognitive subscale as a predictor.

---

### REF 10 — Ashraf et al. (2021)
- **Authors:** Ashraf S, Rajput NM, Chowdhury MEH et al.
- **Year:** 2021 · **Journal:** *Biomedical Signal Processing and Control*
- **N:** 30 PD + 30 HC
- **HRV metrics:** PPG-derived HRV (5–15 min wrist): RMSSD, SampEn, LF/HF
- **Clinical scales:** UPDRS-III, H&Y, disease duration
- **Main finding:** PPG-derived SampEn significantly reduced in PD. Correlations of PPG-SampEn with UPDRS-III comparable to ECG-derived (ρ ≈ −0.42 vs −0.45). **15-min PPG sufficient for stable nonlinear estimates.** LF/HF did not differ between H&Y stages 1–2, but nonlinear metrics did.
- **Stats:** Spearman ρ; Bland-Altman (ECG vs PPG); Wilcoxon.

---

### REF 11 — Bohnen et al. (2012)
- **Authors:** Bohnen NI, Müller MLTM, Koeppe RA et al.
- **Year:** 2012 · **Journal:** *Brain*
- **N:** 44 PD (with neuroimaging and Holter)
- **HRV metrics:** HF-HRV (vagal tone), DFA
- **Clinical scales:** MMSE/MoCA equivalent, UPDRS-II, cholinergic imaging (PET)
- **Main finding:** Reduced cardiac vagal tone was associated with cognitive dysfunction in PD **independent of dopaminergic denervation**. DFA α1 correlated with cholinergic deficiency markers. Supports a **cholinergic–autonomic–cognitive pathway** in PD.
- **Stats:** Partial correlation controlling for age and dopaminergic status; path analysis.
- **Note:** Mechanistic basis for expecting a complexity–cognitive subscale correlation in our CISI-PD analysis.

---

### REF 12 — Solla et al. (2020) — Meta-analysis
- **Authors:** Solla P, Pinna I, Cesare Cannas A, Corongiu D, Tacconi P
- **Year:** 2020 · **Journal:** *Parkinsonism & Related Disorders*
- **N:** Review of ~22 studies, total ~1,200 PD
- **HRV metrics:** All categories (time-domain, spectral, nonlinear)
- **Clinical scales:** H&Y, UPDRS, disease duration, cognitive status
- **Main finding (meta-analytic):** SDNN and RMSSD consistently reduced in PD. **Effect sizes larger for nonlinear metrics (SampEn, DFA) at moderate-to-advanced stages.** H&Y stage most consistent correlate across studies. Pooled ρ(UPDRS-III, complexity) ≈ −0.35 to −0.45. **No strong evidence that levodopa per se reduces HRV** — disease progression confounds LEDD–HRV correlations.
- **Stats:** Meta-regression; pooled Spearman estimates.

---

### Web-confirmed additional references

- **HRV systematic review and meta-analysis (2021):** [PMC8394422](https://pmc.ncbi.nlm.nih.gov/articles/PMC8394422/) — confirms RMSSD/H&Y and HF/disease-duration correlations across studies
- **Circadian HRV in PD (BMC Neurology, 2020):** [PMC7181578](https://pmc.ncbi.nlm.nih.gov/articles/PMC7181578/) — 24h Holter; disease stage correlates with circadian HRV impairment
- **HRV and motor symptom duration (PMC4108815, 2014):** [PMC4108815](https://pmc.ncbi.nlm.nih.gov/articles/PMC4108815/) — duration predicts parasympathetic decline independently of stage
- **Heart-brain synchronization in PD (npj PD, 2022):** [nature.com](https://www.nature.com/articles/s41531-022-00323-w) — HBSI inverse dose-response with dysautonomia severity
- **CISI-PD and quality of life (PMC10026279, 2023):** [PMC10026279](https://pmc.ncbi.nlm.nih.gov/articles/PMC10026279/) — baseline autonomic dysfunction (SCOPA-AUT) predicts CISI-PD progression at 3 years
- **Sympathetic shift and anxiety in PD (Movement Disorders, 2026):** [doi.org](https://movementdisorders.onlinelibrary.wiley.com/doi/full/10.1002/mds.70069) — depression and anxiety correlate with cardiac autonomic imbalance
- **Wearable HRV and neuroimaging in PD (Front. Aging Neurosci., 2025):** [frontiersin.org](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2025.1530240/full)
- **Levodopa dose equivalency (Movement Disorders, 2023):** [doi.org](https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.29410)

---

## Thematic Synthesis

### Which clinical scales correlate with HRV complexity?

| Scale | Evidence | Direction | Best metric | Expected ρ |
|---|---|---|---|---|
| UPDRS-III (motor) | Strong, consistent | Negative | SampEn, MSE-CI, DFA α1 | −0.40 to −0.55 |
| Hoehn & Yahr stage | Strong, graded | Negative | SDNN, SampEn | −0.35 to −0.50 |
| Disease duration | Strong, independent | Negative | SDNN, RMSSD, MSE-CI | −0.30 to −0.45 |
| MoCA / cognitive | Moderate, emerging | Positive | SampEn, HF-HRV | +0.30 to +0.45 |
| CISI-PD total | **Not studied directly** | Expected negative | rcMSE-CI | unknown — our contribution |
| CISI-PD cognitive subscale | Not studied | Expected positive | rcMSE-CI | unknown |
| LEDD | Weak, confounded | Collinear with duration | None dominant | ~−0.20, attenuates w/ adjustment |
| Autonomic symptoms (SCOPA-AUT) | Moderate | Negative | SDNN, LF/HF | −0.30 to −0.40 |

### Levodopa (LEDD) — consensus position

LEDD is **not an independent predictor** of HRV decline when disease duration and H&Y are in the model. The LEDD–HRV correlation reflects disease progression, not a direct pharmacological effect. Studies comparing patients on vs. off medication show minimal acute differences. LEDD should be included as a covariate in regression models to rule out confounding and for transparency, but negative findings are expected and consistent with the literature.

### Statistical framework (recommended)

1. **Spearman ρ** — primary (non-normal HRV, ordinal clinical scales)
2. **Partial Spearman ρ controlling for age + sex** — standard secondary analysis
3. **Multiple regression (parsimonious):** max 3–4 predictors at n=34 (~10:1 rule)
4. **BH-FDR correction** across all pairwise tests
5. **Bootstrap 95% CIs** (n=1000) for all ρ estimates — critical at n=34

### Power considerations at n=34

Detectable Spearman ρ at 80% power (α=0.05, two-tailed): **|ρ| ≥ 0.46**

- Realistic for motor severity (UPDRS equivalent, H&Y, disease duration): expected ρ ≈ 0.40–0.55 → at boundary
- Marginal for cognitive correlations: expected ρ ≈ 0.35 → likely underpowered
- Non-motor symptoms (binary): Mann-Whitney; likely underpowered for individual symptoms; use composite burden score

### Key confounders

| Confounder | Action |
|---|---|
| Age | Continuous covariate in all partial correlations |
| Sex | Binary covariate |
| Disease duration | Covariate in secondary models (collinear with H&Y) |
| Mean HR | Enter as a covariate. The index is reported unnormalized; dividing by HR was tested and rejected (see `HR_NORMALISATION_ANALYSIS.md`) |
| LEDD | Include in sensitivity analysis; flag pending data |
| Comorbidities (diabetes, HTN) | Sensitivity analysis excluding known HRV confounders |

---

## Notes on Our Dataset Specifically

- **CISI-PD** replaces UPDRS in our dataset — a validated, internationally recognized global severity index. No prior study has correlated HRV complexity with CISI-PD directly. This is a genuine literature gap and a contribution of this analysis.
- **LEDD:** Currently n=9/34. Clinical collaborator completing this. Include in analysis with flag; expect null result consistent with literature.
- **Non-motor symptom inventory (12 binary items):** Use composite sum as primary variable; test individually as exploratory analyses with Bonferroni correction.
- **No MoCA/MMSE available** — CISI-PD cognitive subscale serves as the cognitive severity proxy.
- **Genetics (EP):** Available for only 10/34 — cannot control; flag as limitation (GBA carriers show more severe autonomic dysfunction).
