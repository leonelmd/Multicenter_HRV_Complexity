# Figure legends

Legends for all main and appendix figures. Figures carry no titles; all interpretation
lives here. Panels are lettered as drawn.

**Conventions used throughout.** rcMSE denotes refined composite multiscale entropy
(Wu et al. 2014) computed with sample entropy, embedding dimension *m* = 2 and tolerance
*r* = 0.2 × SD of the original (uncoarse-grained) interval series, held constant across
scales. The complexity index is the normalised area under the rcMSE curve, nAUC(1–5)
unless stated, i.e. the trapezoidal integral over scales 1–5 divided by the number of
scales. Indices are reported **unnormalised**; heart rate is entered as a covariate
where adjustment is required. Group contrasts are Mann–Whitney *U* tests and areas under
the receiver-operating-characteristic curve (AUC) oriented as Control > PD, so AUC > 0.5
indicates lower values in PD. Cohorts: CETRAM (Santiago; 15-min supine resting ECG at
1000 Hz; 43 controls, 30 PD), Cruces (Bilbao; 7.40-min finger PPG at 500 Hz recorded
during resting-state fMRI; 21 healthy controls, 31 PD), Nagoya (24-h Holter; 21 disease
controls, 26 PD in the analysis window). Nagoya disease controls were investigated for
essential tremor or for numbness/dizziness with no abnormality found, so the Nagoya
contrast is against a harder comparator than the two healthy-control cohorts.

---

## Figure 1 — Overview of the multicentre design

**Top.** The three contributing cohorts — Santiago (Chile), Bilbao (Spain) and Nagoya
(Japan) — with the acquisition modality of each: resting electrocardiography, finger
photoplethysmography acquired during resting-state functional MRI, and a 24-h wearable
chest-strap recording respectively.

**Bottom.** The analytical contrast drawn in this work. The interbeat-interval series is
summarized either by (A) conventional linear variability indices, which describe the
magnitude of beat-to-beat variation, or by (B) multiscale entropy, in which the series is
progressively coarse-grained and sample entropy SampEn(m, r, N) is evaluated at each
scale, thereby characterizing how the fluctuations are organized across temporal scales.

Prepared manually in Affinity Designer; the source file is `Figure1.af`. It is not
script-generated and therefore does not update when the analysis is re-run.

**Labels requiring correction before submission.** Three items on the current artwork do
not match the analysis: 1) Santiago is labelled 28 PD, whereas the analysed sample is 30;
2) Bilbao is labelled 15 min PPG, whereas the recordings are 7.40 min; and 3) Nagoya is
labelled 24 h Holter, whereas the device is a POLAR V800/H10 chest strap validated
against Holter ECG, a distinction the Discussion draws explicitly. Finally, the Nagoya
counts shown (27 PD, 23 controls) are those of the full cohort rather than of the 47
participants in the 18:00–22:00 analysis window.

---

## Figure 2 — Signal archetypes and conventional autonomic context

**(A–C)** Interbeat-interval traces for all available recordings in each cohort, plotted
faintly with the group mean overlaid. Panel C spans the full 24-h Nagoya record and
therefore includes all 23 disease controls and 27 PD participants; the complexity
analysis elsewhere is restricted to the 47 participants with data in the pre-specified
18:00–22:00 window, so the counts differ by design.

**(D–F)** Mean heart rate by group, shown as violin plots with individual participants
overlaid. PD have higher rates in all three cohorts, significantly so only in Nagoya.

**(G–I)** Age by group. No cohort shows a significant age difference.

**(J–L)** Poincaré plots of successive interval pairs, with 95% confidence ellipses per
group, illustrating how the dispersion and short-term correlation structure of the
interval series differ across acquisition modality.

All *p* values are Mann–Whitney *U*.

---

## Figure 3 — Recording length determines which autonomic marker is applicable

Nagoya 24-h Holter cohort, which is the source cohort of Suzuki et al. (2022); minimum-SDNN
values shown here are a reproduction of that published analysis in the same subjects.
No window, scale or parameter in this figure was selected on the group contrast.

**(A)** Complexity index in consecutive non-overlapping 15-minute windows across the 24-h
recording, averaged within clock hour (mean ± SEM; 3667 windows from 50 subjects, median
1049 beats per window). Fifteen minutes is the shortest window that satisfies the
reliability constraint *N*/τ ≥ 200 at τ = 5 and is also the CETRAM protocol length, making
it the only window at which all three cohorts are directly comparable. Shading marks the
pre-specified 18–22 h window used elsewhere in the paper.

**(B)** AUC for the complexity index and for SDNN at every clock hour. Discrimination is
present across most of the day rather than confined to any one period; no window is
nominated.

**(C)** AUC for all 24 candidate 4-hour windows (nAUC 1–20). Filled symbols survive
Bonferroni correction for 24 comparisons (*p* < 0.05/24 = 0.0021); 7 of 24 do so and 18 of
24 reach uncorrected *p* < 0.05. AUC ranges 0.546–0.881 with a median of 0.746. The
previously used 16–20 h window, which had been chosen for maximal separation and ranked
3rd of 24, is marked, as is its replacement. The 18–22 h window was pre-specified on a
label-blind criterion — the earliest 4-h window with complete subject retention (*n* = 45)
and ≥ 99 % median temporal coverage by retained beats — and discriminates *less* well
(0.773 vs 0.835), confirming it was not selected on the outcome. It also raises median
beat coverage from 89.6 % to 99.6 % and reduces the number of subjects below 80 % coverage
from 21 to 3.

A mixed model over all 15-minute windows, which requires no window selection, gives a
group effect of β_PD = −0.105 (*p* = 0.015) for `cx ~ PD + (1 | subject)`; adding
24-h cosinor terms gives a PD main effect of *p* = 0.0066 with significant PD × time-of-day
interactions (*p* < 0.001), indicating a deficit that is present across the day and also
varies with time of day.

**(D)** AUC of the minimum-across-day statistic as a function of window length, for SDNN
and for the complexity index. SDNN discrimination is maximal in short windows (100 beats,
0.925; 200 beats, 0.940) and declines as windows lengthen (15 min, 0.797; 4 h, 0.752).
The complexity index cannot be estimated below ~15 minutes because *N*/τ falls under 200
at τ = 5 (shaded region), and is 0.845 at 15 min and 0.773 at 4 h. The two markers
therefore have opposite window-length requirements.

**(E)** The four distributional summaries used by Suzuki et al. (minimum, first decile,
first quartile, median across the recording), computed for SDNN in 100-beat windows and
for the complexity index in 15-minute windows. Minimum-SDNN is the strongest index
(0.925) and exceeds the corresponding complexity statistic at every summary level.

**(F)** The same two markers in the 24-h cohort and in CETRAM's 15-minute resting
recordings. In Nagoya, minimum-SDNN (0.925) exceeds complexity (0.845), is correlated with
it at ρ = +0.80, and complexity adds no incremental discrimination in a joint logistic
model (*p* = 0.46). In CETRAM, where only ~10 hundred-beat windows are available per
subject rather than ~920, minimum-SDNN falls to 0.460 while complexity reaches 0.695;
excluding the 11 subjects whose interval series are inconsistent with sinus rhythm
(SDNN > 100 ms in a 15-minute supine recording) the values are 0.532 and 0.628. The
minimum is an extreme-value statistic and requires many windows to identify the transient
low-variability episodes that carry the group difference; that requirement is not met by
short clinical recordings.

**(G)** Scale-resolved sample entropy out to τ = 60 in the pre-specified 18–22 h window
(*n* = 47, mean RR 819 ms), plotted against physical timescale (τ × mean RR). Background
shading marks conventional HRV frequency bands. Resolving scales beyond τ ≈ 20 requires
the beat count that only a multi-hour recording provides.

**(H)** AUC as a function of timescale over the same range. Discrimination peaks at
τ = 4 (3.3 s, respiratory band; AUC 0.870) rather than at scale 1, and declines
monotonically thereafter.

**(I)** Effect of window overlap on the distributional statistics, computed for SDNN in
15-minute windows across all 45 subjects and expressed relative to non-overlapping
windows. Increasing overlap from 0 % to 90 % raises the number of windows per subject from
92 to 911 and lowers the median minimum by 9 %, because the minimum over more draws is
mechanically smaller, while leaving AUC essentially unchanged (within ±0.026, non-monotonic).
The tenth percentile is completely invariant. Overlapping windows therefore add no
information and render minimum-based statistics incomparable across studies; all analyses
here use non-overlapping windows, and the tenth percentile is preferred to the minimum as
the low-tail statistic.

---

## Figure 4 — Multiscale entropy curves and complexity index across cohorts

**(A–D)** Mean rcMSE curves (± SEM) by group for CETRAM, Cruces, Nagoya 07–11 h and
Nagoya 18–22 h. Shaded bands mark the scale range entering the complexity index for that
cohort, set by the *N*/τ ≥ 200 reliability rule. Curve direction differs between cohorts:
CETRAM falls with scale while Cruces and Nagoya rise. This is not an artefact of recording
length — truncating all cohorts to a common 498 beats preserves each shape — but follows
from the entropy at τ = 1, which within each cohort falls as the interval standard
deviation rises, a consequence of setting *r* = 0.2 × SD.

**(E–H)** Complexity index by group for the same four datasets, unnormalised. Box plots
show median and interquartile range with individual subjects overlaid. The upper *p* value
is the unadjusted Mann–Whitney test; the lower line gives the *p* value and AUC with mean
heart rate entered as a covariate, which is the appropriate adjustment (see Appendix 5).
CETRAM: *p* = 0.0048, HR-adjusted *p* = 0.0009, AUC 0.692. Cruces: *p* = 0.275,
HR-adjusted *p* = 0.589, AUC 0.536. Nagoya 07–11 h: *p* = 0.0148, HR-adjusted *p* = 0.0400,
AUC 0.728. Nagoya 18–22 h: *p* = 0.0001, HR-adjusted *p* = 0.0003, AUC 0.835. The Cruces
contrast does not survive heart-rate adjustment.

---

## Figure 5 — Cross-cohort classification

Leave-one-cohort-out cross-validation using handcrafted HRV and complexity features.
Models are trained on two cohorts and tested on the held-out third, so no subject
contributes to both training and testing and no cohort-specific scaling is learned.
Reported are ROC curves and AUCs per held-out cohort together with a pooled estimate.
Feature importance is shown for the full-data model. Leave-one-cohort-out is a
deliberately conservative design: it measures transfer across acquisition modality,
recording length and population simultaneously.

---

## Figure 6 — Age independence of the complexity index

Complexity index against age, by group, in each cohort and window. Lines are ordinary
least-squares fits with 95 % confidence bands; annotations give Spearman ρ and its *p*
value within group. No cohort shows a significant age association in either group, so the
group differences reported elsewhere are not attributable to the age difference between
groups.

---

## Figure 7 — Relationship between complexity and conventional HRV

**(A)** Spearman correlation heatmap between per-scale entropy and conventional
time- and frequency-domain HRV indices, per cohort and group, with Benjamini–Hochberg
control of the false discovery rate within cohort. Correlations use raw per-scale entropy,
not the heart-rate-normalised index.

**(B)** Cross-dataset consistency of those correlation patterns.

**(C)** Scale-resolved physiological interpretation across the three cohorts.

**(D)** Variance decomposition of the complexity index into components attributable to
conventional HRV indices and a residual component.

**(E)** Annotated mean rcMSE curves identifying the scale ranges corresponding to
respiratory, baroreflex and very-low-frequency dynamics.

Absolute entropy values are not comparable across cohorts because *r* = 0.2 × SD makes the
tolerance depend on each recording's low-frequency content; pooled analyses use
within-cohort standardised values.

---

## Figure 8 — Exploratory association with clinical measures (preliminary)

Twenty-five CETRAM PD participants had clinical records matched to the complexity
analysis.

**(A)** Raw Spearman ρ between complexity, RMSSD and DFA α₁ and each clinical variable.
**(B)** The same correlations adjusted for age.
**(C–E)** Scatter plots for CISI-PD total, Hoehn & Yahr stage and disease duration.
**(F–G)** Complexity by Hoehn & Yahr stage and by presence of each non-motor symptom.

No association survived Benjamini–Hochberg correction. The levodopa-equivalent daily dose
(LEDD) panel is **not populated**: the source field is recorded as free text, e.g.
"≈1.475 mg LED/día (estimado)", and has not been converted to numeric values, partly
because the Spanish thousands convention makes automatic parsing unsafe. Eight of the 25
participants have an LEDD entry. This figure carries a preliminary marking until that
conversion is agreed with the clinical collaborator.

The underlying clinical workbook is not redistributed; see DATA_AVAILABILITY.md.

---

## Appendix 1 — Surrogate data analysis

rcMSE curves for the real interval series and for iterated amplitude-adjusted Fourier
transform (IAAFT) surrogates, which preserve the amplitude distribution and the linear
autocorrelation while destroying nonlinear structure. The group difference is largely
preserved in the surrogates (AUC 0.689 vs 0.699 for the real data), indicating that the
PD-versus-control contrast is carried mainly by linear structure. This does not mean the
signals are linear: Appendix 4 shows that the recordings are far less entropic than
spectrally matched noise.

## Appendix 2 — PPG-specific considerations in the Cruces cohort

Effect of pulse-wave morphology and sampling on the interval series derived from finger
PPG. Panels quantify the additional smoothing of the pulse waveform relative to the
R-peak (lag-1 autocorrelation +0.60 in Cruces against +0.39 in both ECG cohorts) and the
2 ms quantisation floor imposed by the 500 Hz sampling rate, which is 2.9 times CETRAM's
relative quantisation noise. Both attenuate fine-scale entropy and therefore the τ = 1
end of the curve.

## Appendix 3 — Sensitivity to recording length

Complexity index as a function of analysed series length, obtained by truncating each
recording to progressively fewer beats. Each cohort is restricted to subjects present at
all of its ladder points so that changes reflect length rather than composition. The
normalised AUC index is unbiased in *N*, so the shorter cohorts are not systematically
displaced; what changes with length is estimator variance, and hence reliability.

## Appendix 4 — Spectral characterisation and the beat-indexed time base

**(A–B)** Power spectral density of the interval series in the beat domain and the
distribution of the spectral exponent β per cohort. CETRAM sits at β = 0.51, midway
between white noise (0) and 1/f (1), with lag-1 autocorrelation +0.46; its falling rcMSE
curve therefore shares a direction with white noise but nothing else. There is no group
difference in β in any cohort (*p* = 0.30–0.72): PD alters the amplitude of the complexity
measure, not the scaling exponent.

**(C–D)** Real series compared against 1/f^β noise matched on length, exponent and
standard deviation. Real recordings are 28–125 % less entropic than their spectrally
matched surrogates at every scale in every cohort, so the data carry deterministic and
higher-order structure that a spectrally equivalent stochastic process does not.

**(E–F)** Consequences of the beat-indexed time base. An interval series is sampled by
beat, not at a fixed frequency, so coarse-graining at τ spans τ × mean RR seconds and the
physical timescale differs between groups whenever heart rate does — a 7 % offset in
CETRAM. Panel F replots entropy against τ × mean RR rather than τ. Measured in the beat
domain the spectral exponents are 0.51 / 0.85 / 0.62, while after interpolation onto a
uniform time base they are 0.84 / 1.05 / 0.97; because rcMSE coarse-grains by beat, it
sees the shallower beat-domain spectrum. Related to Courtiol et al. (2016), who showed for
uniformly sampled signals that coarse-graining acts as an imperfect low-pass filter and
that fine-scale entropy is dominated by broadband low-frequency power.

## Appendix 5 — Heart-rate normalisation

Test of whether the complexity index should be divided by mean heart rate.

**(A)** Mean heart rate by group and cohort; PD have higher rates in all three,
significantly so in Nagoya (*p* = 0.001).

**(B)** Spearman correlation between the index and heart rate, before and after dividing.
The unnormalised index is weakly heart-rate-dependent (ρ = −0.14, −0.26, −0.41); dividing
*increases* the dependence in every cohort (−0.50, −0.67, −0.62).

**(C)** Log–log regression slope of the index on heart rate. A ratio is the correct
correction only if the index scales as HR^+1, because dividing shifts the slope from *b*
to *b* − 1. Measured slopes are −0.09, −0.70 and −1.31, all with confidence intervals
excluding 1, so division moves every cohort further from heart-rate independence.

**(D)** AUC for heart rate alone, for the unnormalised index, for the ratio, and for the
index with heart rate as a covariate. The apparent gain from dividing (+0.046, +0.040,
+0.010) tracks the discriminative power of heart rate itself (0.615, 0.639, 0.788): the
ratio is an undisclosed composite of complexity and heart rate rather than a corrected
measure. Removing heart rate properly gives 0.699, 0.536 and 0.835.

**(E)** Nagoya index against heart rate, before and after division, illustrating the
induced dependence.

**(F)** Change in the correlation between the index and each conventional HRV metric
caused by dividing, plotted against that metric's own heart-rate dependence. Division
inflates every downstream correlation in proportion (Δρ up to +0.30).

The published figures therefore report the unnormalised index, with heart rate entered as
a covariate where adjustment is wanted.

## Appendix 6 — Sensitivity to the entropy parameters

Discrimination and split-half reliability across the two free parameters of sample
entropy: the tolerance multiplier *k* in *r* = *k* × SD(subject), and the embedding
dimension *m*.

**(A–B)** AUC and split-half reliability against *k*, with *m* = 2 drawn bold and *m* = 1
and 3 faint. Both are flat: across *k* = 0.125–0.50 and *m* = 1–3 the AUC moves by ≤ 0.044
(CETRAM), ≤ 0.085 (Cruces) and ≤ 0.020 (Nagoya). Red rings mark settings at which the
estimator is undefined for more than 1 % of a cohort.

**(C–D)** AUC and reliability against *m* at *k* = 0.2. Discrimination is nearly
indifferent to *m*, but reliability falls at *m* = 3 in every cohort because longer
templates match more rarely, so *m* = 2 sits at or near the reliability maximum.

**(E–G)** AUC over the full (*k*, *m*) grid per cohort; the published setting is boxed and
cells where the estimator is undefined for more than 1 % of subjects are marked with a
red cross. The low-*k*, high-*m* corner fails badly — at *k* = 0.05, *m* = 3 the estimator
is undefined for 84.6 % of Cruces and 46.6 % of CETRAM subjects — which is the reason *k*
is not pushed below 0.125.

**(H)** Summary of the published values and the range across the usable grid.

Parameters were fixed a priori by convention and were not optimised on the outcome; the
per-cohort optima disagree in any case (*k* = 0.05, 0.125 and 0.35 respectively). A fixed
absolute tolerance was considered and rejected: it correlates with the interval standard
deviation at ρ = +0.62 to +0.72 and drives CETRAM below chance, i.e. it measures amplitude
rather than complexity.

## Appendix 7 — The Cruces null in the context of the source study

The Cruces cohort is that of Iniguez et al. (npj Parkinsons Dis 2022;8:64), whose
reported group effect was a breakdown in synchronization between HRV and the BOLD signal
across the central autonomic network, not a difference in HRV itself; the HRV metrics in
that work served as inputs to the synchronization analysis rather than as endpoints.

**(a)** Discrimination, with bootstrap 95% confidence intervals, for the complete resting
HRV metric set of that study recomputed in the same 52 participants (21 controls, 31 PD),
together with our complexity index (outlined marker). Colour distinguishes time-domain,
non-linear and spectral families.

**(b)** Corresponding Mann–Whitney *p* values on a logarithmic axis. Every metric points
in the expected direction, yet none attains *p* < 0.05; LF power is closest at 0.050.
NNiqr, the metric the source study selected on the basis of its synchronization maps,
ranks among the weakest (AUC 0.592, *p* = 0.267).

**(c)** For comparison, the clinical autonomic findings reported in that paper: SCOPA-AUT
total, orthostatic hypotension prevalence, Valsalva pressure recovery time, Valsalva ΔSBP
phase IV, and the deep-breathing E/I ratio. **These values are quoted from the
publication and were not recomputed here**, as we did not have access to the
autonomic-test or neuroimaging data; they are plotted on their own axis for that reason.

Taken together, the panels show that the PD group is clinically dysautonomic while its
resting interbeat-interval series does not separate the groups by any conventional
measure. The failure of the complexity index in this cohort is therefore not specific to
the index, and is consistent with a source study that required simultaneous brain imaging
to demonstrate the group effect.

---

## Validation figure — Method verification against reference signals

rcMSE computed for synthetic signals of known complexity — white noise, 1/f noise, AR(1)
processes and the logistic map — alongside the cohort curves, confirming that the
implementation reproduces the published behaviour of the estimator. Also shown are the
variance-scaling exponent and the effective tolerance *r*/SD(τ) as a function of scale,
which together determine whether an rcMSE curve rises or falls under a fixed tolerance.

---

## References cited in the legends

- Costa M, Goldberger AL, Peng C-K. Multiscale entropy analysis of complex physiologic time series. *Phys Rev Lett* 2002;89:068102.
- Courtiol J, Perdikis D, Petkoski S, et al. The multiscale entropy: guidelines for use and interpretation in brain signal analysis. *J Neurosci Methods* 2016;273:175–190.
- Lipponen JA, Tarvainen MP. A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *J Med Eng Technol* 2019;43:173–181.
- Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol* 2000;278:H2039–H2049.
- Suzuki M, Nakamura T, Hirayama M, et al. Wearable sensor device-based detection of decreased heart rate variability in Parkinson's disease. *J Neural Transm* 2022;129:1299–1306.
- Wu S-D, Wu C-W, Lin S-G, Wang C-C, Lee K-Y. Time series analysis using composite multiscale entropy. *Entropy* 2013;15:1069–1084.
