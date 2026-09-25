# -*- coding: utf-8 -*-
# @author: max (max@mdotmonar.ch)
#
# Canonical entropy toolbox for the multicenter cardiac-complexity study.
# This is the authoritative implementation of rcMSE used for every published
# complexity value. The Python mirror in ../complexity_index.py and the
# per-cohort inlined variants must agree with this file.
#
# ---------------------------------------------------------------------------
# THE TOLERANCE r IS FIXED
# ---------------------------------------------------------------------------
# `r` is an argument. It is computed ONCE by the caller from the original
# (un-coarse-grained) series and is never recomputed per scale.
#
# This matters. Coarse-graining at scale tau averages tau consecutive points and
# therefore reduces the variance of the series. Holding r fixed lets that
# variance reduction express itself in the entropy — which is precisely the
# quantity multiscale entropy exists to measure. Recomputing r from each
# coarse-grained series (r = 0.2 * std(coarse_signal)) renormalises every scale
# to its own variance and cancels exactly that term. On synthetic signals the
# per-scale variant fails the canonical check: white-noise entropy stops decaying
# with scale and 1/f noise rises, inverting the standard Costa contrast.
#
# See Multicenter/RCMSE_TOLERANCE_AUDIT.md for the full audit.
#
# ---------------------------------------------------------------------------
# Two documented deviations from the original reference implementation
# ---------------------------------------------------------------------------
#  1. `composite_multiscale_entropy` and `refined_composite_multiscale_entropy`
#     use the Wu et al. (2014) coarse-graining scheme: all `scale` shifts, each
#     coarse-grained series shortened as required. The original used
#     `t in 0:(N % scale)` with a fixed `ratio`, which made the number of
#     averaged coarse-grainings equal to (N mod scale) + 1 — an arbitrary
#     quantity. At scale 20 a subject with N=1300 received 1 coarse-graining
#     while N=1319 received 20, so the variance reduction that "refined
#     composite" exists to provide varied erratically between subjects.
#     Bounds are safe: i*scale + k - 1 <= (N - k + 1) + k - 1 = N.
#
#  2. `-log(A/B)` is guarded to return NaN when A == 0 or B == 0, rather than
#     Inf or a DomainError.
#
# References
#   Richman & Moorman (2000)  Am J Physiol 278:H2039        - SampEn
#   Costa, Goldberger & Peng (2002, 2005) PRL 89:068102; PRE 71:021906 - MSE, fixed r
#   Wu et al. (2014) Phys Lett A 378:1369                    - refined composite
#   Chen et al. (2007) Med Eng Phys 31:61                    - fuzzy entropy

using Statistics
using Combinatorics
using Trapz
using CurveFit

function chebyshev_distance(x, y)
	return maximum(abs.(x - y))
end

function generate_windows(signal, m)
	N = length(signal)
	return [signal[i:i + m - 1] for i in 1:N - m + 1]
end

function sample_entropy_matches(signal, m, r)
	# generate windows from signal
	m_vector = generate_windows(signal, m)
	m1_vector = generate_windows(signal, m + 1)

	# compute the number of matches
	A = sum([(chebyshev_distance(i, j) <= r) ? 1 : 0 for (i, j) in combinations(m1_vector, 2)])
	B = sum([(chebyshev_distance(i, j) <= r) ? 1 : 0 for (i, j) in combinations(m_vector, 2)])

	return A, B
end

# NOTE ON THE FINITE-SIZE FLOOR
# The two embedding orders are counted over different numbers of templates:
# generate_windows yields N-m+1 windows of length m but only N-m of length m+1.
# A perfectly regular series therefore scores log((N-m+1)/(N-m-1)) ~ 2/(N-m),
# not zero. This is the standard Richman & Moorman formulation, not a defect,
# but the floor grows as the series shortens — 0.002 at N=1088, 0.009 at N=217
# (CETRAM coarse-grained to tau=5), 0.021 at N=99 (Cruces at tau=5). It is under
# 2% of the observed entropies at the scales used here, but it does differ
# between cohorts of different length: one more reason to compare effect sizes
# rather than raw entropy across cohorts. Pinned in test_entropy.jl.
function sample_entropy(signal, m, r)
	A, B = sample_entropy_matches(signal, m, r)
	return (A == 0 || B == 0) ? Inf : -log(A / B)
end

function fuzzy_entropy_matches(signal, m, r)
	# generate windows from signal
	m_vector = generate_windows(signal, m)
	m1_vector = generate_windows(signal, m + 1)

	# compute the number of matches
	A = sum([exp( -log(2)*( (chebyshev_distance(i, j)/r)^2 ) ) for (i, j) in combinations(m1_vector, 2)])
	B = sum([exp( -log(2)*( (chebyshev_distance(i, j)/r)^2 ) ) for (i, j) in combinations(m_vector, 2)])

	return A, B
end

function fuzzy_entropy(signal, m, r)
	A, B = fuzzy_entropy_matches(signal, m, r)
	return (A == 0 || B == 0) ? Inf : -log(A / B)
end

function multiscale_entropy(signal, m, r, e, scales = [i for i in 1:trunc(Int, length(signal)/(m+10))])
	N = length(signal)

	en_list = Float64[]

	for scale in scales
		# coarse-graining
		ratio = trunc(Int, N / scale)
		coarse_signal = zeros(ratio)
		for i in 1:ratio
			coarse_signal[i] = sum(signal[(i - 1) * scale + 1:i * scale]) / scale
		end

		# entropy — note r is the caller's r, unchanged
		if e == "sample"
			push!(en_list, sample_entropy(coarse_signal, m, r))
		elseif e == "fuzzy"
			push!(en_list, fuzzy_entropy(coarse_signal, m, r))
		end
	end

	return en_list
end

function composite_multiscale_entropy(signal, m, r, e, scales = [i for i in 1:trunc(Int, length(signal)/(m+10))])
	N = length(signal)

	en_list = Float64[]

	for scale in scales
		cumulative_en = 0.0
		n_used = 0

		# Wu et al. (2014): all `scale` shifts, each shortened as needed
		for k in 1:scale
			ratio = trunc(Int, (N - k + 1) / scale)
			ratio <= m && continue
			coarse_signal = zeros(ratio)
			for i in 1:ratio
				coarse_signal[i] = sum(signal[(i - 1) * scale + k:i * scale + k - 1]) / scale
			end

			if e == "sample"
				cumulative_en += sample_entropy(coarse_signal, m, r)
			elseif e == "fuzzy"
				cumulative_en += fuzzy_entropy(coarse_signal, m, r)
			end
			n_used += 1
		end

		push!(en_list, n_used == 0 ? NaN : cumulative_en / n_used)
	end

	return en_list
end

function refined_composite_multiscale_entropy(signal, m, r, e, scales = [i for i in 1:trunc(Int, length(signal)/(m+10))])
	N = length(signal)

	en_list = Float64[]

	for scale in scales
		# cumulative matches pooled across coarse-grainings
		A = 0
		B = 0

		# Wu et al. (2014): all `scale` shifts, each shortened as needed
		for k in 1:scale
			ratio = trunc(Int, (N - k + 1) / scale)
			ratio <= m && continue
			coarse_signal = zeros(ratio)
			for i in 1:ratio
				coarse_signal[i] = sum(signal[(i - 1) * scale + k:i * scale + k - 1]) / scale
			end

			if e == "sample"
				A_m, B_m = sample_entropy_matches(coarse_signal, m, r)
			elseif e == "fuzzy"
				A_m, B_m = fuzzy_entropy_matches(coarse_signal, m, r)
			end
			A += A_m
			B += B_m
		end

		push!(en_list, (A > 0 && B > 0) ? -log(A / B) : NaN)
	end

	return en_list
end

# NOTE ON NORMALISATION
# compute_nAUC divides the trapezoidal area by the NUMBER OF POINTS n, not by the
# span (n-1). For a constant curve of height c over n scales it returns c*(n-1)/n
# — 0.80c at n=5, 0.95c at n=20. Within a cohort the scale count is fixed, so this
# is a constant rescaling and leaves AUCs, p-values and correlations unchanged.
# Across cohorts analysed over DIFFERENT scale ranges the raw values are not
# directly comparable: a flat curve scores ~19% higher on 1-20 than on 1-5 from
# the formula alone. Cross-cohort comparison should use effect sizes, not raw nAUC.
function compute_nAUC(curve)
	#filter out NaN values
	curve = curve[.!isnan.(curve)]
	#filter out Inf values
	curve = curve[.!isinf.(curve)]

	return trapz([i for i in 1:length(curve)], curve)/length(curve)
end

function compute_LRS(curve, scales)
	a, b = linear_fit(scales, curve)
	return b
end
