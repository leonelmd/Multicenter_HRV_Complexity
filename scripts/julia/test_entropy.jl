# -*- coding: utf-8 -*-
#
# Regression tests for entropy.jl.
#
# These exist because the per-scale-r defect went undetected for six months.
# Test 4 fails loudly for any implementation that recomputes r per scale, and
# test 5 pins the invariance that makes r = factor * SD meaningful at all.
#
# Run:  julia --project=. test_entropy.jl

using Test, Statistics, Random

include(joinpath(@__DIR__, "entropy.jl"))

Random.seed!(42)

# --------------------------------------------------------------- helpers
white(n = 1200) = (x = randn(n); (x .- mean(x)) ./ std(x))

function pink(n = 1200)                      # 1/f noise via spectral synthesis
    f = (1:div(n, 2)) ./ n
    amp = 1 ./ sqrt.(f)
    ph = 2π .* rand(length(f))
    spec = amp .* exp.(im .* ph)
    x = real(ifft_like(spec, n))
    return (x .- mean(x)) ./ std(x)
end

function ifft_like(spec, n)                  # minimal real inverse DFT
    x = zeros(n)
    for k in eachindex(spec)
        @inbounds for t in 1:n
            x[t] += real(spec[k]) * cos(2π * k * (t - 1) / n) -
                    imag(spec[k]) * sin(2π * k * (t - 1) / n)
        end
    end
    return x
end

logistic(n = 1200) = begin
    x = zeros(n); x[1] = 0.4
    for i in 2:n; x[i] = 3.9 * x[i-1] * (1 - x[i-1]); end
    (x .- mean(x)) ./ std(x)
end

@testset "entropy.jl" begin

    # --- 1. SampEn basics ------------------------------------------------
    @testset "sample entropy" begin
        x = white(600)
        se = sample_entropy(x, 2, 0.2 * std(x))
        @test isfinite(se)
        @test 1.0 < se < 3.0                     # white noise, m=2, r=0.2

        # A perfectly regular series does NOT give exactly zero. The two
        # embedding orders are counted over different numbers of templates
        # (Richman & Moorman): generate_windows gives N-m+1 windows of length m
        # but only N-m of length m+1. With every pair matching,
        #     B = C(N-m+1, 2),  A = C(N-m, 2),  A/B = (N-m-1)/(N-m+1)
        # so SampEn = log((N-m+1)/(N-m-1)) ~ 2/(N-m) — a finite-size floor, not
        # a defect. At N=200, m=2 that is log(199/197) = 0.0101011.
        # Pinned exactly so any change to the windowing convention is caught.
        let N = 200, m = 2
            @test sample_entropy(ones(N), m, 0.2) ≈ log((N - m + 1) / (N - m - 1)) atol = 1e-12
            @test sample_entropy(ones(N), m, 0.2) ≈ 0.0101011 atol = 1e-6
            # the floor shrinks as ~2/(N-m): a longer constant series scores lower
            @test sample_entropy(ones(800), m, 0.2) < sample_entropy(ones(N), m, 0.2)
        end
        # deterministic periodic signal is far less entropic than noise
        per = repeat([1.0, 2.0, 3.0, 4.0], 150)
        @test sample_entropy(per, 2, 0.2 * std(per)) < se
    end

    # --- 2. fuzzy entropy behaves like sample entropy --------------------
    @testset "fuzzy entropy" begin
        x = white(500)
        r = 0.2 * std(x)
        @test isfinite(fuzzy_entropy(x, 2, r))
        @test fuzzy_entropy(x, 2, r) > fuzzy_entropy(repeat([1.0, 2.0], 250), 2, r)
    end

    # --- 3. Wu (2014) coarse-graining ------------------------------------
    @testset "coarse-graining scheme" begin
        x = white(1000)
        r = 0.2 * std(x)
        # scale 1 admits exactly one shift, so rcMSE == plain SampEn there
        @test refined_composite_multiscale_entropy(x, 2, r, "sample", [1])[1] ≈
              sample_entropy(x, 2, r) atol = 1e-10
        # all scales defined over a sane range
        c = refined_composite_multiscale_entropy(x, 2, r, "sample", collect(1:5))
        @test length(c) == 5
        @test all(isfinite, c)
        # dropping ONE sample must not move scale-5 entropy appreciably.
        # Under the old `t in 0:(N % scale)` scheme this shifted by up to 3.7%
        # because the number of averaged coarse-grainings changed.
        a = refined_composite_multiscale_entropy(x, 2, r, "sample", [5])[1]
        b = refined_composite_multiscale_entropy(x[1:end-1], 2, r, "sample", [5])[1]
        @test abs(a - b) / a < 0.01
    end

    # --- 4. THE REGRESSION TEST: r must be fixed across scales -----------
    # With fixed r the canonical Costa contrast holds: white-noise entropy
    # decays with scale while 1/f noise stays high. Recomputing r per scale
    # destroys both signatures. Any implementation that reintroduces the bug
    # fails here.
    @testset "fixed r reproduces the Costa signature" begin
        scales = collect(1:10)
        w = white(1200); p = pink(1200)
        cw = refined_composite_multiscale_entropy(w, 2, 0.2 * std(w), "sample", scales)
        cp = refined_composite_multiscale_entropy(p, 2, 0.2 * std(p), "sample", scales)

        @test cw[10] < cw[1]                     # white noise decays
        @test cw[10] < 0.75 * cw[1]              # and substantially so
        @test cp[10] > cw[10]                    # 1/f ends above white noise
        @test compute_nAUC(cp) > compute_nAUC(cw)

        # deterministic chaos sits well below both
        cl = refined_composite_multiscale_entropy(logistic(1200), 2,
                                                  0.2 * std(logistic(1200)),
                                                  "sample", [1])
        @test cl[1] < cw[1]
    end

    # --- 5. scale invariance of r = factor * SD --------------------------
    @testset "amplitude invariance" begin
        x = white(800)
        s1 = refined_composite_multiscale_entropy(x, 2, 0.2 * std(x), "sample", collect(1:4))
        y = 1000.0 .* x .+ 57.0                  # rescale and offset
        s2 = refined_composite_multiscale_entropy(y, 2, 0.2 * std(y), "sample", collect(1:4))
        @test all(abs.(s1 .- s2) .< 1e-9)
    end

    # --- 6. summary statistics -------------------------------------------
    @testset "nAUC and LRS" begin
        # compute_nAUC divides the trapezoidal area by the NUMBER OF POINTS n,
        # not by the span (n-1). For a constant curve of height c over n scales
        # it therefore returns c*(n-1)/n, not c. This is a property of the
        # canonical definition, not a defect, but it means the value carries an
        # n-dependent factor: 0.80 at n=5, 0.95 at n=20. Within a cohort the
        # scale count is fixed so AUCs and p-values are unaffected (monotone
        # rescaling); across cohorts with different scale ranges the raw values
        # are NOT directly comparable. Pinned here so the behaviour cannot drift.
        @test compute_nAUC(fill(2.0, 10)) ≈ 2.0 * 9 / 10 atol = 1e-12
        @test compute_nAUC(fill(1.0, 5))  ≈ 0.80 atol = 1e-12
        @test compute_nAUC(fill(1.0, 20)) ≈ 0.95 atol = 1e-12
        # NaN and Inf are dropped, not propagated
        @test isfinite(compute_nAUC([1.0, NaN, 2.0, Inf, 3.0]))
        # LRS recovers a known slope
        sc = collect(1.0:10.0)
        @test compute_LRS(0.5 .* sc .+ 3.0, sc) ≈ 0.5 atol = 1e-9
    end

    # --- 7. degenerate input is NaN, never an exception -------------------
    @testset "degenerate input" begin
        c = refined_composite_multiscale_entropy(white(300), 2, 1e-12, "sample", [1, 2])
        @test all(x -> isnan(x) || isfinite(x), c)
    end
end

println("\nAll entropy.jl tests passed.")
